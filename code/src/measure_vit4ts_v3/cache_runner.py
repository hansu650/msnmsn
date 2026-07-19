"""Parity-gated, cache-only scorer for the mandatory ViTTrace v3 ablations.

This runner never reads labels and never instantiates an encoder.  It consumes
the already frozen ViT-B/16 token caches and renderer-trace operators, writes
one transactional score per unique computation, and represents duplicate
logical arms as explicit aliases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml

from measure_vit4ts.cache import TOKEN_FILE, TOKEN_MANIFEST, load_clip_cache, make_cache_key
from measure_vit4ts.full_manifest import FullSeriesRecord, load_manifest
from measure_vit4ts.reducers import stitch_column_vectors
from measure_vit4ts.ritp import stitch_native_240

from .cache_registry import (
    REGISTRY_ID,
    CacheOnlyArm,
    CacheOnlyPlan,
    freeze_cache_only_registry,
    load_compute_plan,
    sha256_file,
)
from .core import (
    apply_temporal_operator,
    build_candidate_mask,
    build_linear_nctp,
    build_nearest_full_column_operator,
    column_quantile,
    column_top_fraction_mean,
    fuse_scale_subset,
    harmonic_incidence_projection,
    literal_incidence,
    released_incidence,
)
from .registry import load_arm_registry


RUN_SCHEMA_VERSION = 1
PARITY_GATE_SCHEMA_VERSION = 1
RUN_NAME = "cache_only_b16_w240_s60"
GRID_BY_SCALE = {"P": (14, 14), "M": (13, 13), "L": (12, 12)}
TOKEN_FIELD_BY_SCALE = {
    "P": "patch_tokens",
    "M": "mid_tokens",
    "L": "large_tokens",
}
MASK_FIELD_BY_SCALE = {"M": "mid_mask", "L": "large_mask"}
PARITY_ARMS = ("REL_U", "IHP_LEGACY", "FULL_COLUMN_240")
SOURCE_FILES = ("core.py", "cache_registry.py", "cache_runner.py")


@dataclass(frozen=True)
class ConfigBundle:
    path: Path
    payload: dict[str, Any]
    sha256: str
    manifest_path: Path
    manifest_sha256: str
    records: tuple[FullSeriesRecord, ...]


@dataclass(frozen=True)
class TraceOperators:
    path: Path
    manifest_path: Path
    sha256: str
    manifest_sha256: str
    window_starts: np.ndarray
    soft_data: np.ndarray
    soft_indices: np.ndarray
    soft_indptr: np.ndarray
    soft_offsets: np.ndarray
    hard_winner: np.ndarray


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def cache_runner_source_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in SOURCE_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("score must be a nonempty finite float64 vector")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _load_config(path: Path) -> ConfigBundle:
    config_path = Path(path).resolve(strict=True)
    raw = config_path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("stage") != "vittrace_ablation_full_v3":
        raise ValueError("cache-only scoring requires the isolated v3 config")
    defaults = payload.get("defaults", {})
    if (
        defaults.get("model_name") != "ViT-B-16"
        or int(defaults.get("patch_size", -1)) != 16
        or int(defaults.get("window", -1)) != 240
        or int(defaults.get("stride", -1)) != 60
    ):
        raise ValueError("cache-only scoring is frozen to ViT-B/16, W=240, stride=60")
    manifest_path = Path(payload["manifest"]["path"]).resolve(strict=True)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != str(payload["manifest"]["sha256"]).upper():
        raise ValueError("frozen manifest SHA256 changed")
    _, records = load_manifest(manifest_path)
    if len(records) != int(payload["manifest"]["expected_series"]):
        raise ValueError("frozen manifest series count changed")
    return ConfigBundle(
        config_path,
        payload,
        _sha256_bytes(raw),
        manifest_path,
        manifest_sha,
        records,
    )


def validate_parity_gate(path: Path, bundle: ConfigBundle) -> dict[str, Any]:
    """Fail closed unless the full 492-series parity gate authorizes scoring."""

    gate_path = Path(path).resolve(strict=True)
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    if not isinstance(gate, dict):
        raise ValueError("parity gate must be a JSON mapping")
    if (
        gate.get("schema_version") != PARITY_GATE_SCHEMA_VERSION
        or gate.get("decision") != "PASS"
        or gate.get("passed") is not True
        or int(gate.get("expected_series", -1)) != len(bundle.records)
        or int(gate.get("completed_series", -1)) != len(bundle.records)
        or str(gate.get("config_sha256", "")).upper() != bundle.sha256
        or str(gate.get("manifest_sha256", "")).upper() != bundle.manifest_sha256
    ):
        raise ValueError("full parity gate is absent, failed, incomplete, or stale")
    core_sha = sha256_file(Path(__file__).resolve().parent / "core.py")
    if str(gate.get("core_sha256", "")).upper() != core_sha:
        raise ValueError("parity gate is not bound to the active v3 core")
    arms = gate.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(PARITY_ARMS):
        raise ValueError("parity gate must cover exactly the three registered arms")
    for arm in PARITY_ARMS:
        row = arms[arm]
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            raise ValueError(f"parity arm did not pass: {arm}")
        for key in ("max_abs_error", "max_rel_error"):
            value = float(row[key])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid parity error for {arm}: {key}")
    return gate


def _validate_plan_and_registry(
    plan_path: Path,
    registry_path: Path,
    bundle: ConfigBundle,
) -> tuple[CacheOnlyPlan, dict[str, Any]]:
    plan, plan_payload = load_compute_plan(plan_path)
    if (
        str(plan_payload.get("config_sha256", "")).upper() != bundle.sha256
        or str(plan_payload.get("manifest_sha256", "")).upper()
        != bundle.manifest_sha256
    ):
        raise ValueError("compute plan is stale for the active config/manifest")
    registry = load_arm_registry(registry_path)
    if registry.registry_id != REGISTRY_ID or registry.arm_ids != plan.logical_arm_ids:
        raise ValueError("evaluation registry and compute plan disagree")
    return plan, plan_payload


def _validate_resource_headroom(bundle: ConfigBundle) -> None:
    runtime = bundle.payload["runtime"]
    c_free = shutil.disk_usage("C:\\").free / (1024**3)
    d_free = shutil.disk_usage("D:\\").free / (1024**3)
    ram_free = psutil.virtual_memory().available / (1024**3)
    if c_free < float(runtime["c_drive_floor_gib"]):
        raise RuntimeError("C-drive free space is below the registered floor")
    if d_free < float(runtime["d_drive_floor_gib"]):
        raise RuntimeError("D-drive free space is below the registered floor")
    if ram_free < float(runtime["available_ram_floor_gib"]):
        raise RuntimeError("available RAM is below the registered floor")


def _cache_manifest_key(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("key"), dict):
        raise ValueError("invalid frozen token-cache manifest")
    return payload, payload["key"]


def _load_frozen_cache(bundle: ConfigBundle, record: FullSeriesRecord) -> tuple[Any, Path, str, str]:
    root = Path(bundle.payload["frozen_inputs"]["coordinate_cache_root"])
    candidates = sorted((root / record.series_id / "released").glob(f"*/{TOKEN_MANIFEST}"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one released cache for {record.series_id}")
    manifest_path = candidates[0]
    manifest, key = _cache_manifest_key(manifest_path)
    expected_key = make_cache_key(
        series_id=record.series_id,
        data_sha256=record.expected_sha256,
        renderer="released",
        renderer_sha256=str(key["renderer_sha256"]),
        vendor_commit=str(bundle.payload["vendor"]["commit"]),
        clip_weight_sha256=str(bundle.payload["vendor"]["default_model_sha256"]),
        model_name="ViT-B-16",
        patch_size=16,
        image_size=(224, 224),
    )
    cache_dir = manifest_path.parent
    cache = load_clip_cache(cache_dir, expected_key)
    token_path = cache_dir / TOKEN_FILE
    token_sha = sha256_file(token_path)
    manifest_sha = sha256_file(manifest_path)
    if token_sha != str(manifest.get("sha256", "")).upper():
        raise ValueError("frozen token-cache SHA256 changed")
    if cache.patch_tokens.shape[0] != record.expected_windows:
        raise ValueError("frozen token-cache window count changed")
    return cache, cache_dir, token_sha, manifest_sha


def _load_trace(
    bundle: ConfigBundle,
    record: FullSeriesRecord,
    token_sha: str,
    cache_manifest_sha: str,
) -> TraceOperators:
    root = Path(bundle.payload["frozen_inputs"]["vittrace_run_root"]).parent / "traces"
    # The historical run_root is .../vittrace/runs; traces is its sibling.
    trace_path = root / record.series_id / "ritp_operators.npz"
    manifest_path = root / record.series_id / "ritp_operators.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_sha = sha256_file(trace_path)
    manifest_sha = sha256_file(manifest_path)
    if (
        str(manifest.get("trace_sha256", "")).upper() != trace_sha
        or str(manifest.get("cache_sha256", "")).upper() != token_sha
        or str(manifest.get("cache_manifest_sha256", "")).upper()
        != cache_manifest_sha
        or str(manifest.get("data_sha256", "")).upper()
        != record.expected_sha256.upper()
        or str(manifest.get("full_manifest_sha256", "")).upper()
        != bundle.manifest_sha256
        or int(manifest.get("window_count", -1)) != record.expected_windows
        or int(manifest.get("encoder_calls", -1)) != 0
    ):
        raise ValueError("frozen renderer-trace provenance changed")
    with np.load(trace_path, allow_pickle=False) as archive:
        required = {
            "window_starts",
            "soft_data",
            "soft_indices",
            "soft_indptr",
            "soft_offsets",
            "hard_winner",
            "full_column_data",
            "full_column_indices",
            "full_column_indptr",
        }
        if set(archive.files) != required:
            raise ValueError("frozen trace archive fields changed")
        values = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    starts = values["window_starts"]
    expected_starts = np.arange(record.expected_windows, dtype=np.int64) * 60
    if starts.dtype != np.int64 or not np.array_equal(starts, expected_starts):
        raise ValueError("frozen trace window starts changed")
    if (
        values["soft_indptr"].shape != (record.expected_windows, 241)
        or values["soft_offsets"].shape != (record.expected_windows + 1,)
        or values["hard_winner"].shape != (record.expected_windows, 240)
    ):
        raise ValueError("frozen trace operator shapes changed")
    return TraceOperators(
        trace_path,
        manifest_path,
        trace_sha,
        manifest_sha,
        starts,
        values["soft_data"].astype(np.float64, copy=False),
        values["soft_indices"].astype(np.int64, copy=False),
        values["soft_indptr"].astype(np.int64, copy=False),
        values["soft_offsets"].astype(np.int64, copy=False),
        values["hard_winner"].astype(np.int64, copy=False),
    )


def _legacy_fuse(
    fields: Mapping[str, torch.Tensor],
    valid: Mapping[str, torch.Tensor],
    scales: Sequence[str],
) -> torch.Tensor:
    selected = tuple(scales)
    fused = sum((fields[name].to(torch.float64) for name in selected)) / float(
        len(selected)
    )
    intersection = torch.ones(fused.shape[-1], dtype=torch.bool, device=fused.device)
    for name in selected:
        intersection &= valid[name]
    return torch.where(intersection.unsqueeze(0), fused, torch.zeros_like(fused))


def _legacy_timestamp_score(
    patch_fields: torch.Tensor,
    record: FullSeriesRecord,
    reducer_kind: str,
    reducer_value: float,
    *,
    interpolation_chunk: int = 64,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    values = patch_fields.reshape(-1, 1, 14, 14)
    for start in range(0, values.shape[0], interpolation_chunk):
        maps = (
            F.interpolate(
                values[start : start + interpolation_chunk],
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
        )
        if reducer_kind == "quantile":
            vectors.append(column_quantile(maps, reducer_value))
        elif reducer_kind == "top_fraction":
            vectors.append(column_top_fraction_mean(maps, reducer_value))
        else:  # pragma: no cover - registry validation owns this branch
            raise ValueError("unknown legacy reducer")
    return np.ascontiguousarray(
        stitch_column_vectors(
            np.concatenate(vectors, axis=0),
            record.expected_length,
            240,
            60,
        ),
        dtype=np.float64,
    )


def _apply_soft_trace(trace: TraceOperators, fields: np.ndarray) -> np.ndarray:
    output = np.zeros((fields.shape[0], 240), dtype=np.float64)
    for window in range(fields.shape[0]):
        offset = int(trace.soft_offsets[window])
        indptr = trace.soft_indptr[window]
        for timestamp in range(240):
            left = offset + int(indptr[timestamp])
            right = offset + int(indptr[timestamp + 1])
            indices = trace.soft_indices[left:right]
            output[window, timestamp] = np.dot(
                trace.soft_data[left:right], fields[window, indices]
            )
    return output




def _validate_scope_masks(
    tokens: torch.Tensor,
    candidates: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if tokens.ndim != 3 or min(tokens.shape) <= 0:
        raise ValueError("tokens must have nonempty shape [N,K,D]")
    if not tokens.is_floating_point() or not bool(torch.isfinite(tokens).all()):
        raise ValueError("tokens must be finite floating point")
    if not candidates:
        raise ValueError("at least one matching scope is required")
    cells = int(tokens.shape[1])
    validated: dict[str, torch.Tensor] = {}
    for scope, mask in candidates.items():
        if not isinstance(scope, str) or not scope:
            raise ValueError("matching scope names must be nonempty strings")
        if not isinstance(mask, torch.Tensor):
            raise TypeError("candidate masks must be torch tensors")
        if mask.dtype != torch.bool or mask.shape != (cells, cells):
            raise ValueError("candidate masks must be boolean [K,K] tensors")
        value = mask.to(device=tokens.device)
        if not bool(value.any(dim=1).all()):
            raise ValueError("every query cell needs at least one candidate")
        validated[scope] = value
    return validated


def _multi_scope_median_reference_cost(
    tokens: torch.Tensor,
    candidates: Mapping[str, torch.Tensor],
    *,
    query_chunk_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Compute all candidate scopes from one normalized median GEMM stream."""

    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    masks = _validate_scope_masks(tokens, candidates)
    query = F.normalize(tokens, dim=-1)
    memory = F.normalize(torch.median(tokens, dim=0).values, dim=-1)
    parts: dict[str, list[torch.Tensor]] = {scope: [] for scope in masks}
    for start in range(0, int(tokens.shape[0]), query_chunk_size):
        stop = min(start + query_chunk_size, int(tokens.shape[0]))
        similarity = torch.matmul(query[start:stop], memory.transpose(-1, -2)).clamp(
            -1.0, 1.0
        )
        pair = 0.5 * (1.0 - similarity)
        for scope, mask in masks.items():
            selected = pair.masked_fill(~mask.unsqueeze(0), torch.inf).amin(dim=-1)
            parts[scope].append(selected.to(dtype=torch.float64))
    return {scope: torch.cat(values, dim=0) for scope, values in parts.items()}


def _multi_scope_all_pairs_median_cost(
    tokens: torch.Tensor,
    candidates: Mapping[str, torch.Tensor],
    *,
    query_chunk_size: int = 4,
    reference_chunk_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Share each cosine block across scopes without allocating [N,N,K,K]."""

    if query_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("matching chunk sizes must be positive")
    masks = _validate_scope_masks(tokens, candidates)
    normalized = F.normalize(tokens, dim=-1)
    windows, cells = int(tokens.shape[0]), int(tokens.shape[1])
    rank = (windows - 1) // 2
    outputs: dict[str, list[torch.Tensor]] = {scope: [] for scope in masks}
    for query_start in range(0, windows, query_chunk_size):
        query_stop = min(query_start + query_chunk_size, windows)
        query = normalized[query_start:query_stop]
        window_parts: dict[str, list[torch.Tensor]] = {
            scope: [] for scope in masks
        }
        for reference_start in range(0, windows, reference_chunk_size):
            reference_stop = min(reference_start + reference_chunk_size, windows)
            reference = normalized[reference_start:reference_stop]
            similarity = torch.einsum("bqd,rpd->brqp", query, reference).clamp(
                -1.0, 1.0
            )
            pair = 0.5 * (1.0 - similarity)
            for scope, mask in masks.items():
                selected = pair.masked_fill(
                    ~mask.reshape(1, 1, cells, cells), torch.inf
                ).amin(dim=-1)
                window_parts[scope].append(selected)
        for scope, chunks in window_parts.items():
            per_window = torch.cat(chunks, dim=1)
            order = torch.argsort(per_window, dim=1, stable=True)
            ranked_window = order[:, rank, :]
            selected = torch.gather(
                per_window, 1, ranked_window.unsqueeze(1)
            ).squeeze(1)
            outputs[scope].append(selected.to(dtype=torch.float64))
    return {scope: torch.cat(values, dim=0) for scope, values in outputs.items()}

class CacheOnlyScorer:
    """One-series lazy scorer that shares every expensive cache computation."""

    def __init__(
        self,
        cache: Any,
        trace: TraceOperators,
        record: FullSeriesRecord,
        device: torch.device,
    ) -> None:
        self.cache = cache
        self.trace = trace
        self.record = record
        self.device = device
        self.match_fields: dict[tuple[str, str, str], torch.Tensor] = {}
        self.projected: dict[tuple[str, str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.shared_matching_seconds = 0.0

    def prepare_matching(self) -> None:
        started = time.perf_counter()
        for scale, grid in GRID_BY_SCALE.items():
            tokens = torch.from_numpy(getattr(self.cache, TOKEN_FIELD_BY_SCALE[scale])).to(
                self.device
            )
            candidates = {
                scope: build_candidate_mask(grid, scope, device=self.device)
                for scope in ("position", "row", "column", "global")
            }
            median_costs = _multi_scope_median_reference_cost(
                tokens,
                candidates,
                query_chunk_size=32,
            )
            all_pairs_costs = _multi_scope_all_pairs_median_cost(
                tokens,
                candidates,
                query_chunk_size=4,
                reference_chunk_size=8,
            )
            for scope in candidates:
                self.match_fields[(scale, scope, "median_reference")] = (
                    median_costs[scope].cpu()
                )
                self.match_fields[(scale, scope, "all_pairs")] = (
                    all_pairs_costs[scope].cpu()
                )
            del tokens
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.shared_matching_seconds = time.perf_counter() - started

    def _projected_fields(
        self,
        arm: CacheOnlyArm,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        fields: dict[str, torch.Tensor] = {}
        valid: dict[str, torch.Tensor] = {}
        for scale in ("P", "M", "L"):
            match = self.match_fields[(scale, arm.matching_scope, arm.memory)]
            if scale == "P":
                fields[scale] = match
                valid[scale] = torch.ones(196, dtype=torch.bool)
                continue
            key = (scale, arm.matching_scope, arm.memory, arm.incidence)
            if key not in self.projected:
                mask = torch.from_numpy(getattr(self.cache, MASK_FIELD_BY_SCALE[scale]))
                incidence = (
                    literal_incidence(mask, (14, 14))
                    if arm.incidence == "literal"
                    else released_incidence(mask, (14, 14))
                )
                self.projected[key] = harmonic_incidence_projection(match, incidence)
            fields[scale], valid[scale] = self.projected[key]
        return fields, valid

    def score(self, arm: CacheOnlyArm) -> np.ndarray:
        fields, valid = self._projected_fields(arm)
        if arm.fusion == "active_valid":
            fused = fuse_scale_subset(fields, arm.scales, valid_masks=valid)
        else:
            fused = _legacy_fuse(fields, valid, arm.scales)
        if arm.temporal == "legacy":
            return _legacy_timestamp_score(
                fused,
                self.record,
                str(arm.reducer_kind),
                float(arm.reducer_value),
            )
        patch = np.ascontiguousarray(fused.numpy(), dtype=np.float64)
        if arm.temporal == "nctp_linear":
            local = apply_temporal_operator(build_linear_nctp(240, (14, 14)), patch)
        elif arm.temporal == "nctp_nearest":
            local = apply_temporal_operator(
                build_nearest_full_column_operator(240, (14, 14)), patch
            )
        elif arm.temporal == "trace_soft":
            local = _apply_soft_trace(self.trace, patch)
        elif arm.temporal == "trace_hard":
            local = np.take_along_axis(patch, self.trace.hard_winner, axis=1)
        else:  # pragma: no cover - registry validation owns this branch
            raise ValueError(f"unknown temporal mode: {arm.temporal}")
        return np.ascontiguousarray(
            stitch_native_240(local, self.trace.window_starts, self.record.expected_length),
            dtype=np.float64,
        )


def _series_root(bundle: ConfigBundle, series_id: str) -> Path:
    return Path(bundle.payload["paths"]["run_root"]) / RUN_NAME / series_id


def _failure_path(bundle: ConfigBundle, series_id: str) -> Path:
    return Path(bundle.payload["paths"]["failure_root"]) / RUN_NAME / f"{series_id}.json"


def _verify_complete_series(
    root: Path,
    record: FullSeriesRecord,
    plan: CacheOnlyPlan,
    provenance: Mapping[str, Any],
) -> None:
    success = json.loads((root / "_SUCCESS.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version": RUN_SCHEMA_VERSION,
        "series_id": record.series_id,
        "logical_arm_count": len(plan.arms),
        "unique_computation_count": len(plan.canonical_arms),
        "encoder_calls": 0,
        "config_sha256": provenance["config_sha256"],
        "compute_plan_sha256": provenance["compute_plan_sha256"],
        "source_sha256": provenance["source_sha256"],
    }
    if success != expected:
        raise ValueError("completed cache-only series marker is stale")
    by_id = plan.by_id()
    for arm_id, planned in by_id.items():
        arm_root = root / arm_id
        if planned.is_alias:
            alias = json.loads((arm_root / "alias_manifest.json").read_text(encoding="utf-8"))
            alias_success = json.loads(
                (arm_root / "_SUCCESS.json").read_text(encoding="utf-8")
            )
            if (
                alias.get("series_id") != record.series_id
                or alias.get("arm") != arm_id
                or alias.get("canonical_arm") != planned.canonical_arm
                or alias.get("parameter_sha256") != planned.parameter_sha256
                or alias.get("config_sha256") != provenance["config_sha256"]
                or alias.get("compute_plan_sha256")
                != provenance["compute_plan_sha256"]
                or alias.get("source_sha256") != provenance["source_sha256"]
                or alias.get("canonical_score_path")
                != os.path.relpath(root / planned.canonical_arm / "score.npy", arm_root)
                or alias.get("canonical_score_sha256")
                != sha256_file(root / planned.canonical_arm / "score.npy")
                or int(alias.get("encoder_calls", -1)) != 0
                or (arm_root / "score.npy").exists()
            ):
                raise ValueError("completed alias manifest is stale")
            if alias_success != {
                "series_id": record.series_id,
                "arm": arm_id,
                "canonical_arm": planned.canonical_arm,
                "encoder_calls": 0,
            }:
                raise ValueError("completed alias success marker is stale")
            continue
        score_path = arm_root / "score.npy"
        manifest = json.loads((arm_root / "score_manifest.json").read_text(encoding="utf-8"))
        score_sha = sha256_file(score_path)
        ready = json.loads(
            (arm_root / "_SCORES_READY.json").read_text(encoding="utf-8")
        )
        arm_success = json.loads(
            (arm_root / "_SUCCESS.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("series_id") != record.series_id
            or manifest.get("arm") != arm_id
            or manifest.get("score_sha256") != score_sha
            or manifest.get("parameter_sha256") != planned.parameter_sha256
            or manifest.get("config_sha256") != provenance["config_sha256"]
            or manifest.get("compute_plan_sha256")
            != provenance["compute_plan_sha256"]
            or manifest.get("source_sha256") != provenance["source_sha256"]
            or int(manifest.get("encoder_calls", -1)) != 0
        ):
            raise ValueError("completed canonical score is stale")
        if ready != {
            "series_id": record.series_id,
            "arm": arm_id,
            "score_sha256": score_sha,
            "config_sha256": provenance["config_sha256"],
            "compute_plan_sha256": provenance["compute_plan_sha256"],
            "source_sha256": provenance["source_sha256"],
            "encoder_calls": 0,
        }:
            raise ValueError("completed canonical ready marker is stale")
        if arm_success != {
            "series_id": record.series_id,
            "arm": arm_id,
            "score_sha256": score_sha,
            "encoder_calls": 0,
        }:
            raise ValueError("completed canonical success marker is stale")
        values = np.load(score_path, allow_pickle=False)
        if (
            values.shape != (record.expected_length,)
            or values.dtype != np.float64
            or not np.isfinite(values).all()
        ):
            raise ValueError("completed canonical score payload is invalid")

    runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    required_runtime = {
        "series_id",
        "shared_matching_seconds",
        "canonical_arm_seconds",
        "series_wall_seconds",
        "python_tracemalloc_current_bytes",
        "python_tracemalloc_peak_bytes",
        "process_rss_before_bytes",
        "process_rss_after_bytes",
        "encoder_calls",
    }
    if not required_runtime.issubset(runtime) or runtime.get("series_id") != record.series_id:
        raise ValueError("completed runtime record is incomplete")
    if int(runtime.get("encoder_calls", -1)) != 0:
        raise ValueError("completed runtime record contains an encoder call")
    timings = runtime.get("canonical_arm_seconds")
    if not isinstance(timings, dict) or set(timings) != {
        item.logical.arm_id for item in plan.canonical_arms
    }:
        raise ValueError("completed runtime record has stale arm timings")
    numeric = [
        runtime["shared_matching_seconds"],
        runtime["series_wall_seconds"],
        *timings.values(),
    ]
    if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in numeric):
        raise ValueError("completed runtime record contains an invalid timing")


def _transaction_state(
    root: Path,
    record: FullSeriesRecord,
    plan: CacheOnlyPlan,
    provenance: Mapping[str, Any],
) -> str:
    if (root / "_FAILED.json").is_file():
        raise RuntimeError(f"retained failure blocks rerun: {record.series_id}")
    if (root / "_RUNNING.json").is_file():
        raise RuntimeError(f"active or stale transaction blocks rerun: {record.series_id}")
    if (root / "_SUCCESS.json").is_file():
        _verify_complete_series(root, record, plan, provenance)
        return "complete"
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"partial transaction blocks rerun: {record.series_id}")
    return "pending"


def _commit_canonical(
    root: Path,
    record: FullSeriesRecord,
    arm: CacheOnlyArm,
    parameter_sha: str,
    score: np.ndarray,
    provenance: Mapping[str, Any],
    arm_seconds: float,
) -> None:
    arm_root = root / arm.arm_id
    score_path = arm_root / "score.npy"
    score_sha = _atomic_npy(score_path, score)
    manifest = {
        **dict(provenance),
        "schema_version": RUN_SCHEMA_VERSION,
        "series_id": record.series_id,
        "dataset": record.dataset,
        "track": record.track,
        "paper_group": record.paper_group,
        "signal_name": record.signal_name,
        "arm": arm.arm_id,
        "parameter_sha256": parameter_sha,
        "parameters": arm.parameters(),
        "data_sha256": record.expected_sha256.upper(),
        "score_sha256": score_sha,
        "score_length": record.expected_length,
        "score_dtype": "float64",
        "arm_wall_seconds": float(arm_seconds),
        "encoder_calls": 0,
    }
    _atomic_json(arm_root / "score_manifest.json", manifest)
    _atomic_json(
        arm_root / "_SCORES_READY.json",
        {
            "series_id": record.series_id,
            "arm": arm.arm_id,
            "score_sha256": score_sha,
            "config_sha256": provenance["config_sha256"],
            "compute_plan_sha256": provenance["compute_plan_sha256"],
            "source_sha256": provenance["source_sha256"],
            "encoder_calls": 0,
        },
    )
    _atomic_json(
        arm_root / "_SUCCESS.json",
        {
            "series_id": record.series_id,
            "arm": arm.arm_id,
            "score_sha256": score_sha,
            "encoder_calls": 0,
        },
    )


def _commit_aliases(
    root: Path,
    record: FullSeriesRecord,
    plan: CacheOnlyPlan,
    provenance: Mapping[str, Any],
) -> None:
    by_id = plan.by_id()
    for arm_id, planned in by_id.items():
        if not planned.is_alias:
            continue
        canonical_score = root / planned.canonical_arm / "score.npy"
        alias_root = root / arm_id
        _atomic_json(
            alias_root / "alias_manifest.json",
            {
                **dict(provenance),
                "schema_version": RUN_SCHEMA_VERSION,
                "series_id": record.series_id,
                "arm": arm_id,
                "canonical_arm": planned.canonical_arm,
                "canonical_score_path": os.path.relpath(canonical_score, alias_root),
                "canonical_score_sha256": sha256_file(canonical_score),
                "parameter_sha256": planned.parameter_sha256,
                "encoder_calls": 0,
            },
        )
        _atomic_json(
            alias_root / "_SUCCESS.json",
            {
                "series_id": record.series_id,
                "arm": arm_id,
                "canonical_arm": planned.canonical_arm,
                "encoder_calls": 0,
            },
        )


def run_series(
    bundle: ConfigBundle,
    plan: CacheOnlyPlan,
    plan_path: Path,
    gate_path: Path,
    record: FullSeriesRecord,
) -> str:
    source_sha = cache_runner_source_sha256()
    provenance = {
        "config_sha256": bundle.sha256,
        "full_manifest_sha256": bundle.manifest_sha256,
        "compute_plan_sha256": sha256_file(plan_path),
        "parity_gate_sha256": sha256_file(gate_path),
        "source_sha256": source_sha,
        "encoder_calls": 0,
    }
    root = _series_root(bundle, record.series_id)
    if _transaction_state(root, record, plan, provenance) == "complete":
        return "complete"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        root / "_RUNNING.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "series_id": record.series_id,
            "pid": os.getpid(),
            "logical_arm_count": len(plan.arms),
            "unique_computation_count": len(plan.canonical_arms),
            "encoder_calls": 0,
        },
    )
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    rss_before = int(process.memory_info().rss)
    tracemalloc.start()
    device = torch.device(bundle.payload["runtime"]["device"])
    try:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("registered CUDA device is unavailable")
            torch.cuda.set_device(device.index or 0)
            torch.cuda.reset_peak_memory_stats(device)
        _validate_resource_headroom(bundle)
        data_path = Path(bundle.payload["data"]["root"]) / record.relative_path
        if sha256_file(data_path) != record.expected_sha256.upper():
            raise ValueError("label-free scoring CSV SHA256 changed")
        cache, cache_dir, token_sha, cache_manifest_sha = _load_frozen_cache(
            bundle, record
        )
        trace = _load_trace(bundle, record, token_sha, cache_manifest_sha)
        provenance.update(
            {
                "cache_path": str(cache_dir.resolve()),
                "cache_sha256": token_sha,
                "cache_manifest_sha256": cache_manifest_sha,
                "trace_path": str(trace.path.resolve()),
                "trace_sha256": trace.sha256,
                "trace_manifest_sha256": trace.manifest_sha256,
            }
        )
        scorer = CacheOnlyScorer(cache, trace, record, device)
        scorer.prepare_matching()
        arm_timings: dict[str, float] = {}
        for planned in plan.canonical_arms:
            arm_started = time.perf_counter()
            score = scorer.score(planned.logical)
            seconds = time.perf_counter() - arm_started
            _commit_canonical(
                root,
                record,
                planned.logical,
                planned.parameter_sha256,
                score,
                provenance,
                seconds,
            )
            arm_timings[planned.logical.arm_id] = seconds
        _commit_aliases(root, record, plan, provenance)
        if (
            bundle.sha256 != _sha256_bytes(bundle.path.read_bytes())
            or bundle.manifest_sha256 != sha256_file(bundle.manifest_path)
            or source_sha != cache_runner_source_sha256()
            or token_sha != sha256_file(cache_dir / TOKEN_FILE)
            or cache_manifest_sha != sha256_file(cache_dir / TOKEN_MANIFEST)
            or trace.sha256 != sha256_file(trace.path)
            or provenance["parity_gate_sha256"] != sha256_file(gate_path)
            or provenance["compute_plan_sha256"] != sha256_file(plan_path)
        ):
            raise RuntimeError("frozen input, plan, gate, or scorer changed during scoring")
        current, peak_python = tracemalloc.get_traced_memory()
        rss_after = int(process.memory_info().rss)
        runtime = {
            "series_id": record.series_id,
            "shared_matching_seconds": float(scorer.shared_matching_seconds),
            "canonical_arm_seconds": arm_timings,
            "series_wall_seconds": float(time.perf_counter() - started),
            "python_tracemalloc_current_bytes": int(current),
            "python_tracemalloc_peak_bytes": int(peak_python),
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
            "encoder_calls": 0,
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            runtime.update(
                {
                    "cuda_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
        _atomic_json(root / "runtime.json", runtime)
        _atomic_json(
            root / "_SUCCESS.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "series_id": record.series_id,
                "logical_arm_count": len(plan.arms),
                "unique_computation_count": len(plan.canonical_arms),
                "encoder_calls": 0,
                "config_sha256": bundle.sha256,
                "compute_plan_sha256": provenance["compute_plan_sha256"],
                "source_sha256": source_sha,
            },
        )
        (root / "_RUNNING.json").unlink()
        return "scored"
    except Exception as error:
        failure = {
            "schema_version": RUN_SCHEMA_VERSION,
            "series_id": record.series_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "config_sha256": bundle.sha256,
            "source_sha256": source_sha,
            "encoder_calls": 0,
        }
        _atomic_json(root / "_FAILED.json", failure)
        _atomic_json(_failure_path(bundle, record.series_id), failure)
        if (root / "_RUNNING.json").exists():
            (root / "_RUNNING.json").unlink()
        raise
    finally:
        tracemalloc.stop()


def _select_records(
    records: Sequence[FullSeriesRecord],
    series_ids: Sequence[str] | None,
    max_series: int | None,
) -> tuple[FullSeriesRecord, ...]:
    selected = tuple(records)
    if series_ids:
        wanted = tuple(dict.fromkeys(str(value) for value in series_ids))
        by_id = {record.series_id: record for record in records}
        missing = [series_id for series_id in wanted if series_id not in by_id]
        if missing:
            raise KeyError(f"unknown frozen series ids: {missing}")
        selected = tuple(by_id[series_id] for series_id in wanted)
    if max_series is not None:
        if max_series <= 0:
            raise ValueError("max_series must be positive")
        selected = selected[:max_series]
    return selected


def run_cache_only(
    config_path: Path,
    registry_path: Path,
    plan_path: Path,
    parity_gate_path: Path,
    *,
    series_ids: Sequence[str] | None = None,
    max_series: int | None = None,
) -> dict[str, int]:
    bundle = _load_config(config_path)
    validate_parity_gate(parity_gate_path, bundle)
    plan, _ = _validate_plan_and_registry(plan_path, registry_path, bundle)
    counts = {"scored": 0, "complete": 0, "failed": 0}
    for record in _select_records(bundle.records, series_ids, max_series):
        try:
            state = run_series(bundle, plan, plan_path, parity_gate_path, record)
            counts[state] += 1
        except Exception:
            counts["failed"] += 1
            raise
    return counts


def status_cache_only(config_path: Path, plan_path: Path) -> dict[str, Any]:
    bundle = _load_config(config_path)
    plan, _ = load_compute_plan(plan_path)
    counts = {"complete": 0, "failed": 0, "running": 0, "partial": 0, "pending": 0}
    current: list[str] = []
    for record in bundle.records:
        root = _series_root(bundle, record.series_id)
        if (root / "_SUCCESS.json").is_file():
            counts["complete"] += 1
        elif (root / "_FAILED.json").is_file():
            counts["failed"] += 1
            current.append(record.series_id)
        elif (root / "_RUNNING.json").is_file():
            counts["running"] += 1
            current.append(record.series_id)
        elif root.exists() and any(root.iterdir()):
            counts["partial"] += 1
            current.append(record.series_id)
        else:
            counts["pending"] += 1
    return {
        "run_name": RUN_NAME,
        "total": len(bundle.records),
        "logical_arm_count": len(plan.arms),
        "unique_computation_count": len(plan.canonical_arms),
        "counts": counts,
        "current": current[:8],
        "encoder_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-registry")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--registry", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--parity-gate", type=Path, required=True)
    run.add_argument("--series-id", action="append")
    run.add_argument("--max-series", type=int)
    status = subparsers.add_parser("status")
    status.add_argument("--config", type=Path, required=True)
    status.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-registry":
        registry_path, plan_path = freeze_cache_only_registry(
            args.config, args.output_dir
        )
        print(json.dumps({"registry": str(registry_path), "plan": str(plan_path)}))
    elif args.command == "run":
        print(
            json.dumps(
                run_cache_only(
                    args.config,
                    args.registry,
                    args.plan,
                    args.parity_gate,
                    series_ids=args.series_id,
                    max_series=args.max_series,
                ),
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(status_cache_only(args.config, args.plan), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CacheOnlyScorer",
    "ConfigBundle",
    "RUN_NAME",
    "cache_runner_source_sha256",
    "run_cache_only",
    "run_series",
    "status_cache_only",
    "validate_parity_gate",
]
