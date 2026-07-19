"""Read-only 492-series parity gate for the isolated ViTTrace v3 core.

The command reconstructs ``REL_U``, ``IHP_LEGACY`` and
``FULL_COLUMN_240`` from the immutable released token caches, using only the
dynamic v3 scientific core.  Frozen score transactions are comparators and
are never modified.  The default operational unit is one explicit smoke
series; the complete 492-series mode additionally requires an explicit bulk
approval flag.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from measure_vit4ts.cache import (
    TOKEN_FILE,
    TOKEN_MANIFEST,
    ClipTokenCache,
    load_clip_cache,
    make_cache_key,
)
from measure_vit4ts.full_manifest import EXPECTED_SERIES, FullSeriesRecord, load_manifest

from . import core


SCHEMA_VERSION = 1
EXPECTED_STAGE = "vittrace_ablation_full_v3"
PARITY_ARMS = ("REL_U", "IHP_LEGACY", "FULL_COLUMN_240")
DEFAULT_CONFIG = Path("configs/vittrace_ablation_full_v3.yaml")
_SOURCE_FILES = ("core.py", "parity.py")
_REFERENCE_LAYOUT = {
    "REL_U": ("coordinate_run_root", "coordinate_envelope_full", "REL_U"),
    "IHP_LEGACY": ("coordinate_run_root", "coordinate_envelope_full", "IDX_U"),
    "FULL_COLUMN_240": ("vittrace_run_root", "vittrace_full", "FULL_COLUMN_240"),
}


@dataclass(frozen=True)
class CacheArtifact:
    cache: ClipTokenCache
    cache_path: Path
    manifest_path: Path
    cache_sha256: str
    manifest_sha256: str
    cache_size: int
    cache_mtime_ns: int


@dataclass(frozen=True)
class FrozenReference:
    parity_arm: str
    source_arm: str
    score_path: Path
    manifest_path: Path
    score_sha256: str
    values: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def parity_source_sha256(package_root: Path | None = None) -> str:
    root = Path(package_root) if package_root else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()

def core_sha256(package_root: Path | None = None) -> str:
    root = Path(package_root) if package_root else Path(__file__).resolve().parent
    return _sha256(root / "core.py")



def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _load_config(
    config_path: Path,
) -> tuple[dict[str, Any], str, tuple[FullSeriesRecord, ...], str]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or str(config.get("stage")) != EXPECTED_STAGE:
        raise ValueError(f"parity requires stage={EXPECTED_STAGE}")
    contract = config.get("contracts", {})
    if tuple(contract.get("parity_arms", ())) != PARITY_ARMS:
        raise ValueError("v3 parity arm registry changed")
    if not np.isclose(float(contract.get("parity_atol", np.nan)), 1e-12, rtol=0.0, atol=0.0):
        raise ValueError("v3 parity atol must remain 1e-12")
    if not np.isclose(float(contract.get("parity_rtol", np.nan)), 1e-10, rtol=0.0, atol=0.0):
        raise ValueError("v3 parity rtol must remain 1e-10")
    if int(contract.get("parity_window", -1)) != 240 or int(
        contract.get("parity_stride", -1)
    ) != 60:
        raise ValueError("v3 parity requires the frozen 240/60 interface")
    manifest_path = Path(config["manifest"]["path"])
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != str(config["manifest"]["sha256"]).upper():
        raise ValueError("v3 frozen manifest hash mismatch")
    _, records = load_manifest(manifest_path)
    expected = int(config["manifest"]["expected_series"])
    if expected != EXPECTED_SERIES or len(records) != expected:
        raise ValueError("v3 parity requires exactly 492 manifest series")
    if len({record.series_id for record in records}) != expected:
        raise ValueError("v3 manifest series IDs are not unique")
    return config, _sha256(path), records, manifest_sha


def _cache_key_from_manifest(
    payload: Mapping[str, Any], config: Mapping[str, Any], record: FullSeriesRecord
):
    key = payload.get("key")
    if not isinstance(key, Mapping):
        raise ValueError("released cache manifest is missing its key")
    return make_cache_key(
        series_id=record.series_id,
        data_sha256=record.expected_sha256,
        renderer="released",
        renderer_sha256=str(key.get("renderer_sha256", "")),
        vendor_commit=str(config["vendor"]["commit"]),
        clip_weight_sha256=str(config["vendor"]["default_model_sha256"]),
        model_name=str(config["defaults"]["model_name"]),
        patch_size=int(config["defaults"]["patch_size"]),
        image_size=tuple(int(value) for value in config["defaults"]["image_size"]),
    )


def _load_cache(
    config: Mapping[str, Any], record: FullSeriesRecord
) -> CacheArtifact:
    root = (
        Path(config["frozen_inputs"]["coordinate_cache_root"])
        / record.series_id
        / "released"
    )
    candidates: list[tuple[Path, Mapping[str, Any]]] = []
    for manifest_path in sorted(root.glob(f"*/{TOKEN_MANIFEST}")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = payload.get("key", {})
        if (
            key.get("series_id") == record.series_id
            and str(key.get("data_sha256", "")).upper()
            == record.expected_sha256.upper()
            and key.get("renderer") == "released"
            and str(key.get("vendor_commit", "")).lower()
            == str(config["vendor"]["commit"]).lower()
            and str(key.get("clip_weight_sha256", "")).upper()
            == str(config["vendor"]["default_model_sha256"]).upper()
            and str(key.get("model_name", "")) == str(config["defaults"]["model_name"])
            and int(key.get("patch_size", -1)) == int(config["defaults"]["patch_size"])
        ):
            candidates.append((manifest_path, payload))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one immutable released cache for {record.series_id}, found {len(candidates)}"
        )
    manifest_path, payload = candidates[0]
    expected_key = _cache_key_from_manifest(payload, config, record)
    cache = load_clip_cache(manifest_path.parent, expected_key)
    cache_path = manifest_path.parent / TOKEN_FILE
    cache_sha = _sha256(cache_path)
    manifest_sha = _sha256(manifest_path)
    if cache_sha != str(payload.get("sha256", "")).upper():
        raise ValueError("released cache content hash differs from its manifest")
    if int(cache.patch_tokens.shape[0]) != int(record.expected_windows):
        raise ValueError("released cache window count differs from the frozen manifest")
    stat = cache_path.stat()
    return CacheArtifact(
        cache,
        cache_path,
        manifest_path,
        cache_sha,
        manifest_sha,
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _load_reference(
    config: Mapping[str, Any],
    record: FullSeriesRecord,
    parity_arm: str,
    cache: CacheArtifact,
    manifest_sha: str,
) -> FrozenReference:
    if parity_arm not in _REFERENCE_LAYOUT:
        raise ValueError(f"unsupported parity arm {parity_arm}")
    root_key, stage, source_arm = _REFERENCE_LAYOUT[parity_arm]
    root = (
        Path(config["frozen_inputs"][root_key])
        / stage
        / record.series_id
        / source_arm
    )
    score_path = root / "score.npy"
    transaction_path = root / "score_manifest.json"
    if not score_path.is_file() or not transaction_path.is_file():
        raise FileNotFoundError(f"frozen comparator transaction is incomplete: {root}")
    payload = json.loads(transaction_path.read_text(encoding="utf-8"))
    score_sha = _sha256(score_path)
    if (
        payload.get("series_id") != record.series_id
        or payload.get("arm") != source_arm
        or str(payload.get("data_sha256", "")).upper()
        != record.expected_sha256.upper()
        or str(payload.get("full_manifest_sha256", "")).upper() != manifest_sha
        or str(payload.get("score_sha256", "")).upper() != score_sha
    ):
        raise ValueError(f"stale frozen comparator for {record.series_id}/{parity_arm}")
    if source_arm in {"REL_U", "IDX_U"}:
        cache_sha = str(payload.get("released_token_sha256", "")).upper()
        cache_manifest_sha = str(payload.get("released_manifest_sha256", "")).upper()
    else:
        cache_sha = str(payload.get("cache_sha256", "")).upper()
        cache_manifest_sha = str(payload.get("cache_manifest_sha256", "")).upper()
    if cache_sha != cache.cache_sha256 or cache_manifest_sha != cache.manifest_sha256:
        raise ValueError(f"comparator cache identity mismatch for {parity_arm}")
    values = np.load(score_path, allow_pickle=False)
    if (
        values.shape != (record.expected_length,)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
        or int(payload.get("score_length", -1)) != record.expected_length
        or str(payload.get("score_dtype", "")) != "float64"
    ):
        raise ValueError(f"invalid frozen comparator score for {parity_arm}")
    return FrozenReference(
        parity_arm,
        source_arm,
        score_path,
        transaction_path,
        score_sha,
        np.ascontiguousarray(values),
    )


def _global_cost(tokens: np.ndarray, device: torch.device, chunk_size: int) -> torch.Tensor:
    values = torch.from_numpy(tokens).to(device)
    side = int(round(np.sqrt(values.shape[1])))
    if side * side != int(values.shape[1]):
        raise ValueError("parity supports square token grids only")
    allowed = core.build_candidate_mask((side, side), "global", device=device)
    return core.streamed_median_reference_match(
        # The frozen scorer forms one complete [N,K,K] float32 matmul.  A
        # smaller N chunk can select a different CUDA GEMM kernel (~1e-7).
        values, allowed, query_chunk_size=int(values.shape[0])
    ).cost


def _fuse_legacy_contract(
    large_cost: torch.Tensor,
    mid_cost: torch.Tensor,
    patch_cost: torch.Tensor,
    large_mask: np.ndarray,
    mid_mask: np.ndarray,
    *,
    literal: bool,
) -> torch.Tensor:
    grid = (14, 14)
    large_membership = torch.from_numpy(large_mask).to(large_cost.device)
    mid_membership = torch.from_numpy(mid_mask).to(mid_cost.device)
    incidence = core.literal_incidence if literal else core.released_incidence
    large_incidence = incidence(large_membership, grid)
    mid_incidence = incidence(mid_membership, grid)
    large_field, large_valid = core.harmonic_incidence_projection(
        large_cost, large_incidence
    )
    mid_field, mid_valid = core.harmonic_incidence_projection(mid_cost, mid_incidence)
    if patch_cost.shape != large_field.shape or patch_cost.shape != mid_field.shape:
        raise ValueError("multiscale parity fields do not share [N,196]")
    fused = (large_field + mid_field + patch_cost.to(dtype=torch.float64)) / 3.0
    valid = (large_valid & mid_valid).reshape(1, -1)
    return torch.where(valid, fused, torch.zeros_like(fused))


def _legacy_stitch(
    vectors: np.ndarray, full_length: int, window_size: int, step_size: int
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("legacy vectors must be finite [N,W]")
    if window_size <= 0 or step_size <= 0 or window_size % step_size:
        raise ValueError("legacy window/stride contract is invalid")
    ratio = window_size // step_size
    raster_width = int(values.shape[1])
    raster_step = raster_width // ratio
    raster_length = raster_width + (values.shape[0] - 1) * raster_step
    summed = np.zeros(raster_length, dtype=np.float64)
    count = np.zeros(raster_length, dtype=np.float64)
    for index, vector in enumerate(values):
        start = index * raster_step
        summed[start : start + raster_width] += vector
        count[start : start + raster_width] += 1.0
    stitched = summed / np.maximum(count, 1.0)
    covered = window_size + (values.shape[0] - 1) * step_size
    temporal = np.interp(
        np.linspace(0.0, raster_length - 1, covered),
        np.arange(raster_length, dtype=np.float64),
        stitched,
    )
    if covered < full_length:
        slope = temporal[-1] - temporal[-2] if covered > 1 else 0.0
        temporal = np.concatenate(
            (
                temporal,
                temporal[-1]
                + slope * np.arange(1, full_length - covered + 1, dtype=np.float64),
            )
        )
    return np.ascontiguousarray(temporal[:full_length], dtype=np.float64)


def _native_stitch(
    window_scores: np.ndarray, starts: np.ndarray, full_length: int
) -> np.ndarray:
    scores = np.asarray(window_scores, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[0] == 0 or starts.shape != (scores.shape[0],):
        raise ValueError("native parity scores/starts are misaligned")
    summed = np.zeros(int(full_length), dtype=np.float64)
    count = np.zeros(int(full_length), dtype=np.int64)
    width = int(scores.shape[1])
    for index, start in enumerate(starts):
        stop = int(start) + width
        if start < 0 or stop > full_length:
            raise ValueError("native parity window lies outside the series")
        summed[int(start) : stop] += scores[index]
        count[int(start) : stop] += 1
    supported = count > 0
    if not supported[0]:
        raise ValueError("native parity stitch has an uncovered prefix")
    last = int(np.flatnonzero(supported)[-1])
    if np.any(~supported[: last + 1]):
        raise ValueError("native parity stitch has an interior hole")
    output = np.empty(int(full_length), dtype=np.float64)
    output[: last + 1] = summed[: last + 1] / count[: last + 1]
    if last + 1 < full_length:
        slope = output[last] - output[last - 1] if last > 0 else 0.0
        output[last + 1 :] = output[last] + slope * np.arange(
            1, full_length - last, dtype=np.float64
        )
    return np.ascontiguousarray(output)


def _legacy_score(
    field: torch.Tensor, full_length: int, window: int, stride: int
) -> np.ndarray:
    maps = (
        F.interpolate(
            field.reshape(-1, 1, 14, 14),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(1)
        .detach()
        .cpu()
        .numpy()
    )
    columns = core.column_top_fraction_mean(maps, 0.25)
    return _legacy_stitch(columns, full_length, window, stride)


def _full_column_operator(
    window: int, patch_grid: tuple[int, int], image_size: tuple[int, int]
) -> np.ndarray:
    # The corrected API integrates continuous pixel-x split mass into patch
    # cells.  A naive direct interpolation over grid columns is intentionally
    # rejected because it is not the frozen FULL_COLUMN_240 control.
    function = core.build_linear_nctp
    parameters = inspect.signature(function).parameters
    if "image_size" not in parameters:
        alternative = getattr(core, "build_pixel_linear_nctp", None)
        if alternative is None:
            raise RuntimeError(
                "dynamic core lacks corrected pixel-to-patch full-column NCTP"
            )
        return alternative(window, patch_grid, image_size=image_size)
    return function(window, patch_grid, image_size=image_size)


def reconstruct_parity_arms(
    cache: ClipTokenCache,
    record: FullSeriesRecord,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Reconstruct the three frozen parity arms without labels or CLIP calls."""

    window = int(config["contracts"]["parity_window"])
    stride = int(config["contracts"]["parity_stride"])
    starts = np.arange(int(record.expected_windows), dtype=np.int64) * stride
    if int(starts[-1]) + window > int(record.expected_length):
        raise ValueError("derived parity windows exceed frozen series length")
    chunk = int(config["runtime"]["batch_size"])
    with torch.inference_mode():
        large_cost = _global_cost(cache.large_tokens, device, chunk)
        mid_cost = _global_cost(cache.mid_tokens, device, chunk)
        patch_cost = _global_cost(cache.patch_tokens, device, chunk)
        released_field = _fuse_legacy_contract(
            large_cost,
            mid_cost,
            patch_cost,
            cache.large_mask,
            cache.mid_mask,
            literal=False,
        )
        literal_field = _fuse_legacy_contract(
            large_cost,
            mid_cost,
            patch_cost,
            cache.large_mask,
            cache.mid_mask,
            literal=True,
        )
        rel = _legacy_score(released_field, record.expected_length, window, stride)
        ihp = _legacy_score(literal_field, record.expected_length, window, stride)
        operator = _full_column_operator(
            window,
            (14, 14),
            tuple(int(value) for value in config["defaults"]["image_size"]),
        )
        local = core.apply_temporal_operator(
            operator,
            literal_field.detach().cpu().numpy(),
        )
        full_column = _native_stitch(local, starts, record.expected_length)
    return {
        "REL_U": rel,
        "IHP_LEGACY": ihp,
        "FULL_COLUMN_240": full_column,
    }


def compare_scores(
    actual: np.ndarray, reference: np.ndarray, *, atol: float, rtol: float
) -> dict[str, Any]:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("parity scores must be aligned one-dimensional arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("parity scores must be finite")
    difference = np.abs(left - right)
    tolerance = float(atol) + float(rtol) * np.abs(right)
    denominator = np.maximum(np.abs(right), float(atol))
    relative = difference / denominator
    return {
        "passed": bool(np.all(difference <= tolerance)),
        "max_abs_error": float(np.max(difference, initial=0.0)),
        "max_rel_error": float(np.max(relative, initial=0.0)),
        "mismatch_count": int(np.sum(difference > tolerance)),
        "point_count": int(left.size),
    }


def _series_record_path(config: Mapping[str, Any], series_id: str) -> Path:
    return Path(config["paths"]["output_root"]) / "parity_records" / f"{series_id}.json"


def _resume_payload(
    path: Path,
    *,
    config_sha: str,
    manifest_sha: str,
    source_sha: str,
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("config_sha256") != config_sha
        or payload.get("manifest_sha256") != manifest_sha
        or payload.get("parity_source_sha256") != source_sha
        or set(payload.get("arms", {})) != set(PARITY_ARMS)
    ):
        return None
    cache = payload.get("cache", {})
    cache_path = Path(str(cache.get("path", "")))
    manifest_path = Path(str(cache.get("manifest_path", "")))
    if not cache_path.is_file() or not manifest_path.is_file():
        return None
    stat = cache_path.stat()
    if (
        int(cache.get("size", -1)) != int(stat.st_size)
        or int(cache.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
        or str(cache.get("manifest_sha256", "")) != _sha256(manifest_path)
    ):
        return None
    for arm, result in payload["arms"].items():
        if _sha256(Path(result["source_score_path"])) != result["source_score_sha256"]:
            return None
    return payload


def _failure_path(config: Mapping[str, Any], series_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        Path(config["paths"]["failure_root"])
        / "parity"
        / series_id
        / f"{stamp}_{uuid.uuid4().hex}.json"
    )


def _run_series(
    config: Mapping[str, Any],
    config_sha: str,
    manifest_sha: str,
    source_sha: str,
    record: FullSeriesRecord,
    device: torch.device,
    *,
    retry: bool,
) -> Mapping[str, Any]:
    target = _series_record_path(config, record.series_id)
    if not retry:
        resumed = _resume_payload(
            target,
            config_sha=config_sha,
            manifest_sha=manifest_sha,
            source_sha=source_sha,
        )
        if resumed is not None:
            return resumed
    started = _utc_now()
    cache = _load_cache(config, record)
    references = {
        arm: _load_reference(config, record, arm, cache, manifest_sha)
        for arm in PARITY_ARMS
    }
    reconstructed = reconstruct_parity_arms(cache.cache, record, config, device)
    atol = float(config["contracts"]["parity_atol"])
    rtol = float(config["contracts"]["parity_rtol"])
    arms: dict[str, Any] = {}
    for arm in PARITY_ARMS:
        comparison = compare_scores(
            reconstructed[arm], references[arm].values, atol=atol, rtol=rtol
        )
        arms[arm] = {
            **comparison,
            "source_arm": references[arm].source_arm,
            "source_score_path": str(references[arm].score_path.resolve()),
            "source_manifest_path": str(references[arm].manifest_path.resolve()),
            "source_score_sha256": references[arm].score_sha256,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "series_id": record.series_id,
        "family": record.track,
        "subgroup": record.paper_group,
        "status": "PASS" if all(item["passed"] for item in arms.values()) else "FAIL",
        "passed": bool(all(item["passed"] for item in arms.values())),
        "started_at": started,
        "completed_at": _utc_now(),
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "parity_source_sha256": source_sha,
        "atol": atol,
        "rtol": rtol,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cache": {
            "path": str(cache.cache_path.resolve()),
            "manifest_path": str(cache.manifest_path.resolve()),
            "sha256": cache.cache_sha256,
            "manifest_sha256": cache.manifest_sha256,
            "size": cache.cache_size,
            "mtime_ns": cache.cache_mtime_ns,
        },
        "arms": arms,
    }
    _atomic_json(target, payload)
    return payload


def _latest_failure(config: Mapping[str, Any], series_id: str) -> Mapping[str, Any] | None:
    root = Path(config["paths"]["failure_root"]) / "parity" / series_id
    paths = sorted(root.glob("*.json")) if root.is_dir() else []
    return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None


def _collect_rows(
    config: Mapping[str, Any],
    records: Sequence[FullSeriesRecord],
    *,
    config_sha: str,
    manifest_sha: str,
    source_sha: str,
) -> tuple[pd.DataFrame, dict[str, Mapping[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for record in records:
        payload = _resume_payload(
            _series_record_path(config, record.series_id),
            config_sha=config_sha,
            manifest_sha=manifest_sha,
            source_sha=source_sha,
        )
        if payload is not None:
            payloads[record.series_id] = payload
            for arm in PARITY_ARMS:
                item = payload["arms"][arm]
                rows.append(
                    {
                        "series_id": record.series_id,
                        "family": record.track,
                        "subgroup": record.paper_group,
                        "arm": arm,
                        "status": payload["status"],
                        "passed": bool(item["passed"]),
                        "max_abs_error": float(item["max_abs_error"]),
                        "max_rel_error": float(item["max_rel_error"]),
                        "mismatch_count": int(item["mismatch_count"]),
                        "point_count": int(item["point_count"]),
                        "source_arm": item["source_arm"],
                        "source_score_path": item["source_score_path"],
                        "source_score_sha256": item["source_score_sha256"],
                    }
                )
            continue
        failure = _latest_failure(config, record.series_id)
        for arm in PARITY_ARMS:
            rows.append(
                {
                    "series_id": record.series_id,
                    "family": record.track,
                    "subgroup": record.paper_group,
                    "arm": arm,
                    "status": "ERROR" if failure else "NOT_RUN",
                    "passed": False,
                    "max_abs_error": np.nan,
                    "max_rel_error": np.nan,
                    "mismatch_count": np.nan,
                    "point_count": record.expected_length,
                    "source_arm": _REFERENCE_LAYOUT[arm][2],
                    "source_score_path": "",
                    "source_score_sha256": "",
                }
            )
    return pd.DataFrame(rows), payloads


def _gate_payload(
    config: Mapping[str, Any],
    config_sha: str,
    manifest_sha: str,
    source_sha: str,
    rows: pd.DataFrame,
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    completed = len(payloads)
    errors = int(rows.loc[rows["status"] == "ERROR", "series_id"].nunique())
    failures = int(rows.loc[rows["status"] == "FAIL", "series_id"].nunique())
    full_pass = completed == EXPECTED_SERIES and errors == 0 and failures == 0
    decision = "PASS" if full_pass else ("FAIL" if errors or failures else "INCOMPLETE")
    arms: dict[str, Any] = {}
    for arm in PARITY_ARMS:
        selected = rows.loc[(rows["arm"] == arm) & rows["status"].isin(["PASS", "FAIL"])]
        root_key, stage, source_arm = _REFERENCE_LAYOUT[arm]
        arms[arm] = {
            "passed": bool(
                full_pass
                and len(selected) == EXPECTED_SERIES
                and selected["passed"].astype(bool).all()
            ),
            "series_evaluated": int(len(selected)),
            "max_abs_error": float(selected["max_abs_error"].max())
            if len(selected)
            else None,
            "max_rel_error": float(selected["max_rel_error"].max())
            if len(selected)
            else None,
            "comparator_root": str(Path(config["frozen_inputs"][root_key]).resolve()),
            "comparator_stage": stage,
            "comparator_arm": source_arm,
            "score_pattern": f"<comparator_root>/{stage}/<series_id>/{source_arm}/score.npy",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "passed": full_pass,
        "created_at": _utc_now(),
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "parity_source_sha256": source_sha,
        "core_sha256": core_sha256(),
        "atol": float(config["contracts"]["parity_atol"]),
        "rtol": float(config["contracts"]["parity_rtol"]),
        "expected_series": EXPECTED_SERIES,
        "completed_series": completed,
        "failed_series": failures,
        "error_series": errors,
        "not_run_series": EXPECTED_SERIES - completed - errors,
        "arms": arms,
    }


def _write_summary(
    config: Mapping[str, Any],
    config_sha: str,
    manifest_sha: str,
    source_sha: str,
    records: Sequence[FullSeriesRecord],
) -> tuple[Path, Path]:
    rows, payloads = _collect_rows(
        config,
        records,
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        source_sha=source_sha,
    )
    root = Path(config["paths"]["output_root"])
    csv_path = root / "parity_per_series.csv"
    gate_path = root / "parity_gate.json"
    gate = _gate_payload(config, config_sha, manifest_sha, source_sha, rows, payloads)
    _atomic_csv(csv_path, rows)
    _atomic_json(gate_path, gate)
    # The v3 scorer reads this path and accepts only decision=PASS/passed=true.
    _atomic_json(root / "provenance" / "cache_only_parity_gate.json", gate)
    return csv_path, gate_path


def select_records(
    records: Sequence[FullSeriesRecord],
    *,
    smoke: bool,
    all_series: bool,
    series_ids: Sequence[str],
    approved_bulk: bool,
) -> tuple[FullSeriesRecord, ...]:
    if smoke and all_series:
        raise ValueError("choose either --smoke or --all")
    by_id = {record.series_id: record for record in records}
    requested = tuple(series_ids)
    if len(set(requested)) != len(requested) or any(value not in by_id for value in requested):
        raise ValueError("series IDs must be unique registered manifest entries")
    if all_series:
        if requested:
            raise ValueError("--all cannot be combined with --series-id")
        if not approved_bulk:
            raise PermissionError("--all requires explicit --approved-bulk")
        return tuple(records)
    if smoke:
        if len(requested) > 1:
            raise ValueError("--smoke accepts at most one --series-id")
        return (by_id[requested[0]],) if requested else (records[0],)
    if not requested:
        raise ValueError("choose --smoke, explicit --series-id, or approved --all")
    return tuple(by_id[value] for value in requested)


def run_parity(
    config_path: Path,
    *,
    smoke: bool = False,
    all_series: bool = False,
    series_ids: Sequence[str] = (),
    approved_bulk: bool = False,
    device_name: str | None = None,
    retry: bool = False,
) -> tuple[Path, Path]:
    config, config_sha, records, manifest_sha = _load_config(config_path)
    selected = select_records(
        records,
        smoke=smoke,
        all_series=all_series,
        series_ids=series_ids,
        approved_bulk=approved_bulk,
    )
    device = torch.device(device_name or str(config["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA parity was requested but CUDA is unavailable")
    source_sha = parity_source_sha256()
    for record in selected:
        try:
            _run_series(
                config,
                config_sha,
                manifest_sha,
                source_sha,
                record,
                device,
                retry=retry,
            )
        except Exception as exc:
            _atomic_json(
                _failure_path(config, record.series_id),
                {
                    "schema_version": SCHEMA_VERSION,
                    "series_id": record.series_id,
                    "created_at": _utc_now(),
                    "config_sha256": config_sha,
                    "manifest_sha256": manifest_sha,
                    "parity_source_sha256": source_sha,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            _write_summary(config, config_sha, manifest_sha, source_sha, records)
            raise
    return _write_summary(config, config_sha, manifest_sha, source_sha, records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", dest="all_series", action="store_true")
    parser.add_argument("--series-id", action="append", default=[])
    parser.add_argument("--approved-bulk", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args(argv)
    paths = run_parity(
        args.config,
        smoke=args.smoke,
        all_series=args.all_series,
        series_ids=args.series_id,
        approved_bulk=args.approved_bulk,
        device_name=args.device,
        retry=args.retry,
    )
    print(" ".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
