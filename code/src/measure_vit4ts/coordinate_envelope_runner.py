"""Label-free full runner for the frozen coordinate-envelope candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import yaml

from .cache import (
    TOKEN_FILE,
    TOKEN_MANIFEST,
    ClipTokenCache,
    cache_key_digest,
    compute_renderer_sha256,
    encode_frozen_clip,
    load_clip_cache,
    make_cache_key,
    save_clip_cache,
    verify_vendor_sha,
)
from .data import SeriesData, make_windows, released_preprocess, sha256_file
from .full_manifest import (
    EXPECTED_SERIES,
    FullSeriesRecord,
    load_manifest,
)
from .reducers import released_column_mean, stitch_column_vectors
from .runner import FrozenVendor, _render_windows, load_frozen_vendor
from .topology import (
    ScaleMasks,
    ScaleScores,
    aggregate_multiscale,
    independent_match,
    pairwise_cosine_cost,
)


EXPECTED_STAGE = "coordinate_envelope_full"
COORDINATE_ARMS = ("REL_U", "IDX_U", "COORD_MAX")
_SOURCE_FILES = (
    "data.py",
    "geometry.py",
    "renderers.py",
    "cache.py",
    "reducers.py",
    "runner.py",
    "topology.py",
    "full_manifest.py",
    "coordinate_envelope_runner.py",
)


@dataclass(frozen=True)
class LoadedFullSignal:
    series: SeriesData
    duplicate_timestamps: bool


def coordinate_envelope_source_sha256(package_root: Path | None = None) -> str:
    """Hash every score-producing source file in deterministic order."""

    root = (Path(package_root) if package_root is not None else Path(__file__).parent).resolve(
        strict=True
    )
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        payload = (root / name).read_bytes()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("coordinate-envelope score must be finite float64 [T]")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path).upper()


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], str, tuple[FullSeriesRecord, ...], str]:
    raw = Path(path).read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("coordinate-envelope config must be a mapping")
    if str(config.get("stage")) != EXPECTED_STAGE:
        raise ValueError("runner accepts only the frozen full stage")
    if tuple(config["arms"][EXPECTED_STAGE]) != COORDINATE_ARMS:
        raise ValueError("coordinate-envelope arms differ from the frozen protocol")
    if str(config["arms"].get("headline")) != "COORD_MAX":
        raise ValueError("coordinate-envelope headline arm changed")
    if int(config["data"]["window_size"]) != 240 or int(
        config["data"]["step_size"]
    ) != 60:
        raise ValueError("coordinate-envelope geometry must remain 240/60")
    manifest_path = Path(config["manifest"]["path"])
    manifest_hash = sha256_file(manifest_path).upper()
    if manifest_hash != str(config["manifest"]["sha256"]).upper():
        raise ValueError("full-manifest SHA256 differs from the frozen config")
    payload, records = load_manifest(manifest_path)
    if len(records) != EXPECTED_SERIES:
        raise ValueError("coordinate-envelope full stage requires 492 series")
    index_path = Path(config["data"]["root"]) / str(payload["source_index"])
    if sha256_file(index_path).upper() != str(payload["source_index_sha256"]).upper():
        raise ValueError("datasets.csv differs from the frozen full manifest")
    return (
        config,
        hashlib.sha256(raw).hexdigest().upper(),
        records,
        manifest_hash,
    )


def _safety_check(config: Mapping[str, Any]) -> None:
    c_free = shutil.disk_usage("C:\\").free / (1024**3)
    d_free = shutil.disk_usage("D:\\").free / (1024**3)
    ram_free = psutil.virtual_memory().available / (1024**3)
    if c_free < float(config["runtime"]["c_drive_floor_gib"]):
        raise RuntimeError("C drive is below the frozen safety floor")
    if d_free < float(config["runtime"]["d_drive_floor_gib"]):
        raise RuntimeError("D drive is below the frozen safety floor")
    if ram_free < float(config["runtime"]["available_ram_floor_gib"]):
        raise RuntimeError("available RAM is below the frozen safety floor")


def load_vendor_signal(
    record: FullSeriesRecord,
    data_root: Path,
) -> LoadedFullSignal:
    """Read only timestamp/value and reproduce vendor duplicate sorting."""

    path = Path(data_root) / Path(record.relative_path)
    actual_hash = sha256_file(path).upper()
    if actual_hash != record.expected_sha256.upper():
        raise ValueError(f"data SHA256 mismatch for {record.series_id}")
    frame = pd.read_csv(path, usecols=["timestamp", "value"])
    if frame.empty or len(frame) != record.expected_length:
        raise ValueError(f"full-main length mismatch for {record.series_id}")
    # This deliberately mirrors vendor preprocessing.data_utils.orion_to_internal.
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_numeric(frame["timestamp"], errors="raise").to_numpy()
    values = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(timestamps.astype(np.float64, copy=False)).all():
        raise ValueError("full-main timestamps must be finite")
    if not np.isfinite(values).all():
        raise ValueError("full-main values must be finite")
    duplicate = bool(pd.Index(timestamps).has_duplicates)
    if duplicate != record.duplicate_timestamps:
        raise ValueError("duplicate-timestamp identity differs from the manifest")
    timestamps = np.ascontiguousarray(timestamps)
    values = np.ascontiguousarray(values, dtype=np.float64)
    timestamps.setflags(write=False)
    values.setflags(write=False)
    return LoadedFullSignal(
        SeriesData(
            series_id=record.series_id,
            group=record.track,
            timestamps=timestamps,
            values=values,
            data_sha256=actual_hash.lower(),
        ),
        duplicate,
    )


def _ensure_released_cache(
    vendor: FrozenVendor,
    series: SeriesData,
    batch: Any,
    config: Mapping[str, Any],
) -> tuple[ClipTokenCache, Path, bool, float]:
    started = time.perf_counter()
    images, geometries = _render_windows(batch, "released", config)
    renderer_hash = compute_renderer_sha256(images, geometries)
    key = make_cache_key(
        series_id=series.series_id,
        data_sha256=series.data_sha256,
        renderer="released",
        renderer_sha256=renderer_hash,
        vendor_commit=vendor.commit,
        clip_weight_sha256=vendor.clip_weight_sha256,
        model_name=str(config["vendor"]["model_name"]),
        patch_size=int(config["vision"]["patch_size"]),
        image_size=tuple(int(value) for value in config["vision"]["image_size"]),
    )
    directory = (
        Path(config["paths"]["cache_root"])
        / key.series_id
        / key.renderer
        / cache_key_digest(key)
    )
    created = not (directory / TOKEN_MANIFEST).is_file()
    if created:
        cache = encode_frozen_clip(
            vendor.model,
            images,
            key,
            int(config["vision"]["batch_size"]),
            torch.device(config["runtime"]["device"]),
        )
        actual = save_clip_cache(cache, Path(config["paths"]["cache_root"]))
        if actual != directory:
            raise RuntimeError("coordinate-envelope cache directory identity mismatch")
    else:
        cache = load_clip_cache(directory, key)
    if cache.patch_tokens.shape[0] != batch.values.shape[0]:
        raise ValueError("coordinate-envelope cache window count mismatch")
    return cache, directory, created, time.perf_counter() - started


def _universal_scale_scores(
    cache: ClipTokenCache,
    device: torch.device,
) -> ScaleScores:
    output: dict[str, torch.Tensor] = {}
    for scale, array in (
        ("large", cache.large_tokens),
        ("mid", cache.mid_tokens),
        ("patch", cache.patch_tokens),
    ):
        query = torch.from_numpy(array).to(device)
        memory = torch.median(query, dim=0).values
        cost = pairwise_cosine_cost(query, memory)
        output[scale] = independent_match(cost, "universal").cost
        del query, memory, cost
    return ScaleScores(
        large=output["large"],
        mid=output["mid"],
        patch=output["patch"],
    )


def _timestamp_score(
    scales: ScaleScores,
    masks: ScaleMasks,
    batch: Any,
    released_indexing: bool,
) -> np.ndarray:
    fused = aggregate_multiscale(
        scales,
        masks,
        released_indexing=released_indexing,
    )
    maps = (
        F.interpolate(
            fused.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(1)
        .double()
        .detach()
        .cpu()
        .numpy()
    )
    return np.ascontiguousarray(
        stitch_column_vectors(
            released_column_mean(maps),
            batch.full_length,
            batch.window_size,
            batch.step_size,
        ),
        dtype=np.float64,
    )


def build_coordinate_scores(
    cache: ClipTokenCache,
    batch: Any,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Compute the two coordinate charts and their exact upper envelope."""

    scales = _universal_scale_scores(cache, device)
    masks = ScaleMasks(
        large=torch.from_numpy(cache.large_mask).to(device),
        mid=torch.from_numpy(cache.mid_mask).to(device),
    )
    rel_u = _timestamp_score(scales, masks, batch, released_indexing=True)
    idx_u = _timestamp_score(scales, masks, batch, released_indexing=False)
    coordinate_max = np.ascontiguousarray(np.maximum(rel_u, idx_u), dtype=np.float64)
    if not np.array_equal(coordinate_max, np.maximum(rel_u, idx_u)):
        raise RuntimeError("coordinate envelope is not the exact pointwise maximum")
    return {"REL_U": rel_u, "IDX_U": idx_u, "COORD_MAX": coordinate_max}


def _coordinate_diagnostics(scores: Mapping[str, np.ndarray]) -> dict[str, float]:
    rel_u = np.asarray(scores["REL_U"], dtype=np.float64)
    idx_u = np.asarray(scores["IDX_U"], dtype=np.float64)
    maximum = np.asarray(scores["COORD_MAX"], dtype=np.float64)
    if not np.array_equal(maximum, np.maximum(rel_u, idx_u)):
        raise ValueError("coordinate diagnostic received a non-envelope score")
    return {
        "different_fraction": float(np.mean(rel_u != idx_u)),
        "idx_dominant_fraction": float(np.mean(idx_u > rel_u)),
        "released_dominant_fraction": float(np.mean(rel_u > idx_u)),
        "tie_fraction": float(np.mean(rel_u == idx_u)),
        "mean_absolute_chart_gap": float(np.mean(np.abs(rel_u - idx_u))),
    }


def _commit_score(
    root: Path,
    record: FullSeriesRecord,
    arm: str,
    score: np.ndarray,
    provenance: Mapping[str, Any],
) -> None:
    score_path = root / arm / "score.npy"
    digest = _atomic_npy(score_path, score)
    manifest = {
        **dict(provenance),
        "series_id": record.series_id,
        "dataset": record.dataset,
        "track": record.track,
        "paper_group": record.paper_group,
        "signal_name": record.signal_name,
        "arm": arm,
        "data_sha256": record.expected_sha256.upper(),
        "score_sha256": digest,
        "score_length": int(record.expected_length),
        "score_dtype": "float64",
    }
    _atomic_json(root / arm / "score_manifest.json", manifest)
    _atomic_json(
        root / arm / "_SCORES_READY.json",
        {
            "series_id": record.series_id,
            "arm": arm,
            "score_sha256": digest,
            "config_sha256": provenance["config_sha256"],
            "method_source_sha256": provenance["method_source_sha256"],
            "full_manifest_sha256": provenance["full_manifest_sha256"],
        },
    )
    _atomic_json(
        root / arm / "_SUCCESS.json",
        {"series_id": record.series_id, "arm": arm, "score_sha256": digest},
    )


def _commit_diagnostics(
    root: Path,
    record: FullSeriesRecord,
    diagnostics: Mapping[str, float],
    provenance: Mapping[str, Any],
) -> None:
    target = root / "coordinate_diagnostics"
    manifest_path = target / "diagnostics_manifest.json"
    _atomic_json(
        manifest_path,
        {
            **dict(provenance),
            "series_id": record.series_id,
            "data_sha256": record.expected_sha256.upper(),
            "diagnostics": dict(diagnostics),
        },
    )
    digest = sha256_file(manifest_path).upper()
    _atomic_json(
        target / "_DIAGNOSTICS_READY.json",
        {
            "series_id": record.series_id,
            "diagnostics_manifest_sha256": digest,
            "config_sha256": provenance["config_sha256"],
            "method_source_sha256": provenance["method_source_sha256"],
        },
    )


def _verify_completed_series(
    root: Path,
    record: FullSeriesRecord,
    config_sha: str,
    source_sha: str,
    manifest_sha: str,
) -> None:
    payload = json.loads((root / "_SUCCESS.json").read_text(encoding="utf-8"))
    if payload != {
        "arms": list(COORDINATE_ARMS),
        "count": len(COORDINATE_ARMS),
        "series_id": record.series_id,
    }:
        raise ValueError("coordinate-envelope series success marker is stale")
    arrays: dict[str, np.ndarray] = {}
    for arm in COORDINATE_ARMS:
        arm_root = root / arm
        required = (
            arm_root / "score.npy",
            arm_root / "score_manifest.json",
            arm_root / "_SCORES_READY.json",
            arm_root / "_SUCCESS.json",
        )
        if not all(path.is_file() for path in required):
            raise FileNotFoundError("completed coordinate transaction is incomplete")
        manifest = json.loads(required[1].read_text(encoding="utf-8"))
        ready = json.loads(required[2].read_text(encoding="utf-8"))
        success = json.loads(required[3].read_text(encoding="utf-8"))
        if (
            manifest.get("series_id") != record.series_id
            or manifest.get("arm") != arm
            or str(manifest.get("config_sha256")) != config_sha
            or str(manifest.get("method_source_sha256")) != source_sha
            or str(manifest.get("full_manifest_sha256")) != manifest_sha
            or str(manifest.get("data_sha256")) != record.expected_sha256.upper()
        ):
            raise ValueError("completed coordinate transaction has stale provenance")
        score_path = required[0]
        score_sha = sha256_file(score_path).upper()
        if score_sha != str(manifest.get("score_sha256")):
            raise ValueError("completed coordinate score hash mismatch")
        if ready != {
            "arm": arm,
            "config_sha256": config_sha,
            "full_manifest_sha256": manifest_sha,
            "method_source_sha256": source_sha,
            "score_sha256": score_sha,
            "series_id": record.series_id,
        }:
            raise ValueError("completed coordinate ready marker is stale")
        if success != {
            "arm": arm,
            "score_sha256": score_sha,
            "series_id": record.series_id,
        }:
            raise ValueError("completed coordinate arm success marker is stale")
        array = np.load(score_path, allow_pickle=False)
        if (
            array.shape != (record.expected_length,)
            or array.dtype != np.float64
            or not np.isfinite(array).all()
        ):
            raise ValueError("completed coordinate score payload is invalid")
        arrays[arm] = array
    if not np.array_equal(arrays["COORD_MAX"], np.maximum(arrays["REL_U"], arrays["IDX_U"])):
        raise ValueError("completed COORD_MAX is not the exact pointwise envelope")
    diagnostic_root = root / "coordinate_diagnostics"
    diagnostic_manifest_path = diagnostic_root / "diagnostics_manifest.json"
    diagnostic_ready_path = diagnostic_root / "_DIAGNOSTICS_READY.json"
    if not diagnostic_manifest_path.is_file() or not diagnostic_ready_path.is_file():
        raise FileNotFoundError("completed coordinate diagnostics are incomplete")
    diagnostic_manifest = json.loads(
        diagnostic_manifest_path.read_text(encoding="utf-8")
    )
    diagnostic_sha = sha256_file(diagnostic_manifest_path).upper()
    diagnostic_ready = json.loads(diagnostic_ready_path.read_text(encoding="utf-8"))
    if (
        diagnostic_manifest.get("series_id") != record.series_id
        or str(diagnostic_manifest.get("data_sha256"))
        != record.expected_sha256.upper()
        or str(diagnostic_manifest.get("config_sha256")) != config_sha
        or str(diagnostic_manifest.get("method_source_sha256")) != source_sha
        or str(diagnostic_manifest.get("full_manifest_sha256")) != manifest_sha
    ):
        raise ValueError("completed coordinate diagnostic provenance is stale")
    if diagnostic_ready != {
        "config_sha256": config_sha,
        "diagnostics_manifest_sha256": diagnostic_sha,
        "method_source_sha256": source_sha,
        "series_id": record.series_id,
    }:
        raise ValueError("completed coordinate diagnostic ready marker is stale")


def _transaction_state(
    root: Path,
    record: FullSeriesRecord,
    config_sha: str,
    source_sha: str,
    manifest_sha: str,
) -> str:
    if (root / "_FAILED.json").is_file():
        raise RuntimeError(f"retained coordinate failure blocks rerun: {record.series_id}")
    if (root / "_RUNNING.json").is_file():
        raise RuntimeError(f"active or stale coordinate transaction: {record.series_id}")
    if (root / "_SUCCESS.json").is_file():
        _verify_completed_series(root, record, config_sha, source_sha, manifest_sha)
        return "complete"
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"partial coordinate transaction blocks rerun: {record.series_id}")
    return "pending"


def _run_series(
    config_path: Path,
    config: Mapping[str, Any],
    config_sha: str,
    source_sha: str,
    manifest_sha: str,
    vendor: FrozenVendor,
    record: FullSeriesRecord,
) -> None:
    _safety_check(config)
    root = Path(config["paths"]["run_root"]) / EXPECTED_STAGE / record.series_id
    if _transaction_state(root, record, config_sha, source_sha, manifest_sha) == "complete":
        return
    root.mkdir(parents=True, exist_ok=True)
    status = Path(config["paths"]["log_root"]) / "coordinate_envelope_status.json"
    _atomic_json(root / "_RUNNING.json", {"series_id": record.series_id, "arms": COORDINATE_ARMS})
    started = time.perf_counter()
    device = torch.device(config["runtime"]["device"])
    try:
        _atomic_json(status, {"series_id": record.series_id, "state": "loading", "pid": os.getpid()})
        loaded = load_vendor_signal(record, Path(config["data"]["root"]))
        series = loaded.series
        batch = make_windows(
            released_preprocess(series.values),
            int(config["data"]["window_size"]),
            int(config["data"]["step_size"]),
        )
        if int(batch.values.shape[0]) != record.expected_windows:
            raise ValueError("full-main window count differs from the manifest")
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats()
        _atomic_json(status, {"series_id": record.series_id, "state": "cache", "pid": os.getpid()})
        cache, cache_dir, created, cache_seconds = _ensure_released_cache(
            vendor, series, batch, config
        )
        token_path = cache_dir / TOKEN_FILE
        token_manifest_path = cache_dir / TOKEN_MANIFEST
        token_hash = sha256_file(token_path).upper()
        token_manifest_hash = sha256_file(token_manifest_path).upper()
        _atomic_json(status, {"series_id": record.series_id, "state": "scoring", "pid": os.getpid()})
        scores = build_coordinate_scores(cache, batch, device)
        diagnostics = _coordinate_diagnostics(scores)
        torch.cuda.synchronize()
        if token_hash != sha256_file(token_path).upper() or token_manifest_hash != sha256_file(
            token_manifest_path
        ).upper():
            raise RuntimeError("token cache changed during coordinate scoring")
        vendor_status = subprocess.run(
            ["git", "-C", str(config["vendor"]["root"]), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if vendor_status:
            raise RuntimeError("vendor worktree changed during coordinate scoring")
        if coordinate_envelope_source_sha256() != source_sha:
            raise RuntimeError("coordinate scoring source changed during the active process")
        current_config, current_sha, _, current_manifest_sha = _load_config(config_path)
        if current_sha != config_sha or current_manifest_sha != manifest_sha:
            raise RuntimeError("coordinate config or manifest changed during the active process")
        if current_config["arms"] != config["arms"]:
            raise RuntimeError("coordinate arm identity changed during the active process")
        provenance = {
            "config_sha256": config_sha,
            "method_source_sha256": source_sha,
            "full_manifest_sha256": manifest_sha,
            "vendor_commit": vendor.commit,
            "clip_state_sha256": vendor.clip_weight_sha256.upper(),
            "released_cache_key": cache_key_digest(cache.key),
            "released_token_sha256": token_hash,
            "released_manifest_sha256": token_manifest_hash,
            "cache_created": bool(created),
            "cache_seconds": float(cache_seconds),
            "duplicate_timestamps_preserved": bool(loaded.duplicate_timestamps),
            "clip_forward_calls": (
                int(math.ceil(batch.values.shape[0] / int(config["vision"]["batch_size"])))
                if created
                else 0
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "peak_vram_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
            "wall_seconds_series": float(time.perf_counter() - started),
        }
        for arm in COORDINATE_ARMS:
            _commit_score(root, record, arm, scores[arm], provenance)
        _commit_diagnostics(root, record, diagnostics, provenance)
        _atomic_json(
            root / "_SUCCESS.json",
            {"series_id": record.series_id, "arms": COORDINATE_ARMS, "count": 3},
        )
        (root / "_RUNNING.json").unlink(missing_ok=True)
        _atomic_json(status, {"series_id": record.series_id, "state": "success", "pid": os.getpid()})
    except Exception as exc:
        _atomic_json(
            root / "_FAILED.json",
            {
                "series_id": record.series_id,
                "stage": EXPECTED_STAGE,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "config_sha256": config_sha,
                "method_source_sha256": source_sha,
                "full_manifest_sha256": manifest_sha,
                "pid": os.getpid(),
            },
        )
        _atomic_json(status, {"series_id": record.series_id, "state": "failed", "pid": os.getpid()})
        raise


def run_all(config_path: Path, series_ids: Sequence[str] | None = None) -> None:
    config, config_sha, records, manifest_sha = _load_config(config_path)
    source_sha = coordinate_envelope_source_sha256()
    _safety_check(config)
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("coordinate-envelope full stage requires CUDA")
    verify_vendor_sha(Path(config["vendor"]["root"]), str(config["vendor"]["commit"]))
    vendor = load_frozen_vendor(
        Path(config["vendor"]["root"]),
        str(config["vendor"]["commit"]),
        device,
        str(config["vendor"]["model_name"]),
    )
    if vendor.clip_weight_sha256.upper() != str(config["vendor"]["clip_state_sha256"]).upper():
        raise RuntimeError("frozen CLIP state hash differs from the registered identity")
    by_id = {record.series_id: record for record in records}
    ordered_ids = tuple(series_ids) if series_ids else tuple(by_id)
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("coordinate queue must contain unique registered series")
    if any(series_id not in by_id for series_id in ordered_ids):
        raise ValueError("coordinate queue contains an unregistered series")
    for series_id in ordered_ids:
        _run_series(
            config_path,
            config,
            config_sha,
            source_sha,
            manifest_sha,
            vendor,
            by_id[series_id],
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--series", action="append")
    args = parser.parse_args(argv)
    run_all(args.config, args.series)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
