"""Method-only, label-free runner for measure-consistent frozen ViT4TS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch
import yaml

from .cache import (
    MAP_MANIFEST,
    TOKEN_MANIFEST,
    ClipCacheKey,
    build_frozen_anomaly_maps,
    cache_key_digest,
    compute_renderer_sha256,
    encode_frozen_clip,
    load_anomaly_map_cache,
    load_clip_cache,
    load_frozen_clip,
    make_cache_key,
    save_clip_cache,
    write_anomaly_map_cache,
)
from .data import (
    SeriesData,
    SeriesSpec,
    WindowBatch,
    load_scoring_specs,
    load_signal,
    make_windows,
    released_preprocess,
)
from .geometry import TraceGeometry
from .provenance import score_source_sha256
from .reducers import ArmName, score_arm
from .renderers import (
    RenderedTrace,
    render_official_trace,
    render_patch_normalized_trace,
    render_supersampled_trace,
)


@dataclass(frozen=True)
class FrozenVendor:
    root: Path
    commit: str
    model: torch.nn.Module
    clip_weight_sha256: str


@dataclass(frozen=True)
class RendererResult:
    renderer: str
    maps: np.ndarray
    geometries: tuple[TraceGeometry, ...]
    key: ClipCacheKey
    cache_dir: Path
    wall_seconds: float


@dataclass(frozen=True)
class ScoreArtifact:
    series_id: str
    arm: str
    score_path: Path
    score_sha256: str
    manifest_path: Path


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_npy(path: Path, array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("score must be a non-empty finite vector")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256_file(path)


def _load_config(config_path: Path) -> tuple[dict[str, Any], str]:
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config, hashlib.sha256(raw).hexdigest().upper()


def _safety_check(config: Mapping[str, Any]) -> None:
    floor_disk = float(config["runtime"]["c_drive_floor_gib"])
    floor_ram = float(config["runtime"]["available_ram_floor_gib"])
    c_free = shutil.disk_usage("C:\\").free / (1024**3)
    ram_free = psutil.virtual_memory().available / (1024**3)
    if c_free < floor_disk:
        raise RuntimeError(f"C drive free space {c_free:.2f} GiB is below {floor_disk}")
    if ram_free < floor_ram:
        raise RuntimeError(f"available RAM {ram_free:.2f} GiB is below {floor_ram}")


def load_frozen_vendor(
    vendor_root: Path,
    expected_commit: str,
    device: torch.device,
    model_name: str = "ViT-B-16",
) -> FrozenVendor:
    model, state_sha = load_frozen_clip(vendor_root, expected_commit, device, model_name)
    return FrozenVendor(
        root=vendor_root.resolve(strict=True),
        commit=expected_commit.lower(),
        model=model,
        clip_weight_sha256=state_sha.lower(),
    )


def _render_windows(
    batch: WindowBatch,
    renderer: str,
    config: Mapping[str, Any],
    phase_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, tuple[TraceGeometry, ...]]:
    image_size = tuple(int(value) for value in config["vision"]["image_size"])
    bandwidth = float(config["geometry"]["bandwidth"])
    outputs: list[RenderedTrace] = []
    for window in batch.values:
        if renderer == "released":
            item = render_official_trace(
                window,
                image_size=image_size,
                phase_shift=phase_shift,
                bandwidth=bandwidth,
                expected_length=batch.window_size,
            )
        elif renderer == "distance":
            item = render_patch_normalized_trace(
                window,
                image_size=image_size,
                phase_shift=phase_shift,
                bandwidth=bandwidth,
                expected_length=batch.window_size,
            )
        elif renderer == "supersample4":
            item = render_supersampled_trace(
                window,
                image_size=image_size,
                phase_shift=phase_shift,
                bandwidth=bandwidth,
                expected_length=batch.window_size,
                scale=int(config["controls"]["supersample_scale"]),
            )
        else:
            raise ValueError(f"unsupported renderer {renderer}")
        outputs.append(item)
    images = np.stack([item.image for item in outputs]).astype(np.float32, copy=False)
    geometries = tuple(item.geometry for item in outputs)
    return np.ascontiguousarray(images), geometries


def prepare_renderer_cache(
    vendor: FrozenVendor,
    series: SeriesData,
    batch: WindowBatch,
    renderer: str,
    config: Mapping[str, Any],
    phase_shift: tuple[float, float] = (0.0, 0.0),
) -> RendererResult:
    started = time.perf_counter()
    images, geometries = _render_windows(batch, renderer, config, phase_shift)
    renderer_sha = compute_renderer_sha256(images, geometries)
    key = make_cache_key(
        series_id=series.series_id,
        data_sha256=series.data_sha256,
        renderer=renderer,
        renderer_sha256=renderer_sha,
        vendor_commit=vendor.commit,
        clip_weight_sha256=vendor.clip_weight_sha256,
        model_name=str(config["vendor"]["model_name"]),
        patch_size=int(config["vision"]["patch_size"]),
        image_size=tuple(int(v) for v in config["vision"]["image_size"]),
    )
    cache_dir = (
        Path(config["paths"]["cache_root"])
        / key.series_id
        / key.renderer
        / cache_key_digest(key)
    )
    if (cache_dir / TOKEN_MANIFEST).is_file():
        token_cache = load_clip_cache(cache_dir, key)
    else:
        token_cache = encode_frozen_clip(
            vendor.model,
            images,
            key,
            int(config["vision"]["batch_size"]),
            torch.device(config["runtime"]["device"]),
        )
        actual_dir = save_clip_cache(token_cache, Path(config["paths"]["cache_root"]))
        if actual_dir != cache_dir:
            raise RuntimeError("cache directory identity mismatch")
    if (cache_dir / MAP_MANIFEST).is_file():
        maps = load_anomaly_map_cache(cache_dir, key)
    else:
        maps = build_frozen_anomaly_maps(
            token_cache,
            torch.device(config["runtime"]["device"]),
            int(config["vision"]["map_batch_size"]),
        )
        write_anomaly_map_cache(maps, key, cache_dir)
    if maps.shape[0] != batch.values.shape[0]:
        raise RuntimeError("renderer cache window count mismatch")
    return RendererResult(
        renderer=renderer,
        maps=maps,
        geometries=geometries,
        key=key,
        cache_dir=cache_dir,
        wall_seconds=time.perf_counter() - started,
    )


def commit_score(
    run_dir: Path,
    series: SeriesData,
    arm: str,
    score: np.ndarray,
    provenance: Mapping[str, Any],
) -> ScoreArtifact:
    if np.asarray(score).shape != (series.values.size,):
        raise ValueError("score length must equal the complete series length")
    score_path = run_dir / "score.npy"
    score_sha = atomic_write_npy(score_path, score)
    manifest = {
        **dict(provenance),
        "series_id": series.series_id,
        "arm": arm,
        "data_sha256": series.data_sha256.upper(),
        "score_sha256": score_sha,
        "score_length": int(series.values.size),
        "score_dtype": "float64",
    }
    manifest_path = run_dir / "score_manifest.json"
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        run_dir / "_SCORES_READY.json",
        {
            "series_id": series.series_id,
            "arm": arm,
            "score_sha256": score_sha,
            "method_source_sha256": manifest["method_source_sha256"],
        },
    )
    _atomic_json(
        run_dir / "_SUCCESS.json",
        {"series_id": series.series_id, "arm": arm, "score_sha256": score_sha},
    )
    return ScoreArtifact(series.series_id, arm, score_path, score_sha, manifest_path)


def _runtime_payload(
    config_sha: str,
    vendor: FrozenVendor,
    renderer_results: Mapping[str, RendererResult],
    elapsed: float,
) -> dict[str, Any]:
    device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
    return {
        "config_sha256": config_sha,
        "method_source_sha256": score_source_sha256(),
        "vendor_commit": vendor.commit,
        "clip_state_sha256": vendor.clip_weight_sha256.upper(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device_index) if device_index is not None else "cpu",
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated(device_index) / (1024**2))
            if device_index is not None
            else 0.0
        ),
        "wall_seconds_series": float(elapsed),
        "renderer_wall_seconds": {
            name: float(result.wall_seconds) for name, result in renderer_results.items()
        },
        "renderer_cache_keys": {
            name: cache_key_digest(result.key) for name, result in renderer_results.items()
        },
        "renderer_cache_bytes": {
            name: int(
                sum(path.stat().st_size for path in result.cache_dir.glob("*") if path.is_file())
            )
            for name, result in renderer_results.items()
        },
    }


def run_series(
    config_path: Path,
    series_id: str,
    requested_arms: Sequence[str] | None = None,
    *,
    vendor: FrozenVendor | None = None,
) -> tuple[ScoreArtifact, ...]:
    """Run one registered series without importing or loading ground truth."""

    config, config_sha = _load_config(config_path)
    _safety_check(config)
    specs = {spec.series_id: spec for spec in load_scoring_specs(config)}
    if series_id not in specs:
        raise ValueError(f"series {series_id!r} is not registered")
    stage = str(config["stage"])
    registered_arms = tuple(str(item) for item in config["arms"][stage])
    arms = tuple(requested_arms) if requested_arms is not None else registered_arms
    if not arms or any(arm not in registered_arms for arm in arms):
        raise ValueError("only preregistered arms may be requested")
    device = torch.device(config["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the frozen experiment requires CUDA")
    if vendor is None:
        vendor = load_frozen_vendor(
            Path(config["vendor"]["root"]),
            str(config["vendor"]["commit"]),
            device,
            str(config["vendor"]["model_name"]),
        )

    root = Path(config["paths"]["run_root"]) / stage / series_id
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "_RUNNING.json", {"series_id": series_id, "stage": stage, "arms": arms})
    status_path = Path(config["paths"]["log_root"]) / f"{stage}_status.json"
    _atomic_json(status_path, {"series_id": series_id, "state": "loading", "pid": os.getpid()})
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        series = load_signal(specs[series_id])
        batch = make_windows(
            released_preprocess(series.values),
            int(config["data"]["window_size"]),
            int(config["data"]["step_size"]),
        )
        renderers_needed = {"released", "distance"}
        if "SUPERSAMPLE4" in arms:
            renderers_needed.add("supersample4")
        results: dict[str, RendererResult] = {}
        for renderer_name in ("released", "distance", "supersample4"):
            if renderer_name not in renderers_needed:
                continue
            _atomic_json(
                status_path,
                {"series_id": series_id, "state": f"renderer:{renderer_name}", "pid": os.getpid()},
            )
            results[renderer_name] = prepare_renderer_cache(
                vendor, series, batch, renderer_name, config
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        provenance = _runtime_payload(
            config_sha,
            vendor,
            results,
            time.perf_counter() - started,
        )
        artifacts: list[ScoreArtifact] = []
        for arm in arms:
            _atomic_json(
                status_path,
                {"series_id": series_id, "state": f"score:{arm}", "pid": os.getpid()},
            )
            score = score_arm(
                arm,  # type: ignore[arg-type]
                results["released"].maps,
                results["distance"].maps,
                results.get("supersample4").maps if "supersample4" in results else None,
                results["released"].geometries,
                results["distance"].geometries,
                batch,
                supersample_geometry=(
                    results["supersample4"].geometries
                    if "supersample4" in results
                    else None
                ),
            )
            artifacts.append(commit_score(root / arm, series, arm, score, provenance))
        _atomic_json(
            root / "_SUCCESS.json",
            {"series_id": series_id, "stage": stage, "arms": arms, "count": len(artifacts)},
        )
        (root / "_RUNNING.json").unlink(missing_ok=True)
        _atomic_json(status_path, {"series_id": series_id, "state": "success", "pid": os.getpid()})
        return tuple(artifacts)
    except Exception as exc:
        failure = {
            "series_id": series_id,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "config_sha256": config_sha,
            "vendor_commit": vendor.commit,
            "pid": os.getpid(),
        }
        _atomic_json(root / "_FAILED.json", failure)
        _atomic_json(status_path, {"series_id": series_id, "state": "failed", "pid": os.getpid()})
        raise


def run_all(config_path: Path, series_ids: Sequence[str] | None = None) -> None:
    config, _ = _load_config(config_path)
    specs = load_scoring_specs(config)
    ordered = tuple(series_ids) if series_ids else tuple(spec.series_id for spec in specs)
    known = {spec.series_id for spec in specs}
    if any(series_id not in known for series_id in ordered):
        raise ValueError("run queue contains an unregistered series")
    vendor = load_frozen_vendor(
        Path(config["vendor"]["root"]),
        str(config["vendor"]["commit"]),
        torch.device(config["runtime"]["device"]),
        str(config["vendor"]["model_name"]),
    )
    for series_id in ordered:
        run_series(config_path, series_id, vendor=vendor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--series", action="append")
    args = parser.parse_args(argv)
    run_all(args.config, args.series)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
