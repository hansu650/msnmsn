"""Hash-bound OpenCLIP bridge for the frozen ViTTrace v3 spectrogram route.

The CPU-only spectrogram route owns pixels and scores.  This module is the
single, isolated model-forward bridge between those deterministic pixels and
the existing :mod:`dynamic_cache` schema.  It is fixed to spectrogram,
OpenCLIP ViT-B/16, W=240, stride=60, 224x224, and the registered batch size.

Images are never retained for a complete series.  A first bounded-memory pass
computes the exact renderer identity of the materialized ``[N,3,224,224]``
array.  A second bounded-memory pass renders and encodes chunks through the
same ``encode_dynamic_tokens`` function used by the line encoder.  Completed
series are committed last and can be resumed only after cache/hash validation.
This scorer has no label, anomaly, threshold, or metric surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from measure_vit4ts.data import make_windows, released_preprocess
from measure_vit4ts.full_manifest import EXPECTED_SERIES, FullSeriesRecord

from . import encoder_runner as encoder_stage
from .dynamic_cache import (
    CACHE_FILE,
    CACHE_MANIFEST,
    DynamicCacheKey,
    DynamicTokenCache,
    cache_digest,
    encode_dynamic_tokens,
    load_dynamic_cache,
    save_dynamic_cache,
)
from .spectrogram_registry import (
    SpectrogramRoute,
    load_spectrogram_route,
    make_spectrogram_cache_key,
    renderer_identity,
    validate_spectrogram_cache_key,
)
from .spectrogram_renderer import (
    SCHEMA_VERSION as RENDERER_SCHEMA_VERSION,
    render_spectrogram_windows,
    renderer_config_sha256,
    renderer_source_sha256,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = encoder_stage.DEFAULT_CONFIG
STATUS_FILE = "spectrogram_encoder_stage_status.json"
PATCH_MEAN_FILE = encoder_stage.PATCH_MEAN_FILE
RENDER_MANIFEST_FILE = "spectrogram_render_manifest.json"
_SOURCE_FILES = (
    "spectrogram_renderer.py",
    "spectrogram_registry.py",
    "spectrogram_encoder_runner.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest().upper()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or not np.isfinite(array).all():
        raise ValueError("patch-mean sidecar must be finite float32 [N,D]")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256_file(path)


def spectrogram_encoder_source_sha256(
    package_root: Path | None = None,
    *,
    encoder_source_sha256: str | None = None,
) -> str:
    """Bind this bridge to its renderer/registry and reused encoder sources."""

    root = Path(package_root) if package_root else Path(__file__).resolve().parent
    reused = str(
        encoder_source_sha256
        or encoder_stage.encoder_source_sha256(root if package_root else None)
    ).upper()
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        payload = (root / name).read_bytes()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    digest.update(b"encoder_source_sha256")
    digest.update(reused.encode("ascii"))
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class StreamingRenderIdentity:
    renderer_source_sha256: str
    renderer_config_sha256: str
    image_array_sha256: str
    renderer_sha256: str
    image_shape: tuple[int, int, int, int]
    image_dtype: str
    stft_shape: tuple[int, int, int]


@dataclass(frozen=True)
class EncodedSpectrogram:
    cache: DynamicTokenCache
    encoder_calls: int
    renderer_seconds: float
    encoder_seconds: float


def _validate_windows(windows: np.ndarray, route: SpectrogramRoute) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(windows), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0:
        raise ValueError("spectrogram windows must be nonempty [N,240]")
    if values.shape[1] != route.variant.window or not np.isfinite(values).all():
        raise ValueError("spectrogram windows must be finite W=240")
    return values


def streaming_render_identity(
    windows: np.ndarray,
    route: SpectrogramRoute,
    *,
    chunk_size: int = 64,
) -> StreamingRenderIdentity:
    """Hash a complete render without materializing its full image tensor."""

    values = _validate_windows(windows, route)
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive")
    shape = (int(values.shape[0]), 3, 224, 224)
    dtype = np.dtype(np.float32).str
    image_digest = hashlib.sha256()
    image_digest.update(_canonical_json({"dtype": dtype, "shape": list(shape)}))
    expected_source = renderer_source_sha256()
    expected_config = renderer_config_sha256(route.spec)
    for start in range(0, values.shape[0], chunk):
        rendered = render_spectrogram_windows(values[start : start + chunk], route.spec)
        if rendered.renderer_source_sha256 != expected_source:
            raise RuntimeError("spectrogram renderer source changed between chunks")
        if rendered.renderer_config_sha256 != expected_config:
            raise RuntimeError("spectrogram renderer config changed between chunks")
        image_digest.update(memoryview(rendered.images).cast("B"))
    image_sha = image_digest.hexdigest().upper()
    renderer_sha = _payload_sha256(
        {
            "schema_version": RENDERER_SCHEMA_VERSION,
            "renderer": "spectrogram",
            "renderer_source_sha256": expected_source,
            "renderer_config_sha256": expected_config,
            "image_array_sha256": image_sha,
            "image_shape": list(shape),
            "image_dtype": dtype,
        }
    )
    return StreamingRenderIdentity(
        renderer_source_sha256=expected_source,
        renderer_config_sha256=expected_config,
        image_array_sha256=image_sha,
        renderer_sha256=renderer_sha,
        image_shape=shape,
        image_dtype=dtype,
        stft_shape=(shape[0], route.spec.frequency_bins, route.spec.frame_count),
    )


def encode_spectrogram_windows(
    adapter: torch.nn.Module,
    windows: np.ndarray,
    key: DynamicCacheKey,
    route: SpectrogramRoute,
    *,
    device: torch.device | str,
    chunk_size: int = 64,
) -> EncodedSpectrogram:
    """Render bounded chunks and reuse the canonical dynamic encoder path."""

    values = _validate_windows(windows, route)
    validate_spectrogram_cache_key(route, key)
    chunk = int(chunk_size)
    if chunk <= 0 or chunk > route.variant.batch_size:
        raise ValueError("chunk_size must be in [1, registered batch size]")
    global_parts: list[np.ndarray] = []
    patch_parts: list[np.ndarray] = []
    mid_parts: list[np.ndarray] = []
    large_parts: list[np.ndarray] = []
    patch_grid: tuple[int, int] | None = None
    mid_mask: np.ndarray | None = None
    large_mask: np.ndarray | None = None
    calls = 0
    render_seconds = 0.0
    encode_seconds = 0.0
    for start in range(0, values.shape[0], chunk):
        render_started = time.perf_counter()
        rendered = render_spectrogram_windows(values[start : start + chunk], route.spec)
        render_seconds += time.perf_counter() - render_started
        encode_started = time.perf_counter()
        partial = encode_dynamic_tokens(
            adapter,
            rendered.images,
            key,
            batch_size=route.variant.batch_size,
            device=device,
        )
        encode_seconds += time.perf_counter() - encode_started
        calls += int(math.ceil(rendered.images.shape[0] / route.variant.batch_size))
        if patch_grid is None:
            patch_grid = partial.patch_grid
            mid_mask = partial.mid_mask
            large_mask = partial.large_mask
        elif (
            patch_grid != partial.patch_grid
            or not np.array_equal(mid_mask, partial.mid_mask)
            or not np.array_equal(large_mask, partial.large_mask)
        ):
            raise RuntimeError("dynamic P/M/L geometry changed between render chunks")
        global_parts.append(partial.global_tokens)
        patch_parts.append(partial.patch_tokens)
        mid_parts.append(partial.mid_tokens)
        large_parts.append(partial.large_tokens)
    if patch_grid is None or mid_mask is None or large_mask is None:
        raise RuntimeError("spectrogram encoder produced no chunks")
    cache = DynamicTokenCache(
        key=key,
        patch_grid=patch_grid,
        global_tokens=np.ascontiguousarray(np.concatenate(global_parts), dtype=np.float32),
        patch_tokens=np.ascontiguousarray(np.concatenate(patch_parts), dtype=np.float32),
        mid_tokens=np.ascontiguousarray(np.concatenate(mid_parts), dtype=np.float32),
        large_tokens=np.ascontiguousarray(np.concatenate(large_parts), dtype=np.float32),
        mid_mask=np.ascontiguousarray(mid_mask, dtype=np.int64),
        large_mask=np.ascontiguousarray(large_mask, dtype=np.int64),
    )
    if cache.patch_tokens.shape[0] != values.shape[0]:
        raise RuntimeError("spectrogram encoder window count changed")
    return EncodedSpectrogram(cache, calls, render_seconds, encode_seconds)


def _variant_and_route(
    config: Mapping[str, Any], config_sha: str
) -> tuple[encoder_stage.EncoderVariant, SpectrogramRoute, str]:
    route = load_spectrogram_route(config)
    variant = encoder_stage.resolve_variant(
        config,
        representation="spectrogram",
        model_key="B16",
        window=240,
        stride=60,
        batch_size=64,
        variant_key="spectrogram_B16_W240_S60_B64",
    )
    expected = route.variant.to_payload()
    actual = {**asdict(variant), "image_size": list(variant.image_size)}
    actual.pop("key")
    if actual != expected:
        raise ValueError("encoder variant differs from frozen spectrogram route")
    return variant, route, encoder_stage.variant_sha256(variant, config_sha)


def bridge_identity_sha256(
    *,
    config_sha: str,
    manifest_sha: str,
    route_config_sha: str,
    variant_sha: str,
    encoder_source_sha: str,
    bridge_source_sha: str,
) -> str:
    return _payload_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": str(config_sha).upper(),
            "manifest_sha256": str(manifest_sha).upper(),
            "route_config_sha256": str(route_config_sha).upper(),
            "variant_sha256": str(variant_sha).upper(),
            "encoder_source_sha256": str(encoder_source_sha).upper(),
            "spectrogram_encoder_source_sha256": str(bridge_source_sha).upper(),
        }
    )


def stage_root(
    config: Mapping[str, Any], variant: encoder_stage.EncoderVariant, identity_sha: str
) -> Path:
    return (
        Path(config["paths"]["output_root"])
        / "spectrogram_encoder_stage"
        / f"{variant.key}__{str(identity_sha).upper()[:12]}"
    )


def _record_path(root: Path, series_id: str) -> Path:
    return Path(root) / "records" / f"{series_id}.json"


def _key_from_payload(payload: Mapping[str, Any]) -> DynamicCacheKey:
    values = dict(payload)
    values["image_size"] = tuple(map(int, values["image_size"]))
    return DynamicCacheKey(**values)


def _write_streaming_render_manifest(
    path: Path,
    *,
    config_sha: str,
    route: SpectrogramRoute,
    render: StreamingRenderIdentity,
    key: DynamicCacheKey,
) -> str:
    """Write the render provenance consumed by the CPU score route."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "renderer_schema_version": RENDERER_SCHEMA_VERSION,
        "series_id": key.series_id,
        "config_file_sha256": str(config_sha).upper(),
        "route_config_sha256": route.route_config_sha256,
        "renderer_source_sha256": render.renderer_source_sha256,
        "renderer_config_sha256": render.renderer_config_sha256,
        "image_array_sha256": render.image_array_sha256,
        "image_shape": list(render.image_shape),
        "image_dtype": render.image_dtype,
        "stft_shape": list(render.stft_shape),
        "renderer_sha256": render.renderer_sha256,
        "cache_key": {**asdict(key), "image_size": list(key.image_size)},
        "cache_digest": cache_digest(key),
        "encoder_calls": 0,
        "model_forward": False,
        "materialized_image_file": None,
        "streaming_two_pass": True,
    }
    _atomic_json(path, payload)
    return _sha256_file(path)


def _validate_streaming_render_manifest(
    path: Path,
    *,
    config_sha: str,
    route: SpectrogramRoute,
    key: DynamicCacheKey,
    expected_sha256: str,
) -> Mapping[str, Any]:
    if _sha256_file(path) != str(expected_sha256).upper():
        raise ValueError("spectrogram render-manifest file hash changed")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    key_payload = dict(payload["cache_key"])
    key_payload["image_size"] = tuple(map(int, key_payload["image_size"]))
    manifest_key = DynamicCacheKey(**key_payload)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "renderer_schema_version": RENDERER_SCHEMA_VERSION,
        "series_id": key.series_id,
        "config_file_sha256": str(config_sha).upper(),
        "route_config_sha256": route.route_config_sha256,
        "renderer_source_sha256": renderer_source_sha256(),
        "renderer_config_sha256": renderer_config_sha256(route.spec),
        "renderer_sha256": key.renderer_sha256,
        "cache_digest": cache_digest(key),
        "encoder_calls": 0,
        "model_forward": False,
        "materialized_image_file": None,
        "streaming_two_pass": True,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("spectrogram render-manifest identity changed")
    if manifest_key != key:
        raise ValueError("spectrogram render-manifest cache key changed")
    return payload


def _resume_record(
    path: Path,
    *,
    route: SpectrogramRoute,
    cache_root: Path,
    config_sha: str,
    manifest_sha: str,
    route_config_sha: str,
    encoder_source_sha: str,
    bridge_source_sha: str,
    variant_sha: str,
    bridge_identity_sha: str,
    data_sha: str,
    model_sha: str,
) -> Mapping[str, Any] | None:
    if not Path(path).is_file():
        return None
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "config_sha256": str(config_sha).upper(),
        "manifest_sha256": str(manifest_sha).upper(),
        "route_config_sha256": str(route_config_sha).upper(),
        "encoder_source_sha256": str(encoder_source_sha).upper(),
        "spectrogram_encoder_source_sha256": str(bridge_source_sha).upper(),
        "variant_sha256": str(variant_sha).upper(),
        "bridge_identity_sha256": str(bridge_identity_sha).upper(),
        "data_sha256": str(data_sha).upper(),
        "model_sha256": str(model_sha).upper(),
    }
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if any(payload.get(name) != value for name, value in expected.items()):
            return None
        key = _key_from_payload(payload["cache_key"])
        validate_spectrogram_cache_key(route, key)
        if key.data_sha256.upper() != str(data_sha).upper():
            return None
        if key.model_sha256.upper() != str(model_sha).upper():
            return None
        directory = Path(payload["cache_dir"]).resolve()
        root = Path(cache_root).resolve()
        if not directory.is_relative_to(root):
            return None
        if directory.name != cache_digest(key):
            return None
        if _sha256_file(directory / CACHE_FILE) != payload["cache_file_sha256"]:
            return None
        if _sha256_file(directory / CACHE_MANIFEST) != payload["cache_manifest_sha256"]:
            return None
        render_manifest = Path(payload["render_manifest_path"])
        _validate_streaming_render_manifest(
            render_manifest,
            config_sha=config_sha,
            route=route,
            key=key,
            expected_sha256=payload["render_manifest_sha256"],
        )
        cache = load_dynamic_cache(directory, key)
        if cache.patch_tokens.shape[0] != int(payload["window_count"]):
            return None
        checks = {
            "global_tokens_sha256": cache.global_tokens,
            "patch_tokens_sha256": cache.patch_tokens,
            "mid_tokens_sha256": cache.mid_tokens,
            "large_tokens_sha256": cache.large_tokens,
        }
        if any(
            encoder_stage._array_sha256(value) != payload[name]
            for name, value in checks.items()
        ):
            return None
        mean_path = Path(payload["patch_mean_path"])
        if (
            not mean_path.is_file()
            or _sha256_file(mean_path) != payload["patch_mean_sha256"]
        ):
            return None
        mean = np.load(mean_path, allow_pickle=False)
        if (
            mean.dtype != np.float32
            or mean.shape != tuple(payload["patch_mean_shape"])
            or not np.isfinite(mean).all()
            or encoder_stage._array_sha256(mean)
            != payload["patch_mean_array_sha256"]
        ):
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _failure_path(config: Mapping[str, Any], series_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        Path(config["paths"]["failure_root"])
        / "spectrogram_encoder_stage"
        / series_id
        / f"{stamp}_{uuid.uuid4().hex}.json"
    )


def _run_series(
    config: Mapping[str, Any],
    record: FullSeriesRecord,
    variant: encoder_stage.EncoderVariant,
    route: SpectrogramRoute,
    root: Path,
    encoder: encoder_stage.LoadedEncoder,
    device: torch.device,
    *,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    bridge_source_sha: str,
    variant_sha: str,
    identity_sha: str,
    retry: bool,
) -> Mapping[str, Any]:
    started_at = _utc_now()
    wall_started = time.perf_counter()
    series = encoder_stage._load_series(record, Path(config["data"]["root"]))
    batch = make_windows(released_preprocess(series.values), 240, 60)
    expected_windows = 1 + (record.expected_length - 240) // 60
    if batch.values.shape != (expected_windows, 240):
        raise RuntimeError("spectrogram window count differs from W240/S60 manifest")

    identity_started = time.perf_counter()
    render = streaming_render_identity(
        batch.values, route, chunk_size=route.variant.batch_size
    )
    identity_seconds = time.perf_counter() - identity_started
    key = make_spectrogram_cache_key(
        route,
        render,
        series_id=record.series_id,
        data_sha256=record.expected_sha256,
        model_sha256=encoder.model_sha256,
    )
    cache_root = Path(root) / "caches"
    expected_dir = cache_root / key.series_id / cache_digest(key)
    reused = False
    encoded: EncodedSpectrogram | None = None
    if not retry and (expected_dir / CACHE_FILE).is_file() and (
        expected_dir / CACHE_MANIFEST
    ).is_file():
        cache = load_dynamic_cache(expected_dir, key)
        reused = True
    else:
        encoded = encode_spectrogram_windows(
            encoder.adapter,
            batch.values,
            key,
            route,
            device=device,
            chunk_size=route.variant.batch_size,
        )
        cache = encoded.cache
        actual = save_dynamic_cache(cache, cache_root)
        if actual != expected_dir:
            raise RuntimeError("spectrogram dynamic-cache directory identity mismatch")

    patch_mean = encoder_stage.patch_mean_tokens(cache)
    mean_path = expected_dir / PATCH_MEAN_FILE
    mean_sha = _atomic_npy(mean_path, patch_mean)
    render_manifest_path = expected_dir / RENDER_MANIFEST_FILE
    render_manifest_sha = _write_streaming_render_manifest(
        render_manifest_path,
        config_sha=config_sha,
        route=route,
        render=render,
        key=key,
    )
    encoder_calls = 0 if reused else int(encoded.encoder_calls)
    renderer_seconds = identity_seconds + (0.0 if reused else encoded.renderer_seconds)
    encoder_seconds = 0.0 if reused else encoded.encoder_seconds
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "created_at": _utc_now(),
        "started_at": started_at,
        "series_id": record.series_id,
        "family": record.track,
        "subgroup": record.paper_group,
        "config_sha256": str(config_sha).upper(),
        "manifest_sha256": str(manifest_sha).upper(),
        "route_config_sha256": route.route_config_sha256,
        "encoder_source_sha256": str(encoder_source_sha).upper(),
        "spectrogram_encoder_source_sha256": str(bridge_source_sha).upper(),
        "variant_sha256": str(variant_sha).upper(),
        "bridge_identity_sha256": str(identity_sha).upper(),
        "variant": {**asdict(variant), "image_size": list(variant.image_size)},
        "data_sha256": record.expected_sha256.upper(),
        "model_sha256": encoder.model_sha256.upper(),
        "renderer_identity": renderer_identity(route),
        "renderer_source_sha256": render.renderer_source_sha256,
        "renderer_config_sha256": render.renderer_config_sha256,
        "renderer_sha256": render.renderer_sha256,
        "image_array_sha256": render.image_array_sha256,
        "image_shape": list(render.image_shape),
        "image_dtype": render.image_dtype,
        "stft_shape": list(render.stft_shape),
        "window_count": int(batch.values.shape[0]),
        "encoder_calls": encoder_calls,
        "cache_reused_after_interruption": reused,
        "cache_dir": str(expected_dir.resolve()),
        "cache_key": {**asdict(key), "image_size": list(key.image_size)},
        "cache_file_sha256": _sha256_file(expected_dir / CACHE_FILE),
        "cache_manifest_sha256": _sha256_file(expected_dir / CACHE_MANIFEST),
        "render_manifest_path": str(render_manifest_path.resolve()),
        "render_manifest_sha256": render_manifest_sha,
        "patch_grid": list(cache.patch_grid),
        "global_tokens_shape": list(cache.global_tokens.shape),
        "global_tokens_sha256": encoder_stage._array_sha256(cache.global_tokens),
        "patch_tokens_shape": list(cache.patch_tokens.shape),
        "patch_tokens_sha256": encoder_stage._array_sha256(cache.patch_tokens),
        "mid_tokens_shape": list(cache.mid_tokens.shape),
        "mid_tokens_sha256": encoder_stage._array_sha256(cache.mid_tokens),
        "large_tokens_shape": list(cache.large_tokens.shape),
        "large_tokens_sha256": encoder_stage._array_sha256(cache.large_tokens),
        "patch_mean_path": str(mean_path.resolve()),
        "patch_mean_shape": list(patch_mean.shape),
        "patch_mean_sha256": mean_sha,
        "patch_mean_array_sha256": encoder_stage._array_sha256(patch_mean),
        "renderer_seconds": float(renderer_seconds),
        "encoder_seconds": float(encoder_seconds),
        "wall_seconds": float(time.perf_counter() - wall_started),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "open_clip_version": encoder.open_clip_version,
        "open_clip_module": encoder.open_clip_module,
    }
    _atomic_json(_record_path(root, record.series_id), payload)
    return payload


def _carried_verified_ids(root: Path, identity_sha: str) -> set[str]:
    path = Path(root) / STATUS_FILE
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("bridge_identity_sha256") != str(identity_sha).upper():
            return set()
        return {
            str(row["series_id"])
            for row in payload.get("series", [])
            if row.get("status") == "PASS"
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return set()


def _write_summary(
    root: Path,
    records: Sequence[FullSeriesRecord],
    verified_ids: set[str],
    *,
    config_sha: str,
    manifest_sha: str,
    route_config_sha: str,
    encoder_source_sha: str,
    bridge_source_sha: str,
    variant_sha: str,
    identity_sha: str,
) -> Path:
    rows = [
        {
            "series_id": record.series_id,
            "family": record.track,
            "subgroup": record.paper_group,
            "status": "PASS" if record.series_id in verified_ids else "NOT_VERIFIED",
        }
        for record in records
    ]
    completed = sum(row["status"] == "PASS" for row in rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "COMPLETE" if completed == EXPECTED_SERIES else "INCOMPLETE",
        "expected_series": EXPECTED_SERIES,
        "completed_series": completed,
        "not_verified_series": EXPECTED_SERIES - completed,
        "config_sha256": str(config_sha).upper(),
        "manifest_sha256": str(manifest_sha).upper(),
        "route_config_sha256": str(route_config_sha).upper(),
        "encoder_source_sha256": str(encoder_source_sha).upper(),
        "spectrogram_encoder_source_sha256": str(bridge_source_sha).upper(),
        "variant_sha256": str(variant_sha).upper(),
        "bridge_identity_sha256": str(identity_sha).upper(),
        "series": rows,
    }
    path = Path(root) / STATUS_FILE
    _atomic_json(path, payload)
    return path


def _resolve_stage(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    str,
    tuple[FullSeriesRecord, ...],
    str,
    encoder_stage.EncoderVariant,
    SpectrogramRoute,
    str,
    str,
    str,
    Path,
]:
    config, config_sha, records, manifest_sha = encoder_stage._load_config(config_path)
    variant, route, variant_sha = _variant_and_route(config, config_sha)
    encoder_source_sha = encoder_stage.encoder_source_sha256()
    bridge_source_sha = spectrogram_encoder_source_sha256(
        encoder_source_sha256=encoder_source_sha
    )
    identity_sha = bridge_identity_sha256(
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        route_config_sha=route.route_config_sha256,
        variant_sha=variant_sha,
        encoder_source_sha=encoder_source_sha,
        bridge_source_sha=bridge_source_sha,
    )
    root = stage_root(config, variant, identity_sha)
    return (
        config,
        config_sha,
        records,
        manifest_sha,
        variant,
        route,
        variant_sha,
        encoder_source_sha,
        bridge_source_sha,
        root,
    )


def run_spectrogram_encoder_stage(
    config_path: Path,
    *,
    smoke: bool = False,
    all_series: bool = False,
    series_ids: Sequence[str] = (),
    approved_bulk: bool = False,
    device_name: str | None = None,
    retry: bool = False,
) -> Path:
    (
        config,
        config_sha,
        records,
        manifest_sha,
        variant,
        route,
        variant_sha,
        encoder_source_sha,
        bridge_source_sha,
        root,
    ) = _resolve_stage(config_path)
    selected = encoder_stage.select_records(
        records,
        smoke=smoke,
        all_series=all_series,
        series_ids=series_ids,
        approved_bulk=approved_bulk,
    )
    identity_sha = bridge_identity_sha256(
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        route_config_sha=route.route_config_sha256,
        variant_sha=variant_sha,
        encoder_source_sha=encoder_source_sha,
        bridge_source_sha=bridge_source_sha,
    )
    cache_root = root / "caches"
    model_sha = str(config["vendor"]["default_model_sha256"]).upper()
    verified = _carried_verified_ids(root, identity_sha)
    missing: list[FullSeriesRecord] = []
    for record in selected:
        if retry:
            verified.discard(record.series_id)
            missing.append(record)
            continue
        payload = _resume_record(
            _record_path(root, record.series_id),
            route=route,
            cache_root=cache_root,
            config_sha=config_sha,
            manifest_sha=manifest_sha,
            route_config_sha=route.route_config_sha256,
            encoder_source_sha=encoder_source_sha,
            bridge_source_sha=bridge_source_sha,
            variant_sha=variant_sha,
            bridge_identity_sha=identity_sha,
            data_sha=record.expected_sha256,
            model_sha=model_sha,
        )
        if payload is None:
            verified.discard(record.series_id)
            missing.append(record)
        else:
            verified.add(record.series_id)

    errors: list[tuple[str, Exception]] = []
    encoder: encoder_stage.LoadedEncoder | None = None
    if missing:
        encoder_stage._safety_check(config)
        device = torch.device(device_name or str(config["runtime"]["device"]))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA spectrogram encoder was requested but is unavailable")
        encoder = encoder_stage.load_openclip_encoder(config, variant, device)
        if encoder.model_sha256.upper() != model_sha:
            raise RuntimeError("spectrogram B16 model-state SHA256 differs from config")
        for record in missing:
            try:
                _run_series(
                    config,
                    record,
                    variant,
                    route,
                    root,
                    encoder,
                    device,
                    config_sha=config_sha,
                    manifest_sha=manifest_sha,
                    encoder_source_sha=encoder_source_sha,
                    bridge_source_sha=bridge_source_sha,
                    variant_sha=variant_sha,
                    identity_sha=identity_sha,
                    retry=retry,
                )
                verified.add(record.series_id)
            except Exception as exc:
                _atomic_json(
                    _failure_path(config, record.series_id),
                    {
                        "schema_version": SCHEMA_VERSION,
                        "created_at": _utc_now(),
                        "series_id": record.series_id,
                        "config_sha256": config_sha,
                        "manifest_sha256": manifest_sha,
                        "route_config_sha256": route.route_config_sha256,
                        "encoder_source_sha256": encoder_source_sha,
                        "spectrogram_encoder_source_sha256": bridge_source_sha,
                        "variant_sha256": variant_sha,
                        "bridge_identity_sha256": identity_sha,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                errors.append((record.series_id, exc))
                if not all_series:
                    break

    status = _write_summary(
        root,
        records,
        verified,
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        route_config_sha=route.route_config_sha256,
        encoder_source_sha=encoder_source_sha,
        bridge_source_sha=bridge_source_sha,
        variant_sha=variant_sha,
        identity_sha=identity_sha,
    )
    if errors:
        raise RuntimeError(
            f"spectrogram encoder retained {len(errors)} failures; "
            f"first={errors[0][0]}: {errors[0][1]}"
        )
    return status


class _MockVisionModel(torch.nn.Module):
    def encode_image(self, images: torch.Tensor):
        batch = int(images.shape[0])
        mean = images.mean(dim=(1, 2, 3), keepdim=False)
        global_tokens = mean[:, None] + torch.arange(
            8, dtype=torch.float32, device=images.device
        )[None, :]
        patch = torch.arange(
            197 * 6, dtype=torch.float32, device=images.device
        ).reshape(1, 197, 6)
        patch = patch.repeat(batch, 1, 1) + mean[:, None, None]
        return global_tokens, patch


class _MockAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _MockVisionModel()


def run_mock_gate(config_path: Path, output_directory: Path) -> Path:
    """Run a pure CPU/schema gate without importing or loading OpenCLIP."""

    raw = Path(config_path).read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("v3 config must be a mapping")
    route = load_spectrogram_route(config)
    samples = np.arange(5 * 240, dtype=np.float64).reshape(5, 240)
    windows = np.sin(2.0 * np.pi * samples / 37.0)
    materialized = render_spectrogram_windows(windows, route.spec)
    streamed = streaming_render_identity(windows, route, chunk_size=2)
    if (
        materialized.image_array_sha256 != streamed.image_array_sha256
        or materialized.renderer_sha256 != streamed.renderer_sha256
    ):
        raise RuntimeError("streaming renderer identity differs from materialized render")
    model_sha = str(config["vendor"]["default_model_sha256"]).upper()
    key = make_spectrogram_cache_key(
        route,
        streamed,
        series_id="SYNTHETIC_SPECTROGRAM_GATE",
        data_sha256="D" * 64,
        model_sha256=model_sha,
    )
    encoded = encode_spectrogram_windows(
        _MockAdapter(), windows, key, route, device="cpu", chunk_size=2
    )
    root = Path(output_directory)
    directory = save_dynamic_cache(encoded.cache, root / "caches")
    restored = load_dynamic_cache(directory, key)
    if restored.patch_grid != (14, 14):
        raise RuntimeError("mock gate did not preserve the B16 patch grid")
    line_key = DynamicCacheKey(**{**asdict(key), "renderer": "line.batch64"})
    render_manifest_path = directory / RENDER_MANIFEST_FILE
    render_manifest_sha = _write_streaming_render_manifest(
        render_manifest_path,
        config_sha=hashlib.sha256(raw).hexdigest().upper(),
        route=route,
        render=streamed,
        key=key,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "PASS",
        "device": "cpu",
        "model_forward": "deterministic_mock_only",
        "openclip_loaded": False,
        "renderer_hash_parity": True,
        "spectrogram_line_cache_separation": cache_digest(key) != cache_digest(line_key),
        "cache_schema_round_trip": True,
        "renderer_sha256": streamed.renderer_sha256,
        "image_array_sha256": streamed.image_array_sha256,
        "cache_digest": cache_digest(key),
        "cache_file_sha256": _sha256_file(directory / CACHE_FILE),
        "cache_manifest_sha256": _sha256_file(directory / CACHE_MANIFEST),
        "render_manifest_path": str(render_manifest_path.resolve()),
        "render_manifest_sha256": render_manifest_sha,
        "patch_grid": list(restored.patch_grid),
        "global_tokens_shape": list(restored.global_tokens.shape),
        "patch_tokens_shape": list(restored.patch_tokens.shape),
        "mid_tokens_shape": list(restored.mid_tokens.shape),
        "large_tokens_shape": list(restored.large_tokens.shape),
        "encoder_calls": encoded.encoder_calls,
        "encoder_source_sha256": encoder_stage.encoder_source_sha256(),
        "spectrogram_encoder_source_sha256": spectrogram_encoder_source_sha256(),
    }
    path = root / "spectrogram_encoder_mock_gate.json"
    _atomic_json(path, payload)
    return path


def status_path(config_path: Path) -> Path:
    *_, root = _resolve_stage(config_path)
    return root / STATUS_FILE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)

    gate = commands.add_parser("mock-gate")
    gate.add_argument("--output-dir", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--all", dest="all_series", action="store_true")
    run.add_argument("--series-id", action="append", default=[])
    run.add_argument("--approved-bulk", action="store_true")
    run.add_argument("--device")
    run.add_argument("--retry", action="store_true")

    commands.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "mock-gate":
        path = run_mock_gate(args.config, args.output_dir)
        print(path)
        return 0
    if args.command == "status":
        path = status_path(args.config)
        if path.is_file():
            print(path.read_text(encoding="utf-8"))
        else:
            print(json.dumps({"status": "NOT_STARTED", "path": str(path)}))
        return 0
    path = run_spectrogram_encoder_stage(
        args.config,
        smoke=args.smoke,
        all_series=args.all_series,
        series_ids=args.series_id,
        approved_bulk=args.approved_bulk,
        device_name=args.device,
        retry=args.retry,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EncodedSpectrogram",
    "RENDER_MANIFEST_FILE",
    "StreamingRenderIdentity",
    "bridge_identity_sha256",
    "encode_spectrogram_windows",
    "main",
    "run_mock_gate",
    "run_spectrogram_encoder_stage",
    "spectrogram_encoder_source_sha256",
    "stage_root",
    "status_path",
    "streaming_render_identity",
]
