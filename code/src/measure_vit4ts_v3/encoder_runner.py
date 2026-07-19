"""Label-free, resumable encoder stage for isolated ViTTrace v3 caches.

The mandatory route renders line windows and calls OpenCLIP exactly once per
encoder batch.  It stores the model's true global image embedding, the base
patch tokens, dynamically derived P/M/L tokens, and a separate patch-mean
sidecar.  Frozen caches are read only by the explicit B16 parity-smoke mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from measure_vit4ts.cache import (
    TOKEN_FILE,
    TOKEN_MANIFEST,
    compute_renderer_sha256,
    freeze_model,
    hash_model_state,
    load_clip_cache,
    make_cache_key,
    verify_vendor_sha,
)
from measure_vit4ts.data import SeriesData, make_windows, released_preprocess
from measure_vit4ts.full_manifest import EXPECTED_SERIES, FullSeriesRecord, load_manifest
from measure_vit4ts.renderers import render_official_trace

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


SCHEMA_VERSION = 1
EXPECTED_STAGE = "vittrace_ablation_full_v3"
DEFAULT_CONFIG = Path("configs/vittrace_ablation_full_v3.yaml")
PATCH_MEAN_FILE = "patch_mean_tokens.npy"
PATCH_PARITY_SCHEMA = 1
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_SOURCE_FILES = (
    ("v3", "dynamic_cache.py"),
    ("v3", "encoder_runner.py"),
    ("legacy", "cache.py"),
    ("legacy", "data.py"),
    ("legacy", "geometry.py"),
    ("legacy", "renderers.py"),
)


@dataclass(frozen=True)
class EncoderVariant:
    key: str
    representation: str
    model_key: str
    model_name: str
    pretrained: str
    image_size: tuple[int, int]
    patch_size: int
    window: int
    stride: int
    batch_size: int

    def __post_init__(self) -> None:
        if not _SAFE_KEY.fullmatch(self.key):
            raise ValueError("variant key must be a safe nonempty identifier")
        for value in (self.representation, self.model_key, self.model_name, self.pretrained):
            if not value:
                raise ValueError("variant identity strings must be nonempty")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("variant image size must be positive")
        if min(self.patch_size, self.window, self.stride, self.batch_size) <= 0:
            raise ValueError("variant integer parameters must be positive")
        if self.image_size[0] % self.patch_size or self.image_size[1] % self.patch_size:
            raise ValueError("image size must be divisible by the model patch size")


@dataclass(frozen=True)
class LoadedEncoder:
    adapter: torch.nn.Module
    model_sha256: str
    open_clip_version: str
    open_clip_module: str


class _OpenClipAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model


class PatchParityError(RuntimeError):
    """Raised after the complete patch-parity diagnosis has been persisted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest().upper()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest().upper()


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
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError("patch-mean sidecar must be finite float32 [N,D]")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256(path)


def encoder_source_sha256(package_root: Path | None = None) -> str:
    v3 = Path(package_root) if package_root else Path(__file__).resolve().parent
    legacy = v3.parent / "measure_vit4ts"
    digest = hashlib.sha256()
    for namespace, name in _SOURCE_FILES:
        path = (v3 if namespace == "v3" else legacy) / name
        encoded = f"{namespace}/{name}".encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], str, tuple[FullSeriesRecord, ...], str]:
    raw = Path(path).read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("stage") != EXPECTED_STAGE:
        raise ValueError(f"encoder stage requires stage={EXPECTED_STAGE}")
    manifest_path = Path(config["manifest"]["path"])
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != str(config["manifest"]["sha256"]).upper():
        raise ValueError("frozen manifest SHA256 mismatch")
    _, records = load_manifest(manifest_path)
    if len(records) != EXPECTED_SERIES:
        raise ValueError("encoder stage requires the complete 492-series manifest")
    return config, hashlib.sha256(raw).hexdigest().upper(), records, manifest_sha


def resolve_variant(
    config: Mapping[str, Any],
    *,
    representation: str | None = None,
    model_key: str = "B16",
    window: int | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    variant_key: str | None = None,
) -> EncoderVariant:
    defaults = config["defaults"]
    rep = str(representation or defaults["representation"])
    representations = tuple(str(value) for value in config["grid"]["representations"])
    if rep not in representations:
        raise ValueError(f"unregistered representation {rep!r}")
    backbones = {str(item["key"]): item for item in config["grid"]["backbones"]}
    if model_key not in backbones:
        raise ValueError(f"unregistered model key {model_key!r}")
    backbone = backbones[model_key]
    width = int(window if window is not None else defaults["window"])
    step = int(stride if stride is not None else defaults["stride"])
    batch = int(batch_size if batch_size is not None else config["runtime"]["batch_size"])
    registered_windows = {int(value) for value in config["grid"]["windows"]}
    if width not in registered_windows:
        raise ValueError("window is outside the registered encoder grid")
    if width == 240:
        allowed_steps = {int(value) for value in config["grid"]["strides_w240"]}
    else:
        fraction = float(config["grid"]["window_stride_fraction"])
        product = width * fraction
        if not float(product).is_integer():
            raise ValueError("window stride fraction does not yield an integer")
        allowed_steps = {int(product)}
    if step not in allowed_steps:
        raise ValueError(f"stride {step} is not registered for W={width}")
    name = str(backbone["model_name"])
    pretrained = str(backbone["pretrained"])
    patch = int(backbone["patch_size"])
    image = tuple(int(value) for value in defaults["image_size"])
    key = variant_key or f"{rep}_{model_key}_W{width}_S{step}_B{batch}"
    return EncoderVariant(
        key,
        rep,
        model_key,
        name,
        pretrained,
        image,
        patch,
        width,
        step,
        batch,
    )


def variant_sha256(variant: EncoderVariant, config_sha256: str) -> str:
    return _payload_sha256(
        {"variant": {**asdict(variant), "image_size": list(variant.image_size)},
         "config_sha256": str(config_sha256).upper()}
    )


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
        raise ValueError("series IDs must be unique manifest entries")
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


def _safety_check(config: Mapping[str, Any]) -> None:
    c_free = shutil.disk_usage("C:\\").free / (1024**3)
    d_free = shutil.disk_usage("D:\\").free / (1024**3)
    ram_free = psutil.virtual_memory().available / (1024**3)
    if c_free < float(config["runtime"]["c_drive_floor_gib"]):
        raise RuntimeError("C drive is below the registered safety floor")
    if d_free < float(config["runtime"]["d_drive_floor_gib"]):
        raise RuntimeError("D drive is below the registered safety floor")
    if ram_free < float(config["runtime"]["available_ram_floor_gib"]):
        raise RuntimeError("available RAM is below the registered safety floor")


def _load_series(record: FullSeriesRecord, data_root: Path) -> SeriesData:
    path = Path(data_root) / record.relative_path
    actual_sha = _sha256(path)
    if actual_sha != record.expected_sha256.upper():
        raise ValueError(f"data SHA256 mismatch for {record.series_id}")
    frame = pd.read_csv(path, usecols=["timestamp", "value"])
    if frame.empty or len(frame) != record.expected_length:
        raise ValueError(f"series length mismatch for {record.series_id}")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_numeric(frame["timestamp"], errors="raise").to_numpy()
    values = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(timestamps.astype(np.float64, copy=False)).all() or not np.isfinite(values).all():
        raise ValueError("series contains non-finite timestamp/value data")
    duplicate = bool(pd.Index(timestamps).has_duplicates)
    if duplicate != record.duplicate_timestamps:
        raise ValueError("duplicate-timestamp identity differs from the manifest")
    return SeriesData(
        series_id=record.series_id,
        group=record.track,
        timestamps=np.ascontiguousarray(timestamps),
        values=np.ascontiguousarray(values, dtype=np.float64),
        data_sha256=actual_sha.lower(),
    )


def render_variant_windows(
    windows: np.ndarray, variant: EncoderVariant
) -> tuple[np.ndarray, tuple[Any, ...]]:
    if variant.representation != "line":
        raise NotImplementedError(
            f"representation {variant.representation!r} has a distinct cache key "
            "but no registered renderer implementation"
        )
    outputs = tuple(
        render_official_trace(
            window,
            image_size=variant.image_size,
            expected_length=variant.window,
        )
        for window in np.asarray(windows)
    )
    images = np.ascontiguousarray(
        np.stack([item.image for item in outputs]), dtype=np.float32
    )
    return images, tuple(item.geometry for item in outputs)


def load_openclip_encoder(
    config: Mapping[str, Any], variant: EncoderVariant, device: torch.device
) -> LoadedEncoder:
    vendor_root = Path(config["vendor"]["root"])
    verify_vendor_sha(vendor_root, str(config["vendor"]["commit"]))
    module = importlib.import_module("open_clip")
    model, _, _ = module.create_model_and_transforms(
        variant.model_name,
        pretrained=variant.pretrained,
        vision_cfg={"output_tokens": True},
    )
    adapter = _OpenClipAdapter(model.to(device))
    freeze_model(adapter)
    model_sha = hash_model_state(adapter).upper()
    if variant.model_key == "B16" and variant.pretrained == "openai":
        expected = str(config["vendor"]["default_model_sha256"]).upper()
        if model_sha != expected:
            raise RuntimeError(
                f"B16 model-state SHA256 mismatch: expected {expected}, got {model_sha}"
            )
    return LoadedEncoder(
        adapter,
        model_sha,
        str(getattr(module, "__version__", "unknown")),
        str(Path(module.__file__).resolve()),
    )


def patch_mean_tokens(cache: DynamicTokenCache) -> np.ndarray:
    return np.ascontiguousarray(cache.patch_tokens.mean(axis=1), dtype=np.float32)


def compare_token_arrays(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    left = np.asarray(actual)
    right = np.asarray(reference)
    if left.shape != right.shape:
        return {
            "passed": False,
            "shape_match": False,
            "actual_shape": list(left.shape),
            "reference_shape": list(right.shape),
            "max_abs_error": None,
            "max_rel_error": None,
            "mismatch_count": None,
            "first_mismatch": None,
        }
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    mismatch = difference != 0.0
    first = np.argwhere(mismatch)
    denominator = np.maximum(np.abs(right.astype(np.float64)), np.finfo(np.float32).tiny)
    return {
        "passed": bool(not mismatch.any()),
        "shape_match": True,
        "actual_shape": list(left.shape),
        "reference_shape": list(right.shape),
        "max_abs_error": float(np.max(difference, initial=0.0)),
        "max_rel_error": float(np.max(difference / denominator, initial=0.0)),
        "mismatch_count": int(mismatch.sum()),
        "first_mismatch": first[0].tolist() if len(first) else None,
    }


def _frozen_patch_cache(
    config: Mapping[str, Any], record: FullSeriesRecord
) -> tuple[Any, Path, Mapping[str, Any]]:
    root = (
        Path(config["frozen_inputs"]["coordinate_cache_root"])
        / record.series_id
        / "released"
    )
    candidates: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(root.glob(f"*/{TOKEN_MANIFEST}")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = payload.get("key", {})
        if (
            key.get("series_id") == record.series_id
            and str(key.get("data_sha256", "")).upper() == record.expected_sha256.upper()
            and key.get("renderer") == "released"
            and key.get("model_name") == "ViT-B-16"
            and int(key.get("patch_size", -1)) == 16
            and tuple(key.get("image_size", ())) == (224, 224)
        ):
            candidates.append((path, payload))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one frozen B16 patch cache for {record.series_id}, found {len(candidates)}"
        )
    manifest_path, payload = candidates[0]
    key = payload["key"]
    expected = make_cache_key(
        series_id=record.series_id,
        data_sha256=record.expected_sha256,
        renderer="released",
        renderer_sha256=str(key["renderer_sha256"]),
        vendor_commit=str(key["vendor_commit"]),
        clip_weight_sha256=str(key["clip_weight_sha256"]),
        model_name=str(key["model_name"]),
        patch_size=int(key["patch_size"]),
        image_size=tuple(int(value) for value in key["image_size"]),
    )
    return load_clip_cache(manifest_path.parent, expected), manifest_path, payload


def diagnose_frozen_patch_parity(
    config: Mapping[str, Any],
    record: FullSeriesRecord,
    variant: EncoderVariant,
    dynamic: DynamicTokenCache,
    dynamic_dir: Path,
    model_sha256: str,
) -> dict[str, Any]:
    if not (
        variant.representation == "line"
        and variant.model_key == "B16"
        and variant.model_name == "ViT-B-16"
        and variant.pretrained == "openai"
        and variant.image_size == (224, 224)
        and variant.patch_size == 16
        and variant.window == 240
        and variant.stride == 60
        and variant.batch_size == 64
    ):
        raise ValueError("frozen patch parity is registered only for line B16 W240 S60 B64")
    frozen, frozen_manifest, frozen_payload = _frozen_patch_cache(config, record)
    comparison = compare_token_arrays(dynamic.patch_tokens, frozen.patch_tokens)
    renderer_match = (
        dynamic.key.renderer_sha256.upper()
        == str(frozen_payload["key"]["renderer_sha256"]).upper()
    )
    model_match = model_sha256.upper() == str(
        frozen_payload["key"]["clip_weight_sha256"]
    ).upper()
    patch_mean = patch_mean_tokens(dynamic)
    same_embedding_dim = dynamic.global_tokens.shape[1] == patch_mean.shape[1]
    global_vs_patch_mean = (
        compare_token_arrays(dynamic.global_tokens, patch_mean)
        if same_embedding_dim
        else {
            "passed": False,
            "shape_match": False,
            "actual_shape": list(dynamic.global_tokens.shape),
            "reference_shape": list(patch_mean.shape),
            "reason": "true global and patch mean have distinct embedding dimensions",
        }
    )
    return {
        "schema_version": PATCH_PARITY_SCHEMA,
        "created_at": _utc_now(),
        "series_id": record.series_id,
        "variant": asdict(variant),
        "passed": bool(comparison["passed"] and renderer_match and model_match),
        "patch_comparison": comparison,
        "renderer_sha256_match": renderer_match,
        "model_sha256_match": model_match,
        "dynamic_cache_dir": str(dynamic_dir.resolve()),
        "dynamic_cache_sha256": _sha256(dynamic_dir / CACHE_FILE),
        "frozen_cache_dir": str(frozen_manifest.parent.resolve()),
        "frozen_cache_sha256": _sha256(frozen_manifest.parent / TOKEN_FILE),
        "dynamic_renderer_sha256": dynamic.key.renderer_sha256,
        "frozen_renderer_sha256": str(frozen_payload["key"]["renderer_sha256"]).upper(),
        "dynamic_model_sha256": model_sha256,
        "frozen_model_sha256": str(frozen_payload["key"]["clip_weight_sha256"]).upper(),
        "global_tokens_shape": list(dynamic.global_tokens.shape),
        "patch_mean_shape": list(patch_mean.shape),
        "global_vs_patch_mean": global_vs_patch_mean,
    }


def _variant_root(config: Mapping[str, Any], variant: EncoderVariant, variant_sha: str) -> Path:
    return (
        Path(config["paths"]["output_root"])
        / "encoder_stage"
        / f"{variant.key}__{variant_sha[:12]}"
    )


def _record_path(root: Path, series_id: str) -> Path:
    return root / "records" / f"{series_id}.json"


def _key_from_payload(payload: Mapping[str, Any]) -> DynamicCacheKey:
    values = dict(payload)
    values["image_size"] = tuple(int(value) for value in values["image_size"])
    return DynamicCacheKey(**values)


def _resume_record(
    path: Path,
    *,
    config_sha: str,
    manifest_sha: str,
    source_sha: str,
    variant_sha: str,
    data_sha: str,
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("status") != "PASS"
            or payload.get("config_sha256") != config_sha
            or payload.get("manifest_sha256") != manifest_sha
            or payload.get("encoder_source_sha256") != source_sha
            or payload.get("variant_sha256") != variant_sha
            or str(payload.get("data_sha256", "")).upper() != data_sha.upper()
        ):
            return None
        directory = Path(payload["cache_dir"])
        key = _key_from_payload(payload["cache_key"])
        cache = load_dynamic_cache(directory, key)
        if cache.patch_tokens.shape[0] != int(payload["window_count"]):
            return None
        mean_path = Path(payload["patch_mean_path"])
        if not mean_path.is_file() or _sha256(mean_path) != payload["patch_mean_sha256"]:
            return None
        mean = np.load(mean_path, allow_pickle=False)
        if mean.shape != tuple(payload["patch_mean_shape"]) or mean.dtype != np.float32:
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _failure_path(config: Mapping[str, Any], variant: EncoderVariant, series_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        Path(config["paths"]["failure_root"])
        / "encoder_stage"
        / variant.key
        / series_id
        / f"{stamp}_{uuid.uuid4().hex}.json"
    )


def _run_series(
    config: Mapping[str, Any],
    config_sha: str,
    manifest_sha: str,
    source_sha: str,
    variant: EncoderVariant,
    variant_sha: str,
    root: Path,
    record: FullSeriesRecord,
    encoder: LoadedEncoder,
    device: torch.device,
    *,
    patch_parity: bool,
) -> Mapping[str, Any]:
    started_at = _utc_now()
    wall_started = time.perf_counter()
    series = _load_series(record, Path(config["data"]["root"]))
    batch = make_windows(released_preprocess(series.values), variant.window, variant.stride)
    expected_windows = (record.expected_length - variant.window) // variant.stride + 1
    if batch.values.shape[0] != expected_windows:
        raise RuntimeError("dynamic window count differs from complete-window formula")
    render_started = time.perf_counter()
    images, geometries = render_variant_windows(batch.values, variant)
    renderer_seconds = time.perf_counter() - render_started
    renderer_sha = compute_renderer_sha256(images, geometries).upper()
    key = DynamicCacheKey(
        series_id=record.series_id,
        data_sha256=record.expected_sha256,
        renderer=f"{variant.representation}.batch{variant.batch_size}",
        renderer_sha256=renderer_sha,
        model_name=variant.model_name,
        pretrained=variant.pretrained,
        model_sha256=encoder.model_sha256,
        image_size=variant.image_size,
        patch_size=variant.patch_size,
        window=variant.window,
        stride=variant.stride,
    )
    cache_root = root / "caches"
    expected_dir = cache_root / key.series_id / cache_digest(key)
    encode_started = time.perf_counter()
    cache = encode_dynamic_tokens(
        encoder.adapter,
        images,
        key,
        batch_size=variant.batch_size,
        device=device,
    )
    actual_dir = save_dynamic_cache(cache, cache_root)
    if actual_dir != expected_dir:
        raise RuntimeError("dynamic cache directory identity mismatch")
    patch_mean = patch_mean_tokens(cache)
    mean_path = actual_dir / PATCH_MEAN_FILE
    mean_sha = _atomic_npy(mean_path, patch_mean)
    encode_seconds = time.perf_counter() - encode_started
    parity_payload: Mapping[str, Any] | None = None
    if patch_parity:
        parity_payload = diagnose_frozen_patch_parity(
            config, record, variant, cache, actual_dir, encoder.model_sha256
        )
        parity_path = root / "patch_parity" / f"{record.series_id}.json"
        _atomic_json(parity_path, parity_payload)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if parity_payload is None or parity_payload["passed"] else "FAIL",
        "passed": bool(parity_payload is None or parity_payload["passed"]),
        "series_id": record.series_id,
        "family": record.track,
        "subgroup": record.paper_group,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "encoder_source_sha256": source_sha,
        "variant": {**asdict(variant), "image_size": list(variant.image_size)},
        "variant_sha256": variant_sha,
        "data_sha256": record.expected_sha256.upper(),
        "window_count": int(batch.values.shape[0]),
        "encoder_calls": int(math.ceil(batch.values.shape[0] / variant.batch_size)),
        "cache_dir": str(actual_dir.resolve()),
        "cache_key": {**asdict(key), "image_size": list(key.image_size)},
        "cache_file_sha256": _sha256(actual_dir / CACHE_FILE),
        "cache_manifest_sha256": _sha256(actual_dir / CACHE_MANIFEST),
        "renderer_sha256": renderer_sha,
        "model_sha256": encoder.model_sha256,
        "patch_grid": list(cache.patch_grid),
        "global_tokens_shape": list(cache.global_tokens.shape),
        "global_tokens_sha256": _array_sha256(cache.global_tokens),
        "patch_tokens_shape": list(cache.patch_tokens.shape),
        "patch_tokens_sha256": _array_sha256(cache.patch_tokens),
        "patch_mean_path": str(mean_path.resolve()),
        "patch_mean_shape": list(patch_mean.shape),
        "patch_mean_sha256": mean_sha,
        "patch_mean_array_sha256": _array_sha256(patch_mean),
        "mid_tokens_shape": list(cache.mid_tokens.shape),
        "large_tokens_shape": list(cache.large_tokens.shape),
        "renderer_seconds": float(renderer_seconds),
        "encode_save_seconds": float(encode_seconds),
        "wall_seconds": float(time.perf_counter() - wall_started),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "open_clip_version": encoder.open_clip_version,
        "open_clip_module": encoder.open_clip_module,
        "patch_parity": parity_payload,
    }
    _atomic_json(_record_path(root, record.series_id), payload)
    if not payload["passed"]:
        raise PatchParityError(
            f"frozen patch parity failed for {record.series_id}; diagnosis was preserved"
        )
    return payload


def _write_summary(
    root: Path,
    records: Sequence[FullSeriesRecord],
    *,
    config_sha: str,
    manifest_sha: str,
    source_sha: str,
    variant: EncoderVariant,
    variant_sha: str,
) -> Path:
    rows: list[dict[str, Any]] = []
    for record in records:
        path = _record_path(root, record.series_id)
        payload = _resume_record(
            path,
            config_sha=config_sha,
            manifest_sha=manifest_sha,
            source_sha=source_sha,
            variant_sha=variant_sha,
            data_sha=record.expected_sha256,
        )
        rows.append(
            {
                "series_id": record.series_id,
                "family": record.track,
                "subgroup": record.paper_group,
                "status": "PASS" if payload is not None else "NOT_RUN",
                "window_count": int(payload["window_count"]) if payload else None,
                "encoder_calls": int(payload["encoder_calls"]) if payload else None,
                "cache_dir": payload["cache_dir"] if payload else "",
            }
        )
    completed = sum(row["status"] == "PASS" for row in rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "COMPLETE" if completed == EXPECTED_SERIES else "INCOMPLETE",
        "expected_series": EXPECTED_SERIES,
        "completed_series": completed,
        "not_run_series": EXPECTED_SERIES - completed,
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "encoder_source_sha256": source_sha,
        "variant": {**asdict(variant), "image_size": list(variant.image_size)},
        "variant_sha256": variant_sha,
        "series": rows,
    }
    path = root / "encoder_stage_status.json"
    _atomic_json(path, payload)
    return path


def run_encoder_stage(
    config_path: Path,
    *,
    representation: str | None = None,
    model_key: str = "B16",
    window: int | None = None,
    stride: int | None = None,
    batch_size: int | None = None,
    variant_key: str | None = None,
    smoke: bool = False,
    all_series: bool = False,
    series_ids: Sequence[str] = (),
    approved_bulk: bool = False,
    device_name: str | None = None,
    retry: bool = False,
    patch_parity: bool = False,
) -> Path:
    config, config_sha, records, manifest_sha = _load_config(config_path)
    variant = resolve_variant(
        config,
        representation=representation,
        model_key=model_key,
        window=window,
        stride=stride,
        batch_size=batch_size,
        variant_key=variant_key,
    )
    selected = select_records(
        records,
        smoke=smoke,
        all_series=all_series,
        series_ids=series_ids,
        approved_bulk=approved_bulk,
    )
    if patch_parity and (len(selected) != 1 or all_series):
        raise ValueError("--patch-parity requires exactly one smoke/explicit series")
    source_sha = encoder_source_sha256()
    variant_sha = variant_sha256(variant, config_sha)
    root = _variant_root(config, variant, variant_sha)
    missing = []
    for record in selected:
        resumed = None if retry else _resume_record(
            _record_path(root, record.series_id),
            config_sha=config_sha,
            manifest_sha=manifest_sha,
            source_sha=source_sha,
            variant_sha=variant_sha,
            data_sha=record.expected_sha256,
        )
        if resumed is None or (patch_parity and resumed.get("patch_parity") is None):
            missing.append(record)
    if not missing:
        return _write_summary(
            root,
            records,
            config_sha=config_sha,
            manifest_sha=manifest_sha,
            source_sha=source_sha,
            variant=variant,
            variant_sha=variant_sha,
        )
    _safety_check(config)
    device = torch.device(device_name or str(config["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA encoder stage was requested but CUDA is unavailable")
    encoder = load_openclip_encoder(config, variant, device)
    errors: list[tuple[str, Exception]] = []
    for record in missing:
        try:
            _run_series(
                config,
                config_sha,
                manifest_sha,
                source_sha,
                variant,
                variant_sha,
                root,
                record,
                encoder,
                device,
                patch_parity=patch_parity,
            )
        except Exception as exc:
            _atomic_json(
                _failure_path(config, variant, record.series_id),
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": _utc_now(),
                    "series_id": record.series_id,
                    "variant": {**asdict(variant), "image_size": list(variant.image_size)},
                    "variant_sha256": variant_sha,
                    "config_sha256": config_sha,
                    "manifest_sha256": manifest_sha,
                    "encoder_source_sha256": source_sha,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            errors.append((record.series_id, exc))
            if not all_series:
                _write_summary(
                    root,
                    records,
                    config_sha=config_sha,
                    manifest_sha=manifest_sha,
                    source_sha=source_sha,
                    variant=variant,
                    variant_sha=variant_sha,
                )
                raise
    status = _write_summary(
        root,
        records,
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        source_sha=source_sha,
        variant=variant,
        variant_sha=variant_sha,
    )
    if errors:
        raise RuntimeError(
            f"encoder stage preserved {len(errors)} failures; first={errors[0][0]}: {errors[0][1]}"
        )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--representation")
    parser.add_argument("--model-key", default="B16")
    parser.add_argument("--window", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--variant-key")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", dest="all_series", action="store_true")
    parser.add_argument("--series-id", action="append", default=[])
    parser.add_argument("--approved-bulk", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--patch-parity", action="store_true")
    args = parser.parse_args(argv)
    path = run_encoder_stage(
        args.config,
        representation=args.representation,
        model_key=args.model_key,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
        variant_key=args.variant_key,
        smoke=args.smoke,
        all_series=args.all_series,
        series_ids=args.series_id,
        approved_bulk=args.approved_bulk,
        device_name=args.device,
        retry=args.retry,
        patch_parity=args.patch_parity,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
