"""Hash-bound frozen-CLIP caches and official ViT4TS anomaly-map recovery."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import sha256_file


TOKEN_FILE = "clip_tokens.npz"
TOKEN_MANIFEST = "clip_tokens.json"
MAP_FILE = "anomaly_maps.npy"
MAP_MANIFEST = "anomaly_maps.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClipCacheKey:
    series_id: str
    data_sha256: str
    renderer: Literal["released", "distance", "supersample4"]
    renderer_sha256: str
    vendor_commit: str
    clip_weight_sha256: str
    model_name: str
    patch_size: int
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ClipTokenCache:
    key: ClipCacheKey
    large_tokens: np.ndarray
    mid_tokens: np.ndarray
    patch_tokens: np.ndarray
    large_mask: np.ndarray
    mid_mask: np.ndarray


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _key_payload(key: ClipCacheKey) -> dict[str, Any]:
    payload = asdict(key)
    payload["image_size"] = list(key.image_size)
    return payload


def _key_from_payload(payload: Mapping[str, Any]) -> ClipCacheKey:
    return ClipCacheKey(
        series_id=str(payload["series_id"]),
        data_sha256=str(payload["data_sha256"]),
        renderer=str(payload["renderer"]),
        renderer_sha256=str(payload["renderer_sha256"]),
        vendor_commit=str(payload["vendor_commit"]),
        clip_weight_sha256=str(payload["clip_weight_sha256"]),
        model_name=str(payload["model_name"]),
        patch_size=int(payload["patch_size"]),
        image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
    )


def cache_key_digest(key: ClipCacheKey) -> str:
    """Return a deterministic directory key for a complete cache identity."""

    return sha256(_canonical_json(_key_payload(key))).hexdigest()


def make_cache_key(
    *,
    series_id: str,
    data_sha256: str,
    renderer: Literal["released", "distance", "supersample4"],
    renderer_sha256: str,
    vendor_commit: str,
    clip_weight_sha256: str,
    model_name: str = "ViT-B-16",
    patch_size: int = 16,
    image_size: tuple[int, int] = (224, 224),
) -> ClipCacheKey:
    """Construct and validate a scientific cache identity."""

    key = ClipCacheKey(
        series_id=str(series_id),
        data_sha256=str(data_sha256).lower(),
        renderer=renderer,
        renderer_sha256=str(renderer_sha256).lower(),
        vendor_commit=str(vendor_commit).lower(),
        clip_weight_sha256=str(clip_weight_sha256).lower(),
        model_name=str(model_name),
        patch_size=int(patch_size),
        image_size=(int(image_size[0]), int(image_size[1])),
    )
    if not key.series_id:
        raise ValueError("series_id must be non-empty")
    if key.renderer not in {"released", "distance", "supersample4"}:
        raise ValueError(f"unsupported renderer {key.renderer!r}")
    for name, digest in (
        ("data_sha256", key.data_sha256),
        ("renderer_sha256", key.renderer_sha256),
        ("clip_weight_sha256", key.clip_weight_sha256),
    ):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{name} must be a 64-character hex digest")
    if len(key.vendor_commit) != 40 or any(
        char not in "0123456789abcdef" for char in key.vendor_commit
    ):
        raise ValueError("vendor_commit must be a 40-character Git SHA")
    if key.model_name != "ViT-B-16" or key.patch_size != 16:
        raise ValueError("the frozen route requires ViT-B-16 with patch_size=16")
    if key.image_size != (224, 224):
        raise ValueError("the frozen route requires image_size=(224, 224)")
    return key


def verify_vendor_sha(vendor_root: Path, expected_commit: str) -> str:
    """Fail unless vendor_root is exactly the registered immutable commit."""

    root = Path(vendor_root).resolve(strict=True)
    expected = str(expected_commit).strip().lower()
    if len(expected) != 40:
        raise ValueError("expected_commit must be a full 40-character SHA")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip().lower()
    if actual != expected:
        raise RuntimeError(f"vendor commit mismatch: expected {expected}, got {actual}")
    return actual


def hash_model_state(model: torch.nn.Module) -> str:
    """Hash a model state without constructing one giant byte string."""

    digest = sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    """Put a model in eval mode and permanently disable parameter gradients."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze CLIP model")
    return model


def load_frozen_clip(
    vendor_root: Path,
    expected_commit: str,
    device: torch.device | str,
    model_name: str = "ViT-B-16",
) -> tuple[torch.nn.Module, str]:
    """Import CLIP_AD from the verified vendor and return frozen weights."""

    root = Path(vendor_root).resolve(strict=True)
    verify_vendor_sha(root, expected_commit)
    if model_name != "ViT-B-16":
        raise ValueError("only the frozen ViT-B-16 model is permitted")
    source_root = root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"vendor source directory missing: {source_root}")

    existing_models = sys.modules.get("models")
    if existing_models is not None:
        existing_path = Path(getattr(existing_models, "__file__", "") or ".").resolve()
        if source_root not in existing_path.parents:
            raise RuntimeError(
                "a non-vendor top-level 'models' package is already imported"
            )
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    module = importlib.import_module("models.clip_vision")
    module_path = Path(module.__file__).resolve()
    if source_root not in module_path.parents:
        raise RuntimeError("resolved CLIP_AD outside the verified vendor")
    model = module.CLIP_AD(model_name=model_name, device=torch.device(device))
    freeze_model(model)
    return model, hash_model_state(model)


def _as_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(np.asarray(array))


def _update_array_hash(digest: Any, array: torch.Tensor | np.ndarray) -> None:
    value = _as_numpy(array)
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_json(list(value.shape)))
    digest.update(memoryview(value).cast("B"))


def compute_renderer_sha256(
    images: torch.Tensor | np.ndarray,
    geometry: Any | Sequence[Any] | None = None,
) -> str:
    """Hash rendered pixels and, when provided, ownership/kernel geometry."""

    digest = sha256()
    _update_array_hash(digest, images)
    if geometry is None:
        return digest.hexdigest()
    geometries: Iterable[Any]
    if isinstance(geometry, (list, tuple)):
        geometries = geometry
    else:
        geometries = (geometry,)
    for item in geometries:
        for attribute in ("vertices", "kernel", "time_coordinate"):
            if not hasattr(item, attribute):
                raise TypeError(f"geometry lacks required attribute {attribute!r}")
            _update_array_hash(digest, getattr(item, attribute))
    return digest.hexdigest()


def _validate_cache(cache: ClipTokenCache) -> None:
    arrays = (
        cache.large_tokens,
        cache.mid_tokens,
        cache.patch_tokens,
        cache.large_mask,
        cache.mid_mask,
    )
    if any(not isinstance(array, np.ndarray) for array in arrays):
        raise TypeError("all cache fields must be numpy arrays")
    large, mid, patch = (
        cache.large_tokens,
        cache.mid_tokens,
        cache.patch_tokens,
    )
    if large.ndim != 3 or mid.ndim != 3 or patch.ndim != 3:
        raise ValueError("token arrays must have shape [N, K, D]")
    if large.shape[0] == 0 or not (
        large.shape[0] == mid.shape[0] == patch.shape[0]
    ):
        raise ValueError("token arrays must share a nonzero window count")
    if (large.shape[1], mid.shape[1], patch.shape[1]) != (144, 169, 196):
        raise ValueError("unexpected ViT-B/16 multi-scale token counts")
    if not (large.shape[2] == mid.shape[2] == patch.shape[2] == 768):
        raise ValueError("unexpected ViT-B/16 embedding dimension")
    if any(array.dtype != np.float32 for array in (large, mid, patch)):
        raise ValueError("tokens must be canonical float32 CPU arrays")
    if cache.large_mask.shape != (9, 144) or cache.mid_mask.shape != (4, 169):
        raise ValueError("unexpected frozen patch-mask shapes")
    if cache.large_mask.dtype != np.int64 or cache.mid_mask.dtype != np.int64:
        raise ValueError("patch masks must be int64")
    if any(not np.all(np.isfinite(array)) for array in (large, mid, patch)):
        raise ValueError("token cache contains non-finite values")


def encode_frozen_clip(
    model: torch.nn.Module,
    images: torch.Tensor | np.ndarray,
    key: ClipCacheKey,
    batch_size: int,
    device: torch.device | str,
) -> ClipTokenCache:
    """Encode one complete renderer output exactly once in frozen eval mode."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image_tensor = torch.as_tensor(images, dtype=torch.float32)
    expected_shape = (3, key.image_size[0], key.image_size[1])
    if image_tensor.ndim != 4 or tuple(image_tensor.shape[1:]) != expected_shape:
        raise ValueError(f"images must have shape [N, {expected_shape}]")
    if image_tensor.shape[0] == 0 or not torch.isfinite(image_tensor).all():
        raise ValueError("images must be nonempty and finite")

    freeze_model(model)
    destination = torch.device(device)
    window_count = int(image_tensor.shape[0])
    # Allocate each canonical cache array once.  Retaining every batch and then
    # concatenating duplicates the complete token cache at peak memory, which
    # can exhaust the Windows commit limit on long series even though the final
    # cache itself fits comfortably.
    large_tokens = np.empty((window_count, 144, 768), dtype=np.float32)
    mid_tokens = np.empty((window_count, 169, 768), dtype=np.float32)
    patch_tokens = np.empty((window_count, 196, 768), dtype=np.float32)
    large_mask_ref: np.ndarray | None = None
    mid_mask_ref: np.ndarray | None = None

    with torch.inference_mode():
        for start in range(0, image_tensor.shape[0], batch_size):
            stop = min(start + batch_size, window_count)
            batch = image_tensor[start:stop].to(
                destination, non_blocking=True
            )
            outputs = model.encode_image(batch, key.patch_size)
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
                raise RuntimeError("vendor encode_image returned an unexpected tuple")
            large, mid, patch, _, large_mask, mid_mask = outputs
            if patch.ndim == 4 and patch.shape[2] == 1:
                patch = patch.squeeze(2)
            large_np = np.ascontiguousarray(
                large.detach().float().cpu().numpy(), dtype=np.float32
            )
            mid_np = np.ascontiguousarray(
                mid.detach().float().cpu().numpy(), dtype=np.float32
            )
            patch_np = np.ascontiguousarray(
                patch.detach().float().cpu().numpy(), dtype=np.float32
            )
            large_mask_np = np.ascontiguousarray(
                torch.as_tensor(large_mask).long().cpu().numpy(), dtype=np.int64
            )
            mid_mask_np = np.ascontiguousarray(
                torch.as_tensor(mid_mask).long().cpu().numpy(), dtype=np.int64
            )
            if large_mask_ref is None:
                large_mask_ref, mid_mask_ref = large_mask_np, mid_mask_np
            elif not np.array_equal(large_mask_ref, large_mask_np) or not np.array_equal(
                mid_mask_ref, mid_mask_np
            ):
                raise RuntimeError("frozen patch masks changed between batches")
            expected_batch = stop - start
            if large_np.shape != (expected_batch, 144, 768):
                raise RuntimeError("unexpected large-token batch shape")
            if mid_np.shape != (expected_batch, 169, 768):
                raise RuntimeError("unexpected mid-token batch shape")
            if patch_np.shape != (expected_batch, 196, 768):
                raise RuntimeError("unexpected patch-token batch shape")
            large_tokens[start:stop] = large_np
            mid_tokens[start:stop] = mid_np
            patch_tokens[start:stop] = patch_np

    if large_mask_ref is None or mid_mask_ref is None:
        raise RuntimeError("no CLIP batches were encoded")
    cache = ClipTokenCache(
        key=key,
        large_tokens=large_tokens,
        mid_tokens=mid_tokens,
        patch_tokens=patch_tokens,
        large_mask=large_mask_ref,
        mid_mask=mid_mask_ref,
    )
    _validate_cache(cache)
    return cache


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical_json(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        # ZIP_DEFLATED creates a second compressor output buffer and can fail
        # under a tight Windows commit limit even when the final token arrays
        # fit.  NPZ with stored NPY members is equally lossless and is loaded
        # by the same np.load path without changing any score computation.
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_clip_cache(cache: ClipTokenCache, cache_root: Path) -> Path:
    """Atomically persist a renderer-specific token cache and hash manifest."""

    _validate_cache(cache)
    directory = (
        Path(cache_root)
        / cache.key.series_id
        / cache.key.renderer
        / cache_key_digest(cache.key)
    )
    token_path = directory / TOKEN_FILE
    _atomic_npz(
        token_path,
        {
            "large_tokens": cache.large_tokens,
            "mid_tokens": cache.mid_tokens,
            "patch_tokens": cache.patch_tokens,
            "large_mask": cache.large_mask,
            "mid_mask": cache.mid_mask,
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "key": _key_payload(cache.key),
        "file": TOKEN_FILE,
        "sha256": sha256_file(token_path),
        "shapes": {
            name: list(getattr(cache, name).shape)
            for name in (
                "large_tokens",
                "mid_tokens",
                "patch_tokens",
                "large_mask",
                "mid_mask",
            )
        },
    }
    _atomic_json(directory / TOKEN_MANIFEST, manifest)
    return directory


def load_clip_cache(cache_dir: Path, expected_key: ClipCacheKey) -> ClipTokenCache:
    """Load a cache only after exact identity, hash, dtype, and shape checks."""

    directory = Path(cache_dir)
    with (directory / TOKEN_MANIFEST).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported token-cache schema")
    actual_key = _key_from_payload(manifest["key"])
    if actual_key != expected_key:
        raise ValueError("token-cache key mismatch")
    token_path = directory / str(manifest.get("file"))
    if sha256_file(token_path) != manifest.get("sha256"):
        raise ValueError("token-cache file hash mismatch")
    with np.load(token_path, allow_pickle=False) as archive:
        required = {
            "large_tokens",
            "mid_tokens",
            "patch_tokens",
            "large_mask",
            "mid_mask",
        }
        if set(archive.files) != required:
            raise ValueError("token-cache archive has unexpected fields")
        cache = ClipTokenCache(
            key=actual_key,
            large_tokens=np.ascontiguousarray(archive["large_tokens"]),
            mid_tokens=np.ascontiguousarray(archive["mid_tokens"]),
            patch_tokens=np.ascontiguousarray(archive["patch_tokens"]),
            large_mask=np.ascontiguousarray(archive["large_mask"]),
            mid_mask=np.ascontiguousarray(archive["mid_mask"]),
        )
    _validate_cache(cache)
    declared_shapes = manifest.get("shapes", {})
    for name, shape in declared_shapes.items():
        if list(getattr(cache, name).shape) != list(shape):
            raise ValueError(f"token-cache declared shape mismatch for {name}")
    return cache


def _universal_dissimilarity(
    queries: torch.Tensor,
    median_memory: torch.Tensor,
) -> torch.Tensor:
    query_norm = F.normalize(queries, dim=-1)
    memory_norm = F.normalize(median_memory, dim=-1)
    similarity = torch.matmul(query_norm, memory_norm.T)
    return 0.5 * torch.min(1.0 - similarity, dim=2).values


def _vendor_harmonic_aggregation(
    score_size: tuple[int, int, int],
    similarity: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve the vendor's transpose and one-based membership behavior."""

    batch, height, width = score_size
    values = similarity.double()
    transposed = mask.T.long()
    score = torch.zeros((batch, height * width), device=values.device).double()
    for index in range(height * width):
        membership = [
            bool(torch.isin(index + 1, mask_patch)) for mask_patch in transposed
        ]
        selected = torch.tensor(membership, device=values.device)
        count = selected.sum().item()
        harmonic_sum = torch.sum(1.0 / values[:, selected], dim=-1)
        score[:, index] = count / harmonic_sum
    return score.view(batch, height, width)


def build_frozen_anomaly_maps(
    cache: ClipTokenCache,
    device: torch.device | str,
    map_batch_size: int,
) -> np.ndarray:
    """Reconstruct official all-window median-memory universal anomaly maps."""

    _validate_cache(cache)
    if map_batch_size <= 0:
        raise ValueError("map_batch_size must be positive")
    destination = torch.device(device)
    large_all = torch.from_numpy(cache.large_tokens).to(destination)
    mid_all = torch.from_numpy(cache.mid_tokens).to(destination)
    patch_all = torch.from_numpy(cache.patch_tokens).to(destination)
    large_median = torch.median(large_all, dim=0).values
    mid_median = torch.median(mid_all, dim=0).values
    patch_median = torch.median(patch_all, dim=0).values
    large_mask = torch.from_numpy(cache.large_mask).to(destination)
    mid_mask = torch.from_numpy(cache.mid_mask).to(destination)
    maps = np.empty(
        (cache.large_tokens.shape[0], 224, 224),
        dtype=np.float64,
    )

    with torch.inference_mode():
        for start in range(0, large_all.shape[0], map_batch_size):
            stop = min(start + map_batch_size, large_all.shape[0])
            large_d = _universal_dissimilarity(
                large_all[start:stop], large_median
            )
            mid_d = _universal_dissimilarity(
                mid_all[start:stop], mid_median
            )
            patch_d = _universal_dissimilarity(
                patch_all[start:stop], patch_median
            )
            batch = stop - start
            large_map = _vendor_harmonic_aggregation(
                (batch, 14, 14), large_d, large_mask
            )
            mid_map = _vendor_harmonic_aggregation(
                (batch, 14, 14), mid_d, mid_mask
            )
            patch_map = patch_d.reshape(batch, 14, 14).double()
            fused = torch.nan_to_num(
                (large_map + mid_map + patch_map) / 3.0,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            upsampled = F.interpolate(
                fused.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
            ).squeeze(1)
            maps[start:stop] = upsampled.cpu().numpy()
    if maps.shape != (cache.large_tokens.shape[0], 224, 224):
        raise RuntimeError("unexpected anomaly-map shape")
    if not np.all(np.isfinite(maps)):
        raise RuntimeError("anomaly maps contain non-finite values")
    return maps


def write_anomaly_map_cache(
    maps: np.ndarray,
    key: ClipCacheKey,
    cache_dir: Path,
) -> Path:
    """Atomically persist derived maps beside their exact token cache."""

    array = np.ascontiguousarray(np.asarray(maps), dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (224, 224):
        raise ValueError("maps must have shape [N, 224, 224]")
    if array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("maps must be nonempty and finite")
    directory = Path(cache_dir)
    path = directory / MAP_FILE
    _atomic_npy(path, array)
    _atomic_json(
        directory / MAP_MANIFEST,
        {
            "schema_version": SCHEMA_VERSION,
            "key": _key_payload(key),
            "file": MAP_FILE,
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        },
    )
    return path


def load_anomaly_map_cache(
    cache_dir: Path,
    expected_key: ClipCacheKey,
) -> np.ndarray:
    """Load derived maps after exact cache-key and file-hash validation."""

    directory = Path(cache_dir)
    with (directory / MAP_MANIFEST).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported anomaly-map schema")
    if _key_from_payload(manifest["key"]) != expected_key:
        raise ValueError("anomaly-map cache key mismatch")
    path = directory / str(manifest.get("file"))
    if sha256_file(path) != manifest.get("sha256"):
        raise ValueError("anomaly-map file hash mismatch")
    maps = np.load(path, allow_pickle=False)
    maps = np.ascontiguousarray(maps)
    if maps.dtype != np.float64 or list(maps.shape) != manifest.get("shape"):
        raise ValueError("anomaly-map dtype/shape mismatch")
    if maps.ndim != 3 or maps.shape[1:] != (224, 224) or not np.all(
        np.isfinite(maps)
    ):
        raise ValueError("invalid anomaly-map cache")
    return maps
