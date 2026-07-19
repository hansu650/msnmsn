"""Dynamic, hash-bound vision-token caches for ViTTrace v3.

This module is deliberately independent of the frozen B/16 cache schema.  It
captures the *true* global image embedding returned by OpenCLIP, preserves the
base patch tokens, and derives the ViT4TS P/M/L hierarchy from the observed
patch grid.  A cache produced for one renderer/backbone/window tuple is never
silently reused for another tuple.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 1
CACHE_FILE = "vision_tokens_v3.npz"
CACHE_MANIFEST = "vision_tokens_v3.json"


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


@dataclass(frozen=True)
class DynamicCacheKey:
    series_id: str
    data_sha256: str
    renderer: str
    renderer_sha256: str
    model_name: str
    pretrained: str
    model_sha256: str
    image_size: tuple[int, int]
    patch_size: int
    window: int
    stride: int

    def __post_init__(self) -> None:
        if not self.series_id or not self.renderer or not self.model_name:
            raise ValueError("cache identity strings must be nonempty")
        if len(self.image_size) != 2 or min(map(int, self.image_size)) <= 0:
            raise ValueError("image_size must be a positive (height, width) pair")
        if min(int(self.patch_size), int(self.window), int(self.stride)) <= 0:
            raise ValueError("patch_size, window, and stride must be positive")
        for name in ("data_sha256", "renderer_sha256", "model_sha256"):
            value = str(getattr(self, name)).upper()
            if len(value) != 64 or any(c not in "0123456789ABCDEF" for c in value):
                raise ValueError(f"{name} must be a 64-character hexadecimal digest")


@dataclass(frozen=True)
class DynamicTokenCache:
    key: DynamicCacheKey
    patch_grid: tuple[int, int]
    global_tokens: np.ndarray
    patch_tokens: np.ndarray
    mid_tokens: np.ndarray
    large_tokens: np.ndarray
    mid_mask: np.ndarray
    large_mask: np.ndarray


def derive_patch_grid(
    token_count: int,
    image_size: Sequence[int],
    patch_size: int,
) -> tuple[int, int]:
    height, width = map(int, image_size)
    patch = int(patch_size)
    if patch <= 0 or height % patch or width % patch:
        raise ValueError("image dimensions must be divisible by patch_size")
    grid = (height // patch, width // patch)
    if int(token_count) != grid[0] * grid[1]:
        raise ValueError(
            f"observed {token_count} patch tokens but geometry implies {grid}"
        )
    return grid


def stride1_pool_mask(
    patch_grid: Sequence[int], kernel: int
) -> np.ndarray:
    """Return row-major pooling indices with shape ``[kernel^2, outputs]``."""

    grid_h, grid_w = map(int, patch_grid)
    size = int(kernel)
    if min(grid_h, grid_w, size) <= 0 or size > grid_h or size > grid_w:
        raise ValueError("invalid patch grid or pooling kernel")
    columns: list[list[int]] = []
    for top in range(grid_h - size + 1):
        for left in range(grid_w - size + 1):
            columns.append(
                [
                    (top + dy) * grid_w + left + dx
                    for dy in range(size)
                    for dx in range(size)
                ]
            )
    return np.ascontiguousarray(np.asarray(columns, dtype=np.int64).T)


def pool_patch_tokens(tokens: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    if tokens.ndim != 3 or not tokens.is_floating_point():
        raise ValueError("tokens must have shape [B,K,D]")
    indices = torch.as_tensor(mask, dtype=torch.long, device=tokens.device)
    if indices.ndim != 2 or indices.numel() == 0:
        raise ValueError("mask must be nonempty [support,outputs]")
    if int(indices.min()) < 0 or int(indices.max()) >= tokens.shape[1]:
        raise ValueError("pooling mask index outside the patch grid")
    # [B, support, outputs, D] -> [B, outputs, D]
    return tokens[:, indices.reshape(-1), :].reshape(
        tokens.shape[0], indices.shape[0], indices.shape[1], tokens.shape[2]
    ).mean(dim=1)


def _validate_cache(cache: DynamicTokenCache) -> None:
    grid_h, grid_w = map(int, cache.patch_grid)
    count = grid_h * grid_w
    arrays = (
        cache.global_tokens,
        cache.patch_tokens,
        cache.mid_tokens,
        cache.large_tokens,
        cache.mid_mask,
        cache.large_mask,
    )
    if any(not isinstance(a, np.ndarray) for a in arrays):
        raise TypeError("cache arrays must be numpy arrays")
    global_tokens, patch, mid, large = arrays[:4]
    if global_tokens.ndim != 2 or patch.ndim != 3:
        raise ValueError("global/patch arrays must be [N,D] and [N,K,D]")
    if patch.shape[1] != count or patch.shape[0] != global_tokens.shape[0]:
        raise ValueError("patch grid or window count mismatch")
    if not (
        global_tokens.shape[0] == patch.shape[0] == mid.shape[0] == large.shape[0]
        and patch.shape[2] == mid.shape[2] == large.shape[2]
        and global_tokens.shape[1] > 0
    ):
        raise ValueError("token arrays do not share windows or patch embedding dimension")
    if cache.mid_mask.shape[1] != mid.shape[1] or cache.large_mask.shape[1] != large.shape[1]:
        raise ValueError("pooling masks do not match derived token counts")
    if any(a.dtype != np.float32 for a in arrays[:4]):
        raise ValueError("all token arrays must be canonical float32")
    if any(a.dtype != np.int64 for a in arrays[4:]):
        raise ValueError("pooling masks must be int64")
    if any(not np.isfinite(a).all() for a in arrays[:4]):
        raise ValueError("token cache contains non-finite values")


def encode_dynamic_tokens(
    clip_ad: torch.nn.Module,
    images: torch.Tensor | np.ndarray,
    key: DynamicCacheKey,
    *,
    batch_size: int,
    device: torch.device | str,
) -> DynamicTokenCache:
    """Encode once and capture true global plus deterministic P/M/L tokens."""

    if not hasattr(clip_ad, "model") or not hasattr(clip_ad.model, "encode_image"):
        raise TypeError("clip_ad must expose the underlying OpenCLIP model")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    image_tensor = torch.as_tensor(images, dtype=torch.float32)
    expected = (3, *map(int, key.image_size))
    if image_tensor.ndim != 4 or tuple(image_tensor.shape[1:]) != expected:
        raise ValueError(f"images must have shape [N,{expected}]")
    if image_tensor.shape[0] == 0 or not bool(torch.isfinite(image_tensor).all()):
        raise ValueError("images must be nonempty and finite")

    clip_ad.eval()
    for parameter in clip_ad.parameters():
        parameter.requires_grad_(False)
    destination = torch.device(device)
    global_parts: list[np.ndarray] = []
    patch_parts: list[np.ndarray] = []
    grid: tuple[int, int] | None = None
    mid_mask: np.ndarray | None = None
    large_mask: np.ndarray | None = None
    mid_parts: list[np.ndarray] = []
    large_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, image_tensor.shape[0], int(batch_size)):
            batch = image_tensor[start : start + int(batch_size)].to(
                destination, non_blocking=True
            )
            output = clip_ad.model.encode_image(batch)
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("OpenCLIP encode_image must return (global, patches)")
            global_value, patch_value = output
            if global_value.ndim != 2 or patch_value.ndim != 3:
                raise RuntimeError("unexpected OpenCLIP global/patch token shapes")
            if global_value.shape[0] != batch.shape[0] or patch_value.shape[0] != batch.shape[0]:
                raise RuntimeError("OpenCLIP output batch dimension changed")
            expected_cells = (
                int(key.image_size[0]) // int(key.patch_size)
            ) * (
                int(key.image_size[1]) // int(key.patch_size)
            )
            if int(patch_value.shape[1]) == expected_cells + 1:
                # Some OpenCLIP variants include the visual class token in
                # output_tokens.  The P/M/L hierarchy is patch-cell only.
                patch_value = patch_value[:, 1:, :]
            observed = derive_patch_grid(
                int(patch_value.shape[1]), key.image_size, key.patch_size
            )
            if grid is None:
                grid = observed
                mid_mask = stride1_pool_mask(grid, 2)
                large_mask = stride1_pool_mask(grid, 3)
            elif grid != observed:
                raise RuntimeError("patch grid changed between encoder batches")
            mid_value = pool_patch_tokens(patch_value, mid_mask)
            large_value = pool_patch_tokens(patch_value, large_mask)
            global_parts.append(global_value.detach().float().cpu().numpy())
            patch_parts.append(patch_value.detach().float().cpu().numpy())
            mid_parts.append(mid_value.detach().float().cpu().numpy())
            large_parts.append(large_value.detach().float().cpu().numpy())
    if grid is None or mid_mask is None or large_mask is None:
        raise RuntimeError("no encoder batches were produced")
    cache = DynamicTokenCache(
        key=key,
        patch_grid=grid,
        global_tokens=np.ascontiguousarray(np.concatenate(global_parts), dtype=np.float32),
        patch_tokens=np.ascontiguousarray(np.concatenate(patch_parts), dtype=np.float32),
        mid_tokens=np.ascontiguousarray(np.concatenate(mid_parts), dtype=np.float32),
        large_tokens=np.ascontiguousarray(np.concatenate(large_parts), dtype=np.float32),
        mid_mask=mid_mask,
        large_mask=large_mask,
    )
    _validate_cache(cache)
    return cache


def cache_digest(key: DynamicCacheKey) -> str:
    payload = asdict(key)
    payload["image_size"] = list(key.image_size)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def save_dynamic_cache(cache: DynamicTokenCache, root: Path) -> Path:
    _validate_cache(cache)
    directory = Path(root) / cache.key.series_id / cache_digest(cache.key)
    directory.mkdir(parents=True, exist_ok=True)
    token_path = directory / CACHE_FILE
    temporary = token_path.with_name(f".{token_path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            global_tokens=cache.global_tokens,
            patch_tokens=cache.patch_tokens,
            mid_tokens=cache.mid_tokens,
            large_tokens=cache.large_tokens,
            mid_mask=cache.mid_mask,
            large_mask=cache.large_mask,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, token_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "key": {**asdict(cache.key), "image_size": list(cache.key.image_size)},
        "patch_grid": list(cache.patch_grid),
        "file": CACHE_FILE,
        "sha256": _sha256_file(token_path),
        "shapes": {
            name: list(getattr(cache, name).shape)
            for name in (
                "global_tokens",
                "patch_tokens",
                "mid_tokens",
                "large_tokens",
                "mid_mask",
                "large_mask",
            )
        },
    }
    manifest_path = directory / CACHE_MANIFEST
    manifest_tmp = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    manifest_tmp.write_bytes(_canonical_json(payload))
    os.replace(manifest_tmp, manifest_path)
    return directory


def load_dynamic_cache(directory: Path, expected: DynamicCacheKey) -> DynamicTokenCache:
    directory = Path(directory)
    payload = json.loads((directory / CACHE_MANIFEST).read_text(encoding="utf-8"))
    key_payload = dict(payload["key"])
    key_payload["image_size"] = tuple(map(int, key_payload["image_size"]))
    actual_key = DynamicCacheKey(**key_payload)
    if actual_key != expected or directory.name != cache_digest(expected):
        raise ValueError("dynamic cache identity mismatch")
    token_path = directory / CACHE_FILE
    if _sha256_file(token_path) != str(payload["sha256"]).upper():
        raise ValueError("dynamic cache payload hash mismatch")
    with np.load(token_path, allow_pickle=False) as archive:
        cache = DynamicTokenCache(
            key=actual_key,
            patch_grid=tuple(map(int, payload["patch_grid"])),
            global_tokens=np.ascontiguousarray(archive["global_tokens"]),
            patch_tokens=np.ascontiguousarray(archive["patch_tokens"]),
            mid_tokens=np.ascontiguousarray(archive["mid_tokens"]),
            large_tokens=np.ascontiguousarray(archive["large_tokens"]),
            mid_mask=np.ascontiguousarray(archive["mid_mask"]),
            large_mask=np.ascontiguousarray(archive["large_mask"]),
        )
    _validate_cache(cache)
    return cache


__all__ = [
    "CACHE_FILE",
    "CACHE_MANIFEST",
    "DynamicCacheKey",
    "DynamicTokenCache",
    "cache_digest",
    "derive_patch_grid",
    "encode_dynamic_tokens",
    "load_dynamic_cache",
    "pool_patch_tokens",
    "save_dynamic_cache",
    "stride1_pool_mask",
]
