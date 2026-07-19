from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from measure_vit4ts_v3.dynamic_cache import (
    DynamicCacheKey,
    derive_patch_grid,
    encode_dynamic_tokens,
    load_dynamic_cache,
    pool_patch_tokens,
    save_dynamic_cache,
    stride1_pool_mask,
)


class _TinyOpenClip(torch.nn.Module):
    def encode_image(self, images: torch.Tensor):
        batch = images.shape[0]
        patches = images.mean(dim=1).reshape(batch, 9, 4)
        global_value = patches[:, 0] + 7.0
        return global_value, patches


class _TinyWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyOpenClip()


def _digest(byte: bytes) -> str:
    return hashlib.sha256(byte).hexdigest().upper()


def _key() -> DynamicCacheKey:
    return DynamicCacheKey(
        series_id="toy",
        data_sha256=_digest(b"data"),
        renderer="line",
        renderer_sha256=_digest(b"renderer"),
        model_name="toy",
        pretrained="none",
        model_sha256=_digest(b"model"),
        image_size=(6, 6),
        patch_size=2,
        window=4,
        stride=1,
    )


def test_dynamic_grid_and_masks_follow_row_major_geometry():
    assert derive_patch_grid(196, (224, 224), 16) == (14, 14)
    mask = stride1_pool_mask((3, 4), 2)
    assert mask.shape == (4, 6)
    assert mask[:, 0].tolist() == [0, 1, 4, 5]
    assert mask[:, -1].tolist() == [6, 7, 10, 11]
    with pytest.raises(ValueError):
        derive_patch_grid(195, (224, 224), 16)


def test_pooling_matches_literal_stride_one_average():
    tokens = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    pooled = pool_patch_tokens(tokens, stride1_pool_mask((3, 4), 2))
    assert pooled.shape == (1, 6, 1)
    assert pooled[0, 0, 0].item() == pytest.approx((0 + 1 + 4 + 5) / 4)


def test_true_global_is_not_patch_mean_and_cache_round_trips(tmp_path):
    images = torch.arange(2 * 3 * 6 * 6, dtype=torch.float32).reshape(2, 3, 6, 6)
    cache = encode_dynamic_tokens(
        _TinyWrapper(), images, _key(), batch_size=1, device="cpu"
    )
    assert cache.patch_grid == (3, 3)
    assert cache.patch_tokens.shape == (2, 9, 4)
    assert cache.global_tokens.shape == (2, 4)
    assert not np.array_equal(cache.global_tokens, cache.patch_tokens.mean(axis=1))
    directory = save_dynamic_cache(cache, tmp_path)
    restored = load_dynamic_cache(directory, _key())
    np.testing.assert_array_equal(restored.global_tokens, cache.global_tokens)
    np.testing.assert_array_equal(restored.patch_tokens, cache.patch_tokens)
