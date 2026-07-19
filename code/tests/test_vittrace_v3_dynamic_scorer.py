from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts.ritp import stitch_native_240
from measure_vit4ts_v3 import parity
from measure_vit4ts_v3 import dynamic_score_runner as runner
from measure_vit4ts_v3.dynamic_cache import (
    CACHE_FILE,
    CACHE_MANIFEST,
    DynamicCacheKey,
    DynamicTokenCache,
    save_dynamic_cache,
    stride1_pool_mask,
)
from measure_vit4ts_v3.dynamic_scorer import (
    DYNAMIC_SCORE_ARMS,
    compute_dynamic_scores,
    stitch_native_dynamic,
)
from measure_vit4ts_v3.encoder_runner import EncoderVariant


def _record(series_id: str = "toy", *, length: int = 300) -> FullSeriesRecord:
    return FullSeriesRecord(
        series_id=series_id,
        dataset="artificialWithAnomaly",
        track="NAB",
        paper_group="NAB-Artificial",
        signal_name=series_id,
        relative_path="unused.csv",
        expected_length=length,
        expected_windows=2,
        expected_sha256="D" * 64,
        duplicate_timestamps=False,
    )


def _cache(
    *,
    grid: tuple[int, int] = (14, 14),
    patch_size: int = 16,
    window: int = 240,
    stride: int = 60,
    windows: int = 2,
    feature: int = 4,
) -> DynamicTokenCache:
    rng = np.random.default_rng(2027 + grid[0] * 100 + grid[1])
    patch = rng.normal(size=(windows, grid[0] * grid[1], feature)).astype(np.float32)
    mid_mask = stride1_pool_mask(grid, 2)
    large_mask = stride1_pool_mask(grid, 3)
    mid = patch[:, mid_mask.T, :].mean(axis=2, dtype=np.float32)
    large = patch[:, large_mask.T, :].mean(axis=2, dtype=np.float32)
    key = DynamicCacheKey(
        series_id="toy",
        data_sha256="D" * 64,
        renderer="line.batch64",
        renderer_sha256="E" * 64,
        model_name="synthetic",
        pretrained="synthetic",
        model_sha256="A" * 64,
        image_size=(grid[0] * patch_size, grid[1] * patch_size),
        patch_size=patch_size,
        window=window,
        stride=stride,
    )
    return DynamicTokenCache(
        key=key,
        patch_grid=grid,
        global_tokens=rng.normal(size=(windows, 3)).astype(np.float32),
        patch_tokens=np.ascontiguousarray(patch),
        mid_tokens=np.ascontiguousarray(mid),
        large_tokens=np.ascontiguousarray(large),
        mid_mask=mid_mask,
        large_mask=large_mask,
    )


def test_dynamic_three_arms_match_frozen_b16_parity_formulas_on_cpu():
    cache = _cache()
    record = _record()
    config = {
        "contracts": {"parity_window": 240, "parity_stride": 60},
        "runtime": {"batch_size": 64},
        "defaults": {"image_size": [224, 224]},
    }
    expected = parity.reconstruct_parity_arms(
        cache, record, config, torch.device("cpu")
    )
    actual = compute_dynamic_scores(
        cache,
        full_length=300,
        window=240,
        stride=60,
        image_size=(224, 224),
        device="cpu",
    )
    np.testing.assert_array_equal(actual.arms["REL"].score, expected["REL_U"])
    np.testing.assert_array_equal(actual.arms["IHP"].score, expected["IHP_LEGACY"])
    np.testing.assert_array_equal(
        actual.arms["FULL"].score, expected["FULL_COLUMN_240"]
    )


def test_dynamic_scorer_supports_non_b16_grid_and_window_on_cpu():
    cache = _cache(grid=(7, 7), patch_size=32, window=120, stride=30)
    bundle = compute_dynamic_scores(
        cache,
        full_length=150,
        window=120,
        stride=30,
        image_size=(224, 224),
        device="cpu",
        query_chunk_size=1,
    )
    assert tuple(bundle.arms) == DYNAMIC_SCORE_ARMS
    for arm, result in bundle.arms.items():
        assert result.score.shape == (150,)
        assert result.score.dtype == np.float64
        assert result.window_field.shape == (2, 49)
        assert np.isfinite(result.score).all()
        assert result.metadata["matching_scope"] == "global"
        assert result.metadata["scales"] == "PML"
        if arm == "FULL":
            assert result.metadata["temporal"] == "nctp_linear"


def test_general_native_stitch_is_exact_frozen_w240_stitch():
    rng = np.random.default_rng(9)
    local = rng.random((3, 240), dtype=np.float64)
    starts = np.array([0, 60, 120], dtype=np.int64)
    expected = stitch_native_240(local, starts, 380)
    actual = stitch_native_dynamic(local, starts, 380)
    np.testing.assert_array_equal(actual, expected)


def test_one_series_dynamic_transaction_is_resumable_and_hash_bound(tmp_path):
    cache = _cache()
    cache_dir = save_dynamic_cache(cache, tmp_path / "cache_root")
    root = tmp_path / "variant"
    record = _record()
    record_path = root / "records" / "toy.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text("{}\n", encoding="utf-8")
    encoder_payload = {
        "cache_dir": str(cache_dir),
        "cache_key": {**cache.key.__dict__, "image_size": list(cache.key.image_size)},
        "cache_file_sha256": runner._sha256(cache_dir / CACHE_FILE),
        "cache_manifest_sha256": runner._sha256(cache_dir / CACHE_MANIFEST),
    }
    variant = EncoderVariant(
        "line_B16_W240_S60_B64",
        "line",
        "B16",
        "ViT-B-16",
        "openai",
        (224, 224),
        16,
        240,
        60,
        64,
    )
    kwargs = {
        "config_sha": "C" * 64,
        "manifest_sha": "M" * 64,
        "encoder_source_sha": "E" * 64,
        "score_source_sha": "S" * 64,
        "variant_sha": "V" * 64,
        "score_config_sha": "Q" * 64,
    }
    existing, identity = runner._existing_transactions(
        root, record, encoder_payload, retry=False, **kwargs
    )
    assert existing == {}
    committed = runner._commit_series(
        root,
        record,
        variant,
        encoder_payload,
        cache,
        existing,
        identity,
        device="cpu",
        query_chunk_size=32,
    )
    assert tuple(committed) == DYNAMIC_SCORE_ARMS
    resumed, _ = runner._existing_transactions(
        root, record, encoder_payload, retry=False, **kwargs
    )
    assert tuple(resumed) == DYNAMIC_SCORE_ARMS
    for arm in DYNAMIC_SCORE_ARMS:
        payload = json.loads(
            (root / "dynamic_scores" / "toy" / arm / "score_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["status"] == "PASS"
        assert payload["encoder_calls"] == 0
        assert payload["cache_file_sha256"] == encoder_payload["cache_file_sha256"]
        assert payload["runtime"]["device"] == "cpu"
        score = np.load(payload["score_path"], allow_pickle=False)
        assert score.shape == (300,)
        assert score.dtype == np.float64
    rel_path = root / "dynamic_scores" / "toy" / "REL" / "score.npy"
    tampered = np.load(rel_path, allow_pickle=False)
    tampered[0] += 1.0
    np.save(rel_path, tampered, allow_pickle=False)
    after_tamper, _ = runner._existing_transactions(
        root, record, encoder_payload, retry=False, **kwargs
    )
    assert "REL" not in after_tamper
    assert set(after_tamper) == {"IHP", "FULL"}


def test_dynamic_score_source_is_separate_from_frozen_encoder_source():
    assert len(runner.dynamic_score_source_sha256()) == 64
    names = {name for namespace, name in runner._SOURCE_FILES if namespace == "v3"}
    assert "encoder_runner.py" not in names
    assert "dynamic_cache.py" not in names
