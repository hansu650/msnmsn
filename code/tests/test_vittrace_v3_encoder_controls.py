from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts.reducers import stitch_column_vectors
from measure_vit4ts.ritp import stitch_native_240
from measure_vit4ts_v3 import encoder_control_runner as control_runner
from measure_vit4ts_v3 import encoder_runner as encoder_stage
from measure_vit4ts_v3.dynamic_cache import (
    CACHE_FILE,
    DynamicCacheKey,
    DynamicTokenCache,
    stride1_pool_mask,
)
from measure_vit4ts_v3.encoder_controls import (
    CONTROL_ARMS,
    compute_encoder_controls,
    legacy_scalar_stitch,
    median_cosine_scalar,
    native_w240_scalar_stitch,
)


def _record() -> FullSeriesRecord:
    return FullSeriesRecord(
        series_id="toy",
        dataset="artificialWithAnomaly",
        track="NAB",
        paper_group="NAB-Artificial",
        signal_name="toy",
        relative_path="unused.csv",
        expected_length=300,
        expected_windows=2,
        expected_sha256="D" * 64,
        duplicate_timestamps=False,
    )


def _variant() -> encoder_stage.EncoderVariant:
    return encoder_stage.EncoderVariant(
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


def _cache(tmp_path: Path) -> tuple[DynamicTokenCache, np.ndarray, dict]:
    key = DynamicCacheKey(
        series_id="toy",
        data_sha256="D" * 64,
        renderer="line.batch64",
        renderer_sha256="E" * 64,
        model_name="ViT-B-16",
        pretrained="openai",
        model_sha256="A" * 64,
        image_size=(6, 6),
        patch_size=2,
        window=240,
        stride=60,
    )
    patch = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] * 3,
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]] * 3,
        ],
        dtype=np.float32,
    )
    cache = DynamicTokenCache(
        key=key,
        patch_grid=(3, 3),
        global_tokens=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        patch_tokens=patch,
        mid_tokens=np.zeros((2, 4, 2), dtype=np.float32),
        large_tokens=np.zeros((2, 1, 2), dtype=np.float32),
        mid_mask=stride1_pool_mask((3, 3), 2),
        large_mask=stride1_pool_mask((3, 3), 3),
    )
    directory = tmp_path / "cache"
    directory.mkdir(parents=True)
    (directory / CACHE_FILE).write_bytes(b"cache")
    mean = encoder_stage.patch_mean_tokens(cache)
    payload = {"cache_dir": str(directory), "window_count": 2}
    return cache, mean, payload


def test_median_cosine_scalar_matches_frozen_float32_formula():
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    actual = median_cosine_scalar(embeddings)
    values = torch.from_numpy(embeddings)
    memory = torch.median(values, dim=0).values
    expected = 0.5 * (
        1.0
        - torch.matmul(
            torch.nn.functional.normalize(values, dim=-1),
            torch.nn.functional.normalize(memory, dim=-1),
        )
    )
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected.double().numpy())


def test_scalar_temporal_mappings_are_exact_frozen_stitchers():
    scalar = np.array([1.0, 3.0], dtype=np.float64)
    starts = np.array([0, 60], dtype=np.int64)
    legacy = legacy_scalar_stitch(
        scalar, full_length=300, window=240, stride=60
    )
    native = native_w240_scalar_stitch(scalar, starts, full_length=300)
    np.testing.assert_array_equal(
        legacy,
        stitch_column_vectors(np.repeat(scalar[:, None], 224, axis=1), 300, 240, 60),
    )
    np.testing.assert_array_equal(
        native,
        stitch_native_240(np.repeat(scalar[:, None], 240, axis=1), starts, 300),
    )
    np.testing.assert_array_equal(native[:60], np.ones(60))
    np.testing.assert_array_equal(native[60:240], np.full(180, 2.0))
    np.testing.assert_array_equal(native[240:], np.full(60, 3.0))


def test_controls_keep_true_global_and_patch_mean_semantically_separate():
    true_global = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    patch_mean = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    results = compute_encoder_controls(
        true_global,
        patch_mean,
        np.array([0, 60], dtype=np.int64),
        full_length=300,
        window=240,
        stride=60,
    )
    assert tuple(results) == CONTROL_ARMS
    assert results["CTRL_CLS_LEGACY"].metadata["embedding"] == "openclip_true_global"
    assert results["CTRL_CLS_NATIVE_W240"].metadata["ihp"] == "bypassed"
    patch_metadata = results["CTRL_PATCH_MEAN_NATIVE_W240"].metadata
    assert patch_metadata["embedding"] == "mean_of_base_patch_tokens"
    assert patch_metadata["not_cls"] is True
    assert patch_metadata["nctp"] == "bypassed"
    assert not np.array_equal(
        results["CTRL_CLS_NATIVE_W240"].window_scalar,
        results["CTRL_PATCH_MEAN_NATIVE_W240"].window_scalar,
    )


def test_control_transactions_are_hash_bound_and_resumable(tmp_path):
    root = tmp_path / "variant"
    (root / "records").mkdir(parents=True)
    encoder_record = root / "records" / "toy.json"
    encoder_record.write_text("{}", encoding="utf-8")
    cache, patch_mean, encoder_payload = _cache(tmp_path)
    output = control_runner._run_series(
        root,
        _record(),
        _variant(),
        encoder_payload,
        cache,
        patch_mean,
        config_sha="C" * 64,
        manifest_sha="M" * 64,
        encoder_source_sha="E" * 64,
        control_source_sha="S" * 64,
        variant_sha="V" * 64,
        retry=False,
    )
    assert tuple(output) == CONTROL_ARMS
    created = {}
    for arm in CONTROL_ARMS:
        manifest_path = root / "controls" / "toy" / arm / "score_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        created[arm] = payload["created_at"]
        assert payload["metadata"]["ihp"] == "bypassed"
        assert payload["metadata"]["nctp"] == "bypassed"
        assert np.load(payload["score_path"], allow_pickle=False).shape == (300,)
    resumed = control_runner._run_series(
        root,
        _record(),
        _variant(),
        encoder_payload,
        cache,
        patch_mean,
        config_sha="C" * 64,
        manifest_sha="M" * 64,
        encoder_source_sha="E" * 64,
        control_source_sha="S" * 64,
        variant_sha="V" * 64,
        retry=False,
    )
    assert {arm: resumed[arm]["created_at"] for arm in CONTROL_ARMS} == created


def test_all_mode_preflights_encoder_coverage_before_writing_scores(
    tmp_path, monkeypatch
):
    config = {
        "defaults": {"representation": "line", "image_size": [224, 224], "window": 240, "stride": 60},
        "grid": {
            "representations": ["line"],
            "windows": [240],
            "window_stride_fraction": 0.25,
            "strides_w240": [60],
            "backbones": [{"key": "B16", "model_name": "ViT-B-16", "pretrained": "openai", "patch_size": 16}],
        },
        "runtime": {"batch_size": 64},
        "paths": {"output_root": str(tmp_path / "output")},
    }
    record = _record()
    monkeypatch.setattr(
        encoder_stage,
        "_load_config",
        lambda path: (config, "C" * 64, (record,), "M" * 64),
    )
    monkeypatch.setattr(
        control_runner,
        "_preflight_encoder_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    with pytest.raises(RuntimeError, match="preflight found 1"):
        control_runner.run_encoder_controls(
            Path("unused.yaml"), all_series=True, approved_bulk=True
        )
    assert not list((tmp_path / "output").rglob("score.npy"))


def test_all_mode_streams_one_materialized_encoder_artifact_at_a_time(
    tmp_path, monkeypatch
):
    config = {
        "defaults": {
            "representation": "line",
            "image_size": [224, 224],
            "window": 240,
            "stride": 60,
        },
        "grid": {
            "representations": ["line"],
            "windows": [240],
            "window_stride_fraction": 0.25,
            "strides_w240": [60],
            "backbones": [
                {
                    "key": "B16",
                    "model_name": "ViT-B-16",
                    "pretrained": "openai",
                    "patch_size": 16,
                }
            ],
        },
        "runtime": {"batch_size": 64},
        "paths": {"output_root": str(tmp_path / "output")},
    }
    records = tuple(
        FullSeriesRecord(
            series_id=f"toy_{index}",
            dataset="artificialWithAnomaly",
            track="NAB",
            paper_group="NAB-Artificial",
            signal_name=f"toy_{index}",
            relative_path="unused.csv",
            expected_length=300,
            expected_windows=2,
            expected_sha256=f"{index + 1:064X}",
            duplicate_timestamps=False,
        )
        for index in range(3)
    )
    series_ids = [record.series_id for record in records]
    state = {"active": 0, "preflight": [], "loads": [], "runs": []}

    class Token:
        def __init__(self, series_id):
            self.series_id = series_id
            state["active"] += 1

        def __del__(self):
            state["active"] -= 1

    def fake_preflight(root, record, **kwargs):
        assert state["active"] == 0
        state["preflight"].append(record.series_id)
        return {}

    def fake_load(root, record, **kwargs):
        assert state["preflight"] == series_ids
        assert state["active"] == 0
        state["loads"].append(record.series_id)
        return {}, Token(record.series_id), Token(record.series_id)

    def fake_run(root, record, variant, payload, cache, patch_mean, **kwargs):
        assert state["active"] == 2
        assert cache.series_id == record.series_id
        assert patch_mean.series_id == record.series_id
        state["runs"].append(record.series_id)
        return {}

    monkeypatch.setattr(
        encoder_stage,
        "_load_config",
        lambda path: (config, "C" * 64, records, "M" * 64),
    )
    monkeypatch.setattr(control_runner, "_preflight_encoder_artifact", fake_preflight)
    monkeypatch.setattr(control_runner, "_load_encoder_artifact", fake_load)
    monkeypatch.setattr(control_runner, "_run_series", fake_run)
    expected = tmp_path / "status.json"
    monkeypatch.setattr(control_runner, "_write_summary", lambda *args, **kwargs: expected)

    actual = control_runner.run_encoder_controls(
        Path("unused.yaml"), all_series=True, approved_bulk=True
    )
    assert actual == expected
    assert state["preflight"] == series_ids
    assert state["loads"] == series_ids
    assert state["runs"] == series_ids
    assert state["active"] == 0


def test_control_source_is_separate_from_frozen_encoder_source():
    assert "encoder_controls.py" not in {
        name for namespace, name in encoder_stage._SOURCE_FILES if namespace == "v3"
    }
    assert len(control_runner.control_source_sha256()) == 64
