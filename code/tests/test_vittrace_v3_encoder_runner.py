from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts.renderers import render_official_trace
from measure_vit4ts_v3 import encoder_runner as runner
from measure_vit4ts_v3.dynamic_cache import (
    CACHE_FILE,
    DynamicCacheKey,
    DynamicTokenCache,
    encode_dynamic_tokens,
    save_dynamic_cache,
    stride1_pool_mask,
)


def _config(tmp_path: Path) -> dict:
    return {
        "stage": runner.EXPECTED_STAGE,
        "vendor": {
            "root": "vendor",
            "commit": "1" * 40,
            "default_model_sha256": "A" * 64,
        },
        "defaults": {
            "representation": "line",
            "image_size": [224, 224],
            "window": 240,
            "stride": 60,
        },
        "grid": {
            "representations": ["line", "spectrogram"],
            "windows": [120, 180, 240, 360, 480],
            "window_stride_fraction": 0.25,
            "strides_w240": [30, 60, 120],
            "backbones": [
                {
                    "key": "B16",
                    "model_name": "ViT-B-16",
                    "pretrained": "openai",
                    "patch_size": 16,
                },
                {
                    "key": "B32",
                    "model_name": "ViT-B-32",
                    "pretrained": "openai",
                    "patch_size": 32,
                },
            ],
        },
        "runtime": {
            "batch_size": 64,
            "device": "cpu",
            "c_drive_floor_gib": 0,
            "d_drive_floor_gib": 0,
            "available_ram_floor_gib": 0,
        },
        "paths": {
            "output_root": str(tmp_path / "output"),
            "failure_root": str(tmp_path / "failures"),
        },
        "data": {"root": str(tmp_path / "data")},
    }


def _variant(**changes) -> runner.EncoderVariant:
    values = {
        "key": "line_B16_W240_S60_B64",
        "representation": "line",
        "model_key": "B16",
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "image_size": (224, 224),
        "patch_size": 16,
        "window": 240,
        "stride": 60,
        "batch_size": 64,
    }
    values.update(changes)
    return runner.EncoderVariant(**values)


def _record(series_id: str = "toy") -> FullSeriesRecord:
    return FullSeriesRecord(
        series_id=series_id,
        dataset="artificialWithAnomaly",
        track="NAB",
        paper_group="NAB-Artificial",
        signal_name=series_id,
        relative_path="unused.csv",
        expected_length=480,
        expected_windows=5,
        expected_sha256="D" * 64,
        duplicate_timestamps=False,
    )


def _key() -> DynamicCacheKey:
    return DynamicCacheKey(
        series_id="toy",
        data_sha256="D" * 64,
        renderer="line.batch64",
        renderer_sha256="E" * 64,
        model_name="toy",
        pretrained="none",
        model_sha256="A" * 64,
        image_size=(6, 6),
        patch_size=2,
        window=4,
        stride=1,
    )


def _cache() -> DynamicTokenCache:
    patch = np.arange(2 * 9 * 5, dtype=np.float32).reshape(2, 9, 5)
    return DynamicTokenCache(
        key=_key(),
        patch_grid=(3, 3),
        global_tokens=np.arange(2 * 3, dtype=np.float32).reshape(2, 3),
        patch_tokens=patch,
        mid_tokens=np.zeros((2, 4, 5), dtype=np.float32),
        large_tokens=np.zeros((2, 1, 5), dtype=np.float32),
        mid_mask=stride1_pool_mask((3, 3), 2),
        large_mask=stride1_pool_mask((3, 3), 3),
    )


def test_variant_keys_bind_model_representation_window_stride_and_batch(tmp_path):
    config = _config(tmp_path)
    base = runner.resolve_variant(config)
    short = runner.resolve_variant(config, window=120, stride=30)
    b32 = runner.resolve_variant(config, model_key="B32")
    spectrum = runner.resolve_variant(config, representation="spectrogram")

    assert base.key == "line_B16_W240_S60_B64"
    assert (base.window, base.stride, base.batch_size) == (240, 60, 64)
    assert len({base.key, short.key, b32.key, spectrum.key}) == 4
    assert len({runner.variant_sha256(item, "C" * 64) for item in (base, short, b32, spectrum)}) == 4
    with pytest.raises(ValueError, match="not registered"):
        runner.resolve_variant(config, window=120, stride=60)
    with pytest.raises(NotImplementedError, match="no registered renderer"):
        runner.render_variant_windows(np.zeros((1, 240), dtype=np.float32), spectrum)


def test_bulk_selection_requires_explicit_approval():
    records = (_record("one"), _record("two"))
    assert runner.select_records(
        records,
        smoke=True,
        all_series=False,
        series_ids=(),
        approved_bulk=False,
    ) == (records[0],)
    with pytest.raises(PermissionError, match="approved-bulk"):
        runner.select_records(
            records,
            smoke=False,
            all_series=True,
            series_ids=(),
            approved_bulk=False,
        )


def test_line_renderer_is_the_registered_official_trace():
    variant = runner.EncoderVariant(
        "tiny", "line", "toy", "toy", "none", (6, 6), 2, 4, 1, 2
    )
    window = np.array([[0.0, 0.25, 0.75, 1.0]], dtype=np.float32)
    images, geometries = runner.render_variant_windows(window, variant)
    expected = render_official_trace(window[0], image_size=(6, 6), expected_length=4)
    np.testing.assert_array_equal(images[0], expected.image)
    np.testing.assert_array_equal(geometries[0].vertices, expected.geometry.vertices)


class _ClassTokenModel(torch.nn.Module):
    def encode_image(self, images: torch.Tensor):
        batch = images.shape[0]
        patches = torch.arange(
            batch * 10 * 4, dtype=torch.float32, device=images.device
        ).reshape(batch, 10, 4)
        global_tokens = torch.full((batch, 3), 7.0, device=images.device)
        return global_tokens, patches


class _ClassTokenAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ClassTokenModel()


def test_encoder_removes_optional_class_token_and_preserves_true_global():
    images = torch.zeros((2, 3, 6, 6), dtype=torch.float32)
    cache = encode_dynamic_tokens(
        _ClassTokenAdapter(), images, _key(), batch_size=2, device="cpu"
    )
    assert cache.patch_grid == (3, 3)
    assert cache.patch_tokens.shape == (2, 9, 4)
    assert cache.global_tokens.shape == (2, 3)
    expected_first_patch = torch.arange(40, dtype=torch.float32).reshape(10, 4)[1]
    np.testing.assert_array_equal(cache.patch_tokens[0, 0], expected_first_patch.numpy())


def test_patch_mean_sidecar_is_separate_from_true_global():
    cache = _cache()
    mean = runner.patch_mean_tokens(cache)
    assert mean.shape == (2, 5)
    assert cache.global_tokens.shape == (2, 3)
    np.testing.assert_array_equal(mean, cache.patch_tokens.mean(axis=1))


def test_exact_patch_comparator_reports_first_mismatch():
    reference = np.zeros((2, 3, 4), dtype=np.float32)
    actual = reference.copy()
    assert runner.compare_token_arrays(actual, reference)["passed"] is True
    actual[1, 2, 3] = 1e-7
    result = runner.compare_token_arrays(actual, reference)
    assert result["passed"] is False
    assert result["mismatch_count"] == 1
    assert result["first_mismatch"] == [1, 2, 3]


def test_b16_patch_parity_diagnosis_binds_renderer_model_and_true_global(
    tmp_path, monkeypatch
):
    cache = _cache()
    dynamic_dir = tmp_path / "dynamic"
    dynamic_dir.mkdir()
    (dynamic_dir / CACHE_FILE).write_bytes(b"dynamic")
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    frozen_manifest = frozen_dir / "clip_tokens.json"
    frozen_manifest.write_text("{}", encoding="utf-8")
    (frozen_dir / "clip_tokens.npz").write_bytes(b"frozen")
    frozen = SimpleNamespace(patch_tokens=cache.patch_tokens.copy())
    frozen_payload = {
        "key": {
            "renderer_sha256": cache.key.renderer_sha256,
            "clip_weight_sha256": cache.key.model_sha256,
        }
    }
    monkeypatch.setattr(
        runner,
        "_frozen_patch_cache",
        lambda config, record: (frozen, frozen_manifest, frozen_payload),
    )
    payload = runner.diagnose_frozen_patch_parity(
        {}, _record(), _variant(), cache, dynamic_dir, cache.key.model_sha256
    )
    assert payload["passed"] is True
    assert payload["patch_comparison"]["mismatch_count"] == 0
    assert payload["renderer_sha256_match"] is True
    assert payload["model_sha256_match"] is True
    assert payload["global_tokens_shape"] == [2, 3]
    assert payload["patch_mean_shape"] == [2, 5]
    assert "distinct embedding dimensions" in payload["global_vs_patch_mean"]["reason"]


def test_resume_validates_cache_and_patch_mean_sidecar(tmp_path):
    cache = _cache()
    directory = save_dynamic_cache(cache, tmp_path / "caches")
    mean_path = directory / runner.PATCH_MEAN_FILE
    np.save(mean_path, runner.patch_mean_tokens(cache), allow_pickle=False)
    record_path = tmp_path / "record.json"
    payload = {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "PASS",
        "config_sha256": "C" * 64,
        "manifest_sha256": "M" * 64,
        "encoder_source_sha256": "S" * 64,
        "variant_sha256": "V" * 64,
        "data_sha256": "D" * 64,
        "cache_dir": str(directory),
        "cache_key": {**cache.key.__dict__, "image_size": list(cache.key.image_size)},
        "window_count": 2,
        "patch_mean_path": str(mean_path),
        "patch_mean_sha256": runner._sha256(mean_path),
        "patch_mean_shape": [2, 5],
    }
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    resumed = runner._resume_record(
        record_path,
        config_sha="C" * 64,
        manifest_sha="M" * 64,
        source_sha="S" * 64,
        variant_sha="V" * 64,
        data_sha="D" * 64,
    )
    assert resumed is not None
    assert runner._resume_record(
        record_path,
        config_sha="X" * 64,
        manifest_sha="M" * 64,
        source_sha="S" * 64,
        variant_sha="V" * 64,
        data_sha="D" * 64,
    ) is None


def test_run_preserves_per_series_failure_manifest(tmp_path, monkeypatch):
    config = _config(tmp_path)
    record = _record()
    monkeypatch.setattr(
        runner,
        "_load_config",
        lambda path: (config, "C" * 64, (record,), "M" * 64),
    )
    monkeypatch.setattr(runner, "_safety_check", lambda config: None)
    monkeypatch.setattr(
        runner,
        "load_openclip_encoder",
        lambda config, variant, device: runner.LoadedEncoder(
            _ClassTokenAdapter(), "A" * 64, "test", "test"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_series",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    monkeypatch.setattr(
        runner,
        "_write_summary",
        lambda *args, **kwargs: tmp_path / "status.json",
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run_encoder_stage(Path("unused.yaml"), smoke=True, device_name="cpu")
    failures = list((tmp_path / "failures" / "encoder_stage").rglob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["series_id"] == "toy"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "synthetic failure"
