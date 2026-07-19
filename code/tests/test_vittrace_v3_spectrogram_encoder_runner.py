from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts_v3 import encoder_runner as encoder_stage
from measure_vit4ts_v3 import spectrogram_encoder_runner as runner
from measure_vit4ts_v3.dynamic_cache import (
    CACHE_FILE,
    CACHE_MANIFEST,
    DynamicCacheKey,
    cache_digest,
    load_dynamic_cache,
    save_dynamic_cache,
)
from measure_vit4ts_v3.spectrogram_registry import (
    load_spectrogram_route,
    make_spectrogram_cache_key,
)
from measure_vit4ts_v3.spectrogram_renderer import render_spectrogram_windows


def _config(tmp_path: Path) -> dict:
    return {
        "stage": "vittrace_ablation_full_v3",
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
                }
            ],
        },
        "spectrogram": {
            "nperseg": 64,
            "noverlap": 48,
            "nfft": 128,
            "window": "hann",
            "scaling": "spectrum",
            "frequency_axis": "linear",
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


class _FakeVisionModel(torch.nn.Module):
    def encode_image(self, images: torch.Tensor):
        batch = images.shape[0]
        mean = images.mean(dim=(1, 2, 3))
        global_tokens = mean[:, None] + torch.arange(
            5, dtype=torch.float32, device=images.device
        )[None, :]
        patch = torch.arange(
            197 * 4, dtype=torch.float32, device=images.device
        ).reshape(1, 197, 4)
        return global_tokens, patch.repeat(batch, 1, 1) + mean[:, None, None]


class _FakeAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeVisionModel()


def _windows(count: int = 5) -> np.ndarray:
    samples = np.arange(count * 240, dtype=np.float64).reshape(count, 240)
    return np.sin(2.0 * np.pi * samples / 31.0)


def _route_key(tmp_path: Path, count: int = 5):
    route = load_spectrogram_route(_config(tmp_path))
    windows = _windows(count)
    identity = runner.streaming_render_identity(windows, route, chunk_size=2)
    key = make_spectrogram_cache_key(
        route,
        identity,
        series_id="toy",
        data_sha256="D" * 64,
        model_sha256="A" * 64,
    )
    return route, windows, identity, key


def test_streaming_renderer_identity_matches_materialized_bytes(tmp_path):
    route, windows, identity, _ = _route_key(tmp_path)
    materialized = render_spectrogram_windows(windows, route.spec)
    assert identity.image_shape == materialized.images.shape
    assert identity.image_dtype == materialized.images.dtype.str
    assert identity.stft_shape == materialized.stft_shape
    assert identity.image_array_sha256 == materialized.image_array_sha256
    assert identity.renderer_sha256 == materialized.renderer_sha256
    assert identity.renderer_source_sha256 == materialized.renderer_source_sha256
    assert identity.renderer_config_sha256 == materialized.renderer_config_sha256


def test_mock_bridge_round_trips_canonical_dynamic_cache_and_is_not_line(tmp_path):
    route, windows, _, key = _route_key(tmp_path)
    result = runner.encode_spectrogram_windows(
        _FakeAdapter(), windows, key, route, device="cpu", chunk_size=2
    )
    assert result.encoder_calls == 3
    assert result.cache.patch_grid == (14, 14)
    assert result.cache.global_tokens.shape == (5, 5)
    assert result.cache.patch_tokens.shape == (5, 196, 4)
    assert result.cache.mid_tokens.shape == (5, 169, 4)
    assert result.cache.large_tokens.shape == (5, 144, 4)
    directory = save_dynamic_cache(result.cache, tmp_path / "caches")
    restored = load_dynamic_cache(directory, key)
    np.testing.assert_array_equal(restored.patch_tokens, result.cache.patch_tokens)
    assert key.renderer.startswith("spectrogram.")
    line_key = DynamicCacheKey(**{**asdict(key), "renderer": "line.batch64"})
    assert cache_digest(key) != cache_digest(line_key)


def test_bridge_render_manifest_is_consumable_by_cpu_score_route(tmp_path):
    import yaml
    from measure_vit4ts_v3 import spectrogram_runner as score_route

    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    bundle = score_route.load_config(config_path)
    route, windows, identity, key = _route_key(tmp_path)
    result = runner.encode_spectrogram_windows(
        _FakeAdapter(), windows, key, route, device="cpu", chunk_size=2
    )
    directory = save_dynamic_cache(result.cache, tmp_path / "cache")
    manifest = directory / runner.RENDER_MANIFEST_FILE
    runner._write_streaming_render_manifest(
        manifest,
        config_sha=runner._sha256_file(config_path),
        route=route,
        render=identity,
        key=key,
    )
    status = score_route.score_cache_directory(
        bundle,
        directory,
        manifest,
        tmp_path / "scores",
        full_length=480,
    )
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    assert payload["completed_arms"] == 2


def test_source_digest_binds_reused_encoder_source():
    first = runner.spectrogram_encoder_source_sha256(
        encoder_source_sha256="A" * 64
    )
    second = runner.spectrogram_encoder_source_sha256(
        encoder_source_sha256="B" * 64
    )
    assert len(first) == 64
    assert first != second


def test_resume_accepts_complete_cache_and_rejects_source_tamper(tmp_path):
    route, windows, identity, key = _route_key(tmp_path)
    result = runner.encode_spectrogram_windows(
        _FakeAdapter(), windows, key, route, device="cpu", chunk_size=2
    )
    cache_root = tmp_path / "cache_root"
    directory = save_dynamic_cache(result.cache, cache_root)
    patch_mean = encoder_stage.patch_mean_tokens(result.cache)
    mean_path = directory / runner.PATCH_MEAN_FILE
    np.save(mean_path, patch_mean, allow_pickle=False)
    render_manifest = directory / runner.RENDER_MANIFEST_FILE
    render_manifest_sha = runner._write_streaming_render_manifest(
        render_manifest,
        config_sha="C" * 64,
        route=route,
        render=identity,
        key=key,
    )
    record_path = tmp_path / "record.json"
    common = {
        "schema_version": runner.SCHEMA_VERSION,
        "status": "PASS",
        "config_sha256": "C" * 64,
        "manifest_sha256": "M" * 64,
        "route_config_sha256": route.route_config_sha256,
        "encoder_source_sha256": "E" * 64,
        "spectrogram_encoder_source_sha256": "S" * 64,
        "variant_sha256": "V" * 64,
        "bridge_identity_sha256": "I" * 64,
        "data_sha256": "D" * 64,
        "model_sha256": "A" * 64,
        "cache_dir": str(directory),
        "cache_key": {**asdict(key), "image_size": list(key.image_size)},
        "cache_file_sha256": runner._sha256_file(directory / CACHE_FILE),
        "cache_manifest_sha256": runner._sha256_file(directory / CACHE_MANIFEST),
        "render_manifest_path": str(render_manifest),
        "render_manifest_sha256": render_manifest_sha,
        "window_count": len(windows),
        "global_tokens_sha256": encoder_stage._array_sha256(
            result.cache.global_tokens
        ),
        "patch_tokens_sha256": encoder_stage._array_sha256(
            result.cache.patch_tokens
        ),
        "mid_tokens_sha256": encoder_stage._array_sha256(result.cache.mid_tokens),
        "large_tokens_sha256": encoder_stage._array_sha256(
            result.cache.large_tokens
        ),
        "patch_mean_path": str(mean_path),
        "patch_mean_sha256": runner._sha256_file(mean_path),
        "patch_mean_shape": list(patch_mean.shape),
        "patch_mean_array_sha256": encoder_stage._array_sha256(patch_mean),
        "renderer_sha256": identity.renderer_sha256,
    }
    record_path.write_text(json.dumps(common), encoding="utf-8")
    kwargs = dict(
        route=route,
        cache_root=cache_root,
        config_sha="C" * 64,
        manifest_sha="M" * 64,
        route_config_sha=route.route_config_sha256,
        encoder_source_sha="E" * 64,
        bridge_source_sha="S" * 64,
        variant_sha="V" * 64,
        bridge_identity_sha="I" * 64,
        data_sha="D" * 64,
        model_sha="A" * 64,
    )
    assert runner._resume_record(record_path, **kwargs) is not None
    kwargs["bridge_source_sha"] = "X" * 64
    assert runner._resume_record(record_path, **kwargs) is None


def test_mock_gate_uses_no_openclip_and_writes_hash_bound_schema(tmp_path):
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(tmp_path), sort_keys=False), encoding="utf-8"
    )
    path = runner.run_mock_gate(config_path, tmp_path / "gate")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["device"] == "cpu"
    assert payload["openclip_loaded"] is False
    assert payload["renderer_hash_parity"] is True
    assert payload["spectrogram_line_cache_separation"] is True
    assert payload["cache_schema_round_trip"] is True
    assert payload["patch_grid"] == [14, 14]


def test_bulk_run_requires_explicit_approval_before_model_load(tmp_path, monkeypatch):
    config = _config(tmp_path)
    record = _record()
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
    root = tmp_path / "stage"
    monkeypatch.setattr(
        runner,
        "_resolve_stage",
        lambda path: (
            config,
            "C" * 64,
            (record,),
            "M" * 64,
            variant,
            route,
            "V" * 64,
            "E" * 64,
            "S" * 64,
            root,
        ),
    )
    with pytest.raises(PermissionError, match="approved-bulk"):
        runner.run_spectrogram_encoder_stage(
            Path("unused.yaml"), all_series=True, approved_bulk=False
        )


def test_failure_is_retained_without_silent_restart(tmp_path, monkeypatch):
    config = _config(tmp_path)
    record = _record()
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
    root = tmp_path / "stage"
    monkeypatch.setattr(
        runner,
        "_resolve_stage",
        lambda path: (
            config,
            "C" * 64,
            (record,),
            "M" * 64,
            variant,
            route,
            "V" * 64,
            "E" * 64,
            "S" * 64,
            root,
        ),
    )
    monkeypatch.setattr(runner, "_carried_verified_ids", lambda *args: set())
    monkeypatch.setattr(runner, "_resume_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(encoder_stage, "_safety_check", lambda config: None)
    monkeypatch.setattr(
        encoder_stage,
        "load_openclip_encoder",
        lambda *args: encoder_stage.LoadedEncoder(
            _FakeAdapter(), "A" * 64, "mock", "mock"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_series",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic encoder failure")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_write_summary",
        lambda *args, **kwargs: root / runner.STATUS_FILE,
    )
    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        runner.run_spectrogram_encoder_stage(
            Path("unused.yaml"), smoke=True, device_name="cpu"
        )
    failures = list(
        (tmp_path / "failures" / "spectrogram_encoder_stage").rglob("*.json")
    )
    assert len(failures) == 1
    payload = json.loads(failures[0].read_text(encoding="utf-8"))
    assert payload["series_id"] == "toy"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "synthetic encoder failure"


def test_bridge_source_has_no_label_or_metric_surface():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "anomalies_csv" not in source
    assert "load_labels" not in source
    assert "f1_max" not in source
    assert "--threshold" not in source


def test_powershell_entry_targets_only_the_independent_bridge():
    path = Path(__file__).parents[1] / "scripts" / "vittrace_v3_spectrogram_encoder.ps1"
    source = path.read_text(encoding="utf-8")
    assert "measure_vit4ts_v3.spectrogram_encoder_runner" in source
    assert "HF_HUB_CACHE" in source
    assert "--approved-bulk" in source
