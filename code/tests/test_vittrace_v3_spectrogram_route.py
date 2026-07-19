from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from measure_vit4ts_v3.dynamic_cache import (
    DynamicCacheKey,
    DynamicTokenCache,
    pool_patch_tokens,
    save_dynamic_cache,
    stride1_pool_mask,
)
from measure_vit4ts_v3.spectrogram_registry import (
    ARMS,
    FULL_ARM,
    REL_ARM,
    build_spectrogram_registry,
    load_spectrogram_route,
    make_spectrogram_cache_key,
    renderer_identity,
    validate_spectrogram_cache_key,
)
from measure_vit4ts_v3.spectrogram_renderer import (
    SpectrogramSpec,
    periodic_hann,
    render_spectrogram_windows,
    renderer_config_sha256,
    stft_magnitude,
)
from measure_vit4ts_v3.spectrogram_runner import (
    load_config,
    render_windows_file,
    score_cache_directory,
    score_spectrogram_cache,
)


def _config() -> dict:
    return {
        "stage": "vittrace_ablation_full_v3",
        "vendor": {"default_model_sha256": "A" * 64},
        "defaults": {"image_size": [224, 224]},
        "grid": {
            "representations": ["line", "spectrogram"],
            "windows": [120, 180, 240, 360, 480],
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
        "runtime": {"batch_size": 64},
    }


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "vittrace_v3.yaml"
    path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")
    return path


def _random_cache(key: DynamicCacheKey, windows: int = 3) -> DynamicTokenCache:
    generator = np.random.default_rng(2027)
    patch = generator.normal(size=(windows, 196, 8)).astype(np.float32)
    patch_tensor = torch.from_numpy(patch)
    mid_mask = stride1_pool_mask((14, 14), 2)
    large_mask = stride1_pool_mask((14, 14), 3)
    return DynamicTokenCache(
        key=key,
        patch_grid=(14, 14),
        global_tokens=generator.normal(size=(windows, 4)).astype(np.float32),
        patch_tokens=patch,
        mid_tokens=pool_patch_tokens(patch_tensor, mid_mask).numpy(),
        large_tokens=pool_patch_tokens(patch_tensor, large_mask).numpy(),
        mid_mask=mid_mask,
        large_mask=large_mask,
    )


def _key_and_render(config: dict, windows: int = 3):
    route = load_spectrogram_route(config)
    values = np.arange(windows * 240, dtype=np.float32).reshape(windows, 240)
    values /= float(values.max())
    render = render_spectrogram_windows(values, route.spec)
    key = make_spectrogram_cache_key(
        route,
        render,
        series_id="toy",
        data_sha256="D" * 64,
        model_sha256="A" * 64,
    )
    return route, render, key


def test_mandatory_config_and_registry_are_exact():
    route = load_spectrogram_route(_config())
    assert route.spec == SpectrogramSpec()
    assert route.spec.frame_count == 12
    assert route.spec.frequency_bins == 65
    assert route.variant.to_payload() == {
        "representation": "spectrogram",
        "model_key": "B16",
        "model_name": "ViT-B-16",
        "pretrained": "openai",
        "image_size": [224, 224],
        "patch_size": 16,
        "window": 240,
        "stride": 60,
        "batch_size": 64,
    }
    registry = build_spectrogram_registry()
    assert registry.arm_ids == ARMS == (REL_ARM, FULL_ARM)
    assert registry.primary_arm == FULL_ARM
    assert registry.control_arm == REL_ARM
    assert registry.contrasts[0].candidate == FULL_ARM
    assert registry.contrasts[0].control == REL_ARM


@pytest.mark.parametrize(
    ("block", "field", "value"),
    [
        ("spectrogram", "nfft", 64),
        ("spectrogram", "frequency_axis", "log"),
        ("runtime", "batch_size", 32),
    ],
)
def test_route_rejects_config_drift(block, field, value):
    config = _config()
    config[block][field] = value
    with pytest.raises(ValueError):
        load_spectrogram_route(config)


def test_periodic_hann_and_pure_tone_peak_are_deterministic():
    taper = periodic_hann()
    assert taper.shape == (64,)
    assert taper[0] == 0.0
    assert taper.sum() == pytest.approx(32.0, abs=1e-13)

    samples = np.arange(240, dtype=np.float64)
    tone = np.sin(2.0 * np.pi * 8.0 * samples / 128.0)[None, :]
    first = stft_magnitude(tone)
    second = stft_magnitude(tone.copy())
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 65, 12)
    np.testing.assert_array_equal(np.argmax(first[0], axis=0), np.full(12, 8))


def test_renderer_is_byte_deterministic_and_224_rgb():
    zero = np.zeros((1, 240), dtype=np.float32)
    blank = render_spectrogram_windows(zero)
    assert blank.images.shape == (1, 3, 224, 224)
    assert blank.images.dtype == np.float32
    np.testing.assert_array_equal(blank.images, np.ones_like(blank.images))

    samples = np.arange(240, dtype=np.float32)
    values = np.stack(
        [np.sin(2 * np.pi * samples / 32), np.cos(2 * np.pi * samples / 20)]
    )
    first = render_spectrogram_windows(values)
    second = render_spectrogram_windows(values.copy())
    np.testing.assert_array_equal(first.images, second.images)
    assert first.image_array_sha256 == second.image_array_sha256
    assert first.renderer_sha256 == second.renderer_sha256
    assert first.renderer_config_sha256 == renderer_config_sha256(SpectrogramSpec())
    assert 0.0 <= float(first.images.min()) < float(first.images.max()) <= 1.0
    np.testing.assert_array_equal(first.images[:, 0], first.images[:, 1])
    np.testing.assert_array_equal(first.images[:, 1], first.images[:, 2])


def test_dynamic_cache_identity_is_spectrogram_specific():
    route, render, key = _key_and_render(_config())
    assert key.renderer == renderer_identity(route)
    assert key.renderer.startswith("spectrogram.")
    assert key.renderer_sha256 == render.renderer_sha256
    assert (key.model_name, key.patch_size, key.window, key.stride) == (
        "ViT-B-16",
        16,
        240,
        60,
    )
    validate_spectrogram_cache_key(route, key)
    line_key = DynamicCacheKey(
        **{**key.__dict__, "renderer": "line.batch64"}
    )
    with pytest.raises(ValueError, match="mandatory spectrogram"):
        validate_spectrogram_cache_key(route, line_key)


def test_rel_and_full_score_from_synthetic_dynamic_cache():
    _, _, key = _key_and_render(_config())
    cache = _random_cache(key)
    scores = score_spectrogram_cache(cache, full_length=360)
    assert tuple(scores) == ARMS
    for score in scores.values():
        assert score.shape == (360,)
        assert score.dtype == np.float64
        assert np.isfinite(score).all()
    assert not np.array_equal(scores[REL_ARM], scores[FULL_ARM])
    with pytest.raises(ValueError, match="window count"):
        score_spectrogram_cache(cache, full_length=420)


def test_render_and_score_transactions_round_trip(tmp_path):
    config_path = _write_config(tmp_path)
    bundle = load_config(config_path)
    windows = np.arange(3 * 240, dtype=np.float32).reshape(3, 240)
    windows /= float(windows.max())
    windows_path = tmp_path / "windows.npy"
    np.save(windows_path, windows, allow_pickle=False)
    render_root = tmp_path / "render"
    render_manifest = render_windows_file(
        bundle,
        windows_path,
        render_root,
        series_id="toy",
        data_sha256="D" * 64,
    )
    payload = json.loads(render_manifest.read_text(encoding="utf-8"))
    key_payload = dict(payload["cache_key"])
    key_payload["image_size"] = tuple(key_payload["image_size"])
    key = DynamicCacheKey(**key_payload)
    cache_dir = save_dynamic_cache(_random_cache(key), tmp_path / "cache")

    status = score_cache_directory(
        bundle,
        cache_dir,
        render_manifest,
        tmp_path / "scores",
        full_length=360,
    )
    status_payload = json.loads(status.read_text(encoding="utf-8"))
    assert status_payload["status"] == "COMPLETE"
    assert status_payload["expected_arms"] == list(ARMS)
    assert status_payload["completed_arms"] == 2
    assert status_payload["encoder_calls"] == 0
    assert status_payload["model_forward"] is False
    for arm in ARMS:
        score = np.load(tmp_path / "scores" / arm / "score.npy", allow_pickle=False)
        manifest = json.loads(
            (tmp_path / "scores" / arm / "score_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert score.shape == (360,)
        assert manifest["arm"] == arm
        assert manifest["renderer_sha256"] == key.renderer_sha256
        assert manifest["encoder_calls"] == 0


def test_new_route_has_no_model_forward_or_cuda_surface():
    root = Path(__file__).parents[1] / "src" / "measure_vit4ts_v3"
    for name in (
        "spectrogram_renderer.py",
        "spectrogram_registry.py",
        "spectrogram_runner.py",
    ):
        source = (root / name).read_text(encoding="utf-8")
        assert "encode_image(" not in source
        assert "torch.cuda" not in source
        assert "open_clip" not in source
