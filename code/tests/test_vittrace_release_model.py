from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_fetch_script(root: Path):
    path = root / "code" / "scripts" / "fetch_vittrace_model.py"
    spec = importlib.util.spec_from_file_location("fetch_vittrace_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_model_is_external_frozen_and_hash_bound() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "models" / "vittrace_vit_b16_openai.json").read_text())
    assert payload["architecture"] == "ViT-B-16"
    assert payload["pretrained"] == "openai"
    assert payload["trainable_parameters_added_by_vittrace"] == 0
    assert payload["redistributed_in_repository"] is False
    assert len(payload["checkpoint_sha256"]) == 64
    assert len(payload["canonical_model_state_sha256"]) == 64


def test_fetch_script_passes_explicit_cache_dir_to_openclip(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    module = _load_fetch_script(root)
    calls: list[dict[str, object]] = []

    def fake_create_model_and_transforms(*args, **kwargs):
        calls.append({"args": args, **kwargs})
        return object(), None, None

    monkeypatch.setattr(
        module.open_clip,
        "create_model_and_transforms",
        fake_create_model_and_transforms,
    )
    monkeypatch.setattr(module, "_OpenClipAdapter", lambda model: model)
    monkeypatch.setattr(module, "freeze_model", lambda model: None)
    monkeypatch.setattr(
        module,
        "hash_model_state",
        lambda model: module.EXPECTED_STATE_SHA256.lower(),
    )

    requested = tmp_path / "model-cache"
    result = module.fetch_and_verify_model(requested)

    expected = str(requested.resolve())
    assert requested.is_dir()
    assert calls == [
        {
            "args": ("ViT-B-16",),
            "pretrained": "openai",
            "device": "cpu",
            "vision_cfg": {"output_tokens": True},
            "cache_dir": expected,
        }
    ]
    assert result["cache_dir"] == expected
    assert result["status"] == "VERIFIED"
