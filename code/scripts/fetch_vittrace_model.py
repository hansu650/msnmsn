"""Download and verify the frozen OpenCLIP ViT-B/16 used by ViTTrace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import open_clip

from measure_vit4ts.cache import freeze_model, hash_model_state
from measure_vit4ts_v3.encoder_runner import _OpenClipAdapter


EXPECTED_STATE_SHA256 = "CAAD38C6AB955B9329739541FE1BA110E261936445528FA3B94D0B73E96672CD"


def fetch_and_verify_model(cache_dir: Path | None = None) -> dict[str, object]:
    """Load the registered backbone from ``cache_dir`` and verify its state.

    ``open_clip`` may already be imported before this function runs, so setting
    Hugging Face environment variables alone is not a reliable cache override.
    The resolved directory is therefore also passed directly to OpenCLIP.
    """

    resolved_cache: Path | None = None
    if cache_dir is not None:
        resolved_cache = cache_dir.expanduser().resolve()
        resolved_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(resolved_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(resolved_cache / "hub")

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai",
        device="cpu",
        vision_cfg={"output_tokens": True},
        cache_dir=None if resolved_cache is None else str(resolved_cache),
    )
    adapter = _OpenClipAdapter(model)
    freeze_model(adapter)
    actual = hash_model_state(adapter).upper()
    if actual != EXPECTED_STATE_SHA256:
        raise RuntimeError(
            f"model-state SHA256 mismatch: expected {EXPECTED_STATE_SHA256}, got {actual}"
        )
    return {
        "status": "VERIFIED",
        "architecture": "ViT-B-16",
        "pretrained": "openai",
        "model_state_sha256": actual,
        "cache_dir": None if resolved_cache is None else str(resolved_cache),
        "trainable_parameters_added_by_vittrace": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face/OpenCLIP cache root (prefer a data drive).",
    )
    args = parser.parse_args()
    print(json.dumps(fetch_and_verify_model(args.cache_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
