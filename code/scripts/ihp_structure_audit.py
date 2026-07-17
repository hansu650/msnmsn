"""Create the label-free IHP topology certificate from frozen token caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from measure_vit4ts.ihp import incidence_certificate


VENDOR_COMMIT = "8ab8c16414eb2c1a861dfc3e76f458180035a879"


def _mask_sha256(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(array, dtype=np.int64)
    return hashlib.sha256(normalized.tobytes()).hexdigest().upper()


def _load_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.ascontiguousarray(payload["large_mask"], dtype=np.int64),
            np.ascontiguousarray(payload["mid_mask"], dtype=np.int64),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--token-cache", type=Path)
    source.add_argument(
        "--cache-root",
        type=Path,
        help="Recursively verify every clip_tokens.npz under this directory.",
    )
    parser.add_argument("--vendor-commit", default=VENDOR_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    caches = (
        [args.token_cache]
        if args.token_cache is not None
        else sorted(args.cache_root.rglob("clip_tokens.npz"))
    )
    if not caches:
        raise ValueError("no frozen token caches found")

    large_array, mid_array = _load_masks(caches[0])
    expected_hashes = {
        "large": _mask_sha256(large_array),
        "mid": _mask_sha256(mid_array),
    }
    for cache in caches[1:]:
        large_current, mid_current = _load_masks(cache)
        current_hashes = {
            "large": _mask_sha256(large_current),
            "mid": _mask_sha256(mid_current),
        }
        if current_hashes != expected_hashes:
            raise ValueError(f"mask provenance mismatch: {cache}")

    large = torch.from_numpy(large_array)
    mid = torch.from_numpy(mid_array)
    certificate = {
        "status": "STRUCTURE_CERTIFIED",
        "label_free": True,
        "base_grid": "14x14",
        "cache_files_verified": len(caches),
        "mask_hash_definition": "SHA256 over C-contiguous int64 array bytes",
        "mask_sha256": expected_hashes,
        "vendor": {
            "commit": args.vendor_commit,
            "mask_generation": {
                "large": "src/models/clip_vision.py:33 (kernel_size=48)",
                "mid": "src/models/clip_vision.py:34 (kernel_size=32)",
            },
            "shifted_query": "src/models/model_utils.py:34",
        },
        "large_scale": incidence_certificate(large, 196),
        "mid_scale": incidence_certificate(mid, 196),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
