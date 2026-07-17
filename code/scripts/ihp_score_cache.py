"""Apply IHP to one frozen ViT4TS token cache without reading labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from measure_vit4ts.ihp_cache import score_token_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    maps, columns = score_token_cache(args.token_cache, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, base_maps=maps, window_columns=columns)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
