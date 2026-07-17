"""Create the label-free IHP topology certificate from a frozen token cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from measure_vit4ts.ihp import incidence_certificate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.token_cache, allow_pickle=False) as payload:
        large = torch.from_numpy(payload["large_mask"])
        mid = torch.from_numpy(payload["mid_mask"])
    certificate = {
        "status": "STRUCTURE_CERTIFIED",
        "label_free": True,
        "base_grid": "14x14",
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
