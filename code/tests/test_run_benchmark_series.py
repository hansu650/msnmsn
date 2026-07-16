from __future__ import annotations

from pathlib import Path

import pytest

from paano_k0.run_benchmark_series import parse_args


def _base_args() -> list[str]:
    return [
        "--config", "protocol.yaml",
        "--manifest", "manifest.csv",
        "--series-id", "series",
        "--trajectory", "PAPERNEG_NONOVERLAP",
        "--seed", "2027",
        "--vendor-root", "vendor",
        "--output-root", "results",
        "--device", "cuda",
    ]


def test_full_runner_cli_has_no_label_surface() -> None:
    parsed = parse_args(_base_args())
    assert not hasattr(parsed, "label")
    with pytest.raises(SystemExit):
        parse_args([*_base_args(), "--labels", "forbidden.csv"])


def test_full_runner_rejects_unregistered_arm() -> None:
    args = _base_args()
    args[args.index("PAPERNEG_NONOVERLAP")] = "RAND_BN"
    with pytest.raises(SystemExit):
        parse_args(args)
