"""Run one registered full-benchmark PaAno trajectory without label access."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .benchmark_manifest import load_benchmark_series
from .config import load_protocol
from .run_series import run_job
from .schemas import RunJob, Trajectory


_ALLOWED = (
    Trajectory.OFFICIAL,
    Trajectory.PAPERNEG,
    Trajectory.PAPERNEG_NONOVERLAP,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--series-id", required=True)
    parser.add_argument(
        "--trajectory", choices=tuple(item.value for item in _ALLOWED), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed not in (2027, 2028, 2029):
        raise ValueError("full benchmark seed is not registered")
    protocol = load_protocol(args.config)
    spec = load_benchmark_series(args.manifest, args.series_id)
    job = RunJob(
        series=spec,
        trajectory=Trajectory(args.trajectory),
        seed=args.seed,
        protocol_path=protocol.source_path,
        vendor_root=args.vendor_root,
        output_root=args.output_root,
        device=args.device,
    )
    run_job(job, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
