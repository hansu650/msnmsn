"""Prove exact serial/spawn parity on fixed short U/M benchmark series."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Sequence

from paano_k0.benchmark_manifest import load_benchmark_manifest
from paano_k0.config import load_protocol
from paano_k0.evaluate_benchmark import (
    REGISTERED_BENCHMARK_TRAJECTORIES,
    evaluate_registered_benchmark,
)
from paano_k0.vendor import load_vendor_symbols


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args(argv)


def _short_representatives(series):
    selected = []
    for track in ("U", "M"):
        candidates = tuple(item for item in series if item.track == track)
        if len(candidates) < 2:
            raise ValueError(f"manifest has fewer than two {track} representatives")
        selected.extend(sorted(candidates, key=lambda item: (item.rows, item.series_id))[:2])
    return tuple(selected)


def _compare_outputs(serial_dir: Path, parallel_dir: Path) -> None:
    for name in ("file_metrics.csv", "evaluation_summary.json"):
        if (serial_dir / name).read_bytes() != (parallel_dir / name).read_bytes():
            raise RuntimeError(f"serial/parallel terminal artifact mismatch: {name}")
    serial_metrics = sorted((serial_dir / "metrics").glob("*.json"))
    parallel_metrics = sorted((parallel_dir / "metrics").glob("*.json"))
    if [item.name for item in serial_metrics] != [item.name for item in parallel_metrics]:
        raise RuntimeError("serial/parallel metric coverage mismatch")
    for serial_path, parallel_path in zip(serial_metrics, parallel_metrics, strict=True):
        if serial_path.read_bytes() != parallel_path.read_bytes():
            raise RuntimeError(f"serial/parallel metric mismatch: {serial_path.name}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("parity output root must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(args.config)
    vendor = load_vendor_symbols(args.vendor_root, protocol.baseline.git_sha)
    series = _short_representatives(load_benchmark_manifest(args.manifest))
    serial_dir = output_root / "serial"
    parallel_dir = output_root / "parallel"
    resume_dir = output_root / "resume"
    serial_rows = evaluate_registered_benchmark(
        series,
        args.results_root,
        serial_dir,
        vendor,
        REGISTERED_BENCHMARK_TRAJECTORIES,
        seed=args.seed,
        expected_config_sha256=protocol.source_sha256,
        expected_vendor_sha=protocol.baseline.git_sha,
    )
    parallel_rows = evaluate_registered_benchmark(
        series,
        args.results_root,
        parallel_dir,
        vendor,
        REGISTERED_BENCHMARK_TRAJECTORIES,
        seed=args.seed,
        expected_config_sha256=protocol.source_sha256,
        expected_vendor_sha=protocol.baseline.git_sha,
        workers=4,
        resume_existing=True,
    )
    if [row.to_dict() for row in serial_rows] != [row.to_dict() for row in parallel_rows]:
        raise RuntimeError("serial/parallel MetricRow mismatch")
    _compare_outputs(serial_dir, parallel_dir)

    # Seed a partial cache with half of the exact parallel JSONs.  Reused files
    # must not be rewritten, while missing files must reproduce exact bytes.
    (resume_dir / "metrics").mkdir(parents=True)
    shutil.copy2(
        parallel_dir / "evaluator_contract.json",
        resume_dir / "evaluator_contract.json",
    )
    parallel_metric_paths = sorted((parallel_dir / "metrics").glob("*.json"))
    copied = parallel_metric_paths[::2]
    before = {}
    for source in copied:
        destination = resume_dir / "metrics" / source.name
        shutil.copy2(source, destination)
        before[source.name] = (destination.read_bytes(), destination.stat().st_mtime_ns)
    resumed_rows = evaluate_registered_benchmark(
        series,
        args.results_root,
        resume_dir,
        vendor,
        REGISTERED_BENCHMARK_TRAJECTORIES,
        seed=args.seed,
        expected_config_sha256=protocol.source_sha256,
        expected_vendor_sha=protocol.baseline.git_sha,
        workers=4,
        resume_existing=True,
    )
    if [row.to_dict() for row in parallel_rows] != [row.to_dict() for row in resumed_rows]:
        raise RuntimeError("parallel/resumed MetricRow mismatch")
    _compare_outputs(parallel_dir, resume_dir)
    for name, (payload, mtime_ns) in before.items():
        cached = resume_dir / "metrics" / name
        if cached.read_bytes() != payload or cached.stat().st_mtime_ns != mtime_ns:
            raise RuntimeError(f"resume rewrote a valid cached metric: {name}")
    print(
        "EVALUATOR_PARITY_PASS "
        f"series={','.join(item.series_id for item in series)} "
        f"metrics={len(serial_rows)} resumed={len(copied)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
