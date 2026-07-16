"""Evaluator-only metrics for registered full-benchmark LAST score artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence
import uuid

import numpy as np

from .artifacts import atomic_write_json, verify_committed_score
from .benchmark_manifest import load_benchmark_manifest
from .config import load_protocol
from .evaluate_scores import compute_threshold_free_metrics
from .label_data import read_labels, validate_score_alignment
from .schemas import (
    CheckpointKind,
    MetricRow,
    ScoreManifest,
    SeriesSpec,
    Trajectory,
    make_run_id,
)
from .vendor import VendorSymbols, load_vendor_symbols


REGISTERED_BENCHMARK_TRAJECTORIES = (
    Trajectory.PAPERNEG_NONOVERLAP,
    Trajectory.PAPERNEG,
    Trajectory.OFFICIAL,
)
_REGISTERED_SET = frozenset(REGISTERED_BENCHMARK_TRAJECTORIES)
_FILE_METRIC_FIELDS = (
    "run_id",
    "series_id",
    "family",
    "track",
    "seed",
    "trajectory",
    "checkpoint",
    "arm",
    "vus_pr",
    "auprc",
    "vus_roc",
    "auroc",
    "score_sha256",
    "data_sha256",
    "config_sha256",
    "vendor_sha",
)


@dataclass(frozen=True, slots=True)
class _VerifiedArtifact:
    spec: SeriesSpec
    trajectory: Trajectory
    directory: Path
    manifest: ScoreManifest


def _coerce_trajectories(
    trajectories: Sequence[Trajectory | str],
) -> tuple[Trajectory, ...]:
    values = tuple(
        item if isinstance(item, Trajectory) else Trajectory(str(item))
        for item in trajectories
    )
    if not values:
        raise ValueError("at least one benchmark trajectory is required")
    if len(set(values)) != len(values):
        raise ValueError("benchmark trajectories must be unique")
    unknown = [item.value for item in values if item not in _REGISTERED_SET]
    if unknown:
        raise ValueError(f"unregistered benchmark trajectories: {unknown}")
    return values


def _score_directory(
    results_root: Path,
    spec: SeriesSpec,
    seed: int,
    trajectory: Trajectory,
    checkpoint: CheckpointKind,
) -> Path:
    return (
        Path(results_root)
        / "runs"
        / spec.series_id
        / f"seed_{seed}"
        / trajectory.value
        / "scores"
        / checkpoint.value
    )


def _verify_expected_score(
    directory: Path,
    spec: SeriesSpec,
    trajectory: Trajectory,
    seed: int,
    checkpoint: CheckpointKind,
    *,
    expected_config_sha256: str | None,
    expected_vendor_sha: str | None,
) -> tuple[np.ndarray, ScoreManifest]:
    scores, manifest = verify_committed_score(directory)
    expected_run_id = make_run_id(spec.series_id, seed, trajectory, checkpoint)
    observed = (
        manifest.run_id,
        manifest.series_id,
        manifest.family,
        manifest.track,
        manifest.seed,
        manifest.trajectory,
        manifest.checkpoint,
        manifest.data_sha256,
    )
    expected = (
        expected_run_id,
        spec.series_id,
        spec.family,
        spec.track,
        seed,
        trajectory,
        checkpoint,
        spec.csv_sha256,
    )
    if observed != expected:
        raise ValueError(
            f"score provenance mismatch for {spec.series_id}/{trajectory.value}"
        )
    if expected_config_sha256 is not None and (
        manifest.config_sha256 != expected_config_sha256
    ):
        raise ValueError("score config SHA differs from the registered protocol")
    if expected_vendor_sha is not None and manifest.vendor_sha != expected_vendor_sha:
        raise ValueError("score vendor SHA differs from the registered baseline")
    if scores.shape != (spec.rows,):
        raise ValueError(
            f"score length mismatch for {spec.series_id}: {scores.shape}"
        )
    return scores, manifest


def _atomic_write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _metric_payload(row: MetricRow) -> dict[str, object]:
    payload = row.to_dict()
    payload["arm"] = row.arm
    return payload


def evaluate_registered_benchmark(
    series: Sequence[SeriesSpec],
    results_root: Path,
    output_dir: Path,
    vendor: VendorSymbols,
    trajectories: Sequence[Trajectory | str],
    *,
    seed: int,
    checkpoint: CheckpointKind | str = CheckpointKind.LAST,
    expected_config_sha256: str | None = None,
    expected_vendor_sha: str | None = None,
) -> tuple[MetricRow, ...]:
    """Evaluate exact registered coverage after a global score-hash preflight.

    Phase one verifies every expected committed score while the label reader is
    unreachable.  Only after all hashes and provenance pass does phase two load
    each series label once and reuse it across the requested registered arms.
    """

    specs = tuple(series)
    if not specs or len({item.series_id for item in specs}) != len(specs):
        raise ValueError("benchmark series must be non-empty and unique")
    selected = _coerce_trajectories(trajectories)
    checkpoint_value = (
        checkpoint
        if isinstance(checkpoint, CheckpointKind)
        else CheckpointKind(str(checkpoint))
    )
    if checkpoint_value is not CheckpointKind.LAST:
        raise ValueError("full benchmark evaluation is frozen to LAST only")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    # Global preflight.  Do not move label I/O above this complete loop.
    verified: list[_VerifiedArtifact] = []
    for spec in specs:
        for trajectory in selected:
            directory = _score_directory(
                results_root, spec, seed, trajectory, checkpoint_value
            )
            _, manifest = _verify_expected_score(
                directory,
                spec,
                trajectory,
                seed,
                checkpoint_value,
                expected_config_sha256=expected_config_sha256,
                expected_vendor_sha=expected_vendor_sha,
            )
            verified.append(_VerifiedArtifact(spec, trajectory, directory, manifest))

    expected_count = len(specs) * len(selected)
    if len(verified) != expected_count:
        raise RuntimeError(
            f"benchmark preflight coverage mismatch: {len(verified)} != {expected_count}"
        )
    by_series = {
        spec.series_id: tuple(item for item in verified if item.spec is spec)
        for spec in specs
    }
    if any(len(items) != len(selected) for items in by_series.values()):
        raise RuntimeError("benchmark preflight did not cover every requested arm")

    metric_rows: list[MetricRow] = []
    metric_root = Path(output_dir) / "metrics"
    for spec in specs:
        # The first label read occurs here, after every score artifact passed.
        labels = read_labels(spec)
        for item in by_series[spec.series_id]:
            scores, manifest = _verify_expected_score(
                item.directory,
                spec,
                item.trajectory,
                seed,
                checkpoint_value,
                expected_config_sha256=expected_config_sha256,
                expected_vendor_sha=expected_vendor_sha,
            )
            if manifest.score_sha256 != item.manifest.score_sha256:
                raise RuntimeError("score payload changed after global preflight")
            validate_score_alignment(labels, scores, spec.rows)
            metrics = compute_threshold_free_metrics(
                scores, labels, manifest.sliding_window, vendor, thresholds=250
            )
            row = MetricRow(
                run_id=manifest.run_id,
                series_id=spec.series_id,
                family=spec.family,
                track=spec.track,
                seed=seed,
                trajectory=item.trajectory,
                checkpoint=checkpoint_value,
                vus_pr=metrics["vus_pr"],
                auprc=metrics["auprc"],
                vus_roc=metrics["vus_roc"],
                auroc=metrics["auroc"],
                score_sha256=manifest.score_sha256,
                data_sha256=spec.csv_sha256,
                config_sha256=manifest.config_sha256,
                vendor_sha=manifest.vendor_sha,
            )
            atomic_write_json(metric_root / f"{row.run_id}.json", row.to_dict())
            metric_rows.append(row)

    if len(metric_rows) != expected_count:
        raise RuntimeError(
            f"benchmark evaluation coverage mismatch: {len(metric_rows)} != {expected_count}"
        )
    payload = [_metric_payload(row) for row in metric_rows]
    _atomic_write_csv(Path(output_dir) / "file_metrics.csv", payload, _FILE_METRIC_FIELDS)
    atomic_write_json(
        Path(output_dir) / "evaluation_summary.json",
        {
            "schema_version": "paano-full-evaluation-v1",
            "seed": seed,
            "checkpoint": checkpoint_value.value,
            "trajectories": [item.value for item in selected],
            "series_count": len(specs),
            "metric_count": len(metric_rows),
            "labels_loaded_after_global_preflight": True,
        },
    )
    return tuple(metric_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--trajectories",
        nargs="+",
        choices=tuple(item.value for item in REGISTERED_BENCHMARK_TRAJECTORIES),
        required=True,
    )
    parser.add_argument(
        "--checkpoint", choices=(CheckpointKind.LAST.value,), required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed not in (2027, 2028, 2029):
        raise ValueError("full benchmark seed is not registered")
    protocol = load_protocol(args.config)
    trajectories = _coerce_trajectories(args.trajectories)
    for trajectory in trajectories:
        protocol.trajectory(trajectory)
    series = load_benchmark_manifest(args.manifest)
    vendor = load_vendor_symbols(args.vendor_root, protocol.baseline.git_sha)
    rows = evaluate_registered_benchmark(
        series,
        args.results_root,
        args.output_dir,
        vendor,
        trajectories,
        seed=args.seed,
        checkpoint=args.checkpoint,
        expected_config_sha256=protocol.source_sha256,
        expected_vendor_sha=protocol.baseline.git_sha,
    )
    print(
        f"BENCHMARK_EVALUATION_COMPLETE series={len(series)} "
        f"trajectories={len(trajectories)} metrics={len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

