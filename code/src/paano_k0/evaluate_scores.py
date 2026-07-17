"""Evaluator-only threshold-free metrics for committed score artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import atomic_write_json, verify_committed_score
from .config import expand_primary_jobs, load_protocol, load_series_manifest
from .label_data import read_labels, validate_score_alignment
from .schemas import (
    CheckpointKind,
    MetricRow,
    RunJob,
    SeriesSpec,
    canonicalize_unit_interval_metric,
    make_run_id,
    scored_checkpoints,
)
from .vendor import VendorSymbols, load_vendor_symbols


def compute_threshold_free_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    sliding_window: int,
    vendor: VendorSymbols,
    thresholds: int = 250,
) -> dict[str, float]:
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int8)
    if score_values.ndim != 1 or label_values.shape != score_values.shape:
        raise ValueError("metric inputs must be aligned vectors")
    if not np.isfinite(score_values).all() or sliding_window <= 0 or thresholds <= 1:
        raise ValueError("metric inputs/window/threshold grid are invalid")
    metricor = vendor.basic_metricor()
    auprc = float(metricor.metric_PR(label_values, score_values))
    auroc = float(metricor.metric_ROC(label_values, score_values))
    curve = vendor.generate_curve(
        label_values,
        score_values,
        int(sliding_window),
        "opt",
        int(thresholds),
    )
    if len(curve) != 8:
        raise RuntimeError("vendor generate_curve return surface changed")
    vus_roc = float(curve[-2])
    vus_pr = float(curve[-1])
    raw_values = {"vus_pr": vus_pr, "auprc": auprc, "vus_roc": vus_roc, "auroc": auroc}
    return {
        name: canonicalize_unit_interval_metric(name, value)
        for name, value in raw_values.items()
    }


def evaluate_score_artifact(
    run_dir: Path,
    spec: SeriesSpec,
    vendor: VendorSymbols,
) -> MetricRow:
    # This ordering is a hard leakage boundary: labels are inaccessible until
    # the runner payload, provenance, length, and SHA have all been verified.
    scores, manifest = verify_committed_score(run_dir)
    if manifest.series_id != spec.series_id or manifest.data_sha256 != spec.csv_sha256:
        raise ValueError("score provenance does not match the frozen series")
    labels = read_labels(spec)
    validate_score_alignment(labels, scores, spec.rows)
    metrics = compute_threshold_free_metrics(
        scores, labels, manifest.sliding_window, vendor, thresholds=250
    )
    row = MetricRow(
        run_id=manifest.run_id,
        series_id=manifest.series_id,
        family=manifest.family,
        track=manifest.track,
        seed=manifest.seed,
        trajectory=manifest.trajectory,
        checkpoint=manifest.checkpoint,
        vus_pr=metrics["vus_pr"],
        auprc=metrics["auprc"],
        vus_roc=metrics["vus_roc"],
        auroc=metrics["auroc"],
        score_sha256=manifest.score_sha256,
        data_sha256=manifest.data_sha256,
        config_sha256=manifest.config_sha256,
        vendor_sha=manifest.vendor_sha,
    )
    atomic_write_json(Path(run_dir) / "metrics.json", row.to_dict())
    return row


def score_directory(job: RunJob, checkpoint: CheckpointKind) -> Path:
    return (
        job.output_root
        / "runs"
        / job.series.series_id
        / f"seed_{job.seed}"
        / job.trajectory.value
        / "scores"
        / checkpoint.value
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_protocol(args.config)
    series = load_series_manifest(args.manifest)
    vendor = load_vendor_symbols(args.vendor_root, config.baseline.git_sha)
    jobs = expand_primary_jobs(config, series, args.vendor_root, args.results_root)
    evaluated = 0
    for job in jobs:
        for checkpoint in scored_checkpoints(job.trajectory):
            directory = score_directory(job, checkpoint)
            row = evaluate_score_artifact(directory, job.series, vendor)
            expected_id = make_run_id(
                job.series.series_id, job.seed, job.trajectory, checkpoint
            )
            if row.run_id != expected_id:
                raise ValueError("score run_id differs from the registered job")
            evaluated += 1
    if evaluated != 42:
        raise RuntimeError(f"expected 42 primary score artifacts, evaluated {evaluated}")
    print(f"evaluated={evaluated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
