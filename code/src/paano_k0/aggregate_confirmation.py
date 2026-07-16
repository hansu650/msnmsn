"""Aggregate the fixed three-seed PaAno full-benchmark confirmation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import uuid

import numpy as np

from .aggregate_benchmark import PAPER_REFERENCE_SOURCE, PAPER_REPORTED_VUS_PR
from .artifacts import atomic_write_json
from .benchmark_manifest import EXPECTED_TRACK_COUNTS, load_benchmark_manifest
from .config import load_protocol
from .schemas import (
    CheckpointKind,
    MetricRow,
    SeriesSpec,
    Trajectory,
    make_run_id,
)


CONFIRMATION_SEEDS = (2027, 2028, 2029)
MAIN_TRAJECTORY = Trajectory.PAPERNEG_NONOVERLAP
_METRIC_NAMES = ("vus_pr", "auprc", "vus_roc", "auroc")
_METRIC_FIELDS = tuple(item.name for item in fields(MetricRow))
_SEED_TRACK_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "seed",
    "files",
    "families",
    *_METRIC_NAMES,
)
_TRACK_SUMMARY_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "seeds",
    "seed_count",
    "files_per_seed",
    "families",
    *(f"{name}_{stat}" for name in _METRIC_NAMES for stat in ("mean", "std")),
    "std_ddof",
    "paper_method",
    "paper_vus_pr",
    "paper_reference_source",
    "comparison_type",
)


def _metric_from_mapping(payload: Mapping[str, object]) -> MetricRow:
    missing = [name for name in _METRIC_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"confirmation metric fields missing: {missing}")
    values = {name: payload[name] for name in _METRIC_FIELDS}
    values["seed"] = int(values["seed"])
    for name in _METRIC_NAMES:
        values[name] = float(values[name])
    return MetricRow.from_dict(values)


def _read_metric_source(path: Path) -> tuple[MetricRow, ...]:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise ValueError(f"confirmation metric file must be CSV: {source}")
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = tuple(
                _metric_from_mapping(row) for row in csv.DictReader(handle)
            )
    elif source.is_dir():
        metric_root = source / "metrics" if (source / "metrics").is_dir() else source
        paths = tuple(sorted(metric_root.glob("*.json")))
        if not paths:
            raise ValueError(f"confirmation metric directory is empty: {source}")
        loaded: list[MetricRow] = []
        for metric_path in paths:
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            if "metric" in payload and isinstance(payload["metric"], dict):
                payload = payload["metric"]
            if not isinstance(payload, dict):
                raise ValueError(f"invalid confirmation metric JSON: {metric_path}")
            loaded.append(_metric_from_mapping(payload))
        rows = tuple(loaded)
    else:
        raise FileNotFoundError(f"confirmation metric source is missing: {source}")

    run_ids = [row.run_id for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"duplicate confirmation metric run_id in {source}")
    return rows


def _load_exact_seed_rows(
    series: Sequence[SeriesSpec],
    source: Path,
    seed: int,
    *,
    expected_config_sha256: str | None,
    expected_vendor_sha: str | None,
) -> tuple[MetricRow, ...]:
    rows = _read_metric_source(source)
    specs_by_id = {item.series_id: item for item in series}
    expected_ids = {
        make_run_id(
            spec.series_id,
            seed,
            MAIN_TRAJECTORY,
            CheckpointKind.LAST,
        )
        for spec in series
    }
    actual_ids = {row.run_id for row in rows}
    if actual_ids != expected_ids or len(rows) != len(expected_ids):
        raise ValueError(
            f"confirmation seed {seed} metric coverage mismatch: "
            f"missing={sorted(expected_ids - actual_ids)[:5]}, "
            f"extra={sorted(actual_ids - expected_ids)[:5]}, "
            f"expected={len(expected_ids)}, actual={len(rows)}"
        )

    ordered: list[MetricRow] = []
    by_id = {row.run_id: row for row in rows}
    for run_id in sorted(expected_ids):
        row = by_id[run_id]
        spec = specs_by_id.get(row.series_id)
        if spec is None:
            raise ValueError(f"metric references unknown series {row.series_id}")
        if (
            row.run_id
            != make_run_id(
                spec.series_id,
                seed,
                MAIN_TRAJECTORY,
                CheckpointKind.LAST,
            )
            or row.family != spec.family
            or row.track != spec.track
            or row.seed != seed
            or row.trajectory is not MAIN_TRAJECTORY
            or row.checkpoint is not CheckpointKind.LAST
            or row.data_sha256 != spec.csv_sha256
        ):
            raise ValueError(f"confirmation metric provenance mismatch for {run_id}")
        if expected_config_sha256 is not None and (
            row.config_sha256 != expected_config_sha256
        ):
            raise ValueError("confirmation metric config SHA changed")
        if expected_vendor_sha is not None and row.vendor_sha != expected_vendor_sha:
            raise ValueError("confirmation metric vendor SHA changed")
        ordered.append(row)
    return tuple(ordered)


def _atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
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


def _mean_metrics(rows: Sequence[MetricRow]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty confirmation group")
    return {
        name: float(np.mean([float(getattr(row, name)) for row in rows]))
        for name in _METRIC_NAMES
    }


def aggregate_confirmation(
    series: Sequence[SeriesSpec],
    metric_sources: Mapping[int, Path],
    output_dir: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_vendor_sha: str | None = None,
) -> dict[str, object]:
    """Require exact main-arm LAST coverage for all three registered seeds.

    The function reports every seed before computing fixed-seed dispersion.  It
    deliberately has no success gate, retuning rule, or result-selection path.
    """

    specs = tuple(series)
    if not specs or len({item.series_id for item in specs}) != len(specs):
        raise ValueError("confirmation series must be non-empty and unique")
    track_counts = {
        track: sum(item.track == track for item in specs) for track in ("U", "M")
    }
    if any(count <= 0 for count in track_counts.values()):
        raise ValueError("confirmation requires both U and M series")
    if set(metric_sources) != set(CONFIRMATION_SEEDS):
        raise ValueError(
            "confirmation metric sources must be exactly seeds 2027, 2028, and 2029"
        )

    rows_by_seed = {
        seed: _load_exact_seed_rows(
            specs,
            Path(metric_sources[seed]),
            seed,
            expected_config_sha256=expected_config_sha256,
            expected_vendor_sha=expected_vendor_sha,
        )
        for seed in CONFIRMATION_SEEDS
    }
    all_rows = tuple(
        row for seed in CONFIRMATION_SEEDS for row in rows_by_seed[seed]
    )
    expected_metric_count = len(specs) * len(CONFIRMATION_SEEDS)
    if len(all_rows) != expected_metric_count:
        raise RuntimeError(
            f"confirmation coverage mismatch: {len(all_rows)} != {expected_metric_count}"
        )
    config_shas = {row.config_sha256 for row in all_rows}
    vendor_shas = {row.vendor_sha for row in all_rows}
    if len(config_shas) != 1 or len(vendor_shas) != 1:
        raise ValueError("confirmation provenance differs across registered seeds")

    seed_track_rows: list[dict[str, object]] = []
    for seed in CONFIRMATION_SEEDS:
        for track in ("U", "M"):
            subset = [row for row in rows_by_seed[seed] if row.track == track]
            if len(subset) != track_counts[track]:
                raise ValueError(
                    f"confirmation seed {seed} {track} coverage mismatch: "
                    f"{len(subset)} != {track_counts[track]}"
                )
            seed_track_rows.append(
                {
                    "trajectory": MAIN_TRAJECTORY.value,
                    "checkpoint": CheckpointKind.LAST.value,
                    "arm": f"{MAIN_TRAJECTORY.value}_{CheckpointKind.LAST.value}",
                    "track": track,
                    "seed": seed,
                    "files": len(subset),
                    "families": len({row.family for row in subset}),
                    **_mean_metrics(subset),
                }
            )

    track_summary: list[dict[str, object]] = []
    for track in ("U", "M"):
        subset = [row for row in seed_track_rows if row["track"] == track]
        if [int(row["seed"]) for row in subset] != list(CONFIRMATION_SEEDS):
            raise RuntimeError(f"confirmation did not preserve every seed for {track}")
        metrics: dict[str, float] = {}
        for name in _METRIC_NAMES:
            values = np.asarray([float(row[name]) for row in subset], dtype=np.float64)
            metrics[f"{name}_mean"] = float(np.mean(values))
            metrics[f"{name}_std"] = float(np.std(values, ddof=0))
        track_summary.append(
            {
                "trajectory": MAIN_TRAJECTORY.value,
                "checkpoint": CheckpointKind.LAST.value,
                "arm": f"{MAIN_TRAJECTORY.value}_{CheckpointKind.LAST.value}",
                "track": track,
                "seeds": ";".join(str(seed) for seed in CONFIRMATION_SEEDS),
                "seed_count": len(CONFIRMATION_SEEDS),
                "files_per_seed": track_counts[track],
                "families": len(
                    {item.family for item in specs if item.track == track}
                ),
                **metrics,
                "std_ddof": 0,
                "paper_method": "PaAno (paper-reported)",
                "paper_vus_pr": float(PAPER_REPORTED_VUS_PR[track]),
                "paper_reference_source": PAPER_REFERENCE_SOURCE,
                "comparison_type": "descriptive_external_paper_reported",
            }
        )

    summary: dict[str, object] = {
        "schema_version": "paano-full-confirmation-v1",
        "trajectory": MAIN_TRAJECTORY.value,
        "checkpoint": CheckpointKind.LAST.value,
        "arm": f"{MAIN_TRAJECTORY.value}_{CheckpointKind.LAST.value}",
        "seeds": list(CONFIRMATION_SEEDS),
        "series_per_seed": len(specs),
        "track_files_per_seed": track_counts,
        "metric_count": len(all_rows),
        "seed_track_row_count": len(seed_track_rows),
        "std_ddof": 0,
        "selection_applied": False,
        "retuning_applied": False,
        "result_dropping_applied": False,
        "paper_reference_type": "external_paper_reported_descriptive_only",
        "paper_reference_source": PAPER_REFERENCE_SOURCE,
        "paper_reported_vus_pr": dict(PAPER_REPORTED_VUS_PR),
        "config_sha256": next(iter(config_shas)),
        "vendor_sha": next(iter(vendor_shas)),
        "seed_track_metrics": seed_track_rows,
        "track_summary": track_summary,
    }

    destination = Path(output_dir)
    _atomic_write_csv(
        destination / "confirmation_seed_track_metrics.csv",
        seed_track_rows,
        _SEED_TRACK_FIELDS,
    )
    _atomic_write_csv(
        destination / "confirmation_track_summary.csv",
        track_summary,
        _TRACK_SUMMARY_FIELDS,
    )
    atomic_write_json(destination / "confirmation_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-2027-metrics", type=Path, required=True)
    parser.add_argument("--seed-2028-metrics", type=Path, required=True)
    parser.add_argument("--seed-2029-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.config)
    protocol.trajectory(MAIN_TRAJECTORY)
    series = load_benchmark_manifest(args.manifest)
    observed_counts = {
        track: sum(item.track == track for item in series) for track in ("U", "M")
    }
    if observed_counts != EXPECTED_TRACK_COUNTS:
        raise ValueError("full confirmation manifest track counts changed")
    summary = aggregate_confirmation(
        series,
        {
            2027: args.seed_2027_metrics,
            2028: args.seed_2028_metrics,
            2029: args.seed_2029_metrics,
        },
        args.output_dir,
        expected_config_sha256=protocol.source_sha256,
        expected_vendor_sha=protocol.baseline.git_sha,
    )
    print(
        "CONFIRMATION_AGGREGATE_COMPLETE "
        f"seeds={','.join(str(seed) for seed in CONFIRMATION_SEEDS)} "
        f"metrics={summary['metric_count']} seed_track_rows={summary['seed_track_row_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
