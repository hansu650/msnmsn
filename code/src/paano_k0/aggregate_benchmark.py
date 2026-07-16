"""Aggregate the fixed PaAno full benchmark and paper-reported comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import uuid

import numpy as np

from .artifacts import atomic_write_json
from .benchmark_manifest import EXPECTED_TRACK_COUNTS, load_benchmark_manifest
from .config import load_protocol
from .evaluate_benchmark import REGISTERED_BENCHMARK_TRAJECTORIES
from .schemas import (
    CheckpointKind,
    MetricRow,
    ScoreManifest,
    SeriesSpec,
    Trajectory,
    make_run_id,
)


MAIN_TRAJECTORY = Trajectory.PAPERNEG_NONOVERLAP
PAPER_REPORTED_VUS_PR: Mapping[str, float] = {"U": 0.5296, "M": 0.4263}
PAPER_REFERENCE_SOURCE = "PaAno Table 15 default full-Eval (k=3)"
_METRIC_NAMES = ("vus_pr", "auprc", "vus_roc", "auroc")
_FILE_FIELDS = (
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
_FAMILY_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "family",
    "seed",
    "files",
    *_METRIC_NAMES,
)
_TRACK_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "seed",
    "files",
    "families",
    *_METRIC_NAMES,
)
_OVERALL_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "seed",
    "files",
    "families",
    "tracks",
    *_METRIC_NAMES,
)


def _read_metric(path: Path) -> MetricRow:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "metric" in payload and isinstance(payload["metric"], dict):
        payload = payload["metric"]
    return MetricRow.from_dict(payload)


def _metric_paths(metrics_dir: Path) -> tuple[Path, ...]:
    root = Path(metrics_dir)
    nested = root / "metrics"
    source = nested if nested.is_dir() else root
    return tuple(sorted(source.glob("*.json")))


def _load_exact_rows(
    series: Sequence[SeriesSpec],
    metrics_dir: Path,
    trajectories: Sequence[Trajectory],
    seed: int,
    checkpoint: CheckpointKind,
    *,
    expected_config_sha256: str | None,
    expected_vendor_sha: str | None,
) -> tuple[MetricRow, ...]:
    paths = _metric_paths(metrics_dir)
    all_rows = tuple(_read_metric(path) for path in paths)
    all_ids = [row.run_id for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate benchmark metric run_id detected")
    selected = tuple(row for row in all_rows if row.seed == seed)

    specs_by_id = {item.series_id: item for item in series}
    expected_ids = {
        make_run_id(spec.series_id, seed, trajectory, checkpoint)
        for spec in series
        for trajectory in trajectories
    }
    by_id = {row.run_id: row for row in selected}
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            "benchmark metric coverage mismatch: "
            f"missing={sorted(expected_ids - actual_ids)[:5]}, "
            f"extra={sorted(actual_ids - expected_ids)[:5]}, "
            f"expected={len(expected_ids)}, actual={len(actual_ids)}"
        )

    ordered: list[MetricRow] = []
    for run_id in sorted(expected_ids):
        row = by_id[run_id]
        spec = specs_by_id.get(row.series_id)
        if spec is None:
            raise ValueError(f"metric references unknown series {row.series_id}")
        expected_id = make_run_id(
            spec.series_id, seed, row.trajectory, checkpoint
        )
        if (
            row.run_id != expected_id
            or row.family != spec.family
            or row.track != spec.track
            or row.data_sha256 != spec.csv_sha256
            or row.trajectory not in trajectories
            or row.checkpoint is not checkpoint
        ):
            raise ValueError(f"metric provenance mismatch for {row.run_id}")
        if expected_config_sha256 is not None and (
            row.config_sha256 != expected_config_sha256
        ):
            raise ValueError("metric config SHA differs from the registered protocol")
        if expected_vendor_sha is not None and row.vendor_sha != expected_vendor_sha:
            raise ValueError("metric vendor SHA differs from the registered baseline")
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


def _file_payload(row: MetricRow) -> dict[str, object]:
    payload = row.to_dict()
    payload["arm"] = row.arm
    return payload


def _mean_metrics(rows: Sequence[MetricRow]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric group")
    return {
        name: float(np.mean([float(getattr(row, name)) for row in rows]))
        for name in _METRIC_NAMES
    }


def _family_rows(rows: Sequence[MetricRow]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    keys = sorted(
        {(row.trajectory, row.track, row.family, row.seed) for row in rows},
        key=lambda item: (item[0].value, item[1], item[2], item[3]),
    )
    for trajectory, track, family, seed in keys:
        subset = [
            row
            for row in rows
            if (row.trajectory, row.track, row.family, row.seed)
            == (trajectory, track, family, seed)
        ]
        payload.append(
            {
                "trajectory": trajectory.value,
                "checkpoint": CheckpointKind.LAST.value,
                "arm": f"{trajectory.value}_{CheckpointKind.LAST.value}",
                "track": track,
                "family": family,
                "seed": seed,
                "files": len(subset),
                **_mean_metrics(subset),
            }
        )
    return payload


def _track_rows(rows: Sequence[MetricRow]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    keys = sorted(
        {(row.trajectory, row.track, row.seed) for row in rows},
        key=lambda item: (item[0].value, item[1], item[2]),
    )
    for trajectory, track, seed in keys:
        subset = [
            row
            for row in rows
            if (row.trajectory, row.track, row.seed) == (trajectory, track, seed)
        ]
        payload.append(
            {
                "trajectory": trajectory.value,
                "checkpoint": CheckpointKind.LAST.value,
                "arm": f"{trajectory.value}_{CheckpointKind.LAST.value}",
                "track": track,
                "seed": seed,
                "files": len(subset),
                "families": len({row.family for row in subset}),
                **_mean_metrics(subset),
            }
        )
    return payload


def _overall_rows(rows: Sequence[MetricRow]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    keys = sorted(
        {(row.trajectory, row.seed) for row in rows},
        key=lambda item: (item[0].value, item[1]),
    )
    for trajectory, seed in keys:
        subset = [
            row
            for row in rows
            if (row.trajectory, row.seed) == (trajectory, seed)
        ]
        payload.append(
            {
                "trajectory": trajectory.value,
                "checkpoint": CheckpointKind.LAST.value,
                "arm": f"{trajectory.value}_{CheckpointKind.LAST.value}",
                "seed": seed,
                "files": len(subset),
                "families": len({(row.track, row.family) for row in subset}),
                "tracks": len({row.track for row in subset}),
                **_mean_metrics(subset),
            }
        )
    return payload


def _runtime_rows(
    series: Sequence[SeriesSpec],
    results_root: Path,
    trajectories: Sequence[Trajectory],
    seed: int,
) -> list[dict[str, object]]:
    raw: list[dict[str, object]] = []
    for spec in series:
        for trajectory in trajectories:
            trajectory_dir = (
                Path(results_root)
                / "runs"
                / spec.series_id
                / f"seed_{seed}"
                / trajectory.value
            )
            summary_path = trajectory_dir / "training_summary.json"
            score_manifest_path = (
                trajectory_dir
                / "scores"
                / CheckpointKind.LAST.value
                / "score_manifest.json"
            )
            if not summary_path.is_file() or not score_manifest_path.is_file():
                raise FileNotFoundError(
                    f"missing runtime provenance for {spec.series_id}/{trajectory.value}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            score_manifest = ScoreManifest.from_dict(
                json.loads(score_manifest_path.read_text(encoding="utf-8"))
            )
            if (
                str(summary.get("series_id")) != spec.series_id
                or str(summary.get("family")) != spec.family
                or str(summary.get("track")) != spec.track
                or int(summary.get("seed", -1)) != seed
                or str(summary.get("trajectory")) != trajectory.value
                or score_manifest.run_id
                != make_run_id(
                    spec.series_id, seed, trajectory, CheckpointKind.LAST
                )
            ):
                raise ValueError(
                    f"runtime provenance mismatch for {spec.series_id}/{trajectory.value}"
                )
            raw.append(
                {
                    "trajectory": trajectory,
                    "track": spec.track,
                    "training_runtime_seconds": float(summary["runtime_seconds"]),
                    "training_peak_vram_mib": float(summary["peak_vram_mib"]),
                    "scoring_runtime_seconds": float(score_manifest.runtime_seconds),
                    "scoring_peak_vram_mib": float(score_manifest.peak_vram_mib),
                }
            )

    payload: list[dict[str, object]] = []
    for trajectory in trajectories:
        for track in ("U", "M"):
            subset = [
                item
                for item in raw
                if item["trajectory"] is trajectory and item["track"] == track
            ]
            if not subset:
                continue
            train = [float(item["training_runtime_seconds"]) for item in subset]
            score = [float(item["scoring_runtime_seconds"]) for item in subset]
            payload.append(
                {
                    "trajectory": trajectory.value,
                    "checkpoint": CheckpointKind.LAST.value,
                    "track": track,
                    "seed": seed,
                    "files": len(subset),
                    "training_runtime_seconds_sum": float(np.sum(train)),
                    "training_runtime_seconds_mean": float(np.mean(train)),
                    "scoring_runtime_seconds_sum": float(np.sum(score)),
                    "scoring_runtime_seconds_mean": float(np.mean(score)),
                    "training_peak_vram_mib_max": max(
                        float(item["training_peak_vram_mib"]) for item in subset
                    ),
                    "scoring_peak_vram_mib_max": max(
                        float(item["scoring_peak_vram_mib"]) for item in subset
                    ),
                }
            )
    return payload


def _paper_comparison(
    track_rows: Sequence[Mapping[str, object]],
    expected_track_counts: Mapping[str, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    main = {
        str(row["track"]): row
        for row in track_rows
        if str(row["trajectory"]) == MAIN_TRAJECTORY.value
    }
    comparison: list[dict[str, object]] = []
    track_decisions: dict[str, object] = {}
    for track in ("U", "M"):
        if track not in main:
            raise ValueError(f"missing main-arm track aggregate for {track}")
        row = main[track]
        observed_files = int(row["files"])
        expected_files = int(expected_track_counts[track])
        if observed_files != expected_files:
            raise ValueError(
                f"main-arm {track} coverage mismatch: {observed_files} != {expected_files}"
            )
        ours = float(row["vus_pr"])
        paper = float(PAPER_REPORTED_VUS_PR[track])
        exceeds = bool(ours > paper)
        current = {
            "track": track,
            "files": observed_files,
            "ours_method": f"{MAIN_TRAJECTORY.value}_{CheckpointKind.LAST.value}",
            "ours_vus_pr": ours,
            "paper_method": "PaAno (paper-reported)",
            "paper_vus_pr": paper,
            "paper_reference_source": PAPER_REFERENCE_SOURCE,
            "delta_vus_pr": ours - paper,
            "exceeds_paper_reported": exceeds,
            "comparison_type": "external_paper_reported",
        }
        comparison.append(current)
        track_decisions[track] = current
    passed = all(bool(item["exceeds_paper_reported"]) for item in comparison)
    decision = {
        "schema_version": "paano-full-benchmark-decision-v1",
        "outcome": (
            "CONTINUE_FULL_CONFIRMATION" if passed else "STOP_FULL_MAIN_FAILURE"
        ),
        "main_trajectory": MAIN_TRAJECTORY.value,
        "checkpoint": CheckpointKind.LAST.value,
        "paper_reference_type": "external_paper_reported",
        "paper_reference_source": PAPER_REFERENCE_SOURCE,
        "paper_reported_vus_pr": dict(PAPER_REPORTED_VUS_PR),
        "success_requires_both_tracks": True,
        "both_tracks_exceed": passed,
        "conditional_confirmation_seeds": [2028, 2029] if passed else [],
        "tracks": track_decisions,
        "missing_count": 0,
    }
    return comparison, decision


def aggregate_full_benchmark(
    series: Sequence[SeriesSpec],
    metrics_dir: Path,
    results_root: Path,
    output_dir: Path,
    *,
    seed: int,
    expected_config_sha256: str | None = None,
    expected_vendor_sha: str | None = None,
) -> dict[str, object]:
    """Require complete three-arm LAST coverage and write compact tables."""

    specs = tuple(series)
    if not specs or len({item.series_id for item in specs}) != len(specs):
        raise ValueError("benchmark series must be non-empty and unique")
    trajectories = REGISTERED_BENCHMARK_TRAJECTORIES
    rows = _load_exact_rows(
        specs,
        metrics_dir,
        trajectories,
        seed,
        CheckpointKind.LAST,
        expected_config_sha256=expected_config_sha256,
        expected_vendor_sha=expected_vendor_sha,
    )
    expected_count = len(specs) * len(trajectories)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"benchmark aggregate coverage mismatch: {len(rows)} != {expected_count}"
        )

    file_payload = [_file_payload(row) for row in rows]
    family_payload = _family_rows(rows)
    track_payload = _track_rows(rows)
    overall_payload = _overall_rows(rows)
    observed_track_counts = {
        track: sum(item.track == track for item in specs) for track in ("U", "M")
    }
    comparison, decision = _paper_comparison(
        track_payload, observed_track_counts
    )
    decision.update(
        {
            "seed": seed,
            "series_count": len(specs),
            "metric_count": len(rows),
            "config_sha256": rows[0].config_sha256,
            "vendor_sha": rows[0].vendor_sha,
        }
    )
    runtime_payload = _runtime_rows(specs, results_root, trajectories, seed)

    destination = Path(output_dir)
    _atomic_write_csv(destination / "file_metrics.csv", file_payload, _FILE_FIELDS)
    _atomic_write_csv(
        destination / "family_metrics.csv", family_payload, _FAMILY_FIELDS
    )
    _atomic_write_csv(destination / "track_metrics.csv", track_payload, _TRACK_FIELDS)
    _atomic_write_csv(
        destination / "overall_metrics.csv", overall_payload, _OVERALL_FIELDS
    )

    main_files = [
        item for item in file_payload if item["trajectory"] == MAIN_TRAJECTORY.value
    ]
    main_families = [
        item for item in family_payload if item["trajectory"] == MAIN_TRAJECTORY.value
    ]
    main_tracks = [
        item for item in track_payload if item["trajectory"] == MAIN_TRAJECTORY.value
    ]
    _atomic_write_csv(
        destination / "main_file_metrics.csv", main_files, _FILE_FIELDS
    )
    _atomic_write_csv(
        destination / "main_family_metrics.csv", main_families, _FAMILY_FIELDS
    )
    _atomic_write_csv(
        destination / "main_track_metrics.csv", main_tracks, _TRACK_FIELDS
    )
    _atomic_write_csv(
        destination / "ablation_track_metrics.csv", track_payload, _TRACK_FIELDS
    )
    _atomic_write_csv(
        destination / "paper_reference_comparison.csv",
        comparison,
        (
            "track",
            "files",
            "ours_method",
            "ours_vus_pr",
            "paper_method",
            "paper_vus_pr",
            "paper_reference_source",
            "delta_vus_pr",
            "exceeds_paper_reported",
            "comparison_type",
        ),
    )
    _atomic_write_csv(
        destination / "runtime_summary.csv",
        runtime_payload,
        (
            "trajectory",
            "checkpoint",
            "track",
            "seed",
            "files",
            "training_runtime_seconds_sum",
            "training_runtime_seconds_mean",
            "scoring_runtime_seconds_sum",
            "scoring_runtime_seconds_mean",
            "training_peak_vram_mib_max",
            "scoring_peak_vram_mib_max",
        ),
    )
    atomic_write_json(destination / "decision.json", decision)
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed not in (2027, 2028, 2029):
        raise ValueError("full benchmark seed is not registered")
    protocol = load_protocol(args.config)
    for trajectory in REGISTERED_BENCHMARK_TRAJECTORIES:
        protocol.trajectory(trajectory)
    series = load_benchmark_manifest(args.manifest)
    if {
        track: sum(item.track == track for item in series) for track in ("U", "M")
    } != EXPECTED_TRACK_COUNTS:
        raise ValueError("full benchmark manifest track counts changed")
    decision = aggregate_full_benchmark(
        series,
        args.metrics_dir,
        args.results_root,
        args.output_dir,
        seed=args.seed,
        expected_config_sha256=protocol.source_sha256,
        expected_vendor_sha=protocol.baseline.git_sha,
    )
    print(
        f"BENCHMARK_AGGREGATE_COMPLETE outcome={decision['outcome']} "
        f"metrics={decision['metric_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
