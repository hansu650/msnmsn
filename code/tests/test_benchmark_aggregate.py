from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import paano_k0.aggregate_benchmark as module
from paano_k0.evaluate_benchmark import REGISTERED_BENCHMARK_TRAJECTORIES
from paano_k0.schemas import (
    CheckpointKind,
    MetricRow,
    ScoreManifest,
    SeriesSpec,
    Trajectory,
    make_run_id,
)


CONFIG_SHA = "2" * 64
VENDOR_SHA = "3" * 40


def _spec(tmp_path: Path, series_id: str, family: str, track: str) -> SeriesSpec:
    return SeriesSpec(
        series_id=series_id,
        family=family,
        track=track,
        csv_path=tmp_path / f"{series_id}.csv",
        csv_sha256=("1" if track == "U" else "9") * 64,
        rows=4,
        channels=1,
        train_end=2,
        feature_columns=("value",),
        label_column="Label",
    )


def _score_manifest(spec: SeriesSpec, trajectory: Trajectory) -> ScoreManifest:
    return ScoreManifest(
        schema_version="paano-k0-score-v1",
        run_id=make_run_id(
            spec.series_id, 2027, trajectory, CheckpointKind.LAST
        ),
        series_id=spec.series_id,
        family=spec.family,
        track=spec.track,
        data_sha256=spec.csv_sha256,
        config_sha256=CONFIG_SHA,
        vendor_sha=VENDOR_SHA,
        seed=2027,
        trajectory=trajectory,
        checkpoint=CheckpointKind.LAST,
        initial_state_sha256="4" * 64,
        replay_sha256="5" * 64,
        checkpoint_sha256="6" * 64,
        num_points=spec.rows,
        num_train_patches=2,
        num_full_patches=4,
        channels=1,
        patch_size=2,
        stride=1,
        top_k=1,
        requested_memory_fraction=0.5,
        effective_memory_fraction=0.5,
        memory_count=1,
        memory_sha256="7" * 64,
        score_sha256="8" * 64,
        runtime_seconds=0.25,
        peak_vram_mib=2.0,
        sliding_window=2,
        labels_read=False,
    )


def _write_metric(
    metrics_dir: Path,
    spec: SeriesSpec,
    trajectory: Trajectory,
    vus_pr: float,
) -> None:
    row = MetricRow(
        run_id=make_run_id(
            spec.series_id, 2027, trajectory, CheckpointKind.LAST
        ),
        series_id=spec.series_id,
        family=spec.family,
        track=spec.track,
        seed=2027,
        trajectory=trajectory,
        checkpoint=CheckpointKind.LAST,
        vus_pr=vus_pr,
        auprc=0.4,
        vus_roc=0.7,
        auroc=0.8,
        score_sha256="8" * 64,
        data_sha256=spec.csv_sha256,
        config_sha256=CONFIG_SHA,
        vendor_sha=VENDOR_SHA,
    )
    destination = metrics_dir / "metrics"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{row.run_id}.json").write_text(
        json.dumps(row.to_dict()), encoding="utf-8"
    )


def _write_runtime(
    results_root: Path, spec: SeriesSpec, trajectory: Trajectory
) -> None:
    trajectory_dir = (
        results_root
        / "runs"
        / spec.series_id
        / "seed_2027"
        / trajectory.value
    )
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    (trajectory_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "series_id": spec.series_id,
                "family": spec.family,
                "track": spec.track,
                "seed": 2027,
                "trajectory": trajectory.value,
                "runtime_seconds": 1.5,
                "peak_vram_mib": 10.0,
            }
        ),
        encoding="utf-8",
    )
    score_dir = trajectory_dir / "scores" / CheckpointKind.LAST.value
    score_dir.mkdir(parents=True, exist_ok=True)
    (score_dir / "score_manifest.json").write_text(
        json.dumps(_score_manifest(spec, trajectory).to_dict()), encoding="utf-8"
    )


def _fixture_series(tmp_path: Path) -> tuple[SeriesSpec, ...]:
    return (
        _spec(tmp_path, "u-a", "family-u-a", "U"),
        _spec(tmp_path, "u-b", "family-u-b", "U"),
        _spec(tmp_path, "m-a", "family-m-a", "M"),
        _spec(tmp_path, "m-b", "family-m-b", "M"),
    )


def test_complete_three_arm_aggregate_and_exact_paper_comparison(
    tmp_path: Path,
) -> None:
    series = _fixture_series(tmp_path)
    metrics_dir = tmp_path / "evaluation"
    results_root = tmp_path / "results"
    main_scores = {"u-a": 0.55, "u-b": 0.57, "m-a": 0.44, "m-b": 0.46}
    for spec in series:
        for trajectory in REGISTERED_BENCHMARK_TRAJECTORIES:
            score = (
                main_scores[spec.series_id]
                if trajectory is Trajectory.PAPERNEG_NONOVERLAP
                else 0.40
            )
            _write_metric(metrics_dir, spec, trajectory, score)
            _write_runtime(results_root, spec, trajectory)

    output_dir = tmp_path / "compact"
    decision = module.aggregate_full_benchmark(
        series,
        metrics_dir,
        results_root,
        output_dir,
        seed=2027,
        expected_config_sha256=CONFIG_SHA,
        expected_vendor_sha=VENDOR_SHA,
    )

    assert decision["outcome"] == "CONTINUE_FULL_CONFIRMATION"
    assert decision["both_tracks_exceed"] is True
    assert decision["conditional_confirmation_seeds"] == [2028, 2029]
    assert decision["paper_reported_vus_pr"] == {"U": 0.5296, "M": 0.4263}
    assert decision["tracks"]["U"]["ours_vus_pr"] == pytest.approx(0.56)
    assert decision["tracks"]["M"]["ours_vus_pr"] == pytest.approx(0.45)
    assert decision["paper_reference_source"] == module.PAPER_REFERENCE_SOURCE

    expected_outputs = {
        "file_metrics.csv",
        "family_metrics.csv",
        "track_metrics.csv",
        "overall_metrics.csv",
        "main_file_metrics.csv",
        "main_family_metrics.csv",
        "main_track_metrics.csv",
        "ablation_track_metrics.csv",
        "paper_reference_comparison.csv",
        "runtime_summary.csv",
        "decision.json",
    }
    assert expected_outputs <= {path.name for path in output_dir.iterdir()}
    with (output_dir / "file_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == len(series) * 3
    with (output_dir / "paper_reference_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        comparison = {row["track"]: row for row in csv.DictReader(handle)}
    assert comparison["U"]["paper_vus_pr"] == "0.5296"
    assert comparison["M"]["paper_vus_pr"] == "0.4263"
    assert comparison["M"]["comparison_type"] == "external_paper_reported"


def test_paper_gate_is_strict_at_exact_table_15_value() -> None:
    rows = (
        {
            "trajectory": Trajectory.PAPERNEG_NONOVERLAP.value,
            "track": "U",
            "files": 1,
            "vus_pr": 0.5296,
        },
        {
            "trajectory": Trajectory.PAPERNEG_NONOVERLAP.value,
            "track": "M",
            "files": 1,
            "vus_pr": 0.50,
        },
    )
    comparison, decision = module._paper_comparison(rows, {"U": 1, "M": 1})
    assert comparison[0]["exceeds_paper_reported"] is False
    assert decision["outcome"] == "STOP_FULL_MAIN_FAILURE"
    assert decision["conditional_confirmation_seeds"] == []


def test_track_aggregate_is_file_weighted_not_equal_family_macro(
    tmp_path: Path,
) -> None:
    specs = (
        _spec(tmp_path, "u-large-1", "large", "U"),
        _spec(tmp_path, "u-large-2", "large", "U"),
        _spec(tmp_path, "u-small", "small", "U"),
    )
    values = (1.0, 1.0, 0.0)
    rows = tuple(
        MetricRow(
            run_id=make_run_id(
                spec.series_id,
                2027,
                Trajectory.PAPERNEG_NONOVERLAP,
                CheckpointKind.LAST,
            ),
            series_id=spec.series_id,
            family=spec.family,
            track=spec.track,
            seed=2027,
            trajectory=Trajectory.PAPERNEG_NONOVERLAP,
            checkpoint=CheckpointKind.LAST,
            vus_pr=value,
            auprc=value,
            vus_roc=value,
            auroc=value,
            score_sha256="8" * 64,
            data_sha256=spec.csv_sha256,
            config_sha256=CONFIG_SHA,
            vendor_sha=VENDOR_SHA,
        )
        for spec, value in zip(specs, values, strict=True)
    )
    aggregate = module._track_rows(rows)[0]
    assert aggregate["vus_pr"] == pytest.approx(2.0 / 3.0)
    assert aggregate["vus_pr"] != pytest.approx(0.5)


def test_incomplete_three_arm_metric_coverage_fails_before_runtime_scan(
    tmp_path: Path,
) -> None:
    series = _fixture_series(tmp_path)[:2]
    for spec in series:
        for trajectory in REGISTERED_BENCHMARK_TRAJECTORIES:
            if spec is series[-1] and trajectory is Trajectory.OFFICIAL:
                continue
            _write_metric(tmp_path / "evaluation", spec, trajectory, 0.6)

    with pytest.raises(ValueError, match="coverage mismatch"):
        module.aggregate_full_benchmark(
            series,
            tmp_path / "evaluation",
            tmp_path / "missing-results",
            tmp_path / "compact",
            seed=2027,
            expected_config_sha256=CONFIG_SHA,
            expected_vendor_sha=VENDOR_SHA,
        )
