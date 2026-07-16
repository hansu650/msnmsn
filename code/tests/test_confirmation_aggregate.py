from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import paano_k0.aggregate_confirmation as module
from paano_k0.schemas import (
    CheckpointKind,
    MetricRow,
    SeriesSpec,
    Trajectory,
    make_run_id,
)


CONFIG_SHA = "2" * 64
VENDOR_SHA = "3" * 40


def _spec(
    tmp_path: Path, series_id: str, family: str, track: str, digest: str
) -> SeriesSpec:
    return SeriesSpec(
        series_id=series_id,
        family=family,
        track=track,
        csv_path=tmp_path / f"{series_id}.csv",
        csv_sha256=digest * 64,
        rows=4,
        channels=1,
        train_end=2,
        feature_columns=("value",),
        label_column="Label",
    )


def _series(tmp_path: Path) -> tuple[SeriesSpec, ...]:
    return (
        _spec(tmp_path, "u-a", "family-u-a", "U", "1"),
        _spec(tmp_path, "u-b", "family-u-b", "U", "4"),
        _spec(tmp_path, "m-a", "family-m-a", "M", "5"),
        _spec(tmp_path, "m-b", "family-m-b", "M", "6"),
    )


def _row(spec: SeriesSpec, seed: int, vus_pr: float) -> MetricRow:
    return MetricRow(
        run_id=make_run_id(
            spec.series_id,
            seed,
            Trajectory.PAPERNEG_NONOVERLAP,
            CheckpointKind.LAST,
        ),
        series_id=spec.series_id,
        family=spec.family,
        track=spec.track,
        seed=seed,
        trajectory=Trajectory.PAPERNEG_NONOVERLAP,
        checkpoint=CheckpointKind.LAST,
        vus_pr=vus_pr,
        auprc=vus_pr - 0.05,
        vus_roc=vus_pr + 0.05,
        auroc=vus_pr + 0.08,
        score_sha256=str(seed % 10) * 64,
        data_sha256=spec.csv_sha256,
        config_sha256=CONFIG_SHA,
        vendor_sha=VENDOR_SHA,
    )


def _rows_for_seed(series: tuple[SeriesSpec, ...], seed: int) -> tuple[MetricRow, ...]:
    seed_offset = (seed - 2027) * 0.10
    base = {"u-a": 0.50, "u-b": 0.70, "m-a": 0.30, "m-b": 0.50}
    return tuple(
        _row(spec, seed, base[spec.series_id] + seed_offset) for spec in series
    )


def _write_csv(path: Path, rows: tuple[MetricRow, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        item = row.to_dict()
        item["arm"] = row.arm
        payload.append(item)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(payload[0]))
        writer.writeheader()
        writer.writerows(payload)


def _write_json_directory(path: Path, rows: tuple[MetricRow, ...]) -> None:
    metric_root = path / "metrics"
    metric_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (metric_root / f"{row.run_id}.json").write_text(
            json.dumps(row.to_dict()), encoding="utf-8"
        )


def _sources(
    tmp_path: Path, series: tuple[SeriesSpec, ...]
) -> dict[int, Path]:
    seed_2027 = tmp_path / "main_file_metrics.csv"
    _write_csv(seed_2027, _rows_for_seed(series, 2027))
    seed_2028 = tmp_path / "evaluation_2028"
    seed_2029 = tmp_path / "evaluation_2029"
    _write_json_directory(seed_2028, _rows_for_seed(series, 2028))
    _write_json_directory(seed_2029, _rows_for_seed(series, 2029))
    return {2027: seed_2027, 2028: seed_2028, 2029: seed_2029}


def test_exact_three_seed_track_aggregate_preserves_every_seed(
    tmp_path: Path,
) -> None:
    series = _series(tmp_path)
    output_dir = tmp_path / "compact"
    summary = module.aggregate_confirmation(
        series,
        _sources(tmp_path, series),
        output_dir,
        expected_config_sha256=CONFIG_SHA,
        expected_vendor_sha=VENDOR_SHA,
    )

    assert summary["seeds"] == [2027, 2028, 2029]
    assert summary["metric_count"] == len(series) * 3
    assert summary["seed_track_row_count"] == 6
    assert "outcome" not in summary
    assert summary["selection_applied"] is False
    assert summary["retuning_applied"] is False
    assert summary["result_dropping_applied"] is False
    assert summary["paper_reported_vus_pr"] == {"U": 0.5296, "M": 0.4263}

    seed_rows = {
        (row["track"], int(row["seed"])): row
        for row in summary["seed_track_metrics"]
    }
    assert set(seed_rows) == {
        (track, seed)
        for track in ("U", "M")
        for seed in module.CONFIRMATION_SEEDS
    }
    assert seed_rows[("U", 2027)]["vus_pr"] == pytest.approx(0.60)
    assert seed_rows[("U", 2028)]["vus_pr"] == pytest.approx(0.70)
    assert seed_rows[("U", 2029)]["vus_pr"] == pytest.approx(0.80)

    track_rows = {row["track"]: row for row in summary["track_summary"]}
    assert track_rows["U"]["vus_pr_mean"] == pytest.approx(0.70)
    assert track_rows["U"]["vus_pr_std"] == pytest.approx(
        np.std([0.60, 0.70, 0.80], ddof=0)
    )
    assert track_rows["U"]["std_ddof"] == 0
    assert track_rows["M"]["paper_vus_pr"] == 0.4263
    assert (
        track_rows["M"]["comparison_type"]
        == "descriptive_external_paper_reported"
    )

    assert {
        "confirmation_seed_track_metrics.csv",
        "confirmation_track_summary.csv",
        "confirmation_summary.json",
    } == {path.name for path in output_dir.iterdir()}
    with (output_dir / "confirmation_seed_track_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 6
    assert {int(row["seed"]) for row in written} == {2027, 2028, 2029}


def test_missing_seed_metric_fails_before_any_confirmation_output(
    tmp_path: Path,
) -> None:
    series = _series(tmp_path)
    sources = _sources(tmp_path, series)
    missing = next((sources[2029] / "metrics").glob("*.json"))
    missing.unlink()
    output_dir = tmp_path / "compact"

    with pytest.raises(ValueError, match="seed 2029 metric coverage mismatch"):
        module.aggregate_confirmation(
            series,
            sources,
            output_dir,
            expected_config_sha256=CONFIG_SHA,
            expected_vendor_sha=VENDOR_SHA,
        )
    assert not output_dir.exists()


def test_confirmation_rejects_seed_provenance_drift(
    tmp_path: Path,
) -> None:
    series = _series(tmp_path)
    sources = _sources(tmp_path, series)
    metric_path = next((sources[2028] / "metrics").glob("*.json"))
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    payload["config_sha256"] = "9" * 64
    metric_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config SHA changed"):
        module.aggregate_confirmation(
            series,
            sources,
            tmp_path / "compact",
            expected_config_sha256=CONFIG_SHA,
            expected_vendor_sha=VENDOR_SHA,
        )


def test_confirmation_requires_exact_registered_seed_sources(tmp_path: Path) -> None:
    series = _series(tmp_path)
    sources = _sources(tmp_path, series)
    sources.pop(2029)
    with pytest.raises(ValueError, match="exactly seeds 2027, 2028, and 2029"):
        module.aggregate_confirmation(series, sources, tmp_path / "compact")
