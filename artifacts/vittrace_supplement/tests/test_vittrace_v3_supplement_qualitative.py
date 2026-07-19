from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts_v3.supplement_qualitative import (
    ARM_IDS,
    EXPECTED_COMMON_VALID_SERIES,
    FACTORIAL_ARMS,
    CommittedScore,
    build_case_tables,
    candidate_ranking_frame,
    common_valid_series,
    resolve_committed_score,
    select_representative_cases,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _mask(deltas: dict[str, float], *, invalid: tuple[str, ...] = ()) -> pd.DataFrame:
    rows = []
    for series_id in (*deltas, *invalid):
        valid = series_id not in invalid
        rows.append(
            {
                "series_id": series_id,
                "family": "NAB",
                "subgroup": "NAB-AWS",
                "n_points": 600,
                "n_positive": 5 if valid else 0,
                "valid_f1_max": valid,
                "valid_auprc": valid,
                "valid_vus_pr": valid,
            }
        )
    return pd.DataFrame(rows)


def _metrics(deltas: dict[str, float], *, extras: tuple[str, ...] = ()) -> pd.DataFrame:
    rows = []
    for series_id in (*deltas, *extras):
        delta = deltas.get(series_id, 0.0)
        for arm in ARM_IDS:
            value = 0.4 + (delta if arm == "IHP1_NCTP1" else 0.0)
            rows.append(
                {
                    "series_id": series_id,
                    "arm": arm,
                    "vus_pr": value,
                    "n_points": 600,
                    "n_positive": 5 if series_id in deltas else 0,
                }
            )
    return pd.DataFrame(rows)


def test_case_selection_uses_median_delta_and_series_id_tie_break() -> None:
    deltas = {"A": 0.1, "B": 0.3, "C": 0.5, "D": -0.4, "E": 0.0}
    metrics = _metrics(deltas, extras=("INVALID",))
    valid_mask = _mask(deltas, invalid=("INVALID",))
    ranking = candidate_ranking_frame(metrics, valid_mask)
    result = select_representative_cases(metrics, valid_mask)
    assert list(result["series_id"]) == ["B", "D"]
    by_id = ranking.set_index("series_id")
    assert by_id.loc["B", "positive_median_target"] == pytest.approx(0.3)
    assert by_id.loc["B", "positive_distance"] == pytest.approx(0.0)
    assert by_id.loc["B", "positive_selection_rank"] == pytest.approx(1.0)
    assert by_id.loc["D", "nonpositive_median_target"] == pytest.approx(-0.2)
    assert by_id.loc["D", "nonpositive_distance"] == pytest.approx(0.2)
    assert by_id.loc["D", "nonpositive_selection_rank"] == pytest.approx(1.0)
    assert set(ranking["series_id"]) == set(deltas)
    assert result.iloc[0]["selection_target"] == pytest.approx(0.3)
    assert result.iloc[1]["selection_target"] == pytest.approx(-0.2)
    assert result.iloc[1]["selection_rule"] == (
        "closest_to_median_nonpositive_full_minus_rel_vus_pr"
    )
    assert result["oracle_visualization_only"].all()


def test_common_valid_cohort_gate_requires_frozen_488() -> None:
    with pytest.raises(ValueError, match="expected 488, found 1"):
        common_valid_series(
            _mask({"ONLY": 0.1}), expected_count=EXPECTED_COMMON_VALID_SERIES
        )


def test_case_selection_uses_global_minimum_fallback() -> None:
    deltas = {"A": 0.1, "B": 0.2, "C": 0.9}
    result = select_representative_cases(_metrics(deltas), _mask(deltas))
    assert list(result["series_id"]) == ["B", "A"]
    assert result.iloc[1]["selection_rule"] == "global_minimum_fallback_no_nonpositive"


def test_resolve_committed_direct_and_alias_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "scores"
    series = root / "S1"
    values = np.linspace(0.0, 1.0, 8, dtype=np.float64)

    direct = series / "DIRECT"
    direct.mkdir(parents=True)
    np.save(direct / "score.npy", values, allow_pickle=False)
    score_hash = _hash(direct / "score.npy")
    (direct / "score_manifest.json").write_text(
        json.dumps(
            {"series_id": "S1", "arm": "DIRECT", "score_sha256": score_hash}
        ),
        encoding="utf-8",
    )
    (direct / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "series_id": "S1",
                "arm": "DIRECT",
                "score_sha256": score_hash,
            }
        ),
        encoding="utf-8",
    )
    direct_result = resolve_committed_score(
        root,
        "S1",
        "DIRECT",
        expected_score_sha256=score_hash,
        expected_manifest_sha256=_hash(direct / "score_manifest.json"),
        expected_length=8,
    )
    assert not direct_result.is_alias
    np.testing.assert_array_equal(direct_result.values, values)

    alias = series / "ALIAS"
    alias.mkdir()
    (alias / "alias_manifest.json").write_text(
        json.dumps(
            {
                "series_id": "S1",
                "arm": "ALIAS",
                "canonical_arm": "DIRECT",
                "canonical_score_path": "../DIRECT/score.npy",
                "canonical_score_sha256": score_hash,
            }
        ),
        encoding="utf-8",
    )
    (alias / "_SUCCESS.json").write_text(
        json.dumps(
            {"series_id": "S1", "arm": "ALIAS", "canonical_arm": "DIRECT"}
        ),
        encoding="utf-8",
    )
    alias_result = resolve_committed_score(
        root,
        "S1",
        "ALIAS",
        expected_score_sha256=score_hash,
        expected_manifest_sha256=_hash(alias / "alias_manifest.json"),
        expected_length=8,
    )
    assert alias_result.is_alias and alias_result.canonical_arm == "DIRECT"
    with pytest.raises(ValueError, match="committed score hash mismatch"):
        resolve_committed_score(
            root,
            "S1",
            "ALIAS",
            expected_score_sha256="0" * 64,
            expected_manifest_sha256=_hash(alias / "alias_manifest.json"),
            expected_length=8,
        )


def test_build_case_tables_exports_four_scores_and_oracle_intervals(tmp_path: Path) -> None:
    records = {}
    mask_rows = []
    metric_rows = []
    committed = {}
    cases = []
    for case_order, series_id in enumerate(("S1", "S2")):
        records[series_id] = FullSeriesRecord(
            series_id,
            "synthetic",
            "NAB",
            "NAB-AWS",
            series_id,
            f"synthetic/{series_id}.csv",
            10,
            1,
            "A" * 64,
            False,
        )
        mask_rows.append(
            {
                "series_id": series_id,
                "family": "NAB",
                "subgroup": "NAB-AWS",
                "n_points": 10,
                "n_positive": 2,
                "valid_f1_max": True,
                "valid_auprc": True,
                "valid_vus_pr": True,
            }
        )
        cases.append(
            {
                "case_order": case_order,
                "case_role": "representative_positive" if case_order == 0 else "representative_failure",
                "series_id": series_id,
            }
        )
        for panel_index, (_, arm) in enumerate(FACTORIAL_ARMS):
            values = np.linspace(0.0, 1.0, 10, dtype=np.float64) + panel_index * 0.01
            digest = f"{panel_index + case_order + 1:064X}"[-64:]
            committed[(series_id, arm)] = CommittedScore(
                series_id,
                arm,
                arm,
                tmp_path / f"{series_id}_{arm}.npy",
                digest,
                tmp_path / f"{series_id}_{arm}.json",
                "F" * 64,
                False,
                values,
            )
            metric_rows.append(
                {
                    "series_id": series_id,
                    "arm": arm,
                    "f1_winning_alpha": 0.1,
                    "f1_threshold_evaluator_only": 0.5,
                }
            )

    def loader(record: FullSeriesRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.zeros(10, dtype=np.uint8)
        labels[4:6] = 1
        return np.arange(10, dtype=np.float64), np.linspace(-1.0, 1.0, 10), labels

    tidy, intervals, truth_intervals = build_case_tables(
        pd.DataFrame(cases),
        records,
        pd.DataFrame(mask_rows),
        pd.DataFrame(metric_rows),
        committed,
        loader,
    )
    assert len(tidy) == 20
    for panel, _ in FACTORIAL_ARMS:
        assert {
            f"{panel}_score",
            f"{panel}_score_sha256",
            f"{panel}_f1_winning_alpha",
            f"{panel}_f1_threshold_evaluator_only",
            f"{panel}_prediction_interval_id",
        } <= set(tidy)
    assert tidy["oracle_visualization_only"].all()
    assert not intervals.empty
    assert set(intervals["arm"]) == set(ARM_IDS)
    assert intervals["oracle_visualization_only"].all()
    assert len(truth_intervals) == 2
    assert set(truth_intervals["series_id"]) == {"S1", "S2"}
    assert (truth_intervals["start_index"] == 4).all()
    assert (truth_intervals["end_index"] == 5).all()
    assert truth_intervals["uses_evaluator_labels"].all()
    assert not truth_intervals["oracle_visualization_only"].any()
