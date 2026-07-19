from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from measure_vit4ts_v3.qualitative_outputs import (
    CASE_ROLES,
    nctp_mapping_zoom_data,
    patch_field_heatmap_data,
    score_stack_plot_data,
    select_qualitative_cases,
    structural_case_scores,
    write_qualitative_outputs,
)


SERIES = ("MSL__C-1", "best", "worst", "boundary", "spare")


def _metrics() -> pd.DataFrame:
    deltas = {
        "MSL__C-1": 0.1,
        "best": 0.5,
        "worst": -0.4,
        "boundary": 0.2,
        "spare": 0.0,
    }
    rows = []
    for series_id, delta in deltas.items():
        rows.extend(
            (
                {"series_id": series_id, "arm": "REL", "vus_pr": 0.4},
                {"series_id": series_id, "arm": "FULL", "vus_pr": 0.4 + delta},
            )
        )
    return pd.DataFrame(rows)


def _structural_fields() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fields = {}
    for index, series_id in enumerate(SERIES, start=1):
        released = np.zeros((2, 4), dtype=np.float64)
        literal = np.zeros((2, 4), dtype=np.float64)
        literal[:, [1, 3]] = float(index)
        fields[series_id] = (released, literal)
    # Make the label-free structural case unambiguous and distinct.
    fields["boundary"] = (np.zeros((2, 4)), np.full((2, 4), 20.0))
    return fields


def _cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    structural = structural_case_scores(_structural_fields(), patch_grid=(2, 2))
    cases = select_qualitative_cases(
        _metrics(),
        structural,
        candidate_arm="FULL",
        control_arm="REL",
        metric="vus_pr",
    )
    return cases, structural


def test_case_selection_is_deterministic_and_label_boundary_is_separate() -> None:
    cases, structural = _cases()
    assert tuple(cases["case_role"]) == CASE_ROLES
    assert tuple(cases["series_id"]) == ("MSL__C-1", "best", "worst", "boundary")
    assert cases["series_id"].is_unique
    assert tuple(cases["uses_evaluation_labels"]) == (False, True, True, False)
    row = structural.loc[structural["series_id"] == "boundary"].iloc[0]
    assert row["boundary_cell_count"] == 1
    assert row["terminal_cell_count"] == 1
    assert row["boundary_terminal_score"] == pytest.approx(20.0)


def test_qualitative_tidy_schemas_and_oracle_warning(tmp_path: Path) -> None:
    cases, _ = _cases()
    signals = {series_id: np.asarray([0.0, 1.0, 0.5]) for series_id in SERIES}
    labels = {series_id: np.asarray([0, 1, 0], dtype=np.uint8) for series_id in SERIES}
    scores = {}
    arms = {
        "REL": "REL",
        "IHP": "IHP",
        "REL_NCTP": "REL_NCTP",
        "FULL": "FULL",
    }
    for series_id in SERIES:
        for offset, arm in enumerate(arms.values()):
            scores[(series_id, arm)] = np.asarray([0.1, 0.4, 0.2]) + offset
    threshold = {(series_id, "FULL"): 0.5 for series_id in SERIES}
    stack = score_stack_plot_data(
        cases,
        signals,
        labels,
        scores,
        arms,
        oracle_thresholds=threshold,
    )
    assert set(stack["panel"]) == {"raw_series", *arms}
    assert set(stack.loc[stack["arm"] == "FULL", "threshold_kind"]) == {
        "ORACLE_F1_VISUALIZATION_ONLY"
    }
    assert set(stack.loc[stack["arm"] == "REL", "threshold_kind"]) == {
        "NONE_CONTINUOUS_SCORE"
    }

    fields = {}
    for series_id in SERIES:
        fields[(series_id, "released_patch_field")] = np.arange(8).reshape(2, 2, 2)
        fields[(series_id, "literal_patch_field")] = np.arange(8, 16).reshape(2, 2, 2)
    windows = {series_id: 1 for series_id in SERIES}
    heatmaps = patch_field_heatmap_data(cases, fields, window_indices=windows)
    assert len(heatmaps) == 4 * 2 * 4
    assert set(heatmaps["window_index"]) == {1}

    operator = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    operators = {(series_id, "nctp_linear"): operator for series_id in SERIES}
    ranges = {series_id: (1, 3) for series_id in SERIES}
    mapping = nctp_mapping_zoom_data(
        cases, operators, time_ranges=ranges, patch_grid=(2, 2)
    )
    assert set(mapping["local_time"]) == {1, 2}
    assert set(mapping["operator"]) == {"nctp_linear"}
    outputs = write_qualitative_outputs(
        tmp_path, cases, stack, heatmaps, mapping
    )
    assert len(outputs) == 4 and all(path.is_file() for path in outputs)


def test_missing_mandatory_case_or_mapping_fails_closed() -> None:
    structural = structural_case_scores(_structural_fields(), patch_grid=(2, 2))
    with pytest.raises(ValueError, match="MSL C-1"):
        select_qualitative_cases(
            _metrics().loc[_metrics()["series_id"] != "MSL__C-1"],
            structural,
            candidate_arm="FULL",
            control_arm="REL",
            metric="vus_pr",
        )
    cases, _ = _cases()
    with pytest.raises(ValueError, match="nctp_linear"):
        nctp_mapping_zoom_data(
            cases,
            {(series_id, "legacy"): np.eye(4) for series_id in SERIES},
            time_ranges={series_id: (0, 2) for series_id in SERIES},
            patch_grid=(2, 2),
        )
