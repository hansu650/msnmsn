"""Deterministic qualitative-case selection and plot-data schemas for v3.

Best/worst cases use one explicitly supplied, preregistered metric contrast.
The boundary case is selected without labels from released-versus-literal
field disagreement at the thirteen row ends plus the terminal cell.  Plot
data remain continuous; any evaluator-oracle threshold is visibly marked as
visualization-only metadata.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CASE_ROLES = (
    "fixed_msl_c1",
    "best_improvement",
    "worst_case",
    "boundary_terminal_defect",
)
CASE_COLUMNS = (
    "case_order",
    "case_role",
    "series_id",
    "selection_basis",
    "selection_metric",
    "candidate_arm",
    "control_arm",
    "candidate_value",
    "control_value",
    "delta",
    "structural_score",
    "uses_evaluation_labels",
)
STRUCTURAL_CASE_COLUMNS = (
    "series_id",
    "patch_grid",
    "boundary_cell_count",
    "terminal_cell_count",
    "boundary_mean_abs_delta",
    "terminal_mean_abs_delta",
    "boundary_terminal_score",
    "selection_rule",
    "uses_labels",
)
SCORE_STACK_COLUMNS = (
    "case_order",
    "case_role",
    "series_id",
    "panel_order",
    "panel",
    "arm",
    "time_index",
    "timestamp",
    "value",
    "ground_truth",
    "threshold",
    "threshold_kind",
)
HEATMAP_COLUMNS = (
    "case_order",
    "case_role",
    "series_id",
    "window_index",
    "field",
    "row",
    "column",
    "value",
)
MAPPING_COLUMNS = (
    "case_order",
    "case_role",
    "series_id",
    "operator",
    "local_time",
    "patch_index",
    "patch_row",
    "patch_column",
    "weight",
)


def _grid(patch_grid: Sequence[int]) -> tuple[int, int]:
    if len(patch_grid) != 2:
        raise ValueError("patch_grid must contain height and width")
    height, width = map(int, patch_grid)
    if height <= 0 or width <= 0:
        raise ValueError("patch_grid must be positive")
    return height, width


def structural_case_scores(
    fields: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    patch_grid: Sequence[int] = (14, 14),
) -> pd.DataFrame:
    """Compute label-free boundary/terminal disagreement for each series."""

    height, width = _grid(patch_grid)
    cells = height * width
    boundary = np.arange(width - 1, cells - width, width, dtype=np.int64)
    terminal = np.asarray([cells - 1], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for series_id in sorted(fields):
        released, literal = fields[series_id]
        left = np.asarray(released, dtype=np.float64)
        right = np.asarray(literal, dtype=np.float64)
        if left.shape != right.shape or left.ndim not in (1, 2) or left.shape[-1] != cells:
            raise ValueError("structural fields must be aligned [K] or [N,K]")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("structural fields must be finite")
        difference = np.abs(left - right).reshape(-1, cells)
        boundary_value = float(np.mean(difference[:, boundary]))
        terminal_value = float(np.mean(difference[:, terminal]))
        selected = np.concatenate((boundary, terminal))
        rows.append(
            {
                "series_id": str(series_id),
                "patch_grid": f"{height}x{width}",
                "boundary_cell_count": int(boundary.size),
                "terminal_cell_count": 1,
                "boundary_mean_abs_delta": boundary_value,
                "terminal_mean_abs_delta": terminal_value,
                "boundary_terminal_score": float(np.mean(difference[:, selected])),
                "selection_rule": "mean_abs_released_literal_on_row_ends_plus_terminal",
                "uses_labels": False,
            }
        )
    frame = pd.DataFrame(rows, columns=STRUCTURAL_CASE_COLUMNS)
    if frame.empty or frame["series_id"].duplicated().any():
        raise ValueError("structural-case inputs must be unique and nonempty")
    return frame.sort_values("series_id").reset_index(drop=True)


def _metric_contrast(
    metrics: pd.DataFrame,
    *,
    candidate_arm: str,
    control_arm: str,
    metric: str,
) -> pd.DataFrame:
    required = {"series_id", "arm", metric}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"qualitative metrics are missing columns: {sorted(missing)}")
    subset = metrics.loc[
        metrics["arm"].astype(str).isin((candidate_arm, control_arm)),
        ["series_id", "arm", metric],
    ].copy()
    if subset.duplicated(["series_id", "arm"]).any():
        raise ValueError("qualitative metric rows are not unique")
    pivot = subset.pivot(index="series_id", columns="arm", values=metric)
    if candidate_arm not in pivot or control_arm not in pivot:
        raise ValueError("candidate/control arms are missing from qualitative metrics")
    frame = pivot.loc[:, [candidate_arm, control_arm]].rename(
        columns={candidate_arm: "candidate_value", control_arm: "control_value"}
    )
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("qualitative selection metric must be defined for every candidate")
    frame["delta"] = frame["candidate_value"] - frame["control_value"]
    return frame.reset_index()


def _pick(
    frame: pd.DataFrame,
    value: str,
    *,
    largest: bool,
    excluded: set[str],
) -> pd.Series:
    candidates = frame.loc[~frame["series_id"].astype(str).isin(excluded)].copy()
    if candidates.empty:
        raise ValueError("qualitative roles require distinct available series")
    candidates["series_id"] = candidates["series_id"].astype(str)
    candidates = candidates.sort_values(
        [value, "series_id"], ascending=[not largest, True], kind="mergesort"
    )
    return candidates.iloc[0]


def select_qualitative_cases(
    metrics: pd.DataFrame,
    structural_scores: pd.DataFrame,
    *,
    candidate_arm: str,
    control_arm: str,
    metric: str,
    fixed_series_id: str = "MSL__C-1",
) -> pd.DataFrame:
    """Select the four mandatory cases deterministically and transparently."""

    contrast = _metric_contrast(
        metrics,
        candidate_arm=str(candidate_arm),
        control_arm=str(control_arm),
        metric=str(metric),
    )
    required_structural = {"series_id", "boundary_terminal_score", "selection_rule", "uses_labels"}
    missing = required_structural - set(structural_scores.columns)
    if missing or structural_scores["series_id"].duplicated().any():
        raise ValueError(f"structural case table is invalid: missing={sorted(missing)}")
    structural = structural_scores.copy()
    structural["boundary_terminal_score"] = pd.to_numeric(
        structural["boundary_terminal_score"], errors="raise"
    )
    if not np.isfinite(structural["boundary_terminal_score"]).all():
        raise ValueError("structural case scores must be finite")
    merged = contrast.merge(
        structural.loc[:, ["series_id", "boundary_terminal_score", "selection_rule", "uses_labels"]],
        on="series_id",
        how="inner",
        validate="one_to_one",
    )
    if fixed_series_id not in set(merged["series_id"].astype(str)):
        raise ValueError("the mandatory MSL C-1 case is absent from complete inputs")

    fixed = merged.loc[merged["series_id"].astype(str) == fixed_series_id].iloc[0]
    excluded = {str(fixed_series_id)}
    best = _pick(merged, "delta", largest=True, excluded=excluded)
    excluded.add(str(best["series_id"]))
    worst = _pick(merged, "delta", largest=False, excluded=excluded)
    excluded.add(str(worst["series_id"]))
    boundary = _pick(
        merged, "boundary_terminal_score", largest=True, excluded=excluded
    )
    selections = (
        ("fixed_msl_c1", fixed, "fixed_predeclared_series", False),
        ("best_improvement", best, f"max_{candidate_arm}_minus_{control_arm}_{metric}", True),
        ("worst_case", worst, f"min_{candidate_arm}_minus_{control_arm}_{metric}", True),
        (
            "boundary_terminal_defect",
            boundary,
            str(boundary["selection_rule"]),
            False,
        ),
    )
    rows = []
    for order, (role, row, basis, uses_labels) in enumerate(selections):
        rows.append(
            {
                "case_order": order,
                "case_role": role,
                "series_id": str(row["series_id"]),
                "selection_basis": basis,
                "selection_metric": metric if uses_labels else "",
                "candidate_arm": candidate_arm,
                "control_arm": control_arm,
                "candidate_value": float(row["candidate_value"]),
                "control_value": float(row["control_value"]),
                "delta": float(row["delta"]),
                "structural_score": float(row["boundary_terminal_score"]),
                "uses_evaluation_labels": uses_labels,
            }
        )
    frame = pd.DataFrame(rows, columns=CASE_COLUMNS)
    if tuple(frame["case_role"]) != CASE_ROLES or frame["series_id"].duplicated().any():
        raise RuntimeError("qualitative case selection is incomplete or non-unique")
    return frame


def score_stack_plot_data(
    cases: pd.DataFrame,
    signals: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    scores: Mapping[tuple[str, str], np.ndarray],
    arms: Mapping[str, str],
    *,
    timestamps: Mapping[str, np.ndarray] | None = None,
    oracle_thresholds: Mapping[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Build aligned raw/ground-truth and four-score-stack tidy rows."""

    if tuple(cases["case_role"].astype(str)) != CASE_ROLES:
        raise ValueError("qualitative cases must use the frozen four-role order")
    expected_panels = ("REL", "IHP", "REL_NCTP", "FULL")
    if tuple(arms) != expected_panels:
        raise ValueError("score stacks require REL/IHP/REL_NCTP/FULL in order")
    rows: list[dict[str, Any]] = []
    for case in cases.itertuples(index=False):
        series_id = str(case.series_id)
        signal = np.asarray(signals[series_id], dtype=np.float64)
        truth = np.asarray(labels[series_id])
        time = (
            np.arange(signal.size, dtype=np.float64)
            if timestamps is None
            else np.asarray(timestamps[series_id])
        )
        if signal.ndim != 1 or truth.shape != signal.shape or time.shape != signal.shape:
            raise ValueError("qualitative signal/label/timestamp vectors must align")
        if not np.isfinite(signal).all() or not np.isfinite(time.astype(np.float64)).all():
            raise ValueError("qualitative signal and timestamps must be finite")
        if not np.logical_or(truth == 0, truth == 1).all():
            raise ValueError("qualitative ground truth must be binary")
        panel_values = [("raw_series", "", signal)]
        panel_values.extend(
            (panel, arm, np.asarray(scores[(series_id, arm)], dtype=np.float64))
            for panel, arm in arms.items()
        )
        for panel_order, (panel, arm, values) in enumerate(panel_values):
            if values.shape != signal.shape or not np.isfinite(values).all():
                raise ValueError("qualitative score vectors must align and be finite")
            threshold = np.nan
            kind = "NONE_CONTINUOUS_SCORE"
            if arm and oracle_thresholds is not None and (series_id, arm) in oracle_thresholds:
                threshold = float(oracle_thresholds[(series_id, arm)])
                if not np.isfinite(threshold):
                    raise ValueError("visualization threshold must be finite")
                kind = "ORACLE_F1_VISUALIZATION_ONLY"
            for index in range(signal.size):
                rows.append(
                    {
                        "case_order": int(case.case_order),
                        "case_role": str(case.case_role),
                        "series_id": series_id,
                        "panel_order": panel_order,
                        "panel": panel,
                        "arm": arm,
                        "time_index": index,
                        "timestamp": time[index],
                        "value": float(values[index]),
                        "ground_truth": int(truth[index]),
                        "threshold": threshold,
                        "threshold_kind": kind,
                    }
                )
    return pd.DataFrame(rows, columns=SCORE_STACK_COLUMNS)


def patch_field_heatmap_data(
    cases: pd.DataFrame,
    fields: Mapping[tuple[str, str], np.ndarray],
    *,
    window_indices: Mapping[str, int],
) -> pd.DataFrame:
    """Create released/literal patch-field heatmap rows for fixed windows."""

    required_fields = ("released_patch_field", "literal_patch_field")
    rows: list[dict[str, Any]] = []
    for case in cases.itertuples(index=False):
        series_id = str(case.series_id)
        window = int(window_indices[series_id])
        for field_name in required_fields:
            values = np.asarray(fields[(series_id, field_name)], dtype=np.float64)
            selected = values if values.ndim == 2 else values[window] if values.ndim == 3 else None
            if selected is None or selected.ndim != 2 or not np.isfinite(selected).all():
                raise ValueError("patch fields must be finite [H,W] or [N,H,W]")
            for row, column in np.ndindex(selected.shape):
                rows.append(
                    {
                        "case_order": int(case.case_order),
                        "case_role": str(case.case_role),
                        "series_id": series_id,
                        "window_index": window,
                        "field": field_name,
                        "row": row,
                        "column": column,
                        "value": float(selected[row, column]),
                    }
                )
    return pd.DataFrame(rows, columns=HEATMAP_COLUMNS)


def nctp_mapping_zoom_data(
    cases: pd.DataFrame,
    operators: Mapping[tuple[str, str], np.ndarray],
    *,
    time_ranges: Mapping[str, tuple[int, int]],
    patch_grid: Sequence[int] = (14, 14),
) -> pd.DataFrame:
    """Emit sparse non-zero NCTP/temporal-operator weights for local zooms."""

    height, width = _grid(patch_grid)
    cells = height * width
    rows: list[dict[str, Any]] = []
    names = sorted({name for _, name in operators})
    if "nctp_linear" not in names:
        raise ValueError("mapping zoom requires the nctp_linear operator")
    for case in cases.itertuples(index=False):
        series_id = str(case.series_id)
        start, stop = map(int, time_ranges[series_id])
        if start < 0 or stop <= start:
            raise ValueError("mapping zoom time ranges must be nonempty")
        for name in names:
            operator = np.asarray(operators[(series_id, name)], dtype=np.float64)
            if operator.ndim != 2 or operator.shape[1] != cells or stop > operator.shape[0]:
                raise ValueError("temporal operator must have shape [W,K]")
            if not np.isfinite(operator).all() or (operator < 0.0).any():
                raise ValueError("temporal operator must be finite and non-negative")
            for local_time in range(start, stop):
                support = np.flatnonzero(operator[local_time] > 0.0)
                for patch_index in support:
                    rows.append(
                        {
                            "case_order": int(case.case_order),
                            "case_role": str(case.case_role),
                            "series_id": series_id,
                            "operator": name,
                            "local_time": local_time,
                            "patch_index": int(patch_index),
                            "patch_row": int(patch_index // width),
                            "patch_column": int(patch_index % width),
                            "weight": float(operator[local_time, patch_index]),
                        }
                    )
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_qualitative_outputs(
    output_root: Path,
    cases: pd.DataFrame,
    score_stack: pd.DataFrame,
    heatmaps: pd.DataFrame,
    mapping_zoom: pd.DataFrame,
) -> tuple[Path, ...]:
    """Write the mandatory qualitative manifest and three tidy data tables."""

    expected = (
        ("qualitative_cases.csv", cases, CASE_COLUMNS),
        ("score_stacks.csv", score_stack, SCORE_STACK_COLUMNS),
        ("patch_field_heatmaps.csv", heatmaps, HEATMAP_COLUMNS),
        ("nctp_mapping_zoom.csv", mapping_zoom, MAPPING_COLUMNS),
    )
    outputs = []
    for name, frame, columns in expected:
        if tuple(frame.columns) != columns or frame.empty:
            raise ValueError(f"{name} is empty or has a stale schema")
        path = Path(output_root) / name
        _atomic_csv(path, frame)
        outputs.append(path)
    return tuple(outputs)


__all__ = [
    "CASE_COLUMNS",
    "CASE_ROLES",
    "HEATMAP_COLUMNS",
    "MAPPING_COLUMNS",
    "SCORE_STACK_COLUMNS",
    "STRUCTURAL_CASE_COLUMNS",
    "nctp_mapping_zoom_data",
    "patch_field_heatmap_data",
    "score_stack_plot_data",
    "select_qualitative_cases",
    "structural_case_scores",
    "write_qualitative_outputs",
]
