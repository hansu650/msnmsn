"""Minimal deterministic rough-figure generation for mandatory v3 evidence.

The generator is deliberately presentation-neutral: it serializes every input
row to a canonical tidy CSV and renders vector SVG/PDF diagnostics.  Missing or
invalid inputs become explicit ``BLOCKED``/``NA`` status rows; no values are
imputed and no placeholder curves are fabricated.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "vittrace-v3-rough",
    }
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RoughFigureRecipe:
    name: str
    kind: str
    required_columns: tuple[str, ...]
    x: str
    y: str
    series: str | None = None
    row: str | None = None
    column: str | None = None
    log_x: bool = False


MANDATORY_ROUGH_FIGURES: tuple[RoughFigureRecipe, ...] = (
    RoughFigureRecipe(
        "backbone_accuracy_time",
        "scatter",
        ("backbone", "group", "metric", "value", "elapsed_seconds_per_series"),
        "elapsed_seconds_per_series",
        "value",
        "backbone",
        log_x=True,
    ),
    RoughFigureRecipe(
        "window_sensitivity",
        "line",
        ("window", "arm", "group", "metric", "value"),
        "window",
        "value",
        "arm",
    ),
    RoughFigureRecipe(
        "stride_sensitivity",
        "line",
        ("stride", "arm", "group", "metric", "value"),
        "stride",
        "value",
        "arm",
    ),
    RoughFigureRecipe(
        "ihp_nctp_interaction",
        "interaction",
        ("ihp", "nctp", "group", "metric", "value"),
        "nctp",
        "value",
        row="ihp",
        column="nctp",
    ),
    RoughFigureRecipe(
        "matching_scope",
        "bar",
        ("scope", "arm", "group", "metric", "value"),
        "scope",
        "value",
        "arm",
    ),
    RoughFigureRecipe(
        "scale_subset_heatmap",
        "heatmap",
        ("scale_subset", "group", "metric", "value"),
        "scale_subset",
        "value",
        row="group",
        column="scale_subset",
    ),
    RoughFigureRecipe(
        "reducer_sensitivity",
        "line",
        ("reducer_family", "reducer_setting", "group", "metric", "value"),
        "reducer_setting",
        "value",
        "reducer_family",
    ),
    RoughFigureRecipe(
        "line_vs_spectrogram",
        "bar",
        ("representation", "arm", "group", "metric", "value"),
        "representation",
        "value",
        "arm",
    ),
    RoughFigureRecipe(
        "runtime_memory",
        "scatter",
        ("config_id", "measurement_mode", "median_s", "peak_rss_mb"),
        "median_s",
        "peak_rss_mb",
        "measurement_mode",
        log_x=True,
    ),
    RoughFigureRecipe(
        "qualitative_score_stacks",
        "qualitative",
        ("case_role", "panel", "time_index", "value", "ground_truth"),
        "time_index",
        "value",
        "panel",
    ),
    RoughFigureRecipe(
        "structural_mapping_coverage",
        "mapping",
        ("case_role", "operator", "local_time", "patch_index", "weight"),
        "patch_index",
        "weight",
        row="local_time",
        column="patch_index",
    ),
)
RECIPE_BY_NAME = {recipe.name: recipe for recipe in MANDATORY_ROUGH_FIGURES}
STATUS_COLUMNS = (
    "figure",
    "status",
    "reason",
    "input_rows",
    "plotted_rows",
    "tidy_csv",
    "svg",
    "pdf",
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(frame.columns)
    result = frame.loc[:, columns].copy()
    if columns:
        sortable = [column for column in columns if not result[column].map(type).eq(list).any()]
        try:
            result = result.sort_values(sortable, kind="mergesort", na_position="last")
        except TypeError:
            pass
    return result.reset_index(drop=True)


def _valid_rows(frame: pd.DataFrame, recipe: RoughFigureRecipe) -> pd.DataFrame:
    missing = [column for column in recipe.required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    valid = frame.copy()
    if "status" in valid.columns:
        valid = valid[valid["status"].astype(str).str.upper().isin({"OK", "COMPLETE", "SUCCESS"})]
    for column in (recipe.x, recipe.y):
        if column in valid.columns and column not in {"scope", "representation", "scale_subset"}:
            if recipe.kind not in {"interaction", "mapping"} or column not in {
                recipe.row,
                recipe.column,
            }:
                converted = pd.to_numeric(valid[column], errors="coerce")
                if converted.notna().any():
                    valid[column] = converted
    valid = valid[valid[recipe.y].notna()]
    if valid.empty:
        raise ValueError("no valid finite rows")
    numeric_y = pd.to_numeric(valid[recipe.y], errors="coerce")
    if not np.isfinite(numeric_y).any():
        raise ValueError("no finite y values")
    valid[recipe.y] = numeric_y
    return valid[np.isfinite(valid[recipe.y])].reset_index(drop=True)


def _facet_keys(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    columns = [column for column in ("group", "metric") if column in frame.columns]
    if not columns:
        return [tuple()]
    return sorted(
        {tuple(values) for values in frame[columns].itertuples(index=False, name=None)},
        key=lambda values: tuple(map(str, values)),
    )


def _facet_frame(frame: pd.DataFrame, key: tuple[Any, ...]) -> tuple[pd.DataFrame, str]:
    columns = [column for column in ("group", "metric") if column in frame.columns]
    selected = frame
    labels: list[str] = []
    for column, value in zip(columns, key):
        selected = selected[selected[column] == value]
        labels.append(f"{column}={value}")
    return selected, ", ".join(labels) or "all"


def _axes(count: int) -> tuple[plt.Figure, np.ndarray]:
    columns = min(3, max(1, count))
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 2.8 * rows), squeeze=False)
    flat = axes.ravel()
    for axis in flat[count:]:
        axis.set_visible(False)
    return figure, flat


def _series_values(frame: pd.DataFrame, series: str | None) -> list[tuple[str, pd.DataFrame]]:
    if series is None or series not in frame.columns:
        return [("all", frame)]
    return [
        (str(value), frame[frame[series] == value])
        for value in sorted(frame[series].dropna().unique(), key=str)
    ]


def _plot_standard(recipe: RoughFigureRecipe, frame: pd.DataFrame) -> plt.Figure:
    facets = _facet_keys(frame)
    figure, axes = _axes(len(facets))
    for axis, key in zip(axes, facets):
        selected, title = _facet_frame(frame, key)
        if recipe.kind == "scatter":
            for label, series_frame in _series_values(selected, recipe.series):
                axis.scatter(series_frame[recipe.x], series_frame[recipe.y], s=24, label=label)
        elif recipe.kind == "line":
            for label, series_frame in _series_values(selected, recipe.series):
                ordered = series_frame.sort_values(recipe.x, kind="mergesort")
                axis.plot(ordered[recipe.x], ordered[recipe.y], marker="o", linewidth=1, label=label)
        elif recipe.kind == "bar":
            categories = sorted(selected[recipe.x].astype(str).unique())
            series_values = _series_values(selected, recipe.series)
            width = 0.8 / max(1, len(series_values))
            positions = np.arange(len(categories), dtype=np.float64)
            for index, (label, series_frame) in enumerate(series_values):
                means = series_frame.groupby(series_frame[recipe.x].astype(str))[recipe.y].mean()
                values = [means.get(category, np.nan) for category in categories]
                axis.bar(positions + (index - (len(series_values) - 1) / 2) * width, values, width, label=label)
            axis.set_xticks(positions, categories, rotation=25, ha="right")
        else:
            raise ValueError(f"unsupported standard plot kind: {recipe.kind}")
        if recipe.log_x:
            numeric_x = pd.to_numeric(selected[recipe.x], errors="coerce")
            if (numeric_x <= 0).any():
                raise ValueError("log-scale x values must be positive")
            axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel(recipe.x)
        axis.set_ylabel(recipe.y)
        if recipe.series and selected[recipe.series].nunique(dropna=True) > 1:
            axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    return figure


def _plot_heatmap(recipe: RoughFigureRecipe, frame: pd.DataFrame) -> plt.Figure:
    facets = _facet_keys(frame)
    figure, axes = _axes(len(facets))
    for axis, key in zip(axes, facets):
        selected, title = _facet_frame(frame, key)
        assert recipe.row and recipe.column
        pivot = selected.pivot_table(
            index=recipe.row,
            columns=recipe.column,
            values=recipe.y,
            aggfunc="mean",
            sort=True,
        )
        values = pivot.to_numpy(dtype=np.float64)
        mesh = axis.pcolormesh(
            np.arange(values.shape[1] + 1),
            np.arange(values.shape[0] + 1),
            np.ma.masked_invalid(values),
            shading="flat",
        )
        axis.set_xticks(np.arange(values.shape[1]) + 0.5, map(str, pivot.columns), rotation=30, ha="right")
        axis.set_yticks(np.arange(values.shape[0]) + 0.5, map(str, pivot.index))
        axis.set_xlabel(recipe.column)
        axis.set_ylabel(recipe.row)
        axis.set_title(title)
        figure.colorbar(mesh, ax=axis, shrink=0.8, label=recipe.y)
    return figure


def _plot_qualitative(recipe: RoughFigureRecipe, frame: pd.DataFrame) -> plt.Figure:
    facets = sorted(
        {(str(role), str(panel)) for role, panel in frame[["case_role", "panel"]].itertuples(index=False)},
        key=lambda values: values,
    )
    figure, axes = _axes(len(facets))
    for axis, (role, panel) in zip(axes, facets):
        selected = frame[(frame["case_role"].astype(str) == role) & (frame["panel"].astype(str) == panel)]
        selected = selected.sort_values(recipe.x, kind="mergesort")
        axis.plot(selected[recipe.x], selected[recipe.y], linewidth=0.9)
        truth = pd.to_numeric(selected["ground_truth"], errors="coerce").fillna(0).to_numpy()
        if np.any(truth > 0):
            lower, upper = axis.get_ylim()
            axis.fill_between(selected[recipe.x], lower, upper, where=truth > 0, alpha=0.12, color="red")
        axis.set_title(f"{role}: {panel}")
        axis.set_xlabel(recipe.x)
        axis.set_ylabel(recipe.y)
        axis.grid(alpha=0.2)
    return figure


def _render(recipe: RoughFigureRecipe, frame: pd.DataFrame) -> plt.Figure:
    if recipe.kind in {"scatter", "line", "bar"}:
        return _plot_standard(recipe, frame)
    if recipe.kind == "interaction":
        return _plot_heatmap(recipe, frame)
    if recipe.kind == "heatmap":
        return _plot_heatmap(recipe, frame)
    if recipe.kind == "mapping":
        return _plot_heatmap(recipe, frame)
    if recipe.kind == "qualitative":
        return _plot_qualitative(recipe, frame)
    raise ValueError(f"unknown rough-figure kind: {recipe.kind}")


def render_rough_figure_set(
    inputs: Mapping[str, pd.DataFrame | str | Path],
    *,
    plot_data_root: str | Path,
    figure_root: str | Path,
) -> pd.DataFrame:
    """Render every mandatory recipe or record an explicit blocked status."""

    data_root = Path(plot_data_root)
    output_root = Path(figure_root)
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, Any]] = []
    for recipe in MANDATORY_ROUGH_FIGURES:
        tidy_path = data_root / f"{recipe.name}.csv"
        svg_path = output_root / f"{recipe.name}.svg"
        pdf_path = output_root / f"{recipe.name}.pdf"
        source = inputs.get(recipe.name)
        raw: pd.DataFrame | None = None
        if source is None:
            status_rows.append(
                {
                    "figure": recipe.name,
                    "status": "BLOCKED_MISSING_INPUT",
                    "reason": "no input table supplied; no placeholder values generated",
                    "input_rows": 0,
                    "plotted_rows": 0,
                    "tidy_csv": "",
                    "svg": "",
                    "pdf": "",
                }
            )
            continue
        try:
            raw = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()
            canonical = _canonical(raw)
            _atomic_text(tidy_path, canonical.to_csv(index=False, lineterminator="\n"))
            valid = _valid_rows(canonical, recipe)
            figure = _render(recipe, valid)
            figure.suptitle(recipe.name.replace("_", " "))
            figure.tight_layout()
            figure.savefig(
                svg_path,
                format="svg",
                bbox_inches="tight",
                metadata={"Creator": "ViTTrace v3", "Date": "1980-01-01T00:00:00Z"},
            )
            fixed_date = datetime(1980, 1, 1, tzinfo=timezone.utc)
            figure.savefig(
                pdf_path,
                format="pdf",
                bbox_inches="tight",
                metadata={
                    "Creator": "ViTTrace v3",
                    "CreationDate": fixed_date,
                    "ModDate": fixed_date,
                },
            )
            plt.close(figure)
            status_rows.append(
                {
                    "figure": recipe.name,
                    "status": "COMPLETE",
                    "reason": "",
                    "input_rows": len(raw),
                    "plotted_rows": len(valid),
                    "tidy_csv": tidy_path.as_posix(),
                    "svg": svg_path.as_posix(),
                    "pdf": pdf_path.as_posix(),
                }
            )
        except Exception as error:
            plt.close("all")
            status_rows.append(
                {
                    "figure": recipe.name,
                    "status": "BLOCKED_INVALID_INPUT",
                    "reason": f"{type(error).__name__}: {error}",
                    "input_rows": len(raw) if raw is not None else 0,
                    "plotted_rows": 0,
                    "tidy_csv": tidy_path.as_posix() if tidy_path.exists() else "",
                    "svg": "",
                    "pdf": "",
                }
            )
    status = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    _atomic_text(data_root / "rough_figure_status.csv", status.to_csv(index=False, lineterminator="\n"))
    _atomic_text(
        data_root / "rough_figure_status.json",
        json.dumps(status.to_dict(orient="records"), indent=2, sort_keys=True) + "\n",
    )
    return status


__all__ = [
    "MANDATORY_ROUGH_FIGURES",
    "RECIPE_BY_NAME",
    "RoughFigureRecipe",
    "STATUS_COLUMNS",
    "render_rough_figure_set",
]
