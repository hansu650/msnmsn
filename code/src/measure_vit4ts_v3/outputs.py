"""Tidy table and plot-data writers for the isolated v3 scaffold."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .aggregate import AggregationBundle
from .metrics import ALL_METRICS
from .registry import ArmRegistry


def tidy_table_frame(
    summary: pd.DataFrame,
    registry: ArmRegistry,
    *,
    metrics: Sequence[str] = ALL_METRICS,
) -> pd.DataFrame:
    """Return a stable long-form table ordered by registry arm and metric."""

    required = {
        "aggregation",
        "family",
        "subgroup",
        "arm",
        "metric",
        "value",
        "n_valid",
        "n_total",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing tidy columns: {sorted(missing)}")
    requested = tuple(str(metric) for metric in metrics)
    if not requested or any(metric not in ALL_METRICS for metric in requested):
        raise ValueError("requested metrics differ from the v3 metric registry")
    frame = summary.loc[summary["metric"].astype(str).isin(requested)].copy()
    arm_order = {arm: index for index, arm in enumerate(registry.arm_ids)}
    metric_order = {metric: index for index, metric in enumerate(requested)}
    if not set(frame["arm"].astype(str)).issubset(arm_order):
        raise ValueError("summary contains an unregistered arm")
    frame["_arm_order"] = frame["arm"].map(arm_order)
    frame["_metric_order"] = frame["metric"].map(metric_order)
    return (
        frame.sort_values(
            ["aggregation", "family", "subgroup", "_metric_order", "_arm_order"]
        )
        .drop(columns=["_arm_order", "_metric_order"])
        .reset_index(drop=True)
    )


def subgroup_delta_plot_frame(
    subgroup11: pd.DataFrame,
    registry: ArmRegistry,
    metric: str,
) -> pd.DataFrame:
    """Create one plot-ready row per frozen contrast and subgroup."""

    if metric not in ALL_METRICS:
        raise ValueError(f"unsupported plot metric: {metric}")
    frame = subgroup11.loc[subgroup11["metric"] == metric].copy()
    required_arms = set(registry.arm_ids)
    if set(frame["arm"].astype(str)) != required_arms:
        raise ValueError("subgroup plot input does not contain every registry arm")
    pivot = frame.pivot(index=["family", "subgroup"], columns="arm", values="value")
    output = []
    for contrast_order, contrast in enumerate(registry.contrasts):
        for (family, subgroup), row in pivot.iterrows():
            candidate = float(row[contrast.candidate])
            control = float(row[contrast.control])
            output.append(
                {
                    "contrast_order": contrast_order,
                    "contrast_id": contrast.contrast_id,
                    "contrast_family": contrast.family,
                    "candidate": contrast.candidate,
                    "control": contrast.control,
                    "family": str(family),
                    "subgroup": str(subgroup),
                    "metric": metric,
                    "candidate_value": candidate,
                    "control_value": control,
                    "delta": candidate - control
                    if np.isfinite(candidate) and np.isfinite(control)
                    else np.nan,
                    "zero_reference": 0.0,
                }
            )
    return pd.DataFrame(output).sort_values(
        ["contrast_order", "family", "subgroup"]
    ).reset_index(drop=True)


def bootstrap_plot_frame(bootstrap: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = {
        "contrast_id",
        "contrast_family",
        "candidate",
        "control",
        "metric",
        "delta",
        "ci_lower",
        "ci_upper",
    }
    missing = required - set(bootstrap.columns)
    if missing:
        raise ValueError(f"bootstrap is missing plot columns: {sorted(missing)}")
    frame = bootstrap.loc[bootstrap["metric"] == metric, sorted(required)].copy()
    frame["zero_reference"] = 0.0
    return frame.sort_values(["contrast_family", "contrast_id"]).reset_index(drop=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_tidy_outputs(
    output_root: Path,
    bundle: AggregationBundle,
    registry: ArmRegistry,
    *,
    bootstraps: pd.DataFrame | None = None,
) -> tuple[Path, ...]:
    """Atomically write compact numeric and plot-ready v3 artifacts."""

    root = Path(output_root)
    outputs: list[Path] = []
    tables = {
        "per_series.csv": bundle.per_series,
        "subgroup11.csv": bundle.subgroup11,
        "family3.csv": bundle.family3,
        "equal11.csv": bundle.equal11,
        "fileweighted.csv": bundle.fileweighted,
    }
    for name, frame in tables.items():
        target = root / name
        _atomic_csv(target, frame)
        outputs.append(target)
    registry_path = root / "arm_registry.json"
    _atomic_json(registry_path, registry.to_payload())
    outputs.append(registry_path)
    for metric in ALL_METRICS:
        plot_path = root / f"plot_subgroup_delta_{metric}.csv"
        _atomic_csv(plot_path, subgroup_delta_plot_frame(bundle.subgroup11, registry, metric))
        outputs.append(plot_path)
    if bootstraps is not None:
        bootstrap_path = root / "hierarchical_bootstrap.csv"
        _atomic_csv(bootstrap_path, bootstraps)
        outputs.append(bootstrap_path)
        for metric in sorted(set(bootstraps["metric"].astype(str))):
            target = root / f"plot_bootstrap_{metric}.csv"
            _atomic_csv(target, bootstrap_plot_frame(bootstraps, metric))
            outputs.append(target)
    return tuple(outputs)
