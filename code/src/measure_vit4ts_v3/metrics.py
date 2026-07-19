"""Label-aware v3 metric semantics and the pre-fixed validity mask.

This module is evaluator-only.  It deliberately defines F1-max, AUPRC, and
VUS-PR as undefined for a series without both classes instead of replacing
undefined values by zero or one.  Anomaly-free files receive a separate,
prefixed-threshold false-positive burden record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from measure_vit4ts.metrics import average_precision, paper_f1_max, vus_pr


DETECTION_METRICS = ("f1_max", "auprc", "vus_pr")
ANOMALY_FREE_METRICS = (
    "anomaly_free_fp_rate",
    "anomaly_free_fp_count",
    "anomaly_free_mean_excess",
    "anomaly_free_score_p95",
)
ALL_METRICS = (*DETECTION_METRICS, *ANOMALY_FREE_METRICS)


def _aligned_binary_inputs(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    label_values = np.asarray(labels)
    score_values = np.asarray(scores, dtype=np.float64)
    if label_values.ndim != 1 or score_values.shape != label_values.shape:
        raise ValueError("labels and scores must be nonempty aligned vectors")
    if label_values.size == 0:
        raise ValueError("labels and scores must be nonempty aligned vectors")
    if not np.logical_or(label_values == 0, label_values == 1).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must be finite")
    return label_values.astype(np.uint8), score_values


def anomaly_free_fp_burden(scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Measure false-positive burden at one registry-frozen threshold."""

    values = np.asarray(scores, dtype=np.float64)
    threshold_value = float(threshold)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("anomaly-free scores must be a finite nonempty vector")
    if not np.isfinite(threshold_value):
        raise ValueError("anomaly-free threshold must be finite and prefixed")
    exceedance = np.maximum(values - threshold_value, 0.0)
    flags = values > threshold_value
    return {
        "anomaly_free_fp_rate": float(np.mean(flags)),
        "anomaly_free_fp_count": float(np.sum(flags)),
        "anomaly_free_mean_excess": float(np.mean(exceedance)),
        "anomaly_free_score_p95": float(np.quantile(values, 0.95)),
    }


def evaluate_series(
    labels: np.ndarray,
    scores: np.ndarray,
    timestamps: np.ndarray,
    anomaly_intervals: Sequence[tuple[float, float]],
    *,
    fp_threshold: float,
    alpha_grid: tuple[float, ...] = (0.1, 0.01, 0.001),
    vus_max_window: int = 100,
    f1_fn: Callable[..., tuple[float, float, float]] = paper_f1_max,
    auprc_fn: Callable[[np.ndarray, np.ndarray], float] = average_precision,
    vus_fn: Callable[[np.ndarray, np.ndarray, int], float] = vus_pr,
) -> dict[str, Any]:
    """Evaluate one committed score vector under corrected-primary semantics."""

    label_values, score_values = _aligned_binary_inputs(labels, scores)
    time_values = np.asarray(timestamps)
    if time_values.shape != score_values.shape:
        raise ValueError("timestamps must align with labels and scores")
    n_positive = int(label_values.sum())
    n_points = int(label_values.size)
    has_both_classes = 0 < n_positive < n_points
    if has_both_classes:
        f1, alpha, threshold = f1_fn(
            score_values, time_values, anomaly_intervals, alpha_grid
        )
        auprc = float(auprc_fn(label_values, score_values))
        vus = float(vus_fn(label_values, score_values, int(vus_max_window)))
    else:
        f1 = alpha = threshold = np.nan
        auprc = vus = np.nan
    result: dict[str, Any] = {
        "n_points": n_points,
        "n_positive": n_positive,
        "has_both_classes": has_both_classes,
        "f1_max": float(f1),
        "f1_winning_alpha": float(alpha),
        "f1_threshold_evaluator_only": float(threshold),
        "auprc": auprc,
        "vus_pr": vus,
    }
    if n_positive == 0:
        result.update(anomaly_free_fp_burden(score_values, fp_threshold))
    else:
        result.update({name: np.nan for name in ANOMALY_FREE_METRICS})
    return result


def freeze_valid_series_mask(series_index: pd.DataFrame) -> pd.DataFrame:
    """Freeze one arm-independent valid-series mask from label counts only."""

    required = {"series_id", "family", "subgroup", "n_points", "n_positive"}
    missing = required - set(series_index.columns)
    if missing:
        raise ValueError(f"series index is missing columns: {sorted(missing)}")
    if series_index.duplicated("series_id").any() or series_index.empty:
        raise ValueError("series index must contain unique nonempty series IDs")
    frame = series_index.loc[:, sorted(required)].copy()
    frame["n_points"] = pd.to_numeric(frame["n_points"], errors="raise").astype(int)
    frame["n_positive"] = pd.to_numeric(frame["n_positive"], errors="raise").astype(int)
    if (
        (frame["n_points"] <= 0).any()
        or (frame["n_positive"] < 0).any()
        or (frame["n_positive"] > frame["n_points"]).any()
    ):
        raise ValueError("series label counts are invalid")
    both = (frame["n_positive"] > 0) & (frame["n_positive"] < frame["n_points"])
    no_positive = frame["n_positive"] == 0
    frame["valid_f1_max"] = both
    frame["valid_auprc"] = both
    frame["valid_vus_pr"] = both
    for metric in ANOMALY_FREE_METRICS:
        frame[f"valid_{metric}"] = no_positive
    frame["threshold_free_reason"] = np.select(
        [no_positive, frame["n_positive"] == frame["n_points"]],
        ["NO_POSITIVE", "NO_NEGATIVE"],
        default="DEFINED",
    )
    frame["anomaly_free_reason"] = np.where(
        no_positive, "DEFINED", "HAS_POSITIVE"
    )
    return frame.sort_values("series_id").reset_index(drop=True)


def valid_mask_sha256(mask: pd.DataFrame) -> str:
    required = [
        "series_id",
        "family",
        "subgroup",
        "n_points",
        "n_positive",
        *(f"valid_{metric}" for metric in ALL_METRICS),
    ]
    missing = set(required) - set(mask.columns)
    if missing:
        raise ValueError(f"valid mask is missing columns: {sorted(missing)}")
    records = mask.loc[:, required].sort_values("series_id").to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def validate_metrics_against_mask(
    metrics: pd.DataFrame,
    mask: pd.DataFrame,
    arm_ids: Sequence[str],
) -> pd.DataFrame:
    """Fail closed unless every arm uses the same pre-fixed validity mask."""

    required = {
        "series_id",
        "arm",
        "family",
        "subgroup",
        "n_points",
        "n_positive",
        *ALL_METRICS,
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"per-series metrics are missing columns: {sorted(missing)}")
    arms = tuple(str(value) for value in arm_ids)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arm_ids must be a unique nonempty sequence")
    series_ids = tuple(mask["series_id"].astype(str))
    expected = {(series_id, arm) for series_id in series_ids for arm in arms}
    actual = set(zip(metrics["series_id"].astype(str), metrics["arm"].astype(str)))
    if actual != expected or metrics.duplicated(["series_id", "arm"]).any():
        raise ValueError("per-series metrics must cover the exact mask x arm grid")
    if set(metrics["arm"].astype(str)) != set(arms):
        raise ValueError("per-series metric arms differ from the frozen registry")

    mask_columns = [
        "series_id",
        "family",
        "subgroup",
        "n_points",
        "n_positive",
        *(f"valid_{metric}" for metric in ALL_METRICS),
    ]
    merged = metrics.merge(
        mask.loc[:, mask_columns],
        on="series_id",
        how="left",
        suffixes=("", "_mask"),
        validate="many_to_one",
    )
    for field in ("family", "subgroup", "n_points", "n_positive"):
        if not (merged[field].astype(str) == merged[f"{field}_mask"].astype(str)).all():
            raise ValueError(f"per-series {field} differs from the pre-fixed mask")
        merged = merged.drop(columns=f"{field}_mask")
    for metric in ALL_METRICS:
        values = pd.to_numeric(merged[metric], errors="coerce").to_numpy(dtype=np.float64)
        valid = merged[f"valid_{metric}"].astype(bool).to_numpy()
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
            raise ValueError(
                f"{metric} must be finite exactly on its pre-fixed valid-series mask"
            )
    bounded = ("f1_max", "auprc", "vus_pr", "anomaly_free_fp_rate")
    for metric in bounded:
        finite = pd.to_numeric(merged[metric], errors="coerce").dropna().to_numpy()
        if np.any(finite < 0.0) or np.any(finite > 1.0):
            raise ValueError(f"{metric} must lie in [0,1] when defined")
    return merged.sort_values(["family", "subgroup", "series_id", "arm"]).reset_index(
        drop=True
    )
