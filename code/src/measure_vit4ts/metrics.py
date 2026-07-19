"""Evaluator-only metrics for the measure-consistent ViT4TS experiment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import average_precision_score


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def contextual_interval_f1(
    ground_truth: Sequence[tuple[float, float]],
    detected: Sequence[tuple[float, float]],
) -> tuple[float, float, float]:
    """Reproduce VLM4TS's released unweighted interval-overlap F1."""

    truth = [tuple(map(float, item)) for item in ground_truth]
    predictions = [tuple(map(float, item)) for item in detected]
    true_positive = 0
    false_positive = 0
    for prediction in predictions:
        overlap_count = sum(_intervals_overlap(prediction, item) for item in truth)
        if overlap_count:
            true_positive += overlap_count
        else:
            false_positive += 1
    false_negative = sum(
        not any(_intervals_overlap(item, prediction) for prediction in predictions)
        for item in truth
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def flags_to_intervals(flags: np.ndarray, timestamps: np.ndarray) -> tuple[tuple[float, float], ...]:
    mask = np.asarray(flags, dtype=bool)
    time = np.asarray(timestamps)
    if mask.ndim != 1 or time.shape != mask.shape:
        raise ValueError("flags and timestamps must be aligned vectors")
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return tuple((float(time[start]), float(time[end])) for start, end in zip(starts, ends))


def paper_f1_max(
    scores: np.ndarray,
    timestamps: np.ndarray,
    ground_truth_intervals: Sequence[tuple[float, float]],
    alpha_grid: tuple[float, ...] = (0.1, 0.01, 0.001),
) -> tuple[float, float, float]:
    """Compute the released EWMA/Gaussian threshold sweep after score commit."""

    values = np.asarray(scores, dtype=np.float64)
    time = np.asarray(timestamps)
    if values.ndim != 1 or time.shape != values.shape or values.size == 0:
        raise ValueError("scores and timestamps must be non-empty aligned vectors")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    span = max(1, int(values.size * 0.01))
    smooth = pd.Series(values).ewm(span=span).mean().to_numpy(dtype=np.float64)
    mean = float(np.mean(smooth))
    standard_deviation = float(np.std(smooth))
    best = (-1.0, float("nan"), float("nan"))
    for alpha in alpha_grid:
        alpha_value = float(alpha)
        if not 0.0 < alpha_value < 1.0:
            raise ValueError("alpha values must lie in (0, 1)")
        threshold = mean + float(norm.ppf(1.0 - alpha_value)) * standard_deviation
        detected = flags_to_intervals(smooth > threshold, time)
        _, _, f1 = contextual_interval_f1(ground_truth_intervals, detected)
        if f1 > best[0]:
            best = (f1, alpha_value, threshold)
    return best


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    label_values = np.asarray(labels)
    score_values = np.asarray(scores, dtype=np.float64)
    if label_values.ndim != 1 or score_values.shape != label_values.shape:
        raise ValueError("labels and scores must be aligned vectors")
    if not np.logical_or(label_values == 0, label_values == 1).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must be finite")
    if np.unique(label_values).size < 2:
        raise ValueError("AUPRC requires both normal and anomalous points")
    value = float(average_precision_score(label_values.astype(np.uint8), score_values))
    if not np.isfinite(value) or value < -5e-12 or value > 1.0 + 5e-12:
        raise RuntimeError(f"sklearn returned invalid AUPRC {value}")
    return float(np.clip(value, 0.0, 1.0))


def vus_pr(labels: np.ndarray, scores: np.ndarray, max_window: int) -> float:
    """Compute TSB-AD VUS-PR using its pinned evaluator implementation."""

    label_values = np.asarray(labels, dtype=np.uint8)
    score_values = np.asarray(scores, dtype=np.float64)
    if label_values.ndim != 1 or score_values.shape != label_values.shape:
        raise ValueError("labels and scores must be aligned vectors")
    if not np.logical_or(label_values == 0, label_values == 1).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(score_values).all() or int(max_window) <= 0:
        raise ValueError("VUS inputs must be finite and max_window positive")
    try:
        from TSB_AD.evaluation.basic_metrics import generate_curve
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("install TSB_AD==1.5 with --no-deps for VUS evaluation") from exc
    curve = generate_curve(
        label_values,
        score_values,
        int(max_window),
        version="opt_mem",
        thre=250,
    )
    value = float(curve[-1])
    if not np.isfinite(value) or value < -5e-12 or value > 1.0 + 5e-12:
        raise RuntimeError(f"TSB-AD returned invalid VUS-PR {value}")
    return float(np.clip(value, 0.0, 1.0))


def cosine_drift(reference: np.ndarray, perturbed: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(perturbed, dtype=np.float64)
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("embedding arrays must have the same shape")
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    denom = np.maximum(left_norm * right_norm, np.finfo(np.float64).tiny)
    cosine = np.sum(left * right, axis=-1) / denom
    return float(np.mean(1.0 - np.clip(cosine, -1.0, 1.0)))


def nearest_match_flip(reference_index: np.ndarray, perturbed_index: np.ndarray) -> float:
    left = np.asarray(reference_index)
    right = np.asarray(perturbed_index)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("match-index arrays must have the same non-empty shape")
    return float(np.mean(left != right))


def absolute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    left = np.asarray(x, dtype=np.float64).ravel()
    right = np.asarray(y, dtype=np.float64).ravel()
    if left.shape != right.shape or left.size < 3:
        raise ValueError("Spearman inputs must be aligned and contain at least three points")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return abs(value) if np.isfinite(value) else 0.0


def relative_control_variation(
    scores: np.ndarray, control_axis: int = 0, eps: float = 1e-12
) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("control scores must be finite and non-empty")
    spread = np.std(values, axis=control_axis)
    scale = np.mean(np.abs(values), axis=control_axis)
    return float(np.mean(spread / np.maximum(scale, float(eps))))
