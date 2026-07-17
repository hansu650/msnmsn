"""Exact sufficient-statistics implementation of PaAno's frozen VUS curve.

This module is evaluator-only.  It preserves the threshold grid, range
construction, endpoint conventions, and curve tuple of the frozen vendor
``RangeAUC_volume_opt`` implementation while avoiding a full point scan for
every threshold at every window.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from .schemas import canonicalize_unit_interval_metric
from .vendor import VendorSymbols


_EXACT_THRESHOLD_COUNT = 250


def _validate_inputs(
    labels: np.ndarray,
    scores: np.ndarray,
    sliding_window: int,
    thresholds: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.size == 0:
        raise ValueError("VUS labels must be a non-empty vector")
    try:
        labels_finite = np.isfinite(raw_labels).all()
    except TypeError as exc:
        raise ValueError("VUS labels must be finite binary values") from exc
    if not labels_finite or not np.logical_or(raw_labels == 0, raw_labels == 1).all():
        raise ValueError("VUS labels must be finite binary values")

    try:
        score_values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("VUS scores must be finite numeric values") from exc
    if score_values.ndim != 1 or score_values.shape != raw_labels.shape:
        raise ValueError("VUS scores and labels must be aligned vectors")
    if not np.isfinite(score_values).all():
        raise ValueError("VUS scores must be finite numeric values")

    if isinstance(sliding_window, (bool, np.bool_)) or not isinstance(
        sliding_window, Integral
    ):
        raise ValueError("VUS sliding window must be a non-negative integer")
    window = int(sliding_window)
    if window < 0:
        raise ValueError("VUS sliding window must be a non-negative integer")
    if isinstance(thresholds, (bool, np.bool_)) or not isinstance(thresholds, Integral):
        raise ValueError("VUS threshold count must be exactly 250")
    if int(thresholds) != _EXACT_THRESHOLD_COUNT:
        raise ValueError("VUS threshold count must be exactly 250")

    label_values = raw_labels.astype(np.int8, copy=False)
    positives = int(np.sum(label_values, dtype=np.int64))
    if positives == 0:
        raise ValueError("VUS requires at least one anomaly segment")
    if positives == label_values.size:
        raise ValueError("VUS requires at least one normal point")
    return label_values, score_values, window


def _activation_indices(score_values: np.ndarray, thresholds: int) -> np.ndarray:
    # These are exactly the threshold locations used by the frozen vendor.
    score_sorted = -np.sort(-score_values)
    locations = np.linspace(0, len(score_values) - 1, thresholds).astype(int)
    threshold_values = score_sorted[locations]
    if not np.all(threshold_values[:-1] >= threshold_values[1:]):
        raise RuntimeError("vendor VUS thresholds are not non-increasing")

    # For non-increasing thresholds, a point is active from the first index at
    # which score >= threshold.  side='left' is required for duplicate scores.
    activation = np.searchsorted(
        -threshold_values,
        -score_values,
        side="left",
    ).astype(np.int64, copy=False)
    if activation.shape != score_values.shape or np.any(activation >= thresholds):
        raise RuntimeError("failed to assign every score to the vendor threshold grid")
    return activation


def _curve_from_validated(
    label_values: np.ndarray,
    score_values: np.ndarray,
    sliding_window: int,
    vendor: VendorSymbols,
    thresholds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    metricor = vendor.basic_metricor()
    required = ("range_convers_new", "new_sequence", "sequencing")
    if any(not callable(getattr(metricor, name, None)) for name in required):
        raise RuntimeError("frozen vendor range-construction surface changed")

    sequence = metricor.range_convers_new(label_values)
    if not sequence:
        raise ValueError("VUS requires at least one anomaly segment")

    activation = _activation_indices(score_values, thresholds)
    predicted_count = np.cumsum(
        np.bincount(activation, minlength=thresholds),
        dtype=np.int64,
    )
    if predicted_count.shape != (thresholds,) or np.any(predicted_count <= 0):
        raise RuntimeError("vendor threshold grid produced an invalid prediction count")

    anomaly_mask = label_values == 1
    original_predicted = np.cumsum(
        np.bincount(activation[anomaly_mask], minlength=thresholds),
        dtype=np.int64,
    ).astype(np.float64, copy=False)
    original_positive_count = float(np.sum(label_values, dtype=np.int64))
    point_count = label_values.size
    label_float = label_values.astype(np.float64, copy=False)

    window_values = np.arange(0, sliding_window + 1, 1)
    tpr = np.zeros((sliding_window + 1, thresholds + 2), dtype=np.float64)
    fpr = np.zeros((sliding_window + 1, thresholds + 2), dtype=np.float64)
    precision = np.ones((sliding_window + 1, thresholds + 1), dtype=np.float64)
    auc_by_window = np.zeros(sliding_window + 1, dtype=np.float64)
    ap_by_window = np.zeros(sliding_window + 1, dtype=np.float64)

    for window in window_values:
        # Delegate the two range primitives to the frozen vendor.  Only the
        # threshold-wise reductions below are replaced by sufficient statistics.
        labels_extended = np.asarray(
            metricor.sequencing(label_values, sequence, int(window)),
            dtype=np.float64,
        )
        merged_ranges = metricor.new_sequence(
            labels_extended,
            sequence,
            int(window),
        )
        if labels_extended.shape != label_values.shape or not merged_ranges:
            raise RuntimeError("frozen vendor returned an invalid extended range")
        if not np.isfinite(labels_extended).all():
            raise RuntimeError("frozen vendor returned non-finite range weights")

        halo_weights = labels_extended - label_float
        if np.any(halo_weights < 0):
            raise RuntimeError("frozen vendor returned invalid anomaly-range weights")
        halo_predicted = np.cumsum(
            np.bincount(
                activation,
                weights=halo_weights,
                minlength=thresholds,
            ),
            dtype=np.float64,
        )
        true_positive = original_predicted + halo_predicted
        weighted_label_count = original_positive_count + halo_predicted
        positive_denominator = (
            original_positive_count + weighted_label_count
        ) / 2.0
        negative_denominator = point_count - positive_denominator
        if np.any(positive_denominator <= 0) or np.any(negative_denominator <= 0):
            raise ValueError("VUS range denominator is not strictly positive")

        # The vendor existence term asks whether each merged range contains at
        # least one active point.  A masked reduce-at obtains the minimum
        # activation index per range; its cumulative histogram is the exact
        # threshold-wise existence count.
        range_starts = np.fromiter(
            (int(start) for start, _ in merged_ranges),
            dtype=np.int64,
            count=len(merged_ranges),
        )
        if (
            range_starts.size != len(merged_ranges)
            or np.any(range_starts < 0)
            or np.any(range_starts >= point_count)
            or np.any(range_starts[1:] <= range_starts[:-1])
        ):
            raise RuntimeError("frozen vendor returned invalid merged-range starts")
        masked_activation = np.where(
            labels_extended > 0.0,
            activation,
            thresholds,
        )
        first_activation = np.minimum.reduceat(masked_activation, range_starts)
        if np.any(first_activation >= thresholds):
            raise RuntimeError("frozen vendor returned an empty merged range")
        existence_count = np.cumsum(
            np.bincount(first_activation, minlength=thresholds),
            dtype=np.int64,
        )

        existence_ratio = existence_count / float(len(merged_ranges))
        recall = np.minimum(true_positive / positive_denominator, 1.0)
        threshold_tpr = recall * existence_ratio
        false_positive = predicted_count - true_positive
        threshold_fpr = false_positive / negative_denominator
        threshold_precision = true_positive / predicted_count

        row = int(window)
        tpr[row, 1 : thresholds + 1] = threshold_tpr
        tpr[row, thresholds + 1] = 1.0
        fpr[row, 1 : thresholds + 1] = threshold_fpr
        fpr[row, thresholds + 1] = 1.0
        precision[row, 1:] = threshold_precision

        width = fpr[row, 1:] - fpr[row, :-1]
        height = (tpr[row, 1:] + tpr[row, :-1]) / 2.0
        auc_by_window[row] = np.dot(width, height)
        width_pr = tpr[row, 1:-1] - tpr[row, :-2]
        ap_by_window[row] = np.dot(width_pr, precision[row, 1:])

    surfaces = (tpr, fpr, precision, auc_by_window, ap_by_window)
    if any(not np.isfinite(surface).all() for surface in surfaces):
        raise ValueError("sufficient-statistics VUS produced a non-finite curve")

    # Preserve generate_curve's exact public tuple and flattening order.
    x = np.asarray(tpr).reshape(1, -1).ravel()
    x_ap = np.asarray(tpr)[:, :-1].reshape(1, -1).ravel()
    y = np.asarray(fpr).reshape(1, -1).ravel()
    weights = np.asarray(precision).reshape(1, -1).ravel()
    z = np.repeat(window_values, len(tpr[0]))
    z_ap = np.repeat(window_values, len(tpr[0]) - 1)
    average_auc = float(sum(auc_by_window) / len(window_values))
    average_ap = float(sum(ap_by_window) / len(window_values))
    return y, z, x, x_ap, weights, z_ap, average_auc, average_ap


def generate_curve_exact(
    labels: np.ndarray,
    scores: np.ndarray,
    sliding_window: int,
    vendor: VendorSymbols,
    thresholds: int = _EXACT_THRESHOLD_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return the frozen vendor curve tuple through exact sufficient statistics."""

    label_values, score_values, window = _validate_inputs(
        labels,
        scores,
        sliding_window,
        thresholds,
    )
    return _curve_from_validated(
        label_values,
        score_values,
        window,
        vendor,
        int(thresholds),
    )


def compute_threshold_free_metrics_exact_vus(
    scores: np.ndarray,
    labels: np.ndarray,
    sliding_window: int,
    vendor: VendorSymbols,
    thresholds: int = _EXACT_THRESHOLD_COUNT,
) -> dict[str, float]:
    """Compute evaluator metrics with unchanged point metrics and exact fast VUS."""

    label_values, score_values, window = _validate_inputs(
        labels,
        scores,
        sliding_window,
        thresholds,
    )
    if window <= 0:
        # Preserve the existing benchmark metric wrapper's positive-window
        # contract.  ``generate_curve_exact`` still supports window-zero parity.
        raise ValueError("benchmark VUS sliding window must be positive")
    metricor = vendor.basic_metricor()
    auprc = float(metricor.metric_PR(label_values, score_values))
    auroc = float(metricor.metric_ROC(label_values, score_values))
    curve = _curve_from_validated(
        label_values,
        score_values,
        window,
        vendor,
        int(thresholds),
    )
    if len(curve) != 8:
        raise RuntimeError("exact VUS curve return surface changed")
    raw_values = {
        "vus_pr": float(curve[-1]),
        "auprc": auprc,
        "vus_roc": float(curve[-2]),
        "auroc": auroc,
    }
    return {
        name: canonicalize_unit_interval_metric(name, value)
        for name, value in raw_values.items()
    }
