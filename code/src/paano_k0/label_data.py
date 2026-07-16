"""Evaluator-only label input.

Nothing in the training, memory, or scoring dependency graph imports this module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .schemas import SeriesSpec


def read_labels(spec: SeriesSpec) -> NDArray[np.int8]:
    frame = pd.read_csv(spec.csv_path, usecols=[spec.label_column])
    raw = frame.loc[:, spec.label_column].to_numpy(copy=False)
    if raw.ndim != 1 or raw.shape[0] != spec.rows:
        raise ValueError("label length does not match frozen manifest")
    if not np.isfinite(raw).all():
        raise ValueError("labels contain NaN/Inf")
    unique = set(np.unique(raw).tolist())
    if not unique.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(f"labels must be binary, observed {sorted(unique)}")
    return np.ascontiguousarray(raw.astype(np.int8, copy=False))


def validate_score_alignment(
    labels: NDArray[np.integer], scores: NDArray[np.floating], expected_rows: int
) -> None:
    label_values = np.asarray(labels)
    score_values = np.asarray(scores)
    if label_values.ndim != 1 or score_values.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if label_values.shape[0] != expected_rows or score_values.shape[0] != expected_rows:
        raise ValueError("labels/scores do not match the frozen row count")
    if label_values.shape != score_values.shape:
        raise ValueError("labels and scores are not aligned")
    if not np.isfinite(score_values).all():
        raise ValueError("scores contain NaN/Inf")
    if not set(np.unique(label_values).tolist()).issubset({0, 1}):
        raise ValueError("labels are not binary")

