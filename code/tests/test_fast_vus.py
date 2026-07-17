from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paano_k0.evaluate_scores import compute_threshold_free_metrics
from paano_k0.fast_vus import (
    compute_threshold_free_metrics_exact_vus,
    generate_curve_exact,
)
from paano_k0.vendor import load_vendor_symbols


VENDOR_ROOT = Path(r"C:\Users\qintian\Desktop\msn\vendor\PaAno")
VENDOR_SHA = "d4c67116190efa4592dc6a8a157ced0def68b6af"
PARITY_ATOL = 5e-12


@pytest.fixture(scope="module")
def vendor():
    return load_vendor_symbols(VENDOR_ROOT, VENDOR_SHA)


def _assert_full_curve_parity(
    labels: np.ndarray,
    scores: np.ndarray,
    window: int,
    vendor,
) -> float:
    literal = vendor.generate_curve(labels, scores, window, "opt", 250)
    exact = generate_curve_exact(labels, scores, window, vendor, thresholds=250)
    assert len(literal) == len(exact) == 8
    maximum = 0.0
    for literal_element, exact_element in zip(literal, exact, strict=True):
        literal_values = np.asarray(literal_element)
        exact_values = np.asarray(exact_element)
        assert exact_values.shape == literal_values.shape
        assert np.isfinite(exact_values).all()
        error = float(np.max(np.abs(exact_values - literal_values)))
        maximum = max(maximum, error)
        np.testing.assert_allclose(
            exact_values,
            literal_values,
            rtol=0.0,
            atol=PARITY_ATOL,
        )
    return maximum


@pytest.mark.parametrize(
    ("labels", "scores", "window"),
    (
        # Boundary anomalies at both endpoints.
        (
            np.array([1, 1, 0, 0, 0, 1], dtype=np.int8),
            np.array([0.8, 0.8, 0.1, 0.4, 0.4, 0.9]),
            5,
        ),
        # The two extended ranges merge for larger windows.
        (
            np.array([0, 1, 0, 0, 1, 0, 0], dtype=np.int8),
            np.array([0.2, 0.7, 0.1, 0.3, 0.6, 0.4, 0.5]),
            6,
        ),
        # Highly fragmented ranges and duplicate threshold values.
        (
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8),
            np.array([0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5]),
            9,
        ),
        # Window zero is a required direct-curve boundary case.
        (
            np.array([0, 0, 1, 1, 0, 0], dtype=np.int8),
            np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4]),
            0,
        ),
        # Dense labels retain a valid normal denominator.
        (
            np.array([1, 1, 1, 1, 1, 1, 0, 1], dtype=np.int8),
            np.array([0.7, 0.2, 0.7, 0.4, 0.4, 0.9, 0.1, 0.2]),
            7,
        ),
    ),
)
def test_exact_curve_matches_vendor_boundary_surfaces(
    labels: np.ndarray,
    scores: np.ndarray,
    window: int,
    vendor,
) -> None:
    assert _assert_full_curve_parity(labels, scores, window, vendor) <= PARITY_ATOL


@pytest.mark.parametrize("seed", range(20))
def test_exact_curve_matches_vendor_randomized_surfaces(seed: int, vendor) -> None:
    rng = np.random.default_rng(seed)
    length = int(rng.integers(8, 161))
    density = float(rng.uniform(0.02, 0.85))
    labels = (rng.random(length) < density).astype(np.int8)
    labels[int(rng.integers(0, length))] = 1
    labels[int(rng.integers(0, length))] = 0
    if labels.all():
        labels[-1] = 0
    if not labels.any():
        labels[0] = 1
    # Rounding deliberately creates duplicated score values and thresholds.
    scores = np.round(rng.normal(size=length), int(rng.integers(0, 5)))
    window = int(rng.integers(0, min(length, 11)))
    assert _assert_full_curve_parity(labels, scores, window, vendor) <= PARITY_ATOL


@pytest.mark.parametrize("seed", (101, 202, 303))
def test_exact_metric_wrapper_matches_literal_vendor_metrics(seed: int, vendor) -> None:
    rng = np.random.default_rng(seed)
    labels = (rng.random(127) < 0.25).astype(np.int8)
    labels[0] = 1
    labels[-1] = 0
    scores = np.round(rng.random(127), 3)
    literal = compute_threshold_free_metrics(
        scores,
        labels,
        8,
        vendor,
        thresholds=250,
    )
    exact = compute_threshold_free_metrics_exact_vus(
        scores,
        labels,
        8,
        vendor,
        thresholds=250,
    )
    assert exact.keys() == literal.keys()
    for name in exact:
        assert abs(exact[name] - literal[name]) <= PARITY_ATOL


@pytest.mark.parametrize(
    ("labels", "scores", "window", "thresholds", "message"),
    (
        (np.array([], dtype=np.int8), np.array([]), 1, 250, "non-empty"),
        (np.array([0, 0]), np.array([0.1, 0.2]), 1, 250, "anomaly"),
        (np.array([1, 1]), np.array([0.1, 0.2]), 1, 250, "normal"),
        (np.array([0, 2]), np.array([0.1, 0.2]), 1, 250, "binary"),
        (np.array([0.0, np.nan]), np.array([0.1, 0.2]), 1, 250, "binary"),
        (np.array([0, 1]), np.array([0.1, np.inf]), 1, 250, "finite"),
        (np.array([0, 1]), np.array([0.1]), 1, 250, "aligned"),
        (np.array([0, 1]), np.array([0.1, 0.2]), -1, 250, "non-negative"),
        (np.array([0, 1]), np.array([0.1, 0.2]), 1.5, 250, "integer"),
        (np.array([0, 1]), np.array([0.1, 0.2]), True, 250, "integer"),
        (np.array([0, 1]), np.array([0.1, 0.2]), 1, 249, "exactly 250"),
    ),
)
def test_exact_curve_fails_closed_on_unsupported_inputs(
    labels: np.ndarray,
    scores: np.ndarray,
    window: int,
    thresholds: int,
    message: str,
    vendor,
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_curve_exact(labels, scores, window, vendor, thresholds=thresholds)


def test_metric_wrapper_retains_positive_window_contract(vendor) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_threshold_free_metrics_exact_vus(
            np.array([0.1, 0.9]),
            np.array([0, 1], dtype=np.int8),
            0,
            vendor,
        )
