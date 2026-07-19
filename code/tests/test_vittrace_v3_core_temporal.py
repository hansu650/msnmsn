"""Temporal-operator and reducer invariants for the isolated v3 core."""

from __future__ import annotations

import numpy as np
import pytest

from measure_vit4ts_v3.core import (
    apply_temporal_operator,
    build_linear_nctp,
    build_nearest_full_column_operator,
    column_quantile,
    column_top_fraction_mean,
    validate_temporal_operator,
)


def test_linear_nctp_is_dynamic_float64_and_row_stochastic() -> None:
    operator = build_linear_nctp(5, (2, 3), image_size=(2, 3))
    expected = np.asarray(
        [
            [0.50, 0.00, 0.00, 0.50, 0.00, 0.00],
            [0.25, 0.25, 0.00, 0.25, 0.25, 0.00],
            [0.00, 0.50, 0.00, 0.00, 0.50, 0.00],
            [0.00, 0.25, 0.25, 0.00, 0.25, 0.25],
            [0.00, 0.00, 0.50, 0.00, 0.00, 0.50],
        ],
        dtype=np.float64,
    )

    assert operator.shape == (5, 6)
    assert operator.dtype == np.float64
    np.testing.assert_array_equal(operator, expected)
    np.testing.assert_allclose(operator.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)


def test_nearest_full_column_uses_half_up_not_bankers_rounding() -> None:
    coordinates = np.asarray([0.0, 0.5, 1.49, 1.5, 2.0])
    operator = build_nearest_full_column_operator(
        5,
        (2, 3),
        x_coordinates=coordinates,
        image_size=(2, 3),
    )
    chosen = np.argmax(operator.reshape(5, 2, 3).sum(axis=1), axis=1)

    np.testing.assert_array_equal(chosen, [0, 1, 1, 2, 2])
    np.testing.assert_array_equal(
        operator[1],
        np.asarray([0.0, 0.5, 0.0, 0.0, 0.5, 0.0]),
    )


def test_temporal_operator_preserves_constants_order_and_convex_bounds() -> None:
    operator = build_linear_nctp(7, (3, 4), image_size=(3, 4))
    constant = np.full((2, 12), 7.25, dtype=np.float32)
    projected = apply_temporal_operator(operator, constant)

    assert projected.shape == (2, 7)
    assert projected.dtype == np.float64
    np.testing.assert_allclose(projected, 7.25, rtol=0.0, atol=1e-15)

    # Every patch row carries the same column value, so NCTP must interpolate
    # monotonically between the first and final column without reordering W.
    column_ramp = np.tile(np.arange(4, dtype=np.float64), 3)
    ramp = apply_temporal_operator(operator, column_ramp)
    np.testing.assert_allclose(ramp, np.linspace(0.0, 3.0, 7), atol=1e-15)
    assert np.all(np.diff(ramp) >= 0.0)
    assert ramp.min() >= column_ramp.min() and ramp.max() <= column_ramp.max()


def test_temporal_operator_validation_fails_closed() -> None:
    valid = build_linear_nctp(3, (2, 2))
    validate_temporal_operator(valid, window_length=3, patch_grid=(2, 2))

    bad_sum = valid.copy()
    bad_sum[0, 0] += 0.1
    with pytest.raises(ValueError, match="row-stochastic"):
        validate_temporal_operator(bad_sum)
    negative = valid.copy()
    negative[0, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        validate_temporal_operator(negative)
    with pytest.raises(ValueError, match="outside"):
        build_linear_nctp(
            3,
            (2, 2),
            image_size=(2, 2),
            x_coordinates=[0.0, 0.5, 1.1],
        )


def test_literal_quantile_and_top_fraction_mean_remain_distinct() -> None:
    maps = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 10.0],
            [2.0, 20.0],
            [100.0, 30.0],
        ],
        dtype=np.float64,
    )
    quantile = column_quantile(maps, 0.75)
    top_fraction = column_top_fraction_mean(maps, 0.25)

    np.testing.assert_allclose(quantile, [26.5, 22.5], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(top_fraction, [100.0, 30.0], rtol=0.0, atol=0.0)
    assert not np.array_equal(quantile, top_fraction)

    batched = np.stack((maps, maps + 1.0))
    assert column_quantile(batched, 0.5).shape == (2, 2)
    assert column_top_fraction_mean(batched, 0.5).shape == (2, 2)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_quantile_rejects_invalid_probability(value: float) -> None:
    with pytest.raises(ValueError, match="q must lie"):
        column_quantile(np.ones((2, 2)), value)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, np.nan])
def test_top_fraction_rejects_invalid_fraction(value: float) -> None:
    with pytest.raises(ValueError, match="top_fraction"):
        column_top_fraction_mean(np.ones((2, 2)), value)
