"""Frozen and dynamic pixel-to-patch NCTP parity tests."""

from __future__ import annotations

import numpy as np
import pytest

from measure_vit4ts.ritp import build_full_column_operator
from measure_vit4ts_v3.core import (
    build_linear_nctp,
    build_nearest_full_column_operator,
)


def test_default_b16_w240_linear_nctp_matches_frozen_full_column() -> None:
    frozen = build_full_column_operator()
    actual = build_linear_nctp(240, (14, 14), image_size=(224, 224))

    assert actual.shape == (240, 196)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, frozen, rtol=0.0, atol=1e-12)
    assert float(np.max(np.abs(actual - frozen))) < 1e-12


def test_default_image_size_is_the_frozen_224_frame() -> None:
    explicit = build_linear_nctp(240, (14, 14), image_size=(224, 224))
    implicit = build_linear_nctp(240, (14, 14))
    np.testing.assert_array_equal(implicit, explicit)


def test_rectangular_nondivisible_image_uses_deterministic_pixel_cells() -> None:
    coordinates = np.asarray([0.0, 2.5, 6.0], dtype=np.float64)
    linear = build_linear_nctp(
        3,
        (2, 3),
        image_size=(5, 7),
        x_coordinates=coordinates,
    )
    nearest = build_nearest_full_column_operator(
        3,
        (2, 3),
        image_size=(5, 7),
        x_coordinates=coordinates,
    )

    expected_first = np.asarray([[0.6, 0.0, 0.0], [0.4, 0.0, 0.0]])
    expected_last = np.asarray([[0.0, 0.0, 0.6], [0.0, 0.0, 0.4]])
    np.testing.assert_allclose(linear[0].reshape(2, 3), expected_first, atol=1e-15)
    np.testing.assert_allclose(linear[-1].reshape(2, 3), expected_last, atol=1e-15)
    # Pixel 2 belongs to patch column 0 and pixel 3 to patch column 1.
    np.testing.assert_allclose(
        linear[1].reshape(2, 3),
        np.asarray([[0.3, 0.3, 0.0], [0.2, 0.2, 0.0]]),
        atol=1e-15,
    )
    # Half-up pixel rounding sends x=2.5 to pixel 3, hence patch column 1.
    np.testing.assert_allclose(
        nearest[1].reshape(2, 3),
        np.asarray([[0.0, 0.6, 0.0], [0.0, 0.4, 0.0]]),
        atol=1e-15,
    )
    np.testing.assert_allclose(linear.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(nearest.sum(axis=1), 1.0, atol=1e-15)


def test_patch_grid_cannot_exceed_actual_image_pixels() -> None:
    with pytest.raises(ValueError, match="more cells than image pixels"):
        build_linear_nctp(4, (6, 3), image_size=(5, 7))
