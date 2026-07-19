"""Focused dynamic-grid and multiscale tests for the isolated v3 core."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from measure_vit4ts_v3.core import (
    build_candidate_mask,
    fuse_scale_subset,
    harmonic_incidence_projection,
    literal_incidence,
    patch_count,
    released_incidence,
    validate_patch_grid,
    validate_pooling_mask,
)


def test_patch_grid_and_k_are_dynamic_and_rectangular() -> None:
    assert validate_patch_grid((3, 5)) == (3, 5)
    assert patch_count((3, 5)) == 15
    assert patch_count([7, 4]) == 28  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="grid_h"):
        patch_count((True, 4))
    with pytest.raises(ValueError, match="positive"):
        patch_count((3, 0))
    with pytest.raises(TypeError, match="two-item"):
        patch_count((3,))  # type: ignore[arg-type]


def test_pooling_mask_validation_checks_dtype_bounds_duplicates_and_coverage() -> None:
    valid = torch.tensor([[0, 1, 3], [1, 2, 4], [2, 4, 5]], dtype=torch.int64)
    actual = validate_pooling_mask(valid, (2, 3), require_full_coverage=True)
    torch.testing.assert_close(actual, valid)

    with pytest.raises(TypeError, match="integer"):
        validate_pooling_mask(valid.float(), (2, 3))
    with pytest.raises(ValueError, match="outside"):
        validate_pooling_mask(torch.tensor([[0, 6]]), (2, 3))
    with pytest.raises(ValueError, match="repeats"):
        validate_pooling_mask(torch.tensor([[0, 1], [0, 2]]), (2, 3))
    with pytest.raises(ValueError, match="cover every"):
        validate_pooling_mask(
            np.asarray([[0, 1], [2, 3]], dtype=np.int32),
            (2, 3),
            require_full_coverage=True,
        )


def test_candidate_masks_use_coordinates_without_row_major_leakage() -> None:
    grid = (2, 3)
    position = build_candidate_mask(grid, "position")
    row = build_candidate_mask(grid, "row")
    column = build_candidate_mask(grid, "column")
    global_mask = build_candidate_mask(grid, "global")

    torch.testing.assert_close(position, torch.eye(6, dtype=torch.bool))
    torch.testing.assert_close(row[2], torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool))
    # Flattened cells 2 and 3 are adjacent but lie on different rows; a
    # flattened-interval implementation would leak here.
    assert not bool(row[2, 3])
    torch.testing.assert_close(
        column[2], torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.bool)
    )
    assert bool(global_mask.all())
    assert torch.equal(row, row.T)
    assert torch.equal(column, column.T)


def test_literal_and_released_incidence_have_exact_dynamic_shift() -> None:
    grid = (2, 3)
    identity_membership = torch.arange(6, dtype=torch.int64).reshape(1, 6)
    literal = literal_incidence(
        identity_membership,
        grid,
        require_full_coverage=True,
    )
    released = released_incidence(identity_membership, grid)

    torch.testing.assert_close(literal, torch.eye(6, dtype=torch.bool))
    expected_released = torch.zeros((6, 6), dtype=torch.bool)
    expected_released[torch.arange(5), torch.arange(1, 6)] = True
    torch.testing.assert_close(released, expected_released)
    assert bool(literal.any(dim=1).all())
    assert not bool(released[-1].any())
    assert not bool(released[:, 0].any())


def test_harmonic_projection_preserves_validity_and_zero_extension() -> None:
    incidence = torch.tensor(
        [
            [True, True, False],
            [False, True, False],
            [False, False, False],
        ]
    )
    scores = torch.tensor([[2.0, 4.0, 9.0], [0.0, 4.0, 9.0]], dtype=torch.float32)
    projected, valid = harmonic_incidence_projection(scores, incidence)

    assert projected.dtype == torch.float64
    torch.testing.assert_close(valid, torch.tensor([True, True, False]))
    torch.testing.assert_close(
        projected,
        torch.tensor([[8.0 / 3.0, 4.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float64),
    )


def test_scale_subset_fusion_divides_by_cellwise_active_count() -> None:
    fields = {
        "P": torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32),
        "M": torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32),
        "L": torch.tensor([[100.0, 200.0, 300.0]], dtype=torch.float32),
    }
    validity = {
        "P": torch.tensor([True, True, True]),
        "M": torch.tensor([True, False, True]),
        "L": torch.tensor([False, False, True]),
    }

    pm = fuse_scale_subset(fields, "PM", valid_masks=validity)
    pml = fuse_scale_subset(fields, ("P", "M", "L"), valid_masks=validity)
    m_only = fuse_scale_subset(fields, "M", valid_masks=validity)

    assert pm.dtype == torch.float64
    torch.testing.assert_close(pm, torch.tensor([[6.0, 4.0, 18.0]], dtype=torch.float64))
    torch.testing.assert_close(
        pml,
        torch.tensor([[6.0, 4.0, 112.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        m_only,
        torch.tensor([[10.0, 0.0, 30.0]], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="duplicates"):
        fuse_scale_subset(fields, ("P", "P"))
