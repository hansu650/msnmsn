from __future__ import annotations

import torch
import numpy as np

from measure_vit4ts.ihp import (
    harmonic_projection,
    incidence_certificate,
    index_consistent_harmonic_projection,
    literal_incidence,
)
from measure_vit4ts.ihp_cache import score_token_cache, universal_token_cost


def _mask(side: int, block: int) -> torch.Tensor:
    output_side = 14
    columns = []
    for row in range(side):
        for column in range(side):
            start_row = min(row, output_side - block)
            start_column = min(column, output_side - block)
            indices = [
                (start_row + local_row) * output_side + start_column + local_column
                for local_row in range(block)
                for local_column in range(block)
            ]
            columns.append(indices)
    return torch.tensor(columns, dtype=torch.long).T.contiguous()


def test_literal_incidence_covers_zero_and_terminal_cells() -> None:
    mask = _mask(13, 2)
    incidence = literal_incidence(mask, 196)
    assert incidence.shape == (196, 169)
    assert bool(incidence[0].any())
    assert bool(incidence[-1].any())
    assert bool(incidence.any(dim=1).all())


def test_certificate_exposes_shifted_boundary_and_terminal_failures() -> None:
    certificate = incidence_certificate(_mask(13, 2), 196)
    assert certificate == {
        "output_cells": 196,
        "literal_supported_cells": 196,
        "shifted_supported_cells": 195,
        "shifted_row_boundary_aliases": 13,
        "shifted_terminal_holes": 1,
    }


def test_zero_token_score_has_zero_harmonic_projection() -> None:
    incidence = torch.tensor([[True, True], [False, True]])
    projected, valid = harmonic_projection(
        torch.tensor([[0.0, 2.0]], dtype=torch.float64), incidence
    )
    assert torch.equal(valid, torch.tensor([True, True]))
    assert torch.equal(projected, torch.tensor([[0.0, 2.0]], dtype=torch.float64))


def test_multiscale_projection_matches_manual_harmonic_values() -> None:
    large = torch.tensor([[2.0, 8.0]], dtype=torch.float64)
    mid = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float64)
    patch = torch.tensor([[3.0, 3.0, 3.0, 3.0]], dtype=torch.float64)
    large_mask = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    mid_mask = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    actual = index_consistent_harmonic_projection(
        large, mid, patch, large_mask, mid_mask
    )
    expected = torch.tensor(
        [[[2.0, 7.0 / 3.0], [5.0, 19.0 / 3.0]]], dtype=torch.float64
    )
    assert torch.equal(actual, expected)


def test_universal_cost_is_zero_for_identical_unit_tokens() -> None:
    tokens = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float64,
    )
    assert torch.equal(universal_token_cost(tokens), torch.zeros((2, 2), dtype=torch.float64))


def test_cache_scoring_is_label_free_and_finite(tmp_path) -> None:
    generator = np.random.default_rng(2027)
    cache = tmp_path / "tokens.npz"
    np.savez_compressed(
        cache,
        large_tokens=generator.normal(size=(2, 144, 8)).astype(np.float32),
        mid_tokens=generator.normal(size=(2, 169, 8)).astype(np.float32),
        patch_tokens=generator.normal(size=(2, 196, 8)).astype(np.float32),
        large_mask=_mask(12, 3).numpy(),
        mid_mask=_mask(13, 2).numpy(),
    )
    maps, columns = score_token_cache(cache)
    assert maps.shape == (2, 14, 14)
    assert columns.shape == (2, 224)
    assert np.isfinite(maps).all()
    assert np.isfinite(columns).all()
