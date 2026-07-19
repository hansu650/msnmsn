"""CPU-only streamed matcher tests for the isolated v3 core."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

import measure_vit4ts_v3.core as core
from measure_vit4ts_v3.core import (
    FLOAT32_MATCH_CHUNK_ATOL,
    build_candidate_mask,
    streamed_all_pairs_median_match,
    streamed_median_reference_match,
)


def _dense_all_pairs(tokens: torch.Tensor, allowed: torch.Tensor):
    normalized = F.normalize(tokens, dim=-1)
    similarity = torch.einsum("nqd,mrd->nmqr", normalized, normalized).clamp(-1.0, 1.0)
    pair = (0.5 * (1.0 - similarity)).masked_fill(
        ~allowed.reshape(1, 1, allowed.shape[0], allowed.shape[1]),
        torch.inf,
    )
    per_window_cost, per_window_index = torch.min(pair, dim=-1)
    order = torch.argsort(per_window_cost, dim=1, stable=True)
    rank = (tokens.shape[0] - 1) // 2
    ranked_window = order[:, rank, :]
    selected_cost = torch.gather(
        per_window_cost, 1, ranked_window.unsqueeze(1)
    ).squeeze(1)
    chosen_window = torch.argmax(
        (per_window_cost == selected_cost.unsqueeze(1)).to(dtype=torch.int64),
        dim=1,
    )
    selected_index = torch.gather(
        per_window_index, 1, chosen_window.unsqueeze(1)
    ).squeeze(1)
    return selected_cost.double(), selected_index.long(), chosen_window.long()


@pytest.mark.parametrize("mode", ["position", "row", "column", "global"])
def test_streamed_all_pairs_matches_dense_oracle_for_every_scope(mode: str) -> None:
    generator = torch.Generator().manual_seed(2027)
    tokens = torch.randn((5, 6, 7), generator=generator, dtype=torch.float32)
    allowed = build_candidate_mask((2, 3), mode)  # type: ignore[arg-type]
    expected = _dense_all_pairs(tokens, allowed)

    actual = streamed_all_pairs_median_match(
        tokens,
        allowed,
        query_chunk_size=2,
        reference_chunk_size=3,
    )

    torch.testing.assert_close(actual.cost, expected[0], rtol=0.0, atol=FLOAT32_MATCH_CHUNK_ATOL)
    torch.testing.assert_close(actual.reference_index, expected[1])
    torch.testing.assert_close(actual.reference_window, expected[2])
    assert actual.cost.dtype == torch.float64
    assert bool(actual.valid_mask.all())


def test_all_pairs_chunk_sizes_cannot_change_values_or_output_order() -> None:
    generator = torch.Generator().manual_seed(9)
    tokens = torch.randn((6, 4, 5), generator=generator)
    allowed = build_candidate_mask((2, 2), "global")

    one = streamed_all_pairs_median_match(
        tokens, allowed, query_chunk_size=1, reference_chunk_size=1
    )
    wide = streamed_all_pairs_median_match(
        tokens, allowed, query_chunk_size=99, reference_chunk_size=99
    )

    torch.testing.assert_close(one.cost, wide.cost, rtol=0.0, atol=FLOAT32_MATCH_CHUNK_ATOL)
    torch.testing.assert_close(one.reference_index, wide.reference_index)
    torch.testing.assert_close(one.reference_window, wide.reference_window)


def test_all_pairs_uses_lower_median_and_lowest_window_on_exact_ties() -> None:
    # K=1 leaves only the reference-window reduction.  The four cosine costs
    # for query window zero are [0, 0, 1, 1], whose lower median is 0 and whose
    # stable tie representative is reference window zero.
    tokens = torch.tensor(
        [[[1.0, 0.0]], [[1.0, 0.0]], [[-1.0, 0.0]], [[-1.0, 0.0]]]
    )
    result = streamed_all_pairs_median_match(
        tokens,
        torch.ones((1, 1), dtype=torch.bool),
        query_chunk_size=2,
        reference_chunk_size=2,
    )

    assert result.cost[0, 0] == 0.0
    assert result.reference_window[0, 0] == 0
    assert result.reference_index[0, 0] == 0


def test_median_reference_match_respects_row_major_scope_and_tie_order() -> None:
    # Every vector is identical, so all candidate costs tie.  torch.min must
    # choose the lowest allowed flattened reference index for each query.
    tokens = torch.ones((3, 6, 2), dtype=torch.float32)
    row = streamed_median_reference_match(
        tokens,
        build_candidate_mask((2, 3), "row"),
        query_chunk_size=2,
    )
    column = streamed_median_reference_match(
        tokens,
        build_candidate_mask((2, 3), "column"),
        query_chunk_size=2,
    )

    torch.testing.assert_close(row.reference_index[0], torch.tensor([0, 0, 0, 3, 3, 3]))
    torch.testing.assert_close(
        column.reference_index[0], torch.tensor([0, 1, 2, 0, 1, 2])
    )
    assert bool((row.reference_window == -1).all())


def test_matcher_rejects_empty_candidate_rows_and_nonfinite_tokens() -> None:
    tokens = torch.ones((2, 4, 3), dtype=torch.float32)
    invalid = torch.eye(4, dtype=torch.bool)
    invalid[2] = False
    with pytest.raises(ValueError, match="at least one candidate"):
        streamed_all_pairs_median_match(tokens, invalid)

    tokens[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        streamed_median_reference_match(tokens, torch.ones((4, 4), dtype=torch.bool))


def test_core_is_isolated_from_frozen_scorers_and_external_io() -> None:
    source = inspect.getsource(core)
    assert "from measure_vit4ts" not in source
    assert "import measure_vit4ts" not in source
    assert "open_clip" not in source
    assert "Path(" not in source
