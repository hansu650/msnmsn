"""Float32 reference-order parity for v3 streamed matchers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from measure_vit4ts_v3.core import (
    FLOAT32_MATCH_CHUNK_ATOL,
    build_candidate_mask,
    streamed_median_reference_match,
)


def _full_float32_median_reference(
    tokens: torch.Tensor,
    allowed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = F.normalize(tokens, dim=-1)
    memory = F.normalize(torch.median(tokens, dim=0).values, dim=-1)
    similarity = torch.matmul(query, memory.T).clamp(-1.0, 1.0)
    cost = (0.5 * (1.0 - similarity)).masked_fill(~allowed.unsqueeze(0), torch.inf)
    selected_cost, selected_index = torch.min(cost, dim=-1)
    return selected_cost.double(), selected_index.long()


def test_full_chunk_is_exact_float32_reference_and_output_is_float64() -> None:
    generator = torch.Generator().manual_seed(650)
    tokens = torch.randn((7, 12, 19), generator=generator, dtype=torch.float32)
    allowed = build_candidate_mask((3, 4), "global")
    expected_cost, expected_index = _full_float32_median_reference(tokens, allowed)

    actual = streamed_median_reference_match(
        tokens,
        allowed,
        query_chunk_size=tokens.shape[0],
    )

    torch.testing.assert_close(actual.cost, expected_cost, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.reference_index, expected_index)
    assert actual.cost.dtype == torch.float64


def test_float32_query_chunk_drift_stays_within_documented_tolerance() -> None:
    generator = torch.Generator().manual_seed(651)
    tokens = torch.randn((9, 12, 37), generator=generator, dtype=torch.float32)
    allowed = build_candidate_mask((3, 4), "row")
    full_cost, full_index = _full_float32_median_reference(tokens, allowed)

    streamed = streamed_median_reference_match(tokens, allowed, query_chunk_size=2)

    torch.testing.assert_close(
        streamed.cost,
        full_cost,
        rtol=0.0,
        atol=FLOAT32_MATCH_CHUNK_ATOL,
    )
    torch.testing.assert_close(streamed.reference_index, full_index)
