"""Pure, label-free matching primitives for the temporal-topology K0.

The functions in this module operate only on frozen token caches.  They do
not load data, labels, checkpoints, or configuration files.  All matching
ties are resolved by the smallest flattened reference index; the monotone
dynamic program additionally resolves equal paths lexicographically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import torch
import torch.nn.functional as F


MatchMode = Literal["universal", "row", "position", "local3"]


@dataclass(frozen=True)
class MatchResult:
    """One selected reference and finite cost for every query token.

    Every field has shape ``[..., Q]``.  ``reference_index`` is the
    flattened reference-grid index and ``reference_column`` is its zero-based
    column.  ``valid_mask`` is explicit so downstream transactional code can
    fail closed if a future matcher ever leaves a query unmatched.
    """

    cost: torch.Tensor
    reference_index: torch.Tensor
    reference_column: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.cost.shape
        if self.cost.ndim < 1 or self.cost.shape[-1] == 0:
            raise ValueError("match costs must have shape [..., Q] with Q > 0")
        if (
            self.reference_index.shape != shape
            or self.reference_column.shape != shape
            or self.valid_mask.shape != shape
        ):
            raise ValueError("all MatchResult fields must have identical shapes")
        if not self.cost.is_floating_point() or not bool(torch.isfinite(self.cost).all()):
            raise ValueError("match costs must be finite floating-point values")
        if self.reference_index.dtype != torch.long:
            raise TypeError("reference_index must have dtype torch.long")
        if self.reference_column.dtype != torch.long:
            raise TypeError("reference_column must have dtype torch.long")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        if bool(self.valid_mask.any()):
            if bool((self.reference_index[self.valid_mask] < 0).any()):
                raise ValueError("valid reference indices cannot be negative")
            if bool((self.reference_column[self.valid_mask] < 0).any()):
                raise ValueError("valid reference columns cannot be negative")


@dataclass(frozen=True)
class ScaleScores:
    """Per-token anomaly costs at the frozen large, mid, and patch scales."""

    large: torch.Tensor
    mid: torch.Tensor
    patch: torch.Tensor


@dataclass(frozen=True)
class ScaleMasks:
    """Zero-based vendor pooling masks for the large and mid scales."""

    large: torch.Tensor
    mid: torch.Tensor


def _square_side(size: int, name: str) -> int:
    side = math.isqrt(int(size))
    if size <= 0 or side * side != size:
        raise ValueError(f"{name} token count must be a positive perfect square")
    return side


def _validate_cost_matrix(cost: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    if not isinstance(cost, torch.Tensor):
        raise TypeError("cost must be a torch.Tensor")
    if cost.ndim < 2 or not cost.is_floating_point():
        raise ValueError("cost must be a floating tensor with shape [..., Q, R]")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("cost must contain only finite values")
    queries, references = int(cost.shape[-2]), int(cost.shape[-1])
    query_side = _square_side(queries, "query")
    reference_side = _square_side(references, "reference")
    if query_side != reference_side:
        raise ValueError("query and reference grids must have the same side length")
    return cost, queries, references, query_side


def correct_incidence(mask: torch.Tensor, output_cells: int) -> torch.Tensor:
    """Convert a zero-based pooling mask to ``[output_cells, scale_tokens]``.

    The released masks have shape ``[cells_per_token, scale_tokens]`` and
    contain flattened zero-based output-cell indices.  This projector uses
    those indices literally, avoiding the released ``idx + 1`` shift.
    """

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("mask must be a non-empty [members, scale_tokens] tensor")
    if mask.dtype == torch.bool or mask.is_floating_point() or mask.is_complex():
        raise TypeError("mask must contain integer zero-based indices")
    cells = int(output_cells)
    if cells <= 0:
        raise ValueError("output_cells must be positive")
    indices = mask.to(dtype=torch.long)
    if bool((indices < 0).any()) or bool((indices >= cells).any()):
        raise ValueError("mask contains an index outside the output grid")
    targets = torch.arange(cells, device=indices.device, dtype=torch.long)
    incidence = (indices.unsqueeze(0) == targets[:, None, None]).any(dim=1)
    if incidence.shape != (cells, mask.shape[1]):
        raise RuntimeError("incidence projection returned the wrong shape")
    return incidence


def pairwise_cosine_cost(query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """Return frozen ViT4TS cosine costs ``0.5 * (1 - cosine)``.

    Inputs have shape ``[..., Q, D]`` and ``[..., R, D]``.  Leading batch
    dimensions follow ``torch.matmul`` broadcasting.  Zero vectors remain
    finite through ``torch.nn.functional.normalize``'s epsilon handling.
    """

    if not isinstance(query, torch.Tensor) or not isinstance(memory, torch.Tensor):
        raise TypeError("query and memory must be torch tensors")
    if query.ndim < 2 or memory.ndim < 2:
        raise ValueError("query and memory must have shape [..., tokens, features]")
    if not query.is_floating_point() or not memory.is_floating_point():
        raise TypeError("query and memory must be floating-point tensors")
    if query.shape[-1] != memory.shape[-1] or query.shape[-1] == 0:
        raise ValueError("query and memory feature dimensions must match and be non-empty")
    if query.shape[-2] == 0 or memory.shape[-2] == 0:
        raise ValueError("query and memory token dimensions must be non-empty")
    if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(memory).all()):
        raise ValueError("query and memory must contain only finite values")
    query_norm = F.normalize(query, dim=-1)
    memory_norm = F.normalize(memory, dim=-1)
    similarity = torch.matmul(query_norm, memory_norm.transpose(-1, -2)).clamp(-1.0, 1.0)
    result = 0.5 * (1.0 - similarity)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("pairwise cosine cost produced a non-finite value")
    return result


def _local3_descriptor_cost(cost: torch.Tensor, side: int) -> torch.Tensor:
    """Average aligned three-column costs for every pair of token centres."""

    queries = references = side * side
    device = cost.device
    query_indices = torch.arange(queries, device=device, dtype=torch.long)
    reference_indices = torch.arange(references, device=device, dtype=torch.long)
    query_columns = query_indices.remainder(side)
    reference_columns = reference_indices.remainder(side)
    total = torch.zeros_like(cost)
    count = torch.zeros((queries, references), dtype=torch.long, device=device)

    for offset in (-1, 0, 1):
        query_valid = (query_columns + offset >= 0) & (query_columns + offset < side)
        reference_valid = (
            (reference_columns + offset >= 0)
            & (reference_columns + offset < side)
        )
        shifted_query = query_indices + offset
        shifted_reference = reference_indices + offset
        # Invalid positions are never used after masking; clamping keeps the
        # gather itself in bounds without introducing a boundary candidate.
        shifted_query = shifted_query.clamp(0, queries - 1)
        shifted_reference = shifted_reference.clamp(0, references - 1)
        shifted = cost.index_select(-2, shifted_query).index_select(
            -1, shifted_reference
        )
        joint = query_valid[:, None] & reference_valid[None, :]
        total = total + torch.where(joint, shifted, torch.zeros_like(shifted))
        count = count + joint.to(dtype=torch.long)

    if bool((count == 0).any()):  # pragma: no cover - offset zero is always valid
        raise RuntimeError("local3 descriptor has an empty aligned offset set")
    return total / count.to(dtype=total.dtype)


def independent_match(cost: torch.Tensor, mode: MatchMode) -> MatchResult:
    """Select an independent reference under one frozen candidate rule.

    ``row`` restricts candidates to the same image row. ``position`` uses the
    identical flattened coordinate. ``local3`` compares every pair of token
    centres using the mean of aligned offsets ``{-1, 0, 1}`` that exist in
    both sequences, then minimizes independently over reference centres.
    """

    values, queries, references, side = _validate_cost_matrix(cost)
    if mode not in {"universal", "row", "position", "local3"}:
        raise ValueError(f"unsupported independent match mode {mode!r}")

    if mode == "universal":
        selected_cost, selected_index = torch.min(values, dim=-1)
    elif mode == "position":
        selected_index = torch.arange(
            queries, device=values.device, dtype=torch.long
        ).expand(values.shape[:-2] + (queries,))
        selected_cost = torch.gather(values, -1, selected_index.unsqueeze(-1)).squeeze(-1)
    elif mode == "row":
        flattened = values.reshape((-1, queries, references))
        batch = flattened.shape[0]
        selected_cost_flat = torch.empty(
            (batch, queries), dtype=values.dtype, device=values.device
        )
        selected_index_flat = torch.empty(
            (batch, queries), dtype=torch.long, device=values.device
        )
        for row in range(side):
            query_slice = slice(row * side, (row + 1) * side)
            reference_start = row * side
            candidates = flattened[:, query_slice, reference_start : reference_start + side]
            row_cost, row_offset = torch.min(candidates, dim=-1)
            selected_cost_flat[:, query_slice] = row_cost
            selected_index_flat[:, query_slice] = row_offset + reference_start
        selected_cost = selected_cost_flat.reshape(values.shape[:-2] + (queries,))
        selected_index = selected_index_flat.reshape(values.shape[:-2] + (queries,))
    else:
        descriptor_cost = _local3_descriptor_cost(values, side)
        selected_cost, selected_index = torch.min(descriptor_cost, dim=-1)

    selected_index = selected_index.to(dtype=torch.long)
    selected_column = selected_index.remainder(side)
    valid = torch.ones_like(selected_index, dtype=torch.bool)
    return MatchResult(selected_cost, selected_index, selected_column, valid)


def monotone_column_match(
    cost: torch.Tensor,
    side: int,
    steps: tuple[int, ...] = (0, 1, 2),
) -> MatchResult:
    """Match one open-begin/open-end monotone reference-column path.

    For each query row and query/reference column pair, the cheapest vertical
    reference token is retained.  Dynamic programming then selects a separate
    reference-column path for every query row.  Each path may start and end at
    any reference column, uses only the supplied non-negative increments, and
    has no transition cost.  Exact ties choose the lexicographically smallest
    complete column path and then the smallest flattened reference index.
    """

    values, queries, references, inferred_side = _validate_cost_matrix(cost)
    frozen_side = int(side)
    if frozen_side != inferred_side:
        raise ValueError("side does not match the square cost matrix")
    allowed = tuple(sorted(set(int(step) for step in steps)))
    if not allowed or any(step < 0 for step in allowed):
        raise ValueError("steps must be a non-empty set of non-negative increments")
    # Fixed-length base-(side+1) codes preserve lexicographic path order.
    base = frozen_side + 1
    if base**frozen_side > torch.iinfo(torch.long).max:
        raise ValueError("side is too large for deterministic path tie codes")

    leading = values.shape[:-2]
    flat = values.reshape((-1, queries, references))
    batch = flat.shape[0]
    device = values.device
    query_rows = torch.arange(frozen_side, device=device, dtype=torch.long)
    reference_rows = torch.arange(frozen_side, device=device, dtype=torch.long)
    best_cost = torch.empty(
        (batch, frozen_side, frozen_side, frozen_side),
        dtype=values.dtype,
        device=device,
    )
    best_reference = torch.empty(
        (batch, frozen_side, frozen_side, frozen_side),
        dtype=torch.long,
        device=device,
    )

    for query_column in range(frozen_side):
        query_tokens = query_rows * frozen_side + query_column
        query_block = flat.index_select(1, query_tokens)
        for reference_column in range(frozen_side):
            reference_tokens = reference_rows * frozen_side + reference_column
            block = query_block.index_select(2, reference_tokens)
            local_cost, local_row = torch.min(block, dim=-1)
            best_cost[:, :, query_column, reference_column] = local_cost
            best_reference[:, :, query_column, reference_column] = reference_tokens[
                local_row
            ]

    # Each query row is an independent temporal sequence.  Keeping it in the
    # DP batch dimension avoids a high-cost background row overriding the
    # correspondence preferred by an evidence-bearing row.
    sequence_count = batch * frozen_side
    unary = best_cost.reshape(sequence_count, frozen_side, frozen_side)

    columns = torch.arange(frozen_side, device=device, dtype=torch.long)
    dp = unary[:, 0, :].clone()
    path_code = columns.unsqueeze(0).expand(sequence_count, -1).clone() + 1
    backpointers: list[torch.Tensor | None] = [None]
    max_code = torch.iinfo(torch.long).max

    for query_column in range(1, frozen_side):
        next_dp = torch.empty_like(dp)
        next_code = torch.empty_like(path_code)
        predecessor_table = torch.empty_like(path_code)
        for reference_column in range(frozen_side):
            predecessors = [
                reference_column - step
                for step in allowed
                if reference_column - step >= 0
            ]
            if not predecessors:
                next_dp[:, reference_column] = torch.inf
                next_code[:, reference_column] = max_code
                predecessor_table[:, reference_column] = -1
                continue
            predecessor_tensor = torch.tensor(
                predecessors, device=device, dtype=torch.long
            )
            candidate_value = dp.index_select(1, predecessor_tensor)
            minimum = candidate_value.min(dim=1).values
            candidate_code = (
                path_code.index_select(1, predecessor_tensor) * base
                + reference_column
                + 1
            )
            tied_code = torch.where(
                candidate_value == minimum[:, None],
                candidate_code,
                torch.full_like(candidate_code, max_code),
            )
            chosen_slot = torch.argmin(tied_code, dim=1)
            chosen_predecessor = predecessor_tensor[chosen_slot]
            next_dp[:, reference_column] = (
                minimum + unary[:, query_column, reference_column]
            )
            next_code[:, reference_column] = torch.gather(
                candidate_code, 1, chosen_slot[:, None]
            ).squeeze(1)
            predecessor_table[:, reference_column] = chosen_predecessor
        dp = next_dp
        path_code = next_code
        backpointers.append(predecessor_table)

    terminal_value = dp.min(dim=1).values
    terminal_code = torch.where(
        dp == terminal_value[:, None],
        path_code,
        torch.full_like(path_code, max_code),
    )
    current = torch.argmin(terminal_code, dim=1)
    path = torch.empty(
        (sequence_count, frozen_side), dtype=torch.long, device=device
    )
    path[:, -1] = current
    for query_column in range(frozen_side - 1, 0, -1):
        predecessor_table = backpointers[query_column]
        if predecessor_table is None:  # pragma: no cover - guarded by loop bounds
            raise RuntimeError("missing monotone backpointer")
        current = torch.gather(predecessor_table, 1, current[:, None]).squeeze(1)
        if bool((current < 0).any()):
            raise RuntimeError("monotone dynamic program has no valid open path")
        path[:, query_column - 1] = current

    path = path.reshape(batch, frozen_side, frozen_side)
    output_cost = torch.empty(
        (batch, frozen_side, frozen_side), dtype=values.dtype, device=device
    )
    output_reference = torch.empty(
        (batch, frozen_side, frozen_side), dtype=torch.long, device=device
    )
    for query_column in range(frozen_side):
        chosen_column = path[:, :, query_column]
        output_cost[:, :, query_column] = torch.gather(
            best_cost[:, :, query_column, :], 2, chosen_column.unsqueeze(-1)
        ).squeeze(-1)
        output_reference[:, :, query_column] = torch.gather(
            best_reference[:, :, query_column, :],
            2,
            chosen_column.unsqueeze(-1),
        ).squeeze(-1)

    result_cost = output_cost.reshape(leading + (queries,))
    result_reference = output_reference.reshape(leading + (queries,))
    result_column = result_reference.remainder(frozen_side)
    valid = torch.ones_like(result_reference, dtype=torch.bool)
    return MatchResult(result_cost, result_reference, result_column, valid)


def _released_incidence(mask: torch.Tensor, output_cells: int) -> torch.Tensor:
    """Reproduce the released ``idx + 1`` membership shift."""

    if mask.ndim != 2 or mask.dtype == torch.bool or mask.is_floating_point():
        raise ValueError("released masks must be two-dimensional integer tensors")
    indices = mask.to(dtype=torch.long)
    targets = torch.arange(
        1, int(output_cells) + 1, device=indices.device, dtype=torch.long
    )
    return (indices.unsqueeze(0) == targets[:, None, None]).any(dim=1)


def _harmonic_project(
    score: torch.Tensor, incidence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    values = score.double()
    if values.ndim < 2 or incidence.ndim != 2:
        raise ValueError("scale score and incidence must be [..., tokens] and [cells, tokens]")
    if values.shape[-1] != incidence.shape[-1]:
        raise ValueError("scale score token count does not match its incidence mask")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("scale scores must be finite and non-negative")
    membership = incidence.to(device=values.device, dtype=torch.bool)
    expanded_values = values.unsqueeze(-2)
    expanded_membership = membership.reshape(
        (1,) * (values.ndim - 1) + membership.shape
    )
    reciprocal = torch.where(
        expanded_values > 0,
        expanded_values.reciprocal(),
        torch.full_like(expanded_values, torch.inf),
    )
    denominator = torch.where(
        expanded_membership, reciprocal, torch.zeros_like(reciprocal)
    ).sum(dim=-1)
    count = membership.sum(dim=-1).to(dtype=values.dtype)
    count = count.reshape((1,) * (values.ndim - 1) + (membership.shape[0],))
    valid = membership.any(dim=-1)
    projected = torch.where(
        (count > 0) & torch.isfinite(denominator) & (denominator > 0),
        count / denominator,
        torch.zeros_like(denominator),
    )
    if not bool(torch.isfinite(projected).all()):
        raise RuntimeError("harmonic projection produced a non-finite value")
    return projected, valid


def aggregate_multiscale(
    scores: ScaleScores,
    masks: ScaleMasks,
    released_indexing: bool,
) -> torch.Tensor:
    """Harmonically project and equally fuse the frozen three token scales.

    The corrected branch interprets masks as zero-based indices.  The released
    branch reproduces the vendor's one-based membership query and its final
    uncovered output cell, which becomes zero after the vendor's finite-value
    cleanup.  The returned tensor is finite ``[B, side, side]`` (or with any
    additional leading batch dimensions preserved).
    """

    patch = scores.patch
    if not isinstance(patch, torch.Tensor) or patch.ndim < 2:
        raise ValueError("patch scores must have shape [..., output_cells]")
    output_cells = int(patch.shape[-1])
    side = _square_side(output_cells, "patch")
    if not patch.is_floating_point() or not bool(torch.isfinite(patch).all()):
        raise ValueError("patch scores must be finite floating-point values")
    if bool((patch < 0).any()):
        raise ValueError("patch scores must be non-negative")
    if scores.large.shape[:-1] != patch.shape[:-1] or scores.mid.shape[:-1] != patch.shape[:-1]:
        raise ValueError("all scale scores must share leading batch dimensions")
    if masks.large.device != patch.device or masks.mid.device != patch.device:
        raise ValueError("scale masks and scores must be on the same device")

    if bool(released_indexing):
        large_incidence = _released_incidence(masks.large, output_cells)
        mid_incidence = _released_incidence(masks.mid, output_cells)
    else:
        large_incidence = correct_incidence(masks.large, output_cells)
        mid_incidence = correct_incidence(masks.mid, output_cells)
    large_map, large_valid = _harmonic_project(scores.large, large_incidence)
    mid_map, mid_valid = _harmonic_project(scores.mid, mid_incidence)
    fused = (large_map + mid_map + patch.double()) / 3.0
    if bool(released_indexing):
        valid = (large_valid & mid_valid).reshape(
            (1,) * (fused.ndim - 1) + (output_cells,)
        )
        fused = torch.where(valid, fused, torch.zeros_like(fused))
    fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
    if not bool(torch.isfinite(fused).all()):
        raise RuntimeError("multiscale aggregation produced a non-finite value")
    return fused.reshape(patch.shape[:-1] + (side, side))


def correspondence_diagnostics(result: MatchResult) -> Mapping[str, float]:
    """Return compact deterministic, label-free correspondence statistics."""

    queries = int(result.cost.shape[-1])
    side = _square_side(queries, "query")
    valid = result.valid_mask
    valid_fraction = float(valid.double().mean().detach().cpu())
    if not bool(valid.any()):
        return {
            "valid_fraction": valid_fraction,
            "mean_cost": 0.0,
            "max_cost": 0.0,
            "mean_absolute_column_displacement": 0.0,
            "unique_reference_fraction": 0.0,
        }

    selected_cost = result.cost[valid].double()
    query_column = torch.arange(
        queries, device=result.cost.device, dtype=torch.long
    ).remainder(side)
    query_column = query_column.expand(result.cost.shape)
    displacement = torch.abs(result.reference_column - query_column)[valid].double()
    flattened_reference = result.reference_index.reshape(-1, queries)
    flattened_valid = valid.reshape(-1, queries)
    unique_fractions: list[float] = []
    for references, row_valid in zip(flattened_reference, flattened_valid):
        count = int(row_valid.sum().item())
        fraction = (
            float(torch.unique(references[row_valid]).numel()) / float(count)
            if count > 0
            else 0.0
        )
        unique_fractions.append(fraction)
    output = {
        "valid_fraction": valid_fraction,
        "mean_cost": float(selected_cost.mean().detach().cpu()),
        "max_cost": float(selected_cost.max().detach().cpu()),
        "mean_absolute_column_displacement": float(
            displacement.mean().detach().cpu()
        ),
        "unique_reference_fraction": float(
            sum(unique_fractions) / len(unique_fractions)
        ),
    }
    if not all(math.isfinite(value) for value in output.values()):
        raise RuntimeError("correspondence diagnostics produced a non-finite value")
    return output
