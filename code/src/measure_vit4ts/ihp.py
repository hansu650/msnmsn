"""Index-Consistent Harmonic Projection for frozen visual TSAD scores.

The released VLM4TS pooling masks store zero-based output-cell indices.
IHP interprets those indices literally, projects each coarse token scale to
the base grid with a validity-normalized harmonic mean, and then averages the
two projected maps with the native base-grid scores.  The module is label
free, parameter free, and independent of the experimental runner.
"""

from __future__ import annotations

import math

import torch


def literal_incidence(mask: torch.Tensor, output_cells: int) -> torch.Tensor:
    """Return the literal ``[output_cells, scale_tokens]`` membership matrix.

    ``mask`` has shape ``[members_per_token, scale_tokens]`` and contains
    flattened, zero-based indices into the output grid.
    """

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("mask must be a non-empty [members, tokens] tensor")
    if mask.dtype == torch.bool or mask.is_floating_point() or mask.is_complex():
        raise TypeError("mask must contain integer zero-based indices")
    cells = int(output_cells)
    if cells <= 0:
        raise ValueError("output_cells must be positive")
    indices = mask.to(dtype=torch.long)
    if bool((indices < 0).any()) or bool((indices >= cells).any()):
        raise ValueError("mask contains an index outside the output grid")
    targets = torch.arange(cells, device=indices.device, dtype=torch.long)
    return (indices.unsqueeze(0) == targets[:, None, None]).any(dim=1)


def incidence_certificate(mask: torch.Tensor, output_cells: int) -> dict[str, int]:
    """Certify literal coverage and characterize the released index shift.

    This label-free diagnostic is not part of inference.  It compares literal
    membership against the released one-based query convention and reports
    the number of supported cells, row-boundary coordinate crossings, and
    uncovered terminal cells.
    """

    literal = literal_incidence(mask, output_cells)
    cells = int(output_cells)
    side = math.isqrt(cells)
    if side * side != cells:
        raise ValueError("output_cells must form a square grid")
    indices = mask.to(dtype=torch.long)
    shifted_targets = torch.arange(
        1, cells + 1, device=indices.device, dtype=torch.long
    )
    shifted = (indices.unsqueeze(0) == shifted_targets[:, None, None]).any(dim=1)
    boundary_aliases = sum(
        1
        for cell in range(cells - 1)
        if cell % side == side - 1 and bool(shifted[cell].any())
    )
    return {
        "output_cells": cells,
        "literal_supported_cells": int(literal.any(dim=1).sum().item()),
        "shifted_supported_cells": int(shifted.any(dim=1).sum().item()),
        "shifted_row_boundary_aliases": int(boundary_aliases),
        "shifted_terminal_holes": int((~shifted.any(dim=1)).sum().item()),
    }


def harmonic_projection(
    token_scores: torch.Tensor, incidence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project non-negative token scores with a validity-normalized harmonic mean.

    ``token_scores`` has shape ``[..., scale_tokens]`` and ``incidence`` has
    shape ``[output_cells, scale_tokens]``.  The returned validity vector is
    label free and records which output cells receive at least one token.
    """

    if not isinstance(token_scores, torch.Tensor) or not isinstance(
        incidence, torch.Tensor
    ):
        raise TypeError("token_scores and incidence must be torch tensors")
    values = token_scores.double()
    if values.ndim < 2 or incidence.ndim != 2:
        raise ValueError(
            "token_scores and incidence must be [..., tokens] and [cells, tokens]"
        )
    if values.shape[-1] != incidence.shape[-1]:
        raise ValueError("token count does not match the incidence matrix")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("token scores must be finite and non-negative")

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


def index_consistent_harmonic_projection(
    large_scores: torch.Tensor,
    mid_scores: torch.Tensor,
    patch_scores: torch.Tensor,
    large_mask: torch.Tensor,
    mid_mask: torch.Tensor,
) -> torch.Tensor:
    """Fuse three frozen token scales into an index-consistent base-grid map.

    The returned tensor has shape ``[..., side, side]``.  No labels, learned
    weights, thresholds, or additional backbone passes are used.
    """

    if not isinstance(patch_scores, torch.Tensor) or patch_scores.ndim < 2:
        raise ValueError("patch_scores must have shape [..., output_cells]")
    output_cells = int(patch_scores.shape[-1])
    side = math.isqrt(output_cells)
    if output_cells <= 0 or side * side != output_cells:
        raise ValueError("patch score count must be a positive perfect square")
    if not patch_scores.is_floating_point():
        raise TypeError("patch_scores must be floating point")
    if not bool(torch.isfinite(patch_scores).all()) or bool((patch_scores < 0).any()):
        raise ValueError("patch_scores must be finite and non-negative")
    if large_scores.shape[:-1] != patch_scores.shape[:-1]:
        raise ValueError("large and patch scores must share leading dimensions")
    if mid_scores.shape[:-1] != patch_scores.shape[:-1]:
        raise ValueError("mid and patch scores must share leading dimensions")
    if large_mask.device != patch_scores.device or mid_mask.device != patch_scores.device:
        raise ValueError("masks and scores must be on the same device")

    large_incidence = literal_incidence(large_mask, output_cells)
    mid_incidence = literal_incidence(mid_mask, output_cells)
    large_map, large_valid = harmonic_projection(large_scores, large_incidence)
    mid_map, mid_valid = harmonic_projection(mid_scores, mid_incidence)
    if not bool(large_valid.all()) or not bool(mid_valid.all()):
        raise ValueError("literal masks do not cover the complete output grid")
    fused = (large_map + mid_map + patch_scores.double()) / 3.0
    fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
    if not bool(torch.isfinite(fused).all()):
        raise RuntimeError("IHP produced a non-finite value")
    return fused.reshape(patch_scores.shape[:-1] + (side, side))
