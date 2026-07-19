"""Dynamic scientific primitives for the isolated ViTTrace v3 scorer.

This module contains only deterministic tensor/array transformations.  It
does not read project data, instantiate a vision model, or depend on any of
the frozen 13-arm scoring modules.  Row-major grid conventions are explicit:
flattened cell ``k`` has row ``k // grid_w`` and column ``k % grid_w``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PatchGrid = tuple[int, int]
CandidateMode = Literal["position", "row", "column", "global"]

# Upper bound used by tests for chunk-width-only float32 GEMM drift.
FLOAT32_MATCH_CHUNK_ATOL = 1e-6


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def validate_patch_grid(patch_grid: PatchGrid) -> PatchGrid:
    """Return a validated rectangular ``(grid_h, grid_w)`` patch grid."""

    if (
        not isinstance(patch_grid, (tuple, list))
        or len(patch_grid) != 2
    ):
        raise TypeError("patch_grid must be a two-item (grid_h, grid_w) pair")
    grid_h = _positive_int(patch_grid[0], "grid_h")
    grid_w = _positive_int(patch_grid[1], "grid_w")
    return grid_h, grid_w


def patch_count(patch_grid: PatchGrid) -> int:
    """Return dynamic base-grid size ``K = grid_h * grid_w``."""

    grid_h, grid_w = validate_patch_grid(patch_grid)
    return grid_h * grid_w


def _integer_tensor(mask: torch.Tensor | np.ndarray, name: str) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        tensor = torch.from_numpy(mask)
    elif isinstance(mask, torch.Tensor):
        tensor = mask
    else:
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")
    if (
        tensor.dtype == torch.bool
        or tensor.is_floating_point()
        or tensor.is_complex()
    ):
        raise TypeError(f"{name} must contain integer zero-based indices")
    return tensor


def validate_pooling_mask(
    mask: torch.Tensor | np.ndarray,
    patch_grid: PatchGrid,
    *,
    require_full_coverage: bool = False,
    name: str = "mask",
) -> torch.Tensor:
    """Validate a ``[members, scale_tokens]`` zero-based pooling mask.

    Memberships must be unique within each pooled token and lie in
    ``[0, K)``.  Full base-grid coverage is optional because the released
    incidence ablation deliberately contains an uncovered terminal cell.
    The returned tensor remains on the input device and has ``torch.long``
    dtype.
    """

    indices = _integer_tensor(mask, name)
    if indices.ndim != 2 or indices.shape[0] == 0 or indices.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty [members, scale_tokens] matrix")
    cells = patch_count(patch_grid)
    values = indices.to(dtype=torch.long)
    if bool((values < 0).any()) or bool((values >= cells).any()):
        raise ValueError(f"{name} contains an index outside [0, K)")
    if values.shape[0] > 1:
        ordered = torch.sort(values, dim=0).values
        if bool((ordered[1:] == ordered[:-1]).any()):
            raise ValueError(f"{name} repeats a base cell within a pooled token")
    if require_full_coverage:
        covered = torch.zeros(cells, dtype=torch.bool, device=values.device)
        covered[values.reshape(-1)] = True
        if not bool(covered.all()):
            missing = torch.nonzero(~covered, as_tuple=False).flatten()[:8].tolist()
            raise ValueError(f"{name} does not cover every base cell; missing {missing}")
    return values


def build_candidate_mask(
    patch_grid: PatchGrid,
    mode: CandidateMode,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build an exact row-major candidate relation ``[K, K]``.

    Axis zero indexes query cells and axis one indexes reference cells.
    Coordinate comparisons are used instead of flattened-index intervals, so
    neither row nor column support can wrap across a row-major boundary.
    """

    grid_h, grid_w = validate_patch_grid(patch_grid)
    cells = grid_h * grid_w
    if mode not in {"position", "row", "column", "global"}:
        raise ValueError(f"unsupported candidate mode {mode!r}")
    indices = torch.arange(cells, dtype=torch.long, device=device)
    rows = torch.div(indices, grid_w, rounding_mode="floor")
    columns = indices.remainder(grid_w)
    if mode == "position":
        result = indices[:, None] == indices[None, :]
    elif mode == "row":
        result = rows[:, None] == rows[None, :]
    elif mode == "column":
        result = columns[:, None] == columns[None, :]
    else:
        result = torch.ones((cells, cells), dtype=torch.bool, device=device)
    return result.to(dtype=torch.bool)


# Short public spelling retained for runner code while keeping construction
# explicit at the call site.
candidate_mask = build_candidate_mask


def _validate_candidate_mask(
    mask: torch.Tensor | np.ndarray,
    cells: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        value = torch.from_numpy(mask)
    elif isinstance(mask, torch.Tensor):
        value = mask
    else:
        raise TypeError("candidate_mask must be a torch.Tensor or numpy.ndarray")
    if value.dtype != torch.bool or value.shape != (cells, cells):
        raise ValueError("candidate_mask must be boolean with shape [K, K]")
    if not bool(value.any(dim=1).all()):
        raise ValueError("every query cell needs at least one candidate")
    return value.to(device=device)


def _incidence(
    mask: torch.Tensor | np.ndarray,
    patch_grid: PatchGrid,
    *,
    released: bool,
    require_full_coverage: bool,
) -> torch.Tensor:
    indices = validate_pooling_mask(
        mask,
        patch_grid,
        require_full_coverage=require_full_coverage,
    )
    cells = patch_count(patch_grid)
    start = 1 if released else 0
    targets = torch.arange(
        start,
        cells + start,
        dtype=torch.long,
        device=indices.device,
    )
    incidence = (indices.unsqueeze(0) == targets[:, None, None]).any(dim=1)
    expected = (cells, int(indices.shape[1]))
    if incidence.shape != expected:  # pragma: no cover - shape algebra guard
        raise RuntimeError("incidence construction returned the wrong shape")
    return incidence


def literal_incidence(
    mask: torch.Tensor | np.ndarray,
    patch_grid: PatchGrid,
    *,
    require_full_coverage: bool = False,
) -> torch.Tensor:
    """Project literal zero-based memberships to boolean ``[K, S]``."""

    return _incidence(
        mask,
        patch_grid,
        released=False,
        require_full_coverage=require_full_coverage,
    )


def released_incidence(
    mask: torch.Tensor | np.ndarray,
    patch_grid: PatchGrid,
) -> torch.Tensor:
    """Reproduce the released ``i + 1`` lookup as boolean ``[K, S]``."""

    return _incidence(
        mask,
        patch_grid,
        released=True,
        require_full_coverage=False,
    )


def harmonic_incidence_projection(
    scores: torch.Tensor,
    incidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Harmonically project scale scores ``[..., S]`` to ``[..., K]``.

    A cell with no incident token is returned as zero and marked invalid.  A
    zero incident score gives a zero harmonic mean, matching the mathematical
    continuous extension without introducing an epsilon.
    """

    if not isinstance(scores, torch.Tensor) or not isinstance(incidence, torch.Tensor):
        raise TypeError("scores and incidence must be torch tensors")
    if scores.ndim < 1 or not scores.is_floating_point():
        raise ValueError("scores must be a floating tensor with shape [..., S]")
    if incidence.ndim != 2 or incidence.dtype != torch.bool:
        raise ValueError("incidence must be a boolean [K, S] tensor")
    if scores.shape[-1] != incidence.shape[-1]:
        raise ValueError("score and incidence scale-token dimensions differ")
    if not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()):
        raise ValueError("scores must be finite and non-negative")
    if incidence.device != scores.device:
        raise ValueError("scores and incidence must be on the same device")

    values = scores.to(dtype=torch.float64)
    membership = incidence.reshape(
        (1,) * (values.ndim - 1) + tuple(incidence.shape)
    )
    reciprocal = torch.where(
        values.unsqueeze(-2) > 0.0,
        values.unsqueeze(-2).reciprocal(),
        torch.full_like(values.unsqueeze(-2), torch.inf),
    )
    denominator = torch.where(
        membership,
        reciprocal,
        torch.zeros_like(reciprocal),
    ).sum(dim=-1)
    count = incidence.sum(dim=-1, dtype=torch.long).to(dtype=torch.float64)
    count = count.reshape((1,) * (values.ndim - 1) + (incidence.shape[0],))
    valid = incidence.any(dim=-1)
    projected = torch.where(
        (count > 0.0) & torch.isfinite(denominator) & (denominator > 0.0),
        count / denominator,
        torch.zeros_like(denominator),
    )
    if not bool(torch.isfinite(projected).all()):  # pragma: no cover
        raise RuntimeError("harmonic incidence projection is not finite")
    return projected, valid


def _active_scale_names(
    scale_fields: Mapping[str, torch.Tensor],
    active_scales: Sequence[str] | str,
) -> tuple[str, ...]:
    available = tuple(scale_fields)
    if isinstance(active_scales, str):
        if active_scales in scale_fields:
            active = (active_scales,)
        elif active_scales and all(
            len(name) == 1 and name in scale_fields for name in active_scales
        ):
            active = tuple(active_scales)
        else:
            raise ValueError("active scale string is not representable by available keys")
    else:
        active = tuple(active_scales)
    if not active or any(not isinstance(name, str) or not name for name in active):
        raise ValueError("active_scales must select at least one named scale")
    if len(set(active)) != len(active):
        raise ValueError("active_scales cannot contain duplicates")
    missing = [name for name in active if name not in scale_fields]
    if missing:
        raise ValueError(f"unknown active scales {missing}; available={available}")
    return active


def fuse_scale_subset(
    scale_fields: Mapping[str, torch.Tensor],
    active_scales: Sequence[str] | str,
    *,
    valid_masks: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Fuse a registered scale subset, dividing by its active valid count.

    Every field has shape ``[..., K]``.  Validity is cell-specific; a cell
    missing from one active incidence projection is averaged over the
    remaining active fields rather than being divided by the total configured
    scale count.  Cells with no active valid field remain exactly zero.
    """

    if not isinstance(scale_fields, Mapping) or not scale_fields:
        raise ValueError("scale_fields must be a nonempty mapping")
    active = _active_scale_names(scale_fields, active_scales)
    reference = scale_fields[active[0]]
    if not isinstance(reference, torch.Tensor) or reference.ndim < 1:
        raise ValueError("scale fields must be tensors with shape [..., K]")
    shape = reference.shape
    if shape[-1] == 0:
        raise ValueError("scale fields cannot have an empty K dimension")
    if not reference.is_floating_point():
        raise TypeError("scale fields must be floating-point tensors")
    device = reference.device
    total = torch.zeros(shape, dtype=torch.float64, device=device)
    count = torch.zeros((shape[-1],), dtype=torch.float64, device=device)

    extra_masks = set(valid_masks or {}) - set(scale_fields)
    if extra_masks:
        raise ValueError(f"valid_masks contains unknown scales {sorted(extra_masks)}")
    for name in active:
        field = scale_fields[name]
        if not isinstance(field, torch.Tensor) or field.shape != shape:
            raise ValueError("all active scale fields must have identical shapes")
        if field.device != device or not field.is_floating_point():
            raise ValueError("all active scale fields must share device and floating dtype")
        if not bool(torch.isfinite(field).all()) or bool((field < 0).any()):
            raise ValueError("scale fields must be finite and non-negative")
        if valid_masks is None or name not in valid_masks:
            valid = torch.ones(shape[-1], dtype=torch.bool, device=device)
        else:
            valid = valid_masks[name]
            if (
                not isinstance(valid, torch.Tensor)
                or valid.dtype != torch.bool
                or valid.shape != (shape[-1],)
                or valid.device != device
            ):
                raise ValueError("each validity mask must be boolean [K] on the field device")
        broadcast = valid.reshape((1,) * (len(shape) - 1) + (shape[-1],))
        total = total + torch.where(
            broadcast,
            field.to(dtype=torch.float64),
            torch.zeros_like(total),
        )
        count = count + valid.to(dtype=torch.float64)

    denominator = count.reshape((1,) * (len(shape) - 1) + (shape[-1],))
    fused = torch.where(denominator > 0.0, total / denominator.clamp_min(1.0), total)
    return fused


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise TypeError("image_size must be a two-item (height, width) pair")
    height = _positive_int(image_size[0], "image_height")
    width = _positive_int(image_size[1], "image_width")
    return height, width


def _pixel_to_patch_axis(pixel_count: int, grid_count: int) -> np.ndarray:
    if grid_count > pixel_count:
        raise ValueError("patch grid cannot have more cells than image pixels")
    pixels = np.arange(pixel_count, dtype=np.int64)
    assignment = (pixels * int(grid_count)) // int(pixel_count)
    if assignment[0] != 0 or assignment[-1] != grid_count - 1:
        raise RuntimeError("pixel-to-patch assignment lost an endpoint")
    return assignment


def _temporal_coordinates(
    window_length: int,
    image_width: int,
    x_coordinates: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    length = _positive_int(window_length, "window_length")
    if x_coordinates is None:
        if length == 1:
            coordinates = np.zeros(1, dtype=np.float64)
        else:
            coordinates = np.linspace(0.0, float(image_width - 1), length, dtype=np.float64)
    else:
        coordinates = np.asarray(x_coordinates, dtype=np.float64)
        if coordinates.shape != (length,):
            raise ValueError("x_coordinates must have shape [W]")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("x_coordinates must be finite")
    tolerance = 16.0 * np.finfo(np.float64).eps * max(1, image_width - 1)
    if np.any(coordinates < -tolerance) or np.any(
        coordinates > float(image_width - 1) + tolerance
    ):
        raise ValueError("x_coordinates lie outside the image pixel columns")
    return np.clip(coordinates, 0.0, float(image_width - 1))


def validate_temporal_operator(
    operator: np.ndarray,
    *,
    window_length: int | None = None,
    patch_grid: PatchGrid | None = None,
    atol: float = 1e-12,
) -> np.ndarray:
    """Validate finite, non-negative, row-stochastic ``Q[W, K]``."""

    value = np.asarray(operator)
    if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
        raise TypeError("temporal operator must be a real numeric array")
    q = np.ascontiguousarray(value, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("temporal operator must be a nonempty [W, K] matrix")
    if window_length is not None and q.shape[0] != _positive_int(
        window_length, "window_length"
    ):
        raise ValueError("temporal operator W differs from window_length")
    if patch_grid is not None and q.shape[1] != patch_count(patch_grid):
        raise ValueError("temporal operator K differs from patch_grid")
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("temporal operator must be finite and non-negative")
    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and non-negative")
    row_sum = q.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sum, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("temporal operator must be row-stochastic")
    return q


def _full_column_operator(
    window_length: int,
    patch_grid: PatchGrid,
    x_coordinates: Sequence[float] | np.ndarray | None,
    *,
    image_size: tuple[int, int],
    nearest: bool,
) -> np.ndarray:
    grid_h, grid_w = validate_patch_grid(patch_grid)
    image_h, image_w = _validate_image_size(image_size)
    length = _positive_int(window_length, "window_length")
    coordinates = _temporal_coordinates(length, image_w, x_coordinates)
    q = np.zeros((length, grid_h * grid_w), dtype=np.float64)
    patch_rows = _pixel_to_patch_axis(image_h, grid_h)
    patch_columns = _pixel_to_patch_axis(image_w, grid_w)
    row_mass = np.bincount(patch_rows, minlength=grid_h).astype(np.float64)
    rows = np.arange(grid_h, dtype=np.int64) * grid_w
    if nearest:
        # Half-up rounding is applied in actual image-pixel coordinates.
        pixels = np.floor(coordinates + 0.5).astype(np.int64)
        pixels = np.minimum(pixels, image_w - 1)
        for timestamp, pixel in enumerate(pixels):
            q[timestamp, rows + patch_columns[pixel]] = row_mass
    else:
        left = np.floor(coordinates).astype(np.int64)
        right = np.minimum(left + 1, image_w - 1)
        fraction = coordinates - left.astype(np.float64)
        for timestamp in range(length):
            q[timestamp, rows + patch_columns[left[timestamp]]] += row_mass * (
                1.0 - fraction[timestamp]
            )
            q[timestamp, rows + patch_columns[right[timestamp]]] += (
                row_mass * fraction[timestamp]
            )
    row_sum = q.sum(axis=1, dtype=np.float64)
    if np.any(row_sum <= 0.0):  # pragma: no cover - every full column has mass
        raise RuntimeError("full-column operator contains a zero row")
    q /= row_sum[:, None]
    # Mirror the frozen operator's second normalization, removing the final
    # division residual before the row-stochastic validation.
    q /= q.sum(axis=1, dtype=np.float64)[:, None]
    return validate_temporal_operator(
        q,
        window_length=length,
        patch_grid=(grid_h, grid_w),
    )


def build_linear_nctp(
    window_length: int,
    patch_grid: PatchGrid,
    *,
    image_size: tuple[int, int] = (224, 224),
    x_coordinates: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Build pixel-linear normalized column-to-time projection ``[W,K]``.

    Timestamp coordinates live in the actual image frame. Adjacent pixel
    columns receive linear mass before deterministic pixel-to-patch pooling.
    """

    return _full_column_operator(
        window_length,
        patch_grid,
        x_coordinates,
        image_size=image_size,
        nearest=False,
    )


linear_nctp = build_linear_nctp


def build_nearest_full_column_operator(
    window_length: int,
    patch_grid: PatchGrid,
    *,
    image_size: tuple[int, int] = (224, 224),
    x_coordinates: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Build the pixel-nearest full-column ``floor(x + 0.5)`` control."""

    return _full_column_operator(
        window_length,
        patch_grid,
        x_coordinates,
        image_size=image_size,
        nearest=True,
    )


def apply_temporal_operator(
    operator: np.ndarray,
    patch_scores: np.ndarray,
) -> np.ndarray:
    """Apply row-stochastic ``Q[W,K]`` to scores with trailing dimension K."""

    q = validate_temporal_operator(operator)
    raw = np.asarray(patch_scores)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError("patch_scores must be a real numeric array")
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != q.shape[1]:
        raise ValueError("patch_scores must have trailing dimension K")
    if not np.all(np.isfinite(values)):
        raise ValueError("patch_scores must be finite")
    output = np.matmul(values, q.T)
    if not np.all(np.isfinite(output)):  # pragma: no cover
        raise RuntimeError("temporal projection returned a non-finite value")
    return np.ascontiguousarray(output, dtype=np.float64)


@dataclass(frozen=True)
class MatchResult:
    """Deterministic streamed matching result, with aligned ``[N,K]`` fields."""

    cost: torch.Tensor
    reference_index: torch.Tensor
    reference_window: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.cost.shape
        if self.cost.ndim != 2 or shape[0] == 0 or shape[1] == 0:
            raise ValueError("match cost must be a nonempty [N, K] tensor")
        if (
            self.reference_index.shape != shape
            or self.reference_window.shape != shape
            or self.valid_mask.shape != shape
        ):
            raise ValueError("all MatchResult fields must have shape [N, K]")
        if self.cost.dtype != torch.float64 or not bool(torch.isfinite(self.cost).all()):
            raise ValueError("match cost must be finite float64")
        if self.reference_index.dtype != torch.long:
            raise TypeError("reference_index must have dtype torch.long")
        if self.reference_window.dtype != torch.long:
            raise TypeError("reference_window must have dtype torch.long")
        if self.valid_mask.dtype != torch.bool or not bool(self.valid_mask.all()):
            raise ValueError("every streamed match must be valid")
        if bool((self.reference_index < 0).any()) or bool(
            (self.reference_index >= shape[1]).any()
        ):
            raise ValueError("reference_index lies outside [0, K)")
        if bool((self.reference_window < -1).any()):
            raise ValueError("reference_window must be -1 or a window index")


def _validate_tokens(
    tokens: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, int, int, int]:
    if isinstance(tokens, np.ndarray):
        values = torch.from_numpy(tokens)
    elif isinstance(tokens, torch.Tensor):
        values = tokens
    else:
        raise TypeError("tokens must be a torch.Tensor or numpy.ndarray")
    if values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError("tokens must have nonempty shape [N, K, D]")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("tokens must be finite floating-point values")
    windows, cells, features = map(int, values.shape)
    return values, windows, cells, features


def _chunk_size(value: int, name: str) -> int:
    return _positive_int(value, name)


def _cosine_cost(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    similarity = torch.matmul(query, reference.transpose(-1, -2)).clamp(-1.0, 1.0)
    return 0.5 * (1.0 - similarity)


def streamed_median_reference_match(
    tokens: torch.Tensor | np.ndarray,
    candidate_mask: torch.Tensor | np.ndarray,
    *,
    query_chunk_size: int = 32,
) -> MatchResult:
    """Match streamed queries to the coordinate-wise median reference grid.

    Arithmetic retains the input floating dtype.  Float32 chunk widths may
    differ within ``FLOAT32_MATCH_CHUNK_ATOL``; one full query chunk exactly
    follows the frozen full-tensor reference operation order.
    """

    values, windows, cells, _ = _validate_tokens(tokens)
    chunk = _chunk_size(query_chunk_size, "query_chunk_size")
    allowed = _validate_candidate_mask(candidate_mask, cells, device=values.device)
    # Preserve frozen-cache float32 arithmetic; only selected costs are
    # promoted for downstream projection and serialization.
    query = F.normalize(values, dim=-1)
    memory = F.normalize(
        torch.median(values, dim=0).values, dim=-1
    )
    costs: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, windows, chunk):
        stop = min(start + chunk, windows)
        pair = _cosine_cost(query[start:stop], memory)
        pair = pair.masked_fill(~allowed.unsqueeze(0), torch.inf)
        selected_cost, selected_index = torch.min(pair, dim=-1)
        costs.append(selected_cost.to(dtype=torch.float64))
        indices.append(selected_index.to(dtype=torch.long))
    result_cost = torch.cat(costs, dim=0)
    result_index = torch.cat(indices, dim=0)
    reference_window = torch.full_like(result_index, -1)
    valid = torch.ones_like(result_index, dtype=torch.bool)
    return MatchResult(result_cost, result_index, reference_window, valid)


def streamed_all_pairs_median_match(
    tokens: torch.Tensor | np.ndarray,
    candidate_mask: torch.Tensor | np.ndarray,
    *,
    query_chunk_size: int = 4,
    reference_chunk_size: int = 8,
) -> MatchResult:
    """Return robust all-window matches without materializing ``[N,N,K,K]``.

    For each query token, the nearest allowed reference token is selected
    independently inside every reference window.  The output cost is the
    deterministic lower median of those per-window minima.  Exact cost ties
    choose the lowest reference-window index, while within-window ties choose
    the lowest row-major reference index.

    Cosine arithmetic retains the input dtype.  For float32 caches, changing
    chunk widths can alter costs by at most the documented
    ``FLOAT32_MATCH_CHUNK_ATOL`` without changing output order on separated
    candidates.  Full query/reference chunks reproduce the dense operation.
    """

    values, windows, cells, _ = _validate_tokens(tokens)
    query_chunk = _chunk_size(query_chunk_size, "query_chunk_size")
    reference_chunk = _chunk_size(reference_chunk_size, "reference_chunk_size")
    allowed = _validate_candidate_mask(candidate_mask, cells, device=values.device)
    normalized = F.normalize(values, dim=-1)
    all_costs: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    all_windows: list[torch.Tensor] = []
    rank = (windows - 1) // 2

    for query_start in range(0, windows, query_chunk):
        query_stop = min(query_start + query_chunk, windows)
        query = normalized[query_start:query_stop]
        window_cost_parts: list[torch.Tensor] = []
        window_index_parts: list[torch.Tensor] = []
        for reference_start in range(0, windows, reference_chunk):
            reference_stop = min(reference_start + reference_chunk, windows)
            reference = normalized[reference_start:reference_stop]
            similarity = torch.einsum("bqd,rpd->brqp", query, reference).clamp(
                -1.0, 1.0
            )
            pair = 0.5 * (1.0 - similarity)
            pair = pair.masked_fill(~allowed.reshape(1, 1, cells, cells), torch.inf)
            selected_cost, selected_index = torch.min(pair, dim=-1)
            window_cost_parts.append(selected_cost)
            window_index_parts.append(selected_index.to(dtype=torch.long))

        per_window_cost = torch.cat(window_cost_parts, dim=1)
        per_window_index = torch.cat(window_index_parts, dim=1)
        # Stable ordering makes equal-cost reference-window ties select the
        # earliest window.  The rank is the lower median for even N.
        order = torch.argsort(per_window_cost, dim=1, stable=True)
        ranked_window = order[:, rank, :]
        selected_cost = torch.gather(
            per_window_cost,
            1,
            ranked_window.unsqueeze(1),
        ).squeeze(1)
        # The order statistic defines the lower-median value.  If multiple
        # windows attain that value, report the lowest window index rather
        # than the arbitrary rank slot inside the tied block.
        median_tie = per_window_cost == selected_cost.unsqueeze(1)
        chosen_window = torch.argmax(median_tie.to(dtype=torch.int64), dim=1)
        selected_index = torch.gather(
            per_window_index,
            1,
            chosen_window.unsqueeze(1),
        ).squeeze(1)
        all_costs.append(selected_cost.to(dtype=torch.float64))
        all_indices.append(selected_index.to(dtype=torch.long))
        all_windows.append(chosen_window.to(dtype=torch.long))

    result_cost = torch.cat(all_costs, dim=0)
    result_index = torch.cat(all_indices, dim=0)
    result_window = torch.cat(all_windows, dim=0)
    valid = torch.ones_like(result_index, dtype=torch.bool)
    return MatchResult(result_cost, result_index, result_window, valid)


def _validate_maps(maps: np.ndarray) -> np.ndarray:
    raw = np.asarray(maps)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError("maps must be a real numeric array")
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] == 0 or values.shape[-1] == 0:
        raise ValueError("maps must have shape [..., H, W] with positive H and W")
    if not np.all(np.isfinite(values)):
        raise ValueError("maps must be finite")
    return values


def column_quantile(maps: np.ndarray, q: float) -> np.ndarray:
    """Return the literal linear row quantile for every image column."""

    values = _validate_maps(maps)
    quantile = float(q)
    if not math.isfinite(quantile) or quantile < 0.0 or quantile > 1.0:
        raise ValueError("q must lie in [0, 1]")
    output = np.quantile(values, quantile, axis=-2, method="linear")
    return np.ascontiguousarray(output, dtype=np.float64)


def column_top_fraction_mean(maps: np.ndarray, top_fraction: float) -> np.ndarray:
    """Average the largest ``ceil(H * top_fraction)`` rows per column."""

    values = _validate_maps(maps)
    fraction = float(top_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")
    height = int(values.shape[-2])
    count = max(1, int(math.ceil(height * fraction)))
    split = height - count
    upper = np.partition(values, split, axis=-2)[..., split:, :]
    output = upper.mean(axis=-2, dtype=np.float64)
    return np.ascontiguousarray(output, dtype=np.float64)


__all__ = [
    "CandidateMode",
    "FLOAT32_MATCH_CHUNK_ATOL",
    "MatchResult",
    "PatchGrid",
    "apply_temporal_operator",
    "build_candidate_mask",
    "build_linear_nctp",
    "build_nearest_full_column_operator",
    "candidate_mask",
    "column_quantile",
    "column_top_fraction_mean",
    "fuse_scale_subset",
    "harmonic_incidence_projection",
    "linear_nctp",
    "literal_incidence",
    "patch_count",
    "released_incidence",
    "streamed_all_pairs_median_match",
    "streamed_median_reference_match",
    "validate_patch_grid",
    "validate_pooling_mask",
    "validate_temporal_operator",
]
