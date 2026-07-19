"""Renderer-Indexed Temporal Pullback for frozen IHP patch evidence.

RITP converts effective renderer coverage in the actual ``224 x 224`` frame
to a float64 row-stochastic operator ``Q`` with shape ``[240, 196]``.  Each
covered pixel is assigned to its floor/ceil timestamps by a triangular
ownership kernel and to its row-major ViT-B/16 patch by pixel containment.
"""

from __future__ import annotations

import numpy as np

from .renderer_trace import (
    IMAGE_SHAPE,
    PATCH_GRID,
    PATCH_SHAPE,
    WINDOW_LENGTH,
    RendererTrace,
    linear_full_column_coverage,
)


PATCH_COUNT = PATCH_GRID[0] * PATCH_GRID[1]


def _normalize_rows(mass: np.ndarray, *, context: str) -> np.ndarray:
    value = np.asarray(mass, dtype=np.float64)
    if value.shape != (WINDOW_LENGTH, PATCH_COUNT):
        raise ValueError(
            f"{context} mass must have shape [{WINDOW_LENGTH},{PATCH_COUNT}]"
        )
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError(f"{context} mass must be finite and non-negative")
    row_sum = np.sum(value, axis=1, dtype=np.float64)
    missing = np.flatnonzero(row_sum <= 0.0)
    if missing.size:
        preview = ",".join(str(int(index)) for index in missing[:8])
        raise ValueError(f"{context} has zero-mass timestamp rows: {preview}")
    operator = value / row_sum[:, None]
    # A second normalization removes the final division-rounding residual and
    # makes the serialized row-stochastic contract deterministic.
    operator /= np.sum(operator, axis=1, dtype=np.float64)[:, None]
    if not np.allclose(
        np.sum(operator, axis=1, dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{context} row normalization failed")
    return np.ascontiguousarray(operator, dtype=np.float64)


def _patch_indices() -> np.ndarray:
    rows, columns = np.indices(IMAGE_SHAPE, dtype=np.int64)
    patch_row = rows // PATCH_SHAPE[0]
    patch_column = columns // PATCH_SHAPE[1]
    return patch_row * PATCH_GRID[1] + patch_column


_PIXEL_PATCH = _patch_indices()


def build_soft_ritp(trace: RendererTrace) -> np.ndarray:
    """Build exact-soft ``Q[240,196]`` from one renderer trace."""

    if not isinstance(trace, RendererTrace):
        raise TypeError("trace must be a RendererTrace")
    if trace.window_length != WINDOW_LENGTH:
        raise ValueError("RITP requires exactly 240 local timestamps")

    active = trace.alpha > 0.0
    alpha = trace.alpha[active]
    coordinate = trace.time_coordinate[active]
    patch = _PIXEL_PATCH[active]
    if alpha.size == 0:
        raise ValueError("RITP trace has no positive alpha coverage")

    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - lower.astype(np.float64)
    upper = np.minimum(lower + 1, WINDOW_LENGTH - 1)
    mass = np.zeros((WINDOW_LENGTH, PATCH_COUNT), dtype=np.float64)
    np.add.at(mass, (lower, patch), alpha * (1.0 - fraction))
    upper_weight = alpha * fraction
    has_upper = upper_weight > 0.0
    np.add.at(
        mass,
        (upper[has_upper], patch[has_upper]),
        upper_weight[has_upper],
    )
    return _normalize_rows(mass, context="soft RITP")


def harden_ritp_operator(operator: np.ndarray) -> np.ndarray:
    """Return deterministic one-hot rows; exact ties choose the lowest index."""

    value = np.asarray(operator, dtype=np.float64)
    if value.shape != (WINDOW_LENGTH, PATCH_COUNT):
        raise ValueError("operator must have shape [240,196]")
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("operator must be finite and non-negative")
    row_sum = np.sum(value, axis=1, dtype=np.float64)
    if np.any(row_sum <= 0.0):
        raise ValueError("operator contains a zero row")
    # NumPy argmax returns the first occurrence, which is the required lowest
    # row-major patch index on an exact tie.
    winner = np.argmax(value, axis=1)
    hard = np.zeros_like(value, dtype=np.float64)
    hard[np.arange(WINDOW_LENGTH), winner] = 1.0
    return hard


def build_hard_ritp(trace: RendererTrace) -> np.ndarray:
    """Build the deterministic hard-support deletion ablation."""

    return harden_ritp_operator(build_soft_ritp(trace))


def _patch_integrals(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=np.float64)
    if value.shape != IMAGE_SHAPE:
        raise ValueError("coverage field must have shape [224,224]")
    cells = value.reshape(
        PATCH_GRID[0],
        PATCH_SHAPE[0],
        PATCH_GRID[1],
        PATCH_SHAPE[1],
    ).sum(axis=(1, 3), dtype=np.float64)
    return cells.reshape(PATCH_COUNT)


def build_full_column_operator() -> np.ndarray:
    """Build the parameter-free full-height linear-column control ``Q_x``."""

    mass = np.empty((WINDOW_LENGTH, PATCH_COUNT), dtype=np.float64)
    for timestamp in range(WINDOW_LENGTH):
        coverage = linear_full_column_coverage(timestamp)
        mass[timestamp] = _patch_integrals(coverage)
    return _normalize_rows(mass, context="full-column control")


def apply_ritp(operator: np.ndarray, patch_scores: np.ndarray) -> np.ndarray:
    """Apply ``Q`` to one or more row-major 14x14 patch-score vectors."""

    q = np.asarray(operator, dtype=np.float64)
    values = np.asarray(patch_scores, dtype=np.float64)
    if q.shape != (WINDOW_LENGTH, PATCH_COUNT):
        raise ValueError("operator must have shape [240,196]")
    if values.ndim == 0 or values.shape[-1] != PATCH_COUNT:
        raise ValueError("patch_scores must have trailing dimension 196")
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("operator must be finite and non-negative")
    if not np.all(np.isfinite(values)):
        raise ValueError("patch_scores must be finite")
    if not np.allclose(
        np.sum(q, axis=1, dtype=np.float64), 1.0, rtol=0.0, atol=1e-10
    ):
        raise ValueError("operator is not row-stochastic")
    output = np.matmul(values, q.T)
    return np.ascontiguousarray(output, dtype=np.float64)


def stitch_native_240(
    window_scores: np.ndarray,
    window_starts: np.ndarray,
    full_length: int,
) -> np.ndarray:
    """Stitch native 240-vectors at exact global index ``window_start + t``.

    Overlaps are accumulated in input-window order and normalized by their
    integer coverage counts.  The normal complete-window protocol may leave a
    final suffix shorter than the step; that suffix retains the frozen vendor
    last-slope extrapolation.  Prefix or interior holes fail closed.
    """

    scores = np.asarray(window_scores, dtype=np.float64)
    starts_raw = np.asarray(window_starts)
    length = int(full_length)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] != WINDOW_LENGTH:
        raise ValueError("window_scores must be a nonempty [N,240] array")
    if not np.all(np.isfinite(scores)):
        raise ValueError("window_scores must be finite")
    if starts_raw.ndim != 1 or starts_raw.shape[0] != scores.shape[0]:
        raise ValueError("window_starts must have shape [N]")
    if isinstance(starts_raw, np.ndarray) and (
        not np.issubdtype(starts_raw.dtype, np.integer)
        or np.issubdtype(starts_raw.dtype, np.bool_)
    ):
        raise TypeError("window_starts must contain integers")
    starts = starts_raw.astype(np.int64, copy=False)
    if length <= 0 or np.any(starts < 0):
        raise ValueError("full_length and starts must be non-negative")
    if np.any(np.diff(starts) <= 0):
        raise ValueError("window_starts must be strictly increasing")
    if np.any(starts + WINDOW_LENGTH > length):
        raise ValueError("every native window must fit inside full_length")

    summed = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.int64)
    for index, start in enumerate(starts):
        stop = int(start) + WINDOW_LENGTH
        summed[int(start) : stop] += scores[index]
        count[int(start) : stop] += 1

    supported = count > 0
    if not supported[0]:
        raise ValueError("native stitcher has an uncovered prefix")
    last = int(np.flatnonzero(supported)[-1])
    if np.any(~supported[: last + 1]):
        raise ValueError("native stitcher has an interior coverage hole")
    output = np.empty(length, dtype=np.float64)
    output[: last + 1] = summed[: last + 1] / count[: last + 1]
    if last + 1 < length:
        slope = output[last] - output[last - 1] if last > 0 else 0.0
        output[last + 1 :] = output[last] + slope * np.arange(
            1, length - last, dtype=np.float64
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("native stitcher produced a non-finite value")
    return output


__all__ = [
    "PATCH_COUNT",
    "apply_ritp",
    "build_full_column_operator",
    "build_hard_ritp",
    "build_soft_ritp",
    "harden_ritp_operator",
    "stitch_native_240",
]
