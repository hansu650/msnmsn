"""Label-free reducers from frozen two-dimensional anomaly maps to time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .data import WindowBatch
from .geometry import TraceGeometry


ArmName = Literal[
    "O0",
    "A0",
    "0B",
    "AB",
    "PAPER_Q25",
    "PAPER_Q75",
    "CENTERLINE",
    "SUPPORT_ONLY",
    "UNNORMALIZED",
    "SUPERSAMPLE4",
]


@dataclass(frozen=True)
class WindowPullback:
    """Window-local additive evidence and support mass, both ``[N, L]``."""

    numerator: np.ndarray
    denominator: np.ndarray

    def __post_init__(self) -> None:
        num = np.asarray(self.numerator)
        den = np.asarray(self.denominator)
        if num.ndim != 2 or num.shape != den.shape:
            raise ValueError("pullback arrays must be aligned [N, L] matrices")
        if not np.isfinite(num).all() or not np.isfinite(den).all():
            raise ValueError("pullback arrays must be finite")
        if np.any(den < 0):
            raise ValueError("pullback support mass cannot be negative")


def _validate_maps(maps: np.ndarray) -> np.ndarray:
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (224, 224):
        raise ValueError("anomaly maps must have shape [N, 224, 224]")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("anomaly maps must be non-empty and finite")
    return values


def released_column_mean(maps: np.ndarray, top_fraction: float = 0.25) -> np.ndarray:
    """Reproduce the released top-fraction column mean for every window."""

    values = _validate_maps(maps)
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")
    height = values.shape[1]
    k = max(1, int(np.ceil(height * float(top_fraction))))
    split = height - k
    top = np.partition(values, split, axis=1)[:, split:, :]
    out = top.mean(axis=1, dtype=np.float64)
    if out.shape != (values.shape[0], values.shape[2]):
        raise RuntimeError("released reducer returned the wrong shape")
    return out


def paper_column_quantile(maps: np.ndarray, q: Literal[0.25, 0.75]) -> np.ndarray:
    """Return the literal and intended-upper paper quantile controls."""

    values = _validate_maps(maps)
    if float(q) not in (0.25, 0.75):
        raise ValueError("paper quantile is frozen to 0.25 or 0.75")
    return np.quantile(values, float(q), axis=1, method="linear").astype(np.float64)


def _bilinear_sample(maps: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = _validate_maps(maps)
    coords = np.asarray(xy, dtype=np.float64)
    if coords.shape[:1] != values.shape[:1] or coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("coordinates must have shape [N, L, 2]")
    height, width = values.shape[1:]
    x = coords[..., 0]
    y = coords[..., 1]
    valid = (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    xc = np.clip(x, 0.0, width - 1)
    yc = np.clip(y, 0.0, height - 1)
    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = xc - x0
    wy = yc - y0
    batch = np.arange(values.shape[0], dtype=np.int64)[:, None]
    sampled = (
        values[batch, y0, x0] * (1.0 - wx) * (1.0 - wy)
        + values[batch, y0, x1] * wx * (1.0 - wy)
        + values[batch, y1, x0] * (1.0 - wx) * wy
        + values[batch, y1, x1] * wx * wy
    )
    return sampled * valid, valid.astype(np.float64)


def sample_centerline(maps: np.ndarray, vertices_xy: np.ndarray) -> WindowPullback:
    """Bilinearly sample each local timestamp at its rendered centerline."""

    numerator, denominator = _bilinear_sample(maps, vertices_xy)
    return WindowPullback(numerator.astype(np.float64), denominator)


def _scatter_pixels_to_time(
    sample_coord: np.ndarray,
    pixel_values: np.ndarray,
    pixel_mass: np.ndarray,
    length: int,
) -> WindowPullback:
    coord = np.asarray(sample_coord, dtype=np.float64)
    vals = np.asarray(pixel_values, dtype=np.float64)
    mass = np.asarray(pixel_mass, dtype=np.float64)
    if coord.shape != vals.shape or coord.shape != mass.shape or coord.ndim != 3:
        raise ValueError("pixel coordinate, value, and mass tensors must be aligned [N,H,W]")
    if not np.isfinite(vals).all() or not np.isfinite(mass).all():
        raise ValueError("pixel pullback inputs must be finite")
    if np.any(mass < 0):
        raise ValueError("pixel mass cannot be negative")

    n_windows = coord.shape[0]
    numerator = np.zeros((n_windows, length), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    active = mass > 0
    if np.any(active & ~np.isfinite(coord)):
        raise ValueError("supported pixels require finite time coordinates")
    safe_coord = np.where(active, coord, 0.0)
    flat_coord = np.clip(safe_coord.reshape(n_windows, -1), 0.0, length - 1)
    flat_vals = vals.reshape(n_windows, -1)
    flat_mass = mass.reshape(n_windows, -1)
    lower = np.floor(flat_coord).astype(np.int64)
    upper = np.minimum(lower + 1, length - 1)
    upper_w = flat_coord - lower
    lower_w = 1.0 - upper_w
    for window in range(n_windows):
        lo_mass = flat_mass[window] * lower_w[window]
        hi_mass = flat_mass[window] * upper_w[window]
        np.add.at(denominator[window], lower[window], lo_mass)
        np.add.at(denominator[window], upper[window], hi_mass)
        np.add.at(numerator[window], lower[window], lo_mass * flat_vals[window])
        np.add.at(numerator[window], upper[window], hi_mass * flat_vals[window])
    return WindowPullback(numerator, denominator)


def _geometry_stack(
    geometries: Sequence[TraceGeometry],
    field: Literal["time_coordinate", "kernel", "support_mask", "vertices"],
) -> np.ndarray:
    if not geometries:
        raise ValueError("at least one geometry is required")
    return np.stack([np.asarray(getattr(g, field)) for g in geometries], axis=0)


def pullback_support_only(
    maps: np.ndarray, geometries: Sequence[TraceGeometry]
) -> WindowPullback:
    """Binary trace-support mean with continuous timestamp ownership."""

    values = _validate_maps(maps)
    coord = _geometry_stack(geometries, "time_coordinate")
    if values.shape != coord.shape:
        raise ValueError("maps and geometry must have identical raster shape")
    mass = _geometry_stack(geometries, "support_mask").astype(np.float64)
    return _scatter_pixels_to_time(
        coord,
        values,
        mass,
        geometries[0].window_length,
    )


def pullback_traceback(
    maps: np.ndarray,
    geometries: Sequence[TraceGeometry],
    normalize_support: bool = True,
) -> WindowPullback:
    """Pull frozen anomaly evidence back with the renderer's compact kernel."""

    values = _validate_maps(maps)
    coord = _geometry_stack(geometries, "time_coordinate")
    if values.shape != coord.shape:
        raise ValueError("maps and geometry must have identical raster shape")
    pb = _scatter_pixels_to_time(
        coord,
        values,
        _geometry_stack(geometries, "kernel").astype(np.float64),
        geometries[0].window_length,
    )
    if normalize_support:
        return pb
    # The control preserves weighted evidence but deliberately removes local
    # support normalization. Global overlap averaging remains well-defined.
    denominator = (pb.denominator > 0).astype(np.float64)
    return WindowPullback(pb.numerator, denominator)


def stitch_column_vectors(
    vectors: np.ndarray,
    full_length: int,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Reproduce released raster-column stitching and temporal interpolation."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("column vectors must be a finite [N,W] matrix")
    if window_size <= 0 or step_size <= 0 or window_size % step_size:
        raise ValueError("window and step must define an integer overlap ratio")
    ratio = window_size // step_size
    raster_width = values.shape[1]
    raster_step = raster_width // ratio
    raster_length = raster_width + (values.shape[0] - 1) * raster_step
    summed = np.zeros(raster_length, dtype=np.float64)
    count = np.zeros(raster_length, dtype=np.float64)
    for index, vector in enumerate(values):
        start = index * raster_step
        summed[start : start + raster_width] += vector
        count[start : start + raster_width] += 1.0
    stitched = summed / np.maximum(count, 1.0)
    covered = window_size + (values.shape[0] - 1) * step_size
    temporal = np.interp(
        np.linspace(0.0, raster_length - 1, covered),
        np.arange(raster_length, dtype=np.float64),
        stitched,
    )
    if covered < full_length:
        # Match the released ``align_anomaly_vector`` suffix exactly: extend
        # the final covered-sample slope instead of holding the edge value.
        slope = temporal[-1] - temporal[-2] if covered > 1 else 0.0
        tail = temporal[-1] + slope * np.arange(
            1,
            full_length - covered + 1,
            dtype=np.float64,
        )
        temporal = np.concatenate((temporal, tail))
    return temporal[:full_length]


def stitch_pullbacks(
    pb: WindowPullback,
    starts: np.ndarray,
    full_length: int,
    empty_policy: Literal["edge_hold"] = "edge_hold",
) -> np.ndarray:
    """Coverage-normalize window evidence once in global timestamp space."""

    if empty_policy != "edge_hold":
        raise ValueError("the only registered empty policy is edge_hold")
    start_values = np.asarray(starts, dtype=np.int64)
    if start_values.shape != (pb.numerator.shape[0],):
        raise ValueError("one start index is required per window")
    length = pb.numerator.shape[1]
    numerator = np.zeros(full_length, dtype=np.float64)
    denominator = np.zeros(full_length, dtype=np.float64)
    for index, start in enumerate(start_values):
        end = min(int(start) + length, full_length)
        width = end - int(start)
        if width <= 0:
            continue
        numerator[start:end] += pb.numerator[index, :width]
        denominator[start:end] += pb.denominator[index, :width]
    valid = denominator > 0
    if not np.any(valid):
        raise ValueError("TraceBack produced no supported timestamp")
    output = np.empty(full_length, dtype=np.float64)
    output[valid] = numerator[valid] / denominator[valid]
    missing = ~valid
    if np.any(missing):
        valid_index = np.flatnonzero(valid)
        output[missing] = np.interp(np.flatnonzero(missing), valid_index, output[valid])
    if not np.isfinite(output).all():
        raise RuntimeError("stitched score is not finite")
    return output


def score_arm(
    arm: ArmName,
    released_maps: np.ndarray,
    distance_maps: np.ndarray,
    supersample_maps: np.ndarray | None,
    released_geometry: Sequence[TraceGeometry],
    distance_geometry: Sequence[TraceGeometry],
    batch: WindowBatch,
    *,
    supersample_geometry: Sequence[TraceGeometry] | None = None,
) -> np.ndarray:
    """Route a frozen registered arm without labels or metric-dependent choices."""

    if arm == "O0":
        vectors = released_column_mean(released_maps)
        score = stitch_column_vectors(vectors, batch.full_length, batch.window_size, batch.step_size)
    elif arm == "A0":
        vectors = released_column_mean(distance_maps)
        score = stitch_column_vectors(vectors, batch.full_length, batch.window_size, batch.step_size)
    elif arm == "0B":
        score = stitch_pullbacks(
            pullback_traceback(released_maps, released_geometry), batch.starts, batch.full_length
        )
    elif arm == "AB":
        score = stitch_pullbacks(
            pullback_traceback(distance_maps, distance_geometry), batch.starts, batch.full_length
        )
    elif arm == "PAPER_Q25":
        score = stitch_column_vectors(
            paper_column_quantile(released_maps, 0.25),
            batch.full_length,
            batch.window_size,
            batch.step_size,
        )
    elif arm == "PAPER_Q75":
        score = stitch_column_vectors(
            paper_column_quantile(released_maps, 0.75),
            batch.full_length,
            batch.window_size,
            batch.step_size,
        )
    elif arm == "CENTERLINE":
        score = stitch_pullbacks(
            sample_centerline(distance_maps, _geometry_stack(distance_geometry, "vertices")),
            batch.starts,
            batch.full_length,
        )
    elif arm == "SUPPORT_ONLY":
        score = stitch_pullbacks(
            pullback_support_only(distance_maps, distance_geometry),
            batch.starts,
            batch.full_length,
        )
    elif arm == "UNNORMALIZED":
        score = stitch_pullbacks(
            pullback_traceback(distance_maps, distance_geometry, normalize_support=False),
            batch.starts,
            batch.full_length,
        )
    elif arm == "SUPERSAMPLE4":
        if supersample_maps is None:
            raise ValueError("SUPERSAMPLE4 requires its registered map cache")
        # The renderer owns the pixel-to-time measure.  The released geometry
        # fallback is retained only for callers created before the dedicated
        # supersample geometry was exposed.
        geometry = (
            released_geometry
            if supersample_geometry is None
            else supersample_geometry
        )
        score = stitch_pullbacks(
            pullback_traceback(supersample_maps, geometry),
            batch.starts,
            batch.full_length,
        )
    else:
        raise ValueError(f"unregistered arm: {arm}")
    result = np.asarray(score, dtype=np.float64)
    if result.shape != (batch.full_length,) or not np.isfinite(result).all():
        raise RuntimeError(f"arm {arm} returned an invalid score vector")
    return result
