"""Deterministic trace geometry for the measure-consistent visual interface.

The functions in this module operate in image coordinates: ``x`` increases to
the right and ``y`` increases downwards.  Pixel centers therefore have integer
coordinates ``(column, row)``.  A time-series sample is mapped to a polyline
vertex, while pixels are associated with their nearest *continuous* line
segment rather than with a discretized collection of drawn pixels.

Only geometry is implemented here.  In particular, this module never accepts
labels or anomaly scores and does not reduce an anomaly map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_WINDOW_LENGTH = 240
DEFAULT_BANDWIDTH = 8.0


@dataclass(frozen=True)
class TraceGeometry:
    """Dense pixel-to-polyline geometry returned by a trace renderer.

    Attributes
    ----------
    vertices:
        Polyline vertices with shape ``[T, 2]`` in ``(x, y)`` image
        coordinates.  Vertices may lie slightly outside the image after a
        phase shift; this deliberately represents boundary clipping.
    distance:
        Euclidean distance from each pixel center to the continuous polyline,
        clipped to ``bandwidth``.  Shape ``[H, W]``.
    kernel:
        The exact truncated linear field
        ``max(1 - distance / bandwidth, 0)``.
    nearest_segment:
        Index of the nearest segment for pixels inside the open support tube;
        ``-1`` outside it.  Ties are resolved by the lower segment index.
    segment_fraction:
        Projection coordinate in ``[0, 1]`` along ``nearest_segment``.
    time_coordinate:
        Continuous zero-based sample coordinate
        ``nearest_segment + segment_fraction``.
    normal_coordinate:
        Signed normal offset from the nearest segment in image pixels.  The
        sign follows the oriented 2-D cross product of segment tangent and
        pixel displacement.
    """

    image_size: Tuple[int, int]
    window_length: int
    bandwidth: float
    phase_shift: Tuple[float, float]
    vertices: np.ndarray
    distance: np.ndarray
    kernel: np.ndarray
    nearest_segment: np.ndarray
    segment_fraction: np.ndarray
    time_coordinate: np.ndarray
    normal_coordinate: np.ndarray

    @property
    def support_mask(self) -> np.ndarray:
        """Return pixels with strictly positive truncated-kernel mass."""

        return self.kernel > 0.0

    @property
    def support_mass(self) -> float:
        """Return the total discrete mass of the truncated linear field."""

        return float(np.sum(self.kernel, dtype=np.float64))


def normalize_phase_shift(
    phase_shift: float | Sequence[float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """Normalize a scalar or two-vector phase shift to ``(dx, dy)``.

    A scalar means an equal shift in both image axes.  The experiment uses
    quarter- and half-pixel perturbations, but arbitrary finite sub-pixel
    shifts are accepted to keep the primitive reusable.
    """

    if np.isscalar(phase_shift):
        dx = dy = float(phase_shift)
    else:
        values = tuple(float(value) for value in phase_shift)
        if len(values) != 2:
            raise ValueError("phase_shift must be a scalar or a length-two sequence")
        dx, dy = values
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ValueError("phase_shift must contain finite values")
    return dx, dy


def trace_vertices(
    values: Sequence[float] | np.ndarray,
    *,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    value_range: Tuple[float, float] = (0.0, 1.0),
    phase_shift: float | Sequence[float] = (0.0, 0.0),
    expected_length: int = DEFAULT_WINDOW_LENGTH,
) -> np.ndarray:
    """Map a uniformly sampled trajectory to continuous image coordinates.

    Values are not clipped.  Consequently, an out-of-range trajectory remains
    a well-defined polyline outside the canvas and contributes only where its
    support tube intersects the image.  This is important for representing
    renderer boundary clipping without changing the underlying trajectory.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional sequence")
    if array.size != expected_length:
        raise ValueError(
            f"expected a {expected_length}-sample window, received {array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")

    height, width = (int(image_size[0]), int(image_size[1]))
    if height <= 1 or width <= 1:
        raise ValueError("image dimensions must both be greater than one")
    low, high = (float(value_range[0]), float(value_range[1]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("value_range must be finite and strictly increasing")
    dx, dy = normalize_phase_shift(phase_shift)

    x = np.linspace(0.0, float(width - 1), array.size, dtype=np.float64) + dx
    normalized = (array - low) / (high - low)
    y = (1.0 - normalized) * float(height - 1) + dy
    return np.column_stack((x, y))


def continuous_polyline_geometry(
    vertices: np.ndarray,
    *,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    bandwidth: float = DEFAULT_BANDWIDTH,
    phase_shift: float | Sequence[float] = (0.0, 0.0),
) -> TraceGeometry:
    """Compute an exact truncated distance field for a continuous polyline.

    The implementation is exact for the requested truncated field but avoids
    comparing every pixel with every segment.  For image column ``x`` it only
    considers segments whose horizontal extent is within ``bandwidth`` of
    ``x``.  Any omitted segment is already farther than the truncation radius
    in the horizontal coordinate alone and therefore has zero kernel mass.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("vertices must have shape [T, 2] with T >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("vertices must be finite")
    height, width = (int(image_size[0]), int(image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    radius = float(bandwidth)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    shift = normalize_phase_shift(phase_shift)

    starts = points[:-1]
    vectors = points[1:] - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    if np.any(squared_lengths <= 0.0):
        raise ValueError("consecutive polyline vertices must be distinct")
    segment_min_x = np.minimum(starts[:, 0], points[1:, 0])
    segment_max_x = np.maximum(starts[:, 0], points[1:, 0])

    clipped_distance = np.full((height, width), radius, dtype=np.float64)
    nearest_segment = np.full((height, width), -1, dtype=np.int32)
    segment_fraction = np.full((height, width), np.nan, dtype=np.float64)
    normal_coordinate = np.full((height, width), np.nan, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)[:, None]

    for column in range(width):
        x = float(column)
        candidates = np.flatnonzero(
            (segment_min_x <= x + radius) & (segment_max_x >= x - radius)
        )
        if candidates.size == 0:
            continue

        candidate_starts = starts[candidates]
        candidate_vectors = vectors[candidates]
        candidate_lengths = squared_lengths[candidates]
        delta_x = x - candidate_starts[:, 0]
        delta_y = rows - candidate_starts[None, :, 1]
        projection = (
            delta_x[None, :] * candidate_vectors[None, :, 0]
            + delta_y * candidate_vectors[None, :, 1]
        ) / candidate_lengths[None, :]
        projection = np.clip(projection, 0.0, 1.0)
        closest_x = (
            candidate_starts[None, :, 0]
            + projection * candidate_vectors[None, :, 0]
        )
        closest_y = (
            candidate_starts[None, :, 1]
            + projection * candidate_vectors[None, :, 1]
        )
        residual_x = x - closest_x
        residual_y = rows - closest_y
        squared_distance = residual_x * residual_x + residual_y * residual_y

        local_choice = np.argmin(squared_distance, axis=1)
        row_index = np.arange(height)
        best_squared = squared_distance[row_index, local_choice]
        best_distance = np.sqrt(best_squared)
        inside = best_distance < radius
        if not np.any(inside):
            continue

        chosen_candidates = candidates[local_choice]
        chosen_projection = projection[row_index, local_choice]
        chosen_vectors = vectors[chosen_candidates]
        chosen_norm = np.sqrt(squared_lengths[chosen_candidates])
        chosen_residual_x = residual_x[row_index, local_choice]
        chosen_residual_y = residual_y[row_index, local_choice]
        signed_normal = (
            chosen_vectors[:, 0] * chosen_residual_y
            - chosen_vectors[:, 1] * chosen_residual_x
        ) / chosen_norm

        clipped_distance[inside, column] = best_distance[inside]
        nearest_segment[inside, column] = chosen_candidates[inside]
        segment_fraction[inside, column] = chosen_projection[inside]
        normal_coordinate[inside, column] = signed_normal[inside]

    kernel = np.maximum(1.0 - clipped_distance / radius, 0.0)
    time_coordinate = np.full((height, width), np.nan, dtype=np.float64)
    support = nearest_segment >= 0
    time_coordinate[support] = (
        nearest_segment[support].astype(np.float64) + segment_fraction[support]
    )

    return TraceGeometry(
        image_size=(height, width),
        window_length=int(points.shape[0]),
        bandwidth=radius,
        phase_shift=shift,
        vertices=points.astype(np.float32),
        distance=clipped_distance.astype(np.float32),
        kernel=kernel.astype(np.float32),
        nearest_segment=nearest_segment,
        segment_fraction=segment_fraction.astype(np.float32),
        time_coordinate=time_coordinate.astype(np.float32),
        normal_coordinate=normal_coordinate.astype(np.float32),
    )


def build_trace_geometry(
    values: Sequence[float] | np.ndarray,
    *,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    value_range: Tuple[float, float] = (0.0, 1.0),
    phase_shift: float | Sequence[float] = (0.0, 0.0),
    bandwidth: float = DEFAULT_BANDWIDTH,
    expected_length: int = DEFAULT_WINDOW_LENGTH,
) -> TraceGeometry:
    """Build vertices and their truncated pixel-distance geometry."""

    shift = normalize_phase_shift(phase_shift)
    vertices = trace_vertices(
        values,
        image_size=image_size,
        value_range=value_range,
        phase_shift=shift,
        expected_length=expected_length,
    )
    return continuous_polyline_geometry(
        vertices,
        image_size=image_size,
        bandwidth=bandwidth,
        phase_shift=shift,
    )


def vertex_unit_normals(vertices: np.ndarray) -> np.ndarray:
    """Return stable unit normals at all polyline vertices.

    Endpoint tangents use their single incident segment.  Interior tangents
    use the sum of adjacent unit tangents, falling back to the forward tangent
    for an exact 180-degree reversal.  Normals follow ``(-t_y, t_x)`` in image
    coordinates.
    """

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("vertices must have shape [T, 2] with T >= 2")
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("consecutive polyline vertices must be distinct")
    unit = segments / lengths[:, None]
    tangents = np.empty_like(points)
    tangents[0] = unit[0]
    tangents[-1] = unit[-1]
    if points.shape[0] > 2:
        interior = unit[:-1] + unit[1:]
        interior_norm = np.linalg.norm(interior, axis=1)
        degenerate = interior_norm <= np.finfo(np.float64).eps
        interior[~degenerate] /= interior_norm[~degenerate, None]
        interior[degenerate] = unit[1:][degenerate]
        tangents[1:-1] = interior
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    return normals.astype(np.float32)


def normal_sampling_coordinates(
    vertices: np.ndarray,
    offsets: Iterable[float],
) -> np.ndarray:
    """Generate trace-normal sample coordinates for a later pullback reducer.

    Parameters
    ----------
    vertices:
        Polyline coordinates with shape ``[T, 2]``.
    offsets:
        Signed distances in pixels from each vertex.

    Returns
    -------
    np.ndarray
        Coordinates with shape ``[T, N, 2]`` in ``(x, y)`` order.  Coordinates
        are intentionally not clipped; the downstream sampler can record the
        exact mass lost at image boundaries.
    """

    points = np.asarray(vertices, dtype=np.float64)
    offset_array = np.asarray(tuple(offsets), dtype=np.float64)
    if offset_array.ndim != 1 or offset_array.size == 0:
        raise ValueError("offsets must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(offset_array)):
        raise ValueError("offsets must be finite")
    normals = vertex_unit_normals(points).astype(np.float64)
    coordinates = points[:, None, :] + normals[:, None, :] * offset_array[None, :, None]
    return coordinates.astype(np.float32)
