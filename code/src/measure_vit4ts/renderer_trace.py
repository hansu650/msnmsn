"""Renderer-owned coverage and coordinates for ViTTrace.

The frozen VLM4TS path renders a 240-sample window directly to a
``224 x 224`` black-on-white RGB image.  There is no intermediate 240-pixel
canvas and no 240-to-224 resize in this contract.  This module therefore only
recovers effective line coverage from the actual RGB tensor and binds it to
the continuous timestamp coordinate computed in that same 224-pixel frame.

No labels, anomaly scores, model weights, or filesystem state are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import TraceGeometry


IMAGE_SHAPE = (224, 224)
WINDOW_LENGTH = 240
PATCH_GRID = (14, 14)
PATCH_SHAPE = (16, 16)


def _channels_first_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return a finite float64 RGB tensor with shape ``[3, 224, 224]``."""

    array = np.asarray(rgb)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError("rgb must be a real numeric array")
    if array.shape == (3, *IMAGE_SHAPE):
        channels_first = array
    elif array.shape == (*IMAGE_SHAPE, 3):
        channels_first = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(
            "rgb must have shape [3,224,224] or [224,224,3]"
        )
    value = np.asarray(channels_first, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("rgb must contain only finite values")
    return value


def effective_alpha(rgb: np.ndarray) -> np.ndarray:
    """Recover scalar black-line coverage from a white-background RGB image.

    For black composited over white, every observed channel is ``1 - alpha``.
    The three clipped channel estimates are averaged only to collapse the RGB
    axis; this is exact for the frozen grayscale renderer and remains bounded
    for a numerically perturbed input.
    """

    channels = _channels_first_rgb(rgb)
    channel_alpha = np.clip(1.0 - channels, 0.0, 1.0)
    alpha = np.mean(channel_alpha, axis=0, dtype=np.float64)
    return np.ascontiguousarray(alpha, dtype=np.float64)


@dataclass(frozen=True)
class RendererTrace:
    """Canonical effective coverage and continuous timestamp ownership.

    ``time_coordinate`` may be NaN only where ``alpha`` is exactly zero.
    Active coordinates are zero-based and lie in ``[0, window_length - 1]``.
    Both arrays are immutable contiguous float64 values with shape
    ``[224, 224]``.
    """

    alpha: np.ndarray
    time_coordinate: np.ndarray
    window_length: int = WINDOW_LENGTH

    def __post_init__(self) -> None:
        length = int(self.window_length)
        if length <= 1:
            raise ValueError("window_length must be greater than one")

        alpha = np.ascontiguousarray(np.asarray(self.alpha, dtype=np.float64))
        coordinate = np.ascontiguousarray(
            np.asarray(self.time_coordinate, dtype=np.float64)
        )
        if alpha.shape != IMAGE_SHAPE or coordinate.shape != IMAGE_SHAPE:
            raise ValueError("alpha and time_coordinate must both be [224,224]")
        if not np.all(np.isfinite(alpha)):
            raise ValueError("alpha must contain only finite values")
        if np.any(alpha < 0.0) or np.any(alpha > 1.0):
            raise ValueError("alpha must lie in [0,1]")

        active = alpha > 0.0
        if np.any(~np.isfinite(coordinate[active])):
            raise ValueError("every positive-alpha pixel needs a finite coordinate")
        tolerance = 64.0 * np.finfo(np.float32).eps * float(length - 1)
        if np.any(coordinate[active] < -tolerance) or np.any(
            coordinate[active] > float(length - 1) + tolerance
        ):
            raise ValueError("active time coordinates are outside the window")

        coordinate = coordinate.copy()
        coordinate[active] = np.clip(
            coordinate[active], 0.0, float(length - 1)
        )
        coordinate[~active] = np.nan
        alpha = alpha.copy()
        alpha.setflags(write=False)
        coordinate.setflags(write=False)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "time_coordinate", coordinate)
        object.__setattr__(self, "window_length", length)


def build_renderer_trace(
    rgb: np.ndarray,
    time_coordinate: np.ndarray,
    *,
    window_length: int = WINDOW_LENGTH,
) -> RendererTrace:
    """Bind actual renderer RGB coverage to its 224-frame coordinates."""

    alpha = effective_alpha(rgb)
    coordinate = np.asarray(time_coordinate, dtype=np.float64)
    alpha = alpha.copy()
    alpha[~np.isfinite(coordinate)] = 0.0
    return RendererTrace(
        alpha=alpha,
        time_coordinate=coordinate,
        window_length=window_length,
    )


def renderer_trace_from_geometry(
    rgb: np.ndarray,
    geometry: TraceGeometry,
) -> RendererTrace:
    """Construct a trace from an actual RGB render and matching geometry."""

    if not isinstance(geometry, TraceGeometry):
        raise TypeError("geometry must be a TraceGeometry")
    if tuple(geometry.image_size) != IMAGE_SHAPE:
        raise ValueError("ViTTrace requires the actual 224x224 geometry")
    return build_renderer_trace(
        rgb,
        geometry.time_coordinate,
        window_length=geometry.window_length,
    )


def linear_full_column_coverage(
    timestamp: int,
    *,
    window_length: int = WINDOW_LENGTH,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
) -> np.ndarray:
    """Return the parameter-free full-height column control for timestamp ``t``.

    Timestamp ``t`` owns continuous image column
    ``x_t = (width - 1) * t / (window_length - 1)``.  Its unit coverage is
    split linearly between the floor and ceil columns and repeated over the
    complete image height.
    """

    if isinstance(timestamp, (bool, np.bool_)) or not isinstance(
        timestamp, (int, np.integer)
    ):
        raise TypeError("timestamp must be an integer")
    length = int(window_length)
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if length <= 1 or height <= 0 or width <= 1:
        raise ValueError("window and image dimensions are invalid")
    index = int(timestamp)
    if index < 0 or index >= length:
        raise ValueError("timestamp is outside the window")

    x_coordinate = float(width - 1) * float(index) / float(length - 1)
    left = int(np.floor(x_coordinate))
    right = min(left + 1, width - 1)
    fraction = x_coordinate - float(left)
    coverage = np.zeros((height, width), dtype=np.float64)
    coverage[:, left] += 1.0 - fraction
    coverage[:, right] += fraction
    return coverage


__all__ = [
    "IMAGE_SHAPE",
    "PATCH_GRID",
    "PATCH_SHAPE",
    "WINDOW_LENGTH",
    "RendererTrace",
    "build_renderer_trace",
    "effective_alpha",
    "linear_full_column_coverage",
    "renderer_trace_from_geometry",
]
