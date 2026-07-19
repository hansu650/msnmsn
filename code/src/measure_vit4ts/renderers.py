"""Trace renderers for the frozen VLM4TS visual front end.

Two deterministic renderers share the same trajectory geometry:

``render_official_trace``
    Reproduces the released Matplotlib path as closely as possible.

``render_patch_normalized_trace``
    Renders the continuous polyline with the fixed truncated linear distance
    kernel used by the measure-consistent interface.

Neither renderer accepts labels, thresholds, anomaly maps, or learned state.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from .geometry import (
    DEFAULT_BANDWIDTH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_WINDOW_LENGTH,
    TraceGeometry,
    build_trace_geometry,
    normalize_phase_shift,
)


@dataclass(frozen=True)
class RenderedTrace:
    """An RGB trace image and the geometry that generated it."""

    image: np.ndarray
    geometry: TraceGeometry
    renderer: str


def _validate_image(image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Return a contiguous float32 ``[3, H, W]`` image in ``[0, 1]``."""

    height, width = image_size
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (3, height, width):
        raise RuntimeError(
            f"renderer produced shape {array.shape}, expected {(3, height, width)}"
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError("renderer produced non-finite pixels")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise RuntimeError("renderer produced pixels outside [0, 1]")
    return np.ascontiguousarray(array)


def render_patch_normalized_trace(
    values: Sequence[float] | np.ndarray,
    *,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    value_range: Tuple[float, float] = (0.0, 1.0),
    phase_shift: float | Sequence[float] = (0.0, 0.0),
    bandwidth: float = DEFAULT_BANDWIDTH,
    expected_length: int = DEFAULT_WINDOW_LENGTH,
) -> RenderedTrace:
    """Render a patch-scale normalized continuous distance field.

    For pixel center ``p`` and trajectory polyline ``Gamma``, the black trace
    coverage is

    ``K_h(p, Gamma) = max(1 - dist(p, Gamma) / h, 0)``.

    The returned white-background image is ``1 - K_h`` in every RGB channel.
    The default ``h=8`` is exactly half of the frozen ViT-B/16 patch width and
    is therefore fixed by model geometry rather than tuned on labels.
    """

    geometry = build_trace_geometry(
        values,
        image_size=image_size,
        value_range=value_range,
        phase_shift=phase_shift,
        bandwidth=bandwidth,
        expected_length=expected_length,
    )
    grayscale = (1.0 - geometry.kernel).astype(np.float32, copy=False)
    image = np.repeat(grayscale[None, :, :], 3, axis=0)
    image = _validate_image(image, geometry.image_size)
    return RenderedTrace(image=image, geometry=geometry, renderer="pndf")


def _shift_axis_limits(
    limits: Tuple[float, float],
    phase_pixels: float,
    pixel_span: int,
    *,
    image_axis_is_downward: bool,
) -> Tuple[float, float]:
    """Translate Matplotlib limits while preserving their data span."""

    low, high = (float(limits[0]), float(limits[1]))
    if phase_pixels == 0.0:
        return low, high
    data_per_pixel = (high - low) / float(max(pixel_span - 1, 1))
    # Moving x limits left moves the trace right.  Image y increases downward,
    # so moving y limits upward in data coordinates moves the trace down.
    direction = 1.0 if image_axis_is_downward else -1.0
    delta = direction * phase_pixels * data_per_pixel
    return low + delta, high + delta


def render_official_trace(
    values: Sequence[float] | np.ndarray,
    *,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    value_range: Tuple[float, float] = (0.0, 1.0),
    phase_shift: float | Sequence[float] = (0.0, 0.0),
    bandwidth: float = DEFAULT_BANDWIDTH,
    expected_length: int = DEFAULT_WINDOW_LENGTH,
    dpi: int = 100,
) -> RenderedTrace:
    """Render the released VLM4TS Matplotlib line image.

    The unshifted path mirrors the vendor defaults used by ``ViT4TS``:
    ``('-', 1, '*', 0.1, 'black', (0, 1))``, no ticks, no subplot margins,
    and a PNG round trip before conversion to a float tensor.  Sub-pixel phase
    shifts are implemented only by translating fixed axis limits, so the line
    style and trajectory scale remain unchanged.
    """

    array = np.asarray(values, dtype=np.float64)
    geometry = build_trace_geometry(
        array,
        image_size=image_size,
        value_range=value_range,
        phase_shift=phase_shift,
        bandwidth=bandwidth,
        expected_length=expected_length,
    )
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    height, width = geometry.image_size
    shift_x, shift_y = normalize_phase_shift(phase_shift)
    time_points = np.arange(array.size, dtype=np.float64)

    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    axes.plot(
        time_points,
        array,
        linestyle="-",
        linewidth=1,
        marker="*",
        markersize=0.1,
        color="black",
    )
    axes.set_ylim(value_range)
    axes.set_xticks([])
    axes.set_yticks([])
    figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    axes.margins(0, 0)

    if shift_x != 0.0:
        axes.set_xlim(
            _shift_axis_limits(
                axes.get_xlim(),
                shift_x,
                width,
                image_axis_is_downward=False,
            )
        )
    if shift_y != 0.0:
        axes.set_ylim(
            _shift_axis_limits(
                axes.get_ylim(),
                shift_y,
                height,
                image_axis_is_downward=True,
            )
        )

    buffer = BytesIO()
    figure.savefig(buffer, format="png", pad_inches=0)
    buffer.seek(0)
    with Image.open(buffer) as pil_image:
        rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    image = np.transpose(rgb, (2, 0, 1))
    image = _validate_image(image, geometry.image_size)
    return RenderedTrace(image=image, geometry=geometry, renderer="official")


def render_supersampled_trace(
    values: Sequence[float] | np.ndarray,
    *,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    value_range: Tuple[float, float] = (0.0, 1.0),
    phase_shift: float | Sequence[float] = (0.0, 0.0),
    bandwidth: float = DEFAULT_BANDWIDTH,
    expected_length: int = DEFAULT_WINDOW_LENGTH,
    scale: int = 4,
    dpi: int = 100,
) -> RenderedTrace:
    """Conventional antialiasing control: high-resolution draw then Lanczos.

    Supersampling keeps the released figure size in physical inches fixed and
    multiplies DPI by ``scale``.  Consequently, the one-point line and marker
    widths receive ``scale`` times as many high-resolution pixels and recover
    their original physical scale after Lanczos downsampling.
    """

    if int(scale) != 4:
        raise ValueError("the registered supersampling control is frozen to scale=4")
    height, width = int(image_size[0]), int(image_size[1])
    dx, dy = normalize_phase_shift(phase_shift)
    high = render_official_trace(
        values,
        image_size=(height * scale, width * scale),
        value_range=value_range,
        phase_shift=(dx * scale, dy * scale),
        bandwidth=bandwidth * scale,
        expected_length=expected_length,
        dpi=dpi * scale,
    )
    high_rgb = np.transpose(high.image, (1, 2, 0))
    high_u8 = np.rint(np.clip(high_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    with Image.fromarray(high_u8, mode="RGB") as pil_image:
        low = pil_image.resize((width, height), resample=Image.Resampling.LANCZOS)
        low_rgb = np.asarray(low, dtype=np.float32) / 255.0
    image = _validate_image(np.transpose(low_rgb, (2, 0, 1)), (height, width))
    # Ownership is rebuilt in the downsampled coordinate system for this
    # renderer; callers must not substitute geometry from another rasterizer.
    geometry = build_trace_geometry(
        values,
        image_size=(height, width),
        value_range=value_range,
        phase_shift=phase_shift,
        bandwidth=bandwidth,
        expected_length=expected_length,
    )
    return RenderedTrace(image=image, geometry=geometry, renderer="supersample4")
