"""Deterministic CPU spectrogram renderer for the isolated ViTTrace v3 route.

The YAML file freezes the STFT family.  This module freezes the remaining
pixel semantics that are necessary to make that family reproducible:

* periodic Hann, one-sided magnitude, and no additional detrending;
* ``spectrum`` magnitude scaling by the Hann-window sum;
* per-window ``log1p`` and min/max normalization;
* linear frequency, displayed high-to-low from the top image row;
* high energy rendered dark on a white background; and
* a pure-NumPy half-pixel bilinear resize to 224 x 224.

There is intentionally no model import or encoder call in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1
SPECTROGRAM_KEYS = {
    "nperseg",
    "noverlap",
    "nfft",
    "window",
    "scaling",
    "frequency_axis",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and contiguous bytes instead of bytes alone."""

    array = np.ascontiguousarray(value)
    header = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(map(int, array.shape))}
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest().upper()


def renderer_source_sha256() -> str:
    """Return the independent source identity of the pixel renderer."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()


@dataclass(frozen=True)
class SpectrogramSpec:
    nperseg: int = 64
    noverlap: int = 48
    nfft: int = 128
    window: str = "hann"
    scaling: str = "spectrum"
    frequency_axis: str = "linear"
    image_size: tuple[int, int] = (224, 224)
    signal_length: int = 240
    detrend: str = "none"
    spectrum_mode: str = "magnitude"
    onesided: bool = True
    intensity: str = "log1p_window_minmax_high_energy_dark"
    resize: str = "bilinear_half_pixel_numpy"

    def __post_init__(self) -> None:
        expected = {
            "nperseg": 64,
            "noverlap": 48,
            "nfft": 128,
            "window": "hann",
            "scaling": "spectrum",
            "frequency_axis": "linear",
            "image_size": (224, 224),
            "signal_length": 240,
            "detrend": "none",
            "spectrum_mode": "magnitude",
            "onesided": True,
            "intensity": "log1p_window_minmax_high_energy_dark",
            "resize": "bilinear_half_pixel_numpy",
        }
        actual = asdict(self)
        actual["image_size"] = tuple(actual["image_size"])
        if actual != expected:
            raise ValueError("spectrogram renderer differs from the frozen v3 contract")

    @property
    def hop(self) -> int:
        return self.nperseg - self.noverlap

    @property
    def frame_count(self) -> int:
        return 1 + (self.signal_length - self.nperseg) // self.hop

    @property
    def frequency_bins(self) -> int:
        return self.nfft // 2 + 1

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_size"] = list(self.image_size)
        return payload


@dataclass(frozen=True)
class SpectrogramRenderBatch:
    images: np.ndarray
    stft_shape: tuple[int, int, int]
    renderer_source_sha256: str
    renderer_config_sha256: str
    image_array_sha256: str
    renderer_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.images, np.ndarray)
            or self.images.dtype != np.float32
            or self.images.ndim != 4
            or tuple(self.images.shape[1:]) != (3, 224, 224)
        ):
            raise ValueError("spectrogram images must be float32 [N,3,224,224]")
        if self.images.shape[0] <= 0 or not np.isfinite(self.images).all():
            raise ValueError("spectrogram images must be nonempty and finite")
        if float(self.images.min()) < 0.0 or float(self.images.max()) > 1.0:
            raise ValueError("spectrogram images must lie in [0,1]")
        if self.stft_shape != (self.images.shape[0], 65, 12):
            raise ValueError("spectrogram STFT shape differs from [N,65,12]")


def spec_from_config(block: Mapping[str, Any]) -> SpectrogramSpec:
    if not isinstance(block, Mapping) or set(block) != SPECTROGRAM_KEYS:
        raise ValueError(
            f"spectrogram config keys must be exactly {sorted(SPECTROGRAM_KEYS)}"
        )
    spec = SpectrogramSpec(
        nperseg=int(block["nperseg"]),
        noverlap=int(block["noverlap"]),
        nfft=int(block["nfft"]),
        window=str(block["window"]),
        scaling=str(block["scaling"]),
        frequency_axis=str(block["frequency_axis"]),
    )
    # __post_init__ owns the exact-value check.
    return spec


def renderer_config_sha256(spec: SpectrogramSpec) -> str:
    return _sha256_bytes(_canonical_json(spec.to_payload()))


def _validate_windows(windows: np.ndarray, spec: SpectrogramSpec) -> np.ndarray:
    raw = np.asarray(windows)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError("windows must be a real numeric array")
    values = np.ascontiguousarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("windows must have nonempty shape [N,240]")
    if values.shape[1] != spec.signal_length:
        raise ValueError("spectrogram route is frozen to window length 240")
    if not np.isfinite(values).all():
        raise ValueError("windows contain non-finite values")
    return values


def periodic_hann(length: int = 64) -> np.ndarray:
    size = int(length)
    if size != 64:
        raise ValueError("spectrogram route is frozen to a length-64 Hann window")
    index = np.arange(size, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * index / float(size))


def stft_magnitude(
    windows: np.ndarray,
    spec: SpectrogramSpec | None = None,
) -> np.ndarray:
    """Return deterministic one-sided spectrum magnitudes ``[N,65,12]``."""

    frozen = spec or SpectrogramSpec()
    values = _validate_windows(windows, frozen)
    frames = np.lib.stride_tricks.sliding_window_view(
        values, frozen.nperseg, axis=1
    )[:, :: frozen.hop, :]
    if frames.shape[1] != frozen.frame_count:
        raise RuntimeError("STFT frame count changed")
    taper = periodic_hann(frozen.nperseg)
    spectrum = np.fft.rfft(frames * taper, n=frozen.nfft, axis=-1)
    magnitude = np.abs(spectrum) / taper.sum(dtype=np.float64)
    output = np.transpose(magnitude, (0, 2, 1))
    expected = (values.shape[0], frozen.frequency_bins, frozen.frame_count)
    if output.shape != expected or not np.isfinite(output).all():
        raise RuntimeError("STFT magnitude has an invalid shape or value")
    return np.ascontiguousarray(output, dtype=np.float64)


def _half_pixel_indices(source: int, target: int) -> tuple[np.ndarray, ...]:
    coordinate = (np.arange(target, dtype=np.float64) + 0.5) * (
        float(source) / float(target)
    ) - 0.5
    coordinate = np.clip(coordinate, 0.0, float(source - 1))
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, source - 1)
    weight = coordinate - lower.astype(np.float64)
    return lower, upper, weight


def resize_bilinear_half_pixel(
    images: np.ndarray,
    output_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Pure-NumPy, separable half-pixel bilinear resize for ``[N,H,W]``."""

    values = np.asarray(images, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError("images must be finite [N,H,W]")
    target_h, target_w = map(int, output_size)
    if (target_h, target_w) != (224, 224):
        raise ValueError("spectrogram output is frozen to 224x224")
    source_h, source_w = map(int, values.shape[-2:])
    x0, x1, wx = _half_pixel_indices(source_w, target_w)
    horizontal = values[:, :, x0] * (1.0 - wx)[None, None, :] + values[
        :, :, x1
    ] * wx[None, None, :]
    y0, y1, wy = _half_pixel_indices(source_h, target_h)
    output = horizontal[:, y0, :] * (1.0 - wy)[None, :, None] + horizontal[
        :, y1, :
    ] * wy[None, :, None]
    return np.ascontiguousarray(output, dtype=np.float64)


def _spectra_to_gray(magnitude: np.ndarray) -> np.ndarray:
    values = np.log1p(np.asarray(magnitude, dtype=np.float64))
    minimum = values.min(axis=(1, 2), keepdims=True)
    maximum = values.max(axis=(1, 2), keepdims=True)
    span = maximum - minimum
    normalized = np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values),
        where=span > 0.0,
    )
    # Linear frequency axis with the highest frequency on the top row.  Dark
    # energy matches the existing white-background/black-ink visual family.
    return np.ascontiguousarray(1.0 - normalized[:, ::-1, :], dtype=np.float64)


def render_spectrogram_windows(
    windows: np.ndarray,
    spec: SpectrogramSpec | None = None,
) -> SpectrogramRenderBatch:
    frozen = spec or SpectrogramSpec()
    magnitude = stft_magnitude(windows, frozen)
    gray = _spectra_to_gray(magnitude)
    resized = resize_bilinear_half_pixel(gray, frozen.image_size)
    images = np.ascontiguousarray(
        np.repeat(resized[:, None, :, :], 3, axis=1), dtype=np.float32
    )
    source_sha = renderer_source_sha256()
    config_sha = renderer_config_sha256(frozen)
    image_sha = array_sha256(images)
    renderer_sha = _sha256_bytes(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "renderer": "spectrogram",
                "renderer_source_sha256": source_sha,
                "renderer_config_sha256": config_sha,
                "image_array_sha256": image_sha,
                "image_shape": list(images.shape),
                "image_dtype": images.dtype.str,
            }
        )
    )
    return SpectrogramRenderBatch(
        images=images,
        stft_shape=tuple(map(int, magnitude.shape)),
        renderer_source_sha256=source_sha,
        renderer_config_sha256=config_sha,
        image_array_sha256=image_sha,
        renderer_sha256=renderer_sha,
    )


__all__ = [
    "SCHEMA_VERSION",
    "SpectrogramRenderBatch",
    "SpectrogramSpec",
    "array_sha256",
    "periodic_hann",
    "render_spectrogram_windows",
    "renderer_config_sha256",
    "renderer_source_sha256",
    "resize_bilinear_half_pixel",
    "spec_from_config",
    "stft_magnitude",
]
