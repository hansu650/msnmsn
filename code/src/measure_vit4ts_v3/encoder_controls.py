"""Mandatory true-global and patch-mean controls for ViTTrace v3.

These controls consume only cached embeddings.  They do not use patch-level
matching, IHP, NCTP, labels, thresholds, or an encoder call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from measure_vit4ts.reducers import stitch_column_vectors
from measure_vit4ts.ritp import stitch_native_240


CONTROL_ARMS = (
    "CTRL_CLS_LEGACY",
    "CTRL_CLS_NATIVE_W240",
    "CTRL_PATCH_MEAN_NATIVE_W240",
)


@dataclass(frozen=True)
class ControlResult:
    score: np.ndarray
    window_scalar: np.ndarray
    metadata: Mapping[str, object]


def median_cosine_scalar(
    embeddings: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Return one frozen float32 cosine anomaly scalar per window."""

    values = np.asarray(embeddings)
    if (
        values.ndim != 2
        or min(values.shape) <= 0
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError("embeddings must be finite canonical float32 [N,D]")
    with torch.inference_mode():
        query = torch.from_numpy(values).to(device)
        memory = torch.median(query, dim=0).values
        normalized_query = F.normalize(query, dim=-1)
        normalized_memory = F.normalize(memory, dim=-1)
        similarity = torch.matmul(normalized_query, normalized_memory).clamp(-1.0, 1.0)
        scalar = (0.5 * (1.0 - similarity)).to(dtype=torch.float64)
    output = np.ascontiguousarray(scalar.cpu().numpy(), dtype=np.float64)
    if output.shape != (values.shape[0],) or not np.isfinite(output).all():
        raise RuntimeError("median cosine control returned invalid window scalars")
    return output


def legacy_scalar_stitch(
    scalar: np.ndarray,
    *,
    full_length: int,
    window: int,
    stride: int,
    legacy_columns: int = 224,
) -> np.ndarray:
    values = np.asarray(scalar, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("window scalar must be finite [N]")
    if int(legacy_columns) <= 0:
        raise ValueError("legacy_columns must be positive")
    columns = np.repeat(values[:, None], int(legacy_columns), axis=1)
    score = stitch_column_vectors(columns, int(full_length), int(window), int(stride))
    return np.ascontiguousarray(score, dtype=np.float64)


def native_w240_scalar_stitch(
    scalar: np.ndarray,
    starts: np.ndarray,
    *,
    full_length: int,
) -> np.ndarray:
    values = np.asarray(scalar, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("window scalar must be finite [N]")
    local = np.repeat(values[:, None], 240, axis=1)
    score = stitch_native_240(local, np.asarray(starts), int(full_length))
    return np.ascontiguousarray(score, dtype=np.float64)


def compute_encoder_controls(
    true_global: np.ndarray,
    patch_mean: np.ndarray,
    starts: np.ndarray,
    *,
    full_length: int,
    window: int,
    stride: int,
    device: torch.device | str = "cpu",
) -> dict[str, ControlResult]:
    """Compute all three registered controls without IHP or NCTP."""

    if int(window) != 240:
        raise ValueError("mandatory native controls require W=240")
    global_values = np.asarray(true_global)
    mean_values = np.asarray(patch_mean)
    if global_values.ndim != 2 or mean_values.ndim != 2:
        raise ValueError("true global and patch mean must each be [N,D]")
    if global_values.shape[0] != mean_values.shape[0]:
        raise ValueError("true global and patch mean window counts differ")
    global_scalar = median_cosine_scalar(global_values, device=device)
    patch_mean_scalar = median_cosine_scalar(mean_values, device=device)
    common = {
        "memory": "cross_window_coordinatewise_median",
        "anomaly": "cosine_cost_0.5x1_minus_cosine",
        "matching": "one_embedding_per_window",
        "ihp": "bypassed",
        "nctp": "bypassed",
        "encoder_calls": 0,
    }
    return {
        "CTRL_CLS_LEGACY": ControlResult(
            legacy_scalar_stitch(
                global_scalar,
                full_length=full_length,
                window=window,
                stride=stride,
            ),
            global_scalar,
            {
                **common,
                "embedding": "openclip_true_global",
                "temporal": "broadcast_scalar_to_224_legacy_columns_then_released_stitch",
            },
        ),
        "CTRL_CLS_NATIVE_W240": ControlResult(
            native_w240_scalar_stitch(
                global_scalar, starts, full_length=full_length
            ),
            global_scalar,
            {
                **common,
                "embedding": "openclip_true_global",
                "temporal": "broadcast_scalar_to_native_W240_then_exact_native_stitch",
            },
        ),
        "CTRL_PATCH_MEAN_NATIVE_W240": ControlResult(
            native_w240_scalar_stitch(
                patch_mean_scalar, starts, full_length=full_length
            ),
            patch_mean_scalar,
            {
                **common,
                "embedding": "mean_of_base_patch_tokens",
                "temporal": "broadcast_scalar_to_native_W240_then_exact_native_stitch",
                "not_cls": True,
            },
        ),
    }


__all__ = [
    "CONTROL_ARMS",
    "ControlResult",
    "compute_encoder_controls",
    "legacy_scalar_stitch",
    "median_cosine_scalar",
    "native_w240_scalar_stitch",
]
