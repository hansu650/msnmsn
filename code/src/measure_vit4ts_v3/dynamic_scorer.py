"""Pure cache-only REL/IHP/FULL scoring for dynamic ViTTrace grids."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from measure_vit4ts.reducers import stitch_column_vectors

from .core import (
    apply_temporal_operator,
    build_candidate_mask,
    build_linear_nctp,
    column_top_fraction_mean,
    harmonic_incidence_projection,
    literal_incidence,
    released_incidence,
    streamed_median_reference_match,
    validate_patch_grid,
)
from .dynamic_cache import DynamicTokenCache


DYNAMIC_SCORE_ARMS = ("REL", "IHP", "FULL")
TOKEN_FIELD_BY_SCALE = {
    "P": "patch_tokens",
    "M": "mid_tokens",
    "L": "large_tokens",
}
MASK_FIELD_BY_SCALE = {"M": "mid_mask", "L": "large_mask"}
ARM_PARAMETERS: dict[str, dict[str, Any]] = {
    "REL": {
        "matching_scope": "global",
        "memory": "median_reference",
        "scales": "PML",
        "incidence": "released",
        "fusion": "legacy_intersection",
        "temporal": "legacy",
        "reducer_kind": "top_fraction",
        "reducer_value": 0.25,
    },
    "IHP": {
        "matching_scope": "global",
        "memory": "median_reference",
        "scales": "PML",
        "incidence": "literal",
        "fusion": "legacy_intersection",
        "temporal": "legacy",
        "reducer_kind": "top_fraction",
        "reducer_value": 0.25,
    },
    "FULL": {
        "matching_scope": "global",
        "memory": "median_reference",
        "scales": "PML",
        "incidence": "literal",
        "fusion": "legacy_intersection",
        "temporal": "nctp_linear",
        "reducer_kind": None,
        "reducer_value": None,
    },
}


@dataclass(frozen=True)
class DynamicArmScore:
    score: np.ndarray
    window_field: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        score = np.asarray(self.score)
        field = np.asarray(self.window_field)
        if score.ndim != 1 or score.dtype != np.float64 or not np.isfinite(score).all():
            raise ValueError("dynamic score must be a finite float64 [T] vector")
        if field.ndim != 2 or field.dtype != np.float64 or not np.isfinite(field).all():
            raise ValueError("dynamic window field must be finite float64 [N,K]")


@dataclass(frozen=True)
class DynamicScoreBundle:
    arms: Mapping[str, DynamicArmScore]
    shared_matching_seconds: float
    arm_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        if tuple(self.arms) != DYNAMIC_SCORE_ARMS:
            raise ValueError("dynamic score arm registry changed")
        if set(self.arm_seconds) != set(DYNAMIC_SCORE_ARMS):
            raise ValueError("dynamic arm runtime registry changed")
        values = (self.shared_matching_seconds, *self.arm_seconds.values())
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("dynamic runtimes must be finite and non-negative")


def arm_parameter_sha256(arm: str) -> str:
    if arm not in ARM_PARAMETERS:
        raise KeyError(f"unknown dynamic score arm: {arm}")
    payload = json.dumps(
        ARM_PARAMETERS[arm], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def stitch_native_dynamic(
    window_scores: np.ndarray,
    window_starts: np.ndarray,
    full_length: int,
) -> np.ndarray:
    """Generalized frozen native stitch for any registered window length."""

    scores = np.asarray(window_scores, dtype=np.float64)
    starts_raw = np.asarray(window_starts)
    length = int(full_length)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("window_scores must be a nonempty [N,W] array")
    if not np.isfinite(scores).all():
        raise ValueError("window_scores must be finite")
    if starts_raw.shape != (scores.shape[0],):
        raise ValueError("window_starts must have shape [N]")
    if not np.issubdtype(starts_raw.dtype, np.integer) or np.issubdtype(
        starts_raw.dtype, np.bool_
    ):
        raise TypeError("window_starts must contain integers")
    starts = starts_raw.astype(np.int64, copy=False)
    width = int(scores.shape[1])
    if length <= 0 or np.any(starts < 0) or np.any(starts + width > length):
        raise ValueError("native windows must fit inside positive full_length")
    if starts.size > 1 and np.any(np.diff(starts) <= 0):
        raise ValueError("window_starts must be strictly increasing")

    summed = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.int64)
    for index, start in enumerate(starts):
        stop = int(start) + width
        summed[int(start) : stop] += scores[index]
        count[int(start) : stop] += 1
    supported = count > 0
    if not supported[0]:
        raise ValueError("native stitch has an uncovered prefix")
    last = int(np.flatnonzero(supported)[-1])
    if np.any(~supported[: last + 1]):
        raise ValueError("native stitch has an interior coverage hole")
    output = np.empty(length, dtype=np.float64)
    output[: last + 1] = summed[: last + 1] / count[: last + 1]
    if last + 1 < length:
        slope = output[last] - output[last - 1] if last > 0 else 0.0
        output[last + 1 :] = output[last] + slope * np.arange(
            1, length - last, dtype=np.float64
        )
    if not np.isfinite(output).all():
        raise RuntimeError("native stitch produced a non-finite score")
    return np.ascontiguousarray(output)


def _scale_grids(cache: DynamicTokenCache) -> dict[str, tuple[int, int]]:
    grid_h, grid_w = validate_patch_grid(cache.patch_grid)
    if grid_h < 3 or grid_w < 3:
        raise ValueError("P/M/L scoring requires a base patch grid of at least 3x3")
    return {
        "P": (grid_h, grid_w),
        "M": (grid_h - 1, grid_w - 1),
        "L": (grid_h - 2, grid_w - 2),
    }


def _validate_cache_geometry(cache: DynamicTokenCache) -> dict[str, tuple[int, int]]:
    if not isinstance(cache, DynamicTokenCache):
        raise TypeError("cache must be a DynamicTokenCache")
    grids = _scale_grids(cache)
    windows = int(cache.patch_tokens.shape[0])
    feature = int(cache.patch_tokens.shape[2])
    for scale, grid in grids.items():
        array = np.asarray(getattr(cache, TOKEN_FIELD_BY_SCALE[scale]))
        expected = (windows, grid[0] * grid[1], feature)
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"{scale} tokens differ from dynamic grid metadata")
        if not np.isfinite(array).all():
            raise ValueError(f"{scale} tokens contain non-finite values")
    if cache.mid_mask.shape != (4, grids["M"][0] * grids["M"][1]):
        raise ValueError("mid pooling mask shape differs from dynamic grid")
    if cache.large_mask.shape != (9, grids["L"][0] * grids["L"][1]):
        raise ValueError("large pooling mask shape differs from dynamic grid")
    if cache.mid_mask.dtype != np.int64 or cache.large_mask.dtype != np.int64:
        raise ValueError("dynamic pooling masks must be int64")
    return grids


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_global_matching(
    cache: DynamicTokenCache,
    grids: Mapping[str, tuple[int, int]],
    device: torch.device,
    query_chunk_size: int,
) -> tuple[dict[str, torch.Tensor], float]:
    if int(query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    costs: dict[str, torch.Tensor] = {}
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for scale in ("P", "M", "L"):
            tokens = torch.from_numpy(
                np.asarray(getattr(cache, TOKEN_FIELD_BY_SCALE[scale]))
            ).to(device)
            candidates = build_candidate_mask(grids[scale], "global", device=device)
            result = streamed_median_reference_match(
                tokens,
                candidates,
                query_chunk_size=int(query_chunk_size),
            )
            costs[scale] = result.cost.cpu()
            del tokens, candidates, result
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _synchronize(device)
    return costs, time.perf_counter() - started


def _project_fields(
    cache: DynamicTokenCache,
    costs: Mapping[str, torch.Tensor],
    *,
    literal: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cells = int(cache.patch_grid[0]) * int(cache.patch_grid[1])
    fields = {"P": costs["P"]}
    valid = {"P": torch.ones(cells, dtype=torch.bool)}
    for scale in ("M", "L"):
        mask = torch.from_numpy(np.asarray(getattr(cache, MASK_FIELD_BY_SCALE[scale])))
        if literal:
            incidence = literal_incidence(
                mask, cache.patch_grid, require_full_coverage=True
            )
        else:
            incidence = released_incidence(mask, cache.patch_grid)
        fields[scale], valid[scale] = harmonic_incidence_projection(
            costs[scale], incidence
        )
    return fields, valid


def _legacy_fuse(
    fields: Mapping[str, torch.Tensor], valid: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    fused = (
        fields["L"].to(torch.float64) + fields["M"] + fields["P"]
    ) / 3.0
    intersection = valid["P"] & valid["M"] & valid["L"]
    return torch.where(intersection.unsqueeze(0), fused, torch.zeros_like(fused))


def _legacy_timestamp_score(
    field: torch.Tensor,
    *,
    patch_grid: tuple[int, int],
    image_size: tuple[int, int],
    full_length: int,
    window: int,
    stride: int,
    interpolation_chunk: int = 64,
) -> np.ndarray:
    if int(interpolation_chunk) <= 0:
        raise ValueError("interpolation_chunk must be positive")
    vectors: list[np.ndarray] = []
    values = field.reshape(-1, 1, *patch_grid)
    for start in range(0, int(values.shape[0]), int(interpolation_chunk)):
        maps = (
            F.interpolate(
                values[start : start + int(interpolation_chunk)],
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )
        vectors.append(column_top_fraction_mean(maps, 0.25))
    columns = np.concatenate(vectors, axis=0)
    return np.ascontiguousarray(
        stitch_column_vectors(columns, full_length, window, stride),
        dtype=np.float64,
    )


def compute_dynamic_scores(
    cache: DynamicTokenCache,
    *,
    full_length: int,
    window: int,
    stride: int,
    image_size: Sequence[int] = (224, 224),
    device: torch.device | str = "cpu",
    query_chunk_size: int = 32,
) -> DynamicScoreBundle:
    """Compute registered REL/IHP/FULL scores without labels or encoder calls."""

    grids = _validate_cache_geometry(cache)
    length = int(full_length)
    width = int(window)
    step = int(stride)
    image = tuple(int(value) for value in image_size)
    if length <= 0 or width <= 0 or step <= 0 or len(image) != 2 or min(image) <= 0:
        raise ValueError("full_length/window/stride/image_size must be positive")
    if width % step:
        raise ValueError("legacy temporal scoring requires window divisible by stride")
    expected_windows = (length - width) // step + 1
    if length < width or expected_windows != int(cache.patch_tokens.shape[0]):
        raise ValueError("cache window count differs from full_length/window/stride")
    destination = torch.device(device)
    if destination.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring was requested but CUDA is unavailable")

    costs, matching_seconds = _prepare_global_matching(
        cache, grids, destination, int(query_chunk_size)
    )
    released_fields, released_valid = _project_fields(cache, costs, literal=False)
    literal_fields, literal_valid = _project_fields(cache, costs, literal=True)
    released_field = _legacy_fuse(released_fields, released_valid)
    literal_field = _legacy_fuse(literal_fields, literal_valid)
    window_fields = {
        "REL": released_field,
        "IHP": literal_field,
        "FULL": literal_field,
    }
    starts = np.arange(expected_windows, dtype=np.int64) * step
    outputs: dict[str, DynamicArmScore] = {}
    arm_seconds: dict[str, float] = {}
    for arm in DYNAMIC_SCORE_ARMS:
        started = time.perf_counter()
        field = window_fields[arm]
        if arm in ("REL", "IHP"):
            score = _legacy_timestamp_score(
                field,
                patch_grid=cache.patch_grid,
                image_size=image,
                full_length=length,
                window=width,
                stride=step,
            )
        else:
            operator = build_linear_nctp(
                width, cache.patch_grid, image_size=image
            )
            local = apply_temporal_operator(operator, field.numpy())
            score = stitch_native_dynamic(local, starts, length)
        arm_seconds[arm] = time.perf_counter() - started
        outputs[arm] = DynamicArmScore(
            np.ascontiguousarray(score, dtype=np.float64),
            np.ascontiguousarray(field.numpy(), dtype=np.float64),
            dict(ARM_PARAMETERS[arm]),
        )
    return DynamicScoreBundle(outputs, matching_seconds, arm_seconds)


__all__ = [
    "ARM_PARAMETERS",
    "DYNAMIC_SCORE_ARMS",
    "DynamicArmScore",
    "DynamicScoreBundle",
    "arm_parameter_sha256",
    "compute_dynamic_scores",
    "stitch_native_dynamic",
]
