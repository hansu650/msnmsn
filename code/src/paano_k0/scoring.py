"""PaAno-compatible patch scoring and patch-to-point distribution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .memory import MemoryResult, create_memory_bank


@dataclass(frozen=True)
class ScoreResult:
    patch_scores: np.ndarray
    point_scores: np.ndarray
    memory: MemoryResult
    runtime_seconds: float
    peak_vram_mib: float
    score_sha256: str


def _value(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        return container[name]
    return getattr(container, name)


def _score_sha256(scores: np.ndarray) -> str:
    values = np.ascontiguousarray(scores, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@torch.inference_mode()
def score_patch_store(
    model: nn.Module,
    full_store: Any,
    memory: Tensor,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return mean top-k cosine distance for every full-series patch."""

    if top_k <= 0 or batch_size <= 0:
        raise ValueError("top_k and batch_size must be positive")
    if not isinstance(memory, Tensor) or memory.ndim != 2 or memory.shape[0] < top_k:
        raise ValueError("memory must be [M,D] with M >= top_k")
    model.to(device)
    model.eval()
    normalized_memory = F.normalize(
        memory.to(device=device, dtype=torch.float32), dim=1, eps=1e-12
    )
    scores: list[Tensor] = []
    observed_starts: list[Tensor] = []
    for patches, starts in full_store.iter_batches(batch_size):
        features = model.embedding(
            patches.to(device=device, dtype=torch.float32, non_blocking=True)
        )
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = F.normalize(features, dim=1, eps=1e-12)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        similarities = features @ normalized_memory.T
        similarities = torch.nan_to_num(
            similarities, nan=-1.0, posinf=1.0, neginf=-1.0
        )
        top_similarity = torch.topk(
            similarities, k=top_k, dim=1, largest=True
        ).values
        batch_scores = (1.0 - top_similarity).mean(dim=1)
        batch_scores = torch.nan_to_num(
            batch_scores, nan=1.0, posinf=1.0, neginf=0.0
        )
        scores.append(batch_scores.detach().to(device="cpu", dtype=torch.float32))
        observed_starts.append(starts.detach().to(device="cpu", dtype=torch.int64))
    if not scores:
        raise ValueError("full PatchStore is empty")
    starts = torch.cat(observed_starts).reshape(-1)
    expected_starts = torch.arange(len(full_store), dtype=torch.int64) * int(
        full_store.stride
    )
    if not torch.equal(starts, expected_starts):
        raise RuntimeError("PatchStore scoring order is not chronological")
    result = torch.cat(scores).numpy().astype(np.float32, copy=False)
    if result.shape != (len(full_store),) or not np.isfinite(result).all():
        raise RuntimeError("patch score coverage or finiteness failure")
    return result


def distribute_patch_scores(
    patch_scores: np.ndarray,
    patch_size: int,
    num_points: int,
) -> np.ndarray:
    """Reproduce PaAno's convolutional overlap averaging exactly."""

    if patch_size <= 0 or num_points < patch_size:
        raise ValueError("invalid patch_size/num_points")
    values = np.nan_to_num(
        np.asarray(patch_scores, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if values.ndim != 1:
        raise ValueError("patch_scores must be one-dimensional")
    expected_patches = num_points - patch_size + 1
    if values.shape[0] != expected_patches:
        raise ValueError(
            f"expected {expected_patches} stride-one patch scores, got {values.shape[0]}"
        )
    kernel = np.ones(patch_size, dtype=np.float32)
    sums = np.convolve(values, kernel, mode="full")[:num_points]
    counts = np.convolve(np.ones_like(values), kernel, mode="full")[:num_points]
    point_scores = np.divide(
        sums,
        counts,
        out=np.zeros(num_points, dtype=np.float32),
        where=counts != 0,
    )
    return np.nan_to_num(
        point_scores, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)


def score_checkpoint(
    model: nn.Module,
    checkpoint_state: Mapping[str, Tensor],
    train_store: Any,
    full_store: Any,
    protocol: Any,
    device: torch.device,
) -> ScoreResult:
    """Build a checkpoint-specific memory and commit-ready score vector."""

    hp = _value(protocol, "official_hyperparameters")
    patch_size = int(_value(hp, "patch_size"))
    stride = int(_value(hp, "stride"))
    if stride != 1 or int(full_store.stride) != 1:
        raise ValueError("frozen point distribution requires stride one")
    batch_size = int(_value(hp, "batch_size"))
    requested_fraction = float(_value(hp, "memory_request_fraction"))
    top_k = int(_value(hp, "score_top_k"))
    model.to(device)
    model.load_state_dict(checkpoint_state, strict=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    memory = create_memory_bank(
        model,
        train_store,
        requested_fraction,
        batch_size,
        device,
        random_state=42,
    )
    patch_scores = score_patch_store(
        model, full_store, memory.memory, top_k, batch_size, device
    )
    num_points = (len(full_store) - 1) * stride + patch_size
    point_scores = distribute_patch_scores(patch_scores, patch_size, num_points)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    else:
        peak_vram_mib = 0.0
    runtime_seconds = time.perf_counter() - started
    return ScoreResult(
        patch_scores=patch_scores,
        point_scores=point_scores,
        memory=memory,
        runtime_seconds=runtime_seconds,
        peak_vram_mib=peak_vram_mib,
        score_sha256=_score_sha256(point_scores),
    )


__all__ = [
    "ScoreResult",
    "distribute_patch_scores",
    "score_checkpoint",
    "score_patch_store",
]
