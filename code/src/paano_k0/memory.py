"""PaAno-compatible normal-memory construction with explicit audit fields."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True)
class MemoryResult:
    memory: Tensor
    source_indices: Tensor
    requested_fraction: float
    effective_fraction: float
    memory_count: int
    total_embeddings: int
    sha256: str


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@torch.inference_mode()
def encode_store(
    model: nn.Module,
    store: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Encode a PatchStore in deterministic chronological order on CPU."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(store) <= 0:
        raise ValueError("PatchStore is empty")
    model.to(device)
    model.eval()
    embeddings: list[Tensor] = []
    starts: list[Tensor] = []
    for patches, patch_starts in store.iter_batches(batch_size):
        if patches.ndim != 3:
            raise ValueError("PatchStore batches must have shape [B,C,L]")
        output = model.embedding(
            patches.to(device=device, dtype=torch.float32, non_blocking=True)
        )
        if output.ndim != 2 or output.shape[0] != patches.shape[0]:
            raise ValueError("encoder must emit [B,D]")
        if not torch.isfinite(output).all():
            raise FloatingPointError("encoder emitted NaN or Inf while building memory")
        embeddings.append(output.detach().to(device="cpu", dtype=torch.float32))
        starts.append(patch_starts.detach().to(device="cpu", dtype=torch.int64))
    result = torch.cat(embeddings, dim=0).contiguous()
    indices = torch.cat(starts, dim=0).reshape(-1).contiguous()
    if result.shape[0] != len(store) or indices.shape != (len(store),):
        raise RuntimeError("encoded store coverage mismatch")
    return result, indices


def effective_memory_count(n: int, requested_fraction: float) -> int:
    """Reproduce the released 10%-request/minimum-500 count rule exactly."""

    if n <= 0:
        raise ValueError("n must be positive")
    if (
        not math.isfinite(requested_fraction)
        or requested_fraction <= 0
        or requested_fraction > 1
    ):
        raise ValueError("requested_fraction must be in (0,1]")
    requested = int(round(requested_fraction * n))
    minimum = min(500, max(1, n - 1))
    return int(max(minimum, min(requested, n - 1)))


def create_memory_bank(
    model: nn.Module,
    train_store: Any,
    requested_fraction: float,
    batch_size: int,
    device: torch.device,
    random_state: int = 42,
) -> MemoryResult:
    """Encode, cluster normalized features, and keep nearest real exemplars."""

    embeddings, indices = encode_store(model, train_store, batch_size, device)
    num_samples = embeddings.shape[0]
    count = effective_memory_count(num_samples, requested_fraction)
    if count >= num_samples:
        memory = embeddings
        selected_indices = indices
    else:
        flattened = embeddings.reshape(num_samples, -1)
        normalized = F.normalize(flattened, p=2, dim=1)
        estimator = MiniBatchKMeans(
            n_clusters=count,
            init="k-means++",
            random_state=int(random_state),
            batch_size=max(8192, count),
            max_iter=50,
            n_init=1,
            reassignment_ratio=0.01,
        )
        estimator.fit(normalized.numpy())
        centers = torch.tensor(
            estimator.cluster_centers_, dtype=normalized.dtype, device="cpu"
        )
        distances = torch.cdist(normalized, centers, p=2)
        exemplar_rows = torch.argmin(distances, dim=0)
        memory = embeddings[exemplar_rows].contiguous()
        selected_indices = indices[exemplar_rows].contiguous()
    if memory.ndim != 2 or memory.shape[0] != count:
        raise RuntimeError("memory shape does not match effective count")
    if not torch.isfinite(memory).all():
        raise FloatingPointError("memory contains NaN or Inf")
    return MemoryResult(
        memory=memory,
        source_indices=selected_indices,
        requested_fraction=float(requested_fraction),
        effective_fraction=float(count / num_samples),
        memory_count=count,
        total_embeddings=num_samples,
        sha256=_tensor_sha256(memory),
    )


__all__ = [
    "MemoryResult",
    "create_memory_bank",
    "effective_memory_count",
    "encode_store",
]
