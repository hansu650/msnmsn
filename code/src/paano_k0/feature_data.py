"""Feature-only time-series input and memory-efficient patch access."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch

from .schemas import SeriesSpec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_feature_series(spec: SeriesSpec) -> NDArray[np.float32]:
    """Read only registered feature columns; never materialize the label."""

    path = spec.csv_path.resolve(strict=True)
    if _sha256_file(path) != spec.csv_sha256:
        raise ValueError(f"data SHA256 mismatch for {path.name}")
    columns = list(spec.feature_columns)
    frame = pd.read_csv(path, usecols=columns, dtype=np.float32)
    # pandas may normalize usecols ordering internally; restore the frozen header order.
    frame = frame.loc[:, columns]
    values = np.ascontiguousarray(frame.to_numpy(dtype=np.float32, copy=False))
    if values.shape != (spec.rows, spec.channels):
        raise ValueError(
            f"feature shape mismatch for {spec.series_id}: {values.shape} != "
            f"{(spec.rows, spec.channels)}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite feature value in {spec.series_id}")
    return values


def split_normal_prefix(
    x_full: NDArray[np.float32], train_end: int
) -> NDArray[np.float32]:
    values = np.asarray(x_full)
    if values.dtype != np.float32 or values.ndim != 2:
        raise TypeError("x_full must be float32 [T,C]")
    if not 0 < train_end <= values.shape[0]:
        raise ValueError("train_end outside feature series")
    return np.ascontiguousarray(values[:train_end])


class PatchStore:
    """CPU float32 `[T,C]` series with a stride-one `[N,C,L]` view."""

    def __init__(self, data: torch.Tensor, patch_size: int, stride: int) -> None:
        if data.device.type != "cpu" or data.dtype != torch.float32 or data.ndim != 2:
            raise TypeError("PatchStore data must be a CPU float32 tensor [T,C]")
        if patch_size <= 0 or stride <= 0 or data.shape[0] < patch_size:
            raise ValueError("invalid patch_size/stride for series length")
        self.data = data.contiguous()
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self._patches = (
            self.data.transpose(0, 1)
            .unfold(1, self.patch_size, self.stride)
            .permute(1, 0, 2)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self), int(self.data.shape[1]), self.patch_size)

    def __len__(self) -> int:
        return (int(self.data.shape[0]) - self.patch_size) // self.stride + 1

    def take(self, indices: torch.Tensor | NDArray[np.integer]) -> torch.Tensor:
        starts = torch.as_tensor(indices, dtype=torch.int64, device="cpu")
        if starts.ndim != 1:
            raise ValueError("patch indices must be one-dimensional")
        if starts.numel() and (int(starts.min()) < 0 or int(starts.max()) >= len(self)):
            raise IndexError("patch index outside store")
        return self._patches.index_select(0, starts).contiguous()

    def iter_batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(self)), dtype=torch.int64)
            yield self.take(indices), indices


def make_patch_store(
    x: NDArray[np.float32], patch_size: int, stride: int
) -> PatchStore:
    values = np.asarray(x)
    if values.dtype != np.float32 or values.ndim != 2 or not np.isfinite(values).all():
        raise TypeError("x must be a finite float32 array [T,C]")
    store = PatchStore(torch.from_numpy(np.ascontiguousarray(values)), patch_size, stride)
    first = store.take(torch.tensor([0]))[0].numpy()
    last_index = len(store) - 1
    last_start = last_index * stride
    last = store.take(torch.tensor([last_index]))[0].numpy()
    if not np.array_equal(first, values[:patch_size].T):
        raise RuntimeError("first patch layout verification failed")
    if not np.array_equal(last, values[last_start : last_start + patch_size].T):
        raise RuntimeError("last patch layout verification failed")
    return store

