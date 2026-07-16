from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from paano_k0.memory import create_memory_bank, effective_memory_count, encode_store
from paano_k0.scoring import distribute_patch_scores, score_patch_store


class FixtureStore:
    def __init__(self, patches: torch.Tensor, stride: int = 1) -> None:
        self.patches = patches.to(torch.float32).contiguous()
        self.stride = stride

    def __len__(self) -> int:
        return self.patches.shape[0]

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            indices = torch.arange(start, stop, dtype=torch.int64) * self.stride
            yield self.patches[start:stop], indices


class FixtureEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))

    def embedding(self, patches: torch.Tensor) -> torch.Tensor:
        flattened = patches.flatten(1)
        return torch.stack(
            [flattened.mean(dim=1), flattened[:, 0], flattened[:, -1]], dim=1
        ) * self.scale


def _patches(count: int = 8) -> torch.Tensor:
    values = torch.arange(count * 4, dtype=torch.float32).reshape(count, 1, 4)
    return (values + 1.0) / 10.0


def test_memory_count_and_centers_parity() -> None:
    model = FixtureEncoder()
    store = FixtureStore(_patches())
    actual = create_memory_bank(
        model,
        store,
        requested_fraction=0.1,
        batch_size=3,
        device=torch.device("cpu"),
        random_state=42,
    )

    embeddings, starts = encode_store(model, store, 3, torch.device("cpu"))
    count = effective_memory_count(len(store), 0.1)
    normalized = F.normalize(embeddings, p=2, dim=1)
    estimator = MiniBatchKMeans(
        n_clusters=count,
        init="k-means++",
        random_state=42,
        batch_size=max(8192, count),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    )
    estimator.fit(normalized.numpy())
    centers = torch.tensor(estimator.cluster_centers_, dtype=normalized.dtype)
    expected_rows = torch.argmin(torch.cdist(normalized, centers, p=2), dim=0)
    expected_memory = embeddings[expected_rows]
    expected_indices = starts[expected_rows]

    assert actual.memory_count == count == 7
    assert actual.effective_fraction == count / len(store)
    torch.testing.assert_close(actual.memory, expected_memory, atol=0, rtol=0)
    assert torch.equal(actual.source_indices, expected_indices)


def test_score_and_distribution_parity() -> None:
    model = FixtureEncoder()
    store = FixtureStore(_patches(6))
    memory = model.embedding(_patches(4))
    actual_patch = score_patch_store(
        model,
        store,
        memory,
        top_k=3,
        batch_size=2,
        device=torch.device("cpu"),
    )

    normalized_memory = F.normalize(memory.to(torch.float32), dim=1, eps=1e-12)
    features = model.embedding(store.patches)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = F.normalize(features, dim=1, eps=1e-12)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    similarities = torch.nan_to_num(
        features @ normalized_memory.T, nan=-1.0, posinf=1.0, neginf=-1.0
    )
    expected_patch = (
        1.0 - torch.topk(similarities, k=3, dim=1, largest=True).values
    ).mean(dim=1)
    expected_patch = torch.nan_to_num(
        expected_patch, nan=1.0, posinf=1.0, neginf=0.0
    ).numpy()
    np.testing.assert_allclose(actual_patch, expected_patch, atol=1e-6, rtol=0)

    patch_scores = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    actual_point = distribute_patch_scores(patch_scores, patch_size=3, num_points=6)
    kernel = np.ones(3, dtype=np.float32)
    sums = np.convolve(patch_scores, kernel, mode="full")[:6]
    counts = np.convolve(np.ones_like(patch_scores), kernel, mode="full")[:6]
    expected_point = np.divide(
        sums,
        counts,
        out=np.zeros(6, dtype=np.float32),
        where=counts != 0,
    )
    np.testing.assert_allclose(actual_point, expected_point, atol=1e-6, rtol=0)


def test_effective_memory_minimum_rule() -> None:
    assert effective_memory_count(1, 0.1) == 1
    assert effective_memory_count(8, 0.1) == 7
    assert effective_memory_count(1000, 0.1) == 500
    assert effective_memory_count(10000, 0.1) == 1000
