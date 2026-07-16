"""Deterministic model initialization and cross-arm minibatch replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Callable, Mapping
import uuid

import numpy as np
from numpy.typing import NDArray
import torch

from .schemas import IterationReplay, ReplayPlan


@dataclass(frozen=True, slots=True)
class ReplayIdentity:
    series_id: str
    seed: int
    n_train_patches: int
    batch_size: int
    num_iterations: int


def stable_seed(base_seed: int, series_sha256: str, namespace: str) -> int:
    if base_seed < 0 or not namespace:
        raise ValueError("base_seed must be non-negative and namespace non-empty")
    payload = f"{base_seed}\0{series_sha256}\0{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        # Flatten first because PyTorch does not permit a zero-dimensional
        # integral tensor (for example BatchNorm.num_batches_tracked) to be
        # reinterpreted as bytes directly.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_initial_state(
    build_fn: Callable[[], torch.nn.Module], init_seed: int
) -> tuple[dict[str, torch.Tensor], str]:
    seed_everything(init_seed, deterministic=True)
    model = build_fn()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_fn must return torch.nn.Module")
    state = {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in model.state_dict().items()
    }
    return state, state_dict_sha256(state)


def _plan_payload_sha256(
    series_id: str,
    seed: int,
    n_patches: int,
    batch_size: int,
    records: tuple[IterationReplay, ...],
) -> str:
    digest = hashlib.sha256()
    identity = {
        "series_id": series_id,
        "seed": seed,
        "n_train_patches": n_patches,
        "batch_size": batch_size,
        "num_iterations": len(records),
    }
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for step in records:
        for array in (step.anchor_indices, step.positive_uniform, step.unadjacent_uniform):
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


_FLOAT32_BELOW_ONE = np.nextafter(np.float32(1.0), np.float32(0.0))


def _store_unit_uniforms(draws: NDArray[np.float64]) -> NDArray[np.float32]:
    """Store float64 draws in float32 without closing the [0, 1) interval.

    A valid float64 draw in the upper half-ULP below one can round to exactly
    ``1.0`` when cast to float32.  Canonicalizing only that representation to
    the greatest float32 below one preserves the original RNG stream and the
    intended final candidate ordinal.
    """

    stored = np.ascontiguousarray(draws, dtype=np.float32)
    np.minimum(stored, _FLOAT32_BELOW_ONE, out=stored)
    return stored


def build_replay_plan(
    n_patches: int,
    batch_size: int,
    iterations: int,
    seed: int,
    num_unadjacent: int = 5,
    *,
    series_id: str = "",
) -> ReplayPlan:
    if n_patches < 2 or batch_size < 2 or iterations <= 0 or seed < 0:
        raise ValueError("replay requires >=2 patches/batch and positive iterations")
    if num_unadjacent <= 0:
        raise ValueError("num_unadjacent must be positive")
    rng = np.random.default_rng(seed)
    records: list[IterationReplay] = []
    while len(records) < iterations:
        permutation = rng.permutation(n_patches).astype(np.int64, copy=False)
        for start in range(0, n_patches, batch_size):
            if len(records) >= iterations:
                break
            anchors = np.ascontiguousarray(permutation[start : start + batch_size], dtype=np.int64)
            if anchors.size < 2:
                # A singleton minibatch cannot reproduce PaAno's non-self pretext task.
                # Merge it into a fresh epoch rather than inventing a self-negative.
                continue
            positive = _store_unit_uniforms(rng.random(anchors.size))
            unadjacent = _store_unit_uniforms(
                rng.random((anchors.size, num_unadjacent))
            )
            records.append(IterationReplay(anchors, positive, unadjacent))
    frozen = tuple(records)
    payload_hash = _plan_payload_sha256(series_id, seed, n_patches, batch_size, frozen)
    return ReplayPlan(
        series_id=series_id,
        seed=seed,
        n_train_patches=n_patches,
        batch_size=batch_size,
        iterations=frozen,
        payload_sha256=payload_hash,
    )


def materialize_positive_indices(
    anchor_indices: NDArray[np.int64],
    n_patches: int,
    offsets: tuple[int, ...],
    uniform: NDArray[np.float32],
) -> NDArray[np.int64]:
    anchors = np.asarray(anchor_indices)
    draws = np.asarray(uniform)
    if anchors.dtype != np.int64 or anchors.ndim != 1:
        raise TypeError("anchor_indices must be int64 [B]")
    if draws.dtype != np.float32 or draws.shape != anchors.shape:
        raise TypeError("uniform must be float32 [B]")
    if not offsets or 0 in offsets or len(set(offsets)) != len(offsets):
        raise ValueError("positive offsets must be unique, nonzero, and non-empty")
    if np.any((draws < 0) | (draws >= 1)):
        raise ValueError("uniform draws must lie in [0,1)")
    result = np.empty_like(anchors)
    for row, (anchor, draw) in enumerate(zip(anchors.tolist(), draws.tolist(), strict=True)):
        candidates = [anchor + offset for offset in offsets if 0 <= anchor + offset < n_patches]
        if not candidates:
            raise ValueError(f"anchor {anchor} has no valid positive candidate")
        ordinal = min(int(float(draw) * len(candidates)), len(candidates) - 1)
        result[row] = candidates[ordinal]
    selected_offsets = result - anchors
    if set(offsets) == {-96, 96}:
        if not np.all(np.abs(selected_offsets) == 96):
            raise AssertionError("non-overlap arm did not preserve a 96-step offset")
    return result


def materialize_unadjacent_indices(
    batch_size: int, uniform: NDArray[np.float32]
) -> NDArray[np.int64]:
    draws = np.asarray(uniform)
    if batch_size < 2 or draws.dtype != np.float32 or draws.ndim != 2:
        raise TypeError("uniform must be float32 [B,K] with B>=2")
    if draws.shape[0] != batch_size:
        raise ValueError("uniform leading dimension must equal batch_size")
    if np.any((draws < 0) | (draws >= 1)):
        raise ValueError("uniform draws must lie in [0,1)")
    offsets = np.floor(draws.astype(np.float64) * (batch_size - 1)).astype(np.int64) + 1
    rows = np.arange(batch_size, dtype=np.int64)[:, None]
    indices = (rows + offsets) % batch_size
    if np.any(indices == rows):
        raise AssertionError("unadjacent replay produced a self-pair")
    return np.ascontiguousarray(indices)


def _paths(path: Path) -> tuple[Path, Path]:
    base = Path(path)
    npz_path = base if base.suffix.lower() == ".npz" else base.with_suffix(".npz")
    return npz_path, npz_path.with_suffix(".json")


def save_replay_plan(plan: ReplayPlan, path: Path) -> str:
    npz_path, json_path = _paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    expected = _plan_payload_sha256(
        plan.series_id,
        plan.seed,
        plan.n_train_patches,
        plan.batch_size,
        plan.iterations,
    )
    if expected != plan.payload_sha256:
        raise ValueError("ReplayPlan payload hash is invalid")
    arrays: dict[str, np.ndarray] = {}
    for index, step in enumerate(plan.iterations):
        arrays[f"anchor_{index:03d}"] = step.anchor_indices
        arrays[f"positive_{index:03d}"] = step.positive_uniform
        arrays[f"unadjacent_{index:03d}"] = step.unadjacent_uniform
    temp_npz = npz_path.with_name(f".{npz_path.name}.{uuid.uuid4().hex}.tmp")
    temp_json = json_path.with_name(f".{json_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        metadata = {
            "series_id": plan.series_id,
            "seed": plan.seed,
            "n_train_patches": plan.n_train_patches,
            "batch_size": plan.batch_size,
            "num_iterations": len(plan.iterations),
            "payload_sha256": plan.payload_sha256,
            "npz_sha256": hashlib.sha256(temp_npz.read_bytes()).hexdigest(),
        }
        with temp_json.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_npz, npz_path)
        os.replace(temp_json, json_path)
    finally:
        temp_npz.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)
    return plan.payload_sha256


def load_replay_plan(
    path: Path, expected: ReplayIdentity | ReplayPlan | None = None
) -> ReplayPlan:
    npz_path, json_path = _paths(path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if hashlib.sha256(npz_path.read_bytes()).hexdigest() != metadata["npz_sha256"]:
        raise ValueError("replay NPZ file hash mismatch")
    count = int(metadata["num_iterations"])
    records: list[IterationReplay] = []
    with np.load(npz_path, allow_pickle=False) as payload:
        expected_keys = {
            f"{prefix}_{index:03d}"
            for index in range(count)
            for prefix in ("anchor", "positive", "unadjacent")
        }
        if set(payload.files) != expected_keys:
            raise ValueError("replay NPZ schema mismatch")
        for index in range(count):
            records.append(
                IterationReplay(
                    np.ascontiguousarray(payload[f"anchor_{index:03d}"], dtype=np.int64),
                    np.ascontiguousarray(payload[f"positive_{index:03d}"], dtype=np.float32),
                    np.ascontiguousarray(payload[f"unadjacent_{index:03d}"], dtype=np.float32),
                )
            )
    plan = ReplayPlan(
        series_id=str(metadata["series_id"]),
        seed=int(metadata["seed"]),
        n_train_patches=int(metadata["n_train_patches"]),
        batch_size=int(metadata["batch_size"]),
        iterations=tuple(records),
        payload_sha256=str(metadata["payload_sha256"]),
    )
    actual_payload_hash = _plan_payload_sha256(
        plan.series_id,
        plan.seed,
        plan.n_train_patches,
        plan.batch_size,
        plan.iterations,
    )
    if actual_payload_hash != plan.payload_sha256:
        raise ValueError("replay payload SHA256 mismatch")
    if expected is not None:
        expected_identity = (
            ReplayIdentity(
                expected.series_id,
                expected.seed,
                expected.n_train_patches,
                expected.batch_size,
                len(expected.iterations),
            )
            if isinstance(expected, ReplayPlan)
            else expected
        )
        observed = ReplayIdentity(
            plan.series_id,
            plan.seed,
            plan.n_train_patches,
            plan.batch_size,
            len(plan.iterations),
        )
        if observed != expected_identity:
            raise ValueError(f"replay identity mismatch: {observed} != {expected_identity}")
    return plan
