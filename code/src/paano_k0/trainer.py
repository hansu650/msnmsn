"""Deterministic optimized trajectories and the RAND_BN replay control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .instrumentation import IterationRecorder, build_iteration_record
from .objectives import (
    compute_pretext_batch,
    compute_triplet_batch,
    encoder_gradient_diagnostics,
    pretext_weight,
)
from .replay import (
    materialize_positive_indices,
    materialize_unadjacent_indices,
    state_dict_sha256,
)
from .schemas import (
    CheckpointKind,
    ReplayPlan,
    TrainingSummary,
    Trajectory,
)


@dataclass(frozen=True)
class TrainingResult:
    """Complete label-free output from one trajectory."""

    checkpoints: Mapping[CheckpointKind, Mapping[str, Tensor]]
    summary: TrainingSummary
    iteration_log_sha256: str

    @property
    def best_state(self) -> Mapping[str, Tensor] | None:
        return self.checkpoints.get(CheckpointKind.BEST)

    @property
    def last_state(self) -> Mapping[str, Tensor] | None:
        return self.checkpoints.get(CheckpointKind.LAST)


def _clone_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().to(device="cpu").contiguous().clone()
        for key, value in model.state_dict().items()
    }


def _hyperparameters(protocol: Any) -> Any:
    hp = getattr(protocol, "official_hyperparameters", None)
    if hp is None and isinstance(protocol, Mapping):
        hp = protocol.get("official_hyperparameters")
    if hp is None:
        raise TypeError("protocol has no official_hyperparameters")
    return hp


def _value(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        if name not in container:
            raise KeyError(name)
        return container[name]
    return getattr(container, name)


def _trajectory_offsets(protocol: Any, trajectory: Trajectory) -> tuple[int, ...]:
    if hasattr(protocol, "trajectory"):
        registered = protocol.trajectory(trajectory)
        return tuple(int(value) for value in registered.positive_offsets)
    trajectories = _value(protocol, "trajectories")
    for registered in trajectories:
        identity = _value(registered, "id")
        identity = identity.value if isinstance(identity, Trajectory) else str(identity)
        if identity == trajectory.value:
            return tuple(int(value) for value in _value(registered, "positive_offsets"))
    raise ValueError(f"trajectory {trajectory.value} is not registered")


def _encoder_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("projection_head.")
        and not name.startswith("classification_head.")
        and parameter.requires_grad
    )


def _allocated_vram_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0**2))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_pretext_patches(
    store: Any,
    anchor_indices: np.ndarray,
    pretext_step: int,
) -> tuple[Tensor, Tensor]:
    targets = anchor_indices.astype(np.int64, copy=False) - int(pretext_step)
    valid = (targets >= 0) & (targets < len(store))
    clamped = np.clip(targets, 0, len(store) - 1)
    patches = store.take(torch.from_numpy(clamped.astype(np.int64, copy=False))).clone()
    valid_tensor = torch.from_numpy(valid.astype(np.bool_, copy=False))
    if (~valid_tensor).any():
        patches[~valid_tensor] = 0.0
    return patches, valid_tensor


def cosine_learning_rate(
    iteration: int,
    total: int,
    initial_lr: float,
    final_ratio: float = 0.1,
) -> float:
    """Exact released cosine schedule, evaluated at one-based iteration."""

    if total <= 0 or iteration < 0 or iteration > total:
        raise ValueError("iteration must be in [0,total] with total > 0")
    if initial_lr <= 0 or not math.isfinite(initial_lr):
        raise ValueError("initial_lr must be finite and positive")
    if final_ratio <= 0 or final_ratio > 1 or not math.isfinite(final_ratio):
        raise ValueError("final_ratio must be in (0,1]")
    final_lr = initial_lr * final_ratio
    t = min(iteration, total)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t / total))
    return float(final_lr + (initial_lr - final_lr) * cosine_factor)


def train_trajectory(
    model: nn.Module,
    initial_state: Mapping[str, Tensor],
    store: Any,
    replay: ReplayPlan,
    trajectory: Trajectory,
    protocol: Any,
    device: torch.device,
    recorder: IterationRecorder,
) -> TrainingResult:
    """Run one of the three registered optimized trajectories."""

    trajectory = Trajectory(trajectory)
    if trajectory is Trajectory.RAND_BN:
        raise ValueError("use replay_rand_bn for RAND_BN")
    hp = _hyperparameters(protocol)
    total_iterations = int(_value(hp, "iterations"))
    if len(replay.iterations) != total_iterations:
        raise ValueError("replay length differs from the frozen iteration count")
    if replay.n_train_patches != len(store):
        raise ValueError("replay patch count differs from PatchStore")
    margin = float(_value(hp, "margin"))
    divisor = float(_value(hp, "triplet_divisor"))
    initial_lr = float(_value(hp, "learning_rate"))
    weight_decay = float(_value(hp, "weight_decay"))
    pretext_step = int(_value(hp, "pretext_step"))
    patch_size = int(_value(hp, "patch_size"))
    offsets = _trajectory_offsets(protocol, trajectory)

    model.to(device)
    model.load_state_dict(initial_state, strict=True)
    initial_hash = state_dict_sha256(initial_state)
    if state_dict_sha256(model.state_dict()) != initial_hash:
        raise RuntimeError("model did not load the shared initial state exactly")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.ones(1, device=device), reduction="none"
    )
    encoder_parameters = _encoder_parameters(model)
    best_loss = float("inf")
    best_iteration = 0
    best_state: dict[str, Tensor] | None = None
    final_loss_value = float("nan")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    try:
        for iteration, step in enumerate(replay.iterations, start=1):
            iteration_started = time.perf_counter()
            learning_rate = cosine_learning_rate(
                iteration, total_iterations, initial_lr
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)

            # Replay arrays are deliberately immutable.  Copy before exposing
            # them through torch.from_numpy so PyTorch never receives a
            # non-writable buffer (the values and replay hash are unchanged).
            anchor_np = np.array(step.anchor_indices, dtype=np.int64, copy=True)
            positive_np = materialize_positive_indices(
                anchor_np,
                len(store),
                offsets,
                np.asarray(step.positive_uniform, dtype=np.float32),
            )
            anchors = store.take(torch.from_numpy(anchor_np)).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            positives = store.take(torch.from_numpy(positive_np)).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            batch_size = anchors.shape[0]
            if batch_size < 2:
                raise ValueError("triplet training requires at least two anchors")
            lambda_pretext = pretext_weight(iteration, total_iterations)
            pretext_result = None
            if lambda_pretext > 0.0:
                pretext_cpu, valid_cpu = _make_pretext_patches(
                    store, anchor_np, pretext_step
                )
                pretext_patches = pretext_cpu.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                all_patches = torch.cat([anchors, positives, pretext_patches], dim=0)
                embeddings = model.embedding(all_patches)
                h_anchor = embeddings[:batch_size]
                h_positive = embeddings[batch_size : 2 * batch_size]
                h_pretext = embeddings[2 * batch_size :]
                unadjacent_np = materialize_unadjacent_indices(
                    batch_size,
                    np.asarray(step.unadjacent_uniform, dtype=np.float32),
                )
                pretext_result = compute_pretext_batch(
                    model,
                    h_anchor,
                    h_pretext,
                    valid_cpu.to(device=device),
                    torch.from_numpy(unadjacent_np).to(device=device),
                    criterion,
                )
            else:
                embeddings = model.embedding(torch.cat([anchors, positives], dim=0))
                h_anchor = embeddings[:batch_size]
                h_positive = embeddings[batch_size:]

            anchor_tensor = torch.from_numpy(anchor_np).to(device=device)
            positive_tensor = torch.from_numpy(positive_np).to(device=device)
            triplet = compute_triplet_batch(
                model,
                h_anchor,
                h_positive,
                anchor_tensor,
                positive_tensor,
                trajectory,
                margin,
                divisor,
                1.0,
            )
            if pretext_result is None:
                weighted_pretext = triplet.scaled_loss.new_zeros(())
            else:
                weighted_pretext = lambda_pretext * pretext_result.loss
            final_loss = triplet.scaled_loss + weighted_pretext
            if not torch.isfinite(final_loss):
                raise FloatingPointError(
                    f"non-finite loss at iteration {iteration} in {trajectory.value}"
                )
            gradients = encoder_gradient_diagnostics(
                triplet.scaled_loss, weighted_pretext, encoder_parameters
            )
            final_loss_value = float(final_loss.detach().cpu().item())
            final_loss.backward()
            optimizer.step()

            if final_loss_value < best_loss:
                best_loss = final_loss_value
                best_iteration = iteration
                best_state = _clone_state(model)
            _synchronize(device)
            iteration_runtime = time.perf_counter() - iteration_started
            record = build_iteration_record(
                iteration=iteration,
                anchor_indices=anchor_tensor.detach().cpu(),
                positive_indices=positive_tensor.detach().cpu(),
                triplet=triplet,
                pretext=pretext_result,
                gradients=gradients,
                learning_rate=learning_rate,
                lambda_pretext=lambda_pretext,
                final_loss=final_loss_value,
                iteration_runtime_seconds=iteration_runtime,
                allocated_vram_mib=_allocated_vram_mib(device),
                best_checkpoint_iteration=best_iteration,
                patch_size=patch_size,
                series_id=replay.series_id,
                trajectory=trajectory.value,
                seed=replay.seed,
            )
            recorder.append(record)
    except BaseException:
        recorder.abort()
        raise

    if best_state is None or best_iteration == 0:
        recorder.abort()
        raise RuntimeError("training produced no finite checkpoint")
    last_state = _clone_state(model)
    iteration_log_sha256 = recorder.close()
    _synchronize(device)
    runtime_seconds = time.perf_counter() - started
    checkpoints: dict[CheckpointKind, Mapping[str, Tensor]] = {
        CheckpointKind.BEST: best_state,
        CheckpointKind.LAST: last_state,
    }
    checkpoint_hashes = {
        checkpoint.value: state_dict_sha256(state)
        for checkpoint, state in checkpoints.items()
    }
    summary = TrainingSummary(
        trajectory=trajectory,
        seed=int(replay.seed),
        initial_state_sha256=initial_hash,
        replay_sha256=str(replay.payload_sha256),
        best_iteration=best_iteration,
        best_loss=best_loss,
        last_iteration=total_iterations,
        runtime_seconds=runtime_seconds,
        peak_vram_mib=_allocated_vram_mib(device),
        checkpoint_sha256=checkpoint_hashes,
    )
    return TrainingResult(checkpoints, summary, iteration_log_sha256)


def replay_rand_bn(
    model: nn.Module,
    initial_state: Mapping[str, Tensor],
    store: Any,
    replay: ReplayPlan,
    protocol: Any,
    device: torch.device,
    recorder: IterationRecorder,
) -> TrainingResult:
    """Replay identical BN exposure without projection, backward, or optimizer."""

    hp = _hyperparameters(protocol)
    total_iterations = int(_value(hp, "iterations"))
    if len(replay.iterations) != total_iterations:
        raise ValueError("replay length differs from the frozen iteration count")
    if replay.n_train_patches != len(store):
        raise ValueError("replay patch count differs from PatchStore")
    patch_size = int(_value(hp, "patch_size"))
    pretext_step = int(_value(hp, "pretext_step"))
    offsets = _trajectory_offsets(protocol, Trajectory.OFFICIAL)
    model.to(device)
    model.load_state_dict(initial_state, strict=True)
    initial_hash = state_dict_sha256(initial_state)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for iteration, step in enumerate(replay.iterations, start=1):
                iteration_started = time.perf_counter()
                anchor_np = np.array(step.anchor_indices, dtype=np.int64, copy=True)
                positive_np = materialize_positive_indices(
                    anchor_np,
                    len(store),
                    offsets,
                    np.asarray(step.positive_uniform, dtype=np.float32),
                )
                anchors = store.take(torch.from_numpy(anchor_np)).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                positives = store.take(torch.from_numpy(positive_np)).to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                lambda_pretext = pretext_weight(iteration, total_iterations)
                if lambda_pretext > 0.0:
                    pretext_cpu, _ = _make_pretext_patches(
                        store, anchor_np, pretext_step
                    )
                    pretext_patches = pretext_cpu.to(
                        device=device, dtype=torch.float32, non_blocking=True
                    )
                    model.embedding(torch.cat([anchors, positives, pretext_patches], dim=0))
                else:
                    model.embedding(torch.cat([anchors, positives], dim=0))
                _synchronize(device)
                record = build_iteration_record(
                    iteration=iteration,
                    anchor_indices=torch.from_numpy(anchor_np),
                    positive_indices=torch.from_numpy(positive_np),
                    triplet=None,
                    pretext=None,
                    gradients=None,
                    learning_rate=cosine_learning_rate(
                        iteration,
                        total_iterations,
                        float(_value(hp, "learning_rate")),
                    ),
                    lambda_pretext=lambda_pretext,
                    final_loss=0.0,
                    iteration_runtime_seconds=time.perf_counter() - iteration_started,
                    allocated_vram_mib=_allocated_vram_mib(device),
                    best_checkpoint_iteration=0,
                    patch_size=patch_size,
                    series_id=replay.series_id,
                    trajectory=Trajectory.RAND_BN.value,
                    seed=replay.seed,
                )
                recorder.append(record)
    except BaseException:
        recorder.abort()
        raise

    calibrated_state = _clone_state(model)
    initial_parameters = dict(model.named_parameters())
    for name, parameter in initial_parameters.items():
        expected = initial_state[name].detach().to(device="cpu")
        actual = parameter.detach().to(device="cpu")
        if not torch.equal(actual, expected):
            recorder.abort()
            raise RuntimeError(f"RAND_BN changed trainable parameter {name}")
    iteration_log_sha256 = recorder.close()
    _synchronize(device)
    runtime_seconds = time.perf_counter() - started
    checkpoint_hash = state_dict_sha256(calibrated_state)
    summary = TrainingSummary(
        trajectory=Trajectory.RAND_BN,
        seed=int(replay.seed),
        initial_state_sha256=initial_hash,
        replay_sha256=str(replay.payload_sha256),
        best_iteration=None,
        best_loss=None,
        last_iteration=total_iterations,
        runtime_seconds=runtime_seconds,
        peak_vram_mib=_allocated_vram_mib(device),
        checkpoint_sha256={CheckpointKind.BN_CALIBRATED.value: checkpoint_hash},
    )
    return TrainingResult(
        {CheckpointKind.BN_CALIBRATED: calibrated_state},
        summary,
        iteration_log_sha256,
    )


__all__ = [
    "TrainingResult",
    "cosine_learning_rate",
    "replay_rand_bn",
    "train_trajectory",
]
