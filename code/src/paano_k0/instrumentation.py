"""Transactional per-iteration instrumentation for PaAno K0."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import torch
from torch import Tensor

from .objectives import GradDiagnostics, PretextBatch, TripletBatch


_REQUIRED_RECORD_KEYS = frozenset(
    {
        "iteration",
        "batch_size",
        "learning_rate",
        "lambda_pretext",
        "positive_offset",
        "raw_overlap_ratio",
        "positive_distance_quantiles",
        "negative_distance_quantiles",
        "hinge_margin_quantiles",
        "active_hinge_fraction",
        "triplet_loss_before_divisor",
        "triplet_loss_after_divisor",
        "pretext_loss",
        "pretext_accuracy",
        "encoder_triplet_grad_norm",
        "encoder_pretext_grad_norm",
        "encoder_gradient_cosine",
        "selected_negative_temporal_gap",
        "selected_negative_index_collision_rate",
        "best_checkpoint_iteration",
        "final_loss",
        "iteration_runtime_seconds",
        "allocated_vram_mib",
    }
)


def quantile_summary(
    x: Tensor,
    probabilities: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, float]:
    """Return a stable five-number summary from a finite tensor."""

    if not isinstance(x, Tensor) or x.numel() == 0:
        raise ValueError("quantile input must be a non-empty tensor")
    if probabilities != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("K0 instrumentation requires the frozen five probabilities")
    values = x.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("quantile input contains NaN or Inf")
    q = torch.quantile(values, torch.tensor(probabilities, dtype=torch.float64))
    names = ("min", "q1", "median", "q3", "max")
    return {name: float(value.item()) for name, value in zip(names, q, strict=True)}


def _finite_float(name: str, value: float | int) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def build_iteration_record(
    *,
    iteration: int,
    anchor_indices: Tensor,
    positive_indices: Tensor,
    triplet: TripletBatch | None,
    pretext: PretextBatch | None,
    gradients: GradDiagnostics | None,
    learning_rate: float,
    lambda_pretext: float,
    final_loss: float,
    iteration_runtime_seconds: float,
    allocated_vram_mib: float,
    best_checkpoint_iteration: int,
    patch_size: int = 96,
    series_id: str | None = None,
    trajectory: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe row containing every frozen mechanism field."""

    if iteration < 1:
        raise ValueError("iteration must be one-based")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if anchor_indices.ndim != 1 or positive_indices.shape != anchor_indices.shape:
        raise ValueError("anchor_indices and positive_indices must be equal vectors")
    offsets = positive_indices.to(torch.int64) - anchor_indices.to(torch.int64)
    overlap = torch.clamp(
        1.0 - offsets.abs().to(torch.float64) / float(patch_size), min=0.0
    )

    if triplet is None:
        zeros = torch.zeros(anchor_indices.numel(), dtype=torch.float32)
        positive_distance = zeros
        negative_distance = zeros
        hinge_margin = zeros
        active_fraction = 0.0
        triplet_before = 0.0
        triplet_after = 0.0
        temporal_gap = zeros
        collision_rate = 0.0
    else:
        positive_distance = triplet.d_positive
        negative_distance = triplet.d_negative
        hinge_margin = triplet.hinge_margin
        active_fraction = float(triplet.active_mask.float().mean().detach().cpu().item())
        triplet_before = float(triplet.unscaled_loss.detach().cpu().item())
        triplet_after = float(triplet.scaled_loss.detach().cpu().item())
        temporal_gap = triplet.temporal_gap
        collision_rate = float(
            triplet.index_collision.float().mean().detach().cpu().item()
        )

    if pretext is None:
        pretext_loss = 0.0
        pretext_accuracy: float | None = None
    else:
        pretext_loss = float(pretext.loss.detach().cpu().item())
        pretext_accuracy = _finite_float("pretext_accuracy", pretext.accuracy)

    if gradients is None:
        triplet_grad_norm = 0.0
        pretext_grad_norm = 0.0
        gradient_cosine: float | None = None
    else:
        triplet_grad_norm = _finite_float("triplet_grad_norm", gradients.triplet_norm)
        pretext_grad_norm = _finite_float("pretext_grad_norm", gradients.pretext_norm)
        gradient_cosine = (
            None
            if gradients.cosine is None
            else _finite_float("gradient_cosine", gradients.cosine)
        )

    record: dict[str, Any] = {
        "iteration": int(iteration),
        "batch_size": int(anchor_indices.numel()),
        "learning_rate": _finite_float("learning_rate", learning_rate),
        "lambda_pretext": _finite_float("lambda_pretext", lambda_pretext),
        "positive_offset": quantile_summary(offsets),
        "raw_overlap_ratio": quantile_summary(overlap),
        "positive_distance_quantiles": quantile_summary(positive_distance),
        "negative_distance_quantiles": quantile_summary(negative_distance),
        "hinge_margin_quantiles": quantile_summary(hinge_margin),
        "active_hinge_fraction": _finite_float("active_hinge_fraction", active_fraction),
        "triplet_loss_before_divisor": _finite_float(
            "triplet_loss_before_divisor", triplet_before
        ),
        "triplet_loss_after_divisor": _finite_float(
            "triplet_loss_after_divisor", triplet_after
        ),
        "pretext_loss": _finite_float("pretext_loss", pretext_loss),
        "pretext_accuracy": pretext_accuracy,
        "encoder_triplet_grad_norm": triplet_grad_norm,
        "encoder_pretext_grad_norm": pretext_grad_norm,
        "encoder_gradient_cosine": gradient_cosine,
        "selected_negative_temporal_gap": quantile_summary(temporal_gap),
        "selected_negative_index_collision_rate": _finite_float(
            "selected_negative_index_collision_rate", collision_rate
        ),
        "best_checkpoint_iteration": int(best_checkpoint_iteration),
        "final_loss": _finite_float("final_loss", final_loss),
        "iteration_runtime_seconds": _finite_float(
            "iteration_runtime_seconds", iteration_runtime_seconds
        ),
        "allocated_vram_mib": _finite_float("allocated_vram_mib", allocated_vram_mib),
    }
    if series_id is not None:
        record["series_id"] = str(series_id)
    if trajectory is not None:
        record["trajectory"] = str(trajectory)
    if seed is not None:
        record["seed"] = int(seed)
    return record


def _validate_json_value(value: Any, path: str = "record") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            _validate_json_value(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains non-JSON type {type(value).__name__}")


class IterationRecorder:
    """Write one trajectory JSONL atomically and validate complete coverage."""

    def __init__(self, path: Path, expected_iterations: int) -> None:
        if expected_iterations <= 0:
            raise ValueError("expected_iterations must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.expected_iterations = int(expected_iterations)
        self._temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        self._handle = self._temporary.open("x", encoding="utf-8", newline="\n")
        self._count = 0
        self._closed = False

    @property
    def count(self) -> int:
        return self._count

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("recorder is already closed")
        missing = _REQUIRED_RECORD_KEYS.difference(record)
        if missing:
            raise ValueError(f"iteration record missing keys: {sorted(missing)}")
        expected_iteration = self._count + 1
        if record["iteration"] != expected_iteration:
            raise ValueError(
                f"expected iteration {expected_iteration}, got {record['iteration']}"
            )
        _validate_json_value(record)
        self._handle.write(
            json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._handle.flush()
        self._count += 1

    def close(self) -> str:
        if self._closed:
            raise RuntimeError("recorder is already closed")
        if self._count != self.expected_iterations:
            self.abort()
            raise ValueError(
                f"expected {self.expected_iterations} rows, recorded {self._count}"
            )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.replace(self._temporary, self.path)
        self._closed = True
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def abort(self) -> None:
        if self._closed:
            return
        self._handle.close()
        self._temporary.unlink(missing_ok=True)
        self._closed = True

    def __enter__(self) -> "IterationRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


__all__ = ["IterationRecorder", "build_iteration_record", "quantile_summary"]
