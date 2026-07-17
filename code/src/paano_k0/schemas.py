"""Immutable cross-module contracts for the frozen PaAno K0 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Any, Literal, Mapping

import numpy as np


UNIT_INTERVAL_TOLERANCE = 5e-12


def canonicalize_unit_interval_metric(name: str, value: float) -> float:
    """Return a mathematically bounded metric with numeric endpoint tolerance."""

    metric = float(value)
    if (
        not np.isfinite(metric)
        or metric < -UNIT_INTERVAL_TOLERANCE
        or metric > 1.0 + UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError(f"{name} must be finite and in [0,1] within numeric tolerance")
    return float(np.clip(metric, 0.0, 1.0))
from numpy.typing import NDArray


Track = Literal["U", "M"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Trajectory(str, Enum):
    OFFICIAL = "OFFICIAL"
    PAPERNEG = "PAPERNEG"
    PAPERNEG_NONOVERLAP = "PAPERNEG_NONOVERLAP"
    RAND_BN = "RAND_BN"


class CheckpointKind(str, Enum):
    BEST = "BEST"
    LAST = "LAST"
    BN_CALIBRATED = "BN_CALIBRATED"


def _coerce_trajectory(value: Trajectory | str) -> Trajectory:
    return value if isinstance(value, Trajectory) else Trajectory(str(value))


def _coerce_checkpoint(value: CheckpointKind | str) -> CheckpointKind:
    return value if isinstance(value, CheckpointKind) else CheckpointKind(str(value))


def _validate_sha256(value: str, field_name: str, *, allow_empty: bool = False) -> None:
    if allow_empty and not value:
        return
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA256")


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    series_id: str
    family: str
    track: Track
    csv_path: Path
    csv_sha256: str
    rows: int
    channels: int
    train_end: int
    feature_columns: tuple[str, ...]
    label_column: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "csv_path", Path(self.csv_path))
        if not self.series_id or not self.family:
            raise ValueError("series_id and family must be non-empty")
        if self.track not in ("U", "M"):
            raise ValueError("track must be 'U' or 'M'")
        _validate_sha256(self.csv_sha256, "csv_sha256")
        if self.rows <= 0 or self.channels <= 0:
            raise ValueError("rows and channels must be positive")
        if not 0 < self.train_end <= self.rows:
            raise ValueError("train_end must be in (0, rows]")
        if len(self.feature_columns) != self.channels:
            raise ValueError("feature_columns length must equal channels")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature columns must be unique")
        if not self.label_column or self.label_column in self.feature_columns:
            raise ValueError("label_column must be non-empty and feature-disjoint")


@dataclass(frozen=True, slots=True)
class IterationReplay:
    anchor_indices: NDArray[np.int64]
    positive_uniform: NDArray[np.float32]
    unadjacent_uniform: NDArray[np.float32]

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchor_indices)
        positive = np.asarray(self.positive_uniform)
        unadjacent = np.asarray(self.unadjacent_uniform)
        if anchors.dtype != np.int64 or anchors.ndim != 1:
            raise TypeError("anchor_indices must be int64 [B]")
        if positive.dtype != np.float32 or positive.shape != anchors.shape:
            raise TypeError("positive_uniform must be float32 [B]")
        if unadjacent.dtype != np.float32 or unadjacent.ndim != 2:
            raise TypeError("unadjacent_uniform must be float32 [B,K]")
        if unadjacent.shape[0] != anchors.shape[0]:
            raise ValueError("replay arrays must share the leading dimension")
        if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(unadjacent)):
            raise ValueError("replay uniforms must be finite")
        if np.any((positive < 0) | (positive >= 1)) or np.any(
            (unadjacent < 0) | (unadjacent >= 1)
        ):
            raise ValueError("replay uniforms must lie in [0,1)")
        anchors.setflags(write=False)
        positive.setflags(write=False)
        unadjacent.setflags(write=False)


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    series_id: str
    seed: int
    n_train_patches: int
    batch_size: int
    iterations: tuple[IterationReplay, ...]
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.seed < 0 or self.n_train_patches <= 0 or self.batch_size <= 0:
            raise ValueError("invalid replay identity")
        if not self.iterations:
            raise ValueError("a replay plan must contain at least one iteration")
        if any(np.any(step.anchor_indices >= self.n_train_patches) for step in self.iterations):
            raise ValueError("anchor index outside training patch store")
        if any(np.any(step.anchor_indices < 0) for step in self.iterations):
            raise ValueError("anchor indices must be non-negative")
        _validate_sha256(self.payload_sha256, "payload_sha256")

    @property
    def records(self) -> tuple[IterationReplay, ...]:
        """Compatibility alias used by the trainer implementation."""

        return self.iterations


@dataclass(frozen=True, slots=True)
class RunJob:
    series: SeriesSpec
    trajectory: Trajectory
    seed: int
    protocol_path: Path
    vendor_root: Path
    output_root: Path
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory", _coerce_trajectory(self.trajectory))
        object.__setattr__(self, "protocol_path", Path(self.protocol_path))
        object.__setattr__(self, "vendor_root", Path(self.vendor_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.seed < 0 or not self.device:
            raise ValueError("invalid run job seed/device")

    @property
    def spec(self) -> SeriesSpec:
        return self.series

    @property
    def series_spec(self) -> SeriesSpec:
        return self.series


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    trajectory: Trajectory
    seed: int
    initial_state_sha256: str
    replay_sha256: str
    best_iteration: int | None
    best_loss: float | None
    last_iteration: int
    runtime_seconds: float
    peak_vram_mib: float
    checkpoint_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory", _coerce_trajectory(self.trajectory))
        if self.seed < 0 or self.last_iteration <= 0:
            raise ValueError("invalid summary iteration/seed")
        if self.best_iteration is not None and not 1 <= self.best_iteration <= self.last_iteration:
            raise ValueError("best_iteration outside trained range")
        if self.runtime_seconds < 0 or self.peak_vram_mib < 0:
            raise ValueError("runtime and VRAM must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trajectory"] = self.trajectory.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingSummary":
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ScoreManifest:
    schema_version: str
    run_id: str
    series_id: str
    family: str
    track: Track
    data_sha256: str
    config_sha256: str
    vendor_sha: str
    seed: int
    trajectory: Trajectory
    checkpoint: CheckpointKind
    initial_state_sha256: str
    replay_sha256: str
    checkpoint_sha256: str
    num_points: int
    num_train_patches: int
    num_full_patches: int
    channels: int
    patch_size: int
    stride: int
    top_k: int
    requested_memory_fraction: float
    effective_memory_fraction: float
    memory_count: int
    memory_sha256: str
    score_sha256: str
    runtime_seconds: float
    peak_vram_mib: float
    sliding_window: int
    labels_read: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory", _coerce_trajectory(self.trajectory))
        object.__setattr__(self, "checkpoint", _coerce_checkpoint(self.checkpoint))
        if self.track not in ("U", "M"):
            raise ValueError("track must be U or M")
        if not self.schema_version or not self.run_id or not self.series_id:
            raise ValueError("manifest identity fields must be non-empty")
        if self.labels_read:
            raise ValueError("runner score manifests must record labels_read=false")
        for name in (
            "data_sha256",
            "config_sha256",
            "initial_state_sha256",
            "replay_sha256",
            "checkpoint_sha256",
            "memory_sha256",
            "score_sha256",
        ):
            _validate_sha256(str(getattr(self, name)), name)
        if not re.fullmatch(r"[0-9a-f]{40}", self.vendor_sha):
            raise ValueError("vendor_sha must be a lowercase 40-character Git SHA")
        positive_ints = (
            self.num_points,
            self.num_train_patches,
            self.num_full_patches,
            self.channels,
            self.patch_size,
            self.stride,
            self.top_k,
            self.memory_count,
            self.sliding_window,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("manifest count/configuration fields must be positive")
        if not 0 < self.requested_memory_fraction <= 1:
            raise ValueError("requested memory fraction must be in (0,1]")
        if not 0 < self.effective_memory_fraction <= 1:
            raise ValueError("effective memory fraction must be in (0,1]")
        if self.runtime_seconds < 0 or self.peak_vram_mib < 0:
            raise ValueError("runtime and VRAM must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trajectory"] = self.trajectory.value
        payload["checkpoint"] = self.checkpoint.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScoreManifest":
        """Build a typed manifest from a JSON-compatible mapping."""

        values = dict(payload)
        values["trajectory"] = _coerce_trajectory(values["trajectory"])
        values["checkpoint"] = _coerce_checkpoint(values["checkpoint"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class MetricRow:
    run_id: str
    series_id: str
    family: str
    track: Track
    seed: int
    trajectory: Trajectory
    checkpoint: CheckpointKind
    vus_pr: float
    auprc: float
    vus_roc: float
    auroc: float
    score_sha256: str
    data_sha256: str
    config_sha256: str
    vendor_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory", _coerce_trajectory(self.trajectory))
        object.__setattr__(self, "checkpoint", _coerce_checkpoint(self.checkpoint))
        if self.track not in ("U", "M"):
            raise ValueError("track must be U or M")
        for name in ("vus_pr", "auprc", "vus_roc", "auroc"):
            value = canonicalize_unit_interval_metric(name, getattr(self, name))
            object.__setattr__(self, name, value)

    @property
    def arm(self) -> str:
        return f"{self.trajectory.value}_{self.checkpoint.value}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trajectory"] = self.trajectory.value
        payload["checkpoint"] = self.checkpoint.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricRow":
        return cls(**dict(payload))


def scored_checkpoints(trajectory: Trajectory | str) -> tuple[CheckpointKind, ...]:
    trajectory = _coerce_trajectory(trajectory)
    if trajectory is Trajectory.RAND_BN:
        return (CheckpointKind.BN_CALIBRATED,)
    return (CheckpointKind.BEST, CheckpointKind.LAST)


def make_run_id(
    series_id: str,
    seed: int,
    trajectory: Trajectory | str,
    checkpoint: CheckpointKind | str,
) -> str:
    trajectory = _coerce_trajectory(trajectory)
    checkpoint = _coerce_checkpoint(checkpoint)
    if checkpoint not in scored_checkpoints(trajectory):
        raise ValueError(f"checkpoint {checkpoint.value} is invalid for {trajectory.value}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    safe_series = re.sub(r"[^A-Za-z0-9._-]+", "-", series_id).strip(" .-")
    if not safe_series:
        raise ValueError("series_id has no filesystem-safe characters")
    return f"{safe_series}__seed_{seed}__{trajectory.value}__{checkpoint.value}"
