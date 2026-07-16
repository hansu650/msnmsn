"""PaAno K0 execution-fidelity audit overlay.

The package deliberately separates the label-free runner path from the
evaluator-only label path.  Public contracts live in :mod:`paano_k0.schemas`.
"""

from .schemas import (
    CheckpointKind,
    IterationReplay,
    MetricRow,
    ReplayPlan,
    RunJob,
    ScoreManifest,
    SeriesSpec,
    TrainingSummary,
    Trajectory,
    make_run_id,
    scored_checkpoints,
)

__all__ = [
    "CheckpointKind",
    "IterationReplay",
    "MetricRow",
    "ReplayPlan",
    "RunJob",
    "ScoreManifest",
    "SeriesSpec",
    "TrainingSummary",
    "Trajectory",
    "make_run_id",
    "scored_checkpoints",
]

