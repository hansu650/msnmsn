"""Strict parsing of the frozen K0 protocol and deterministic job expansion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date as Date
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .schemas import CheckpointKind, RunJob, SeriesSpec, Trajectory


class ProtocolError(ValueError):
    """Raised when a protocol or manifest departs from the frozen design."""


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    name: str
    venue: str
    git_url: str
    git_sha: str
    paper_vus_pr_u: float
    paper_vus_pr_m: float
    comparison_note: str


@dataclass(frozen=True, slots=True)
class OfficialHyperparameters:
    patch_size: int
    stride: int
    iterations: int
    batch_size: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    margin: float
    triplet_divisor: int
    positive_radius: int
    pretext_step: int
    pretext_nonzero_iterations: int
    memory_request_fraction: float
    memory_minimum_exemplars: int
    score_top_k: int
    use_revin: bool


@dataclass(frozen=True, slots=True)
class SeedsConfig:
    primary: tuple[int, ...]
    conditional_confirmation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    id: Trajectory
    save_checkpoints: tuple[CheckpointKind, ...]
    negative_pool: str | None = None
    negative_selection_space: str | None = None
    positive_offsets: tuple[int, ...] = ()
    optimizer_updates: bool = True
    batchnorm_forward_replay_batches: int | None = None


@dataclass(frozen=True, slots=True)
class InstrumentationConfig:
    per_iteration: tuple[str, ...]
    per_score_artifact: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabelPolicyConfig:
    runner_accepts_labels: bool
    score_written_before_evaluation: bool
    evaluator_only_metrics: tuple[str, ...]
    forbidden_selection_metrics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LowActivityGate:
    post_pretext_iteration_range: tuple[int, int]
    median_active_hinge_fraction_lte: float
    minimum_families: int


@dataclass(frozen=True, slots=True)
class EarlyCheckpointGate:
    best_iteration_lte: int
    minimum_families: int


@dataclass(frozen=True, slots=True)
class PerformanceGate:
    macro_vus_pr_delta_gte: float
    minimum_positive_families: int
    worst_family_delta_gte: float
    macro_auprc_delta_gt: float


@dataclass(frozen=True, slots=True)
class OutcomeConfig:
    last_only_passes: str
    paperneg_only_passes: str
    nonoverlap_unique_passes_with_low_activity: str
    activity_not_low_and_no_execution_gain: str
    activity_low_but_no_performance_gain: str


@dataclass(frozen=True, slots=True)
class GatesConfig:
    low_activity: LowActivityGate
    early_checkpoint: EarlyCheckpointGate
    performance: PerformanceGate
    outcomes: OutcomeConfig


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    project: str
    status: str
    date: str
    baseline: BaselineConfig
    official_hyperparameters: OfficialHyperparameters
    seeds: SeedsConfig
    trajectories: tuple[TrajectoryConfig, ...]
    instrumentation: InstrumentationConfig
    label_policy: LabelPolicyConfig
    gates: GatesConfig
    forbidden: tuple[str, ...]
    source_path: Path
    source_sha256: str

    def trajectory(self, trajectory: Trajectory | str) -> TrajectoryConfig:
        wanted = trajectory if isinstance(trajectory, Trajectory) else Trajectory(trajectory)
        matches = [item for item in self.trajectories if item.id is wanted]
        if len(matches) != 1:
            raise ProtocolError(f"trajectory {wanted.value} is not registered exactly once")
        return matches[0]


_TOP_KEYS = {
    "project",
    "status",
    "date",
    "baseline",
    "official_hyperparameters",
    "seeds",
    "trajectories",
    "instrumentation",
    "label_policy",
    "gates",
    "forbidden",
}


def _expect_keys(mapping: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ProtocolError(f"{context} keys differ; missing={missing}, unknown={unknown}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_trajectory(payload: Mapping[str, Any]) -> TrajectoryConfig:
    trajectory = Trajectory(payload.get("id"))
    common = {"id", "save_checkpoints"}
    if trajectory is Trajectory.RAND_BN:
        _expect_keys(
            payload,
            common | {"optimizer_updates", "batchnorm_forward_replay_batches"},
            f"trajectory.{trajectory.value}",
        )
        return TrajectoryConfig(
            id=trajectory,
            save_checkpoints=tuple(CheckpointKind(v) for v in payload["save_checkpoints"]),
            optimizer_updates=bool(payload["optimizer_updates"]),
            batchnorm_forward_replay_batches=int(payload["batchnorm_forward_replay_batches"]),
        )
    _expect_keys(
        payload,
        common | {"negative_pool", "negative_selection_space", "positive_offsets"},
        f"trajectory.{trajectory.value}",
    )
    return TrajectoryConfig(
        id=trajectory,
        save_checkpoints=tuple(CheckpointKind(v) for v in payload["save_checkpoints"]),
        negative_pool=str(payload["negative_pool"]),
        negative_selection_space=str(payload["negative_selection_space"]),
        positive_offsets=tuple(int(v) for v in payload["positive_offsets"]),
    )


def load_protocol(path: Path) -> ProtocolConfig:
    """Load and strictly validate the sole scientific YAML configuration."""

    source = Path(path).resolve(strict=True)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ProtocolError("protocol root must be a mapping")
    _expect_keys(raw, _TOP_KEYS, "protocol")

    baseline = raw["baseline"]
    hyper = raw["official_hyperparameters"]
    seeds = raw["seeds"]
    instrumentation = raw["instrumentation"]
    labels = raw["label_policy"]
    gates = raw["gates"]
    _expect_keys(
        baseline,
        {
            "name",
            "venue",
            "git_url",
            "git_sha",
            "paper_vus_pr_u",
            "paper_vus_pr_m",
            "comparison_note",
        },
        "baseline",
    )
    _expect_keys(hyper, set(OfficialHyperparameters.__dataclass_fields__), "official_hyperparameters")
    _expect_keys(seeds, {"primary", "conditional_confirmation"}, "seeds")
    _expect_keys(instrumentation, {"per_iteration", "per_score_artifact"}, "instrumentation")
    _expect_keys(
        labels,
        {
            "runner_accepts_labels",
            "score_written_before_evaluation",
            "evaluator_only_metrics",
            "forbidden_selection_metrics",
        },
        "label_policy",
    )
    _expect_keys(gates, {"low_activity", "early_checkpoint", "performance", "outcomes"}, "gates")
    _expect_keys(
        gates["low_activity"], set(LowActivityGate.__dataclass_fields__), "gates.low_activity"
    )
    _expect_keys(
        gates["early_checkpoint"],
        set(EarlyCheckpointGate.__dataclass_fields__),
        "gates.early_checkpoint",
    )
    _expect_keys(
        gates["performance"], set(PerformanceGate.__dataclass_fields__), "gates.performance"
    )
    _expect_keys(gates["outcomes"], set(OutcomeConfig.__dataclass_fields__), "gates.outcomes")

    raw_date = raw["date"]
    date_text = raw_date.isoformat() if isinstance(raw_date, Date) else str(raw_date)
    config = ProtocolConfig(
        project=str(raw["project"]),
        status=str(raw["status"]),
        date=date_text,
        baseline=BaselineConfig(**baseline),
        official_hyperparameters=OfficialHyperparameters(**hyper),
        seeds=SeedsConfig(
            primary=tuple(int(v) for v in seeds["primary"]),
            conditional_confirmation=tuple(int(v) for v in seeds["conditional_confirmation"]),
        ),
        trajectories=tuple(_parse_trajectory(item) for item in raw["trajectories"]),
        instrumentation=InstrumentationConfig(
            per_iteration=tuple(str(v) for v in instrumentation["per_iteration"]),
            per_score_artifact=tuple(str(v) for v in instrumentation["per_score_artifact"]),
        ),
        label_policy=LabelPolicyConfig(
            runner_accepts_labels=bool(labels["runner_accepts_labels"]),
            score_written_before_evaluation=bool(labels["score_written_before_evaluation"]),
            evaluator_only_metrics=tuple(str(v) for v in labels["evaluator_only_metrics"]),
            forbidden_selection_metrics=tuple(str(v) for v in labels["forbidden_selection_metrics"]),
        ),
        gates=GatesConfig(
            low_activity=LowActivityGate(
                post_pretext_iteration_range=tuple(
                    int(v) for v in gates["low_activity"]["post_pretext_iteration_range"]
                ),
                median_active_hinge_fraction_lte=float(
                    gates["low_activity"]["median_active_hinge_fraction_lte"]
                ),
                minimum_families=int(gates["low_activity"]["minimum_families"]),
            ),
            early_checkpoint=EarlyCheckpointGate(**gates["early_checkpoint"]),
            performance=PerformanceGate(**gates["performance"]),
            outcomes=OutcomeConfig(**gates["outcomes"]),
        ),
        forbidden=tuple(str(v) for v in raw["forbidden"]),
        source_path=source,
        source_sha256=_sha256_file(source),
    )
    validate_protocol(config)
    return config


def validate_protocol(config: ProtocolConfig) -> None:
    """Reject scientific drift from the pre-registered K0 YAML."""

    expected_hyper = OfficialHyperparameters(
        patch_size=96,
        stride=1,
        iterations=100,
        batch_size=512,
        learning_rate=0.0001,
        optimizer="AdamW",
        weight_decay=0.0001,
        margin=0.1,
        triplet_divisor=10,
        positive_radius=2,
        pretext_step=96,
        pretext_nonzero_iterations=19,
        memory_request_fraction=0.10,
        memory_minimum_exemplars=500,
        score_top_k=3,
        use_revin=True,
    )
    if (
        config.project != "paano_execution_activity_k0"
        or config.status != "frozen_for_implementation"
        or config.date != "2026-07-16"
    ):
        raise ProtocolError("project/status is not the frozen K0 protocol")
    if config.baseline.git_sha != "d4c67116190efa4592dc6a8a157ced0def68b6af":
        raise ProtocolError("baseline Git SHA changed")
    if config.baseline.name != "PaAno" or config.baseline.venue != "ICLR_2026":
        raise ProtocolError("baseline identity changed")
    if (
        config.baseline.git_url != "https://github.com/jinnnju/PaAno"
        or config.baseline.paper_vus_pr_u != 0.530
        or config.baseline.paper_vus_pr_m != 0.431
        or config.baseline.comparison_note
        != "external_paper_reference_not_same_file_paired"
    ):
        raise ProtocolError("baseline external-reference metadata changed")
    if config.official_hyperparameters != expected_hyper:
        raise ProtocolError("official hyperparameters changed")
    if config.seeds != SeedsConfig((2027,), (2027, 2028, 2029)):
        raise ProtocolError("seed protocol changed")
    if tuple(item.id for item in config.trajectories) != tuple(Trajectory):
        raise ProtocolError("trajectory order or membership changed")
    expected_checkpoints = {
        Trajectory.OFFICIAL: (CheckpointKind.BEST, CheckpointKind.LAST),
        Trajectory.PAPERNEG: (CheckpointKind.BEST, CheckpointKind.LAST),
        Trajectory.PAPERNEG_NONOVERLAP: (CheckpointKind.BEST, CheckpointKind.LAST),
        Trajectory.RAND_BN: (CheckpointKind.BN_CALIBRATED,),
    }
    for item in config.trajectories:
        if item.save_checkpoints != expected_checkpoints[item.id]:
            raise ProtocolError(f"checkpoint protocol changed for {item.id.value}")
    if config.trajectory(Trajectory.OFFICIAL).positive_offsets != (-2, -1, 1, 2):
        raise ProtocolError("official positive offsets changed")
    if config.trajectory(Trajectory.PAPERNEG).positive_offsets != (-2, -1, 1, 2):
        raise ProtocolError("paper-negative positive offsets changed")
    if config.trajectory(Trajectory.PAPERNEG_NONOVERLAP).positive_offsets != (-96, 96):
        raise ProtocolError("non-overlap offsets changed")
    official = config.trajectory(Trajectory.OFFICIAL)
    paperneg = config.trajectory(Trajectory.PAPERNEG)
    nonoverlap = config.trajectory(Trajectory.PAPERNEG_NONOVERLAP)
    if (official.negative_pool, official.negative_selection_space) != (
        "positive_views",
        "projection",
    ):
        raise ProtocolError("official negative semantics changed")
    if (paperneg.negative_pool, paperneg.negative_selection_space) != (
        "anchor_views",
        "encoder",
    ) or (nonoverlap.negative_pool, nonoverlap.negative_selection_space) != (
        "anchor_views",
        "encoder",
    ):
        raise ProtocolError("paper-faithful negative semantics changed")
    rand = config.trajectory(Trajectory.RAND_BN)
    if rand.optimizer_updates or rand.batchnorm_forward_replay_batches != 100:
        raise ProtocolError("RAND_BN update semantics changed")
    if config.label_policy.runner_accepts_labels or not config.label_policy.score_written_before_evaluation:
        raise ProtocolError("label-isolation policy changed")
    expected_iteration_fields = (
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
    )
    expected_score_fields = (
        "vus_pr",
        "auprc",
        "vus_roc",
        "runtime_seconds",
        "peak_vram_mib",
        "requested_memory_fraction",
        "effective_memory_fraction",
        "score_sha256",
    )
    if config.instrumentation != InstrumentationConfig(
        expected_iteration_fields, expected_score_fields
    ):
        raise ProtocolError("instrumentation fields changed")
    if config.label_policy.evaluator_only_metrics != ("vus_pr", "auprc", "vus_roc"):
        raise ProtocolError("evaluator-only metrics changed")
    if config.label_policy.forbidden_selection_metrics != (
        "point_f1",
        "best_f1",
        "range_f1",
    ):
        raise ProtocolError("forbidden selection metrics changed")
    if config.gates.low_activity != LowActivityGate((20, 100), 0.10, 4):
        raise ProtocolError("activity gate changed")
    if config.gates.early_checkpoint != EarlyCheckpointGate(30, 4):
        raise ProtocolError("checkpoint gate changed")
    if config.gates.performance != PerformanceGate(0.010, 3, -0.010, 0.0):
        raise ProtocolError("performance gate changed")
    if config.gates.outcomes != OutcomeConfig(
        "SIMPLE_CHECKPOINT_FIX",
        "SIMPLE_PAPER_PARITY_FIX",
        "GO_METHOD_DESIGN",
        "STOP_NO_ACTIVITY_FAILURE",
        "STOP_NO_PERFORMANCE_HEADROOM",
    ):
        raise ProtocolError("decision outcomes changed")
    expected_forbidden = (
        "learned_selector",
        "family_specific_parameters",
        "eval_label_tuning",
        "score_calibration_novelty",
        "attention_or_large_backbone",
        "post_failure_rescue_module",
    )
    if config.forbidden != expected_forbidden:
        raise ProtocolError("forbidden-operation list changed")


def load_series_manifest(path: Path) -> tuple[SeriesSpec, ...]:
    """Verify and load the six frozen files without materializing labels."""

    manifest_path = Path(path).resolve(strict=True)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {
            "family",
            "track",
            "file",
            "sha256",
            "rows",
            "channels",
            "train_end",
            "bytes",
            "local_path",
        }
        if set(reader.fieldnames or ()) != expected_columns:
            raise ProtocolError("K0 manifest columns changed")
        records = tuple(reader)
    if len(records) != 6:
        raise ProtocolError(f"expected six K0 series, found {len(records)}")

    specs: list[SeriesSpec] = []
    for record in records:
        csv_path = Path(record["local_path"]).resolve(strict=True)
        if csv_path.name != record["file"]:
            raise ProtocolError(f"manifest filename mismatch: {csv_path}")
        expected_bytes = int(record["bytes"])
        if csv_path.stat().st_size != expected_bytes:
            raise ProtocolError(f"byte count mismatch for {csv_path.name}")
        digest = _sha256_file(csv_path)
        if digest != record["sha256"]:
            raise ProtocolError(f"SHA256 mismatch for {csv_path.name}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
            row_count = sum(1 for _ in handle)
        label_column = "Label"
        if label_column not in header or header[-1] != label_column:
            raise ProtocolError(f"expected final Label column in {csv_path.name}")
        feature_columns = tuple(column for column in header if column != label_column)
        rows = int(record["rows"])
        channels = int(record["channels"])
        if row_count != rows or len(feature_columns) != channels:
            raise ProtocolError(f"shape mismatch for {csv_path.name}")
        train_end = int(record["train_end"])
        if train_end < 96:
            raise ProtocolError(f"normal prefix too short for {csv_path.name}")
        specs.append(
            SeriesSpec(
                series_id=csv_path.stem,
                family=record["family"],
                track=record["track"],
                csv_path=csv_path,
                csv_sha256=digest,
                rows=rows,
                channels=channels,
                train_end=train_end,
                feature_columns=feature_columns,
                label_column=label_column,
            )
        )
    if len({spec.series_id for spec in specs}) != 6 or len({spec.family for spec in specs}) != 6:
        raise ProtocolError("series IDs and families must be unique")
    if {spec.track for spec in specs} != {"U", "M"}:
        raise ProtocolError("both U and M tracks are required")
    return tuple(specs)


def expand_primary_jobs(
    config: ProtocolConfig,
    series: Sequence[SeriesSpec],
    vendor_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
) -> tuple[RunJob, ...]:
    if len(series) != 6:
        raise ProtocolError("primary K0 requires exactly six frozen series")
    return tuple(
        RunJob(
            series=spec,
            trajectory=trajectory.id,
            seed=seed,
            protocol_path=config.source_path,
            vendor_root=vendor_root,
            output_root=output_root,
            device=device,
        )
        for spec in series
        for seed in config.seeds.primary
        for trajectory in config.trajectories
    )


def expand_confirmation_jobs(
    config: ProtocolConfig,
    decision_path: Path,
    series: Sequence[SeriesSpec],
    vendor_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
) -> tuple[RunJob, ...]:
    with Path(decision_path).open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    outcome = str(decision.get("outcome", ""))
    winner_for_outcome = {
        "SIMPLE_CHECKPOINT_FIX": Trajectory.OFFICIAL,
        "SIMPLE_PAPER_PARITY_FIX": Trajectory.PAPERNEG,
        "GO_METHOD_DESIGN": Trajectory.PAPERNEG_NONOVERLAP,
    }
    if outcome not in winner_for_outcome:
        raise ProtocolError(f"confirmation is forbidden for outcome {outcome!r}")
    trajectories = (Trajectory.OFFICIAL, winner_for_outcome[outcome])
    unique_trajectories = tuple(dict.fromkeys(trajectories))
    new_seeds = tuple(
        seed
        for seed in config.seeds.conditional_confirmation
        if seed not in config.seeds.primary
    )
    return tuple(
        RunJob(
            series=spec,
            trajectory=trajectory,
            seed=seed,
            protocol_path=config.source_path,
            vendor_root=vendor_root,
            output_root=output_root,
            device=device,
        )
        for spec in series
        for seed in new_seeds
        for trajectory in unique_trajectories
    )
