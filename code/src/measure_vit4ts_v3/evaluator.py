"""Delayed-label evaluator for the parity-gated ViTTrace v3 cache run.

Every score transaction, alias, frozen input identity, and retained failure is
checked before the ground-truth loader is imported.  Evaluation artifacts are
written outside the score tree so the label-free scorer remains immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from measure_vit4ts.full_manifest import FullSeriesRecord, load_manifest

from .cache_registry import CacheOnlyPlan, PlannedArm, load_compute_plan, sha256_file
from .metrics import (
    evaluate_series,
    freeze_valid_series_mask,
    valid_mask_sha256,
    validate_metrics_against_mask,
)
from .registry import ArmRegistry, load_arm_registry


EVALUATOR_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
RUN_NAME = "cache_only_b16_w240_s60"
EVALUATION_DIRECTORY = "corrected_primary_evaluation"


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ScoreRecord:
    series_id: str
    arm: str
    canonical_arm: str
    score_path: Path
    score_sha256: str
    score_manifest: Mapping[str, Any]
    transaction_manifest: Mapping[str, Any]

    @property
    def is_alias(self) -> bool:
        return self.arm != self.canonical_arm


@dataclass(frozen=True)
class EvaluationProtocol:
    config_path: Path
    config: Mapping[str, Any]
    config_identity: FileIdentity
    manifest_identity: FileIdentity
    records: tuple[FullSeriesRecord, ...]
    registry_path: Path
    registry_identity: FileIdentity
    registry: ArmRegistry
    plan_path: Path
    plan_identity: FileIdentity
    plan: CacheOnlyPlan
    plan_payload: Mapping[str, Any]
    parity_gate_identity: FileIdentity
    stage_root: Path


@dataclass(frozen=True)
class EvaluationPreflight:
    scores: Mapping[tuple[str, str], ScoreRecord]
    score_files: Mapping[str, FileIdentity]
    data_files: Mapping[str, FileIdentity]
    source_sha256: str
    parity_gate_sha256: str
    score_index_sha256: str


@dataclass(frozen=True)
class EvaluationResult:
    per_series: pd.DataFrame
    valid_mask: pd.DataFrame
    provenance: Mapping[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdefABCDEF" for character in text)


def _identity(path: Path, expected: str | None, context: str) -> FileIdentity:
    target = Path(path).resolve(strict=True)
    if not target.is_file():  # pragma: no cover - resolve(strict=True) is the normal guard
        raise FileNotFoundError(f"{context} is missing: {target}")
    digest = sha256_file(target).upper()
    if expected is not None and digest != str(expected).upper():
        raise ValueError(f"{context} SHA256 mismatch: {target}")
    stat = target.stat()
    return FileIdentity(target, digest, int(stat.st_size), int(stat.st_mtime_ns))


def _json(path: Path, context: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} JSON must be a mapping: {path}")
    return payload


def _default_artifact_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(config["paths"]["output_root"])
    return (
        root / "manifests" / "cache_only_arm_registry.json",
        root / "manifests" / "cache_only_compute_plan.json",
        root / "provenance" / "cache_only_parity_gate.json",
    )


def load_evaluation_protocol(
    config_path: Path,
    registry_path: Path | None = None,
    plan_path: Path | None = None,
    parity_gate_path: Path | None = None,
) -> EvaluationProtocol:
    """Load the exact frozen protocol without opening anomaly metadata."""

    config_path = Path(config_path).resolve(strict=True)
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, Mapping) or config.get("stage") != "vittrace_ablation_full_v3":
        raise ValueError("v3 evaluator accepts only stage=vittrace_ablation_full_v3")
    config_identity = _identity(config_path, _sha256_bytes(raw), "v3 config")

    manifest_path = Path(config["manifest"]["path"])
    manifest_identity = _identity(
        manifest_path, str(config["manifest"]["sha256"]), "v3 full manifest"
    )
    _, records = load_manifest(manifest_identity.path)
    expected_series = int(config["manifest"]["expected_series"])
    if len(records) != expected_series or len({row.series_id for row in records}) != expected_series:
        raise ValueError("v3 evaluator requires the exact unique manifest series set")

    default_registry, default_plan, default_gate = _default_artifact_paths(config)
    registry_path = Path(registry_path or default_registry)
    plan_path = Path(plan_path or default_plan)
    parity_gate_path = Path(parity_gate_path or default_gate)
    registry_identity = _identity(registry_path, None, "v3 arm registry")
    plan_identity = _identity(plan_path, None, "v3 compute plan")
    parity_identity = _identity(parity_gate_path, None, "v3 parity gate")
    registry = load_arm_registry(registry_identity.path)
    plan, plan_payload = load_compute_plan(plan_identity.path)
    if registry.arm_ids != plan.logical_arm_ids:
        raise ValueError("v3 arm registry and compute plan disagree")
    if (
        str(plan_payload.get("config_sha256", "")).upper() != config_identity.sha256
        or str(plan_payload.get("manifest_sha256", "")).upper()
        != manifest_identity.sha256
    ):
        raise ValueError("v3 compute plan is stale for the active protocol")
    gate = _json(parity_identity.path, "v3 parity gate")
    if (
        int(gate.get("schema_version", -1)) != 1
        or gate.get("decision") != "PASS"
        or gate.get("passed") is not True
        or int(gate.get("expected_series", -1)) != expected_series
        or int(gate.get("completed_series", -1)) != expected_series
        or str(gate.get("config_sha256", "")).upper()
        != config_identity.sha256
        or str(gate.get("manifest_sha256", "")).upper() != manifest_identity.sha256
    ):
        raise ValueError("v3 evaluator requires the complete passing parity gate")
    stage_root = Path(config["paths"]["run_root"]) / RUN_NAME
    return EvaluationProtocol(
        config_path,
        config,
        config_identity,
        manifest_identity,
        tuple(records),
        registry_identity.path,
        registry_identity,
        registry,
        plan_identity.path,
        plan_identity,
        plan,
        plan_payload,
        parity_identity,
        stage_root,
    )


def _validate_common_provenance(
    payload: Mapping[str, Any],
    protocol: EvaluationProtocol,
    record: FullSeriesRecord,
    planned: PlannedArm,
) -> None:
    required_sha = (
        "config_sha256",
        "full_manifest_sha256",
        "compute_plan_sha256",
        "parity_gate_sha256",
        "source_sha256",
    )
    if int(payload.get("schema_version", -1)) != RUN_SCHEMA_VERSION:
        raise ValueError("v3 score transaction schema changed")
    if payload.get("series_id") != record.series_id or payload.get("arm") != planned.logical.arm_id:
        raise ValueError("v3 score transaction identity mismatch")
    if any(not _is_sha256(payload.get(field)) for field in required_sha):
        raise ValueError("v3 score transaction is missing hash provenance")
    if (
        str(payload["config_sha256"]).upper() != protocol.config_identity.sha256
        or str(payload["full_manifest_sha256"]).upper()
        != protocol.manifest_identity.sha256
        or str(payload["compute_plan_sha256"]).upper() != protocol.plan_identity.sha256
        or str(payload["parity_gate_sha256"]).upper()
        != protocol.parity_gate_identity.sha256
        or str(payload.get("parameter_sha256", "")).upper()
        != planned.parameter_sha256.upper()
        or int(payload.get("encoder_calls", -1)) != 0
    ):
        raise ValueError("v3 score transaction provenance differs from the frozen protocol")


def _validate_canonical_score(
    series_root: Path,
    protocol: EvaluationProtocol,
    record: FullSeriesRecord,
    planned: PlannedArm,
) -> tuple[ScoreRecord, FileIdentity]:
    arm_root = series_root / planned.logical.arm_id
    paths = {
        name: arm_root / name
        for name in (
            "score.npy",
            "score_manifest.json",
            "_SCORES_READY.json",
            "_SUCCESS.json",
        )
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"incomplete v3 canonical score transaction: {arm_root}")
    manifest = _json(paths["score_manifest.json"], "v3 score manifest")
    _validate_common_provenance(manifest, protocol, record, planned)
    if (
        manifest.get("dataset") != record.dataset
        or manifest.get("track") != record.track
        or manifest.get("paper_group") != record.paper_group
        or manifest.get("signal_name") != record.signal_name
        or str(manifest.get("data_sha256", "")).upper()
        != record.expected_sha256.upper()
    ):
        raise ValueError("v3 score hierarchy/data identity differs from the full manifest")
    score_identity = _identity(
        paths["score.npy"], str(manifest.get("score_sha256", "")), "v3 score"
    )
    values = np.load(score_identity.path, allow_pickle=False)
    if (
        values.shape != (record.expected_length,)
        or values.dtype != np.float64
        or not np.isfinite(values).all()
        or int(manifest.get("score_length", -1)) != record.expected_length
        or str(manifest.get("score_dtype", "")) != "float64"
    ):
        raise ValueError("v3 score must be finite float64 with the frozen series length")
    ready = _json(paths["_SCORES_READY.json"], "v3 score-ready marker")
    expected_ready = {
        "series_id": record.series_id,
        "arm": planned.logical.arm_id,
        "score_sha256": score_identity.sha256,
        "config_sha256": protocol.config_identity.sha256,
        "compute_plan_sha256": protocol.plan_identity.sha256,
        "source_sha256": str(manifest["source_sha256"]).upper(),
        "encoder_calls": 0,
    }
    success = _json(paths["_SUCCESS.json"], "v3 arm success marker")
    expected_success = {
        "series_id": record.series_id,
        "arm": planned.logical.arm_id,
        "score_sha256": score_identity.sha256,
        "encoder_calls": 0,
    }
    if ready != expected_ready or success != expected_success:
        raise ValueError("v3 canonical score marker payload is not exact")
    return (
        ScoreRecord(
            record.series_id,
            planned.logical.arm_id,
            planned.logical.arm_id,
            score_identity.path,
            score_identity.sha256,
            manifest,
            manifest,
        ),
        score_identity,
    )


def _validate_alias(
    series_root: Path,
    protocol: EvaluationProtocol,
    record: FullSeriesRecord,
    planned: PlannedArm,
    canonical: ScoreRecord,
) -> ScoreRecord:
    arm_root = series_root / planned.logical.arm_id
    manifest_path = arm_root / "alias_manifest.json"
    success_path = arm_root / "_SUCCESS.json"
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(f"incomplete v3 alias transaction: {arm_root}")
    if any((arm_root / name).exists() for name in ("score.npy", "score_manifest.json", "_SCORES_READY.json")):
        raise ValueError("v3 alias must not duplicate a canonical score transaction")
    manifest = _json(manifest_path, "v3 alias manifest")
    _validate_common_provenance(manifest, protocol, record, planned)
    if manifest.get("canonical_arm") != planned.canonical_arm:
        raise ValueError("v3 alias canonical arm differs from the compute plan")
    relative = manifest.get("canonical_score_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("v3 alias canonical score path is missing")
    resolved = (arm_root / relative).resolve(strict=True)
    expected = canonical.score_path.resolve(strict=True)
    if resolved != expected:
        raise ValueError("v3 alias score path escapes or differs from its canonical arm")
    if str(manifest.get("canonical_score_sha256", "")).upper() != canonical.score_sha256:
        raise ValueError("v3 alias score hash differs from its canonical arm")
    success = _json(success_path, "v3 alias success marker")
    if success != {
        "series_id": record.series_id,
        "arm": planned.logical.arm_id,
        "canonical_arm": planned.canonical_arm,
        "encoder_calls": 0,
    }:
        raise ValueError("v3 alias success marker payload is not exact")
    return ScoreRecord(
        record.series_id,
        planned.logical.arm_id,
        planned.canonical_arm,
        canonical.score_path,
        canonical.score_sha256,
        canonical.score_manifest,
        manifest,
    )


def _retained_failures(protocol: EvaluationProtocol) -> tuple[Path, ...]:
    paths = list(protocol.stage_root.rglob("_FAILED.json")) if protocol.stage_root.exists() else []
    failure_root = Path(protocol.config["paths"]["failure_root"]) / RUN_NAME
    if failure_root.exists():
        # Scorer failures block label access. Evaluator audit failures are
        # retained in a child directory but do not permanently poison a later
        # corrected evaluator invocation.
        paths.extend(path for path in failure_root.glob("*.json") if path.is_file())
    return tuple(sorted({path.resolve() for path in paths}, key=lambda value: str(value).lower()))


def preflight_scores(protocol: EvaluationProtocol) -> EvaluationPreflight:
    """Validate the complete logical score grid before any label import/read."""

    if not protocol.stage_root.is_dir():
        raise FileNotFoundError("v3 cache-only score root is missing")
    failures = _retained_failures(protocol)
    running = tuple(protocol.stage_root.rglob("_RUNNING.json"))
    if failures or running:
        raise RuntimeError("retained failed or active v3 transactions block label access")
    expected_series = {record.series_id for record in protocol.records}
    actual_series = {path.name for path in protocol.stage_root.iterdir() if path.is_dir()}
    if actual_series != expected_series:
        raise RuntimeError("v3 score-directory coverage differs from the frozen manifest")

    scores: dict[tuple[str, str], ScoreRecord] = {}
    score_files: dict[str, FileIdentity] = {}
    data_files: dict[str, FileIdentity] = {}
    sources: set[str] = set()
    parity_hashes: set[str] = set()
    by_id = protocol.plan.by_id()
    for record in protocol.records:
        series_root = protocol.stage_root / record.series_id
        marker = _json(series_root / "_SUCCESS.json", "v3 series success marker")
        source_sha = str(marker.get("source_sha256", "")).upper()
        expected_marker = {
            "schema_version": RUN_SCHEMA_VERSION,
            "series_id": record.series_id,
            "logical_arm_count": len(protocol.plan.arms),
            "unique_computation_count": len(protocol.plan.canonical_arms),
            "encoder_calls": 0,
            "config_sha256": protocol.config_identity.sha256,
            "compute_plan_sha256": protocol.plan_identity.sha256,
            "source_sha256": source_sha,
        }
        if marker != expected_marker or not _is_sha256(source_sha):
            raise ValueError("v3 series success marker is stale or incomplete")
        canonical: dict[str, ScoreRecord] = {}
        for planned in protocol.plan.canonical_arms:
            artifact, identity = _validate_canonical_score(
                series_root, protocol, record, planned
            )
            canonical[planned.logical.arm_id] = artifact
            scores[(record.series_id, planned.logical.arm_id)] = artifact
            score_files[str(identity.path).lower()] = identity
            sources.add(str(artifact.score_manifest["source_sha256"]).upper())
            parity_hashes.add(str(artifact.score_manifest["parity_gate_sha256"]).upper())
        for arm_id in protocol.plan.logical_arm_ids:
            planned = by_id[arm_id]
            if not planned.is_alias:
                continue
            artifact = _validate_alias(
                series_root,
                protocol,
                record,
                planned,
                canonical[planned.canonical_arm],
            )
            scores[(record.series_id, arm_id)] = artifact
            sources.add(str(artifact.transaction_manifest["source_sha256"]).upper())
            parity_hashes.add(
                str(artifact.transaction_manifest["parity_gate_sha256"]).upper()
            )
        data_path = Path(protocol.config["data"]["root"]) / record.relative_path
        data_files[record.series_id] = _identity(
            data_path, record.expected_sha256, "v3 source data"
        )

    expected_count = len(protocol.records) * len(protocol.plan.arms)
    if len(scores) != expected_count:
        raise RuntimeError("v3 evaluator preflight found incomplete logical-arm coverage")
    if len(sources) != 1 or len(parity_hashes) != 1:
        raise ValueError("v3 score transactions do not share source/parity provenance")
    parity_sha = next(iter(parity_hashes))
    if parity_sha != protocol.parity_gate_identity.sha256:
        raise ValueError("v3 score transactions reference a different parity gate")
    digest = hashlib.sha256()
    for key, artifact in sorted(scores.items()):
        line = "|".join(
            (key[0], key[1], artifact.canonical_arm, artifact.score_sha256)
        ).encode("utf-8")
        digest.update(len(line).to_bytes(4, "big"))
        digest.update(line)
    return EvaluationPreflight(
        scores,
        score_files,
        data_files,
        next(iter(sources)),
        parity_sha,
        digest.hexdigest().upper(),
    )


def _assert_identity_unchanged(identity: FileIdentity, context: str) -> None:
    stat = identity.path.stat()
    if int(stat.st_size) != identity.size or int(stat.st_mtime_ns) != identity.mtime_ns:
        raise RuntimeError(f"{context} changed after v3 global preflight")


def assert_preflight_immutable(
    protocol: EvaluationProtocol, preflight: EvaluationPreflight
) -> None:
    for identity, context in (
        (protocol.config_identity, "v3 config"),
        (protocol.manifest_identity, "v3 manifest"),
        (protocol.registry_identity, "v3 registry"),
        (protocol.plan_identity, "v3 compute plan"),
        (protocol.parity_gate_identity, "v3 parity gate"),
    ):
        _assert_identity_unchanged(identity, context)
        if sha256_file(identity.path).upper() != identity.sha256:
            raise RuntimeError(f"{context} hash changed after v3 global preflight")
    for identity in (*preflight.score_files.values(), *preflight.data_files.values()):
        _assert_identity_unchanged(identity, "v3 score/data input")


def _load_ground_truth_function() -> Callable[[Mapping[str, Any], str, np.ndarray], Any]:
    # This is the sole label-module import and is reached only after preflight.
    from measure_vit4ts.evaluator import load_ground_truth

    return load_ground_truth


def _default_timestamps(record: FullSeriesRecord, data_root: Path) -> np.ndarray:
    from measure_vit4ts.coordinate_envelope_runner import load_vendor_signal

    return load_vendor_signal(record, data_root).series.timestamps


def _label_config(
    config: Mapping[str, Any], records: Sequence[FullSeriesRecord]
) -> dict[str, Any]:
    payload = dict(config)
    payload["data"] = dict(config["data"])
    payload["scoring"] = {
        "series": [
            {"series_id": record.series_id, "relative_path": record.relative_path}
            for record in records
        ]
    }
    return payload


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_preflight(
    protocol: EvaluationProtocol,
    preflight: EvaluationPreflight,
    *,
    timestamp_loader: Callable[[FullSeriesRecord, Path], np.ndarray] = _default_timestamps,
    ground_truth_loader: Callable[[Mapping[str, Any], str, np.ndarray], Any] | None = None,
    f1_fn: Callable[..., tuple[float, float, float]] | None = None,
    auprc_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    vus_fn: Callable[[np.ndarray, np.ndarray, int], float] | None = None,
) -> EvaluationResult:
    """Load labels once, freeze one validity mask, and evaluate every arm."""

    assert_preflight_immutable(protocol, preflight)
    if ground_truth_loader is None:
        ground_truth_loader = _load_ground_truth_function()
    anomalies_identity = _identity(
        Path(protocol.config["data"]["anomalies_csv"]), None, "v3 anomaly table"
    )
    data_root = Path(protocol.config["data"]["root"])
    label_config = _label_config(protocol.config, protocol.records)
    timestamps: dict[str, np.ndarray] = {}
    truths: dict[str, Any] = {}
    index_rows: list[dict[str, Any]] = []
    for record in protocol.records:
        time_values = np.asarray(timestamp_loader(record, data_root))
        if time_values.ndim != 1 or time_values.size != record.expected_length:
            raise ValueError("v3 evaluator timestamp vector differs from the manifest")
        truth = ground_truth_loader(label_config, record.series_id, time_values)
        labels = np.asarray(truth.point_labels)
        if labels.shape != time_values.shape or not np.logical_or(labels == 0, labels == 1).all():
            raise ValueError("v3 evaluator ground truth must be aligned binary labels")
        timestamps[record.series_id] = np.ascontiguousarray(time_values)
        truths[record.series_id] = truth
        index_rows.append(
            {
                "series_id": record.series_id,
                "family": record.track,
                "subgroup": record.paper_group,
                "n_points": int(labels.size),
                "n_positive": int(labels.sum()),
            }
        )
    valid_mask = freeze_valid_series_mask(pd.DataFrame(index_rows))
    alpha_grid = tuple(float(value) for value in protocol.config["statistics"].get(
        "alpha_grid", (0.1, 0.01, 0.001)
    ))
    vus_window = int(protocol.config["statistics"]["vus_max_window"])
    planned = protocol.plan.by_id()
    rows: list[dict[str, Any]] = []
    for record in protocol.records:
        truth = truths[record.series_id]
        label_values = np.asarray(truth.point_labels, dtype=np.uint8)
        for arm in protocol.registry.arm_ids:
            artifact = preflight.scores[(record.series_id, arm)]
            _assert_identity_unchanged(
                preflight.score_files[str(artifact.score_path).lower()], "v3 score"
            )
            score = np.load(artifact.score_path, allow_pickle=False)
            kwargs: dict[str, Any] = {}
            if f1_fn is not None:
                kwargs["f1_fn"] = f1_fn
            if auprc_fn is not None:
                kwargs["auprc_fn"] = auprc_fn
            if vus_fn is not None:
                kwargs["vus_fn"] = vus_fn
            values = evaluate_series(
                label_values,
                score,
                timestamps[record.series_id],
                truth.intervals,
                fp_threshold=protocol.registry.thresholds[arm],
                alpha_grid=alpha_grid,
                vus_max_window=vus_window,
                **kwargs,
            )
            item = planned[arm]
            manifest = artifact.score_manifest
            row = {
                "series_id": record.series_id,
                "family": record.track,
                "subgroup": record.paper_group,
                "dataset": record.dataset,
                "signal_name": record.signal_name,
                "arm": arm,
                "arm_order": protocol.registry.arm_ids.index(arm),
                "arm_role": next(spec.role for spec in protocol.registry.arms if spec.arm_id == arm),
                "ablation_family": item.logical.family,
                "canonical_arm": artifact.canonical_arm,
                "is_alias": artifact.is_alias,
                "parameter_sha256": item.parameter_sha256,
                **{key: _plain(value) for key, value in values.items()},
                "score_sha256": artifact.score_sha256,
                "data_sha256": record.expected_sha256.upper(),
                "config_sha256": protocol.config_identity.sha256,
                "full_manifest_sha256": protocol.manifest_identity.sha256,
                "compute_plan_sha256": protocol.plan_identity.sha256,
                "registry_sha256": protocol.registry_identity.sha256,
                "parity_gate_sha256": preflight.parity_gate_sha256,
                "score_source_sha256": preflight.source_sha256,
                "cache_sha256": str(manifest.get("cache_sha256", "")).upper(),
                "cache_manifest_sha256": str(
                    manifest.get("cache_manifest_sha256", "")
                ).upper(),
                "trace_sha256": str(manifest.get("trace_sha256", "")).upper(),
                "trace_manifest_sha256": str(
                    manifest.get("trace_manifest_sha256", "")
                ).upper(),
                "anomalies_sha256": anomalies_identity.sha256,
            }
            rows.append(row)
    per_series = validate_metrics_against_mask(
        pd.DataFrame(rows), valid_mask, protocol.registry.arm_ids
    )
    provenance = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "stage": "vittrace_ablation_full_v3",
        "run_name": RUN_NAME,
        "registry_id": protocol.registry.registry_id,
        "config_sha256": protocol.config_identity.sha256,
        "full_manifest_sha256": protocol.manifest_identity.sha256,
        "registry_sha256": protocol.registry_identity.sha256,
        "compute_plan_sha256": protocol.plan_identity.sha256,
        "parity_gate_sha256": preflight.parity_gate_sha256,
        "score_source_sha256": preflight.source_sha256,
        "score_index_sha256": preflight.score_index_sha256,
        "anomalies_sha256": anomalies_identity.sha256,
        "valid_mask_sha256": valid_mask_sha256(valid_mask),
        "series_count": len(protocol.records),
        "logical_arm_count": len(protocol.plan.arms),
        "unique_computation_count": len(protocol.plan.canonical_arms),
        "score_record_count": len(per_series),
        "alias_record_count": int(per_series["is_alias"].sum()),
        "label_load_count": len(protocol.records),
        "retained_failure_count": 0,
    }
    return EvaluationResult(per_series, valid_mask, provenance)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_evaluation_outputs(
    output_root: Path, result: EvaluationResult
) -> tuple[Path, Path, Path, Path]:
    root = Path(output_root)
    metrics_path = root / "per_series_metrics.csv"
    mask_path = root / "valid_series_mask.csv"
    provenance_path = root / "evaluation_provenance.json"
    marker_path = root / "_EVALUATION_COMPLETE.json"
    _atomic_csv(metrics_path, result.per_series)
    _atomic_csv(mask_path, result.valid_mask)
    provenance = dict(result.provenance)
    provenance.update(
        {
            "per_series_metrics_sha256": sha256_file(metrics_path).upper(),
            "valid_series_mask_file_sha256": sha256_file(mask_path).upper(),
        }
    )
    _atomic_json(provenance_path, provenance)
    _atomic_json(
        marker_path,
        {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "series_count": int(provenance["series_count"]),
            "logical_arm_count": int(provenance["logical_arm_count"]),
            "score_record_count": int(provenance["score_record_count"]),
            "valid_mask_sha256": str(provenance["valid_mask_sha256"]),
            "per_series_metrics_sha256": str(provenance["per_series_metrics_sha256"]),
            "valid_series_mask_file_sha256": str(
                provenance["valid_series_mask_file_sha256"]
            ),
            "evaluation_provenance_sha256": sha256_file(provenance_path).upper(),
        },
    )
    return metrics_path, mask_path, provenance_path, marker_path


def _preserve_failure(
    config: Mapping[str, Any] | None,
    config_path: Path,
    error: BaseException,
) -> Path | None:
    if config is None:
        return None
    root = Path(config["paths"]["failure_root"]) / RUN_NAME / "evaluator"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"evaluation_failure_{stamp}_{os.getpid()}.json"
    _atomic_json(
        path,
        {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "stage": "vittrace_ablation_full_v3",
            "config_path": str(Path(config_path).resolve()),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )
    return path


def evaluate_cache_only(
    config_path: Path,
    registry_path: Path | None = None,
    plan_path: Path | None = None,
    parity_gate_path: Path | None = None,
    output_root: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Run complete score preflight, delayed-label evaluation, and commit."""

    config: Mapping[str, Any] | None = None
    try:
        protocol = load_evaluation_protocol(
            config_path, registry_path, plan_path, parity_gate_path
        )
        config = protocol.config
        preflight = preflight_scores(protocol)
        result = evaluate_preflight(protocol, preflight)
        root = Path(
            output_root
            or Path(protocol.config["paths"]["result_root"]) / EVALUATION_DIRECTORY
        )
        return write_evaluation_outputs(root, result)
    except Exception as error:
        _preserve_failure(config, Path(config_path), error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--parity-gate", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    paths = evaluate_cache_only(
        args.config,
        args.registry,
        args.plan,
        args.parity_gate,
        args.output_root,
    )
    print(json.dumps({"outputs": [str(path) for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EVALUATION_DIRECTORY",
    "EvaluationPreflight",
    "EvaluationProtocol",
    "EvaluationResult",
    "FileIdentity",
    "RUN_NAME",
    "ScoreRecord",
    "assert_preflight_immutable",
    "evaluate_cache_only",
    "evaluate_preflight",
    "load_evaluation_protocol",
    "preflight_scores",
    "write_evaluation_outputs",
]
