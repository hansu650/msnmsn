"""Fail-closed delayed-label evaluator across all ViTTrace v3 score stages.

The command has three deliberate phases: freeze a stage registry, preflight the
entire registered score grid without importing any label code, and only then
evaluate.  Incomplete stages produce a durable ``BLOCKED`` marker and can be
rechecked after more score transactions arrive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from measure_vit4ts.full_manifest import FullSeriesRecord, load_manifest

from . import evaluator as cache_evaluator
from .cache_registry import load_compute_plan
from .combined_protocol import (
    CombinedArmSpec,
    CombinedProtocolSpec,
    CombinedStageSpec,
    EXPECTED_VALID_SERIES,
    canonical_json_sha256,
    load_combined_protocol,
    sha256_file,
)
from .metrics import (
    DETECTION_METRICS,
    evaluate_series,
    freeze_valid_series_mask,
    valid_mask_sha256,
    validate_metrics_against_mask,
)
from .registry import ArmRegistry, ArmSpec, ContrastSpec


SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/vittrace_ablation_full_v3.yaml")
OUTPUT_DIRECTORY = "combined_evaluation"
_SOURCE_FILES = ("combined_protocol.py", "combined_evaluator.py", "combined_aggregate.py")


class CombinedBlocked(RuntimeError):
    """Raised after a durable BLOCKED report is available."""


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class CombinedScoreRecord:
    series_id: str
    stage_id: str
    stage_kind: str
    arm: str
    source_arm: str
    score_path: Path
    score_identity: FileIdentity
    manifest_path: Path
    manifest_identity: FileIdentity
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class StagePreflight:
    stage: CombinedStageSpec
    status: str
    reason: str
    expected_series: int
    completed_series: int
    expected_rows: int
    completed_rows: int
    status_path: Path | None
    status_sha256: str | None
    scores: Mapping[tuple[str, str], CombinedScoreRecord]


@dataclass(frozen=True)
class CombinedPreflight:
    protocol: CombinedProtocolSpec
    config: Mapping[str, Any]
    records: tuple[FullSeriesRecord, ...]
    data_files: Mapping[str, FileIdentity]
    stages: tuple[StagePreflight, ...]
    status: str
    reason: str
    score_index_sha256: str | None

    @property
    def scores(self) -> dict[tuple[str, str], CombinedScoreRecord]:
        output: dict[tuple[str, str], CombinedScoreRecord] = {}
        for stage in self.stages:
            output.update(stage.scores)
        return output


@dataclass(frozen=True)
class CombinedEvaluationResult:
    per_series: pd.DataFrame
    valid_mask: pd.DataFrame
    arm_metadata: pd.DataFrame
    provenance: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity(path: Path, expected: str | None = None) -> FileIdentity:
    target = Path(path).resolve(strict=True)
    digest = sha256_file(target)
    if expected is not None and digest != str(expected).upper():
        raise ValueError(f"SHA256 mismatch: {target}")
    stat = target.stat()
    return FileIdentity(target, digest, int(stat.st_size), int(stat.st_mtime_ns))


def _assert_unchanged(identity: FileIdentity, context: str) -> None:
    stat = identity.path.stat()
    if int(stat.st_size) != identity.size or int(stat.st_mtime_ns) != identity.mtime_ns:
        raise RuntimeError(f"{context} changed after combined preflight")
    if sha256_file(identity.path) != identity.sha256:
        raise RuntimeError(f"{context} hash changed after combined preflight")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def combined_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        path = root / name
        if not path.is_file():
            continue
        payload = path.read_bytes()
        digest.update(name.encode("ascii"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def _load_bound_inputs(
    protocol: CombinedProtocolSpec,
) -> tuple[Mapping[str, Any], tuple[FullSeriesRecord, ...]]:
    config = yaml.safe_load(protocol.config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("stage") != "vittrace_ablation_full_v3":
        raise ValueError("combined evaluator accepts only the frozen v3 stage")
    if Path(config["manifest"]["path"]).resolve() != protocol.manifest_path:
        raise ValueError("combined protocol manifest path differs from config")
    _, records = load_manifest(protocol.manifest_path)
    if (
        len(records) != protocol.expected_series
        or len({record.series_id for record in records}) != protocol.expected_series
    ):
        raise ValueError("combined protocol requires the exact unique manifest series set")
    return config, tuple(records)


def _preflight_data(
    config: Mapping[str, Any], records: Sequence[FullSeriesRecord]
) -> dict[str, FileIdentity]:
    root = Path(config["data"]["root"])
    return {
        record.series_id: _identity(root / record.relative_path, record.expected_sha256)
        for record in records
    }


def _failure_paths(
    config: Mapping[str, Any], stage: CombinedStageSpec
) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    if stage.root.exists():
        candidates.update(stage.root.rglob("_FAILED.json"))
        candidates.update(stage.root.rglob("*.failed.json"))
    failure_root = Path(config["paths"]["failure_root"])
    aliases = {
        stage.stage_id,
        stage.configuration_id,
        stage.stage_kind,
        stage.stage_kind.replace("_line", ""),
    }
    for name in aliases:
        path = failure_root / name
        if path.exists():
            for item in path.rglob("*.json"):
                relative = item.relative_to(path)
                if relative.parts and relative.parts[0].lower() == "evaluator":
                    continue
                candidates.add(item)
    return tuple(sorted((path.resolve() for path in candidates), key=str))


def _blocked(stage: CombinedStageSpec, reason: str, completed_series: int = 0,
             completed_rows: int = 0, status_path: Path | None = None) -> StagePreflight:
    return StagePreflight(
        stage,
        "BLOCKED",
        reason,
        stage.expected_series,
        int(completed_series),
        stage.expected_series * len(stage.arms),
        int(completed_rows),
        status_path,
        sha256_file(status_path) if status_path and status_path.is_file() else None,
        {},
    )


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _score_record(
    stage: CombinedStageSpec,
    arm: CombinedArmSpec,
    record: FullSeriesRecord,
    manifest_path: Path,
    *,
    score_hash_field: str,
    required_identity: Mapping[str, str],
    require_status: bool = True,
) -> CombinedScoreRecord:
    manifest_identity = _identity(manifest_path)
    payload = _read_json(manifest_path, "combined score manifest")
    if require_status and payload.get("status") != "PASS":
        raise ValueError("combined score transaction is not PASS")
    if payload.get("series_id", record.series_id) != record.series_id:
        raise ValueError("combined score series identity differs")
    if payload.get("arm") != arm.source_arm:
        raise ValueError("combined source-arm identity differs")
    if str(payload.get("data_sha256", record.expected_sha256)).upper() != record.expected_sha256.upper():
        raise ValueError("combined score data provenance differs")
    for key, value in required_identity.items():
        if str(payload.get(key, "")).upper() != str(value).upper():
            raise ValueError(f"combined score provenance differs: {key}")
    score_path = Path(payload["score_path"]).resolve(strict=True)
    score_identity = _identity(score_path, str(payload[score_hash_field]))
    score = np.load(score_path, mmap_mode="r", allow_pickle=False)
    if score.dtype != np.float64 or score.shape != (record.expected_length,):
        raise ValueError("combined score must be canonical float64 [T]")
    # A chunked finite check avoids retaining any complete score in RAM.
    for start in range(0, score.size, 1 << 18):
        if not np.isfinite(score[start : start + (1 << 18)]).all():
            raise ValueError("combined score contains non-finite values")
    return CombinedScoreRecord(
        record.series_id,
        stage.stage_id,
        stage.stage_kind,
        arm.arm_id,
        arm.source_arm,
        score_path,
        score_identity,
        manifest_path.resolve(),
        manifest_identity,
        payload,
    )


def _relaxed_cache_protocol(
    protocol: CombinedProtocolSpec,
    config: Mapping[str, Any],
    records: tuple[FullSeriesRecord, ...],
    stage: CombinedStageSpec,
) -> cache_evaluator.EvaluationProtocol:
    output_root = Path(config["paths"]["output_root"])
    registry_path = output_root / "manifests" / "cache_only_arm_registry.json"
    plan_path = output_root / "manifests" / "cache_only_compute_plan.json"
    gate_path = output_root / "provenance" / "cache_only_parity_gate.json"
    registry_payload = _read_json(registry_path, "cache-only arm registry")
    plan, plan_payload = load_compute_plan(plan_path)
    arm_rows = registry_payload["arms"]
    arms = tuple(
        ArmSpec(str(row["id"]), str(row["role"]), int(row["order"]), float(row["fp_threshold"]))
        for row in arm_rows
    )
    contrasts = tuple(
        ContrastSpec(str(row["id"]), str(row["family"]), str(row["candidate"]), str(row["control"]))
        for row in registry_payload["contrasts"]
    )
    registry = ArmRegistry(
        str(registry_payload["registry_id"]),
        str(registry_payload["primary_arm"]),
        str(registry_payload["control_arm"]),
        arms,
        contrasts,
    )
    if registry.arm_ids != plan.logical_arm_ids:
        raise ValueError("cache-only registry and plan disagree")
    gate = _read_json(gate_path, "cache-only parity gate")
    if (
        gate.get("decision") != "PASS"
        or gate.get("passed") is not True
        or int(gate.get("completed_series", -1)) != protocol.expected_series
    ):
        raise ValueError("cache-only parity gate is not complete PASS")
    return cache_evaluator.EvaluationProtocol(
        protocol.config_path,
        config,
        cache_evaluator._identity(protocol.config_path, protocol.config_sha256, "config"),
        cache_evaluator._identity(protocol.manifest_path, protocol.manifest_sha256, "manifest"),
        records,
        registry_path,
        cache_evaluator._identity(registry_path, None, "registry"),
        registry,
        plan_path,
        cache_evaluator._identity(plan_path, None, "plan"),
        plan,
        plan_payload,
        cache_evaluator._identity(gate_path, None, "gate"),
        stage.root,
    )


def _preflight_cache_only(
    protocol: CombinedProtocolSpec,
    config: Mapping[str, Any],
    records: tuple[FullSeriesRecord, ...],
    stage: CombinedStageSpec,
) -> StagePreflight:
    expected_ids = {record.series_id for record in records}
    if not stage.root.is_dir():
        return _blocked(stage, "score root is missing")
    actual = {path.name for path in stage.root.iterdir() if path.is_dir()}
    complete = sum((stage.root / series_id / "_SUCCESS.json").is_file() for series_id in expected_ids)
    if actual != expected_ids or complete != stage.expected_series:
        return _blocked(
            stage,
            f"cache-only stage incomplete: {complete}/{stage.expected_series} series",
            complete,
            complete * len(stage.arms),
        )
    try:
        cache_protocol = _relaxed_cache_protocol(protocol, config, records, stage)
        source = cache_evaluator.preflight_scores(cache_protocol)
        by_source = {arm.source_arm: arm for arm in stage.arms}
        all_sources = set(cache_protocol.registry.arm_ids)
        if not set(by_source).issubset(all_sources):
            raise ValueError("combined cache arm map is not a frozen-plan subset")
        scores: dict[tuple[str, str], CombinedScoreRecord] = {}
        for (series_id, source_arm), item in source.scores.items():
            if source_arm not in by_source:
                continue
            arm = by_source[source_arm]
            manifest_path = (
                item.score_path.parent / "score_manifest.json"
                if not item.is_alias
                else item.score_path.parent.parent / source_arm / "alias_manifest.json"
            )
            # Alias metadata lives in its own directory, while its score is the
            # immutable canonical file already verified by cache_evaluator.
            if item.is_alias:
                manifest_path = stage.root / series_id / source_arm / "alias_manifest.json"
            scores[(series_id, arm.arm_id)] = CombinedScoreRecord(
                series_id,
                stage.stage_id,
                stage.stage_kind,
                arm.arm_id,
                source_arm,
                item.score_path,
                _identity(item.score_path, item.score_sha256),
                manifest_path,
                _identity(manifest_path),
                item.transaction_manifest,
            )
        return StagePreflight(
            stage,
            "READY",
            "complete fail-closed cache-only preflight",
            stage.expected_series,
            stage.expected_series,
            stage.expected_series * len(stage.arms),
            len(scores),
            None,
            source.score_index_sha256,
            scores,
        )
    except Exception as exc:
        return _blocked(stage, f"{type(exc).__name__}: {exc}", complete)


def _status_stage(
    protocol: CombinedProtocolSpec,
    records: tuple[FullSeriesRecord, ...],
    stage: CombinedStageSpec,
    *,
    status_name: str,
    transaction_directory: str,
    score_hash_field: str,
    source_hash_field: str,
) -> StagePreflight:
    status_path = stage.status_path or stage.root / status_name
    if not status_path.is_file():
        return _blocked(stage, "stage status file is missing", status_path=status_path)
    try:
        status = _read_json(status_path, "combined stage status")
        completed_series = int(status.get("completed_series", 0))
        completed_rows = int(status.get("completed_rows", 0))
        expected_rows = stage.expected_series * len(stage.arms)
        if (
            status.get("status") != "COMPLETE"
            or int(status.get("expected_series", -1)) != stage.expected_series
            or int(status.get("expected_rows", -1)) != expected_rows
            or completed_series != stage.expected_series
            or completed_rows != expected_rows
        ):
            return _blocked(
                stage,
                f"stage status incomplete: {completed_series}/{stage.expected_series} series, "
                f"{completed_rows}/{expected_rows} rows",
                completed_series,
                completed_rows,
                status_path,
            )
        expected_sources = [arm.source_arm for arm in stage.arms]
        if list(status.get("expected_arms", [])) != expected_sources:
            raise ValueError("stage status arm registry differs")
        rows = status.get("rows", [])
        grid = {(str(row.get("series_id")), str(row.get("arm"))) for row in rows if row.get("status") == "PASS"}
        wanted = {(record.series_id, arm.source_arm) for record in records for arm in stage.arms}
        if grid != wanted or len(rows) != expected_rows:
            raise ValueError("stage status PASS grid differs from manifest x arms")
        required = {
            "config_sha256": protocol.config_sha256,
            "manifest_sha256": protocol.manifest_sha256,
            "encoder_source_sha256": str(status["encoder_source_sha256"]),
            source_hash_field: str(status[source_hash_field]),
            "variant_sha256": str(status["variant_sha256"]),
        }
        if str(status["config_sha256"]).upper() != protocol.config_sha256:
            raise ValueError("stage config hash differs")
        if str(status["manifest_sha256"]).upper() != protocol.manifest_sha256:
            raise ValueError("stage manifest hash differs")
        # Dynamic transactions also bind one scoring configuration.
        if "scoring_config_sha256" in status:
            required["scoring_config_sha256"] = str(status["scoring_config_sha256"])
        scores: dict[tuple[str, str], CombinedScoreRecord] = {}
        for record in records:
            for arm in stage.arms:
                path = stage.root / transaction_directory / record.series_id / arm.source_arm / "score_manifest.json"
                item = _score_record(
                    stage,
                    arm,
                    record,
                    path,
                    score_hash_field=score_hash_field,
                    required_identity=required,
                )
                scores[(record.series_id, arm.arm_id)] = item
        return StagePreflight(
            stage,
            "READY",
            "complete fail-closed transactional preflight",
            stage.expected_series,
            completed_series,
            expected_rows,
            len(scores),
            status_path,
            sha256_file(status_path),
            scores,
        )
    except Exception as exc:
        return _blocked(stage, f"{type(exc).__name__}: {exc}", status_path=status_path)


def _preflight_spectrogram(
    protocol: CombinedProtocolSpec,
    records: tuple[FullSeriesRecord, ...],
    stage: CombinedStageSpec,
) -> StagePreflight:
    scores: dict[tuple[str, str], CombinedScoreRecord] = {}
    completed_series = 0
    status_digests: list[str] = []
    try:
        for record in records:
            series_root = stage.root / record.series_id
            status_path = series_root / "spectrogram_scores_status.json"
            if not status_path.is_file():
                continue
            status = _read_json(status_path, "spectrogram series status")
            if (
                status.get("status") != "COMPLETE"
                or int(status.get("completed_arms", -1)) != len(stage.arms)
                or list(status.get("expected_arms", [])) != [arm.source_arm for arm in stage.arms]
                or str(status.get("config_file_sha256", "")).upper() != protocol.config_sha256
            ):
                raise ValueError(f"invalid spectrogram status: {record.series_id}")
            status_digests.append(sha256_file(status_path))
            required = {
                "config_file_sha256": protocol.config_sha256,
                "route_config_sha256": str(status["route_config_sha256"]),
                "spectrogram_source_sha256": str(status["spectrogram_source_sha256"]),
                "renderer_source_sha256": str(status["renderer_source_sha256"]),
            }
            for arm in stage.arms:
                item = _score_record(
                    stage,
                    arm,
                    record,
                    series_root / arm.source_arm / "score_manifest.json",
                    score_hash_field="score_file_sha256",
                    required_identity=required,
                    require_status=False,
                )
                render_path = Path(item.manifest["render_manifest_path"]).resolve(strict=True)
                if sha256_file(render_path) != str(item.manifest["render_manifest_sha256"]).upper():
                    raise ValueError("spectrogram render manifest hash differs")
                render = _read_json(render_path, "spectrogram render manifest")
                if (
                    render.get("series_id") != record.series_id
                    or str(render.get("data_sha256", "")).upper() != record.expected_sha256.upper()
                ):
                    raise ValueError("spectrogram render/data identity differs")
                scores[(record.series_id, arm.arm_id)] = item
            completed_series += 1
        expected_rows = stage.expected_series * len(stage.arms)
        if completed_series != stage.expected_series or len(scores) != expected_rows:
            return _blocked(
                stage,
                f"spectrogram stage incomplete: {completed_series}/{stage.expected_series} series",
                completed_series,
                len(scores),
                stage.status_path,
            )
        status_sha = canonical_json_sha256(status_digests)
        return StagePreflight(
            stage,
            "READY",
            "complete fail-closed per-series spectrogram preflight",
            stage.expected_series,
            completed_series,
            expected_rows,
            len(scores),
            stage.status_path,
            status_sha,
            scores,
        )
    except Exception as exc:
        return _blocked(
            stage,
            f"{type(exc).__name__}: {exc}",
            completed_series,
            len(scores),
            stage.status_path,
        )


def preflight_combined(protocol: CombinedProtocolSpec) -> CombinedPreflight:
    """Preflight every registered stage and source-data hash, without labels."""

    config, records = _load_bound_inputs(protocol)
    data_files = _preflight_data(config, records)
    stages: list[StagePreflight] = []
    for stage in protocol.stages:
        failures = _failure_paths(config, stage)
        if failures:
            stages.append(_blocked(stage, f"{len(failures)} retained stage failure(s)"))
            continue
        if stage.stage_kind == "cache_only":
            result = _preflight_cache_only(protocol, config, records, stage)
        elif stage.stage_kind == "encoder_controls":
            result = _status_stage(
                protocol,
                records,
                stage,
                status_name="encoder_controls_status.json",
                transaction_directory="controls",
                score_hash_field="score_sha256",
                source_hash_field="control_source_sha256",
            )
        elif stage.stage_kind == "dynamic_line":
            result = _status_stage(
                protocol,
                records,
                stage,
                status_name="dynamic_scores_status.json",
                transaction_directory="dynamic_scores",
                score_hash_field="score_sha256",
                source_hash_field="dynamic_score_source_sha256",
            )
        else:
            result = _preflight_spectrogram(protocol, records, stage)
        stages.append(result)
    blocked = [row for row in stages if row.status != "READY"]
    if blocked:
        status = "BLOCKED"
        reason = "; ".join(f"{row.stage.stage_id}: {row.reason}" for row in blocked)
        score_index = None
    else:
        status = "READY"
        reason = "all registered stages passed immutable preflight"
        digest = hashlib.sha256()
        for row in stages:
            for key, item in sorted(row.scores.items()):
                value = "|".join((row.stage.stage_id, key[0], key[1], item.score_identity.sha256)).encode("utf-8")
                digest.update(len(value).to_bytes(4, "big"))
                digest.update(value)
        score_index = digest.hexdigest().upper()
    return CombinedPreflight(
        protocol,
        config,
        records,
        data_files,
        tuple(stages),
        status,
        reason,
        score_index,
    )


def _preflight_payload(preflight: CombinedPreflight) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": preflight.status,
        "reason": preflight.reason,
        "protocol_id": preflight.protocol.protocol_id,
        "protocol_sha256": preflight.protocol.payload_sha256,
        "config_sha256": preflight.protocol.config_sha256,
        "manifest_sha256": preflight.protocol.manifest_sha256,
        "expected_series": preflight.protocol.expected_series,
        "expected_valid_series": preflight.protocol.expected_valid_series,
        "score_index_sha256": preflight.score_index_sha256,
        "label_modules_imported": False,
        "stages": [
            {
                "stage_id": row.stage.stage_id,
                "stage_group": row.stage.stage_group,
                "stage_kind": row.stage.stage_kind,
                "configuration_id": row.stage.configuration_id,
                "status": row.status,
                "reason": row.reason,
                "expected_series": row.expected_series,
                "completed_series": row.completed_series,
                "expected_rows": row.expected_rows,
                "completed_rows": row.completed_rows,
                "status_path": str(row.status_path) if row.status_path else None,
                "status_sha256": row.status_sha256,
            }
            for row in preflight.stages
        ],
    }


def write_preflight(output_root: Path, preflight: CombinedPreflight) -> Path:
    root = Path(output_root)
    path = root / "combined_preflight_status.json"
    payload = _preflight_payload(preflight)
    _atomic_json(path, payload)
    marker = root / (
        "_COMBINED_PREFLIGHT_READY.json"
        if preflight.status == "READY"
        else "_COMBINED_EVALUATION_BLOCKED.json"
    )
    _atomic_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "status": preflight.status,
            "reason": preflight.reason,
            "protocol_sha256": preflight.protocol.payload_sha256,
            "preflight_sha256": sha256_file(path),
            "label_modules_imported": False,
        },
    )
    (root / ("_COMBINED_EVALUATION_BLOCKED.json" if preflight.status == "READY" else "_COMBINED_PREFLIGHT_READY.json")).unlink(missing_ok=True)
    _atomic_json(root / "stage_evaluation_index.json", _stage_index(root, preflight, None))
    return path


def _default_timestamp_loader(record: FullSeriesRecord, data_root: Path) -> np.ndarray:
    from measure_vit4ts.coordinate_envelope_runner import load_vendor_signal

    return load_vendor_signal(record, data_root).series.timestamps


def _ground_truth_loader() -> Callable[[Mapping[str, Any], str, np.ndarray], Any]:
    from measure_vit4ts.evaluator import load_ground_truth

    return load_ground_truth


def _label_config(config: Mapping[str, Any], records: Sequence[FullSeriesRecord]) -> dict[str, Any]:
    payload = dict(config)
    payload["data"] = dict(config["data"])
    payload["scoring"] = {
        "series": [
            {"series_id": record.series_id, "relative_path": record.relative_path}
            for record in records
        ]
    }
    return payload


def _arm_metadata_frame(protocol: CombinedProtocolSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in protocol.stages:
        for arm in stage.arms:
            metadata = dict(arm.metadata)
            fixed = dict(arm.fixed_factors)
            model = metadata.get("model_name", metadata.get("model_key", fixed.get("model_key")))
            patch_grid = metadata.get("patch_grid", fixed.get("patch_grid"))
            rows.append(
                {
                    "arm": arm.arm_id,
                    "source_arm": arm.source_arm,
                    "stage_id": stage.stage_id,
                    "stage_group": stage.stage_group,
                    "stage_kind": stage.stage_kind,
                    "configuration_id": stage.configuration_id,
                    "arm_role": arm.role,
                    "arm_order": arm.order,
                    "experiment_group": arm.experiment_group,
                    "changed_factor": arm.changed_factor,
                    "display_name": metadata.get("display_name", arm.arm_id),
                    "is_final": metadata.get("is_final"),
                    "fixed_factors_json": json.dumps(fixed, sort_keys=True),
                    "arm_metadata_json": json.dumps(metadata, sort_keys=True),
                    "backbone": model,
                    "representation": metadata.get("representation", fixed.get("representation", "line")),
                    "window": metadata.get("window", fixed.get("window")),
                    "stride": metadata.get("stride", fixed.get("stride")),
                    "patch_size": metadata.get("patch_size", fixed.get("patch_size")),
                    "patch_grid": json.dumps(patch_grid) if patch_grid is not None else None,
                    "matching_scope": metadata.get("matching_scope"),
                    "memory": metadata.get("memory"),
                    "scale_subset": json.dumps(metadata.get("scales")) if metadata.get("scales") is not None else None,
                    "incidence": metadata.get("incidence"),
                    "temporal": metadata.get("temporal"),
                    "reducer_family": metadata.get("reducer_kind"),
                    "reducer_setting": metadata.get("reducer_value"),
                    "ihp": metadata.get("ihp", "literal" if metadata.get("incidence") == "literal" else "released"),
                    "nctp": metadata.get("nctp", str(metadata.get("temporal", "")).startswith("nctp")),
                    "encoder_calls": metadata.get("encoder_calls", 0),
                    "elapsed_seconds_per_series": metadata.get("elapsed_seconds_per_series"),
                    "fp_threshold": arm.fp_threshold,
                }
            )
    return pd.DataFrame(rows).sort_values("arm_order").reset_index(drop=True)


def evaluate_combined(
    preflight: CombinedPreflight,
    *,
    timestamp_loader: Callable[[FullSeriesRecord, Path], np.ndarray] = _default_timestamp_loader,
    ground_truth_loader: Callable[[Mapping[str, Any], str, np.ndarray], Any] | None = None,
    f1_fn: Callable[..., tuple[float, float, float]] | None = None,
    auprc_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    vus_fn: Callable[[np.ndarray, np.ndarray, int], float] | None = None,
) -> CombinedEvaluationResult:
    if preflight.status != "READY":
        raise CombinedBlocked(preflight.reason)
    for identity in preflight.data_files.values():
        _assert_unchanged(identity, "source data")
    for item in preflight.scores.values():
        _assert_unchanged(item.score_identity, "score")
        _assert_unchanged(item.manifest_identity, "score manifest")
    loader = ground_truth_loader or _ground_truth_loader()
    data_root = Path(preflight.config["data"]["root"])
    label_config = _label_config(preflight.config, preflight.records)
    timestamps: dict[str, np.ndarray] = {}
    truths: dict[str, Any] = {}
    index_rows: list[dict[str, Any]] = []
    for record in preflight.records:
        time_values = np.asarray(timestamp_loader(record, data_root))
        if time_values.shape != (record.expected_length,):
            raise ValueError("combined timestamp vector differs from manifest")
        truth = loader(label_config, record.series_id, time_values)
        labels = np.asarray(truth.point_labels)
        if labels.shape != time_values.shape or not np.logical_or(labels == 0, labels == 1).all():
            raise ValueError("combined ground truth must be aligned binary labels")
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
    valid_counts = {metric: int(valid_mask[f"valid_{metric}"].sum()) for metric in DETECTION_METRICS}
    if set(valid_counts.values()) != {preflight.protocol.expected_valid_series}:
        raise ValueError(
            f"combined common validity mask changed: {valid_counts}, expected "
            f"{preflight.protocol.expected_valid_series}"
        )
    alpha_grid = tuple(float(value) for value in preflight.config["statistics"].get("alpha_grid", (0.1, 0.01, 0.001)))
    vus_window = int(preflight.config["statistics"]["vus_max_window"])
    arm_by_id = {arm.arm_id: arm for arm in preflight.protocol.arms}
    stage_by_arm = {
        arm.arm_id: stage
        for stage in preflight.protocol.stages
        for arm in stage.arms
    }
    rows: list[dict[str, Any]] = []
    for record in preflight.records:
        truth = truths[record.series_id]
        labels = np.asarray(truth.point_labels, dtype=np.uint8)
        for arm_id in preflight.protocol.arm_ids:
            arm = arm_by_id[arm_id]
            stage = stage_by_arm[arm_id]
            artifact = preflight.scores[(record.series_id, arm_id)]
            score = np.load(artifact.score_path, allow_pickle=False)
            kwargs: dict[str, Any] = {}
            if f1_fn is not None:
                kwargs["f1_fn"] = f1_fn
            if auprc_fn is not None:
                kwargs["auprc_fn"] = auprc_fn
            if vus_fn is not None:
                kwargs["vus_fn"] = vus_fn
            values = evaluate_series(
                labels,
                score,
                timestamps[record.series_id],
                truth.intervals,
                fp_threshold=arm.fp_threshold,
                alpha_grid=alpha_grid,
                vus_max_window=vus_window,
                **kwargs,
            )
            rows.append(
                {
                    "series_id": record.series_id,
                    "family": record.track,
                    "subgroup": record.paper_group,
                    "dataset": record.dataset,
                    "signal_name": record.signal_name,
                    "stage_id": stage.stage_id,
                    "stage_group": stage.stage_group,
                    "stage_kind": stage.stage_kind,
                    "configuration_id": stage.configuration_id,
                    "arm": arm_id,
                    "source_arm": arm.source_arm,
                    "arm_order": arm.order,
                    "arm_role": arm.role,
                    "experiment_group": arm.experiment_group,
                    "changed_factor": arm.changed_factor,
                    **values,
                    "score_sha256": artifact.score_identity.sha256,
                    "score_manifest_sha256": artifact.manifest_identity.sha256,
                    "data_sha256": record.expected_sha256.upper(),
                    "config_sha256": preflight.protocol.config_sha256,
                    "full_manifest_sha256": preflight.protocol.manifest_sha256,
                    "combined_protocol_sha256": preflight.protocol.payload_sha256,
                    "combined_source_sha256": combined_source_sha256(),
                }
            )
    registry = arm_registry(preflight.protocol)
    per_series = validate_metrics_against_mask(
        pd.DataFrame(rows), valid_mask, registry.arm_ids
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "protocol_id": preflight.protocol.protocol_id,
        "protocol_sha256": preflight.protocol.payload_sha256,
        "config_sha256": preflight.protocol.config_sha256,
        "manifest_sha256": preflight.protocol.manifest_sha256,
        "score_index_sha256": preflight.score_index_sha256,
        "combined_source_sha256": combined_source_sha256(),
        "valid_mask_sha256": valid_mask_sha256(valid_mask),
        "expected_series": preflight.protocol.expected_series,
        "valid_series": preflight.protocol.expected_valid_series,
        "excluded_series": preflight.protocol.expected_series - preflight.protocol.expected_valid_series,
        "stage_count": len(preflight.protocol.stages),
        "arm_count": len(preflight.protocol.arms),
        "score_record_count": len(per_series),
        "label_load_count": len(preflight.records),
        "bootstrap_seed": preflight.protocol.bootstrap_seed,
        "bootstrap_replicates": preflight.protocol.bootstrap_replicates,
    }
    return CombinedEvaluationResult(
        per_series,
        valid_mask,
        _arm_metadata_frame(preflight.protocol),
        provenance,
    )


def arm_registry(protocol: CombinedProtocolSpec) -> ArmRegistry:
    arms = tuple(
        ArmSpec(arm.arm_id, arm.role, arm.order, arm.fp_threshold)
        for arm in protocol.arms
    )
    contrasts = tuple(
        ContrastSpec(row.contrast_id, row.family, row.candidate, row.control)
        for row in protocol.contrasts
    )
    primary = contrasts[0].candidate
    control = contrasts[0].control
    return ArmRegistry(
        protocol.protocol_id.upper(),
        primary,
        control,
        arms,
        contrasts,
        protocol.bootstrap_seed,
        protocol.bootstrap_replicates,
    )


def _stage_index(
    root: Path,
    preflight: CombinedPreflight,
    result: CombinedEvaluationResult | None,
) -> Mapping[str, Any]:
    stages: list[dict[str, Any]] = []
    for row in preflight.stages:
        stage_root = root / "stages" / row.stage.stage_id
        metrics_path = stage_root / "per_series_metrics.csv"
        metadata_path = stage_root / "arm_metadata.csv"
        marker_path = stage_root / "_EVALUATION_COMPLETE.json"
        ready = result is not None and row.status == "READY"
        stages.append(
            {
                "stage_id": row.stage.stage_id,
                "stage_group": row.stage.stage_group,
                "stage_kind": row.stage.stage_kind,
                "configuration_id": row.stage.configuration_id,
                "status": "COMPLETE" if ready else row.status,
                "reason": "stage metrics materialized" if ready else row.reason,
                "metrics_path": str(metrics_path.resolve()) if ready else None,
                "metrics_sha256": sha256_file(metrics_path) if ready else None,
                "marker_path": str(marker_path.resolve()) if ready else None,
                "marker_sha256": sha256_file(marker_path) if ready else None,
                "arm_metadata_path": str(metadata_path.resolve()) if ready else None,
                "arm_metadata_sha256": sha256_file(metadata_path) if ready else None,
                "expected_series": row.expected_series,
                "completed_series": row.completed_series,
                "expected_rows": row.expected_rows,
                "completed_rows": row.completed_rows,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": preflight.protocol.payload_sha256,
        "manifest_sha256": preflight.protocol.manifest_sha256,
        "stages": stages,
    }


def write_evaluation_outputs(
    output_root: Path,
    preflight: CombinedPreflight,
    result: CombinedEvaluationResult,
) -> tuple[Path, ...]:
    root = Path(output_root)
    metrics_path = root / "per_series_metrics.csv"
    mask_path = root / "valid_series_mask.csv"
    arm_path = root / "arm_metadata.csv"
    provenance_path = root / "evaluation_provenance.json"
    marker_path = root / "_COMBINED_EVALUATION_COMPLETE.json"
    _atomic_csv(metrics_path, result.per_series)
    _atomic_csv(mask_path, result.valid_mask)
    _atomic_csv(arm_path, result.arm_metadata)
    for stage in preflight.stages:
        stage_root = root / "stages" / stage.stage.stage_id
        stage_metrics = result.per_series.loc[
            result.per_series["stage_id"] == stage.stage.stage_id
        ].copy()
        stage_arms = result.arm_metadata.loc[
            result.arm_metadata["stage_id"] == stage.stage.stage_id
        ].copy()
        stage_metrics_path = stage_root / "per_series_metrics.csv"
        stage_arm_path = stage_root / "arm_metadata.csv"
        stage_marker_path = stage_root / "_EVALUATION_COMPLETE.json"
        _atomic_csv(stage_metrics_path, stage_metrics)
        _atomic_csv(stage_arm_path, stage_arms)
        _atomic_json(
            stage_marker_path,
            {
                "schema_version": SCHEMA_VERSION,
                "stage_id": stage.stage.stage_id,
                "expected_series": stage.expected_series,
                "arm_count": len(stage.stage.arms),
                "row_count": len(stage_metrics),
                "metrics_sha256": sha256_file(stage_metrics_path),
                "arm_metadata_sha256": sha256_file(stage_arm_path),
                "valid_mask_sha256": result.provenance["valid_mask_sha256"],
            },
        )
    provenance = dict(result.provenance)
    provenance.update(
        {
            "per_series_metrics_sha256": sha256_file(metrics_path),
            "valid_series_mask_file_sha256": sha256_file(mask_path),
            "arm_metadata_sha256": sha256_file(arm_path),
        }
    )
    _atomic_json(provenance_path, provenance)
    _atomic_json(
        marker_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "protocol_sha256": preflight.protocol.payload_sha256,
            "series_count": preflight.protocol.expected_series,
            "valid_series_count": preflight.protocol.expected_valid_series,
            "stage_count": len(preflight.stages),
            "arm_count": len(preflight.protocol.arms),
            "per_series_metrics_sha256": sha256_file(metrics_path),
            "valid_series_mask_file_sha256": sha256_file(mask_path),
            "arm_metadata_sha256": sha256_file(arm_path),
            "evaluation_provenance_sha256": sha256_file(provenance_path),
        },
    )
    index_path = root / "stage_evaluation_index.json"
    _atomic_json(index_path, _stage_index(root, preflight, result))
    return metrics_path, mask_path, arm_path, provenance_path, marker_path, index_path


def _arm_row(
    arm: str,
    source: str,
    role: str,
    order: int,
    group: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "arm": arm,
        "source_arm": source,
        "role": role,
        "order": order,
        "fp_threshold": 0.0,
        "experiment_group": group,
        "changed_factor": group,
        "fixed_factors": {"protocol": "VLM4TS_492", "metrics_mask": "COMMON_488"},
        "metadata": dict(parameters),
    }


def build_current_registry(config_path: Path) -> dict[str, Any]:
    """Freeze the current cache-only and completed CLS-control stages."""

    config_path = Path(config_path).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["paths"]["output_root"])
    manifest_path = Path(config["manifest"]["path"]).resolve(strict=True)
    registry_payload = _read_json(
        output_root / "manifests" / "cache_only_arm_registry.json", "cache registry"
    )
    plan_payload = _read_json(
        output_root / "manifests" / "cache_only_compute_plan.json", "cache plan"
    )
    planned = {str(row["id"]): row for row in plan_payload["arms"]}
    stages: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    for row in registry_payload["arms"]:
        item = planned[str(row["id"])]
        parameters = dict(item["parameters"])
        parameters["display_name"] = str(row["id"])
        parameters["is_final"] = str(row["id"]) == "FINAL_DEFAULT"
        arms.append(
            _arm_row(
                str(row["id"]),
                str(row["id"]),
                str(row["role"]),
                int(row["order"]),
                str(item["family"]),
                parameters,
            )
        )
    stages.append(
        {
            "stage_id": "CACHE_ONLY_B16_W240_S60",
            "stage_group": "cache_only_controls",
            "stage_kind": "cache_only",
            "configuration_id": "CACHE_B16_W240_S60",
            "root": str(Path(config["paths"]["run_root"]) / cache_evaluator.RUN_NAME),
            "status_path": None,
            "expected_series": int(config["manifest"]["expected_series"]),
            "arms": arms,
        }
    )
    variants = sorted(
        path.parent
        for path in (output_root / "encoder_stage").glob("*/encoder_controls_status.json")
    )
    if len(variants) != 1:
        raise ValueError("current registry builder requires exactly one encoder-control variant")
    control_root = variants[0]
    status = _read_json(control_root / "encoder_controls_status.json", "control status")
    order = len(arms)
    control_arms = []
    for source in status["expected_arms"]:
        metadata = {
            "representation": "line",
            "model_key": "B16",
            "window": 240,
            "stride": 60,
            "embedding": "true_global" if "CLS" in source else "patch_mean",
            "encoder_calls": 0,
            "ihp": "bypassed",
            "nctp": "bypassed",
            "display_name": source,
            "is_final": False,
        }
        control_arms.append(
            _arm_row(source, source, "inherited_control", order, "PATCH_EMBEDDING", metadata)
        )
        order += 1
    stages.append(
        {
            "stage_id": "ENCODER_CONTROLS_B16_W240_S60",
            "stage_group": "encoder_controls",
            "stage_kind": "encoder_controls",
            "configuration_id": "ENCODER_B16_W240_S60",
            "root": str(control_root),
            "status_path": str(control_root / "encoder_controls_status.json"),
            "expected_series": int(config["manifest"]["expected_series"]),
            "arms": control_arms,
        }
    )
    contrasts = list(registry_payload["contrasts"])
    for arm in control_arms:
        contrasts.append(
            {
                "id": f"{arm['arm']}_VS_LEGACY",
                "family": "PATCH_EMBEDDING",
                "candidate": arm["arm"],
                "control": "LEGACY_DEFAULT",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "VITTRACE_V3_COMBINED_492",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_series": int(config["manifest"]["expected_series"]),
        "expected_valid_series": EXPECTED_VALID_SERIES,
        "stages": stages,
        "contrasts": contrasts,
        "bootstrap": {
            "seed": 2027,
            "n_resamples": 10000,
            "shared_indices": True,
            "hierarchy": ["subgroup", "series"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-current")
    init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    init.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--registry", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--registry", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "init-current":
        payload = build_current_registry(args.config)
        _atomic_json(args.output, payload)
        load_combined_protocol(args.output)
        print(args.output)
        return 0
    protocol = load_combined_protocol(args.registry)
    preflight = preflight_combined(protocol)
    write_preflight(args.output_dir, preflight)
    if args.command == "preflight":
        print(json.dumps(_preflight_payload(preflight), indent=2))
        return 0 if preflight.status == "READY" else 2
    if preflight.status != "READY":
        # The BLOCKED marker was already written before this exit.
        print(preflight.reason)
        return 2
    result = evaluate_combined(preflight)
    paths = write_evaluation_outputs(args.output_dir, preflight, result)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CombinedBlocked",
    "CombinedEvaluationResult",
    "CombinedPreflight",
    "CombinedScoreRecord",
    "StagePreflight",
    "arm_registry",
    "build_current_registry",
    "combined_source_sha256",
    "evaluate_combined",
    "main",
    "preflight_combined",
    "write_evaluation_outputs",
    "write_preflight",
]
