"""Frozen cross-stage registry for delayed-label ViTTrace v3 evaluation.

The score-producing stages deliberately use different transaction schemas.  This
module provides one small, label-free registry that namespaces their arms and
binds every stage to the active 492-series protocol.  Adding a completed stage
is an append-only registry operation; it never changes an existing score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .registry import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED


SCHEMA_VERSION = 1
EXPECTED_VALID_SERIES = 488
STAGE_KINDS = ("cache_only", "encoder_controls", "dynamic_line", "spectrogram")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def _safe_id(value: Any, context: str) -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text):
        raise ValueError(f"{context} is not a safe identifier: {text!r}")
    return text


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a sequence")
    return value


@dataclass(frozen=True)
class CombinedArmSpec:
    arm_id: str
    source_arm: str
    role: str
    order: int
    fp_threshold: float
    experiment_group: str
    changed_factor: str
    fixed_factors: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CombinedStageSpec:
    stage_id: str
    stage_group: str
    stage_kind: str
    configuration_id: str
    root: Path
    status_path: Path | None
    expected_series: int
    arms: tuple[CombinedArmSpec, ...]


@dataclass(frozen=True)
class CombinedContrastSpec:
    contrast_id: str
    family: str
    candidate: str
    control: str


@dataclass(frozen=True)
class CombinedProtocolSpec:
    protocol_id: str
    config_path: Path
    config_sha256: str
    manifest_path: Path
    manifest_sha256: str
    expected_series: int
    expected_valid_series: int
    stages: tuple[CombinedStageSpec, ...]
    contrasts: tuple[CombinedContrastSpec, ...]
    bootstrap_seed: int
    bootstrap_replicates: int
    payload_sha256: str

    @property
    def arms(self) -> tuple[CombinedArmSpec, ...]:
        return tuple(arm for stage in self.stages for arm in stage.arms)

    @property
    def arm_ids(self) -> tuple[str, ...]:
        return tuple(arm.arm_id for arm in self.arms)

    @property
    def thresholds(self) -> dict[str, float]:
        return {arm.arm_id: arm.fp_threshold for arm in self.arms}


def validate_combined_protocol(payload: Mapping[str, Any]) -> CombinedProtocolSpec:
    root = _mapping(payload, "combined stage registry")
    required = {
        "schema_version",
        "protocol_id",
        "config_path",
        "config_sha256",
        "manifest_path",
        "manifest_sha256",
        "expected_series",
        "expected_valid_series",
        "stages",
        "contrasts",
        "bootstrap",
    }
    if set(root) != required:
        raise ValueError(f"combined registry keys must be exactly {sorted(required)}")
    if int(root["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("combined registry schema_version must equal one")
    protocol_id = _safe_id(root["protocol_id"], "protocol_id")
    config_path = Path(root["config_path"]).resolve(strict=True)
    manifest_path = Path(root["manifest_path"]).resolve(strict=True)
    config_sha = str(root["config_sha256"]).upper()
    manifest_sha = str(root["manifest_sha256"]).upper()
    if sha256_file(config_path) != config_sha or sha256_file(manifest_path) != manifest_sha:
        raise ValueError("combined registry config/manifest hash binding is stale")
    expected_series = int(root["expected_series"])
    expected_valid = int(root["expected_valid_series"])
    if expected_series <= 0 or not 0 < expected_valid <= expected_series:
        raise ValueError("combined registry series counts are invalid")

    stages: list[CombinedStageSpec] = []
    arms: list[CombinedArmSpec] = []
    stage_ids: set[str] = set()
    for stage_index, raw_stage in enumerate(_sequence(root["stages"], "stages")):
        row = _mapping(raw_stage, f"stage[{stage_index}]")
        required_stage = {
            "stage_id",
            "stage_group",
            "stage_kind",
            "configuration_id",
            "root",
            "status_path",
            "expected_series",
            "arms",
        }
        if set(row) != required_stage:
            raise ValueError(f"stage[{stage_index}] keys changed")
        stage_id = _safe_id(row["stage_id"], f"stage[{stage_index}].stage_id")
        if stage_id in stage_ids:
            raise ValueError("combined stage identifiers must be unique")
        stage_ids.add(stage_id)
        stage_kind = str(row["stage_kind"])
        if stage_kind not in STAGE_KINDS:
            raise ValueError(f"unsupported combined stage kind: {stage_kind}")
        stage_arms: list[CombinedArmSpec] = []
        for arm_index, raw_arm in enumerate(_sequence(row["arms"], "stage arms")):
            item = _mapping(raw_arm, f"stage[{stage_index}].arm[{arm_index}]")
            required_arm = {
                "arm",
                "source_arm",
                "role",
                "order",
                "fp_threshold",
                "experiment_group",
                "changed_factor",
                "fixed_factors",
                "metadata",
            }
            if set(item) != required_arm:
                raise ValueError("combined arm keys changed")
            threshold = float(item["fp_threshold"])
            if not math.isfinite(threshold):
                raise ValueError("combined arm threshold must be finite")
            arm = CombinedArmSpec(
                _safe_id(item["arm"], "combined arm"),
                _safe_id(item["source_arm"], "combined source arm"),
                str(item["role"] or "ablation"),
                int(item["order"]),
                threshold,
                str(item["experiment_group"] or row["stage_group"]),
                str(item["changed_factor"] or "NA"),
                dict(_mapping(item["fixed_factors"], "fixed_factors")),
                dict(_mapping(item["metadata"], "metadata")),
            )
            stage_arms.append(arm)
            arms.append(arm)
        if not stage_arms:
            raise ValueError("every combined stage requires at least one arm")
        status = None if row["status_path"] in (None, "") else Path(row["status_path"])
        stages.append(
            CombinedStageSpec(
                stage_id,
                str(row["stage_group"]),
                stage_kind,
                _safe_id(row["configuration_id"], "configuration_id"),
                Path(row["root"]),
                status,
                int(row["expected_series"]),
                tuple(stage_arms),
            )
        )
    arm_ids = tuple(arm.arm_id for arm in arms)
    if len(arm_ids) < 2 or len(set(arm_ids)) != len(arm_ids):
        raise ValueError("combined registry requires globally unique arms")
    if sorted(arm.order for arm in arms) != list(range(len(arms))):
        raise ValueError("combined arm order must be globally contiguous")
    if any(stage.expected_series != expected_series for stage in stages):
        raise ValueError("all combined stages must cover the same series count")

    contrasts: list[CombinedContrastSpec] = []
    for index, raw in enumerate(_sequence(root["contrasts"], "contrasts")):
        item = _mapping(raw, f"contrast[{index}]")
        if set(item) != {"id", "family", "candidate", "control"}:
            raise ValueError("combined contrast keys changed")
        contrast = CombinedContrastSpec(
            _safe_id(item["id"], "contrast id"),
            str(item["family"]),
            _safe_id(item["candidate"], "contrast candidate"),
            _safe_id(item["control"], "contrast control"),
        )
        if contrast.candidate not in arm_ids or contrast.control not in arm_ids:
            raise ValueError("combined contrast endpoint is not registered")
        if contrast.candidate == contrast.control:
            raise ValueError("combined contrast endpoints must differ")
        contrasts.append(contrast)
    if not contrasts or len({row.contrast_id for row in contrasts}) != len(contrasts):
        raise ValueError("combined contrasts must be unique and nonempty")
    bootstrap = _mapping(root["bootstrap"], "bootstrap")
    if dict(bootstrap) != {
        "seed": BOOTSTRAP_SEED,
        "n_resamples": BOOTSTRAP_REPLICATES,
        "shared_indices": True,
        "hierarchy": ["subgroup", "series"],
    }:
        raise ValueError("combined bootstrap protocol changed")
    return CombinedProtocolSpec(
        protocol_id,
        config_path,
        config_sha,
        manifest_path,
        manifest_sha,
        expected_series,
        expected_valid,
        tuple(stages),
        tuple(contrasts),
        BOOTSTRAP_SEED,
        BOOTSTRAP_REPLICATES,
        canonical_json_sha256(root),
    )


def load_combined_protocol(path: Path) -> CombinedProtocolSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return validate_combined_protocol(_mapping(payload, "combined registry JSON"))


__all__ = [
    "CombinedArmSpec",
    "CombinedContrastSpec",
    "CombinedProtocolSpec",
    "CombinedStageSpec",
    "EXPECTED_VALID_SERIES",
    "SCHEMA_VERSION",
    "STAGE_KINDS",
    "canonical_json_sha256",
    "load_combined_protocol",
    "sha256_file",
    "validate_combined_protocol",
]
