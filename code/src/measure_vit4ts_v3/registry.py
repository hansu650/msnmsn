"""Frozen arm-registry schema for corrected-primary v3 evaluation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 3
BOOTSTRAP_SEED = 2027
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_SUBGROUPS = 11
EXPECTED_FAMILIES = 3
_SAFE_ID = re.compile(r"^[A-Z0-9][A-Z0-9_.-]*$")
_VALIDITY_POLICY = {
    "f1_max": "both_classes",
    "auprc": "both_classes",
    "vus_pr": "both_classes",
    "anomaly_free_fp": "no_positive",
}
# Cache-only scores were frozen before the corrected-primary common-mask fix.
# Accept that exact artifact policy for provenance replay; all newly emitted
# registries and all metric outputs use _VALIDITY_POLICY above.
_LEGACY_ARTIFACT_VALIDITY_POLICY = {**_VALIDITY_POLICY, "f1_max": "all_series"}


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    role: str
    order: int
    fp_threshold: float


@dataclass(frozen=True)
class ContrastSpec:
    contrast_id: str
    family: str
    candidate: str
    control: str


@dataclass(frozen=True)
class ArmRegistry:
    registry_id: str
    primary_arm: str
    control_arm: str
    arms: tuple[ArmSpec, ...]
    contrasts: tuple[ContrastSpec, ...]
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    expected_subgroups: int = EXPECTED_SUBGROUPS
    expected_families: int = EXPECTED_FAMILIES

    @property
    def arm_ids(self) -> tuple[str, ...]:
        return tuple(arm.arm_id for arm in self.arms)

    @property
    def thresholds(self) -> dict[str, float]:
        return {arm.arm_id: arm.fp_threshold for arm in self.arms}

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "registry_id": self.registry_id,
            "primary_arm": self.primary_arm,
            "control_arm": self.control_arm,
            "arms": [
                {
                    "id": arm.arm_id,
                    "role": arm.role,
                    "order": arm.order,
                    "fp_threshold": arm.fp_threshold,
                }
                for arm in self.arms
            ],
            "contrasts": [
                {
                    "id": contrast.contrast_id,
                    "family": contrast.family,
                    "candidate": contrast.candidate,
                    "control": contrast.control,
                }
                for contrast in self.contrasts
            ],
            "validity_policy": dict(_VALIDITY_POLICY),
            "bootstrap": {
                "seed": self.bootstrap_seed,
                "n_resamples": self.bootstrap_replicates,
                "shared_indices": True,
                "hierarchy": ["subgroup", "series"],
            },
            "groups": {
                "expected_subgroups": self.expected_subgroups,
                "expected_families": self.expected_families,
            },
        }


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a sequence")
    return value


def _safe_id(value: Any, context: str) -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text):
        raise ValueError(f"{context} is not a safe frozen identifier: {text!r}")
    return text


def validate_arm_registry(payload: Mapping[str, Any]) -> ArmRegistry:
    """Validate and freeze the complete v3 arm/contrast/analysis registry."""

    root = _mapping(payload, "arm registry")
    required = {
        "schema_version",
        "registry_id",
        "primary_arm",
        "control_arm",
        "arms",
        "contrasts",
        "validity_policy",
        "bootstrap",
        "groups",
    }
    if set(root) != required:
        raise ValueError(f"arm-registry keys must be exactly {sorted(required)}")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("v3 arm-registry schema_version must equal 3")
    registry_id = _safe_id(root["registry_id"], "registry_id")

    arm_rows = _sequence(root["arms"], "arms")
    if len(arm_rows) < 2:
        raise ValueError("v3 registry requires at least two arms")
    arms: list[ArmSpec] = []
    for index, raw in enumerate(arm_rows):
        row = _mapping(raw, f"arm[{index}]")
        if set(row) != {"id", "role", "order", "fp_threshold"}:
            raise ValueError("each arm requires id/role/order/fp_threshold only")
        arm_id = _safe_id(row["id"], f"arm[{index}].id")
        role = str(row["role"] or "")
        order = int(row["order"])
        threshold = float(row["fp_threshold"])
        if not role or isinstance(row["order"], bool) or order != index:
            raise ValueError("arm roles must be nonempty and order must be contiguous")
        if not math.isfinite(threshold):
            raise ValueError("arm fp_threshold must be finite and prefixed")
        arms.append(ArmSpec(arm_id, role, order, threshold))
    arm_ids = tuple(arm.arm_id for arm in arms)
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("arm identifiers must be unique")

    primary = _safe_id(root["primary_arm"], "primary_arm")
    control = _safe_id(root["control_arm"], "control_arm")
    if primary not in arm_ids or control not in arm_ids or primary == control:
        raise ValueError("primary/control arms must be distinct registered arms")

    contrast_rows = _sequence(root["contrasts"], "contrasts")
    if not contrast_rows:
        raise ValueError("v3 registry requires at least one frozen contrast")
    contrasts: list[ContrastSpec] = []
    for index, raw in enumerate(contrast_rows):
        row = _mapping(raw, f"contrast[{index}]")
        if set(row) != {"id", "family", "candidate", "control"}:
            raise ValueError("each contrast requires id/family/candidate/control only")
        contrast_id = _safe_id(row["id"], f"contrast[{index}].id")
        family = str(row["family"] or "")
        candidate = _safe_id(row["candidate"], f"contrast[{index}].candidate")
        reference = _safe_id(row["control"], f"contrast[{index}].control")
        if not family or candidate not in arm_ids or reference not in arm_ids:
            raise ValueError("contrast family and registered endpoints are required")
        if candidate == reference:
            raise ValueError("contrast endpoints must differ")
        contrasts.append(ContrastSpec(contrast_id, family, candidate, reference))
    if len({item.contrast_id for item in contrasts}) != len(contrasts):
        raise ValueError("contrast identifiers must be unique")
    if not any(
        item.candidate == primary and item.control == control for item in contrasts
    ):
        raise ValueError("registry must contain the corrected primary contrast")

    artifact_policy = dict(_mapping(root["validity_policy"], "validity_policy"))
    if artifact_policy not in (
        _VALIDITY_POLICY, _LEGACY_ARTIFACT_VALIDITY_POLICY
    ):
        raise ValueError("v3 validity policy changed")
    bootstrap = _mapping(root["bootstrap"], "bootstrap")
    expected_bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "n_resamples": BOOTSTRAP_REPLICATES,
        "shared_indices": True,
        "hierarchy": ["subgroup", "series"],
    }
    if dict(bootstrap) != expected_bootstrap:
        raise ValueError("v3 bootstrap must use shared 10000-draw indices and seed 2027")
    groups = _mapping(root["groups"], "groups")
    if dict(groups) != {
        "expected_subgroups": EXPECTED_SUBGROUPS,
        "expected_families": EXPECTED_FAMILIES,
    }:
        raise ValueError("v3 aggregation scope must remain 11 subgroups / 3 families")
    return ArmRegistry(
        registry_id,
        primary,
        control,
        tuple(arms),
        tuple(contrasts),
    )


def load_arm_registry(path: Path) -> ArmRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return validate_arm_registry(_mapping(payload, "arm-registry JSON"))
