"""Frozen cache-only B/16, W=240, stride=60 ablation registry.

The registry separates *logical* paper rows from unique computations.  Arms
with identical scientific parameters become explicit aliases, so a runner
never recomputes or duplicates their score arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .registry import ArmRegistry, ArmSpec, ContrastSpec, validate_arm_registry


PLAN_SCHEMA_VERSION = 1
REGISTRY_ID = "VITTRACE_V3_CACHE_ONLY_B16_W240_S60"
PRIMARY_ARM = "FINAL_DEFAULT"
CONTROL_ARM = "LEGACY_DEFAULT"

MATCHING_SCOPES = ("position", "row", "column", "global")
SCALE_SUBSETS = ("P", "M", "L", "PM", "PL", "ML", "PML")
MEMORY_MODES = ("median_reference", "all_pairs")


@dataclass(frozen=True)
class CacheOnlyArm:
    """One logical arm and its complete label-free computation parameters."""

    arm_id: str
    family: str
    role: str
    matching_scope: str
    memory: str
    scales: tuple[str, ...]
    incidence: str
    fusion: str
    temporal: str
    reducer_kind: str | None = None
    reducer_value: float | None = None

    def parameters(self) -> dict[str, Any]:
        return {
            "model_name": "ViT-B-16",
            "patch_grid": [14, 14],
            "window": 240,
            "stride": 60,
            "matching_scope": self.matching_scope,
            "memory": self.memory,
            "scales": list(self.scales),
            "incidence": self.incidence,
            "fusion": self.fusion,
            "temporal": self.temporal,
            "reducer_kind": self.reducer_kind,
            "reducer_value": self.reducer_value,
        }


@dataclass(frozen=True)
class PlannedArm:
    logical: CacheOnlyArm
    canonical_arm: str
    parameter_sha256: str

    @property
    def is_alias(self) -> bool:
        return self.logical.arm_id != self.canonical_arm


@dataclass(frozen=True)
class CacheOnlyPlan:
    arms: tuple[PlannedArm, ...]

    @property
    def logical_arm_ids(self) -> tuple[str, ...]:
        return tuple(item.logical.arm_id for item in self.arms)

    @property
    def canonical_arms(self) -> tuple[PlannedArm, ...]:
        return tuple(item for item in self.arms if not item.is_alias)

    def by_id(self) -> dict[str, PlannedArm]:
        return {item.logical.arm_id: item for item in self.arms}

    def to_payload(
        self,
        *,
        config_path: Path,
        config_sha256: str,
        manifest_sha256: str,
    ) -> dict[str, Any]:
        aliases: dict[str, list[str]] = {
            item.logical.arm_id: [] for item in self.canonical_arms
        }
        for item in self.arms:
            aliases[item.canonical_arm].append(item.logical.arm_id)
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "registry_id": REGISTRY_ID,
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": str(config_sha256).upper(),
            "manifest_sha256": str(manifest_sha256).upper(),
            "encoder_calls": 0,
            "logical_arm_count": len(self.arms),
            "unique_computation_count": len(self.canonical_arms),
            "arms": [
                {
                    "id": item.logical.arm_id,
                    "family": item.logical.family,
                    "role": item.logical.role,
                    "canonical_arm": item.canonical_arm,
                    "is_alias": item.is_alias,
                    "parameter_sha256": item.parameter_sha256,
                    "parameters": item.logical.parameters(),
                }
                for item in self.arms
            ],
            "unique_computations": [
                {
                    "canonical_arm": item.logical.arm_id,
                    "parameter_sha256": item.parameter_sha256,
                    "aliases": aliases[item.logical.arm_id],
                    "parameters": item.logical.parameters(),
                }
                for item in self.canonical_arms
            ],
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def parameter_sha256(parameters: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(parameters), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _arm(
    arm_id: str,
    family: str,
    *,
    matching_scope: str = "global",
    memory: str = "median_reference",
    scales: str = "PML",
    incidence: str = "literal",
    fusion: str = "active_valid",
    temporal: str = "nctp_linear",
    reducer_kind: str | None = None,
    reducer_value: float | None = None,
    role: str = "ablation",
) -> CacheOnlyArm:
    result = CacheOnlyArm(
        arm_id=arm_id,
        family=family,
        role=role,
        matching_scope=matching_scope,
        memory=memory,
        scales=tuple(scales),
        incidence=incidence,
        fusion=fusion,
        temporal=temporal,
        reducer_kind=reducer_kind,
        reducer_value=reducer_value,
    )
    _validate_arm(result)
    return result


def _legacy(arm_id: str, family: str, **changes: Any) -> CacheOnlyArm:
    values: dict[str, Any] = {
        "matching_scope": "global",
        "memory": "median_reference",
        "scales": "PML",
        "incidence": "released",
        "fusion": "legacy_intersection",
        "temporal": "legacy",
        "reducer_kind": "top_fraction",
        "reducer_value": 0.25,
    }
    values.update(changes)
    return _arm(arm_id, family, **values)


def _final(arm_id: str, family: str, **changes: Any) -> CacheOnlyArm:
    values: dict[str, Any] = {
        "matching_scope": "global",
        "memory": "median_reference",
        "scales": "PML",
        "incidence": "literal",
        "fusion": "active_valid",
        "temporal": "nctp_linear",
    }
    values.update(changes)
    return _arm(arm_id, family, **values)


def _validate_arm(arm: CacheOnlyArm) -> None:
    if arm.matching_scope not in MATCHING_SCOPES:
        raise ValueError(f"unsupported matching scope: {arm.matching_scope}")
    if arm.memory not in MEMORY_MODES:
        raise ValueError(f"unsupported memory: {arm.memory}")
    scale_key = "".join(arm.scales)
    if scale_key not in SCALE_SUBSETS or len(set(arm.scales)) != len(arm.scales):
        raise ValueError(f"unsupported active scale subset: {scale_key}")
    if arm.incidence not in {"released", "literal"}:
        raise ValueError("incidence must be released or literal")
    if arm.fusion not in {"legacy_intersection", "active_valid"}:
        raise ValueError("unsupported scale fusion")
    if arm.temporal not in {
        "legacy",
        "nctp_linear",
        "nctp_nearest",
        "trace_soft",
        "trace_hard",
    }:
        raise ValueError("unsupported temporal mode")
    if arm.temporal == "legacy":
        if arm.reducer_kind not in {"quantile", "top_fraction"}:
            raise ValueError("legacy temporal mode requires an explicit reducer")
        if arm.reducer_value is None or not 0.0 < float(arm.reducer_value) <= 1.0:
            raise ValueError("legacy reducer value must lie in (0,1]")
    elif arm.reducer_kind is not None or arm.reducer_value is not None:
        raise ValueError("non-legacy temporal modes cannot carry a row reducer")


def _validate_config_grid(config: Mapping[str, Any]) -> None:
    defaults = config["defaults"]
    grid = config["grid"]
    if (
        str(defaults["model_name"]) != "ViT-B-16"
        or int(defaults["patch_size"]) != 16
        or int(defaults["window"]) != 240
        or int(defaults["stride"]) != 60
    ):
        raise ValueError("cache-only registry is frozen to B/16, W=240, stride=60")
    if tuple(grid["matching_scopes"]) != MATCHING_SCOPES:
        raise ValueError("matching-scope grid differs from the mandatory registry")
    if tuple(grid["scale_subsets"]) != SCALE_SUBSETS:
        raise ValueError("P/M/L scale-subset grid differs from the mandatory registry")
    if tuple(grid["memories"]) != MEMORY_MODES:
        raise ValueError("memory grid differs from median-reference/all-pairs")
    expected = (0.10, 0.25, 0.50)
    if tuple(float(v) for v in grid["reducer_quantiles"]) != expected:
        raise ValueError("quantile grid must be Q10/Q25/Q50")
    if tuple(float(v) for v in grid["reducer_top_fractions"]) != expected:
        raise ValueError("top-fraction grid must be 10/25/50 percent")


def build_cache_only_plan(config: Mapping[str, Any]) -> CacheOnlyPlan:
    """Build all mandatory logical arms and deduplicate exact configurations."""

    _validate_config_grid(config)
    logical: list[CacheOnlyArm] = [
        _legacy("LEGACY_DEFAULT", "DEFAULT", role="control"),
        _final("FINAL_DEFAULT", "DEFAULT", role="method"),
    ]
    for scope in MATCHING_SCOPES:
        logical.append(
            _legacy(
                f"MATCH_LEGACY_{scope.upper()}",
                "MATCHING_LEGACY",
                matching_scope=scope,
            )
        )
    for scope in MATCHING_SCOPES:
        logical.append(
            _final(
                f"MATCH_FINAL_{scope.upper()}",
                "MATCHING_FINAL",
                matching_scope=scope,
            )
        )
    for scales in SCALE_SUBSETS:
        logical.append(_final(f"SCALE_{scales}", "SCALE_SUBSET", scales=scales))
    # Mandatory scale deletion controls on the released legacy-map path.
    # PML is an explicit paper row even though it aliases LEGACY_DEFAULT.
    logical.extend(
        [
            _legacy("SCALE_LEGACY_P", "SCALE_SUBSET_LEGACY", scales="P"),
            _legacy("SCALE_LEGACY_PML", "SCALE_SUBSET_LEGACY", scales="PML"),
        ]
    )
    logical.extend(
        [
            _final("MEMORY_MEDIAN", "MEMORY", memory="median_reference"),
            _final("MEMORY_ALLPAIRS", "MEMORY", memory="all_pairs"),
        ]
    )
    for value, suffix in ((0.10, "10"), (0.25, "25"), (0.50, "50")):
        logical.append(
            _legacy(
                f"REDUCER_Q{suffix}",
                "REDUCER_QUANTILE",
                reducer_kind="quantile",
                reducer_value=value,
            )
        )
    for value, suffix in ((0.10, "10"), (0.25, "25"), (0.50, "50")):
        logical.append(
            _legacy(
                f"REDUCER_TOP{suffix}",
                "REDUCER_TOP_FRACTION",
                reducer_kind="top_fraction",
                reducer_value=value,
            )
        )

    # Exact IHP (incidence) x NCTP (temporal) component interaction.
    logical.extend(
        [
            _legacy("IHP0_NCTP0", "IHP_X_NCTP"),
            _final(
                "IHP1_NCTP0",
                "IHP_X_NCTP",
                temporal="legacy",
                reducer_kind="top_fraction",
                reducer_value=0.25,
            ),
            _legacy(
                "IHP0_NCTP1",
                "IHP_X_NCTP",
                temporal="nctp_linear",
                reducer_kind=None,
                reducer_value=None,
            ),
            _final("IHP1_NCTP1", "IHP_X_NCTP"),
        ]
    )
    logical.extend(
        [
            _final("TEMP_NCTP_LINEAR", "TEMPORAL"),
            _final("TEMP_NCTP_NEAREST", "TEMPORAL", temporal="nctp_nearest"),
            _final("TEMP_TRACE_SOFT", "TEMPORAL", temporal="trace_soft"),
            _final("TEMP_TRACE_HARD", "TEMPORAL", temporal="trace_hard"),
            _final(
                "TEMP_LEGACY",
                "TEMPORAL",
                temporal="legacy",
                reducer_kind="top_fraction",
                reducer_value=0.25,
            ),
        ]
    )

    seen: dict[str, str] = {}
    planned: list[PlannedArm] = []
    for arm in logical:
        digest = parameter_sha256(arm.parameters())
        canonical = seen.setdefault(digest, arm.arm_id)
        planned.append(PlannedArm(arm, canonical, digest))
    plan = CacheOnlyPlan(tuple(planned))
    if len(plan.logical_arm_ids) != 36 or len(plan.canonical_arms) != 26:
        raise RuntimeError("mandatory cache-only registry cardinality changed")
    if len(set(plan.logical_arm_ids)) != len(plan.logical_arm_ids):
        raise RuntimeError("logical cache-only arm identifiers are not unique")
    return plan


def build_evaluation_registry(plan: CacheOnlyPlan) -> ArmRegistry:
    arms = tuple(
        ArmSpec(item.logical.arm_id, item.logical.role, index, 0.0)
        for index, item in enumerate(plan.arms)
    )
    contrasts: list[ContrastSpec] = [
        ContrastSpec("FINAL_VS_LEGACY", "PRIMARY", PRIMARY_ARM, CONTROL_ARM)
    ]
    for item in plan.arms:
        arm_id = item.logical.arm_id
        if arm_id in {PRIMARY_ARM, CONTROL_ARM}:
            continue
        contrasts.append(
            ContrastSpec(
                f"{arm_id}_VS_FINAL",
                item.logical.family,
                arm_id,
                PRIMARY_ARM,
            )
        )
    registry = ArmRegistry(
        REGISTRY_ID,
        PRIMARY_ARM,
        CONTROL_ARM,
        arms,
        tuple(contrasts),
    )
    # Round-trip through the public strict validator before writing it.
    return validate_arm_registry(registry.to_payload())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def freeze_cache_only_registry(
    config_path: Path,
    output_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Write the evaluation registry and deduplicated computation plan."""

    raw = Path(config_path).read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, Mapping) or config.get("stage") != "vittrace_ablation_full_v3":
        raise ValueError("cache-only registry requires the isolated v3 config")
    manifest_path = Path(config["manifest"]["path"])
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != str(config["manifest"]["sha256"]).upper():
        raise ValueError("frozen 11-group manifest SHA256 changed")
    config_sha = _sha256_bytes(raw)
    plan = build_cache_only_plan(config)
    registry = build_evaluation_registry(plan)
    root = (
        Path(output_directory)
        if output_directory is not None
        else Path(config["paths"]["output_root"]) / "manifests"
    )
    registry_path = root / "cache_only_arm_registry.json"
    plan_path = root / "cache_only_compute_plan.json"
    _atomic_json(registry_path, registry.to_payload())
    _atomic_json(
        plan_path,
        plan.to_payload(
            config_path=Path(config_path),
            config_sha256=config_sha,
            manifest_sha256=manifest_sha,
        ),
    )
    return registry_path, plan_path


def load_compute_plan(path: Path) -> tuple[CacheOnlyPlan, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported cache-only compute-plan schema")
    if payload.get("registry_id") != REGISTRY_ID or payload.get("encoder_calls") != 0:
        raise ValueError("cache-only compute-plan identity changed")
    rows = payload.get("arms")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("compute plan arms must be a sequence")
    planned: list[PlannedArm] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("compute-plan arm must be a mapping")
        parameters = row.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("compute-plan parameters must be a mapping")
        arm = CacheOnlyArm(
            arm_id=str(row["id"]),
            family=str(row["family"]),
            role=str(row["role"]),
            matching_scope=str(parameters["matching_scope"]),
            memory=str(parameters["memory"]),
            scales=tuple(str(v) for v in parameters["scales"]),
            incidence=str(parameters["incidence"]),
            fusion=str(parameters["fusion"]),
            temporal=str(parameters["temporal"]),
            reducer_kind=(
                None
                if parameters.get("reducer_kind") is None
                else str(parameters["reducer_kind"])
            ),
            reducer_value=(
                None
                if parameters.get("reducer_value") is None
                else float(parameters["reducer_value"])
            ),
        )
        _validate_arm(arm)
        digest = parameter_sha256(arm.parameters())
        if digest != str(row["parameter_sha256"]).upper():
            raise ValueError("compute-plan parameter hash mismatch")
        planned.append(PlannedArm(arm, str(row["canonical_arm"]), digest))
    plan = CacheOnlyPlan(tuple(planned))
    expected = build_cache_only_plan(
        {
            "defaults": {
                "model_name": "ViT-B-16",
                "patch_size": 16,
                "window": 240,
                "stride": 60,
            },
            "grid": {
                "matching_scopes": list(MATCHING_SCOPES),
                "scale_subsets": list(SCALE_SUBSETS),
                "memories": list(MEMORY_MODES),
                "reducer_quantiles": [0.10, 0.25, 0.50],
                "reducer_top_fractions": [0.10, 0.25, 0.50],
            },
        }
    )
    if plan != expected:
        raise ValueError("compute plan differs from the mandatory frozen registry")
    if int(payload.get("logical_arm_count", -1)) != len(plan.arms) or int(
        payload.get("unique_computation_count", -1)
    ) != len(plan.canonical_arms):
        raise ValueError("compute-plan cardinalities are stale")
    return plan, payload


__all__ = [
    "CacheOnlyArm",
    "CacheOnlyPlan",
    "PlannedArm",
    "REGISTRY_ID",
    "build_cache_only_plan",
    "build_evaluation_registry",
    "freeze_cache_only_registry",
    "load_compute_plan",
    "parameter_sha256",
    "sha256_file",
]
