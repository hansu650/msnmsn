"""Corrected-primary aggregation CLI and publication-table schemas for v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .aggregate import (
    AggregationBundle,
    aggregate_metrics,
    paired_hierarchical_bootstrap,
)
from .cache_registry import CacheOnlyPlan, load_compute_plan, sha256_file
from .evaluator import EVALUATION_DIRECTORY, RUN_NAME
from .metrics import ALL_METRICS, DETECTION_METRICS, valid_mask_sha256
from .outputs import bootstrap_plot_frame, subgroup_delta_plot_frame
from .registry import ArmRegistry, load_arm_registry
from .structural_audit import structural_audit_frame


REPORTING_SCHEMA_VERSION = 1
FACTORIAL_ARMS = (
    "IHP0_NCTP0",
    "IHP1_NCTP0",
    "IHP0_NCTP1",
    "IHP1_NCTP1",
)
FACTOR_LEVELS = {
    "IHP0_NCTP0": (0, 0),
    "IHP1_NCTP0": (1, 0),
    "IHP0_NCTP1": (0, 1),
    "IHP1_NCTP1": (1, 1),
}
TABLE3_FAMILIES = ("NAB", "NASA", "Yahoo")


def arm_registry_frame(registry: ArmRegistry, plan: CacheOnlyPlan) -> pd.DataFrame:
    """Join strict evaluation metadata to the deduplicated computation plan."""

    if registry.arm_ids != plan.logical_arm_ids:
        raise ValueError("arm registry and compute plan disagree")
    specs = {item.arm_id: item for item in registry.arms}
    rows: list[dict[str, Any]] = []
    for item in plan.arms:
        arm = item.logical
        spec = specs[arm.arm_id]
        rows.append(
            {
                "registry_id": registry.registry_id,
                "arm": arm.arm_id,
                "arm_order": spec.order,
                "arm_role": spec.role,
                "fp_threshold": spec.fp_threshold,
                "ablation_family": arm.family,
                "canonical_arm": item.canonical_arm,
                "is_alias": item.is_alias,
                "parameter_sha256": item.parameter_sha256,
                "matching_scope": arm.matching_scope,
                "memory": arm.memory,
                "scales": "+".join(arm.scales),
                "incidence": arm.incidence,
                "fusion": arm.fusion,
                "temporal": arm.temporal,
                "reducer_kind": arm.reducer_kind,
                "reducer_value": arm.reducer_value,
            }
        )
    frame = pd.DataFrame(rows).sort_values("arm_order").reset_index(drop=True)
    if frame["arm"].duplicated().any() or tuple(frame["arm"]) != registry.arm_ids:
        raise RuntimeError("arm-registry join is not one-to-one and ordered")
    return frame


def join_arm_registry(frame: pd.DataFrame, arm_table: pd.DataFrame) -> pd.DataFrame:
    """Attach frozen arm metadata while rejecting conflicting existing fields."""

    if "arm" not in frame or "arm" not in arm_table:
        raise ValueError("arm-registry joins require an arm column")
    if arm_table.empty or arm_table["arm"].duplicated().any():
        raise ValueError("arm registry table must be unique and nonempty")
    unknown = set(frame["arm"].astype(str)) - set(arm_table["arm"].astype(str))
    if unknown:
        raise ValueError(f"table contains unregistered arms: {sorted(unknown)}")
    metadata = [column for column in arm_table.columns if column != "arm"]
    overlapping = [column for column in metadata if column in frame.columns]
    joined = frame.merge(
        arm_table,
        on="arm",
        how="left",
        validate="many_to_one",
        suffixes=("", "_registry"),
    )
    for column in overlapping:
        expected = joined[f"{column}_registry"]
        actual = joined[column]
        equal = (actual.isna() & expected.isna()) | (
            actual.astype(str) == expected.astype(str)
        )
        if not bool(equal.all()):
            raise ValueError(f"existing {column} conflicts with the frozen arm registry")
        joined = joined.drop(columns=f"{column}_registry")
    return joined


def table3_style_frame(
    bundle: AggregationBundle,
    arm_table: pd.DataFrame,
    *,
    metrics: Sequence[str] = DETECTION_METRICS,
) -> pd.DataFrame:
    """Wide NAB/NASA/Yahoo/equal-11 table following VLM4TS Table 3 layout."""

    requested = tuple(str(metric) for metric in metrics)
    if not requested or any(metric not in ALL_METRICS for metric in requested):
        raise ValueError("Table3 metrics differ from the v3 metric registry")
    families = set(bundle.family3["family"].astype(str))
    if families != set(TABLE3_FAMILIES):
        raise ValueError("Table3 schema requires exactly NAB, NASA, and Yahoo")
    family_values = bundle.family3.loc[
        bundle.family3["metric"].astype(str).isin(requested)
    ].pivot(index=["arm", "metric"], columns="family", values="value")
    equal = bundle.equal11.loc[
        bundle.equal11["metric"].astype(str).isin(requested),
        ["arm", "metric", "value"],
    ].rename(columns={"value": "equal11"})
    frame = family_values.reset_index().merge(
        equal, on=["arm", "metric"], how="inner", validate="one_to_one"
    )
    frame = join_arm_registry(frame, arm_table)
    metric_order = {metric: index for index, metric in enumerate(requested)}
    frame["_metric_order"] = frame["metric"].map(metric_order)
    first = [
        "registry_id",
        "arm",
        "arm_order",
        "arm_role",
        "ablation_family",
        "canonical_arm",
        "is_alias",
        "metric",
        *TABLE3_FAMILIES,
        "equal11",
    ]
    remainder = [
        column for column in frame.columns if column not in first and column != "_metric_order"
    ]
    return (
        frame.sort_values(["_metric_order", "arm_order"])
        .loc[:, first + remainder]
        .reset_index(drop=True)
    )


def long11_frame(
    bundle: AggregationBundle, arm_table: pd.DataFrame
) -> pd.DataFrame:
    """Complete long-form 11-subgroup schema for every registered metric."""

    frame = join_arm_registry(bundle.subgroup11.copy(), arm_table)
    metric_order = {metric: index for index, metric in enumerate(ALL_METRICS)}
    frame["metric_order"] = frame["metric"].map(metric_order)
    return frame.sort_values(
        ["metric_order", "arm_order", "family", "subgroup"]
    ).reset_index(drop=True)


def factorial_2x2_frame(
    bundle: AggregationBundle, arm_table: pd.DataFrame
) -> pd.DataFrame:
    """Return the frozen IHP x NCTP cells in Table3-compatible form."""

    if not set(FACTORIAL_ARMS).issubset(set(arm_table["arm"].astype(str))):
        raise ValueError("IHP x NCTP 2x2 arms are incomplete")
    table = table3_style_frame(bundle, arm_table)
    frame = table.loc[table["arm"].isin(FACTORIAL_ARMS)].copy()
    frame["ihp"] = frame["arm"].map(lambda arm: FACTOR_LEVELS[str(arm)][0])
    frame["nctp"] = frame["arm"].map(lambda arm: FACTOR_LEVELS[str(arm)][1])
    frame["cell_order"] = frame["arm"].map(
        {arm: index for index, arm in enumerate(FACTORIAL_ARMS)}
    )
    metric_order = {metric: index for index, metric in enumerate(DETECTION_METRICS)}
    frame["metric_order"] = frame["metric"].map(metric_order)
    return frame.sort_values(["metric_order", "cell_order"]).reset_index(drop=True)


def plot_tidy_frame(
    bundle: AggregationBundle, arm_table: pd.DataFrame
) -> pd.DataFrame:
    """One stable plotting schema spanning subgroup/family/overall views."""

    sources = (
        ("subgroup11", bundle.subgroup11),
        ("family3", bundle.family3),
        ("equal11", bundle.equal11),
        ("fileweighted", bundle.fileweighted),
    )
    frames = []
    for view, source in sources:
        current = source.copy()
        current["view"] = view
        frames.append(current)
    frame = join_arm_registry(pd.concat(frames, ignore_index=True), arm_table)
    frame["valid_fraction"] = np.where(
        frame["n_total"].to_numpy(dtype=np.float64) > 0,
        frame["n_valid"].to_numpy(dtype=np.float64)
        / frame["n_total"].to_numpy(dtype=np.float64),
        np.nan,
    )
    frame["view_order"] = frame["view"].map(
        {name: index for index, (name, _) in enumerate(sources)}
    )
    frame["metric_order"] = frame["metric"].map(
        {metric: index for index, metric in enumerate(ALL_METRICS)}
    )
    return frame.sort_values(
        ["view_order", "metric_order", "arm_order", "family", "subgroup"]
    ).reset_index(drop=True)


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


def _load_evaluation(
    root: Path,
    registry_identity: str,
    plan_identity: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any], str]:
    root = Path(root)
    paths = {
        "metrics": root / "per_series_metrics.csv",
        "mask": root / "valid_series_mask.csv",
        "provenance": root / "evaluation_provenance.json",
        "marker": root / "_EVALUATION_COMPLETE.json",
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("complete v3 evaluation artifacts are required")
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("v3 evaluation marker/provenance must be mappings")
    if (
        sha256_file(paths["metrics"]).upper()
        != str(marker.get("per_series_metrics_sha256", "")).upper()
        or sha256_file(paths["mask"]).upper()
        != str(marker.get("valid_series_mask_file_sha256", "")).upper()
        or sha256_file(paths["provenance"]).upper()
        != str(marker.get("evaluation_provenance_sha256", "")).upper()
        or str(provenance.get("registry_sha256", "")).upper()
        != registry_identity.upper()
        or str(provenance.get("compute_plan_sha256", "")).upper()
        != plan_identity.upper()
    ):
        raise ValueError("v3 evaluation artifacts are stale or provenance-mismatched")
    metrics = pd.read_csv(paths["metrics"])
    mask = pd.read_csv(paths["mask"])
    if valid_mask_sha256(mask) != str(provenance.get("valid_mask_sha256", "")).upper():
        raise ValueError("v3 valid-series mask content hash changed")
    return metrics, mask, provenance, sha256_file(paths["marker"]).upper()


def write_reporting_outputs(
    output_root: Path,
    bundle: AggregationBundle,
    registry: ArmRegistry,
    plan: CacheOnlyPlan,
    *,
    provenance: Mapping[str, Any],
    bootstraps: pd.DataFrame | None = None,
) -> tuple[Path, ...]:
    """Write all numeric/table/plot schemas, with a completion marker last."""

    root = Path(output_root)
    arm_table = arm_registry_frame(registry, plan)
    tables: dict[str, pd.DataFrame] = {
        "per_series.csv": join_arm_registry(bundle.per_series, arm_table),
        "subgroup11.csv": join_arm_registry(bundle.subgroup11, arm_table),
        "family3.csv": join_arm_registry(bundle.family3, arm_table),
        "equal11.csv": join_arm_registry(bundle.equal11, arm_table),
        "fileweighted.csv": join_arm_registry(bundle.fileweighted, arm_table),
        "arm_registry.csv": arm_table,
        "table3_style.csv": table3_style_frame(bundle, arm_table),
        "long11.csv": long11_frame(bundle, arm_table),
        "ihp_nctp_2x2.csv": factorial_2x2_frame(bundle, arm_table),
        "plot_tidy.csv": plot_tidy_frame(bundle, arm_table),
        "structural_audit.csv": structural_audit_frame(),
    }
    for metric in ALL_METRICS:
        tables[f"plot_subgroup_delta_{metric}.csv"] = subgroup_delta_plot_frame(
            bundle.subgroup11, registry, metric
        )
    if bootstraps is not None:
        tables["hierarchical_bootstrap.csv"] = bootstraps
        for metric in sorted(set(bootstraps["metric"].astype(str))):
            tables[f"plot_bootstrap_{metric}.csv"] = bootstrap_plot_frame(
                bootstraps, metric
            )
    outputs: list[Path] = []
    output_hashes: dict[str, str] = {}
    for name, frame in tables.items():
        target = root / name
        _atomic_csv(target, frame)
        outputs.append(target)
        output_hashes[name] = sha256_file(target).upper()
    registry_json = root / "arm_registry.json"
    _atomic_json(registry_json, registry.to_payload())
    outputs.append(registry_json)
    output_hashes[registry_json.name] = sha256_file(registry_json).upper()
    report_provenance = root / "reporting_provenance.json"
    payload = {
        **dict(provenance),
        "schema_version": REPORTING_SCHEMA_VERSION,
        "registry_id": registry.registry_id,
        "logical_arm_count": len(plan.arms),
        "unique_computation_count": len(plan.canonical_arms),
        "bootstrap_executed": bootstraps is not None,
        "output_sha256": output_hashes,
    }
    _atomic_json(report_provenance, payload)
    outputs.append(report_provenance)
    marker = root / "_REPORTING_COMPLETE.json"
    _atomic_json(
        marker,
        {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "registry_id": registry.registry_id,
            "logical_arm_count": len(plan.arms),
            "bootstrap_executed": bootstraps is not None,
            "reporting_provenance_sha256": sha256_file(report_provenance).upper(),
            "structural_audit_sha256": output_hashes["structural_audit.csv"],
            "output_count": len(output_hashes),
        },
    )
    outputs.append(marker)
    return tuple(outputs)


def _preserve_reporting_failure(
    config: Mapping[str, Any] | None, config_path: Path, error: BaseException
) -> Path | None:
    if config is None:
        return None
    root = Path(config["paths"]["failure_root"]) / RUN_NAME / "reporting"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"reporting_failure_{stamp}_{os.getpid()}.json"
    _atomic_json(
        path,
        {
            "schema_version": REPORTING_SCHEMA_VERSION,
            "config_path": str(Path(config_path).resolve()),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )
    return path


def aggregate_evaluation(
    config_path: Path,
    registry_path: Path | None = None,
    plan_path: Path | None = None,
    evaluation_root: Path | None = None,
    output_root: Path | None = None,
    *,
    with_bootstrap: bool = False,
    bootstrap_replicates: int | None = None,
) -> tuple[Path, ...]:
    """Aggregate a completed evaluator commit; bootstrap is explicit opt-in."""

    config: Mapping[str, Any] | None = None
    try:
        config_path = Path(config_path).resolve(strict=True)
        raw = config_path.read_bytes()
        config = yaml.safe_load(raw)
        if not isinstance(config, Mapping) or config.get("stage") != "vittrace_ablation_full_v3":
            raise ValueError("reporting accepts only the isolated v3 config")
        defaults = Path(config["paths"]["output_root"]) / "manifests"
        registry_path = Path(
            registry_path or defaults / "cache_only_arm_registry.json"
        ).resolve(strict=True)
        plan_path = Path(
            plan_path or defaults / "cache_only_compute_plan.json"
        ).resolve(strict=True)
        registry_sha = sha256_file(registry_path).upper()
        plan_sha = sha256_file(plan_path).upper()
        registry = load_arm_registry(registry_path)
        plan, plan_payload = load_compute_plan(plan_path)
        config_sha = hashlib.sha256(raw).hexdigest().upper()
        if (
            registry.arm_ids != plan.logical_arm_ids
            or str(plan_payload.get("config_sha256", "")).upper() != config_sha
        ):
            raise ValueError("v3 registry/plan differs from the active config")
        evaluation_root = Path(
            evaluation_root
            or Path(config["paths"]["result_root"]) / EVALUATION_DIRECTORY
        )
        metrics, mask, evaluation_provenance, marker_sha = _load_evaluation(
            evaluation_root, registry_sha, plan_sha
        )
        bundle = aggregate_metrics(metrics, mask, registry)
        bootstraps: pd.DataFrame | None = None
        if with_bootstrap:
            frames = [
                paired_hierarchical_bootstrap(
                    metrics,
                    mask,
                    registry,
                    metric,
                    n_boot=bootstrap_replicates,
                    seed=registry.bootstrap_seed,
                )
                for metric in DETECTION_METRICS
            ]
            bootstraps = pd.concat(frames, ignore_index=True)
        provenance = {
            "config_sha256": config_sha,
            "registry_sha256": registry_sha,
            "compute_plan_sha256": plan_sha,
            "evaluation_marker_sha256": marker_sha,
            "evaluation_score_index_sha256": str(
                evaluation_provenance["score_index_sha256"]
            ),
            "valid_mask_sha256": valid_mask_sha256(mask),
        }
        root = Path(output_root or config["paths"]["compact_root"])
        return write_reporting_outputs(
            root,
            bundle,
            registry,
            plan,
            provenance=provenance,
            bootstraps=bootstraps,
        )
    except Exception as error:
        _preserve_reporting_failure(config, Path(config_path), error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--with-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args(argv)
    outputs = aggregate_evaluation(
        args.config,
        args.registry,
        args.plan,
        args.evaluation_root,
        args.output_root,
        with_bootstrap=args.with_bootstrap,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FACTORIAL_ARMS",
    "REPORTING_SCHEMA_VERSION",
    "aggregate_evaluation",
    "arm_registry_frame",
    "factorial_2x2_frame",
    "join_arm_registry",
    "long11_frame",
    "plot_tidy_frame",
    "table3_style_frame",
    "write_reporting_outputs",
]
