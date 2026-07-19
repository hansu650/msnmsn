"""Build the narrowed, paper-matched ViTTrace ablation result package.

This module is deliberately post-processing only.  It accepts one immutable
combined evaluation transaction and its completed 10,000-draw aggregation,
validates the exact 11-arm/492-series protocol, derives compact tables, and
delegates archive construction to :mod:`measure_vit4ts_v3.result_package`.
It never imports an encoder or scorer and never reads score arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .combined_evaluator import arm_registry
from .combined_protocol import CombinedProtocolSpec, load_combined_protocol, sha256_file
from .metrics import ALL_METRICS, DETECTION_METRICS, valid_mask_sha256, validate_metrics_against_mask
from .result_package import build_result_package, verify_result_zip


SCHEMA_VERSION = 1
EXPECTED_SERIES = 492
EXPECTED_VALID_SERIES = 488
EXPECTED_BOOTSTRAP_REPLICATES = 10_000
EXPECTED_BOOTSTRAP_SEED = 2027

EXPECTED_ARMS = (
    "FINAL_DEFAULT",
    "MATCH_FINAL_POSITION",
    "MATCH_FINAL_ROW",
    "SCALE_P",
    "IHP0_NCTP0",
    "IHP1_NCTP0",
    "IHP0_NCTP1",
    "IHP1_NCTP1",
    "CTRL_CLS_LEGACY",
    "CTRL_CLS_NATIVE_W240",
    "CTRL_PATCH_MEAN_NATIVE_W240",
)
EXPECTED_CONTRASTS = (
    "IHP_ONLY_VS_REL",
    "NCTP_ONLY_VS_REL",
    "FULL_VS_REL",
    "FULL_VS_IHP_ONLY",
)
TABLE3_ROWS = (
    ("w/o patch-level embedding", "CTRL_CLS_LEGACY"),
    ("w/o cross-patch comparison", "MATCH_FINAL_POSITION"),
    ("w/o column-wise comparison", "MATCH_FINAL_ROW"),
    ("w/o multi-scale embedding", "SCALE_P"),
    ("Full method", "FINAL_DEFAULT"),
)
PAPER_ARM_BY_LOCAL_DISPLAY = {
    "w/o patch-level embedding": "w/o patch-level embedding",
    "w/o cross-patch comparison": "w/o cross-patch comparison",
    "w/o column-wise comparison": "w/o column-wise comparison",
    "w/o multi-scale embedding": "w/o multi-scale embedding",
    "Full method": "ViT4TS (ours)",
}
FACTORIAL_ROWS = (
    ("REL", 0, 0, "IHP0_NCTP0"),
    ("IHP only", 1, 0, "IHP1_NCTP0"),
    ("NCTP only", 0, 1, "IHP0_NCTP1"),
    ("IHP + NCTP", 1, 1, "IHP1_NCTP1"),
)
FAMILY_COLUMNS = ("NAB", "NASA", "YAHOO")
SUBGROUP_COLUMNS = ("Art", "AWS", "AdEx", "Traf", "Tweets", "MSL", "SMAP", "A1", "A2", "A3", "A4")
SUBGROUP_SOURCE_BY_DISPLAY = {
    "Art": "NAB-Artificial",
    "AWS": "NAB-AWS",
    "AdEx": "NAB-AdExchange",
    "Traf": "NAB-Traffic",
    "Tweets": "NAB-Tweets",
    "MSL": "NASA-MSL",
    "SMAP": "NASA-SMAP",
    "A1": "Yahoo-A1",
    "A2": "Yahoo-A2",
    "A3": "Yahoo-A3",
    "A4": "Yahoo-A4",
}
METRIC_DISPLAY = {"f1_max": "F1-max", "auprc": "AUPRC", "vus_pr": "VUS-PR"}


@dataclass(frozen=True)
class ValidatedInputs:
    protocol: CombinedProtocolSpec
    metrics: pd.DataFrame
    mask: pd.DataFrame
    arm_metadata: pd.DataFrame
    subgroup11: pd.DataFrame
    family3: pd.DataFrame
    equal11: pd.DataFrame
    fileweighted: pd.DataFrame
    bootstrap: pd.DataFrame
    evaluation_marker: Mapping[str, Any]
    aggregation_marker: Mapping[str, Any]
    external_reference: pd.DataFrame


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"cannot read {context}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _boolean(series: pd.Series, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError(f"{context} is not boolean")
    return mapped.astype(bool)


def _require_hash(path: Path, expected: Any, context: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected or "").upper():
        raise ValueError(f"{context} hash binding failed: {path}")


def _require_complete_markers(
    protocol: CombinedProtocolSpec, evaluation: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    evaluation_marker = _read_json(
        evaluation / "_COMBINED_EVALUATION_COMPLETE.json", "combined evaluation marker"
    )
    aggregation_marker = _read_json(
        evaluation / "_COMBINED_AGGREGATION_COMPLETE.json", "combined aggregation marker"
    )
    common = {
        "status": "COMPLETE",
        "protocol_sha256": protocol.payload_sha256,
        "series_count": EXPECTED_SERIES,
        "valid_series_count": EXPECTED_VALID_SERIES,
        "arm_count": len(EXPECTED_ARMS),
    }
    for name, marker in (("evaluation", evaluation_marker), ("aggregation", aggregation_marker)):
        for key, expected in common.items():
            if marker.get(key) != expected:
                raise ValueError(f"{name} marker {key} differs: {marker.get(key)!r}")
    if aggregation_marker.get("contrast_count") != len(EXPECTED_CONTRASTS):
        raise ValueError("aggregation marker contrast count differs")
    if aggregation_marker.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED:
        raise ValueError("aggregation marker bootstrap seed differs")
    if aggregation_marker.get("bootstrap_replicates") != EXPECTED_BOOTSTRAP_REPLICATES:
        raise ValueError("aggregation marker bootstrap replicate count differs")
    return evaluation_marker, aggregation_marker


def _validate_protocol(protocol: CombinedProtocolSpec) -> None:
    if protocol.expected_series != EXPECTED_SERIES or protocol.expected_valid_series != EXPECTED_VALID_SERIES:
        raise ValueError("paper-matched delivery requires the frozen 492/488 protocol")
    if protocol.arm_ids != EXPECTED_ARMS:
        raise ValueError("paper-matched delivery requires the exact ordered 11-arm subset")
    if tuple(row.contrast_id for row in protocol.contrasts) != EXPECTED_CONTRASTS:
        raise ValueError("paper-matched delivery requires the exact four contrasts")
    if protocol.bootstrap_seed != EXPECTED_BOOTSTRAP_SEED or protocol.bootstrap_replicates != EXPECTED_BOOTSTRAP_REPLICATES:
        raise ValueError("paper-matched delivery bootstrap protocol differs")


def _validate_mask_and_metrics(
    protocol: CombinedProtocolSpec,
    evaluation: Path,
    evaluation_marker: Mapping[str, Any],
    aggregation_marker: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = evaluation / "per_series_metrics.csv"
    mask_path = evaluation / "valid_series_mask.csv"
    arm_path = evaluation / "arm_metadata.csv"
    provenance_path = evaluation / "evaluation_provenance.json"
    _require_hash(metrics_path, evaluation_marker.get("per_series_metrics_sha256"), "per-series metrics")
    _require_hash(mask_path, evaluation_marker.get("valid_series_mask_file_sha256"), "valid mask")
    _require_hash(arm_path, evaluation_marker.get("arm_metadata_sha256"), "arm metadata")
    _require_hash(provenance_path, evaluation_marker.get("evaluation_provenance_sha256"), "evaluation provenance")
    metrics = pd.read_csv(metrics_path)
    mask = pd.read_csv(mask_path)
    arms = pd.read_csv(arm_path)
    if len(mask) != EXPECTED_SERIES or mask["series_id"].astype(str).nunique() != EXPECTED_SERIES:
        raise ValueError("valid mask must contain exactly 492 unique series")
    valid_columns = [f"valid_{metric}" for metric in DETECTION_METRICS]
    for column in valid_columns:
        mask[column] = _boolean(mask[column], column)
    if not all(int(mask[column].sum()) == EXPECTED_VALID_SERIES for column in valid_columns):
        raise ValueError("each detection metric must use the common 488-series mask")
    if not all(mask[column].equals(mask[valid_columns[0]]) for column in valid_columns[1:]):
        raise ValueError("detection-metric validity masks differ")
    if valid_mask_sha256(mask) != str(aggregation_marker.get("valid_mask_sha256", "")).upper():
        raise ValueError("aggregation valid-mask identity differs")
    if len(metrics) != EXPECTED_SERIES * len(EXPECTED_ARMS):
        raise ValueError("per-series metrics must contain exactly 11 x 492 rows")
    validated = validate_metrics_against_mask(metrics, mask, arm_registry(protocol).arm_ids)
    if len(arms) != len(EXPECTED_ARMS) or tuple(arms.sort_values("arm_order")["arm"].astype(str)) != EXPECTED_ARMS:
        raise ValueError("arm metadata does not cover the exact ordered arm subset")
    return validated, mask, arms


def _validate_aggregate_frame(
    frame: pd.DataFrame,
    name: str,
    expected_rows: int,
) -> None:
    required = {"aggregation", "family", "subgroup", "arm", "metric", "value", "n_valid", "n_total"}
    if set(frame.columns) != required or len(frame) != expected_rows:
        raise ValueError(f"{name} aggregate schema/shape differs")
    if set(frame["arm"].astype(str)) != set(EXPECTED_ARMS) or set(frame["metric"].astype(str)) != set(ALL_METRICS):
        raise ValueError(f"{name} aggregate arm/metric grid differs")
    # Detection metrics must be defined everywhere. Anomaly-free burden is
    # intentionally undefined in subgroups with no anomaly-free series.
    detection = frame.loc[frame["metric"].isin(DETECTION_METRICS), "value"]
    detection_values = pd.to_numeric(detection, errors="coerce")
    if detection_values.isna().any() or (detection_values < 0.0).any() or (detection_values > 1.0).any():
        raise ValueError(f"{name} detection aggregate contains undefined values")
    auxiliary = frame.loc[~frame["metric"].isin(DETECTION_METRICS)].copy()
    auxiliary_values = pd.to_numeric(auxiliary["value"], errors="coerce")
    auxiliary_counts = pd.to_numeric(auxiliary["n_valid"], errors="raise")
    if not np.isfinite(auxiliary_values.loc[auxiliary_counts > 0]).all():
        raise ValueError(f"{name} defined anomaly-free aggregate is not finite")
    if auxiliary_values.loc[auxiliary_counts == 0].notna().any():
        raise ValueError(f"{name} undefined anomaly-free aggregate must be NA")


def _check_reconstructed_views(
    subgroup11: pd.DataFrame, family3: pd.DataFrame, equal11: pd.DataFrame
) -> None:
    detection = subgroup11.loc[subgroup11["metric"].isin(DETECTION_METRICS)].copy()
    reconstructed_family = (
        detection.groupby(["family", "arm", "metric"], as_index=False)["value"].mean()
    )
    authoritative_family = family3.loc[family3["metric"].isin(DETECTION_METRICS), ["family", "arm", "metric", "value"]]
    merged = reconstructed_family.merge(
        authoritative_family, on=["family", "arm", "metric"], suffixes=("_rebuilt", "_stored"), validate="one_to_one"
    )
    if len(merged) != 3 * len(EXPECTED_ARMS) * len(DETECTION_METRICS) or not np.allclose(
        merged["value_rebuilt"], merged["value_stored"], atol=1e-12, rtol=1e-10
    ):
        raise ValueError("family3 values do not reconstruct from subgroup11")
    reconstructed_equal = detection.groupby(["arm", "metric"], as_index=False)["value"].mean()
    authoritative_equal = equal11.loc[equal11["metric"].isin(DETECTION_METRICS), ["arm", "metric", "value"]]
    merged_equal = reconstructed_equal.merge(
        authoritative_equal, on=["arm", "metric"], suffixes=("_rebuilt", "_stored"), validate="one_to_one"
    )
    if len(merged_equal) != len(EXPECTED_ARMS) * len(DETECTION_METRICS) or not np.allclose(
        merged_equal["value_rebuilt"], merged_equal["value_stored"], atol=1e-12, rtol=1e-10
    ):
        raise ValueError("equal11 values do not reconstruct from subgroup11")


def _validate_aggregates(
    evaluation: Path, aggregation_marker: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = ("subgroup11", "family3", "equal11", "fileweighted", "bootstrap")
    paths = {name: evaluation / ("bootstrap_ci.csv" if name == "bootstrap" else f"{name}_metrics.csv") for name in names}
    validated_path = evaluation / "per_series_metrics_validated.csv"
    for name, path in {"per_series": validated_path, **paths}.items():
        _require_hash(path, aggregation_marker.get(f"{name}_sha256"), f"aggregated {name}")
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    _validate_aggregate_frame(frames["subgroup11"], "subgroup11", 11 * len(EXPECTED_ARMS) * len(ALL_METRICS))
    _validate_aggregate_frame(frames["family3"], "family3", 3 * len(EXPECTED_ARMS) * len(ALL_METRICS))
    _validate_aggregate_frame(frames["equal11"], "equal11", len(EXPECTED_ARMS) * len(ALL_METRICS))
    _validate_aggregate_frame(frames["fileweighted"], "fileweighted", len(EXPECTED_ARMS) * len(ALL_METRICS))
    _check_reconstructed_views(frames["subgroup11"], frames["family3"], frames["equal11"])
    bootstrap = frames["bootstrap"]
    required_bootstrap = {
        "contrast_id", "contrast_family", "candidate", "control", "metric", "delta", "ci_lower", "ci_upper",
        "n_boot", "seed", "shared_indices", "resample_plan_sha256", "valid_mask_sha256", "n_valid_series",
        "n_subgroups", "resampling_unit",
    }
    if set(bootstrap.columns) != required_bootstrap or len(bootstrap) != len(EXPECTED_CONTRASTS) * len(DETECTION_METRICS):
        raise ValueError("bootstrap schema/shape differs")
    expected_grid = {(contrast, metric) for contrast in EXPECTED_CONTRASTS for metric in DETECTION_METRICS}
    if set(zip(bootstrap["contrast_id"].astype(str), bootstrap["metric"].astype(str))) != expected_grid:
        raise ValueError("bootstrap contrast/metric grid differs")
    shared = _boolean(bootstrap["shared_indices"], "bootstrap shared_indices")
    numeric_gate = (
        (pd.to_numeric(bootstrap["n_boot"]) == EXPECTED_BOOTSTRAP_REPLICATES).all()
        and (pd.to_numeric(bootstrap["seed"]) == EXPECTED_BOOTSTRAP_SEED).all()
        and (pd.to_numeric(bootstrap["n_valid_series"]) == EXPECTED_VALID_SERIES).all()
        and (pd.to_numeric(bootstrap["n_subgroups"]) == 11).all()
        and shared.all()
    )
    if not numeric_gate or bootstrap["resample_plan_sha256"].astype(str).nunique() != 1:
        raise ValueError("bootstrap frozen protocol differs")
    for column in ("delta", "ci_lower", "ci_upper"):
        if not np.isfinite(pd.to_numeric(bootstrap[column], errors="coerce")).all():
            raise ValueError(f"bootstrap {column} is not finite")
    return frames["subgroup11"], frames["family3"], frames["equal11"], frames["fileweighted"], bootstrap


def _validate_external_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"paper", "page", "table_or_figure", "arm", "group", "metric", "value", "value_status", "notes"}
    if set(frame.columns) != required:
        raise ValueError("external reference schema differs")
    table3 = frame.loc[(frame["table_or_figure"] == "Table 3") & (frame["metric"] == "F1-max")].copy()
    expected_arms = set(PAPER_ARM_BY_LOCAL_DISPLAY.values())
    if len(table3) != 15 or set(table3["arm"].astype(str)) != expected_arms or set(table3["group"].astype(str).str.upper()) != set(FAMILY_COLUMNS):
        raise ValueError("external Table 3 reference must contain the exact 5 x 3 rows")
    if set(table3["value_status"].astype(str)) != {"exact_paper_reported"}:
        raise ValueError("external Table 3 values must be explicitly paper-reported")
    if not np.isfinite(pd.to_numeric(table3["value"], errors="coerce")).all():
        raise ValueError("external Table 3 values are not finite")
    table1 = frame.loc[(frame["table_or_figure"] == "Table 1") & (frame["arm"] == "ViT4TS") & (frame["metric"] == "F1-max")]
    if len(table1) != 12 or set(table1["group"].astype(str)) != set((*SUBGROUP_COLUMNS, "equal-11")):
        raise ValueError("external Table 1 reference must contain the exact 11 subgroup plus equal-11 rows")
    if set(table1["value_status"].astype(str)) != {"exact_paper_reported"}:
        raise ValueError("external Table 1 values must be explicitly paper-reported")
    if not np.isfinite(pd.to_numeric(table1["value"], errors="coerce")).all():
        raise ValueError("external Table 1 values are not finite")
    return frame


def validate_delivery_inputs(
    registry_path: Path, evaluation_dir: Path, external_reference: Path
) -> ValidatedInputs:
    protocol = load_combined_protocol(Path(registry_path))
    _validate_protocol(protocol)
    evaluation = Path(evaluation_dir)
    evaluation_marker, aggregation_marker = _require_complete_markers(protocol, evaluation)
    metrics, mask, arms = _validate_mask_and_metrics(
        protocol, evaluation, evaluation_marker, aggregation_marker
    )
    subgroup11, family3, equal11, fileweighted, bootstrap = _validate_aggregates(
        evaluation, aggregation_marker
    )
    external = _validate_external_reference(Path(external_reference))
    return ValidatedInputs(
        protocol, metrics, mask, arms, subgroup11, family3, equal11,
        fileweighted, bootstrap, evaluation_marker, aggregation_marker, external
    )


def _value_lookup(frame: pd.DataFrame, arm: str, metric: str, key: str, value: str) -> float:
    selected = frame.loc[
        (frame["arm"].astype(str) == arm)
        & (frame["metric"].astype(str) == metric)
        & (frame[key].astype(str).str.upper() == value.upper()),
        "value",
    ]
    if len(selected) != 1:
        raise ValueError(f"missing unique aggregate value for {arm}/{metric}/{value}")
    return float(selected.iloc[0])


def _subgroup_source(frame: pd.DataFrame, display: str) -> str:
    observed = set(frame["subgroup"].astype(str))
    source = SUBGROUP_SOURCE_BY_DISPLAY[display]
    matches = [candidate for candidate in (source, display) if candidate in observed]
    if len(matches) != 1:
        raise ValueError(f"missing unique subgroup mapping for {display}: {matches}")
    return matches[0]


def build_table3(inputs: ValidatedInputs, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for display_name, arm in TABLE3_ROWS:
        row: dict[str, Any] = {
            "display_name": display_name,
            "arm": arm,
            "metric": METRIC_DISPLAY[metric],
            "value_status": "local_same_protocol_corrected_primary",
            "is_final": arm == "FINAL_DEFAULT",
        }
        for family in FAMILY_COLUMNS:
            row[family] = _value_lookup(inputs.family3, arm, metric, "family", family)
        row["equal11"] = _value_lookup(inputs.equal11, arm, metric, "family", "ALL")
        row["fileweighted"] = _value_lookup(inputs.fileweighted, arm, metric, "family", "ALL")
        rows.append(row)
    return pd.DataFrame(rows)


def build_factorial_table(inputs: ValidatedInputs, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for display_name, ihp, nctp, arm in FACTORIAL_ROWS:
        row: dict[str, Any] = {
            "display_name": display_name,
            "arm": arm,
            "ihp": ihp,
            "nctp": nctp,
            "metric": METRIC_DISPLAY[metric],
            "value_status": "local_same_protocol_corrected_primary",
            "is_final": arm == "IHP1_NCTP1",
        }
        for family in FAMILY_COLUMNS:
            row[family] = _value_lookup(inputs.family3, arm, metric, "family", family)
        row["equal11"] = _value_lookup(inputs.equal11, arm, metric, "family", "ALL")
        row["fileweighted"] = _value_lookup(inputs.fileweighted, arm, metric, "family", "ALL")
        rows.append(row)
    return pd.DataFrame(rows)


def build_final_long11(inputs: ValidatedInputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in DETECTION_METRICS:
        row: dict[str, Any] = {
            "arm": "FINAL_DEFAULT",
            "metric": METRIC_DISPLAY[metric],
            "value_status": "local_same_protocol_corrected_primary",
            "is_final": True,
        }
        for subgroup in SUBGROUP_COLUMNS:
            source = _subgroup_source(inputs.subgroup11, subgroup)
            row[subgroup] = _value_lookup(inputs.subgroup11, "FINAL_DEFAULT", metric, "subgroup", source)
        row["equal11"] = _value_lookup(inputs.equal11, "FINAL_DEFAULT", metric, "family", "ALL")
        row["fileweighted"] = _value_lookup(inputs.fileweighted, "FINAL_DEFAULT", metric, "family", "ALL")
        rows.append(row)
    return pd.DataFrame(rows)


def build_external_comparison(inputs: ValidatedInputs) -> pd.DataFrame:
    external = inputs.external_reference.loc[
        (inputs.external_reference["table_or_figure"] == "Table 3")
        & (inputs.external_reference["metric"] == "F1-max")
    ].copy()
    rows: list[dict[str, Any]] = []
    for local_display, local_arm in TABLE3_ROWS:
        paper_arm = PAPER_ARM_BY_LOCAL_DISPLAY[local_display]
        for family in FAMILY_COLUMNS:
            match = external.loc[
                (external["arm"].astype(str) == paper_arm)
                & (external["group"].astype(str).str.upper() == family)
            ]
            if len(match) != 1:
                raise ValueError("external descriptive comparison row is not unique")
            paper_value = float(match.iloc[0]["value"])
            local_value = _value_lookup(inputs.family3, local_arm, "f1_max", "family", family)
            rows.append(
                {
                    "display_name": local_display,
                    "local_arm": local_arm,
                    "external_paper_arm": paper_arm,
                    "family": family,
                    "metric": "F1-max",
                    "local_value": local_value,
                    "external_paper_value": paper_value,
                    "descriptive_delta": local_value - paper_value,
                    "local_value_status": "local_same_protocol_corrected_primary",
                    "external_value_status": "exact_paper_reported",
                    "comparison_status": "descriptive_external_only_no_paired_ci",
                    "paired_ci_applicable": False,
                }
            )
    return pd.DataFrame(rows)


def build_main_vs_paper_f1(inputs: ValidatedInputs) -> pd.DataFrame:
    external = inputs.external_reference.loc[
        (inputs.external_reference["table_or_figure"] == "Table 1")
        & (inputs.external_reference["arm"] == "ViT4TS")
        & (inputs.external_reference["metric"] == "F1-max")
    ].copy()
    rows: list[dict[str, Any]] = []
    for display in (*SUBGROUP_COLUMNS, "equal-11"):
        if display == "equal-11":
            local_value = _value_lookup(inputs.equal11, "FINAL_DEFAULT", "f1_max", "family", "ALL")
            external_group = "equal-11"
        else:
            source = _subgroup_source(inputs.subgroup11, display)
            local_value = _value_lookup(inputs.subgroup11, "FINAL_DEFAULT", "f1_max", "subgroup", source)
            external_group = display
        match = external.loc[external["group"].astype(str).str.lower() == external_group.lower()]
        if len(match) != 1:
            raise ValueError(f"external Table 1 row is not unique: {external_group}")
        paper_value = float(match.iloc[0]["value"])
        rows.append(
            {
                "group": display,
                "local_arm": "FINAL_DEFAULT",
                "external_paper_arm": "ViT4TS",
                "metric": "F1-max",
                "local_value": local_value,
                "external_paper_value": paper_value,
                "descriptive_delta": local_value - paper_value,
                "local_value_status": "local_same_protocol_corrected_primary",
                "external_value_status": "exact_paper_reported",
                "comparison_status": "descriptive_external_not_paired",
                "paired_ci_applicable": False,
            }
        )
    return pd.DataFrame(rows)


def _registry_csv(protocol: CombinedProtocolSpec) -> pd.DataFrame:
    stage_by_arm = {
        arm.arm_id: stage
        for stage in protocol.stages
        for arm in stage.arms
    }
    rows: list[dict[str, Any]] = []
    for arm in protocol.arms:
        stage = stage_by_arm[arm.arm_id]
        rows.append(
            {
                "arm": arm.arm_id,
                "source_arm": arm.source_arm,
                "order": arm.order,
                "role": arm.role,
                "stage_id": stage.stage_id,
                "stage_kind": stage.stage_kind,
                "experiment_group": arm.experiment_group,
                "changed_factor": arm.changed_factor,
                "metadata_json": json.dumps(dict(arm.metadata), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(strict=True), destination)


def _stage_tree(
    inputs: ValidatedInputs,
    registry_path: Path,
    evaluation_dir: Path,
    external_reference: Path,
    repo_root: Path,
    output_root: Path,
) -> None:
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"paper-matched output root is not empty: {root}")
    for directory in (
        "config", "code", "tests", "manifests", "provenance", "failures",
        "results", "tables", "plot_data", "rough_figures",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)

    evaluation = Path(evaluation_dir)
    authoritative = (
        "per_series_metrics.csv", "per_series_metrics_validated.csv", "valid_series_mask.csv",
        "arm_metadata.csv", "evaluation_provenance.json", "subgroup11_metrics.csv",
        "family3_metrics.csv", "equal11_metrics.csv", "fileweighted_metrics.csv", "bootstrap_ci.csv",
    )
    for name in authoritative:
        _copy(evaluation / name, root / "results" / name)
    _copy(registry_path, root / "arm_registry.json")
    _copy(registry_path, root / "config" / "vittrace_paper_ablation_registry.json")
    _atomic_csv(root / "arm_registry.csv", _registry_csv(inputs.protocol))
    _copy(external_reference, root / "external_vit4ts_reference.csv")
    _copy(
        evaluation / "_COMBINED_EVALUATION_COMPLETE.json",
        root / "manifests" / "_COMBINED_EVALUATION_COMPLETE.json",
    )
    _copy(
        evaluation / "_COMBINED_AGGREGATION_COMPLETE.json",
        root / "manifests" / "_COMBINED_AGGREGATION_COMPLETE.json",
    )

    tables: dict[str, pd.DataFrame] = {}
    for metric in DETECTION_METRICS:
        tables[f"table3_style_{metric}.csv"] = build_table3(inputs, metric)
        tables[f"table_ihp_nctp_2x2_{metric}.csv"] = build_factorial_table(inputs, metric)
    tables["table_final_default_long11.csv"] = build_final_long11(inputs)
    tables["external_descriptive_comparison.csv"] = build_external_comparison(inputs)
    tables["main_vs_paper_f1.csv"] = build_main_vs_paper_f1(inputs)
    for name, frame in tables.items():
        _atomic_csv(root / "tables" / name, frame)
    _atomic_csv(
        root / "plot_data" / "ihp_nctp_2x2_tidy.csv",
        pd.concat([build_factorial_table(inputs, metric) for metric in DETECTION_METRICS], ignore_index=True),
    )
    _atomic_csv(root / "plot_data" / "external_descriptive_comparison.csv", tables["external_descriptive_comparison.csv"])
    _atomic_text(root / "rough_figures" / "README.md", "No figures are generated in the narrowed data-only delivery.\n")

    source = repo_root / "code" / "src" / "measure_vit4ts_v3" / "paper_matched_delivery.py"
    test = repo_root / "code" / "tests" / "test_vittrace_paper_matched_delivery.py"
    _copy(source, root / "code" / source.name)
    _copy(test, root / "tests" / test.name)
    _atomic_csv(
        root / "failures" / "failure_manifest.csv",
        pd.DataFrame(columns=["failure_id", "stage", "arm", "series_id", "status", "reason"]),
    )
    input_paths = {
        "registry": Path(registry_path),
        "evaluation_marker": evaluation / "_COMBINED_EVALUATION_COMPLETE.json",
        "aggregation_marker": evaluation / "_COMBINED_AGGREGATION_COMPLETE.json",
        "per_series_metrics": evaluation / "per_series_metrics.csv",
        "valid_series_mask": evaluation / "valid_series_mask.csv",
        "bootstrap_ci": evaluation / "bootstrap_ci.csv",
        "external_reference": Path(external_reference),
        "delivery_source": source,
        "delivery_test": test,
    }
    provenance_rows = [
        {"artifact": name, "source_path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in input_paths.items()
    ]
    _atomic_csv(root / "provenance" / "input_identities.csv", pd.DataFrame(provenance_rows))
    _atomic_json(
        root / "provenance" / "protocol.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": inputs.protocol.protocol_id,
            "protocol_sha256": inputs.protocol.payload_sha256,
            "series_count": EXPECTED_SERIES,
            "valid_series_count": EXPECTED_VALID_SERIES,
            "arm_count": len(EXPECTED_ARMS),
            "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
            "bootstrap_replicates": EXPECTED_BOOTSTRAP_REPLICATES,
            "comparison_policy": "external paper values are descriptive only and receive no paired CI",
        },
    )
    _atomic_text(
        root / "README.md",
        "# ViTTrace paper-matched ablation results\n\n"
        "This compact package contains the completed 492-series, common-488 corrected-primary "
        "evaluation for the ViT4TS Table 3-style local controls and the IHP x NCTP 2x2. "
        "Published ViT4TS values are external descriptive references, not same-run paired baselines.\n",
    )
    _atomic_text(
        root / "STATUS.md",
        "# Status\n\nCOMPLETE: 11 arms x 492 series; common valid mask 488; "
        "F1-max/AUPRC/VUS-PR; 10,000 paired hierarchical bootstrap draws (seed 2027).\n",
    )
    _atomic_text(
        root / "EXPERIMENT_LOG.md",
        "# Experiment log\n\nNo encoder or scorer was executed by this delivery step. "
        "All tables were derived from the hash-bound COMPLETE combined evaluation and aggregation.\n",
    )
    _atomic_json(
        root / "manifests" / "_PAPER_MATCHED_DELIVERY_COMPLETE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "protocol_sha256": inputs.protocol.payload_sha256,
            "series_count": EXPECTED_SERIES,
            "valid_series_count": EXPECTED_VALID_SERIES,
            "arm_count": len(EXPECTED_ARMS),
            "table3_row_count": len(TABLE3_ROWS),
            "factorial_row_count": len(FACTORIAL_ROWS),
            "external_comparison_status": "DESCRIPTIVE_ONLY_NO_PAIRED_CI",
        },
    )


def assemble_paper_matched_delivery(
    registry_path: Path,
    evaluation_dir: Path,
    external_reference: Path,
    repo_root: Path,
    output_root: Path,
    *,
    zip_path: Path | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    """Validate completed evidence, stage compact tables, and verify the ZIP."""

    inputs = validate_delivery_inputs(registry_path, evaluation_dir, external_reference)
    _stage_tree(
        inputs, Path(registry_path), Path(evaluation_dir), Path(external_reference),
        Path(repo_root), Path(output_root)
    )
    archive, manifest = build_result_package(
        output_root, zip_path=zip_path, allow_incomplete=False
    )
    verified = verify_result_zip(archive)
    if verified.get("sha256sums_sha256") != manifest.get("sha256sums_sha256"):
        raise RuntimeError("result ZIP verification manifest differs")
    return archive, verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--external-reference", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args(argv)
    archive, manifest = assemble_paper_matched_delivery(
        args.registry, args.evaluation_dir, args.external_reference, args.repo_root,
        args.output_root, zip_path=args.zip
    )
    print(json.dumps({"archive": str(archive), **dict(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ARMS",
    "EXPECTED_CONTRASTS",
    "FACTORIAL_ROWS",
    "TABLE3_ROWS",
    "ValidatedInputs",
    "assemble_paper_matched_delivery",
    "build_external_comparison",
    "build_factorial_table",
    "build_final_long11",
    "build_main_vs_paper_f1",
    "build_table3",
    "validate_delivery_inputs",
]
