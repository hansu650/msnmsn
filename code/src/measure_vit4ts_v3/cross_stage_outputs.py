"""Strict cross-stage metric assembly for the mandatory ViTTrace v3 grid.

The scorer/evaluator stages remain independent transactions.  This module
joins only evaluator-committed per-series metric tables whose paths and
SHA256 identities are registered in ``stage_evaluation_index.json``.  A
missing, failed, or stale stage is represented as ``BLOCKED``; it is never
replaced by a blank/zero row or a paper-reported value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
METRICS = ("f1_max", "auprc", "vus_pr")
FAMILIES = ("NAB", "NASA", "Yahoo")
ANOMALY_FREE_FIELDS = (
    "anomaly_free_fp_rate",
    "anomaly_free_fp_count",
    "anomaly_free_mean_excess",
    "anomaly_free_score_p95",
)
SUBGROUP_DISPLAY = {
    "NAB-Artificial": "Art",
    "NAB-AWS": "AWS",
    "NAB-AdExchange": "AdEx",
    "NAB-Traffic": "Traf",
    "NAB-Tweets": "Tweets",
    "NASA-MSL": "MSL",
    "NASA-SMAP": "SMAP",
    "Yahoo-A1": "A1",
    "Yahoo-A2": "A2",
    "Yahoo-A3": "A3",
    "Yahoo-A4": "A4",
}
SUBGROUPS = tuple(SUBGROUP_DISPLAY)

MANDATORY_GROUP_COUNTS: dict[str, int] = {
    "cache_only_controls": 1,
    "encoder_controls": 1,
    "window_sensitivity": 5,
    "stride_sensitivity": 3,
    "backbone_sensitivity": 4,
    "representation_sensitivity": 2,
}

ARM_METADATA_COLUMNS = (
    "arm",
    "source_arm",
    "display_name",
    "stage_kind",
    "arm_role",
    "arm_order",
    "arm_metadata_json",
    "experiment_group",
    "changed_factor",
    "fixed_factors",
    "fixed_factors_json",
    "backbone",
    "representation",
    "window",
    "stride",
    "patch_size",
    "patch_grid",
    "matching_scope",
    "memory",
    "scale_subset",
    "incidence",
    "temporal",
    "reducer_family",
    "reducer_setting",
    "ihp",
    "nctp",
    "encoder_calls",
    "elapsed_seconds_per_series",
    "source_cache_sha256",
    "model_sha256",
    "renderer_sha256",
    "token_shapes",
    "status",
    "failure_reason",
    "is_final",
)

RECIPE_GROUPS: dict[str, frozenset[str]] = {
    "ihp_nctp_interaction": frozenset({"IHP_X_NCTP"}),
    "matching_scope": frozenset({"MATCHING_LEGACY", "MATCHING_FINAL"}),
    "scale_subset_heatmap": frozenset({"SCALE_SUBSET", "SCALE_SUBSET_LEGACY"}),
    "reducer_sensitivity": frozenset({"REDUCER_QUANTILE", "REDUCER_TOP_FRACTION"}),
}


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class StageEvaluation:
    stage_id: str
    stage_group: str
    configuration_id: str
    status: str
    reason: str
    metrics_path: Path | None
    metrics_sha256: str
    marker_path: Path | None
    marker_sha256: str
    arm_metadata_path: Path | None
    arm_metadata_sha256: str
    expected_series: int


def _path(value: Any) -> Path | None:
    text = str(value or "")
    return Path(text).resolve() if text else None


def _digest(value: Any, context: str, *, allow_empty: bool = False) -> str:
    text = str(value or "").upper()
    if not text and allow_empty:
        return ""
    if len(text) != 64 or any(character not in "0123456789ABCDEF" for character in text):
        raise ValueError(f"{context} must be a SHA256 digest")
    return text


def load_stage_index(path: Path) -> tuple[str, tuple[StageEvaluation, ...]]:
    """Load and validate the immutable cross-stage evaluator index."""

    target = Path(path).resolve(strict=True)
    payload = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping) or "schema_version" not in payload or "stages" not in payload:
        raise ValueError("stage evaluation index has a stale schema")
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("stage evaluation index schema_version changed")
    protocol_value = payload.get("protocol_sha256", payload.get("manifest_sha256"))
    protocol_sha = _digest(protocol_value, "protocol/manifest_sha256")
    raw_stages = payload["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stage evaluation index must contain stages")
    stages: list[StageEvaluation] = []
    seen: set[str] = set()
    required = {
        "stage_id",
        "stage_group",
        "configuration_id",
        "status",
        "reason",
        "metrics_path",
        "metrics_sha256",
        "marker_path",
        "marker_sha256",
        "arm_metadata_path",
        "arm_metadata_sha256",
        "expected_series",
    }
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping) or not required.issubset(set(raw)):
            raise ValueError(f"stage[{index}] has a stale schema")
        stage_id = str(raw["stage_id"])
        if not stage_id or stage_id in seen:
            raise ValueError("stage identifiers must be unique and nonempty")
        seen.add(stage_id)
        status = str(raw["status"]).upper()
        reason = str(raw["reason"] or "")
        if status not in {"COMPLETE", "BLOCKED", "FAILED"}:
            raise ValueError("stage status must be COMPLETE/BLOCKED/FAILED")
        metrics_path = _path(raw["metrics_path"])
        marker_path = _path(raw["marker_path"])
        metadata_path = _path(raw["arm_metadata_path"])
        if status == "COMPLETE":
            if None in (metrics_path, marker_path, metadata_path):
                raise ValueError("complete stage lacks committed artifact paths")
        elif not reason:
            raise ValueError("blocked/failed stage requires an explicit reason")
        stages.append(
            StageEvaluation(
                stage_id=stage_id,
                stage_group=str(raw["stage_group"]),
                configuration_id=str(raw["configuration_id"]),
                status=status,
                reason=reason,
                metrics_path=metrics_path,
                metrics_sha256=_digest(
                    raw["metrics_sha256"], "metrics_sha256", allow_empty=status != "COMPLETE"
                ),
                marker_path=marker_path,
                marker_sha256=_digest(
                    raw["marker_sha256"], "marker_sha256", allow_empty=status != "COMPLETE"
                ),
                arm_metadata_path=metadata_path,
                arm_metadata_sha256=_digest(
                    raw["arm_metadata_sha256"],
                    "arm_metadata_sha256",
                    allow_empty=status != "COMPLETE",
                ),
                expected_series=int(raw["expected_series"]),
            )
        )
    return protocol_sha, tuple(stages)


def stage_status_frame(stages: Sequence[StageEvaluation]) -> pd.DataFrame:
    rows = [
        {
            "stage_id": stage.stage_id,
            "stage_group": stage.stage_group,
            "configuration_id": stage.configuration_id,
            "status": stage.status,
            "reason": stage.reason,
            "expected_series": stage.expected_series,
            "metrics_path": str(stage.metrics_path or ""),
            "metrics_sha256": stage.metrics_sha256,
            "marker_path": str(stage.marker_path or ""),
            "marker_sha256": stage.marker_sha256,
            "arm_metadata_path": str(stage.arm_metadata_path or ""),
            "arm_metadata_sha256": stage.arm_metadata_sha256,
        }
        for stage in stages
    ]
    frame = pd.DataFrame(rows)
    coverage = frame.loc[frame["status"] == "COMPLETE"].groupby("stage_group")[
        "configuration_id"
    ].nunique()
    for group, expected in MANDATORY_GROUP_COUNTS.items():
        actual = int(coverage.get(group, 0))
        if actual < expected:
            missing = expected - actual
            frame = pd.concat(
                [
                    frame,
                    pd.DataFrame(
                        [
                            {
                                "stage_id": f"MISSING::{group}",
                                "stage_group": group,
                                "configuration_id": "",
                                "status": "BLOCKED",
                                "reason": f"mandatory configuration coverage {actual}/{expected}; missing {missing}",
                                "expected_series": 0,
                                "metrics_path": "",
                                "metrics_sha256": "",
                                "marker_path": "",
                                "marker_sha256": "",
                                "arm_metadata_path": "",
                                "arm_metadata_sha256": "",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return frame.sort_values(["stage_group", "configuration_id", "stage_id"]).reset_index(
        drop=True
    )


def _load_complete_stage(stage: StageEvaluation) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stage.status != "COMPLETE":
        raise ValueError("only COMPLETE stages may be loaded")
    assert stage.metrics_path and stage.marker_path and stage.arm_metadata_path
    identities = (
        (stage.metrics_path, stage.metrics_sha256, "metrics"),
        (stage.marker_path, stage.marker_sha256, "marker"),
        (stage.arm_metadata_path, stage.arm_metadata_sha256, "arm metadata"),
    )
    for path, expected, context in identities:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{stage.stage_id} {context} identity mismatch")
    metrics = pd.read_csv(stage.metrics_path)
    metadata = pd.read_csv(stage.arm_metadata_path)
    required = {"series_id", "family", "subgroup", "arm", *METRICS}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"{stage.stage_id} metrics missing {sorted(missing)}")
    if metrics.duplicated(["series_id", "arm"]).any() or metrics.empty:
        raise ValueError(f"{stage.stage_id} metrics are not a unique nonempty grid")
    if metrics["series_id"].nunique() != stage.expected_series:
        raise ValueError(f"{stage.stage_id} series count differs from its committed index")
    if set(metrics["family"].astype(str)) - set(FAMILIES):
        raise ValueError(f"{stage.stage_id} contains an unknown family")
    if set(metrics["subgroup"].astype(str)) - set(SUBGROUPS):
        raise ValueError(f"{stage.stage_id} contains an unknown subgroup")
    if "arm" not in metadata or metadata["arm"].duplicated().any() or metadata.empty:
        raise ValueError(f"{stage.stage_id} arm metadata are invalid")
    unknown = set(metrics["arm"].astype(str)) - set(metadata["arm"].astype(str))
    if unknown:
        raise ValueError(f"{stage.stage_id} has metrics without arm metadata: {sorted(unknown)}")
    parsed_metadata: list[Mapping[str, Any]] = []
    if "arm_metadata_json" in metadata:
        for row_index, value in enumerate(metadata["arm_metadata_json"]):
            try:
                parsed = json.loads(str(value)) if pd.notna(value) else {}
            except json.JSONDecodeError as error:
                raise ValueError(f"{stage.stage_id} arm metadata JSON is invalid") from error
            if not isinstance(parsed, Mapping):
                raise ValueError(f"{stage.stage_id} arm metadata JSON must contain objects")
            parsed_metadata.append(parsed)
    else:
        parsed_metadata = [{} for _ in range(len(metadata))]
    for column in (
        "display_name", "is_final", "source_cache_sha256", "model_sha256",
        "renderer_sha256", "token_shapes", "status", "failure_reason",
    ):
        if column not in metadata:
            values = [row.get(column, np.nan) for row in parsed_metadata]
            metadata[column] = [
                json.dumps(value, sort_keys=True) if isinstance(value, (Mapping, list, tuple)) else value
                for value in values
            ]
    if "fixed_factors" not in metadata and "fixed_factors_json" in metadata:
        metadata["fixed_factors"] = metadata["fixed_factors_json"]
    if "fixed_factors_json" not in metadata and "fixed_factors" in metadata:
        metadata["fixed_factors_json"] = metadata["fixed_factors"]
    if "display_name" not in metadata:
        metadata["display_name"] = metadata["arm"]
    for column in ARM_METADATA_COLUMNS:
        if column not in metadata:
            metadata[column] = np.nan
    metadata = metadata.loc[:, ARM_METADATA_COLUMNS].copy()
    duplicate_metadata = [
        column for column in ARM_METADATA_COLUMNS if column != "arm" and column in metrics
    ]
    metrics = metrics.drop(columns=duplicate_metadata)
    metrics = metrics.merge(metadata, on="arm", how="left", validate="many_to_one")
    metrics.insert(0, "stage_id", stage.stage_id)
    metrics.insert(1, "stage_group", stage.stage_group)
    metrics.insert(2, "configuration_id", stage.configuration_id)
    return metrics, metadata


def _aggregate_stage(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    id_columns = ["stage_id", "stage_group", "configuration_id", *ARM_METADATA_COLUMNS]
    long = metrics.melt(
        id_vars=[*id_columns, "series_id", "family", "subgroup"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["is_valid"] = np.isfinite(long["value"].to_numpy(dtype=np.float64))
    group_keys = [*id_columns, "metric"]
    subgroup = (
        long.groupby([*group_keys, "family", "subgroup"], dropna=False, sort=True)
        .agg(value=("value", "mean"), n_valid=("is_valid", "sum"), n_total=("series_id", "nunique"))
        .reset_index()
    )
    family = (
        subgroup.groupby([*group_keys, "family"], dropna=False, sort=True)
        .agg(value=("value", "mean"), n_valid=("n_valid", "sum"), n_total=("n_total", "sum"))
        .reset_index()
    )
    equal11 = (
        subgroup.groupby(group_keys, dropna=False, sort=True)
        .agg(value=("value", "mean"), n_valid=("n_valid", "sum"), n_total=("n_total", "sum"))
        .reset_index()
    )
    equal11["family"] = "ALL"
    equal11["subgroup"] = "equal11"
    fileweighted = (
        long.groupby(group_keys, dropna=False, sort=True)
        .agg(value=("value", "mean"), n_valid=("is_valid", "sum"), n_total=("series_id", "nunique"))
        .reset_index()
    )
    fileweighted["family"] = "ALL"
    fileweighted["subgroup"] = "fileweighted"
    subgroup["view"] = "subgroup11"
    family["subgroup"] = family["family"]
    family["view"] = "family3"
    equal11["view"] = "equal11"
    fileweighted["view"] = "fileweighted"
    return {
        "per_series": metrics,
        "subgroup11": subgroup,
        "family3": family,
        "equal11": equal11,
        "fileweighted": fileweighted,
        "plot_tidy": pd.concat([subgroup, family, equal11, fileweighted], ignore_index=True),
    }


def _presentation_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["stage_id", "stage_group", "configuration_id", "arm"]
    columns = ["display_name", "experiment_group", "changed_factor", "is_final"]
    metadata = frame.loc[:, [*keys, *columns]].drop_duplicates().reset_index(drop=True)
    if metadata.duplicated(keys).any():
        raise ValueError("presentation metadata changed within one committed arm")
    return metadata


def _table3(family: pd.DataFrame, equal11: pd.DataFrame) -> pd.DataFrame:
    keys = ["stage_id", "stage_group", "configuration_id", "arm", "metric"]
    pivot = family.pivot_table(index=keys, columns="family", values="value", aggfunc="first")
    equal = equal11.loc[:, [*keys, "value", "n_valid", "n_total"]].rename(
        columns={"value": "equal11", "n_valid": "equal11_n", "n_total": "equal11_total"}
    )
    frame = pivot.reset_index().merge(equal, on=keys, how="outer", validate="one_to_one")
    presentation = _presentation_metadata(family)
    arm_keys = ["stage_id", "stage_group", "configuration_id", "arm"]
    frame = frame.merge(presentation, on=arm_keys, how="left", validate="many_to_one")
    for family_name in FAMILIES:
        if family_name not in frame:
            frame[family_name] = np.nan
    return frame.loc[
        :, [*keys, "display_name", "experiment_group", "changed_factor", "is_final",
            *FAMILIES, "equal11", "equal11_n", "equal11_total"]
    ]


def _long11_wide(subgroup: pd.DataFrame, equal11: pd.DataFrame) -> pd.DataFrame:
    keys = ["stage_id", "stage_group", "configuration_id", "arm", "metric"]
    wide = subgroup.pivot_table(index=keys, columns="subgroup", values="value", aggfunc="first")
    wide = wide.rename(columns=SUBGROUP_DISPLAY).reset_index()
    equal = equal11.loc[:, [*keys, "value", "n_valid", "n_total"]].rename(
        columns={"value": "equal11", "n_valid": "n_valid", "n_total": "n_total"}
    )
    frame = wide.merge(equal, on=keys, how="outer", validate="one_to_one")
    presentation = _presentation_metadata(subgroup)
    arm_keys = ["stage_id", "stage_group", "configuration_id", "arm"]
    frame = frame.merge(presentation, on=arm_keys, how="left", validate="many_to_one")
    columns = [
        *keys, "display_name", "experiment_group", "changed_factor", "is_final",
        *SUBGROUP_DISPLAY.values(), "equal11", "n_valid", "n_total",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame.loc[:, columns]


def _recipe_frames(plot: pd.DataFrame) -> dict[str, pd.DataFrame]:
    source = plot.loc[plot["view"].isin(("family3", "equal11"))].copy()
    source["group"] = np.where(source["view"] == "equal11", "equal11", source["family"])
    recipes: dict[str, tuple[pd.Series, Mapping[str, str]]] = {
        "backbone_accuracy_time": (
            source["experiment_group"].eq("backbone_sensitivity"),
            {"elapsed_seconds_per_series": "elapsed_seconds_per_series"},
        ),
        "window_sensitivity": (source["experiment_group"].eq("window_sensitivity"), {}),
        "stride_sensitivity": (source["experiment_group"].eq("stride_sensitivity"), {}),
        "ihp_nctp_interaction": (source["experiment_group"].isin(RECIPE_GROUPS["ihp_nctp_interaction"]), {}),
        "matching_scope": (
            source["experiment_group"].isin(RECIPE_GROUPS["matching_scope"]),
            {"matching_scope": "scope"},
        ),
        "scale_subset_heatmap": (source["experiment_group"].isin(RECIPE_GROUPS["scale_subset_heatmap"]), {}),
        "reducer_sensitivity": (source["experiment_group"].isin(RECIPE_GROUPS["reducer_sensitivity"]), {}),
        "line_vs_spectrogram": (source["experiment_group"].eq("representation_sensitivity"), {}),
    }
    outputs: dict[str, pd.DataFrame] = {}
    common = ["stage_id", "configuration_id", "arm", "group", "metric", "value", "n_valid", "n_total"]
    extra_by_name = {
        "backbone_accuracy_time": ["backbone", "elapsed_seconds_per_series"],
        "window_sensitivity": ["window"],
        "stride_sensitivity": ["stride"],
        "ihp_nctp_interaction": ["ihp", "nctp"],
        "matching_scope": ["matching_scope"],
        "scale_subset_heatmap": ["scale_subset"],
        "reducer_sensitivity": ["reducer_family", "reducer_setting"],
        "line_vs_spectrogram": ["representation"],
    }
    for name, (mask, rename) in recipes.items():
        frame = source.loc[mask, [*common, *extra_by_name[name]]].copy()
        frame = frame.rename(columns=dict(rename))
        frame.insert(0, "recipe", name)
        outputs[name] = frame
    return outputs


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def assemble_cross_stage_outputs(index_path: Path, output_root: Path) -> tuple[Path, ...]:
    """Verify all committed stages and write cross-stage numeric/plot tables."""

    protocol_sha, stages = load_stage_index(index_path)
    status = stage_status_frame(stages)
    complete = [stage for stage in stages if stage.status == "COMPLETE"]
    bundles = [_aggregate_stage(_load_complete_stage(stage)[0]) for stage in complete]
    if not bundles:
        raise ValueError("no complete metric stage is available for assembly")
    combined = {
        name: pd.concat([bundle[name] for bundle in bundles], ignore_index=True, sort=False)
        for name in bundles[0]
    }
    root = Path(output_root)
    outputs: list[Path] = []
    anomaly_columns = [
        "stage_id", "stage_group", "configuration_id", "series_id", "family",
        "subgroup", "arm", "n_points", "n_positive", *ANOMALY_FREE_FIELDS,
    ]
    per_series = combined["per_series"]
    if "n_positive" in per_series:
        anomaly_free = per_series.loc[
            pd.to_numeric(per_series["n_positive"], errors="coerce").eq(0)
        ].copy()
    else:
        anomaly_free = pd.DataFrame()
    for column in anomaly_columns:
        if column not in anomaly_free:
            anomaly_free[column] = np.nan
    anomaly_free = anomaly_free.loc[:, anomaly_columns]
    score_index_columns = [
        "stage_id", "stage_group", "configuration_id", "series_id", "arm",
        "score_sha256", "score_manifest_sha256", "data_sha256",
    ]
    score_index = per_series.copy()
    for column in score_index_columns:
        if column not in score_index:
            score_index[column] = np.nan
    score_index = score_index.loc[:, score_index_columns]
    tables = {
        "results/per_series_metrics_all_stages.csv": combined["per_series"],
        "results/per_series_scores_index.csv": score_index,
        "results/subgroup11_metrics_all_stages.csv": combined["subgroup11"],
        "results/family3_metrics_all_stages.csv": combined["family3"],
        "results/equal11_metrics_all_stages.csv": combined["equal11"],
        "results/fileweighted_metrics_all_stages.csv": combined["fileweighted"],
        "results/anomaly_free_fp_burden.csv": anomaly_free,
        "tables/table3_style_all_stages.csv": _table3(combined["family3"], combined["equal11"]),
        "tables/table_long11_all_stages.csv": _long11_wide(combined["subgroup11"], combined["equal11"]),
        "plot_data/metrics_all_stages_tidy.csv": combined["plot_tidy"],
        "manifests/stage_status.csv": status,
    }
    recipe_frames = _recipe_frames(combined["plot_tidy"])
    tables["tables/table_component_2x2_all_metrics.csv"] = recipe_frames["ihp_nctp_interaction"]
    for name, frame in recipe_frames.items():
        tables[f"plot_data/{name}.csv"] = frame
    hashes: dict[str, str] = {}
    for relative, frame in tables.items():
        path = root / relative
        _atomic_csv(path, frame)
        outputs.append(path)
        hashes[relative] = sha256_file(path)
    blocked = status.loc[status["status"] != "COMPLETE"]
    marker = root / "manifests" / (
        "_CROSS_STAGE_BLOCKED.json" if not blocked.empty else "_CROSS_STAGE_COMPLETE.json"
    )
    _atomic_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_sha256": protocol_sha,
            "stage_index_sha256": sha256_file(Path(index_path)),
            "complete_stage_count": len(complete),
            "blocked_stage_count": int(len(blocked)),
            "status": "BLOCKED" if not blocked.empty else "COMPLETE",
            "output_sha256": hashes,
        },
    )
    outputs.append(marker)
    return tuple(outputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble SHA-bound ViTTrace-v3 evaluator stages into compact tables."
    )
    parser.add_argument("--stage-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    outputs = assemble_cross_stage_outputs(arguments.stage_index, arguments.output_root)
    blocked = any(path.name == "_CROSS_STAGE_BLOCKED.json" for path in outputs)
    print(
        json.dumps(
            {
                "status": "BLOCKED" if blocked else "COMPLETE",
                "output_root": str(arguments.output_root.resolve()),
                "output_count": len(outputs),
            },
            sort_keys=True,
        )
    )
    return 2 if blocked else 0


__all__ = [
    "ANOMALY_FREE_FIELDS",
    "ARM_METADATA_COLUMNS",
    "FAMILIES",
    "MANDATORY_GROUP_COUNTS",
    "METRICS",
    "RECIPE_GROUPS",
    "SCHEMA_VERSION",
    "SUBGROUPS",
    "StageEvaluation",
    "assemble_cross_stage_outputs",
    "load_stage_index",
    "main",
    "sha256_file",
    "stage_status_frame",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
