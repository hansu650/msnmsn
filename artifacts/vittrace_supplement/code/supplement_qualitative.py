"""Data-only qualitative supplement from committed ViTTrace v3 scores.

This module never invokes an encoder or scorer.  It verifies the completed
delayed-label evaluation, indexes the four frozen IHP x NCTP score artifacts,
selects two deterministic representative cases from the evaluator-only
FULL-minus-REL VUS-PR contrast, and exports tidy numeric data.  F1 thresholds
are copied from evaluator output and are explicitly marked as oracle values
for visualization only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from measure_vit4ts.full_manifest import FullSeriesRecord, load_manifest
from measure_vit4ts.metrics import flags_to_intervals

from .metrics import valid_mask_sha256


SCHEMA_VERSION = 2
FACTORIAL_ARMS: tuple[tuple[str, str], ...] = (
    ("rel", "IHP0_NCTP0"),
    ("ihp", "IHP1_NCTP0"),
    ("rel_nctp", "IHP0_NCTP1"),
    ("full", "IHP1_NCTP1"),
)
ARM_IDS = tuple(arm for _, arm in FACTORIAL_ARMS)
BASE_ARM = "IHP0_NCTP0"
FULL_ARM = "IHP1_NCTP1"
MIN_CASE_LENGTH = 480
EXPECTED_COMMON_VALID_SERIES = 488


@dataclass(frozen=True)
class CommittedScore:
    series_id: str
    arm: str
    canonical_arm: str
    score_path: Path
    score_sha256: str
    manifest_path: Path
    manifest_sha256: str
    is_alias: bool
    values: np.ndarray


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_object(path: Path, context: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _digest(value: Any, context: str) -> str:
    text = str(value or "").upper()
    if len(text) != 64 or any(character not in "0123456789ABCDEF" for character in text):
        raise ValueError(f"{context} must be a SHA256 digest")
    return text


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def resolve_committed_score(
    score_root: Path,
    series_id: str,
    arm: str,
    *,
    expected_score_sha256: str,
    expected_manifest_sha256: str,
    expected_length: int,
) -> CommittedScore:
    """Resolve and verify one direct or aliased committed score transaction."""

    expected_score = _digest(expected_score_sha256, "expected score hash")
    expected_manifest = _digest(expected_manifest_sha256, "expected manifest hash")
    series_root = Path(score_root).resolve(strict=True) / str(series_id)
    arm_root = series_root / str(arm)
    success_path = arm_root / "_SUCCESS.json"
    if not success_path.is_file():
        raise FileNotFoundError(f"committed success marker missing: {series_id}/{arm}")
    success = _read_object(success_path, "score success marker")
    if success.get("series_id") != series_id or success.get("arm") != arm:
        raise ValueError(f"score success identity mismatch: {series_id}/{arm}")

    direct_manifest = arm_root / "score_manifest.json"
    alias_manifest = arm_root / "alias_manifest.json"
    if direct_manifest.is_file() == alias_manifest.is_file():
        raise ValueError(f"score transaction must be exactly direct or alias: {series_id}/{arm}")

    if direct_manifest.is_file():
        manifest_path = direct_manifest
        manifest = _read_object(manifest_path, "score manifest")
        score_path = arm_root / "score.npy"
        canonical_arm = arm
        declared_score = _digest(manifest.get("score_sha256"), "declared score hash")
        is_alias = False
        if success.get("score_sha256") and str(success["score_sha256"]).upper() != declared_score:
            raise ValueError(f"score success hash mismatch: {series_id}/{arm}")
    else:
        manifest_path = alias_manifest
        manifest = _read_object(manifest_path, "alias manifest")
        canonical_arm = str(manifest.get("canonical_arm", ""))
        relative_score = str(manifest.get("canonical_score_path", ""))
        if not canonical_arm or not relative_score:
            raise ValueError(f"alias target is incomplete: {series_id}/{arm}")
        score_path = (arm_root / relative_score).resolve(strict=True)
        if not _inside(score_path, series_root):
            raise ValueError(f"alias score escapes its series transaction: {series_id}/{arm}")
        declared_score = _digest(
            manifest.get("canonical_score_sha256"), "declared canonical score hash"
        )
        if success.get("canonical_arm") and success.get("canonical_arm") != canonical_arm:
            raise ValueError(f"alias success target mismatch: {series_id}/{arm}")
        is_alias = True

    if manifest.get("series_id") != series_id or manifest.get("arm") != arm:
        raise ValueError(f"score manifest identity mismatch: {series_id}/{arm}")
    actual_manifest = sha256_file(manifest_path)
    if actual_manifest != expected_manifest:
        raise ValueError(f"committed score manifest hash mismatch: {series_id}/{arm}")
    actual_score = sha256_file(score_path)
    if actual_score != declared_score or actual_score != expected_score:
        raise ValueError(f"committed score hash mismatch: {series_id}/{arm}")

    values = np.load(score_path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float64 or values.shape != (int(expected_length),):
        raise ValueError(f"committed score shape/dtype mismatch: {series_id}/{arm}")
    for start in range(0, values.size, 1 << 18):
        if not np.isfinite(values[start : start + (1 << 18)]).all():
            raise ValueError(f"committed score contains non-finite values: {series_id}/{arm}")
    materialized = np.ascontiguousarray(values, dtype=np.float64)
    return CommittedScore(
        str(series_id),
        str(arm),
        canonical_arm,
        score_path,
        actual_score,
        manifest_path.resolve(),
        actual_manifest,
        is_alias,
        materialized,
    )


def common_valid_series(
    valid_mask: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> pd.DataFrame:
    required = {
        "series_id",
        "family",
        "subgroup",
        "n_points",
        "n_positive",
        "valid_f1_max",
        "valid_auprc",
        "valid_vus_pr",
    }
    missing = required - set(valid_mask)
    if missing:
        raise ValueError(f"valid mask lacks columns: {sorted(missing)}")
    if valid_mask.empty or valid_mask.duplicated("series_id").any():
        raise ValueError("valid mask must contain unique nonempty series IDs")
    frame = valid_mask.copy()
    for column in ("valid_f1_max", "valid_auprc", "valid_vus_pr"):
        if not frame[column].isin((True, False)).all():
            raise ValueError(f"{column} must be Boolean")
    common = frame.loc[
        frame[["valid_f1_max", "valid_auprc", "valid_vus_pr"]].all(axis=1)
    ].copy()
    common["n_points"] = pd.to_numeric(common["n_points"], errors="raise").astype(int)
    common["n_positive"] = pd.to_numeric(common["n_positive"], errors="raise").astype(int)
    if common.empty or (common["n_positive"] <= 0).any() or (
        common["n_positive"] >= common["n_points"]
    ).any():
        raise ValueError("common-valid series must contain both label classes")
    if expected_count is not None and len(common) != int(expected_count):
        raise ValueError(
            "common-valid cohort size mismatch: "
            f"expected {int(expected_count)}, found {len(common)}"
        )
    return common.sort_values("series_id").reset_index(drop=True)


def candidate_ranking_frame(
    metrics: pd.DataFrame,
    valid_mask: pd.DataFrame,
    *,
    min_length: int = MIN_CASE_LENGTH,
) -> pd.DataFrame:
    """Build auditable FULL-minus-REL ranking evidence for every valid series."""

    required = {"series_id", "arm", "vus_pr", "n_points", "n_positive"}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"metrics lack columns: {sorted(missing)}")
    common = common_valid_series(valid_mask)
    common_ids = set(common["series_id"].astype(str))
    subset = metrics.loc[
        metrics["arm"].astype(str).isin(ARM_IDS)
        & metrics["series_id"].astype(str).isin(common_ids),
        ["series_id", "arm", "vus_pr", "n_points", "n_positive"],
    ].copy()
    subset["series_id"] = subset["series_id"].astype(str)
    subset["arm"] = subset["arm"].astype(str)
    expected_rows = len(common) * len(ARM_IDS)
    if len(subset) != expected_rows or subset.duplicated(["series_id", "arm"]).any():
        raise ValueError("factorial metric grid is incomplete or duplicated")
    if set(subset["series_id"]) != common_ids:
        raise ValueError("factorial metrics and common-valid mask disagree")
    subset["vus_pr"] = pd.to_numeric(subset["vus_pr"], errors="raise")
    subset["n_points"] = pd.to_numeric(subset["n_points"], errors="raise").astype(int)
    subset["n_positive"] = pd.to_numeric(
        subset["n_positive"], errors="raise"
    ).astype(int)
    if not np.isfinite(subset["vus_pr"]).all():
        raise ValueError("FULL-minus-REL selection requires defined VUS-PR")

    counts = subset.groupby("series_id", sort=False)[["n_points", "n_positive"]].agg(
        ["min", "max"]
    )
    if (counts[("n_points", "min")] != counts[("n_points", "max")]).any() or (
        counts[("n_positive", "min")] != counts[("n_positive", "max")]
    ).any():
        raise ValueError("factorial metric rows disagree on series label counts")
    common_counts = common.set_index("series_id")[["n_points", "n_positive"]].astype(int)
    metric_counts = counts.xs("min", axis=1, level=1).loc[common_counts.index]
    if not metric_counts.equals(common_counts):
        raise ValueError("factorial metric counts differ from the committed valid mask")

    pivot = subset.pivot(index="series_id", columns="arm", values="vus_pr")
    if set(pivot.columns) != set(ARM_IDS):
        raise ValueError("factorial arms differ from the frozen four-cell registry")
    ranking = common.set_index("series_id").join(
        pivot[[BASE_ARM, FULL_ARM]], how="inner", validate="one_to_one"
    )
    ranking["rel_vus_pr"] = ranking[BASE_ARM]
    ranking["full_vus_pr"] = ranking[FULL_ARM]
    ranking["delta_vus_pr"] = ranking["full_vus_pr"] - ranking["rel_vus_pr"]
    ranking["eligible"] = (
        (ranking["n_points"] >= int(min_length)) & (ranking["n_positive"] > 0)
    )
    ranking["positive_pool"] = ranking["eligible"] & (ranking["delta_vus_pr"] > 0.0)
    ranking["nonpositive_pool"] = ranking["eligible"] & (
        ranking["delta_vus_pr"] <= 0.0
    )

    positive_values = ranking.loc[ranking["positive_pool"], "delta_vus_pr"].to_numpy(
        dtype=np.float64
    )
    nonpositive_values = ranking.loc[
        ranking["nonpositive_pool"], "delta_vus_pr"
    ].to_numpy(dtype=np.float64)
    positive_target = float(np.median(positive_values)) if positive_values.size else np.nan
    nonpositive_target = (
        float(np.median(nonpositive_values)) if nonpositive_values.size else np.nan
    )
    ranking["positive_median_target"] = positive_target
    ranking["positive_distance"] = np.where(
        ranking["positive_pool"],
        np.abs(ranking["delta_vus_pr"] - positive_target),
        np.nan,
    )
    ranking["nonpositive_median_target"] = nonpositive_target
    ranking["nonpositive_distance"] = np.where(
        ranking["nonpositive_pool"],
        np.abs(ranking["delta_vus_pr"] - nonpositive_target),
        np.nan,
    )
    ranking = ranking.reset_index()

    ascending = ranking.sort_values(
        ["delta_vus_pr", "series_id"], kind="mergesort"
    ).index
    descending = ranking.sort_values(
        ["delta_vus_pr", "series_id"], ascending=[False, True], kind="mergesort"
    ).index
    ranking["delta_rank_ascending"] = pd.Series(
        np.arange(1, len(ranking) + 1, dtype=np.int64), index=ascending
    )
    ranking["delta_rank_descending"] = pd.Series(
        np.arange(1, len(ranking) + 1, dtype=np.int64), index=descending
    )
    positive_order = ranking.loc[ranking["positive_pool"]].sort_values(
        ["positive_distance", "series_id"], kind="mergesort"
    ).index
    nonpositive_order = ranking.loc[ranking["nonpositive_pool"]].sort_values(
        ["nonpositive_distance", "series_id"], kind="mergesort"
    ).index
    ranking["positive_selection_rank"] = np.nan
    ranking["nonpositive_selection_rank"] = np.nan
    ranking.loc[positive_order, "positive_selection_rank"] = np.arange(
        1, len(positive_order) + 1, dtype=np.int64
    )
    ranking.loc[nonpositive_order, "nonpositive_selection_rank"] = np.arange(
        1, len(nonpositive_order) + 1, dtype=np.int64
    )
    columns = (
        "series_id",
        "family",
        "subgroup",
        "n_points",
        "n_positive",
        "rel_vus_pr",
        "full_vus_pr",
        "delta_vus_pr",
        "eligible",
        "positive_pool",
        "nonpositive_pool",
        "positive_median_target",
        "positive_distance",
        "nonpositive_median_target",
        "nonpositive_distance",
        "delta_rank_ascending",
        "delta_rank_descending",
        "positive_selection_rank",
        "nonpositive_selection_rank",
    )
    return ranking.loc[:, columns].sort_values("series_id").reset_index(drop=True)


def select_representative_cases(
    metrics: pd.DataFrame,
    valid_mask: pd.DataFrame,
    *,
    min_length: int = MIN_CASE_LENGTH,
) -> pd.DataFrame:
    """Select median-positive and median-nonpositive FULL-minus-REL cases."""

    ranking = candidate_ranking_frame(metrics, valid_mask, min_length=min_length)
    eligible = ranking.loc[ranking["eligible"]].copy()
    if eligible.empty:
        raise ValueError("no common-valid positive-label series satisfies the length floor")

    positive = eligible.loc[eligible["positive_pool"]].copy()
    if positive.empty:
        raise ValueError("no positive FULL-minus-REL VUS-PR case is available")
    positive_target = float(positive["positive_median_target"].iloc[0])
    selected_positive = positive.sort_values(
        ["positive_distance", "series_id"], kind="mergesort"
    ).iloc[0]

    remaining = eligible.loc[eligible["series_id"] != selected_positive["series_id"]].copy()
    nonpositive = remaining.loc[remaining["nonpositive_pool"]].copy()
    if not nonpositive.empty:
        failure_target = float(nonpositive["nonpositive_median_target"].iloc[0])
        selected_failure = nonpositive.sort_values(
            ["nonpositive_distance", "series_id"], kind="mergesort"
        ).iloc[0]
        failure_rule = "closest_to_median_nonpositive_full_minus_rel_vus_pr"
    else:
        if remaining.empty:
            raise ValueError("a distinct failure/fallback case is unavailable")
        failure_target = float(remaining["delta_vus_pr"].min())
        selected_failure = remaining.sort_values(
            ["delta_vus_pr", "series_id"], kind="mergesort"
        ).iloc[0]
        failure_rule = "global_minimum_fallback_no_nonpositive"

    rows: list[dict[str, Any]] = []
    for order, role, row, target, rule in (
        (
            0,
            "representative_positive",
            selected_positive,
            positive_target,
            "closest_to_median_positive_full_minus_rel_vus_pr",
        ),
        (1, "representative_failure", selected_failure, failure_target, failure_rule),
    ):
        rows.append(
            {
                "case_order": order,
                "case_role": role,
                "series_id": str(row["series_id"]),
                "family": str(row["family"]),
                "subgroup": str(row["subgroup"]),
                "n_points": int(row["n_points"]),
                "n_positive": int(row["n_positive"]),
                "rel_vus_pr": float(row["rel_vus_pr"]),
                "full_vus_pr": float(row["full_vus_pr"]),
                "delta_vus_pr": float(row["delta_vus_pr"]),
                "selection_target": float(target),
                "selection_distance": float(abs(float(row["delta_vus_pr"]) - target)),
                "selection_rule": rule,
                "uses_evaluator_labels": True,
                "oracle_visualization_only": True,
            }
        )
    result = pd.DataFrame(rows)
    if result["series_id"].duplicated().any():
        raise RuntimeError("qualitative selections must be distinct")
    return result


def _prediction_data(
    score: np.ndarray,
    timestamps: np.ndarray,
    *,
    alpha: Any,
    threshold: Any,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], bool]:
    alpha_value = pd.to_numeric(pd.Series([alpha]), errors="coerce").iloc[0]
    threshold_value = pd.to_numeric(pd.Series([threshold]), errors="coerce").iloc[0]
    alpha_defined = bool(np.isfinite(alpha_value))
    threshold_defined = bool(np.isfinite(threshold_value))
    if alpha_defined != threshold_defined:
        raise ValueError("evaluator F1 alpha/threshold metadata are partially missing")
    if not alpha_defined:
        return (
            np.full(score.shape, np.nan, dtype=np.float64),
            np.full(score.shape, -1, dtype=np.int64),
            [],
            False,
        )
    if not 0.0 < float(alpha_value) < 1.0:
        raise ValueError("evaluator F1 alpha must lie in (0,1)")
    span = max(1, int(score.size * 0.01))
    smooth = pd.Series(score).ewm(span=span).mean().to_numpy(dtype=np.float64)
    flags = smooth > float(threshold_value)
    interval_ids = np.full(score.shape, -1, dtype=np.int64)
    changes = np.diff(np.pad(flags.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    intervals = flags_to_intervals(flags, timestamps)
    rows: list[dict[str, Any]] = []
    for interval_id, (start, end, bounds) in enumerate(zip(starts, ends, intervals)):
        interval_ids[start : end + 1] = interval_id
        rows.append(
            {
                "interval_id": int(interval_id),
                "start_index": int(start),
                "end_index": int(end),
                "start_time": float(bounds[0]),
                "end_time": float(bounds[1]),
            }
        )
    return smooth, interval_ids, rows, True


SeriesLoader = Callable[[FullSeriesRecord], tuple[np.ndarray, np.ndarray, np.ndarray]]


def build_case_tables(
    cases: pd.DataFrame,
    records: Mapping[str, FullSeriesRecord],
    valid_mask: pd.DataFrame,
    metrics: pd.DataFrame,
    committed: Mapping[tuple[str, str], CommittedScore],
    series_loader: SeriesLoader,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build pointwise data plus prediction and ground-truth intervals."""

    mask_by_id = valid_mask.set_index("series_id", verify_integrity=True)
    metric_by_key = metrics.set_index(["series_id", "arm"], verify_integrity=True)
    tidy_frames: list[pd.DataFrame] = []
    interval_rows: list[dict[str, Any]] = []
    truth_interval_rows: list[dict[str, Any]] = []
    for case in cases.itertuples(index=False):
        series_id = str(case.series_id)
        record = records[series_id]
        timestamps, raw, labels = series_loader(record)
        time = np.asarray(timestamps)
        signal = np.asarray(raw, dtype=np.float64)
        truth = np.asarray(labels)
        expected_shape = (record.expected_length,)
        if time.shape != expected_shape or signal.shape != expected_shape or truth.shape != expected_shape:
            raise ValueError(f"raw/time/label length mismatch: {series_id}")
        if not np.isfinite(time).all() or not np.isfinite(signal).all():
            raise ValueError(f"raw/time values must be finite: {series_id}")
        if not np.logical_or(truth == 0, truth == 1).all():
            raise ValueError(f"labels must be binary: {series_id}")
        mask_row = mask_by_id.loc[series_id]
        if int(truth.sum()) != int(mask_row["n_positive"]):
            raise ValueError(f"label count differs from the committed valid mask: {series_id}")
        truth_flags = truth.astype(bool)
        truth_changes = np.diff(np.pad(truth_flags.astype(np.int8), (1, 1)))
        truth_starts = np.flatnonzero(truth_changes == 1)
        truth_ends = np.flatnonzero(truth_changes == -1) - 1
        truth_bounds = flags_to_intervals(truth_flags, time)
        for interval_id, (start, end, bounds) in enumerate(
            zip(truth_starts, truth_ends, truth_bounds)
        ):
            truth_interval_rows.append(
                {
                    "case_order": int(case.case_order),
                    "case_role": str(case.case_role),
                    "series_id": series_id,
                    "interval_id": int(interval_id),
                    "start_index": int(start),
                    "end_index": int(end),
                    "start_time": float(bounds[0]),
                    "end_time": float(bounds[1]),
                    "uses_evaluator_labels": True,
                    "oracle_visualization_only": False,
                }
            )
        frame = pd.DataFrame(
            {
                "case_order": int(case.case_order),
                "case_role": str(case.case_role),
                "series_id": series_id,
                "family": record.track,
                "subgroup": record.paper_group,
                "dataset": record.dataset,
                "signal_name": record.signal_name,
                "point_index": np.arange(record.expected_length, dtype=np.int64),
                "time": time,
                "raw_value": signal,
                "label": truth.astype(np.uint8),
                "valid_f1_max": bool(mask_row["valid_f1_max"]),
                "valid_auprc": bool(mask_row["valid_auprc"]),
                "valid_vus_pr": bool(mask_row["valid_vus_pr"]),
                "oracle_visualization_only": True,
            }
        )
        for panel, arm in FACTORIAL_ARMS:
            artifact = committed[(series_id, arm)]
            metric = metric_by_key.loc[(series_id, arm)]
            alpha = metric.get("f1_winning_alpha", np.nan)
            threshold = metric.get("f1_threshold_evaluator_only", np.nan)
            smooth, interval_ids, intervals, available = _prediction_data(
                artifact.values, time, alpha=alpha, threshold=threshold
            )
            frame[f"{panel}_score"] = artifact.values
            frame[f"{panel}_smoothed_score"] = smooth
            frame[f"{panel}_score_sha256"] = artifact.score_sha256
            frame[f"{panel}_f1_winning_alpha"] = float(alpha) if available else np.nan
            frame[f"{panel}_f1_threshold_evaluator_only"] = (
                float(threshold) if available else np.nan
            )
            frame[f"{panel}_prediction_interval_id"] = interval_ids
            frame[f"{panel}_prediction_metadata_available"] = available
            for interval in intervals:
                interval_rows.append(
                    {
                        "case_order": int(case.case_order),
                        "case_role": str(case.case_role),
                        "series_id": series_id,
                        "panel": panel,
                        "arm": arm,
                        "score_sha256": artifact.score_sha256,
                        "f1_winning_alpha": float(alpha),
                        "f1_threshold_evaluator_only": float(threshold),
                        "oracle_visualization_only": True,
                        **interval,
                    }
                )
        tidy_frames.append(frame)
    tidy = pd.concat(tidy_frames, ignore_index=True)
    interval_columns = (
        "case_order",
        "case_role",
        "series_id",
        "panel",
        "arm",
        "score_sha256",
        "f1_winning_alpha",
        "f1_threshold_evaluator_only",
        "oracle_visualization_only",
        "interval_id",
        "start_index",
        "end_index",
        "start_time",
        "end_time",
    )
    intervals = pd.DataFrame(interval_rows, columns=interval_columns)
    truth_interval_columns = (
        "case_order",
        "case_role",
        "series_id",
        "interval_id",
        "start_index",
        "end_index",
        "start_time",
        "end_time",
        "uses_evaluator_labels",
        "oracle_visualization_only",
    )
    truth_intervals = pd.DataFrame(truth_interval_rows, columns=truth_interval_columns)
    return tidy, intervals, truth_intervals


def _load_evaluation(evaluation_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    root = Path(evaluation_root).resolve(strict=True)
    marker_path = root / "_COMBINED_EVALUATION_COMPLETE.json"
    metrics_path = root / "per_series_metrics.csv"
    mask_path = root / "valid_series_mask.csv"
    provenance_path = root / "evaluation_provenance.json"
    marker = _read_object(marker_path, "combined evaluation marker")
    if marker.get("status") != "COMPLETE":
        raise ValueError("combined evaluation is not COMPLETE")
    checks = (
        (metrics_path, marker.get("per_series_metrics_sha256"), "per-series metrics"),
        (mask_path, marker.get("valid_series_mask_file_sha256"), "valid mask"),
        (provenance_path, marker.get("evaluation_provenance_sha256"), "evaluation provenance"),
    )
    for path, expected, context in checks:
        if sha256_file(path) != _digest(expected, f"{context} marker hash"):
            raise ValueError(f"{context} hash mismatch")
    provenance = _read_object(provenance_path, "evaluation provenance")
    metrics = pd.read_csv(metrics_path)
    valid_mask = pd.read_csv(mask_path)
    semantic_mask_hash = valid_mask_sha256(valid_mask)
    if semantic_mask_hash != _digest(provenance.get("valid_mask_sha256"), "valid mask semantic hash"):
        raise ValueError("valid mask semantic identity mismatch")
    return metrics, valid_mask, {
        "marker_path": str(marker_path),
        "marker_sha256": sha256_file(marker_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "valid_mask_path": str(mask_path),
        "valid_mask_file_sha256": sha256_file(mask_path),
        "valid_mask_sha256": semantic_mask_hash,
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "config_sha256": _digest(provenance.get("config_sha256"), "config hash"),
        "manifest_sha256": _digest(provenance.get("manifest_sha256"), "manifest hash"),
    }


def export_supplement_qualitative(
    config_path: Path,
    evaluation_root: Path,
    score_root: Path,
    output_root: Path,
    *,
    series_loader: SeriesLoader | None = None,
) -> tuple[Path, ...]:
    """Export the complete data-only supplement transaction."""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("supplement qualitative output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    metrics, valid_mask, evaluation_identity = _load_evaluation(evaluation_root)
    config_target = Path(config_path).resolve(strict=True)
    config_sha = sha256_file(config_target)
    if config_sha != evaluation_identity["config_sha256"]:
        raise ValueError("supplement config differs from evaluated config")
    config = yaml.safe_load(config_target.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("stage") != "vittrace_ablation_full_v3":
        raise ValueError("supplement accepts only the frozen v3 config")
    manifest_path = Path(config["manifest"]["path"]).resolve(strict=True)
    if sha256_file(manifest_path) != evaluation_identity["manifest_sha256"]:
        raise ValueError("supplement manifest differs from evaluated manifest")
    _, records_tuple = load_manifest(manifest_path)
    records = {record.series_id: record for record in records_tuple}
    common = common_valid_series(
        valid_mask, expected_count=EXPECTED_COMMON_VALID_SERIES
    )
    if not set(common["series_id"]).issubset(records):
        raise ValueError("common-valid mask contains an unknown manifest series")

    metric_keys = metrics.set_index(["series_id", "arm"], verify_integrity=True)
    index_rows: list[dict[str, Any]] = []
    committed: dict[tuple[str, str], CommittedScore] = {}
    for row in common.itertuples(index=False):
        record = records[str(row.series_id)]
        if int(row.n_points) != record.expected_length:
            raise ValueError(f"valid-mask length differs from manifest: {record.series_id}")
        for panel, arm in FACTORIAL_ARMS:
            if (record.series_id, arm) not in metric_keys.index:
                raise ValueError(f"factorial metric row missing: {record.series_id}/{arm}")
            metric = metric_keys.loc[(record.series_id, arm)]
            artifact = resolve_committed_score(
                score_root,
                record.series_id,
                arm,
                expected_score_sha256=str(metric["score_sha256"]),
                expected_manifest_sha256=str(metric["score_manifest_sha256"]),
                expected_length=record.expected_length,
            )
            committed[(record.series_id, arm)] = artifact
            index_rows.append(
                {
                    "series_id": record.series_id,
                    "family": record.track,
                    "subgroup": record.paper_group,
                    "panel": panel,
                    "arm": arm,
                    "canonical_arm": artifact.canonical_arm,
                    "is_alias": artifact.is_alias,
                    "score_path": str(artifact.score_path),
                    "score_sha256": artifact.score_sha256,
                    "manifest_path": str(artifact.manifest_path),
                    "manifest_sha256": artifact.manifest_sha256,
                    "score_length": artifact.values.size,
                    "common_valid": True,
                }
            )
    score_index = pd.DataFrame(index_rows).sort_values(["series_id", "panel"])
    expected_index_rows = len(common) * len(FACTORIAL_ARMS)
    if len(score_index) != expected_index_rows or score_index.duplicated(["series_id", "arm"]).any():
        raise RuntimeError("supplement score index is incomplete or duplicated")

    ranking = candidate_ranking_frame(metrics, valid_mask)
    cases = select_representative_cases(metrics, valid_mask)
    if series_loader is None:
        data_root = Path(config["data"]["root"])
        label_config = dict(config)
        label_config["data"] = dict(config["data"])
        label_config["scoring"] = {
            "series": [
                {"series_id": record.series_id, "relative_path": record.relative_path}
                for record in records_tuple
            ]
        }

        def default_loader(record: FullSeriesRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            from measure_vit4ts.coordinate_envelope_runner import load_vendor_signal
            from measure_vit4ts.evaluator import load_ground_truth

            data_path = data_root / record.relative_path
            if sha256_file(data_path) != record.expected_sha256.upper():
                raise ValueError(f"source data hash mismatch: {record.series_id}")
            loaded = load_vendor_signal(record, data_root)
            time = np.asarray(loaded.series.timestamps)
            raw = np.asarray(loaded.series.values, dtype=np.float64)
            truth = load_ground_truth(label_config, record.series_id, time)
            return time, raw, np.asarray(truth.point_labels, dtype=np.uint8)

        loader = default_loader
    else:
        loader = series_loader
    tidy, intervals, truth_intervals = build_case_tables(
        cases, records, valid_mask, metrics, committed, loader
    )

    score_index_path = output / "score_index.csv"
    ranking_path = output / "candidate_ranking.csv"
    cases_path = output / "selected_cases.csv"
    tidy_path = output / "case_tidy.csv"
    intervals_path = output / "prediction_intervals.csv"
    truth_intervals_path = output / "ground_truth_intervals.csv"
    _atomic_csv(score_index_path, score_index)
    _atomic_csv(ranking_path, ranking)
    _atomic_csv(cases_path, cases)
    _atomic_csv(tidy_path, tidy)
    _atomic_csv(intervals_path, intervals)
    _atomic_csv(truth_intervals_path, truth_intervals)
    output_hashes = {
        path.name: sha256_file(path)
        for path in (
            score_index_path,
            ranking_path,
            cases_path,
            tidy_path,
            intervals_path,
            truth_intervals_path,
        )
    }
    selection_path = output / "selection_manifest.json"
    _atomic_json(
        selection_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "factorial_arms": [
                {"panel": panel, "arm": arm} for panel, arm in FACTORIAL_ARMS
            ],
            "contrast": f"{FULL_ARM}-{BASE_ARM}",
            "metric": "vus_pr",
            "minimum_length": MIN_CASE_LENGTH,
            "positive_rule": "closest_to_median_positive_full_minus_rel_vus_pr_then_series_id",
            "failure_rule": "closest_to_median_nonpositive_else_global_minimum_then_series_id",
            "uses_evaluator_labels": True,
            "oracle_visualization_only": True,
            "common_valid_series": int(len(common)),
            "score_index_rows": int(len(score_index)),
            "cases": cases.to_dict(orient="records"),
            "config_path": str(config_target),
            "config_sha256": config_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "score_root": str(Path(score_root).resolve(strict=True)),
            "evaluation": dict(evaluation_identity),
            "output_sha256": output_hashes,
        },
    )
    complete_path = output / "_SUPPLEMENT_QUALITATIVE_COMPLETE.json"
    _atomic_json(
        complete_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "selection_manifest_sha256": sha256_file(selection_path),
            "output_sha256": {**output_hashes, selection_path.name: sha256_file(selection_path)},
            "model_or_encoder_calls": 0,
            "figures_emitted": 0,
        },
    )
    return (
        score_index_path,
        ranking_path,
        cases_path,
        tidy_path,
        intervals_path,
        truth_intervals_path,
        selection_path,
        complete_path,
    )


def run_supplement_qualitative(
    config_path: Path,
    evaluation_root: Path,
    score_root: Path,
    output_root: Path,
) -> tuple[Path, ...]:
    try:
        return export_supplement_qualitative(
            config_path, evaluation_root, score_root, output_root
        )
    except Exception as error:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            root / "_SUPPLEMENT_QUALITATIVE_BLOCKED.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "BLOCKED",
                "error_type": type(error).__name__,
                "reason": str(error),
                "model_or_encoder_calls": 0,
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = run_supplement_qualitative(
        args.config, args.evaluation_root, args.score_root, args.output_root
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARM_IDS",
    "BASE_ARM",
    "EXPECTED_COMMON_VALID_SERIES",
    "FACTORIAL_ARMS",
    "FULL_ARM",
    "MIN_CASE_LENGTH",
    "CommittedScore",
    "build_case_tables",
    "candidate_ranking_frame",
    "common_valid_series",
    "export_supplement_qualitative",
    "resolve_committed_score",
    "run_supplement_qualitative",
    "select_representative_cases",
]
