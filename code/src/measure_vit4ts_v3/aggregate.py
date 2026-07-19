"""Corrected-primary v3 aggregation with arm-independent masks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    ALL_METRICS,
    valid_mask_sha256,
    validate_metrics_against_mask,
)
from .registry import ArmRegistry, BOOTSTRAP_SEED, ContrastSpec


@dataclass(frozen=True)
class AggregationBundle:
    per_series: pd.DataFrame
    subgroup11: pd.DataFrame
    family3: pd.DataFrame
    equal11: pd.DataFrame
    fileweighted: pd.DataFrame


def _validate_hierarchy(mask: pd.DataFrame, registry: ArmRegistry) -> pd.DataFrame:
    hierarchy = mask[["series_id", "family", "subgroup"]].copy()
    if hierarchy.duplicated("series_id").any():
        raise ValueError("valid-series mask has duplicate series IDs")
    mapping = hierarchy[["family", "subgroup"]].drop_duplicates()
    if mapping["subgroup"].nunique() != registry.expected_subgroups:
        raise ValueError("v3 aggregation requires exactly 11 subgroups")
    if mapping["family"].nunique() != registry.expected_families:
        raise ValueError("v3 aggregation requires exactly three families")
    if mapping.groupby("subgroup")["family"].nunique().max() != 1:
        raise ValueError("each subgroup must belong to exactly one family")
    return hierarchy


def _subgroup_summary(
    rows: pd.DataFrame, mask: pd.DataFrame, registry: ArmRegistry
) -> pd.DataFrame:
    descriptors = (
        mask[["family", "subgroup"]]
        .drop_duplicates()
        .sort_values(["family", "subgroup"])
        .reset_index(drop=True)
    )
    totals = mask.groupby(["family", "subgroup"])["series_id"].nunique()
    output: list[dict[str, Any]] = []
    for descriptor in descriptors.itertuples(index=False):
        key = (str(descriptor.family), str(descriptor.subgroup))
        subset = rows.loc[
            (rows["family"].astype(str) == key[0])
            & (rows["subgroup"].astype(str) == key[1])
        ]
        for arm in registry.arm_ids:
            arm_rows = subset.loc[subset["arm"].astype(str) == arm]
            for metric in ALL_METRICS:
                valid = arm_rows[f"valid_{metric}"].astype(bool)
                values = arm_rows.loc[valid, metric].to_numpy(dtype=np.float64)
                output.append(
                    {
                        "aggregation": "subgroup11",
                        "family": key[0],
                        "subgroup": key[1],
                        "arm": arm,
                        "metric": metric,
                        "value": float(np.mean(values)) if values.size else np.nan,
                        "n_valid": int(values.size),
                        "n_total": int(totals.loc[key]),
                    }
                )
    return pd.DataFrame(output)


def _family_summary(subgroups: pd.DataFrame, registry: ArmRegistry) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    families = tuple(sorted(subgroups["family"].astype(str).unique()))
    for family in families:
        family_rows = subgroups.loc[subgroups["family"].astype(str) == family]
        total_groups = int(family_rows["subgroup"].nunique())
        for arm in registry.arm_ids:
            for metric in ALL_METRICS:
                selected = family_rows.loc[
                    (family_rows["arm"] == arm) & (family_rows["metric"] == metric),
                    "value",
                ].dropna()
                output.append(
                    {
                        "aggregation": "family3",
                        "family": family,
                        "subgroup": "ALL",
                        "arm": arm,
                        "metric": metric,
                        "value": float(selected.mean()) if len(selected) else np.nan,
                        "n_valid": int(len(selected)),
                        "n_total": total_groups,
                    }
                )
    return pd.DataFrame(output)


def _equal11_summary(subgroups: pd.DataFrame, registry: ArmRegistry) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for arm in registry.arm_ids:
        for metric in ALL_METRICS:
            selected = subgroups.loc[
                (subgroups["arm"] == arm) & (subgroups["metric"] == metric), "value"
            ].dropna()
            output.append(
                {
                    "aggregation": "equal11",
                    "family": "ALL",
                    "subgroup": "ALL",
                    "arm": arm,
                    "metric": metric,
                    "value": float(selected.mean()) if len(selected) else np.nan,
                    "n_valid": int(len(selected)),
                    "n_total": registry.expected_subgroups,
                }
            )
    return pd.DataFrame(output)


def _fileweighted_summary(
    rows: pd.DataFrame, mask: pd.DataFrame, registry: ArmRegistry
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    n_total = int(mask["series_id"].nunique())
    for arm in registry.arm_ids:
        arm_rows = rows.loc[rows["arm"].astype(str) == arm]
        for metric in ALL_METRICS:
            valid = arm_rows[f"valid_{metric}"].astype(bool)
            values = arm_rows.loc[valid, metric].to_numpy(dtype=np.float64)
            output.append(
                {
                    "aggregation": "fileweighted",
                    "family": "ALL",
                    "subgroup": "ALL",
                    "arm": arm,
                    "metric": metric,
                    "value": float(np.mean(values)) if values.size else np.nan,
                    "n_valid": int(values.size),
                    "n_total": n_total,
                }
            )
    return pd.DataFrame(output)


def aggregate_metrics(
    metrics: pd.DataFrame, mask: pd.DataFrame, registry: ArmRegistry
) -> AggregationBundle:
    """Produce the five required views from one pre-fixed series mask."""

    _validate_hierarchy(mask, registry)
    rows = validate_metrics_against_mask(metrics, mask, registry.arm_ids)
    subgroups = _subgroup_summary(rows, mask, registry)
    families = _family_summary(subgroups, registry)
    equal11 = _equal11_summary(subgroups, registry)
    weighted = _fileweighted_summary(rows, mask, registry)
    return AggregationBundle(rows, subgroups, families, equal11, weighted)


def _contrast_arrays(
    paired: pd.DataFrame,
    contrasts: Sequence[ContrastSpec],
) -> tuple[np.ndarray, tuple[str, ...]]:
    arrays = []
    identifiers = []
    for contrast in contrasts:
        arrays.append(
            paired[contrast.candidate].to_numpy(dtype=np.float64)
            - paired[contrast.control].to_numpy(dtype=np.float64)
        )
        identifiers.append(contrast.contrast_id)
    return np.column_stack(arrays), tuple(identifiers)


def paired_hierarchical_bootstrap(
    metrics: pd.DataFrame,
    mask: pd.DataFrame,
    registry: ArmRegistry,
    metric: str,
    *,
    n_boot: int | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap all contrasts with one shared subgroup/series index plan."""

    if metric not in ALL_METRICS:
        raise ValueError(f"unsupported v3 metric: {metric}")
    if int(seed) != BOOTSTRAP_SEED:
        raise ValueError("v3 paired bootstrap seed is frozen to 2027")
    draws_count = registry.bootstrap_replicates if n_boot is None else int(n_boot)
    if draws_count <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    _validate_hierarchy(mask, registry)
    rows = validate_metrics_against_mask(metrics, mask, registry.arm_ids)
    valid_ids = set(
        mask.loc[mask[f"valid_{metric}"].astype(bool), "series_id"].astype(str)
    )
    selected = rows.loc[rows["series_id"].astype(str).isin(valid_ids)]
    pivot = selected.pivot(
        index=["family", "subgroup", "series_id"], columns="arm", values=metric
    )
    if tuple(pivot.columns.astype(str)) != tuple(sorted(registry.arm_ids)):
        # Pandas sorts pivot columns; compare sets while still rejecting holes below.
        if set(pivot.columns.astype(str)) != set(registry.arm_ids):
            raise ValueError("bootstrap pivot does not contain every frozen arm")
    if pivot.isna().any().any():
        raise ValueError("bootstrap requires complete finite paired arm rows")
    groups = tuple(sorted({str(index[1]) for index in pivot.index}))
    if len(groups) != registry.expected_subgroups:
        raise ValueError(
            f"{metric} bootstrap requires at least one pre-fixed valid series in all 11 subgroups"
        )
    group_frames: dict[str, pd.DataFrame] = {}
    group_deltas: dict[str, np.ndarray] = {}
    for group in groups:
        frame = pivot.xs(group, level="subgroup").sort_index(level="series_id")
        group_frames[group] = frame
        group_deltas[group], _ = _contrast_arrays(frame, registry.contrasts)
        if frame.empty:
            raise ValueError("bootstrap found an empty subgroup")

    rng = np.random.default_rng(int(seed))
    n_contrasts = len(registry.contrasts)
    samples = np.empty((draws_count, n_contrasts), dtype=np.float64)
    plan_digest = hashlib.sha256()
    for draw_index in range(draws_count):
        sampled_group_indices = rng.integers(0, len(groups), size=len(groups), dtype=np.int32)
        plan_digest.update(sampled_group_indices.tobytes())
        group_means = np.empty((len(groups), n_contrasts), dtype=np.float64)
        for position, group_index in enumerate(sampled_group_indices):
            group = groups[int(group_index)]
            values = group_deltas[group]
            sampled_series = rng.integers(0, values.shape[0], size=values.shape[0], dtype=np.int32)
            plan_digest.update(sampled_series.tobytes())
            group_means[position] = np.mean(values[sampled_series], axis=0)
        samples[draw_index] = np.mean(group_means, axis=0)

    original_group_means = np.vstack(
        [np.mean(group_deltas[group], axis=0) for group in groups]
    )
    points = np.mean(original_group_means, axis=0)
    mask_digest = valid_mask_sha256(mask)
    shared_digest = plan_digest.hexdigest().upper()
    output: list[dict[str, Any]] = []
    for index, contrast in enumerate(registry.contrasts):
        output.append(
            {
                "contrast_id": contrast.contrast_id,
                "contrast_family": contrast.family,
                "candidate": contrast.candidate,
                "control": contrast.control,
                "metric": metric,
                "delta": float(points[index]),
                "ci_lower": float(np.quantile(samples[:, index], 0.025)),
                "ci_upper": float(np.quantile(samples[:, index], 0.975)),
                "n_boot": draws_count,
                "seed": int(seed),
                "shared_indices": True,
                "resample_plan_sha256": shared_digest,
                "valid_mask_sha256": mask_digest,
                "n_valid_series": len(valid_ids),
                "n_subgroups": len(groups),
                "resampling_unit": "11_subgroups_then_paired_series",
            }
        )
    return pd.DataFrame(output)
