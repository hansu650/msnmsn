"""Data-only factorial statistics for the four IHP/NCTP arms.

The module consumes a hash-bound COMPLETE combined-evaluation transaction,
selects the exact 2x2 factorial arms, and creates one shared hierarchical
subgroup-to-series bootstrap plan for every metric and contrast.  It never
loads score arrays, labels, encoders, or scorers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .combined_protocol import sha256_file
from .metrics import DETECTION_METRICS


SCHEMA_VERSION = 1
EXPECTED_SERIES = 492
EXPECTED_VALID_SERIES = 488
EXPECTED_SUBGROUPS = 11
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2027

FACTORIAL_ARMS = (
    ("REL", 0, 0, "IHP0_NCTP0"),
    ("IHP", 1, 0, "IHP1_NCTP0"),
    ("NCTP", 0, 1, "IHP0_NCTP1"),
    ("FULL", 1, 1, "IHP1_NCTP1"),
)
ARM_IDS = tuple(row[3] for row in FACTORIAL_ARMS)
CONTRASTS = (
    ("IHP_MINUS_REL", "IHP1_NCTP0", "IHP0_NCTP0", "IHP - REL"),
    ("NCTP_MINUS_REL", "IHP0_NCTP1", "IHP0_NCTP0", "NCTP - REL"),
    ("FULL_MINUS_REL", "IHP1_NCTP1", "IHP0_NCTP0", "FULL - REL"),
    ("FULL_MINUS_IHP", "IHP1_NCTP1", "IHP1_NCTP0", "FULL - IHP"),
    ("FULL_MINUS_NCTP", "IHP1_NCTP1", "IHP0_NCTP1", "FULL - NCTP"),
    (
        "FACTORIAL_INTERACTION",
        "IHP1_NCTP1",
        "IHP0_NCTP0",
        "FULL - IHP - NCTP + REL",
    ),
)


@dataclass(frozen=True)
class SupplementInputs:
    metrics: pd.DataFrame
    arm_metadata: pd.DataFrame
    marker: Mapping[str, Any]
    valid_series: tuple[str, ...]


@dataclass(frozen=True)
class SupplementOutputs:
    contrasts: pd.DataFrame
    factorial_summary: pd.DataFrame
    per_series_deltas: pd.DataFrame
    plan_sha256: str


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


def _read_marker(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError("combined evaluation marker must be a JSON object")
    return payload


def _validate_marker(
    marker: Mapping[str, Any], metrics_path: Path, arm_path: Path
) -> None:
    if marker.get("status") != "COMPLETE":
        raise ValueError("supplement statistics require a COMPLETE evaluation")
    if int(marker.get("series_count", -1)) != EXPECTED_SERIES:
        raise ValueError("supplement statistics require exactly 492 series")
    if int(marker.get("valid_series_count", -1)) != EXPECTED_VALID_SERIES:
        raise ValueError("supplement statistics require the common 488-series mask")
    if sha256_file(metrics_path) != str(marker.get("per_series_metrics_sha256", "")).upper():
        raise ValueError("per-series metrics hash differs from COMPLETE marker")
    if sha256_file(arm_path) != str(marker.get("arm_metadata_sha256", "")).upper():
        raise ValueError("arm metadata hash differs from COMPLETE marker")


def _finite_metric_mask(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=np.float64))
    return pd.Series(finite, index=frame.index)


def load_supplement_inputs(evaluation_dir: Path) -> SupplementInputs:
    """Load and validate the immutable four-arm/common-488 metric grid."""

    root = Path(evaluation_dir)
    marker_path = root / "_COMBINED_EVALUATION_COMPLETE.json"
    metrics_path = root / "per_series_metrics.csv"
    arm_path = root / "arm_metadata.csv"
    marker = _read_marker(marker_path)
    _validate_marker(marker, metrics_path, arm_path)
    metrics = pd.read_csv(metrics_path)
    metadata = pd.read_csv(arm_path)
    required_metrics = {"series_id", "family", "subgroup", "arm", *DETECTION_METRICS}
    if not required_metrics.issubset(metrics.columns):
        raise ValueError("per-series metrics miss supplement columns")
    if "arm" not in metadata.columns or metadata["arm"].astype(str).duplicated().any():
        raise ValueError("arm metadata must contain unique arm rows")
    metric_arms = set(metrics["arm"].astype(str))
    metadata_arms = set(metadata["arm"].astype(str))
    if metric_arms != metadata_arms or int(marker.get("arm_count", -1)) != len(metric_arms):
        raise ValueError("metric/metadata/marker arm grids differ")
    if not set(ARM_IDS).issubset(metric_arms):
        raise ValueError("the exact four IHP/NCTP arms are not complete")
    selected = metrics.loc[metrics["arm"].astype(str).isin(ARM_IDS)].copy()
    if len(selected) != EXPECTED_SERIES * len(ARM_IDS):
        raise ValueError("factorial metrics must contain exactly 4 x 492 rows")
    if selected.duplicated(["series_id", "arm"]).any():
        raise ValueError("factorial metrics contain duplicate series/arm rows")
    if selected["series_id"].astype(str).nunique() != EXPECTED_SERIES:
        raise ValueError("factorial metrics differ from the 492-series manifest")

    descriptor_counts = selected.groupby("series_id")[["family", "subgroup"]].nunique()
    if (descriptor_counts != 1).any().any():
        raise ValueError("family/subgroup metadata differ across factorial arms")
    masks: list[set[str]] = []
    for metric in DETECTION_METRICS:
        for arm in ARM_IDS:
            arm_rows = selected.loc[selected["arm"].astype(str) == arm]
            valid = arm_rows.loc[_finite_metric_mask(arm_rows, metric), "series_id"].astype(str)
            values = pd.to_numeric(
                arm_rows.loc[_finite_metric_mask(arm_rows, metric), metric], errors="raise"
            ).to_numpy(dtype=np.float64)
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"{metric} must lie in [0,1]")
            masks.append(set(valid))
    if not masks or any(mask != masks[0] for mask in masks[1:]):
        raise ValueError("factorial arms/metrics do not share one validity mask")
    if len(masks[0]) != EXPECTED_VALID_SERIES:
        raise ValueError("factorial common validity mask must contain 488 series")
    invalid = set(selected["series_id"].astype(str)) - masks[0]
    for metric in DETECTION_METRICS:
        invalid_values = pd.to_numeric(
            selected.loc[selected["series_id"].astype(str).isin(invalid), metric],
            errors="coerce",
        )
        if invalid_values.notna().any():
            raise ValueError("invalid series must remain undefined for every detection metric")
    valid_series = tuple(sorted(masks[0]))
    return SupplementInputs(selected, metadata, marker, valid_series)


def _factorial_cube(inputs: SupplementInputs) -> tuple[pd.DataFrame, np.ndarray]:
    valid = inputs.metrics.loc[
        inputs.metrics["series_id"].astype(str).isin(inputs.valid_series)
    ].copy()
    descriptors = (
        valid[["series_id", "family", "subgroup"]]
        .drop_duplicates()
        .sort_values(["subgroup", "series_id"])
        .reset_index(drop=True)
    )
    if len(descriptors) != EXPECTED_VALID_SERIES:
        raise ValueError("valid descriptor table differs from common 488 mask")
    if descriptors["subgroup"].astype(str).nunique() != EXPECTED_SUBGROUPS:
        raise ValueError("supplement bootstrap requires exactly 11 subgroups")
    arrays: list[np.ndarray] = []
    index: pd.MultiIndex | None = None
    for metric in DETECTION_METRICS:
        pivot = valid.pivot(index="series_id", columns="arm", values=metric)
        pivot = pivot.loc[:, list(ARM_IDS)]
        if pivot.isna().any().any():
            raise ValueError("factorial metric cube contains an incomplete paired row")
        ordered = descriptors["series_id"].astype(str)
        pivot = pivot.loc[ordered]
        arrays.append(pivot.to_numpy(dtype=np.float64))
        index = pivot.index
    if index is None:
        raise RuntimeError("factorial metric cube is empty")
    cube = np.stack(arrays, axis=2)  # series x arms x metrics
    return descriptors, cube


def _contrast_cube(arm_cube: np.ndarray) -> np.ndarray:
    rel, ihp, nctp, full = (arm_cube[:, index, :] for index in range(4))
    return np.stack(
        (
            ihp - rel,
            nctp - rel,
            full - rel,
            full - ihp,
            full - nctp,
            full - ihp - nctp + rel,
        ),
        axis=1,
    )  # series x contrasts x metrics


def _bootstrap_shared(
    descriptors: pd.DataFrame,
    contrast_cube: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray, str]:
    if int(n_boot) <= 0 or int(seed) != BOOTSTRAP_SEED:
        raise ValueError("supplement bootstrap is frozen to a positive seed-2027 plan")
    groups = tuple(sorted(descriptors["subgroup"].astype(str).unique()))
    if len(groups) != EXPECTED_SUBGROUPS:
        raise ValueError("supplement bootstrap requires exactly 11 subgroups")
    group_rows = {
        group: np.flatnonzero(descriptors["subgroup"].astype(str).to_numpy() == group)
        for group in groups
    }
    original = np.stack(
        [contrast_cube[group_rows[group]].mean(axis=0) for group in groups], axis=0
    ).mean(axis=0)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(
        (int(n_boot), contrast_cube.shape[1], contrast_cube.shape[2]), dtype=np.float64
    )
    digest = hashlib.sha256()
    for draw_index in range(int(n_boot)):
        sampled_groups = rng.integers(0, len(groups), size=len(groups), dtype=np.int32)
        digest.update(sampled_groups.tobytes())
        group_means = np.empty(
            (len(groups), contrast_cube.shape[1], contrast_cube.shape[2]), dtype=np.float64
        )
        for position, group_index in enumerate(sampled_groups):
            rows = group_rows[groups[int(group_index)]]
            sampled_series = rng.integers(0, len(rows), size=len(rows), dtype=np.int32)
            digest.update(sampled_series.tobytes())
            group_means[position] = contrast_cube[rows[sampled_series]].mean(axis=0)
        draws[draw_index] = group_means.mean(axis=0)
    return original, draws, digest.hexdigest().upper()


def compute_supplement_stats(
    inputs: SupplementInputs,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
) -> SupplementOutputs:
    """Compute deterministic factorial summaries and one shared-plan bootstrap."""

    descriptors, arm_cube = _factorial_cube(inputs)
    contrast_cube = _contrast_cube(arm_cube)
    points, draws, plan_sha = _bootstrap_shared(
        descriptors, contrast_cube, n_boot=n_boot, seed=BOOTSTRAP_SEED
    )

    contrast_rows: list[dict[str, Any]] = []
    for contrast_index, (identifier, candidate, control, formula) in enumerate(CONTRASTS):
        for metric_index, metric in enumerate(DETECTION_METRICS):
            samples = draws[:, contrast_index, metric_index]
            lower, upper = np.quantile(samples, (0.025, 0.975))
            contrast_rows.append(
                {
                    "contrast_id": identifier,
                    "candidate": candidate,
                    "control": control,
                    "formula": formula,
                    "metric": metric,
                    "point_delta": float(points[contrast_index, metric_index]),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "crosses_zero": bool(lower <= 0.0 <= upper),
                    "proportion_gt_zero": float(np.mean(samples > 0.0)),
                    "effective_n": EXPECTED_VALID_SERIES,
                    "n_subgroups": EXPECTED_SUBGROUPS,
                    "n_boot": int(n_boot),
                    "seed": BOOTSTRAP_SEED,
                    "shared_plan_sha256": plan_sha,
                    "resampling_unit": "11_subgroups_then_paired_series",
                }
            )

    factorial_rows: list[dict[str, Any]] = []
    subgroups = descriptors["subgroup"].astype(str).to_numpy()
    group_names = tuple(sorted(set(subgroups)))
    for arm_index, (display, ihp, nctp, arm) in enumerate(FACTORIAL_ARMS):
        for metric_index, metric in enumerate(DETECTION_METRICS):
            group_means = [
                arm_cube[subgroups == group, arm_index, metric_index].mean()
                for group in group_names
            ]
            factorial_rows.append(
                {
                    "display_name": display,
                    "arm": arm,
                    "ihp": ihp,
                    "nctp": nctp,
                    "metric": metric,
                    "equal11_value": float(np.mean(group_means)),
                    "fileweighted_value": float(arm_cube[:, arm_index, metric_index].mean()),
                    "effective_n": EXPECTED_VALID_SERIES,
                    "n_subgroups": EXPECTED_SUBGROUPS,
                }
            )

    per_series_rows: list[dict[str, Any]] = []
    for series_index, descriptor in enumerate(descriptors.itertuples(index=False)):
        for contrast_index, (identifier, candidate, control, formula) in enumerate(CONTRASTS):
            for metric_index, metric in enumerate(DETECTION_METRICS):
                per_series_rows.append(
                    {
                        "series_id": str(descriptor.series_id),
                        "family": str(descriptor.family),
                        "subgroup": str(descriptor.subgroup),
                        "contrast_id": identifier,
                        "candidate": candidate,
                        "control": control,
                        "formula": formula,
                        "metric": metric,
                        "paired_delta": float(
                            contrast_cube[series_index, contrast_index, metric_index]
                        ),
                    }
                )
    return SupplementOutputs(
        pd.DataFrame(contrast_rows),
        pd.DataFrame(factorial_rows),
        pd.DataFrame(per_series_rows),
        plan_sha,
    )


def write_supplement_outputs(
    evaluation_dir: Path,
    output_dir: Path,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
) -> tuple[Path, ...]:
    """Validate inputs, compute supplement statistics, and commit compact CSVs."""

    inputs = load_supplement_inputs(evaluation_dir)
    outputs = compute_supplement_stats(inputs, n_boot=n_boot)
    root = Path(output_dir)
    paths = (
        root / "supplement_contrasts.csv",
        root / "supplement_factorial_summary.csv",
        root / "supplement_per_series_deltas.csv",
    )
    _atomic_csv(paths[0], outputs.contrasts)
    _atomic_csv(paths[1], outputs.factorial_summary)
    _atomic_csv(paths[2], outputs.per_series_deltas)
    marker = root / "_SUPPLEMENT_STATS_COMPLETE.json"
    _atomic_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "series_count": EXPECTED_SERIES,
            "valid_series_count": EXPECTED_VALID_SERIES,
            "arm_count": len(ARM_IDS),
            "contrast_count": len(CONTRASTS),
            "metric_count": len(DETECTION_METRICS),
            "bootstrap_replicates": int(n_boot),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shared_plan_sha256": outputs.plan_sha256,
            "contrast_rows": len(outputs.contrasts),
            "factorial_rows": len(outputs.factorial_summary),
            "per_series_delta_rows": len(outputs.per_series_deltas),
            "input_metrics_sha256": sha256_file(Path(evaluation_dir) / "per_series_metrics.csv"),
            "input_arm_metadata_sha256": sha256_file(Path(evaluation_dir) / "arm_metadata.csv"),
            "supplement_contrasts_sha256": sha256_file(paths[0]),
            "supplement_factorial_summary_sha256": sha256_file(paths[1]),
            "supplement_per_series_deltas_sha256": sha256_file(paths[2]),
        },
    )
    return (*paths, marker)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = write_supplement_outputs(args.evaluation_dir, args.output_dir)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARM_IDS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CONTRASTS",
    "FACTORIAL_ARMS",
    "SupplementInputs",
    "SupplementOutputs",
    "compute_supplement_stats",
    "load_supplement_inputs",
    "write_supplement_outputs",
]
