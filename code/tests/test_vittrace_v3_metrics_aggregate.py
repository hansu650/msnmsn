from __future__ import annotations

import numpy as np
import pandas as pd

from measure_vit4ts_v3.aggregate import aggregate_metrics, paired_hierarchical_bootstrap
from measure_vit4ts_v3.metrics import ANOMALY_FREE_METRICS, freeze_valid_series_mask
from measure_vit4ts_v3.outputs import bootstrap_plot_frame, subgroup_delta_plot_frame, tidy_table_frame
from measure_vit4ts_v3.registry import validate_arm_registry


GROUPS = tuple(
    [("F1", f"G{i}") for i in range(1, 6)]
    + [("F2", f"G{i}") for i in range(6, 8)]
    + [("F3", f"G{i}") for i in range(8, 12)]
)


def _registry():
    return validate_arm_registry(
        {
            "schema_version": 3,
            "registry_id": "VITTRACE_V3_AGG",
            "primary_arm": "A",
            "control_arm": "B",
            "arms": [
                {"id": "A", "role": "primary", "order": 0, "fp_threshold": 0.5},
                {"id": "B", "role": "control", "order": 1, "fp_threshold": 0.5},
            ],
            "contrasts": [
                {"id": "A_MINUS_B", "family": "main", "candidate": "A", "control": "B"},
                {"id": "B_MINUS_A", "family": "symmetry", "candidate": "B", "control": "A"},
            ],
            "validity_policy": {
                "f1_max": "both_classes",
                "auprc": "both_classes",
                "vus_pr": "both_classes",
                "anomaly_free_fp": "no_positive",
            },
            "bootstrap": {
                "seed": 2027,
                "n_resamples": 10000,
                "shared_indices": True,
                "hierarchy": ["subgroup", "series"],
            },
            "groups": {"expected_subgroups": 11, "expected_families": 3},
        }
    )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    index_rows = []
    metric_rows = []
    for family, subgroup in GROUPS:
        for kind, positives in (("positive", 1), ("normal", 0)):
            series_id = f"{subgroup}_{kind}"
            index_rows.append(
                {
                    "series_id": series_id,
                    "family": family,
                    "subgroup": subgroup,
                    "n_points": 10,
                    "n_positive": positives,
                }
            )
            for arm in ("A", "B"):
                candidate = arm == "A"
                row = {
                    "series_id": series_id,
                    "arm": arm,
                    "family": family,
                    "subgroup": subgroup,
                    "n_points": 10,
                    "n_positive": positives,
                    "f1_max": (0.6 if candidate else 0.5) if positives else np.nan,
                    "auprc": (0.7 if candidate else 0.5) if positives else np.nan,
                    "vus_pr": (0.8 if candidate else 0.6) if positives else np.nan,
                }
                fp_values = {
                    "anomaly_free_fp_rate": 0.1 if candidate else 0.2,
                    "anomaly_free_fp_count": 1.0 if candidate else 2.0,
                    "anomaly_free_mean_excess": 0.01 if candidate else 0.02,
                    "anomaly_free_score_p95": 0.4 if candidate else 0.5,
                }
                row.update(fp_values if not positives else {name: np.nan for name in ANOMALY_FREE_METRICS})
                metric_rows.append(row)
    return pd.DataFrame(index_rows), pd.DataFrame(metric_rows)


def test_all_required_aggregation_views_use_the_same_valid_mask() -> None:
    index, metrics = _inputs()
    mask = freeze_valid_series_mask(index)
    registry = _registry()
    bundle = aggregate_metrics(metrics, mask, registry)
    assert len(bundle.per_series) == 44
    assert bundle.subgroup11["subgroup"].nunique() == 11
    assert bundle.family3["family"].nunique() == 3
    auprc_a = bundle.equal11.loc[
        (bundle.equal11["arm"] == "A") & (bundle.equal11["metric"] == "auprc")
    ].iloc[0]
    assert np.isclose(auprc_a["value"], 0.7)
    assert auprc_a["n_valid"] == 11
    detection_a = bundle.equal11.loc[
        (bundle.equal11["arm"] == "A")
        & bundle.equal11["metric"].isin(("f1_max", "auprc", "vus_pr"))
    ]
    assert set(detection_a["n_valid"]) == {11}
    for view in (
        bundle.subgroup11,
        bundle.family3,
        bundle.equal11,
        bundle.fileweighted,
    ):
        detection = view.loc[view["metric"].isin(("f1_max", "auprc", "vus_pr"))]
        grouped = detection.groupby(["family", "subgroup", "arm"])["n_valid"]
        assert grouped.nunique().eq(1).all()
    weighted_fp = bundle.fileweighted.loc[
        (bundle.fileweighted["arm"] == "A")
        & (bundle.fileweighted["metric"] == "anomaly_free_fp_rate")
    ].iloc[0]
    assert np.isclose(weighted_fp["value"], 0.1)
    assert weighted_fp["n_valid"] == 11

    table = tidy_table_frame(bundle.equal11, registry, metrics=("auprc", "vus_pr"))
    assert list(table["metric"].unique()) == ["auprc", "vus_pr"]
    plot = subgroup_delta_plot_frame(bundle.subgroup11, registry, "auprc")
    assert len(plot) == 22
    assert np.allclose(plot.loc[plot["contrast_id"] == "A_MINUS_B", "delta"], 0.2)


def test_bootstrap_reuses_one_paired_index_plan_for_all_contrasts_and_metrics() -> None:
    index, metrics = _inputs()
    mask = freeze_valid_series_mask(index)
    registry = _registry()
    auprc = paired_hierarchical_bootstrap(
        metrics, mask, registry, "auprc", n_boot=100, seed=2027
    )
    repeat = paired_hierarchical_bootstrap(
        metrics, mask, registry, "auprc", n_boot=100, seed=2027
    )
    vus = paired_hierarchical_bootstrap(
        metrics, mask, registry, "vus_pr", n_boot=100, seed=2027
    )
    assert auprc.equals(repeat)
    assert auprc["resample_plan_sha256"].nunique() == 1
    assert auprc.iloc[0]["resample_plan_sha256"] == vus.iloc[0]["resample_plan_sha256"]
    main = auprc.loc[auprc["contrast_id"] == "A_MINUS_B"].iloc[0]
    inverse = auprc.loc[auprc["contrast_id"] == "B_MINUS_A"].iloc[0]
    assert np.isclose(main["delta"], 0.2)
    assert np.isclose(main["ci_lower"], 0.2)
    assert np.isclose(inverse["delta"], -0.2)
    plot = bootstrap_plot_frame(auprc, "auprc")
    assert set(plot["zero_reference"]) == {0.0}
