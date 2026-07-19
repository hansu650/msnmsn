from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from measure_vit4ts_v3.metrics import (
    ANOMALY_FREE_METRICS,
    evaluate_series,
    freeze_valid_series_mask,
    valid_mask_sha256,
    validate_metrics_against_mask,
)
from measure_vit4ts_v3.registry import validate_arm_registry


def _registry_payload() -> dict:
    return {
        "schema_version": 3,
        "registry_id": "VITTRACE_V3_TEST",
        "primary_arm": "A",
        "control_arm": "B",
        "arms": [
            {"id": "A", "role": "primary", "order": 0, "fp_threshold": 0.5},
            {"id": "B", "role": "control", "order": 1, "fp_threshold": 0.5},
        ],
        "contrasts": [
            {"id": "A_MINUS_B", "family": "main", "candidate": "A", "control": "B"}
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


@pytest.mark.parametrize(
    ("labels", "is_anomaly_free"),
    [
        (np.zeros(4, dtype=np.uint8), True),
        (np.ones(4, dtype=np.uint8), False),
    ],
)
def test_single_class_detection_metrics_are_undefined_without_metric_calls(
    labels: np.ndarray, is_anomaly_free: bool
) -> None:
    scores = np.asarray([0.1, 0.6, 0.7, 0.2], dtype=np.float64)

    def forbidden(*args, **kwargs):
        raise AssertionError("undefined threshold-free metrics must not be called")

    result = evaluate_series(
        labels,
        scores,
        np.arange(4),
        (),
        fp_threshold=0.5,
        f1_fn=forbidden,
        auprc_fn=forbidden,
        vus_fn=forbidden,
    )
    assert np.isnan(result["f1_max"])
    assert np.isnan(result["f1_winning_alpha"])
    assert np.isnan(result["f1_threshold_evaluator_only"])
    assert np.isnan(result["auprc"])
    assert np.isnan(result["vus_pr"])
    if is_anomaly_free:
        assert result["anomaly_free_fp_count"] == 2.0
        assert result["anomaly_free_fp_rate"] == 0.5
        assert np.isclose(result["anomaly_free_mean_excess"], 0.075)
    else:
        assert all(np.isnan(result[metric]) for metric in ANOMALY_FREE_METRICS)


def test_valid_series_mask_is_prefixed_once_and_identical_for_every_arm() -> None:
    index = pd.DataFrame(
        [
            {"series_id": "normal", "family": "F1", "subgroup": "G1", "n_points": 4, "n_positive": 0},
            {"series_id": "mixed", "family": "F1", "subgroup": "G1", "n_points": 4, "n_positive": 1},
            {"series_id": "all_positive", "family": "F1", "subgroup": "G1", "n_points": 4, "n_positive": 4},
        ]
    )
    mask = freeze_valid_series_mask(index)
    normal = mask.set_index("series_id").loc["normal"]
    mixed = mask.set_index("series_id").loc["mixed"]
    all_positive = mask.set_index("series_id").loc["all_positive"]
    assert not bool(normal["valid_f1_max"])
    assert not bool(normal["valid_auprc"])
    assert bool(normal["valid_anomaly_free_fp_rate"])
    assert bool(mixed["valid_f1_max"])
    assert bool(mixed["valid_auprc"]) and bool(mixed["valid_vus_pr"])
    assert not bool(all_positive["valid_f1_max"])
    assert not bool(all_positive["valid_auprc"])
    assert not bool(all_positive["valid_anomaly_free_fp_rate"])
    assert valid_mask_sha256(mask) == valid_mask_sha256(mask.sample(frac=1.0, random_state=1))

    rows = []
    for record in mask.itertuples(index=False):
        for arm in ("A", "B"):
            row = {
                "series_id": record.series_id,
                "arm": arm,
                "family": record.family,
                "subgroup": record.subgroup,
                "n_points": record.n_points,
                "n_positive": record.n_positive,
                "f1_max": 0.5 if record.valid_f1_max else np.nan,
                "auprc": 0.5 if record.valid_auprc else np.nan,
                "vus_pr": 0.5 if record.valid_vus_pr else np.nan,
            }
            row.update(
                {
                    metric: 0.1 if getattr(record, f"valid_{metric}") else np.nan
                    for metric in ANOMALY_FREE_METRICS
                }
            )
            rows.append(row)
    validated = validate_metrics_against_mask(pd.DataFrame(rows), mask, ("A", "B"))
    assert len(validated) == 6
    corrupted = pd.DataFrame(rows)
    corrupted.loc[
        (corrupted["series_id"] == "normal") & (corrupted["arm"] == "A"), "auprc"
    ] = 0.0
    with pytest.raises(ValueError, match="pre-fixed valid-series mask"):
        validate_metrics_against_mask(corrupted, mask, ("A", "B"))


def test_arm_registry_rejects_analysis_or_schema_drift() -> None:
    registry = validate_arm_registry(_registry_payload())
    assert registry.arm_ids == ("A", "B")
    assert registry.bootstrap_seed == 2027
    changed = copy.deepcopy(_registry_payload())
    changed["bootstrap"]["seed"] = 9
    with pytest.raises(ValueError, match="seed 2027"):
        validate_arm_registry(changed)
    changed = copy.deepcopy(_registry_payload())
    changed["arms"][1]["id"] = "A"
    with pytest.raises(ValueError, match="unique"):
        validate_arm_registry(changed)
