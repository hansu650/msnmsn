from __future__ import annotations

import copy

import pytest

from measure_vit4ts_v3.registry import validate_arm_registry


def _payload() -> dict:
    return {
        "schema_version": 3,
        "registry_id": "VITTRACE_V3_LEGACY_COMPAT_TEST",
        "primary_arm": "FINAL",
        "control_arm": "BASE",
        "arms": [
            {"id": "BASE", "role": "control", "order": 0, "fp_threshold": 0.0},
            {"id": "FINAL", "role": "method", "order": 1, "fp_threshold": 0.0},
        ],
        "contrasts": [
            {
                "id": "FINAL_VS_BASE",
                "family": "PRIMARY",
                "candidate": "FINAL",
                "control": "BASE",
            }
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


def test_exact_legacy_artifact_policy_is_accepted_but_emission_is_corrected() -> None:
    payload = _payload()
    payload["validity_policy"]["f1_max"] = "all_series"
    registry = validate_arm_registry(payload)
    assert registry.arm_ids == ("BASE", "FINAL")
    emitted = registry.to_payload()["validity_policy"]
    assert emitted == {
        "f1_max": "both_classes",
        "auprc": "both_classes",
        "vus_pr": "both_classes",
        "anomaly_free_fp": "no_positive",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("f1_max", "always_defined"),
        ("auprc", "all_series"),
        ("vus_pr", "all_series"),
        ("anomaly_free_fp", "both_classes"),
    ],
)
def test_any_nonexact_policy_variant_is_rejected(field: str, value: str) -> None:
    payload = copy.deepcopy(_payload())
    payload["validity_policy"][field] = value
    with pytest.raises(ValueError, match="validity policy changed"):
        validate_arm_registry(payload)
