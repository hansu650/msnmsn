from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from measure_vit4ts_v3.combined_protocol import sha256_file
from measure_vit4ts_v3.metrics import DETECTION_METRICS
from measure_vit4ts_v3.supplement_stats import (
    ARM_IDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SupplementInputs,
    compute_supplement_stats,
    load_supplement_inputs,
    write_supplement_outputs,
)


SUBGROUPS = (
    ("NAB", "NAB-Artificial"),
    ("NAB", "NAB-AWS"),
    ("NAB", "NAB-AdExchange"),
    ("NAB", "NAB-Traffic"),
    ("NAB", "NAB-Tweets"),
    ("NASA", "NASA-MSL"),
    ("NASA", "NASA-SMAP"),
    ("YAHOO", "Yahoo-A1"),
    ("YAHOO", "Yahoo-A2"),
    ("YAHOO", "Yahoo-A3"),
    ("YAHOO", "Yahoo-A4"),
)

ARM_OFFSETS = {
    "IHP0_NCTP0": 0.00,
    "IHP1_NCTP0": 0.01,
    "IHP0_NCTP1": 0.02,
    "IHP1_NCTP1": 0.05,
}

EXPECTED_DELTAS = {
    "IHP_MINUS_REL": 0.01,
    "NCTP_MINUS_REL": 0.02,
    "FULL_MINUS_REL": 0.05,
    "FULL_MINUS_IHP": 0.04,
    "FULL_MINUS_NCTP": 0.03,
    "FACTORIAL_INTERACTION": 0.02,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _complete_evaluation(tmp_path: Path) -> Path:
    evaluation = tmp_path / "evaluation"
    rows: list[dict[str, object]] = []
    for series_index in range(492):
        family, subgroup = SUBGROUPS[series_index % len(SUBGROUPS)]
        valid = series_index >= 4
        for arm in ARM_IDS:
            base = 0.30 + 0.001 * (series_index % len(SUBGROUPS)) + 1e-6 * series_index
            value = base + ARM_OFFSETS[arm]
            rows.append(
                {
                    "series_id": f"S{series_index:03d}",
                    "family": family,
                    "subgroup": subgroup,
                    "arm": arm,
                    "f1_max": value if valid else np.nan,
                    "auprc": value + 0.03 if valid else np.nan,
                    "vus_pr": value + 0.06 if valid else np.nan,
                }
            )
    metrics = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "arm": ARM_IDS,
            "arm_order": range(len(ARM_IDS)),
            "stage_id": "IHP_NCTP_FACTORIAL",
        }
    )
    _write_csv(evaluation / "per_series_metrics.csv", metrics)
    _write_csv(evaluation / "arm_metadata.csv", metadata)
    _write_json(
        evaluation / "_COMBINED_EVALUATION_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "series_count": 492,
            "valid_series_count": 488,
            "arm_count": len(ARM_IDS),
            "per_series_metrics_sha256": sha256_file(evaluation / "per_series_metrics.csv"),
            "arm_metadata_sha256": sha256_file(evaluation / "arm_metadata.csv"),
        },
    )
    return evaluation


def test_writes_full_shared_plan_factorial_supplement(tmp_path: Path) -> None:
    evaluation = _complete_evaluation(tmp_path)
    output = tmp_path / "supplement"
    paths = write_supplement_outputs(evaluation, output)

    contrasts = pd.read_csv(output / "supplement_contrasts.csv")
    factorial = pd.read_csv(output / "supplement_factorial_summary.csv")
    per_series = pd.read_csv(output / "supplement_per_series_deltas.csv")
    marker = json.loads((output / "_SUPPLEMENT_STATS_COMPLETE.json").read_text())

    assert len(paths) == 4
    assert contrasts.shape[0] == 6 * len(DETECTION_METRICS)
    assert factorial.shape[0] == 4 * len(DETECTION_METRICS)
    assert per_series.shape[0] == 488 * 6 * len(DETECTION_METRICS)
    assert contrasts["shared_plan_sha256"].nunique() == 1
    assert contrasts.iloc[0]["shared_plan_sha256"] == marker["shared_plan_sha256"]
    assert set(contrasts["n_boot"]) == {BOOTSTRAP_REPLICATES}
    assert set(contrasts["seed"]) == {BOOTSTRAP_SEED}
    assert set(contrasts["effective_n"]) == {488}
    assert not contrasts["crosses_zero"].astype(bool).any()
    assert np.allclose(contrasts["proportion_gt_zero"], 1.0)
    for contrast_id, expected in EXPECTED_DELTAS.items():
        selected = contrasts.loc[contrasts["contrast_id"] == contrast_id]
        assert np.allclose(selected["point_delta"], expected, atol=1e-14)
        assert np.allclose(selected["ci_lower"], expected, atol=1e-14)
        assert np.allclose(selected["ci_upper"], expected, atol=1e-14)
    assert marker["status"] == "COMPLETE"
    assert marker["contrast_rows"] == 18
    assert marker["factorial_rows"] == 12
    assert marker["per_series_delta_rows"] == 8784
    for path in paths[:3]:
        assert marker[f"{path.stem}_sha256"] == sha256_file(path)


def test_shared_plan_is_invariant_to_metric_row_order(tmp_path: Path) -> None:
    inputs = load_supplement_inputs(_complete_evaluation(tmp_path))
    first = compute_supplement_stats(inputs, n_boot=128)
    shuffled = SupplementInputs(
        inputs.metrics.sample(frac=1.0, random_state=13).reset_index(drop=True),
        inputs.arm_metadata,
        inputs.marker,
        inputs.valid_series,
    )
    second = compute_supplement_stats(shuffled, n_boot=128)

    assert first.plan_sha256 == second.plan_sha256
    pd.testing.assert_frame_equal(first.contrasts, second.contrasts)
    pd.testing.assert_frame_equal(first.factorial_summary, second.factorial_summary)
    pd.testing.assert_frame_equal(first.per_series_deltas, second.per_series_deltas)


def test_rejects_metric_specific_validity_drift_after_hash_rebind(tmp_path: Path) -> None:
    evaluation = _complete_evaluation(tmp_path)
    metrics_path = evaluation / "per_series_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    target = (metrics["series_id"] == "S004") & (metrics["arm"] == "IHP1_NCTP0")
    metrics.loc[target, "vus_pr"] = np.nan
    _write_csv(metrics_path, metrics)
    marker_path = evaluation / "_COMBINED_EVALUATION_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["per_series_metrics_sha256"] = sha256_file(metrics_path)
    _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="do not share one validity mask"):
        load_supplement_inputs(evaluation)
