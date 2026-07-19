from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from measure_vit4ts_v3.aggregate import aggregate_metrics
from measure_vit4ts_v3.combined_evaluator import arm_registry
from measure_vit4ts_v3.combined_protocol import (
    canonical_json_sha256,
    load_combined_protocol,
    sha256_file,
)
from measure_vit4ts_v3.metrics import ALL_METRICS, DETECTION_METRICS, freeze_valid_series_mask, valid_mask_sha256
from measure_vit4ts_v3.paper_matched_delivery import (
    EXPECTED_ARMS,
    EXPECTED_CONTRASTS,
    assemble_paper_matched_delivery,
    validate_delivery_inputs,
)
from measure_vit4ts_v3.result_package import verify_result_zip


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
CONTRAST_ENDPOINTS = (
    ("IHP_ONLY_VS_REL", "IHP1_NCTP0", "IHP0_NCTP0"),
    ("NCTP_ONLY_VS_REL", "IHP0_NCTP1", "IHP0_NCTP0"),
    ("FULL_VS_REL", "IHP1_NCTP1", "IHP0_NCTP0"),
    ("FULL_VS_IHP_ONLY", "IHP1_NCTP1", "IHP1_NCTP0"),
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _protocol(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    manifest = tmp_path / "manifest.json"
    config.write_text("stage: vittrace_ablation_full_v3\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    arms = []
    for order, arm in enumerate(EXPECTED_ARMS):
        arms.append(
            {
                "arm": arm,
                "source_arm": arm,
                "role": "method" if arm == "FINAL_DEFAULT" else "ablation",
                "order": order,
                "fp_threshold": 0.0,
                "experiment_group": "PAPER_MATCHED",
                "changed_factor": "PAPER_MATCHED",
                "fixed_factors": {"metrics_mask": "COMMON_488"},
                "metadata": {"display_name": arm, "is_final": arm == "FINAL_DEFAULT"},
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_id": "VITTRACE_PAPER_MATCHED_ABLATIONS_V1",
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "expected_series": 492,
        "expected_valid_series": 488,
        "stages": [
            {
                "stage_id": "PAPER_MATCHED_STAGE",
                "stage_group": "paper_matched",
                "stage_kind": "cache_only",
                "configuration_id": "PAPER_MATCHED_CONFIG",
                "root": str(stage_root),
                "status_path": None,
                "expected_series": 492,
                "arms": arms,
            }
        ],
        "contrasts": [
            {"id": identifier, "family": "IHP_X_NCTP", "candidate": candidate, "control": control}
            for identifier, candidate, control in CONTRAST_ENDPOINTS
        ],
        "bootstrap": {
            "seed": 2027,
            "n_resamples": 10_000,
            "shared_indices": True,
            "hierarchy": ["subgroup", "series"],
        },
    }
    registry = tmp_path / "registry.json"
    _write_json(registry, payload)
    return registry


def _metrics_and_mask() -> tuple[pd.DataFrame, pd.DataFrame]:
    series_rows = []
    for index in range(492):
        family, subgroup = SUBGROUPS[index % len(SUBGROUPS)]
        series_rows.append(
            {
                "series_id": f"S{index:03d}",
                "family": family,
                "subgroup": subgroup,
                "n_points": 100,
                "n_positive": 0 if index < 4 else 5,
            }
        )
    mask = freeze_valid_series_mask(pd.DataFrame(series_rows))
    rows = []
    by_id = mask.set_index("series_id")
    for series_index, series_id in enumerate(mask["series_id"].astype(str)):
        descriptor = by_id.loc[series_id]
        valid = bool(descriptor["valid_f1_max"])
        subgroup_index = next(
            i for i, item in enumerate(SUBGROUPS) if item == (descriptor["family"], descriptor["subgroup"])
        )
        for arm_index, arm in enumerate(EXPECTED_ARMS):
            value = 0.30 + 0.01 * arm_index + 0.001 * subgroup_index + 0.000001 * series_index
            row = {
                "series_id": series_id,
                "family": descriptor["family"],
                "subgroup": descriptor["subgroup"],
                "n_points": int(descriptor["n_points"]),
                "n_positive": int(descriptor["n_positive"]),
                "arm": arm,
                "f1_max": value if valid else np.nan,
                "auprc": value + 0.01 if valid else np.nan,
                "vus_pr": value + 0.02 if valid else np.nan,
                "anomaly_free_fp_rate": np.nan if valid else 0.1,
                "anomaly_free_fp_count": np.nan if valid else 10.0,
                "anomaly_free_mean_excess": np.nan if valid else 0.02,
                "anomaly_free_score_p95": np.nan if valid else 0.4,
            }
            rows.append(row)
    return pd.DataFrame(rows), mask


def _external_reference(path: Path) -> None:
    paper_arms = (
        "w/o patch-level embedding",
        "w/o cross-patch comparison",
        "w/o column-wise comparison",
        "w/o multi-scale embedding",
        "ViT4TS (ours)",
    )
    rows = []
    for arm_index, arm in enumerate(paper_arms):
        for family_index, family in enumerate(("NAB", "NASA", "YAHOO")):
            rows.append(
                {
                    "paper": "VLM4TS_AAAI2026",
                    "page": 7,
                    "table_or_figure": "Table 3",
                    "arm": arm,
                    "group": family,
                    "metric": "F1-max",
                    "value": 0.5 + 0.01 * arm_index + 0.001 * family_index,
                    "value_status": "exact_paper_reported",
                    "notes": "descriptive only",
                }
            )
    table1 = {"Art": 0.545, "AWS": 0.400, "AdEx": 0.615, "Traf": 0.615, "Tweets": 0.597, "MSL": 0.543, "SMAP": 0.726, "A1": 0.614, "A2": 0.892, "A3": 0.614, "A4": 0.565, "equal-11": 0.612}
    for group, value in table1.items():
        rows.append({"paper": "VLM4TS_AAAI2026", "page": 6, "table_or_figure": "Table 1", "arm": "ViT4TS", "group": group, "metric": "F1-max", "value": value, "value_status": "exact_paper_reported", "notes": "descriptive only"})
    _write_csv(path, pd.DataFrame(rows))


def _complete_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    registry_path = _protocol(tmp_path)
    protocol = load_combined_protocol(registry_path)
    registry = arm_registry(protocol)
    evaluation = tmp_path / "evaluation"
    metrics, mask = _metrics_and_mask()
    bundle = aggregate_metrics(metrics, mask, registry)
    arm_rows = pd.DataFrame(
        {"arm": EXPECTED_ARMS, "arm_order": range(len(EXPECTED_ARMS)), "stage_id": "PAPER_MATCHED_STAGE"}
    )
    provenance = {"status": "COMPLETE", "protocol_sha256": protocol.payload_sha256}
    _write_csv(evaluation / "per_series_metrics.csv", metrics)
    _write_csv(evaluation / "valid_series_mask.csv", mask)
    _write_csv(evaluation / "arm_metadata.csv", arm_rows)
    _write_json(evaluation / "evaluation_provenance.json", provenance)
    _write_csv(evaluation / "per_series_metrics_validated.csv", bundle.per_series)
    _write_csv(evaluation / "subgroup11_metrics.csv", bundle.subgroup11)
    _write_csv(evaluation / "family3_metrics.csv", bundle.family3)
    _write_csv(evaluation / "equal11_metrics.csv", bundle.equal11)
    _write_csv(evaluation / "fileweighted_metrics.csv", bundle.fileweighted)
    bootstrap_rows = []
    for contrast_index, (identifier, candidate, control) in enumerate(CONTRAST_ENDPOINTS):
        for metric_index, metric in enumerate(DETECTION_METRICS):
            bootstrap_rows.append(
                {
                    "contrast_id": identifier,
                    "contrast_family": "IHP_X_NCTP",
                    "candidate": candidate,
                    "control": control,
                    "metric": metric,
                    "delta": 0.01 * (contrast_index + 1),
                    "ci_lower": -0.001 + 0.01 * contrast_index,
                    "ci_upper": 0.02 + 0.01 * contrast_index,
                    "n_boot": 10_000,
                    "seed": 2027,
                    "shared_indices": True,
                    "resample_plan_sha256": "A" * 64,
                    "valid_mask_sha256": valid_mask_sha256(mask),
                    "n_valid_series": 488,
                    "n_subgroups": 11,
                    "resampling_unit": "11_subgroups_then_paired_series",
                }
            )
    _write_csv(evaluation / "bootstrap_ci.csv", pd.DataFrame(bootstrap_rows))
    evaluation_marker = {
        "schema_version": 1,
        "status": "COMPLETE",
        "protocol_sha256": protocol.payload_sha256,
        "series_count": 492,
        "valid_series_count": 488,
        "stage_count": 1,
        "arm_count": 11,
        "per_series_metrics_sha256": sha256_file(evaluation / "per_series_metrics.csv"),
        "valid_series_mask_file_sha256": sha256_file(evaluation / "valid_series_mask.csv"),
        "arm_metadata_sha256": sha256_file(evaluation / "arm_metadata.csv"),
        "evaluation_provenance_sha256": sha256_file(evaluation / "evaluation_provenance.json"),
    }
    _write_json(evaluation / "_COMBINED_EVALUATION_COMPLETE.json", evaluation_marker)
    aggregation_marker = {
        "schema_version": 1,
        "status": "COMPLETE",
        "protocol_sha256": protocol.payload_sha256,
        "valid_mask_sha256": valid_mask_sha256(mask),
        "series_count": 492,
        "valid_series_count": 488,
        "arm_count": 11,
        "contrast_count": 4,
        "bootstrap_seed": 2027,
        "bootstrap_replicates": 10_000,
        "resample_plan_sha256": "A" * 64,
        **{
            f"{name}_sha256": sha256_file(evaluation / filename)
            for name, filename in {
                "per_series": "per_series_metrics_validated.csv",
                "subgroup11": "subgroup11_metrics.csv",
                "family3": "family3_metrics.csv",
                "equal11": "equal11_metrics.csv",
                "fileweighted": "fileweighted_metrics.csv",
                "bootstrap": "bootstrap_ci.csv",
            }.items()
        },
    }
    _write_json(evaluation / "_COMBINED_AGGREGATION_COMPLETE.json", aggregation_marker)
    external = tmp_path / "external.csv"
    _external_reference(external)
    return registry_path, evaluation, external


def test_builds_verified_compact_paper_matched_package(tmp_path: Path) -> None:
    registry, evaluation, external = _complete_evidence(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "delivery"
    archive, manifest = assemble_paper_matched_delivery(
        registry,
        evaluation,
        external,
        repo_root,
        output,
        zip_path=tmp_path / "delivery.zip",
    )

    assert archive == (tmp_path / "delivery.zip").resolve()
    assert verify_result_zip(archive)["sha256sums_sha256"] == manifest["sha256sums_sha256"]
    table3 = pd.read_csv(output / "tables" / "table3_style_f1_max.csv")
    assert table3["display_name"].tolist() == [
        "w/o patch-level embedding",
        "w/o cross-patch comparison",
        "w/o column-wise comparison",
        "w/o multi-scale embedding",
        "Full method",
    ]
    assert table3.shape[0] == 5 and int(table3["is_final"].sum()) == 1
    factorial = pd.read_csv(output / "tables" / "table_ihp_nctp_2x2_vus_pr.csv")
    assert set(zip(factorial["ihp"], factorial["nctp"])) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert "fileweighted" in factorial
    long11 = pd.read_csv(output / "tables" / "table_final_default_long11.csv")
    assert long11.shape == (3, 17)
    comparison = pd.read_csv(output / "tables" / "external_descriptive_comparison.csv")
    assert len(comparison) == 15
    assert set(comparison["comparison_status"]) == {"descriptive_external_only_no_paired_ci"}
    assert not comparison["paired_ci_applicable"].astype(bool).any()
    main_comparison = pd.read_csv(output / "tables" / "main_vs_paper_f1.csv")
    assert len(main_comparison) == 12
    assert set(main_comparison["comparison_status"]) == {"descriptive_external_not_paired"}
    assert not (output / "results" / "scores.npy").exists()
    marker = json.loads((output / "manifests" / "_PAPER_MATCHED_DELIVERY_COMPLETE.json").read_text())
    assert marker["status"] == "COMPLETE" and marker["valid_series_count"] == 488


def test_rejects_bootstrap_protocol_drift_even_when_hash_is_rebound(tmp_path: Path) -> None:
    registry, evaluation, external = _complete_evidence(tmp_path)
    bootstrap_path = evaluation / "bootstrap_ci.csv"
    bootstrap = pd.read_csv(bootstrap_path)
    bootstrap.loc[0, "n_boot"] = 9_999
    _write_csv(bootstrap_path, bootstrap)
    marker_path = evaluation / "_COMBINED_AGGREGATION_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["bootstrap_sha256"] = sha256_file(bootstrap_path)
    _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="bootstrap frozen protocol differs"):
        validate_delivery_inputs(registry, evaluation, external)


def test_rejects_incomplete_arm_series_grid(tmp_path: Path) -> None:
    registry, evaluation, external = _complete_evidence(tmp_path)
    metrics_path = evaluation / "per_series_metrics.csv"
    metrics = pd.read_csv(metrics_path).iloc[:-1]
    _write_csv(metrics_path, metrics)
    marker_path = evaluation / "_COMBINED_EVALUATION_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["per_series_metrics_sha256"] = sha256_file(metrics_path)
    _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="11 x 492"):
        validate_delivery_inputs(registry, evaluation, external)
