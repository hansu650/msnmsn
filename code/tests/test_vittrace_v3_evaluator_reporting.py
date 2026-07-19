from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts_v3.aggregate import aggregate_metrics
from measure_vit4ts_v3.cache_registry import (
    CacheOnlyArm,
    CacheOnlyPlan,
    PlannedArm,
    parameter_sha256,
    sha256_file,
)
from measure_vit4ts_v3.evaluator import (
    EvaluationProtocol,
    FileIdentity,
    RUN_NAME,
    evaluate_preflight,
    preflight_scores,
    write_evaluation_outputs,
)
from measure_vit4ts_v3.metrics import ANOMALY_FREE_METRICS, freeze_valid_series_mask
from measure_vit4ts_v3.registry import validate_arm_registry
from measure_vit4ts_v3.reporting import (
    arm_registry_frame,
    factorial_2x2_frame,
    long11_frame,
    plot_tidy_frame,
    table3_style_frame,
    write_reporting_outputs,
)
from measure_vit4ts_v3.structural_audit import structural_audit_frame


def _identity(path: Path) -> FileIdentity:
    target = path.resolve(strict=True)
    stat = target.stat()
    return FileIdentity(
        target,
        sha256_file(target).upper(),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _mini_arm(arm_id: str, incidence: str) -> CacheOnlyArm:
    return CacheOnlyArm(
        arm_id=arm_id,
        family="TEST",
        role="method" if arm_id == "A" else "control",
        matching_scope="global",
        memory="median_reference",
        scales=("P", "M", "L"),
        incidence=incidence,
        fusion="active_valid_harmonic",
        temporal="nctp_linear",
    )


def _mini_registry():
    return validate_arm_registry(
        {
            "schema_version": 3,
            "registry_id": "VITTRACE_V3_EVAL_TEST",
            "primary_arm": "A",
            "control_arm": "B",
            "arms": [
                {"id": "A", "role": "method", "order": 0, "fp_threshold": 0.5},
                {"id": "B", "role": "control", "order": 1, "fp_threshold": 0.5},
            ],
            "contrasts": [
                {"id": "A_VS_B", "family": "TEST", "candidate": "A", "control": "B"}
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


def _build_protocol(tmp_path: Path) -> EvaluationProtocol:
    data_root = tmp_path / "data"
    data_root.mkdir()
    records = []
    for series_id, family, subgroup in (
        ("normal", "NAB", "NAB-Artificial"),
        ("mixed", "NASA", "NASA-MSL"),
    ):
        path = data_root / f"{series_id}.csv"
        pd.DataFrame({"timestamp": np.arange(4), "value": np.arange(4)}).to_csv(
            path, index=False
        )
        records.append(
            FullSeriesRecord(
                series_id,
                series_id,
                family,
                subgroup,
                series_id,
                path.name,
                4,
                1,
                sha256_file(path).upper(),
                False,
            )
        )
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    plan_path = tmp_path / "plan.json"
    gate_path = tmp_path / "gate.json"
    anomalies_path = tmp_path / "anomalies.csv"
    for path, payload in (
        (config_path, "stage: vittrace_ablation_full_v3\n"),
        (manifest_path, "{}\n"),
        (registry_path, "{}\n"),
        (plan_path, "{}\n"),
        (gate_path, "{}\n"),
        (anomalies_path, "signal,events\n"),
    ):
        path.write_text(payload, encoding="utf-8")
    config = {
        "stage": "vittrace_ablation_full_v3",
        "paths": {
            "run_root": str(tmp_path / "runs"),
            "failure_root": str(tmp_path / "failures"),
        },
        "data": {"root": str(data_root), "anomalies_csv": str(anomalies_path)},
        "statistics": {"vus_max_window": 2},
    }
    arm_a = _mini_arm("A", "literal")
    arm_b = _mini_arm("B", "literal")
    digest = parameter_sha256(arm_a.parameters())
    assert digest == parameter_sha256(arm_b.parameters())
    plan = CacheOnlyPlan(
        (
            PlannedArm(arm_a, "A", digest),
            PlannedArm(arm_b, "A", digest),
        )
    )
    registry = _mini_registry()
    protocol = EvaluationProtocol(
        config_path.resolve(),
        config,
        _identity(config_path),
        _identity(manifest_path),
        tuple(records),
        registry_path.resolve(),
        _identity(registry_path),
        registry,
        plan_path.resolve(),
        _identity(plan_path),
        plan,
        {},
        _identity(gate_path),
        Path(config["paths"]["run_root"]) / RUN_NAME,
    )
    source_sha = "A" * 64
    for record in records:
        series_root = protocol.stage_root / record.series_id
        arm_root = series_root / "A"
        arm_root.mkdir(parents=True)
        score = np.asarray([0.1, 0.6, 0.8, 0.2], dtype=np.float64)
        score_path = arm_root / "score.npy"
        np.save(score_path, score, allow_pickle=False)
        score_sha = sha256_file(score_path).upper()
        common = {
            "schema_version": 1,
            "series_id": record.series_id,
            "config_sha256": protocol.config_identity.sha256,
            "full_manifest_sha256": protocol.manifest_identity.sha256,
            "compute_plan_sha256": protocol.plan_identity.sha256,
            "parity_gate_sha256": protocol.parity_gate_identity.sha256,
            "source_sha256": source_sha,
            "encoder_calls": 0,
        }
        manifest = {
            **common,
            "arm": "A",
            "parameter_sha256": digest,
            "dataset": record.dataset,
            "track": record.track,
            "paper_group": record.paper_group,
            "signal_name": record.signal_name,
            "data_sha256": record.expected_sha256,
            "score_sha256": score_sha,
            "score_length": 4,
            "score_dtype": "float64",
            "cache_sha256": "B" * 64,
            "cache_manifest_sha256": "C" * 64,
            "trace_sha256": "D" * 64,
            "trace_manifest_sha256": "E" * 64,
        }
        _write_json(arm_root / "score_manifest.json", manifest)
        _write_json(
            arm_root / "_SCORES_READY.json",
            {
                "series_id": record.series_id,
                "arm": "A",
                "score_sha256": score_sha,
                "config_sha256": protocol.config_identity.sha256,
                "compute_plan_sha256": protocol.plan_identity.sha256,
                "source_sha256": source_sha,
                "encoder_calls": 0,
            },
        )
        _write_json(
            arm_root / "_SUCCESS.json",
            {
                "series_id": record.series_id,
                "arm": "A",
                "score_sha256": score_sha,
                "encoder_calls": 0,
            },
        )
        alias_root = series_root / "B"
        alias_root.mkdir()
        _write_json(
            alias_root / "alias_manifest.json",
            {
                **common,
                "arm": "B",
                "parameter_sha256": digest,
                "canonical_arm": "A",
                "canonical_score_path": "../A/score.npy",
                "canonical_score_sha256": score_sha,
            },
        )
        _write_json(
            alias_root / "_SUCCESS.json",
            {
                "series_id": record.series_id,
                "arm": "B",
                "canonical_arm": "A",
                "encoder_calls": 0,
            },
        )
        _write_json(
            series_root / "_SUCCESS.json",
            {
                "schema_version": 1,
                "series_id": record.series_id,
                "logical_arm_count": 2,
                "unique_computation_count": 1,
                "encoder_calls": 0,
                "config_sha256": protocol.config_identity.sha256,
                "compute_plan_sha256": protocol.plan_identity.sha256,
                "source_sha256": source_sha,
            },
        )
    return protocol


def test_alias_preflight_precedes_labels_and_no_positive_mask_is_fixed(
    tmp_path: Path,
) -> None:
    protocol = _build_protocol(tmp_path)
    retained = protocol.stage_root / "normal" / "_FAILED.json"
    _write_json(retained, {"error": "preserved"})
    with pytest.raises(RuntimeError, match="block label access"):
        preflight_scores(protocol)
    assert retained.is_file()
    retained.unlink()

    preflight = preflight_scores(protocol)
    assert len(preflight.scores) == 4
    alias = preflight.scores[("normal", "B")]
    assert alias.is_alias and alias.canonical_arm == "A"
    assert alias.score_path == preflight.scores[("normal", "A")].score_path

    reads: list[str] = []

    def ground_truth(config, series_id, timestamps):
        reads.append(series_id)
        labels = (
            np.zeros(4, dtype=np.uint8)
            if series_id == "normal"
            else np.asarray([0, 1, 0, 0], dtype=np.uint8)
        )
        return SimpleNamespace(point_labels=labels, intervals=())

    result = evaluate_preflight(
        protocol,
        preflight,
        timestamp_loader=lambda record, root: np.arange(4),
        ground_truth_loader=ground_truth,
        f1_fn=lambda *args: (0.5, 0.1, 0.5),
        auprc_fn=lambda labels, scores: 0.75,
        vus_fn=lambda labels, scores, window: 0.8,
    )
    assert reads == ["normal", "mixed"]
    normal = result.per_series.loc[result.per_series["series_id"] == "normal"]
    mixed = result.per_series.loc[result.per_series["series_id"] == "mixed"]
    assert normal["f1_max"].isna().all()
    assert normal["auprc"].isna().all() and normal["vus_pr"].isna().all()
    assert set(normal["anomaly_free_fp_count"]) == {2.0}
    assert set(mixed["auprc"]) == {0.75}
    assert mixed["anomaly_free_fp_rate"].isna().all()
    assert result.provenance["alias_record_count"] == 2

    outputs = write_evaluation_outputs(tmp_path / "evaluation", result)
    assert all(path.is_file() for path in outputs)
    marker = json.loads(outputs[-1].read_text(encoding="utf-8"))
    assert marker["score_record_count"] == 4


FAMILIES = (
    *(('NAB', f'NAB-G{i}') for i in range(1, 6)),
    *(('NASA', f'NASA-G{i}') for i in range(1, 3)),
    *(('Yahoo', f'Yahoo-G{i}') for i in range(1, 5)),
)
REPORT_ARMS = (
    "LEGACY_DEFAULT",
    "FINAL_DEFAULT",
    "IHP0_NCTP0",
    "IHP1_NCTP0",
    "IHP0_NCTP1",
    "IHP1_NCTP1",
)


def _report_plan_registry() -> tuple[CacheOnlyPlan, object]:
    logical = []
    for order, arm_id in enumerate(REPORT_ARMS):
        ihp, nctp = {
            "LEGACY_DEFAULT": (0, 0),
            "FINAL_DEFAULT": (1, 1),
            "IHP0_NCTP0": (0, 0),
            "IHP1_NCTP0": (1, 0),
            "IHP0_NCTP1": (0, 1),
            "IHP1_NCTP1": (1, 1),
        }[arm_id]
        logical.append(
            CacheOnlyArm(
                arm_id,
                "IHP_X_NCTP" if arm_id.startswith("IHP") else "DEFAULT",
                "method" if arm_id == "FINAL_DEFAULT" else "control",
                "global",
                "median_reference",
                ("P", "M", "L"),
                "literal" if ihp else "released",
                "active_valid_harmonic" if ihp else "released_harmonic",
                "nctp_linear" if nctp else "legacy",
                None if nctp else "top_fraction",
                None if nctp else 0.25,
            )
        )
    seen: dict[str, str] = {}
    planned = []
    for arm in logical:
        digest = parameter_sha256(arm.parameters())
        canonical = seen.setdefault(digest, arm.arm_id)
        planned.append(PlannedArm(arm, canonical, digest))
    plan = CacheOnlyPlan(tuple(planned))
    registry = validate_arm_registry(
        {
            "schema_version": 3,
            "registry_id": "VITTRACE_V3_REPORT_TEST",
            "primary_arm": "FINAL_DEFAULT",
            "control_arm": "LEGACY_DEFAULT",
            "arms": [
                {"id": arm, "role": logical[i].role, "order": i, "fp_threshold": 0.5}
                for i, arm in enumerate(REPORT_ARMS)
            ],
            "contrasts": [
                {
                    "id": "FINAL_VS_LEGACY",
                    "family": "PRIMARY",
                    "candidate": "FINAL_DEFAULT",
                    "control": "LEGACY_DEFAULT",
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
    )
    return plan, registry


def _report_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    index_rows = []
    metric_rows = []
    for family, subgroup in FAMILIES:
        for suffix, positives in (("mixed", 1), ("normal", 0)):
            series_id = f"{subgroup}_{suffix}"
            index_rows.append(
                {
                    "series_id": series_id,
                    "family": family,
                    "subgroup": subgroup,
                    "n_points": 10,
                    "n_positive": positives,
                }
            )
            for arm_order, arm in enumerate(REPORT_ARMS):
                value = 0.2 + 0.05 * arm_order
                row = {
                    "series_id": series_id,
                    "arm": arm,
                    "family": family,
                    "subgroup": subgroup,
                    "n_points": 10,
                    "n_positive": positives,
                    "f1_max": value if positives else np.nan,
                    "auprc": value if positives else np.nan,
                    "vus_pr": value + 0.1 if positives else np.nan,
                }
                row.update(
                    {
                        metric: (value if not positives else np.nan)
                        for metric in ANOMALY_FREE_METRICS
                    }
                )
                metric_rows.append(row)
    return pd.DataFrame(index_rows), pd.DataFrame(metric_rows)


def test_reporting_schemas_and_static_structural_audit(tmp_path: Path) -> None:
    plan, registry = _report_plan_registry()
    index, metrics = _report_inputs()
    mask = freeze_valid_series_mask(index)
    bundle = aggregate_metrics(metrics, mask, registry)
    arm_table = arm_registry_frame(registry, plan)

    table3 = table3_style_frame(bundle, arm_table)
    assert len(table3) == len(REPORT_ARMS) * 3
    assert {"NAB", "NASA", "Yahoo", "equal11"} <= set(table3.columns)
    long11 = long11_frame(bundle, arm_table)
    assert len(long11) == len(REPORT_ARMS) * 11 * 7
    factorial = factorial_2x2_frame(bundle, arm_table)
    assert len(factorial) == 4 * 3
    assert set(zip(factorial["ihp"], factorial["nctp"])) == {
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    }
    tidy = plot_tidy_frame(bundle, arm_table)
    assert set(tidy["view"]) == {"subgroup11", "family3", "equal11", "fileweighted"}
    assert tidy["valid_fraction"].between(0.0, 1.0).all()

    audit = structural_audit_frame()
    released = audit.loc[
        (audit["audit_type"] == "incidence") & (audit["operator"] == "released")
    ]
    literal = audit.loc[
        (audit["audit_type"] == "incidence") & (audit["operator"] == "literal")
    ]
    temporal = audit.loc[
        (audit["audit_type"] == "temporal_operator")
        & (audit["availability"] == "static_core")
    ]
    deferred = audit.loc[
        audit["availability"].isin(["requires_frozen_trace_npz", "not_applicable"])
    ]
    assert set(released["effective_patch_count"]) == {195}
    assert set(released["row_crossings"]) == {13.0}
    assert set(released["terminal_holes"]) == {1.0}
    assert set(literal["effective_patch_count"]) == {196}
    assert set(literal["row_crossings"]) == {0.0}
    assert set(literal["terminal_holes"]) == {0.0}
    assert set(temporal["shape_rows"]) == {240}
    assert set(temporal["shape_columns"]) == {196}
    assert temporal["nonnegative"].all()
    assert np.allclose(temporal["row_sum_min"], 1.0)
    assert np.allclose(temporal["row_sum_max"], 1.0)
    assert set(temporal["zero_rows"]) == {0}
    assert set(deferred["operator"]) == {"trace_soft", "trace_hard", "legacy"}
    assert deferred["na_reason"].str.len().gt(0).all()

    supplied = np.zeros((240, 196), dtype=np.float64)
    supplied[:, 0] = 1.0
    with_trace = structural_audit_frame(
        extra_temporal_operators={"trace_soft": supplied}
    )
    trace = with_trace.loc[with_trace["operator"] == "trace_soft"].iloc[0]
    assert trace["availability"] == "provided_read_only" and trace["zero_rows"] == 0

    outputs = write_reporting_outputs(
        tmp_path / "report",
        bundle,
        registry,
        plan,
        provenance={"evaluation_marker_sha256": "A" * 64},
    )
    names = {path.name for path in outputs}
    assert {
        "table3_style.csv",
        "long11.csv",
        "ihp_nctp_2x2.csv",
        "plot_tidy.csv",
        "structural_audit.csv",
        "_REPORTING_COMPLETE.json",
    } <= names
    assert "hierarchical_bootstrap.csv" not in names
    marker = json.loads(
        (tmp_path / "report" / "_REPORTING_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert marker["bootstrap_executed"] is False
