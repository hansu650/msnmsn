from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from measure_vit4ts_v3.cross_stage_outputs import (
    ARM_METADATA_COLUMNS,
    assemble_cross_stage_outputs,
    load_stage_index,
)
from measure_vit4ts_v3.delivery_assembler import (
    _local_status,
    build_cache_index,
    build_failure_manifest,
    combined_arm_registry,
)
from measure_vit4ts_v3.microbenchmark_runner import benchmark_callable
from measure_vit4ts_v3.qualitative_runner import (
    FIELD_FILE,
    FIELD_MANIFEST,
    load_field_cache,
    preselected_qualitative_cases,
)


SUBGROUPS = (
    ("NAB", "NAB-Artificial"),
    ("NAB", "NAB-AWS"),
    ("NAB", "NAB-AdExchange"),
    ("NAB", "NAB-Traffic"),
    ("NAB", "NAB-Tweets"),
    ("NASA", "NASA-MSL"),
    ("NASA", "NASA-SMAP"),
    ("Yahoo", "Yahoo-A1"),
    ("Yahoo", "Yahoo-A2"),
    ("Yahoo", "Yahoo-A3"),
    ("Yahoo", "Yahoo-A4"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _stage(tmp_path: Path) -> Path:
    rows = []
    for group_index, (family, subgroup) in enumerate(SUBGROUPS):
        for series_index in range(2):
            series_id = f"{subgroup}__{series_index}"
            for arm, offset in (("REL", 0.0), ("FULL", 0.1)):
                rows.append(
                    {
                        "series_id": series_id,
                        "family": family,
                        "subgroup": subgroup,
                        "arm": arm,
                        "experiment_group": "DEFAULT",
                        "changed_factor": "ihp_nctp",
                        "n_points": 100,
                        "n_positive": 0 if series_index == 0 else 1,
                        "anomaly_free_fp_rate": 0.05 if series_index == 0 else np.nan,
                        "anomaly_free_fp_count": 5 if series_index == 0 else np.nan,
                        "anomaly_free_mean_excess": 0.01 if series_index == 0 else np.nan,
                        "anomaly_free_score_p95": 0.2 if series_index == 0 else np.nan,
                        "f1_max": group_index / 100 + offset,
                        "auprc": 0.2 + offset,
                        "vus_pr": 0.3 + offset,
                    }
                )
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics, index=False)
    metadata = pd.DataFrame(
        [
            {
                "arm": arm,
                "arm_metadata_json": json.dumps({"display_name": arm, "is_final": arm == "FULL"}),
                "experiment_group": "DEFAULT",
                "changed_factor": "ihp_nctp",
                "fixed_factors": "B16/W240/S60",
                "ihp": int(arm == "FULL"),
                "nctp": int(arm == "FULL"),
            }
            for arm in ("REL", "FULL")
        ]
    )
    arm_path = tmp_path / "arms.csv"
    metadata.to_csv(arm_path, index=False)
    marker = tmp_path / "marker.json"
    marker.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    payload = {
        "schema_version": 1,
        "manifest_sha256": "A" * 64,
        "stages": [
            {
                "stage_id": "base",
                "stage_group": "cache_only_controls",
                "configuration_id": "B16_W240_S60",
                "status": "COMPLETE",
                "reason": "stage metrics materialized",
                "metrics_path": str(metrics),
                "metrics_sha256": _sha(metrics),
                "marker_path": str(marker),
                "marker_sha256": _sha(marker),
                "arm_metadata_path": str(arm_path),
                "arm_metadata_sha256": _sha(arm_path),
                "expected_series": 22,
                "stage_kind": "cache_only",
                "completed_rows": 44,
            }
        ],
    }
    index = tmp_path / "stage_index.json"
    index.write_text(json.dumps(payload), encoding="utf-8")
    return index


def test_cross_stage_tables_are_real_and_missing_stages_block(tmp_path: Path) -> None:
    index = _stage(tmp_path)
    protocol, stages = load_stage_index(index)
    assert protocol == "A" * 64 and stages[0].stage_id == "base"
    outputs = assemble_cross_stage_outputs(index, tmp_path / "out")
    assert any(path.name == "_CROSS_STAGE_BLOCKED.json" for path in outputs)
    table = pd.read_csv(tmp_path / "out" / "tables" / "table3_style_all_stages.csv")
    full = table.loc[(table["arm"] == "FULL") & (table["metric"] == "vus_pr")].iloc[0]
    assert full["NAB"] == pytest.approx(0.4)
    assert full["NASA"] == pytest.approx(0.4)
    assert full["Yahoo"] == pytest.approx(0.4)
    assert full["equal11"] == pytest.approx(0.4)
    assert full["display_name"] == "FULL"
    assert bool(full["is_final"])
    assert full["experiment_group"] == "DEFAULT"
    registry, provenance = combined_arm_registry(index)
    registry_full = registry.loc[registry["arm"] == "FULL"].iloc[0]
    anomaly_free = pd.read_csv(tmp_path / "out" / "results" / "anomaly_free_fp_burden.csv")
    assert len(anomaly_free) == 22
    assert set(anomaly_free["n_positive"]) == {0}
    assert set(anomaly_free["anomaly_free_fp_count"]) == {5}
    assert bool(registry_full["is_final"])
    assert provenance["record_count"] == 2
    long11 = pd.read_csv(tmp_path / "out" / "tables" / "table_long11_all_stages.csv")
    assert {"Art", "AWS", "AdEx", "Traf", "Tweets", "MSL", "SMAP", "A1", "A2", "A3", "A4"}.issubset(long11.columns)


def test_stale_stage_hash_fails_closed(tmp_path: Path) -> None:
    index = _stage(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    Path(payload["stages"][0]["metrics_path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        assemble_cross_stage_outputs(index, tmp_path / "out")


def test_structural_case_is_preselected_before_metric_best_worst() -> None:
    series = ("MSL__C-1", "structural", "best", "worst", "spare")
    deltas = {"MSL__C-1": 0.0, "structural": 5.0, "best": 1.0, "worst": -1.0, "spare": 0.2}
    rows = []
    for series_id in series:
        rows.extend(
            [
                {"series_id": series_id, "arm": "REL", "vus_pr": 0.2},
                {"series_id": series_id, "arm": "FULL", "vus_pr": 0.2 + deltas[series_id]},
            ]
        )
    structural = pd.DataFrame(
        {
            "series_id": series,
            "boundary_terminal_score": [0.0, 10.0, 1.0, 2.0, 3.0],
        }
    )
    cases = preselected_qualitative_cases(
        pd.DataFrame(rows),
        structural,
        candidate_arm="FULL",
        control_arm="REL",
        metric="vus_pr",
        structural_series_id="structural",
    )
    assert tuple(cases["series_id"]) == ("MSL__C-1", "best", "worst", "structural")
    assert tuple(cases["uses_evaluation_labels"]) == (False, True, True, False)


def test_hash_bound_field_cache_detects_tampering(tmp_path: Path) -> None:
    series_root = tmp_path / "S"
    series_root.mkdir()
    payload_path = series_root / FIELD_FILE
    with payload_path.open("wb") as handle:
        np.savez(handle, released_field=np.zeros((2, 4)), literal_field=np.ones((2, 4)))
    source_cache_sha = "B" * 64
    source_sha = "C" * 64
    manifest = {
        "schema_version": 1,
        "series_id": "S",
        "shape": [2, 4],
        "source_cache_manifest_sha256": source_cache_sha,
        "source_sha256": source_sha,
        "field_payload_sha256": _sha(payload_path),
        "labels_read": False,
    }
    (series_root / FIELD_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    released, literal, _ = load_field_cache(
        tmp_path,
        "S",
        source_cache_manifest_sha256=source_cache_sha,
        expected_source_sha256=source_sha,
    )
    assert released.shape == literal.shape == (2, 4)
    payload_path.write_bytes(payload_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_field_cache(
            tmp_path,
            "S",
            source_cache_manifest_sha256=source_cache_sha,
            expected_source_sha256=source_sha,
        )


def test_microbenchmark_is_exact_5_plus_30_and_score_preserving() -> None:
    reference = np.asarray([0.1, 0.2, 0.3])
    frame = benchmark_callable(
        lambda: reference.copy(),
        reference,
        identity={
            "experiment_id": "TEST",
            "arm": "FULL",
            "series_id": "MSL__C-1",
            "family": "NASA",
            "subgroup": "NASA-MSL",
        },
    )
    assert len(frame) == 35
    assert int(frame["is_warmup"].sum()) == 5
    assert set(frame["encoder_calls"]) == {0}
    with pytest.raises(RuntimeError, match="differs"):
        benchmark_callable(
            lambda: reference + 1,
            reference,
            identity={
                "experiment_id": "TEST",
                "arm": "FULL",
                "series_id": "MSL__C-1",
                "family": "NASA",
                "subgroup": "NASA-MSL",
            },
        )


def test_failure_manifest_and_cache_index_preserve_evidence(tmp_path: Path) -> None:
    failures = tmp_path / "failures"
    failures.mkdir()
    (failures / "one.json").write_text(
        json.dumps({"stage": "s", "series_id": "x", "status": "FAILED", "error": "boom"}),
        encoding="utf-8",
    )
    frame = build_failure_manifest([failures], tmp_path / "delivery" / "failures")
    assert len(frame) == 1 and frame.iloc[0]["reason"] == "boom"

    cache = tmp_path / "cache" / "x" / "key"
    cache.mkdir(parents=True)
    payload = cache / "vision_tokens_v3.npz"
    payload.write_bytes(b"tokens")
    (cache / "vision_tokens_v3.json").write_text(
        json.dumps(
            {
                "key": {"series_id": "x", "renderer": "line"},
                "file": payload.name,
                "sha256": _sha(payload),
            }
        ),
        encoding="utf-8",
    )
    index = build_cache_index([tmp_path / "cache"], tmp_path / "delivery" / "caches" / "cache_index.csv")
    assert len(index) == 1 and bool(index.iloc[0]["payload_present"])
    assert index.iloc[0]["payload_sha256_declared"] == _sha(payload)


def test_delivery_status_requires_hash_bound_10000_bootstrap_and_all_figures(
    tmp_path: Path,
) -> None:
    cross = tmp_path / "cross"
    experiment = tmp_path / "experiment"
    external = tmp_path / "external.csv"
    (cross / "manifests").mkdir(parents=True)
    (cross / "results").mkdir(parents=True)
    (cross / "plot_data").mkdir(parents=True)
    (experiment / "results" / "qualitative_plot_data").mkdir(parents=True)
    (experiment / "results" / "runtime").mkdir(parents=True)
    (cross / "manifests" / "_CROSS_STAGE_COMPLETE.json").write_text(
        '{"status":"COMPLETE"}\n', encoding="utf-8"
    )
    bootstrap = cross / "results" / "bootstrap_ci.csv"
    pd.DataFrame([{"contrast": "FULL-REL", "metric": "vus_pr", "delta": 0.01}]).to_csv(
        bootstrap, index=False
    )
    (cross / "results" / "_COMBINED_AGGREGATION_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "bootstrap_replicates": 10_000,
                "bootstrap_sha256": _sha(bootstrap),
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"figure": f"figure_{index}", "status": "COMPLETE"} for index in range(11)]
    ).to_csv(cross / "plot_data" / "rough_figure_status.csv", index=False)
    (experiment / "results" / "qualitative_plot_data" / "_QUALITATIVE_COMPLETE.json").write_text(
        "{}", encoding="utf-8"
    )
    (experiment / "results" / "runtime" / "_MICROBENCHMARK_COMPLETE.json").write_text(
        "{}", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "measurement_mode": "encoder_inclusive",
                "stage": "total",
                "aggregation": "config",
                "encoder_calls_max": 1,
            }
        ]
    ).to_csv(experiment / "results" / "runtime" / "runtime.csv", index=False)
    external.write_text("paper,value\nViT4TS,0.1\n", encoding="utf-8")
    failures = pd.DataFrame(columns=["failure_path"])
    caches = pd.DataFrame([{"payload_present": True, "hash_binding": "RUNNER_MANIFEST_DECLARED"}])
    status = _local_status(cross, experiment, external, failures, caches)
    assert len(status) == 9
    assert set(status["status"]) == {"COMPLETE"}

    marker_path = cross / "results" / "_COMBINED_AGGREGATION_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["bootstrap_replicates"] = 9_999
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    blocked = _local_status(cross, experiment, external, failures, caches)
    assert blocked.loc[blocked["requirement"] == "bootstrap_10000", "status"].item() == "BLOCKED"
