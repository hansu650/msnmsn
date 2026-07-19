from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts_v3.combined_aggregate import aggregate_combined
from measure_vit4ts_v3.combined_evaluator import (
    CombinedBlocked,
    CombinedPreflight,
    CombinedScoreRecord,
    StagePreflight,
    _failure_paths,
    _identity,
    _status_stage,
    evaluate_combined,
    write_evaluation_outputs,
)
from measure_vit4ts_v3.combined_protocol import (
    CombinedArmSpec,
    CombinedContrastSpec,
    CombinedProtocolSpec,
    CombinedStageSpec,
    canonical_json_sha256,
)


GROUPS = tuple(
    [("NAB", f"NAB-G{i}") for i in range(5)]
    + [("NASA", f"NASA-G{i}") for i in range(2)]
    + [("Yahoo", f"Yahoo-G{i}") for i in range(4)]
)


def _record(tmp_path: Path, index: int, family: str, subgroup: str) -> FullSeriesRecord:
    data = tmp_path / "data" / f"s{index}.csv"
    data.parent.mkdir(exist_ok=True)
    data.write_text("timestamp,value\n0,0\n1,1\n2,0\n3,1\n", encoding="utf-8")
    from measure_vit4ts_v3.combined_protocol import sha256_file

    return FullSeriesRecord(
        f"s{index}",
        f"d{index}",
        family,
        subgroup,
        f"signal{index}",
        data.name,
        4,
        1,
        sha256_file(data),
        False,
    )


def _arm(arm: str, order: int) -> CombinedArmSpec:
    return CombinedArmSpec(
        arm,
        arm,
        "method" if order == 0 else "control",
        order,
        0.0,
        "TEST",
        "TEST",
        {},
        {"representation": "line", "encoder_calls": 0},
    )


def _protocol(tmp_path: Path, records: tuple[FullSeriesRecord, ...]) -> CombinedProtocolSpec:
    config = tmp_path / "config.yaml"
    manifest = tmp_path / "manifest.json"
    config.write_text("stage: vittrace_ablation_full_v3\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    arms = (_arm("A", 0), _arm("B", 1))
    stage = CombinedStageSpec(
        "TEST_STAGE",
        "test",
        "encoder_controls",
        "TEST_CONFIG",
        tmp_path / "scores",
        None,
        len(records),
        arms,
    )
    payload_hash = canonical_json_sha256({"test": "protocol"})
    from measure_vit4ts_v3.combined_protocol import sha256_file

    return CombinedProtocolSpec(
        "TEST_COMBINED",
        config,
        sha256_file(config),
        manifest,
        sha256_file(manifest),
        len(records),
        len(records),
        (stage,),
        (CombinedContrastSpec("A_VS_B", "TEST", "A", "B"),),
        2027,
        10000,
        payload_hash,
    )


def _ready_preflight(tmp_path: Path) -> CombinedPreflight:
    records = tuple(_record(tmp_path, index, *group) for index, group in enumerate(GROUPS))
    protocol = _protocol(tmp_path, records)
    scores: dict[tuple[str, str], CombinedScoreRecord] = {}
    for record in records:
        for arm, value in (("A", 0.8), ("B", 0.2)):
            root = tmp_path / "scores" / record.series_id / arm
            root.mkdir(parents=True)
            score_path = root / "score.npy"
            np.save(score_path, np.asarray([0.1, value, value, 0.1], dtype=np.float64))
            manifest_path = root / "score_manifest.json"
            manifest_path.write_text(json.dumps({"arm": arm}) + "\n", encoding="utf-8")
            scores[(record.series_id, arm)] = CombinedScoreRecord(
                record.series_id,
                "TEST_STAGE",
                "encoder_controls",
                arm,
                arm,
                score_path,
                _identity(score_path),
                manifest_path,
                _identity(manifest_path),
                {"arm": arm},
            )
    data_files = {
        record.series_id: _identity(tmp_path / "data" / record.relative_path)
        for record in records
    }
    stage = StagePreflight(
        protocol.stages[0],
        "READY",
        "test ready",
        len(records),
        len(records),
        len(records) * 2,
        len(records) * 2,
        None,
        "A" * 64,
        scores,
    )
    config = {
        "data": {"root": str(tmp_path / "data"), "anomalies_csv": str(tmp_path / "a.csv")},
        "statistics": {"vus_max_window": 2},
    }
    return CombinedPreflight(
        protocol,
        config,
        records,
        data_files,
        (stage,),
        "READY",
        "ready",
        "B" * 64,
    )


def test_evaluator_failure_audit_does_not_poison_corrected_rerun(tmp_path: Path) -> None:
    records = tuple(_record(tmp_path, index, *group) for index, group in enumerate(GROUPS[:2]))
    protocol = _protocol(tmp_path, records)
    failure_root = tmp_path / "failures"
    audit = failure_root / protocol.stages[0].stage_id / "evaluator" / "old.json"
    audit.parent.mkdir(parents=True)
    audit.write_text("{}\n", encoding="utf-8")
    config = {"paths": {"failure_root": str(failure_root)}}
    assert _failure_paths(config, protocol.stages[0]) == ()

    scorer_failure = audit.parent.parent / "scorer.json"
    scorer_failure.write_text("{}\n", encoding="utf-8")
    assert _failure_paths(config, protocol.stages[0]) == (scorer_failure.resolve(),)


def test_blocked_preflight_never_calls_ground_truth(tmp_path: Path) -> None:
    ready = _ready_preflight(tmp_path)
    blocked = CombinedPreflight(
        ready.protocol,
        ready.config,
        ready.records,
        ready.data_files,
        (StagePreflight(ready.stages[0].stage, "BLOCKED", "incomplete", 11, 10, 22, 20, None, None, {}),),
        "BLOCKED",
        "stage incomplete",
        None,
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("labels must not be loaded")

    try:
        evaluate_combined(blocked, ground_truth_loader=forbidden)
    except CombinedBlocked:
        pass
    else:  # pragma: no cover
        raise AssertionError("blocked evaluation must raise")
    assert calls == []


def test_status_stage_is_incremental_and_hash_checked(tmp_path: Path) -> None:
    records = tuple(_record(tmp_path, index, *group) for index, group in enumerate(GROUPS[:2]))
    base = _protocol(tmp_path, records)
    arm = CombinedArmSpec("CTRL", "CTRL", "control", 0, 0.0, "TEST", "TEST", {}, {})
    stage = CombinedStageSpec(
        "CTRL_STAGE",
        "encoder_controls",
        "encoder_controls",
        "CTRL_CONFIG",
        tmp_path / "control",
        tmp_path / "control" / "encoder_controls_status.json",
        2,
        (arm,),
    )
    status = {
        "status": "INCOMPLETE",
        "expected_series": 2,
        "completed_series": 1,
        "expected_rows": 2,
        "completed_rows": 1,
        "expected_arms": ["CTRL"],
        "config_sha256": base.config_sha256,
        "manifest_sha256": base.manifest_sha256,
        "encoder_source_sha256": "E" * 64,
        "control_source_sha256": "C" * 64,
        "variant_sha256": "D" * 64,
        "rows": [{"series_id": records[0].series_id, "arm": "CTRL", "status": "PASS"}],
    }
    stage.status_path.parent.mkdir(parents=True)
    stage.status_path.write_text(json.dumps(status), encoding="utf-8")
    blocked = _status_stage(
        base,
        records,
        stage,
        status_name="encoder_controls_status.json",
        transaction_directory="controls",
        score_hash_field="score_sha256",
        source_hash_field="control_source_sha256",
    )
    assert blocked.status == "BLOCKED"
    assert blocked.completed_rows == 1

    rows = []
    for record in records:
        root = stage.root / "controls" / record.series_id / "CTRL"
        root.mkdir(parents=True)
        score_path = root / "score.npy"
        np.save(score_path, np.arange(4, dtype=np.float64))
        from measure_vit4ts_v3.combined_protocol import sha256_file

        (root / "score_manifest.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "series_id": record.series_id,
                    "arm": "CTRL",
                    "data_sha256": record.expected_sha256,
                    "config_sha256": base.config_sha256,
                    "manifest_sha256": base.manifest_sha256,
                    "encoder_source_sha256": "E" * 64,
                    "control_source_sha256": "C" * 64,
                    "variant_sha256": "D" * 64,
                    "score_path": str(score_path),
                    "score_sha256": sha256_file(score_path),
                }
            ),
            encoding="utf-8",
        )
        rows.append({"series_id": record.series_id, "arm": "CTRL", "status": "PASS"})
    status.update(
        {"status": "COMPLETE", "completed_series": 2, "completed_rows": 2, "rows": rows}
    )
    stage.status_path.write_text(json.dumps(status), encoding="utf-8")
    ready = _status_stage(
        base,
        records,
        stage,
        status_name="encoder_controls_status.json",
        transaction_directory="controls",
        score_hash_field="score_sha256",
        source_hash_field="control_source_sha256",
    )
    assert ready.status == "READY"
    assert len(ready.scores) == 2

    score_path = next(iter(ready.scores.values())).score_path
    np.save(score_path, np.ones(4, dtype=np.float64))
    corrupt = _status_stage(
        base,
        records,
        stage,
        status_name="encoder_controls_status.json",
        transaction_directory="controls",
        score_hash_field="score_sha256",
        source_hash_field="control_source_sha256",
    )
    assert corrupt.status == "BLOCKED"


def test_combined_outputs_and_shared_bootstrap(tmp_path: Path) -> None:
    preflight = _ready_preflight(tmp_path)
    labels = np.asarray([0, 1, 0, 0], dtype=np.uint8)

    def timestamp_loader(record, root):
        return np.arange(record.expected_length, dtype=np.float64)

    def ground_truth_loader(config, series_id, timestamps):
        return SimpleNamespace(point_labels=labels, intervals=((1.0, 1.0),))

    result = evaluate_combined(
        preflight,
        timestamp_loader=timestamp_loader,
        ground_truth_loader=ground_truth_loader,
        f1_fn=lambda scores, timestamps, intervals, alpha: (float(np.max(scores)), 0.1, 0.5),
        auprc_fn=lambda labels, scores: float(np.mean(scores)),
        vus_fn=lambda labels, scores, window: float(np.mean(scores)),
    )
    assert len(result.per_series) == 22
    assert set(result.valid_mask["valid_auprc"]) == {True}
    evaluation = tmp_path / "evaluation"
    write_evaluation_outputs(evaluation, preflight, result)
    registry_path = tmp_path / "combined.json"
    # aggregate_combined loads a validated JSON registry; the production path
    # is covered by init-current/preflight integration tests.  Here we verify
    # the aggregation primitive directly to keep synthetic fixtures compact.
    from measure_vit4ts_v3.aggregate import aggregate_metrics, paired_hierarchical_bootstrap
    from measure_vit4ts_v3.combined_evaluator import arm_registry

    registry = arm_registry(preflight.protocol)
    bundle = aggregate_metrics(result.per_series, result.valid_mask, registry)
    assert bundle.subgroup11["subgroup"].nunique() == 11
    assert bundle.family3["family"].nunique() == 3
    assert len(bundle.equal11) > 0 and len(bundle.fileweighted) > 0
    auprc = paired_hierarchical_bootstrap(
        result.per_series, result.valid_mask, registry, "auprc", n_boot=8, seed=2027
    )
    vus = paired_hierarchical_bootstrap(
        result.per_series, result.valid_mask, registry, "vus_pr", n_boot=8, seed=2027
    )
    assert auprc.iloc[0]["resample_plan_sha256"] == vus.iloc[0]["resample_plan_sha256"]
    assert (evaluation / "stage_evaluation_index.json").is_file()
