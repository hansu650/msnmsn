from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import measure_vit4ts_v3.cache_runner as cache_runner
from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts_v3.cache_registry import build_cache_only_plan, sha256_file
from measure_vit4ts_v3.core import (
    FLOAT32_MATCH_CHUNK_ATOL,
    build_candidate_mask,
    streamed_all_pairs_median_match,
    streamed_median_reference_match,
)
from measure_vit4ts_v3.cache_runner import (
    CacheOnlyScorer,
    ConfigBundle,
    TraceOperators,
    _atomic_json,
    _commit_aliases,
    _commit_canonical,
    _verify_complete_series,
    run_cache_only,
    run_series,
    validate_parity_gate,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "vittrace_ablation_full_v3.yaml"


def _record(windows: int = 2) -> FullSeriesRecord:
    return FullSeriesRecord(
        series_id="synthetic__signal",
        dataset="artificialWithAnomaly",
        track="NAB",
        paper_group="NAB-Artificial",
        signal_name="signal",
        relative_path="artificialWithAnomaly/signal.csv",
        expected_length=240 + (windows - 1) * 60,
        expected_windows=windows,
        expected_sha256="A" * 64,
        duplicate_timestamps=False,
    )


def _pooling_masks() -> tuple[np.ndarray, np.ndarray]:
    mid = []
    for row in range(13):
        for column in range(13):
            mid.append(
                [
                    row * 14 + column,
                    row * 14 + column + 1,
                    (row + 1) * 14 + column,
                    (row + 1) * 14 + column + 1,
                ]
            )
    large = []
    for row in range(12):
        for column in range(12):
            large.append(
                [
                    (row + dy) * 14 + column + dx
                    for dy in range(3)
                    for dx in range(3)
                ]
            )
    return np.asarray(mid, dtype=np.int64).T, np.asarray(large, dtype=np.int64).T


def _cache(windows: int = 2) -> SimpleNamespace:
    rng = np.random.default_rng(2027)
    mid_mask, large_mask = _pooling_masks()
    return SimpleNamespace(
        patch_tokens=rng.normal(size=(windows, 196, 8)).astype(np.float32),
        mid_tokens=rng.normal(size=(windows, 169, 8)).astype(np.float32),
        large_tokens=rng.normal(size=(windows, 144, 8)).astype(np.float32),
        mid_mask=mid_mask,
        large_mask=large_mask,
    )


def _trace(tmp_path: Path, windows: int = 2) -> TraceOperators:
    winner = np.floor(np.arange(240, dtype=np.float64) * 196.0 / 240.0).astype(
        np.int64
    )
    hard = np.tile(winner, (windows, 1))
    data = np.ones(windows * 240, dtype=np.float64)
    indices = np.tile(winner, windows)
    indptr = np.tile(np.arange(241, dtype=np.int64), (windows, 1))
    offsets = np.arange(windows + 1, dtype=np.int64) * 240
    path = tmp_path / "trace.npz"
    manifest = tmp_path / "trace.json"
    path.write_bytes(b"trace")
    manifest.write_text("{}\n", encoding="utf-8")
    return TraceOperators(
        path=path,
        manifest_path=manifest,
        sha256="B" * 64,
        manifest_sha256="C" * 64,
        window_starts=np.arange(windows, dtype=np.int64) * 60,
        soft_data=data,
        soft_indices=indices,
        soft_indptr=indptr,
        soft_offsets=offsets,
        hard_winner=hard,
    )


def test_full_cache_only_grid_scores_without_encoder(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    plan = build_cache_only_plan(config)
    scorer = CacheOnlyScorer(
        _cache(),
        _trace(tmp_path),
        _record(),
        torch.device("cpu"),
    )
    scorer.prepare_matching()
    assert len(scorer.match_fields) == 24
    for planned in plan.canonical_arms:
        score = scorer.score(planned.logical)
        assert score.shape == (_record().expected_length,)
        assert score.dtype == np.float64
        assert np.isfinite(score).all()


def test_multi_scope_matching_is_exact_and_shares_cosine_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(3107)
    tokens = torch.randn((5, 6, 7), generator=generator, dtype=torch.float32)
    masks = {
        scope: build_candidate_mask((2, 3), scope)
        for scope in ("position", "row", "column", "global")
    }
    expected_median = {
        scope: streamed_median_reference_match(
            tokens, mask, query_chunk_size=2
        ).cost
        for scope, mask in masks.items()
    }
    expected_all_pairs = {
        scope: streamed_all_pairs_median_match(
            tokens,
            mask,
            query_chunk_size=2,
            reference_chunk_size=3,
        ).cost
        for scope, mask in masks.items()
    }

    original_matmul = torch.matmul
    original_einsum = torch.einsum
    matmul_calls = 0
    pair_shapes: list[tuple[int, int]] = []

    def counted_matmul(*args, **kwargs):
        nonlocal matmul_calls
        matmul_calls += 1
        return original_matmul(*args, **kwargs)

    def counted_einsum(equation, query, reference):
        pair_shapes.append((int(query.shape[0]), int(reference.shape[0])))
        return original_einsum(equation, query, reference)

    monkeypatch.setattr(cache_runner.torch, "matmul", counted_matmul)
    monkeypatch.setattr(cache_runner.torch, "einsum", counted_einsum)
    actual_median = cache_runner._multi_scope_median_reference_cost(
        tokens, masks, query_chunk_size=2
    )
    actual_all_pairs = cache_runner._multi_scope_all_pairs_median_cost(
        tokens, masks, query_chunk_size=2, reference_chunk_size=3
    )

    assert matmul_calls == 3
    assert len(pair_shapes) == 6
    assert max(q for q, _ in pair_shapes) <= 2

    assert max(r for _, r in pair_shapes) <= 3
    for scope in masks:
        torch.testing.assert_close(
            actual_median[scope], expected_median[scope],
            rtol=0.0, atol=FLOAT32_MATCH_CHUNK_ATOL,
        )
        torch.testing.assert_close(
            actual_all_pairs[scope], expected_all_pairs[scope],
            rtol=0.0, atol=FLOAT32_MATCH_CHUNK_ATOL,
        )

def test_full_parity_gate_must_bind_active_core(tmp_path: Path) -> None:
    record = _record(1)
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.json"
    config_path.write_text("stage: synthetic\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    bundle = ConfigBundle(
        config_path,
        {},
        "D" * 64,
        manifest_path,
        "E" * 64,
        (record,),
    )
    from measure_vit4ts_v3 import cache_runner

    gate = {
        "schema_version": 1,
        "decision": "PASS",
        "passed": True,
        "expected_series": 1,
        "completed_series": 1,
        "config_sha256": bundle.sha256,
        "manifest_sha256": bundle.manifest_sha256,
        "core_sha256": sha256_file(Path(cache_runner.__file__).parent / "core.py"),
        "arms": {
            arm: {
                "passed": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "source_paths": [],
            }
            for arm in ("REL_U", "IHP_LEGACY", "FULL_COLUMN_240")
        },
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert validate_parity_gate(gate_path, bundle)["passed"] is True

    gate["completed_series"] = 0
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        validate_parity_gate(gate_path, bundle)
    gate["completed_series"] = 1

    gate["core_sha256"] = "0" * 64
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="active v3 core"):
        validate_parity_gate(gate_path, bundle)


def test_canonical_scores_and_aliases_are_transactional(tmp_path: Path) -> None:
    plan = build_cache_only_plan(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    record = _record(1)
    root = tmp_path / record.series_id
    provenance = {
        "config_sha256": "1" * 64,
        "full_manifest_sha256": "2" * 64,
        "compute_plan_sha256": "3" * 64,
        "parity_gate_sha256": "4" * 64,
        "source_sha256": "5" * 64,
        "encoder_calls": 0,
    }
    score = np.linspace(0.0, 1.0, record.expected_length, dtype=np.float64)
    for planned in plan.canonical_arms:
        _commit_canonical(
            root,
            record,
            planned.logical,
            planned.parameter_sha256,
            score,
            provenance,
            0.01,
        )
    _commit_aliases(root, record, plan, provenance)

    for planned in plan.arms:
        arm_root = root / planned.logical.arm_id
        if planned.is_alias:
            assert (arm_root / "alias_manifest.json").is_file()
            assert not (arm_root / "score.npy").exists()
        else:
            assert (arm_root / "score.npy").is_file()

    _atomic_json(
        root / "_SUCCESS.json",
        {
            "schema_version": 1,
            "series_id": record.series_id,
            "logical_arm_count": len(plan.arms),
            "unique_computation_count": len(plan.canonical_arms),
            "encoder_calls": 0,
            "config_sha256": provenance["config_sha256"],
            "compute_plan_sha256": provenance["compute_plan_sha256"],
            "source_sha256": provenance["source_sha256"],
        },
    )
    _atomic_json(
        root / "runtime.json",
        {
            "series_id": record.series_id,
            "shared_matching_seconds": 0.1,
            "canonical_arm_seconds": {
                item.logical.arm_id: 0.01 for item in plan.canonical_arms
            },
            "series_wall_seconds": 0.5,
            "python_tracemalloc_current_bytes": 1,
            "python_tracemalloc_peak_bytes": 2,
            "process_rss_before_bytes": 3,
            "process_rss_after_bytes": 4,
            "encoder_calls": 0,
        },
    )
    _verify_complete_series(root, record, plan, provenance)


def test_run_checks_parity_gate_before_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = ConfigBundle(
        tmp_path / "config.yaml",
        {},
        "1" * 64,
        tmp_path / "manifest.json",
        "2" * 64,
        (_record(1),),
    )
    calls: list[str] = []

    monkeypatch.setattr(cache_runner, "_load_config", lambda _: bundle)

    def reject_gate(*_: object) -> None:
        calls.append("gate")
        raise RuntimeError("parity gate rejected")

    def reject_plan(*_: object) -> None:
        calls.append("plan")
        raise AssertionError("registry must not be read before parity")

    monkeypatch.setattr(cache_runner, "validate_parity_gate", reject_gate)
    monkeypatch.setattr(cache_runner, "_validate_plan_and_registry", reject_plan)
    with pytest.raises(RuntimeError, match="parity gate rejected"):
        run_cache_only(
            tmp_path / "config.yaml",
            tmp_path / "registry.json",
            tmp_path / "plan.json",
            tmp_path / "gate.json",
        )
    assert calls == ["gate"]


def test_failure_manifest_is_retained_and_blocks_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_path = data_root / "artificialWithAnomaly" / "signal.csv"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("timestamp,value\n0,0\n", encoding="utf-8")
    original = _record(1)
    record = FullSeriesRecord(
        series_id=original.series_id,
        dataset=original.dataset,
        track=original.track,
        paper_group=original.paper_group,
        signal_name=original.signal_name,
        relative_path=original.relative_path,
        expected_length=original.expected_length,
        expected_windows=original.expected_windows,
        expected_sha256=sha256_file(data_path),
        duplicate_timestamps=False,
    )
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.json"
    plan_path = tmp_path / "plan.json"
    gate_path = tmp_path / "gate.json"
    config_path.write_text("stage: synthetic\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    plan_path.write_text("{}\n", encoding="utf-8")
    gate_path.write_text("{}\n", encoding="utf-8")
    payload = {
        "runtime": {
            "device": "cpu",
            "c_drive_floor_gib": 0,
            "d_drive_floor_gib": 0,
            "available_ram_floor_gib": 0,
        },
        "paths": {
            "run_root": str(tmp_path / "runs"),
            "failure_root": str(tmp_path / "failures"),
        },
        "data": {"root": str(data_root)},
    }
    bundle = ConfigBundle(
        config_path,
        payload,
        sha256_file(config_path),
        manifest_path,
        sha256_file(manifest_path),
        (record,),
    )
    plan = build_cache_only_plan(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    monkeypatch.setattr(cache_runner, "_validate_resource_headroom", lambda _: None)

    def fail_cache(*_: object) -> None:
        raise RuntimeError("synthetic cache failure")

    monkeypatch.setattr(cache_runner, "_load_frozen_cache", fail_cache)
    with pytest.raises(RuntimeError, match="synthetic cache failure"):
        run_series(bundle, plan, plan_path, gate_path, record)

    series_root = Path(payload["paths"]["run_root"]) / cache_runner.RUN_NAME / record.series_id
    retained = (
        Path(payload["paths"]["failure_root"])
        / cache_runner.RUN_NAME
        / f"{record.series_id}.json"
    )
    assert (series_root / "_FAILED.json").is_file()
    assert retained.is_file()
    assert not (series_root / "_RUNNING.json").exists()
    failure = json.loads(retained.read_text(encoding="utf-8"))
    assert failure["encoder_calls"] == 0
    assert failure["error"] == "synthetic cache failure"

    with pytest.raises(RuntimeError, match="retained failure blocks rerun"):
        run_series(bundle, plan, plan_path, gate_path, record)


def test_runner_source_contains_no_encoder_instantiation() -> None:
    source = (REPO / "code" / "src" / "measure_vit4ts_v3" / "cache_runner.py").read_text(
        encoding="utf-8"
    )
    assert "open_clip" not in source
    assert "load_frozen_clip" not in source
    assert "encoder_calls" in source
