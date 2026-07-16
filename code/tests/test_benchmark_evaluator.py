from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import paano_k0.evaluate_benchmark as module
from paano_k0.schemas import (
    CheckpointKind,
    ScoreManifest,
    SeriesSpec,
    Trajectory,
    make_run_id,
)


CONFIG_SHA = "2" * 64
VENDOR_SHA = "3" * 40


def _spec(tmp_path: Path, series_id: str, track: str = "U") -> SeriesSpec:
    return SeriesSpec(
        series_id=series_id,
        family=f"family-{track}",
        track=track,
        csv_path=tmp_path / f"{series_id}.csv",
        csv_sha256="1" * 64,
        rows=4,
        channels=1,
        train_end=2,
        feature_columns=("value",),
        label_column="Label",
    )


def _manifest(spec: SeriesSpec, trajectory: Trajectory) -> ScoreManifest:
    return ScoreManifest(
        schema_version="paano-k0-score-v1",
        run_id=make_run_id(
            spec.series_id, 2027, trajectory, CheckpointKind.LAST
        ),
        series_id=spec.series_id,
        family=spec.family,
        track=spec.track,
        data_sha256=spec.csv_sha256,
        config_sha256=CONFIG_SHA,
        vendor_sha=VENDOR_SHA,
        seed=2027,
        trajectory=trajectory,
        checkpoint=CheckpointKind.LAST,
        initial_state_sha256="4" * 64,
        replay_sha256="5" * 64,
        checkpoint_sha256="6" * 64,
        num_points=spec.rows,
        num_train_patches=2,
        num_full_patches=4,
        channels=spec.channels,
        patch_size=2,
        stride=1,
        top_k=1,
        requested_memory_fraction=0.5,
        effective_memory_fraction=0.5,
        memory_count=1,
        memory_sha256="7" * 64,
        score_sha256="8" * 64,
        runtime_seconds=0.1,
        peak_vram_mib=1.0,
        sliding_window=2,
        labels_read=False,
    )


def _artifact_from_path(
    directory: Path, specs: dict[str, SeriesSpec]
) -> tuple[np.ndarray, ScoreManifest]:
    trajectory = Trajectory(directory.parents[1].name)
    spec = specs[directory.parents[3].name]
    scores = np.linspace(0.1, 0.9, spec.rows, dtype=np.float32)
    return scores, _manifest(spec, trajectory)


def _vendor(tmp_path: Path):
    root = tmp_path / "vendor"
    metric_path = root / "utils" / "basic_metrics.py"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text("# frozen test metric\n", encoding="utf-8")
    return SimpleNamespace(
        fingerprint=SimpleNamespace(root=root, git_sha=VENDOR_SHA)
    )


def _fixed_metrics(*_args, **_kwargs) -> dict[str, float]:
    return {
        "vus_pr": 0.6,
        "auprc": 0.5,
        "vus_roc": 0.7,
        "auroc": 0.8,
    }


@pytest.mark.parametrize(
    ("available", "requested", "expected"),
    ((41.0, 4, 4), (29.0, 4, 3), (28.9, 4, 2), (23.0, 4, 1)),
)
def test_worker_count_preserves_twenty_gib_ram_floor(
    monkeypatch: pytest.MonkeyPatch,
    available: float,
    requested: int,
    expected: int,
) -> None:
    monkeypatch.setattr(module, "_available_memory_gib", lambda: available)
    assert module._effective_worker_count(requested) == expected


def test_worker_count_refuses_to_cross_twenty_gib_ram_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "_available_memory_gib", lambda: 22.99)
    with pytest.raises(RuntimeError, match="20 GiB floor"):
        module._effective_worker_count(4)


@pytest.mark.parametrize("workers", (1, 2))
def test_all_score_hashes_preflight_before_first_label_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    specs = (_spec(tmp_path, "series-a"), _spec(tmp_path, "series-b", "M"))
    by_id = {spec.series_id: spec for spec in specs}
    verification_calls = 0
    label_calls = 0

    def verify(directory: Path):
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 2:
            raise ValueError("fixture score hash mismatch")
        return _artifact_from_path(directory, by_id)

    def read_labels(_spec: SeriesSpec):
        nonlocal label_calls
        label_calls += 1
        raise AssertionError("labels were read before global preflight completed")

    monkeypatch.setattr(module, "verify_committed_score", verify)
    monkeypatch.setattr(module, "read_labels", read_labels)

    with pytest.raises(ValueError, match="score hash mismatch"):
        module.evaluate_registered_benchmark(
            specs,
            tmp_path / "results",
            tmp_path / "evaluation",
            object(),
            (Trajectory.OFFICIAL,),
            seed=2027,
            expected_config_sha256=CONFIG_SHA,
            expected_vendor_sha=VENDOR_SHA,
            workers=workers,
            resume_existing=workers > 1,
        )
    assert verification_calls == 2
    assert label_calls == 0


def test_resumable_cache_reuses_valid_rows_and_only_evaluates_missing_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = (_spec(tmp_path, "series-a"), _spec(tmp_path, "series-b", "M"))
    by_id = {spec.series_id: spec for spec in specs}
    label_ids: list[str] = []

    monkeypatch.setattr(
        module,
        "verify_committed_score",
        lambda directory: _artifact_from_path(directory, by_id),
    )
    monkeypatch.setattr(
        module,
        "read_labels",
        lambda spec: label_ids.append(spec.series_id)
        or np.array([0, 1, 0, 1], dtype=np.int8),
    )
    monkeypatch.setattr(module, "compute_threshold_free_metrics", _fixed_metrics)
    output = tmp_path / "evaluation"
    vendor = _vendor(tmp_path)

    module.evaluate_registered_benchmark(
        specs,
        tmp_path / "results",
        output,
        vendor,
        (Trajectory.OFFICIAL,),
        seed=2027,
        expected_config_sha256=CONFIG_SHA,
        expected_vendor_sha=VENDOR_SHA,
        workers=1,
        resume_existing=True,
    )
    cached_path = output / "metrics" / f"{_manifest(specs[0], Trajectory.OFFICIAL).run_id}.json"
    missing_path = output / "metrics" / f"{_manifest(specs[1], Trajectory.OFFICIAL).run_id}.json"
    cached_bytes = cached_path.read_bytes()
    cached_mtime = cached_path.stat().st_mtime_ns
    missing_path.unlink()
    (output / "file_metrics.csv").unlink()
    (output / "evaluation_summary.json").unlink()
    label_ids.clear()

    rows = module.evaluate_registered_benchmark(
        specs,
        tmp_path / "results",
        output,
        vendor,
        (Trajectory.OFFICIAL,),
        seed=2027,
        expected_config_sha256=CONFIG_SHA,
        expected_vendor_sha=VENDOR_SHA,
        workers=1,
        resume_existing=True,
    )

    assert label_ids == ["series-b"]
    assert cached_path.read_bytes() == cached_bytes
    assert cached_path.stat().st_mtime_ns == cached_mtime
    assert [row.series_id for row in rows] == ["series-a", "series-b"]
    with (output / "file_metrics.csv").open(encoding="utf-8", newline="") as handle:
        assert [row["series_id"] for row in csv.DictReader(handle)] == [
            "series-a",
            "series-b",
        ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("run_id", "wrong-run"),
        ("series_id", "wrong-series"),
        ("family", "wrong-family"),
        ("track", "M"),
        ("seed", 2028),
        ("trajectory", "PAPERNEG"),
        ("checkpoint", "BEST"),
        ("score_sha256", "9" * 64),
        ("data_sha256", "9" * 64),
        ("config_sha256", "9" * 64),
        ("vendor_sha", "9" * 40),
    ),
)
def test_resumable_cache_fails_closed_on_provenance_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    spec = _spec(tmp_path, "series-a")
    by_id = {spec.series_id: spec}
    label_calls = 0

    monkeypatch.setattr(
        module,
        "verify_committed_score",
        lambda directory: _artifact_from_path(directory, by_id),
    )

    def labels(_spec: SeriesSpec) -> np.ndarray:
        nonlocal label_calls
        label_calls += 1
        return np.array([0, 1, 0, 1], dtype=np.int8)

    monkeypatch.setattr(module, "read_labels", labels)
    monkeypatch.setattr(module, "compute_threshold_free_metrics", _fixed_metrics)
    output = tmp_path / "evaluation"
    vendor = _vendor(tmp_path)
    module.evaluate_registered_benchmark(
        (spec,),
        tmp_path / "results",
        output,
        vendor,
        (Trajectory.OFFICIAL,),
        seed=2027,
        expected_config_sha256=CONFIG_SHA,
        expected_vendor_sha=VENDOR_SHA,
        workers=1,
        resume_existing=True,
    )
    metric_path = next((output / "metrics").glob("*.json"))
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    payload[field] = replacement
    metric_path.write_text(json.dumps(payload), encoding="utf-8")
    label_calls = 0

    with pytest.raises((ValueError, TypeError)):
        module.evaluate_registered_benchmark(
            (spec,),
            tmp_path / "results",
            output,
            vendor,
            (Trajectory.OFFICIAL,),
            seed=2027,
            expected_config_sha256=CONFIG_SHA,
            expected_vendor_sha=VENDOR_SHA,
            workers=1,
            resume_existing=True,
        )
    assert label_calls == 0


def test_one_label_read_is_reused_across_all_registered_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, "series-a")
    by_id = {spec.series_id: spec}
    label_calls = 0
    verification_calls = 0

    def verify(directory: Path):
        nonlocal verification_calls
        verification_calls += 1
        return _artifact_from_path(directory, by_id)

    def read_labels(_spec: SeriesSpec) -> np.ndarray:
        nonlocal label_calls
        label_calls += 1
        return np.array([0, 1, 0, 1], dtype=np.int8)

    monkeypatch.setattr(module, "verify_committed_score", verify)
    monkeypatch.setattr(module, "read_labels", read_labels)
    monkeypatch.setattr(
        module,
        "compute_threshold_free_metrics",
        lambda *_args, **_kwargs: {
            "vus_pr": 0.6,
            "auprc": 0.5,
            "vus_roc": 0.7,
            "auroc": 0.8,
        },
    )

    rows = module.evaluate_registered_benchmark(
        (spec,),
        tmp_path / "results",
        tmp_path / "evaluation",
        object(),
        module.REGISTERED_BENCHMARK_TRAJECTORIES,
        seed=2027,
        expected_config_sha256=CONFIG_SHA,
        expected_vendor_sha=VENDOR_SHA,
    )

    assert len(rows) == 3
    assert {row.trajectory for row in rows} == set(
        module.REGISTERED_BENCHMARK_TRAJECTORIES
    )
    assert {row.checkpoint for row in rows} == {CheckpointKind.LAST}
    assert verification_calls == 6  # global preflight plus immutable recheck
    assert label_calls == 1
    assert len(tuple((tmp_path / "evaluation" / "metrics").glob("*.json"))) == 3
    with (tmp_path / "evaluation" / "file_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 3


@pytest.mark.parametrize(
    ("trajectories", "checkpoint", "message"),
    (
        ((Trajectory.RAND_BN,), CheckpointKind.LAST, "unregistered"),
        ((Trajectory.OFFICIAL,), CheckpointKind.BEST, "LAST only"),
    ),
)
def test_evaluator_rejects_unregistered_arm_or_non_last_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trajectories: tuple[Trajectory, ...],
    checkpoint: CheckpointKind,
    message: str,
) -> None:
    monkeypatch.setattr(
        module,
        "verify_committed_score",
        lambda *_args, **_kwargs: pytest.fail("score I/O must not start"),
    )
    with pytest.raises(ValueError, match=message):
        module.evaluate_registered_benchmark(
            (_spec(tmp_path, "series-a"),),
            tmp_path / "results",
            tmp_path / "evaluation",
            object(),
            trajectories,
            seed=2027,
            checkpoint=checkpoint,
        )
