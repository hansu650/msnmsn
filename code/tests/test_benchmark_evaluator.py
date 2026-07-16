from __future__ import annotations

import csv
from pathlib import Path

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


def test_all_score_hashes_preflight_before_first_label_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        )
    assert verification_calls == 2
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
