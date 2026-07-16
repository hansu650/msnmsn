from pathlib import Path

import numpy as np
import pytest

from paano_k0.artifacts import (
    atomic_save_numpy,
    commit_score_artifact,
    verify_committed_score,
)
from paano_k0.schemas import CheckpointKind, ScoreManifest, Trajectory


def _manifest(num_points: int = 4) -> ScoreManifest:
    digest = "1" * 64
    return ScoreManifest(
        schema_version="paano-k0-score-v1",
        run_id="fixture__seed_2027__OFFICIAL__LAST",
        series_id="fixture",
        family="NAB",
        track="U",
        data_sha256=digest,
        config_sha256="2" * 64,
        vendor_sha="3" * 40,
        seed=2027,
        trajectory=Trajectory.OFFICIAL,
        checkpoint=CheckpointKind.LAST,
        initial_state_sha256="4" * 64,
        replay_sha256="5" * 64,
        checkpoint_sha256="6" * 64,
        num_points=num_points,
        num_train_patches=10,
        num_full_patches=20,
        channels=1,
        patch_size=96,
        stride=1,
        top_k=3,
        requested_memory_fraction=0.1,
        effective_memory_fraction=0.5,
        memory_count=5,
        memory_sha256="7" * 64,
        score_sha256="0" * 64,
        runtime_seconds=0.25,
        peak_vram_mib=12.0,
        sliding_window=10,
        labels_read=False,
    )


def test_score_commit_round_trip(tmp_path: Path) -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    marker = commit_score_artifact(tmp_path, scores, _manifest())
    assert marker.name == "_SUCCESS"
    loaded, manifest = verify_committed_score(tmp_path)
    np.testing.assert_array_equal(loaded, scores)
    assert manifest.score_sha256 != "0" * 64
    assert manifest.labels_read is False


def test_success_marker_is_last(tmp_path: Path) -> None:
    atomic_save_numpy(tmp_path / "scores.npy", np.ones(4, dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="success marker"):
        verify_committed_score(tmp_path)


def test_score_hash_detects_mutation(tmp_path: Path) -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    commit_score_artifact(tmp_path, scores, _manifest())
    score_path = tmp_path / "scores.npy"
    payload = bytearray(score_path.read_bytes())
    payload[-1] ^= 1
    score_path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_committed_score(tmp_path)
