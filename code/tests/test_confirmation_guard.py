from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import paano_k0.confirmation_guard as module
from paano_k0.schemas import CheckpointKind, ScoreManifest, Trajectory, make_run_id


CONFIG_SHA = "2" * 64
VENDOR_SHA = "3" * 40


def _protocol() -> SimpleNamespace:
    return SimpleNamespace(
        source_sha256=CONFIG_SHA,
        baseline=SimpleNamespace(git_sha=VENDOR_SHA),
        official_hyperparameters=SimpleNamespace(
            patch_size=96, stride=1, score_top_k=3
        ),
        trajectory=lambda trajectory: trajectory,
    )


def _decision() -> dict[str, object]:
    tracks = {
        track: {
            "track": track,
            "files": files,
            "exceeds_paper_reported": True,
        }
        for track, files in module.EXPECTED_TRACK_COUNTS.items()
    }
    return {
        "schema_version": "paano-full-benchmark-decision-v1",
        "outcome": "CONTINUE_FULL_CONFIRMATION",
        "main_trajectory": "PAPERNEG_NONOVERLAP",
        "checkpoint": "LAST",
        "paper_reference_type": "external_paper_reported",
        "paper_reference_source": module.PAPER_REFERENCE_SOURCE,
        "paper_reported_vus_pr": dict(module.PAPER_REPORTED_VUS_PR),
        "success_requires_both_tracks": True,
        "both_tracks_exceed": True,
        "conditional_confirmation_seeds": [2028, 2029],
        "seed": 2027,
        "series_count": 530,
        "metric_count": 1590,
        "missing_count": 0,
        "config_sha256": CONFIG_SHA,
        "vendor_sha": VENDOR_SHA,
        "tracks": tracks,
    }


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def vendor_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "vendor"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "guard@example.invalid")
    _git(repo, "config", "user.name", "Guard Test")
    (repo / ".gitignore").write_text(
        "__pycache__/\n*.pyc\nignored.py\n", encoding="utf-8"
    )
    (repo / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "model.py")
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_confirmation_authorization_binds_current_config_and_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(_decision()), encoding="utf-8")
    monkeypatch.setattr(module, "load_protocol", lambda path: _protocol())
    monkeypatch.setattr(
        module,
        "verify_vendor_repo",
        lambda root, sha: SimpleNamespace(dirty=False, git_sha=sha),
    )
    monkeypatch.setattr(
        module, "_assert_vendor_has_no_semantic_changes", lambda fingerprint: None
    )

    result = module.validate_confirmation_authorization(
        tmp_path / "config.yaml", tmp_path / "vendor", decision_path
    )
    assert result["conditional_confirmation_seeds"] == [2028, 2029]

    stale = _decision()
    stale["config_sha256"] = "9" * 64
    decision_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
        module.validate_confirmation_authorization(
            tmp_path / "config.yaml", tmp_path / "vendor", decision_path
        )


def test_vendor_guard_allows_only_direct_runtime_bytecode(vendor_repo: Path) -> None:
    for relative in (
        Path("__pycache__/model.cpython-311.pyc"),
        Path("utils/__pycache__/metrics.cpython-311.pyc"),
        Path("路径 空格/__pycache__/模块 1.pyc"),
    ):
        path = vendor_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime bytecode")
    module._assert_vendor_has_no_semantic_changes(
        SimpleNamespace(dirty=False, root=vendor_repo)
    )


@pytest.mark.parametrize(
    "drift_kind",
    (
        "tracked_unstaged",
        "tracked_staged",
        "ordinary_untracked",
        "ignored_untracked",
        "nested_bytecode",
        "non_bytecode_cache_file",
    ),
)
def test_vendor_guard_rejects_semantic_drift(
    vendor_repo: Path,
    drift_kind: str,
) -> None:
    if drift_kind == "tracked_unstaged":
        (vendor_repo / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif drift_kind == "tracked_staged":
        (vendor_repo / "staged.py").write_text("VALUE = 2\n", encoding="utf-8")
        _git(vendor_repo, "add", "staged.py")
    elif drift_kind == "ordinary_untracked":
        (vendor_repo / "ordinary.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif drift_kind == "ignored_untracked":
        (vendor_repo / "ignored.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif drift_kind == "nested_bytecode":
        path = vendor_repo / "__pycache__/nested/model.pyc"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"nested")
    else:
        path = vendor_repo / "__pycache__/README.txt"
        path.parent.mkdir(parents=True)
        path.write_text("not bytecode", encoding="utf-8")
    with pytest.raises(ValueError, match="semantic changes"):
        module._assert_vendor_has_no_semantic_changes(
            SimpleNamespace(dirty=False, root=vendor_repo)
        )


def _score_manifest() -> ScoreManifest:
    return ScoreManifest(
        schema_version="paano-k0-score-v1",
        run_id=make_run_id(
            "series-a", 2028, Trajectory.PAPERNEG_NONOVERLAP, CheckpointKind.LAST
        ),
        series_id="series-a",
        family="NAB",
        track="U",
        data_sha256="1" * 64,
        config_sha256=CONFIG_SHA,
        vendor_sha=VENDOR_SHA,
        seed=2028,
        trajectory=Trajectory.PAPERNEG_NONOVERLAP,
        checkpoint=CheckpointKind.LAST,
        initial_state_sha256="4" * 64,
        replay_sha256="5" * 64,
        checkpoint_sha256="6" * 64,
        num_points=4,
        num_train_patches=2,
        num_full_patches=4,
        channels=1,
        patch_size=96,
        stride=1,
        top_k=3,
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


def test_confirmation_resume_validates_hash_manifest_summary_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    spec = SimpleNamespace(
        series_id="series-a",
        family="NAB",
        track="U",
        csv_sha256="1" * 64,
        rows=4,
        channels=1,
    )
    manifest = _score_manifest()
    run_dir = (
        tmp_path
        / "runs"
        / "series-a"
        / "seed_2028"
        / "PAPERNEG_NONOVERLAP"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "_SUCCESS").write_text(
        "series-a seed=2028 trajectory=PAPERNEG_NONOVERLAP\n",
        encoding="ascii",
    )
    summary = {
        "series_id": "series-a",
        "family": "NAB",
        "track": "U",
        "seed": 2028,
        "trajectory": "PAPERNEG_NONOVERLAP",
        "data_sha256": "1" * 64,
        "config_sha256": CONFIG_SHA,
        "vendor_sha": VENDOR_SHA,
        "vendor_dirty": False,
        "scored_checkpoints": ["BEST", "LAST"],
        "initial_state_sha256": "4" * 64,
        "replay_sha256": "5" * 64,
        "checkpoint_sha256": {"BEST": "9" * 64, "LAST": "6" * 64},
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    monkeypatch.setattr(module, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(module, "load_benchmark_series", lambda path, sid: spec)
    monkeypatch.setattr(
        module,
        "verify_vendor_repo",
        lambda root, sha: SimpleNamespace(dirty=False, git_sha=sha),
    )
    monkeypatch.setattr(
        module, "_assert_vendor_has_no_semantic_changes", lambda fingerprint: None
    )
    monkeypatch.setattr(
        module,
        "verify_committed_score",
        lambda path: (np.zeros(4, dtype=np.float32), manifest),
    )

    assert module.validate_existing_confirmation_run(
        tmp_path / "config.yaml",
        tmp_path / "manifest.csv",
        tmp_path / "vendor",
        tmp_path,
        "series-a",
        2028,
    ) == run_dir

    corrupted = replace(manifest, config_sha256="9" * 64)
    monkeypatch.setattr(
        module,
        "verify_committed_score",
        lambda path: (np.zeros(4, dtype=np.float32), corrupted),
    )
    with pytest.raises(ValueError, match="config_sha256"):
        module.validate_existing_confirmation_run(
            tmp_path / "config.yaml",
            tmp_path / "manifest.csv",
            tmp_path / "vendor",
            tmp_path,
            "series-a",
            2028,
        )
