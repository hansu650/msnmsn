"""Label-free, fail-fast runner for one registered PaAno K0 trajectory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import traceback
from typing import Sequence
import uuid

import torch

from .artifacts import (
    atomic_save_checkpoint,
    atomic_write_json,
    commit_score_artifact,
)
from .config import ProtocolConfig, load_protocol, load_series_manifest, validate_protocol
from .feature_data import make_patch_store, read_feature_series, split_normal_prefix
from .instrumentation import IterationRecorder
from .replay import (
    ReplayIdentity,
    build_initial_state,
    build_replay_plan,
    load_replay_plan,
    save_replay_plan,
    stable_seed,
)
from .schemas import (
    RunJob,
    ScoreManifest,
    Trajectory,
    make_run_id,
    scored_checkpoints,
)
from .scoring import score_checkpoint
from .trainer import replay_rand_bn, train_trajectory
from .vendor import build_encoder, compute_baseline_window, load_vendor_symbols


_SCORE_SCHEMA = "paano-k0-score-v1"


def _trajectory_directory(job: RunJob) -> Path:
    return (
        job.output_root
        / "runs"
        / job.series.series_id
        / f"seed_{job.seed}"
        / job.trajectory.value
    )


def _score_directory(job: RunJob, checkpoint: str) -> Path:
    """Mirror ``evaluate_scores.score_directory`` without importing label code."""

    return _trajectory_directory(job) / "scores" / checkpoint


def _atomic_success(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(payload.rstrip("\n") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_or_create_replay(
    job: RunJob,
    protocol: ProtocolConfig,
    n_train_patches: int,
    replay_seed: int,
):
    hp = protocol.official_hyperparameters
    base = (
        job.output_root
        / "replay"
        / job.series.series_id
        / f"seed_{job.seed}"
    )
    npz_path = base.with_suffix(".npz")
    metadata_path = base.with_suffix(".json")
    expected = ReplayIdentity(
        series_id=job.series.series_id,
        seed=replay_seed,
        n_train_patches=n_train_patches,
        batch_size=hp.batch_size,
        num_iterations=hp.iterations,
    )
    exists = (npz_path.is_file(), metadata_path.is_file())
    if exists == (True, True):
        return load_replay_plan(npz_path, expected)
    if exists != (False, False):
        raise RuntimeError(f"partial shared replay artifact: {base}")
    plan = build_replay_plan(
        n_train_patches,
        hp.batch_size,
        hp.iterations,
        replay_seed,
        series_id=job.series.series_id,
    )
    save_replay_plan(plan, npz_path)
    return load_replay_plan(npz_path, expected)


def run_job(job: RunJob, protocol: ProtocolConfig) -> tuple[Path, ...]:
    """Execute and durably commit exactly one registered, feature-only job."""

    trajectory_dir = _trajectory_directory(job)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    success_path = trajectory_dir / "_SUCCESS"
    failure_path = trajectory_dir / "_FAILED.json"
    success_path.unlink(missing_ok=True)

    try:
        validate_protocol(protocol)
        if job.protocol_path.resolve(strict=True) != protocol.source_path:
            raise ValueError("RunJob protocol path differs from the loaded protocol")
        registered = protocol.trajectory(job.trajectory)
        if registered.id is not job.trajectory:
            raise ValueError("job trajectory is not registered by the frozen protocol")
        if tuple(registered.save_checkpoints) != scored_checkpoints(job.trajectory):
            raise ValueError("trajectory checkpoint surface differs from the registry")

        device = torch.device(job.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        vendor = load_vendor_symbols(job.vendor_root, protocol.baseline.git_sha)
        hp = protocol.official_hyperparameters

        x_full = read_feature_series(job.series)
        x_train = split_normal_prefix(x_full, job.series.train_end)
        train_store = make_patch_store(x_train, hp.patch_size, hp.stride)
        full_store = make_patch_store(x_full, hp.patch_size, hp.stride)
        sliding_window = compute_baseline_window(vendor, x_full[:, 0])

        init_seed = stable_seed(job.seed, job.series.csv_sha256, "model_init")
        replay_seed = stable_seed(job.seed, job.series.csv_sha256, "training_replay")
        build_cpu = lambda: build_encoder(  # noqa: E731 - intentional seedable factory
            vendor, job.series.channels, hp.use_revin, torch.device("cpu")
        )
        initial_state, initial_state_sha256 = build_initial_state(build_cpu, init_seed)
        replay = _load_or_create_replay(
            job, protocol, len(train_store), replay_seed
        )

        model = build_encoder(vendor, job.series.channels, hp.use_revin, device)
        recorder = IterationRecorder(
            trajectory_dir / "iteration_metrics.jsonl",
            expected_iterations=hp.iterations,
        )
        if job.trajectory is Trajectory.RAND_BN:
            training = replay_rand_bn(
                model,
                initial_state,
                train_store,
                replay,
                protocol,
                device,
                recorder,
            )
        else:
            training = train_trajectory(
                model,
                initial_state,
                train_store,
                replay,
                job.trajectory,
                protocol,
                device,
                recorder,
            )
        if training.summary.initial_state_sha256 != initial_state_sha256:
            raise RuntimeError("training summary lost the shared initial-state identity")
        if training.summary.replay_sha256 != replay.payload_sha256:
            raise RuntimeError("training summary lost the shared replay identity")

        committed: list[Path] = []
        for checkpoint in scored_checkpoints(job.trajectory):
            if checkpoint not in training.checkpoints:
                raise RuntimeError(f"trainer omitted registered checkpoint {checkpoint.value}")
            checkpoint_state = training.checkpoints[checkpoint]
            checkpoint_path = trajectory_dir / "checkpoints" / f"{checkpoint.value}.pt"
            atomic_save_checkpoint(checkpoint_path, checkpoint_state)
            scored = score_checkpoint(
                model,
                checkpoint_state,
                train_store,
                full_store,
                protocol,
                device,
            )
            if scored.point_scores.shape != (job.series.rows,):
                raise RuntimeError("scorer output is not aligned to the feature series")
            checkpoint_sha256 = training.summary.checkpoint_sha256[checkpoint.value]
            manifest = ScoreManifest(
                schema_version=_SCORE_SCHEMA,
                run_id=make_run_id(
                    job.series.series_id, job.seed, job.trajectory, checkpoint
                ),
                series_id=job.series.series_id,
                family=job.series.family,
                track=job.series.track,
                data_sha256=job.series.csv_sha256,
                config_sha256=protocol.source_sha256,
                vendor_sha=vendor.fingerprint.git_sha,
                seed=job.seed,
                trajectory=job.trajectory,
                checkpoint=checkpoint,
                initial_state_sha256=initial_state_sha256,
                replay_sha256=replay.payload_sha256,
                checkpoint_sha256=checkpoint_sha256,
                num_points=job.series.rows,
                num_train_patches=len(train_store),
                num_full_patches=len(full_store),
                channels=job.series.channels,
                patch_size=hp.patch_size,
                stride=hp.stride,
                top_k=hp.score_top_k,
                requested_memory_fraction=scored.memory.requested_fraction,
                effective_memory_fraction=scored.memory.effective_fraction,
                memory_count=scored.memory.memory_count,
                memory_sha256=scored.memory.sha256,
                score_sha256="0" * 64,
                runtime_seconds=scored.runtime_seconds,
                peak_vram_mib=scored.peak_vram_mib,
                sliding_window=sliding_window,
                labels_read=False,
            )
            score_dir = _score_directory(job, checkpoint.value)
            commit_score_artifact(score_dir, scored.point_scores, manifest)
            committed.append(score_dir)

        if tuple(training.checkpoints) != scored_checkpoints(job.trajectory):
            raise RuntimeError("trainer produced an unregistered checkpoint surface")
        summary = training.summary.to_dict()
        # Training internals use the per-series replay seed.  The public run seed
        # remains the experiment seed used by the score manifests and job matrix.
        summary.update(
            {
                "seed": job.seed,
                "series_id": job.series.series_id,
                "family": job.series.family,
                "track": job.series.track,
                "data_sha256": job.series.csv_sha256,
                "config_sha256": protocol.source_sha256,
                "vendor_sha": vendor.fingerprint.git_sha,
                "vendor_dirty": vendor.fingerprint.dirty,
                "model_init_seed": init_seed,
                "training_replay_seed": replay_seed,
                "iteration_log_sha256": training.iteration_log_sha256,
                "scored_checkpoints": [item.value for item in scored_checkpoints(job.trajectory)],
            }
        )
        atomic_write_json(trajectory_dir / "training_summary.json", summary)
        failure_path.unlink(missing_ok=True)
        _atomic_success(
            success_path,
            f"{job.series.series_id} seed={job.seed} trajectory={job.trajectory.value}",
        )
        return tuple(committed)
    except BaseException as exc:
        success_path.unlink(missing_ok=True)
        atomic_write_json(
            failure_path,
            {
                "series_id": job.series.series_id,
                "family": job.series.family,
                "track": job.series.track,
                "seed": job.seed,
                "trajectory": job.trajectory.value,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--series-id", required=True)
    parser.add_argument(
        "--trajectory", choices=tuple(item.value for item in Trajectory), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.config)
    matches = [
        spec
        for spec in load_series_manifest(args.manifest)
        if spec.series_id == args.series_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"--series-id must identify exactly one frozen series; found {len(matches)}"
        )
    job = RunJob(
        series=matches[0],
        trajectory=Trajectory(args.trajectory),
        seed=args.seed,
        protocol_path=protocol.source_path,
        vendor_root=args.vendor_root,
        output_root=args.output_root,
        device=args.device,
    )
    run_job(job, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
