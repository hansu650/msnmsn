"""Evaluator-only metrics for registered full-benchmark LAST score artifacts."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
from typing import Sequence
import uuid

import numpy as np

from .artifacts import atomic_write_json, sha256_file, verify_committed_score
from .benchmark_manifest import load_benchmark_manifest
from .config import load_protocol
from .evaluate_scores import compute_threshold_free_metrics
from .label_data import read_labels, validate_score_alignment
from .schemas import (
    CheckpointKind,
    MetricRow,
    ScoreManifest,
    SeriesSpec,
    Trajectory,
    make_run_id,
)
from .vendor import VendorSymbols, load_vendor_symbols


REGISTERED_BENCHMARK_TRAJECTORIES = (
    Trajectory.PAPERNEG_NONOVERLAP,
    Trajectory.PAPERNEG,
    Trajectory.OFFICIAL,
)
_REGISTERED_SET = frozenset(REGISTERED_BENCHMARK_TRAJECTORIES)
_FILE_METRIC_FIELDS = (
    "run_id",
    "series_id",
    "family",
    "track",
    "seed",
    "trajectory",
    "checkpoint",
    "arm",
    "vus_pr",
    "auprc",
    "vus_roc",
    "auroc",
    "score_sha256",
    "data_sha256",
    "config_sha256",
    "vendor_sha",
)
_METRIC_ROW_FIELDS = frozenset(MetricRow.__dataclass_fields__)
_THRESHOLDS = 250
_MAX_WORKERS = 4
_RAM_FLOOR_GIB = 20.0
_RAM_PER_WORKER_GIB = 3.0
_WORKER_VENDOR: VendorSymbols | None = None


@dataclass(frozen=True, slots=True)
class _VerifiedArtifact:
    spec: SeriesSpec
    trajectory: Trajectory
    directory: Path
    manifest: ScoreManifest


@dataclass(frozen=True, slots=True)
class _SeriesTask:
    spec: SeriesSpec
    artifacts: tuple[_VerifiedArtifact, ...]
    seed: int
    checkpoint: CheckpointKind
    expected_config_sha256: str | None
    expected_vendor_sha: str | None


def _coerce_trajectories(
    trajectories: Sequence[Trajectory | str],
) -> tuple[Trajectory, ...]:
    values = tuple(
        item if isinstance(item, Trajectory) else Trajectory(str(item))
        for item in trajectories
    )
    if not values:
        raise ValueError("at least one benchmark trajectory is required")
    if len(set(values)) != len(values):
        raise ValueError("benchmark trajectories must be unique")
    unknown = [item.value for item in values if item not in _REGISTERED_SET]
    if unknown:
        raise ValueError(f"unregistered benchmark trajectories: {unknown}")
    return values


def _score_directory(
    results_root: Path,
    spec: SeriesSpec,
    seed: int,
    trajectory: Trajectory,
    checkpoint: CheckpointKind,
) -> Path:
    return (
        Path(results_root)
        / "runs"
        / spec.series_id
        / f"seed_{seed}"
        / trajectory.value
        / "scores"
        / checkpoint.value
    )


def _verify_expected_score(
    directory: Path,
    spec: SeriesSpec,
    trajectory: Trajectory,
    seed: int,
    checkpoint: CheckpointKind,
    *,
    expected_config_sha256: str | None,
    expected_vendor_sha: str | None,
) -> tuple[np.ndarray, ScoreManifest]:
    scores, manifest = verify_committed_score(directory)
    expected_run_id = make_run_id(spec.series_id, seed, trajectory, checkpoint)
    observed = (
        manifest.run_id,
        manifest.series_id,
        manifest.family,
        manifest.track,
        manifest.seed,
        manifest.trajectory,
        manifest.checkpoint,
        manifest.data_sha256,
    )
    expected = (
        expected_run_id,
        spec.series_id,
        spec.family,
        spec.track,
        seed,
        trajectory,
        checkpoint,
        spec.csv_sha256,
    )
    if observed != expected:
        raise ValueError(
            f"score provenance mismatch for {spec.series_id}/{trajectory.value}"
        )
    if expected_config_sha256 is not None and (
        manifest.config_sha256 != expected_config_sha256
    ):
        raise ValueError("score config SHA differs from the registered protocol")
    if expected_vendor_sha is not None and manifest.vendor_sha != expected_vendor_sha:
        raise ValueError("score vendor SHA differs from the registered baseline")
    if scores.shape != (spec.rows,):
        raise ValueError(
            f"score length mismatch for {spec.series_id}: {scores.shape}"
        )
    return scores, manifest


def _atomic_write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _metric_payload(row: MetricRow) -> dict[str, object]:
    payload = row.to_dict()
    payload["arm"] = row.arm
    return payload


def _row_from_metrics(
    spec: SeriesSpec,
    trajectory: Trajectory,
    seed: int,
    checkpoint: CheckpointKind,
    manifest: ScoreManifest,
    metrics: dict[str, float],
) -> MetricRow:
    return MetricRow(
        run_id=manifest.run_id,
        series_id=spec.series_id,
        family=spec.family,
        track=spec.track,
        seed=seed,
        trajectory=trajectory,
        checkpoint=checkpoint,
        vus_pr=metrics["vus_pr"],
        auprc=metrics["auprc"],
        vus_roc=metrics["vus_roc"],
        auroc=metrics["auroc"],
        score_sha256=manifest.score_sha256,
        data_sha256=spec.csv_sha256,
        config_sha256=manifest.config_sha256,
        vendor_sha=manifest.vendor_sha,
    )


def _evaluate_series(task: _SeriesTask, vendor: VendorSymbols) -> tuple[MetricRow, ...]:
    """Recheck every pending score, then read one label vector for the series."""

    reverified: list[tuple[_VerifiedArtifact, np.ndarray, ScoreManifest]] = []
    for item in task.artifacts:
        scores, manifest = _verify_expected_score(
            item.directory,
            task.spec,
            item.trajectory,
            task.seed,
            task.checkpoint,
            expected_config_sha256=task.expected_config_sha256,
            expected_vendor_sha=task.expected_vendor_sha,
        )
        if manifest.score_sha256 != item.manifest.score_sha256:
            raise RuntimeError("score payload changed after global preflight")
        reverified.append((item, scores, manifest))

    # No label is reachable until all pending arms for this series are immutable.
    labels = read_labels(task.spec)
    rows: list[MetricRow] = []
    for item, scores, manifest in reverified:
        validate_score_alignment(labels, scores, task.spec.rows)
        metrics = compute_threshold_free_metrics(
            scores,
            labels,
            manifest.sliding_window,
            vendor,
            thresholds=_THRESHOLDS,
        )
        rows.append(
            _row_from_metrics(
                task.spec,
                item.trajectory,
                task.seed,
                task.checkpoint,
                manifest,
                metrics,
            )
        )
    return tuple(rows)


def _configure_worker_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _worker_initializer(vendor_root: str, expected_vendor_sha: str) -> None:
    global _WORKER_VENDOR
    _configure_worker_threads()
    _WORKER_VENDOR = load_vendor_symbols(Path(vendor_root), expected_vendor_sha)


def _worker_evaluate_series(task: _SeriesTask) -> tuple[MetricRow, ...]:
    if _WORKER_VENDOR is None:
        raise RuntimeError("worker vendor was not initialized")
    return _evaluate_series(task, _WORKER_VENDOR)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def _available_memory_gib() -> float:
    if os.name == "nt":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.ullAvailPhys) / float(1 << 30)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    return float(page_size * available_pages) / float(1 << 30)


def _effective_worker_count(requested: int) -> int:
    if requested < 1 or requested > _MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{_MAX_WORKERS}]")
    available = _available_memory_gib()
    capacity = int((available - _RAM_FLOOR_GIB) // _RAM_PER_WORKER_GIB)
    if capacity < 1:
        raise RuntimeError(
            f"available RAM {available:.2f} GiB cannot preserve the "
            f"{_RAM_FLOOR_GIB:.0f} GiB floor"
        )
    return min(requested, capacity)


def _request_digest(verified: Sequence[_VerifiedArtifact]) -> str:
    payload = [
        {
            "run_id": item.manifest.run_id,
            "series_id": item.spec.series_id,
            "trajectory": item.trajectory.value,
            "score_sha256": item.manifest.score_sha256,
            "data_sha256": item.spec.csv_sha256,
            "score_manifest_sha256": sha256(
                json.dumps(
                    item.manifest.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        for item in verified
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _evaluator_contract(
    verified: Sequence[_VerifiedArtifact],
    vendor: VendorSymbols,
    selected: Sequence[Trajectory],
    *,
    seed: int,
    checkpoint: CheckpointKind,
    expected_config_sha256: str | None,
    expected_vendor_sha: str | None,
) -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    sources = {
        name: sha256_file(package_root / name)
        for name in (
            "evaluate_benchmark.py",
            "evaluate_scores.py",
            "label_data.py",
            "artifacts.py",
            "schemas.py",
            "vendor.py",
        )
    }
    vendor_metric = vendor.fingerprint.root / "utils" / "basic_metrics.py"
    sources["vendor/utils/basic_metrics.py"] = sha256_file(vendor_metric)
    return {
        "schema_version": "paano-evaluator-contract-v1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "thresholds": _THRESHOLDS,
        "seed": seed,
        "checkpoint": checkpoint.value,
        "trajectories": [item.value for item in selected],
        "metric_count": len(verified),
        "request_sha256": _request_digest(verified),
        "config_sha256": expected_config_sha256,
        "vendor_sha": expected_vendor_sha,
        "source_sha256": sources,
    }


def _validate_contract(output_dir: Path, expected: dict[str, object]) -> None:
    root = Path(output_dir)
    metric_root = root / "metrics"
    contract_path = root / "evaluator_contract.json"
    existing_metrics = tuple(metric_root.glob("*.json")) if metric_root.is_dir() else ()
    if contract_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("evaluator cache contract does not match current code")
        return
    if existing_metrics:
        raise ValueError("metric cache exists without an evaluator contract")
    atomic_write_json(contract_path, expected)


def _validate_cached_row(
    path: Path,
    item: _VerifiedArtifact,
    *,
    seed: int,
    checkpoint: CheckpointKind,
) -> MetricRow:
    if path.name != f"{item.manifest.run_id}.json":
        raise ValueError(f"metric cache filename mismatch: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or frozenset(payload) != _METRIC_ROW_FIELDS:
        raise ValueError(f"metric cache schema mismatch: {path}")
    row = MetricRow.from_dict(payload)
    observed = (
        row.run_id,
        row.series_id,
        row.family,
        row.track,
        row.seed,
        row.trajectory,
        row.checkpoint,
        row.score_sha256,
        row.data_sha256,
        row.config_sha256,
        row.vendor_sha,
    )
    expected = (
        item.manifest.run_id,
        item.spec.series_id,
        item.spec.family,
        item.spec.track,
        seed,
        item.trajectory,
        checkpoint,
        item.manifest.score_sha256,
        item.spec.csv_sha256,
        item.manifest.config_sha256,
        item.manifest.vendor_sha,
    )
    if observed != expected:
        raise ValueError(f"metric cache provenance mismatch: {path}")
    return row


def _evaluate_resumable(
    specs: tuple[SeriesSpec, ...],
    verified: tuple[_VerifiedArtifact, ...],
    output_dir: Path,
    vendor: VendorSymbols,
    selected: tuple[Trajectory, ...],
    *,
    seed: int,
    checkpoint: CheckpointKind,
    expected_config_sha256: str | None,
    expected_vendor_sha: str | None,
    workers: int,
) -> tuple[MetricRow, ...]:
    contract = _evaluator_contract(
        verified,
        vendor,
        selected,
        seed=seed,
        checkpoint=checkpoint,
        expected_config_sha256=expected_config_sha256,
        expected_vendor_sha=expected_vendor_sha,
    )
    _validate_contract(output_dir, contract)

    expected_by_id = {item.manifest.run_id: item for item in verified}
    if len(expected_by_id) != len(verified):
        raise RuntimeError("duplicate run ID in benchmark preflight")
    metric_root = Path(output_dir) / "metrics"
    metric_root.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in metric_root.glob("*.json")
        if path.stem not in expected_by_id
    )
    if unexpected:
        raise ValueError(f"unexpected metric cache files: {unexpected[:3]}")

    cached: dict[str, MetricRow] = {}
    pending_by_series: dict[str, list[_VerifiedArtifact]] = {
        spec.series_id: [] for spec in specs
    }
    for item in verified:
        path = metric_root / f"{item.manifest.run_id}.json"
        if not path.is_file():
            pending_by_series[item.spec.series_id].append(item)
            continue
        # Recheck the score payload before accepting a cached metric row.
        _, manifest = _verify_expected_score(
            item.directory,
            item.spec,
            item.trajectory,
            seed,
            checkpoint,
            expected_config_sha256=expected_config_sha256,
            expected_vendor_sha=expected_vendor_sha,
        )
        if manifest.score_sha256 != item.manifest.score_sha256:
            raise RuntimeError("cached score changed after global preflight")
        cached[item.manifest.run_id] = _validate_cached_row(
            path, item, seed=seed, checkpoint=checkpoint
        )

    pending_tasks = tuple(
        _SeriesTask(
            spec=spec,
            artifacts=tuple(pending_by_series[spec.series_id]),
            seed=seed,
            checkpoint=checkpoint,
            expected_config_sha256=expected_config_sha256,
            expected_vendor_sha=expected_vendor_sha,
        )
        for spec in specs
        if pending_by_series[spec.series_id]
    )
    terminal_outputs = (
        Path(output_dir) / "file_metrics.csv",
        Path(output_dir) / "evaluation_summary.json",
    )
    if pending_tasks and any(path.exists() for path in terminal_outputs):
        raise ValueError("partial cache conflicts with existing terminal outputs")
    effective_workers = _effective_worker_count(workers)
    if pending_tasks and effective_workers == 1:
        for task in pending_tasks:
            for row in _evaluate_series(task, vendor):
                atomic_write_json(metric_root / f"{row.run_id}.json", row.to_dict())
    elif pending_tasks:
        _configure_worker_threads()
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=get_context("spawn"),
            initializer=_worker_initializer,
            initargs=(str(vendor.fingerprint.root), vendor.fingerprint.git_sha),
        ) as executor:
            futures = {
                executor.submit(_worker_evaluate_series, task): task.spec.series_id
                for task in pending_tasks
            }
            for future in as_completed(futures):
                for row in future.result():
                    atomic_write_json(
                        metric_root / f"{row.run_id}.json", row.to_dict()
                    )

    # Final fail-closed validation reconstructs canonical manifest/arm order.
    metric_rows = tuple(
        _validate_cached_row(
            metric_root / f"{item.manifest.run_id}.json",
            item,
            seed=seed,
            checkpoint=checkpoint,
        )
        for item in verified
    )
    if len(metric_rows) != len(verified):
        raise RuntimeError("resumable evaluator did not produce exact coverage")
    payload = [_metric_payload(row) for row in metric_rows]
    _atomic_write_csv(Path(output_dir) / "file_metrics.csv", payload, _FILE_METRIC_FIELDS)
    atomic_write_json(
        Path(output_dir) / "evaluation_summary.json",
        {
            "schema_version": "paano-full-evaluation-v1",
            "seed": seed,
            "checkpoint": checkpoint.value,
            "trajectories": [item.value for item in selected],
            "series_count": len(specs),
            "metric_count": len(metric_rows),
            "labels_loaded_after_global_preflight": True,
        },
    )
    return metric_rows


def evaluate_registered_benchmark(
    series: Sequence[SeriesSpec],
    results_root: Path,
    output_dir: Path,
    vendor: VendorSymbols,
    trajectories: Sequence[Trajectory | str],
    *,
    seed: int,
    checkpoint: CheckpointKind | str = CheckpointKind.LAST,
    expected_config_sha256: str | None = None,
    expected_vendor_sha: str | None = None,
    workers: int = 1,
    resume_existing: bool = False,
) -> tuple[MetricRow, ...]:
    """Evaluate exact registered coverage after a global score-hash preflight.

    Phase one verifies every expected committed score while the label reader is
    unreachable.  Only after all hashes and provenance pass does phase two load
    each series label once and reuse it across the requested registered arms.
    """

    specs = tuple(series)
    if not specs or len({item.series_id for item in specs}) != len(specs):
        raise ValueError("benchmark series must be non-empty and unique")
    selected = _coerce_trajectories(trajectories)
    checkpoint_value = (
        checkpoint
        if isinstance(checkpoint, CheckpointKind)
        else CheckpointKind(str(checkpoint))
    )
    if checkpoint_value is not CheckpointKind.LAST:
        raise ValueError("full benchmark evaluation is frozen to LAST only")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if workers < 1 or workers > _MAX_WORKERS:
        raise ValueError(f"workers must be in [1,{_MAX_WORKERS}]")

    # Global preflight.  Do not move label I/O above this complete loop.
    verified: list[_VerifiedArtifact] = []
    for spec in specs:
        for trajectory in selected:
            directory = _score_directory(
                results_root, spec, seed, trajectory, checkpoint_value
            )
            _, manifest = _verify_expected_score(
                directory,
                spec,
                trajectory,
                seed,
                checkpoint_value,
                expected_config_sha256=expected_config_sha256,
                expected_vendor_sha=expected_vendor_sha,
            )
            verified.append(_VerifiedArtifact(spec, trajectory, directory, manifest))

    expected_count = len(specs) * len(selected)
    if len(verified) != expected_count:
        raise RuntimeError(
            f"benchmark preflight coverage mismatch: {len(verified)} != {expected_count}"
        )
    by_series = {
        spec.series_id: tuple(item for item in verified if item.spec is spec)
        for spec in specs
    }
    if any(len(items) != len(selected) for items in by_series.values()):
        raise RuntimeError("benchmark preflight did not cover every requested arm")

    if resume_existing or workers > 1:
        if not hasattr(vendor, "fingerprint"):
            raise TypeError("resumable evaluation requires a frozen vendor fingerprint")
        return _evaluate_resumable(
            specs,
            tuple(verified),
            output_dir,
            vendor,
            selected,
            seed=seed,
            checkpoint=checkpoint_value,
            expected_config_sha256=expected_config_sha256,
            expected_vendor_sha=expected_vendor_sha,
            workers=workers,
        )

    metric_rows: list[MetricRow] = []
    metric_root = Path(output_dir) / "metrics"
    for spec in specs:
        # The first label read occurs here, after every score artifact passed.
        labels = read_labels(spec)
        for item in by_series[spec.series_id]:
            scores, manifest = _verify_expected_score(
                item.directory,
                spec,
                item.trajectory,
                seed,
                checkpoint_value,
                expected_config_sha256=expected_config_sha256,
                expected_vendor_sha=expected_vendor_sha,
            )
            if manifest.score_sha256 != item.manifest.score_sha256:
                raise RuntimeError("score payload changed after global preflight")
            validate_score_alignment(labels, scores, spec.rows)
            metrics = compute_threshold_free_metrics(
                scores, labels, manifest.sliding_window, vendor, thresholds=_THRESHOLDS
            )
            row = _row_from_metrics(
                spec,
                item.trajectory,
                seed,
                checkpoint_value,
                manifest,
                metrics,
            )
            atomic_write_json(metric_root / f"{row.run_id}.json", row.to_dict())
            metric_rows.append(row)

    if len(metric_rows) != expected_count:
        raise RuntimeError(
            f"benchmark evaluation coverage mismatch: {len(metric_rows)} != {expected_count}"
        )
    payload = [_metric_payload(row) for row in metric_rows]
    _atomic_write_csv(Path(output_dir) / "file_metrics.csv", payload, _FILE_METRIC_FIELDS)
    atomic_write_json(
        Path(output_dir) / "evaluation_summary.json",
        {
            "schema_version": "paano-full-evaluation-v1",
            "seed": seed,
            "checkpoint": checkpoint_value.value,
            "trajectories": [item.value for item in selected],
            "series_count": len(specs),
            "metric_count": len(metric_rows),
            "labels_loaded_after_global_preflight": True,
        },
    )
    return tuple(metric_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--trajectories",
        nargs="+",
        choices=tuple(item.value for item in REGISTERED_BENCHMARK_TRAJECTORIES),
        required=True,
    )
    parser.add_argument(
        "--checkpoint", choices=(CheckpointKind.LAST.value,), required=True
    )
    parser.add_argument("--workers", type=int, choices=range(1, _MAX_WORKERS + 1), default=1)
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed not in (2027, 2028, 2029):
        raise ValueError("full benchmark seed is not registered")
    protocol = load_protocol(args.config)
    trajectories = _coerce_trajectories(args.trajectories)
    for trajectory in trajectories:
        protocol.trajectory(trajectory)
    series = load_benchmark_manifest(args.manifest)
    vendor = load_vendor_symbols(args.vendor_root, protocol.baseline.git_sha)
    rows = evaluate_registered_benchmark(
        series,
        args.results_root,
        args.output_dir,
        vendor,
        trajectories,
        seed=args.seed,
        checkpoint=args.checkpoint,
        expected_config_sha256=protocol.source_sha256,
        expected_vendor_sha=protocol.baseline.git_sha,
        workers=args.workers,
        resume_existing=args.resume_existing,
    )
    print(
        f"BENCHMARK_EVALUATION_COMPLETE series={len(series)} "
        f"trajectories={len(trajectories)} metrics={len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
