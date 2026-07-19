"""Data-only runtime supplement for the four IHP x NCTP factorial arms.

This module deliberately benchmarks only work performed *after* a frozen
cache has been loaded.  It does not import a vision model, renderer, dataset,
or any production scorer.  Callers provide four small loader/compute
callables, one for each registered factorial arm.  The benchmark interleaves
the arms by rotating their order on every round, records load and compute
times separately, and always reports ``encoder_calls=0``.

The confirmation-cohort helper is intentionally fail closed.  It only audits
pre-existing files; it never creates a split or cohort marker.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil


SCHEMA_VERSION = 1
SCOPE = "post_cache_projection_scoring_only"
MEASUREMENT_MODE = "post_cache_only"
DEFAULT_WARMUPS = 2
DEFAULT_REPEATS = 5
MIN_REPEATS = 5

FACTORIAL_ARMS = (
    "IHP0_NCTP0",
    "IHP0_NCTP1",
    "IHP1_NCTP0",
    "IHP1_NCTP1",
)
ARM_FACTORS: dict[str, tuple[int, int, str]] = {
    "IHP0_NCTP0": (0, 0, "LEGACY_DEFAULT"),
    "IHP0_NCTP1": (0, 1, "IHP0_NCTP1"),
    "IHP1_NCTP0": (1, 0, "IHP1_NCTP0"),
    "IHP1_NCTP1": (1, 1, "FINAL_DEFAULT"),
}

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

FAILURE_MANIFEST_COLUMNS = (
    "schema_version",
    "created_at_utc",
    "stage",
    "scope",
    "status",
    "series_id",
    "arm",
    "phase",
    "round_index",
    "error_type",
    "reason",
    "traceback",
    "recoverable",
    "encoder_calls",
)

SAMPLE_COLUMNS = (
    "schema_version",
    "scope",
    "measurement_mode",
    "series_id",
    "arm",
    "canonical_arm",
    "ihp",
    "nctp",
    "windows_per_series",
    "round_index",
    "order_position",
    "is_warmup",
    "protocol_warmups",
    "protocol_repeats",
    "load_seconds",
    "compute_seconds",
    "total_seconds",
    "peak_rss_bytes",
    "thread_count",
    "thread_metadata_sha256",
    "encoder_calls",
    "status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "scope",
    "measurement_mode",
    "series_id",
    "arm",
    "canonical_arm",
    "ihp",
    "nctp",
    "phase",
    "unit",
    "windows_per_series",
    "n_warmup",
    "n_measured",
    "protocol_warmups",
    "protocol_repeats",
    "median_ms",
    "iqr_ms",
    "p95_ms",
    "peak_rss_bytes",
    "thread_count",
    "thread_metadata_sha256",
    "encoder_calls",
    "status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class ThreadMetadata:
    """Explicit thread identity recorded without mutating process settings."""

    thread_count: int
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        if isinstance(self.thread_count, bool) or int(self.thread_count) <= 0:
            raise ValueError("thread_count must be a positive integer")
        unknown = set(self.environment) - set(THREAD_ENV_KEYS)
        if unknown:
            raise ValueError(f"unknown thread environment keys: {sorted(unknown)}")

    def payload(self) -> dict[str, Any]:
        return {
            "thread_count": int(self.thread_count),
            "environment": {
                key: str(self.environment.get(key, "UNSET")) for key in THREAD_ENV_KEYS
            },
        }

    @property
    def sha256(self) -> str:
        return _canonical_json_sha256(self.payload())


def capture_thread_metadata(thread_count: int) -> ThreadMetadata:
    """Capture, but never change, the registered process-thread metadata."""

    return ThreadMetadata(
        int(thread_count),
        {key: os.environ.get(key, "UNSET") for key in THREAD_ENV_KEYS},
    )


@dataclass(frozen=True)
class PostCacheOperation:
    """One data-only loader and projection/scoring callable."""

    arm: str
    load: Callable[[], Any]
    compute: Callable[[Any], Any]
    windows_per_series: int

    def __post_init__(self) -> None:
        if self.arm not in FACTORIAL_ARMS:
            raise ValueError(f"unregistered factorial arm: {self.arm}")
        if not callable(self.load) or not callable(self.compute):
            raise TypeError("load and compute must be callable")
        if isinstance(self.windows_per_series, bool) or int(self.windows_per_series) <= 0:
            raise ValueError("windows_per_series must be a positive integer")


@dataclass(frozen=True)
class SupplementRuntimeResult:
    samples: pd.DataFrame
    summary: pd.DataFrame
    failures: pd.DataFrame
    environment: Mapping[str, Any]
    protocol_text: str

    @property
    def complete(self) -> bool:
        return self.failures.empty and not self.samples.empty and not self.summary.empty


def _validate_operations(
    operations: Sequence[PostCacheOperation],
) -> tuple[PostCacheOperation, ...]:
    values = tuple(operations)
    if len(values) != len(FACTORIAL_ARMS):
        raise ValueError("supplement requires exactly the four factorial arms")
    by_arm = {operation.arm: operation for operation in values}
    if len(by_arm) != len(values) or set(by_arm) != set(FACTORIAL_ARMS):
        raise ValueError("supplement arm set differs from the frozen four-arm factorial")
    return tuple(by_arm[arm] for arm in FACTORIAL_ARMS)


def _process_peak_rss(process: psutil.Process) -> int:
    memory = process.memory_info()
    # Windows exposes the process-lifetime peak working set as ``peak_wset``.
    # Other platforms retain the best available RSS observation.
    return int(max(int(memory.rss), int(getattr(memory, "peak_wset", memory.rss))))


def _failure_frame(rows: Sequence[Mapping[str, Any]] = ()) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=FAILURE_MANIFEST_COLUMNS)
    if frame.empty:
        return frame
    if set(frame["status"].astype(str)) != {"FAILED"}:
        raise ValueError("failure manifest status must be FAILED")
    if set(pd.to_numeric(frame["encoder_calls"], errors="raise")) != {0}:
        raise ValueError("post-cache failure rows must record encoder_calls=0")
    return frame.sort_values(["round_index", "arm", "phase"]).reset_index(drop=True)


def _summarize(samples: pd.DataFrame) -> pd.DataFrame:
    measured = samples.loc[(~samples["is_warmup"]) & samples["status"].eq("PASS")]
    rows: list[dict[str, Any]] = []
    for arm in FACTORIAL_ARMS:
        arm_rows = measured.loc[measured["arm"].eq(arm)]
        all_rows = samples.loc[samples["arm"].eq(arm)]
        if arm_rows.empty:
            continue
        operation = ARM_FACTORS[arm]
        windows = int(arm_rows["windows_per_series"].iloc[0])
        for phase in ("load", "compute", "total"):
            seconds = arm_rows[f"{phase}_seconds"].to_numpy(dtype=np.float64)
            milliseconds = seconds * 1000.0
            for unit, values in (
                ("series", milliseconds),
                ("window", milliseconds / float(windows)),
            ):
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scope": SCOPE,
                        "measurement_mode": MEASUREMENT_MODE,
                        "series_id": str(arm_rows["series_id"].iloc[0]),
                        "arm": arm,
                        "canonical_arm": operation[2],
                        "ihp": operation[0],
                        "nctp": operation[1],
                        "phase": phase,
                        "unit": unit,
                        "windows_per_series": windows,
                        "n_warmup": int(all_rows["is_warmup"].sum()),
                        "n_measured": int(values.size),
                        "protocol_warmups": int(all_rows["protocol_warmups"].iloc[0]),
                        "protocol_repeats": int(all_rows["protocol_repeats"].iloc[0]),
                        "median_ms": float(np.median(values)),
                        "iqr_ms": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                        "p95_ms": float(np.quantile(values, 0.95)),
                        "peak_rss_bytes": int(all_rows["peak_rss_bytes"].max()),
                        "thread_count": int(all_rows["thread_count"].iloc[0]),
                        "thread_metadata_sha256": str(
                            all_rows["thread_metadata_sha256"].iloc[0]
                        ),
                        "encoder_calls": 0,
                        "status": "PASS",
                    }
                )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def runtime_environment(thread_metadata: ThreadMetadata) -> dict[str, Any]:
    """Return the environment payload, explicitly scoped to post-cache work."""

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "measurement_mode": MEASUREMENT_MODE,
        "wording": "post-cache projection/scoring only; excludes render, model, encoder, and cache creation",
        "encoder_calls": 0,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "psutil": psutil.__version__,
        "thread_metadata": thread_metadata.payload(),
        "thread_metadata_sha256": thread_metadata.sha256,
    }


def runtime_protocol_text(*, warmups: int, repeats: int) -> str:
    return (
        "ViTTrace factorial runtime supplement\n"
        "=======================================\n"
        "Scope: post-cache projection/scoring only.\n"
        "Excluded: rendering, model loading, encoder inference, cache creation, and GPU work.\n"
        f"Arms: {', '.join(FACTORIAL_ARMS)}.\n"
        f"Protocol: {warmups} warmup rounds and {repeats} measured rounds per arm.\n"
        "Order: deterministic cyclic interleaving across the four arms on every round.\n"
        "Timing: cache load and compute are measured separately; total is their sum.\n"
        "Units: median, IQR, and p95 are reported in ms/series and ms/window.\n"
        "Memory: peak process RSS/working-set observation is reported.\n"
        "Encoder calls: 0 for every sample and summary row.\n"
    )


def benchmark_post_cache_factorial(
    operations: Sequence[PostCacheOperation],
    *,
    series_id: str,
    thread_metadata: ThreadMetadata,
    warmups: int = DEFAULT_WARMUPS,
    repeats: int = DEFAULT_REPEATS,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    process: psutil.Process | None = None,
) -> SupplementRuntimeResult:
    """Benchmark exactly four post-cache arms with cyclic interleaving."""

    ordered = _validate_operations(operations)
    warmup_count = int(warmups)
    repeat_count = int(repeats)
    if warmup_count < 1:
        raise ValueError("at least one warmup round is required")
    if repeat_count < MIN_REPEATS:
        raise ValueError(f"at least {MIN_REPEATS} measured repeats are required")
    if not str(series_id):
        raise ValueError("series_id must be nonempty")
    if not callable(clock_ns):
        raise TypeError("clock_ns must be callable")
    runtime_process = process or psutil.Process()
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    failed_arms: set[str] = set()
    total_rounds = warmup_count + repeat_count
    for round_index in range(total_rounds):
        shift = round_index % len(ordered)
        round_operations = ordered[shift:] + ordered[:shift]
        for order_position, operation in enumerate(round_operations):
            if operation.arm in failed_arms:
                continue
            phase = "load"
            try:
                peak = _process_peak_rss(runtime_process)
                load_start = int(clock_ns())
                payload = operation.load()
                load_stop = int(clock_ns())
                peak = max(peak, _process_peak_rss(runtime_process))
                phase = "compute"
                compute_start = int(clock_ns())
                operation.compute(payload)
                compute_stop = int(clock_ns())
                peak = max(peak, _process_peak_rss(runtime_process))
                load_seconds = (load_stop - load_start) / 1_000_000_000.0
                compute_seconds = (compute_stop - compute_start) / 1_000_000_000.0
                if load_seconds < 0.0 or compute_seconds < 0.0:
                    raise RuntimeError("benchmark clock moved backwards")
                ihp, nctp, canonical = ARM_FACTORS[operation.arm]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scope": SCOPE,
                        "measurement_mode": MEASUREMENT_MODE,
                        "series_id": str(series_id),
                        "arm": operation.arm,
                        "canonical_arm": canonical,
                        "ihp": ihp,
                        "nctp": nctp,
                        "windows_per_series": int(operation.windows_per_series),
                        "round_index": round_index,
                        "order_position": order_position,
                        "is_warmup": round_index < warmup_count,
                        "protocol_warmups": warmup_count,
                        "protocol_repeats": repeat_count,
                        "load_seconds": load_seconds,
                        "compute_seconds": compute_seconds,
                        "total_seconds": load_seconds + compute_seconds,
                        "peak_rss_bytes": peak,
                        "thread_count": int(thread_metadata.thread_count),
                        "thread_metadata_sha256": thread_metadata.sha256,
                        "encoder_calls": 0,
                        "status": "PASS",
                    }
                )
            except Exception as error:  # preserve one durable row per failed arm
                failed_arms.add(operation.arm)
                failure_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "created_at_utc": _utc_now(),
                        "stage": "supplement_runtime",
                        "scope": SCOPE,
                        "status": "FAILED",
                        "series_id": str(series_id),
                        "arm": operation.arm,
                        "phase": phase,
                        "round_index": round_index,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "traceback": traceback.format_exc(),
                        "recoverable": False,
                        "encoder_calls": 0,
                    }
                )
    samples = pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
    failures = _failure_frame(failure_rows)
    summary = _summarize(samples)
    environment = runtime_environment(thread_metadata)
    protocol = runtime_protocol_text(warmups=warmup_count, repeats=repeat_count)
    return SupplementRuntimeResult(samples, summary, failures, environment, protocol)


def write_supplement_outputs(
    output_root: Path,
    result: SupplementRuntimeResult,
) -> tuple[Path, ...]:
    """Write compact data-only supplement outputs and a hash-bound marker."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    samples_path = root / "post_cache_runtime_samples.csv"
    summary_path = root / "post_cache_runtime_summary.csv"
    environment_path = root / "post_cache_runtime_environment.json"
    protocol_path = root / "POST_CACHE_RUNTIME_PROTOCOL.txt"
    failure_path = root / "failure_manifest.csv"
    _atomic_csv(samples_path, result.samples.loc[:, SAMPLE_COLUMNS])
    _atomic_csv(summary_path, result.summary.loc[:, SUMMARY_COLUMNS])
    _atomic_json(environment_path, result.environment)
    _atomic_text(protocol_path, result.protocol_text)
    _atomic_csv(failure_path, result.failures.loc[:, FAILURE_MANIFEST_COLUMNS])
    marker = root / (
        "_POST_CACHE_RUNTIME_COMPLETE.json"
        if result.complete
        else "_POST_CACHE_RUNTIME_BLOCKED.json"
    )
    _atomic_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE" if result.complete else "BLOCKED",
            "scope": SCOPE,
            "measurement_mode": MEASUREMENT_MODE,
            "wording": "post-cache projection/scoring only",
            "arm_count": int(result.samples["arm"].nunique()) if not result.samples.empty else 0,
            "encoder_calls": 0,
            "samples_sha256": sha256_file(samples_path),
            "summary_sha256": sha256_file(summary_path),
            "environment_sha256": sha256_file(environment_path),
            "protocol_sha256": sha256_file(protocol_path),
            "failure_manifest_sha256": sha256_file(failure_path),
            "failure_count": len(result.failures),
        },
    )
    return samples_path, summary_path, environment_path, protocol_path, failure_path, marker


def _parse_utc(value: Any, field: str) -> datetime:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def audit_confirmation_cohort(
    cohort_marker_path: Path | None,
    arm_selection_record_path: Path | None,
    *,
    selection_mtime_tolerance_seconds: float = 300.0,
) -> dict[str, Any]:
    """Audit pre-existing cohort evidence without creating or changing it.

    ``CONFIRMATION_READY`` is returned only when an already-existing arm
    selection record hash-binds the unchanged cohort marker and manifest, and
    both cohort files demonstrably predate the recorded arm-selection time.
    Every missing, stale, post-selection, or unbound case is blocked.
    """

    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    marker_path = Path(cohort_marker_path) if cohort_marker_path is not None else None
    selection_path = (
        Path(arm_selection_record_path) if arm_selection_record_path is not None else None
    )
    marker: Mapping[str, Any] | None = None
    selection: Mapping[str, Any] | None = None
    if marker_path is None or not marker_path.is_file() or marker_path.is_symlink():
        reasons.append("pre-existing regular cohort marker is absent")
    else:
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, Mapping):
                raise ValueError("cohort marker must contain a JSON object")
            marker = payload
            evidence["cohort_marker_path"] = str(marker_path.resolve())
            evidence["cohort_marker_sha256"] = sha256_file(marker_path)
        except Exception as error:
            reasons.append(f"invalid cohort marker: {type(error).__name__}: {error}")
    if selection_path is None or not selection_path.is_file() or selection_path.is_symlink():
        reasons.append("pre-existing arm-selection record is absent")
    else:
        try:
            payload = json.loads(selection_path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, Mapping):
                raise ValueError("arm-selection record must contain a JSON object")
            selection = payload
            evidence["arm_selection_record_path"] = str(selection_path.resolve())
            evidence["arm_selection_record_sha256"] = sha256_file(selection_path)
        except Exception as error:
            reasons.append(f"invalid arm-selection record: {type(error).__name__}: {error}")
    if marker is not None and selection is not None:
        if marker.get("status") != "FROZEN" or not str(marker.get("cohort_id", "")):
            reasons.append("cohort marker is not a named FROZEN cohort")
        if selection.get("status") != "ARM_SELECTION_FROZEN":
            reasons.append("arm-selection record is not frozen")
        try:
            marker_created = _parse_utc(marker.get("created_at_utc"), "created_at_utc")
            selected_at = _parse_utc(
                selection.get("arm_selection_at_utc"), "arm_selection_at_utc"
            )
            marker_mtime = datetime.fromtimestamp(
                marker_path.stat().st_mtime, tz=timezone.utc
            )
            selection_mtime = datetime.fromtimestamp(
                selection_path.stat().st_mtime, tz=timezone.utc
            )
            evidence["cohort_created_at_utc"] = marker_created.isoformat()
            evidence["arm_selection_at_utc"] = selected_at.isoformat()
            if marker_created >= selected_at or marker_mtime >= selected_at:
                reasons.append("cohort marker does not demonstrably predate arm selection")
            tolerance = float(selection_mtime_tolerance_seconds)
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("selection_mtime_tolerance_seconds must be non-negative")
            if abs((selection_mtime - selected_at).total_seconds()) > tolerance:
                reasons.append("arm-selection record mtime is not contemporaneous with selection")
        except Exception as error:
            reasons.append(f"invalid temporal evidence: {type(error).__name__}: {error}")
        marker_sha = evidence.get("cohort_marker_sha256", "")
        if str(selection.get("confirmation_marker_sha256", "")).upper() != marker_sha:
            reasons.append("arm-selection record does not hash-bind the current cohort marker")
        manifest_value = marker.get("cohort_manifest_path")
        manifest_path = Path(str(manifest_value)) if manifest_value else None
        if manifest_path is None or not manifest_path.is_file() or manifest_path.is_symlink():
            reasons.append("pre-existing regular cohort manifest is absent")
        else:
            manifest_sha = sha256_file(manifest_path)
            evidence["cohort_manifest_path"] = str(manifest_path.resolve())
            evidence["cohort_manifest_sha256"] = manifest_sha
            if str(marker.get("cohort_manifest_sha256", "")).upper() != manifest_sha:
                reasons.append("cohort manifest changed after marker freeze")
            if str(selection.get("confirmation_manifest_sha256", "")).upper() != manifest_sha:
                reasons.append("arm-selection record does not hash-bind the cohort manifest")
            try:
                selected_at = _parse_utc(
                    selection.get("arm_selection_at_utc"), "arm_selection_at_utc"
                )
                manifest_mtime = datetime.fromtimestamp(
                    manifest_path.stat().st_mtime, tz=timezone.utc
                )
                if manifest_mtime >= selected_at:
                    reasons.append("cohort manifest does not predate arm selection")
            except Exception:
                pass
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "CONFIRMATION_READY" if not reasons else "CONFIRMATION_BLOCKED",
        "reason": "; ".join(reasons),
        "reasons": reasons,
        "audit_is_read_only": True,
        "created_split": False,
        "evidence": evidence,
    }


__all__ = [
    "ARM_FACTORS",
    "DEFAULT_REPEATS",
    "DEFAULT_WARMUPS",
    "FACTORIAL_ARMS",
    "FAILURE_MANIFEST_COLUMNS",
    "MEASUREMENT_MODE",
    "MIN_REPEATS",
    "PostCacheOperation",
    "SAMPLE_COLUMNS",
    "SCHEMA_VERSION",
    "SCOPE",
    "SUMMARY_COLUMNS",
    "SupplementRuntimeResult",
    "ThreadMetadata",
    "audit_confirmation_cohort",
    "benchmark_post_cache_factorial",
    "capture_thread_metadata",
    "runtime_environment",
    "runtime_protocol_text",
    "sha256_file",
    "write_supplement_outputs",
]
