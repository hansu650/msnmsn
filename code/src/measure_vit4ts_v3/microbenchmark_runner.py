"""5-warmup + 30-repeat cache-only post-processing microbenchmarks.

The benchmark opens one hash-bound frozen token cache and the already
committed score vectors.  Encoder calls are impossible in this module.  Each
timed post-processing output must remain exactly equal to its committed score,
so timing cannot silently introduce a new scoring implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

from .cache_registry import CacheOnlyPlan, PlannedArm, load_compute_plan, sha256_file
from .cache_runner import (
    RUN_NAME,
    CacheOnlyScorer,
    _load_config,
    _load_frozen_cache,
    _load_trace,
    _validate_plan_and_registry,
    validate_parity_gate,
)
from .runtime_outputs import RUNTIME_SCHEMA_VERSION, normalize_runtime_samples, write_runtime_outputs


SCHEMA_VERSION = 1
DEFAULT_WARMUP = 5
DEFAULT_REPEATS = 30


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cuda_peaks(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return np.nan, np.nan
    torch.cuda.synchronize(device)
    return (
        float(torch.cuda.max_memory_allocated(device)),
        float(torch.cuda.max_memory_reserved(device)),
    )


def benchmark_callable(
    operation: Callable[[], np.ndarray],
    reference: np.ndarray,
    *,
    identity: Mapping[str, Any],
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """Time a score-preserving cache-only callable under the fixed protocol."""

    warmup_count = int(warmup)
    repeat_count = int(repeats)
    if warmup_count != DEFAULT_WARMUP or repeat_count != DEFAULT_REPEATS:
        raise ValueError("v3 microbenchmark protocol is fixed to 5 warmup + 30 repeats")
    destination = torch.device(device)
    expected = np.asarray(reference, dtype=np.float64)
    if expected.ndim != 1 or not np.isfinite(expected).all():
        raise ValueError("microbenchmark reference must be a finite score vector")
    process = psutil.Process()
    rows: list[dict[str, Any]] = []
    for index in range(warmup_count + repeat_count):
        if destination.type == "cuda":
            torch.cuda.reset_peak_memory_stats(destination)
            torch.cuda.synchronize(destination)
        started = time.perf_counter()
        actual = np.asarray(operation(), dtype=np.float64)
        if destination.type == "cuda":
            torch.cuda.synchronize(destination)
        elapsed = time.perf_counter() - started
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            difference = (
                float(np.max(np.abs(actual - expected)))
                if actual.shape == expected.shape
                else float("inf")
            )
            raise RuntimeError(f"microbenchmark output differs from committed score: max_abs={difference}")
        allocated, reserved = _cuda_peaks(destination)
        rows.append(
            {
                "schema_version": RUNTIME_SCHEMA_VERSION,
                "sample_kind": "benchmark_repeat",
                "scope": "cache_only_postprocess",
                "experiment_id": str(identity["experiment_id"]),
                "arm": str(identity["arm"]),
                "series_id": str(identity["series_id"]),
                "family": str(identity["family"]),
                "subgroup": str(identity["subgroup"]),
                "measurement_mode": "cached",
                "stage": str(identity.get("stage", "postprocess")),
                "backbone": str(identity.get("backbone", "B16")),
                "representation": str(identity.get("representation", "line")),
                "window": int(identity.get("window", 240)),
                "stride": int(identity.get("stride", 60)),
                "patch_size": int(identity.get("patch_size", 16)),
                "batch_size": 0,
                "device": str(destination),
                "threads": int(torch.get_num_threads()),
                "repeat_index": index,
                "is_warmup": index < warmup_count,
                "protocol_warmup": warmup_count,
                "protocol_repeats": repeat_count,
                "elapsed_seconds": elapsed,
                "peak_rss_bytes": float(process.memory_info().rss),
                "cuda_peak_allocated_bytes": allocated,
                "cuda_peak_reserved_bytes": reserved,
                "encoder_calls": 0,
                "status": "PASS",
                "partial_reason": "",
            }
        )
    return normalize_runtime_samples(rows)


def _committed_canonical_score(run_root: Path, series_id: str, planned: PlannedArm) -> tuple[np.ndarray, str]:
    path = Path(run_root) / RUN_NAME / series_id / planned.canonical_arm / "score.npy"
    if not path.is_file():
        raise FileNotFoundError(f"committed canonical score is missing: {path}")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("committed microbenchmark score is invalid")
    return np.ascontiguousarray(values, dtype=np.float64), sha256_file(path)


def run_microbenchmark(
    config_path: Path,
    plan_path: Path,
    registry_path: Path,
    parity_gate_path: Path,
    output_root: Path,
    *,
    series_id: str = "MSL__C-1",
    arm_ids: Sequence[str] | None = None,
    device: str = "cpu",
) -> tuple[Path, ...]:
    """Run exact post-processing repeats for selected canonical v3 arms."""

    root = Path(output_root)
    try:
        bundle = _load_config(config_path)
        validate_parity_gate(parity_gate_path, bundle)
        plan, _ = _validate_plan_and_registry(plan_path, registry_path, bundle)
        records = {record.series_id: record for record in bundle.records}
        if series_id not in records:
            raise ValueError("microbenchmark series is absent from the frozen manifest")
        record = records[series_id]
        selected_ids = tuple(arm_ids or ("IHP0_NCTP0", "IHP1_NCTP0", "IHP0_NCTP1", "IHP1_NCTP1"))
        by_id = plan.by_id()
        if len(set(selected_ids)) != len(selected_ids) or any(arm not in by_id for arm in selected_ids):
            raise ValueError("microbenchmark arm selection is invalid")
        # Deduplicate aliases while preserving the requested logical order.
        selected: list[PlannedArm] = []
        seen: set[str] = set()
        for arm_id in selected_ids:
            canonical = by_id[arm_id].canonical_arm
            planned = by_id[canonical]
            if canonical not in seen:
                selected.append(planned)
                seen.add(canonical)

        cache, _, token_sha, cache_manifest_sha = _load_frozen_cache(bundle, record)
        trace = _load_trace(bundle, record, token_sha, cache_manifest_sha)
        destination = torch.device(device)
        if destination.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA microbenchmark requested but unavailable")
        scorer = CacheOnlyScorer(cache, trace, record, destination)
        scorer.prepare_matching()  # setup is outside post-processing repeats
        frames: list[pd.DataFrame] = []
        score_hashes: dict[str, str] = {}
        for planned in selected:
            reference, score_sha = _committed_canonical_score(
                Path(bundle.payload["paths"]["run_root"]), series_id, planned
            )
            score_hashes[planned.canonical_arm] = score_sha

            def operation(item: PlannedArm = planned) -> np.ndarray:
                # Projection memoization is cleared so every repeat includes
                # incidence projection, scale fusion, and temporal return.
                scorer.projected.clear()
                return scorer.score(item.logical)

            frames.append(
                benchmark_callable(
                    operation,
                    reference,
                    identity={
                        "experiment_id": "B16_W240_S60_CACHE_ONLY_5W30R",
                        "arm": planned.canonical_arm,
                        "series_id": series_id,
                        "family": record.track,
                        "subgroup": record.paper_group,
                    },
                    device=destination,
                )
            )
        samples = pd.concat(frames, ignore_index=True)
        sample_path, summary_path = write_runtime_outputs(root, samples)
        marker = root / "_MICROBENCHMARK_COMPLETE.json"
        _atomic_json(
            marker,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "COMPLETE",
                "series_id": series_id,
                "warmup": DEFAULT_WARMUP,
                "repeats": DEFAULT_REPEATS,
                "encoder_calls": 0,
                "matching_setup_seconds_not_in_postprocess": float(scorer.shared_matching_seconds),
                "token_cache_sha256": token_sha,
                "token_cache_manifest_sha256": cache_manifest_sha,
                "committed_score_sha256": score_hashes,
                "runtime_samples_sha256": sha256_file(sample_path),
                "runtime_summary_sha256": sha256_file(summary_path),
            },
        )
        blocked = root / "_MICROBENCHMARK_BLOCKED.json"
        if blocked.exists():
            blocked.unlink()
        return sample_path, summary_path, marker
    except FileNotFoundError as error:
        marker = root / "_MICROBENCHMARK_BLOCKED.json"
        _atomic_json(
            marker,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "BLOCKED",
                "reason": f"required upstream stage is incomplete: {error}",
                "error_type": type(error).__name__,
            },
        )
        return (marker,)
    except Exception as error:
        failure = root / "_MICROBENCHMARK_FAILED.json"
        _atomic_json(
            failure,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--parity-gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--series-id", default="MSL__C-1")
    parser.add_argument("--arm", action="append", dest="arms")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    outputs = run_microbenchmark(
        args.config,
        args.plan,
        args.registry,
        args.parity_gate,
        args.output_root,
        series_id=args.series_id,
        arm_ids=args.arms,
        device=args.device,
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, sort_keys=True))
    return 2 if any("BLOCKED" in path.name for path in outputs) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_WARMUP",
    "SCHEMA_VERSION",
    "benchmark_callable",
    "run_microbenchmark",
]
