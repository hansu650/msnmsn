"""Transactional label-free runner for dynamic-cache REL/IHP/FULL scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
import tracemalloc
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch

from measure_vit4ts.full_manifest import EXPECTED_SERIES, FullSeriesRecord

from . import encoder_control_runner as artifact_checks
from . import encoder_runner as encoder_stage
from .dynamic_cache import CACHE_FILE, DynamicCacheKey, load_dynamic_cache
from .dynamic_scorer import (
    ARM_PARAMETERS,
    DYNAMIC_SCORE_ARMS,
    arm_parameter_sha256,
    compute_dynamic_scores,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = encoder_stage.DEFAULT_CONFIG
_SOURCE_FILES = (
    ("v3", "core.py"),
    ("v3", "dynamic_scorer.py"),
    ("v3", "dynamic_score_runner.py"),
    ("v3", "encoder_control_runner.py"),
    ("legacy", "reducers.py"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def dynamic_score_source_sha256(package_root: Path | None = None) -> str:
    v3 = Path(package_root) if package_root else Path(__file__).resolve().parent
    legacy = v3.parent / "measure_vit4ts"
    digest = hashlib.sha256()
    for namespace, name in _SOURCE_FILES:
        path = (v3 if namespace == "v3" else legacy) / name
        payload = path.read_bytes()
        digest.update(namespace.encode("ascii"))
        digest.update(name.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def scoring_config_sha256(
    variant_sha: str, *, device: str, query_chunk_size: int
) -> str:
    return _payload_sha256(
        {
            "variant_sha256": str(variant_sha).upper(),
            "device": str(torch.device(device)),
            "query_chunk_size": int(query_chunk_size),
            "arms": {arm: ARM_PARAMETERS[arm] for arm in DYNAMIC_SCORE_ARMS},
        }
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_score(path: Path, score: np.ndarray, expected_length: int) -> str:
    values = np.ascontiguousarray(np.asarray(score), dtype=np.float64)
    if values.shape != (int(expected_length),) or not np.isfinite(values).all():
        raise ValueError("dynamic score must be finite float64 [T]")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256(path)


def _key_from_payload(payload: Mapping[str, Any]) -> DynamicCacheKey:
    values = dict(payload)
    values["image_size"] = tuple(int(value) for value in values["image_size"])
    return DynamicCacheKey(**values)


def _transaction_path(root: Path, series_id: str, arm: str) -> Path:
    return root / "dynamic_scores" / series_id / arm / "score_manifest.json"


def _series_identity(
    root: Path,
    record: FullSeriesRecord,
    encoder_payload: Mapping[str, Any],
    *,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    score_source_sha: str,
    variant_sha: str,
    score_config_sha: str,
) -> dict[str, Any]:
    record_path = root / "records" / f"{record.series_id}.json"
    return {
        "series_id": record.series_id,
        "data_sha256": record.expected_sha256.upper(),
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "encoder_source_sha256": encoder_source_sha,
        "dynamic_score_source_sha256": score_source_sha,
        "variant_sha256": variant_sha,
        "scoring_config_sha256": score_config_sha,
        "encoder_record_sha256": _sha256(record_path),
        "cache_file_sha256": str(encoder_payload["cache_file_sha256"]).upper(),
        "cache_manifest_sha256": str(
            encoder_payload["cache_manifest_sha256"]
        ).upper(),
    }


def _resume_transaction(
    path: Path,
    record: FullSeriesRecord,
    arm: str,
    identity: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("status") != "PASS"
            or payload.get("arm") != arm
            or payload.get("arm_parameter_sha256") != arm_parameter_sha256(arm)
            or any(payload.get(key) != value for key, value in identity.items())
            or payload.get("metadata") != ARM_PARAMETERS[arm]
        ):
            return None
        score_path = Path(payload["score_path"])
        score = np.load(score_path, allow_pickle=False)
        if (
            _sha256(score_path) != payload["score_sha256"]
            or score.dtype != np.float64
            or score.shape != (record.expected_length,)
            or not np.isfinite(score).all()
            or encoder_stage._array_sha256(score) != payload["score_array_sha256"]
        ):
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _existing_transactions(
    root: Path,
    record: FullSeriesRecord,
    encoder_payload: Mapping[str, Any],
    *,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    score_source_sha: str,
    variant_sha: str,
    score_config_sha: str,
    retry: bool,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    identity = _series_identity(
        root,
        record,
        encoder_payload,
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        encoder_source_sha=encoder_source_sha,
        score_source_sha=score_source_sha,
        variant_sha=variant_sha,
        score_config_sha=score_config_sha,
    )
    existing: dict[str, Mapping[str, Any]] = {}
    if not retry:
        for arm in DYNAMIC_SCORE_ARMS:
            payload = _resume_transaction(
                _transaction_path(root, record.series_id, arm),
                record,
                arm,
                identity,
            )
            if payload is not None:
                existing[arm] = payload
    return existing, identity


def _load_cache(encoder_payload: Mapping[str, Any]):
    cache = load_dynamic_cache(
        Path(encoder_payload["cache_dir"]),
        _key_from_payload(encoder_payload["cache_key"]),
    )
    if (
        encoder_stage._array_sha256(cache.global_tokens)
        != encoder_payload["global_tokens_sha256"]
    ):
        raise ValueError("true-global array hash differs from encoder transaction")
    if (
        encoder_stage._array_sha256(cache.patch_tokens)
        != encoder_payload["patch_tokens_sha256"]
    ):
        raise ValueError("base-patch array hash differs from encoder transaction")
    return cache


def _commit_series(
    root: Path,
    record: FullSeriesRecord,
    variant: encoder_stage.EncoderVariant,
    encoder_payload: Mapping[str, Any],
    cache: Any,
    existing: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Any],
    *,
    device: str,
    query_chunk_size: int,
) -> dict[str, Mapping[str, Any]]:
    process = psutil.Process(os.getpid())
    rss_before = int(process.memory_info().rss)
    wall_started = time.perf_counter()
    tracemalloc.start()
    try:
        bundle = compute_dynamic_scores(
            cache,
            full_length=record.expected_length,
            window=variant.window,
            stride=variant.stride,
            image_size=variant.image_size,
            device=device,
            query_chunk_size=query_chunk_size,
        )
        _, peak_python = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    compute_wall = time.perf_counter() - wall_started
    rss_after = int(process.memory_info().rss)
    output: dict[str, Mapping[str, Any]] = dict(existing)
    for arm in DYNAMIC_SCORE_ARMS:
        if arm in existing:
            continue
        result = bundle.arms[arm]
        manifest_path = _transaction_path(root, record.series_id, arm)
        score_path = manifest_path.parent / "score.npy"
        score_sha = _atomic_score(score_path, result.score, record.expected_length)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "created_at": _utc_now(),
            **dict(identity),
            "family": record.track,
            "subgroup": record.paper_group,
            "arm": arm,
            "arm_parameter_sha256": arm_parameter_sha256(arm),
            "metadata": dict(result.metadata),
            "variant": {**asdict(variant), "image_size": list(variant.image_size)},
            "cache_dir": str(Path(encoder_payload["cache_dir"]).resolve()),
            "cache_key": dict(encoder_payload["cache_key"]),
            "patch_grid": list(cache.patch_grid),
            "window_count": int(cache.patch_tokens.shape[0]),
            "window_field_shape": list(result.window_field.shape),
            "window_field_sha256": encoder_stage._array_sha256(result.window_field),
            "score_path": str(score_path.resolve()),
            "score_sha256": score_sha,
            "score_array_sha256": encoder_stage._array_sha256(result.score),
            "score_length": int(result.score.size),
            "score_dtype": "float64",
            "encoder_calls": 0,
            "runtime": {
                "device": str(torch.device(device)),
                "query_chunk_size": int(query_chunk_size),
                "shared_matching_seconds": float(bundle.shared_matching_seconds),
                "arm_seconds": float(bundle.arm_seconds[arm]),
                "series_compute_seconds": float(compute_wall),
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "tracemalloc_peak_bytes": int(peak_python),
            },
        }
        _atomic_json(manifest_path, payload)
        output[arm] = payload
    return output


def _failure_path(config: Mapping[str, Any], variant_key: str, series_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        Path(config["paths"]["failure_root"])
        / "dynamic_scores"
        / variant_key
        / series_id
        / f"{stamp}_{uuid.uuid4().hex}.json"
    )


def _summary_row_status(
    path: Path,
    record: FullSeriesRecord,
    arm: str,
    expected: Mapping[str, Any],
) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("status") != "PASS"
            or payload.get("series_id") != record.series_id
            or payload.get("arm") != arm
            or payload.get("data_sha256") != record.expected_sha256.upper()
            or any(payload.get(key) != value for key, value in expected.items())
        ):
            return "NOT_RUN"
        score_path = Path(payload["score_path"])
        shape, dtype, fortran = artifact_checks._npy_header(score_path)
        if (
            shape != (record.expected_length,)
            or dtype != np.dtype(np.float64)
            or fortran
            or _sha256(score_path) != payload["score_sha256"]
        ):
            return "NOT_RUN"
        return "PASS"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return "NOT_RUN"


def _write_summary(
    root: Path,
    records: Sequence[FullSeriesRecord],
    *,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    score_source_sha: str,
    variant_sha: str,
    score_config_sha: str,
) -> Path:
    expected = {
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "encoder_source_sha256": encoder_source_sha,
        "dynamic_score_source_sha256": score_source_sha,
        "variant_sha256": variant_sha,
        "scoring_config_sha256": score_config_sha,
    }
    rows: list[dict[str, str]] = []
    complete_series = 0
    for record in records:
        series_pass = True
        for arm in DYNAMIC_SCORE_ARMS:
            status = _summary_row_status(
                _transaction_path(root, record.series_id, arm),
                record,
                arm,
                expected,
            )
            rows.append({"series_id": record.series_id, "arm": arm, "status": status})
            series_pass &= status == "PASS"
        complete_series += int(series_pass)
    complete_rows = sum(row["status"] == "PASS" for row in rows)
    expected_rows = EXPECTED_SERIES * len(DYNAMIC_SCORE_ARMS)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "COMPLETE" if complete_rows == expected_rows else "INCOMPLETE",
        "expected_series": EXPECTED_SERIES,
        "expected_arms": list(DYNAMIC_SCORE_ARMS),
        "expected_rows": expected_rows,
        "completed_series": complete_series,
        "completed_rows": complete_rows,
        **expected,
        "rows": rows,
    }
    path = root / "dynamic_scores_status.json"
    _atomic_json(path, payload)
    return path


def run_dynamic_scores(
    config_path: Path,
    *,
    model_key: str = "B16",
    window: int = 240,
    stride: int | None = None,
    batch_size: int | None = None,
    variant_key: str | None = None,
    device: str = "cpu",
    query_chunk_size: int = 32,
    smoke: bool = False,
    all_series: bool = False,
    series_ids: Sequence[str] = (),
    approved_bulk: bool = False,
    retry: bool = False,
) -> Path:
    config, config_sha, records, manifest_sha = encoder_stage._load_config(config_path)
    variant = encoder_stage.resolve_variant(
        config,
        representation="line",
        model_key=model_key,
        window=window,
        stride=stride,
        batch_size=batch_size,
        variant_key=variant_key,
    )
    if variant.representation != "line":
        raise ValueError("dynamic scorer consumes registered line caches only")
    if int(query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    selected = encoder_stage.select_records(
        records,
        smoke=smoke,
        all_series=all_series,
        series_ids=series_ids,
        approved_bulk=approved_bulk,
    )
    encoder_source_sha = encoder_stage.encoder_source_sha256()
    score_source_sha = dynamic_score_source_sha256()
    variant_sha = encoder_stage.variant_sha256(variant, config_sha)
    score_config_sha = scoring_config_sha256(
        variant_sha, device=device, query_chunk_size=query_chunk_size
    )
    root = encoder_stage._variant_root(config, variant, variant_sha)

    invalid: list[str] = []
    for record in selected:
        try:
            artifact_checks._preflight_encoder_artifact(
                root,
                record,
                config_sha=config_sha,
                manifest_sha=manifest_sha,
                encoder_source_sha=encoder_source_sha,
                variant_sha=variant_sha,
            )
        except Exception as exc:
            invalid.append(f"{record.series_id}: {type(exc).__name__}: {exc}")
    if invalid:
        raise RuntimeError(
            f"dynamic-score preflight found {len(invalid)} invalid encoder transactions; "
            f"first={invalid[0]}"
        )

    for record in selected:
        cache = None
        try:
            encoder_payload = artifact_checks._preflight_encoder_artifact(
                root,
                record,
                config_sha=config_sha,
                manifest_sha=manifest_sha,
                encoder_source_sha=encoder_source_sha,
                variant_sha=variant_sha,
            )
            existing, identity = _existing_transactions(
                root,
                record,
                encoder_payload,
                config_sha=config_sha,
                manifest_sha=manifest_sha,
                encoder_source_sha=encoder_source_sha,
                score_source_sha=score_source_sha,
                variant_sha=variant_sha,
                score_config_sha=score_config_sha,
                retry=retry,
            )
            if len(existing) == len(DYNAMIC_SCORE_ARMS):
                continue
            cache = _load_cache(encoder_payload)
            _commit_series(
                root,
                record,
                variant,
                encoder_payload,
                cache,
                existing,
                identity,
                device=device,
                query_chunk_size=query_chunk_size,
            )
        except Exception as exc:
            _atomic_json(
                _failure_path(config, variant.key, record.series_id),
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": _utc_now(),
                    "series_id": record.series_id,
                    "variant": {**asdict(variant), "image_size": list(variant.image_size)},
                    "variant_sha256": variant_sha,
                    "config_sha256": config_sha,
                    "manifest_sha256": manifest_sha,
                    "encoder_source_sha256": encoder_source_sha,
                    "dynamic_score_source_sha256": score_source_sha,
                    "scoring_config_sha256": score_config_sha,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            cache = None

    return _write_summary(
        root,
        records,
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        encoder_source_sha=encoder_source_sha,
        score_source_sha=score_source_sha,
        variant_sha=variant_sha,
        score_config_sha=score_config_sha,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-key", default="B16")
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--variant-key")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", dest="all_series", action="store_true")
    parser.add_argument("--series-id", action="append", default=[])
    parser.add_argument("--approved-bulk", action="store_true")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args(argv)
    path = run_dynamic_scores(
        args.config,
        model_key=args.model_key,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
        variant_key=args.variant_key,
        device=args.device,
        query_chunk_size=args.query_chunk_size,
        smoke=args.smoke,
        all_series=args.all_series,
        series_ids=args.series_id,
        approved_bulk=args.approved_bulk,
        retry=args.retry,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
