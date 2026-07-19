"""Cache-only CLI for the three mandatory ViTTrace v3 encoder controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
import uuid
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from measure_vit4ts.full_manifest import EXPECTED_SERIES, FullSeriesRecord

from . import encoder_runner as encoder_stage
from .dynamic_cache import (
    CACHE_FILE,
    CACHE_MANIFEST,
    DynamicCacheKey,
    cache_digest,
    load_dynamic_cache,
)
from .encoder_controls import CONTROL_ARMS, compute_encoder_controls


SCHEMA_VERSION = 1
DEFAULT_CONFIG = encoder_stage.DEFAULT_CONFIG
_SOURCE_FILES = ("encoder_controls.py", "encoder_control_runner.py")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def control_source_sha256(package_root: Path | None = None) -> str:
    root = Path(package_root) if package_root else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _SOURCE_FILES:
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


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
        raise ValueError("control score must be finite float64 [T]")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _sha256(path)


def _mandatory_variant(variant: encoder_stage.EncoderVariant) -> None:
    if not (
        variant.representation == "line"
        and variant.model_key == "B16"
        and variant.model_name == "ViT-B-16"
        and variant.pretrained == "openai"
        and variant.image_size == (224, 224)
        and variant.patch_size == 16
        and variant.window == 240
    ):
        raise ValueError("encoder controls require the registered line B16 W240 cache")


def _key_from_payload(payload: Mapping[str, Any]) -> DynamicCacheKey:
    values = dict(payload)
    values["image_size"] = tuple(int(value) for value in values["image_size"])
    return DynamicCacheKey(**values)

def _expected_sha256(value: Any, label: str) -> str:
    digest = str(value).upper()
    if len(digest) != 64 or any(
        character not in "0123456789ABCDEF" for character in digest
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal digest")
    return digest


def _shape(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a shape list")
    shape = tuple(int(dimension) for dimension in value)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{label} must contain positive dimensions")
    return shape


def _read_npy_header(handle: Any) -> tuple[tuple[int, ...], np.dtype[Any], bool]:
    version = np.lib.format.read_magic(handle)
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError(f"unsupported NPY version: {version}")
    shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
    canonical = np.dtype(dtype)
    if canonical.hasobject:
        raise ValueError("object arrays are forbidden in encoder transactions")
    return tuple(int(value) for value in shape), canonical, bool(fortran_order)


def _npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype[Any], bool]:
    with Path(path).open("rb") as handle:
        return _read_npy_header(handle)


def _npz_headers(
    path: Path,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any], bool]]:
    expected = (
        "global_tokens",
        "patch_tokens",
        "mid_tokens",
        "large_tokens",
        "mid_mask",
        "large_mask",
    )
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.infolist()
        expected_members = {f"{name}.npy" for name in expected}
        if (
            len(members) != len(expected_members)
            or {item.filename for item in members} != expected_members
        ):
            raise ValueError("dynamic NPZ member registry changed")
        headers: dict[str, tuple[tuple[int, ...], np.dtype[Any], bool]] = {}
        by_name = {item.filename: item for item in members}
        for name in expected:
            with archive.open(by_name[f"{name}.npy"], "r") as handle:
                headers[name] = _read_npy_header(handle)
    return headers


def _preflight_encoder_artifact(
    root: Path,
    record: FullSeriesRecord,
    *,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    variant_sha: str,
) -> Mapping[str, Any]:
    """Validate one encoder transaction without materializing token arrays."""

    record_path = root / "records" / f"{record.series_id}.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"encoder record is missing: {record.series_id}")
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("encoder record must be a JSON object")
    if (
        payload.get("schema_version") != encoder_stage.SCHEMA_VERSION
        or payload.get("status") != "PASS"
        or payload.get("series_id") != record.series_id
        or payload.get("config_sha256") != config_sha
        or payload.get("manifest_sha256") != manifest_sha
        or payload.get("encoder_source_sha256") != encoder_source_sha
        or payload.get("variant_sha256") != variant_sha
        or str(payload.get("data_sha256", "")).upper()
        != record.expected_sha256.upper()
    ):
        raise ValueError("encoder record identity/hash binding differs")

    key = _key_from_payload(payload["cache_key"])
    if (
        key.series_id != record.series_id
        or key.data_sha256.upper() != record.expected_sha256.upper()
    ):
        raise ValueError("dynamic cache key differs from the manifest record")
    cache_dir = Path(payload["cache_dir"])
    if not cache_dir.is_dir() or cache_dir.name != cache_digest(key):
        raise ValueError("dynamic cache directory identity differs")
    cache_path = cache_dir / CACHE_FILE
    cache_manifest_path = cache_dir / CACHE_MANIFEST
    patch_mean_path = Path(payload["patch_mean_path"])
    if (
        not cache_path.is_file()
        or not cache_manifest_path.is_file()
        or not patch_mean_path.is_file()
    ):
        raise FileNotFoundError("encoder transaction payload/manifest/sidecar is missing")
    if patch_mean_path.resolve() != (
        cache_dir / encoder_stage.PATCH_MEAN_FILE
    ).resolve():
        raise ValueError("patch-mean sidecar path differs from the registered cache path")

    cache_sha = _sha256(cache_path)
    cache_manifest_sha = _sha256(cache_manifest_path)
    patch_mean_sha = _sha256(patch_mean_path)
    if cache_sha != _expected_sha256(
        payload["cache_file_sha256"], "cache_file_sha256"
    ):
        raise ValueError("dynamic cache file hash differs from the encoder record")
    if cache_manifest_sha != _expected_sha256(
        payload["cache_manifest_sha256"], "cache_manifest_sha256"
    ):
        raise ValueError("dynamic cache manifest hash differs from the encoder record")
    if patch_mean_sha != _expected_sha256(
        payload["patch_mean_sha256"], "patch_mean_sha256"
    ):
        raise ValueError("patch-mean file hash differs from the encoder record")

    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(cache_manifest, dict)
        or cache_manifest.get("schema_version") != 1
    ):
        raise ValueError("dynamic cache manifest schema differs")
    if cache_manifest.get("file") != CACHE_FILE:
        raise ValueError("dynamic cache manifest file registry differs")
    if _key_from_payload(cache_manifest["key"]) != key:
        raise ValueError("dynamic cache manifest key differs from the encoder record")
    if cache_sha != _expected_sha256(
        cache_manifest["sha256"], "cache manifest payload hash"
    ):
        raise ValueError("dynamic cache payload hash differs from its manifest")

    shape_payload = cache_manifest.get("shapes")
    required_shapes = {
        "global_tokens",
        "patch_tokens",
        "mid_tokens",
        "large_tokens",
        "mid_mask",
        "large_mask",
    }
    if not isinstance(shape_payload, dict) or set(shape_payload) != required_shapes:
        raise ValueError("dynamic cache shape registry changed")
    shapes = {name: _shape(shape_payload[name], name) for name in required_shapes}
    headers = _npz_headers(cache_path)
    token_names = ("global_tokens", "patch_tokens", "mid_tokens", "large_tokens")
    for name in required_shapes:
        actual_shape, actual_dtype, fortran_order = headers[name]
        expected_dtype = np.dtype(np.float32 if name in token_names else np.int64)
        if (
            actual_shape != shapes[name]
            or actual_dtype != expected_dtype
            or fortran_order
        ):
            raise ValueError(f"dynamic cache {name} header differs from its metadata")

    global_shape = shapes["global_tokens"]
    patch_shape = shapes["patch_tokens"]
    mid_shape = shapes["mid_tokens"]
    large_shape = shapes["large_tokens"]
    mid_mask_shape = shapes["mid_mask"]
    large_mask_shape = shapes["large_mask"]
    patch_grid = _shape(cache_manifest["patch_grid"], "patch_grid")
    expected_grid = (
        key.image_size[0] // key.patch_size,
        key.image_size[1] // key.patch_size,
    )
    window_count = int(payload["window_count"])
    expected_windows = (record.expected_length - key.window) // key.stride + 1
    if (
        len(global_shape) != 2
        or len(patch_shape) != 3
        or len(mid_shape) != 3
        or len(large_shape) != 3
        or len(mid_mask_shape) != 2
        or len(large_mask_shape) != 2
        or patch_grid != expected_grid
        or tuple(payload["patch_grid"]) != patch_grid
        or window_count != expected_windows
        or global_shape[0] != window_count
        or patch_shape[0] != window_count
        or mid_shape[0] != window_count
        or large_shape[0] != window_count
        or patch_shape[1] != patch_grid[0] * patch_grid[1]
        or not (patch_shape[2] == mid_shape[2] == large_shape[2])
        or mid_shape[1] != (patch_grid[0] - 1) * (patch_grid[1] - 1)
        or large_shape[1] != (patch_grid[0] - 2) * (patch_grid[1] - 2)
        or mid_mask_shape != (4, mid_shape[1])
        or large_mask_shape != (9, large_shape[1])
    ):
        raise ValueError("dynamic cache shapes are internally inconsistent")
    encoder_shapes = {
        "global_tokens": "global_tokens_shape",
        "patch_tokens": "patch_tokens_shape",
        "mid_tokens": "mid_tokens_shape",
        "large_tokens": "large_tokens_shape",
    }
    for cache_name, record_name in encoder_shapes.items():
        if _shape(payload[record_name], record_name) != shapes[cache_name]:
            raise ValueError(f"{record_name} differs from dynamic cache metadata")

    patch_mean_shape, patch_mean_dtype, patch_mean_fortran = _npy_header(
        patch_mean_path
    )
    if (
        patch_mean_shape != _shape(payload["patch_mean_shape"], "patch_mean_shape")
        or patch_mean_shape != (window_count, patch_shape[2])
        or patch_mean_dtype != np.dtype(np.float32)
        or patch_mean_fortran
    ):
        raise ValueError("patch-mean NPY header differs from the encoder transaction")
    _expected_sha256(payload["global_tokens_sha256"], "global_tokens_sha256")
    _expected_sha256(payload["patch_tokens_sha256"], "patch_tokens_sha256")
    _expected_sha256(
        payload["patch_mean_array_sha256"], "patch_mean_array_sha256"
    )
    return payload


def _load_encoder_artifact(
    root: Path,
    record: FullSeriesRecord,
    *,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    variant_sha: str,
) -> tuple[Mapping[str, Any], Any, np.ndarray]:
    payload = _preflight_encoder_artifact(
        root,
        record,
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        encoder_source_sha=encoder_source_sha,
        variant_sha=variant_sha,
    )
    cache = load_dynamic_cache(Path(payload["cache_dir"]), _key_from_payload(payload["cache_key"]))
    if encoder_stage._array_sha256(cache.global_tokens) != payload["global_tokens_sha256"]:
        raise ValueError("true-global array hash differs from the encoder transaction")
    if encoder_stage._array_sha256(cache.patch_tokens) != payload["patch_tokens_sha256"]:
        raise ValueError("base-patch array hash differs from the encoder transaction")
    mean_path = Path(payload["patch_mean_path"])
    patch_mean = np.load(mean_path, allow_pickle=False)
    if (
        patch_mean.dtype != np.float32
        or patch_mean.shape != tuple(payload["patch_mean_shape"])
        or encoder_stage._array_sha256(patch_mean) != payload["patch_mean_array_sha256"]
    ):
        raise ValueError("patch-mean sidecar differs from the encoder transaction")
    derived = encoder_stage.patch_mean_tokens(cache)
    if not np.array_equal(patch_mean, derived):
        raise ValueError("patch-mean sidecar is not the exact mean of base patch tokens")
    if cache.global_tokens.shape[0] != patch_mean.shape[0]:
        raise ValueError("true-global and patch-mean window counts differ")
    return payload, cache, np.ascontiguousarray(patch_mean)


def _transaction_path(root: Path, series_id: str, arm: str) -> Path:
    return root / "controls" / series_id / arm / "score_manifest.json"


def _resume_transaction(
    path: Path,
    *,
    arm: str,
    record: FullSeriesRecord,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    control_source_sha: str,
    variant_sha: str,
    encoder_record_sha: str,
    cache_sha: str,
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("status") != "PASS"
            or payload.get("series_id") != record.series_id
            or payload.get("arm") != arm
            or str(payload.get("data_sha256", "")).upper() != record.expected_sha256.upper()
            or payload.get("config_sha256") != config_sha
            or payload.get("manifest_sha256") != manifest_sha
            or payload.get("encoder_source_sha256") != encoder_source_sha
            or payload.get("control_source_sha256") != control_source_sha
            or payload.get("variant_sha256") != variant_sha
            or payload.get("encoder_record_sha256") != encoder_record_sha
            or payload.get("cache_file_sha256") != cache_sha
        ):
            return None
        score_path = Path(payload["score_path"])
        score = np.load(score_path, allow_pickle=False)
        if (
            _sha256(score_path) != payload["score_sha256"]
            or score.dtype != np.float64
            or score.shape != (record.expected_length,)
            or not np.isfinite(score).all()
        ):
            return None
        metadata = payload.get("metadata", {})
        if metadata.get("ihp") != "bypassed" or metadata.get("nctp") != "bypassed":
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _run_series(
    root: Path,
    record: FullSeriesRecord,
    variant: encoder_stage.EncoderVariant,
    encoder_payload: Mapping[str, Any],
    cache: Any,
    patch_mean: np.ndarray,
    *,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    control_source_sha: str,
    variant_sha: str,
    retry: bool,
) -> dict[str, Mapping[str, Any]]:
    encoder_record_path = root / "records" / f"{record.series_id}.json"
    encoder_record_sha = _sha256(encoder_record_path)
    cache_sha = _sha256(Path(encoder_payload["cache_dir"]) / CACHE_FILE)
    existing: dict[str, Mapping[str, Any]] = {}
    for arm in CONTROL_ARMS:
        if not retry:
            transaction = _resume_transaction(
                _transaction_path(root, record.series_id, arm),
                arm=arm,
                record=record,
                config_sha=config_sha,
                manifest_sha=manifest_sha,
                encoder_source_sha=encoder_source_sha,
                control_source_sha=control_source_sha,
                variant_sha=variant_sha,
                encoder_record_sha=encoder_record_sha,
                cache_sha=cache_sha,
            )
            if transaction is not None:
                existing[arm] = transaction
    if len(existing) == len(CONTROL_ARMS):
        return existing
    window_count = int(encoder_payload["window_count"])
    starts = np.arange(window_count, dtype=np.int64) * int(variant.stride)
    results = compute_encoder_controls(
        cache.global_tokens,
        patch_mean,
        starts,
        full_length=record.expected_length,
        window=variant.window,
        stride=variant.stride,
        device="cpu",
    )
    if set(results) != set(CONTROL_ARMS):
        raise RuntimeError("mandatory encoder-control registry changed")
    output: dict[str, Mapping[str, Any]] = {}
    for arm in CONTROL_ARMS:
        result = results[arm]
        manifest_path = _transaction_path(root, record.series_id, arm)
        score_path = manifest_path.parent / "score.npy"
        score_sha = _atomic_score(score_path, result.score, record.expected_length)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "created_at": _utc_now(),
            "series_id": record.series_id,
            "family": record.track,
            "subgroup": record.paper_group,
            "arm": arm,
            "data_sha256": record.expected_sha256.upper(),
            "config_sha256": config_sha,
            "manifest_sha256": manifest_sha,
            "encoder_source_sha256": encoder_source_sha,
            "control_source_sha256": control_source_sha,
            "variant_sha256": variant_sha,
            "encoder_record_path": str(encoder_record_path.resolve()),
            "encoder_record_sha256": encoder_record_sha,
            "cache_file_sha256": cache_sha,
            "true_global_shape": list(cache.global_tokens.shape),
            "true_global_sha256": encoder_stage._array_sha256(cache.global_tokens),
            "patch_mean_shape": list(patch_mean.shape),
            "patch_mean_sha256": encoder_stage._array_sha256(patch_mean),
            "window_scalar_sha256": encoder_stage._array_sha256(result.window_scalar),
            "score_path": str(score_path.resolve()),
            "score_sha256": score_sha,
            "score_length": int(result.score.size),
            "score_dtype": "float64",
            "metadata": dict(result.metadata),
        }
        _atomic_json(manifest_path, payload)
        output[arm] = payload
    return output


def _failure_path(config: Mapping[str, Any], variant_key: str, series_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        Path(config["paths"]["failure_root"])
        / "encoder_controls"
        / variant_key
        / series_id
        / f"{stamp}_{uuid.uuid4().hex}.json"
    )


def _write_summary(
    root: Path,
    records: Sequence[FullSeriesRecord],
    *,
    config_sha: str,
    manifest_sha: str,
    encoder_source_sha: str,
    control_source_sha: str,
    variant_sha: str,
) -> Path:
    rows: list[dict[str, Any]] = []
    for record in records:
        for arm in CONTROL_ARMS:
            path = _transaction_path(root, record.series_id, arm)
            status = "NOT_RUN"
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("config_sha256") == config_sha
                    and payload.get("manifest_sha256") == manifest_sha
                    and payload.get("encoder_source_sha256") == encoder_source_sha
                    and payload.get("control_source_sha256") == control_source_sha
                    and payload.get("variant_sha256") == variant_sha
                    and payload.get("status") == "PASS"
                ):
                    status = "PASS"
            rows.append({"series_id": record.series_id, "arm": arm, "status": status})
    completed = len({row["series_id"] for row in rows if row["status"] == "PASS"})
    complete_rows = sum(row["status"] == "PASS" for row in rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "COMPLETE" if complete_rows == EXPECTED_SERIES * len(CONTROL_ARMS) else "INCOMPLETE",
        "expected_series": EXPECTED_SERIES,
        "expected_arms": list(CONTROL_ARMS),
        "expected_rows": EXPECTED_SERIES * len(CONTROL_ARMS),
        "completed_series": completed,
        "completed_rows": complete_rows,
        "config_sha256": config_sha,
        "manifest_sha256": manifest_sha,
        "encoder_source_sha256": encoder_source_sha,
        "control_source_sha256": control_source_sha,
        "variant_sha256": variant_sha,
        "rows": rows,
    }
    path = root / "encoder_controls_status.json"
    _atomic_json(path, payload)
    return path


def run_encoder_controls(
    config_path: Path,
    *,
    stride: int | None = None,
    batch_size: int | None = None,
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
        model_key="B16",
        window=240,
        stride=stride,
        batch_size=batch_size,
    )
    _mandatory_variant(variant)
    selected = encoder_stage.select_records(
        records,
        smoke=smoke,
        all_series=all_series,
        series_ids=series_ids,
        approved_bulk=approved_bulk,
    )
    encoder_source_sha = encoder_stage.encoder_source_sha256()
    control_source_sha = control_source_sha256()
    variant_sha = encoder_stage.variant_sha256(variant, config_sha)
    root = encoder_stage._variant_root(config, variant, variant_sha)

    # Metadata-only global preflight: never retain DynamicTokenCache arrays here.
    invalid: list[str] = []
    for record in selected:
        try:
            _preflight_encoder_artifact(
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
            f"encoder-control preflight found {len(invalid)} missing/invalid encoder transactions; "
            f"first={invalid[0]}"
        )
    for record in selected:
        artifact: tuple[Mapping[str, Any], Any, np.ndarray] | None = None
        try:
            # Load, score, and release exactly one series at a time.
            artifact = _load_encoder_artifact(
                root,
                record,
                config_sha=config_sha,
                manifest_sha=manifest_sha,
                encoder_source_sha=encoder_source_sha,
                variant_sha=variant_sha,
            )
            _run_series(
                root,
                record,
                variant,
                artifact[0],
                artifact[1],
                artifact[2],
                config_sha=config_sha,
                manifest_sha=manifest_sha,
                encoder_source_sha=encoder_source_sha,
                control_source_sha=control_source_sha,
                variant_sha=variant_sha,
                retry=retry,
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
                    "control_source_sha256": control_source_sha,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            # Prevent a one-iteration overlap when the next large cache loads.
            artifact = None
    return _write_summary(
        root,
        records,
        config_sha=config_sha,
        manifest_sha=manifest_sha,
        encoder_source_sha=encoder_source_sha,
        control_source_sha=control_source_sha,
        variant_sha=variant_sha,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", dest="all_series", action="store_true")
    parser.add_argument("--series-id", action="append", default=[])
    parser.add_argument("--approved-bulk", action="store_true")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args(argv)
    path = run_encoder_controls(
        args.config,
        stride=args.stride,
        batch_size=args.batch_size,
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
