"""Transactional experiment artifact I/O.

Score payloads become evaluator-visible only after their hashes and manifest have
been durably written.  This module intentionally has no dataset or label I/O.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np
import torch

from .schemas import ScoreManifest


def sha256_file(path: Path, chunk_bytes: int = 8_388_608) -> str:
    """Return the SHA256 of *path* without loading the complete file."""

    path = Path(path)
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _replace_flushed(temp_path: Path, final_path: Path) -> None:
    os.replace(temp_path, final_path)
    # On Windows there is no portable directory fsync. os.replace still gives
    # the required atomic name transition within one directory.


def atomic_write_json(path: Path, payload: Mapping[str, Any] | Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_flushed(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_save_numpy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        with temp_path.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_flushed(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_save_checkpoint(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        with temp_path.open("wb") as handle:
            torch.save(dict(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_flushed(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _manifest_payload(manifest: ScoreManifest, score_hash: str) -> dict[str, Any]:
    payload = _jsonable(manifest)
    if not isinstance(payload, dict):
        raise TypeError("ScoreManifest must serialize to a mapping")
    existing = payload.get("score_sha256")
    # A frozen ScoreManifest validates SHA syntax at construction time, so the
    # runner uses 64 zeroes as the explicit pre-commit sentinel.  The durable
    # file hash replaces that sentinel here before the manifest is exposed.
    if existing not in (None, "", "0" * 64, score_hash):
        raise ValueError("manifest score_sha256 conflicts with committed payload")
    payload["score_sha256"] = score_hash
    payload["labels_read"] = False
    return payload


def commit_score_artifact(
    run_dir: Path,
    scores: np.ndarray,
    manifest: ScoreManifest,
) -> Path:
    """Commit scores, manifest, then a success marker in that exact order."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    success_path = run_dir / "_SUCCESS"
    success_path.unlink(missing_ok=True)

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite float32 vector")
    if int(manifest.num_points) != int(values.size):
        raise ValueError("score length does not match manifest num_points")

    score_path = run_dir / "scores.npy"
    atomic_save_numpy(score_path, values)
    score_hash = sha256_file(score_path)
    atomic_write_json(run_dir / "score_manifest.json", _manifest_payload(manifest, score_hash))

    # Verify both durable payloads before making the artifact discoverable.
    loaded = np.load(score_path, allow_pickle=False)
    if loaded.dtype != np.float32 or loaded.shape != values.shape:
        raise RuntimeError("committed score payload failed round-trip verification")
    if sha256_file(score_path) != score_hash:
        raise RuntimeError("score hash changed before success commit")
    with success_path.open("x", encoding="ascii") as handle:
        handle.write(f"{score_hash}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return success_path


def verify_committed_score(run_dir: Path) -> tuple[np.ndarray, ScoreManifest]:
    """Verify a committed score without touching labels or metric code."""

    run_dir = Path(run_dir)
    success_path = run_dir / "_SUCCESS"
    if not success_path.is_file():
        raise FileNotFoundError(f"missing score success marker: {success_path}")
    manifest_path = run_dir / "score_manifest.json"
    score_path = run_dir / "scores.npy"
    if not manifest_path.is_file() or not score_path.is_file():
        raise FileNotFoundError("score artifact is incomplete")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("labels_read") is not False:
        raise ValueError("runner manifest must record labels_read=false")
    expected_hash = str(payload.get("score_sha256", ""))
    actual_hash = sha256_file(score_path)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("score payload SHA256 mismatch")
    marker_hash = success_path.read_text(encoding="ascii").strip()
    if marker_hash != expected_hash:
        raise ValueError("success marker does not match the score payload")

    scores = np.load(score_path, allow_pickle=False)
    if scores.dtype != np.float32 or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("committed score payload has an invalid schema")
    manifest = ScoreManifest.from_dict(payload)
    if scores.size != manifest.num_points:
        raise ValueError("committed score length does not match manifest")
    return scores, manifest
