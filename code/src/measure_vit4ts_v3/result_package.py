"""Build and verify a compact, provenance-preserving v3 result package.

The archive is intentionally allow-list based.  It contains code, compact
tables, manifests, and vector figures, but never datasets, model weights, or
token-cache payloads.  ``SHA256SUMS.csv`` covers every payload member; the
non-recursive ``PACKAGE_MANIFEST.json`` records the checksum-file digest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PACKAGE_SCHEMA_VERSION = "vittrace-v3-results-package/1"
CHECKSUM_COLUMNS = ("relative_path", "size_bytes", "sha256", "category")
CHECKSUM_NAME = "SHA256SUMS.csv"
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
FIXED_ZIP_DATETIME = (1980, 1, 1, 0, 0, 0)

ALLOWED_TOP_LEVEL_FILES = frozenset(
    {
        "README.md",
        "STATUS.md",
        "EXPERIMENT_LOG.md",
        "arm_registry.csv",
        "arm_registry.json",
        "external_vit4ts_reference.csv",
        "protocol.json",
        "input_identities.csv",
        "_SUPPLEMENT_DELIVERY_COMPLETE.json",
        "CONFIRMATION_BLOCKED.md",
    }
)
ALLOWED_TOP_LEVEL_DIRS = frozenset(
    {
        "config",
        "code",
        "tests",
        "manifests",
        "provenance",
        "failures",
        "results",
        "tables",
        "plot_data",
        "rough_figures",
        "qualitative",
        "runtime",
    }
)
REQUIRED_PATHS = (
    "README.md",
    "STATUS.md",
    "EXPERIMENT_LOG.md",
    "config",
    "code",
    "tests",
    "arm_registry.csv",
    "arm_registry.json",
    "manifests",
    "provenance",
    "failures/failure_manifest.csv",
    "results",
    "tables",
    "plot_data",
    "rough_figures",
    "external_vit4ts_reference.csv",
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".pb",
        ".joblib",
        ".pickle",
        ".pkl",
    }
)
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".latex-build",
        "datasets",
        "dataset",
        "data",
        "models",
        "model_weights",
        "weights",
        "checkpoints",
        "token_cache",
        "token_caches",
        "runs",
        "logs",
        "__pycache__",
        ".pytest_cache",
    }
)


class PackageValidationError(ValueError):
    """Raised when a package would violate the compact-results contract."""


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _is_reparse_point(path: Path) -> bool:
    info = path.lstat()
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_attribute)


def _category(relative: str) -> str:
    first = PurePosixPath(relative).parts[0]
    return "top_level" if len(PurePosixPath(relative).parts) == 1 else first


def _iter_allowed_files(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield allowed files while rejecting unsafe entries in allowed trees."""

    for name in sorted(ALLOWED_TOP_LEVEL_FILES):
        path = root / name
        if path.exists():
            if not path.is_file() or _is_reparse_point(path):
                raise PackageValidationError(f"unsafe top-level package entry: {path}")
            yield name, path

    for directory_name in sorted(ALLOWED_TOP_LEVEL_DIRS):
        directory = root / directory_name
        if not directory.exists():
            continue
        if not directory.is_dir() or _is_reparse_point(directory):
            raise PackageValidationError(f"unsafe package directory: {directory}")
        for current, directories, files in os.walk(directory, followlinks=False):
            current_path = Path(current)
            safe_directories: list[str] = []
            for child_name in sorted(directories):
                child = current_path / child_name
                relative_parts = child.relative_to(root).parts
                if child_name.lower() in FORBIDDEN_PARTS:
                    continue
                if _is_reparse_point(child):
                    raise PackageValidationError(f"reparse point in package tree: {child}")
                if any(part.lower() in FORBIDDEN_PARTS for part in relative_parts):
                    continue
                safe_directories.append(child_name)
            directories[:] = safe_directories

            for filename in sorted(files):
                path = current_path / filename
                relative_path = path.relative_to(root)
                if _is_reparse_point(path):
                    raise PackageValidationError(f"reparse point in package tree: {path}")
                if any(part.lower() in FORBIDDEN_PARTS for part in relative_path.parts):
                    continue
                if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                    raise PackageValidationError(f"model/artifact payload is forbidden: {path}")
                yield relative_path.as_posix(), path

    cache_index = root / "caches" / "cache_index.csv"
    if cache_index.exists():
        caches = root / "caches"
        if _is_reparse_point(caches) or _is_reparse_point(cache_index):
            raise PackageValidationError("cache index may not be reached through a reparse point")
        if not cache_index.is_file():
            raise PackageValidationError("caches/cache_index.csv is not a regular file")
        yield "caches/cache_index.csv", cache_index


def collect_package_payload(
    root: str | Path,
    *,
    allow_incomplete: bool = False,
    max_file_bytes: int = 128 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Collect and hash every compact payload file in deterministic order."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise PackageValidationError(f"result root does not exist: {root_path}")
    if _is_reparse_point(root_path):
        raise PackageValidationError(f"result root is a reparse point: {root_path}")
    if not allow_incomplete:
        missing = [relative for relative in REQUIRED_PATHS if not (root_path / relative).exists()]
        if missing:
            raise PackageValidationError(f"incomplete v3 result tree; missing: {missing}")
    if max_file_bytes <= 0:
        raise PackageValidationError("max_file_bytes must be positive")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative, path in _iter_allowed_files(root_path):
        if relative in {CHECKSUM_NAME, MANIFEST_NAME}:
            continue
        if relative in seen:
            raise PackageValidationError(f"duplicate package member: {relative}")
        size = path.stat().st_size
        if size > max_file_bytes:
            raise PackageValidationError(
                f"package member exceeds {max_file_bytes} bytes: {relative} ({size})"
            )
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": int(size),
                "sha256": _sha256(path),
                "category": _category(relative),
            }
        )
        seen.add(relative)
    rows.sort(key=lambda item: item["relative_path"])
    if not rows:
        raise PackageValidationError("no allow-listed result files were found")
    return rows


def _checksum_csv(rows: Iterable[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CHECKSUM_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in CHECKSUM_COLUMNS})
    return buffer.getvalue()


def _stable_build_time(rows: list[dict[str, Any]], root: Path) -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        timestamp = int(epoch)
    else:
        timestamp = max(int((root / row["relative_path"]).stat().st_mtime) for row in rows)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def write_package_manifests(
    root: str | Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the payload checksum file and its non-recursive package manifest."""

    root_path = Path(root).resolve()
    checksum_path = root_path / CHECKSUM_NAME
    _atomic_text(checksum_path, _checksum_csv(rows))
    manifest: dict[str, Any] = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "build_time_utc": _stable_build_time(rows, root_path),
        "payload_file_count": len(rows),
        "payload_bytes": int(sum(row["size_bytes"] for row in rows)),
        "sha256sums_file": CHECKSUM_NAME,
        "sha256sums_sha256": _sha256(checksum_path),
        "package_members": [row["relative_path"] for row in rows]
        + [CHECKSUM_NAME, MANIFEST_NAME],
        "exclusion_contract": {
            "datasets": True,
            "model_weights": True,
            "large_token_caches": True,
            "git_and_build_state": True,
            "only_cache_index_is_allowed": True,
        },
        "zip_metadata": {
            "member_order": "lexicographic",
            "member_timestamp": "1980-01-01T00:00:00",
            "member_mode": "0644",
        },
    }
    _atomic_text(
        root_path / MANIFEST_NAME,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return manifest


def verify_package_root(root: str | Path) -> list[dict[str, Any]]:
    """Verify both manifests and every payload file at the result root."""

    root_path = Path(root).resolve()
    checksum_path = root_path / CHECKSUM_NAME
    manifest_path = root_path / MANIFEST_NAME
    if not checksum_path.is_file() or not manifest_path.is_file():
        raise PackageValidationError("package manifests are missing")
    with checksum_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(rows[0].keys()) != CHECKSUM_COLUMNS if rows else True:
        raise PackageValidationError("SHA256SUMS.csv has an invalid schema")
    if [row["relative_path"] for row in rows] != sorted(row["relative_path"] for row in rows):
        raise PackageValidationError("SHA256SUMS.csv is not sorted")
    for row in rows:
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PackageValidationError(f"unsafe checksum path: {relative}")
        path = root_path.joinpath(*relative.parts)
        if not path.is_file() or _is_reparse_point(path):
            raise PackageValidationError(f"missing or unsafe payload file: {relative}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise PackageValidationError(f"payload size mismatch: {relative}")
        if _sha256(path) != row["sha256"]:
            raise PackageValidationError(f"payload SHA256 mismatch: {relative}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise PackageValidationError("package manifest schema mismatch")
    if manifest.get("sha256sums_sha256") != _sha256(checksum_path):
        raise PackageValidationError("SHA256SUMS.csv manifest digest mismatch")
    expected_members = [row["relative_path"] for row in rows] + [CHECKSUM_NAME, MANIFEST_NAME]
    if manifest.get("package_members") != expected_members:
        raise PackageValidationError("package member list mismatch")
    return rows


def _write_reproducible_zip(root: Path, destination: Path, members: Iterable[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(members):
            path = root.joinpath(*PurePosixPath(relative).parts)
            info = zipfile.ZipInfo(relative, FIXED_ZIP_DATETIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits |= 0x800
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(temporary, destination)


def verify_result_zip(path: str | Path) -> dict[str, Any]:
    """Verify member safety, deterministic metadata, and payload checksums."""

    zip_path = Path(path).resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names) or len(names) != len(set(names)):
            raise PackageValidationError("ZIP members are not unique and lexicographically sorted")
        for info in infos:
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or info.is_dir():
                raise PackageValidationError(f"unsafe ZIP member: {info.filename}")
            if info.date_time != FIXED_ZIP_DATETIME:
                raise PackageValidationError(f"non-reproducible ZIP timestamp: {info.filename}")
        if CHECKSUM_NAME not in names or MANIFEST_NAME not in names:
            raise PackageValidationError("ZIP manifests are missing")
        checksum_bytes = archive.read(CHECKSUM_NAME)
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        if hashlib.sha256(checksum_bytes).hexdigest() != manifest.get("sha256sums_sha256"):
            raise PackageValidationError("ZIP checksum-file digest mismatch")
        rows = list(csv.DictReader(checksum_bytes.decode("utf-8").splitlines()))
        expected = sorted([row["relative_path"] for row in rows] + [CHECKSUM_NAME, MANIFEST_NAME])
        if names != expected or manifest.get("package_members") != [
            row["relative_path"] for row in rows
        ] + [CHECKSUM_NAME, MANIFEST_NAME]:
            raise PackageValidationError("ZIP member set does not match manifests")
        for row in rows:
            payload = archive.read(row["relative_path"])
            if len(payload) != int(row["size_bytes"]):
                raise PackageValidationError(f"ZIP member size mismatch: {row['relative_path']}")
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise PackageValidationError(f"ZIP member digest mismatch: {row['relative_path']}")
    return manifest


def build_result_package(
    root: str | Path,
    *,
    zip_path: str | Path | None = None,
    allow_incomplete: bool = False,
    max_file_bytes: int = 128 * 1024 * 1024,
) -> tuple[Path, dict[str, Any]]:
    """Build, immediately verify, and return a compact v3 result archive."""

    root_path = Path(root).resolve()
    rows = collect_package_payload(
        root_path,
        allow_incomplete=allow_incomplete,
        max_file_bytes=max_file_bytes,
    )
    manifest = write_package_manifests(root_path, rows)
    verify_package_root(root_path)
    if zip_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = root_path.parent / f"ViTTrace_ablation_full_v3_results_{stamp}.zip"
    else:
        destination = Path(zip_path).resolve()
    _write_reproducible_zip(root_path, destination, manifest["package_members"])
    verified_manifest = verify_result_zip(destination)
    return destination, verified_manifest


__all__ = [
    "ALLOWED_TOP_LEVEL_DIRS",
    "ALLOWED_TOP_LEVEL_FILES",
    "CHECKSUM_COLUMNS",
    "CHECKSUM_NAME",
    "FIXED_ZIP_DATETIME",
    "MANIFEST_NAME",
    "PACKAGE_SCHEMA_VERSION",
    "PackageValidationError",
    "build_result_package",
    "collect_package_payload",
    "verify_package_root",
    "verify_result_zip",
    "write_package_manifests",
]
