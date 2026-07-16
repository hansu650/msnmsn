"""Build and verify the complete paper-compatible TSB-AD Eval manifest."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import re
from typing import Iterable, Mapping
import uuid

from .config import ProtocolError
from .schemas import SeriesSpec


MANIFEST_COLUMNS = (
    "family",
    "track",
    "file",
    "sha256",
    "rows",
    "channels",
    "train_end",
    "bytes",
    "local_path",
)
EXPECTED_TRACK_COUNTS = {"U": 350, "M": 180}
_FAMILY_RE = re.compile(r"^\d+_([^_]+)_id_")
_TRAIN_RE = re.compile(r"_tr_(\d+)_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_eval_names(path: Path, column: str, expected: int) -> tuple[str, ...]:
    source = Path(path).resolve(strict=True)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or ()):
            raise ProtocolError(f"missing {column!r} in Eval list {source}")
        names = tuple(str(row[column]).strip() for row in reader)
    if len(names) != expected or len(set(names)) != expected:
        raise ProtocolError(
            f"Eval list {source.name} must contain {expected} unique filenames"
        )
    for name in names:
        if not name or Path(name).name != name or not name.lower().endswith(".csv"):
            raise ProtocolError(f"unsafe Eval filename: {name!r}")
    return names


def _parse_identity(filename: str) -> tuple[str, int]:
    family_match = _FAMILY_RE.search(filename)
    train_match = _TRAIN_RE.search(filename)
    if family_match is None or train_match is None:
        raise ProtocolError(f"cannot parse family/train_end from {filename}")
    return family_match.group(1), int(train_match.group(1))


def _inspect_csv(path: Path) -> tuple[int, int, tuple[str, ...], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ProtocolError(f"empty CSV: {path}") from exc
        rows = sum(1 for _ in reader)
    if not header or header[-1] != "Label" or header.count("Label") != 1:
        raise ProtocolError(f"expected one final Label column in {path.name}")
    feature_columns = tuple(header[:-1])
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ProtocolError(f"invalid feature header in {path.name}")
    return rows, len(feature_columns), feature_columns, "Label"


def _record_for_file(track: str, root: Path, filename: str) -> dict[str, str]:
    path = (Path(root).resolve(strict=True) / filename).resolve(strict=True)
    if path.parent != Path(root).resolve(strict=True):
        raise ProtocolError(f"Eval file escaped registered root: {filename}")
    family, train_end = _parse_identity(filename)
    rows, channels, _, _ = _inspect_csv(path)
    if train_end < 96 or train_end > rows:
        raise ProtocolError(f"invalid normal prefix in {filename}")
    return {
        "family": family,
        "track": track,
        "file": filename,
        "sha256": _sha256_file(path),
        "rows": str(rows),
        "channels": str(channels),
        "train_end": str(train_end),
        "bytes": str(path.stat().st_size),
        "local_path": path.as_posix(),
    }


def build_manifest_from_eval_lists(
    u_eval_list: Path,
    m_eval_list: Path,
    u_root: Path,
    m_root: Path,
    output_path: Path,
) -> Path:
    """Create the fixed 350/180 manifest; external score columns are ignored."""

    u_names = _read_eval_names(u_eval_list, "file_name", 350)
    m_names = _read_eval_names(m_eval_list, "file_list", 180)
    records = [
        *(_record_for_file("U", u_root, name) for name in u_names),
        *(_record_for_file("M", m_root, name) for name in m_names),
    ]
    if len({record["file"] for record in records}) != 530:
        raise ProtocolError("U/M Eval filename collision detected")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(strict=True)


def _read_manifest_records(path: Path) -> tuple[dict[str, str], ...]:
    manifest = Path(path).resolve(strict=True)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ProtocolError("full benchmark manifest columns changed")
        records = tuple(dict(row) for row in reader)
    counts = {track: sum(row["track"] == track for row in records) for track in ("U", "M")}
    if counts != EXPECTED_TRACK_COUNTS:
        raise ProtocolError(f"full benchmark track counts changed: {counts}")
    series_ids = [Path(row["file"]).stem for row in records]
    expected_total = sum(EXPECTED_TRACK_COUNTS.values())
    if len(records) != expected_total or len(set(series_ids)) != expected_total:
        raise ProtocolError(
            f"full benchmark requires {expected_total} unique series"
        )
    return records


def _spec_from_record(record: Mapping[str, str]) -> SeriesSpec:
    csv_path = Path(record["local_path"]).resolve(strict=True)
    if csv_path.name != record["file"]:
        raise ProtocolError(f"manifest filename mismatch: {csv_path}")
    if csv_path.stat().st_size != int(record["bytes"]):
        raise ProtocolError(f"byte count mismatch for {csv_path.name}")
    digest = _sha256_file(csv_path)
    if digest != record["sha256"]:
        raise ProtocolError(f"SHA256 mismatch for {csv_path.name}")
    rows, channels, features, label = _inspect_csv(csv_path)
    if rows != int(record["rows"]) or channels != int(record["channels"]):
        raise ProtocolError(f"shape mismatch for {csv_path.name}")
    family, train_end = _parse_identity(csv_path.name)
    if family != record["family"] or train_end != int(record["train_end"]):
        raise ProtocolError(f"identity mismatch for {csv_path.name}")
    return SeriesSpec(
        series_id=csv_path.stem,
        family=family,
        track=record["track"],
        csv_path=csv_path,
        csv_sha256=digest,
        rows=rows,
        channels=channels,
        train_end=train_end,
        feature_columns=features,
        label_column=label,
    )


def load_benchmark_manifest(path: Path) -> tuple[SeriesSpec, ...]:
    """Fully verify all 530 registered files once before a benchmark phase."""

    return tuple(_spec_from_record(record) for record in _read_manifest_records(path))


def load_benchmark_series(path: Path, series_id: str) -> SeriesSpec:
    """Verify only the requested payload while retaining global manifest coverage checks."""

    matches = [
        record
        for record in _read_manifest_records(path)
        if Path(record["file"]).stem == series_id
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"series_id must identify exactly one full benchmark row; found {len(matches)}"
        )
    return _spec_from_record(matches[0])
