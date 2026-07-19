"""Label-free manifest builder for the eleven VLM4TS paper groups."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import sha256_file


SCHEMA_VERSION = 1
WINDOW_SIZE = 240
STEP_SIZE = 60
OFFICIAL_GROUPS = (
    ("artificialWithAnomaly", "NAB", "NAB-Artificial", 6),
    ("realAWSCloudwatch", "NAB", "NAB-AWS", 17),
    ("realAdExchange", "NAB", "NAB-AdExchange", 5),
    ("realTraffic", "NAB", "NAB-Traffic", 7),
    ("realTweets", "NAB", "NAB-Tweets", 10),
    ("MSL", "NASA", "NASA-MSL", 27),
    ("SMAP", "NASA", "NASA-SMAP", 53),
    ("YAHOOA1", "Yahoo", "Yahoo-A1", 67),
    ("YAHOOA2", "Yahoo", "Yahoo-A2", 100),
    ("YAHOOA3", "Yahoo", "Yahoo-A3", 100),
    ("YAHOOA4", "Yahoo", "Yahoo-A4", 100),
)
EXPECTED_SERIES = 492
EXPECTED_POINTS = 1_541_932
EXPECTED_WINDOWS = 24_057
EXPECTED_DUPLICATE_TIMESTAMP_FILES = 16
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_FORBIDDEN_KEY_PARTS = ("label", "target", "anomaly", "groundtruth")


@dataclass(frozen=True)
class FullSeriesRecord:
    """One immutable scoring record with no representable ground truth."""

    series_id: str
    dataset: str
    track: str
    paper_group: str
    signal_name: str
    relative_path: str
    expected_length: int
    expected_windows: int
    expected_sha256: str
    duplicate_timestamps: bool


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _reject_label_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in {"y", "gt"} or any(
                part in normalized for part in _FORBIDDEN_KEY_PARTS
            ):
                raise ValueError(f"label-like manifest key is forbidden: {key!r}")
            _reject_label_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_label_keys(child)


def _read_dataset_index(path: Path) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"datasets.csv row {number} must contain two fields")
            dataset = str(row[0])
            parsed = ast.literal_eval(row[1])
            if not isinstance(parsed, tuple) or not parsed:
                raise ValueError(f"datasets.csv row {number} is not a non-empty tuple")
            names = tuple(str(value) for value in parsed)
            if len(names) != len(set(names)) or any(not name for name in names):
                raise ValueError(f"datasets.csv row {number} has invalid signal names")
            if dataset in output:
                raise ValueError(f"duplicate datasets.csv dataset {dataset}")
            output[dataset] = names
    return output


def _read_signal_metadata(path: Path) -> tuple[int, bool]:
    timestamps: list[float] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["timestamp", "value"]:
            raise ValueError(f"unexpected scoring CSV schema: {path}")
        count = 0
        for row in reader:
            timestamp = float(row["timestamp"])
            value = float(row["value"])
            if not math.isfinite(timestamp) or not math.isfinite(value):
                raise ValueError(f"non-finite scoring value: {path}")
            timestamps.append(timestamp)
            count += 1
    if count < WINDOW_SIZE:
        raise ValueError(f"series is shorter than the frozen window: {path}")
    return count, len(set(timestamps)) != count


def build_manifest(data_root: Path) -> dict[str, Any]:
    """Build the exact label-free full-main manifest from the public index."""

    root = Path(data_root).resolve(strict=True)
    index_path = root / "datasets.csv"
    index = _read_dataset_index(index_path)
    records: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for dataset, track, paper_group, expected_count in OFFICIAL_GROUPS:
        if dataset not in index:
            raise KeyError(f"official dataset missing from datasets.csv: {dataset}")
        signals = index[dataset]
        if len(signals) != expected_count:
            raise ValueError(f"official count mismatch for {dataset}")
        directory = root / dataset
        actual = {path.stem for path in directory.glob("*.csv")}
        if actual != set(signals):
            raise ValueError(f"datasets.csv/file-set mismatch for {dataset}")
        group_rows.append(
            {
                "dataset": dataset,
                "track": track,
                "paper_group": paper_group,
                "series_count": expected_count,
            }
        )
        for signal_name in signals:
            series_id = f"{dataset}__{signal_name}"
            if not _SAFE_ID.fullmatch(series_id):
                raise ValueError(f"unsafe full-main series id: {series_id}")
            path = directory / f"{signal_name}.csv"
            length, duplicate_timestamps = _read_signal_metadata(path)
            windows = (length - WINDOW_SIZE) // STEP_SIZE + 1
            records.append(
                {
                    "series_id": series_id,
                    "dataset": dataset,
                    "track": track,
                    "paper_group": paper_group,
                    "signal_name": signal_name,
                    "relative_path": f"{dataset}/{signal_name}.csv",
                    "expected_length": length,
                    "expected_windows": windows,
                    "expected_sha256": sha256_file(path).upper(),
                    "duplicate_timestamps": bool(duplicate_timestamps),
                }
            )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_index": "datasets.csv",
        "source_index_sha256": sha256_file(index_path).upper(),
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "groups": group_rows,
        "series": records,
        "totals": {
            "groups": len(group_rows),
            "series": len(records),
            "points": sum(int(row["expected_length"]) for row in records),
            "windows": sum(int(row["expected_windows"]) for row in records),
            "duplicate_timestamp_files": sum(
                bool(row["duplicate_timestamps"]) for row in records
            ),
        },
    }
    validate_manifest(payload)
    return payload


def validate_manifest(payload: Mapping[str, Any]) -> tuple[FullSeriesRecord, ...]:
    """Fail closed unless payload is the complete official 11-group manifest."""

    if not isinstance(payload, Mapping):
        raise TypeError("full manifest must be a mapping")
    _reject_label_keys(payload)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported full-manifest schema")
    if int(payload.get("window_size", -1)) != WINDOW_SIZE or int(
        payload.get("step_size", -1)
    ) != STEP_SIZE:
        raise ValueError("full-manifest window geometry changed")
    index_hash = str(payload.get("source_index_sha256", "")).upper()
    if len(index_hash) != 64 or any(char not in "0123456789ABCDEF" for char in index_hash):
        raise ValueError("full-manifest source index hash is invalid")

    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) != len(OFFICIAL_GROUPS):
        raise ValueError("full manifest must contain exactly eleven groups")
    for row, expected in zip(groups, OFFICIAL_GROUPS, strict=True):
        dataset, track, paper_group, count = expected
        if row != {
            "dataset": dataset,
            "track": track,
            "paper_group": paper_group,
            "series_count": count,
        }:
            raise ValueError(f"full-manifest group identity changed: {paper_group}")

    raw_records = payload.get("series")
    if not isinstance(raw_records, list) or len(raw_records) != EXPECTED_SERIES:
        raise ValueError("full manifest must contain exactly 492 series")
    allowed = {
        "series_id",
        "dataset",
        "track",
        "paper_group",
        "signal_name",
        "relative_path",
        "expected_length",
        "expected_windows",
        "expected_sha256",
        "duplicate_timestamps",
    }
    group_map = {dataset: (track, paper_group, count) for dataset, track, paper_group, count in OFFICIAL_GROUPS}
    counts = {paper_group: 0 for _, _, paper_group, _ in OFFICIAL_GROUPS}
    records: list[FullSeriesRecord] = []
    seen: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != allowed:
            raise ValueError("full-manifest series fields differ from the frozen schema")
        record = FullSeriesRecord(**raw)
        if record.series_id in seen or not _SAFE_ID.fullmatch(record.series_id):
            raise ValueError(f"duplicate or unsafe full-main series id: {record.series_id}")
        seen.add(record.series_id)
        if record.dataset not in group_map:
            raise ValueError(f"unknown full-main dataset: {record.dataset}")
        track, paper_group, _ = group_map[record.dataset]
        if record.track != track or record.paper_group != paper_group:
            raise ValueError("full-main track/group identity mismatch")
        if record.series_id != f"{record.dataset}__{record.signal_name}":
            raise ValueError("full-main series id is not canonical")
        if record.relative_path != f"{record.dataset}/{record.signal_name}.csv":
            raise ValueError("full-main relative path is not canonical")
        if record.expected_length < WINDOW_SIZE:
            raise ValueError("full-main series is shorter than the frozen window")
        expected_windows = (record.expected_length - WINDOW_SIZE) // STEP_SIZE + 1
        if record.expected_windows != expected_windows:
            raise ValueError("full-main expected window count is inconsistent")
        digest = str(record.expected_sha256).upper()
        if len(digest) != 64 or any(char not in "0123456789ABCDEF" for char in digest):
            raise ValueError("full-main file hash is invalid")
        if not isinstance(record.duplicate_timestamps, bool):
            raise ValueError("duplicate_timestamps must be boolean")
        counts[paper_group] += 1
        records.append(record)

    for _, _, paper_group, expected_count in OFFICIAL_GROUPS:
        if counts[paper_group] != expected_count:
            raise ValueError(f"full-main group count mismatch: {paper_group}")
    totals = payload.get("totals")
    expected_totals = {
        "groups": len(OFFICIAL_GROUPS),
        "series": EXPECTED_SERIES,
        "points": EXPECTED_POINTS,
        "windows": EXPECTED_WINDOWS,
        "duplicate_timestamp_files": EXPECTED_DUPLICATE_TIMESTAMP_FILES,
    }
    if totals != expected_totals:
        raise ValueError("full-manifest totals differ from the frozen official workload")
    return tuple(records)


def load_manifest(path: Path) -> tuple[dict[str, Any], tuple[FullSeriesRecord, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("full manifest JSON must be a mapping")
    return payload, validate_manifest(payload)


def write_manifest(data_root: Path, output: Path) -> Path:
    payload = build_manifest(data_root)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_manifest(args.data_root, args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
