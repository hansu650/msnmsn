"""Label-free data access for the measure-consistent ViT4TS route.

This module deliberately exposes no ground-truth loader. The scoring process
can read only timestamp and value columns declared by SeriesSpec.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
from scipy.signal import detrend


_ALLOWED_GROUPS = frozenset({"NAB", "NASA", "Yahoo"})
_FORBIDDEN_FIELD_PARTS = ("label", "target", "anomaly", "groundtruth")


@dataclass(frozen=True)
class SeriesSpec:
    """Scoring-only series description with no representable label field."""

    series_id: str
    group: Literal["NAB", "NASA", "Yahoo"]
    csv_path: Path
    timestamp_column: str
    value_column: str
    expected_sha256: str


@dataclass(frozen=True)
class SeriesData:
    """Hash-bound, chronologically ordered univariate signal."""

    series_id: str
    group: str
    timestamps: np.ndarray
    values: np.ndarray
    data_sha256: str


@dataclass(frozen=True)
class WindowBatch:
    """Complete sliding windows and their global starting indices."""

    values: np.ndarray
    starts: np.ndarray
    full_length: int
    window_size: int
    step_size: int


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    """Return the lowercase SHA256 digest of path using bounded memory."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_field_name(name: object) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def _reject_label_like_fields(record: Mapping[str, Any]) -> None:
    for key in record:
        normalized = _normalized_field_name(key)
        if normalized in {"y", "gt"} or any(
            part in normalized for part in _FORBIDDEN_FIELD_PARTS
        ):
            raise ValueError(
                f"scoring series field {key!r} is label-like and is forbidden"
            )


def load_scoring_specs(config: Mapping[str, Any]) -> tuple[SeriesSpec, ...]:
    """Parse the label-free scoring.series block from a loaded config."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    scoring = config.get("scoring")
    if not isinstance(scoring, Mapping):
        raise ValueError("config.scoring must be a mapping")
    records = scoring.get("series")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("config.scoring.series must be a non-empty list")

    data_section = config.get("data", {})
    if data_section is None:
        data_section = {}
    if not isinstance(data_section, Mapping):
        raise ValueError("config.data must be a mapping")
    data_root = Path(str(data_section.get("root", "."))).expanduser()
    allowed_fields = {
        "series_id",
        "dataset",
        "group",
        "csv_path",
        "relative_path",
        "timestamp_column",
        "value_column",
        "expected_sha256",
    }
    required_fields = {"series_id", "group", "expected_sha256"}

    specs: list[SeriesSpec] = []
    seen_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"scoring.series[{index}] must be a mapping")
        _reject_label_like_fields(raw_record)
        unexpected = set(raw_record) - allowed_fields
        if unexpected:
            raise ValueError(
                f"scoring.series[{index}] has unsupported fields: "
                + ", ".join(sorted(str(item) for item in unexpected))
            )
        missing = required_fields - set(raw_record)
        if missing:
            raise ValueError(
                f"scoring.series[{index}] is missing: "
                + ", ".join(sorted(missing))
            )

        series_id = str(raw_record["series_id"]).strip()
        if not series_id:
            raise ValueError("series_id must be non-empty")
        if series_id in seen_ids:
            raise ValueError(f"duplicate series_id: {series_id}")
        seen_ids.add(series_id)

        group = str(raw_record["group"]).strip()
        if group not in _ALLOWED_GROUPS:
            raise ValueError(f"unsupported group {group!r}")

        if ("csv_path" in raw_record) == ("relative_path" in raw_record):
            raise ValueError("each scoring record needs exactly one of csv_path or relative_path")
        path_value = raw_record.get("csv_path", raw_record.get("relative_path"))
        csv_path = Path(str(path_value)).expanduser()
        if not csv_path.is_absolute():
            csv_path = data_root / csv_path
        timestamp_column = str(
            raw_record.get("timestamp_column", data_section.get("timestamp_column", "timestamp"))
        ).strip()
        value_column = str(
            raw_record.get("value_column", data_section.get("value_column", "value"))
        ).strip()
        if not timestamp_column or not value_column:
            raise ValueError("timestamp_column and value_column must be non-empty")
        if timestamp_column == value_column:
            raise ValueError("timestamp_column and value_column must differ")

        expected_hash = str(raw_record["expected_sha256"]).strip().lower()
        if len(expected_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_hash
        ):
            raise ValueError("expected_sha256 must be a 64-character hex digest")

        specs.append(
            SeriesSpec(
                series_id=series_id,
                group=group,
                csv_path=csv_path.resolve(strict=False),
                timestamp_column=timestamp_column,
                value_column=value_column,
                expected_sha256=expected_hash,
            )
        )
    return tuple(specs)


def load_signal(spec: SeriesSpec) -> SeriesData:
    """Load and validate only timestamp/value columns for scoring."""

    actual_hash = sha256_file(spec.csv_path)
    if actual_hash != spec.expected_sha256.lower():
        raise ValueError(
            f"data SHA256 mismatch for {spec.series_id}: "
            f"expected {spec.expected_sha256}, got {actual_hash}"
        )
    frame = pd.read_csv(
        spec.csv_path,
        usecols=[spec.timestamp_column, spec.value_column],
    )
    if frame.empty:
        raise ValueError(f"series {spec.series_id} is empty")
    frame = frame[[spec.timestamp_column, spec.value_column]]
    timestamps = pd.to_numeric(frame[spec.timestamp_column], errors="raise").to_numpy()
    values = pd.to_numeric(frame[spec.value_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    if timestamps.ndim != 1 or values.ndim != 1 or timestamps.size != values.size:
        raise ValueError("timestamp/value columns must be aligned vectors")
    if not np.all(np.isfinite(timestamps.astype(np.float64, copy=False))):
        raise ValueError("timestamps must be finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")

    order = np.argsort(timestamps, kind="stable")
    timestamps = np.ascontiguousarray(timestamps[order])
    values = np.ascontiguousarray(values[order], dtype=np.float64)
    if timestamps.size > 1 and np.any(timestamps[1:] == timestamps[:-1]):
        raise ValueError("timestamps must be unique")
    timestamps.setflags(write=False)
    values.setflags(write=False)
    return SeriesData(
        series_id=spec.series_id,
        group=spec.group,
        timestamps=timestamps,
        values=values,
        data_sha256=actual_hash,
    )


def released_preprocess(values: np.ndarray) -> np.ndarray:
    """Reproduce vendor detrending and full-series min-max normalization."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    detrended = np.asarray(detrend(array), dtype=np.float64)
    minimum = float(np.min(detrended))
    span = float(np.max(detrended)) - minimum
    normalized = (
        (detrended - minimum) / span if span > 0.0 else np.zeros_like(detrended)
    )
    return np.ascontiguousarray(normalized, dtype=np.float64)


def make_windows(
    values: np.ndarray,
    window_size: int,
    step_size: int,
) -> WindowBatch:
    """Extract complete, fixed-step windows from a one-dimensional signal."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    length, window, step = int(array.size), int(window_size), int(step_size)
    if window <= 1 or step <= 0:
        raise ValueError("window_size must exceed one and step_size must be positive")
    if length < window:
        raise ValueError(
            f"series length {length} is shorter than window_size {window}"
        )
    starts = np.arange(0, length - window + 1, step, dtype=np.int64)
    sliding = np.lib.stride_tricks.sliding_window_view(array, window)
    batch = WindowBatch(
        values=np.ascontiguousarray(sliding[starts], dtype=np.float32),
        starts=np.ascontiguousarray(starts, dtype=np.int64),
        full_length=length,
        window_size=window,
        step_size=step,
    )
    validate_window_batch(batch)
    return batch


def validate_window_batch(
    batch: WindowBatch,
    *,
    frozen_window_size: int | None = None,
    frozen_step_size: int | None = None,
) -> None:
    """Validate shape, chronology, and optional frozen K0 geometry."""

    if batch.values.ndim != 2:
        raise ValueError("window values must have shape [N, L]")
    if batch.starts.ndim != 1 or batch.starts.size != batch.values.shape[0]:
        raise ValueError("starts must have shape [N]")
    if batch.values.shape[1] != batch.window_size:
        raise ValueError("window width disagrees with window_size")
    expected_n = (batch.full_length - batch.window_size) // batch.step_size + 1
    if batch.values.shape[0] != expected_n:
        raise ValueError("window count disagrees with complete-window formula")
    expected_starts = np.arange(
        0,
        batch.full_length - batch.window_size + 1,
        batch.step_size,
        dtype=np.int64,
    )
    if not np.array_equal(batch.starts, expected_starts):
        raise ValueError("window starts are not chronological fixed-step indices")
    if batch.values.dtype != np.float32 or batch.starts.dtype != np.int64:
        raise ValueError("windows must be float32 and starts must be int64")
    if not np.all(np.isfinite(batch.values)):
        raise ValueError("window values must be finite")
    if frozen_window_size is not None and batch.window_size != frozen_window_size:
        raise ValueError("window_size differs from frozen protocol")
    if frozen_step_size is not None and batch.step_size != frozen_step_size:
        raise ValueError("step_size differs from frozen protocol")
