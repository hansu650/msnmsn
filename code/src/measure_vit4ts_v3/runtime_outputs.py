"""CPU-only runtime schemas and aggregation for ViTTrace v3.

The module keeps cache-only post-processing distinct from encoder-inclusive
execution.  It accepts either explicit repeated measurements or the compact
``runtime.json`` / encoder-stage records emitted by the v3 runners.  No model,
score, cache, or label is opened here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUNTIME_SCHEMA_VERSION = 1
MEASUREMENT_MODES = ("cached", "encoder_inclusive")
SAMPLE_KINDS = ("benchmark_repeat", "series_observation")
RUNTIME_SAMPLE_COLUMNS = (
    "schema_version",
    "sample_kind",
    "scope",
    "experiment_id",
    "arm",
    "series_id",
    "family",
    "subgroup",
    "measurement_mode",
    "stage",
    "backbone",
    "representation",
    "window",
    "stride",
    "patch_size",
    "batch_size",
    "device",
    "threads",
    "repeat_index",
    "is_warmup",
    "protocol_warmup",
    "protocol_repeats",
    "elapsed_seconds",
    "peak_rss_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "encoder_calls",
    "status",
    "partial_reason",
)
RUNTIME_SUMMARY_COLUMNS = (
    "aggregation",
    "scope",
    "experiment_id",
    "arm",
    "series_id",
    "family",
    "subgroup",
    "measurement_mode",
    "stage",
    "backbone",
    "representation",
    "window",
    "stride",
    "patch_size",
    "batch_size",
    "device",
    "threads",
    "n_series",
    "n_samples",
    "n_warmup",
    "n_measured",
    "protocol_warmup",
    "protocol_repeats",
    "protocol_complete",
    "status",
    "partial_reason",
    "median_seconds",
    "p90_seconds",
    "mean_seconds",
    "std_seconds",
    "peak_rss_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "encoder_calls_total",
    "encoder_calls_max",
)

_TEXT_COLUMNS = {
    "sample_kind",
    "scope",
    "experiment_id",
    "arm",
    "series_id",
    "family",
    "subgroup",
    "measurement_mode",
    "stage",
    "backbone",
    "representation",
    "device",
    "status",
    "partial_reason",
}
_INT_COLUMNS = {
    "schema_version",
    "window",
    "stride",
    "patch_size",
    "batch_size",
    "threads",
    "repeat_index",
    "protocol_warmup",
    "protocol_repeats",
    "encoder_calls",
}
_FLOAT_COLUMNS = {
    "elapsed_seconds",
    "peak_rss_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
}


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == ""


def _finite_nonnegative(values: pd.Series, *, allow_missing: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not allow_missing and numeric.isna().any():
        raise ValueError(f"{values.name} must be present")
    finite = numeric.dropna().to_numpy(dtype=np.float64)
    if not np.isfinite(finite).all() or bool((finite < 0.0).any()):
        raise ValueError(f"{values.name} must be finite and non-negative")
    return numeric


def normalize_runtime_samples(samples: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return the strict canonical runtime-sample table.

    ``benchmark_repeat`` rows obey the registered warmup/repeat protocol.
    ``series_observation`` rows are measurements naturally emitted once per
    series and are retained without pretending that they are the requested
    5+30 microbenchmark repetitions.
    """

    frame = samples.copy() if isinstance(samples, pd.DataFrame) else pd.DataFrame(samples)
    missing = set(RUNTIME_SAMPLE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"runtime samples are missing columns: {sorted(missing)}")
    frame = frame.loc[:, RUNTIME_SAMPLE_COLUMNS].copy()
    if frame.empty:
        raise ValueError("runtime samples must be nonempty")
    for column in _TEXT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str)
    for column in _INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    for column in _FLOAT_COLUMNS:
        frame[column] = _finite_nonnegative(frame[column], allow_missing=True)
    frame["is_warmup"] = frame["is_warmup"].astype(bool)

    if set(frame["sample_kind"]) - set(SAMPLE_KINDS):
        raise ValueError("runtime sample_kind is not registered")
    if set(frame["measurement_mode"]) - set(MEASUREMENT_MODES):
        raise ValueError("runtime measurement_mode is not cached/encoder_inclusive")
    if (frame["schema_version"] != RUNTIME_SCHEMA_VERSION).any():
        raise ValueError("runtime schema version changed")
    required_text = ("scope", "experiment_id", "stage", "device", "status")
    if any((frame[column].str.len() == 0).any() for column in required_text):
        raise ValueError("runtime identity/status fields must be nonempty")
    if set(frame["status"]) - {"PASS", "PARTIAL_RESOURCE_LIMIT", "BLOCKED"}:
        raise ValueError("runtime status is not PASS/PARTIAL_RESOURCE_LIMIT/BLOCKED")

    integers = frame.loc[:, list(_INT_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    nonnegative = (integers.drop(columns="repeat_index") < 0).any(axis=None)
    if bool(nonnegative) or bool((integers["repeat_index"] < -1).any()):
        raise ValueError("runtime integer fields are invalid")
    if frame["sample_kind"].eq("benchmark_repeat").any():
        repeated = frame.loc[frame["sample_kind"] == "benchmark_repeat"]
        if (
            repeated["protocol_warmup"].isna().any()
            or repeated["protocol_repeats"].isna().any()
            or repeated["repeat_index"].isna().any()
        ):
            raise ValueError("benchmark repeats require protocol counts and repeat indices")
    observed = frame["sample_kind"] == "series_observation"
    if bool((frame.loc[observed, "is_warmup"]).any()):
        raise ValueError("series observations cannot be marked as warmup")

    blocked = frame["status"] == "BLOCKED"
    measured = ~frame["is_warmup"] & ~blocked
    if frame.loc[measured, "elapsed_seconds"].isna().any():
        raise ValueError("measured PASS/PARTIAL rows require elapsed_seconds")
    if frame.loc[blocked, "partial_reason"].str.len().eq(0).any():
        raise ValueError("BLOCKED runtime rows require an explicit reason")
    partial = frame["status"] == "PARTIAL_RESOURCE_LIMIT"
    if frame.loc[partial, "partial_reason"].str.len().eq(0).any():
        raise ValueError("partial runtime rows require an explicit resource reason")

    cached = frame["measurement_mode"] == "cached"
    if (frame.loc[cached, "encoder_calls"].fillna(-1) != 0).any():
        raise ValueError("cached timing must record encoder_calls=0")
    inclusive_total = (
        frame["measurement_mode"].eq("encoder_inclusive")
        & frame["stage"].eq("total")
        & ~blocked
    )
    if (frame.loc[inclusive_total, "encoder_calls"].fillna(0) <= 0).any():
        raise ValueError("encoder-inclusive total timing must contain an encoder call")
    return frame.sort_values(
        ["scope", "experiment_id", "series_id", "stage", "is_warmup", "repeat_index"]
    ).reset_index(drop=True)


_CONFIG_GROUP = (
    "scope",
    "experiment_id",
    "arm",
    "measurement_mode",
    "stage",
    "backbone",
    "representation",
    "window",
    "stride",
    "patch_size",
    "batch_size",
    "device",
    "threads",
)


def _summary_rows(frame: pd.DataFrame, *, by_series: bool) -> list[dict[str, Any]]:
    group_columns = list(_CONFIG_GROUP)
    if by_series:
        group_columns.extend(("series_id", "family", "subgroup"))
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        identity = dict(zip(group_columns, keys, strict=True))
        blocked = group["status"].eq("BLOCKED")
        warmup = group["is_warmup"] & ~blocked
        measured = ~group["is_warmup"] & ~blocked
        values = group.loc[measured, "elapsed_seconds"].dropna().to_numpy(dtype=np.float64)
        warmup_required = int(group["protocol_warmup"].dropna().max()) if group["protocol_warmup"].notna().any() else 0
        repeats_required = int(group["protocol_repeats"].dropna().max()) if group["protocol_repeats"].notna().any() else 1
        sample_kinds = set(group["sample_kind"])
        if len(sample_kinds) != 1:
            raise ValueError("one runtime summary group mixes benchmark and series samples")
        is_benchmark = next(iter(sample_kinds)) == "benchmark_repeat"
        complete = (
            int(warmup.sum()) >= warmup_required
            and int(measured.sum()) >= repeats_required
            and not bool(blocked.any())
        ) if is_benchmark else (values.size > 0 and not bool(blocked.any()))
        reasons = sorted({value for value in group["partial_reason"] if value})
        if is_benchmark and not complete and not reasons:
            raise ValueError("incomplete benchmark timing requires an explicit reason")
        status = "PASS" if complete else ("BLOCKED" if bool(blocked.all()) else "PARTIAL_RESOURCE_LIMIT")
        row: dict[str, Any] = {
            "aggregation": "per_series" if by_series else "config",
            **identity,
            "series_id": identity.get("series_id", "ALL"),
            "family": identity.get("family", "ALL"),
            "subgroup": identity.get("subgroup", "ALL"),
            "n_series": int(group["series_id"].replace("", np.nan).nunique()),
            "n_samples": int(len(group)),
            "n_warmup": int(warmup.sum()),
            "n_measured": int(measured.sum()),
            "protocol_warmup": warmup_required,
            "protocol_repeats": repeats_required,
            "protocol_complete": bool(complete),
            "status": status,
            "partial_reason": " | ".join(reasons),
            "median_seconds": float(np.median(values)) if values.size else np.nan,
            "p90_seconds": float(np.quantile(values, 0.9)) if values.size else np.nan,
            "mean_seconds": float(np.mean(values)) if values.size else np.nan,
            "std_seconds": float(np.std(values, ddof=0)) if values.size else np.nan,
            "peak_rss_bytes": float(group["peak_rss_bytes"].max()) if group["peak_rss_bytes"].notna().any() else np.nan,
            "cuda_peak_allocated_bytes": float(group["cuda_peak_allocated_bytes"].max()) if group["cuda_peak_allocated_bytes"].notna().any() else np.nan,
            "cuda_peak_reserved_bytes": float(group["cuda_peak_reserved_bytes"].max()) if group["cuda_peak_reserved_bytes"].notna().any() else np.nan,
            "encoder_calls_total": int(group.loc[measured, "encoder_calls"].fillna(0).sum()),
            "encoder_calls_max": int(group.loc[measured, "encoder_calls"].fillna(0).max()) if measured.any() else 0,
        }
        rows.append(row)
    return rows


def aggregate_runtime_samples(samples: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Aggregate exact runtime samples into per-series and config views."""

    frame = normalize_runtime_samples(samples)
    rows = _summary_rows(frame, by_series=True) + _summary_rows(frame, by_series=False)
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("runtime aggregation produced no rows")
    return summary.loc[:, RUNTIME_SUMMARY_COLUMNS].sort_values(
        ["aggregation", "scope", "experiment_id", "series_id", "stage"]
    ).reset_index(drop=True)


def _base_sample(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "sample_kind": "series_observation",
        "scope": "",
        "experiment_id": "",
        "arm": "",
        "series_id": "",
        "family": "",
        "subgroup": "",
        "measurement_mode": "cached",
        "stage": "",
        "backbone": "",
        "representation": "",
        "window": pd.NA,
        "stride": pd.NA,
        "patch_size": pd.NA,
        "batch_size": pd.NA,
        "device": "unknown",
        "threads": pd.NA,
        "repeat_index": 0,
        "is_warmup": False,
        "protocol_warmup": 0,
        "protocol_repeats": 1,
        "elapsed_seconds": np.nan,
        "peak_rss_bytes": np.nan,
        "cuda_peak_allocated_bytes": np.nan,
        "cuda_peak_reserved_bytes": np.nan,
        "encoder_calls": 0,
        "status": "PASS",
        "partial_reason": "",
    }
    row.update(overrides)
    return row


def cache_runtime_samples(
    payload: Mapping[str, Any],
    *,
    family: str = "",
    subgroup: str = "",
    experiment_id: str = "B16_W240_S60_CACHE_ONLY",
    device: str = "cuda:0",
) -> pd.DataFrame:
    """Flatten one cache-runner ``runtime.json`` into canonical samples."""

    required = {
        "series_id",
        "shared_matching_seconds",
        "canonical_arm_seconds",
        "series_wall_seconds",
        "process_rss_before_bytes",
        "process_rss_after_bytes",
        "encoder_calls",
    }
    if required - set(payload):
        raise ValueError("cache runtime payload is incomplete")
    if int(payload["encoder_calls"]) != 0:
        raise ValueError("cache runtime payload unexpectedly encoded")
    series_id = str(payload["series_id"])
    rss = max(int(payload["process_rss_before_bytes"]), int(payload["process_rss_after_bytes"]))
    common = {
        "scope": "cache_only_postprocess",
        "experiment_id": experiment_id,
        "series_id": series_id,
        "family": str(family),
        "subgroup": str(subgroup),
        "measurement_mode": "cached",
        "backbone": "B16",
        "representation": "line",
        "window": 240,
        "stride": 60,
        "patch_size": 16,
        "device": device,
        "peak_rss_bytes": rss,
        "cuda_peak_allocated_bytes": payload.get("cuda_peak_allocated_bytes", np.nan),
        "cuda_peak_reserved_bytes": payload.get("cuda_peak_reserved_bytes", np.nan),
        "encoder_calls": 0,
    }
    rows = [
        _base_sample(
            **common,
            arm="SHARED",
            stage="matching",
            elapsed_seconds=float(payload["shared_matching_seconds"]),
        )
    ]
    timings = payload["canonical_arm_seconds"]
    if not isinstance(timings, Mapping) or not timings:
        raise ValueError("cache runtime canonical_arm_seconds must be a mapping")
    rows.extend(
        _base_sample(
            **common,
            arm=str(arm),
            stage="postprocess",
            elapsed_seconds=float(seconds),
        )
        for arm, seconds in sorted(timings.items())
    )
    rows.append(
        _base_sample(
            **common,
            arm="ALL",
            stage="total",
            elapsed_seconds=float(payload["series_wall_seconds"]),
        )
    )
    return normalize_runtime_samples(rows)


def encoder_runtime_samples(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten one encoder-stage result into renderer/encode/total samples."""

    required = {
        "series_id",
        "family",
        "subgroup",
        "variant",
        "renderer_seconds",
        "encode_save_seconds",
        "wall_seconds",
        "encoder_calls",
        "device",
    }
    if required - set(payload) or not isinstance(payload["variant"], Mapping):
        raise ValueError("encoder runtime payload is incomplete")
    variant = payload["variant"]
    common = {
        "scope": "encoder_variant",
        "experiment_id": str(variant.get("key", "ENCODER_VARIANT")),
        "arm": "ENCODER",
        "series_id": str(payload["series_id"]),
        "family": str(payload["family"]),
        "subgroup": str(payload["subgroup"]),
        "measurement_mode": "encoder_inclusive",
        "backbone": str(variant.get("model_key", variant.get("model_name", ""))),
        "representation": str(variant.get("representation", "")),
        "window": variant.get("window", pd.NA),
        "stride": variant.get("stride", pd.NA),
        "patch_size": variant.get("patch_size", pd.NA),
        "batch_size": variant.get("batch_size", pd.NA),
        "device": str(payload["device"]),
    }
    calls = int(payload["encoder_calls"])
    rows = [
        _base_sample(**common, stage="render", elapsed_seconds=float(payload["renderer_seconds"]), encoder_calls=0),
        _base_sample(**common, stage="encode_save", elapsed_seconds=float(payload["encode_save_seconds"]), encoder_calls=calls),
        _base_sample(**common, stage="total", elapsed_seconds=float(payload["wall_seconds"]), encoder_calls=calls),
    ]
    return normalize_runtime_samples(rows)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_runtime_outputs(
    output_root: Path,
    samples: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Atomically write ``runtime_samples.csv`` and aggregated ``runtime.csv``."""

    root = Path(output_root)
    normalized = normalize_runtime_samples(samples)
    summary = aggregate_runtime_samples(normalized)
    sample_path = root / "runtime_samples.csv"
    summary_path = root / "runtime.csv"
    _atomic_csv(sample_path, normalized)
    _atomic_csv(summary_path, summary)
    return sample_path, summary_path


def load_runtime_jsons(paths: Iterable[Path]) -> pd.DataFrame:
    """Auto-detect and flatten cache-runner or encoder-stage JSON records."""

    frames: list[pd.DataFrame] = []
    for path in sorted(map(Path, paths)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "canonical_arm_seconds" in payload:
            frames.append(cache_runtime_samples(payload))
        elif "renderer_seconds" in payload and "encode_save_seconds" in payload:
            frames.append(encoder_runtime_samples(payload))
        else:
            raise ValueError(f"unsupported runtime JSON schema: {path}")
    if not frames:
        raise ValueError("no runtime JSON records were provided")
    return normalize_runtime_samples(pd.concat(frames, ignore_index=True))


__all__ = [
    "MEASUREMENT_MODES",
    "RUNTIME_SAMPLE_COLUMNS",
    "RUNTIME_SCHEMA_VERSION",
    "RUNTIME_SUMMARY_COLUMNS",
    "aggregate_runtime_samples",
    "cache_runtime_samples",
    "encoder_runtime_samples",
    "load_runtime_jsons",
    "normalize_runtime_samples",
    "write_runtime_outputs",
]
