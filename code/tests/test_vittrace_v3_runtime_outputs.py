from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from measure_vit4ts_v3.runtime_outputs import (
    RUNTIME_SAMPLE_COLUMNS,
    aggregate_runtime_samples,
    cache_runtime_samples,
    encoder_runtime_samples,
    load_runtime_jsons,
    normalize_runtime_samples,
    write_runtime_outputs,
)


def _repeat(index: int, *, warmup: bool, reason: str = "") -> dict[str, object]:
    return {
        "schema_version": 1,
        "sample_kind": "benchmark_repeat",
        "scope": "end_to_end",
        "experiment_id": "B16_LINE_W240",
        "arm": "FULL",
        "series_id": "MSL__C-1",
        "family": "NASA",
        "subgroup": "NASA-MSL",
        "measurement_mode": "encoder_inclusive",
        "stage": "total",
        "backbone": "B16",
        "representation": "line",
        "window": 240,
        "stride": 60,
        "patch_size": 16,
        "batch_size": 64,
        "device": "cuda:0",
        "threads": 1,
        "repeat_index": index,
        "is_warmup": warmup,
        "protocol_warmup": 5,
        "protocol_repeats": 30,
        "elapsed_seconds": 0.1 + 0.001 * index,
        "peak_rss_bytes": 1024 + index,
        "cuda_peak_allocated_bytes": 2048 + index,
        "cuda_peak_reserved_bytes": 4096 + index,
        "encoder_calls": 1,
        "status": "PASS" if not reason else "PARTIAL_RESOURCE_LIMIT",
        "partial_reason": reason,
    }


def test_benchmark_runtime_protocol_and_statistics() -> None:
    rows = [_repeat(i, warmup=i < 5) for i in range(35)]
    normalized = normalize_runtime_samples(rows)
    assert tuple(normalized.columns) == RUNTIME_SAMPLE_COLUMNS
    summary = aggregate_runtime_samples(normalized)
    per_series = summary.loc[summary["aggregation"] == "per_series"].iloc[0]
    assert per_series["n_warmup"] == 5
    assert per_series["n_measured"] == 30
    assert bool(per_series["protocol_complete"])
    measured = np.asarray([0.1 + 0.001 * i for i in range(5, 35)])
    assert per_series["median_seconds"] == pytest.approx(float(np.median(measured)))
    assert per_series["p90_seconds"] == pytest.approx(float(np.quantile(measured, 0.9)))
    assert per_series["encoder_calls_total"] == 30


def test_incomplete_benchmark_requires_explicit_resource_reason() -> None:
    rows = [_repeat(i, warmup=i < 5) for i in range(15)]
    with pytest.raises(ValueError, match="explicit reason"):
        aggregate_runtime_samples(rows)
    rows = [
        {**row, "status": "PARTIAL_RESOURCE_LIMIT", "partial_reason": "H14 runtime budget"}
        for row in rows
    ]
    summary = aggregate_runtime_samples(rows)
    row = summary.loc[summary["aggregation"] == "per_series"].iloc[0]
    assert not bool(row["protocol_complete"])
    assert row["status"] == "PARTIAL_RESOURCE_LIMIT"
    assert row["partial_reason"] == "H14 runtime budget"


def test_cache_and_encoder_records_remain_distinct(tmp_path: Path) -> None:
    cache_payload = {
        "series_id": "MSL__C-1",
        "shared_matching_seconds": 1.0,
        "canonical_arm_seconds": {"REL": 0.2, "FULL": 0.3},
        "series_wall_seconds": 1.6,
        "python_tracemalloc_current_bytes": 100,
        "python_tracemalloc_peak_bytes": 200,
        "process_rss_before_bytes": 1000,
        "process_rss_after_bytes": 2000,
        "cuda_peak_allocated_bytes": 3000,
        "cuda_peak_reserved_bytes": 4000,
        "encoder_calls": 0,
    }
    cached = cache_runtime_samples(cache_payload, family="NASA", subgroup="NASA-MSL")
    assert set(cached["measurement_mode"]) == {"cached"}
    assert set(cached["encoder_calls"]) == {0}
    assert set(cached["stage"]) == {"matching", "postprocess", "total"}

    encoder_payload = {
        "series_id": "MSL__C-1",
        "family": "NASA",
        "subgroup": "NASA-MSL",
        "variant": {
            "key": "LINE_B16_W240_S60",
            "model_key": "B16",
            "representation": "line",
            "window": 240,
            "stride": 60,
            "patch_size": 16,
            "batch_size": 64,
        },
        "renderer_seconds": 0.4,
        "encode_save_seconds": 2.0,
        "wall_seconds": 2.5,
        "encoder_calls": 2,
        "device": "cuda:0",
    }
    encoded = encoder_runtime_samples(encoder_payload)
    assert set(encoded["measurement_mode"]) == {"encoder_inclusive"}
    assert int(encoded.loc[encoded["stage"] == "total", "encoder_calls"].iloc[0]) == 2

    first = tmp_path / "cache.json"
    second = tmp_path / "encoder.json"
    first.write_text(json.dumps(cache_payload), encoding="utf-8")
    second.write_text(json.dumps(encoder_payload), encoding="utf-8")
    loaded = load_runtime_jsons([second, first])
    assert set(loaded["measurement_mode"]) == {"cached", "encoder_inclusive"}
    paths = write_runtime_outputs(tmp_path / "out", loaded)
    assert all(path.is_file() for path in paths)
    summary = pd.read_csv(paths[1])
    assert set(summary["measurement_mode"]) == {"cached", "encoder_inclusive"}


def test_cached_rows_cannot_claim_encoder_calls() -> None:
    row = _repeat(0, warmup=False)
    row.update(
        {
            "sample_kind": "series_observation",
            "measurement_mode": "cached",
            "scope": "cache_only_postprocess",
            "stage": "total",
            "protocol_warmup": 0,
            "protocol_repeats": 1,
            "encoder_calls": 1,
        }
    )
    with pytest.raises(ValueError, match="encoder_calls=0"):
        normalize_runtime_samples([row])
