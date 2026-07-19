from __future__ import annotations

import numpy as np
import pytest
import torch

from measure_vit4ts.full_manifest import FullSeriesRecord
from measure_vit4ts.reducers import stitch_column_vectors
from measure_vit4ts.ihp_cache import universal_token_cost
from measure_vit4ts.ritp import (
    build_full_column_operator,
    stitch_native_240,
)
from measure_vit4ts_v3 import parity


def _record(series_id: str) -> FullSeriesRecord:
    return FullSeriesRecord(
        series_id=series_id,
        dataset="artificialWithAnomaly",
        track="artificialWithAnomaly",
        paper_group="art_daily",
        signal_name=series_id,
        relative_path="unused.csv",
        expected_sha256="0" * 64,
        expected_length=480,
        expected_windows=5,
        duplicate_timestamps=False,
    )


def test_compare_scores_uses_registered_mixed_tolerance() -> None:
    reference = np.array([0.0, 10.0], dtype=np.float64)
    within = np.array([0.5e-12, 10.0 + 0.9e-9], dtype=np.float64)
    outside = np.array([1.1e-12, 10.0 + 1.2e-9], dtype=np.float64)

    passed = parity.compare_scores(within, reference, atol=1e-12, rtol=1e-10)
    failed = parity.compare_scores(outside, reference, atol=1e-12, rtol=1e-10)

    assert passed["passed"] is True
    assert passed["mismatch_count"] == 0
    assert failed["passed"] is False
    assert failed["mismatch_count"] == 2


def test_global_cost_uses_full_frozen_float32_matmul() -> None:
    generator = torch.Generator().manual_seed(2027)
    tokens = torch.randn(65, 4, 7, generator=generator, dtype=torch.float32).numpy()
    actual = parity._global_cost(tokens, torch.device("cpu"), chunk_size=64)
    expected = universal_token_cost(torch.from_numpy(tokens)).to(dtype=torch.float64)
    assert torch.equal(actual, expected)


def test_local_stitchers_are_frozen_parity_equivalents() -> None:
    rng = np.random.default_rng(17)
    raster = rng.normal(size=(5, 224))
    native = rng.normal(size=(5, 240))
    starts = np.arange(5, dtype=np.int64) * 60

    assert np.array_equal(
        parity._legacy_stitch(raster, 503, 240, 60),
        stitch_column_vectors(raster, 503, 240, 60),
    )
    assert np.array_equal(
        parity._native_stitch(native, starts, 503),
        stitch_native_240(native, starts, 503),
    )


def test_record_selection_has_explicit_bulk_guard() -> None:
    records = tuple(_record(f"series_{index}") for index in range(3))
    assert parity.select_records(
        records,
        smoke=True,
        all_series=False,
        series_ids=(),
        approved_bulk=False,
    ) == (records[0],)
    with pytest.raises(PermissionError, match="approved-bulk"):
        parity.select_records(
            records,
            smoke=False,
            all_series=True,
            series_ids=(),
            approved_bulk=False,
        )
    assert parity.select_records(
        records,
        smoke=False,
        all_series=True,
        series_ids=(),
        approved_bulk=True,
    ) == records


def test_dynamic_full_column_matches_frozen_control() -> None:
    dynamic = parity._full_column_operator(240, (14, 14), (224, 224))
    frozen = build_full_column_operator()
    assert dynamic.shape == frozen.shape == (240, 196)
    assert np.allclose(dynamic, frozen, rtol=1e-13, atol=3e-15)
    assert float(np.max(np.abs(dynamic - frozen))) <= 3e-15


def test_gate_cannot_pass_partial_results() -> None:
    rows = []
    for arm in parity.PARITY_ARMS:
        rows.append(
            {
                "series_id": "only_one",
                "arm": arm,
                "status": "PASS",
                "passed": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
            }
        )
    config = {
        "contracts": {"parity_atol": 1e-12, "parity_rtol": 1e-10},
        "frozen_inputs": {
            "coordinate_run_root": ".",
            "vittrace_run_root": ".",
        },
    }
    gate = parity._gate_payload(
        config,
        "CONFIG",
        "MANIFEST",
        "SOURCE",
        __import__("pandas").DataFrame(rows),
        {"only_one": {"status": "PASS"}},
    )
    assert gate["decision"] == "INCOMPLETE"
    assert gate["passed"] is False
    assert gate["completed_series"] == 1
    assert gate["core_sha256"] == parity.core_sha256()
