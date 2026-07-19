from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from measure_vit4ts_v3.supplement_runtime import (
    FACTORIAL_ARMS,
    FAILURE_MANIFEST_COLUMNS,
    MEASUREMENT_MODE,
    PostCacheOperation,
    SCOPE,
    ThreadMetadata,
    audit_confirmation_cohort,
    benchmark_post_cache_factorial,
    sha256_file,
    write_supplement_outputs,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        return self.value

    def advance(self, nanoseconds: int) -> None:
        self.value += int(nanoseconds)


def _operations(clock: FakeClock, order: list[str], *, fail: str = ""):
    output = []
    for index, arm in enumerate(reversed(FACTORIAL_ARMS)):
        load_ns = (index + 1) * 1_000_000
        compute_ns = (index + 1) * 2_000_000

        def loader(
            arm_id: str = arm,
            duration: int = load_ns,
        ) -> dict[str, str]:
            order.append(f"load:{arm_id}")
            clock.advance(duration)
            if arm_id == fail:
                raise RuntimeError("registered synthetic load failure")
            return {"arm": arm_id}

        def compute(
            payload: dict[str, str],
            arm_id: str = arm,
            duration: int = compute_ns,
        ) -> str:
            assert payload["arm"] == arm_id
            order.append(f"compute:{arm_id}")
            clock.advance(duration)
            return arm_id

        output.append(PostCacheOperation(arm, loader, compute, windows_per_series=7))
    return output


def _threads() -> ThreadMetadata:
    return ThreadMetadata(1, {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})


def test_four_arm_runtime_is_interleaved_post_cache_only_and_summarized() -> None:
    clock = FakeClock()
    call_order: list[str] = []
    result = benchmark_post_cache_factorial(
        _operations(clock, call_order),
        series_id="MSL__C-1",
        thread_metadata=_threads(),
        warmups=2,
        repeats=5,
        clock_ns=clock,
    )
    assert result.complete
    assert len(result.samples) == 4 * (2 + 5)
    assert set(result.samples["arm"]) == set(FACTORIAL_ARMS)
    assert set(result.samples["encoder_calls"]) == {0}
    assert set(result.samples["scope"]) == {SCOPE}
    assert set(result.samples["measurement_mode"]) == {MEASUREMENT_MODE}
    round_zero = result.samples.loc[result.samples["round_index"] == 0]
    round_one = result.samples.loc[result.samples["round_index"] == 1]
    assert tuple(round_zero.sort_values("order_position")["arm"]) == FACTORIAL_ARMS
    assert tuple(round_one.sort_values("order_position")["arm"]) == (
        *FACTORIAL_ARMS[1:],
        FACTORIAL_ARMS[0],
    )
    assert len(result.summary) == 4 * 3 * 2
    assert set(result.summary["phase"]) == {"load", "compute", "total"}
    assert set(result.summary["unit"]) == {"series", "window"}
    assert set(result.summary["n_measured"]) == {5}
    assert set(result.summary["n_warmup"]) == {2}
    assert set(result.summary["encoder_calls"]) == {0}
    assert (result.summary["p95_ms"] >= result.summary["median_ms"]).all()
    assert result.environment["encoder_calls"] == 0
    assert "post-cache projection/scoring only" in result.environment["wording"]
    assert "Encoder calls: 0" in result.protocol_text


def test_runtime_protocol_rejects_partial_factorial_and_too_few_repeats() -> None:
    clock = FakeClock()
    operations = _operations(clock, [])
    with pytest.raises(ValueError, match="exactly the four factorial arms"):
        benchmark_post_cache_factorial(
            operations[:-1],
            series_id="x",
            thread_metadata=_threads(),
            clock_ns=clock,
        )
    with pytest.raises(ValueError, match="at least 5"):
        benchmark_post_cache_factorial(
            operations,
            series_id="x",
            thread_metadata=_threads(),
            repeats=4,
            clock_ns=clock,
        )


def test_failure_manifest_is_schema_stable_and_blocks_complete_marker(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    result = benchmark_post_cache_factorial(
        _operations(clock, [], fail="IHP0_NCTP1"),
        series_id="x",
        thread_metadata=_threads(),
        repeats=5,
        clock_ns=clock,
    )
    assert not result.complete
    assert tuple(result.failures.columns) == FAILURE_MANIFEST_COLUMNS
    assert len(result.failures) == 1
    failure = result.failures.iloc[0]
    assert failure["arm"] == "IHP0_NCTP1"
    assert failure["phase"] == "load"
    assert failure["encoder_calls"] == 0
    paths = write_supplement_outputs(tmp_path / "supplement", result)
    marker = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert paths[-1].name == "_POST_CACHE_RUNTIME_BLOCKED.json"
    assert marker["status"] == "BLOCKED"
    assert marker["failure_count"] == 1
    assert pd.read_csv(paths[-2]).columns.tolist() == list(FAILURE_MANIFEST_COLUMNS)


def _timestamp(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def test_confirmation_audit_is_read_only_and_fails_closed(tmp_path: Path) -> None:
    missing = audit_confirmation_cohort(None, None)
    assert missing["status"] == "CONFIRMATION_BLOCKED"
    assert not missing["created_split"]

    manifest = tmp_path / "confirmation_manifest.json"
    marker = tmp_path / "confirmation_marker.json"
    selection = tmp_path / "arm_selection.json"
    manifest.write_text('{"series":["held-out"]}\n', encoding="utf-8")
    os.utime(manifest, (_timestamp("2020-01-01T00:00:00Z"),) * 2)
    marker.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "cohort_id": "PREEXISTING_CONFIRMATION",
                "created_at_utc": "2020-01-02T00:00:00Z",
                "cohort_manifest_path": str(manifest),
                "cohort_manifest_sha256": sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    os.utime(marker, (_timestamp("2020-01-02T00:00:00Z"),) * 2)
    selection.write_text(
        json.dumps(
            {
                "status": "ARM_SELECTION_FROZEN",
                "arm_selection_at_utc": "2020-02-01T00:00:00Z",
                "confirmation_marker_sha256": sha256_file(marker),
                "confirmation_manifest_sha256": sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    os.utime(selection, (_timestamp("2020-02-01T00:00:00Z"),) * 2)
    before = sorted(path.name for path in tmp_path.iterdir())
    ready = audit_confirmation_cohort(marker, selection)
    after = sorted(path.name for path in tmp_path.iterdir())
    assert ready["status"] == "CONFIRMATION_READY"
    assert ready["audit_is_read_only"]
    assert before == after

    manifest.write_text('{"series":["tampered"]}\n', encoding="utf-8")
    blocked = audit_confirmation_cohort(marker, selection)
    assert blocked["status"] == "CONFIRMATION_BLOCKED"
    assert "manifest changed" in blocked["reason"]


def test_complete_output_marker_hash_binds_data_only_payloads(tmp_path: Path) -> None:
    clock = FakeClock()
    result = benchmark_post_cache_factorial(
        _operations(clock, []),
        series_id="x",
        thread_metadata=_threads(),
        repeats=5,
        clock_ns=clock,
    )
    paths = write_supplement_outputs(tmp_path / "supplement", result)
    marker = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert paths[-1].name == "_POST_CACHE_RUNTIME_COMPLETE.json"
    assert marker["status"] == "COMPLETE"
    assert marker["encoder_calls"] == 0
    assert marker["arm_count"] == 4
    assert marker["samples_sha256"] == sha256_file(paths[0])
    assert marker["summary_sha256"] == sha256_file(paths[1])
