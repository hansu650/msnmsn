from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from measure_vit4ts_v3.cache_registry import (
    build_cache_only_plan,
    freeze_cache_only_registry,
    load_compute_plan,
)
from measure_vit4ts_v3.registry import load_arm_registry


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "vittrace_ablation_full_v3.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_mandatory_cache_only_registry_is_complete_and_deduplicated() -> None:
    plan = build_cache_only_plan(_config())
    assert len(plan.arms) == 36
    assert len(plan.canonical_arms) == 26
    assert len(set(plan.logical_arm_ids)) == 36
    identifiers = set(plan.logical_arm_ids)
    assert {
        "MATCH_LEGACY_POSITION",
        "MATCH_LEGACY_ROW",
        "MATCH_LEGACY_COLUMN",
        "MATCH_LEGACY_GLOBAL",
        "MATCH_FINAL_POSITION",
        "MATCH_FINAL_ROW",
        "MATCH_FINAL_COLUMN",
        "MATCH_FINAL_GLOBAL",
    } <= identifiers
    assert {f"SCALE_{value}" for value in ("P", "M", "L", "PM", "PL", "ML", "PML")} <= identifiers
    assert {"SCALE_LEGACY_P", "SCALE_LEGACY_PML"} <= identifiers
    assert {"MEMORY_MEDIAN", "MEMORY_ALLPAIRS"} <= identifiers
    assert {f"REDUCER_Q{value}" for value in (10, 25, 50)} <= identifiers
    assert {f"REDUCER_TOP{value}" for value in (10, 25, 50)} <= identifiers
    assert {
        "IHP0_NCTP0",
        "IHP1_NCTP0",
        "IHP0_NCTP1",
        "IHP1_NCTP1",
        "TEMP_NCTP_LINEAR",
        "TEMP_NCTP_NEAREST",
        "TEMP_TRACE_SOFT",
        "TEMP_TRACE_HARD",
        "TEMP_LEGACY",
    } <= identifiers

    by_id = plan.by_id()
    assert by_id["MATCH_FINAL_GLOBAL"].canonical_arm == "FINAL_DEFAULT"
    assert by_id["SCALE_PML"].canonical_arm == "FINAL_DEFAULT"
    assert by_id["MEMORY_MEDIAN"].canonical_arm == "FINAL_DEFAULT"
    assert by_id["IHP1_NCTP1"].canonical_arm == "FINAL_DEFAULT"
    assert by_id["TEMP_NCTP_LINEAR"].canonical_arm == "FINAL_DEFAULT"
    assert by_id["MATCH_LEGACY_GLOBAL"].canonical_arm == "LEGACY_DEFAULT"
    assert by_id["IHP0_NCTP0"].canonical_arm == "LEGACY_DEFAULT"
    assert by_id["SCALE_LEGACY_PML"].canonical_arm == "LEGACY_DEFAULT"
    assert by_id["REDUCER_TOP25"].canonical_arm == "LEGACY_DEFAULT"
    assert by_id["IHP1_NCTP0"].canonical_arm == "IHP1_NCTP0"
    assert by_id["TEMP_LEGACY"].canonical_arm == "IHP1_NCTP0"

    # B4 reducer sensitivity changes only the legacy row reducer.  It must not
    # silently include the literal-incidence/active-valid IHP changes.
    for arm_id in {f"REDUCER_Q{value}" for value in (10, 25, 50)} | {
        f"REDUCER_TOP{value}" for value in (10, 25, 50)
    }:
        arm = by_id[arm_id].logical
        assert arm.incidence == "released"
        assert arm.fusion == "legacy_intersection"
        assert arm.temporal == "legacy"


def test_freeze_round_trip_binds_config_and_manifest(tmp_path: Path) -> None:
    registry_path, plan_path = freeze_cache_only_registry(CONFIG, tmp_path)
    registry = load_arm_registry(registry_path)
    plan, payload = load_compute_plan(plan_path)
    assert registry.arm_ids == plan.logical_arm_ids
    assert payload["logical_arm_count"] == 36
    assert payload["unique_computation_count"] == 26
    assert payload["encoder_calls"] == 0


def test_registry_rejects_nonmandatory_grid() -> None:
    config = _config()
    config["grid"]["matching_scopes"] = ["global"]
    with pytest.raises(ValueError, match="matching-scope"):
        build_cache_only_plan(config)
