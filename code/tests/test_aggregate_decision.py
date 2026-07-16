from pathlib import Path

import pytest

from paano_k0.aggregate import (
    ContrastRow,
    GateResult,
    decide_k0,
    load_metric_rows,
    performance_gate,
)
from paano_k0.config import expand_primary_jobs, load_protocol, load_series_manifest


ROOT = Path(__file__).resolve().parents[2]


def _config():
    return load_protocol(ROOT / "configs" / "k0_protocol.yaml")


def _contrasts(delta_vus: float = 0.01, delta_pr: float = 0.001):
    return tuple(
        ContrastRow(
            contrast="fixture",
            series_id=f"series-{index}",
            family=f"family-{index}",
            seed=2027,
            treatment="A",
            control="B",
            delta_vus_pr=delta_vus,
            delta_auprc=delta_pr,
            delta_vus_roc=0.0,
        )
        for index in range(6)
    )


def test_gate_boundaries_and_precedence() -> None:
    config = _config()
    passing = performance_gate(_contrasts(), config, name="overlap")
    assert passing.passed
    pr_boundary = performance_gate(_contrasts(delta_pr=0.0), config, name="paper_negative")
    assert not pr_boundary.passed  # macro AUPRC is strictly greater than zero

    low = GateResult("low_activity", True, {"count": 6})
    early = GateResult("early_checkpoint", True, {"count": 6})
    checkpoint = GateResult("checkpoint", True, {})
    paper = GateResult("paper_negative", True, {})
    overlap = GateResult("overlap", True, {})
    decision = decide_k0(config, checkpoint, paper, overlap, low, early)
    assert decision["outcome"] == "GO_METHOD_DESIGN"

    decision = decide_k0(
        config,
        checkpoint,
        paper,
        GateResult("overlap", False, {}),
        low,
        early,
    )
    assert decision["outcome"] == "SIMPLE_PAPER_PARITY_FIX"


def test_incomplete_family_fails(tmp_path: Path) -> None:
    config = _config()
    series = load_series_manifest(ROOT / "docs" / "K0_DATA_MANIFEST.csv")
    jobs = expand_primary_jobs(config, series, ROOT / "vendor-placeholder", tmp_path)
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_metric_rows(tmp_path, jobs)
