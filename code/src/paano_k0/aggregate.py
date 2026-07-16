"""Pre-registered K0 contrasts, gates, and terminal decision."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_json
from .config import ProtocolConfig, load_protocol, load_series_manifest
from .schemas import (
    CheckpointKind,
    MetricRow,
    RunJob,
    Trajectory,
    make_run_id,
    scored_checkpoints,
)


CONTRASTS: Mapping[str, tuple[str, str]] = {
    "checkpoint": ("OFFICIAL_LAST", "OFFICIAL_BEST"),
    "paper_negative": ("PAPERNEG_LAST", "OFFICIAL_LAST"),
    "overlap": ("PAPERNEG_NONOVERLAP_LAST", "PAPERNEG_LAST"),
}


@dataclass(frozen=True, slots=True)
class ContrastRow:
    contrast: str
    series_id: str
    family: str
    seed: int
    treatment: str
    control: str
    delta_vus_pr: float
    delta_auprc: float
    delta_vus_roc: float


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    details: Mapping[str, object]


def _read_metric(path: Path) -> MetricRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metric" in payload and isinstance(payload["metric"], dict):
        payload = payload["metric"]
    return MetricRow.from_dict(payload)


def load_metric_rows(
    metrics_root: Path,
    expected_jobs: Sequence[RunJob],
) -> tuple[MetricRow, ...]:
    """Load exactly the expected 42 primary scored arms; never average partial runs."""

    expected: set[str] = set()
    for job in expected_jobs:
        for checkpoint in scored_checkpoints(job.trajectory):
            expected.add(make_run_id(job.series.series_id, job.seed, job.trajectory, checkpoint))
    paths = tuple(sorted(Path(metrics_root).rglob("metrics.json")))
    rows = tuple(_read_metric(path) for path in paths)
    by_id = {row.run_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate metric run_id detected")
    actual = set(by_id)
    if actual != expected:
        raise ValueError(
            f"metric coverage mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    return tuple(by_id[run_id] for run_id in sorted(expected))


def family_macro(rows: Sequence[MetricRow], arm: str, metric: str) -> float:
    selected = [row for row in rows if row.arm == arm]
    if not selected:
        raise ValueError(f"no metric rows for arm {arm}")
    families: dict[str, list[float]] = {}
    for row in selected:
        families.setdefault(row.family, []).append(float(getattr(row, metric)))
    return float(np.mean([np.mean(values) for values in families.values()]))


def contrast_rows(
    rows: Sequence[MetricRow],
    treatment: str,
    control: str,
    *,
    contrast: str = "custom",
) -> tuple[ContrastRow, ...]:
    treatment_rows = {
        (row.series_id, row.family, row.seed): row for row in rows if row.arm == treatment
    }
    control_rows = {
        (row.series_id, row.family, row.seed): row for row in rows if row.arm == control
    }
    if set(treatment_rows) != set(control_rows) or not treatment_rows:
        raise ValueError(f"unpaired metric rows for {treatment} versus {control}")
    return tuple(
        ContrastRow(
            contrast=contrast,
            series_id=key[0],
            family=key[1],
            seed=key[2],
            treatment=treatment,
            control=control,
            delta_vus_pr=treatment_rows[key].vus_pr - control_rows[key].vus_pr,
            delta_auprc=treatment_rows[key].auprc - control_rows[key].auprc,
            delta_vus_roc=treatment_rows[key].vus_roc - control_rows[key].vus_roc,
        )
        for key in sorted(treatment_rows)
    )


def performance_gate(
    contrasts: Sequence[ContrastRow],
    config: ProtocolConfig,
    *,
    name: str,
) -> GateResult:
    if not contrasts:
        raise ValueError("performance gate requires paired contrasts")
    gate = config.gates.performance
    family_vus: dict[str, list[float]] = {}
    family_pr: dict[str, list[float]] = {}
    for row in contrasts:
        family_vus.setdefault(row.family, []).append(row.delta_vus_pr)
        family_pr.setdefault(row.family, []).append(row.delta_auprc)
    family_delta_vus = {key: float(np.mean(value)) for key, value in family_vus.items()}
    family_delta_pr = {key: float(np.mean(value)) for key, value in family_pr.items()}
    macro_vus = float(np.mean(list(family_delta_vus.values())))
    macro_pr = float(np.mean(list(family_delta_pr.values())))
    positives = sum(value > 0 for value in family_delta_vus.values())
    worst = min(family_delta_vus.values())
    passed = (
        macro_vus >= gate.macro_vus_pr_delta_gte
        and macro_pr > gate.macro_auprc_delta_gt
        and positives >= gate.minimum_positive_families
        and worst >= gate.worst_family_delta_gte
    )
    return GateResult(
        name=name,
        passed=bool(passed),
        details={
            "macro_delta_vus_pr": macro_vus,
            "macro_delta_auprc": macro_pr,
            "positive_families": positives,
            "worst_family_delta_vus_pr": worst,
            "family_delta_vus_pr": family_delta_vus,
        },
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def activity_gate(
    iteration_logs: Sequence[Path],
    config: ProtocolConfig,
    series_to_family: Mapping[str, str],
) -> GateResult:
    gate = config.gates.low_activity
    start, end = gate.post_pretext_iteration_range
    medians: dict[str, float] = {}
    for path in iteration_logs:
        records = _load_jsonl(path)
        if not records:
            continue
        if str(records[0].get("trajectory")) != Trajectory.OFFICIAL.value:
            continue
        series_id = str(records[0]["series_id"])
        if series_id not in series_to_family:
            raise ValueError(f"unregistered activity series_id: {series_id}")
        family = series_to_family[series_id]
        values = [
            float(record["active_hinge_fraction"])
            for record in records
            if start <= int(record["iteration"]) <= end
        ]
        if len(values) != end - start + 1:
            raise ValueError(f"incomplete OFFICIAL activity log: {path}")
        medians[family] = float(np.median(values))
    if len(medians) != 6:
        raise ValueError(f"expected six OFFICIAL activity families, found {len(medians)}")
    low = {family: value <= gate.median_active_hinge_fraction_lte for family, value in medians.items()}
    count = sum(low.values())
    return GateResult(
        name="low_activity",
        passed=count >= gate.minimum_families,
        details={"family_medians": medians, "family_low_activity": low, "count": count},
    )


def checkpoint_gate(summary_paths: Sequence[Path], config: ProtocolConfig) -> GateResult:
    gate = config.gates.early_checkpoint
    best: dict[str, int] = {}
    for path in summary_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("trajectory")) != Trajectory.OFFICIAL.value:
            continue
        best[str(payload["family"])] = int(payload["best_iteration"])
    if len(best) != 6:
        raise ValueError(f"expected six OFFICIAL summaries, found {len(best)}")
    early = {family: value <= gate.best_iteration_lte for family, value in best.items()}
    count = sum(early.values())
    return GateResult(
        name="early_checkpoint",
        passed=count >= gate.minimum_families,
        details={"best_iteration": best, "family_early": early, "count": count},
    )


def decide_k0(
    config: ProtocolConfig,
    checkpoint_performance: GateResult,
    paper_negative_performance: GateResult,
    overlap_performance: GateResult,
    low_activity: GateResult,
    early_checkpoint: GateResult,
) -> dict[str, object]:
    outcomes = config.gates.outcomes
    if overlap_performance.passed and low_activity.passed:
        outcome = outcomes.nonoverlap_unique_passes_with_low_activity
        winner = Trajectory.PAPERNEG_NONOVERLAP.value
    elif paper_negative_performance.passed:
        outcome = outcomes.paperneg_only_passes
        winner = Trajectory.PAPERNEG.value
    elif checkpoint_performance.passed:
        outcome = outcomes.last_only_passes
        winner = Trajectory.OFFICIAL.value
    elif not low_activity.passed:
        outcome = outcomes.activity_not_low_and_no_execution_gain
        winner = None
    else:
        outcome = outcomes.activity_low_but_no_performance_gain
        winner = None
    return {
        "schema_version": "paano-k0-decision-v1",
        "outcome": outcome,
        "winning_trajectory": winner,
        "method_frozen": False,
        "missing_count": 0,
        "config_sha256": config.source_sha256,
        "gates": {
            item.name: {"passed": item.passed, **dict(item.details)}
            for item in (
                checkpoint_performance,
                paper_negative_performance,
                overlap_performance,
                low_activity,
                early_checkpoint,
            )
        },
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temp.replace(path)


def write_aggregate_outputs(
    output_dir: Path,
    rows: Sequence[MetricRow],
    contrasts: Sequence[ContrastRow],
    decision: Mapping[str, object],
) -> None:
    output_dir = Path(output_dir)
    metric_rows = [row.to_dict() for row in rows]
    _write_csv(output_dir / "file_metrics.csv", metric_rows, tuple(metric_rows[0]))
    contrast_payload = [asdict(row) for row in contrasts]
    _write_csv(
        output_dir / "paired_contrasts.csv",
        contrast_payload,
        tuple(contrast_payload[0]),
    )
    arms = sorted({row.arm for row in rows})
    family_payload: list[dict[str, object]] = []
    for arm in arms:
        selected = [row for row in rows if row.arm == arm]
        for family in sorted({row.family for row in selected}):
            subset = [row for row in selected if row.family == family]
            family_payload.append(
                {
                    "arm": arm,
                    "family": family,
                    "n": len(subset),
                    "vus_pr": float(np.mean([row.vus_pr for row in subset])),
                    "auprc": float(np.mean([row.auprc for row in subset])),
                    "vus_roc": float(np.mean([row.vus_roc for row in subset])),
                }
            )
    _write_csv(output_dir / "family_metrics.csv", family_payload, tuple(family_payload[0]))
    atomic_write_json(output_dir / "decision.json", decision)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from .config import expand_primary_jobs

    args = parse_args(argv)
    config = load_protocol(args.config)
    series = load_series_manifest(args.manifest)
    jobs = expand_primary_jobs(config, series, args.vendor_root, args.results_root)
    rows = load_metric_rows(args.results_root, jobs)
    all_contrasts: list[ContrastRow] = []
    gate_results: dict[str, GateResult] = {}
    for name, (treatment, control) in CONTRASTS.items():
        paired = contrast_rows(rows, treatment, control, contrast=name)
        all_contrasts.extend(paired)
        gate_results[name] = performance_gate(paired, config, name=name)
    logs = tuple(args.results_root.rglob("iteration_metrics.jsonl"))
    summaries = tuple(args.results_root.rglob("training_summary.json"))
    activity = activity_gate(logs, config, {item.series_id: item.family for item in series})
    early = checkpoint_gate(summaries, config)
    decision = decide_k0(
        config,
        gate_results["checkpoint"],
        gate_results["paper_negative"],
        gate_results["overlap"],
        activity,
        early,
    )
    write_aggregate_outputs(args.output_dir, rows, all_contrasts, decision)
    print(decision["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
