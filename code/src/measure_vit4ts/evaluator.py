"""Post-commit evaluator; this is the only module allowed to load labels."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .data import SeriesSpec, load_signal
from .metrics import average_precision, paper_f1_max, vus_pr
from .provenance import score_source_sha256


@dataclass(frozen=True)
class GroundTruth:
    point_labels: np.ndarray
    intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ValidatedScore:
    series_id: str
    arm: str
    score_path: Path
    score_sha256: str
    manifest: Mapping[str, Any]


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_config(config_path: Path) -> tuple[dict[str, Any], str]:
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a mapping")
    return config, hashlib.sha256(raw).hexdigest().upper()


def _series_record(config: Mapping[str, Any], series_id: str) -> Mapping[str, Any]:
    matches = [item for item in config["scoring"]["series"] if item["series_id"] == series_id]
    if len(matches) != 1:
        raise ValueError(f"expected one scoring record for {series_id}")
    return matches[0]


def _series_spec(config: Mapping[str, Any], record: Mapping[str, Any]) -> SeriesSpec:
    data = config["data"]
    return SeriesSpec(
        series_id=str(record["series_id"]),
        group=str(record["group"]),
        csv_path=Path(data["root"]) / str(record["relative_path"]),
        timestamp_column=str(data["timestamp_column"]),
        value_column=str(data["value_column"]),
        expected_sha256=str(record["expected_sha256"]),
    )


def validate_score_commit(
    run_dir: Path,
    expected_series: str,
    expected_arm: str,
    expected_config_sha256: str,
    expected_vendor_commit: str,
) -> ValidatedScore:
    """Validate a complete score transaction before any label is opened."""

    ready_path = run_dir / "_SCORES_READY.json"
    manifest_path = run_dir / "score_manifest.json"
    score_path = run_dir / "score.npy"
    if not ready_path.is_file() or not manifest_path.is_file() or not score_path.is_file():
        raise FileNotFoundError(f"incomplete score transaction: {run_dir}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for payload in (ready, manifest):
        if payload.get("series_id") != expected_series or payload.get("arm") != expected_arm:
            raise ValueError(f"score identity mismatch in {run_dir}")
    if manifest.get("config_sha256") != expected_config_sha256:
        raise ValueError("score config hash mismatch")
    if manifest.get("vendor_commit") != expected_vendor_commit:
        raise ValueError("score vendor commit mismatch")
    expected_source_sha = score_source_sha256()
    if (
        manifest.get("method_source_sha256") != expected_source_sha
        or ready.get("method_source_sha256") != expected_source_sha
    ):
        raise ValueError("score method-source hash mismatch")
    actual_hash = _sha256_file(score_path)
    if actual_hash != manifest.get("score_sha256") or actual_hash != ready.get("score_sha256"):
        raise ValueError("score hash mismatch")
    score = np.load(score_path, allow_pickle=False)
    if score.ndim != 1 or score.size != int(manifest.get("score_length", -1)):
        raise ValueError("score length mismatch")
    if score.dtype != np.float64 or not np.isfinite(score).all():
        raise ValueError("committed score must be finite float64")
    return ValidatedScore(expected_series, expected_arm, score_path, actual_hash, manifest)


def _read_anomaly_table(path: Path) -> dict[str, tuple[tuple[float, float], ...]]:
    table: dict[str, tuple[tuple[float, float], ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["signal", "events"]:
            raise ValueError("anomalies.csv schema changed")
        for row in reader:
            events = ast.literal_eval(row["events"])
            table[row["signal"]] = tuple((float(a), float(b)) for a, b in events)
    return table


def load_ground_truth(
    config: Mapping[str, Any], series_id: str, timestamps: np.ndarray
) -> GroundTruth:
    """Load interval labels after all requested score commits pass preflight."""

    record = _series_record(config, series_id)
    signal_name = Path(str(record["relative_path"])).stem
    anomaly_table = _read_anomaly_table(Path(config["data"]["anomalies_csv"]))
    if signal_name not in anomaly_table:
        raise KeyError(f"missing ground truth for {signal_name}")
    intervals = anomaly_table[signal_name]
    time = np.asarray(timestamps)
    labels = np.zeros(time.shape, dtype=np.uint8)
    for start, end in intervals:
        labels[(time >= start) & (time <= end)] = 1
    if intervals and not np.any(labels):
        raise ValueError(f"{series_id} has no aligned anomaly event")
    if np.all(labels):
        raise ValueError(f"{series_id} has no aligned normal point")
    return GroundTruth(labels, intervals)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def evaluate_series(
    config_path: Path, series_id: str, arms: Sequence[str]
) -> list[dict[str, Any]]:
    config, config_sha = _load_config(config_path)
    stage = str(config["stage"])
    run_root = Path(config["paths"]["run_root"]) / stage / series_id
    validated = [
        validate_score_commit(
            run_root / arm,
            series_id,
            arm,
            config_sha,
            str(config["vendor"]["commit"]),
        )
        for arm in arms
    ]
    record = _series_record(config, series_id)
    signal = load_signal(_series_spec(config, record))
    ground_truth = load_ground_truth(config, series_id, signal.timestamps)
    rows: list[dict[str, Any]] = []
    for artifact in validated:
        score = np.load(artifact.score_path, allow_pickle=False)
        f1, alpha, threshold = paper_f1_max(
            score,
            signal.timestamps,
            ground_truth.intervals,
            tuple(float(v) for v in config["evaluation"]["alpha_grid"]),
        )
        row = {
            "series_id": series_id,
            "dataset": str(record["dataset"]),
            "group": str(record["group"]),
            "arm": artifact.arm,
            "f1_max": f1,
            "f1_winning_alpha": alpha,
            "f1_threshold_evaluator_only": threshold,
            "auprc": average_precision(ground_truth.point_labels, score),
            "vus_pr": vus_pr(
                ground_truth.point_labels,
                score,
                int(config["evaluation"]["vus_max_window"]),
            ),
            "score_sha256": artifact.score_sha256,
            "n_points": int(score.size),
            "n_anomaly_points": int(ground_truth.point_labels.sum()),
        }
        rows.append(row)
        _atomic_json(run_root / artifact.arm / "metrics.json", row)
        _atomic_json(
            run_root / artifact.arm / "_EVALUATED.json",
            {"series_id": series_id, "arm": artifact.arm, "score_sha256": artifact.score_sha256},
        )
    return rows


def evaluate_all(config_path: Path, stage: str = "k0") -> Path:
    config, _ = _load_config(config_path)
    if stage != str(config["stage"]):
        raise ValueError("requested stage does not match the frozen config")
    arms = tuple(str(item) for item in config["arms"][stage])
    records = tuple(config["scoring"]["series"])
    # Complete score preflight occurs for every file before the first label read.
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest().upper()
    for record in records:
        series_id = str(record["series_id"])
        for arm in arms:
            validate_score_commit(
                Path(config["paths"]["run_root"]) / stage / series_id / arm,
                series_id,
                arm,
                config_sha,
                str(config["vendor"]["commit"]),
            )
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(evaluate_series(config_path, str(record["series_id"]), arms))
    frame = pd.DataFrame(rows).sort_values(["group", "dataset", "series_id", "arm"])
    output = Path(config["paths"]["metrics_root"]) / stage / "file_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", default="k0")
    args = parser.parse_args(argv)
    print(evaluate_all(args.config, args.stage))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
