"""Terminal, compact-only claim audit for the PaAno full benchmark.

This module deliberately uses only the Python standard library.  It consumes
the compact outputs produced by :mod:`aggregate_benchmark` and, at the
confirmation stage, :mod:`aggregate_confirmation`.  It never opens labels,
scores, datasets, checkpoints, logs, or caches and it cannot authorize more
compute or select a result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Mapping, Sequence


_SCHEMA_VERSION = "paano-terminal-claim-gate-v1"
_DECISION_SCHEMA = "paano-full-benchmark-decision-v1"
_CONFIRMATION_SCHEMA = "paano-full-confirmation-v1"
_STAGES = ("seed-2027", "confirmation")
_TRACKS = ("U", "M")
_TRACK_FILES = {"U": 350, "M": 180}
_SEEDS = (2027, 2028, 2029)
_MAIN_TRAJECTORY = "PAPERNEG_NONOVERLAP"
_TRAJECTORIES = (_MAIN_TRAJECTORY, "PAPERNEG", "OFFICIAL")
_CHECKPOINT = "LAST"
_ARM = f"{_MAIN_TRAJECTORY}_{_CHECKPOINT}"
_PAPER_REFERENCES = {"U": 0.5296, "M": 0.4263}
_PAPER_REFERENCE_SOURCE = "PaAno Table 15 default full-Eval (k=3)"
_METRICS = ("vus_pr", "auprc", "vus_roc", "auroc")
_FLOAT_TOLERANCE = 1e-12
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

_TRACK_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "seed",
    "files",
    "families",
    *_METRICS,
)
_PAPER_FIELDS = (
    "track",
    "files",
    "ours_method",
    "ours_vus_pr",
    "paper_method",
    "paper_vus_pr",
    "paper_reference_source",
    "delta_vus_pr",
    "exceeds_paper_reported",
    "comparison_type",
)
_SEED_TRACK_FIELDS = _TRACK_FIELDS
_TRACK_SUMMARY_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "seeds",
    "seed_count",
    "files_per_seed",
    "families",
    *(f"{name}_{stat}" for name in _METRICS for stat in ("mean", "std")),
    "std_ddof",
    "paper_method",
    "paper_vus_pr",
    "paper_reference_source",
    "comparison_type",
)
_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "outcome",
        "main_trajectory",
        "checkpoint",
        "paper_reference_type",
        "paper_reference_source",
        "paper_reported_vus_pr",
        "success_requires_both_tracks",
        "both_tracks_exceed",
        "conditional_confirmation_seeds",
        "tracks",
        "missing_count",
        "seed",
        "series_count",
        "metric_count",
        "config_sha256",
        "vendor_sha",
    }
)
_DECISION_TRACK_FIELDS = frozenset(
    {
        "track",
        "files",
        "ours_method",
        "ours_vus_pr",
        "paper_method",
        "paper_vus_pr",
        "paper_reference_source",
        "delta_vus_pr",
        "exceeds_paper_reported",
        "comparison_type",
    }
)
_CONFIRMATION_FIELDS = frozenset(
    {
        "schema_version",
        "trajectory",
        "checkpoint",
        "arm",
        "seeds",
        "series_per_seed",
        "track_files_per_seed",
        "metric_count",
        "seed_track_row_count",
        "std_ddof",
        "selection_applied",
        "retuning_applied",
        "result_dropping_applied",
        "paper_reference_type",
        "paper_reference_source",
        "paper_reported_vus_pr",
        "config_sha256",
        "vendor_sha",
        "seed_track_metrics",
        "track_summary",
    }
)
_SEED_TRACK_JSON_FIELDS = frozenset(_SEED_TRACK_FIELDS)
_TRACK_SUMMARY_JSON_FIELDS = frozenset(_TRACK_SUMMARY_FIELDS)
_UNSUPPORTED_CLAIMS = (
    "eval_label_selection",
    "family_specific_selection",
    "local_baseline_reproduction",
    "paired_improvement_over_paper_reported_paano",
    "statistical_significance_versus_paper_reported_paano",
)


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _read_json_object(path: Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required compact JSON is missing: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid compact JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"compact JSON must contain one object: {source}")
    return payload


def _read_csv_exact(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required compact CSV is missing: {source}")
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            parsed = list(csv.reader(handle))
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 compact CSV: {source}") from exc
    if not parsed:
        raise ValueError(f"compact CSV is empty: {source}")
    header = tuple(parsed[0])
    expected = tuple(fields)
    if header != expected:
        raise ValueError(
            f"compact CSV schema mismatch for {source.name}: "
            f"observed={header!r}, expected={expected!r}"
        )
    rows: list[dict[str, str]] = []
    for line_number, values in enumerate(parsed[1:], start=2):
        if len(values) != len(expected):
            raise ValueError(
                f"compact CSV row width mismatch in {source.name}:{line_number}"
            )
        if not any(value != "" for value in values):
            raise ValueError(f"blank compact CSV row in {source.name}:{line_number}")
        rows.append(dict(zip(expected, values, strict=True)))
    if not rows:
        raise ValueError(f"compact CSV has no data rows: {source}")
    return rows


def _require_keys(payload: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    observed = frozenset(payload)
    if observed != expected:
        raise ValueError(
            f"{where} schema mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_equal(observed: Any, expected: Any, where: str) -> None:
    if observed != expected:
        raise ValueError(
            f"{where} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be a boolean")
    return value


def _require_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer")
    return value


def _require_float(value: Any, where: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise ValueError(f"{where} must lie in [0, 1]")
    return result


def _parse_int(value: str, where: str) -> int:
    if not re.fullmatch(r"0|-?[1-9][0-9]*", value):
        raise ValueError(f"{where} is not a canonical integer: {value!r}")
    return int(value)


def _parse_float(value: str, where: str, *, unit_interval: bool = False) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{where} is not a number: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise ValueError(f"{where} must lie in [0, 1]")
    return result


def _parse_bool(value: str, where: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"{where} must be exactly True or False")
    return value == "True"


def _assert_close(observed: float, expected: float, where: str) -> None:
    if not math.isclose(
        observed, expected, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE
    ):
        raise ValueError(
            f"{where} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _validate_references(payload: Any, where: str) -> dict[str, float]:
    if not isinstance(payload, dict) or set(payload) != set(_TRACKS):
        raise ValueError(f"{where} must contain exactly U and M")
    result: dict[str, float] = {}
    for track in _TRACKS:
        value = _require_float(payload[track], f"{where}.{track}", unit_interval=True)
        _require_equal(value, _PAPER_REFERENCES[track], f"{where}.{track}")
        result[track] = value
    return result


def _validate_decision(payload: Mapping[str, Any], stage: str) -> dict[str, Any]:
    _require_keys(payload, _DECISION_FIELDS, "decision.json")
    expected_outcome = (
        "STOP_FULL_MAIN_FAILURE"
        if stage == "seed-2027"
        else "CONTINUE_FULL_CONFIRMATION"
    )
    expected_pass = stage == "confirmation"
    expected_seeds = [2028, 2029] if expected_pass else []
    fixed = {
        "schema_version": _DECISION_SCHEMA,
        "outcome": expected_outcome,
        "main_trajectory": _MAIN_TRAJECTORY,
        "checkpoint": _CHECKPOINT,
        "paper_reference_type": "external_paper_reported",
        "paper_reference_source": _PAPER_REFERENCE_SOURCE,
        "success_requires_both_tracks": True,
        "both_tracks_exceed": expected_pass,
        "conditional_confirmation_seeds": expected_seeds,
        "missing_count": 0,
        "seed": 2027,
        "series_count": sum(_TRACK_FILES.values()),
        "metric_count": sum(_TRACK_FILES.values()) * len(_TRAJECTORIES),
    }
    for field, expected in fixed.items():
        _require_equal(payload[field], expected, f"decision.json.{field}")
    _require_bool(payload["success_requires_both_tracks"], "decision success flag")
    _require_bool(payload["both_tracks_exceed"], "decision pass flag")
    for field in ("missing_count", "seed", "series_count", "metric_count"):
        _require_int(payload[field], f"decision.json.{field}")
    _validate_references(payload["paper_reported_vus_pr"], "decision paper references")

    config_sha = payload["config_sha256"]
    vendor_sha = payload["vendor_sha"]
    if not isinstance(config_sha, str) or _SHA256_RE.fullmatch(config_sha) is None:
        raise ValueError("decision config_sha256 must be a lowercase SHA-256")
    if not isinstance(vendor_sha, str) or _GIT_SHA_RE.fullmatch(vendor_sha) is None:
        raise ValueError("decision vendor_sha must be a full lowercase Git SHA")

    tracks = payload["tracks"]
    if not isinstance(tracks, dict) or set(tracks) != set(_TRACKS):
        raise ValueError("decision tracks must contain exactly U and M")
    normalized: dict[str, dict[str, Any]] = {}
    passes: list[bool] = []
    for track in _TRACKS:
        item = tracks[track]
        if not isinstance(item, dict):
            raise ValueError(f"decision tracks.{track} must be an object")
        _require_keys(item, _DECISION_TRACK_FIELDS, f"decision tracks.{track}")
        fixed_track = {
            "track": track,
            "files": _TRACK_FILES[track],
            "ours_method": _ARM,
            "paper_method": "PaAno (paper-reported)",
            "paper_reference_source": _PAPER_REFERENCE_SOURCE,
            "comparison_type": "external_paper_reported",
        }
        for field, expected in fixed_track.items():
            _require_equal(item[field], expected, f"decision tracks.{track}.{field}")
        _require_int(item["files"], f"decision tracks.{track}.files")
        ours = _require_float(
            item["ours_vus_pr"],
            f"decision tracks.{track}.ours_vus_pr",
            unit_interval=True,
        )
        paper = _require_float(
            item["paper_vus_pr"],
            f"decision tracks.{track}.paper_vus_pr",
            unit_interval=True,
        )
        _require_equal(paper, _PAPER_REFERENCES[track], f"decision {track} reference")
        delta = _require_float(item["delta_vus_pr"], f"decision tracks.{track}.delta")
        _assert_close(delta, ours - paper, f"decision tracks.{track}.delta")
        observed_pass = _require_bool(
            item["exceeds_paper_reported"],
            f"decision tracks.{track}.exceeds_paper_reported",
        )
        recomputed_pass = ours > paper
        _require_equal(
            observed_pass,
            recomputed_pass,
            f"decision tracks.{track}.strict comparison",
        )
        passes.append(recomputed_pass)
        normalized[track] = {
            "files": _TRACK_FILES[track],
            "observed_vus_pr": ours,
            "reference_vus_pr": paper,
            "delta_vus_pr": ours - paper,
            "strictly_exceeds": recomputed_pass,
        }
    _require_equal(
        all(passes), expected_pass, "decision both_tracks_exceed recomputation"
    )
    return normalized


def _normalize_track_rows(
    rows: Sequence[Mapping[str, str]], *, expected_trajectories: Sequence[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    expected_keys = {
        (trajectory, track)
        for trajectory in expected_trajectories
        for track in _TRACKS
    }
    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows, start=2):
        where = f"track metrics row {index}"
        trajectory = row["trajectory"]
        track = row["track"]
        key = (trajectory, track)
        if key in normalized:
            raise ValueError(f"duplicate compact track row: {key}")
        if key not in expected_keys:
            raise ValueError(f"unexpected compact track row: {key}")
        fixed = {
            "checkpoint": _CHECKPOINT,
            "arm": f"{trajectory}_{_CHECKPOINT}",
            "seed": 2027,
            "files": _TRACK_FILES[track],
        }
        for field, expected in fixed.items():
            observed: Any = row[field]
            if field in ("seed", "files"):
                observed = _parse_int(observed, f"{where}.{field}")
            _require_equal(observed, expected, f"{where}.{field}")
        families = _parse_int(row["families"], f"{where}.families")
        if families <= 0 or families > _TRACK_FILES[track]:
            raise ValueError(f"{where}.families is outside the valid coverage range")
        metrics = {
            name: _parse_float(row[name], f"{where}.{name}", unit_interval=True)
            for name in _METRICS
        }
        normalized[key] = {
            "trajectory": trajectory,
            "track": track,
            "families": families,
            **metrics,
        }
    if set(normalized) != expected_keys:
        raise ValueError(
            "compact track coverage mismatch: "
            f"missing={sorted(expected_keys - set(normalized))}, "
            f"extra={sorted(set(normalized) - expected_keys)}"
        )
    return normalized


def _validate_seed_compact_inputs(
    artifacts_dir: Path, decision: Mapping[str, Any], decision_tracks: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    main_rows = _read_csv_exact(artifacts_dir / "main_track_metrics.csv", _TRACK_FIELDS)
    ablation_rows = _read_csv_exact(
        artifacts_dir / "ablation_track_metrics.csv", _TRACK_FIELDS
    )
    paper_rows = _read_csv_exact(
        artifacts_dir / "paper_reference_comparison.csv", _PAPER_FIELDS
    )
    main = _normalize_track_rows(main_rows, expected_trajectories=(_MAIN_TRAJECTORY,))
    ablations = _normalize_track_rows(
        ablation_rows, expected_trajectories=_TRAJECTORIES
    )

    for track in _TRACKS:
        main_row = main[(_MAIN_TRAJECTORY, track)]
        ablation_main = ablations[(_MAIN_TRAJECTORY, track)]
        if main_row != ablation_main:
            raise ValueError(f"main/ablation compact row disagreement for track {track}")
        if main_row["families"] != ablations[("PAPERNEG", track)]["families"] or (
            main_row["families"] != ablations[("OFFICIAL", track)]["families"]
        ):
            raise ValueError(f"ablation family coverage differs for track {track}")
        _assert_close(
            main_row["vus_pr"],
            float(decision_tracks[track]["observed_vus_pr"]),
            f"main track/decision VUS-PR for {track}",
        )

    paper_by_track: dict[str, Mapping[str, str]] = {}
    for index, row in enumerate(paper_rows, start=2):
        track = row["track"]
        if track not in _TRACKS or track in paper_by_track:
            raise ValueError(f"invalid or duplicate paper comparison track at row {index}")
        paper_by_track[track] = row
    if set(paper_by_track) != set(_TRACKS):
        raise ValueError("paper comparison must contain exactly U and M")
    for track in _TRACKS:
        row = paper_by_track[track]
        fixed: dict[str, Any] = {
            "files": _TRACK_FILES[track],
            "ours_method": _ARM,
            "paper_method": "PaAno (paper-reported)",
            "paper_reference_source": _PAPER_REFERENCE_SOURCE,
            "comparison_type": "external_paper_reported",
        }
        for field, expected in fixed.items():
            observed: Any = row[field]
            if field == "files":
                observed = _parse_int(observed, f"paper comparison {track}.files")
            _require_equal(observed, expected, f"paper comparison {track}.{field}")
        ours = _parse_float(
            row["ours_vus_pr"], f"paper comparison {track}.ours", unit_interval=True
        )
        paper = _parse_float(
            row["paper_vus_pr"], f"paper comparison {track}.paper", unit_interval=True
        )
        delta = _parse_float(row["delta_vus_pr"], f"paper comparison {track}.delta")
        exceeds = _parse_bool(
            row["exceeds_paper_reported"], f"paper comparison {track}.exceeds"
        )
        _require_equal(paper, _PAPER_REFERENCES[track], f"paper comparison {track} ref")
        _assert_close(ours, main[(_MAIN_TRAJECTORY, track)]["vus_pr"], f"paper/main {track}")
        _assert_close(delta, ours - paper, f"paper comparison {track}.delta")
        _require_equal(exceeds, ours > paper, f"paper comparison {track}.strict pass")
        _require_equal(
            exceeds,
            bool(decision_tracks[track]["strictly_exceeds"]),
            f"paper comparison/decision pass for {track}",
        )

    component_tracks: dict[str, dict[str, Any]] = {}
    for track in _TRACKS:
        full = ablations[(_MAIN_TRAJECTORY, track)]["vus_pr"]
        paperneg = ablations[("PAPERNEG", track)]["vus_pr"]
        official = ablations[("OFFICIAL", track)]["vus_pr"]
        full_gt_paperneg = full > paperneg
        full_gt_official = full > official
        component_tracks[track] = {
            "full_vus_pr": full,
            "paperneg_vus_pr": paperneg,
            "official_vus_pr": official,
            "full_minus_paperneg": full - paperneg,
            "full_minus_official": full - official,
            "full_strictly_exceeds_paperneg": full_gt_paperneg,
            "full_strictly_exceeds_official": full_gt_official,
            "track_supports_attribution": full_gt_paperneg and full_gt_official,
        }
    component_supported = all(
        item["track_supports_attribution"] for item in component_tracks.values()
    )
    component = {
        "cautiously_supported": component_supported,
        "requires_strict_full_over_both_removals_on_both_tracks": True,
        "tie_blocks_attribution": True,
        "tracks": component_tracks,
    }
    return component, ablations


def _normalize_seed_track_json(
    payload: Mapping[str, Any], where: str
) -> dict[str, Any]:
    _require_keys(payload, _SEED_TRACK_JSON_FIELDS, where)
    trajectory = payload["trajectory"]
    track = payload["track"]
    seed = _require_int(payload["seed"], f"{where}.seed")
    if track not in _TRACKS or seed not in _SEEDS:
        raise ValueError(f"{where} has an unregistered track or seed")
    fixed = {
        "trajectory": _MAIN_TRAJECTORY,
        "checkpoint": _CHECKPOINT,
        "arm": _ARM,
        "files": _TRACK_FILES[track],
    }
    for field, expected in fixed.items():
        _require_equal(payload[field], expected, f"{where}.{field}")
    _require_int(payload["files"], f"{where}.files")
    families = _require_int(payload["families"], f"{where}.families")
    if families <= 0 or families > _TRACK_FILES[track]:
        raise ValueError(f"{where}.families is outside the valid coverage range")
    metrics = {
        name: _require_float(payload[name], f"{where}.{name}", unit_interval=True)
        for name in _METRICS
    }
    return {
        "trajectory": trajectory,
        "track": track,
        "seed": seed,
        "files": _TRACK_FILES[track],
        "families": families,
        **metrics,
    }


def _seed_track_csv_to_json(row: Mapping[str, str], where: str) -> dict[str, Any]:
    return _normalize_seed_track_json(
        {
            "trajectory": row["trajectory"],
            "checkpoint": row["checkpoint"],
            "arm": row["arm"],
            "track": row["track"],
            "seed": _parse_int(row["seed"], f"{where}.seed"),
            "files": _parse_int(row["files"], f"{where}.files"),
            "families": _parse_int(row["families"], f"{where}.families"),
            **{
                name: _parse_float(row[name], f"{where}.{name}", unit_interval=True)
                for name in _METRICS
            },
        },
        where,
    )


def _normalize_track_summary_json(
    payload: Mapping[str, Any], where: str
) -> dict[str, Any]:
    _require_keys(payload, _TRACK_SUMMARY_JSON_FIELDS, where)
    track = payload["track"]
    if track not in _TRACKS:
        raise ValueError(f"{where}.track is unregistered")
    fixed = {
        "trajectory": _MAIN_TRAJECTORY,
        "checkpoint": _CHECKPOINT,
        "arm": _ARM,
        "seeds": ";".join(str(seed) for seed in _SEEDS),
        "seed_count": len(_SEEDS),
        "files_per_seed": _TRACK_FILES[track],
        "std_ddof": 0,
        "paper_method": "PaAno (paper-reported)",
        "paper_reference_source": _PAPER_REFERENCE_SOURCE,
        "comparison_type": "descriptive_external_paper_reported",
    }
    for field, expected in fixed.items():
        _require_equal(payload[field], expected, f"{where}.{field}")
    for field in ("seed_count", "files_per_seed", "families", "std_ddof"):
        _require_int(payload[field], f"{where}.{field}")
    families = int(payload["families"])
    if families <= 0 or families > _TRACK_FILES[track]:
        raise ValueError(f"{where}.families is outside the valid coverage range")
    paper = _require_float(
        payload["paper_vus_pr"], f"{where}.paper_vus_pr", unit_interval=True
    )
    _require_equal(paper, _PAPER_REFERENCES[track], f"{where}.paper_vus_pr")
    metrics: dict[str, float] = {}
    for name in _METRICS:
        metrics[f"{name}_mean"] = _require_float(
            payload[f"{name}_mean"], f"{where}.{name}_mean", unit_interval=True
        )
        std = _require_float(payload[f"{name}_std"], f"{where}.{name}_std")
        if std < 0.0:
            raise ValueError(f"{where}.{name}_std must be non-negative")
        metrics[f"{name}_std"] = std
    return {"track": track, "families": families, **metrics}


def _track_summary_csv_to_json(row: Mapping[str, str], where: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trajectory": row["trajectory"],
        "checkpoint": row["checkpoint"],
        "arm": row["arm"],
        "track": row["track"],
        "seeds": row["seeds"],
        "seed_count": _parse_int(row["seed_count"], f"{where}.seed_count"),
        "files_per_seed": _parse_int(
            row["files_per_seed"], f"{where}.files_per_seed"
        ),
        "families": _parse_int(row["families"], f"{where}.families"),
        "std_ddof": _parse_int(row["std_ddof"], f"{where}.std_ddof"),
        "paper_method": row["paper_method"],
        "paper_vus_pr": _parse_float(
            row["paper_vus_pr"], f"{where}.paper_vus_pr", unit_interval=True
        ),
        "paper_reference_source": row["paper_reference_source"],
        "comparison_type": row["comparison_type"],
    }
    for name in _METRICS:
        payload[f"{name}_mean"] = _parse_float(
            row[f"{name}_mean"], f"{where}.{name}_mean", unit_interval=True
        )
        payload[f"{name}_std"] = _parse_float(
            row[f"{name}_std"], f"{where}.{name}_std"
        )
    return _normalize_track_summary_json(payload, where)


def _assert_mapping_close(
    observed: Mapping[str, Any], expected: Mapping[str, Any], where: str
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{where} normalized key mismatch")
    for field in observed:
        left = observed[field]
        right = expected[field]
        if isinstance(left, float) and isinstance(right, float):
            _assert_close(left, right, f"{where}.{field}")
        else:
            _require_equal(left, right, f"{where}.{field}")


def _validate_confirmation(
    artifacts_dir: Path,
    decision: Mapping[str, Any],
    seed_2027_track_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    seed_csv = _read_csv_exact(
        artifacts_dir / "confirmation_seed_track_metrics.csv", _SEED_TRACK_FIELDS
    )
    summary_csv = _read_csv_exact(
        artifacts_dir / "confirmation_track_summary.csv", _TRACK_SUMMARY_FIELDS
    )
    summary = _read_json_object(artifacts_dir / "confirmation_summary.json")
    _require_keys(summary, _CONFIRMATION_FIELDS, "confirmation_summary.json")
    fixed = {
        "schema_version": _CONFIRMATION_SCHEMA,
        "trajectory": _MAIN_TRAJECTORY,
        "checkpoint": _CHECKPOINT,
        "arm": _ARM,
        "seeds": list(_SEEDS),
        "series_per_seed": sum(_TRACK_FILES.values()),
        "track_files_per_seed": dict(_TRACK_FILES),
        "metric_count": sum(_TRACK_FILES.values()) * len(_SEEDS),
        "seed_track_row_count": len(_SEEDS) * len(_TRACKS),
        "std_ddof": 0,
        "selection_applied": False,
        "retuning_applied": False,
        "result_dropping_applied": False,
        "paper_reference_type": "external_paper_reported_descriptive_only",
        "paper_reference_source": _PAPER_REFERENCE_SOURCE,
    }
    for field, expected in fixed.items():
        _require_equal(summary[field], expected, f"confirmation_summary.json.{field}")
    for field in (
        "series_per_seed",
        "metric_count",
        "seed_track_row_count",
        "std_ddof",
    ):
        _require_int(summary[field], f"confirmation_summary.json.{field}")
    for field in ("selection_applied", "retuning_applied", "result_dropping_applied"):
        _require_bool(summary[field], f"confirmation_summary.json.{field}")
    _validate_references(
        summary["paper_reported_vus_pr"], "confirmation summary paper references"
    )
    _require_equal(
        summary["config_sha256"],
        decision["config_sha256"],
        "confirmation/decision config SHA",
    )
    _require_equal(
        summary["vendor_sha"], decision["vendor_sha"], "confirmation/decision vendor SHA"
    )

    json_seed_rows = summary["seed_track_metrics"]
    if not isinstance(json_seed_rows, list):
        raise ValueError("confirmation seed_track_metrics must be an array")
    normalized_json_seed: dict[tuple[int, str], dict[str, Any]] = {}
    for index, item in enumerate(json_seed_rows):
        if not isinstance(item, dict):
            raise ValueError("confirmation seed_track_metrics entries must be objects")
        row = _normalize_seed_track_json(item, f"confirmation seed JSON row {index}")
        key = (row["seed"], row["track"])
        if key in normalized_json_seed:
            raise ValueError(f"duplicate confirmation seed-track JSON row: {key}")
        normalized_json_seed[key] = row
    expected_seed_keys = {(seed, track) for seed in _SEEDS for track in _TRACKS}
    if set(normalized_json_seed) != expected_seed_keys:
        raise ValueError("confirmation seed-track JSON coverage is not exact")

    normalized_csv_seed: dict[tuple[int, str], dict[str, Any]] = {}
    for index, item in enumerate(seed_csv, start=2):
        row = _seed_track_csv_to_json(item, f"confirmation seed CSV row {index}")
        key = (row["seed"], row["track"])
        if key in normalized_csv_seed:
            raise ValueError(f"duplicate confirmation seed-track CSV row: {key}")
        normalized_csv_seed[key] = row
    if set(normalized_csv_seed) != expected_seed_keys:
        raise ValueError("confirmation seed-track CSV coverage is not exact")
    for key in sorted(expected_seed_keys):
        _assert_mapping_close(
            normalized_csv_seed[key],
            normalized_json_seed[key],
            f"confirmation seed-track cross-file row {key}",
        )
    for track in _TRACKS:
        confirmation_row = normalized_json_seed[(2027, track)]
        seed_row = seed_2027_track_rows[(_MAIN_TRAJECTORY, track)]
        _require_equal(
            confirmation_row["families"],
            seed_row["families"],
            f"confirmation/seed-2027 {track} family coverage",
        )
        for metric in _METRICS:
            _assert_close(
                confirmation_row[metric],
                seed_row[metric],
                f"confirmation/seed-2027 {track} {metric}",
            )

    json_summary_rows = summary["track_summary"]
    if not isinstance(json_summary_rows, list):
        raise ValueError("confirmation track_summary must be an array")
    normalized_json_summary: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(json_summary_rows):
        if not isinstance(item, dict):
            raise ValueError("confirmation track_summary entries must be objects")
        row = _normalize_track_summary_json(item, f"confirmation summary JSON row {index}")
        track = row["track"]
        if track in normalized_json_summary:
            raise ValueError(f"duplicate confirmation track summary JSON row: {track}")
        normalized_json_summary[track] = row
    normalized_csv_summary: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(summary_csv, start=2):
        row = _track_summary_csv_to_json(item, f"confirmation summary CSV row {index}")
        track = row["track"]
        if track in normalized_csv_summary:
            raise ValueError(f"duplicate confirmation track summary CSV row: {track}")
        normalized_csv_summary[track] = row
    if set(normalized_json_summary) != set(_TRACKS) or set(normalized_csv_summary) != set(
        _TRACKS
    ):
        raise ValueError("confirmation track summary coverage is not exact")
    for track in _TRACKS:
        _assert_mapping_close(
            normalized_csv_summary[track],
            normalized_json_summary[track],
            f"confirmation track-summary cross-file row {track}",
        )
        families = {normalized_json_seed[(seed, track)]["families"] for seed in _SEEDS}
        if families != {normalized_json_summary[track]["families"]}:
            raise ValueError(f"confirmation family coverage differs for track {track}")
        for metric in _METRICS:
            values = [normalized_json_seed[(seed, track)][metric] for seed in _SEEDS]
            mean = statistics.fmean(values)
            std = statistics.pstdev(values)
            _assert_close(
                normalized_json_summary[track][f"{metric}_mean"],
                mean,
                f"confirmation {track} recomputed {metric} mean",
            )
            _assert_close(
                normalized_json_summary[track][f"{metric}_std"],
                std,
                f"confirmation {track} recomputed {metric} population std",
            )

    seed_results: list[dict[str, Any]] = []
    all_seeds_both_tracks = True
    for seed in _SEEDS:
        tracks: dict[str, dict[str, Any]] = {}
        for track in _TRACKS:
            observed = normalized_json_seed[(seed, track)]["vus_pr"]
            reference = _PAPER_REFERENCES[track]
            tracks[track] = {
                "observed_vus_pr": observed,
                "reference_vus_pr": reference,
                "delta_vus_pr": observed - reference,
                "strictly_exceeds": observed > reference,
            }
        both = all(item["strictly_exceeds"] for item in tracks.values())
        all_seeds_both_tracks = all_seeds_both_tracks and both
        seed_results.append(
            {"seed": seed, "both_tracks_strictly_exceed": both, "tracks": tracks}
        )

    track_means: dict[str, dict[str, Any]] = {}
    for track in _TRACKS:
        observed = normalized_json_summary[track]["vus_pr_mean"]
        reference = _PAPER_REFERENCES[track]
        track_means[track] = {
            "observed_vus_pr_mean": observed,
            "observed_vus_pr_population_std": normalized_json_summary[track][
                "vus_pr_std"
            ],
            "reference_vus_pr": reference,
            "delta_vus_pr": observed - reference,
            "strictly_exceeds": observed > reference,
        }
    mean_both_tracks = all(item["strictly_exceeds"] for item in track_means.values())
    if mean_both_tracks and all_seeds_both_tracks:
        branch = "STABLE_EXTERNAL_EXCEEDANCE"
    elif mean_both_tracks:
        branch = "MEAN_PASS_WITH_SEED_INSTABILITY"
    else:
        branch = "MEAN_FAILURE"
    confirmation = {
        "branch": branch,
        "fixed_seeds": list(_SEEDS),
        "all_seeds_both_tracks_strictly_exceed": all_seeds_both_tracks,
        "mean_both_tracks_strictly_exceeds": mean_both_tracks,
        "population_std_ddof": 0,
        "seed_results": seed_results,
        "track_means": track_means,
    }
    return branch, confirmation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_claim_gate(artifacts_dir: Path, output: Path, stage: str) -> dict[str, Any]:
    """Validate compact terminal artifacts and atomically write the claim gate.

    ``stage='seed-2027'`` is valid only for a frozen
    ``STOP_FULL_MAIN_FAILURE``.  ``stage='confirmation'`` is valid only after a
    frozen ``CONTINUE_FULL_CONFIRMATION`` and exact fixed-three-seed compact
    aggregation.  Any disagreement raises before the output is replaced.
    """

    if stage not in _STAGES:
        raise ValueError(f"claim-gate stage must be one of {_STAGES!r}")
    root = Path(artifacts_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"compact artifact directory is missing: {root}")
    decision_path = root / "decision.json"
    decision = _read_json_object(decision_path)
    decision_tracks = _validate_decision(decision, stage)
    component, seed_2027_track_rows = _validate_seed_compact_inputs(
        root, decision, decision_tracks
    )

    consumed = [
        decision_path,
        root / "main_track_metrics.csv",
        root / "ablation_track_metrics.csv",
        root / "paper_reference_comparison.csv",
    ]
    confirmation: dict[str, Any] | None = None
    if stage == "confirmation":
        claim_outcome, confirmation = _validate_confirmation(
            root, decision, seed_2027_track_rows
        )
        consumed.extend(
            [
                root / "confirmation_seed_track_metrics.csv",
                root / "confirmation_track_summary.csv",
                root / "confirmation_summary.json",
            ]
        )
    else:
        claim_outcome = "STOP_FULL_MAIN_FAILURE"

    external_pass_count = sum(
        bool(item["strictly_exceeds"]) for item in decision_tracks.values()
    )
    external_branch = {
        0: "SEED2027_NO_TRACK_STRICTLY_EXCEEDS",
        1: "SEED2027_ONE_TRACK_STRICTLY_EXCEEDS",
        2: "SEED2027_BOTH_TRACKS_STRICTLY_EXCEED",
    }[external_pass_count]
    component_branch = (
        "COMPONENT_EVIDENCE_CAUTIOUSLY_ALLOWED"
        if component["cautiously_supported"]
        else "COMPONENT_ATTRIBUTION_BLOCKED_MIXED_OR_DOMINATED"
    )

    external = {
        "paper_reference_type": "external_paper_reported",
        "paper_reference_source": _PAPER_REFERENCE_SOURCE,
        "paper_reported_vus_pr": dict(_PAPER_REFERENCES),
        "seed_2027": {
            "both_tracks_strictly_exceed": all(
                item["strictly_exceeds"] for item in decision_tracks.values()
            ),
            "tracks": dict(decision_tracks),
        },
    }
    claim_permissions = {
        "cautious_component_attribution": bool(component["cautiously_supported"]),
        "descriptive_fixed_three_seed_mean_exceedance": bool(
            confirmation is not None
            and confirmation["mean_both_tracks_strictly_exceeds"]
        ),
        "stable_fixed_three_seed_external_exceedance": bool(
            claim_outcome == "STABLE_EXTERNAL_EXCEEDANCE"
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "stage": stage,
        "terminal": True,
        "claim_only": True,
        "experiment_decision": decision["outcome"],
        "claim_outcome": claim_outcome,
        "external_branch": external_branch,
        "component_branch": component_branch,
        "confirmation_branch": (
            confirmation["branch"] if confirmation is not None else None
        ),
        "external_performance": external,
        "component_attribution": component,
        "confirmation": confirmation,
        "claim_permissions": claim_permissions,
        "unsupported_claims": list(_UNSUPPORTED_CLAIMS),
        "input_sha256": {
            path.name: _sha256_file(path) for path in sorted(consumed, key=lambda p: p.name)
        },
        "selection_applied": False,
        "result_dropping_applied": False,
        "authorizes_compute": False,
    }
    _atomic_write_json(Path(output), payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=_STAGES, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = evaluate_claim_gate(args.artifacts_dir, args.output, args.stage)
    print(
        "TERMINAL_CLAIM_GATE_COMPLETE "
        f"stage={payload['stage']} claim_outcome={payload['claim_outcome']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
