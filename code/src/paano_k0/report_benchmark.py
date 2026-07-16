"""Render the frozen full-benchmark report from compact aggregates only.

This module deliberately uses only the Python standard library.  In particular,
it has no evaluator, dataset, label, or score-artifact dependency: the seven
registered compact aggregate files are its complete input surface.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence
import uuid


EXPECTED_TRACK_COUNTS: Mapping[str, int] = {"U": 350, "M": 180}
REGISTERED_TRAJECTORIES = (
    "PAPERNEG_NONOVERLAP",
    "PAPERNEG",
    "OFFICIAL",
)
MAIN_TRAJECTORY = "PAPERNEG_NONOVERLAP"
REGISTERED_SEED = 2027
CHECKPOINT = "LAST"
PAPER_REPORTED_VUS_PR: Mapping[str, float] = {"U": 0.5296, "M": 0.4263}
PAPER_REFERENCE_SOURCE = "PaAno Table 15 default full-Eval (k=3)"
# SHA-256 of sorted lines ``track<TAB>family<TAB>series_id<TAB>data_sha256``
# from docs/TSB_AD_FULL_EVAL_MANIFEST.csv, each terminated by ``\n``.  This
# lets the compact-only renderer prove exact registered membership without
# reopening the manifest, dataset, raw scores, or labels.
CANONICAL_MANIFEST_SERIES_SHA256 = (
    "df77993c171208570d65d639a8a1849c00518ad9a78cb0fd6dcfb6e40df2f02e"
)

_TRACKS = tuple(EXPECTED_TRACK_COUNTS)
_METRICS = ("vus_pr", "auprc", "vus_roc", "auroc")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_GIT_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")

_FILE_FIELDS = (
    "run_id",
    "series_id",
    "family",
    "track",
    "seed",
    "trajectory",
    "checkpoint",
    "arm",
    "vus_pr",
    "auprc",
    "vus_roc",
    "auroc",
    "score_sha256",
    "data_sha256",
    "config_sha256",
    "vendor_sha",
)
_FAMILY_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "family",
    "seed",
    "files",
    *_METRICS,
)
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
_RUNTIME_FIELDS = (
    "trajectory",
    "checkpoint",
    "track",
    "seed",
    "files",
    "training_runtime_seconds_sum",
    "training_runtime_seconds_mean",
    "scoring_runtime_seconds_sum",
    "scoring_runtime_seconds_mean",
    "training_peak_vram_mib_max",
    "scoring_peak_vram_mib_max",
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


@dataclass(frozen=True, slots=True)
class _ReportData:
    main_files: tuple[dict[str, str], ...]
    main_families: tuple[dict[str, str], ...]
    main_tracks: tuple[dict[str, str], ...]
    ablation_tracks: tuple[dict[str, str], ...]
    paper_comparison: tuple[dict[str, str], ...]
    runtime: tuple[dict[str, str], ...]
    decision: dict[str, object]


def _registered_arm(trajectory: str) -> str:
    return f"{trajectory}_{CHECKPOINT}"


def _display_arm(trajectory: str) -> str:
    return f"{trajectory}-{CHECKPOINT}"


def _expected_run_id(series_id: str) -> str:
    safe_series = re.sub(r"[^A-Za-z0-9._-]+", "-", series_id).strip(" .-")
    if not safe_series:
        raise ValueError("main_file_metrics.csv contains an invalid series_id")
    return (
        f"{safe_series}__seed_{REGISTERED_SEED}__"
        f"{MAIN_TRAJECTORY}__{CHECKPOINT}"
    )


def _read_csv_exact(
    path: Path, fields: Sequence[str]
) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"missing compact benchmark artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        expected = tuple(fields)
        if actual != expected:
            raise ValueError(
                f"{path.name} columns changed: expected={expected}, actual={actual}"
            )
        rows = tuple(dict(row) for row in reader)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing compact benchmark artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return payload


def _int_value(row: Mapping[str, str], field: str, source: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}.{field} must be an integer") from error
    return value


def _float_value(row: Mapping[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}.{field} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{source}.{field} must be finite")
    return value


def _bool_value(row: Mapping[str, str], field: str, source: str) -> bool:
    raw = row.get(field)
    if raw == "True":
        return True
    if raw == "False":
        return False
    raise ValueError(f"{source}.{field} must be True or False")


def _json_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"decision.json.{field} must be an integer")
    return value


def _json_float(payload: Mapping[str, object], field: str, source: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}.{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source}.{field} must be finite")
    return result


def _expect_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: expected={expected!r}, actual={actual!r}")


def _expect_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{message}: expected={expected!r}, actual={actual!r}")


def _validate_metric_values(row: Mapping[str, str], source: str) -> None:
    for metric in _METRICS:
        value = _float_value(row, metric, source)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{source}.{metric} must lie in [0, 1]")


def _validate_arm_row(
    row: Mapping[str, str], trajectory: str, track: str, source: str
) -> None:
    _expect_equal(row.get("trajectory"), trajectory, f"{source} trajectory")
    _expect_equal(row.get("checkpoint"), CHECKPOINT, f"{source} checkpoint")
    _expect_equal(row.get("arm"), _registered_arm(trajectory), f"{source} arm")
    _expect_equal(row.get("track"), track, f"{source} track")
    _expect_equal(_int_value(row, "seed", source), REGISTERED_SEED, f"{source} seed")


def _index_exact(
    rows: Sequence[dict[str, str]],
    keys: Sequence[tuple[str, ...]],
    fields: Sequence[str],
    source: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if key in indexed:
            raise ValueError(f"{source} contains duplicate row {key}")
        indexed[key] = row
    expected = set(keys)
    actual = set(indexed)
    if actual != expected:
        raise ValueError(
            f"{source} registered rows changed: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return indexed


def _mean(rows: Sequence[Mapping[str, str]], metric: str, source: str) -> float:
    if not rows:
        raise ValueError(f"cannot average empty {source}")
    return math.fsum(_float_value(row, metric, source) for row in rows) / len(rows)


def _validate_main_files(rows: Sequence[dict[str, str]]) -> None:
    _expect_equal(len(rows), 530, "main_file_metrics.csv row count")
    run_ids: set[str] = set()
    series_ids: set[str] = set()
    config_hashes: set[str] = set()
    vendor_hashes: set[str] = set()
    counts = {track: 0 for track in _TRACKS}
    for index, row in enumerate(rows, start=2):
        source = f"main_file_metrics.csv row {index}"
        track = row.get("track", "")
        if track not in EXPECTED_TRACK_COUNTS:
            raise ValueError(f"{source} has unregistered track {track!r}")
        _validate_arm_row(row, MAIN_TRAJECTORY, track, source)
        _validate_metric_values(row, source)
        run_id = row.get("run_id", "")
        series_id = row.get("series_id", "")
        family = row.get("family", "")
        if not run_id or not series_id or not family:
            raise ValueError(f"{source} has an empty run_id, series_id, or family")
        _expect_equal(run_id, _expected_run_id(series_id), f"{source} run_id")
        if run_id in run_ids or series_id in series_ids:
            raise ValueError(f"{source} duplicates a run_id or series_id")
        run_ids.add(run_id)
        series_ids.add(series_id)
        for field in ("score_sha256", "data_sha256", "config_sha256"):
            if not _HEX_64.fullmatch(row.get(field, "")):
                raise ValueError(f"{source}.{field} is not a lowercase SHA-256")
        if not _HEX_40.fullmatch(row.get("vendor_sha", "")):
            raise ValueError(f"{source}.vendor_sha is not a lowercase Git SHA-1")
        config_hashes.add(row["config_sha256"])
        vendor_hashes.add(row["vendor_sha"])
        counts[track] += 1
    _expect_equal(counts, dict(EXPECTED_TRACK_COUNTS), "main file track coverage")
    _expect_equal(len(config_hashes), 1, "main file config provenance count")
    _expect_equal(len(vendor_hashes), 1, "main file vendor provenance count")
    membership_lines = sorted(
        "\t".join(
            (
                row["track"],
                row["family"],
                row["series_id"],
                row["data_sha256"],
            )
        )
        for row in rows
    )
    membership_payload = ("\n".join(membership_lines) + "\n").encode("utf-8")
    membership_sha256 = hashlib.sha256(membership_payload).hexdigest()
    _expect_equal(
        membership_sha256,
        CANONICAL_MANIFEST_SERIES_SHA256,
        "main file canonical manifest membership SHA-256",
    )


def _validate_main_families(
    family_rows: Sequence[dict[str, str]], file_rows: Sequence[dict[str, str]]
) -> dict[tuple[str, str], dict[str, str]]:
    expected_keys = sorted({(row["track"], row["family"]) for row in file_rows})
    indexed = _index_exact(
        family_rows,
        expected_keys,
        ("track", "family"),
        "main_family_metrics.csv",
    )
    for key, row in indexed.items():
        track, family = key
        source = f"main_family_metrics.csv {track}/{family}"
        _validate_arm_row(row, MAIN_TRAJECTORY, track, source)
        _validate_metric_values(row, source)
        subset = [
            item
            for item in file_rows
            if item["track"] == track and item["family"] == family
        ]
        _expect_equal(_int_value(row, "files", source), len(subset), f"{source} files")
        for metric in _METRICS:
            _expect_close(
                _float_value(row, metric, source),
                _mean(subset, metric, source),
                f"{source} {metric} mean",
            )
    return indexed


def _validate_main_tracks(
    track_rows: Sequence[dict[str, str]],
    file_rows: Sequence[dict[str, str]],
    family_rows: Mapping[tuple[str, str], dict[str, str]],
) -> dict[str, dict[str, str]]:
    indexed_tuple = _index_exact(
        track_rows,
        [(track,) for track in _TRACKS],
        ("track",),
        "main_track_metrics.csv",
    )
    indexed = {key[0]: value for key, value in indexed_tuple.items()}
    for track, row in indexed.items():
        source = f"main_track_metrics.csv {track}"
        _validate_arm_row(row, MAIN_TRAJECTORY, track, source)
        _validate_metric_values(row, source)
        subset = [item for item in file_rows if item["track"] == track]
        families = sum(key[0] == track for key in family_rows)
        _expect_equal(
            _int_value(row, "files", source),
            EXPECTED_TRACK_COUNTS[track],
            f"{source} files",
        )
        _expect_equal(
            _int_value(row, "families", source), families, f"{source} families"
        )
        for metric in _METRICS:
            _expect_close(
                _float_value(row, metric, source),
                _mean(subset, metric, source),
                f"{source} {metric} mean",
            )
    return indexed


def _validate_ablation_tracks(
    rows: Sequence[dict[str, str]], main_tracks: Mapping[str, dict[str, str]]
) -> dict[tuple[str, str], dict[str, str]]:
    expected = [
        (trajectory, track)
        for trajectory in REGISTERED_TRAJECTORIES
        for track in _TRACKS
    ]
    indexed = _index_exact(
        rows,
        expected,
        ("trajectory", "track"),
        "ablation_track_metrics.csv",
    )
    for (trajectory, track), row in indexed.items():
        source = f"ablation_track_metrics.csv {trajectory}/{track}"
        _validate_arm_row(row, trajectory, track, source)
        _validate_metric_values(row, source)
        _expect_equal(
            _int_value(row, "files", source),
            EXPECTED_TRACK_COUNTS[track],
            f"{source} files",
        )
        _expect_equal(
            _int_value(row, "families", source),
            _int_value(main_tracks[track], "families", f"main track {track}"),
            f"{source} families",
        )
        if trajectory == MAIN_TRAJECTORY:
            main = main_tracks[track]
            for field in ("seed", "files", "families", *_METRICS):
                if field in _METRICS:
                    _expect_close(
                        _float_value(row, field, source),
                        _float_value(main, field, f"main track {track}"),
                        f"{source} differs from main_track_metrics.csv",
                    )
                else:
                    _expect_equal(
                        row[field],
                        main[field],
                        f"{source} differs from main_track_metrics.csv field {field}",
                    )
    return indexed


def _validate_paper_comparison(
    rows: Sequence[dict[str, str]], main_tracks: Mapping[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    indexed_tuple = _index_exact(
        rows,
        [(track,) for track in _TRACKS],
        ("track",),
        "paper_reference_comparison.csv",
    )
    indexed = {key[0]: value for key, value in indexed_tuple.items()}
    for track, row in indexed.items():
        source = f"paper_reference_comparison.csv {track}"
        ours = _float_value(main_tracks[track], "vus_pr", f"main track {track}")
        paper = PAPER_REPORTED_VUS_PR[track]
        exceeds = ours > paper
        _expect_equal(
            _int_value(row, "files", source),
            EXPECTED_TRACK_COUNTS[track],
            f"{source} files",
        )
        _expect_equal(
            row.get("ours_method"),
            _registered_arm(MAIN_TRAJECTORY),
            f"{source} ours_method",
        )
        _expect_close(_float_value(row, "ours_vus_pr", source), ours, f"{source} ours")
        _expect_equal(
            row.get("paper_method"), "PaAno (paper-reported)", f"{source} method"
        )
        _expect_close(
            _float_value(row, "paper_vus_pr", source), paper, f"{source} paper value"
        )
        _expect_equal(
            row.get("paper_reference_source"),
            PAPER_REFERENCE_SOURCE,
            f"{source} paper source",
        )
        _expect_close(
            _float_value(row, "delta_vus_pr", source),
            ours - paper,
            f"{source} delta",
        )
        _expect_equal(
            _bool_value(row, "exceeds_paper_reported", source),
            exceeds,
            f"{source} strict comparison",
        )
        _expect_equal(
            row.get("comparison_type"),
            "external_paper_reported",
            f"{source} comparison type",
        )
    return indexed


def _validate_runtime(rows: Sequence[dict[str, str]]) -> None:
    expected = [
        (trajectory, track)
        for trajectory in REGISTERED_TRAJECTORIES
        for track in _TRACKS
    ]
    indexed = _index_exact(
        rows, expected, ("trajectory", "track"), "runtime_summary.csv"
    )
    for (trajectory, track), row in indexed.items():
        source = f"runtime_summary.csv {trajectory}/{track}"
        _expect_equal(row.get("checkpoint"), CHECKPOINT, f"{source} checkpoint")
        _expect_equal(_int_value(row, "seed", source), REGISTERED_SEED, f"{source} seed")
        files = _int_value(row, "files", source)
        _expect_equal(files, EXPECTED_TRACK_COUNTS[track], f"{source} files")
        values: dict[str, float] = {}
        for field in _RUNTIME_FIELDS[5:]:
            values[field] = _float_value(row, field, source)
            if values[field] < 0.0:
                raise ValueError(f"{source}.{field} must be non-negative")
        _expect_close(
            values["training_runtime_seconds_sum"],
            values["training_runtime_seconds_mean"] * files,
            f"{source} training sum/mean",
        )
        _expect_close(
            values["scoring_runtime_seconds_sum"],
            values["scoring_runtime_seconds_mean"] * files,
            f"{source} scoring sum/mean",
        )


def _validate_decision(
    decision: dict[str, object],
    main_files: Sequence[dict[str, str]],
    paper: Mapping[str, dict[str, str]],
) -> None:
    if set(decision) != _DECISION_FIELDS:
        raise ValueError(
            "decision.json fields changed: "
            f"missing={sorted(_DECISION_FIELDS - set(decision))}, "
            f"extra={sorted(set(decision) - _DECISION_FIELDS)}"
        )
    _expect_equal(
        decision["schema_version"],
        "paano-full-benchmark-decision-v1",
        "decision schema",
    )
    _expect_equal(decision["main_trajectory"], MAIN_TRAJECTORY, "decision trajectory")
    _expect_equal(decision["checkpoint"], CHECKPOINT, "decision checkpoint")
    _expect_equal(
        decision["paper_reference_type"],
        "external_paper_reported",
        "decision reference type",
    )
    _expect_equal(
        decision["paper_reference_source"],
        PAPER_REFERENCE_SOURCE,
        "decision paper source",
    )
    _expect_equal(decision["success_requires_both_tracks"], True, "decision gate")
    _expect_equal(_json_int(decision, "missing_count"), 0, "decision missing count")
    _expect_equal(_json_int(decision, "seed"), REGISTERED_SEED, "decision seed")
    _expect_equal(_json_int(decision, "series_count"), 530, "decision series count")
    _expect_equal(_json_int(decision, "metric_count"), 1590, "decision metric count")

    references = decision["paper_reported_vus_pr"]
    if not isinstance(references, dict) or set(references) != set(_TRACKS):
        raise ValueError("decision paper_reported_vus_pr must contain exactly U and M")
    for track in _TRACKS:
        _expect_close(
            _json_float(references, track, "decision paper_reported_vus_pr"),
            PAPER_REPORTED_VUS_PR[track],
            f"decision paper value {track}",
        )

    config_hashes = {row["config_sha256"] for row in main_files}
    vendor_hashes = {row["vendor_sha"] for row in main_files}
    _expect_equal(decision["config_sha256"], next(iter(config_hashes)), "decision config SHA")
    _expect_equal(decision["vendor_sha"], next(iter(vendor_hashes)), "decision vendor SHA")

    tracks = decision["tracks"]
    if not isinstance(tracks, dict) or set(tracks) != set(_TRACKS):
        raise ValueError("decision tracks must contain exactly U and M")
    observed_passes: list[bool] = []
    for track in _TRACKS:
        item = tracks[track]
        if not isinstance(item, dict) or set(item) != set(_PAPER_FIELDS):
            raise ValueError(f"decision track {track} fields changed")
        csv_row = paper[track]
        source = f"decision track {track}"
        for field in (
            "track",
            "ours_method",
            "paper_method",
            "paper_reference_source",
            "comparison_type",
        ):
            _expect_equal(item[field], csv_row[field], f"{source} {field}")
        _expect_equal(item["files"], int(csv_row["files"]), f"{source} files")
        for field in ("ours_vus_pr", "paper_vus_pr", "delta_vus_pr"):
            _expect_close(
                _json_float(item, field, source),
                float(csv_row[field]),
                f"{source} {field}",
            )
        csv_pass = _bool_value(csv_row, "exceeds_paper_reported", source)
        _expect_equal(item["exceeds_paper_reported"], csv_pass, f"{source} pass")
        observed_passes.append(csv_pass)

    passed = all(observed_passes)
    _expect_equal(decision["both_tracks_exceed"], passed, "decision both-track result")
    expected_outcome = (
        "CONTINUE_FULL_CONFIRMATION" if passed else "STOP_FULL_MAIN_FAILURE"
    )
    _expect_equal(decision["outcome"], expected_outcome, "decision outcome")
    _expect_equal(
        decision["conditional_confirmation_seeds"],
        [2028, 2029] if passed else [],
        "decision conditional seeds",
    )


def _load_and_validate(artifacts_dir: Path) -> _ReportData:
    root = Path(artifacts_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"compact benchmark artifact directory is missing: {root}")
    main_files = _read_csv_exact(root / "main_file_metrics.csv", _FILE_FIELDS)
    main_families = _read_csv_exact(
        root / "main_family_metrics.csv", _FAMILY_FIELDS
    )
    main_tracks = _read_csv_exact(root / "main_track_metrics.csv", _TRACK_FIELDS)
    ablation_tracks = _read_csv_exact(
        root / "ablation_track_metrics.csv", _TRACK_FIELDS
    )
    paper_comparison = _read_csv_exact(
        root / "paper_reference_comparison.csv", _PAPER_FIELDS
    )
    runtime = _read_csv_exact(root / "runtime_summary.csv", _RUNTIME_FIELDS)
    decision = _read_json_object(root / "decision.json")

    _validate_main_files(main_files)
    family_index = _validate_main_families(main_families, main_files)
    track_index = _validate_main_tracks(main_tracks, main_files, family_index)
    _validate_ablation_tracks(ablation_tracks, track_index)
    paper_index = _validate_paper_comparison(paper_comparison, track_index)
    _validate_runtime(runtime)
    _validate_decision(decision, main_files, paper_index)
    return _ReportData(
        main_files=main_files,
        main_families=main_families,
        main_tracks=main_tracks,
        ablation_tracks=ablation_tracks,
        paper_comparison=paper_comparison,
        runtime=runtime,
        decision=decision,
    )


def _markdown_text(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _metric(row: Mapping[str, str], name: str) -> str:
    return f"{float(row[name]):.6f}"


def _render_markdown(data: _ReportData, git_sha: str | None) -> str:
    main_tracks = {row["track"]: row for row in data.main_tracks}
    ablations = {
        (row["trajectory"], row["track"]): row for row in data.ablation_tracks
    }
    paper = {row["track"]: row for row in data.paper_comparison}
    runtime = {(row["trajectory"], row["track"]): row for row in data.runtime}
    family_counts = {
        track: sum(row["track"] == track for row in data.main_families)
        for track in _TRACKS
    }
    decision = data.decision

    lines = [
        "# PaAno Full-Benchmark Main Results",
        "",
        "> Numeric report rendered exclusively from the seven registered compact aggregate outputs; no raw score, label, or dataset file was reopened.",
    ]
    if git_sha is not None:
        lines.append(f"> Report code Git SHA: `{git_sha.lower()}`")
    lines.extend(
        [
            "",
            "## Protocol and complete coverage",
            "",
            "All results use the frozen seed 2027 endpoint, file-weighted aggregation, and the `LAST` checkpoint. The same complete Eval lists are used for every registered arm.",
            "",
            "| Track | Eval series | Main-arm families |",
            "|---|---:|---:|",
        ]
    )
    for track in _TRACKS:
        lines.append(
            f"| TSB-AD-{track} | {EXPECTED_TRACK_COUNTS[track]} | {family_counts[track]} |"
        )
    lines.extend(
        [
            f"| **Total** | **{sum(EXPECTED_TRACK_COUNTS.values())}** | **{len(data.main_families)}** |",
            "",
            "The registered arms are `PAPERNEG_NONOVERLAP-LAST` (full arm), `PAPERNEG-LAST` (remove non-overlap positives), and `OFFICIAL-LAST` (remove both registered execution changes). No arm, track, family, or result is selected after evaluation.",
            "",
            "## Main results",
            "",
            "| Arm | Track | Files | VUS-PR | AUPRC | VUS-ROC | AUROC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for track in _TRACKS:
        row = main_tracks[track]
        lines.append(
            f"| `{_display_arm(MAIN_TRAJECTORY)}` | {track} | {row['files']} | "
            f"{_metric(row, 'vus_pr')} | {_metric(row, 'auprc')} | "
            f"{_metric(row, 'vus_roc')} | {_metric(row, 'auroc')} |"
        )

    lines.extend(
        [
            "",
            "### Complete main-arm family results",
            "",
            "| Track | Family | Files | VUS-PR | AUPRC | VUS-ROC | AUROC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        data.main_families,
        key=lambda item: (_TRACKS.index(item["track"]), item["family"]),
    ):
        lines.append(
            f"| {row['track']} | {_markdown_text(row['family'])} | {row['files']} | "
            f"{_metric(row, 'vus_pr')} | {_metric(row, 'auprc')} | "
            f"{_metric(row, 'vus_roc')} | {_metric(row, 'auroc')} |"
        )

    lines.extend(
        [
            "",
            "## Component-removal ablations",
            "",
            "These are matched seed-2027 controls on the same files and frozen `LAST` endpoint; they are not the paper's external ablation rows.",
            "",
            "| Registered arm | Track | Files | VUS-PR | AUPRC | VUS-ROC | AUROC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for trajectory in REGISTERED_TRAJECTORIES:
        for track in _TRACKS:
            row = ablations[(trajectory, track)]
            lines.append(
                f"| `{_display_arm(trajectory)}` | {track} | {row['files']} | "
                f"{_metric(row, 'vus_pr')} | {_metric(row, 'auprc')} | "
                f"{_metric(row, 'vus_roc')} | {_metric(row, 'auroc')} |"
            )

    lines.extend(
        [
            "",
            "## External paper-reported comparison",
            "",
            f"The fixed external reference is **{PAPER_REFERENCE_SOURCE}**. These values are paper-reported ten-seed results, not a local reproduction or a paired same-seed baseline.",
            "",
            "| Track | Our full arm VUS-PR | PaAno paper-reported VUS-PR | Delta | Strictly exceeds |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for track in _TRACKS:
        row = paper[track]
        lines.append(
            f"| TSB-AD-{track} | {float(row['ours_vus_pr']):.6f} | "
            f"{float(row['paper_vus_pr']):.4f} | {float(row['delta_vus_pr']):+.6f} | "
            f"{'Yes' if row['exceeds_paper_reported'] == 'True' else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Runtime and peak VRAM",
            "",
            "Runtime is reported separately from protocol alignment. Totals are sums over files; means are per file; peak VRAM is the maximum observed within each registered arm/track group.",
            "",
            "| Arm | Track | Train total (s) | Train mean (s) | Score total (s) | Score mean (s) | Train peak (MiB) | Score peak (MiB) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for trajectory in REGISTERED_TRAJECTORIES:
        for track in _TRACKS:
            row = runtime[(trajectory, track)]
            lines.append(
                f"| `{_display_arm(trajectory)}` | {track} | "
                f"{float(row['training_runtime_seconds_sum']):.3f} | "
                f"{float(row['training_runtime_seconds_mean']):.3f} | "
                f"{float(row['scoring_runtime_seconds_sum']):.3f} | "
                f"{float(row['scoring_runtime_seconds_mean']):.3f} | "
                f"{float(row['training_peak_vram_mib_max']):.3f} | "
                f"{float(row['scoring_peak_vram_mib_max']):.3f} |"
            )

    passed = bool(decision["both_tracks_exceed"])
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            f"**Outcome: `{decision['outcome']}`.**",
            "",
        ]
    )
    if passed:
        lines.append(
            "Both tracks strictly exceed their fixed paper-reported VUS-PR references at seed 2027. The frozen protocol therefore authorizes only the preregistered main-arm confirmation seeds 2028 and 2029; the component ablations remain seed 2027."
        )
    else:
        failed = [
            track for track in _TRACKS if paper[track]["exceeds_paper_reported"] == "False"
        ]
        lines.append(
            "The full arm does not strictly exceed the fixed paper-reported VUS-PR reference on "
            f"{', '.join('TSB-AD-' + track for track in failed)}. The frozen protocol stops without confirmation seeds or a post-hoc variant."
        )

    lines.extend(
        [
            "",
            "## Six-file K0 negative caveat",
            "",
            "The earlier six-file same-code K0 established objective inactivity and early checkpointing, but the registered execution changes did not pass its matched performance gate. This full-coverage external comparison does not erase that negative result and, by itself, cannot establish the proposed causal mechanism.",
            "",
            "## Compact provenance",
            "",
            f"- Frozen config SHA-256: `{decision['config_sha256']}`",
            f"- Frozen PaAno vendor commit: `{decision['vendor_sha']}`",
            f"- Evaluated metric rows: `{decision['metric_count']}` (530 series x 3 arms x 1 seed)",
            "- Inputs: `main_file_metrics.csv`, `main_family_metrics.csv`, `main_track_metrics.csv`, `ablation_track_metrics.csv`, `paper_reference_comparison.csv`, `runtime_summary.csv`, and `decision.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def render_full_benchmark_report(
    artifacts_dir: Path, output: Path, git_sha: str | None = None
) -> Path:
    """Validate all compact aggregates and atomically render the English report."""

    if git_sha is not None:
        git_sha = git_sha.strip()
        if not _GIT_SHA.fullmatch(git_sha):
            raise ValueError("git_sha must be a 7-64 character hexadecimal Git object ID")
    data = _load_and_validate(Path(artifacts_dir))
    destination = Path(output)
    _atomic_write_text(destination, _render_markdown(data, git_sha))
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-sha")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = render_full_benchmark_report(
        args.artifacts_dir, args.output, git_sha=args.git_sha
    )
    print(
        "FULL_BENCHMARK_REPORT_COMPLETE "
        f"output={output} coverage=530 arms=3 checkpoint=LAST"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
