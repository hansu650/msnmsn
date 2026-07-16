from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import pytest

import paano_k0.report_benchmark as module


FILE_FIELDS = (
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
FAMILY_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "family",
    "seed",
    "files",
    "vus_pr",
    "auprc",
    "vus_roc",
    "auroc",
)
TRACK_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "seed",
    "files",
    "families",
    "vus_pr",
    "auprc",
    "vus_roc",
    "auroc",
)
PAPER_FIELDS = (
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
RUNTIME_FIELDS = (
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
CONFIG_SHA = "c" * 64
VENDOR_SHA = "d" * 40
TRAJECTORIES = ("PAPERNEG_NONOVERLAP", "PAPERNEG", "OFFICIAL")
COUNTS = {"U": 350, "M": 180}
VALUES = {
    ("PAPERNEG_NONOVERLAP", "U"): (0.54, 0.48, 0.89, 0.87),
    ("PAPERNEG_NONOVERLAP", "M"): (0.43, 0.38, 0.79, 0.76),
    ("PAPERNEG", "U"): (0.51, 0.46, 0.86, 0.84),
    ("PAPERNEG", "M"): (0.40, 0.35, 0.76, 0.73),
    ("OFFICIAL", "U"): (0.50, 0.45, 0.85, 0.83),
    ("OFFICIAL", "M"): (0.39, 0.34, 0.75, 0.72),
}


def _synthetic_membership_sha256() -> str:
    family_names = {"U": ("NAB", "IOPS"), "M": ("SMD", "SMAP")}
    lines: list[str] = []
    for track, count in COUNTS.items():
        for index in range(count):
            series_id = f"{track.lower()}-{index:03d}"
            family = family_names[track][index % 2]
            data_sha256 = hashlib.sha256(series_id.encode("ascii")).hexdigest()
            lines.append(
                "\t".join((track, family, series_id, data_sha256))
            )
    payload = ("\n".join(sorted(lines)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(autouse=True)
def _register_synthetic_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "CANONICAL_MANIFEST_SERIES_SHA256",
        _synthetic_membership_sha256(),
    )


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric_payload(trajectory: str, track: str) -> dict[str, float]:
    vus_pr, auprc, vus_roc, auroc = VALUES[(trajectory, track)]
    return {
        "vus_pr": vus_pr,
        "auprc": auprc,
        "vus_roc": vus_roc,
        "auroc": auroc,
    }


def _write_valid_compact(root: Path) -> None:
    main_files: list[dict[str, object]] = []
    family_names = {"U": ("NAB", "IOPS"), "M": ("SMD", "SMAP")}
    for track, count in COUNTS.items():
        for index in range(count):
            series_id = f"{track.lower()}-{index:03d}"
            digest = hashlib.sha256(series_id.encode("ascii")).hexdigest()
            family = family_names[track][index % 2]
            main_files.append(
                {
                    "run_id": (
                        f"{series_id}__seed_2027__"
                        "PAPERNEG_NONOVERLAP__LAST"
                    ),
                    "series_id": series_id,
                    "family": family,
                    "track": track,
                    "seed": 2027,
                    "trajectory": "PAPERNEG_NONOVERLAP",
                    "checkpoint": "LAST",
                    "arm": "PAPERNEG_NONOVERLAP_LAST",
                    **_metric_payload("PAPERNEG_NONOVERLAP", track),
                    "score_sha256": digest,
                    "data_sha256": digest,
                    "config_sha256": CONFIG_SHA,
                    "vendor_sha": VENDOR_SHA,
                }
            )
    _write_csv(root / "main_file_metrics.csv", FILE_FIELDS, main_files)

    family_rows: list[dict[str, object]] = []
    for track, names in family_names.items():
        for family in names:
            files = sum(
                row["track"] == track and row["family"] == family
                for row in main_files
            )
            family_rows.append(
                {
                    "trajectory": "PAPERNEG_NONOVERLAP",
                    "checkpoint": "LAST",
                    "arm": "PAPERNEG_NONOVERLAP_LAST",
                    "track": track,
                    "family": family,
                    "seed": 2027,
                    "files": files,
                    **_metric_payload("PAPERNEG_NONOVERLAP", track),
                }
            )
    _write_csv(root / "main_family_metrics.csv", FAMILY_FIELDS, family_rows)

    main_tracks: list[dict[str, object]] = []
    for track, count in COUNTS.items():
        main_tracks.append(
            {
                "trajectory": "PAPERNEG_NONOVERLAP",
                "checkpoint": "LAST",
                "arm": "PAPERNEG_NONOVERLAP_LAST",
                "track": track,
                "seed": 2027,
                "files": count,
                "families": 2,
                **_metric_payload("PAPERNEG_NONOVERLAP", track),
            }
        )
    _write_csv(root / "main_track_metrics.csv", TRACK_FIELDS, main_tracks)

    ablation_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for trajectory_index, trajectory in enumerate(TRAJECTORIES):
        for track, count in COUNTS.items():
            ablation_rows.append(
                {
                    "trajectory": trajectory,
                    "checkpoint": "LAST",
                    "arm": f"{trajectory}_LAST",
                    "track": track,
                    "seed": 2027,
                    "files": count,
                    "families": 2,
                    **_metric_payload(trajectory, track),
                }
            )
            training_mean = 1.25 + trajectory_index
            scoring_mean = 0.20 + trajectory_index * 0.05
            runtime_rows.append(
                {
                    "trajectory": trajectory,
                    "checkpoint": "LAST",
                    "track": track,
                    "seed": 2027,
                    "files": count,
                    "training_runtime_seconds_sum": training_mean * count,
                    "training_runtime_seconds_mean": training_mean,
                    "scoring_runtime_seconds_sum": scoring_mean * count,
                    "scoring_runtime_seconds_mean": scoring_mean,
                    "training_peak_vram_mib_max": 512.0 + trajectory_index,
                    "scoring_peak_vram_mib_max": 128.0 + trajectory_index,
                }
            )
    _write_csv(root / "ablation_track_metrics.csv", TRACK_FIELDS, ablation_rows)
    _write_csv(root / "runtime_summary.csv", RUNTIME_FIELDS, runtime_rows)

    paper_rows: list[dict[str, object]] = []
    references = {"U": 0.5296, "M": 0.4263}
    for track, count in COUNTS.items():
        ours = VALUES[("PAPERNEG_NONOVERLAP", track)][0]
        paper_rows.append(
            {
                "track": track,
                "files": count,
                "ours_method": "PAPERNEG_NONOVERLAP_LAST",
                "ours_vus_pr": ours,
                "paper_method": "PaAno (paper-reported)",
                "paper_vus_pr": references[track],
                "paper_reference_source": module.PAPER_REFERENCE_SOURCE,
                "delta_vus_pr": ours - references[track],
                "exceeds_paper_reported": True,
                "comparison_type": "external_paper_reported",
            }
        )
    _write_csv(root / "paper_reference_comparison.csv", PAPER_FIELDS, paper_rows)

    decision_tracks = {
        row["track"]: {
            key: value for key, value in row.items()
        }
        for row in paper_rows
    }
    decision = {
        "schema_version": "paano-full-benchmark-decision-v1",
        "outcome": "CONTINUE_FULL_CONFIRMATION",
        "main_trajectory": "PAPERNEG_NONOVERLAP",
        "checkpoint": "LAST",
        "paper_reference_type": "external_paper_reported",
        "paper_reference_source": module.PAPER_REFERENCE_SOURCE,
        "paper_reported_vus_pr": references,
        "success_requires_both_tracks": True,
        "both_tracks_exceed": True,
        "conditional_confirmation_seeds": [2028, 2029],
        "tracks": decision_tracks,
        "missing_count": 0,
        "seed": 2027,
        "series_count": 530,
        "metric_count": 1590,
        "config_sha256": CONFIG_SHA,
        "vendor_sha": VENDOR_SHA,
    }
    (root / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )


def test_renders_complete_english_report_from_seven_compact_inputs(
    tmp_path: Path,
) -> None:
    compact = tmp_path / "compact"
    compact.mkdir()
    _write_valid_compact(compact)
    output = tmp_path / "docs" / "PAANO_FULL_MAIN_RESULTS.md"

    result = module.render_full_benchmark_report(
        compact, output, git_sha="a1b2c3d4e5f678901234567890abcdef12345678"
    )

    assert result == output
    report = output.read_text(encoding="utf-8")
    assert "TSB-AD-U | 350" in report
    assert "TSB-AD-M | 180" in report
    assert "PAPERNEG_NONOVERLAP-LAST" in report
    assert "PAPERNEG-LAST" in report
    assert "OFFICIAL-LAST" in report
    assert "NAB" in report and "IOPS" in report
    assert "SMD" in report and "SMAP" in report
    assert "0.5296" in report and "0.4263" in report
    assert module.PAPER_REFERENCE_SOURCE in report
    assert "CONTINUE_FULL_CONFIRMATION" in report
    assert "Runtime and peak VRAM" in report
    assert "Six-file K0 negative caveat" in report
    assert "did not pass its matched performance gate" in report
    assert "a1b2c3d4e5f678901234567890abcdef12345678" in report
    assert "![" not in report


def test_rejects_missing_registered_arm_and_preserves_existing_output(
    tmp_path: Path,
) -> None:
    compact = tmp_path / "compact"
    compact.mkdir()
    _write_valid_compact(compact)
    path = compact / "ablation_track_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if (row["trajectory"], row["track"]) != ("OFFICIAL", "M")
    ]
    _write_csv(path, TRACK_FIELDS, rows)
    output = tmp_path / "result.md"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="registered rows changed"):
        module.render_full_benchmark_report(compact, output)

    assert output.read_text(encoding="utf-8") == "existing\n"


def test_rejects_changed_paper_value_or_source(tmp_path: Path) -> None:
    compact = tmp_path / "compact"
    compact.mkdir()
    _write_valid_compact(compact)
    path = compact / "paper_reference_comparison.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["paper_vus_pr"] = "0.431"
    rows[1]["paper_reference_source"] = "wrong table"
    _write_csv(path, PAPER_FIELDS, rows)

    with pytest.raises(ValueError, match="paper value"):
        module.render_full_benchmark_report(compact, tmp_path / "result.md")


def test_atomic_replace_failure_leaves_previous_report_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compact = tmp_path / "compact"
    compact.mkdir()
    _write_valid_compact(compact)
    output = tmp_path / "result.md"
    output.write_text("previous\n", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        module.render_full_benchmark_report(compact, output)

    assert output.read_text(encoding="utf-8") == "previous\n"
    assert not tuple(tmp_path.glob(".result.md.*.tmp"))


def test_report_rejects_noncanonical_series_membership(tmp_path: Path) -> None:
    compact = tmp_path / "compact"
    compact.mkdir()
    _write_valid_compact(compact)
    path = compact / "main_file_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["data_sha256"] = "e" * 64
    _write_csv(path, FILE_FIELDS, rows)

    with pytest.raises(ValueError, match="canonical manifest membership"):
        module.render_full_benchmark_report(compact, tmp_path / "result.md")


def test_report_module_has_no_project_data_or_evaluator_imports() -> None:
    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_imports: list[str] = []
    allowed_roots = {
        "argparse",
        "csv",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "typing",
        "uuid",
        "__future__",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_roots:
                    forbidden_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                forbidden_imports.append(f"relative:{node.module or ''}")
            else:
                root = (node.module or "").split(".", 1)[0]
                if root not in allowed_roots:
                    forbidden_imports.append(node.module or "")
    assert forbidden_imports == []
