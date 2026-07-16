from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import pytest

import paano_k0.claim_gate as module


CONFIG_SHA = "c" * 64
VENDOR_SHA = "d" * 40
PAPER_SOURCE = "PaAno Table 15 default full-Eval (k=3)"
REFERENCES = {"U": 0.5296, "M": 0.4263}
TRACK_FILES = {"U": 350, "M": 180}
TRAJECTORIES = ("PAPERNEG_NONOVERLAP", "PAPERNEG", "OFFICIAL")
MAIN_TRAJECTORY = TRAJECTORIES[0]
ARM = f"{MAIN_TRAJECTORY}_LAST"
SEEDS = (2027, 2028, 2029)
METRICS = ("vus_pr", "auprc", "vus_roc", "auroc")

TRACK_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "seed",
    "files",
    "families",
    *METRICS,
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
SEED_TRACK_FIELDS = TRACK_FIELDS
TRACK_SUMMARY_FIELDS = (
    "trajectory",
    "checkpoint",
    "arm",
    "track",
    "seeds",
    "seed_count",
    "files_per_seed",
    "families",
    *(f"{name}_{stat}" for name in METRICS for stat in ("mean", "std")),
    "std_ddof",
    "paper_method",
    "paper_vus_pr",
    "paper_reference_source",
    "comparison_type",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _replace_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields, _ = _read_csv(path)
    _write_csv(path, tuple(fields), rows)


def _metric_values(vus_pr: float) -> dict[str, float]:
    return {
        "vus_pr": vus_pr,
        "auprc": max(0.0, vus_pr - 0.08),
        "vus_roc": min(1.0, vus_pr + 0.20),
        "auroc": min(1.0, vus_pr + 0.15),
    }


def _track_row(trajectory: str, track: str, vus_pr: float) -> dict[str, object]:
    return {
        "trajectory": trajectory,
        "checkpoint": "LAST",
        "arm": f"{trajectory}_LAST",
        "track": track,
        "seed": 2027,
        "files": TRACK_FILES[track],
        "families": 2,
        **_metric_values(vus_pr),
    }


def _paper_row(track: str, ours: float) -> dict[str, object]:
    paper = REFERENCES[track]
    return {
        "track": track,
        "files": TRACK_FILES[track],
        "ours_method": ARM,
        "ours_vus_pr": ours,
        "paper_method": "PaAno (paper-reported)",
        "paper_vus_pr": paper,
        "paper_reference_source": PAPER_SOURCE,
        "delta_vus_pr": ours - paper,
        "exceeds_paper_reported": ours > paper,
        "comparison_type": "external_paper_reported",
    }


def _write_seed_compact(
    root: Path,
    *,
    main: dict[str, float] | None = None,
    paperneg: dict[str, float] | None = None,
    official: dict[str, float] | None = None,
) -> None:
    main = dict(main or {"U": 0.52, "M": 0.42})
    paperneg = dict(
        paperneg or {track: value - 0.03 for track, value in main.items()}
    )
    official = dict(
        official or {track: value - 0.04 for track, value in main.items()}
    )

    main_rows = [_track_row(MAIN_TRAJECTORY, track, main[track]) for track in ("U", "M")]
    ablation_rows = [
        _track_row(trajectory, track, values[track])
        for trajectory, values in (
            (MAIN_TRAJECTORY, main),
            ("PAPERNEG", paperneg),
            ("OFFICIAL", official),
        )
        for track in ("U", "M")
    ]
    paper_rows = [_paper_row(track, main[track]) for track in ("U", "M")]
    both = all(main[track] > REFERENCES[track] for track in ("U", "M"))
    decision_tracks = {row["track"]: row for row in paper_rows}
    decision = {
        "schema_version": "paano-full-benchmark-decision-v1",
        "outcome": (
            "CONTINUE_FULL_CONFIRMATION" if both else "STOP_FULL_MAIN_FAILURE"
        ),
        "main_trajectory": MAIN_TRAJECTORY,
        "checkpoint": "LAST",
        "paper_reference_type": "external_paper_reported",
        "paper_reference_source": PAPER_SOURCE,
        "paper_reported_vus_pr": dict(REFERENCES),
        "success_requires_both_tracks": True,
        "both_tracks_exceed": both,
        "conditional_confirmation_seeds": [2028, 2029] if both else [],
        "tracks": decision_tracks,
        "missing_count": 0,
        "seed": 2027,
        "series_count": 530,
        "metric_count": 1590,
        "config_sha256": CONFIG_SHA,
        "vendor_sha": VENDOR_SHA,
    }
    _write_csv(root / "main_track_metrics.csv", TRACK_FIELDS, main_rows)
    _write_csv(root / "ablation_track_metrics.csv", TRACK_FIELDS, ablation_rows)
    _write_csv(root / "paper_reference_comparison.csv", PAPER_FIELDS, paper_rows)
    _write_json(root / "decision.json", decision)


def _write_confirmation_compact(
    root: Path,
    seed_vus_pr: dict[str, tuple[float, float, float]],
) -> None:
    seed_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(SEEDS):
        for track in ("U", "M"):
            values = _metric_values(seed_vus_pr[track][seed_index])
            seed_rows.append(
                {
                    "trajectory": MAIN_TRAJECTORY,
                    "checkpoint": "LAST",
                    "arm": ARM,
                    "track": track,
                    "seed": seed,
                    "files": TRACK_FILES[track],
                    "families": 2,
                    **values,
                }
            )

    summary_rows: list[dict[str, object]] = []
    for track in ("U", "M"):
        subset = [row for row in seed_rows if row["track"] == track]
        aggregate_metrics: dict[str, float] = {}
        for name in METRICS:
            values = [float(row[name]) for row in subset]
            aggregate_metrics[f"{name}_mean"] = statistics.fmean(values)
            aggregate_metrics[f"{name}_std"] = statistics.pstdev(values)
        summary_rows.append(
            {
                "trajectory": MAIN_TRAJECTORY,
                "checkpoint": "LAST",
                "arm": ARM,
                "track": track,
                "seeds": "2027;2028;2029",
                "seed_count": 3,
                "files_per_seed": TRACK_FILES[track],
                "families": 2,
                **aggregate_metrics,
                "std_ddof": 0,
                "paper_method": "PaAno (paper-reported)",
                "paper_vus_pr": REFERENCES[track],
                "paper_reference_source": PAPER_SOURCE,
                "comparison_type": "descriptive_external_paper_reported",
            }
        )

    summary = {
        "schema_version": "paano-full-confirmation-v1",
        "trajectory": MAIN_TRAJECTORY,
        "checkpoint": "LAST",
        "arm": ARM,
        "seeds": list(SEEDS),
        "series_per_seed": 530,
        "track_files_per_seed": dict(TRACK_FILES),
        "metric_count": 1590,
        "seed_track_row_count": 6,
        "std_ddof": 0,
        "selection_applied": False,
        "retuning_applied": False,
        "result_dropping_applied": False,
        "paper_reference_type": "external_paper_reported_descriptive_only",
        "paper_reference_source": PAPER_SOURCE,
        "paper_reported_vus_pr": dict(REFERENCES),
        "config_sha256": CONFIG_SHA,
        "vendor_sha": VENDOR_SHA,
        "seed_track_metrics": seed_rows,
        "track_summary": summary_rows,
    }
    _write_csv(
        root / "confirmation_seed_track_metrics.csv",
        SEED_TRACK_FIELDS,
        seed_rows,
    )
    _write_csv(
        root / "confirmation_track_summary.csv",
        TRACK_SUMMARY_FIELDS,
        summary_rows,
    )
    _write_json(root / "confirmation_summary.json", summary)


def _confirmation_fixture(
    root: Path,
    seed_vus_pr: dict[str, tuple[float, float, float]],
) -> None:
    _write_seed_compact(
        root,
        main={track: values[0] for track, values in seed_vus_pr.items()},
    )
    _write_confirmation_compact(root, seed_vus_pr)


def _evaluate(root: Path, stage: str) -> tuple[dict[str, object], Path]:
    output = root.parent / f"claim-{stage}.json"
    result = module.evaluate_claim_gate(root, output, stage)
    assert result == _read_json(output)
    return result, output


def test_seed_stop_one_track_external_and_strict_component_branches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root, main={"U": 0.55, "M": 0.42})

    result, _ = _evaluate(root, "seed-2027")

    assert result["external_branch"] == "SEED2027_ONE_TRACK_STRICTLY_EXCEEDS"
    assert result["component_branch"] == "COMPONENT_EVIDENCE_CAUTIOUSLY_ALLOWED"
    assert result["confirmation_branch"] is None


def test_seed_stop_no_track_external_branch(tmp_path: Path) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root, main={"U": 0.52, "M": 0.42})

    result, _ = _evaluate(root, "seed-2027")

    assert result["external_branch"] == "SEED2027_NO_TRACK_STRICTLY_EXCEEDS"
    assert result["confirmation_branch"] is None


def test_seed_stage_rejects_a_positive_two_track_experiment_decision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root, main={"U": 0.55, "M": 0.45})

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "seed-2027")


@pytest.mark.parametrize(
    ("paperneg", "official"),
    (
        ({"U": 0.52, "M": 0.42}, {"U": 0.48, "M": 0.38}),
        ({"U": 0.49, "M": 0.39}, {"U": 0.48, "M": 0.43}),
    ),
    ids=("tie", "dominated"),
)
def test_component_tie_or_dominance_blocks_attribution(
    tmp_path: Path,
    paperneg: dict[str, float],
    official: dict[str, float],
) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(
        root,
        main={"U": 0.52, "M": 0.42},
        paperneg=paperneg,
        official=official,
    )

    result, _ = _evaluate(root, "seed-2027")

    assert (
        result["component_branch"]
        == "COMPONENT_ATTRIBUTION_BLOCKED_MIXED_OR_DOMINATED"
    )


@pytest.mark.parametrize(
    ("values", "expected_branch"),
    (
        (
            {"U": (0.55, 0.56, 0.57), "M": (0.45, 0.46, 0.47)},
            "STABLE_EXTERNAL_EXCEEDANCE",
        ),
        (
            {"U": (0.55, 0.50, 0.55), "M": (0.45, 0.40, 0.44)},
            "MEAN_PASS_WITH_SEED_INSTABILITY",
        ),
        (
            {"U": (0.55, 0.50, 0.50), "M": (0.45, 0.44, 0.44)},
            "MEAN_FAILURE",
        ),
    ),
    ids=("stable", "mean-pass-unstable", "mean-failure"),
)
def test_confirmation_branch_is_derived_from_all_fixed_seeds(
    tmp_path: Path,
    values: dict[str, tuple[float, float, float]],
    expected_branch: str,
) -> None:
    root = tmp_path / "compact"
    _confirmation_fixture(root, values)

    result, _ = _evaluate(root, "confirmation")

    assert result["external_branch"] == "SEED2027_BOTH_TRACKS_STRICTLY_EXCEED"
    assert result["confirmation_branch"] == expected_branch


@pytest.mark.parametrize(
    "missing_name",
    (
        "confirmation_seed_track_metrics.csv",
        "confirmation_track_summary.csv",
        "confirmation_summary.json",
    ),
)
def test_partial_confirmation_package_is_rejected(
    tmp_path: Path,
    missing_name: str,
) -> None:
    root = tmp_path / "compact"
    _confirmation_fixture(
        root,
        {"U": (0.55, 0.56, 0.57), "M": (0.45, 0.46, 0.47)},
    )
    (root / missing_name).unlink()
    output = tmp_path / "claim.json"

    with pytest.raises((FileNotFoundError, ValueError)):
        module.evaluate_claim_gate(root, output, "confirmation")
    assert not output.exists()


@pytest.mark.parametrize("mutation", ("wrong-version", "missing-key", "extra-key"))
def test_decision_schema_mismatch_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root)
    path = root / "decision.json"
    payload = _read_json(path)
    if mutation == "wrong-version":
        payload["schema_version"] = "paano-full-benchmark-decision-v0"
    elif mutation == "missing-key":
        payload.pop("missing_count")
    else:
        payload["unexpected"] = True
    _write_json(path, payload)

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "seed-2027")


@pytest.mark.parametrize("mutation", ("wrong-version", "missing-key", "extra-key"))
def test_confirmation_summary_schema_mismatch_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "compact"
    _confirmation_fixture(
        root,
        {"U": (0.55, 0.56, 0.57), "M": (0.45, 0.46, 0.47)},
    )
    path = root / "confirmation_summary.json"
    payload = _read_json(path)
    if mutation == "wrong-version":
        payload["schema_version"] = "paano-full-confirmation-v0"
    elif mutation == "missing-key":
        payload.pop("selection_applied")
    else:
        payload["outcome"] = "forbidden"
    _write_json(path, payload)

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "confirmation")


def test_exact_csv_header_schema_is_required(tmp_path: Path) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root)
    path = root / "main_track_metrics.csv"
    fields, rows = _read_csv(path)
    fields[-1] = "unexpected_metric"
    for row in rows:
        row["unexpected_metric"] = row.pop("auroc")
    _write_csv(path, tuple(fields), rows)

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "seed-2027")


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root)
    path = root / "decision.json"
    original = path.read_text(encoding="utf-8").rstrip()
    assert original.endswith("}")
    path.write_text(
        original[:-1] + ',\n  "schema_version": "paano-full-benchmark-decision-v1"\n}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "seed-2027")


@pytest.mark.parametrize("target", ("decision", "paper-csv", "confirmation"))
def test_reference_value_or_source_mismatch_is_rejected(
    tmp_path: Path,
    target: str,
) -> None:
    root = tmp_path / "compact"
    _confirmation_fixture(
        root,
        {"U": (0.55, 0.56, 0.57), "M": (0.45, 0.46, 0.47)},
    )
    if target == "decision":
        path = root / "decision.json"
        payload = _read_json(path)
        assert isinstance(payload["paper_reported_vus_pr"], dict)
        payload["paper_reported_vus_pr"]["M"] = 0.431  # type: ignore[index]
        _write_json(path, payload)
    elif target == "paper-csv":
        path = root / "paper_reference_comparison.csv"
        fields, rows = _read_csv(path)
        rows[1]["paper_reference_source"] = "unregistered source"
        _write_csv(path, tuple(fields), rows)
    else:
        path = root / "confirmation_summary.json"
        payload = _read_json(path)
        payload["paper_reference_source"] = "unregistered source"
        _write_json(path, payload)

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "confirmation")


@pytest.mark.parametrize("field", ("config_sha256", "vendor_sha"))
def test_confirmation_provenance_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    root = tmp_path / "compact"
    _confirmation_fixture(
        root,
        {"U": (0.55, 0.56, 0.57), "M": (0.45, 0.46, 0.47)},
    )
    path = root / "confirmation_summary.json"
    payload = _read_json(path)
    payload[field] = ("e" * 64) if field == "config_sha256" else ("f" * 40)
    _write_json(path, payload)

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "confirmation")


@pytest.mark.parametrize("field", ("vus_pr_mean", "vus_pr_std"))
def test_confirmation_mean_or_population_std_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    root = tmp_path / "compact"
    _confirmation_fixture(
        root,
        {"U": (0.55, 0.56, 0.57), "M": (0.45, 0.46, 0.47)},
    )
    path = root / "confirmation_track_summary.csv"
    fields, rows = _read_csv(path)
    rows[0][field] = str(float(rows[0][field]) + 0.01)
    _write_csv(path, tuple(fields), rows)
    summary_path = root / "confirmation_summary.json"
    summary = _read_json(summary_path)
    assert isinstance(summary["track_summary"], list)
    assert isinstance(summary["track_summary"][0], dict)  # type: ignore[index]
    summary["track_summary"][0][field] = float(rows[0][field])  # type: ignore[index]
    _write_json(summary_path, summary)

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "confirmation")


def test_confirmation_seed_2027_must_match_frozen_main_track_row(
    tmp_path: Path,
) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root, main={"U": 0.55, "M": 0.45})
    _write_confirmation_compact(
        root,
        {"U": (0.56, 0.57, 0.58), "M": (0.46, 0.47, 0.48)},
    )

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "confirmation")


@pytest.mark.parametrize("bad", ("nan", "inf", "-0.01", "1.01"))
def test_nonfinite_or_out_of_range_metric_is_rejected(
    tmp_path: Path,
    bad: str,
) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root)
    path = root / "main_track_metrics.csv"
    fields, rows = _read_csv(path)
    rows[0]["vus_pr"] = bad
    _write_csv(path, tuple(fields), rows)

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, tmp_path / "claim.json", "seed-2027")


def test_validation_failure_preserves_existing_output(tmp_path: Path) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root)
    decision_path = root / "decision.json"
    decision = _read_json(decision_path)
    decision["schema_version"] = "invalid"
    _write_json(decision_path, decision)
    output = tmp_path / "claim.json"
    sentinel = b'{"prior":"preserve"}\n'
    output.write_bytes(sentinel)

    with pytest.raises(ValueError):
        module.evaluate_claim_gate(root, output, "seed-2027")

    assert output.read_bytes() == sentinel


def test_atomic_replace_failure_preserves_existing_output_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "compact"
    _write_seed_compact(root)
    output = tmp_path / "claim.json"
    sentinel = b'{"prior":"preserve"}\n'
    output.write_bytes(sentinel)

    def _fail_replace(_source: object, _destination: object) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(module.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        module.evaluate_claim_gate(root, output, "seed-2027")

    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(".*.tmp"))


def test_input_hashes_cover_exact_consumed_compact_files_and_output_is_deterministic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "compact"
    _confirmation_fixture(
        root,
        {"U": (0.55, 0.56, 0.57), "M": (0.45, 0.46, 0.47)},
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    result = module.evaluate_claim_gate(root, first, "confirmation")
    module.evaluate_claim_gate(root, second, "confirmation")

    expected_names = {
        "decision.json",
        "main_track_metrics.csv",
        "ablation_track_metrics.csv",
        "paper_reference_comparison.csv",
        "confirmation_seed_track_metrics.csv",
        "confirmation_track_summary.csv",
        "confirmation_summary.json",
    }
    expected_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in expected_names
    }
    assert result["input_sha256"] == expected_hashes
    assert first.read_bytes() == second.read_bytes()


def test_claim_gate_implementation_imports_only_python_standard_library() -> None:
    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported: set[str] = set()
    relative: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative.append(node.module or "")
            elif node.module:
                imported.add(node.module.partition(".")[0])

    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    assert relative == []
    assert imported <= stdlib, f"non-stdlib imports: {sorted(imported - stdlib)}"
