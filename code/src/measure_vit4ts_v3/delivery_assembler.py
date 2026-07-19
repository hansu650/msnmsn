"""Assemble and strictly verify the compact mandatory ViTTrace v3 delivery.

Only compact evidence is copied.  Raw datasets, model weights, score arrays,
token arrays, runner logs, and active run directories are never included.
Incomplete mandatory stages remain visible as ``BLOCKED`` in ``STATUS.md``;
the final ZIP is emitted only when every scientific and delivery requirement
has authoritative completion evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .cross_stage_outputs import (
    ARM_METADATA_COLUMNS,
    assemble_cross_stage_outputs,
    load_stage_index,
    sha256_file,
    stage_status_frame,
)
from .result_package import build_result_package, collect_package_payload
from .rough_figure_outputs import render_rough_figure_set
from .structural_audit import structural_audit_frame


SCHEMA_VERSION = 1
COMPACT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".yaml",
    ".yml",
    ".txt",
    ".svg",
    ".pdf",
    ".py",
    ".ps1",
    ".toml",
}
FORBIDDEN_COPY_PARTS = {
    "runs",
    "logs",
    "tmp",
    "datasets",
    "data",
    "models",
    "weights",
    "__pycache__",
    ".pytest_cache",
    ".git",
}
LOCAL_REQUIREMENTS = (
    "cross_stage_metrics",
    "bootstrap_10000",
    "qualitative_four_cases",
    "microbenchmark_5w30r",
    "runtime_encoder_inclusive",
    "rough_figures_11",
    "external_reference",
    "failure_manifest",
    "cache_index",
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _copy_file(source: Path, destination: Path) -> None:
    source = Path(source).resolve(strict=True)
    if source.suffix.lower() not in COMPACT_SUFFIXES:
        raise ValueError(f"non-compact delivery source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_compact_tree(source: Path, destination: Path) -> None:
    source = Path(source)
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part.lower() in FORBIDDEN_COPY_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in COMPACT_SUFFIXES or path.stat().st_size > 128 * 1024 * 1024:
            continue
        _copy_file(path, destination / relative)


def build_failure_manifest(failure_roots: Iterable[Path], output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in map(Path, failure_roots):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as error:  # malformed failure evidence remains visible
                payload = {"error_type": type(error).__name__, "error": str(error)}
            rows.append(
                {
                    "failure_path": str(path.resolve()),
                    "failure_sha256": sha256_file(path),
                    "stage": str(payload.get("stage", path.parent.name)),
                    "series_id": str(payload.get("series_id", "")),
                    "arm": str(payload.get("arm", "")),
                    "status": str(payload.get("status", "FAILED")),
                    "error_type": str(payload.get("error_type", "")),
                    "reason": str(payload.get("reason", payload.get("error", "")))[:2000],
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=(
            "failure_path",
            "failure_sha256",
            "stage",
            "series_id",
            "arm",
            "status",
            "error_type",
            "reason",
        ),
    )
    if not frame.empty:
        frame = frame.sort_values(["stage", "series_id", "arm", "failure_path"]).reset_index(drop=True)
    _atomic_csv(Path(output_root) / "failure_manifest.csv", frame)
    return frame


def _cache_manifest_rows(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = {"vision_tokens_v3.json", "clip_tokens.json", "qualitative_patch_fields.json"}
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            if path.name not in names:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                key = payload.get("key", {}) if isinstance(payload, Mapping) else {}
                series_id = str(payload.get("series_id", key.get("series_id", path.parent.parent.name)))
                file_name = str(payload.get("file", ""))
                if path.name == "qualitative_patch_fields.json":
                    file_name = "qualitative_patch_fields.npz"
                    declared = str(payload.get("field_payload_sha256", ""))
                else:
                    declared = str(payload.get("sha256", ""))
                data_path = path.parent / file_name if file_name else None
                rows.append(
                    {
                        "cache_type": path.stem,
                        "series_id": series_id,
                        "variant": str(key.get("renderer", key.get("model_name", ""))),
                        "manifest_path": str(path.resolve()),
                        "manifest_sha256": sha256_file(path),
                        "payload_path": str(data_path.resolve()) if data_path and data_path.exists() else "",
                        "payload_sha256_declared": declared.upper(),
                        "payload_size_bytes": int(data_path.stat().st_size) if data_path and data_path.exists() else 0,
                        "payload_present": bool(data_path and data_path.is_file()),
                        "hash_binding": "RUNNER_MANIFEST_DECLARED",
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "cache_type": path.stem,
                        "series_id": "",
                        "variant": "",
                        "manifest_path": str(path.resolve()),
                        "manifest_sha256": sha256_file(path),
                        "payload_path": "",
                        "payload_sha256_declared": "",
                        "payload_size_bytes": 0,
                        "payload_present": False,
                        "hash_binding": f"INVALID_MANIFEST:{type(error).__name__}:{error}",
                    }
                )
    return rows


def build_cache_index(cache_roots: Iterable[Path], output_path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        _cache_manifest_rows(cache_roots),
        columns=(
            "cache_type",
            "series_id",
            "variant",
            "manifest_path",
            "manifest_sha256",
            "payload_path",
            "payload_sha256_declared",
            "payload_size_bytes",
            "payload_present",
            "hash_binding",
        ),
    )
    if not frame.empty:
        frame = frame.sort_values(["cache_type", "variant", "series_id", "manifest_path"]).reset_index(drop=True)
    _atomic_csv(output_path, frame)
    return frame


def combined_arm_registry(stage_index_path: Path) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    protocol_sha, stages = load_stage_index(stage_index_path)
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []
    for stage in stages:
        if stage.status != "COMPLETE" or stage.arm_metadata_path is None:
            continue
        if sha256_file(stage.arm_metadata_path) != stage.arm_metadata_sha256:
            raise ValueError(f"stale arm metadata for {stage.stage_id}")
        frame = pd.read_csv(stage.arm_metadata_path)
        if "arm" not in frame:
            raise ValueError(f"arm metadata lacks arm: {stage.stage_id}")
        if "arm_metadata_json" in frame:
            parsed: list[Mapping[str, Any]] = []
            for value in frame["arm_metadata_json"]:
                item = json.loads(str(value)) if pd.notna(value) else {}
                if not isinstance(item, Mapping):
                    raise ValueError(f"arm metadata JSON must contain objects: {stage.stage_id}")
                parsed.append(item)
            for column in ("display_name", "is_final"):
                if column not in frame:
                    frame[column] = [item.get(column, pd.NA) for item in parsed]
        if "display_name" not in frame:
            frame["display_name"] = frame["arm"]
        if "fixed_factors" not in frame and "fixed_factors_json" in frame:
            frame["fixed_factors"] = frame["fixed_factors_json"]
        if "fixed_factors_json" not in frame and "fixed_factors" in frame:
            frame["fixed_factors_json"] = frame["fixed_factors"]
        for column in ARM_METADATA_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame.loc[:, ARM_METADATA_COLUMNS].copy()
        frame.insert(0, "stage_id", stage.stage_id)
        frame.insert(1, "stage_group", stage.stage_group)
        frame.insert(2, "configuration_id", stage.configuration_id)
        frames.append(frame)
        sources.append(
            {
                "stage_id": stage.stage_id,
                "arm_metadata_path": str(stage.arm_metadata_path),
                "arm_metadata_sha256": stage.arm_metadata_sha256,
            }
        )
    if not frames:
        raise ValueError("no complete arm metadata are available")
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["stage_id", "arm"]).any():
        raise ValueError("combined arm registry has duplicate stage/arm keys")
    return combined, {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "sources": sources,
        "record_count": int(len(combined)),
    }


def _local_status(
    cross_root: Path,
    experiment_root: Path,
    external_reference: Path,
    failure_manifest: pd.DataFrame,
    cache_index: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[tuple[str, bool, str]] = []
    cross_complete = cross_root / "manifests" / "_CROSS_STAGE_COMPLETE.json"
    checks.append(("cross_stage_metrics", cross_complete.is_file(), str(cross_complete)))
    bootstrap = cross_root / "results" / "bootstrap_ci.csv"
    aggregate_marker = cross_root / "results" / "_COMBINED_AGGREGATION_COMPLETE.json"
    bootstrap_ok = False
    bootstrap_reason = "10,000-draw bootstrap output/marker missing"
    if bootstrap.is_file() and aggregate_marker.is_file():
        try:
            aggregate = json.loads(aggregate_marker.read_text(encoding="utf-8-sig"))
            bootstrap_frame = pd.read_csv(bootstrap)
            bootstrap_ok = (
                aggregate.get("status") == "COMPLETE"
                and int(aggregate.get("bootstrap_replicates", -1)) == 10_000
                and str(aggregate.get("bootstrap_sha256", "")).upper() == sha256_file(bootstrap)
                and not bootstrap_frame.empty
            )
            bootstrap_reason = str(bootstrap) if bootstrap_ok else "bootstrap marker/hash/count mismatch"
        except Exception as error:
            bootstrap_reason = f"invalid bootstrap evidence: {type(error).__name__}: {error}"
    checks.append(("bootstrap_10000", bootstrap_ok, bootstrap_reason))
    qualitative = experiment_root / "results" / "qualitative_plot_data" / "_QUALITATIVE_COMPLETE.json"
    checks.append(("qualitative_four_cases", qualitative.is_file(), str(qualitative)))
    runtime_root = experiment_root / "results" / "runtime"
    micro_candidates = list(runtime_root.rglob("_MICROBENCHMARK_COMPLETE.json"))
    checks.append(
        (
            "microbenchmark_5w30r",
            len(micro_candidates) == 1,
            str(micro_candidates[0]) if len(micro_candidates) == 1 else f"markers={len(micro_candidates)}",
        )
    )
    runtime_path = runtime_root / "runtime.csv"
    encoder_ok = False
    encoder_reason = "consolidated encoder-inclusive runtime.csv missing"
    if runtime_path.is_file():
        try:
            runtime_frame = pd.read_csv(runtime_path)
            rows = runtime_frame.loc[
                runtime_frame["measurement_mode"].astype(str).eq("encoder_inclusive")
                & runtime_frame["stage"].astype(str).eq("total")
                & runtime_frame["aggregation"].astype(str).eq("config")
            ]
            encoder_ok = not rows.empty and bool(rows["encoder_calls_max"].gt(0).all())
            encoder_reason = str(runtime_path) if encoder_ok else "encoder-inclusive total rows absent/invalid"
        except Exception as error:
            encoder_reason = f"invalid consolidated runtime: {type(error).__name__}: {error}"
    checks.append(("runtime_encoder_inclusive", encoder_ok, encoder_reason))
    figure_status_candidates = list((cross_root / "plot_data").glob("rough_figure_status.csv"))
    figures_ok = False
    figure_reason = "rough figure status missing"
    if figure_status_candidates:
        figure_frame = pd.read_csv(figure_status_candidates[0])
        figures_ok = len(figure_frame) == 11 and bool((figure_frame["status"] == "COMPLETE").all())
        figure_reason = str(figure_status_candidates[0]) if figures_ok else "not all 11 rough figures PASS"
    checks.append(("rough_figures_11", figures_ok, figure_reason))
    checks.append(("external_reference", external_reference.is_file(), str(external_reference)))
    checks.append(("failure_manifest", list(failure_manifest.columns) != [], f"rows={len(failure_manifest)}"))
    cache_ok = not cache_index.empty and bool(cache_index["payload_present"].all()) and not bool(
        cache_index["hash_binding"].astype(str).str.startswith("INVALID").any()
    )
    checks.append(("cache_index", cache_ok, f"rows={len(cache_index)}" if cache_ok else "cache index empty/invalid"))
    return pd.DataFrame(
        [
            {
                "requirement": name,
                "status": "COMPLETE" if passed else "BLOCKED",
                "evidence_or_reason": reason,
            }
            for name, passed, reason in checks
        ]
    )


def _status_markdown(stage_status: pd.DataFrame, local_status: pd.DataFrame) -> str:
    complete = bool((stage_status["status"] == "COMPLETE").all()) and bool(
        (local_status["status"] == "COMPLETE").all()
    )
    lines = [
        "# ViTTrace v3 Delivery Status",
        "",
        f"**Overall: {'COMPLETE' if complete else 'BLOCKED'}**",
        "",
        "Missing or failed stages are never imputed. A final ZIP is emitted only for COMPLETE status.",
        "",
        "## Scientific stages",
        "",
        "| Stage | Group | Configuration | Status | Evidence / reason |",
        "|---|---|---|---|---|",
    ]
    for row in stage_status.itertuples(index=False):
        evidence = row.reason or row.marker_path or row.metrics_path
        lines.append(
            f"| {row.stage_id} | {row.stage_group} | {row.configuration_id} | {row.status} | {str(evidence).replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Delivery requirements",
            "",
            "| Requirement | Status | Evidence / reason |",
            "|---|---|---|",
        ]
    )
    for row in local_status.itertuples(index=False):
        lines.append(
            f"| {row.requirement} | {row.status} | {str(row.evidence_or_reason).replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def _partial_checksums(root: Path) -> Path:
    rows = collect_package_payload(root, allow_incomplete=True)
    path = root / "provenance" / "PARTIAL_SHA256SUMS.csv"
    frame = pd.DataFrame(rows)
    _atomic_csv(path, frame)
    return path


def assemble_delivery(
    config_path: Path,
    repo_root: Path,
    experiment_root: Path,
    stage_index_path: Path,
    external_reference: Path,
    output_root: Path,
    *,
    zip_path: Path | None = None,
) -> tuple[Path | None, Mapping[str, Any]]:
    """Build a new compact delivery tree and ZIP only when fully complete."""

    config_path = Path(config_path).resolve(strict=True)
    repo_root = Path(repo_root).resolve(strict=True)
    experiment_root = Path(experiment_root).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("delivery output_root must be new or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluation_root = Path(stage_index_path).resolve(strict=True).parent
    cross_root = output_root
    assemble_cross_stage_outputs(stage_index_path, cross_root)

    _copy_file(config_path, output_root / "config" / config_path.name)
    source_root = repo_root / "code" / "src" / "measure_vit4ts_v3"
    test_root = repo_root / "code" / "tests"
    _copy_compact_tree(source_root, output_root / "code" / "measure_vit4ts_v3")
    for test in sorted(test_root.glob("test_vittrace_v3_*.py")):
        # Python source is explicitly allowed for code/tests snapshots.
        destination = output_root / "tests" / test.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(test, destination)
    for script in sorted((repo_root / "code" / "scripts").glob("*vittrace*v3*.ps1")):
        destination = output_root / "code" / "scripts" / script.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, destination)
    _copy_compact_tree(experiment_root / "manifests", output_root / "manifests" / "experiment")
    _copy_compact_tree(experiment_root / "provenance", output_root / "provenance" / "experiment")
    _copy_compact_tree(experiment_root / "results" / "qualitative_plot_data", output_root / "results" / "qualitative_plot_data")
    _copy_compact_tree(experiment_root / "results" / "runtime", output_root / "results" / "runtime")
    for name in ("bootstrap_ci.csv", "_COMBINED_AGGREGATION_COMPLETE.json"):
        source = evaluation_root / name
        if source.is_file():
            _copy_file(source, output_root / "results" / name)
    bootstrap_copy = output_root / "results" / "bootstrap_ci.csv"
    if bootstrap_copy.is_file():
        _atomic_csv(output_root / "results" / "contrasts.csv", pd.read_csv(bootstrap_copy))
    defaults = config["defaults"]
    structural = structural_audit_frame(
        patch_grid=(14, 14), window_length=int(defaults["window"]), image_size=tuple(defaults["image_size"])
    )
    _atomic_csv(output_root / "results" / "structural_audit.csv", structural)

    figure_inputs: dict[str, Path | pd.DataFrame] = {
        name: output_root / "plot_data" / f"{name}.csv"
        for name in (
            "backbone_accuracy_time",
            "window_sensitivity",
            "stride_sensitivity",
            "ihp_nctp_interaction",
            "matching_scope",
            "scale_subset_heatmap",
            "reducer_sensitivity",
            "line_vs_spectrogram",
        )
    }
    runtime_path = output_root / "results" / "runtime" / "runtime.csv"
    if runtime_path.is_file():
        runtime_frame = pd.read_csv(runtime_path)
        if "config_id" not in runtime_frame and "experiment_id" in runtime_frame:
            runtime_frame["config_id"] = runtime_frame["experiment_id"]
        figure_inputs["runtime_memory"] = runtime_frame
    qualitative_root = output_root / "results" / "qualitative_plot_data"
    figure_inputs["qualitative_score_stacks"] = qualitative_root / "score_stacks.csv"
    figure_inputs["structural_mapping_coverage"] = qualitative_root / "nctp_mapping_zoom.csv"
    render_rough_figure_set(
        figure_inputs,
        plot_data_root=output_root / "plot_data",
        figure_root=output_root / "rough_figures",
    )
    if (experiment_root / "EXPERIMENT_LOG.md").is_file():
        _copy_file(experiment_root / "EXPERIMENT_LOG.md", output_root / "EXPERIMENT_LOG.md")
    else:
        _atomic_text(output_root / "EXPERIMENT_LOG.md", "# Experiment Log\n\nBLOCKED: source experiment log missing.\n")

    failure_manifest = build_failure_manifest(
        [experiment_root / "failures"], output_root / "failures"
    )
    cache_index = build_cache_index(
        [
            experiment_root / "encoder_stage",
            experiment_root / "caches" / "qualitative_fields",
            Path(config["frozen_inputs"]["coordinate_cache_root"]),
        ],
        output_root / "caches" / "cache_index.csv",
    )
    registry_frame, registry_payload = combined_arm_registry(stage_index_path)
    _atomic_csv(output_root / "arm_registry.csv", registry_frame)
    _atomic_json(output_root / "arm_registry.json", registry_payload)
    if external_reference.is_file():
        _copy_file(external_reference, output_root / "external_vit4ts_reference.csv")

    _, stages = load_stage_index(stage_index_path)
    stage_status = stage_status_frame(stages)
    local_status = _local_status(
        cross_root,
        experiment_root,
        external_reference,
        failure_manifest,
        cache_index,
    )
    _atomic_csv(output_root / "manifests" / "delivery_stage_status.csv", stage_status)
    _atomic_csv(output_root / "manifests" / "delivery_local_status.csv", local_status)
    _atomic_text(output_root / "STATUS.md", _status_markdown(stage_status, local_status))
    _atomic_text(
        output_root / "README.md",
        "# ViTTrace / IHP-NCTP v3 compact results\n\n"
        "This tree contains verified compact metrics, tables, plot data, provenance, failures, "
        "cache identities, source snapshots, and vector rough figures. It excludes datasets, "
        "model weights, score arrays, and token arrays. See `STATUS.md` before using any result.\n",
    )
    complete = bool((stage_status["status"] == "COMPLETE").all()) and bool(
        (local_status["status"] == "COMPLETE").all()
    )
    if not complete:
        partial = _partial_checksums(output_root)
        marker = output_root / "manifests" / "_DELIVERY_BLOCKED.json"
        _atomic_json(
            marker,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "BLOCKED",
                "stage_status_sha256": sha256_file(output_root / "manifests" / "delivery_stage_status.csv"),
                "local_status_sha256": sha256_file(output_root / "manifests" / "delivery_local_status.csv"),
                "partial_checksums_sha256": sha256_file(partial),
                "zip_emitted": False,
            },
        )
        return None, json.loads(marker.read_text(encoding="utf-8"))

    destination, manifest = build_result_package(
        output_root,
        zip_path=zip_path,
        allow_incomplete=False,
    )
    return destination, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--stage-index", type=Path, required=True)
    parser.add_argument("--external-reference", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args(argv)
    archive, payload = assemble_delivery(
        args.config,
        args.repo_root,
        args.experiment_root,
        args.stage_index,
        args.external_reference,
        args.output_root,
        zip_path=args.zip,
    )
    print(json.dumps({"archive": str(archive or ""), **dict(payload)}, sort_keys=True))
    return 0 if archive is not None else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LOCAL_REQUIREMENTS",
    "SCHEMA_VERSION",
    "assemble_delivery",
    "build_cache_index",
    "build_failure_manifest",
    "combined_arm_registry",
]
