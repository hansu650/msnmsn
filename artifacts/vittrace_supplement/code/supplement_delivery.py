"""Assemble the data-only ViTTrace supplement delivery.

The delivery reuses one immutable 492-series evaluation, committed score
transactions, and the historical cache-only runtime records.  It never runs
an encoder, changes a score, emits a figure, edits a manuscript, or touches
Git state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch

from .cache_registry import sha256_file
from measure_vit4ts.full_manifest import load_manifest
from .result_package import build_result_package, verify_result_zip
from .supplement_qualitative import export_supplement_qualitative
from .supplement_runtime import audit_confirmation_cohort
from .supplement_stats import write_supplement_outputs as write_stats


SCHEMA_VERSION = "vittrace-supplement-data-only/1"
EXPECTED_SERIES = 492
EXPECTED_VALID_SERIES = 488
FACTORIAL_RUNTIME_ARMS = (
    ("REL", "IHP0_NCTP0", "LEGACY_DEFAULT"),
    ("IHP only", "IHP1_NCTP0", "IHP1_NCTP0"),
    ("NCTP only", "IHP0_NCTP1", "IHP0_NCTP1"),
    ("Full", "IHP1_NCTP1", "FINAL_DEFAULT"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root.as_posix()}", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _bool_column(frame: pd.DataFrame, name: str) -> pd.Series:
    values = frame[name]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError(f"{name} is not boolean")
    return mapped.astype(bool)


def _copy_base_package(previous: Path, output: Path) -> None:
    source = previous.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not (source / "PACKAGE_MANIFEST.json").is_file():
        raise ValueError("previous compact package is incomplete")
    shutil.copytree(source, output)
    for stale in ("PACKAGE_MANIFEST.json", "SHA256SUMS.csv"):
        target = output / stale
        if target.exists():
            target.unlink()


def _copy_delivery_sources(repo: Path, output: Path) -> None:
    source_root = repo / "code" / "src" / "measure_vit4ts_v3"
    test_root = repo / "code" / "tests"
    for name in (
        "paper_matched_delivery.py",
        "result_package.py",
        "supplement_stats.py",
        "supplement_qualitative.py",
        "supplement_runtime.py",
        "supplement_delivery.py",
    ):
        shutil.copy2(source_root / name, output / "code" / name)
    for name in (
        "test_vittrace_paper_matched_delivery.py",
        "test_vittrace_v3_result_package.py",
        "test_vittrace_v3_supplement_stats.py",
        "test_vittrace_v3_supplement_qualitative.py",
        "test_vittrace_v3_supplement_runtime.py",
    ):
        shutil.copy2(test_root / name, output / "tests" / name)


def _common_valid_mask(evaluation: Path) -> pd.DataFrame:
    mask = pd.read_csv(evaluation / "valid_series_mask.csv")
    flags = [_bool_column(mask, f"valid_{metric}") for metric in ("f1_max", "auprc", "vus_pr")]
    common = flags[0] & flags[1] & flags[2]
    if not flags[0].equals(flags[1]) or not flags[0].equals(flags[2]):
        raise ValueError("detection metrics do not use one common validity mask")
    selected = mask.loc[common].copy()
    if len(mask) != EXPECTED_SERIES or len(selected) != EXPECTED_VALID_SERIES:
        raise ValueError("supplement requires the frozen 492/488 cohort")
    return selected.sort_values("series_id").reset_index(drop=True)


def _table3_row(evaluation: Path) -> pd.DataFrame:
    family = pd.read_csv(evaluation / "family3_metrics.csv")
    rows = family.loc[(family["arm"] == "FINAL_DEFAULT") & (family["metric"] == "f1_max")]
    values = {str(row.family).upper(): float(row.value) for row in rows.itertuples()}
    if set(values) != {"NAB", "NASA", "YAHOO"}:
        raise ValueError("Table 3 family grid is incomplete")
    return pd.DataFrame(
        [
            {
                "method": "ViTTrace / IHP-240 (ours)",
                "metric": "F1-max",
                "NAB": values["NAB"],
                "NASA": values["NASA"],
                "YAHOO": values["YAHOO"],
                "aggregation": "macro mean of 5/2/4 subdataset F1-max values",
                "paired_ci_available": True,
                "is_final": True,
            }
        ]
    )


def _stage_statistics(evaluation: Path, output: Path) -> dict[str, Path]:
    transaction = output / "results" / "supplement_statistics"
    paths = write_stats(evaluation, transaction)
    contrasts = pd.read_csv(paths[0])
    factorial = pd.read_csv(paths[1])
    deltas = pd.read_csv(paths[2])
    aliases = {
        "bootstrap": output / "results" / "bootstrap_ci_complete.csv",
        "factorial": output / "results" / "factorial_2x2_summary.csv",
        "interaction": output / "results" / "factorial_interaction.csv",
        "deltas": output / "results" / "per_series_paired_deltas.csv",
        "ablation": output / "tables" / "table_ablation_ready.csv",
        "fig4_data": output / "plot_data" / "fig4_plot_data.csv",
        "table3": output / "tables" / "table3_ours_row.csv",
    }
    _atomic_csv(aliases["bootstrap"], contrasts)
    _atomic_csv(
        aliases["interaction"],
        contrasts.loc[contrasts["contrast_id"] == "FACTORIAL_INTERACTION"].reset_index(drop=True),
    )
    _atomic_csv(aliases["factorial"], factorial)
    _atomic_csv(aliases["deltas"], deltas)
    ready = factorial.pivot_table(
        index=["display_name", "arm", "ihp", "nctp", "effective_n", "n_subgroups"],
        columns="metric",
        values="equal11_value",
    ).reset_index()
    ready.columns.name = None
    _atomic_csv(aliases["ablation"], ready)
    _atomic_csv(aliases["fig4_data"], factorial)
    _atomic_csv(aliases["table3"], _table3_row(evaluation))
    return aliases


def _stage_historical_runtime(
    config_path: Path,
    evaluation: Path,
    run_root: Path,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))))
    manifest_path = Path(config["manifest"]["path"]).resolve(strict=True)
    _, records_tuple = load_manifest(manifest_path)
    records = {record.series_id: record for record in records_tuple}
    common = _common_valid_mask(evaluation)
    rows: list[dict[str, Any]] = []
    for mask_row in common.itertuples(index=False):
        series_id = str(mask_row.series_id)
        if series_id not in records:
            raise ValueError(f"runtime series missing from manifest: {series_id}")
        runtime_path = run_root / series_id / "runtime.json"
        success_path = run_root / series_id / "_SUCCESS.json"
        if not runtime_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(f"committed runtime transaction missing: {series_id}")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if runtime.get("series_id") != series_id or int(runtime.get("encoder_calls", -1)) != 0:
            raise ValueError(f"runtime provenance mismatch: {series_id}")
        if int(success.get("encoder_calls", -1)) != 0:
            raise ValueError(f"runtime success marker is not cache-only: {series_id}")
        timings = runtime.get("canonical_arm_seconds")
        if not isinstance(timings, Mapping):
            raise ValueError(f"canonical timing map missing: {series_id}")
        windows = int(records[series_id].expected_windows)
        for display, logical, canonical in FACTORIAL_RUNTIME_ARMS:
            seconds = float(timings[canonical])
            if not np.isfinite(seconds) or seconds < 0.0:
                raise ValueError(f"invalid post-cache timing: {series_id}/{logical}")
            rows.append(
                {
                    "series_id": series_id,
                    "family": str(mask_row.family),
                    "subgroup": str(mask_row.subgroup),
                    "display_name": display,
                    "arm": logical,
                    "canonical_arm": canonical,
                    "seconds_per_series": seconds,
                    "ms_per_series": seconds * 1000.0,
                    "windows_per_series": windows,
                    "ms_per_window": seconds * 1000.0 / windows,
                    "process_rss_before_bytes": int(runtime["process_rss_before_bytes"]),
                    "process_rss_after_bytes": int(runtime["process_rss_after_bytes"]),
                    "python_tracemalloc_peak_bytes": int(runtime["python_tracemalloc_peak_bytes"]),
                    "encoder_calls": 0,
                    "measurement_scope": "historical committed post-cache projection/scoring",
                    "runtime_json_sha256": sha256_file(runtime_path),
                }
            )
    samples = pd.DataFrame(rows)
    if len(samples) != EXPECTED_VALID_SERIES * len(FACTORIAL_RUNTIME_ARMS):
        raise RuntimeError("runtime sample grid is incomplete")
    summaries: list[dict[str, Any]] = []
    for (display, arm, canonical), group in samples.groupby(
        ["display_name", "arm", "canonical_arm"], sort=False
    ):
        series_ms = group["ms_per_series"].to_numpy(dtype=float)
        window_ms = group["ms_per_window"].to_numpy(dtype=float)
        q25, q75 = np.quantile(series_ms, (0.25, 0.75))
        summaries.append(
            {
                "display_name": display,
                "arm": arm,
                "canonical_arm": canonical,
                "sample_count": len(group),
                "median_ms_per_series": float(np.median(series_ms)),
                "iqr_ms_per_series": float(q75 - q25),
                "p95_ms_per_series": float(np.quantile(series_ms, 0.95)),
                "mean_ms_per_series": float(np.mean(series_ms)),
                "std_ms_per_series": float(np.std(series_ms, ddof=1)),
                "median_ms_per_window": float(np.median(window_ms)),
                "p95_ms_per_window": float(np.quantile(window_ms, 0.95)),
                "peak_observed_process_rss_bytes": int(
                    group[["process_rss_before_bytes", "process_rss_after_bytes"]].max().max()
                ),
                "encoder_calls": 0,
                "scope": "post-cache projection/scoring overhead",
                "protocol_deviation": "one committed pass per series; not repeated interleaved microbenchmark",
            }
        )
    summary = pd.DataFrame(summaries)
    runtime_dir = output / "runtime"
    _atomic_csv(runtime_dir / "runtime_postcache_samples.csv", samples)
    _atomic_csv(runtime_dir / "runtime_postcache.csv", summary)
    _atomic_json(
        runtime_dir / "runtime_environment.json",
        {
            "created_at_utc": _utc_now(),
            "scope": "post-cache projection/scoring overhead",
            "coverage": EXPECTED_VALID_SERIES,
            "encoder_calls": 0,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "torch": torch.__version__,
            "cuda_available_at_packaging": torch.cuda.is_available(),
            "historical_measurement_note": "timings were read from immutable cache-only runtime transactions",
        },
    )
    _atomic_text(
        runtime_dir / "runtime_protocol.md",
        "# Runtime protocol\n\n"
        "These values are **post-cache projection/scoring overhead** only; they exclude rendering, "
        "model loading, encoder inference, cache creation, and disk loading. The table covers all 488 "
        "common-valid series using their immutable one-pass runtime transactions. Each series supplies "
        "one observation per arm. This is not a repeated, interleaved warm-cache microbenchmark, and "
        "the package does not claim end-to-end latency or fixed-thread causal timing.\n",
    )
    return samples, summary


def _input_identities(paths: Mapping[str, Path], git_commit: str) -> pd.DataFrame:
    rows = [
        {
            "identity": name,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "status": "VERIFIED",
        }
        for name, path in paths.items()
    ]
    rows.append(
        {
            "identity": "git_commit",
            "path": "repository HEAD at packaging",
            "size_bytes": 0,
            "sha256": git_commit,
            "status": "RECORDED_DIRTY_WORKTREE_PRESERVED",
        }
    )
    return pd.DataFrame(rows)


def build_data_only_supplement(
    *,
    repo_root: Path,
    evaluation_dir: Path,
    score_root: Path,
    run_root: Path,
    previous_package: Path,
    config_path: Path,
    registry_path: Path,
    external_reference_path: Path,
    output_root: Path,
    zip_path: Path,
) -> tuple[Path, Mapping[str, Any]]:
    repo = repo_root.resolve(strict=True)
    evaluation = evaluation_dir.resolve(strict=True)
    config = config_path.resolve(strict=True)
    registry = registry_path.resolve(strict=True)
    external = external_reference_path.resolve(strict=True)
    output = output_root.resolve()
    _copy_base_package(previous_package, output)
    _copy_delivery_sources(repo, output)
    git_commit = _git_commit(repo)

    _common_valid_mask(evaluation)
    aliases = _stage_statistics(evaluation, output)
    export_supplement_qualitative(config, evaluation, score_root, output / "qualitative")
    runtime_samples, runtime_summary = _stage_historical_runtime(
        config, evaluation, run_root, output
    )

    confirmation = audit_confirmation_cohort(None, None)
    _atomic_json(output / "provenance" / "confirmation_audit.json", confirmation)
    _atomic_text(
        output / "CONFIRMATION_BLOCKED.md",
        "# Confirmation blocked\n\nNo pre-existing untouched cohort and pre-selection hash binding were found. "
        "The 492-series results remain exploratory/post-selected; no retrospective split is presented "
        "as confirmation.\n",
    )

    identity_paths = {
        "config": config,
        "registry": registry,
        "external_reference": external,
        "evaluation_complete": evaluation / "_COMBINED_EVALUATION_COMPLETE.json",
        "aggregation_complete": evaluation / "_COMBINED_AGGREGATION_COMPLETE.json",
        "per_series_metrics": evaluation / "per_series_metrics.csv",
        "valid_series_mask": evaluation / "valid_series_mask.csv",
        "family3_metrics": evaluation / "family3_metrics.csv",
        "previous_package_manifest": previous_package / "PACKAGE_MANIFEST.json",
    }
    _atomic_csv(output / "input_identities.csv", _input_identities(identity_paths, git_commit))
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "git_commit": git_commit,
        "series_count": EXPECTED_SERIES,
        "common_valid_series": EXPECTED_VALID_SERIES,
        "factorial_arms": [logical for _, logical, _ in FACTORIAL_RUNTIME_ARMS],
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 2027,
        "model_or_encoder_runs": 0,
        "figures_emitted": 0,
        "manuscript_edits": 0,
        "confirmation_status": confirmation.get("status"),
        "runtime_coverage": len(runtime_samples) // 4,
        "runtime_protocol_deviation": "historical one-pass per-series records; no repeated interleaving",
    }
    _atomic_json(output / "protocol.json", protocol)
    _atomic_text(
        output / "README.md",
        "# ViTTrace data-only supplement\n\n"
        "Compact statistics, paired deltas, Table 3 data, deterministic qualitative case data, "
        "and post-cache runtime records. No figures, datasets, weights, token caches, or model runs "
        "are included.\n",
    )
    _atomic_text(
        output / "STATUS.md",
        f"COMPLETE: 492 series, 488 common-valid, 0 failures, 0 model/encoder runs, "
        f"0 figures. Git {git_commit}.\n",
    )
    _atomic_text(
        output / "EXPERIMENT_LOG.md",
        f"{_utc_now()} Reused immutable evaluation/cache transactions; computed only data-only "
        "statistics and exports. No scores or protocols were changed.\n",
    )
    failure_manifest = output / "failures" / "failure_manifest.csv"
    failures = pd.read_csv(failure_manifest)
    if len(failures) != 0:
        raise RuntimeError("base failure manifest is not empty")

    forbidden = [
        path for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".svg", ".png", ".jpg", ".jpeg", ".npy", ".npz"}
    ]
    if forbidden:
        raise RuntimeError(f"data-only package contains forbidden payloads: {forbidden[:4]}")

    key_outputs = {
        "bootstrap_ci_complete": aliases["bootstrap"],
        "factorial_2x2_summary": aliases["factorial"],
        "factorial_interaction": aliases["interaction"],
        "table3_ours_row": aliases["table3"],
        "qualitative_complete": output / "qualitative" / "_SUPPLEMENT_QUALITATIVE_COMPLETE.json",
        "runtime_postcache": output / "runtime" / "runtime_postcache.csv",
        "protocol": output / "protocol.json",
        "failure_manifest": failure_manifest,
    }
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "created_at_utc": _utc_now(),
        "series_count": EXPECTED_SERIES,
        "common_valid_series": EXPECTED_VALID_SERIES,
        "failure_count": 0,
        "model_or_encoder_runs": 0,
        "figures_emitted": 0,
        "runtime_rows": len(runtime_samples),
        "runtime_summary_rows": len(runtime_summary),
        "key_outputs": {
            name: {"relative_path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)}
            for name, path in key_outputs.items()
        },
    }
    _atomic_json(output / "_SUPPLEMENT_DELIVERY_COMPLETE.json", marker)
    destination, manifest = build_result_package(output, zip_path=zip_path)
    verify_result_zip(destination)
    return destination, manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo_root", "evaluation_dir", "score_root", "run_root", "previous_package",
        "config_path", "registry_path", "external_reference_path", "output_root", "zip_path",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args(argv)
    destination, manifest = build_data_only_supplement(**vars(args))
    print(json.dumps({"zip_path": str(destination), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
