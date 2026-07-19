"""Hash-bound qualitative extraction for the mandatory ViTTrace v3 cases.

The structural boundary/terminal case is frozen from patch-field geometry
before this module opens metric or label artifacts.  When patch fields were
not persisted by a scorer, they are recomputed from the exact registered
dynamic token cache and committed with input/output SHA256 identities.  The
committed score vectors are only read and verified; they are never replaced
by recomputed scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from measure_vit4ts.full_manifest import FullSeriesRecord, load_manifest

from .core import apply_temporal_operator, build_linear_nctp
from .dynamic_cache import (
    CACHE_FILE,
    CACHE_MANIFEST,
    DynamicCacheKey,
    load_dynamic_cache,
)
from .dynamic_scorer import compute_dynamic_scores, stitch_native_dynamic
from .qualitative_outputs import (
    CASE_COLUMNS,
    CASE_ROLES,
    nctp_mapping_zoom_data,
    patch_field_heatmap_data,
    score_stack_plot_data,
    structural_case_scores,
    write_qualitative_outputs,
)


SCHEMA_VERSION = 1
PANEL_ARMS = {
    "REL": "IHP0_NCTP0",
    "IHP": "IHP1_NCTP0",
    "REL_NCTP": "IHP0_NCTP1",
    "FULL": "IHP1_NCTP1",
}
FIELD_FILE = "qualitative_patch_fields.npz"
FIELD_MANIFEST = "qualitative_patch_fields.json"


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest().upper()


def source_sha256() -> str:
    paths = (
        Path(__file__),
        Path(__file__).with_name("dynamic_scorer.py"),
        Path(__file__).with_name("core.py"),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _load_config(path: Path) -> tuple[Mapping[str, Any], str]:
    target = Path(path).resolve(strict=True)
    raw = target.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, Mapping) or payload.get("stage") != "vittrace_ablation_full_v3":
        raise ValueError("qualitative extraction accepts only the isolated v3 config")
    return payload, hashlib.sha256(raw).hexdigest().upper()


def _dynamic_key(payload: Mapping[str, Any]) -> DynamicCacheKey:
    key = dict(payload["key"])
    key["image_size"] = tuple(map(int, key["image_size"]))
    return DynamicCacheKey(**key)


def locate_dynamic_cache(cache_root: Path, record: FullSeriesRecord) -> tuple[Path, DynamicCacheKey, str, str]:
    candidates = sorted((Path(cache_root) / record.series_id).glob(f"*/{CACHE_MANIFEST}"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one bound dynamic cache for {record.series_id}")
    manifest_path = candidates[0]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = _dynamic_key(payload)
    if key.series_id != record.series_id or key.data_sha256.upper() != record.expected_sha256.upper():
        raise ValueError("dynamic cache series/data identity mismatch")
    directory = manifest_path.parent
    cache = load_dynamic_cache(directory, key)
    del cache
    return directory, key, sha256_file(manifest_path), sha256_file(directory / CACHE_FILE)


def _field_root(root: Path, series_id: str) -> Path:
    return Path(root) / series_id


def load_field_cache(
    root: Path,
    series_id: str,
    *,
    source_cache_manifest_sha256: str,
    expected_source_sha256: str,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    directory = _field_root(root, series_id)
    manifest_path = directory / FIELD_MANIFEST
    payload_path = directory / FIELD_FILE
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        int(payload.get("schema_version", -1)) != SCHEMA_VERSION
        or payload.get("series_id") != series_id
        or str(payload.get("source_cache_manifest_sha256", "")).upper()
        != source_cache_manifest_sha256.upper()
        or str(payload.get("source_sha256", "")).upper() != expected_source_sha256.upper()
        or str(payload.get("field_payload_sha256", "")).upper() != sha256_file(payload_path)
        or payload.get("labels_read") is not False
    ):
        raise ValueError("qualitative patch-field cache identity mismatch")
    with np.load(payload_path, allow_pickle=False) as archive:
        released = np.ascontiguousarray(archive["released_field"], dtype=np.float64)
        literal = np.ascontiguousarray(archive["literal_field"], dtype=np.float64)
    if released.shape != literal.shape or released.ndim != 2 or not np.isfinite(released).all() or not np.isfinite(literal).all():
        raise ValueError("qualitative patch fields are invalid")
    if list(released.shape) != list(payload.get("shape", [])):
        raise ValueError("qualitative patch-field shape changed")
    return released, literal, payload


def recompute_field_cache(
    output_root: Path,
    cache_directory: Path,
    key: DynamicCacheKey,
    *,
    full_length: int,
    device: str,
    source_cache_manifest_sha256: str,
    source_cache_payload_sha256: str,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Recompute label-free released/literal fields and commit their hash."""

    cache = load_dynamic_cache(cache_directory, key)
    bundle = compute_dynamic_scores(
        cache,
        full_length=int(full_length),
        window=int(key.window),
        stride=int(key.stride),
        image_size=key.image_size,
        device=device,
    )
    released = np.ascontiguousarray(bundle.arms["REL"].window_field, dtype=np.float64)
    literal = np.ascontiguousarray(bundle.arms["IHP"].window_field, dtype=np.float64)
    directory = _field_root(output_root, key.series_id)
    payload_path = directory / FIELD_FILE
    field_sha = _atomic_npz(payload_path, released_field=released, literal_field=literal)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "series_id": key.series_id,
        "shape": list(released.shape),
        "source_cache_directory": str(cache_directory.resolve()),
        "source_cache_manifest_sha256": source_cache_manifest_sha256.upper(),
        "source_cache_payload_sha256": source_cache_payload_sha256.upper(),
        "source_sha256": source_sha256(),
        "field_payload_sha256": field_sha,
        "matching_seconds": float(bundle.matching_seconds),
        "device": str(device),
        "encoder_calls": 0,
        "labels_read": False,
        "recomputed_from_bound_token_cache": True,
    }
    _atomic_json(directory / FIELD_MANIFEST, payload)
    return released, literal, payload


def _resolve_committed_score(stage_root: Path, series_id: str, arm: str) -> tuple[np.ndarray, str]:
    arm_root = Path(stage_root) / series_id / arm
    direct = arm_root / "score.npy"
    if direct.is_file():
        manifest_path = arm_root / "score_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest.get("score_sha256", manifest.get("sha256", ""))).upper()
        actual = sha256_file(direct)
        if expected and expected != actual:
            raise ValueError(f"committed score hash mismatch: {series_id}/{arm}")
        score_path = direct
    else:
        alias_path = arm_root / "alias_manifest.json"
        alias = json.loads(alias_path.read_text(encoding="utf-8"))
        score_path = (arm_root / str(alias["canonical_score_path"])).resolve(strict=True)
        actual = sha256_file(score_path)
        if actual != str(alias["canonical_score_sha256"]).upper():
            raise ValueError(f"committed alias hash mismatch: {series_id}/{arm}")
    values = np.load(score_path, allow_pickle=False)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("committed qualitative score is not a finite vector")
    return np.ascontiguousarray(values, dtype=np.float64), actual


def preselected_qualitative_cases(
    metrics: pd.DataFrame,
    structural_scores_frame: pd.DataFrame,
    *,
    candidate_arm: str,
    control_arm: str,
    metric: str,
    structural_series_id: str,
    fixed_series_id: str = "MSL__C-1",
) -> pd.DataFrame:
    """Freeze a label-free structural case, then select distinct metric cases."""

    required = {"series_id", "arm", metric}
    if required - set(metrics):
        raise ValueError("qualitative metrics lack the requested contrast")
    subset = metrics.loc[
        metrics["arm"].astype(str).isin((candidate_arm, control_arm)),
        ["series_id", "arm", metric],
    ]
    pivot = subset.pivot(index="series_id", columns="arm", values=metric)
    if candidate_arm not in pivot or control_arm not in pivot:
        raise ValueError("qualitative candidate/control metrics are missing")
    pivot = pivot.rename(columns={candidate_arm: "candidate_value", control_arm: "control_value"})
    pivot["delta"] = pivot["candidate_value"] - pivot["control_value"]
    if not np.isfinite(pivot[["candidate_value", "control_value", "delta"]].to_numpy()).all():
        raise ValueError("qualitative contrast must be defined on every candidate")
    contrast = pivot.reset_index()
    structural = structural_scores_frame.set_index("series_id", verify_integrity=True)
    required_ids = {fixed_series_id, structural_series_id}
    if not required_ids.issubset(set(contrast["series_id"])) or not required_ids.issubset(set(structural.index)):
        raise ValueError("fixed/structural qualitative series are unavailable")
    excluded = set(required_ids)
    remaining = contrast.loc[~contrast["series_id"].isin(excluded)].copy()
    remaining["series_id"] = remaining["series_id"].astype(str)
    best = remaining.sort_values(["delta", "series_id"], ascending=[False, True], kind="mergesort").iloc[0]
    remaining = remaining.loc[remaining["series_id"] != best["series_id"]]
    worst = remaining.sort_values(["delta", "series_id"], ascending=[True, True], kind="mergesort").iloc[0]
    selections = (
        ("fixed_msl_c1", fixed_series_id, "fixed_predeclared_series", False),
        ("best_improvement", str(best["series_id"]), f"max_{candidate_arm}_minus_{control_arm}_{metric}", True),
        ("worst_case", str(worst["series_id"]), f"min_{candidate_arm}_minus_{control_arm}_{metric}", True),
        ("boundary_terminal_defect", structural_series_id, "preselected_max_boundary_terminal_field_disagreement", False),
    )
    by_series = contrast.set_index("series_id")
    rows: list[dict[str, Any]] = []
    for order, (role, series_id, basis, uses_labels) in enumerate(selections):
        metric_row = by_series.loc[series_id]
        structural_row = structural.loc[series_id]
        rows.append(
            {
                "case_order": order,
                "case_role": role,
                "series_id": series_id,
                "selection_basis": basis,
                "selection_metric": metric if uses_labels else "",
                "candidate_arm": candidate_arm,
                "control_arm": control_arm,
                "candidate_value": float(metric_row["candidate_value"]),
                "control_value": float(metric_row["control_value"]),
                "delta": float(metric_row["delta"]),
                "structural_score": float(structural_row["boundary_terminal_score"]),
                "uses_evaluation_labels": uses_labels,
            }
        )
    frame = pd.DataFrame(rows, columns=CASE_COLUMNS)
    if tuple(frame["case_role"]) != CASE_ROLES or frame["series_id"].duplicated().any():
        raise RuntimeError("qualitative cases are not the frozen four distinct roles")
    return frame


def _blocked(root: Path, reason: str, error: BaseException | None = None) -> Path:
    marker = Path(root) / "_QUALITATIVE_BLOCKED.json"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED",
        "reason": reason,
        "source_sha256": source_sha256(),
    }
    if error is not None:
        payload.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    _atomic_json(marker, payload)
    return marker


def run_qualitative_extraction(
    config_path: Path,
    metrics_path: Path,
    dynamic_cache_root: Path,
    score_stage_root: Path,
    output_root: Path,
    *,
    device: str = "cpu",
    metric: str = "vus_pr",
) -> tuple[Path, ...]:
    """Execute the complete four-case qualitative-data transaction."""

    root = Path(output_root)
    try:
        config, config_sha = _load_config(config_path)
        manifest_path = Path(config["manifest"]["path"]).resolve(strict=True)
        if sha256_file(manifest_path) != str(config["manifest"]["sha256"]).upper():
            raise ValueError("qualitative manifest hash mismatch")
        _, records_tuple = load_manifest(manifest_path)
        records = {record.series_id: record for record in records_tuple}
        field_root = Path(config["paths"]["output_root"]) / "caches" / "qualitative_fields"
        structural_rows: list[pd.DataFrame] = []
        cache_index_rows: list[dict[str, Any]] = []
        fields_by_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        source = source_sha256()

        # Label-free pass.  No metrics/anomaly file is opened before the
        # structural case and its complete provenance table are committed.
        for record in records_tuple:
            cache_dir, key, manifest_sha, payload_sha = locate_dynamic_cache(dynamic_cache_root, record)
            try:
                released, literal, field_meta = load_field_cache(
                    field_root,
                    record.series_id,
                    source_cache_manifest_sha256=manifest_sha,
                    expected_source_sha256=source,
                )
                reused = True
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                released, literal, field_meta = recompute_field_cache(
                    field_root,
                    cache_dir,
                    key,
                    full_length=record.expected_length,
                    device=device,
                    source_cache_manifest_sha256=manifest_sha,
                    source_cache_payload_sha256=payload_sha,
                )
                reused = False
            fields_by_series[record.series_id] = (released, literal)
            structural_rows.append(
                structural_case_scores({record.series_id: (released, literal)}, patch_grid=(14, 14))
            )
            cache_index_rows.append(
                {
                    "series_id": record.series_id,
                    "source_cache_manifest_sha256": manifest_sha,
                    "source_cache_payload_sha256": payload_sha,
                    "field_payload_sha256": field_meta["field_payload_sha256"],
                    "field_shape": "x".join(map(str, released.shape)),
                    "cache_reused": reused,
                    "labels_read": False,
                }
            )
        structural_frame = pd.concat(structural_rows, ignore_index=True)
        structural_frame = structural_frame.sort_values("series_id").reset_index(drop=True)
        structural_path = root / "structural_case_scores.csv"
        cache_index_path = root / "qualitative_field_cache_index.csv"
        _atomic_csv(structural_path, structural_frame)
        _atomic_csv(cache_index_path, pd.DataFrame(cache_index_rows).sort_values("series_id"))
        ordered = structural_frame.sort_values(
            ["boundary_terminal_score", "series_id"], ascending=[False, True], kind="mergesort"
        )
        structural_series_id = str(ordered.iloc[0]["series_id"])
        structural_selection_path = root / "structural_case_selection.json"
        _atomic_json(
            structural_selection_path,
            {
                "schema_version": SCHEMA_VERSION,
                "series_id": structural_series_id,
                "selection_rule": "max_boundary_terminal_field_disagreement_then_series_id",
                "uses_labels": False,
                "structural_scores_sha256": sha256_file(structural_path),
                "field_cache_index_sha256": sha256_file(cache_index_path),
            },
        )

        metrics_target = Path(metrics_path).resolve(strict=True)
        metrics = pd.read_csv(metrics_target)
        cases = preselected_qualitative_cases(
            metrics,
            structural_frame,
            candidate_arm=PANEL_ARMS["FULL"],
            control_arm=PANEL_ARMS["REL"],
            metric=metric,
            structural_series_id=structural_series_id,
        )

        # Only now may labels be loaded for visualization rows.
        from measure_vit4ts.coordinate_envelope_runner import load_vendor_signal
        from measure_vit4ts.evaluator import load_ground_truth

        label_config = dict(config)
        label_config["data"] = dict(config["data"])
        label_config["scoring"] = {
            "series": [
                {"series_id": record.series_id, "relative_path": record.relative_path}
                for record in records_tuple
            ]
        }
        signals: dict[str, np.ndarray] = {}
        timestamps: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        scores: dict[tuple[str, str], np.ndarray] = {}
        score_hash_rows: list[dict[str, Any]] = []
        heatmap_fields: dict[tuple[str, str], np.ndarray] = {}
        window_indices: dict[str, int] = {}
        operators: dict[tuple[str, str], np.ndarray] = {}
        time_ranges: dict[str, tuple[int, int]] = {}
        nctp = build_linear_nctp(240, (14, 14), image_size=(224, 224))
        for case in cases.itertuples(index=False):
            series_id = str(case.series_id)
            record = records[series_id]
            loaded = load_vendor_signal(record, Path(config["data"]["root"]))
            signal = np.asarray(loaded.series.values, dtype=np.float64)
            time = np.asarray(loaded.series.timestamps)
            truth = load_ground_truth(label_config, series_id, time)
            signals[series_id] = signal
            timestamps[series_id] = time
            labels[series_id] = np.asarray(truth.point_labels, dtype=np.uint8)
            for panel, arm in PANEL_ARMS.items():
                score, score_sha = _resolve_committed_score(score_stage_root, series_id, arm)
                if score.size != record.expected_length:
                    raise ValueError("qualitative committed score length differs from manifest")
                scores[(series_id, arm)] = score
                score_hash_rows.append(
                    {"series_id": series_id, "panel": panel, "arm": arm, "score_sha256": score_sha}
                )
            released, literal = fields_by_series[series_id]
            delta = np.abs(released - literal)
            cells = np.r_[np.arange(13, 195, 14), 195]
            window_index = int(np.argmax(delta[:, cells].mean(axis=1)))
            window_indices[series_id] = window_index
            heatmap_fields[(series_id, "released_patch_field")] = released.reshape(-1, 14, 14)
            heatmap_fields[(series_id, "literal_patch_field")] = literal.reshape(-1, 14, 14)
            operators[(series_id, "nctp_linear")] = nctp
            time_ranges[series_id] = (112, 128)

        score_stack = score_stack_plot_data(
            cases,
            signals,
            labels,
            scores,
            PANEL_ARMS,
            timestamps=timestamps,
        )
        heatmaps = patch_field_heatmap_data(cases, heatmap_fields, window_indices=window_indices)
        mapping = nctp_mapping_zoom_data(
            cases, operators, time_ranges=time_ranges, patch_grid=(14, 14)
        )
        outputs = list(write_qualitative_outputs(root, cases, score_stack, heatmaps, mapping))
        score_hash_path = root / "qualitative_score_index.csv"
        _atomic_csv(score_hash_path, pd.DataFrame(score_hash_rows))
        outputs.extend(
            [structural_path, cache_index_path, structural_selection_path, score_hash_path]
        )
        hashes = {path.name: sha256_file(path) for path in outputs}
        marker = root / "_QUALITATIVE_COMPLETE.json"
        _atomic_json(
            marker,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "COMPLETE",
                "config_sha256": config_sha,
                "manifest_sha256": sha256_file(manifest_path),
                "metrics_sha256": sha256_file(metrics_target),
                "source_sha256": source,
                "case_series": list(cases["series_id"].astype(str)),
                "structural_case_uses_labels": False,
                "output_sha256": hashes,
            },
        )
        outputs.append(marker)
        blocked = root / "_QUALITATIVE_BLOCKED.json"
        if blocked.exists():
            blocked.unlink()
        return tuple(outputs)
    except FileNotFoundError as error:
        _blocked(root, f"required upstream stage is incomplete: {error}", error)
        return (root / "_QUALITATIVE_BLOCKED.json",)
    except Exception as error:
        _blocked(root, f"qualitative extraction failed: {error}", error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--dynamic-cache-root", type=Path, required=True)
    parser.add_argument("--score-stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric", default="vus_pr", choices=("f1_max", "auprc", "vus_pr"))
    args = parser.parse_args(argv)
    outputs = run_qualitative_extraction(
        args.config,
        args.metrics,
        args.dynamic_cache_root,
        args.score_stage_root,
        args.output_root,
        device=args.device,
        metric=args.metric,
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, sort_keys=True))
    return 2 if any(path.name.endswith("BLOCKED.json") for path in outputs) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FIELD_FILE",
    "FIELD_MANIFEST",
    "PANEL_ARMS",
    "SCHEMA_VERSION",
    "load_field_cache",
    "locate_dynamic_cache",
    "preselected_qualitative_cases",
    "recompute_field_cache",
    "run_qualitative_extraction",
    "source_sha256",
]
