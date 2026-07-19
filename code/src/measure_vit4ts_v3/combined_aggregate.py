"""Unified 11-subgroup/3-family aggregation for combined v3 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .aggregate import aggregate_metrics, paired_hierarchical_bootstrap
from .combined_evaluator import arm_registry
from .combined_protocol import load_combined_protocol, sha256_file
from .metrics import DETECTION_METRICS, valid_mask_sha256


SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def aggregate_combined(
    registry_path: Path,
    evaluation_root: Path,
    output_root: Path | None = None,
    *,
    n_boot: int | None = None,
) -> tuple[Path, ...]:
    """Aggregate one complete combined evaluator transaction.

    ``n_boot`` exists for focused tests.  Production invocations omit it and
    therefore use the registry-frozen 10,000 shared seed-2027 draws.
    """

    protocol = load_combined_protocol(registry_path)
    evaluation = Path(evaluation_root)
    marker_path = evaluation / "_COMBINED_EVALUATION_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "COMPLETE"
        or marker.get("protocol_sha256") != protocol.payload_sha256
        or int(marker.get("series_count", -1)) != protocol.expected_series
        or int(marker.get("valid_series_count", -1)) != protocol.expected_valid_series
    ):
        raise ValueError("combined aggregation requires the exact complete evaluator marker")
    metrics_path = evaluation / "per_series_metrics.csv"
    mask_path = evaluation / "valid_series_mask.csv"
    if sha256_file(metrics_path) != marker["per_series_metrics_sha256"]:
        raise ValueError("combined per-series metrics changed after evaluation")
    if sha256_file(mask_path) != marker["valid_series_mask_file_sha256"]:
        raise ValueError("combined validity mask changed after evaluation")
    metrics = pd.read_csv(metrics_path)
    mask = pd.read_csv(mask_path)
    for column in [name for name in mask.columns if name.startswith("valid_")]:
        mask[column] = mask[column].astype(str).str.lower().map({"true": True, "false": False})
        if mask[column].isna().any():
            raise ValueError(f"combined validity column is not boolean: {column}")
    registry = arm_registry(protocol)
    bundle = aggregate_metrics(metrics, mask, registry)
    root = Path(output_root or evaluation)
    outputs = {
        "per_series": root / "per_series_metrics_validated.csv",
        "subgroup11": root / "subgroup11_metrics.csv",
        "family3": root / "family3_metrics.csv",
        "equal11": root / "equal11_metrics.csv",
        "fileweighted": root / "fileweighted_metrics.csv",
        "bootstrap": root / "bootstrap_ci.csv",
    }
    _atomic_csv(outputs["per_series"], bundle.per_series)
    _atomic_csv(outputs["subgroup11"], bundle.subgroup11)
    _atomic_csv(outputs["family3"], bundle.family3)
    _atomic_csv(outputs["equal11"], bundle.equal11)
    _atomic_csv(outputs["fileweighted"], bundle.fileweighted)
    bootstrap = pd.concat(
        [
            paired_hierarchical_bootstrap(
                metrics,
                mask,
                registry,
                metric,
                n_boot=n_boot,
                seed=protocol.bootstrap_seed,
            )
            for metric in DETECTION_METRICS
        ],
        ignore_index=True,
    )
    if bootstrap["resample_plan_sha256"].nunique() != 1:
        raise RuntimeError("combined bootstrap metrics did not share one index plan")
    _atomic_csv(outputs["bootstrap"], bootstrap)
    complete = root / "_COMBINED_AGGREGATION_COMPLETE.json"
    _atomic_json(
        complete,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "protocol_sha256": protocol.payload_sha256,
            "valid_mask_sha256": valid_mask_sha256(mask),
            "series_count": protocol.expected_series,
            "valid_series_count": protocol.expected_valid_series,
            "arm_count": len(protocol.arms),
            "contrast_count": len(protocol.contrasts),
            "bootstrap_seed": protocol.bootstrap_seed,
            "bootstrap_replicates": protocol.bootstrap_replicates if n_boot is None else int(n_boot),
            "resample_plan_sha256": str(bootstrap.iloc[0]["resample_plan_sha256"]),
            **{f"{name}_sha256": sha256_file(path) for name, path in outputs.items()},
        },
    )
    return (*outputs.values(), complete)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    paths = aggregate_combined(args.registry, args.evaluation_dir, args.output_dir)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["aggregate_combined", "main"]
