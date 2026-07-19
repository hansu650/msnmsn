"""Command-line entry points for v3 compact outputs and packaging."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from .qualitative_outputs import select_qualitative_cases
from .result_package import build_result_package
from .rough_figure_outputs import MANDATORY_ROUGH_FIGURES, render_rough_figure_set
from .runtime_outputs import load_runtime_jsons, write_runtime_outputs
from .structural_audit import structural_audit_frame


def _pair(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected HEIGHTxWIDTH")
    try:
        result = tuple(map(int, parts))
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair values must be integers") from error
    if result[0] <= 0 or result[1] <= 0:
        raise argparse.ArgumentTypeError("pair values must be positive")
    return result


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _structural(args: argparse.Namespace) -> int:
    frame = structural_audit_frame(
        patch_grid=args.patch_grid,
        window_length=args.window,
        image_size=args.image_size,
    )
    _atomic_csv(args.output, frame)
    print(json.dumps({"output": str(args.output), "rows": len(frame)}))
    return 0


def _runtime(args: argparse.Namespace) -> int:
    if args.samples_csv:
        samples = pd.read_csv(args.samples_csv)
    else:
        samples = load_runtime_jsons(args.runtime_json)
    sample_path, summary_path = write_runtime_outputs(args.output_root, samples)
    print(json.dumps({"samples": str(sample_path), "summary": str(summary_path)}))
    return 0


def _qualitative_select(args: argparse.Namespace) -> int:
    metrics = pd.read_csv(args.metrics)
    structural = pd.read_csv(args.structural)
    cases = select_qualitative_cases(
        metrics,
        structural,
        candidate_arm=args.candidate,
        control_arm=args.control,
        metric=args.metric,
        fixed_series_id=args.fixed_series_id,
    )
    _atomic_csv(args.output, cases)
    print(json.dumps({"output": str(args.output), "cases": len(cases)}))
    return 0


def _rough(args: argparse.Namespace) -> int:
    inputs = {
        recipe.name: path
        for recipe in MANDATORY_ROUGH_FIGURES
        if (path := args.input_root / f"{recipe.name}.csv").is_file()
    }
    status = render_rough_figure_set(
        inputs,
        plot_data_root=args.plot_data_root,
        figure_root=args.figure_root,
    )
    counts = status["status"].value_counts().to_dict()
    print(json.dumps({"status_counts": counts}, sort_keys=True))
    return 0 if (status["status"] == "COMPLETE").all() else 2


def _package(args: argparse.Namespace) -> int:
    archive, manifest = build_result_package(
        args.root,
        zip_path=args.zip,
        allow_incomplete=args.allow_incomplete,
        max_file_bytes=args.max_file_mib * 1024 * 1024,
    )
    print(
        json.dumps(
            {
                "archive": str(archive),
                "payload_file_count": manifest["payload_file_count"],
                "payload_bytes": manifest["payload_bytes"],
                "sha256sums_sha256": manifest["sha256sums_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    structural = commands.add_parser("structural", help="write structural_audit.csv")
    structural.add_argument("--patch-grid", type=_pair, default=(14, 14))
    structural.add_argument("--window", type=int, default=240)
    structural.add_argument("--image-size", type=_pair, default=(224, 224))
    structural.add_argument("--output", type=Path, required=True)
    structural.set_defaults(func=_structural)

    runtime = commands.add_parser("runtime", help="write runtime samples and summary")
    runtime_source = runtime.add_mutually_exclusive_group(required=True)
    runtime_source.add_argument("--samples-csv", type=Path)
    runtime_source.add_argument("--runtime-json", type=Path, nargs="+")
    runtime.add_argument("--output-root", type=Path, required=True)
    runtime.set_defaults(func=_runtime)

    qualitative = commands.add_parser("qualitative-select", help="freeze four qualitative cases")
    qualitative.add_argument("--metrics", type=Path, required=True)
    qualitative.add_argument("--structural", type=Path, required=True)
    qualitative.add_argument("--candidate", required=True)
    qualitative.add_argument("--control", required=True)
    qualitative.add_argument("--metric", required=True)
    qualitative.add_argument("--fixed-series-id", default="MSL__C-1")
    qualitative.add_argument("--output", type=Path, required=True)
    qualitative.set_defaults(func=_qualitative_select)

    rough = commands.add_parser("rough", help="render available mandatory rough figures")
    rough.add_argument("--input-root", type=Path, required=True)
    rough.add_argument("--plot-data-root", type=Path, required=True)
    rough.add_argument("--figure-root", type=Path, required=True)
    rough.set_defaults(func=_rough)

    package = commands.add_parser("package", help="build verified compact result ZIP")
    package.add_argument("--root", type=Path, required=True)
    package.add_argument("--zip", type=Path)
    package.add_argument("--allow-incomplete", action="store_true")
    package.add_argument("--max-file-mib", type=int, default=128)
    package.set_defaults(func=_package)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
