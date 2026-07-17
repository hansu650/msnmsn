"""Create the final IHP mechanism and evidence figures from frozen artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6B7280"
LIGHT = "#F3F4F6"


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output / f"{stem}.{suffix}",
            # PDF/SVG are the manuscript-facing vector assets.  Keep a
            # 600-dpi PNG fallback for IEEE line-art/combination workflows.
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def mechanism_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.26, 0.16, 0.48, "Frozen ViT4TS\nmultiscale costs", LIGHT),
        (0.23, 0.26, 0.17, 0.48, "Literal incidence\nrepair  $A_s$", "#E8F3FA"),
        (0.45, 0.26, 0.18, 0.48, "Inherited harmonic\nprojection", "#E8F6F1"),
        (0.68, 0.26, 0.13, 0.48, "Equal-scale\nfusion", "#FFF1E8"),
        (0.86, 0.26, 0.12, 0.48, "Timestamp\nscores", LIGHT),
    ]
    for x, y, width, height, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                linewidth=1.2,
                edgecolor="#374151",
                facecolor=color,
            )
        )
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    for start, end in ((0.18, 0.23), (0.40, 0.45), (0.63, 0.68), (0.81, 0.86)):
        ax.add_patch(
            FancyArrowPatch(
                (start + 0.006, 0.50),
                (end - 0.006, 0.50),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color="#374151",
            )
        )
    ax.text(
        0.535,
        0.10,
        r"$p_s(i)=n_s(i)\,/\!\sum_{k:A_s(i,k)=1} d_{s,k}^{-1}$",
        ha="center",
        va="center",
        fontsize=10,
        color="#111827",
    )
    ax.text(
        0.50,
        0.91,
        "IHP: Coordinate-Contract Repair",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    ax.text(0.315, 0.79, "Changed", ha="center", va="center", fontsize=7.5, color=BLUE, weight="bold")
    ax.text(0.54, 0.79, "Unchanged", ha="center", va="center", fontsize=7.5, color=GREEN, weight="bold")
    _save(fig, output, "ihp_method")


def evidence_figure(artifacts: Path, output: Path) -> None:
    comparison = pd.read_csv(artifacts / "ihp_external_vit4ts_comparison.csv")
    bootstrap = pd.read_csv(artifacts / "ihp_hierarchical_bootstrap.csv")
    order = comparison["paper_group"].tolist()
    short = [
        name.replace("NAB-", "").replace("NASA-", "").replace("Yahoo-", "")
        for name in order
    ]
    x = np.arange(len(comparison))

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [2.15, 1.0]}
    )
    left.scatter(
        x - 0.18,
        comparison["paper_vit4ts_f1_max"],
        marker="D",
        color=GRAY,
        s=22,
        label="ViT4TS (paper-reported)",
        zorder=3,
    )
    left.scatter(
        x,
        comparison["control_f1_max"],
        marker="s",
        color=ORANGE,
        s=23,
        label="REL-U (same cache)",
        zorder=3,
    )
    left.scatter(
        x + 0.18,
        comparison["ihp_f1_max"],
        marker="o",
        color=BLUE,
        s=25,
        label="IHP (same cache)",
        zorder=3,
    )
    left.set_xticks(x, short, rotation=45, ha="right")
    left.set_ylabel("F1-max")
    left.set_ylim(0.35, 0.94)
    left.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    left.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, fontsize=7.1, loc="lower center", ncol=1)
    left.set_title("(a) External context and same-cache control", fontsize=9.5)
    left.axvline(4.5, color="#CBD5E1", linewidth=0.8)
    left.axvline(6.5, color="#CBD5E1", linewidth=0.8)
    left.text(2.0, 0.925, "NAB", ha="center", fontsize=7.5, color="#4B5563")
    left.text(5.5, 0.925, "NASA", ha="center", fontsize=7.5, color="#4B5563")
    left.text(8.5, 0.925, "Yahoo", ha="center", fontsize=7.5, color="#4B5563")

    metrics = ["f1_max", "auprc", "vus_pr"]
    labels = ["F1-max", "AUPRC", "VUS-PR"]
    colors = [BLUE, ORANGE, GREEN]
    values = []
    lower = []
    upper = []
    for metric in metrics:
        row = bootstrap.loc[bootstrap["metric"] == metric].iloc[0]
        values.append(float(row["delta"]))
        lower.append(float(row["delta"] - row["ci_lower"]))
        upper.append(float(row["ci_upper"] - row["delta"]))
    positions = np.arange(3)
    right.bar(positions, values, color=colors, width=0.62, alpha=0.9)
    right.errorbar(
        positions,
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#111827",
        elinewidth=1.0,
        capsize=3,
    )
    right.axhline(0, color="#374151", linewidth=0.8)
    right.set_xticks(positions, labels, rotation=24, ha="right")
    right.set_ylabel(r"IHP $-$ released control")
    right.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    right.spines[["top", "right"]].set_visible(False)
    right.set_title("(b) Same-cache ablation", fontsize=9.5)
    for index, value in enumerate(values):
        right.text(index, value + 0.003, f"{value:+.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout(w_pad=1.2)
    _save(fig, output, "ihp_results")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mechanism_figure(args.output)
    evidence_figure(args.artifacts, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
