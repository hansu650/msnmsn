from __future__ import annotations

from pathlib import Path

import pandas as pd

from measure_vit4ts_v3.rough_figure_outputs import (
    MANDATORY_ROUGH_FIGURES,
    render_rough_figure_set,
)


def _all_inputs() -> dict[str, pd.DataFrame]:
    return {
        "backbone_accuracy_time": pd.DataFrame(
            {
                "backbone": ["B16", "B32"],
                "group": ["NAB", "NAB"],
                "metric": ["F1-max", "F1-max"],
                "value": [0.6, 0.5],
                "elapsed_seconds_per_series": [2.0, 1.0],
            }
        ),
        "window_sensitivity": pd.DataFrame(
            {
                "window": [120, 240, 120, 240],
                "arm": ["REL", "REL", "FULL", "FULL"],
                "group": ["NAB"] * 4,
                "metric": ["F1-max"] * 4,
                "value": [0.5, 0.52, 0.55, 0.57],
            }
        ),
        "stride_sensitivity": pd.DataFrame(
            {
                "stride": [30, 60, 30, 60],
                "arm": ["REL", "REL", "FULL", "FULL"],
                "group": ["NAB"] * 4,
                "metric": ["F1-max"] * 4,
                "value": [0.51, 0.52, 0.56, 0.57],
            }
        ),
        "ihp_nctp_interaction": pd.DataFrame(
            {
                "ihp": [False, False, True, True],
                "nctp": [False, True, False, True],
                "group": ["NAB"] * 4,
                "metric": ["F1-max"] * 4,
                "value": [0.50, 0.52, 0.53, 0.57],
            }
        ),
        "matching_scope": pd.DataFrame(
            {
                "scope": ["position", "global"],
                "arm": ["FINAL", "FINAL"],
                "group": ["NAB", "NAB"],
                "metric": ["F1-max", "F1-max"],
                "value": [0.50, 0.57],
            }
        ),
        "scale_subset_heatmap": pd.DataFrame(
            {
                "scale_subset": ["P", "PML", "P", "PML"],
                "group": ["NAB", "NAB", "NASA", "NASA"],
                "metric": ["F1-max"] * 4,
                "value": [0.51, 0.57, 0.49, 0.54],
            }
        ),
        "reducer_sensitivity": pd.DataFrame(
            {
                "reducer_family": ["quantile", "quantile", "topfrac", "topfrac"],
                "reducer_setting": ["q10", "q25", "10", "25"],
                "group": ["NAB"] * 4,
                "metric": ["F1-max"] * 4,
                "value": [0.50, 0.52, 0.49, 0.51],
            }
        ),
        "line_vs_spectrogram": pd.DataFrame(
            {
                "representation": ["line", "spectrogram", "line", "spectrogram"],
                "arm": ["REL", "REL", "FULL", "FULL"],
                "group": ["NAB"] * 4,
                "metric": ["F1-max"] * 4,
                "value": [0.50, 0.48, 0.57, 0.53],
            }
        ),
        "runtime_memory": pd.DataFrame(
            {
                "config_id": ["REL_cached", "FULL_e2e"],
                "measurement_mode": ["cached", "encoder_inclusive"],
                "median_s": [0.02, 2.5],
                "peak_rss_mb": [120.0, 1024.0],
            }
        ),
        "qualitative_score_stacks": pd.DataFrame(
            {
                "case_role": ["fixed_msl_c1"] * 6,
                "panel": ["raw_series"] * 3 + ["REL"] * 3,
                "time_index": [0, 1, 2, 0, 1, 2],
                "value": [0.0, 1.0, 0.5, 0.1, 0.8, 0.2],
                "ground_truth": [0, 1, 0, 0, 1, 0],
            }
        ),
        "structural_mapping_coverage": pd.DataFrame(
            {
                "case_role": ["boundary_terminal_defect"] * 4,
                "operator": ["nctp_linear"] * 4,
                "local_time": [0, 0, 1, 1],
                "patch_index": [0, 1, 0, 1],
                "weight": [1.0, 0.0, 0.4, 0.6],
            }
        ),
    }


def test_all_mandatory_rough_figures_write_tidy_svg_and_pdf(tmp_path: Path) -> None:
    inputs = _all_inputs()
    assert set(inputs) == {recipe.name for recipe in MANDATORY_ROUGH_FIGURES}
    status = render_rough_figure_set(
        inputs,
        plot_data_root=tmp_path / "plot_data",
        figure_root=tmp_path / "rough_figures",
    )
    assert set(status["status"]) == {"COMPLETE"}
    assert (status["plotted_rows"] > 0).all()
    for recipe in MANDATORY_ROUGH_FIGURES:
        tidy = tmp_path / "plot_data" / f"{recipe.name}.csv"
        svg = tmp_path / "rough_figures" / f"{recipe.name}.svg"
        pdf = tmp_path / "rough_figures" / f"{recipe.name}.pdf"
        assert tidy.is_file() and pd.read_csv(tidy).shape[0] > 0
        assert "<svg" in svg.read_text(encoding="utf-8")
        assert pdf.read_bytes().startswith(b"%PDF")


def test_missing_and_invalid_inputs_are_explicit_not_fabricated(tmp_path: Path) -> None:
    status = render_rough_figure_set(
        {"backbone_accuracy_time": pd.DataFrame({"backbone": ["B16"]})},
        plot_data_root=tmp_path / "plot_data",
        figure_root=tmp_path / "rough_figures",
    ).set_index("figure")
    assert status.loc["backbone_accuracy_time", "status"] == "BLOCKED_INVALID_INPUT"
    assert "missing required columns" in status.loc["backbone_accuracy_time", "reason"]
    assert status.loc["window_sensitivity", "status"] == "BLOCKED_MISSING_INPUT"
    assert not (tmp_path / "rough_figures" / "backbone_accuracy_time.svg").exists()
    assert (tmp_path / "plot_data" / "rough_figure_status.csv").is_file()
