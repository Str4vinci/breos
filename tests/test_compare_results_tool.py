"""Tests for the source-tree result comparison tool."""

from pathlib import Path

import pandas as pd

from tools import compare_results as compare_results_tool


def test_compare_results_supports_two_folders_with_custom_labels(tmp_path, monkeypatch):
    first = pd.DataFrame(
        {
            "Year": [0, 1, 2],
            "Savings_Cumulative_NPV": [-500.0, -100.0, 200.0],
            "Cost_System_Cumulative_NPV": [5000.0, 5200.0, 5400.0],
            "Cost_No_Sys_Cumulative_NPV": [0.0, 3000.0, 6000.0],
        }
    )
    second = pd.DataFrame(
        {
            "Year": [0, 1, 2],
            "Savings_Cumulative_NPV": [-800.0, -400.0, 50.0],
            "Cost_System_Cumulative_NPV": [6500.0, 6600.0, 6700.0],
            "Cost_No_Sys_Cumulative_NPV": [0.0, 3400.0, 6800.0],
        }
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first.to_csv(first_dir / "cost_projection.csv", index=False)
    second.to_csv(second_dir / "cost_projection.csv", index=False)

    rendered = {}

    def fake_plot(cost_dfs, labels, colors, output_dir, filename):
        rendered.update(
            cost_dfs=cost_dfs,
            labels=labels,
            colors=colors,
            output_dir=output_dir,
            filename=filename,
        )
        Path(output_dir, filename).write_bytes(b"rendered")

    monkeypatch.setattr(compare_results_tool, "plot_breakeven_comparison", fake_plot)
    output_dir = tmp_path / "comparison"

    result = compare_results_tool.compare_results(
        [str(first_dir), str(second_dir)],
        labels=["Reference", "Alternative"],
        output_dir=str(output_dir),
    )

    expected_output = output_dir / "breakeven_comparison.png"
    assert result == str(expected_output)
    assert expected_output.read_bytes() == b"rendered"
    assert rendered["labels"] == ["Reference", "Alternative"]
    assert rendered["output_dir"] == str(output_dir)
    assert rendered["filename"] == "breakeven_comparison.png"
    assert rendered["colors"] == list(compare_results_tool.BREAKEVEN_COLORS[:2])
    assert len(rendered["cost_dfs"]) == 2
    pd.testing.assert_frame_equal(rendered["cost_dfs"][0], first)
    pd.testing.assert_frame_equal(rendered["cost_dfs"][1], second)
