"""What the plot functions return, whether they leave it open, and what they draw.

Every plot returns its figure, or a dict of figures keyed by file name. With
no output directory it saves nothing and leaves the figure open; with one it
saves the figure, closes it and still returns it. The inputs are small
synthetic frames shaped like the results each plot documents.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from breos import plotting  # noqa: E402
from breos.constants import DEFAULT_EOL_PERCENTAGE  # noqa: E402
from tests.conftest import _build_synthetic_weather  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _results(year: int = 2023) -> pd.DataFrame:
    """A year of hourly results with the energy, battery and cell-temperature columns."""
    index = pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz="UTC")
    hour = index.hour.to_numpy()
    pv = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 2000.0
    load = np.full(len(index), 500.0)
    return pd.DataFrame(
        {
            "Datetime": index,
            "PV_Production": pv,
            "Houseload": load,
            "Import_From_Grid": np.clip(load - pv, 0, None),
            "PV_AC_Export": np.clip(pv - load, 0, None),
            "Battery_Energy": 2000.0 + 500.0 * np.sin(hour / 24 * 2 * np.pi),
            "Battery_SOH": np.linspace(100.0, 96.0, len(index)),
            "T_cell": 20.0 + 5.0 * np.sin(hour / 24 * 2 * np.pi),
        }
    )


def _degradation() -> pd.DataFrame:
    days = pd.date_range("2023-01-01", periods=365, freq="D", tz="UTC")
    ramp = np.linspace(0.0, 0.04, len(days))
    return pd.DataFrame(
        {
            "Datetime": days,
            "SOH": 100.0 - 100.0 * ramp,
            "Cumulative_Cycle_Degradation": ramp / 2,
            "Cumulative_Calendar_Degradation": ramp / 2,
            "Cumulative_FEC": np.arange(len(days), dtype=float),
            "Resistance_Growth": ramp,
            "Effective_RTE": 0.95 - ramp,
        }
    )


def _cost_projection(years: int = 10) -> pd.DataFrame:
    """A cost projection that pays back in year 5, with the CO2 columns."""
    year = np.arange(1, years + 1)
    no_system = 1000.0 * year
    with_system = 5000.0 + 0.0 * year
    co2_total = np.full(years, 1700.0)
    co2_self = np.full(years, 800.0)
    frame = pd.DataFrame(
        {
            "Year": year,
            "Cost_No_Sys_Cumulative_NPV": no_system,
            "Cost_System_Cumulative_NPV": with_system,
            "Cost_System_Annual_NPV": np.zeros(years),
            "Savings_Cumulative_NPV": no_system - with_system,
            "CO2_Avoided_Total_kg": co2_total,
            "CO2_Avoided_SelfConsumed_kg": co2_self,
            "CO2_Avoided_Total_Cumulative_kg": np.cumsum(co2_total),
            "CO2_Avoided_SelfConsumed_Cumulative_kg": np.cumsum(co2_self),
        }
    )
    frame.attrs.update(total_investment=5000.0, currency="EUR")
    return frame


def _mc_runs(npv_mean: float = 5300.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    runs = pd.DataFrame(
        {
            "run": np.arange(40),
            "npv_savings": rng.normal(npv_mean, 100.0, 40),
            "payback_year": [8.0] * 30 + [np.nan] * 10,
            "payback_year_interpolated": list(np.linspace(6.0, 12.0, 30)) + [np.nan] * 10,
            "lcoe_per_kwh": rng.normal(0.2, 0.01, 40),
            "final_soh_pct": rng.normal(80.0, 1.0, 40),
            "mean_grid_independence_pct": rng.normal(55.0, 2.0, 40),
        }
    )
    runs.attrs["currency"] = "EUR"
    return runs


def _sweep() -> pd.DataFrame:
    rows = [
        {"param_n_modules": n, "param_battery_kwh": kwh, "grid_independence_pct": 40 + n + kwh, "npv_savings": 100 * n}
        for n in (4, 8)
        for kwh in (0.0, 5.0)
    ]
    return pd.DataFrame(rows)


def _orientation() -> pd.DataFrame:
    rows = [
        {"param_tilt": tilt, "param_azimuth": azimuth, "usable_ac_system_production_kwh": 1000 - abs(azimuth - 180)}
        for tilt in (10, 35)
        for azimuth in (90, 180, 270)
    ]
    return pd.DataFrame(rows)


def _pareto() -> pd.DataFrame:
    gi = np.linspace(45.0, 90.0, 6)
    frame = pd.DataFrame(
        {
            "Modules": 10,
            "Battery_kWh": np.linspace(0.0, 20.0, 6),
            "Grid_Independence_%": gi,
            "NPV": 4000.0 - 10.0 * (gi - 45.0) ** 2,
            "Projected_CO2_Avoided_SelfConsumed_kg": 100.0 * gi,
            "Projected_NPV": 4000.0 - 10.0 * (gi - 45.0) ** 2,
        }
    )
    frame.attrs["currency"] = "EUR"
    return frame


def _waterfall(bifacial_enabled=None, rear_gain_kwh: float = 0.0) -> dict:
    stages = [
        {"key": "horizontal_reference_dc", "label": "Horizontal reference", "energy_kwh": 1200.0},
        {
            "key": "transposition",
            "label": "Plane-of-array transposition",
            "energy_kwh": 1300.0,
            "delta_kwh": 100.0,
            "delta_pct_of_previous": 8.33,
        },
        {
            "key": "bifacial_rear_gain",
            "label": "Bifacial rear gain",
            "energy_kwh": 1300.0 + rear_gain_kwh,
            "delta_kwh": rear_gain_kwh,
            "delta_pct_of_previous": rear_gain_kwh / 13.0,
        },
        {
            "key": "temperature",
            "label": "Cell temperature",
            "energy_kwh": 1200.0,
            "delta_kwh": -100.0 - rear_gain_kwh,
            "delta_pct_of_previous": -7.7,
        },
    ]
    waterfall = {"stages": stages}
    if bifacial_enabled is not None:
        waterfall["bifacial"] = {"enabled": bifacial_enabled}
    return waterfall


_HISTORICAL = {2021: _build_synthetic_weather(2021), 2022: _build_synthetic_weather(2022)}

# Each plot, called with the output directory (None or a path) it is given.
PLOTS = {
    "yearly_graphs": lambda d: plotting.yearly_graphs(_results(), d),
    "weekly_graphs": lambda d: plotting.weekly_graphs(_results(), 10, d),
    "plot_monthly_comparison": lambda d: plotting.plot_monthly_comparison(_results(), d),
    "plot_monthly_balance": lambda d: plotting.plot_monthly_balance(_results(), d),
    "plot_timeseries": lambda d: plotting.plot_timeseries(_results().set_index("Datetime"), ["Houseload"], d),
    "plot_cell_temperature": lambda d: plotting.plot_cell_temperature(_results(), d),
    "plot_battery_soh_timeseries": lambda d: plotting.plot_battery_soh_timeseries(_results(), d),
    "degradation_plots": lambda d: plotting.degradation_plots(_degradation(), d),
    "plot_resistance_and_efficiency": lambda d: plotting.plot_resistance_and_efficiency(_degradation(), d),
    "plot_breakeven": lambda d: plotting.plot_breakeven(_cost_projection(), d),
    "plot_breakeven_comparison": lambda d: plotting.plot_breakeven_comparison([_cost_projection()], ["PV"], d),
    "plot_co2_savings": lambda d: plotting.plot_co2_savings(_cost_projection(), d),
    "plot_montecarlo_simulation": lambda d: plotting.plot_montecarlo_simulation(_mc_runs(), d, verbose=False),
    "plot_montecarlo_npv_distribution": lambda d: plotting.plot_montecarlo_npv_distribution(_mc_runs(), d),
    "plot_montecarlo_grid_independence_distribution": (
        lambda d: plotting.plot_montecarlo_grid_independence_distribution(_mc_runs(), d)
    ),
    "plot_montecarlo_final_soh_distribution": lambda d: plotting.plot_montecarlo_final_soh_distribution(_mc_runs(), d),
    "plot_breakeven_distribution": lambda d: plotting.plot_breakeven_distribution([6.0, 7.5, 9.0], 4, d),
    "plot_breakeven_cdf": lambda d: plotting.plot_breakeven_cdf([6.0, 7.5, 9.0, np.nan], d),
    "plot_breakeven_summary_bar": lambda d: plotting.plot_breakeven_summary_bar(3, 4, d),
    "plot_weather_monthly_comparison": (
        lambda d: plotting.plot_weather_monthly_comparison(_build_synthetic_weather(), _HISTORICAL, d)
    ),
    "plot_weather_annual_ghi_distribution": (
        lambda d: plotting.plot_weather_annual_ghi_distribution(_build_synthetic_weather(), _HISTORICAL, d)
    ),
    "plot_sweep_heatmap": lambda d: plotting.plot_sweep_heatmap(_sweep(), "grid_independence_pct", d),
    "plot_orientation_landscape": (
        lambda d: plotting.plot_orientation_landscape(_orientation(), "usable_ac_system_production_kwh", d)
    ),
    "plot_pareto_front": lambda d: plotting.plot_pareto_front(_pareto(), d, color_by="Battery_kWh"),
}

# The figures of the plots that draw several, by file name.
MULTI_FIGURE = {
    "degradation_plots": {
        "battery_degradation_soh",
        "battery_degradation_components_per_battery",
        "battery_degradation_fec",
        "battery_resistance_growth",
        "battery_effective_rte",
    },
    "plot_resistance_and_efficiency": {"battery_resistance_growth", "battery_effective_rte"},
    "plot_breakeven": {"breakeven_cumulative", "breakeven_annual"},
    "plot_co2_savings": {"co2_avoided_yearly", "co2_avoided_cumulative"},
    "plot_montecarlo_simulation": {
        "montecarlo_npv_distribution",
        "montecarlo_grid_independence_distribution",
        "montecarlo_final_soh_distribution",
        "montecarlo_lcoe_distribution",
        "breakeven_histogram",
        "breakeven_cdf",
        "breakeven_summary_bar",
    },
}


def _figures(name, returned):
    if name in MULTI_FIGURE:
        assert isinstance(returned, dict)
        assert set(returned) == MULTI_FIGURE[name]
        return list(returned.values())
    return [returned]


def test_every_public_plot_is_covered():
    import inspect

    public = {
        name
        for name, obj in inspect.getmembers(plotting, inspect.isfunction)
        if obj.__module__ == plotting.__name__ and not name.startswith("_")
    }
    assert public - set(PLOTS) == {"plot_pv_loss_waterfall", "set_presentation_mode"}


@pytest.mark.parametrize("name", sorted(PLOTS))
def test_without_a_directory_the_figure_is_returned_open(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    figures = _figures(name, PLOTS[name](None))

    assert figures and all(isinstance(fig, Figure) for fig in figures)
    assert sorted(fig.number for fig in figures) == sorted(plt.get_fignums())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", sorted(PLOTS))
def test_with_a_directory_the_figure_is_saved_closed_and_returned(name, tmp_path):
    figures = _figures(name, PLOTS[name](str(tmp_path)))

    assert figures and all(isinstance(fig, Figure) for fig in figures)
    assert plt.get_fignums() == []
    written = list(tmp_path.rglob("*.png"))
    assert len(written) == len(figures)
    assert all(path.stat().st_size > 0 for path in written)


def test_plots_with_nothing_to_draw_return_none_or_an_empty_dict():
    results = _results()
    assert plotting.weekly_graphs(results.iloc[:0], 10) is None
    assert plotting.plot_cell_temperature(results.drop(columns="T_cell")) is None
    assert plotting.plot_battery_soh_timeseries(results.drop(columns="Battery_SOH")) is None
    assert plotting.plot_breakeven_cdf([np.nan, np.nan]) is None
    assert plotting.plot_co2_savings(_cost_projection().drop(columns="CO2_Avoided_Total_kg")) == {}
    assert plotting.degradation_plots(pd.DataFrame()) == {}
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# PV loss waterfall
# ---------------------------------------------------------------------------


def _stage_labels(fig):
    return {text.get_text() for text in fig.axes[0].texts}


@pytest.mark.parametrize(
    ("waterfall", "shown"),
    [
        (_waterfall(bifacial_enabled=False), False),
        (_waterfall(bifacial_enabled=True), True),
        (_waterfall(bifacial_enabled=True, rear_gain_kwh=50.0), True),
        (_waterfall(), False),
        (_waterfall(rear_gain_kwh=50.0), True),
    ],
    ids=["monofacial", "bifacial-no-gain", "bifacial", "no-block-no-gain", "no-block-gain"],
)
def test_waterfall_hides_the_rear_gain_stage_of_a_monofacial_system(waterfall, shown):
    fig = plotting.plot_pv_loss_waterfall(waterfall)

    assert ("Bifacial rear gain" in _stage_labels(fig)) is shown
    assert "Cell temperature" in _stage_labels(fig)


def test_waterfall_is_closed_once_saved_and_open_otherwise(tmp_path):
    open_fig = plotting.plot_pv_loss_waterfall(_waterfall())
    assert plt.get_fignums() == [open_fig.number]
    plt.close("all")

    saved = plotting.plot_pv_loss_waterfall(_waterfall(), output_path=str(tmp_path / "waterfall.png"))
    assert isinstance(saved, Figure)
    assert plt.get_fignums() == []
    assert (tmp_path / "waterfall.png").stat().st_size > 0


# ---------------------------------------------------------------------------
# Break-even CDF and histogram
# ---------------------------------------------------------------------------


def _cdf_line(fig):
    return np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)


def test_breakeven_cdf_is_normalised_by_every_run():
    fig = plotting.plot_breakeven_cdf([6.0, 8.0, np.nan, None])

    y = _cdf_line(fig)
    np.testing.assert_allclose(y, [0.0, 0.25, 0.5])
    plateau = fig.axes[0].lines[1]
    assert plateau.get_ydata()[0] == pytest.approx(0.5)
    assert plateau.get_label() == "Ever pay back: 50.0%"
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Years to break-even"
    assert ax.get_ylabel() == "Share of runs paid back (of 4)"
    # 2.5% and 25% are reached; 50% only at the plateau; 75% and 97.5% never.
    marked = sorted(round(float(offsets[0, 1]), 3) for offsets in (c.get_offsets() for c in ax.collections))
    assert marked == [0.025, 0.25, 0.5]


def test_breakeven_cdf_takes_the_run_count_for_a_list_of_paid_back_runs():
    fig = plotting.plot_breakeven_cdf([6.0, 8.0], total_runs=8)

    assert _cdf_line(fig).max() == pytest.approx(0.25)
    with pytest.raises(ValueError, match="total_runs is 1"):
        plotting.plot_breakeven_cdf([6.0, 8.0], total_runs=1)


def test_breakeven_cdf_of_runs_that_all_pay_back_reaches_one():
    fig = plotting.plot_breakeven_cdf([6.0, 7.0, 8.0, 9.0])

    assert _cdf_line(fig).max() == pytest.approx(1.0)


def test_montecarlo_cdf_counts_runs_that_never_pay_back():
    figures = plotting.plot_montecarlo_simulation(_mc_runs(), verbose=False)

    assert _cdf_line(figures["breakeven_cdf"]).max() == pytest.approx(30 / 40)


def test_breakeven_histogram_skips_runs_that_never_pay_back():
    fig = plotting.plot_breakeven_distribution([6.0, 7.0, np.nan], 3)

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Years to break-even"
    assert sum(patch.get_height() for patch in ax.patches) == 2


# ---------------------------------------------------------------------------
# Battery SOH
# ---------------------------------------------------------------------------


def _eol_line(fig):
    (line,) = [line for line in fig.axes[0].lines if str(line.get_label()).startswith("End of Life")]
    return line


def test_soh_end_of_life_line_defaults_to_the_battery_default():
    assert DEFAULT_EOL_PERCENTAGE == pytest.approx(0.70)
    line = _eol_line(plotting.plot_battery_soh_timeseries(_results()))

    assert line.get_ydata()[0] == pytest.approx(70.0)
    assert line.get_label() == "End of Life (70%)"


def test_soh_end_of_life_line_follows_the_threshold_given():
    fig = plotting.plot_battery_soh_timeseries(_results(), eol_percentage=0.8)
    line = _eol_line(fig)

    assert line.get_ydata()[0] == pytest.approx(80.0)
    assert line.get_label() == "End of Life (80%)"
    assert fig.axes[0].get_ylim()[0] <= 75.0


@pytest.mark.parametrize("eol", [0.0, 1.0, 70.0])
def test_soh_end_of_life_must_be_a_fraction(eol):
    with pytest.raises(ValueError, match="eol_percentage"):
        plotting.plot_battery_soh_timeseries(_results(), eol_percentage=eol)


def test_battery_config_and_app_default_share_the_constant():
    from breos.app_config import APP_CONFIG_FIELDS
    from breos.battery import BatteryConfig

    assert BatteryConfig(nominal_energy_wh=5000.0).eol_percentage == DEFAULT_EOL_PERCENTAGE
    assert APP_CONFIG_FIELDS["battery_eol_percentage"].default == DEFAULT_EOL_PERCENTAGE


# ---------------------------------------------------------------------------
# Weekly profile
# ---------------------------------------------------------------------------


def test_weekly_graph_has_a_legend_and_a_title():
    fig = plotting.weekly_graphs(_results(), 10)

    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    assert len(legends) == 1
    assert [text.get_text() for text in legends[0].get_texts()] == ["PV production", "Load", "Battery energy (kWh)"]
    assert fig.axes[0].get_title(loc="left") == "Week 10: 06 Mar 2023 to 12 Mar 2023"


def test_weekly_graph_without_a_battery_has_a_two_entry_legend():
    fig = plotting.weekly_graphs(_results().drop(columns="Battery_Energy"), 10)

    (ax,) = fig.axes
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["PV production", "Load"]


# ---------------------------------------------------------------------------
# Monte Carlo NPV histogram
# ---------------------------------------------------------------------------


def _zero_lines(fig):
    return [line for line in fig.axes[0].lines if np.allclose(line.get_xdata(), 0.0)]


def test_npv_histogram_far_from_zero_keeps_its_scale_and_says_so():
    fig = plotting.plot_montecarlo_npv_distribution(_mc_runs(npv_mean=5300.0))

    ax = fig.axes[0]
    assert not _zero_lines(fig)
    assert ax.get_xlim()[0] > 4000.0
    assert any("All 40 runs are above break-even" in text.get_text() for text in ax.texts)


def test_npv_histogram_near_zero_draws_the_break_even_line():
    fig = plotting.plot_montecarlo_npv_distribution(_mc_runs(npv_mean=150.0))

    (line,) = _zero_lines(fig)
    assert line.get_label() == "Break-even (0)"
    assert not fig.axes[0].texts


@pytest.mark.parametrize(("zero_line", "count"), [(True, 1), (False, 0)])
def test_npv_histogram_zero_line_can_be_forced(zero_line, count):
    fig = plotting.plot_montecarlo_npv_distribution(_mc_runs(npv_mean=5300.0), zero_line=zero_line)

    assert len(_zero_lines(fig)) == count
    assert not fig.axes[0].texts


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


def test_pareto_front_labels_optimizer_columns_with_units():
    fig = plotting.plot_pareto_front(_pareto(), x="Projected_CO2_Avoided_SelfConsumed_kg", y="Projected_NPV")

    ax = fig.axes[0]
    assert ax.get_xlabel() == "CO$_2$ avoided by self-consumed PV over the projection (kg CO$_2$eq)"
    assert ax.get_ylabel() == "NPV savings (EUR)"


def test_pareto_front_labels_other_columns_from_their_name():
    frame = _pareto().rename(columns={"Projected_CO2_Avoided_SelfConsumed_kg": "Projected_PV_DC_Year1_kWh"})
    fig = plotting.plot_pareto_front(frame, x="Projected_PV_DC_Year1_kWh")

    assert fig.axes[0].get_xlabel() == "PV DC Year1 (kWh)"


def test_pareto_front_legend_marker_matches_coloured_points():
    fig = plotting.plot_pareto_front(_pareto(), color_by="Battery_kWh")

    legend = fig.axes[0].get_legend()
    (handle,) = legend.legend_handles
    assert legend.get_texts()[0].get_text() == "Pareto front (6)"
    assert handle.get_markerfacecolor() == "none"
    assert fig.axes[1].get_ylabel() == "Battery capacity (kWh)"


def test_pareto_front_in_one_colour_keeps_its_filled_marker():
    fig = plotting.plot_pareto_front(_pareto())

    (handle,) = fig.axes[0].get_legend().legend_handles
    np.testing.assert_allclose(handle.get_facecolor()[0], matplotlib.colors.to_rgba("tab:blue"))


# ---------------------------------------------------------------------------
# CO2 savings
# ---------------------------------------------------------------------------


def test_co2_cumulative_has_whole_year_ticks_and_labels_inside_the_axes():
    figures = plotting.plot_co2_savings(_cost_projection(years=20))
    ax = figures["co2_avoided_cumulative"].axes[0]

    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, MaxNLocator)
    ticks = [tick for tick in ax.get_xticks() if ax.get_xlim()[0] <= tick <= ax.get_xlim()[1]]
    assert ticks and all(float(tick).is_integer() for tick in ticks)
    # The final-value labels sit right of year 20, inside the widened axes.
    assert ax.get_xlim()[1] > 20.0
    assert [text.get_text() for text in ax.texts] == ["34.0 t", "16.0 t"]
    renderer = figures["co2_avoided_cumulative"].canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    for text in ax.texts:
        box = text.get_window_extent(renderer)
        assert axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1
        assert axes_box.y0 <= box.y0 and box.y1 <= axes_box.y1


def test_co2_yearly_ticks_are_the_projection_years():
    ax = plotting.plot_co2_savings(_cost_projection(years=5))["co2_avoided_yearly"].axes[0]

    assert [label.get_text() for label in ax.get_xticklabels()] == ["1", "2", "3", "4", "5"]
