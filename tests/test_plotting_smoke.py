"""Smoke tests for every public function in ``breos.plotting``.

Each test builds the smallest realistic input the function documents, calls it
with the Agg backend and checks that it wrote its figure. Inputs come from one
offline App run and one small offline Monte Carlo study where the function
consumes those result shapes; the rest follow the docstrings.
``plot_pv_loss_waterfall`` and ``plot_montecarlo_simulation`` have their own
tests in ``test_plotting.py`` and ``test_montecarlo_plotting.py``.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt  # noqa: E402

import breos.app as app_module  # noqa: E402
import breos.runners.app as app_runner  # noqa: E402
from breos import App, plotting  # noqa: E402
from breos.montecarlo import MonteCarloSettings, run_montecarlo  # noqa: E402
from tests.conftest import _build_synthetic_weather  # noqa: E402

TESTED_ELSEWHERE = {"plot_pv_loss_waterfall", "plot_montecarlo_simulation"}

# Cheap enough that both the App run and every Monte Carlo trajectory pay back
# inside the short projection, so the break-even plots have points to draw.
_BASE_CONFIG = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "cost_preset": "residential_pt",
    "costs": {"electricity_cost": 0.6, "storage_cost_per_kwh": 100.0, "installation_cost_per_module": 50.0},
    "emissions_country": "PT",
    "resolution": "h",
    "projection_years": 3,
}

_DEAD_PVBAT_REASON = "#186: dead pvbat leftover; BREOS never produces the input schema it expects"
_DEAD_LEGACY_MC_REASON = (
    "#186: reachable only through the legacy run-year branch of plot_montecarlo_simulation, "
    "whose run_number/year columns BREOS never produces"
)


def _fake_tmy(weather):
    def _fetch(*args, **kwargs):
        return weather.copy(), {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}

    return _fetch


def _run_app(config, weather):
    """Run App offline and keep the frames it builds its result from."""
    captured = {"degradation": []}
    run_simulation = app_runner.run_app_simulation
    simulate_energy_balance = app_runner.simulate_energy_balance

    def _capture_run(*args):
        captured["artifacts"] = run_simulation(*args)
        return captured["artifacts"]

    def _capture_year(*args, **kwargs):
        output = simulate_energy_balance(*args, **kwargs)
        captured["degradation"].append(output[5])
        return output

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_module, "fetch_tmy_weather_data", _fake_tmy(weather))
        mp.setattr(app_module, "load_weather", lambda **kw: None)
        mp.setattr(app_module, "run_app_simulation", _capture_run)
        mp.setattr(app_runner, "simulate_energy_balance", _capture_year)
        App(config).simulate()

    artifacts = captured["artifacts"]
    return SimpleNamespace(
        results=artifacts.first_year_results_df,
        cost_projection=artifacts.cost_projection,
        costs=artifacts.costs,
        yearly=artifacts.yearly_df,
        payback_year=artifacts.payback_year,
        degradation=captured["degradation"][0],
    )


@pytest.fixture(scope="module")
def battery_run():
    config = {**_BASE_CONFIG, "battery_kwh": 5.0, "enable_resistance_fade": True}
    return _run_app(config, _build_synthetic_weather())


@pytest.fixture(scope="module")
def pv_only_run():
    return _run_app({**_BASE_CONFIG, "battery_kwh": 0.0}, _build_synthetic_weather())


@pytest.fixture(scope="module")
def historical_weather():
    """Two synthetic historical years; the second is 6 % sunnier."""
    first = _build_synthetic_weather(2021)
    second = _build_synthetic_weather(2022)
    second[["ghi", "dni", "dhi"]] *= 1.06
    return pd.concat([first, second])


@pytest.fixture(scope="module")
def mc_runs(tmp_path_factory, historical_weather):
    """One-row-per-run table from a small offline Monte Carlo study."""
    weather_file = tmp_path_factory.mktemp("mc") / "weather.csv"
    historical_weather.rename_axis("date").reset_index().to_csv(weather_file, index=False)
    settings = MonteCarloSettings(weather_file=str(weather_file), n_runs=5, seed=1, load_uncertainty=0.2)
    return run_montecarlo({**_BASE_CONFIG, "battery_kwh": 5.0}, settings).runs


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _assert_written(directory, *names):
    for name in names:
        path = directory / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def _monthly_ghi_kwh_m2(weather):
    return weather["ghi"].resample("ME").sum() / 1000.0


def test_every_public_plotting_function_has_a_smoke_test():
    public = {
        name
        for name, obj in inspect.getmembers(plotting, inspect.isfunction)
        if obj.__module__ == plotting.__name__ and not name.startswith("_")
    }
    missing = sorted(name for name in public - TESTED_ELSEWHERE if f"test_{name}" not in globals())
    assert not missing


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------


def test_set_presentation_mode():
    with matplotlib.rc_context():
        plotting.set_presentation_mode(True, scale=2.0)
        assert plt.rcParams["font.size"] == 28
        plotting.set_presentation_mode(False)
        assert plt.rcParams["font.size"] == matplotlib.rcParamsDefault["font.size"]
        assert matplotlib.get_backend().lower() == "agg"


# ---------------------------------------------------------------------------
# App energy balance (first-year results frame)
# ---------------------------------------------------------------------------


def test_monthly_graphs(battery_run, tmp_path):
    plotting.monthly_graphs(battery_run.results, str(tmp_path))
    _assert_written(tmp_path, "monthly_energy.png")


def test_yearly_graphs(battery_run, tmp_path):
    plotting.yearly_graphs(battery_run.results, str(tmp_path))
    _assert_written(tmp_path, "yearly_energy.png")


def test_weekly_graphs(battery_run, tmp_path):
    plotting.weekly_graphs(battery_run.results, 10, str(tmp_path))
    _assert_written(tmp_path, "week_10_profile.png")


def test_plot_monthly_comparison(battery_run, tmp_path):
    plotting.plot_monthly_comparison(battery_run.results, str(tmp_path), scenario_name="battery")
    _assert_written(tmp_path, "monthly_comparison_battery.png")


def test_plot_monthly_balance(battery_run, tmp_path):
    plotting.plot_monthly_balance(battery_run.results, str(tmp_path))
    _assert_written(tmp_path, "monthly_balance.png")


def test_plot_timeseries(battery_run, tmp_path):
    frame = battery_run.results.set_index("Datetime")
    plotting.plot_timeseries(frame, ["PV_Production", "Houseload"], str(tmp_path))
    _assert_written(tmp_path, "timeseries.png")


def test_plot_cell_temperature(battery_run, tmp_path):
    plotting.plot_cell_temperature(battery_run.results, str(tmp_path))
    _assert_written(tmp_path, "battery_cell_temperature.png")


def test_plot_battery_soh_timeseries(battery_run, tmp_path):
    plotting.plot_battery_soh_timeseries(battery_run.results, str(tmp_path), scenario_name="battery")
    _assert_written(tmp_path, "battery_soh_timeseries_battery.png")


def test_plot_battery_soh_timeseries_date_window(battery_run, tmp_path):
    plotting.plot_battery_soh_timeseries(
        battery_run.results, str(tmp_path), start_date="2023-03-01", end_date="2023-06-30"
    )


# ---------------------------------------------------------------------------
# App degradation (daily degradation frame)
# ---------------------------------------------------------------------------


def test_degradation_plots(battery_run, tmp_path):
    plotting.degradation_plots(battery_run.degradation, str(tmp_path))
    _assert_written(
        tmp_path,
        "battery_degradation_soh.png",
        "battery_degradation_components_per_battery.png",
        "battery_degradation_fec.png",
        "battery_resistance_growth.png",
        "battery_effective_rte.png",
    )


def test_plot_resistance_and_efficiency(battery_run, tmp_path):
    plotting.plot_resistance_and_efficiency(battery_run.degradation, str(tmp_path))
    _assert_written(tmp_path, "battery_resistance_growth.png", "battery_effective_rte.png")


def test_plot_calendar_aging_sensitivity(battery_run, tmp_path):
    soh = battery_run.yearly["Battery_SOH_%"].to_numpy()
    trajectories = {f"k₀ × {scale}": list(100 - scale * (100 - soh)) for scale in (0.5, 1, 2)}
    plotting.plot_calendar_aging_sensitivity(trajectories, 80.0, str(tmp_path))
    _assert_written(tmp_path, "calendar_aging_sensitivity.png")


# ---------------------------------------------------------------------------
# Degradation validation (measured vs predicted SOH)
# ---------------------------------------------------------------------------


@pytest.fixture
def soh_pair(battery_run):
    """Daily predicted SOH from App and a sparse noisy 'measured' series."""
    predicted = battery_run.degradation.set_index("Datetime")["SOH"] / 100.0
    measured = predicted.iloc[::30] + np.random.default_rng(0).normal(0.0, 0.002, len(predicted.iloc[::30]))
    return measured, predicted.loc[measured.index]


def test_plot_validation_soh_comparison(soh_pair, tmp_path):
    measured, predicted = soh_pair
    plotting.plot_validation_soh_comparison(
        measured, predicted, str(tmp_path), x_label="Date", metrics={"RMSE": 0.002, "MAE": 0.0015, "R2": 0.9}
    )
    _assert_written(tmp_path, "validation_soh_comparison.png")


def test_plot_validation_residuals(soh_pair, tmp_path):
    plotting.plot_validation_residuals(*soh_pair, str(tmp_path))
    _assert_written(tmp_path, "validation_residuals.png")


def test_plot_validation_parity(soh_pair, tmp_path):
    plotting.plot_validation_parity(*soh_pair, str(tmp_path), metrics={"R2": 0.9})
    _assert_written(tmp_path, "validation_parity.png")


@pytest.mark.xfail(
    not hasattr(matplotlib.cm, "get_cmap"),
    raises=AttributeError,
    strict=True,
    reason="#219: plt.cm.get_cmap was removed in matplotlib 3.11",
)
def test_plot_validation_multi_system(soh_pair, tmp_path):
    measured, predicted = soh_pair
    system = {
        "simulation": pd.DataFrame({"date": predicted.index, "predicted_soh": predicted.to_numpy()}),
        "truth": pd.DataFrame({"date": measured.index, "measured_soh": measured.to_numpy()}),
    }
    plotting.plot_validation_multi_system({"A": system, "B": system}, str(tmp_path))
    _assert_written(tmp_path, "validation_multi_system.png")


def test_plot_validation_degradation_split(battery_run, tmp_path):
    frame = battery_run.degradation
    simulation = pd.DataFrame(
        {
            "date": frame["Datetime"],
            "cal_loss": frame["Cumulative_Calendar_Degradation"],
            "cycle_loss": frame["Cumulative_Cycle_Degradation"],
        }
    )
    plotting.plot_validation_degradation_split(simulation, str(tmp_path), system_label="A")
    _assert_written(tmp_path, "validation_degradation_split_A.png")


# ---------------------------------------------------------------------------
# App economics and emissions (cost projection)
# ---------------------------------------------------------------------------


def test_create_cost_plots(battery_run, tmp_path):
    plotting.create_cost_plots(
        battery_run.cost_projection, battery_run.costs["total_initial_cost"], str(tmp_path), scenario_name="battery"
    )
    _assert_written(tmp_path, "cost_projection_battery.png")


def test_plot_breakeven(battery_run, tmp_path):
    plotting.plot_breakeven(battery_run.cost_projection, str(tmp_path), scenario_name="battery")
    _assert_written(tmp_path, "breakeven_cumulative_battery.png", "breakeven_annual_battery.png")


def test_plot_breakeven_comparison(battery_run, pv_only_run, tmp_path):
    plotting.plot_breakeven_comparison(
        [battery_run.cost_projection, pv_only_run.cost_projection],
        ["PV + battery", "PV only"],
        ["tab:blue", "tab:orange"],
        str(tmp_path),
    )
    _assert_written(tmp_path, "breakeven_comparison.png")


def test_plot_breakeven_two(battery_run, pv_only_run, tmp_path):
    plotting.plot_breakeven_two(
        battery_run.cost_projection,
        "PV + battery",
        battery_run.payback_year,
        pv_only_run.cost_projection,
        "PV only",
        pv_only_run.payback_year,
        str(tmp_path),
    )
    _assert_written(tmp_path, "breakeven_two.png")


def test_plot_co2_savings(battery_run, tmp_path):
    plotting.plot_co2_savings(battery_run.cost_projection, str(tmp_path), scenario_name="battery")
    _assert_written(tmp_path, "co2_avoided_yearly_battery.png", "co2_avoided_cumulative_battery.png")


# ---------------------------------------------------------------------------
# Monte Carlo (one-row-per-run table)
# ---------------------------------------------------------------------------


def test_plot_montecarlo_npv_distribution(mc_runs, tmp_path):
    plotting.plot_montecarlo_npv_distribution(mc_runs, str(tmp_path))
    _assert_written(tmp_path, "montecarlo_npv_distribution.png")


def test_plot_montecarlo_grid_independence_distribution(mc_runs, tmp_path):
    plotting.plot_montecarlo_grid_independence_distribution(mc_runs, str(tmp_path))
    _assert_written(tmp_path, "montecarlo_grid_independence_distribution.png")


def test_plot_montecarlo_final_soh_distribution(mc_runs, tmp_path):
    plotting.plot_montecarlo_final_soh_distribution(mc_runs, str(tmp_path))
    _assert_written(tmp_path, "montecarlo_final_soh_distribution.png")


def test_plot_breakeven_distribution(mc_runs, tmp_path):
    payback = mc_runs["payback_year_exact"].dropna().tolist()
    assert payback
    plotting.plot_breakeven_distribution(payback, len(mc_runs), str(tmp_path))
    _assert_written(tmp_path, "breakeven_histogram.png")


def test_plot_breakeven_cdf(mc_runs, tmp_path):
    payback = mc_runs["payback_year_exact"].dropna().tolist()
    assert payback
    plotting.plot_breakeven_cdf(payback, str(tmp_path))
    _assert_written(tmp_path, "breakeven_cdf.png")


def test_plot_breakeven_summary_bar(mc_runs, tmp_path):
    plotting.plot_breakeven_summary_bar(int(mc_runs["payback_year"].notna().sum()), len(mc_runs), str(tmp_path))
    _assert_written(tmp_path, "breakeven_summary_bar.png")


@pytest.mark.skip(reason=_DEAD_LEGACY_MC_REASON)
def test_plot_montecarlo_cost_overlay():
    pass


@pytest.mark.skip(reason=_DEAD_LEGACY_MC_REASON)
def test_plot_montecarlo_soh_overlay():
    pass


@pytest.mark.skip(reason=_DEAD_LEGACY_MC_REASON)
def test_plot_montecarlo_soh_traces():
    pass


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


def test_plot_weather_monthly_comparison(historical_weather, tmp_path):
    monthly = _monthly_ghi_kwh_m2(historical_weather)
    by_month = monthly.groupby(monthly.index.month)
    mean = by_month.mean()
    half_width = 1.96 * by_month.std() / np.sqrt(by_month.count())
    stats = pd.DataFrame(
        {
            "mean": mean,
            "ci_low": mean - half_width,
            "ci_high": mean + half_width,
            "min": by_month.min(),
            "max": by_month.max(),
        }
    )
    tmy = _monthly_ghi_kwh_m2(_build_synthetic_weather()).to_numpy()
    plotting.plot_weather_monthly_comparison(
        tmy, stats, "GHI (kWh/m²)", "pvgis-sarah3", str(tmp_path), "monthly_ghi_comparison.png"
    )
    _assert_written(tmp_path, "monthly_ghi_comparison.png")


def test_plot_weather_annual_ghi_distribution(historical_weather, tmp_path):
    annual = historical_weather["ghi"].groupby(historical_weather.index.year).sum() / 1000.0
    tmy_annual = _build_synthetic_weather()["ghi"].sum() / 1000.0
    plotting.plot_weather_annual_ghi_distribution(annual, tmy_annual, float(annual.mean()), str(tmp_path))
    _assert_written(tmp_path, "annual_ghi_distribution.png")


# ---------------------------------------------------------------------------
# Orientation and sizing sweeps
# ---------------------------------------------------------------------------


@pytest.fixture
def orientation_grid():
    """Azimuth x tilt lattice with a smooth optimum near south at 35 degrees."""
    azimuth, tilt = np.meshgrid([90.0, 135.0, 180.0, 225.0, 270.0], [0.0, 20.0, 35.0, 50.0])
    metric = 1500.0 * np.cos(np.radians(azimuth - 180.0) / 2) * np.cos(np.radians(tilt - 35.0))
    return pd.DataFrame({"Azimuth": azimuth.ravel(), "Tilt": tilt.ravel(), "Metric": metric.ravel()})


def test_plot_tilt_optimization(orientation_grid, tmp_path):
    south = orientation_grid[orientation_grid["Azimuth"] == 180.0].reset_index(drop=True)
    tilt_results = south.rename(columns={"Metric": "Total_PV_Production_kWh"})
    plotting.plot_tilt_optimization(tilt_results, str(tmp_path), scenario_name="porto")
    _assert_written(tmp_path, "tilt_optimization_porto.png")


def test_plot_azitilt_landscape_2d(orientation_grid, tmp_path):
    plotting.plot_azitilt_landscape_2d(orientation_grid, 180.0, 35.0, str(tmp_path))
    _assert_written(tmp_path, "optimization_landscape_2d.png")


def test_plot_azitilt_landscape_3d(orientation_grid, tmp_path):
    best = orientation_grid.loc[orientation_grid["Metric"].idxmax()]
    plotting.plot_azitilt_landscape_3d(orientation_grid, best["Azimuth"], best["Tilt"], best["Metric"], str(tmp_path))
    _assert_written(tmp_path, "optimization_landscape_3d.png")


def test_plot_azitilt_ew_1d(orientation_grid, tmp_path):
    east = orientation_grid[orientation_grid["Azimuth"] == 90.0]
    best = east.loc[east["Metric"].idxmax()]
    plotting.plot_azitilt_ew_1d(
        east["Tilt"].to_numpy(), east["Metric"].tolist(), best["Tilt"], best["Metric"], str(tmp_path)
    )
    _assert_written(tmp_path, "optimization_1d_tilt_ew.png")


@pytest.fixture
def sizing_pivot():
    """Grid independence (%) with battery kWh as index and module count as columns."""
    return pd.DataFrame(
        [[28.0, 32.5, 35.1], [41.2, 52.8, 58.3], [44.0, 60.1, 68.9]],
        index=[0.0, 5.0, 10.0],
        columns=[4, 8, 12],
    )


def test_plot_grid_independence_heatmap(sizing_pivot, tmp_path):
    plotting.plot_grid_independence_heatmap(sizing_pivot, str(tmp_path), "Porto", vmin=0.0, vmax=100.0)
    _assert_written(tmp_path, "grid_independence_heatmap.png")


def test_plot_location_comparison_delta(sizing_pivot, tmp_path):
    delta = sizing_pivot - sizing_pivot.to_numpy().mean()
    plotting.plot_location_comparison_delta(delta, str(tmp_path), "Porto", "Berlin")
    _assert_written(tmp_path, "grid_independence_delta.png")


# Matplotlib 3.11 returns the colour cycle as RGB tuples, which scatter's c=
# reads as values to colour-map when a group has exactly three points.
@pytest.mark.filterwarnings("ignore:\\*c\\* argument looks like a single numeric RGB")
def test_plot_pareto_front_analysis(tmp_path):
    rng = np.random.default_rng(0)
    rows = [
        {
            "Consumption_kWh": consumption,
            "Tariff": tariff,
            "Detailed_Strategy": strategy,
            "Net_Cost_Eur": consumption * 0.2 - 40.0 * point + rng.normal(0.0, 20.0),
            "Grid_Independence_%": 20.0 + 8.0 * point + rng.normal(0.0, 2.0),
        }
        for consumption in (3000, 5000)
        for tariff in ("flat", "bi-hourly")
        for strategy in ("PV only", "PV + battery")
        for point in range(4)
    ]
    plotting.plot_pareto_front_analysis(pd.DataFrame(rows), [3000, 5000], str(tmp_path))
    _assert_written(tmp_path, "pareto_front_refined.png", "pareto_front_3000.csv", "pareto_front_5000.csv")


# ---------------------------------------------------------------------------
# Tariff comparison
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason=_DEAD_PVBAT_REASON)
def test_plot_tariff_comparison():
    pass


@pytest.mark.skip(reason=_DEAD_PVBAT_REASON)
def test_plot_tariff_comparison_manual():
    pass


def _berlin_csv_frame(frame, tmp_path, name):
    """Round-trip a results frame through CSV on Berlin time: text with two UTC offsets."""
    berlin = frame.copy()
    berlin["Datetime"] = pd.DatetimeIndex(berlin["Datetime"]).tz_convert("Europe/Berlin")
    path = tmp_path / f"{name}.csv"
    berlin.to_csv(path, index=False)
    return pd.read_csv(path)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda r, d: plotting.monthly_graphs(r, d), "monthly_energy.png"),
        (lambda r, d: plotting.yearly_graphs(r, d), "yearly_energy.png"),
        (lambda r, d: plotting.weekly_graphs(r, 13, d), "week_13_profile.png"),
        (lambda r, d: plotting.plot_cell_temperature(r, d), "battery_cell_temperature.png"),
        (
            lambda r, d: plotting.plot_battery_soh_timeseries(r, d, scenario_name="dst"),
            "battery_soh_timeseries_dst.png",
        ),
        (lambda r, d: plotting.plot_monthly_comparison(r, d, scenario_name="dst"), "monthly_comparison_dst.png"),
        (lambda r, d: plotting.plot_monthly_balance(r, d), "monthly_balance.png"),
    ],
    ids=["monthly", "yearly", "weekly", "cell_temperature", "soh", "monthly_comparison", "monthly_balance"],
)
def test_plots_read_dst_crossing_result_csvs(battery_run, tmp_path, call, expected):
    # A DST-zone run writes two UTC offsets, and pandas 3 refused to parse
    # them as one column (#216).
    results = _berlin_csv_frame(battery_run.results, tmp_path, "results")
    assert results["Datetime"].str.endswith("+02:00").any() and results["Datetime"].str.endswith("+01:00").any()

    call(results, str(tmp_path))
    _assert_written(tmp_path, expected)


def test_degradation_plots_read_dst_crossing_csvs(battery_run, tmp_path):
    degradation = _berlin_csv_frame(battery_run.degradation, tmp_path, "degradation")

    plotting.degradation_plots(degradation, str(tmp_path))
    plotting.plot_resistance_and_efficiency(degradation, str(tmp_path))
    _assert_written(tmp_path, "battery_degradation_soh.png", "battery_resistance_growth.png")


def test_load_results_reads_dst_csvs_and_path_inputs(battery_run, tmp_path):
    from breos.io import load_results

    berlin = battery_run.results.copy()
    berlin["Datetime"] = pd.DatetimeIndex(berlin["Datetime"]).tz_convert("Europe/Berlin")
    path = tmp_path / "results.csv"
    berlin.to_csv(path, index=False)

    loaded = load_results(path)  # a Path, which used to raise AttributeError

    assert len(loaded) == len(berlin)
    # Each row keeps its Berlin wall-clock time.
    np.testing.assert_array_equal(
        loaded.index.to_numpy(), pd.DatetimeIndex(berlin["Datetime"]).tz_localize(None).to_numpy()
    )
    assert load_results(str(path)).index.equals(loaded.index)
