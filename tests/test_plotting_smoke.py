"""Smoke tests for every public function in ``breos.plotting``.

Each test builds the smallest realistic input the function documents, calls it
with the Agg backend and checks that it wrote its figure. Inputs come from one
offline App run, one small offline Monte Carlo study and two small offline
``breos sweep`` runs where the function consumes those result shapes; the rest
follow the docstrings.
``plot_pv_loss_waterfall`` and ``plot_montecarlo_simulation`` have their own
tests in ``test_plotting.py`` and ``test_montecarlo_plotting.py``.
"""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt  # noqa: E402

import breos.app as app_module  # noqa: E402
import breos.projection as projection_module  # noqa: E402
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


def _fake_tmy(weather):
    def _fetch(*args, **kwargs):
        return weather.copy(), {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}

    return _fetch


def _run_app(config, weather):
    """Run App offline and keep the frames it builds its result from."""
    captured = {"degradation": []}
    run_simulation = app_runner.run_app_simulation
    simulate_energy_balance = projection_module.simulate_energy_balance

    def _capture_run(*args):
        captured["artifacts"] = run_simulation(*args)
        return captured["artifacts"]

    def _capture_year(*args, **kwargs):
        output = simulate_energy_balance(*args, **kwargs)
        captured["degradation"].append(output[4])
        return output

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_module, "fetch_tmy_weather_data", _fake_tmy(weather))
        mp.setattr(app_module, "load_weather", lambda **kw: None)
        mp.setattr(app_module, "run_app_simulation", _capture_run)
        mp.setattr(projection_module, "simulate_energy_balance", _capture_year)
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
        backend = matplotlib.get_backend()
        plotting.set_presentation_mode(True, scale=2.0)
        assert plt.rcParams["font.size"] == 28
        plotting.set_presentation_mode(False)
        assert plt.rcParams["font.size"] == matplotlib.rcParamsDefault["font.size"]
        assert matplotlib.get_backend() == backend


# ---------------------------------------------------------------------------
# App energy balance (first-year results frame)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# App economics and emissions (cost projection)
# ---------------------------------------------------------------------------


def test_plot_breakeven(battery_run, tmp_path):
    plotting.plot_breakeven(battery_run.cost_projection, str(tmp_path), scenario_name="battery")
    _assert_written(tmp_path, "breakeven_cumulative_battery.png", "breakeven_annual_battery.png")


def test_plot_breakeven_comparison(battery_run, pv_only_run, tmp_path):
    plotting.plot_breakeven_comparison(
        [battery_run.cost_projection, pv_only_run.cost_projection], ["PV + battery", "PV only"], str(tmp_path)
    )
    _assert_written(tmp_path, "breakeven_comparison.png")


def test_plot_breakeven_comparison_of_app_results(tmp_path):
    results = []
    for battery_kwh in (0.0, 5.0):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(app_module, "fetch_tmy_weather_data", _fake_tmy(_build_synthetic_weather()))
            mp.setattr(app_module, "load_weather", lambda **kw: None)
            app = App({**_BASE_CONFIG, "battery_kwh": battery_kwh})
            app.simulate()
        results.append(app.result())
    plotting.plot_breakeven_comparison(results, ["PV only", "PV + battery"], str(tmp_path), filename="app.png")
    _assert_written(tmp_path, "app.png")


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
    payback = mc_runs["payback_year_interpolated"].dropna().tolist()
    assert payback
    plotting.plot_breakeven_distribution(payback, len(mc_runs), str(tmp_path))
    _assert_written(tmp_path, "breakeven_histogram.png")


def test_plot_breakeven_cdf(mc_runs, tmp_path):
    payback = mc_runs["payback_year_interpolated"].dropna().tolist()
    assert payback
    plotting.plot_breakeven_cdf(payback, str(tmp_path))
    _assert_written(tmp_path, "breakeven_cdf.png")


def test_plot_breakeven_summary_bar(mc_runs, tmp_path):
    plotting.plot_breakeven_summary_bar(int(mc_runs["payback_year"].notna().sum()), len(mc_runs), str(tmp_path))
    _assert_written(tmp_path, "breakeven_summary_bar.png")


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def historical_weather_file(tmp_path_factory, historical_weather):
    """The multi-year weather CSV a Monte Carlo study reads."""
    weather_file = tmp_path_factory.mktemp("weather") / "historical.csv"
    historical_weather.rename_axis("date").reset_index().to_csv(weather_file, index=False)
    return weather_file


def test_plot_weather_monthly_comparison(historical_weather_file, tmp_path):
    plotting.plot_weather_monthly_comparison(_build_synthetic_weather(), historical_weather_file, str(tmp_path))
    plotting.plot_weather_monthly_comparison(
        _build_synthetic_weather(), historical_weather_file, str(tmp_path), variable="temp_air"
    )
    _assert_written(tmp_path, "weather_monthly_ghi.png", "weather_monthly_temp_air.png")


def test_plot_weather_annual_ghi_distribution(historical_weather_file, tmp_path):
    plotting.plot_weather_annual_ghi_distribution(_build_synthetic_weather(), historical_weather_file, str(tmp_path))
    _assert_written(tmp_path, "annual_ghi_distribution.png")


# ---------------------------------------------------------------------------
# Sweeps and the optimizer front
# ---------------------------------------------------------------------------


def _run_sweep(directory, config):
    """Run ``breos sweep`` offline on synthetic weather; return its CSV."""
    from breos.cli import main

    config_file = directory / "sweep.json"
    config_file.write_text(json.dumps(config))
    output = directory / "sweep.csv"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_module, "fetch_tmy_weather_data", _fake_tmy(_build_synthetic_weather()))
        mp.setattr(app_module, "load_weather", lambda **kw: None)
        assert main(["sweep", "--config", str(config_file), "--output", str(output)]) == 0
    return output


@pytest.fixture(scope="module")
def sizing_sweep(tmp_path_factory):
    """Module count x battery capacity."""
    sweep = {"n_modules": [4, 8], "battery_kwh": [0.0, 5.0]}
    return _run_sweep(tmp_path_factory.mktemp("sizing"), {**_BASE_CONFIG, "sweep": sweep})


@pytest.fixture(scope="module")
def orientation_sweep(tmp_path_factory):
    """Tilt x azimuth, PV only."""
    sweep = {"tilt": [10, 35, 60], "azimuth": [90, 180, 270]}
    return _run_sweep(tmp_path_factory.mktemp("orientation"), {**_BASE_CONFIG, "battery_kwh": 0.0, "sweep": sweep})


def test_plot_sweep_heatmap(sizing_sweep, tmp_path):
    plotting.plot_sweep_heatmap(sizing_sweep, "grid_independence_pct", str(tmp_path))
    # A second sweep whose NPV is 100 lower per kWh of battery: real, non-zero differences.
    other = pd.read_csv(sizing_sweep)
    other["npv_savings"] -= 100.0 * other["param_battery_kwh"]
    with pytest.MonkeyPatch.context() as mp:
        drawn = []
        mp.setattr(plotting.plt, "close", lambda *args, **kwargs: drawn.append(plotting.plt.gcf()))
        plotting.plot_sweep_heatmap(sizing_sweep, "npv_savings", str(tmp_path), diff=other, currency="EUR")
    image = drawn[0].axes[0].images[0]
    np.testing.assert_allclose(np.ma.filled(image.get_array(), np.nan), [[0.0, 0.0], [500.0, 500.0]], atol=1e-6)
    assert (image.norm.vmin, image.norm.vmax) == pytest.approx((-500.0, 500.0))
    assert drawn[0].axes[1].get_ylabel() == "NPV savings difference (EUR)"
    plotting.plt.close("all")
    _assert_written(tmp_path, "sweep_grid_independence_pct.png", "sweep_npv_savings_diff.png")


def test_plot_orientation_landscape(orientation_sweep, tmp_path):
    plotting.plot_orientation_landscape(orientation_sweep, "usable_ac_system_production_kwh", str(tmp_path))
    tilt_only = pd.read_csv(orientation_sweep).query("param_azimuth == 180")
    plotting.plot_orientation_landscape(
        tilt_only.drop(columns="param_azimuth"),
        "usable_ac_system_production_kwh",
        str(tmp_path),
        filename="tilt.png",
    )
    _assert_written(tmp_path, "orientation_landscape.png", "tilt.png")


def test_plot_pareto_front(sizing_sweep, tmp_path):
    plotting.plot_pareto_front(
        sizing_sweep, str(tmp_path), x="grid_independence_pct", y="npv_savings", color_by="battery_kwh"
    )
    _assert_written(tmp_path, "pareto_front.png")


def test_plot_cell_temperature_leaves_months_without_data_empty(tmp_path, monkeypatch):
    # Months without data used to be drawn at 0 °C (#219).
    index = pd.date_range("2025-06-01", "2025-06-30 23:00", freq="h", tz="UTC")
    results = pd.DataFrame({"Datetime": index, "T_cell": 20.0})
    drawn = []
    monkeypatch.setattr(plotting.plt, "close", lambda *args, **kwargs: drawn.append(plotting.plt.gcf()))

    plotting.plot_cell_temperature(results, str(tmp_path))

    mean_line = drawn[0].axes[0].lines[0]
    y = np.asarray(mean_line.get_ydata(), dtype=float)
    assert y[5] == pytest.approx(20.0)
    assert np.isnan(np.delete(y, 5)).all()
    plotting.plt.close("all")


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
    ids=["yearly", "weekly", "cell_temperature", "soh", "monthly_comparison", "monthly_balance"],
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
    # Mixed offsets: the wall-clock index repeats an hour, so the instants are kept.
    assert loaded["Datetime_UTC"].is_monotonic_increasing
    assert loaded["Datetime_UTC"].diff().dropna().nunique() == 1


def test_load_results_keeps_one_zone_files_as_they_were(battery_run, tmp_path):
    from breos.io import load_results

    path = tmp_path / "results.csv"
    battery_run.results.to_csv(path, index=False)

    loaded = load_results(path)

    assert "Datetime_UTC" not in loaded.columns
    assert str(loaded.index.tz) == "UTC"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda r, d: plotting.yearly_graphs(r, d), "yearly_energy.png"),
        (lambda r, d: plotting.plot_monthly_comparison(r, d, scenario_name="dst"), "monthly_comparison_dst.png"),
        (lambda r, d: plotting.plot_monthly_balance(r, d), "monthly_balance.png"),
    ],
    ids=["yearly", "monthly_comparison", "monthly_balance"],
)
def test_energy_plots_read_dst_csvs_loaded_with_load_results(battery_run, tmp_path, call, expected):
    # load_results turns the Datetime column into a wall-clock index, which
    # repeats an hour in autumn; the plots used to find no regular step.
    from breos.io import load_results

    berlin = battery_run.results.copy()
    berlin["Datetime"] = pd.DatetimeIndex(berlin["Datetime"]).tz_convert("Europe/Berlin")
    path = tmp_path / "results.csv"
    berlin.to_csv(path, index=False)

    call(load_results(path), str(tmp_path))
    _assert_written(tmp_path, expected)
