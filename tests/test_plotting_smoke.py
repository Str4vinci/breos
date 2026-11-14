"""The plots read the frames the real producers write.

Inputs come from one offline App run, one small offline Monte Carlo study and
two small offline ``breos sweep`` runs, so a renamed or reshaped producer
column fails here. That every plot writes, closes and returns its figure is
owned by ``test_plotting_figures.py`` on small synthetic frames.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt  # noqa: E402

import breos.app as app_module  # noqa: E402
import breos.projection as projection_module  # noqa: E402
import breos.runners.app as app_runner  # noqa: E402
from breos import App, plotting  # noqa: E402
from breos.montecarlo import MonteCarloSettings, run_montecarlo  # noqa: E402
from tests.conftest import _build_synthetic_weather, write_instant_at_labels_sidecar  # noqa: E402

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


@contextmanager
def _offline():
    """Serve App the synthetic TMY instead of fetching weather."""
    weather = _build_synthetic_weather()
    site = {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_module, "fetch_tmy_weather_data", lambda *args, **kwargs: (weather.copy(), site))
        mp.setattr(app_module, "load_weather", lambda **kw: None)
        yield mp


def _run_app(config):
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

    with _offline() as mp:
        mp.setattr(app_module, "run_app_simulation", _capture_run)
        mp.setattr(projection_module, "simulate_energy_balance", _capture_year)
        App(config).simulate()

    artifacts = captured["artifacts"]
    return SimpleNamespace(
        results=artifacts.first_year_results_df,
        cost_projection=artifacts.cost_projection,
        degradation=captured["degradation"][0],
    )


@pytest.fixture(scope="module")
def battery_run():
    return _run_app({**_BASE_CONFIG, "battery_kwh": 5.0, "enable_resistance_fade": True})


@pytest.fixture(scope="module")
def pv_only_run():
    return _run_app({**_BASE_CONFIG, "battery_kwh": 0.0})


@pytest.fixture(scope="module")
def historical_weather():
    """Two synthetic historical years; the second is 6 % sunnier."""
    first = _build_synthetic_weather(2021)
    second = _build_synthetic_weather(2022)
    second[["ghi", "dni", "dhi"]] *= 1.06
    return pd.concat([first, second])


@pytest.fixture(scope="module")
def historical_weather_file(tmp_path_factory, historical_weather):
    """The multi-year weather CSV a Monte Carlo study reads."""
    weather_file = tmp_path_factory.mktemp("weather") / "historical.csv"
    historical_weather.rename_axis("date").reset_index().to_csv(weather_file, index=False)
    write_instant_at_labels_sidecar(weather_file)
    return weather_file


@pytest.fixture(scope="module")
def mc_runs(historical_weather_file):
    """One-row-per-run table from a small offline Monte Carlo study."""
    settings = MonteCarloSettings(weather_file=str(historical_weather_file), n_runs=5, seed=1, load_uncertainty=0.2)
    return run_montecarlo({**_BASE_CONFIG, "battery_kwh": 5.0}, settings).runs


def _run_sweep(directory, config):
    """Run ``breos sweep`` offline on synthetic weather; return its CSV."""
    from breos.cli import main

    config_file = directory / "sweep.json"
    config_file.write_text(json.dumps(config))
    output = directory / "sweep.csv"
    with _offline():
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


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _assert_written(directory, *names):
    for name in names:
        path = directory / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


# Each plot that reads a producer's frame as it is, called with a fixture
# getter and the output directory, and the files it writes.
REAL_PLOTS = {
    "breakeven": (
        lambda get, d: plotting.plot_breakeven(get("battery_run").cost_projection, d, scenario_name="battery"),
        ["breakeven_cumulative_battery.png", "breakeven_annual_battery.png"],
    ),
    "breakeven_comparison": (
        lambda get, d: plotting.plot_breakeven_comparison(
            [get("battery_run").cost_projection, get("pv_only_run").cost_projection], ["PV + battery", "PV only"], d
        ),
        ["breakeven_comparison.png"],
    ),
    "co2_savings": (
        lambda get, d: plotting.plot_co2_savings(get("battery_run").cost_projection, d, scenario_name="battery"),
        ["co2_avoided_yearly_battery.png", "co2_avoided_cumulative_battery.png"],
    ),
    "weather_annual_ghi_distribution": (
        lambda get, d: plotting.plot_weather_annual_ghi_distribution(
            _build_synthetic_weather(), get("historical_weather_file"), d
        ),
        ["annual_ghi_distribution.png"],
    ),
    "pareto_front": (
        lambda get, d: plotting.plot_pareto_front(
            get("sizing_sweep"), d, x="grid_independence_pct", y="npv_savings", color_by="battery_kwh"
        ),
        ["pareto_front.png"],
    ),
}


@pytest.mark.parametrize("name", sorted(REAL_PLOTS))
def test_a_plot_reads_the_frame_its_producer_writes(name, request, tmp_path):
    draw, expected = REAL_PLOTS[name]
    draw(request.getfixturevalue, str(tmp_path))
    _assert_written(tmp_path, *expected)


# ---------------------------------------------------------------------------
# App energy balance and economics
# ---------------------------------------------------------------------------


def test_plot_battery_soh_timeseries_date_window(battery_run, tmp_path):
    fig = plotting.plot_battery_soh_timeseries(
        battery_run.results, str(tmp_path), start_date="2023-03-01", end_date="2023-06-30"
    )

    drawn = pd.DatetimeIndex(fig.axes[0].lines[0].get_xdata())
    assert (drawn.min().date().isoformat(), drawn.max().date().isoformat()) == ("2023-03-01", "2023-06-30")


def test_plot_breakeven_comparison_of_app_results(tmp_path):
    results = []
    for battery_kwh in (0.0, 5.0):
        with _offline():
            app = App({**_BASE_CONFIG, "battery_kwh": battery_kwh})
            app.simulate()
        results.append(app.result())
    plotting.plot_breakeven_comparison(results, ["PV only", "PV + battery"], str(tmp_path), filename="app.png")
    _assert_written(tmp_path, "app.png")


# ---------------------------------------------------------------------------
# Monte Carlo (one-row-per-run table)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("older_table", [False, True], ids=["real", "without-interpolated-payback"])
def test_plot_montecarlo_simulation_draws_a_real_study(mc_runs, tmp_path, older_table):
    # Older run tables carry only the integer payback year; the payback
    # plots fall back to it.
    runs = mc_runs.drop(columns="payback_year_interpolated") if older_table else mc_runs
    assert runs["payback_year"].notna().any()

    plotting.plot_montecarlo_simulation(runs, str(tmp_path), verbose=False)

    _assert_written(
        tmp_path / "plots",
        "breakeven_histogram.png",
        "breakeven_cdf.png",
        "breakeven_summary_bar.png",
        "montecarlo_npv_distribution.png",
        "montecarlo_grid_independence_distribution.png",
        "montecarlo_final_soh_distribution.png",
        "montecarlo_lcoe_distribution.png",
    )


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------


def test_plot_weather_monthly_comparison(historical_weather_file, tmp_path):
    plotting.plot_weather_monthly_comparison(_build_synthetic_weather(), historical_weather_file, str(tmp_path))
    plotting.plot_weather_monthly_comparison(
        _build_synthetic_weather(), historical_weather_file, str(tmp_path), variable="temp_air"
    )
    _assert_written(tmp_path, "weather_monthly_ghi.png", "weather_monthly_temp_air.png")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


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
