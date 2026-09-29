"""How the sweep, orientation, Pareto, weather and break-even plots read their inputs."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from breos import plotting  # noqa: E402
from breos.economics import find_payback_year_interpolated  # noqa: E402
from breos.weather import preload_weather_by_year  # noqa: E402


@pytest.fixture
def drawn(monkeypatch):
    """The figures the plots close, kept open for inspection."""
    figures = []
    monkeypatch.setattr(plotting.plt, "close", lambda *args, **kwargs: figures.append(plotting.plt.gcf()))
    yield figures
    plotting.plt.close("all")


def _sweep(values, n_modules=(4, 8), battery_kwh=(0.0, 5.0), **extra):
    """A ``breos sweep`` table: one row per module count and battery size, in sweep order."""
    rows = []
    for n, kwh in [(n, kwh) for n in n_modules for kwh in battery_kwh]:
        rows.append({"run": len(rows) + 1, "param_n_modules": n, "param_battery_kwh": kwh, **extra})
    frame = pd.DataFrame(rows)
    frame["grid_independence_pct"] = values
    return frame


# ---------------------------------------------------------------------------
# plot_sweep_heatmap
# ---------------------------------------------------------------------------


def test_sweep_heatmap_puts_the_first_swept_key_on_x(tmp_path, drawn):
    plotting.plot_sweep_heatmap(_sweep([40.0, 60.0, 45.0, 70.0]), "grid_independence_pct", str(tmp_path))

    ax = drawn[0].axes[0]
    # Rows are battery sizes, columns module counts, both ascending.
    np.testing.assert_array_equal(ax.images[0].get_array(), [[40.0, 45.0], [60.0, 70.0]])
    assert [label.get_text() for label in ax.get_xticklabels()] == ["4", "8"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["0", "5"]
    assert ax.get_xlabel() == "Number of PV modules"
    assert ax.get_ylabel() == "Battery capacity (kWh)"
    assert [text.get_text() for text in ax.texts] == ["40.0", "45.0", "60.0", "70.0"]
    assert (tmp_path / "sweep_grid_independence_pct.png").is_file()


def test_sweep_heatmap_diff_draws_the_difference_on_the_shared_cells(tmp_path, drawn):
    a = _sweep([40.0, 60.0, 45.0, 70.0])
    b = _sweep([30.0, 55.0, 35.0, 0.0]).iloc[:3]

    plotting.plot_sweep_heatmap(a, "grid_independence_pct", str(tmp_path), diff=b, labels=("Porto", "Berlin"))

    ax = drawn[0].axes[0]
    data = np.ma.filled(ax.images[0].get_array(), np.nan)
    np.testing.assert_array_equal(data, [[10.0, 10.0], [5.0, np.nan]])
    # A diverging scale centred on zero.
    norm = ax.images[0].norm
    assert (norm.vmin, norm.vcenter, norm.vmax) == (-10.0, 0.0, 10.0)
    assert [text.get_text() for text in ax.texts] == ["+10.0", "+10.0", "+5.0"]
    assert drawn[0].axes[1].get_ylabel() == "Grid independence (%) (Porto − Berlin)"
    assert (tmp_path / "sweep_grid_independence_pct_diff.png").is_file()


def test_sweep_heatmap_reads_the_csv_and_named_parameters(tmp_path, drawn):
    path = tmp_path / "sweep.csv"
    _sweep([40.0, 60.0, 45.0, 70.0]).to_csv(path, index=False)

    plotting.plot_sweep_heatmap(path, "grid_independence_pct", str(tmp_path), x="battery_kwh", y="param_n_modules")

    np.testing.assert_array_equal(drawn[0].axes[0].images[0].get_array(), [[40.0, 60.0], [45.0, 70.0]])


def test_sweep_heatmap_needs_x_and_y_for_a_third_swept_key(tmp_path):
    frame = pd.concat([_sweep([1.0] * 4, param_price=0.2), _sweep([2.0] * 4, param_price=0.3)])
    frame["param_price"] = [0.2] * 4 + [0.3] * 4

    with pytest.raises(ValueError, match="varies 3 parameters"):
        plotting.plot_sweep_heatmap(frame, "grid_independence_pct", str(tmp_path))
    with pytest.raises(ValueError, match="also varies param_price"):
        plotting.plot_sweep_heatmap(frame, "grid_independence_pct", str(tmp_path), x="n_modules", y="battery_kwh")

    plotting.plot_sweep_heatmap(
        frame[frame["param_price"] == 0.2], "grid_independence_pct", str(tmp_path), x="n_modules", y="battery_kwh"
    )


def test_sweep_heatmap_rejects_sweeps_without_a_shared_cell(tmp_path):
    other = _sweep([1.0] * 4, n_modules=(10, 12))

    with pytest.raises(ValueError, match="share no"):
        plotting.plot_sweep_heatmap(_sweep([1.0] * 4), "grid_independence_pct", str(tmp_path), diff=other)


def test_sweep_heatmap_labels_money_in_the_frame_currency(tmp_path, drawn):
    frame = _sweep([1.0] * 4)
    frame["npv_savings"] = [100.0, 200.0, 300.0, 400.0]
    frame.attrs["currency"] = "CHF"

    plotting.plot_sweep_heatmap(frame, "npv_savings", str(tmp_path))

    assert drawn[0].axes[1].get_ylabel() == "NPV savings (CHF)"
    assert [text.get_text() for text in drawn[0].axes[0].texts] == ["100", "300", "200", "400"]


# ---------------------------------------------------------------------------
# plot_orientation_landscape
# ---------------------------------------------------------------------------


def _orientation_sweep():
    tilts, azimuths = [10, 30, 50], [90, 180, 270]
    rows = [
        {
            "param_tilt": tilt,
            "param_azimuth": azimuth,
            # Best at 30 degrees facing south; west a little better than east.
            "pv_production_kwh": 1000.0 - (tilt - 30) ** 2 - abs(azimuth - 185),
        }
        for tilt in tilts
        for azimuth in azimuths
    ]
    return pd.DataFrame(rows)


def test_orientation_landscape_marks_the_optimum_and_the_east_west_profile(tmp_path, drawn):
    plotting.plot_orientation_landscape(_orientation_sweep(), "pv_production_kwh", str(tmp_path))

    ax_map, ax_profile = drawn[0].axes[0], drawn[0].axes[1]
    np.testing.assert_array_equal(ax_map.collections[1].get_offsets(), [[180.0, 30.0]])
    profile = ax_profile.lines[0]
    np.testing.assert_array_equal(profile.get_xdata(), [90.0, 180.0, 270.0])
    np.testing.assert_array_equal(profile.get_ydata(), [905.0, 995.0, 915.0])
    assert profile.get_label() == "30° tilt"
    assert [label.get_text() for label in ax_profile.get_xticklabels()] == [
        "90° E",
        "135° SE",
        "180° S",
        "225° SW",
        "270° W",
    ]
    assert (tmp_path / "orientation_landscape.png").is_file()


def test_orientation_landscape_minimises_when_asked(tmp_path, drawn):
    frame = _orientation_sweep()
    frame["lcoe_per_kwh"] = 1.0 / frame["pv_production_kwh"]

    plotting.plot_orientation_landscape(frame, "lcoe_per_kwh", str(tmp_path), maximize=False)

    np.testing.assert_array_equal(drawn[0].axes[0].collections[1].get_offsets(), [[180.0, 30.0]])


def test_orientation_landscape_draws_a_tilt_sweep_as_a_profile(tmp_path, drawn):
    # An east-west roof: both arrays take the swept tilt, so only tilt varies.
    frame = pd.DataFrame({"param_tilt": [30, 10, 50], "pv_production_kwh": [950.0, 900.0, 920.0]})

    plotting.plot_orientation_landscape(frame, "pv_production_kwh", str(tmp_path))

    (ax,) = drawn[0].axes
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [10, 30, 50])
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [30])
    assert ax.lines[1].get_label() == "Optimum: 30° tilt"


def test_orientation_landscape_rejects_a_sweep_with_another_varied_key(tmp_path):
    frame = pd.DataFrame({"param_tilt": [10, 10], "param_n_modules": [4, 8], "pv_production_kwh": [1.0, 2.0]})

    with pytest.raises(ValueError, match="more than one row"):
        plotting.plot_orientation_landscape(frame, "pv_production_kwh", str(tmp_path))


# ---------------------------------------------------------------------------
# plot_pareto_front
# ---------------------------------------------------------------------------


def test_pareto_mask_keeps_the_non_dominated_points():
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0])
    y = np.array([3.0, 2.0, 1.0, 1.0, 3.0, 0.5, 2.0])

    # (2, 1) and (3, 0.5) are dominated; equal points on the front both stay.
    np.testing.assert_array_equal(plotting._pareto_mask(x, y), [True, True, True, False, True, False, True])


def test_pareto_front_reads_the_optimizer_result(tmp_path, drawn):
    pareto = pd.DataFrame(
        {
            "Modules": [4, 8, 12],
            "Battery_kWh": [0.0, 5.0, 10.0],
            "Grid_Independence_%": [40.0, 70.0, 85.0],
            "NPV": [2000.0, 500.0, -1500.0],
            "Projected_Initial_Cost": [2000.0, 7000.0, 6000.0],
        }
    )
    pareto.attrs["currency"] = "EUR"
    result = SimpleNamespace(details={"pareto": pareto})

    plotting.plot_pareto_front(result, str(tmp_path), color_by="Battery_kWh")
    # With cost minimised, the 12-module design beats the 8-module one.
    plotting.plot_pareto_front(
        pareto, str(tmp_path), x="Projected_Initial_Cost", y="Grid_Independence_%", maximize=(False, True)
    )

    first, second = drawn[0].axes[0], drawn[1].axes[0]
    np.testing.assert_array_equal(first.lines[0].get_xdata(), [40.0, 70.0, 85.0])
    assert first.get_ylabel() == "NPV savings (EUR)"
    assert drawn[0].axes[1].get_ylabel() == "Battery capacity (kWh)"
    np.testing.assert_array_equal(second.lines[0].get_xdata(), [2000.0, 6000.0])
    np.testing.assert_array_equal(second.collections[0].get_offsets(), [[7000.0, 70.0]])
    assert second.get_xlabel() == "Investment (EUR)"


def test_pareto_front_skips_designs_without_both_objectives(tmp_path, drawn):
    frame = pd.DataFrame({"grid_independence_pct": [40.0, 60.0, np.nan], "npv_savings": [100.0, 50.0, 900.0]})

    plotting.plot_pareto_front(frame, str(tmp_path), x="grid_independence_pct", y="npv_savings")

    np.testing.assert_array_equal(drawn[0].axes[0].lines[0].get_xdata(), [40.0, 60.0])
    with pytest.raises(ValueError, match="No design"):
        plotting.plot_pareto_front(frame.iloc[2:], str(tmp_path), x="grid_independence_pct", y="npv_savings")


# ---------------------------------------------------------------------------
# Weather comparison
# ---------------------------------------------------------------------------


def _weather_year(year, ghi=100.0, temperature=10.0, columns=("ghi", "temp_air")):
    index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    index = index[~((index.month == 2) & (index.day == 29))]
    return pd.DataFrame({columns[0]: ghi, columns[1]: temperature}, index=index)


def test_weather_monthly_totals_irradiance_and_averages_temperature():
    weather = _weather_year(2023, ghi=1000.0, temperature=12.5)

    ghi = plotting._weather_monthly(weather, "ghi")
    assert ghi[1] == pytest.approx(31 * 24.0)
    assert ghi[2] == pytest.approx(28 * 24.0)
    assert plotting._weather_monthly(weather, "temp_air").tolist() == [12.5] * 12
    # 15-minute steps give the same energy.
    quarter = weather.resample("15min").ffill()
    assert plotting._weather_monthly(quarter, "ghi")[3] == pytest.approx(31 * 24.0)


def test_weather_monthly_stats_read_a_monte_carlo_weather_file(tmp_path):
    # Open-Meteo column names, as a historical file has them.
    years = [
        _weather_year(y, ghi=g, columns=("shortwave_radiation", "temperature_2m"))
        for y, g in ((2021, 100.0), (2022, 200.0))
    ]
    path = tmp_path / "historical.csv"
    pd.concat(years).rename_axis("date").reset_index().to_csv(path, index=False)

    from_file = plotting._weather_monthly_stats(plotting._historical_weather_years(str(path)), "ghi")
    from_frames = plotting._weather_monthly_stats(preload_weather_by_year(str(path)), "ghi")

    pd.testing.assert_frame_equal(from_file, from_frames)
    january = from_file.loc[1]
    assert january["min"] == pytest.approx(31 * 2.4)
    assert january["max"] == pytest.approx(31 * 4.8)
    assert january["mean"] == pytest.approx(31 * 3.6)
    # Two years: the t quantile with one degree of freedom, 12.706.
    half_width = 12.7062047 * np.std([74.4, 148.8], ddof=1) / np.sqrt(2)
    assert january["ci_high"] - january["mean"] == pytest.approx(half_width)


def test_weather_plots_draw_the_tmy_against_the_historical_years(tmp_path, drawn, monkeypatch):
    historical = {2021: _weather_year(2021, ghi=100.0), 2022: _weather_year(2022, ghi=200.0)}
    tmy = _weather_year(2023, ghi=160.0)
    lines = []
    real_axvline = plotting.plt.Axes.axvline

    def axvline(self, x=0, *args, **kwargs):
        lines.append(float(x))
        return real_axvline(self, x, *args, **kwargs)

    monkeypatch.setattr(plotting.plt.Axes, "axvline", axvline)

    plotting.plot_weather_monthly_comparison(tmy, historical, str(tmp_path), tmy_label="TMY (PVGIS)")
    plotting.plot_weather_annual_ghi_distribution(tmy, historical, str(tmp_path))

    tmy_line = drawn[0].axes[0].lines[-1]
    assert tmy_line.get_label() == "TMY (PVGIS)"
    assert tmy_line.get_ydata()[0] == pytest.approx(31 * 24 * 0.16)
    assert drawn[0].axes[0].get_ylabel() == "GHI (kWh/m²)"
    assert lines == [pytest.approx(8760 * 0.16), pytest.approx(8760 * 0.15)]
    assert (tmp_path / "weather_monthly_ghi.png").is_file()
    assert (tmp_path / "annual_ghi_distribution.png").is_file()


def test_weather_plots_name_a_missing_variable(tmp_path):
    with pytest.raises(ValueError, match="no dni column"):
        plotting.plot_weather_monthly_comparison(
            _weather_year(2023), {2021: _weather_year(2021)}, str(tmp_path), variable="dni"
        )
    with pytest.raises(ValueError, match="variable must be one of"):
        plotting.plot_weather_monthly_comparison(
            _weather_year(2023), {2021: _weather_year(2021)}, str(tmp_path), variable="wind"
        )


# ---------------------------------------------------------------------------
# plot_breakeven_comparison
# ---------------------------------------------------------------------------


def _app_result(investment, savings, currency="EUR"):
    """The ``financial`` rows and currency of an App result."""
    rows = [{"year": 0, "balance": -investment, "reference": 0.0}]
    for year, (balance, no_system) in enumerate(savings, start=1):
        rows.append(
            {
                "year": year,
                "balance": balance,
                "reference": 0.0,
                "cost_with_system": no_system - balance,
                "cost_without_system": no_system,
            }
        )
    return {"financial": rows, "total_investment": investment, "provenance": {"currency": currency}}


def test_breakeven_comparison_reads_app_results(tmp_path, drawn, monkeypatch):
    lines = []
    real_axvline = plotting.plt.Axes.axvline
    monkeypatch.setattr(
        plotting.plt.Axes,
        "axvline",
        lambda self, x=0, *a, **k: lines.append(float(x)) or real_axvline(self, x, *a, **k),
    )
    result = _app_result(1000.0, [(-400.0, 600.0), (200.0, 1200.0)])

    plotting.plot_breakeven_comparison([result], ["PV"], str(tmp_path))

    ax = drawn[0].axes[0]
    no_system, with_system = ax.lines[0], ax.lines[1]
    # Year 0 is the investment against nothing spent.
    np.testing.assert_array_equal(no_system.get_xdata(), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(no_system.get_ydata(), [0.0, 600.0, 1200.0])
    np.testing.assert_array_equal(with_system.get_ydata(), [1000.0, 1000.0, 1000.0])
    frame = plotting._breakeven_projection(result)
    assert frame.attrs["total_investment"] == 1000.0
    assert lines == [pytest.approx(find_payback_year_interpolated(frame))] == [pytest.approx(1 + 400 / 600)]
    assert ax.get_ylabel() == "Cumulative Cost (EUR)"


def test_breakeven_comparison_checks_labels_colours_and_currencies(tmp_path):
    eur, chf = _app_result(1000.0, [(-400.0, 600.0)]), _app_result(1000.0, [(-400.0, 600.0)], currency="CHF")

    with pytest.raises(ValueError, match="need 2 labels"):
        plotting.plot_breakeven_comparison([eur, eur], ["one"], str(tmp_path))
    with pytest.raises(ValueError, match="need 2 colors"):
        plotting.plot_breakeven_comparison([eur, eur], ["one", "two"], str(tmp_path), colors=["C0"])
    with pytest.raises(ValueError, match="different currencies"):
        plotting.plot_breakeven_comparison([eur, chf], ["one", "two"], str(tmp_path))
    with pytest.raises(ValueError, match="no 'financial'"):
        plotting.plot_breakeven_comparison([{"financial": None}], ["period"], str(tmp_path))
