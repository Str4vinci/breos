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
    close = plotting.plt.close
    monkeypatch.setattr(plotting.plt, "close", lambda *args, **kwargs: figures.append(plotting.plt.gcf()))
    yield figures
    close("all")


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
    assert ax.images[0].get_cmap().name == "RdBu"
    assert [text.get_text() for text in ax.texts] == ["+10.0", "+10.0", "+5.0"]
    # A difference of percentages is in percentage points.
    assert drawn[0].axes[1].get_ylabel() == "Grid independence, Porto − Berlin (percentage points)"
    assert (tmp_path / "sweep_grid_independence_pct_diff.png").is_file()


def test_sweep_heatmap_diff_labels_without_names_and_of_money(tmp_path, drawn):
    a, b = _sweep([40.0, 60.0, 45.0, 70.0]), _sweep([30.0, 55.0, 35.0, 50.0])
    a["npv_savings"], b["npv_savings"] = [100.0, 200.0, 300.0, 400.0], [150.0, 150.0, 150.0, 150.0]

    plotting.plot_sweep_heatmap(a, "grid_independence_pct", str(tmp_path), diff=b)
    plotting.plot_sweep_heatmap(a, "npv_savings", str(tmp_path), diff=b, labels=("A", "B"), currency="EUR")

    assert drawn[0].axes[1].get_ylabel() == "Grid independence difference (percentage points)"
    assert drawn[1].axes[1].get_ylabel() == "NPV savings, A − B (EUR)"
    np.testing.assert_array_equal(drawn[1].axes[0].images[0].get_array(), [[-50.0, 150.0], [50.0, 250.0]])
    assert (drawn[1].axes[0].images[0].norm.vmin, drawn[1].axes[0].images[0].norm.vmax) == (-250.0, 250.0)


def test_sweep_heatmap_diff_reads_keys_named_differently_in_the_two_sweeps(tmp_path, drawn):
    a = _sweep([40.0, 60.0, 45.0, 70.0])
    b = _sweep([30.0, 55.0, 35.0, 50.0]).rename(
        columns={"param_n_modules": "n_modules", "param_battery_kwh": "battery_kwh"}
    )

    plotting.plot_sweep_heatmap(a, "grid_independence_pct", str(tmp_path), diff=b)

    np.testing.assert_array_equal(drawn[0].axes[0].images[0].get_array(), [[10.0, 10.0], [5.0, 20.0]])


def test_sweep_heatmap_colours_cell_text_by_the_cell_luminance(tmp_path, drawn):
    a = _sweep([10.0, 0.0, -10.0, 0.5])

    plotting.plot_sweep_heatmap(a, "grid_independence_pct", str(tmp_path), diff=_sweep([0.0] * 4))

    # The two ends of RdBu are dark, its centre is near white.
    colours = {text.get_text(): text.get_color() for text in drawn[0].axes[0].texts}
    assert colours == {"+10.0": "white", "+0.0": "black", "\u221210.0": "white", "+0.5": "black"}


def test_sweep_heatmap_reads_the_csv_and_named_parameters(tmp_path, drawn):
    path = tmp_path / "sweep.csv"
    _sweep([40.0, 60.0, 45.0, 70.0]).to_csv(path, index=False)

    plotting.plot_sweep_heatmap(path, "grid_independence_pct", str(tmp_path), x="battery_kwh", y="param_n_modules")

    np.testing.assert_array_equal(drawn[0].axes[0].images[0].get_array(), [[40.0, 60.0], [45.0, 70.0]])


def test_sweep_heatmap_reads_a_bare_key_from_the_swept_column(tmp_path, drawn):
    # A sweep CSV also has result columns named like the keys; here the App
    # resolved other values than were swept.
    frame = _sweep([40.0, 60.0, 45.0, 70.0])
    frame["n_modules"] = frame["param_n_modules"] + 1
    frame["battery_kwh"] = frame["param_battery_kwh"] + 1.0

    plotting.plot_sweep_heatmap(frame, "grid_independence_pct", str(tmp_path), x="n_modules", y="battery_kwh")

    ax = drawn[0].axes[0]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["4", "8"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["0", "5"]

    third = pd.concat([frame.assign(param_price=0.2), frame.assign(param_price=0.3)])
    with pytest.raises(ValueError, match=r"also varies param_price, so") as error:
        plotting.plot_sweep_heatmap(third, "grid_independence_pct", str(tmp_path), x="n_modules", y="battery_kwh")
    assert "param_n_modules" not in str(error.value).split("also varies")[1]


def test_sweep_heatmap_names_the_valid_columns_of_a_bad_key(tmp_path):
    frame = _sweep([40.0, 60.0, 45.0, 70.0])

    with pytest.raises(ValueError, match=r"no 'npv' column; its numeric columns are run, param_n_modules"):
        plotting.plot_sweep_heatmap(frame, "npv", str(tmp_path))
    with pytest.raises(ValueError, match=r"no 'param_tilt' or 'tilt' column; its columns are run, param_n_modules"):
        plotting.plot_sweep_heatmap(frame, "grid_independence_pct", str(tmp_path), x="tilt", y="battery_kwh")
    with pytest.raises(ValueError, match="pass diff= as well"):
        plotting.plot_sweep_heatmap(frame, "grid_independence_pct", str(tmp_path), labels=("A", "B"))


def test_tick_text_keeps_large_values_readable():
    assert plotting._tick_text(10.0) == "10"
    assert plotting._tick_text(np.int64(4)) == "4"
    assert plotting._tick_text(1234567.0) == "1234567"
    assert plotting._tick_text(2_500_000) == "2500000"
    assert plotting._tick_text(0.25) == "0.25"
    assert plotting._tick_text(0.1 + 0.2) == "0.3"
    assert plotting._tick_text("flat") == "flat"


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
    # Only positive values: a sequential scale.
    assert drawn[0].axes[0].images[0].get_cmap().name == "YlGnBu"


def test_sweep_heatmap_takes_the_currency_of_a_csv(tmp_path, drawn):
    path = tmp_path / "sweep.csv"
    frame = _sweep([1.0] * 4)
    frame["npv_savings"] = [100.0, 200.0, 300.0, 400.0]
    frame.to_csv(path, index=False)

    plotting.plot_sweep_heatmap(path, "npv_savings", str(tmp_path), currency="USD")
    # A CSV records no currency: without currency=, the label names none.
    plotting.plot_sweep_heatmap(path, "npv_savings", str(tmp_path))

    assert drawn[0].axes[1].get_ylabel() == "NPV savings (USD)"
    assert drawn[1].axes[1].get_ylabel() == "NPV savings"
    frame.attrs["currency"] = "CHF"
    with pytest.raises(ValueError, match="records its currency as 'CHF'"):
        plotting.plot_sweep_heatmap(frame, "npv_savings", str(tmp_path), currency="USD")
    other = frame.copy()
    other.attrs["currency"] = "USD"
    with pytest.raises(ValueError, match=r"different currencies \(CHF, USD\)"):
        plotting.plot_sweep_heatmap(frame, "npv_savings", str(tmp_path), diff=other)


def test_sweep_heatmap_reads_the_currency_column_breos_writes(tmp_path, drawn):
    path = tmp_path / "sweep.csv"
    frame = _sweep([1.0] * 4)
    frame["npv_savings"] = [100.0, 200.0, 300.0, 400.0]
    frame["currency"] = "JPY"
    frame.to_csv(path, index=False)

    plotting.plot_sweep_heatmap(path, "npv_savings", str(tmp_path))

    assert drawn[0].axes[1].get_ylabel() == "NPV savings (JPY)"
    with pytest.raises(ValueError, match="records its currency as 'JPY'"):
        plotting.plot_sweep_heatmap(path, "npv_savings", str(tmp_path), currency="EUR")


def test_sweep_heatmap_centres_gains_and_losses_on_zero(tmp_path, drawn):
    frame = _sweep([1.0] * 4)
    frame["npv_savings"] = [-100.0, 200.0, 300.0, 400.0]

    plotting.plot_sweep_heatmap(frame, "npv_savings", str(tmp_path))
    plotting.plot_sweep_heatmap(frame, "npv_savings", str(tmp_path), vmin=-100.0, vmax=400.0)

    diverging, fixed = drawn[0].axes[0].images[0], drawn[1].axes[0].images[0]
    assert (diverging.norm.vmin, diverging.norm.vcenter, diverging.norm.vmax) == (-400.0, 0.0, 400.0)
    assert diverging.get_cmap().name == "RdBu"
    # Explicit limits keep the sequential scale.
    assert (fixed.norm.vmin, fixed.norm.vmax) == (-100.0, 400.0)
    assert fixed.get_cmap().name == "YlGnBu"


def test_sweep_heatmap_centres_an_all_loss_sweep_on_zero(tmp_path, drawn):
    frame = _sweep([1.0] * 4)
    frame["npv_savings"] = [-100.0, -200.0, -300.0, -400.0]

    plotting.plot_sweep_heatmap(frame, "npv_savings", str(tmp_path))

    ax = drawn[0].axes[0]
    # Losses only still read against break-even, not as a sequential scale.
    assert (ax.images[0].norm.vmin, ax.images[0].norm.vcenter, ax.images[0].norm.vmax) == (-400.0, 0.0, 400.0)
    assert ax.images[0].get_cmap().name == "RdBu"
    # Cell text uses the minus sign the colour bar uses.
    assert [text.get_text() for text in ax.texts] == ["\u2212100", "\u2212300", "\u2212200", "\u2212400"]


def test_sweep_heatmap_ignores_currencies_when_nothing_is_money(tmp_path, drawn):
    a, b = _sweep([40.0, 60.0, 45.0, 70.0]), _sweep([30.0, 55.0, 35.0, 50.0])
    a.attrs["currency"], b.attrs["currency"] = "CHF", "EUR"

    plotting.plot_sweep_heatmap(a, "grid_independence_pct", str(tmp_path), diff=b, labels=("Zurich", "Porto"))

    assert drawn[0].axes[1].get_ylabel() == "Grid independence, Zurich − Porto (percentage points)"
    a["npv_savings"], b["npv_savings"] = [1.0] * 4, [2.0] * 4
    with pytest.raises(ValueError, match=r"different currencies \(CHF, EUR\)"):
        plotting.plot_sweep_heatmap(a, "npv_savings", str(tmp_path), diff=b)


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
            "usable_ac_system_production_kwh": 1000.0 - (tilt - 30) ** 2 - abs(azimuth - 185),
        }
        for tilt in tilts
        for azimuth in azimuths
    ]
    return pd.DataFrame(rows)


def test_orientation_landscape_marks_the_optimum_and_the_east_west_profile(tmp_path, drawn):
    plotting.plot_orientation_landscape(_orientation_sweep(), "usable_ac_system_production_kwh", str(tmp_path))

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
    frame["lcoe_per_kwh"] = 1.0 / frame["usable_ac_system_production_kwh"]

    plotting.plot_orientation_landscape(frame, "lcoe_per_kwh", str(tmp_path), maximize=False)

    np.testing.assert_array_equal(drawn[0].axes[0].collections[1].get_offsets(), [[180.0, 30.0]])


def test_orientation_landscape_draws_a_tilt_sweep_as_a_profile(tmp_path, drawn):
    # An east-west roof: both arrays take the swept tilt, so only tilt varies.
    frame = pd.DataFrame({"param_tilt": [30, 10, 50], "usable_ac_system_production_kwh": [950.0, 900.0, 920.0]})

    plotting.plot_orientation_landscape(frame, "usable_ac_system_production_kwh", str(tmp_path))

    (ax,) = drawn[0].axes
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [10, 30, 50])
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [30])
    assert ax.lines[1].get_label() == "Optimum: 30° tilt"


def test_orientation_landscape_ticks_negative_azimuths(tmp_path, drawn):
    # A southern-hemisphere sweep: north is 0, east and west are -90 and 90.
    frame = pd.DataFrame(
        [
            {
                "param_tilt": tilt,
                "param_azimuth": azimuth,
                "usable_ac_system_production_kwh": 1000.0 - tilt - abs(azimuth),
            }
            for tilt in (10, 30)
            for azimuth in (-90, -45, 0, 45, 90)
        ]
    )

    plotting.plot_orientation_landscape(frame, "usable_ac_system_production_kwh", str(tmp_path))

    for ax in drawn[0].axes[:2]:
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            "−90° W",
            "−45° NW",
            "0° N",
            "45° NE",
            "90° E",
        ]


def test_orientation_landscape_labels_a_flat_optimum_without_an_azimuth(tmp_path, drawn):
    frame = _orientation_sweep()
    frame["param_tilt"] = frame["param_tilt"].replace({10: 0})
    frame["flat_best"] = 100.0 - frame["param_tilt"]

    plotting.plot_orientation_landscape(frame, "flat_best", str(tmp_path))

    assert drawn[0].axes[0].collections[1].get_label() == "Optimum: 0° tilt"


def test_orientation_landscape_names_a_bad_metric_and_an_empty_one(tmp_path):
    frame = _orientation_sweep()
    frame["npv_savings"] = np.nan

    with pytest.raises(ValueError, match=r"no 'pv_production' column; its numeric columns are param_tilt"):
        plotting.plot_orientation_landscape(frame, "pv_production", str(tmp_path))
    with pytest.raises(ValueError, match="no finite npv_savings value"):
        plotting.plot_orientation_landscape(frame, "npv_savings", str(tmp_path))
    tilt_only = frame[frame["param_azimuth"] == 180].drop(columns="param_azimuth")
    with pytest.raises(ValueError, match="no finite npv_savings value"):
        plotting.plot_orientation_landscape(tilt_only, "npv_savings", str(tmp_path))


def test_orientation_landscape_takes_the_currency_of_a_csv(tmp_path, drawn):
    frame = _orientation_sweep()
    frame["npv_savings"] = frame["usable_ac_system_production_kwh"]

    plotting.plot_orientation_landscape(frame, "npv_savings", str(tmp_path), currency="USD")
    plotting.plot_orientation_landscape(frame, "npv_savings", str(tmp_path))

    assert drawn[0].axes[1].get_ylabel() == "NPV savings (USD)"
    assert drawn[1].axes[1].get_ylabel() == "NPV savings"


def test_orientation_landscape_rejects_a_sweep_with_another_varied_key(tmp_path):
    frame = pd.DataFrame(
        {"param_tilt": [10, 10], "param_n_modules": [4, 8], "usable_ac_system_production_kwh": [1.0, 2.0]}
    )

    with pytest.raises(ValueError, match="more than one row"):
        plotting.plot_orientation_landscape(frame, "usable_ac_system_production_kwh", str(tmp_path))


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


def test_pareto_front_takes_the_currency_of_a_csv(tmp_path, drawn):
    path = tmp_path / "designs.csv"
    pd.DataFrame({"grid_independence_pct": [40.0, 60.0], "npv_savings": [100.0, 50.0]}).to_csv(path, index=False)

    plotting.plot_pareto_front(path, str(tmp_path), x="grid_independence_pct", y="npv_savings", currency="USD")
    plotting.plot_pareto_front(path, str(tmp_path), x="grid_independence_pct", y="npv_savings")

    assert drawn[0].axes[0].get_ylabel() == "NPV savings (USD)"
    assert drawn[1].axes[0].get_ylabel() == "NPV savings"
    with pytest.raises(ValueError, match=r"no 'param_npv' or 'npv' column; its columns are grid_independence_pct"):
        plotting.plot_pareto_front(path, str(tmp_path), x="grid_independence_pct", y="npv")


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
    assert [line.get_label() for line in drawn[0].axes[0].lines] == [
        "Monthly minimum",
        "Monthly maximum",
        "Historical mean (2 years)",
        "TMY (PVGIS)",
    ]
    assert tmy_line.get_ydata()[0] == pytest.approx(31 * 24 * 0.16)
    assert drawn[0].axes[0].get_ylabel() == "GHI (kWh/m²)"
    assert lines == [pytest.approx(8760 * 0.16), pytest.approx(8760 * 0.15)]
    assert (tmp_path / "weather_monthly_ghi.png").is_file()
    assert (tmp_path / "annual_ghi_distribution.png").is_file()


def test_weather_plots_report_missing_values_and_months(tmp_path):
    historical = {2021: _weather_year(2021), 2022: _weather_year(2022)}
    gappy = _weather_year(2023)
    gappy.iloc[:5, 0] = np.nan
    no_march = _weather_year(2023)
    no_march.loc[no_march.index.month == 3, "ghi"] = np.nan

    with pytest.warns(UserWarning, match="The TMY has 5 missing ghi values"):
        plotting.plot_weather_annual_ghi_distribution(gappy, historical, str(tmp_path))
    with pytest.raises(ValueError, match="The TMY has no GHI for Mar"), pytest.warns(UserWarning, match="744 missing"):
        plotting.plot_weather_annual_ghi_distribution(no_march, historical, str(tmp_path))
    with (
        pytest.raises(ValueError, match="Historical year 2022 has no GHI for Mar"),
        pytest.warns(UserWarning, match="Historical year 2022 has 744 missing"),
    ):
        plotting.plot_weather_annual_ghi_distribution(gappy.fillna(0.0), {2022: no_march}, str(tmp_path))


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


def _app_result(investment, savings, currency="EUR", system_costs=None):
    """The ``financial`` rows and currency of an App result.

    ``savings`` holds each year's cumulative balance and no-system cost. The
    system cost is the no-system cost less the balance, unless ``system_costs``
    gives it.
    """
    rows = [{"year": 0, "balance": -investment, "reference": 0.0}]
    for year, (balance, no_system) in enumerate(savings, start=1):
        with_system = no_system - balance if system_costs is None else system_costs[year - 1]
        rows.append(
            {
                "year": year,
                "balance": balance,
                "reference": 0.0,
                "cost_with_system": with_system,
                "cost_without_system": no_system,
            }
        )
    return {"financial": rows, "total_investment": investment, "provenance": {"currency": currency}}


def _captured_vlines(monkeypatch):
    lines = []
    real_axvline = plotting.plt.Axes.axvline
    monkeypatch.setattr(
        plotting.plt.Axes,
        "axvline",
        lambda self, x=0, *a, **k: lines.append(float(x)) or real_axvline(self, x, *a, **k),
    )
    return lines


def test_breakeven_comparison_reads_app_results(tmp_path, drawn, monkeypatch):
    lines = _captured_vlines(monkeypatch)
    # Year 1 costs 1100 with the system, not the 1000 invested at year 0.
    result = _app_result(1000.0, [(-500.0, 600.0), (100.0, 1300.0)])

    plotting.plot_breakeven_comparison([result], ["PV"], str(tmp_path))

    ax = drawn[0].axes[0]
    no_system, with_system = ax.lines[0], ax.lines[1]
    # Year 0 is the investment against nothing spent.
    np.testing.assert_array_equal(no_system.get_xdata(), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(no_system.get_ydata(), [0.0, 600.0, 1300.0])
    np.testing.assert_array_equal(with_system.get_ydata(), [1000.0, 1100.0, 1200.0])
    assert no_system.get_label() == "No system"
    frame = plotting._breakeven_projection(result)
    assert frame.attrs["total_investment"] == 1000.0
    assert lines == [pytest.approx(find_payback_year_interpolated(frame))] == [pytest.approx(1 + 500 / 600)]
    # The payback line carries its year.
    assert [text.get_text().strip() for text in ax.texts] == ["1.8 years"]
    assert ax.get_ylabel() == "Cumulative Cost (EUR)"


def test_breakeven_comparison_anchors_year_zero_at_the_investment_not_the_first_row(tmp_path, drawn):
    result = _app_result(800.0, [(-500.0, 600.0), (100.0, 1300.0)], system_costs=[1100.0, 1200.0])

    plotting.plot_breakeven_comparison([result], ["PV"], str(tmp_path))

    np.testing.assert_array_equal(drawn[0].axes[0].lines[1].get_ydata(), [800.0, 1100.0, 1200.0])


def test_breakeven_comparison_shares_one_baseline_and_names_the_groups(tmp_path, drawn):
    one = _app_result(1000.0, [(-400.0, 600.0), (200.0, 1200.0)])
    two = _app_result(2000.0, [(-1400.0, 600.0), (-800.0, 1200.0)])
    other = _app_result(1000.0, [(-300.0, 700.0), (400.0, 1400.0)])

    plotting.plot_breakeven_comparison([one, two], ["PV only", "PV + 5 kWh"], str(tmp_path))
    plotting.plot_breakeven_comparison([one, two, other], ["PV only", "PV + 5 kWh", "Berlin"], str(tmp_path))

    shared, grouped = drawn[0].axes[0], drawn[1].axes[0]
    labels = [line.get_label() for line in shared.lines if line.get_linestyle() == "--"]
    assert labels == ["No system"]
    assert shared.lines[0].get_color() == "black"
    labels = [line.get_label() for line in grouped.lines if line.get_linestyle() == "--"]
    assert labels == ["No system (PV only, PV + 5 kWh)", "No system (Berlin)"]
    assert [text.get_text().strip() for text in grouped.texts] == ["1.7 years", "1.4 years"]


def test_breakeven_comparison_labels_csv_projections_by_the_known_currency(tmp_path, drawn):
    usd = _app_result(1000.0, [(-400.0, 600.0), (200.0, 1200.0)], currency="USD")
    path = tmp_path / "projection.csv"
    plotting._breakeven_projection(_app_result(1000.0, [(-300.0, 700.0)])).to_csv(path, index=False)
    from_csv = pd.read_csv(path)

    # The CSV records no currency, so it does not conflict with the USD result.
    plotting.plot_breakeven_comparison([usd, from_csv], ["App", "CSV"], str(tmp_path))
    plotting.plot_breakeven_comparison([from_csv], ["CSV"], str(tmp_path), currency="CHF")
    plotting.plot_breakeven_comparison([from_csv], ["CSV"], str(tmp_path))

    assert drawn[0].axes[0].get_ylabel() == "Cumulative Cost (USD)"
    assert drawn[1].axes[0].get_ylabel() == "Cumulative Cost (CHF)"
    assert drawn[2].axes[0].get_ylabel() == "Cumulative Cost"
    formatter = drawn[2].axes[0].yaxis.get_major_formatter()
    assert formatter(1500.0, 0) == "1500"
    assert formatter(-12345.0, 0) == "-12 345"
    with pytest.raises(ValueError, match="records its currency as 'USD'"):
        plotting.plot_breakeven_comparison([usd, from_csv], ["App", "CSV"], str(tmp_path), currency="EUR")


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
