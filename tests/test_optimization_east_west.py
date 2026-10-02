"""East-West layouts in the optimizer search space (#384)."""

import numpy as np
import pandas as pd
import pytest

from breos.optimization import evaluate_projected_design
from breos.optimization_config import resolve_optimization_config

MODULE = "Suntech_STP550S_STC"
COSTS = {
    "electricity_cost": 0.25,
    "electricity_sold_cost": 0.07,
    "daily_power_cost": 0.50,
    "module_cost_per_w": 0.15,
    "storage_cost_per_kwh": 400.0,
    "installation_cost_per_module": 300.0,
    "maintenance_cost_per_panel": 12.0,
    "maintenance_cost": 30.0,
}
FINANCIALS = {"inflation_rate": 0.03, "sell_price_inflation": 0.015, "discount_rate": 0.04}
DC_AC_RATIO = 1.25
YEARS = 2
DEGRADATION = 0.007
LOCATION = {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"}
BATTERY = {"temperature": 20.0, "indoor_model": {"enabled": False}}


def _optimizer_config(**mode):
    return {
        "location": dict(LOCATION),
        "simulation": {"resolution": "h", "years_projection": YEARS},
        "constraints": {"budget": 100000.0, "max_area_m2": 100.0, "max_modules": 9, "max_battery_kwh": 4.0},
        "optimization": {"early_stop": False},
        "mode": mode,
        "pv": {"module": MODULE, "degradation_rate": DEGRADATION},
        "battery": dict(BATTERY),
        "costs": dict(COSTS, dc_ac_ratio=DC_AC_RATIO),
        "financials": dict(FINANCIALS),
    }


@pytest.fixture
def year_inputs(open_meteo_weather):
    idx = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
    weather = open_meteo_weather(idx)
    houseload = pd.DataFrame({"Load": 400.0 + 300.0 * (idx.hour >= 17)}, index=idx)
    return weather, houseload


def _app_run(monkeypatch, weather, houseload, battery_kwh, arrays):
    from breos.app import App
    from breos.app_inputs import AppRuntimeDependencies
    from breos.weather import build_battery_temperature_series

    dependencies = AppRuntimeDependencies(
        load_profile=lambda **kwargs: houseload.copy(),
        load_weather=lambda **kwargs: None,
        fetch_tmy_weather_data=lambda **kwargs: (weather.copy(), {}),
        resample_to_15min=lambda frame, **kwargs: frame,
        build_battery_temperature_series=build_battery_temperature_series,
    )
    monkeypatch.setattr(App, "_runtime_dependencies", staticmethod(lambda: dependencies))
    app = App(
        {
            "location": dict(LOCATION),
            "annual_consumption_kwh": float(houseload["Load"].sum() / 1000.0),
            "battery_kwh": float(battery_kwh),
            "pv_module": MODULE,
            "pv_arrays": arrays,
            "projection_years": YEARS,
            "resolution": "h",
            "start_date": "2023-01-01",
            "costs": dict(COSTS),
            "inverter_loading_ratio": DC_AC_RATIO,
            **FINANCIALS,
            "pv_degradation_rate": DEGRADATION,
            "battery_temperature": BATTERY["temperature"],
            "battery_indoor_model": BATTERY["indoor_model"],
            "execution_backend": "python",
        }
    )
    app.simulate()
    return app


def _assert_frames_identical(app_frame, optimizer_frame):
    """Every column both frames carry holds the same floats, to the last bit."""
    shared = [column for column in optimizer_frame.columns if column in app_frame.columns]
    assert shared
    for column in shared:
        expected = app_frame[column].to_numpy()
        actual = optimizer_frame[column].to_numpy()
        if expected.dtype.kind == "f":
            assert np.array_equal(expected, actual, equal_nan=True), column
        else:
            assert (expected == actual).all(), column


@pytest.mark.parametrize(("n_modules", "arrays"), [(7, [(3, 90), (4, 270)]), (1, [(1, 270)])])
def test_an_east_west_design_is_the_app_pv_arrays_run(year_inputs, monkeypatch, n_modules, arrays):
    """The optimizer's East-West design gives the App's floats for the same two arrays.

    floor(n / 2) modules face east and the rest west, so one module is a
    single west array, which is how the App runs it: it refuses an empty array.
    """
    weather, houseload = year_inputs
    result = evaluate_projected_design(
        weather,
        houseload,
        _optimizer_config(),
        n_modules=n_modules,
        battery_kwh=3.0,
        tilt=10.0,
        layout="east_west",
        execution_backend="python",
    )
    app = _app_run(
        monkeypatch,
        weather,
        houseload,
        3.0,
        [{"modules": modules, "tilt": 10, "azimuth": azimuth} for modules, azimuth in arrays],
    )

    _assert_frames_identical(app._artifacts.yearly_df, result.yearly)
    _assert_frames_identical(app._artifacts.cost_projection, result.financial)
    assert result.metrics["Layout"] == "east_west"
    assert result.metrics["Tilt"] == 10.0
    assert np.isnan(result.metrics["Azimuth"])
    assert result.metrics["Projected_Initial_Cost"] == app._artifacts.costs["total_initial_cost"]


def test_a_single_design_reports_its_layout(year_inputs):
    weather, houseload = year_inputs
    result = evaluate_projected_design(
        weather, houseload, _optimizer_config(), n_modules=4, battery_kwh=0.0, tilt=30.0, azimuth=180.0
    )
    assert (result.metrics["Layout"], result.metrics["Tilt"], result.metrics["Azimuth"]) == ("single", 30.0, 180.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"tilt": 30.0}, "single layout needs an azimuth"),
        ({"tilt": 10.0, "azimuth": 90.0, "layout": "east_west"}, "do not pass an azimuth"),
        ({"tilt": 10.0, "azimuth": 180.0, "layout": "north_south"}, "layout must be one of"),
    ],
)
def test_a_design_names_a_layout_it_can_build(arguments, message):
    with pytest.raises(ValueError, match=message):
        evaluate_projected_design(
            pd.DataFrame(), pd.DataFrame(), _optimizer_config(), n_modules=4, battery_kwh=0.0, **arguments
        )


def test_the_mode_table_defaults_to_the_single_layout():
    mode = resolve_optimization_config(_optimizer_config())["mode"]
    assert mode == {"fixed_azimuth": None, "layouts": ["single"], "east_west_tilt_deg": 10.0}


def test_layouts_are_stored_once_each_in_gene_order():
    mode = resolve_optimization_config(_optimizer_config(layouts=["East_West", "single", "east_west"]))["mode"]
    assert mode["layouts"] == ["single", "east_west"]
    assert resolve_optimization_config({**_optimizer_config(), "mode": mode})["mode"] == mode


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [
        ({"layouts": []}, ValueError, "at least 1 entry"),
        ({"layouts": "east_west"}, TypeError, "must be a list"),
        ({"layouts": ["south"]}, ValueError, r"mode\.layouts\[0\]' must be one of"),
        ({"east_west_tilt_deg": 95}, ValueError, "between 0 and 90"),
        ({"layouts": ["east_west"], "fixed_azimuth": 180}, ValueError, "fixed_azimuth sets the azimuth"),
    ],
)
def test_an_invalid_mode_raises_before_the_search(mode, error, message):
    with pytest.raises(error, match=message):
        resolve_optimization_config(_optimizer_config(**mode))


def _problem(year_inputs, **mode):
    pytest.importorskip("pymoo")
    from breos.optimization import SolarDesignProblem

    weather, houseload = year_inputs
    return SolarDesignProblem(weather, houseload, _optimizer_config(**mode), execution_backend="python")


@pytest.mark.parametrize(
    ("mode", "n_var", "layout_gene"),
    [
        ({}, 4, None),
        ({"fixed_azimuth": 180}, 3, None),
        ({"layouts": ["single", "east_west"]}, 5, 4),
        ({"layouts": ["single", "east_west"], "fixed_azimuth": 180}, 4, 3),
        ({"layouts": ["east_west"]}, 2, None),
    ],
)
def test_the_layout_gene_is_added_only_when_layouts_are_compared(year_inputs, mode, n_var, layout_gene):
    problem = _problem(year_inputs, **mode)
    assert (problem.n_var, problem.layout_gene, len(problem.gene_steps)) == (n_var, layout_gene, n_var)
    if layout_gene is not None:
        assert (problem.xl[layout_gene], problem.xu[layout_gene]) == (0, 1)


def test_the_repair_sets_the_unread_genes_of_an_east_west_design(year_inputs):
    from breos.optimization import DiscreteGridRepair

    problem = _problem(year_inputs, layouts=["single", "east_west"])
    X = np.array([[4.2, 1.4, 32.0, 201.0, 0.3], [4.2, 1.4, 32.0, 201.0, 0.7], [4.4, 0.6, 57.0, 140.0, 0.9]])
    repaired = DiscreteGridRepair()._do(problem, X.copy())
    # The single design keeps its snapped orientation; both East-West designs
    # land on the lowest tilt and azimuth, so the first two are duplicates.
    np.testing.assert_array_equal(repaired[0], [4.0, 1.0, 30.0, 200.0, 0.0])
    np.testing.assert_array_equal(repaired[1], [4.0, 1.0, 10.0, 90.0, 1.0])
    np.testing.assert_array_equal(repaired[2], [4.0, 1.0, 10.0, 90.0, 1.0])


def test_the_search_scores_an_east_west_gene_as_the_fixed_design(year_inputs):
    """An East-West candidate is the evaluate_projected_design result, and its constraints count every module."""
    problem = _problem(year_inputs, layouts=["single", "east_west"], east_west_tilt_deg=15)
    weather, houseload = year_inputs
    out: dict = {}
    problem._evaluate(np.array([7.0, 2.0, 40.0, 200.0, 1.0]), out)
    fixed = evaluate_projected_design(
        weather,
        houseload,
        _optimizer_config(),
        n_modules=7,
        battery_kwh=2.0,
        tilt=15.0,
        layout="east_west",
        execution_backend="python",
    )
    for key, value in fixed.metrics.items():
        if key.startswith("Projected_"):
            assert out[key] == value or (np.isnan(value) and np.isnan(out[key])), key

    single: dict = {}
    problem._evaluate(np.array([7.0, 2.0, 40.0, 200.0, 0.0]), single)
    # The same modules and inverter: the CAPEX and area do not depend on the layout.
    assert single["Projected_Initial_Cost"] == out["Projected_Initial_Cost"]
    assert out["G"][0] == out["Projected_Initial_Cost"] - 100000.0
    assert out["G"][1] == pytest.approx(7 * problem.module_area_m2 - 100.0)
    assert single["Projected_PV_DC_Year1_kWh"] != out["Projected_PV_DC_Year1_kWh"]


def test_a_search_over_both_layouts_reports_the_layout_of_every_design(year_inputs):
    pytest.importorskip("pymoo")
    from breos.optimization import optimize_system_multi_objective

    weather, houseload = year_inputs
    config = _optimizer_config(layouts=["single", "east_west"])
    config["simulation"]["years_projection"] = 1
    result = optimize_system_multi_objective(
        weather, houseload, config, pop_size=12, n_gen=2, seed=3, execution_backend="python"
    )
    pareto = result.details["pareto"]
    assert list(pareto.columns[:5]) == ["Modules", "Battery_kWh", "Tilt", "Azimuth", "Layout"]
    assert set(pareto["Layout"]) <= {"single", "east_west"}
    east_west = pareto[pareto["Layout"] == "east_west"]
    assert not east_west.empty and (pareto["Layout"] == "single").any()
    assert (east_west["Tilt"] == 10.0).all() and east_west["Azimuth"].isna().all()
    assert pareto.loc[pareto["Layout"] == "single", "Azimuth"].notna().all()
    assert result.details["provenance"]["mode"]["layouts"] == ["single", "east_west"]


def test_an_east_west_only_search_has_no_orientation_genes(year_inputs):
    pytest.importorskip("pymoo")
    from breos.optimization import optimize_system_multi_objective

    weather, houseload = year_inputs
    config = _optimizer_config(layouts=["east_west"], east_west_tilt_deg=20)
    config["simulation"]["years_projection"] = 1
    pareto = optimize_system_multi_objective(
        weather, houseload, config, pop_size=6, n_gen=1, seed=3, execution_backend="python"
    ).details["pareto"]
    assert (pareto["Layout"] == "east_west").all()
    assert (pareto["Tilt"] == 20.0).all() and pareto["Azimuth"].isna().all()
