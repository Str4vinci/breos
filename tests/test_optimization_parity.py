"""Optimizer/App parity guardrails (2026-07 audit follow-up).

The NSGA-II optimizer must score candidate designs with the same model the
App reports for the winning design:

- projection: candidates run the shared multi-year projection loop and its
  economics, so a returned design reproduces through the App;
- inverter: candidates are simulated with the AC nameplate their CAPEX pays
  for (``pv_peak / dc_ac_ratio``), i.e. clipping applies during scoring;
- load alignment: the raw load frame reaches ``simulate_energy_balance``
  unmangled, so its timezone-aware alignment applies (no positional
  re-stamping).
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _stub_projection_balance

COSTS_CONFIG = {
    "electricity_cost": 0.25,
    "electricity_sold_cost": 0.07,
    "daily_power_cost": 0.50,  # cancels out of savings; nonzero to prove it
    "module_cost_per_w": 0.15,
    "storage_cost_per_kwh": 400.0,
    "dc_ac_ratio": 1.25,
    "installation_cost_per_module": 300.0,
    "maintenance_cost_per_panel": 12.0,
    "maintenance_cost": 30.0,
}
FINANCIALS_CONFIG = {
    "inflation_rate": 0.03,
    "sell_price_inflation": 0.015,
    "discount_rate": 0.04,
    "pv_degradation_rate": 0.007,
    "project_lifespan": 20,
}


# ---------------------------------------------------------------------------
# SolarDesignProblem wiring (requires pymoo)
# ---------------------------------------------------------------------------


def _problem_config(dc_ac_ratio: float = 1.6):
    return {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h", "years_projection": 1},
        "constraints": {"budget": 100000, "max_area_m2": 100.0, "max_modules": 5},
        "mode": {"fixed_azimuth": 180},
        "pv": {"module": "Suntech_STP550S_STC"},
        "battery": {"temperature": 20.0, "indoor_model": {"enabled": False}},
        "costs": dict(COSTS_CONFIG, dc_ac_ratio=dc_ac_ratio),
        "financials": FINANCIALS_CONFIG,
    }


def _run_evaluate(monkeypatch, config, houseload, tmy_index):
    pytest.importorskip("pymoo")
    from breos.optimization import SolarDesignProblem

    tmy_data = pd.DataFrame({"temp_air": 15.0, "ghi": 0.0}, index=tmy_index)
    dc = pd.Series(0.0, index=tmy_index)
    captured: dict = {}

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    _stub_projection_balance(monkeypatch, tmy_index, captured, Houseload=500.0, Import_From_Grid=500.0)

    problem = SolarDesignProblem(tmy_data, houseload, config)
    out: dict = {}
    problem._evaluate(np.array([2.0, 1.0, 10.0], dtype=float), out)
    captured["out"] = out
    return captured


def test_optimizer_applies_capex_matched_ac_clipping(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    captured = _run_evaluate(monkeypatch, _problem_config(dc_ac_ratio=1.6), houseload, idx)

    # 2 modules x 550 Wp / 1.6 — the same nameplate the CAPEX pays for
    assert captured["battery_config"].inverter_ac_capacity_w == pytest.approx(2 * 550 / 1.6)


def test_optimizer_prefers_app_top_level_inverter_efficiency(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    config = _problem_config()
    config["inverter_efficiency"] = 0.80
    config["inverter"] = {"efficiency": 0.91}

    captured = _run_evaluate(monkeypatch, config, houseload, idx)

    assert captured["battery_config"].inverter_efficiency == pytest.approx(0.80)


def test_optimizer_propagates_battery_power_caps(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    config = _problem_config()
    config["battery"].update(
        {
            "max_charge_power_w": 1750.0,
            "max_discharge_power_w": 2250.0,
        }
    )

    captured = _run_evaluate(monkeypatch, config, houseload, idx)

    # Frozen convention: charge is DC input-side power; discharge is AC
    # delivered-to-load power. BatteryConfig enforces both during dispatch.
    assert captured["battery_config"].max_charge_power_w == pytest.approx(1750.0)
    assert captured["battery_config"].max_discharge_power_w == pytest.approx(2250.0)


def test_optimizer_passes_load_with_original_timestamps(monkeypatch):
    tmy_index = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    load_index = pd.date_range("2024-01-01 00:00", periods=2, freq="h", tz="Europe/Lisbon")
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=load_index)
    captured = _run_evaluate(monkeypatch, _problem_config(), houseload, tmy_index)

    # The load must reach simulate_energy_balance with its real timestamps —
    # its internal alignment (UTC instants, year remap) does the rest.
    assert captured["houseload"].index.equals(load_index)


def test_projected_design_forwards_pv_model_options(monkeypatch):
    """A configured PV chain must reach the PV model, not just provenance.

    The optimization paths once dropped ``transposition_model`` and its
    siblings on the floor: ``app_config`` validated them and
    ``reproduction.json`` recorded them, while every call ran the defaults. The
    output was byte-identical to an unconfigured run, so nothing downstream
    could notice — only the provenance record was wrong. Assert the kwargs
    arrive rather than asserting on numbers, since the whole failure mode is
    that the numbers do not move.
    """
    import breos.optimization as optimization

    seen: dict[str, object] = {}

    class _Stop(RuntimeError):
        pass

    def _spy(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(optimization, "calculate_pv_production_dc", _spy)

    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon"},
        "simulation": {"resolution": "h"},
        "pv": {"degradation_rate": 0.005},
        "battery": {},
        "transposition_model": "perez",
        "solar_position": "mid-interval",
        "diffuse_iam": "marion",
    }

    with pytest.raises(_Stop):
        optimization.evaluate_projected_design(
            tmy_data=pd.DataFrame(),
            houseload=pd.DataFrame(),
            config=config,
            n_modules=9,
            battery_kwh=5.0,
            tilt=35.0,
            azimuth=180.0,
        )

    assert seen["transposition_model"] == "perez"
    assert seen["solar_position"] == "mid-interval"
    assert seen["diffuse_iam"] == "marion"


def test_config_model_options_omits_absent_keys():
    """Absent keys stay absent so the PV model's own defaults still apply."""
    from breos.pv.model_options import configured_pv_model_kwargs

    assert configured_pv_model_kwargs({}) == {}
    assert configured_pv_model_kwargs({"albedo": 0.25}) == {"albedo": 0.25}


def test_optimizer_battery_defaults_match_app_defaults():
    """An unset battery setting resolves to the App's default, not a local one.

    The optimizer used to carry its own fallbacks (10-90% became 20-80%, and
    0.9795 each way instead of a 95% round trip), so a spec that omitted them
    gave a 5 kWh pack a 3 kWh window against the App's 4 kWh.
    """
    from breos.app_config import DEFAULTS
    from breos.battery import BatteryConfig
    from breos.optimization import _build_battery_config_from_spec

    optimizer = _build_battery_config_from_spec({}, nominal_energy_wh=5000.0)
    reference = BatteryConfig(nominal_energy_wh=5000.0)

    assert optimizer.min_soc == DEFAULTS["battery_min_soc"] == reference.min_soc
    assert optimizer.max_soc == DEFAULTS["battery_max_soc"] == reference.max_soc
    assert optimizer.eol_percentage == DEFAULTS["battery_eol_percentage"] == reference.eol_percentage
    # The App leaves efficiency to BatteryConfig unless battery_rte is set.
    assert DEFAULTS["battery_rte"] is None
    assert optimizer.charge_efficiency * optimizer.discharge_efficiency == pytest.approx(0.95)
    for field in (
        "charge_efficiency",
        "discharge_efficiency",
        "standby_loss_wh",
        "battery_type",
        "dc_coupled",
        "calendar_model",
        "enable_resistance_fade",
    ):
        assert getattr(optimizer, field) == getattr(reference, field), field


def test_optimizer_scores_an_unset_battery_window_with_app_defaults(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)

    captured = _run_evaluate(monkeypatch, _problem_config(), houseload, idx)

    assert captured["battery_config"].min_soc == pytest.approx(0.10)
    assert captured["battery_config"].max_soc == pytest.approx(0.90)


def test_projected_budget_constraint_gates_the_reported_capex(synthetic_weather, sample_load):
    """The budget checks the CAPEX the projected result reports.

    The constraint used the steady-state CAPEX, which the removed
    ``costs.panel_wp`` could price at 400 W while the reported figure priced
    the selected 550 W module, so a EUR 480 budget accepted a design reported
    at EUR 490.03.
    """
    pytest.importorskip("pymoo")
    from breos.optimization import SolarDesignProblem

    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h", "years_projection": 1},
        "constraints": {"budget": 480.0, "max_area_m2": 100.0, "max_modules": 5},
        "optimization": {"objective_basis": "projected"},
        "mode": {"fixed_azimuth": 180},
        "pv": {"module": "Suntech_STP550S_STC"},
        "battery": {"temperature": 20.0, "indoor_model": {"enabled": False}},
    }
    problem = SolarDesignProblem(synthetic_weather, sample_load, config)
    out: dict = {}
    problem._evaluate(np.array([1.0, 0.0, 35.0], dtype=float), out)

    assert out["Projected_Initial_Cost"] == pytest.approx(490.03, abs=0.01)
    assert out["G"][0] == pytest.approx(out["Projected_Initial_Cost"] - 480.0)
    assert out["G"][0] > 0.0


def test_optimizer_honours_an_explicit_replacement_cost(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)

    import breos.optimization as optimization

    priced_at = []
    real_projection = optimization.cost_analysis_projection

    def spy(*args, **kwargs):
        priced_at.append(kwargs["costs"]["replacement_cost_each"])
        return real_projection(*args, **kwargs)

    monkeypatch.setattr(optimization, "cost_analysis_projection", spy)
    _run_evaluate(monkeypatch, _problem_config(), houseload, idx)
    config = _problem_config()
    config["battery"]["replacement_cost"] = 1234.0
    _run_evaluate(monkeypatch, config, houseload, idx)

    # 1 kWh at the configured 400/kWh, unless the config names a cost. The
    # economics prices it; the battery carries no money (ADR 0003 E4).
    assert priced_at == pytest.approx([400.0, 1234.0])


def test_projected_optimizer_candidate_matches_app(open_meteo_weather, monkeypatch):
    """A returned projected design reproduces through the public App facade.

    The one-year horizon makes the optimizer's aggregate projected grid
    independence the same basis as App's first-year headline. App rounds its
    public JSON-facing values to two decimals, so 0.0051 is the serialization
    bound in percentage points or euros, not a model tolerance.
    """
    pytest.importorskip("pymoo")

    from breos.app import App
    from breos.app_inputs import AppRuntimeDependencies
    from breos.optimization import optimize_system_multi_objective
    from breos.weather import build_battery_temperature_series

    # A full year: App rejects weather that does not cover the calendar year
    # of start_date, and over a single January day the site-altitude
    # difference between App and the optimizer (0 m instead of pvlib's
    # elevation lookup) stayed inside App's rounding and went unnoticed.
    idx = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
    weather = open_meteo_weather(idx)
    houseload = pd.DataFrame({"Load": [500.0] * len(idx)}, index=idx)
    financials = dict(FINANCIALS_CONFIG, project_lifespan=1)
    optimizer_config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h", "years_projection": 1},
        "constraints": {
            "budget": 100000.0,
            "max_area_m2": 100.0,
            "max_modules": 4,
            "max_battery_kwh": 2.0,
            "max_tilt_deg": 30.0,
        },
        "optimization": {"objective_basis": "projected", "early_stop": False},
        "mode": {"fixed_azimuth": 180},
        "pv": {"module": "Suntech_STP550S_STC", "degradation_rate": financials["pv_degradation_rate"]},
        # Leave battery window, efficiency, and degradation settings unset in
        # both paths so the public defaults resolve the candidate identically.
        "battery": {"temperature": 20.0, "indoor_model": {"enabled": False}},
        "costs": COSTS_CONFIG,
        "financials": financials,
    }
    optimized = optimize_system_multi_objective(
        weather,
        houseload,
        optimizer_config,
        pop_size=8,
        n_gen=1,
        seed=42,
        verbose=False,
    )
    pareto = optimized.details["pareto"]
    battery_candidates = pareto.loc[pareto["Battery_kWh"] > 0.0]
    assert not battery_candidates.empty
    candidate = battery_candidates.iloc[0]

    # The App facade normally gets these inputs from its weather and load
    # providers. Injecting the same frames keeps this end-to-end check
    # offline and makes both workflows simulate exactly the same interval.
    dependencies = AppRuntimeDependencies(
        load_profile=lambda **kwargs: houseload.copy(),
        load_weather=lambda **kwargs: None,
        fetch_tmy_weather_data=lambda **kwargs: (weather.copy(), {}),
        resample_to_15min=lambda frame, **kwargs: frame,
        build_battery_temperature_series=build_battery_temperature_series,
    )
    monkeypatch.setattr(App, "_runtime_dependencies", staticmethod(lambda: dependencies))

    app_costs = {key: value for key, value in COSTS_CONFIG.items() if key != "dc_ac_ratio"}
    app_config = {
        "location": optimizer_config["location"],
        "n_modules": int(candidate["Modules"]),
        "annual_consumption_kwh": float(houseload["Load"].sum() / 1000.0),
        "battery_kwh": float(candidate["Battery_kWh"]),
        "pv_module": optimizer_config["pv"]["module"],
        "tilt": float(candidate["Tilt"]),
        "azimuth": float(candidate["Azimuth"]),
        "projection_years": financials["project_lifespan"],
        "resolution": "h",
        "start_date": "2023-01-01",
        "costs": app_costs,
        "inverter_loading_ratio": COSTS_CONFIG["dc_ac_ratio"],
        "inflation_rate": financials["inflation_rate"],
        "sell_price_inflation": financials["sell_price_inflation"],
        "discount_rate": financials["discount_rate"],
        "pv_degradation_rate": financials["pv_degradation_rate"],
        "battery_temperature": optimizer_config["battery"]["temperature"],
        "battery_indoor_model": optimizer_config["battery"]["indoor_model"],
        "execution_backend": "python",
    }
    app = App(app_config)
    app.simulate()
    app_result = app.result()

    app_rounding = {"abs": 0.0051, "rel": 0.0}
    assert app_result["grid_independence_pct"] == pytest.approx(
        candidate["Projected_Grid_Independence_%"], **app_rounding
    )
    assert app_result["npv_savings"] == pytest.approx(candidate["Projected_NPV"], **app_rounding)
    assert app_result["total_investment"] == pytest.approx(candidate["Projected_Initial_Cost"], **app_rounding)


def test_optimizer_site_uses_the_same_altitude_as_app():
    pvlib_location = pytest.importorskip("pvlib.location")
    from breos.optimization import _site_location

    looked_up = _site_location({"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon"})
    app_site = pvlib_location.Location(41.15, -8.61, tz="Europe/Lisbon")
    assert looked_up.altitude == app_site.altitude
    assert looked_up.altitude > 0.0  # Porto is not at sea level

    explicit = _site_location({"latitude": 41.15, "longitude": -8.61, "altitude": 0})
    assert explicit.altitude == 0.0
