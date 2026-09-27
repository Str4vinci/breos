"""App, Monte Carlo and projected optimization age a battery identically.

Each of the three multi-year loops runs one simulation span per project year.
Native rainflow cycling has to continue across those spans: the loops feed
each year's degradation state into the next with ``finalize_degradation``
off, and close the remaining half cycles only at the end of the horizon. A
loop that finalized every year, or dropped the rainflow state, would charge a
different number of cycles for the same design and inputs.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import breos.montecarlo as montecarlo_module
import breos.optimization as optimization_module
from breos.app_config import resolve_app_config
from breos.app_inputs import PreparedSimulationInputs
from breos.battery import align_simulation_inputs
from breos.montecarlo import MonteCarloSettings, _simulate_trajectory
from breos.runners import app as app_runner
from breos.solar import PVProductionBreakdown

PROJECTION_YEARS = 3
STATE_KEYS = (
    "soh_fraction",
    "fec_cum",
    "cumulative_calendar_seconds",
    "cumulative_cycle_degradation",
    "cumulative_calendar_degradation",
    "native_rainflow_state",
)


def _year_inputs():
    """One hourly year whose evening discharge runs past midnight."""
    idx = pd.date_range("2025-01-01 00:00", periods=8760, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    pv = np.where((hour >= 9) & (hour < 16), 2600.0, 0.0)
    load = np.where((hour >= 19) | (hour < 3), 700.0, 250.0)
    return (
        pd.Series(pv, index=idx),
        pd.DataFrame({"Load": load}, index=idx),
        pd.Series(20.0, index=idx),
    )


def _pv_breakdown(pv: pd.Series) -> PVProductionBreakdown:
    zeros = pd.Series(0.0, index=pv.index)
    return PVProductionBreakdown(
        horizontal_reference_dc=pv,
        poa_global_dc=pv,
        front_effective_irradiance_dc=pv,
        rear_gain_dc=zeros,
        effective_irradiance_dc=pv,
        module_dc=pv,
        dc_after_static_losses=pv,
        dc_after_losses=pv,
        pvwatts_component_losses={},
        pvwatts_components_pct={},
        pvwatts_combined_pct=0.0,
        age_degradation_pct=0.0,
        age_degradation_loss=zeros,
    )


def _state_record(state, n_replacements):
    return {"n_replacements": int(n_replacements), **{key: state[key] for key in STATE_KEYS}}


def _spy(monkeypatch, module, name, records, calls):
    real = getattr(module, name)

    def _wrapped(*args, **kwargs):
        calls.append(kwargs.get("finalize_degradation"))
        output = real(*args, **kwargs)
        if name == "simulate_energy_balance_summary":
            records.append(_state_record(output.final_degradation_state, output.n_replacements))
        else:
            records.append(_state_record(output[-1], output[4]))
        return output

    monkeypatch.setattr(module, name, _wrapped)


@pytest.mark.parametrize("eol_percentage", [0.70, 0.99], ids=["no-replacement", "replacement"])
def test_app_montecarlo_and_projected_optimization_age_the_battery_identically(monkeypatch, eol_percentage):
    pv, load, temperature = _year_inputs()
    resolved = resolve_app_config(
        {
            "location": "porto",
            "n_modules": 8,
            "annual_consumption_kwh": 4000,
            "battery_kwh": 5.0,
            "cost_preset": "residential_pt",
            "resolution": "h",
            "projection_years": PROJECTION_YEARS,
            "pv_degradation_rate": 0.005,
            "battery_eol_percentage": eol_percentage,
        }
    )
    cfg = resolved.cfg
    assert cfg["degradation_engine"] == "native"
    inverter_ac_capacity_w = cfg["n_modules"] * resolved.avg_module_power_w / cfg["inverter_loading_ratio"]

    # App: its own year loop over the prepared inputs.
    app_records, app_calls = [], []
    inputs = PreparedSimulationInputs(
        weather=pd.DataFrame(index=pv.index),
        dc_system_base=pv,
        load_data=load,
        temperature_series=temperature,
        pv_breakdown=_pv_breakdown(pv),
    )
    monkeypatch.setattr(app_runner, "prepare_simulation_inputs", lambda cfg, resolved, deps: inputs)
    _spy(monkeypatch, app_runner, "simulate_energy_balance", app_records, app_calls)
    app_artifacts = app_runner.run_app_simulation(cfg, resolved, deps=SimpleNamespace())

    # Monte Carlo: one trajectory with a single weather year and no load
    # uncertainty, so the draw is the App's inputs exactly.
    mc_records, mc_calls = [], []
    _spy(monkeypatch, montecarlo_module, "simulate_energy_balance_summary", mc_records, mc_calls)
    mc_metrics, _trajectory = _simulate_trajectory(
        cfg,
        resolved,
        np.array([2025]),
        PROJECTION_YEARS,
        MonteCarloSettings(weather_file="unused.csv", load_uncertainty=0.0),
        np.random.default_rng(0),
        {2025: align_simulation_inputs(pv, load, temperature, freq="h")},
        None,
    )

    # Projected optimization: the year loop behind evaluate_projected_design.
    opt_records, opt_calls = [], []
    _spy(monkeypatch, optimization_module, "simulate_energy_balance", opt_records, opt_calls)
    batt_spec = {
        "min_soc": cfg["battery_min_soc"],
        "max_soc": cfg["battery_max_soc"],
        "eol_percentage": cfg["battery_eol_percentage"],
        "dc_coupled": cfg["dc_coupled"],
        "calendar_model": cfg["calendar_model"],
        "max_charge_power_w": cfg["battery_max_charge_power_w"],
        "max_discharge_power_w": cfg["battery_max_discharge_power_w"],
        "power_limit_c_rate": cfg["battery_power_limit_c_rate"],
        "enable_resistance_fade": cfg["enable_resistance_fade"],
    }
    opt_metrics = optimization_module._evaluate_projected_design_metrics(
        base_dc_power=pv,
        tmy_data=pd.DataFrame(index=pv.index),
        houseload=load,
        temperature_series=temperature,
        pv_params=resolved.pv_params,
        batt_spec=batt_spec,
        costs_cfg={},
        fin_cfg={},
        freq="h",
        years_projection=PROJECTION_YEARS,
        degradation_rate=cfg["pv_degradation_rate"],
        n_modules=cfg["n_modules"],
        battery_kwh=cfg["battery_kwh"],
        inverter_efficiency=cfg["inverter_efficiency"],
        inverter_ac_capacity_w=inverter_ac_capacity_w,
    )

    last_year_only = [False] * (PROJECTION_YEARS - 1) + [True]
    assert app_calls == mc_calls == opt_calls == last_year_only
    assert len(app_records) == len(mc_records) == len(opt_records) == PROJECTION_YEARS
    for year, (app_year, mc_year, opt_year) in enumerate(zip(app_records, mc_records, opt_records, strict=True)):
        for key in ("n_replacements", *STATE_KEYS):
            assert mc_year[key] == pytest.approx(app_year[key], rel=1e-12, abs=1e-12), (year, key)
            assert opt_year[key] == pytest.approx(app_year[key], rel=1e-12, abs=1e-12), (year, key)

    # The rainflow residue stays open between years and is closed at the end.
    assert all(record["native_rainflow_state"]["current"] is not None for record in app_records[:-1])
    assert app_records[-1]["native_rainflow_state"]["residue"] == []
    assert app_records[-1]["fec_cum"] > 0.0

    final_soh = app_artifacts.yearly_df["Battery_SOH_%"].iloc[-1]
    assert mc_metrics["final_soh_pct"] == pytest.approx(final_soh, rel=1e-12)
    assert opt_metrics["Projected_Final_SOH_%"] == pytest.approx(final_soh, rel=1e-12)
    total_replacements = int(app_artifacts.yearly_df["Replacements"].sum())
    assert mc_metrics["total_replacements"] == total_replacements
    assert opt_metrics["Projected_Total_Replacements"] == total_replacements
    if eol_percentage == 0.99:
        assert total_replacements > 0
