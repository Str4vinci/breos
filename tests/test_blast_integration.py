"""End-to-end BLAST integration tests through the real dispatch and runner.

These exercise ``simulate_energy_balance`` and ``run_app_simulation`` with real
BLAST degradation (no mocked degradation model), covering sub-hourly resolution,
multi-replacement projections, and split/continuous threading equivalence.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from breos.app_inputs import PreparedSimulationInputs
from breos.battery import BatteryConfig, simulate_energy_balance
from breos.runners import app as app_runner
from breos.solar import PVProductionBreakdown

# Energy-ledger closure identities (Wh; hourly => W and Wh are numerically equal).
# Both hold on every timestep row, including battery-replacement rows.


def _assert_energy_ledger_closes(results: pd.DataFrame, atol: float = 1e-6) -> None:
    delta = (
        results["Battery_Charge_Stored"]
        - results["Battery_Discharge_DC"]
        - results["Standby_Loss"]
        - results["Capacity_Window_Loss"]
        - results["Battery_Replacement_Energy_Removed"]
        + results["Battery_Replacement_Energy_Added"]
    )
    np.testing.assert_allclose(results["Battery_Energy_Delta"], delta, atol=atol)

    lhs = results["PV_DC"] + results["Battery_Replacement_Energy_Added"]
    rhs = (
        results["PV_AC_To_Load"]
        + results["PV_AC_Export"]
        + results["Battery_AC_To_Load"]
        + results["PV_DC_Curtailed"]
        + results["Battery_Charge_Loss"]
        + results["Battery_Discharge_Loss"]
        + results["PV_Direct_Inverter_Loss"]
        + results["Battery_Inverter_Loss"]
        + results["Standby_Loss"]
        + results["Capacity_Window_Loss"]
        + results["Battery_Replacement_Energy_Removed"]
        + results["Battery_Energy_Delta"]
    )
    np.testing.assert_allclose(lhs, rhs, atol=atol)


# ---------------------------------------------------------------------------
# Test 2: BLAST at 15-minute resolution
# ---------------------------------------------------------------------------


def test_blast_15min_resolution_two_days():
    """Two full 15-minute days: 96 post-step samples + a prepended anchor each."""
    steps = 192  # two full days at 15-minute resolution
    idx = pd.date_range("2025-06-01 00:00", periods=steps, freq="15min", tz="UTC")
    quarter_of_day = np.arange(steps) % 96
    pv_dc = pd.Series(np.where((quarter_of_day >= 32) & (quarter_of_day < 64), 4200.0, 0.0), index=idx)
    houseload = pd.DataFrame(
        {"Load": np.where((quarter_of_day < 32) | (quarter_of_day >= 64), 1400.0, 250.0)},
        index=idx,
    )
    temperature = pd.Series(28.0, index=idx)
    config = BatteryConfig(nominal_energy_wh=5000, standby_loss_wh=0.0, enable_replacement=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_df, _total_pv, summary_df, _cost, _n_rep, degradation_df, degradation_state = simulate_energy_balance(
            pv_dc=pv_dc,
            houseload=houseload,
            battery_config=config,
            freq="15min",
            temperature_series=temperature,
            degradation_engine="blast",
            blast_model="nmc811_grsi_lgm50_5ah",
            return_degradation_state=True,
        )

    # Exactly two daily degradation updates (one per 96-sample day).
    assert len(degradation_df) == 2

    snapshot = degradation_state["blast_engine"]
    stressors = snapshot["stressors"]
    # BLAST derives per-day elapsed time from the anchor-prepended endpoint grid,
    # so two days advance the model clock by exactly two days.
    assert stressors["t_days"][-1] == pytest.approx(2.0)
    # Each daily endpoint grid holds 96 post-step samples plus the prior anchor.
    assert stressors["delta_t_days"][-1] == pytest.approx(1.0)

    # FEC is consistent across the engine, the daily frame, and the carry state.
    engine_efc = stressors["efc"][-1]
    assert degradation_df["Cumulative_FEC"].iloc[-1] == pytest.approx(engine_efc)
    assert degradation_state["fec_cum"] == pytest.approx(engine_efc)
    assert engine_efc > 0.0

    # Finite SOH and finite newest internal states/outputs.
    final_soh = summary_df["Final SOH [%]"].iloc[0]
    assert np.isfinite(final_soh)
    assert 0.0 < final_soh <= 100.0
    for group_name in ("states", "outputs"):
        for values in snapshot[group_name].values():
            assert np.isfinite(values[-1])

    _assert_energy_ledger_closes(results_df)


# ---------------------------------------------------------------------------
# Test 3: multiple replacements through run_app_simulation
# ---------------------------------------------------------------------------


def _pv_breakdown(pv: pd.Series) -> PVProductionBreakdown:
    zeros = pd.Series(0.0, index=pv.index)
    return PVProductionBreakdown(
        horizontal_reference_dc=pv,
        poa_global_dc=pv,
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


def test_blast_multiple_replacements_through_runner(monkeypatch):
    """A 20-year projection with the real LG MJ1 model triggers several replacements."""
    idx = pd.date_range("2025-01-01 00:00", periods=8760, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    # Deterministic aggressive daily cycling at a warm cell temperature: charge
    # hard midday, discharge deeply overnight, repeated every year.
    pv = pd.Series(np.where((hour >= 8) & (hour < 16), 5000.0, 0.0), index=idx)
    load = pd.DataFrame({"Load": np.where((hour < 8) | (hour >= 16), 1600.0, 200.0)}, index=idx)
    temperature = pd.Series(35.0, index=idx)
    inputs = PreparedSimulationInputs(
        weather=pd.DataFrame(index=idx),
        dc_system_base=pv,
        load_data=load,
        temperature_series=temperature,
        pv_breakdown=_pv_breakdown(pv),
    )

    costs = {
        "electricity_cost": 0.25,
        "electricity_sold_cost": 0.05,
        "daily_power_cost": 0.3,
        "annual_operation_cost": 50.0,
        "total_initial_cost": 12000.0,
    }
    monkeypatch.setattr(app_runner, "prepare_simulation_inputs", lambda cfg, resolved, deps: inputs)
    monkeypatch.setattr(app_runner, "build_costs_dict", lambda cfg, resolved: costs)

    # Wrap (do not mock) the real degradation call to retain each year's frames.
    captured_years: list[tuple] = []
    real_simulate = app_runner.simulate_energy_balance

    def _capturing_simulate(*args, **kwargs):
        result = real_simulate(*args, **kwargs)
        captured_years.append(result)
        return result

    monkeypatch.setattr(app_runner, "simulate_energy_balance", _capturing_simulate)

    inflation_rate = 0.03
    max_soc = 0.9
    battery_kwh = 5.0
    cfg = {
        "resolution": "h",
        "battery_kwh": battery_kwh,
        "projection_years": 20,
        "pv_degradation_rate": 0.005,
        "n_modules": 10,
        "inverter_loading_ratio": 1.25,
        "battery_rte": None,
        "battery_max_charge_power_w": None,
        "battery_max_discharge_power_w": None,
        "enable_resistance_fade": False,
        "battery_eol_percentage": 0.8,
        "battery_max_soc": max_soc,
        "battery_min_soc": 0.1,
        "dc_coupled": True,
        "inverter_efficiency": 0.96,
        "calendar_model": "naumann_lam_field_calibrated",
        "inflation_rate": inflation_rate,
        "sell_price_inflation": 0.0,
        "discount_rate": 0.04,
        "degradation_engine": "blast",
        "blast_model": "nmc811_grsi_lgmj1_4ah",
    }
    resolved = SimpleNamespace(
        cost_params=SimpleNamespace(battery_cost_per_kwh=500.0),
        avg_module_power_w=400.0,
        emissions_params=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifacts = app_runner.run_app_simulation(cfg, resolved, deps=SimpleNamespace())

    assert len(captured_years) == cfg["projection_years"]
    per_year_counts = [year_result[4] for year_result in captured_years]
    per_year_costs = [year_result[3] for year_result in captured_years]

    # Multiple replacement events occur (stable minimum for this committed fixture;
    # the deterministic run currently produces 11).
    assert artifacts.total_replacements >= 5

    # Per-year counts and costs sum to the artifact totals.
    assert sum(per_year_counts) == artifacts.total_replacements
    assert sum(per_year_costs) == pytest.approx(artifacts.total_replacement_cost)

    # Replacement events match the degradation summary.
    expected_events = [
        {"year": year_idx + 1, "count": count} for year_idx, count in enumerate(per_year_counts) if count
    ]
    assert artifacts.degradation_summary["replacement_events"] == expected_events
    assert len(expected_events) >= 5

    # cost_projection replacement column carries the per-year cost with the
    # configured inflation treatment: cost * (1 + inflation)^(year - 1).
    cost_projection = artifacts.cost_projection
    inflation_factors = (1 + inflation_rate) ** (cost_projection["Year"].to_numpy() - 1)
    expected_replacement_cost = artifacts.yearly_df["Replacement_Cost"].to_numpy() * inflation_factors
    np.testing.assert_allclose(cost_projection["Cost_Replacement"].to_numpy(), expected_replacement_cost, atol=1e-9)

    replacement_reset_energy = battery_kwh * 1000.0 * max_soc
    replacement_years_seen = 0
    for year_idx, year_result in enumerate(captured_years):
        results_df = year_result[0]
        degradation_df = year_result[5]
        year_replacements = year_result[4]

        # Ledger closes on every row of every year, including replacement rows.
        _assert_energy_ledger_closes(results_df)

        if not year_replacements:
            continue
        replacement_years_seen += 1
        replacement_rows = results_df[results_df["Battery_Replaced"]]
        assert len(replacement_rows) == year_replacements

        # SOH sawtooth reset: replacement rows snap SOH and stored energy back to
        # a fresh full-health pack.
        np.testing.assert_allclose(replacement_rows["Battery_SOH"].to_numpy(), 100.0, atol=1e-6)
        np.testing.assert_allclose(replacement_rows["Battery_Energy"].to_numpy(), replacement_reset_energy, atol=1e-6)
        np.testing.assert_allclose(
            replacement_rows["Battery_Replacement_Energy_Added"].to_numpy(), replacement_reset_energy, atol=1e-6
        )

        # Engine EFC/state reset: the replacement day records a zeroed FEC and a
        # restored SOH, and the daily SOH series dips to EOL then jumps back up.
        reset_days = degradation_df[degradation_df["SOH"] >= 99.999]
        assert not reset_days.empty
        np.testing.assert_allclose(reset_days["Cumulative_FEC"].to_numpy(), 0.0, atol=1e-9)
        # SOH degrades to within ~1 point of the EOL threshold before resetting;
        # the crossing day is snapped straight to 100, so the recorded low sits
        # just above EOL.
        eol_pct = cfg["battery_eol_percentage"] * 100.0
        assert degradation_df["SOH"].min() <= eol_pct + 1.0
        assert np.diff(degradation_df["SOH"].to_numpy()).max() > 10.0

    assert replacement_years_seen >= 5


# ---------------------------------------------------------------------------
# Test 4: varying-profile split/thread equivalence
# ---------------------------------------------------------------------------


def _two_year_hourly_profile() -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    # Start at 10:00 so the year boundary (hour 8760) lands mid-morning while the
    # battery is actively charging — a genuine mid-swing split, not idle/saturated.
    idx = pd.date_range("2025-01-01 10:00", periods=2 * 8760, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    day_of_year = idx.dayofyear.to_numpy()
    season = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    pv = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, 1) * (2500 + 3000 * season)
    load = (
        350.0
        + 650.0 * np.exp(-(((hour - 20) % 24) ** 2) / 6.0)
        + 450.0 * np.exp(-(((hour - 7) % 24) ** 2) / 6.0)
        + 200.0
    )
    return (
        pd.Series(pv, index=idx),
        pd.DataFrame({"Load": load}, index=idx),
        pd.Series(20.0 + 8.0 * season, index=idx),
    )


@pytest.mark.parametrize(
    "blast_model",
    [
        "nca_grsi_sonymurata_2p5ah",  # sigmoid loss states
        "nmc_lto_10ah",  # capacity-gain and power loss states
    ],
)
def test_blast_split_matches_continuous_at_mid_swing_boundary(blast_model):
    pv, load, temperature = _two_year_hourly_profile()
    boundary = 8760
    config = BatteryConfig(nominal_energy_wh=8000, standby_loss_wh=0.0, enable_replacement=False)
    common = dict(
        battery_config=config,
        freq="h",
        degradation_engine="blast",
        blast_model=blast_model,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        continuous_results, *_, continuous_degradation, continuous_state = simulate_energy_balance(
            pv_dc=pv,
            houseload=load,
            temperature_series=temperature,
            return_degradation_state=True,
            **common,
        )

        year_one_results, *_, year_one_degradation, year_one_state = simulate_energy_balance(
            pv_dc=pv.iloc[:boundary],
            houseload=load.iloc[:boundary],
            temperature_series=temperature.iloc[:boundary],
            return_degradation_state=True,
            **common,
        )
        carried_energy = float(year_one_results["Battery_Energy_End"].iloc[-1])
        carried_pv_origin = float(year_one_results["Battery_PV_Origin_Energy_End"].iloc[-1])

        year_two_results, *_, year_two_degradation, year_two_state = simulate_energy_balance(
            pv_dc=pv.iloc[boundary:],
            houseload=load.iloc[boundary:],
            temperature_series=temperature.iloc[boundary:],
            initial_energy_wh=carried_energy,
            initial_pv_origin_energy_wh=carried_pv_origin,
            initial_degradation_state=year_one_state,
            return_degradation_state=True,
            **common,
        )

    # The split really occurs at a varying/mid-swing boundary, not a rail.
    soc = continuous_results["Battery_SOC_Absolute"].to_numpy()
    boundary_soc = soc[boundary]
    assert 0.15 < boundary_soc < 0.85, boundary_soc
    assert abs(soc[boundary] - soc[boundary - 1]) > 0.05
    assert abs(soc[boundary + 1] - soc[boundary]) > 0.05

    # Final SOH matches to a tight tolerance.
    assert year_two_degradation["SOH"].iloc[-1] == pytest.approx(continuous_degradation["SOH"].iloc[-1], abs=1e-9)

    # Every current engine state / output / stressor matches (skip initial NaN
    # sentinels by comparing only the newest value of each array).
    for group_name in ("states", "outputs", "stressors"):
        continuous_group = continuous_state["blast_engine"][group_name]
        split_group = year_two_state["blast_engine"][group_name]
        assert continuous_group.keys() == split_group.keys()
        for field in continuous_group:
            continuous_value = continuous_group[field][-1]
            split_value = split_group[field][-1]
            if np.isnan(continuous_value) and np.isnan(split_value):
                continue
            assert split_value == pytest.approx(continuous_value, abs=1e-9, rel=1e-12), f"{group_name}.{field}"

    # FEC, stored energy, and PV-origin inventory all match.
    assert year_two_state["blast_engine"]["stressors"]["efc"][-1] == pytest.approx(
        continuous_state["blast_engine"]["stressors"]["efc"][-1], abs=1e-9
    )
    assert year_two_results["Battery_Energy_End"].iloc[-1] == pytest.approx(
        continuous_results["Battery_Energy_End"].iloc[-1], abs=1e-6
    )
    assert year_two_results["Battery_PV_Origin_Energy_End"].iloc[-1] == pytest.approx(
        continuous_results["Battery_PV_Origin_Energy_End"].iloc[-1], abs=1e-6
    )
