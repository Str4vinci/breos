"""Aligned inputs carry their resolution; PV_Production has one definition (#214)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from breos.battery import (
    BatteryConfig,
    align_simulation_inputs,
    simulate_energy_balance,
    simulate_energy_balance_summary,
)

_BACKENDS = ["python"] + (["numba"] if importlib.util.find_spec("numba") else [])


def _inputs(freq="h", periods=48):
    index = pd.date_range("2025-01-01", periods=periods, freq=freq, tz="UTC")
    steps_per_day = periods // 2
    phase = np.arange(periods) % steps_per_day / steps_per_day
    pv_dc = pd.Series(6000.0 * np.clip(np.sin((phase - 0.25) * 2.0 * np.pi), 0.0, None), index=index)
    load = pd.DataFrame({"Load": 400.0 + 600.0 * (phase > 0.75)}, index=index)
    return pv_dc, load


def test_aligned_inputs_run_at_their_own_resolution():
    # 96 quarter-hours of 1 kW are 24 kWh; reading them as hours gave 96 kWh.
    pv_dc, load = _inputs("15min", periods=96)
    load["Load"] = 1000.0
    aligned = align_simulation_inputs(pv_dc, load, freq="15min")

    summary = simulate_energy_balance_summary(aligned=aligned)

    assert aligned.freq == "15min"
    assert summary.hours_per_step == 0.25
    assert summary.summary_row["Total Load [kWh]"] == pytest.approx(24.0)
    # An alias for the same step is accepted.
    assert simulate_energy_balance_summary(aligned=aligned, freq="15T").summary_row == summary.summary_row


def test_a_freq_that_disagrees_with_aligned_inputs_raises():
    pv_dc, load = _inputs("15min", periods=96)
    aligned = align_simulation_inputs(pv_dc, load, freq="15min")

    with pytest.raises(ValueError, match="freq='h' disagrees with the aligned inputs, which step at '15min'"):
        simulate_energy_balance_summary(aligned=aligned, freq="h")
    with pytest.raises(ValueError, match="disagrees with the aligned inputs"):
        aligned.with_pv_only_chain(BatteryConfig(nominal_energy_wh=0.0), freq="h")


def test_pv_only_chain_defaults_to_the_inputs_resolution():
    pv_dc, load = _inputs("15min", periods=96)
    config = BatteryConfig(nominal_energy_wh=0.0, inverter_ac_capacity_w=3000.0)
    aligned = align_simulation_inputs(pv_dc, load, freq="15min")

    cached = simulate_energy_balance_summary(battery_config=config, aligned=aligned.with_pv_only_chain(config))
    plain = simulate_energy_balance_summary(battery_config=config, aligned=aligned)

    assert cached.column_sums == plain.column_sums


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("inverter_ac_capacity_w", [None, 10000.0, 2500.0])
@pytest.mark.parametrize("battery_wh", [0.0, 3000.0])
def test_pv_production_is_ac_to_load_and_export_plus_dc_to_battery(backend, inverter_ac_capacity_w, battery_wh):
    # Without an inverter rating, DC sent to the battery used to be counted at
    # the inverter efficiency; with one, at its DC value.
    pv_dc, load = _inputs()
    config = BatteryConfig(nominal_energy_wh=battery_wh, inverter_ac_capacity_w=inverter_ac_capacity_w)

    results, total_pv_wh, *_ = simulate_energy_balance(
        pv_dc=pv_dc, houseload=load, battery_config=config, freq="h", execution_backend=backend
    )
    delivered = results["PV_AC_To_Load"] + results["PV_AC_Export"] + results["PV_DC_To_Battery"]

    np.testing.assert_allclose(results["PV_Production"], delivered, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        results["PV_Production"],
        results["PV_DC"] - results["PV_DC_Curtailed"] - results["PV_Direct_Inverter_Loss"],
        rtol=1e-12,
        atol=1e-9,
    )
    assert total_pv_wh == pytest.approx(results["PV_Production"].sum())
    if battery_wh:
        assert results["PV_DC_To_Battery"].sum() > 0
