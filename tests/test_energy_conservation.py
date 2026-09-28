"""Per-step energy conservation across backends, engines and a full year (#184)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from breos.battery import BatteryConfig, simulate_energy_balance
from tests.energy_conservation import LEDGER_COLUMNS, assert_energy_conservation, assert_origin_reconciliation

_BACKENDS = [
    "python",
    pytest.param(
        "numba", marks=pytest.mark.skipif(not importlib.util.find_spec("numba"), reason="numba not installed")
    ),
]


def _year(freq="h"):
    """One year of PV, load and a cold-winter battery temperature."""
    steps_per_hour = 4 if freq == "15min" else 1
    index = pd.date_range("2025-01-01", periods=8760 * steps_per_hour, freq=freq, tz="UTC")
    hours = np.arange(len(index)) / steps_per_hour
    hour_of_day, day = hours % 24, hours / 24
    season = 0.6 + 0.4 * np.sin((day - 80) / 365 * 2 * np.pi)
    pv = pd.Series(6000.0 * season * np.clip(np.sin((hour_of_day - 6) / 12 * np.pi), 0, 1), index=index)
    load = pd.DataFrame({"Load": 350.0 + 900.0 * ((hour_of_day >= 18) & (hour_of_day < 22))}, index=index)
    # Below freezing in winter, so the capacity derating is exercised.
    temperature = pd.Series(8.0 + 14.0 * np.sin((day - 110) / 365 * 2 * np.pi), index=index)
    return pv, load, temperature


@pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("engine", "config_kwargs"),
    [
        ("native", {"enable_resistance_fade": True}),
        ("native", {"inverter_ac_capacity_w": 3500.0, "max_charge_power_w": 1500.0}),
        ("blast", {"inverter_ac_capacity_w": 5000.0}),
    ],
)
def test_full_year_conserves_energy_at_every_step(backend, engine, config_kwargs):
    pv, load, temperature = _year()
    # A raised end of life puts pack replacements inside the year.
    config = BatteryConfig(nominal_energy_wh=5000.0, eol_percentage=0.985, standby_loss_wh=2.0, **config_kwargs)
    results, *_, n_replacements, _ = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=config,
        freq="h",
        temperature_series=temperature,
        degradation_engine=engine,
        blast_model="nmc_gr_50ah_b1" if engine == "blast" else None,
        execution_backend=backend,
    )

    assert n_replacements >= 1
    # Cold-weather derating shrinks the usable window and spills stored energy.
    assert results["Capacity_Window_Loss"].sum() > 0.0
    assert_energy_conservation(results, config, atol=1e-7)
    assert_origin_reconciliation(results, 1.0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_15min_year_conserves_energy_at_every_step(backend):
    pv, load, temperature = _year("15min")
    config = BatteryConfig(nominal_energy_wh=5000.0, eol_percentage=0.985, inverter_ac_capacity_w=4000.0)
    results, *_ = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=config,
        freq="15min",
        temperature_series=temperature,
        execution_backend=backend,
    )

    assert_energy_conservation(results, config, atol=1e-7)
    assert_origin_reconciliation(results, 0.25)


def _two_days():
    pv, load, temperature = _year()
    config = BatteryConfig(nominal_energy_wh=5000.0, inverter_ac_capacity_w=4000.0)
    results, *_ = simulate_energy_balance(
        pv_dc=pv.iloc[:48],
        houseload=load.iloc[:48],
        battery_config=config,
        freq="h",
        temperature_series=temperature.iloc[:48],
    )
    return results, config


def test_checker_counts_grid_charging_as_an_input():
    results, config = _two_days()
    # Grid charging that is imported and stored balances; one that is only
    # imported does not.
    drawn = 100.0 / config.charge_efficiency  # grid energy that stores 100 Wh
    charged = results.copy()
    charged["Grid_AC_To_Battery"] = drawn
    charged["Grid_DC_To_Battery"] = drawn
    charged["Grid_Charge_Conversion_Loss"] = 0.0
    charged["Import_From_Grid"] += drawn
    charged["Battery_Charge_Input"] += drawn
    charged["Battery_Charge_Loss"] += drawn - 100.0
    charged["Battery_Charge_Stored"] += 100.0
    charged["Battery_Energy_Delta"] += 100.0
    assert_energy_conservation(charged, config)

    leaked = results.copy()
    leaked["Import_From_Grid"] += 100.0
    with pytest.raises(AssertionError, match="load supply"):
        assert_energy_conservation(leaked, config)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_checker_rejects_non_finite_ledger_rows(value):
    results, config = _two_days()
    assert_energy_conservation(results, config)
    # A row that is non-finite in every column satisfies each identity
    # under assert_allclose's equal_nan, so it must be caught up front.
    corrupted = results.copy()
    corrupted.loc[corrupted.index[12], list(LEDGER_COLUMNS)] = value
    with pytest.raises(AssertionError, match="non-finite ledger values"):
        assert_energy_conservation(corrupted, config)

    one_cell = results.copy()
    one_cell.iloc[12, one_cell.columns.get_loc("Battery_Charge_Loss")] = value
    with pytest.raises(AssertionError, match="Battery_Charge_Loss"):
        assert_energy_conservation(one_cell, config)


def test_checker_applies_atol_as_an_absolute_bound():
    results, config = _two_days()
    step = results["PV_DC"].idxmax()
    # Above 1 kW, NumPy's default rtol of 1e-7 alone would allow the 1e-4 W
    # discrepancy below.
    assert results.loc[step, "PV_DC"] > 1000.0
    off = results.copy()
    off.loc[step, "PV_DC_Curtailed"] += 1e-4
    with pytest.raises(AssertionError, match="PV DC split"):
        assert_energy_conservation(off, config, atol=1e-7)
    assert_energy_conservation(off, config, atol=1e-3)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_carried_grid_origin_is_drawn_down_in_proportion(backend):
    """Grid-origin energy carried in from an earlier span leaves in proportion to its share."""
    pv, load, temperature = _year()
    config = BatteryConfig(nominal_energy_wh=5000.0, eol_percentage=0.985, standby_loss_wh=2.0)
    results, *_, n_replacements, _ = simulate_energy_balance(
        pv_dc=pv.iloc[: 24 * 60],
        houseload=load.iloc[: 24 * 60],
        battery_config=config,
        freq="h",
        temperature_series=temperature.iloc[: 24 * 60],
        initial_energy_wh=4000.0,
        initial_pv_origin_energy_wh=1000.0,
        initial_grid_origin_energy_wh=2000.0,
        execution_backend=backend,
    )

    assert_energy_conservation(results, config)
    assert_origin_reconciliation(results, 1.0)
    first = results.iloc[0]
    assert first["Battery_Grid_Origin_Energy_Beginning"] == 2000.0
    # The first night discharges; the grid origin supplies half of it.
    night = results.iloc[:6]
    assert night["Battery_Discharge_DC"].sum() > 0.0
    np.testing.assert_allclose(
        night["Grid_Origin_Battery_Discharge_DC"], night["Battery_Discharge_DC"] * 0.5, rtol=1e-6, atol=1e-9
    )
    assert results["Grid_Origin_Standby_Loss"].sum() > 0.0
    assert results["Grid_Origin_Capacity_Window_Loss"].sum() > 0.0
    # Nothing charges the grid origin yet, so it only ever falls.
    assert (results["Battery_Grid_Origin_Energy_End"] <= results["Battery_Grid_Origin_Energy_Beginning"]).all()


def test_replacement_hands_every_origin_out_and_starts_unattributed():
    pv, load, temperature = _year()
    config = BatteryConfig(nominal_energy_wh=5000.0, eol_percentage=0.985)
    results, *_, n_replacements, _ = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=config,
        freq="h",
        temperature_series=temperature,
        initial_energy_wh=4000.0,
        initial_pv_origin_energy_wh=1000.0,
        initial_grid_origin_energy_wh=500.0,
    )

    assert n_replacements >= 1
    swaps = results[results["Battery_Replaced"]]
    assert (swaps["PV_Origin_Replacement_Energy_Removed"] > 0.0).all()
    assert (swaps["Battery_PV_Origin_Energy_End"] == 0.0).all()
    assert (swaps["Battery_Grid_Origin_Energy_End"] == 0.0).all()
    assert_origin_reconciliation(results, 1.0)


def test_origin_checker_catches_an_unbooked_origin_change():
    results, _config = _two_days()
    assert_origin_reconciliation(results, 1.0)
    drifted = results.copy()
    drifted.loc[drifted.index[20], "Battery_PV_Origin_Energy_End"] += 1.0
    with pytest.raises(AssertionError, match="PV-origin balance"):
        assert_origin_reconciliation(drifted, 1.0)


def test_carried_grid_origin_cannot_exceed_the_non_pv_energy():
    pv, load, temperature = _year()
    with pytest.raises(ValueError, match="initial_grid_origin_energy_wh must be between 0"):
        simulate_energy_balance(
            pv_dc=pv.iloc[:24],
            houseload=load.iloc[:24],
            battery_config=BatteryConfig(nominal_energy_wh=5000.0),
            freq="h",
            initial_energy_wh=3000.0,
            initial_pv_origin_energy_wh=2000.0,
            initial_grid_origin_energy_wh=1500.0,
        )
