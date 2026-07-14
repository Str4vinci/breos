"""Parity tests for the optional Numba kernels (breos[fast])."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")


def test_numba_energy_balance_capability_contract_is_explicit():
    from breos.numba_kernels import ENERGY_BALANCE_CAPABILITIES

    assert ENERGY_BALANCE_CAPABILITIES == {
        "status": "approximate_screening_only",
        "production_caller": False,
        "inverter_clipping": False,
        "dc_coupling_ledger": False,
        "battery_power_limits": False,
        "replacement_events": False,
    }


def test_lfp_capacity_factor_parity_with_python_reference():
    # The kernel hand-duplicates the LFP derate constants from constants.py;
    # pin them against the reference implementation across both derate
    # branches, including sub-zero temperatures and the 0.5 floor.
    from breos.battery import lfp_capacity_factor
    from breos.numba_kernels import _lfp_capacity_factor_numba

    for temp in np.linspace(-60.0, 50.0, 221):
        assert _lfp_capacity_factor_numba(float(temp)) == pytest.approx(lfp_capacity_factor(float(temp)), abs=1e-12)


def test_energy_balance_kernel_matches_reference_where_models_coincide():
    # The kernel omits inverter conversion; with unity inverter efficiency,
    # 25 degC, and a single day (no degradation step applied to stored
    # results), it must reproduce the reference energy balance exactly.
    from breos.battery import BatteryConfig, simulate_energy_balance
    from breos.numba_kernels import energy_balance_kernel

    idx = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
    pv = np.array([0.0] * 7 + [500.0, 1500.0, 2500.0, 3000.0, 2800.0, 2000.0, 1200.0, 400.0] + [0.0] * 9)
    load = np.array([600.0] * 6 + [900.0] * 3 + [700.0] * 8 + [1100.0] * 5 + [800.0] * 2)

    config = BatteryConfig(
        nominal_energy_wh=5000,
        inverter_efficiency=1.0,
        standby_loss_wh=5.0,
        enable_replacement=False,
    )
    results_df, *_ = simulate_energy_balance(
        pv_dc=pd.Series(pv, index=idx),
        houseload=pd.DataFrame({"Load": load}, index=idx),
        battery_config=config,
        freq="h",
        temperature_series=pd.Series(25.0, index=idx),
    )

    import_wh, sell_wh, battery_wh, _soc_norm, _soc_abs = energy_balance_kernel(
        pv,
        load,
        5000.0,
        config.max_soc,
        config.min_soc,
        config.charge_efficiency,
        config.discharge_efficiency,
        5.0,
        1.0,
        1.0,
    )

    np.testing.assert_allclose(results_df["Import_From_Grid"].to_numpy(), import_wh, atol=1e-9)
    np.testing.assert_allclose(results_df["Sell_To_Grid"].to_numpy(), sell_wh, atol=1e-9)
    np.testing.assert_allclose(results_df["Battery_Energy"].to_numpy(), battery_wh, atol=1e-9)
