"""power_limit_c_rate limits the stored energy in both directions (#155)."""

import numpy as np
import pandas as pd
import pytest

from breos.battery import BatteryConfig, simulate_energy_balance

BACKENDS = ["python", "numba"]


def _two_days():
    index = pd.date_range("2025-06-01", periods=192, freq="15min", tz="UTC")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    pv = pd.Series(np.where((hour >= 10) & (hour < 15), 12_000.0, 0.0), index=index)
    load = pd.DataFrame({"Load": np.where(hour >= 18, 9_000.0, 300.0)}, index=index)
    return pv, load


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_c_stores_and_releases_the_same_power(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    pv, load = _two_days()
    config = BatteryConfig(
        nominal_energy_wh=5_000.0,
        power_limit_c_rate=1.0,
        inverter_ac_capacity_w=50_000.0,
        min_soc=0.0,
        max_soc=1.0,
    )

    results = simulate_energy_balance(
        pv_dc=pv, houseload=load, battery_config=config, freq="15min", execution_backend=backend
    )[0]

    stored = results["Battery_Charge_Stored"].to_numpy()
    released = results["Battery_Discharge_DC"].to_numpy()
    # On develop charging stopped at 4,873 W stored (5 kW of DC input) and
    # discharge drew 5,553 W (5 kW of AC output): 0.97 C against 1.11 C.
    assert stored.max() == pytest.approx(5_000.0, rel=1e-9)
    assert released.max() == pytest.approx(5_000.0, rel=1e-9)
    assert results["Battery_Charge_Input"].max() > 5_000.0
    assert results["Battery_AC_To_Load"].max() < 5_000.0
