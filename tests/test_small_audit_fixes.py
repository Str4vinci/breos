"""Small fixes from the 0.7 readiness audit (#175)."""

import numpy as np
import pandas as pd
import pytest

from breos.battery import BatteryConfig, simulate_energy_balance
from breos.montecarlo import _summarize


def test_cli_reports_a_missing_optional_extra_without_a_traceback(monkeypatch, capsys):
    # NumbaUnavailableError is an ImportError, which main() did not catch (#175).
    import breos.cli as cli

    def missing_numba(_args):
        raise ImportError('execution_backend="numba" requires the optional Numba dependency')

    monkeypatch.setattr(cli, "_run", missing_numba)

    assert cli.main(["run", "--config", "unused.toml"]) == 1
    assert "breos: error: execution_backend" in capsys.readouterr().err


def _resistance_run(**kwargs):
    index = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    config = BatteryConfig(nominal_energy_wh=5000.0, enable_resistance_fade=True, initial_resistance_growth=0.3)
    results, *_ = simulate_energy_balance(
        pv_dc=pd.Series(3000.0 * (index.hour.isin(range(9, 16))), index=index),
        houseload=pd.DataFrame({"Load": 600.0}, index=index),
        battery_config=config,
        freq="h",
        **kwargs,
    )
    return results


def test_a_carried_zero_resistance_growth_is_used():
    # A replaced pack carries 0.0 into the next year; it used to be read as
    # "not supplied" and replaced by the configured 0.3 (#175).
    fresh = _resistance_run(initial_resistance_growth=0.0)
    configured = _resistance_run()
    aged = _resistance_run(initial_resistance_growth=0.3)

    pd.testing.assert_frame_equal(configured, aged)
    assert fresh["Battery_Charge_Loss"].sum() < configured["Battery_Charge_Loss"].sum()


def test_montecarlo_summary_leaves_out_non_finite_values():
    # An infinite LCOE made the mean infinite and the spread NaN (#175).
    runs = pd.DataFrame({"lcoe_eur_kwh": [0.10, 0.20, np.inf, np.nan]})

    entry = _summarize(runs)["lcoe_eur_kwh"]

    assert entry["count"] == 2
    assert entry["n_runs"] == 4
    assert entry["mean"] == pytest.approx(0.15)
    assert np.isfinite(entry["std"])
