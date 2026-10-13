"""AC-coupled batteries (ADR 0004): configuration, ledger columns and the run-time refusal."""

import dataclasses
import importlib.util

import numpy as np
import pandas as pd
import pytest

import breos.battery as battery_module
from breos._dispatch import _LEDGER_COLUMNS, LEDGER_SCHEMA_VERSION
from breos.battery import BatteryConfig, simulate_energy_balance, simulate_energy_balance_summary
from tests.energy_conservation import assert_energy_conservation

_BACKENDS = [
    "python",
    pytest.param(
        "numba", marks=pytest.mark.skipif(not importlib.util.find_spec("numba"), reason="numba not installed")
    ),
]
AC_COLUMNS = ("PV_AC_To_Battery", "PV_Charge_Conversion_Loss")


def _day():
    index = pd.date_range("2025-06-01", periods=24, freq="h", tz="UTC")
    hours = np.arange(24)
    pv = pd.Series(np.clip(np.sin((hours - 6) / 12 * np.pi), 0.0, None) * 4000.0, index=index)
    load = pd.DataFrame({"Load": 600.0 + 400.0 * (hours >= 18)}, index=index)
    return pv, load, pd.Series(25.0, index=index)


def _simulate(config, backend="python"):
    pv, load, temperature = _day()
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=config,
        freq="h",
        temperature_series=temperature,
        execution_backend=backend,
    )[0]


def _ac(**overrides):
    return BatteryConfig(
        **{
            "nominal_energy_wh": 10000.0,
            "coupling": "ac",
            "battery_inverter_efficiency": 0.96,
            **overrides,
        }
    )


def test_coupling_defaults_to_dc_without_a_battery_inverter():
    config = BatteryConfig(nominal_energy_wh=10000.0)
    assert config.coupling == "dc"
    assert config.battery_inverter_ac_capacity_w is None
    assert config.battery_inverter_efficiency is None


def test_ac_coupling_accepts_and_coerces_its_inverter():
    config = _ac(battery_inverter_ac_capacity_w=3000, battery_inverter_efficiency=np.float32(0.5))
    assert config.battery_inverter_ac_capacity_w == 3000.0
    assert isinstance(config.battery_inverter_ac_capacity_w, float)
    assert config.battery_inverter_efficiency == 0.5
    assert _ac().battery_inverter_ac_capacity_w is None


@pytest.mark.parametrize("coupling", ["DC", "hybrid", "", None, True])
def test_unknown_coupling_is_refused(coupling):
    with pytest.raises(ValueError, match="coupling must be one of: dc, ac"):
        BatteryConfig(nominal_energy_wh=10000.0, coupling=coupling)


@pytest.mark.parametrize("field", [{"battery_inverter_ac_capacity_w": 3000.0}, {"battery_inverter_efficiency": 0.96}])
def test_dc_coupling_refuses_the_battery_inverter(field):
    with pytest.raises(ValueError, match="needs coupling = 'ac'"):
        BatteryConfig(nominal_energy_wh=10000.0, **field)


def test_ac_coupling_requires_the_battery_inverter_efficiency():
    with pytest.raises(ValueError, match="needs battery_inverter_efficiency; it has no default"):
        BatteryConfig(nominal_energy_wh=10000.0, coupling="ac")


@pytest.mark.parametrize("value", [0.0, -0.1, 1.01, float("nan"), float("inf"), True, np.bool_(True), "high"])
def test_battery_inverter_efficiency_must_be_a_fraction(value):
    with pytest.raises(ValueError, match="battery_inverter_efficiency"):
        _ac(battery_inverter_efficiency=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, np.bool_(False), "big"])
def test_battery_inverter_rating_must_be_finite_and_non_negative(value):
    with pytest.raises(ValueError, match="battery_inverter_ac_capacity_w must be a finite non-negative number"):
        _ac(battery_inverter_ac_capacity_w=value)


def test_ac_config_survives_the_controller_round_trip():
    # A day controller rebuilds the config from its dataclass fields.
    config = _ac(battery_inverter_ac_capacity_w=3000.0)
    assert BatteryConfig(**dataclasses.asdict(config)) == config


def test_ledger_adds_the_pv_ac_charge_columns_after_the_grid_charge_columns():
    position = _LEDGER_COLUMNS.index("Grid_Charge_Conversion_Loss")
    assert _LEDGER_COLUMNS[position + 1 : position + 3] == AC_COLUMNS
    assert LEDGER_SCHEMA_VERSION == "3.1"


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ac_coupled_battery_is_refused_before_any_step(backend):
    with pytest.raises(NotImplementedError, match="AC-coupled battery dispatch is not implemented yet"):
        _simulate(_ac(battery_inverter_ac_capacity_w=3000.0), backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dc_coupled_battery_writes_the_ac_columns_as_zero(backend):
    config = BatteryConfig(nominal_energy_wh=10000.0, inverter_ac_capacity_w=3000.0)
    frame = _simulate(config, backend)
    assert frame["PV_DC_To_Battery"].sum() > 0.0
    for column in AC_COLUMNS:
        assert (frame[column] == 0.0).all()
    assert_energy_conservation(frame, config)


def test_ac_coupling_without_a_battery_is_the_pv_only_run():
    # ADR 0004 P2 at the BatteryConfig level: no battery, no battery inverter.
    dc = _simulate(BatteryConfig(nominal_energy_wh=0.0, inverter_ac_capacity_w=3000.0))
    ac = _simulate(_ac(nominal_energy_wh=0.0, inverter_ac_capacity_w=3000.0, battery_inverter_ac_capacity_w=2000.0))
    pd.testing.assert_frame_equal(dc, ac, check_exact=True)
    for column in AC_COLUMNS:
        assert (ac[column] == 0.0).all()


def test_ac_coupling_accepts_the_boundaries():
    config = _ac(battery_inverter_ac_capacity_w=0.0, battery_inverter_efficiency=1.0)
    assert config.battery_inverter_ac_capacity_w == 0.0
    assert config.battery_inverter_efficiency == 1.0


def test_pv_only_summary_and_retired_pack_report_the_ac_columns_as_zero():
    pv, load, temperature = _day()
    kwargs = {"pv_dc": pv, "houseload": load, "freq": "h", "temperature_series": temperature}
    summary = simulate_energy_balance_summary(battery_config=BatteryConfig(nominal_energy_wh=0.0), **kwargs)
    retired = simulate_energy_balance(
        battery_config=BatteryConfig(nominal_energy_wh=10000.0), battery_retired=True, initial_energy_wh=0.0, **kwargs
    )[0]
    for column in AC_COLUMNS:
        assert summary.column_sums[column] == 0.0
        assert (retired[column] == 0.0).all()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_every_buffer_row_is_written_whatever_the_buffer_held(monkeypatch, backend):
    # The matrix comes from np.empty: poison it so an unwritten row shows.
    original = battery_module._ResultBuffers.__init__

    def poisoned(self, n_steps):
        original(self, n_steps)
        self.matrix.fill(np.nan)

    monkeypatch.setattr(battery_module._ResultBuffers, "__init__", poisoned)
    config = BatteryConfig(nominal_energy_wh=10000.0, inverter_ac_capacity_w=3000.0)
    frame = _simulate(config, backend)
    numeric = frame.drop(columns=["Datetime"]).to_numpy(dtype=float)
    assert np.isfinite(numeric).all()
    for column in AC_COLUMNS:
        assert (frame[column] == 0.0).all()
