"""The per-step ledger is laid out once: one name per column, one row per name."""

import numpy as np
import pandas as pd
import pytest

from breos import _dispatch
from breos.battery import (
    _FRAME_COLUMNS,
    LEDGER_SCHEMA_VERSION,
    BatteryConfig,
    _PvOnlySummaryBuffers,
    _ResultBuffers,
    simulate_energy_balance,
    simulate_energy_balance_summary,
)


def test_every_matrix_row_has_exactly_one_row_constant():
    constants = {name: row for name, row in vars(_dispatch).items() if name[:2] in ("R_", "L_") and name.isupper()}
    assert sorted(constants.values()) == list(range(_dispatch._N_ROWS))
    ledger_rows = {row for name, row in constants.items() if name.startswith("L_")}
    assert ledger_rows == {_dispatch._ROW[name] for name in _dispatch._LEDGER_COLUMNS}


@pytest.mark.parametrize("buffers", [_ResultBuffers, _PvOnlySummaryBuffers])
def test_buffers_expose_the_frame_columns_read_only(buffers):
    out = buffers(4)
    assert tuple(out.columns) == _FRAME_COLUMNS
    with pytest.raises(KeyError):
        out.columns["Battery_Enregy"]
    with pytest.raises(TypeError):
        out.columns["Battery_Enregy"] = np.zeros(4)


def test_full_buffer_columns_are_views_of_the_kernel_matrix():
    out = _ResultBuffers(3)
    out.zero_fill()
    for name in _dispatch._ROW_COLUMNS:
        out.matrix[_dispatch._ROW[name], 1] = 7.0
        assert out.columns[name][1] == 7.0, name
        out.matrix[_dispatch._ROW[name], 1] = 0.0
    assert out.columns["Battery_Energy_End"] is out.columns["Battery_Energy"]


def _inputs():
    index = pd.date_range("2025-06-01", periods=48, freq="h", tz="UTC")
    pv = pd.Series(np.where((index.hour >= 9) & (index.hour < 17), 3000.0, 0.0), index=index)
    return pv, pd.DataFrame({"Load": 600.0}, index=index)


@pytest.mark.parametrize("capacity_wh", [0.0, 5000.0])
def test_frame_reports_the_layout_columns_in_order(capacity_wh):
    pv, load = _inputs()
    results_df, *_ = simulate_energy_balance(
        pv_dc=pv, houseload=load, battery_config=BatteryConfig(nominal_energy_wh=capacity_wh), freq="h"
    )
    assert tuple(results_df.columns) == ("Datetime", *_FRAME_COLUMNS)


def test_summary_reports_the_ledger_schema_version():
    pv, load = _inputs()
    summary = simulate_energy_balance_summary(
        pv_dc=pv, houseload=load, battery_config=BatteryConfig(nominal_energy_wh=5000.0), freq="h"
    )
    assert summary.ledger_schema_version == LEDGER_SCHEMA_VERSION == "3.0"
