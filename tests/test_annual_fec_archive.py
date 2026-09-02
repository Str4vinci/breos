"""Exact annual full-equivalent cycles, and the annual-archive column join.

Two defects in the Batch B2 lattice archive are covered here. Cumulative FEC
restarts at zero when a pack is replaced, so it cannot answer how many cycles
a projected year actually accumulated; and concatenating the yearly and
financial ledgers emitted two ``PV_Production_kWh`` and two ``Export_kWh``
columns, which makes a reader's column selection depend on its CSV parser.
"""

import numpy as np
import pandas as pd
import pytest

from breos.battery import BatteryConfig, simulate_energy_balance
from breos.optimization import ProjectedDesignResult, _evaluate_projected_design_metrics
from breos.pv_modules import get_module
from tools.revision.grid_eval import _annual_archive_frame, _prepare_archive_directory, _reject_duplicate_columns

# The itemised, undiscounted annual cash flows a Task 7 re-ranking needs, plus
# the energy, battery and replacement series the schema gate requires.
REQUIRED_ARCHIVE_COLUMNS = (
    "Design_ID",
    "Year",
    "PV_Production_kWh",
    "Load_kWh",
    "Import_kWh",
    "Export_kWh",
    "Grid_Independence_%",
    "Battery_SOH_%",
    "Battery_Charge_Throughput_kWh",
    "Battery_Discharge_Throughput_kWh",
    "Battery_SOC_Normalized_Mean_%",
    "Battery_SOC_Absolute_Mean_%",
    "Battery_Annual_FEC",
    "Battery_Cumulative_FEC",
    "Replacements",
    "Replacement_Cost",
    "Cost_Import",
    "Revenue_Export",
    "Cost_Operation",
    "Cost_Daily",
    "Cost_Replacement",
    "Cost_System_Annual",
    "Cost_No_Sys_Annual",
)


def _cycling_day(pv_w: float = 3000.0, load_w: float = 900.0):
    return [0.0] * 8 + [pv_w] * 8 + [0.0] * 8, [load_w] * 8 + [0.0] * 8 + [load_w] * 8


def _cycling_span(days: int):
    idx = pd.date_range("2025-01-01 00:00", periods=days * 24, freq="h", tz="UTC")
    pv_day, load_day = _cycling_day()
    return (
        idx,
        pd.Series(pv_day * days, index=idx),
        pd.DataFrame({"Load": load_day * days}, index=idx),
        pd.Series(25.0, index=idx),
    )


def _run_span(days: int, config: BatteryConfig, backend: str = "python"):
    _idx, pv_dc, houseload, temperature = _cycling_span(days)
    *_, degradation_df, _state = simulate_energy_balance(
        pv_dc=pv_dc,
        houseload=houseload,
        battery_config=config,
        freq="h",
        temperature_series=temperature,
        return_degradation_state=True,
        execution_backend=backend,
    )
    return degradation_df


def _near_eol_config(**overrides) -> BatteryConfig:
    """A pack a hair above end of life, so day 1 retires it."""
    settings = dict(
        nominal_energy_wh=5000.0,
        standby_loss_wh=0.0,
        initial_soh=70.000001,
        eol_percentage=0.70,
        enable_replacement=True,
    )
    settings.update(overrides)
    return BatteryConfig(**settings)


def test_lifetime_fec_keeps_the_retired_pack_when_cumulative_fec_resets():
    degradation = _run_span(4, _near_eol_config())

    # Day 1 retires the pack: the installed-pack counter is back at zero and
    # SOH is back at 100, but the day's own cycles are already banked.
    assert degradation["Cumulative_FEC"].iloc[0] == pytest.approx(0.0)
    assert degradation["SOH"].iloc[0] == pytest.approx(100.0)
    retired_pack_fec = float(degradation["Cumulative_FEC_All_Packs"].iloc[0])
    assert retired_pack_fec > 0.0

    all_packs = degradation["Cumulative_FEC_All_Packs"].to_numpy(dtype=float)
    assert (np.diff(all_packs) >= 0.0).all()
    # Exactly the two packs' contributions, with nothing lost at the seam.
    assert all_packs[-1] == pytest.approx(retired_pack_fec + float(degradation["Cumulative_FEC"].iloc[-1]), abs=1e-12)
    assert all_packs[-1] > float(degradation["Cumulative_FEC"].iloc[-1])


def test_lifetime_fec_equals_cumulative_fec_when_no_pack_is_replaced():
    degradation = _run_span(4, BatteryConfig(nominal_energy_wh=5000.0, standby_loss_wh=0.0, enable_replacement=False))

    assert float(degradation["Cumulative_FEC"].iloc[-1]) > 0.0
    np.testing.assert_allclose(
        degradation["Cumulative_FEC_All_Packs"].to_numpy(dtype=float),
        degradation["Cumulative_FEC"].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_lifetime_fec_is_backend_independent():
    python_run = _run_span(4, _near_eol_config(), backend="python")
    numba_run = _run_span(4, _near_eol_config(), backend="numba")

    np.testing.assert_array_equal(
        python_run["Cumulative_FEC_All_Packs"].to_numpy(dtype=float),
        numba_run["Cumulative_FEC_All_Packs"].to_numpy(dtype=float),
    )


def _projected_result(*, battery_kwh: float, batt_spec: dict, years: int = 2, days: int = 30):
    idx, base_dc, houseload, temperature = _cycling_span(days)
    metrics = _evaluate_projected_design_metrics(
        base_dc_power=base_dc,
        tmy_data=pd.DataFrame({"temp_air": 20.0, "ghi": 500.0}, index=idx),
        houseload=houseload,
        temperature_series=temperature,
        pv_params=get_module("Suntech_STP550S_STC"),
        batt_spec=batt_spec,
        costs_cfg={"storage_cost_per_kwh": 500.0},
        fin_cfg={"project_lifespan": years},
        freq="h",
        years_projection=years,
        degradation_rate=0.005,
        n_modules=6,
        battery_kwh=battery_kwh,
        inverter_efficiency=0.96,
        inverter_ac_capacity_w=2640.0,
        return_tables=True,
    )
    return ProjectedDesignResult(
        metrics=metrics,
        yearly=metrics.pop("_yearly_summary_df"),
        financial=metrics.pop("_cost_projection_df"),
    )


def test_archive_frame_has_unique_headers_and_the_required_schema():
    result = _projected_result(
        battery_kwh=5.0,
        batt_spec={"min_soc": 0.1, "max_soc": 0.9, "eol_percentage": 0.70, "enable_replacement": True},
    )

    frame = _annual_archive_frame(17, result)

    assert frame.columns.duplicated().sum() == 0
    _reject_duplicate_columns(frame)
    missing = [column for column in REQUIRED_ARCHIVE_COLUMNS if column not in frame.columns]
    assert missing == []
    assert frame["Design_ID"].unique().tolist() == [17]
    assert frame["Year"].tolist() == [1, 2]

    # One canonical copy, carrying the yearly ledger's values, and every
    # financial column the projection reported is still present.
    np.testing.assert_array_equal(frame["PV_Production_kWh"].to_numpy(), result.yearly["PV_Production_kWh"].to_numpy())
    np.testing.assert_array_equal(frame["Export_kWh"].to_numpy(), result.yearly["Export_kWh"].to_numpy())
    for column in result.financial.columns:
        assert column in frame.columns


def test_archive_frame_preserves_every_column_of_both_ledgers():
    result = _projected_result(
        battery_kwh=0.0,
        batt_spec={"min_soc": 0.1, "max_soc": 0.9, "enable_replacement": True},
    )

    frame = _annual_archive_frame(0, result)

    expected = (
        ["Design_ID"]
        + list(result.yearly.columns)
        + [column for column in result.financial.columns if column not in result.yearly.columns]
    )
    assert frame.columns.tolist() == expected
    assert frame["Battery_Annual_FEC"].tolist() == [0.0, 0.0]


def test_archive_frame_rejects_a_disagreement_between_the_two_ledgers():
    result = _projected_result(
        battery_kwh=5.0,
        batt_spec={"min_soc": 0.1, "max_soc": 0.9, "eol_percentage": 0.70, "enable_replacement": True},
    )
    result.financial.loc[0, "Export_kWh"] = result.financial.loc[0, "Export_kWh"] + 1.0

    with pytest.raises(ValueError, match="disagree on Export_kWh"):
        _annual_archive_frame(3, result)


def test_duplicate_columns_are_rejected_before_a_shard_is_written():
    frame = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["Year", "Export_kWh", "Export_kWh"])

    with pytest.raises(ValueError, match=r"duplicate column names: \['Export_kWh'\]"):
        _reject_duplicate_columns(frame)


def test_archive_directory_must_be_empty(tmp_path):
    archive = tmp_path / "archive"
    _prepare_archive_directory(archive)
    (archive / "annual_00000.csv").write_text("Design_ID,Year\n0,1\n")

    with pytest.raises(FileExistsError, match="refusing to reuse nonempty archive directory"):
        _prepare_archive_directory(archive)
