"""Tests for the App result serialization helpers."""

import io

import pandas as pd
import pytest

from breos.app_results import monthly_to_dicts

_LEDGER_COLUMNS = [
    "PV_DC",
    "PV_Production",
    "PV_AC_To_Load",
    "PV_Origin_Battery_AC_To_Load",
    "Houseload",
    "Import_From_Grid",
    "PV_AC_Export",
    "PV_DC_Curtailed",
]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _constant_year(tz: str) -> pd.DataFrame:
    """One hourly 2023 year with a balanced AC energy flow and a ``Datetime`` column."""
    index = pd.date_range("2023-01-01", periods=8760, freq="h", tz=tz)
    values = {
        "PV_DC": 1200.0,
        "PV_Production": 1100.0,
        "PV_AC_To_Load": 400.0,
        "PV_Origin_Battery_AC_To_Load": 100.0,
        "Houseload": 1500.0,
        "Import_From_Grid": 1000.0,
        "PV_AC_Export": 500.0,
        "PV_DC_Curtailed": 0.0,
    }
    return pd.DataFrame({column: values[column] for column in _LEDGER_COLUMNS}, index=index).reset_index(
        names="Datetime"
    )


@pytest.mark.parametrize("tz", ["Europe/Berlin", "Australia/Sydney", "Etc/GMT-1"])
def test_monthly_rows_group_on_the_local_calendar_east_of_utc(tz):
    # East of UTC, 1 January 00:00 local is still 31 December in UTC. Grouping
    # in UTC made a one-hour December stub the first row and a 13th row.
    rows = monthly_to_dicts(_constant_year(tz), "h")

    assert [row["month"] for row in rows] == _MONTHS
    assert rows[0]["usable_ac_system_production_kwh"] == 744.0
    assert rows[-1]["usable_ac_system_production_kwh"] == 744.0
    assert sum(row["usable_ac_system_production_kwh"] for row in rows) == pytest.approx(8760.0)
    assert all("pv_kwh" not in row for row in rows)
    assert all(row["grid_import_kwh"] == row["consumption_kwh"] - row["self_consumption_kwh"] for row in rows)
    assert all(
        row["usable_ac_system_production_kwh"]
        == row["direct_pv_ac_load_kwh"] + row["pv_origin_battery_ac_load_kwh"] + row["grid_export_kwh"]
        for row in rows
    )


def test_monthly_rows_from_a_mixed_offset_csv_keep_civil_months():
    # A CSV of an IANA-zone run has +01:00 and +02:00 rows. Each row keeps its
    # own wall-clock time, so the DST months have 743 and 745 hours.
    buffer = io.StringIO()
    _constant_year("Europe/Berlin").to_csv(buffer, index=False)
    buffer.seek(0)
    rows = monthly_to_dicts(pd.read_csv(buffer), "h")

    hours = {row["month"]: row["usable_ac_system_production_kwh"] for row in rows}
    assert list(hours) == _MONTHS
    assert hours["Jan"] == 744.0
    assert hours["Mar"] == 743.0
    assert hours["Oct"] == 745.0


def _constant_irradiance(year: pd.DataFrame) -> pd.DataFrame:
    """500 W/m2 GHI and 2000 W of STC power at the plane-of-array irradiance, every step."""
    return pd.DataFrame({"GHI": 500.0, "POA_STC_DC": 2000.0}, index=year.index)


def test_monthly_rows_carry_irradiance_and_performance_ratio():
    year = _constant_year("UTC")
    rows = monthly_to_dicts(year, "h", irradiance=_constant_irradiance(year), system_kwp=4.0)

    january = rows[0]
    assert january["global_horizontal_kwh_m2"] == pytest.approx(744 * 0.5)
    # 2000 W of STC power on a 4 kWp system is 500 W/m2 in the plane.
    assert january["global_incident_kwh_m2"] == pytest.approx(744 * 0.5)
    # 1000 W of usable AC (direct + battery + export) over 2000 W of STC power.
    assert january["performance_ratio_pct"] == pytest.approx(50.0)


def test_monthly_rows_without_irradiance_have_no_irradiance_keys():
    rows = monthly_to_dicts(_constant_year("UTC"), "h")

    assert not {"global_horizontal_kwh_m2", "global_incident_kwh_m2", "performance_ratio_pct"} & set(rows[0])


def test_monthly_rows_of_a_window_that_starts_on_a_midnight_dst_change():
    # Chile moves its clocks forward at local midnight: 3 September 2023 has no
    # 00:00. A [period] window from that day starts at 01:00, and grouping by
    # month normalised that to the missing midnight and raised.
    index = pd.date_range("2023-09-03 01:00", "2023-11-30 23:00", freq="h", tz="America/Santiago")
    frame = _constant_year("UTC").iloc[: len(index)].drop(columns="Datetime").set_index(index)

    rows = monthly_to_dicts(frame, "h", timezone="America/Santiago")

    assert [row["month"] for row in rows] == ["Sep", "Oct", "Nov"]
    assert sum(row["usable_ac_system_production_kwh"] for row in rows) == pytest.approx(len(index) * 1.0)
