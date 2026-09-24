"""Tests for the App result serialization helpers."""

import io

import pandas as pd
import pytest

from breos.app_results import monthly_to_dicts

_LEDGER_COLUMNS = [
    "PV_DC",
    "PV_Production",
    "PV_AC_To_Load",
    "Battery_AC_To_Load_PV",
    "Houseload",
    "Import_From_Grid",
    "PV_AC_Export",
    "PV_DC_Curtailed",
]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _constant_year(tz: str) -> pd.DataFrame:
    """One hourly 2023 year at a constant 1 kW, with a ``Datetime`` column."""
    index = pd.date_range("2023-01-01", periods=8760, freq="h", tz=tz)
    return pd.DataFrame({column: 1000.0 for column in _LEDGER_COLUMNS}, index=index).reset_index(names="Datetime")


@pytest.mark.parametrize("tz", ["Europe/Berlin", "Australia/Sydney", "Etc/GMT-1"])
def test_monthly_rows_group_on_the_local_calendar_east_of_utc(tz):
    # East of UTC, 1 January 00:00 local is still 31 December in UTC. Grouping
    # in UTC made a one-hour December stub the first row and a 13th row.
    rows = monthly_to_dicts(_constant_year(tz), "h")

    assert [row["month"] for row in rows] == _MONTHS
    assert rows[0]["pv_kwh"] == 744.0
    assert rows[-1]["pv_kwh"] == 744.0
    assert sum(row["pv_kwh"] for row in rows) == pytest.approx(8760.0)


def test_monthly_rows_from_a_mixed_offset_csv_keep_civil_months():
    # A CSV of an IANA-zone run has +01:00 and +02:00 rows. Each row keeps its
    # own wall-clock time, so the DST months have 743 and 745 hours.
    buffer = io.StringIO()
    _constant_year("Europe/Berlin").to_csv(buffer, index=False)
    buffer.seek(0)
    rows = monthly_to_dicts(pd.read_csv(buffer), "h")

    hours = {row["month"]: row["pv_kwh"] for row in rows}
    assert list(hours) == _MONTHS
    assert hours["Jan"] == 744.0
    assert hours["Mar"] == 743.0
    assert hours["Oct"] == 745.0
