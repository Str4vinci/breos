"""Explicit, reported repair of measured load and PV input (#194)."""

import json

import numpy as np
import pandas as pd
import pytest

import breos
from breos import App
from breos.io import InputRepairReport, repair_series

# 2025-01-06 is a Monday.
_MONDAY = "2025-01-06"


def _load(days=21, freq="h", tz=None, start=_MONDAY):
    """Load whose value encodes the day number and the time of day.

    Day ``d`` (0 = 6 January) at wall-clock hour ``h`` reads
    ``1000 * d + 10 * h`` W, so a filled value says exactly which days fed it.
    """
    end = pd.Timestamp(start) + pd.Timedelta(days=days)
    idx = pd.date_range(start, end, freq=freq, tz=tz, inclusive="left")
    wall = idx.tz_localize(None) if idx.tz is not None else idx
    day = (wall.normalize() - pd.Timestamp(start)).days.to_numpy()
    return pd.Series(1000.0 * day + 10.0 * wall.hour.to_numpy(), index=idx, name="load_w")


def _value(day, hour):
    return 1000.0 * day + 10.0 * hour


# --- gaps -------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["drop", np.nan, np.inf, -np.inf])
def test_gaps_raise_by_default(bad):
    load = _load()
    stamp = load.index[24 * 9 + 5]
    load = load.drop(stamp) if bad == "drop" else load.where(load.index != stamp, bad)

    with pytest.raises(ValueError, match="1 missing or non-finite steps in 1 gap.*gap_fill='nearby_days'"):
        repair_series(load)


def test_weekday_gap_is_the_mean_of_the_nearest_weekdays():
    load = _load()
    # Wednesday 15 January (day 9), 05:00-07:00, three missing timestamps.
    missing = load.index[24 * 9 + 5 : 24 * 9 + 8]
    repaired, report = repair_series(load.drop(missing), gap_fill="nearby_days")

    # Tuesday (day 8) and Thursday (day 10) at the same hours.
    for hour in (5, 6, 7):
        assert repaired[f"2025-01-15 {hour:02d}:00"] == pytest.approx((_value(8, hour) + _value(10, hour)) / 2)
    (event,) = report.events
    assert (event.issue, event.method, event.steps, event.missing_timestamps, event.non_finite_values) == (
        "gap",
        "nearby_days",
        3,
        3,
        0,
    )
    assert event.start == "2025-01-15T05:00:00"
    assert event.end == "2025-01-15T07:00:00"
    assert (event.replaced_min_w, event.replaced_mean_w, event.replaced_max_w) == (9050.0, 9060.0, 9070.0)
    # Every step outside the gap is untouched.
    untouched = repaired.drop(missing)
    pd.testing.assert_series_equal(untouched, load.drop(missing))


def test_weekend_gap_uses_weekend_days_and_monday_uses_the_nearest_weekdays():
    load = _load()
    load["2025-01-18 12:00"] = np.nan  # Saturday, day 12
    load["2025-01-20 12:00"] = np.nan  # Monday, day 14
    repaired, report = repair_series(load, gap_fill="nearby_days")

    # Saturday: Sunday 19th (+1) and Sunday 12th (-6); Friday 17th is nearer but a weekday.
    assert repaired["2025-01-18 12:00"] == (_value(13, 12) + _value(6, 12)) / 2
    # Monday: Tuesday 21st (+1) and Wednesday 22nd (+2); Friday 17th (-3) is further.
    assert repaired["2025-01-20 12:00"] == (_value(15, 12) + _value(16, 12)) / 2
    assert [event.non_finite_values for event in report.events] == [1, 1]


def test_other_day_type_is_used_only_when_no_same_type_day_is_in_the_window():
    load = _load()
    load["2025-01-18 12:00"] = np.nan  # Saturday, day 12; Sunday 19th is also invalid
    load["2025-01-19 12:00"] = np.nan
    repaired, _ = repair_series(load, gap_fill="nearby_days", window_days=2)

    # No valid weekend day within two days of Saturday, so weekdays: Friday 17th
    # (-1), then Thursday 16th (-2) before Monday 20th (+2), the earlier first.
    assert repaired["2025-01-18 12:00"] == (_value(11, 12) + _value(10, 12)) / 2
    # Filled steps are never used as donors: Sunday gets Friday (-2) and Monday (+1).
    assert repaired["2025-01-19 12:00"] == (_value(14, 12) + _value(11, 12)) / 2


def test_ties_prefer_the_earlier_day_and_neighbour_days_sets_the_count():
    load = _load()
    load["2025-01-15 03:00"] = np.nan  # Wednesday, day 9
    one, _ = repair_series(load, gap_fill="nearby_days", neighbour_days=1)
    four, _ = repair_series(load, gap_fill="nearby_days", neighbour_days=4)

    assert one["2025-01-15 03:00"] == _value(8, 3)
    # Tue, Thu, Mon, Fri.
    assert four["2025-01-15 03:00"] == np.mean([_value(d, 3) for d in (8, 10, 7, 11)])


def test_gap_without_a_valid_day_in_the_window_raises():
    load = _load()
    load[load.index.normalize() <= pd.Timestamp("2025-01-15")] = np.nan  # days 0-9

    with pytest.raises(ValueError, match="no valid reading at the same time of day within 7 days"):
        repair_series(load, gap_fill="nearby_days")


def test_target_index_turns_missing_leading_steps_into_gaps():
    load = _load()
    target = pd.date_range("2025-01-05", periods=len(load) + 24, freq="h")
    repaired, report = repair_series(load, gap_fill="nearby_days", index=target)

    assert repaired.index.equals(target)
    # Sunday 5 January takes the weekend days in reach: Saturday 11th (+6) and
    # Sunday 12th (+7).
    assert repaired["2025-01-05 10:00"] == (_value(5, 10) + _value(6, 10)) / 2
    assert report.events[0].missing_timestamps == 24

    with pytest.raises(ValueError, match="not on index"):
        repair_series(load, index=target[30:])


# --- DST --------------------------------------------------------------------


def test_lisbon_local_index_fills_by_legal_clock_across_dst():
    # 2025-03-30: Lisbon springs forward at 01:00 (23-hour day); 2025-10-26:
    # it falls back at 02:00 (25-hour day, 01:00 repeated).
    spring = _load(days=14, tz="Europe/Lisbon", start="2025-03-24")
    assert len(spring) == 14 * 24 - 1

    # Nothing to repair across the short day: the index is regular in absolute time.
    same, report = repair_series(spring)
    assert not report.repaired
    pd.testing.assert_series_equal(same, spring, check_freq=False)

    # Saturday 29 March 08:00 WET (08:00 UTC) takes Friday 28th and Sunday 30th
    # at 08:00 on the legal clock. On the Sunday that is 07:00 UTC.
    stamp = pd.Timestamp("2025-03-29 08:00", tz="Europe/Lisbon")
    repaired, _ = repair_series(spring.drop(stamp), gap_fill="nearby_days", same_day_type=False)
    assert repaired[stamp] == (_value(4, 8) + _value(6, 8)) / 2

    # Tuesday 1 April 08:00 WEST: Monday 31 March and Wednesday 2 April.
    stamp = pd.Timestamp("2025-04-01 08:00", tz="Europe/Lisbon")
    repaired, _ = repair_series(spring.drop(stamp), gap_fill="nearby_days")
    assert repaired[stamp] == (_value(7, 8) + _value(9, 8)) / 2

    autumn = _load(days=14, tz="Europe/Lisbon", start="2025-10-20")
    assert len(autumn) == 14 * 24 + 1
    # Both 01:00 steps of Sunday 26 October are missing. Each takes 01:00 on the
    # nearest weekend days in the series: Saturday 25th (-1) and Saturday 1 November (+6).
    repeated = autumn.index[autumn.index.tz_localize(None) == pd.Timestamp("2025-10-26 01:00")]
    assert len(repeated) == 2
    repaired, report = repair_series(autumn.drop(repeated), gap_fill="nearby_days")
    assert repaired[repeated].tolist() == [(_value(5, 1) + _value(12, 1)) / 2] * 2
    assert report.events[0].steps == 2


def test_pv_fills_by_utc_clock_and_ignores_day_type():
    idx = pd.date_range("2025-03-24", periods=14 * 24 - 1, freq="h", tz="Europe/Lisbon")
    utc_hour = idx.tz_convert("UTC").hour.to_numpy()
    pv = pd.Series(np.where((utc_hour >= 7) & (utc_hour <= 18), 100.0 * utc_hour, 0.0), index=idx, name="pv_w")
    # Monday 31 March 12:00 WEST is 11:00 UTC.
    stamp = pd.Timestamp("2025-03-31 12:00", tz="Europe/Lisbon")
    repaired, report = repair_series(pv.drop(stamp), kind="pv", gap_fill="nearby_days")

    assert repaired[stamp] == 1100.0
    assert (report.series, report.clock, report.same_day_type) == ("pv", "utc", False)


# --- negatives --------------------------------------------------------------


def test_small_negatives_are_clipped_and_reported():
    load = _load(freq="15min")
    load.iloc[10] = -3.0
    load.iloc[11] = -10.0
    load.iloc[500] = -0.5
    repaired, report = repair_series(load)

    assert repaired.iloc[[10, 11, 500]].tolist() == [0.0, 0.0, 0.0]
    assert [(e.issue, e.method, e.steps, e.original_min_w) for e in report.events] == [
        ("negative", "clip_to_zero", 2, -10.0),
        ("negative", "clip_to_zero", 1, -0.5),
    ]
    # 13 W for a quarter hour, then 0.5 W for a quarter hour.
    assert [e.energy_wh for e in report.events] == [3.25, 0.125]
    assert report.energy_added_wh == 3.375


def test_large_negative_load_raises_as_net_metering():
    load = _load()
    load.iloc[40] = -250.0

    with pytest.raises(ValueError, match=r"below -10 W at 1 steps.*net flow \(load minus on-site PV\)"):
        repair_series(load)
    with pytest.raises(ValueError, match="below -0 W"):
        small = _load()
        small.iloc[40] = -1.0
        repair_series(small, negative_clip_w=0.0)
    # A larger threshold is the caller's explicit choice.
    repaired, _ = repair_series(load, negative_clip_w=300.0)
    assert repaired.iloc[40] == 0.0


def test_sustained_small_negative_load_raises():
    load = _load(freq="15min")
    load.iloc[100:105] = -2.0  # 75 minutes

    with pytest.raises(ValueError, match="negative for 5 consecutive steps.*longer than max_negative_run"):
        repair_series(load)
    load.iloc[104] = 5.0  # 60 minutes is allowed
    _, report = repair_series(load)
    assert report.events[0].steps == 4


def test_large_negative_pv_raises():
    pv = _load().rename("pv_w")
    pv.iloc[3] = -50.0

    with pytest.raises(ValueError, match="PV output is not negative beyond inverter standby"):
        repair_series(pv, kind="pv")


# --- index checks -----------------------------------------------------------


def test_duplicate_unsorted_and_irregular_indexes_raise():
    load = _load(days=8)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        repair_series(pd.concat([load.iloc[:5], load.iloc[4:]]))
    with pytest.raises(ValueError, match="not sorted"):
        repair_series(load.iloc[::-1])
    shifted = load.index.to_series()
    shifted.iloc[50:] += pd.Timedelta(minutes=20)
    with pytest.raises(ValueError, match="irregular"):
        repair_series(pd.Series(load.to_numpy(), index=pd.DatetimeIndex(shifted)), freq="h")
    with pytest.raises(TypeError, match="DatetimeIndex"):
        repair_series(load.reset_index(drop=True))
    with pytest.raises(ValueError, match="gap_fill"):
        repair_series(load, gap_fill="linear")


# --- report -----------------------------------------------------------------


def test_report_energy_matches_the_actual_change():
    load = _load(freq="15min")
    load.iloc[7] = -6.0
    load.iloc[300:310] = np.nan
    load.iloc[900] = np.inf
    load = load.drop(load.index[1200:1204])
    repaired, report = repair_series(load, gap_fill="nearby_days")

    hours = 0.25
    finite = load[np.isfinite(load)]
    assert report.energy_before_wh == pytest.approx(finite.sum() * hours)
    assert report.energy_after_wh == pytest.approx(repaired.sum() * hours)
    assert report.energy_after_wh - report.energy_before_wh == pytest.approx(
        report.energy_added_wh - report.energy_removed_wh
    )
    assert report.energy_added_wh == pytest.approx(sum(e.energy_wh for e in report.events))
    assert report.energy_removed_wh == 0.0
    assert report.steps_repaired == 1 + 10 + 1 + 4 == sum(e.steps for e in report.events)
    assert [e.issue for e in report.events] == ["negative", "gap", "gap", "gap"]


def test_report_is_strict_json():
    load = _load()
    load.iloc[3] = -1.0
    load.iloc[100] = np.nan
    _, report = repair_series(load, gap_fill="nearby_days")

    text = json.dumps(report.to_dict(), allow_nan=False)
    assert json.loads(text) == report.to_dict()
    assert isinstance(report, InputRepairReport)
    assert report.repaired


def test_clean_series_returns_an_empty_report():
    load = _load().astype("int64")
    repaired, report = repair_series(load)

    assert repaired.dtype == float
    assert repaired.to_numpy().tolist() == load.to_numpy(dtype=float).tolist()
    assert not report.repaired
    assert report.events == ()
    assert report.energy_added_wh == report.energy_removed_wh == 0.0
    assert report.to_dict()["events"] == []


# --- App provenance ---------------------------------------------------------


_CONFIG = {
    "location": "porto",
    "n_modules": 4,
    "annual_consumption_kwh": 3000,
    "load_profile": "8",
    "projection_years": 1,
}


def _write_measured_load(directory):
    """Write a repaired 2023 hourly measured load as the generic RLP file."""
    idx = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC", name="Datetime")
    load = pd.Series(300.0 + 200.0 * np.sin(idx.hour.to_numpy() / 24 * 2 * np.pi) ** 2, index=idx, name="W")
    load = load.drop(idx[1000:1003])
    load.iloc[20] = -2.0
    repaired, report = repair_series(load, gap_fill="nearby_days")
    repaired.to_csv(directory / "REE_2026_2.0TD_1000kwh_hourly.csv")
    return report


@pytest.mark.usefixtures("_patch_weather")
def test_repair_report_reaches_app_provenance(tmp_path):
    report = _write_measured_load(tmp_path)
    config = {**_CONFIG, "rlp_directory": str(tmp_path)}

    app = App(config, input_repairs=[report])
    app.simulate()
    provenance = app.result()["provenance"]

    assert provenance["input_repairs"] == [report.to_dict()]
    json.dumps(provenance["input_repairs"], allow_nan=False)

    # A report saved as JSON and loaded back is accepted too, and a lone report.
    assert App(config, input_repairs=json.loads(json.dumps(report.to_dict())))._input_repairs == [report.to_dict()]
    assert App(config, input_repairs=report)._input_repairs == [report.to_dict()]


@pytest.mark.usefixtures("_patch_weather")
def test_app_without_repairs_has_no_input_repairs_key(tmp_path):
    _write_measured_load(tmp_path)
    app = App({**_CONFIG, "rlp_directory": str(tmp_path)})
    app.simulate()

    assert "input_repairs" not in app.result()["provenance"]


def test_app_rejects_input_repairs_that_are_not_reports():
    with pytest.raises(TypeError, match="input_repairs"):
        App(_CONFIG, input_repairs=["load repaired"])
    with pytest.raises(ValueError, match="not strict JSON"):
        App(_CONFIG, input_repairs=[{"energy_added_wh": float("nan")}])


def test_repair_api_is_exported_from_io_and_the_top_level():
    import breos.repair

    assert breos.repair_series is repair_series is breos.repair.repair_series
    assert breos.InputRepairReport is InputRepairReport
