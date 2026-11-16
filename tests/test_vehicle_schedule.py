"""Civil-time behavior, energy and parked time of the pure vehicle calendar."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from breos.vehicle import VehicleScheduleSpec, resolve_vehicle_schedule


def rules(depart="08:00", arrive="18:00", trip_kwh=10, **extra):
    day = {"depart": depart, "arrive": arrive, "trip_kwh": trip_kwh, **extra}
    return {"weekday": day, "weekend": dict(day)}


def day_index(day, *, timezone="Europe/Lisbon", freq="h"):
    start = pd.Timestamp(day, tz=timezone)
    return pd.date_range(start, start + pd.DateOffset(days=1), freq=freq, inclusive="left").tz_convert("UTC")


def csv_schedule(tmp_path, rows, index, *, timezone="Europe/Lisbon"):
    path = tmp_path / "trips.csv"
    path.write_text("start,end,kind,energy_kwh,away_charging\n" + rows)
    return resolve_vehicle_schedule({"file": str(path)}, index, timezone=timezone, reference_year=index[0].year)


@pytest.mark.parametrize(
    ("date", "depart", "arrive", "energy"),
    [
        ("2026-01-09", 8, 18, 10000),
        ("2026-01-10", 10, 14, 12000),
    ],
)
def test_departure_civil_day_selects_weekday_or_weekend(date, depart, arrive, energy):
    schedule = rules()
    schedule["weekend"] = {"depart": "10:00", "arrive": "14:00", "trip_kwh": 12}
    index = day_index(date)
    resolved = resolve_vehicle_schedule(schedule, index, timezone="Europe/Lisbon", reference_year=2026)
    np.testing.assert_array_equal(resolved.plugged, (index.hour < depart) | (index.hour >= arrive))
    assert resolved.drive_first_wh[depart] == resolved.drive_second_wh[arrive - 1] == energy / 2
    assert (resolved.drive_first_wh + resolved.drive_second_wh).sum() == energy
    assert resolved.park_hours.sum() == arrive - depart - 2
    assert resolved.steps_to_departure[depart - 1] == 1


@pytest.mark.parametrize("zone", ["Europe/Lisbon", "Europe/Berlin"])
@pytest.mark.parametrize(("date", "hours", "away_hours"), [("2026-03-29", 23, 3), ("2026-10-25", 25, 5)])
@pytest.mark.parametrize("freq", ["h", "15min"])
def test_spring_and_autumn_use_elapsed_hours_in_the_configured_zone(zone, date, hours, away_hours, freq):
    index = day_index(date, timezone=zone, freq=freq)
    resolved = resolve_vehicle_schedule(rules("00:00", "04:00"), index, timezone=zone, reference_year=2026)
    step_hours = 1 if freq == "h" else 0.25
    assert len(index) * step_hours == hours
    assert (~resolved.plugged).sum() * step_hours == away_hours
    assert resolved.park_hours.sum() == away_hours - 2 * step_hours
    assert (resolved.drive_first_wh + resolved.drive_second_wh).sum() == 10000
    # Changing only the index display zone changes none of the calendar.
    other = resolve_vehicle_schedule(
        rules("00:00", "04:00"), index.tz_convert(zone), timezone=zone, reference_year=2026
    )
    assert resolved.schedule_hash() == other.schedule_hash()


@pytest.mark.parametrize(("zone", "repeated"), [("Europe/Lisbon", "01:30"), ("Europe/Berlin", "02:30")])
def test_repeated_departure_uses_first_occurrence_and_arrival_uses_last(zone, repeated):
    index = day_index("2026-10-25", timezone=zone, freq="15min")
    departing = resolve_vehicle_schedule(rules(repeated, "05:00"), index, timezone=zone, reference_year=2026)
    arriving = resolve_vehicle_schedule(rules("00:00", repeated), index, timezone=zone, reference_year=2026)
    first = pd.Timestamp(f"2026-10-25 {repeated}").tz_localize(zone, ambiguous=True)
    last = pd.Timestamp(f"2026-10-25 {repeated}").tz_localize(zone, ambiguous=False)
    assert not departing.plugged[index.get_loc(first)]
    assert not departing.plugged[index.get_loc(last)]
    assert not arriving.plugged[index.get_loc(first)]
    assert arriving.plugged[index.get_loc(last)]


@pytest.mark.parametrize(
    ("zone", "missing", "transition"),
    [
        ("Europe/Lisbon", "01:30", "02:00"),
        ("Europe/Berlin", "02:30", "03:00"),
    ],
)
def test_nonexistent_template_time_meets_the_spring_transition(zone, missing, transition):
    index = day_index("2026-03-29", timezone=zone, freq="15min")
    resolved = resolve_vehicle_schedule(rules(missing, "05:00"), index, timezone=zone, reference_year=2026)
    boundary = pd.Timestamp(f"2026-03-29 {transition}", tz=zone)
    pos = index.get_loc(boundary)
    assert resolved.plugged[pos - 1] and not resolved.plugged[pos]
    assert resolved.drive_first_wh[pos] == 5000


@pytest.mark.parametrize(
    ("freq", "first", "last", "parked"),
    [
        ("h", "07:00", "18:00", 10),
        ("15min", "07:45", "18:30", 10.5),
    ],
)
def test_template_conservatively_rounds_partial_steps(freq, first, last, parked):
    index = day_index("2026-01-05", freq=freq)
    resolved = resolve_vehicle_schedule(rules("07:50", "18:35"), index, timezone="Europe/Lisbon", reference_year=2026)
    positions = np.flatnonzero(~resolved.plugged)
    assert index[positions[0]].strftime("%H:%M") == first
    assert index[positions[-1]].strftime("%H:%M") == last
    assert resolved.park_hours.sum() == parked
    assert resolved.drive_first_wh.sum() == resolved.drive_second_wh.sum() == 5000
    assert resolved.steps_to_departure[positions[0] - 1] == 1


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_one_away_step_combines_both_legs_and_never_charges(freq):
    index = day_index("2026-01-05", freq=freq)
    resolved = resolve_vehicle_schedule(rules("08:01", "08:10"), index, timezone="Europe/Lisbon", reference_year=2026)
    pos = np.flatnonzero(~resolved.plugged)
    assert len(pos) == 1
    assert resolved.drive_first_wh[pos[0]] == resolved.drive_second_wh[pos[0]] == 5000
    assert not resolved.park_hours.any()


def test_day_rule_can_disable_away_charging_without_changing_gross_driving():
    index = day_index("2026-01-05")
    enabled = resolve_vehicle_schedule(rules(), index, timezone="Europe/Lisbon", reference_year=2026)
    disabled = resolve_vehicle_schedule(
        rules(away_charging=False), index, timezone="Europe/Lisbon", reference_year=2026
    )
    assert disabled.park_hours.sum() == 0 and enabled.park_hours.sum() == 8
    np.testing.assert_array_equal(enabled.drive_first_wh, disabled.drive_first_wh)
    np.testing.assert_array_equal(enabled.drive_second_wh, disabled.drive_second_wh)


def test_period_end_uses_next_departure_beyond_the_window():
    index = pd.date_range("2026-01-05 20:00", periods=2, freq="h", tz="UTC")
    resolved = resolve_vehicle_schedule(rules(), index, timezone="Europe/Lisbon", reference_year=2026)
    np.testing.assert_array_equal(resolved.steps_to_departure, [12, 11])


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_calendar_seam_repeats_its_own_day_types_and_carries_overnight_legs(freq):
    index = pd.date_range("2023-01-01", "2024-01-01", inclusive="left", freq=freq, tz="UTC")
    spec = rules("22:00", "02:00")
    spec["weekend"] = {"depart": "21:00", "arrive": "03:00", "trip_kwh": 20}
    resolved = resolve_vehicle_schedule(spec, index, timezone="Europe/Lisbon", reference_year=2023)
    assert not resolved.plugged[0] and not resolved.plugged[-1]
    assert (
        resolved.drive_second_wh[index.get_loc("2023-01-01 02:45" if freq == "15min" else "2023-01-01 02:00")] == 10000
    )
    assert resolved.drive_first_wh[index.get_loc("2023-12-31 21:00")] == 10000
    assert (resolved.drive_first_wh + resolved.drive_second_wh).sum() == (260 * 10 + 105 * 20) * 1000
    # Jan 1 2023 is Sunday, whereas Jan 1 2024 is Monday. The replay's
    # next departure uses Sunday's 10:00, not Monday's 08:00.
    daytime = rules()
    daytime["weekend"] = {"depart": "10:00", "arrive": "14:00", "trip_kwh": 12}
    day = resolve_vehicle_schedule(daytime, index, timezone="Europe/Lisbon", reference_year=2023)
    assert day.steps_to_departure[-1] == (10 * (4 if freq == "15min" else 1)) + 1


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_multiday_trip_file_carries_legs_and_actual_parked_hours_across_the_seam(tmp_path, freq):
    index = pd.date_range("2023-01-01", "2024-01-01", inclusive="left", freq=freq, tz="UTC")
    resolved = csv_schedule(
        tmp_path,
        "2023-12-30T08:30Z,2023-12-30T09:00Z,drive,2,\n"
        "2023-12-30T09:00Z,2024-01-02T11:30Z,park,,true\n"
        "2024-01-02T11:30Z,2024-01-02T12:00Z,drive,4,\n",
        index,
    )
    assert not resolved.plugged[0] and not resolved.plugged[-1]
    assert resolved.plugged[index.get_loc("2023-01-02 12:00")]
    assert resolved.park_hours[0] == (1 if freq == "h" else 0.25)
    assert (resolved.drive_first_wh + resolved.drive_second_wh).sum() == 6000
    assert resolved.park_hours.sum() == 74.5


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_trip_file_distributes_gross_energy_and_keeps_partial_charging_windows(tmp_path, freq):
    index = pd.date_range("2026-01-05 07:00", periods=16 if freq == "15min" else 4, freq=freq, tz="UTC")
    resolved = csv_schedule(
        tmp_path,
        "2026-01-05 07:50,2026-01-05 08:10,drive,4,\n"
        "2026-01-05 08:10,2026-01-05 08:40,park,,true\n"
        "2026-01-05 08:40,2026-01-05 09:00,drive,2,\n"
        "2026-01-05 09:00,2026-01-05 09:30,park,,false\n"
        "2026-01-05 09:30,2026-01-05 10:00,drive,1,\n",
        index,
    )
    assert (resolved.drive_first_wh + resolved.drive_second_wh).sum() == 7000
    assert resolved.park_hours.sum() == 0.5
    assert resolved.plugged[-1 if freq == "h" else -4]
    if freq == "h":
        np.testing.assert_allclose(resolved.drive_first_wh, [2000, 2000, 0, 0])
        np.testing.assert_allclose(resolved.drive_second_wh, [0, 2000, 1000, 0])
        np.testing.assert_allclose(resolved.park_hours, [0, 0.5, 0, 0])
    else:
        # 07:45 is away even though its first five minutes were at home.
        assert resolved.plugged[2] and not resolved.plugged[3]
        assert resolved.drive_first_wh[3] == 2000
        assert resolved.park_hours[4] == pytest.approx(5 / 60)


def test_multiple_parks_collapse_driving_after_first_park_into_second_leg(tmp_path):
    index = pd.date_range("2026-01-05 08:00", periods=1, freq="h", tz="UTC")
    resolved = csv_schedule(
        tmp_path,
        "2026-01-05 08:00,2026-01-05 08:10,drive,1,\n"
        "2026-01-05 08:10,2026-01-05 08:20,park,,true\n"
        "2026-01-05 08:20,2026-01-05 08:30,drive,2,\n"
        "2026-01-05 08:30,2026-01-05 08:40,park,,true\n"
        "2026-01-05 08:40,2026-01-05 09:00,drive,3,\n",
        index,
    )
    assert resolved.drive_first_wh[0] == 1000 and resolved.drive_second_wh[0] == 5000
    assert resolved.park_hours[0] == pytest.approx(1 / 3)


def test_separate_short_absences_in_one_step_share_the_first_park_boundary(tmp_path):
    index = pd.date_range("2026-01-05 08:00", periods=1, freq="h", tz="UTC")
    resolved = csv_schedule(
        tmp_path,
        "2026-01-05 08:00,2026-01-05 08:10,drive,1,\n"
        "2026-01-05 08:10,2026-01-05 08:20,park,,true\n"
        "2026-01-05 08:30,2026-01-05 08:40,drive,2,\n"
        "2026-01-05 08:40,2026-01-05 09:00,park,,true\n",
        index,
    )
    assert resolved.drive_first_wh[0] == 1000 and resolved.drive_second_wh[0] == 2000
    assert resolved.park_hours[0] == 0.5


@pytest.mark.parametrize(
    "rows",
    [
        "2026-01-05 08:00,2026-01-05 09:00,drive,1,\n2026-01-05 08:30,2026-01-05 10:00,park,,true\n",
        "2026-01-05 09:00,2026-01-05 08:00,drive,1,\n",
        "2026-01-05 08:00,2026-01-05 09:00,drive,-1,\n",
        "2026-01-05 08:00,2026-01-05 09:00,drive,nan,\n",
        "2026-01-05 08:00,2026-01-05 09:00,drive,true,\n",
        "2026-01-05 08:00,2026-01-05 09:00,drive,,\n",
        "2026-01-05 08:00,2026-01-05 09:00,park,,maybe\n",
        "2026-01-05 08:00,2026-01-05 09:00,park,1,true\n",
        "2026-01-05 08:00,2026-01-05 09:00,drive,1,true\n",
        "2026-01-05 08:00,2026-01-05 09:00,home,,\n",
    ],
)
def test_invalid_trip_segments_are_refused(tmp_path, rows):
    with pytest.raises(ValueError, match="Vehicle"):
        csv_schedule(tmp_path, rows, day_index("2026-01-05"))


@pytest.mark.parametrize(("zone", "clock"), [("Europe/Lisbon", "01:30"), ("Europe/Berlin", "02:30")])
@pytest.mark.parametrize("date", ["2026-03-29", "2026-10-25"])
def test_naive_csv_dst_transition_times_need_explicit_offsets(tmp_path, zone, clock, date):
    with pytest.raises(ValueError, match="explicit offset"):
        csv_schedule(tmp_path, f"{date} {clock},{date} 05:00,drive,1,\n", day_index(date, timezone=zone), timezone=zone)


def test_offset_qualified_file_keeps_both_occurrences_of_repeated_hour(tmp_path):
    index = day_index("2026-10-25", freq="15min")
    resolved = csv_schedule(tmp_path, "2026-10-25T01:30+01:00,2026-10-25T01:30+00:00,park,,true\n", index)
    assert resolved.park_hours.sum() == 1
    assert (~resolved.plugged).sum() == 4


def test_driving_across_repeated_hour_does_not_change_duration_in_replayed_halo(tmp_path):
    index = day_index("2026-10-25", freq="15min")
    resolved = csv_schedule(tmp_path, "2026-10-25T01:40+01:00,2026-10-25T01:20+00:00,drive,4,\n", index)
    assert (resolved.drive_first_wh + resolved.drive_second_wh).sum() == 4000
    assert (~resolved.plugged).sum() == 4


@pytest.mark.parametrize("year", [2023, 2024])
def test_file_departure_count_crosses_the_reference_year_seam(tmp_path, year):
    index = pd.date_range(f"{year}-12-31 23:00", periods=1, freq="h", tz="UTC")
    resolved = csv_schedule(tmp_path, f"{year}-01-05 08:00,{year}-01-05 09:00,drive,1,\n", index)
    assert resolved.plugged[0] and resolved.steps_to_departure[0] == 105


def test_file_with_wrong_reference_year_is_refused_instead_of_ignoring_trips(tmp_path):
    with pytest.raises(ValueError, match="reference calendar year"):
        csv_schedule(tmp_path, "2025-01-05 08:00,2025-01-05 09:00,drive,1,\n", day_index("2026-01-05"))


def test_file_refuses_an_unrelated_absence_outside_its_reference_year(tmp_path):
    index = pd.date_range("2025-01-01", "2026-01-01", inclusive="left", freq="h", tz="UTC")
    with pytest.raises(ValueError, match=r"2026-06-02.*reference calendar year.*2025"):
        csv_schedule(
            tmp_path,
            "2025-06-02 08:00,2025-06-02 09:00,drive,3,\n2026-06-02 08:00,2026-06-02 09:00,drive,4,\n",
            index,
        )


@pytest.mark.parametrize("year", [1985, 1986])
@pytest.mark.parametrize("file", [False, True])
def test_year_bounds_use_the_first_real_instant_when_local_midnight_is_missing(tmp_path, year, file):
    # Lima advanced the clock at midnight on 1 January 1986. Exercise both
    # a missing upper year bound and a missing lower bound.
    index = pd.date_range(f"{year}-06-02", periods=24, freq="h", tz="UTC")
    if file:
        resolved = csv_schedule(
            tmp_path, f"{year}-06-02 08:00,{year}-06-02 09:00,drive,3,\n", index, timezone="America/Lima"
        )
    else:
        resolved = resolve_vehicle_schedule(rules(), index, timezone="America/Lima", reference_year=year)
    assert (resolved.drive_first_wh + resolved.drive_second_wh).sum() == (3000 if file else 10000)


def test_empty_file_means_home_with_no_departure(tmp_path):
    resolved = csv_schedule(tmp_path, "", day_index("2026-01-05"))
    assert resolved.plugged.all() and not resolved.steps_to_departure.any()
    assert not (resolved.drive_first_wh + resolved.drive_second_wh + resolved.park_hours).any()


def test_hash_is_stable_content_based_and_sensitive_to_every_resolved_array(tmp_path):
    index = day_index("2026-01-05")
    spec = VehicleScheduleSpec(**rules())
    baseline = resolve_vehicle_schedule(spec, index, timezone="Europe/Lisbon", reference_year=2026)
    assert len(baseline.schedule_hash()) == 64
    assert (
        baseline.schedule_hash()
        == resolve_vehicle_schedule(rules(), index, timezone="Europe/Lisbon", reference_year=2026).schedule_hash()
    )
    # Independent copies and pandas timestamp units have the same hash.
    assert replace(baseline, index=index.as_unit("us")).schedule_hash() == baseline.schedule_hash()
    for name in ("plugged", "steps_to_departure", "drive_first_wh", "drive_second_wh", "park_hours"):
        array = getattr(baseline, name)
        assert not array.flags.writeable
        changed = array.copy()
        changed[0] = not changed[0] if name == "plugged" else changed[0] + 1
        assert replace(baseline, **{name: changed}).schedule_hash() != baseline.schedule_hash()
    assert replace(baseline, index=index + pd.Timedelta(hours=1)).schedule_hash() != baseline.schedule_hash()
    assert replace(baseline, hours_per_step=0.25).schedule_hash() != baseline.schedule_hash()
    # Filename and CSV formatting are not provenance content.
    rows = "2026-01-05 08:00,2026-01-05 09:00,drive,1,\n"
    original = csv_schedule(tmp_path, rows, index)
    assert csv_schedule(tmp_path, rows.replace(",1,", ",1.0,"), index).schedule_hash() == original.schedule_hash()
    assert csv_schedule(tmp_path, rows.replace(",1,", ",2,"), index).schedule_hash() != original.schedule_hash()


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2026-01-05", periods=2, freq="h"),
        pd.DatetimeIndex([], tz="UTC"),
        pd.DatetimeIndex(["2026-01-05", "2026-01-05"], tz="UTC"),
        pd.DatetimeIndex(["2026-01-05 01:00", "2026-01-05"], tz="UTC"),
        pd.DatetimeIndex(["2026-01-05", "2026-01-05 01:00", "2026-01-05 03:00"], tz="UTC"),
    ],
)
def test_schedule_refuses_an_invalid_simulation_index(index):
    with pytest.raises(ValueError, match="simulation index"):
        resolve_vehicle_schedule(rules(), index, timezone="Europe/Lisbon", reference_year=2026)
