"""App weather must cover the whole simulated calendar year (#242).

The simulation calendar runs from the first to the last weather row, so
weather missing its first or last week used to simulate about 51 weeks while
the economics treated the run as a full year.
"""

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.app_inputs import remap_tmy_year, require_full_year_weather

_PORTO = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5,
    "projection_years": 1,
    "start_date": "2025-01-01",
}


def _year(start: str, periods: int = 8760, freq: str = "h", tz: str = "UTC") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    frame = pd.DataFrame({"ghi": np.zeros(len(index)), "temp_air": np.full(len(index), 15.0)}, index=index)
    frame.attrs["breos_weather_metadata"] = {"source": "synthetic"}
    return frame


def _patch_fetch(monkeypatch, weather: pd.DataFrame) -> None:
    def fetch(*_args, **_kwargs):
        frame = weather.copy()
        frame.attrs["breos_weather_metadata"] = {
            "source": "PVGIS_TMY",
            "horizon": {"status": "not_applied", "provider": "pvgis", "profile": None},
        }
        return frame, {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", fetch)
    monkeypatch.setattr("breos.app.load_weather", lambda **_kwargs: None)


def test_cached_weather_file_missing_its_first_week_raises(tmp_path, monkeypatch, synthetic_weather):
    """The reported case: a cached TMY starting 8 January used to run as a full year."""
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir()
    path = weather_dir / "porto_tmy_2005_2023_pvgis-sarah3.csv"
    synthetic_weather.iloc[7 * 24 :].to_csv(path)
    monkeypatch.chdir(tmp_path)

    app = App(_PORTO)
    with pytest.raises(ValueError, match=r"leading 7 days 00:00:00 \(168 steps") as excinfo:
        app.simulate()

    message = str(excinfo.value)
    assert str(path.resolve()) in message
    assert "2025-01-01 00:00:00+00:00 to 2025-01-07 23:00:00+00:00" in message
    assert "trailing" not in message


def test_weather_missing_its_last_week_raises(monkeypatch, synthetic_weather):
    _patch_fetch(monkeypatch, synthetic_weather.iloc[: -7 * 24])

    app = App(_PORTO)
    with pytest.raises(ValueError, match=r"Weather from PVGIS_TMY .* trailing 7 days 00:00:00 \(168 steps") as excinfo:
        app.simulate()

    assert "2025-12-25 00:00:00+00:00 to 2025-12-31 23:00:00+00:00" in str(excinfo.value)
    assert "leading" not in str(excinfo.value)


def test_weather_missing_both_ends_names_both_spans():
    weather = _year("2025-01-02", periods=8760 - 48)
    with pytest.raises(ValueError, match=r"leading 1 days .* and the trailing 1 days .* are missing"):
        require_full_year_weather(weather, 2025, "h", "UTC")


@pytest.mark.parametrize("tz", ["UTC", "Europe/Lisbon", "Etc/GMT-1"])
def test_a_complete_year_runs_whatever_its_timezone(monkeypatch, synthetic_weather, tz):
    """A UTC year respelled in each timezone, and a fixed-offset PVGIS year."""
    respelled = synthetic_weather.copy()
    respelled.index = respelled.index.tz_convert(tz)
    _patch_fetch(monkeypatch, respelled)

    app = App(_PORTO)
    app.simulate()

    assert app.result()["consumption_kwh"] == pytest.approx(4000.0, rel=1e-9)


@pytest.mark.parametrize(
    ("weather", "timezone"),
    [
        # PVGIS TMY: local midnight to local midnight at a fixed offset.
        (_year("2025-01-01", tz="Etc/GMT-1"), "Europe/Berlin"),
        # A UTC year spelled in the location's civil time.
        (_year("2025-01-01").tz_convert("Europe/Berlin"), "Europe/Berlin"),
        # A civil year west of UTC spelled in UTC.
        (_year("2025-01-01", tz="America/New_York").tz_convert("UTC"), "America/New_York"),
        # Labels offset from the hour by less than one step.
        (_year("2025-01-01 00:30"), "UTC"),
    ],
    ids=["fixed-offset", "utc-year-in-civil-time", "civil-year-in-utc", "half-hour-labels"],
)
def test_complete_years_on_any_accepted_clock_pass(weather, timezone):
    require_full_year_weather(weather, 2025, "h", timezone)


def test_15min_run_on_resampled_hourly_weather_passes(monkeypatch, synthetic_weather):
    """The resampler fills the last hour's quarter-hours from the last hourly row."""
    _patch_fetch(monkeypatch, synthetic_weather)

    app = App({**_PORTO, "resolution": "15min"})
    app.simulate()

    assert app.result()["consumption_kwh"] == pytest.approx(4000.0, rel=1e-9)


def test_15min_weather_ending_on_the_last_hour_is_three_steps_short():
    weather = _year("2025-01-01", periods=8760 * 4 - 3, freq="15min")
    require_full_year_weather(_year("2025-01-01", periods=8760 * 4, freq="15min"), 2025, "15min", "UTC")
    with pytest.raises(ValueError, match=r"trailing 0 days 00:45:00 \(3 steps"):
        require_full_year_weather(weather, 2025, "15min", "UTC")


def test_a_tmy_restamped_onto_a_leap_year_passes():
    weather = remap_tmy_year(_year("2025-01-01"), 2028)

    assert len(weather) == 8784
    require_full_year_weather(weather, 2028, "h", "UTC")


def test_a_leap_year_without_its_last_day_raises():
    """8,760 rows from 1 January are one day short of a leap year."""
    with pytest.raises(ValueError, match=r"trailing 1 days 00:00:00 \(24 steps"):
        require_full_year_weather(_year("2028-01-01"), 2028, "h", "UTC")
