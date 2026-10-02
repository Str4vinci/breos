"""The timezone of the simulation result index, per weather source (#180).

The result index takes the weather index's clock. It is not the configured
IANA timezone, so a tariff resolver must ``tz_convert`` to
``ResolvedAppConfig.timezone`` itself instead of reading ``index.tz``. These
tests pin today's clocks so that a change to them is deliberate:

- a PVGIS fetch gives the fixed offset of 1 January (``Etc/GMT-N``), all year;
- a cached weather CSV keeps a single fixed offset written in its stamps
  (such as a saved PVGIS fetch), and is read as UTC when its stamps are UTC,
  naive, or carry more than one offset (an IANA zone across DST);
- Monte Carlo historical weather is indexed in UTC.
"""

import pandas as pd
import pytest

from breos.app import App
from breos.montecarlo import _index_weather
from breos.runners.app import run_app_simulation

_BASE = {"n_modules": 6, "annual_consumption_kwh": 3000, "projection_years": 1, "start_date": "2025-01-01"}


def _result_index(config):
    app = App(config)
    artifacts = run_app_simulation(app._resolved, app._runtime_dependencies())
    return pd.DatetimeIndex(artifacts.first_year_results_df["Datetime"]), app._resolved.timezone


@pytest.fixture
def fake_pvgis(monkeypatch, tmp_path, synthetic_weather):
    """Serve the synthetic year through pvlib's own PVGIS TMY roll, offline."""
    from pvlib.iotools.pvgis import _coerce_and_roll_tmy

    def fake_get_pvgis_tmy(latitude, longitude, *args, roll_utc_offset=None, coerce_year=None, **kwargs):
        data = _coerce_and_roll_tmy(synthetic_weather.copy(), roll_utc_offset, coerce_year)
        return data, {"inputs": {"location": {"latitude": latitude, "longitude": longitude, "elevation": 0}}}

    monkeypatch.setattr("breos.weather.pvlib.iotools.get_pvgis_tmy", fake_get_pvgis_tmy)
    # No weather/ directory in the working directory, so App fetches.
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    ("location", "expected_tz"),
    [
        ({"latitude": 52.52, "longitude": 13.405, "timezone": "Europe/Berlin"}, "Etc/GMT-1"),
        ({"latitude": 40.71, "longitude": -74.01, "timezone": "America/New_York"}, "Etc/GMT+5"),
        # 1 January is daylight time in Melbourne, so its whole year runs on UTC+11.
        ({"latitude": -37.81, "longitude": 144.96, "timezone": "Australia/Melbourne"}, "Etc/GMT-11"),
    ],
    ids=["berlin", "new_york", "melbourne"],
)
@pytest.mark.parametrize("resolution", ["h", "15min"])
def test_pvgis_weather_gives_the_fixed_offset_of_1_january(fake_pvgis, location, expected_tz, resolution):
    index, configured_tz = _result_index({**_BASE, "location": location, "resolution": resolution})

    assert str(index.tz) == expected_tz
    assert str(index.tz) != configured_tz
    # pvlib rolls the rows so the year starts at local midnight on this clock.
    assert index[0] == pd.Timestamp("2025-01-01 00:00", tz=expected_tz)
    # The offset stays fixed all year, so in July (daylight time in Berlin and
    # New York, standard time in Melbourne) it differs from civil time.
    july = index[index.month == 7][0]
    assert july.utcoffset() == index[0].utcoffset()
    assert july.utcoffset() != july.tz_convert(configured_tz).utcoffset()


def _write_porto_csv(directory, weather):
    directory.mkdir()
    weather.to_csv(directory / "porto_tmy_2005_2023_pvgis-sarah3.csv")


@pytest.mark.parametrize(
    ("stamps", "expected_tz"),
    [("utc", "UTC"), ("naive", "UTC"), ("lisbon_dst_offsets", "UTC"), ("fixed_offset", "UTC+01:00")],
)
def test_cached_weather_csv_clock(monkeypatch, tmp_path, synthetic_weather, stamps, expected_tz):
    weather = synthetic_weather.copy()
    if stamps == "naive":
        weather.index = weather.index.tz_localize(None)
    elif stamps == "lisbon_dst_offsets":
        weather.index = weather.index.tz_convert("Europe/Lisbon")
    elif stamps == "fixed_offset":
        weather.index = weather.index.tz_convert("Etc/GMT-1")
    _write_porto_csv(tmp_path / "weather", weather)
    monkeypatch.chdir(tmp_path)

    def _no_fetch(*args, **kwargs):
        raise AssertionError("App fetched PVGIS weather instead of using the local cache")

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _no_fetch)

    index, configured_tz = _result_index({**_BASE, "location": "porto"})

    assert str(index.tz) == expected_tz
    assert configured_tz == "Europe/Lisbon"
    # The synthetic year starts at 00:00 UTC, and keeps that instant on every clock.
    assert index[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC")


@pytest.mark.parametrize("tz", [None, "UTC", "Europe/Berlin", "Etc/GMT-1"])
def test_montecarlo_historical_weather_is_indexed_in_utc(synthetic_weather, tz):
    weather = synthetic_weather.copy()
    weather.index = weather.index.tz_localize(None) if tz is None else weather.index.tz_convert(tz)

    indexed = _index_weather(weather.reset_index(names="date"))

    assert str(indexed.index.tz) == "UTC"
    assert indexed.index[0] == pd.Timestamp("2023-01-01 00:00", tz="UTC")
