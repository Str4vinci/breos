"""Tests for weather and weather-derived helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breos.weather import (
    build_battery_temperature_series,
    fetch_tmy_weather_data,
    fetch_weather_data,
    load_weather,
    parse_weather_filename,
    preload_weather_by_year,
    read_epw_file,
    relabel_right_labeled_interval_means,
    resample_tmy_to_15min,
    resample_to_15min,
    save_weather_csv,
    select_random_year_and_replace_datetime,
    weather_representative_time_offset,
)


def _timed_weather(*, basis: str, label: str, irradiance_offset_hours: float = 0.0):
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    weather = pd.DataFrame({"ghi": [0.0, 1.0, 0.0]}, index=index)
    weather.attrs["breos_weather_metadata"] = {
        "radiation_time_basis": basis,
        "timestamp_label_basis": label,
        "irradiance_time_offset_hours": irradiance_offset_hours,
    }
    return weather


def test_weather_representative_time_offsets_cover_provider_left_and_right_labels():
    instant = _timed_weather(basis="instant", label="provider_hour", irradiance_offset_hours=0.1714)
    left = _timed_weather(basis="interval_mean", label="left")
    right = _timed_weather(basis="interval_mean", label="right")

    assert weather_representative_time_offset(instant, "h") == pd.Timedelta(hours=0.1714)
    assert weather_representative_time_offset(left, "h") == pd.Timedelta(minutes=30)
    assert weather_representative_time_offset(right, "h") == pd.Timedelta(minutes=-30)


def test_right_labeled_interval_means_are_reindexed_to_interval_start():
    weather = _timed_weather(basis="interval_mean", label="right")

    relabeled = relabel_right_labeled_interval_means(weather)

    assert relabeled.index.equals(weather.index - pd.Timedelta(hours=1))
    metadata = relabeled.attrs["breos_weather_metadata"]
    assert metadata["timestamp_label_basis"] == "left"
    assert metadata["source_timestamp_label_basis"] == "right"
    assert weather.attrs["breos_weather_metadata"]["timestamp_label_basis"] == "right"


def test_resampling_right_labeled_means_preserves_physical_interval_start():
    weather = _timed_weather(basis="interval_mean", label="right")

    resampled = resample_to_15min(weather, method="linear")

    assert resampled.index[0] == weather.index[0] - pd.Timedelta(hours=1)
    assert weather_representative_time_offset(resampled, "15min") == pd.Timedelta(minutes=7.5)


def test_pvgis_provider_offset_survives_resampling_exactly():
    weather = _timed_weather(basis="instant", label="provider_hour", irradiance_offset_hours=0.1714)

    resampled = resample_to_15min(weather, method="linear")

    assert weather_representative_time_offset(resampled, "15min") == pd.Timedelta(hours=0.1714)


def test_preloaded_years_restore_content_bound_timestamp_metadata(tmp_path):
    path = tmp_path / "historical.csv"
    dates = pd.date_range("2025-01-01 01:00", periods=8760, freq="h")
    pd.DataFrame({"date": dates, "shortwave_radiation": 0.0}).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {
        "schema_version": 1,
        "weather_sha256": digest,
        "breos_weather_metadata": {
            "radiation_time_basis": "interval_mean",
            "timestamp_label_basis": "right",
            "raw_radiation_variables": ["shortwave_radiation"],
        },
    }
    Path(f"{path}.metadata.json").write_text(json.dumps(sidecar))

    loaded = preload_weather_by_year(str(path), target_year=2025)[2025]

    assert loaded["date"].iloc[0] == pd.Timestamp("2025-01-01 00:00")
    assert loaded["date"].iloc[-1] == pd.Timestamp("2025-12-31 23:00")
    assert loaded.attrs["breos_weather_metadata"]["timestamp_label_basis"] == "left"
    assert loaded.attrs["breos_weather_metadata"]["source_timestamp_label_basis"] == "right"
    assert loaded.attrs["breos_weather_metadata"]["sha256"] == digest


def _write_leap_year_15min_weather(tmp_path):
    idx = pd.date_range("2024-01-01 00:00", "2024-12-31 23:45", freq="15min")
    df = pd.DataFrame({"date": idx, "temp_air": range(len(idx))})
    path = tmp_path / "weather.csv"
    df.to_csv(path, index=False)
    return path, df


def _install_fake_openmeteo(monkeypatch, captured):
    """Serve hourly labels from start_date 00:00 to end_date 23:00, as Open-Meteo does."""

    class FakeSession:
        def mount(self, *_args, **_kwargs):
            pass

    class FakeVariable:
        def __init__(self, values):
            self.values = values

        def ValuesAsNumpy(self):
            return self.values

    class FakeHourly:
        def __init__(self, params):
            self.start = pd.Timestamp(params["start_date"])
            self.end = pd.Timestamp(params["end_date"]) + pd.Timedelta(days=1)

        def Time(self):
            return int(self.start.timestamp())

        def TimeEnd(self):
            return int(self.end.timestamp())

        def Interval(self):
            return 3600

        def Variables(self, index):
            hours = int((self.end - self.start) / pd.Timedelta(hours=1))
            return FakeVariable(np.arange(hours, dtype=float) + 1000.0 * index)

    class FakeResponse:
        def __init__(self, params):
            self.params = params

        def Hourly(self):
            return FakeHourly(self.params)

    class FakeClient:
        def __init__(self, *, session):
            captured["session"] = session

        def weather_api(self, url, *, params):
            captured["url"] = url
            captured["params"] = params
            return [FakeResponse(params)]

    monkeypatch.setattr("breos.weather.requests_cache.CachedSession", lambda *_args, **_kwargs: FakeSession())
    monkeypatch.setattr("breos.weather.openmeteo_requests.Client", FakeClient)


@pytest.mark.parametrize(
    ("radiation_time_basis", "suffix", "label_basis"),
    [("interval_mean", "", "right"), ("instant", "_instant", "instant")],
)
def test_fetch_weather_data_requests_selected_openmeteo_radiation(
    monkeypatch, radiation_time_basis, suffix, label_basis
):
    captured = {}
    _install_fake_openmeteo(monkeypatch, captured)

    weather = fetch_weather_data(
        latitude=41.1579,
        longitude=-8.6291,
        start_date="2024-06-01",
        end_date="2024-06-01",
        tilt=0,
        azimuth=0,
        save_to_file=False,
        radiation_time_basis=radiation_time_basis,
    )

    assert captured["params"]["hourly"] == [
        "temperature_2m",
        "wind_speed_10m",
        f"shortwave_radiation{suffix}",
        f"direct_radiation{suffix}",
        f"diffuse_radiation{suffix}",
        f"direct_normal_irradiance{suffix}",
        f"global_tilted_irradiance{suffix}",
        f"terrestrial_radiation{suffix}",
    ]
    assert list(weather.columns) == [
        "temperature_2m",
        "wind_speed_10m",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "terrestrial_radiation",
    ]
    metadata = weather.attrs["breos_weather_metadata"]
    assert metadata["radiation_time_basis"] == radiation_time_basis
    assert metadata["timestamp_label_basis"] == label_basis


def test_fetch_weather_data_rejects_unknown_radiation_time_basis():
    with pytest.raises(ValueError, match="radiation_time_basis"):
        fetch_weather_data(
            latitude=41.1579,
            longitude=-8.6291,
            start_date="2024-06-01",
            end_date="2024-06-01",
            tilt=0,
            azimuth=0,
            save_to_file=False,
            radiation_time_basis="unknown",
        )


@pytest.mark.parametrize(
    ("radiation_time_basis", "requested_end", "first_label", "last_label"),
    [
        ("interval_mean", "2025-01-01", "2024-01-01 01:00", "2025-01-01 00:00"),
        ("instant", "2024-12-31", "2024-01-01 00:00", "2024-12-31 23:00"),
    ],
)
def test_fetch_weather_data_covers_the_requested_hours(
    monkeypatch, radiation_time_basis, requested_end, first_label, last_label
):
    captured = {}
    _install_fake_openmeteo(monkeypatch, captured)

    weather = fetch_weather_data(
        latitude=41.1579,
        longitude=-8.6291,
        start_date="2024-01-01",
        end_date="2024-12-31",
        tilt=0,
        azimuth=0,
        save_to_file=False,
        radiation_time_basis=radiation_time_basis,
    )

    assert captured["params"]["end_date"] == requested_end
    assert len(weather) == 8784
    assert weather.index[0] == pd.Timestamp(first_label)
    assert weather.index[-1] == pd.Timestamp(last_label)


def test_fetched_right_labelled_file_keeps_its_last_year(monkeypatch, tmp_path):
    captured = {}
    _install_fake_openmeteo(monkeypatch, captured)
    fetched = fetch_weather_data(
        latitude=41.1579,
        longitude=-8.6291,
        start_date="2022-01-01",
        end_date="2024-12-31",
        tilt=0,
        azimuth=0,
        location_name="porto",
        output_dir=str(tmp_path),
    )
    path = tmp_path / "porto_historical_2022_2024_openmeteo.csv"

    by_year = preload_weather_by_year(str(path), target_year=2025)

    assert sorted(by_year) == [2022, 2023, 2024]
    last = by_year[2024]
    assert len(last) == 8760
    assert last["date"].iloc[-1] == pd.Timestamp("2025-12-31 23:00")
    # The mean over 2024-12-31 23:00-24:00 is labelled at the following midnight.
    midnight_mean = fetched.loc[pd.Timestamp("2025-01-01 00:00"), "shortwave_radiation"]
    assert last["shortwave_radiation"].iloc[-1] == midnight_mean


def _write_right_labelled_file_without_trailing_midnight(tmp_path):
    path = tmp_path / "historical.csv"
    dates = pd.date_range("2022-01-01 00:00", "2024-12-31 23:00", freq="h")
    pd.DataFrame({"date": dates, "shortwave_radiation": 0.0}).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {
        "schema_version": 1,
        "weather_sha256": digest,
        "breos_weather_metadata": {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "right"},
    }
    Path(f"{path}.metadata.json").write_text(json.dumps(sidecar))
    return path


def test_preload_weather_by_year_warns_about_the_years_it_skips(tmp_path, caplog):
    path = _write_right_labelled_file_without_trailing_midnight(tmp_path)

    with caplog.at_level("WARNING", logger="breos.weather"):
        by_year = preload_weather_by_year(str(path), target_year=2025)

    assert sorted(by_year) == [2022, 2023]
    assert "2021 (1 of 8760 rows), 2024 (8759 of 8760 rows)" in caplog.text
    assert "midnight after the last day" in caplog.text


def test_select_random_year_only_picks_complete_years(tmp_path):
    path = _write_right_labelled_file_without_trailing_midnight(tmp_path)

    picked = set()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        selected, selected_year = select_random_year_and_replace_datetime(str(path), target_year=2025, rng=rng)
        assert len(selected) == 8760
        picked.add(selected_year)

    assert picked == {2022, 2023}


def test_select_random_year_draws_from_the_given_generator(tmp_path):
    path = _write_right_labelled_file_without_trailing_midnight(tmp_path)
    np.random.seed(0)
    global_state = np.random.get_state()[1].copy()

    picks = [
        select_random_year_and_replace_datetime(str(path), rng=np.random.default_rng(seed))[1] for seed in range(8)
    ]

    assert picks == [int(np.random.default_rng(seed).choice([2022, 2023])) for seed in range(8)]
    assert set(picks) == {2022, 2023}
    np.testing.assert_array_equal(np.random.get_state()[1], global_state)


def test_battery_temperature_helper_applies_indoor_default():
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    weather = pd.DataFrame({"temp_air": [10.0, 20.0]}, index=idx)

    temps = build_battery_temperature_series("weather", index=idx, weather_df=weather)

    assert float(temps.iloc[0]) == pytest.approx(18.4)
    assert float(temps.iloc[1]) == pytest.approx(21.4)


def test_battery_temperature_helper_can_disable_indoor_model():
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    weather = pd.DataFrame({"temp_air": [10.0, 20.0]}, index=idx)

    temps = build_battery_temperature_series("weather", index=idx, weather_df=weather, indoor_model={"enabled": False})

    assert list(temps) == [10.0, 20.0]


def test_weather_filename_parser_accepts_locations_with_underscores():
    parsed = parse_weather_filename("new_york_city_historical_2020_2024_openmeteo.csv")

    assert parsed == {
        "location": "new_york_city",
        "type": "historical",
        "year_start": "2020",
        "year_end": "2024",
        "source": "openmeteo",
    }


def test_resample_to_15min_keeps_all_slots_in_last_hour():
    idx = pd.date_range("2025-01-01 00:00", periods=3, freq="h")
    weather = pd.DataFrame({"temp_air": [0.0, 4.0, 8.0]}, index=idx)

    resampled = resample_to_15min(weather, method="linear")

    assert len(resampled) == 12
    assert resampled.index[-1] == pd.Timestamp("2025-01-01 02:45")


def test_resample_to_15min_makima_holds_last_observation_through_last_hour():
    idx = pd.date_range("2025-01-01 00:00", periods=4, freq="h")
    weather = pd.DataFrame({"temp_air": [0.0, 4.0, 8.0, 12.0]}, index=idx)

    resampled = resample_to_15min(weather)

    assert len(resampled) == 16
    assert resampled.index[-1] == pd.Timestamp("2025-01-01 03:45")
    assert resampled["temp_air"].notna().all()
    assert resampled.loc["2025-01-01 03:00":, "temp_air"].tolist() == pytest.approx([12.0] * 4)


def test_resample_to_15min_can_preserve_each_hours_irradiance_energy():
    idx = pd.date_range("2025-06-21 10:00", periods=4, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            "ghi": [500.0, 700.0, 600.0, 300.0],
            "dni": [400.0, 600.0, 500.0, 200.0],
            "dhi": [250.0, 300.0, 350.0, 200.0],
            "temp_air": [20.0, 21.0, 22.0, 21.0],
        },
        index=idx,
    )

    resampled = resample_to_15min(
        weather,
        method="linear",
        latitude=41.1579,
        longitude=-8.6291,
        preserve_irradiance_energy=True,
    )

    for column in ("ghi", "dni", "dhi"):
        hourly_means = resampled[column].to_numpy().reshape(len(weather), 4).mean(axis=1)
        assert hourly_means == pytest.approx(weather[column].to_numpy())


def test_clear_sky_resampling_does_not_attenuate_values_at_source_timestamps():
    idx = pd.date_range("2025-06-21 10:00", periods=4, freq="h", tz="UTC")
    weather = pd.DataFrame({"ghi": [50.0, 100.0, 75.0, 25.0]}, index=idx)

    resampled = resample_to_15min(weather, latitude=41.1579, longitude=-8.6291)

    assert resampled.loc[idx, "ghi"].to_numpy() == pytest.approx(weather["ghi"].to_numpy())


def _three_day_hourly_weather(unit: str) -> pd.DataFrame:
    idx = pd.date_range("2025-06-20", periods=72, freq="h", tz="UTC").as_unit(unit)
    hour = np.arange(72) % 24
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0.0, None)
    return pd.DataFrame(
        {
            "ghi": 800.0 * daylight,
            "dni": 600.0 * daylight,
            "dhi": 200.0 * daylight,
            "temp_air": 18.0 + 6.0 * np.sin((hour - 9) / 24 * 2 * np.pi),
            "wind_speed": 2.0 + hour / 12.0,
        },
        index=idx,
    )


_TMY_METADATA = {"inputs": {"location": {"latitude": 41.1579, "longitude": -8.6291, "elevation": 100.0}}}
_RESAMPLERS = {
    "clear_sky_makima": lambda df: resample_to_15min(df, latitude=41.1579, longitude=-8.6291),
    "clear_sky_linear": lambda df: resample_to_15min(df, method="linear", latitude=41.1579, longitude=-8.6291),
    "direct_makima": lambda df: resample_to_15min(df),
    "tmy": lambda df: resample_tmy_to_15min(df, _TMY_METADATA),
}


@pytest.mark.parametrize("resampler", sorted(_RESAMPLERS))
@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_resamplers_do_not_depend_on_the_index_resolution(resampler, unit):
    """pandas 3 parses timestamps as microseconds, pandas 2 as nanoseconds.

    Both resamplers used to divide the raw integers by 10**9, which jittered
    the microsecond abscissa and made second-resolution input fail outright.
    """
    reference = _RESAMPLERS[resampler](_three_day_hourly_weather("ns"))
    resampled = _RESAMPLERS[resampler](_three_day_hourly_weather(unit))

    assert resampled.index.equals(reference.index)
    assert resampled.columns.tolist() == reference.columns.tolist()
    np.testing.assert_array_equal(resampled.to_numpy(), reference.to_numpy())


def test_tmy_resampling_keeps_the_weather_columns_and_bounds_humidity():
    weather = _three_day_hourly_weather("ns")
    weather["relative_humidity"] = np.linspace(40.0, 140.0, len(weather))
    weather["pressure"] = 101325.0
    weather.attrs["breos_weather_metadata"] = {"source": "PVGIS_TMY", "radiation_time_basis": "instant"}

    resampled = resample_tmy_to_15min(weather, _TMY_METADATA)

    assert resampled.columns.tolist() == ["ghi", "dni", "dhi", "temp_air", "relative_humidity", "wind_speed"]
    assert resampled["relative_humidity"].max() == 100.0
    assert resampled.attrs["breos_weather_metadata"]["irradiance_resampling_method"] == "makima_clear_sky"


def test_fetch_tmy_weather_accepts_hourly_frequency_alias_and_uses_horizon_by_default(monkeypatch):
    tmy = pd.DataFrame({"ghi": [0.0]}, index=pd.date_range("2020-01-01 00:00", periods=1, freq="h"))
    captured = {}

    def fake_get_pvgis_tmy(*args, **kwargs):
        captured.update(kwargs)
        return tmy.copy(), {}

    monkeypatch.setattr("breos.weather.pvlib.iotools.get_pvgis_tmy", fake_get_pvgis_tmy)

    weather, _metadata = fetch_tmy_weather_data(41.0, -8.0, sample_year=None, freq="H")

    assert len(weather) == 1
    assert captured["usehorizon"] is True
    assert weather.attrs["breos_weather_metadata"]["source"] == "PVGIS_TMY"
    assert weather.attrs["breos_weather_metadata"]["horizon"] == {
        "status": "applied",
        "provider": "pvgis",
        "profile": "provider_default",
    }


def test_fetch_tmy_can_request_unshaded_pvgis_weather(monkeypatch):
    tmy = pd.DataFrame({"ghi": [0.0]}, index=pd.date_range("2020-01-01 00:00", periods=1, freq="h"))
    captured = {}

    def fake_get_pvgis_tmy(*args, **kwargs):
        captured.update(kwargs)
        return tmy.copy(), {}

    monkeypatch.setattr("breos.weather.pvlib.iotools.get_pvgis_tmy", fake_get_pvgis_tmy)

    weather, _metadata = fetch_tmy_weather_data(41.0, -8.0, sample_year=None, use_horizon=False)

    assert captured["usehorizon"] is False
    assert weather.attrs["breos_weather_metadata"]["horizon"] == {
        "status": "not_applied",
        "provider": "pvgis",
        "profile": None,
    }


def test_local_weather_records_path_and_hash(tmp_path):
    path = tmp_path / "porto_tmy_2005_2023_pvgis-sarah3.csv"
    pd.DataFrame(
        {"ghi": [0.0, 1.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
    ).to_csv(path)

    weather = load_weather("porto", data_type="tmy", weather_dir=str(tmp_path))

    metadata = weather.attrs["breos_weather_metadata"]
    assert metadata["source"] == "local_file"
    assert metadata["path"] == str(path.resolve())
    assert len(metadata["sha256"]) == 64
    assert metadata["horizon"] == {"status": "unknown", "provider": None, "profile": None}


def test_saved_pvgis_weather_round_trips_horizon_metadata(monkeypatch, tmp_path):
    tmy = pd.DataFrame(
        {"ghi": [0.0, 1.0]},
        index=pd.date_range("2020-01-01 00:00", periods=2, freq="h", tz="UTC"),
    )
    api_metadata = {
        "inputs": {
            "location": {"latitude": 41.0, "longitude": -8.0},
            "meteo_data": {"radiation_db": "SARAH3", "year_min": 2005, "year_max": 2023},
        }
    }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "breos.weather.pvlib.iotools.get_pvgis_tmy",
        lambda *args, **kwargs: (tmy.copy(), api_metadata),
    )

    fetch_tmy_weather_data(41.0, -8.0, sample_year=None, save_to_file=True, use_horizon=False)

    csv_path = tmp_path / "weather" / "lat41_lon-8_tmy_2005_2023_pvgis-sarah3.csv"
    sidecar_path = csv_path.with_name(f"{csv_path.name}.metadata.json")
    payload = json.loads(sidecar_path.read_text())
    assert payload["schema_version"] == 1
    assert len(payload["weather_sha256"]) == 64

    loaded = load_weather("lat41_lon-8", data_type="tmy", weather_dir=str(csv_path.parent))
    metadata = loaded.attrs["breos_weather_metadata"]
    assert metadata["source"] == "local_file"
    assert metadata["upstream_source"] == "PVGIS_TMY"
    assert metadata["api_metadata"] == api_metadata
    assert metadata["horizon"] == {"status": "not_applied", "provider": "pvgis", "profile": None}
    assert metadata["metadata_sidecar"] == str(sidecar_path)


def test_stale_weather_sidecar_is_ignored(monkeypatch, tmp_path, caplog):
    tmy = pd.DataFrame(
        {"ghi": [0.0, 1.0]},
        index=pd.date_range("2020-01-01 00:00", periods=2, freq="h", tz="UTC"),
    )
    api_metadata = {
        "inputs": {
            "location": {"latitude": 41.0, "longitude": -8.0},
            "meteo_data": {"radiation_db": "SARAH3", "year_min": 2005, "year_max": 2023},
        }
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "breos.weather.pvlib.iotools.get_pvgis_tmy",
        lambda *args, **kwargs: (tmy.copy(), api_metadata),
    )
    fetch_tmy_weather_data(41.0, -8.0, sample_year=None, save_to_file=True)
    csv_path = tmp_path / "weather" / "lat41_lon-8_tmy_2005_2023_pvgis-sarah3.csv"
    csv_path.write_text(csv_path.read_text() + "\n")

    loaded = load_weather("lat41_lon-8", data_type="tmy", weather_dir=str(csv_path.parent))

    assert loaded.attrs["breos_weather_metadata"]["horizon"]["status"] == "unknown"
    assert "digest does not match" in caplog.text


def test_resample_to_15min_preserves_weather_metadata():
    idx = pd.date_range("2025-01-01 00:00", periods=3, freq="h", tz="UTC")
    weather = pd.DataFrame({"temp_air": [0.0, 4.0, 8.0]}, index=idx)
    weather.attrs["breos_weather_metadata"] = {
        "source": "test",
        "horizon": {"status": "not_applied", "provider": "test", "profile": None},
    }

    resampled = resample_to_15min(weather, method="linear")

    metadata = resampled.attrs["breos_weather_metadata"]
    assert metadata["source"] == "test"
    assert metadata["horizon"] == weather.attrs["breos_weather_metadata"]["horizon"]
    assert metadata["input_resolution"] == "h"
    assert metadata["output_resolution"] == "15min"
    assert metadata["irradiance_resampling_method"] == "linear"
    assert metadata is not weather.attrs["breos_weather_metadata"]


def test_resample_tmy_to_15min_preserves_weather_metadata():
    idx = pd.date_range("2025-01-01 00:00", periods=4, freq="h", tz="UTC")
    weather = pd.DataFrame({"temp_air": [0.0, 4.0, 8.0, 12.0]}, index=idx)
    weather.attrs["breos_weather_metadata"] = {
        "source": "test",
        "horizon": {"status": "applied", "provider": "test", "profile": "test"},
    }
    api_metadata = {"inputs": {"location": {"latitude": 41.0, "longitude": -8.0, "elevation": 0.0}}}

    resampled = resample_tmy_to_15min(weather, api_metadata)

    metadata = resampled.attrs["breos_weather_metadata"]
    assert metadata["source"] == "test"
    assert metadata["horizon"] == weather.attrs["breos_weather_metadata"]["horizon"]
    assert metadata["input_resolution"] == "h"
    assert metadata["output_resolution"] == "15min"
    assert metadata["irradiance_resampling_method"] == "makima_clear_sky"
    assert metadata is not weather.attrs["breos_weather_metadata"]


def test_fetch_tmy_keeps_utc_instants_for_non_utc_location(monkeypatch):
    # PVGIS serves UTC-ordered rows; synthetic GHI peaks at 11:00 UTC
    # (solar noon near Berlin's longitude). The fetch must roll the data
    # to local midnight without breaking each row's UTC instant — the old
    # relabeling bug shifted irradiance against solar position by the
    # location's full UTC offset.
    from pvlib.iotools.pvgis import _coerce_and_roll_tmy

    utc_idx = pd.date_range("1990-01-01", periods=8760, freq="h", tz="UTC")
    ghi = np.where(
        utc_idx.hour == 11,
        800.0,
        np.where(np.abs(utc_idx.hour - 11) <= 3, 300.0, 0.0),
    )
    raw = pd.DataFrame({"ghi": ghi, "temp_air": 10.0}, index=utc_idx)

    def fake_get_pvgis_tmy(latitude, longitude, *args, roll_utc_offset=None, coerce_year=1990, **kwargs):
        data = raw.copy()
        if not (roll_utc_offset is None and coerce_year is None):
            data = _coerce_and_roll_tmy(data, roll_utc_offset, coerce_year or 1990)
        return data, {"inputs": {}}

    monkeypatch.setattr("breos.weather.pvlib.iotools.get_pvgis_tmy", fake_get_pvgis_tmy)

    tmy_data, _ = fetch_tmy_weather_data(
        latitude=52.52,
        longitude=13.405,
        sample_year=2025,
        timezone="Europe/Berlin",
    )

    assert len(tmy_data) == 8760
    # Series starts at local midnight of the sample year (UTC+1 standard time)
    assert tmy_data.index[0] == pd.Timestamp("2025-01-01 00:00", tz="Etc/GMT-1")
    # Parsed back as UTC instants (as the pipeline does), the GHI peak must
    # stay at 11:00 UTC; the relabeling bug moved it to 10:00 UTC.
    utc_hours = tmy_data.index.tz_convert("UTC").hour
    peak_utc_hour = tmy_data.groupby(utc_hours)["ghi"].mean().idxmax()
    assert peak_utc_hour == 11
    # On the local clock the mean-GHI peak lands at midday, not the UTC peak.
    peak_local_hour = tmy_data.groupby(tmy_data.index.hour)["ghi"].mean().idxmax()
    assert 11 <= peak_local_hour <= 15


def test_fetch_tmy_without_sample_year_keeps_original_index(monkeypatch):
    tmy = pd.DataFrame({"ghi": [0.0]}, index=pd.date_range("2009-01-01 00:00", periods=1, freq="h", tz="UTC"))
    captured = {}

    def fake_get_pvgis_tmy(*args, **kwargs):
        captured.update(kwargs)
        return tmy.copy(), {}

    monkeypatch.setattr("breos.weather.pvlib.iotools.get_pvgis_tmy", fake_get_pvgis_tmy)

    weather, _metadata = fetch_tmy_weather_data(41.0, -8.0, sample_year=None)

    assert captured["roll_utc_offset"] is None
    assert captured["coerce_year"] is None
    assert weather.index[0] == tmy.index[0]


def test_fetch_tmy_rejects_fractional_hour_timezone_before_request(monkeypatch):
    requested = False

    def fake_get_pvgis_tmy(*args, **kwargs):
        nonlocal requested
        requested = True

    monkeypatch.setattr("breos.weather.pvlib.iotools.get_pvgis_tmy", fake_get_pvgis_tmy)

    with pytest.raises(ValueError, match="fractional-hour timezone"):
        fetch_tmy_weather_data(22.57, 88.36, sample_year=2025, timezone="Asia/Kolkata")

    assert requested is False


def test_read_epw_accepts_15t_frequency_alias(monkeypatch):
    epw = pd.DataFrame(
        {
            "ghi": [0.0, 10.0],
            "dni": [0.0, 5.0],
            "dhi": [0.0, 5.0],
            "temp_air": [12.0, 13.0],
            "wind_speed": [1.0, 1.5],
        },
        index=pd.date_range("2025-01-01 00:00", periods=2, freq="h"),
    )
    calls = {}

    def fake_read_epw(_filepath):
        return epw.copy(), {"latitude": 41.0, "longitude": -8.0}

    def fake_resample(df, method="makima", latitude=None, longitude=None, **_kwargs):
        calls["method"] = method
        calls["latitude"] = latitude
        calls["longitude"] = longitude
        return df

    monkeypatch.setattr("breos.weather.pvlib.iotools.read_epw", fake_read_epw)
    monkeypatch.setattr("breos.weather.resample_to_15min", fake_resample)

    weather = read_epw_file("dummy.epw", freq="15T")

    assert calls == {"method": "makima", "latitude": 41.0, "longitude": -8.0}
    assert weather.attrs["breos_weather_metadata"]["horizon"] == {
        "status": "unknown",
        "provider": "epw",
        "profile": None,
    }


def test_select_random_year_accepts_15min_leap_year_after_dropping_feb_29(tmp_path):
    weather_path, source = _write_leap_year_15min_weather(tmp_path)

    selected, selected_year = select_random_year_and_replace_datetime(str(weather_path), target_year=2025)

    dates = pd.to_datetime(selected["date"])
    source_march_1 = source.loc[source["date"] == pd.Timestamp("2024-03-01 00:00"), "temp_air"].item()
    mapped_march_1 = selected.loc[dates == pd.Timestamp("2025-03-01 00:00"), "temp_air"].item()

    assert selected_year == 2024
    assert len(selected) == 35040
    assert not ((dates.dt.month == 2) & (dates.dt.day == 29)).any()
    assert dates.iloc[0] == pd.Timestamp("2025-01-01 00:00")
    assert dates.iloc[-1] == pd.Timestamp("2025-12-31 23:45")
    assert mapped_march_1 == source_march_1


def test_preload_weather_by_year_accepts_15min_leap_year_after_dropping_feb_29(tmp_path):
    weather_path, source = _write_leap_year_15min_weather(tmp_path)

    by_year = preload_weather_by_year(str(weather_path), target_year=2025)
    selected = by_year[2024]

    dates = pd.to_datetime(selected["date"])
    source_march_1 = source.loc[source["date"] == pd.Timestamp("2024-03-01 00:00"), "temp_air"].item()
    mapped_march_1 = selected.loc[dates == pd.Timestamp("2025-03-01 00:00"), "temp_air"].item()

    assert len(selected) == 35040
    assert not ((dates.dt.month == 2) & (dates.dt.day == 29)).any()
    assert dates.iloc[0] == pd.Timestamp("2025-01-01 00:00")
    assert dates.iloc[-1] == pd.Timestamp("2025-12-31 23:45")
    assert mapped_march_1 == source_march_1


def test_save_weather_csv_writes_content_bound_metadata(tmp_path):
    weather = _timed_weather(basis="interval_mean", label="right")
    path = tmp_path / "weather.csv"

    save_weather_csv(weather, path)

    payload = json.loads(Path(f"{path}.metadata.json").read_text())
    assert payload["weather_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["breos_weather_metadata"]["radiation_time_basis"] == "interval_mean"
