"""App weather named directly through ``weather_file`` (#394)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breos import cli
from breos.app import App
from breos.app_inputs import input_configuration_key, resample_hourly_weather
from breos.montecarlo import MonteCarloSettings, run_montecarlo

# The weather CSVs these tests write carry no metadata sidecar, so the sun
# position is pinned at the step labels rather than left to "auto".
BASE = {
    "location": "porto",
    "n_modules": 6,
    "annual_consumption_kwh": 3000,
    "projection_years": 1,
    "solar_position": "interval-start",
}
COMMITTED_TMY = (
    Path(__file__).resolve().parents[1] / "validation" / "data" / "weather" / "porto_tmy_2005_2023_pvgis-sarah3.csv.gz"
)


@pytest.fixture
def inputs(tmp_path, monkeypatch, synthetic_weather, no_fetch):
    """A working directory with a weather CSV under inputs/, named outside the cache convention."""
    directory = tmp_path / "inputs"
    directory.mkdir()
    synthetic_weather.to_csv(directory / "site_weather.csv")
    monkeypatch.chdir(tmp_path)
    return directory


def _run(config):
    app = App(config)
    app.simulate()
    return app.result()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_weather_file_is_read_and_recorded_with_its_digest(inputs):
    result = _run({**BASE, "weather_file": "inputs/site_weather.csv"})

    weather = result["provenance"]["weather"]
    path = inputs / "site_weather.csv"
    assert weather["source"] == "local_file"
    assert weather["path"] == str(path.resolve())
    assert weather["sha256"] == _sha256(path)
    assert "parsed_filename" not in weather
    # The path as given; the absolute path is in provenance.weather.
    assert result["provenance"]["resolved_config"]["weather_file"] == "inputs/site_weather.csv"
    assert result["usable_ac_system_production_kwh"] > 0


def test_weather_file_gives_the_same_result_as_the_same_file_in_the_cache(inputs, synthetic_weather):
    cache = inputs.parent / "weather"
    cache.mkdir()
    synthetic_weather.to_csv(cache / "porto_tmy_2005_2023_pvgis-sarah3.csv")

    cached = App(BASE)
    cached.simulate()
    named = App({**BASE, "weather_file": str(inputs / "site_weather.csv")})
    named.simulate()

    pd.testing.assert_frame_equal(named.timeseries(), cached.timeseries(), check_exact=True)
    without_provenance = {key: value for key, value in cached.result().items() if key != "provenance"}
    assert {key: value for key, value in named.result().items() if key != "provenance"} == without_provenance


def test_weather_file_wins_over_the_cache(inputs, synthetic_weather):
    cache = inputs.parent / "weather"
    cache.mkdir()
    dimmer = synthetic_weather.copy()
    dimmer[["ghi", "dni", "dhi"]] *= 0.5
    dimmer.to_csv(cache / "porto_tmy_2005_2023_pvgis-sarah3.csv")

    cached = _run(BASE)
    named = _run({**BASE, "weather_file": "inputs/site_weather.csv"})

    assert named["provenance"]["weather"]["path"] == str((inputs / "site_weather.csv").resolve())
    assert named["usable_ac_system_production_kwh"] > cached["usable_ac_system_production_kwh"]


def test_weather_file_serves_a_coordinate_location(inputs):
    location = {"latitude": 41.15, "longitude": -8.63, "timezone": "Europe/Lisbon"}

    result = _run({**BASE, "location": location, "weather_file": "inputs/site_weather.csv"})

    assert result["provenance"]["weather"]["path"] == str((inputs / "site_weather.csv").resolve())


def test_a_compressed_weather_file_keeps_its_sidecar_metadata(tmp_path, monkeypatch, no_fetch):
    target = tmp_path / "porto.csv.gz"
    target.write_bytes(COMMITTED_TMY.read_bytes())
    Path(f"{target}.metadata.json").write_bytes(Path(f"{COMMITTED_TMY}.metadata.json").read_bytes())
    monkeypatch.chdir(tmp_path)

    result = _run({**BASE, "weather_file": Path("porto.csv.gz")})

    weather = result["provenance"]["weather"]
    assert weather["path"] == str(target.resolve())
    assert weather["sha256"] == _sha256(COMMITTED_TMY)
    assert weather["upstream_source"] == "PVGIS_TMY"
    assert weather["metadata_sidecar"] == f"{target.resolve()}.metadata.json"
    assert result["provenance"]["resolved_config"]["weather_file"] == "porto.csv.gz"


def test_a_missing_weather_file_is_rejected_at_construction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="weather_file not found: missing.csv"):
        App({**BASE, "weather_file": "missing.csv"})


def test_weather_file_and_weather_source_are_alternatives(inputs):
    with pytest.raises(ValueError, match="set one of them"):
        App({**BASE, "weather_file": "inputs/site_weather.csv", "weather_source": "nsrdb"})


def test_a_multi_year_weather_file_is_refused(tmp_path, monkeypatch, synthetic_weather, no_fetch):
    next_year = synthetic_weather.copy()
    next_year.index = next_year.index + pd.Timedelta(days=365)
    pd.concat([synthetic_weather, next_year]).to_csv(tmp_path / "two_years.csv")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="more than one year"):
        _run({**BASE, "weather_file": "two_years.csv"})


def test_cli_weather_file_flag_replaces_a_config_files_weather_source(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('location = "porto"\nweather_source = "nsrdb"\n')
    args = cli.build_parser().parse_args(["run", "--config", str(config), "--weather-file", "inputs/tmy.csv"])

    built = cli._build_config(args)

    assert built["weather_file"] == "inputs/tmy.csv"
    assert "weather_source" not in built


def test_two_weather_files_are_two_input_configurations(inputs, synthetic_weather):
    synthetic_weather.to_csv(inputs / "other.csv")

    first = input_configuration_key({**BASE, "weather_file": "inputs/site_weather.csv"})
    second = input_configuration_key({**BASE, "weather_file": "inputs/other.csv"})

    assert first != second


def test_revalue_refuses_a_new_weather_file(inputs, synthetic_weather):
    synthetic_weather.to_csv(inputs / "other.csv")
    app = App({**BASE, "weather_file": "inputs/site_weather.csv"})
    app.simulate()

    with pytest.raises(ValueError, match="weather_file is not a price key"):
        app.revalue({"weather_file": "inputs/other.csv"})


def test_monte_carlo_refuses_the_app_weather_file(inputs, write_multiyear_weather):
    historical = write_multiyear_weather(inputs / "historical.csv")
    settings = MonteCarloSettings(weather_file=str(historical), n_runs=1, years_per_run=1, target_year=2023)

    with pytest.raises(ValueError, match=r"\[montecarlo\]\.weather_file"):
        run_montecarlo({**BASE, "weather_file": "inputs/site_weather.csv"}, settings)


@pytest.mark.parametrize("resolution", ["h", "15min"])
@pytest.mark.parametrize(("step", "hours"), [("2h", 2), ("3h", 3), ("D", 24)])
def test_weather_coarser_than_hourly_is_refused(
    tmp_path, monkeypatch, synthetic_weather, no_fetch, resolution, step, hours
):
    # Two-hourly rows were read as hourly at 15 minutes, stretching the year.
    synthetic_weather.resample(step).mean().to_csv(tmp_path / "coarse.csv")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=f"steps of {hours} hours"):
        _run({**BASE, "resolution": resolution, "weather_file": "coarse.csv"})


def test_an_hourly_file_missing_its_second_row_is_not_read_as_two_hourly(
    tmp_path, monkeypatch, synthetic_weather, no_fetch
):
    synthetic_weather.drop(synthetic_weather.index[1]).to_csv(tmp_path / "gap.csv")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must step evenly: row 1 is"):
        _run({**BASE, "resolution": "15min", "weather_file": "gap.csv"})


@pytest.mark.parametrize("resolution", ["h", "15min"])
def test_an_hourly_file_missing_a_row_is_refused_at_every_resolution(
    tmp_path, monkeypatch, synthetic_weather, no_fetch, resolution
):
    # The 15-minute resampler filled the missing hour without a message.
    synthetic_weather.drop(synthetic_weather.index[2000]).to_csv(tmp_path / "gap.csv")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="step evenly"):
        _run({**BASE, "resolution": resolution, "weather_file": "gap.csv"})


def _hourly(n=48):
    index = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"ghi": np.arange(n, dtype=float)}, index=index)


def _check(weather, freq="15min"):
    return resample_hourly_weather(weather, freq, latitude=41.0, longitude=-8.0, resample=lambda w, **_: w.iloc[:0])


def test_hourly_weather_out_of_order_is_sorted_before_its_step_is_checked():
    weather = _hourly()
    swapped = weather.iloc[[0, 1, 3, 2, *range(4, len(weather))]]

    assert _check(swapped).empty


def test_one_stray_row_does_not_hide_a_two_hourly_step():
    weather = _hourly(96).iloc[::2]
    stray = _hourly(96).iloc[[10]].set_axis(weather.index[[5]] + pd.Timedelta("30min"))
    mixed = pd.concat([weather, stray]).sort_index()

    for freq in ("h", "15min"):
        with pytest.raises(ValueError, match="steps of 2 hours"):
            _check(mixed, freq)


def test_a_repeated_hourly_timestamp_is_named():
    weather = _hourly()
    repeated = pd.concat([weather, weather.iloc[[7]]]).sort_index()

    with pytest.raises(ValueError, match="repeats the timestamp"):
        _check(repeated)


def test_weather_without_a_datetime_index_is_returned_unchanged():
    weather = pd.DataFrame({"ghi": [1.0, 2.0]}, index=pd.Index(["metadata", 1], dtype=object))

    assert _check(weather) is weather
