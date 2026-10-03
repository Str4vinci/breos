"""App weather named directly through ``weather_file`` (#394)."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from breos import cli
from breos.app import App
from breos.app_inputs import input_configuration_key
from breos.montecarlo import MonteCarloSettings, run_montecarlo

BASE = {"location": "porto", "n_modules": 6, "annual_consumption_kwh": 3000, "projection_years": 1}
COMMITTED_TMY = (
    Path(__file__).resolve().parents[1] / "validation" / "data" / "weather" / "porto_tmy_2005_2023_pvgis-sarah3.csv.gz"
)


@pytest.fixture
def no_fetch(monkeypatch):
    """Fail if App falls back to a PVGIS fetch; these tests are offline."""

    def _fail(*args, **kwargs):
        raise AssertionError("App fetched PVGIS weather instead of reading weather_file")

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _fail)


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
