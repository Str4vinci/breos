"""App selection among cached TMY weather files through ``weather_source``."""

from pathlib import Path

import pytest

from breos import cli
from breos.app import App

BASE = {"location": "porto", "n_modules": 6, "annual_consumption_kwh": 3000, "projection_years": 1}
SARAH3_FILE = "porto_tmy_2005_2023_pvgis-sarah3.csv"
NSRDB_FILE = "porto_tmy_2014_nsrdb.csv"


@pytest.fixture
def no_fetch(monkeypatch):
    """Fail if App falls back to a PVGIS fetch; these tests are offline."""

    def _fail(*args, **kwargs):
        raise AssertionError("App fetched PVGIS weather instead of using the local cache")

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _fail)


@pytest.fixture
def weather_dir(tmp_path, monkeypatch, synthetic_weather, no_fetch):
    """A working directory whose weather/ cache holds one Porto TMY file."""
    directory = tmp_path / "weather"
    directory.mkdir()
    synthetic_weather.to_csv(directory / SARAH3_FILE)
    monkeypatch.chdir(tmp_path)
    return directory


def _add_dimmer_nsrdb_file(directory, synthetic_weather):
    dimmer = synthetic_weather.copy()
    dimmer[["ghi", "dni", "dhi"]] *= 0.5
    dimmer.to_csv(directory / NSRDB_FILE)


def _run(config):
    app = App(config)
    app.simulate()
    return app.result()


def test_single_cached_tmy_file_is_used_without_weather_source(weather_dir):
    result = _run(BASE)

    weather = result["provenance"]["weather"]
    assert weather["source"] == "local_file"
    assert weather["path"] == str((weather_dir / SARAH3_FILE).resolve())
    assert weather["parsed_filename"]["source"] == "pvgis-sarah3"
    assert result["provenance"]["resolved_config"]["weather_source"] is None


def test_two_cached_tmy_files_raise_an_error_naming_the_config_key(weather_dir, synthetic_weather):
    _add_dimmer_nsrdb_file(weather_dir, synthetic_weather)

    with pytest.raises(ValueError, match="Several cached TMY weather files") as excinfo:
        _run(BASE)

    message = str(excinfo.value)
    assert "'weather_source'" in message
    assert "--weather-source" in message
    assert f"{SARAH3_FILE}, {NSRDB_FILE}" in message
    assert "nsrdb, pvgis-sarah3" in message


@pytest.mark.parametrize(
    ("source", "filename"),
    [("pvgis-sarah3", SARAH3_FILE), ("nsrdb", NSRDB_FILE)],
)
def test_weather_source_selects_the_matching_cached_file(weather_dir, synthetic_weather, source, filename):
    _add_dimmer_nsrdb_file(weather_dir, synthetic_weather)

    result = _run({**BASE, "weather_source": source})

    weather = result["provenance"]["weather"]
    assert weather["path"] == str((weather_dir / filename).resolve())
    assert weather["parsed_filename"]["source"] == source
    assert result["provenance"]["resolved_config"]["weather_source"] == source


def test_weather_source_changes_the_simulated_production(weather_dir, synthetic_weather):
    _add_dimmer_nsrdb_file(weather_dir, synthetic_weather)

    sarah3 = _run({**BASE, "weather_source": "pvgis-sarah3"})
    nsrdb = _run({**BASE, "weather_source": "nsrdb"})

    assert nsrdb["usable_ac_system_production_kwh"] < sarah3["usable_ac_system_production_kwh"]


def test_weather_source_without_a_matching_file_does_not_fetch(weather_dir):
    with pytest.raises(FileNotFoundError, match="porto_tmy_<years>_nsrdb.csv"):
        _run({**BASE, "weather_source": "nsrdb"})


@pytest.mark.parametrize("value", ["", "pvgis sarah3", "../nsrdb", 3, ["nsrdb"]])
def test_invalid_weather_source_is_rejected_at_construction(value):
    with pytest.raises(ValueError, match="'weather_source' must be"):
        App({**BASE, "weather_source": value})


def test_weather_source_is_rejected_for_coordinate_locations():
    location = {"latitude": 41.15, "longitude": -8.63, "timezone": "Europe/Lisbon"}
    with pytest.raises(ValueError, match="coordinate-dict locations"):
        App({**BASE, "location": location, "weather_source": "nsrdb"})


def test_cli_weather_source_flag_reaches_the_config():
    args = cli.build_parser().parse_args(["run", "--location", "porto", "--weather-source", "nsrdb"])

    assert cli._build_config(args)["weather_source"] == "nsrdb"


def test_a_gzip_compressed_cached_tmy_file_is_used(tmp_path, monkeypatch, no_fetch):
    """The committed Porto TMY is a .csv.gz; App reads it from weather/ without decompressing it first."""
    committed = Path(__file__).resolve().parents[1] / "validation" / "data" / "weather" / f"{SARAH3_FILE}.gz"
    directory = tmp_path / "weather"
    directory.mkdir()
    target = directory / committed.name
    target.write_bytes(committed.read_bytes())
    Path(f"{target}.metadata.json").write_bytes(Path(f"{committed}.metadata.json").read_bytes())
    monkeypatch.chdir(tmp_path)

    result = _run({**BASE, "weather_source": "pvgis-sarah3"})

    weather = result["provenance"]["weather"]
    assert weather["source"] == "local_file"
    assert weather["path"] == str(target.resolve())
    assert weather["upstream_source"] == "PVGIS_TMY"
    assert weather["parsed_filename"]["source"] == "pvgis-sarah3"
    assert result["usable_ac_system_production_kwh"] > 0
