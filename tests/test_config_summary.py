"""The resolved-config summary is built from the config registry (#181)."""

import json

import pytest

from breos import cli
from breos.app_config import APP_CONFIG_FIELDS

BASE = {"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000}
# Runner sections and the removed legacy selector are not App settings.
NOT_SUMMARISED = {"montecarlo", "sweep", "battery_type"}


def test_every_app_key_has_one_place_in_the_summary():
    places = {name: field.summary for name, field in APP_CONFIG_FIELDS.items() if field.summary is not None}
    assert set(APP_CONFIG_FIELDS) - set(places) == NOT_SUMMARISED
    assert len(set(places.values())) == len(places)
    for place in places.values():
        section, _, key = place.partition(".")
        assert section in cli._SUMMARY_SECTIONS and key and "." not in key


def test_summary_reports_each_key_at_its_place():
    summary = cli._resolved_config_summary(BASE)
    for name, field in APP_CONFIG_FIELDS.items():
        if field.summary is not None:
            section, key = field.summary.split(".")
            assert key in summary[section], name


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("enable_resistance_fade", True),
        ("battery_temperature", 25.0),
        ("calendar_model", "naumann_lam_field_calibrated_v2"),
        ("horizon_profile", [[0.0, 5.0], [180.0, 10.0]]),
        ("execution_backend", "python"),
        ("costs", {"electricity_cost": 0.25}),
    ],
)
def test_keys_the_hand_built_summary_left_out_are_reported(name, value):
    summary = cli._resolved_config_summary({**BASE, name: value})
    section, key = APP_CONFIG_FIELDS[name].summary.split(".")
    assert summary[section][key] == value


def test_summary_reports_what_the_resolver_derives():
    config = {
        "location": {"latitude": 52.5, "longitude": 13.4, "timezone": "Europe/Berlin"},
        "annual_consumption_kwh": 3500,
        "tracking": "single_axis",
        "battery_kwh": 5.0,
        "pv_arrays": [{"modules": 4, "azimuth": 90}, {"modules": 3, "azimuth": 270}],
    }
    summary = cli._resolved_config_summary(config)
    resolved = cli.resolve_app_config(config)
    assert summary["location"] == {
        "key": None,
        "latitude": 52.5,
        "longitude": 13.4,
        "timezone": "Europe/Berlin",
    }
    assert summary["pv"]["n_modules"] == 7
    assert summary["pv"]["tilt"] == resolved.tilt
    assert summary["pv"]["azimuth"] == resolved.azimuth
    assert summary["pv"]["axis_azimuth"] == resolved.axis_azimuth == 180.0
    assert [array["azimuth"] for array in summary["pv"]["arrays"]] == [90.0, 270.0]
    assert summary["pv"]["module"] == resolved.pv_module_key
    assert summary["inverter"]["ac_rating_kw"] == resolved.inverter_ac_capacity_w / 1000
    assert summary["battery"]["round_trip_efficiency"] == 0.95


def test_summary_writes_a_toml_date_as_text(tmp_path, capsys):
    config_path = tmp_path / "tariff.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
resolution = "15min"

[tariff]
schedule = "pt_mainland_2027_daily_tri"
currency = "EUR"
import_prices = { all = 0.20 }
export_prices = { all = 0.05 }
study_date = 2027-07-01
""".strip(),
        encoding="utf-8",
    )

    assert cli.main(["validate-config", str(config_path), "--json"]) == 0
    tariff = json.loads(capsys.readouterr().out)["economics"]["tariff"]
    assert tariff["study_date"] == "2027-07-01"
    assert tariff["schedule"] == "pt_mainland_2027_daily_tri"
