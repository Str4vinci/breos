"""Offline checks for PVWatts reference provenance and trust."""

import copy
import json
from unittest.mock import Mock

import pytest

from validation import compare, fetch_references

INPUTS = {
    "lat": "41.1579",
    "lon": "-8.6291",
    "system_capacity": "4",
    "module_type": "0",
    "array_type": "0",
    "tilt": "35",
    "azimuth": "180",
    "losses": "14.08",
    "dc_ac_ratio": "1.2",
    "inv_eff": "96",
    "albedo": "0.2",
    "radius": "0",
    "dataset": "nsrdb",
}


@pytest.fixture
def pvwatts_response():
    return {
        "inputs": dict(INPUTS),
        "errors": [],
        "warnings": [],
        "version": "8.5.0",
        "ssc_info": {"version": 280},
        "station_info": {
            "lat": 41.17,
            "lon": -8.62,
            "elev": 122.0,
            "location": "112652",
            "distance": 1551,
            "weather_data_source": "NSRDB PSM V4 Meteosat Prime Meridian tmy-2022 1.0.0",
        },
        "outputs": {"ac_monthly": [100.0] * 12, "ac_annual": 5620.92, "solrad_annual": 4.1234},
    }


@pytest.fixture
def fetch(monkeypatch, pvwatts_response):
    monkeypatch.setenv("NREL_API_KEY", "offline-test-secret")
    monkeypatch.setattr(fetch_references, "_today", lambda: "2026-10-22")
    get = Mock(return_value=Mock(json=lambda: copy.deepcopy(pvwatts_response)))
    monkeypatch.setattr(fetch_references.requests, "get", get)
    system, locations = fetch_references.load_spec()

    def run():
        result = fetch_references.fetch_pvwatts(locations["porto"], system, {"breos_losses_pct": 14.08})
        serialized = json.dumps(result)
        assert "api_key" not in serialized
        assert "offline-test-secret" not in serialized
        return result, get

    return run


def test_matching_echo_is_trusted(fetch, pvwatts_response):
    # Even if the API echoes credentials, they must not enter the stored record.
    pvwatts_response["inputs"]["api_key"] = "offline-test-secret"
    result, get = fetch()

    assert result["trusted"] is True
    assert result["mismatched_inputs"] == []
    assert result["fetched"] == "2026-10-22"
    assert result["version"] == "8.5.0"
    assert result["ssc_version"] == 280
    assert result["weather_data_source"] == pvwatts_response["station_info"]["weather_data_source"]
    assert result["station_latitude"] == 41.17
    assert result["station_longitude"] == -8.62
    assert result["station_elevation_m"] == 122.0
    assert result["monthly_kwh"] == [100.0] * 12
    assert result["annual_kwh"] == 5620.92
    assert result["solrad_annual_kwh_m2_day"] == 4.123
    assert "warnings" not in result
    get.assert_called_once_with(
        "https://developer.nlr.gov/api/pvwatts/v8.json",
        params={**result["params"], "api_key": "offline-test-secret"},
        timeout=60,
    )
    assert {
        key: str(value) for key, value in result["params"].items() if key not in ("system_capacity", "inv_eff")
    } == {key: value for key, value in INPUTS.items() if key not in ("system_capacity", "inv_eff")}
    assert result["params"]["system_capacity"] == 4.0
    assert result["params"]["inv_eff"] == 96.0


@pytest.mark.parametrize("parameter", ["tilt", "losses", "inv_eff"])
@pytest.mark.parametrize("change", ["different", "missing"])
def test_mismatched_echo_is_untrusted(fetch, pvwatts_response, parameter, change):
    if change == "missing":
        del pvwatts_response["inputs"][parameter]
    else:
        pvwatts_response["inputs"][parameter] = "intl" if parameter == "dataset" else "999"

    result, get = fetch()

    assert result["trusted"] is False
    assert result["mismatched_inputs"] == [parameter]
    assert result["annual_kwh"] == 5620.92
    get.assert_called_once()


def test_missing_inputs_is_untrusted(fetch, pvwatts_response):
    del pvwatts_response["inputs"]
    result, _ = fetch()
    assert result["trusted"] is False
    assert set(result["mismatched_inputs"]) == set(INPUTS)


def test_nonempty_warnings_are_stored(fetch, pvwatts_response):
    pvwatts_response["warnings"] = ["Offline warning fixture"]
    result, _ = fetch()
    assert result["warnings"] == pvwatts_response["warnings"]
    assert result["trusted"] is True


@pytest.mark.parametrize("distance,trusted", [(199_999, True), (200_000, True), (200_001, False), (None, False)])
def test_intl_distance_trust_rule(fetch, pvwatts_response, distance, trusted):
    pvwatts_response["inputs"]["dataset"] = "intl"
    pvwatts_response["station_info"]["distance"] = distance
    get = fetch_references.requests.get
    get.side_effect = [Mock(json=lambda: {"errors": ["No NSRDB data"]}), get.return_value]
    result, _ = fetch()
    assert result["dataset"] == "intl"
    assert result["mismatched_inputs"] == []
    assert result["trusted"] is trusted
    assert get.call_count == 2


def test_input_check_compares_numbers_and_strings():
    assert (
        fetch_references.mismatched_pvwatts_inputs(
            {"system_capacity": 4.0, "inv_eff": 96.0, "dataset": "nsrdb", "api_key": "offline-test-secret"},
            {"system_capacity": "4", "inv_eff": "96", "dataset": "nsrdb"},
        )
        == []
    )
    assert fetch_references.mismatched_pvwatts_inputs({"tilt": 35, "dataset": "nsrdb"}, {"tilt": "bad"}) == [
        "tilt",
        "dataset",
    ]


def test_report_notes_name_mismatched_inputs(monkeypatch):
    monkeypatch.setattr(
        compare,
        "load_reference",
        lambda key: {
            "pvwatts_v8": {
                "dataset": "nsrdb",
                "station_distance_m": 1551,
                "annual_kwh": 5620.92,
                "trusted": False,
                "mismatched_inputs": ["tilt", "losses"],
            }
        },
    )
    model = {"annual_kwh": 5600, "monthly_kwh": [100.0] * 12}
    rows = compare.build_rows({"locations": {"porto": {"models": {"isotropic": model, "perez": model}}}})
    assert "mismatched inputs: tilt, losses" in rows[0]["notes"]
    assert "columns are empty" not in " ".join(compare._pvwatts_status(rows))
