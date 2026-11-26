"""An App run across both DST transitions, and timezone spellings of one instant (#184)."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.load_profiles import load_profile
from breos.runners.app import run_app_simulation
from tests.conftest import INSTANT_AT_LABELS

_BERLIN = {
    "location": "berlin",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5,
    "resolution": "15min",
    "projection_years": 1,
    "start_date": "2025-01-01",
}
_EXCLUDED = ("provenance.execution", "monthly")


def _fetch_spelled_in(tz, synthetic_weather):
    def fetch(*_args, **_kwargs):
        weather = synthetic_weather.copy()
        weather.index = weather.index.tz_convert(tz)
        weather.attrs["breos_weather_metadata"] = {
            "source": "synthetic",
            **INSTANT_AT_LABELS,
            "horizon": {"status": "not_applied", "provider": "pvgis", "profile": None},
        }
        return weather, {"inputs": {"location": {"latitude": 52.52, "longitude": 13.405, "elevation": 0}}}

    return fetch


def _run(synthetic_weather, tz="UTC"):
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fetch_spelled_in(tz, synthetic_weather)),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App(_BERLIN)
        artifacts = run_app_simulation(app._resolved, app._runtime_dependencies())
        app.simulate()
    return app.result(), artifacts.first_year_results_df


def _civil_days(frame):
    local = pd.DatetimeIndex(frame["Datetime"]).tz_convert("Europe/Berlin")
    return frame.set_index(local), pd.Series(local.date).value_counts()


def test_berlin_15min_run_keeps_civil_days_across_both_transitions(synthetic_weather):
    _, results = _run(synthetic_weather)
    local, steps_per_day = _civil_days(results)
    profile = load_profile("demandlib_h0", 4000, start_date="2025-01-01", freq="15min", timezone="Europe/Berlin").iloc[
        :, 0
    ]

    spring, autumn, summer = (pd.Timestamp(d).date() for d in ("2025-03-30", "2025-10-26", "2025-06-15"))
    assert steps_per_day[spring] == 92
    assert steps_per_day[autumn] == 100
    assert steps_per_day[summer] == 96
    for day in (spring, autumn, summer):
        simulated = local.loc[local.index.date == day, "Houseload"].sum()
        source = profile[profile.index.date == day].sum()
        assert simulated == pytest.approx(source, rel=1e-12)

    # The wall-clock load shape holds in winter and in summer.
    for day in ("2025-01-15", "2025-07-15"):
        simulated = local.loc[local.index.date == pd.Timestamp(day).date(), "Houseload"].to_numpy()
        np.testing.assert_allclose(simulated, profile[day].to_numpy(), rtol=1e-12)

    delivered = local["PV_AC_To_Load"] + local["PV_AC_Export"] + local["PV_DC_To_Battery"]
    np.testing.assert_allclose(local["PV_Production"], delivered, rtol=0, atol=1e-9)


@pytest.mark.parametrize("tz", ["Europe/Berlin", "Etc/GMT-1"])
def test_weather_timezone_spelling_does_not_change_results(synthetic_weather, tz):
    from tools.parity.app_parity import flatten

    reference, _ = _run(synthetic_weather, "UTC")
    respelled, _ = _run(synthetic_weather, tz)
    reference, respelled = (
        {k: v for k, v in flatten(result).items() if not k.startswith(_EXCLUDED)} for result in (reference, respelled)
    )

    assert respelled == reference


def test_monthly_rows_do_not_depend_on_the_weather_timezone_spelling(synthetic_weather):
    # A UTC year spelled in Berlin time covers the year on UTC, so its months
    # are cut on UTC, as for the UTC spelling (#180, #514).
    reference, _ = _run(synthetic_weather, "UTC")
    respelled, _ = _run(synthetic_weather, "Europe/Berlin")

    assert respelled["monthly"] == reference["monthly"]


def test_annual_monthly_rows_follow_the_civil_year_of_local_time_weather(tmp_path, monkeypatch, no_fetch):
    # Local-time weather across a DST change is saved with two UTC offsets and
    # read back as UTC. It covers the year on the location's clock, so its
    # months are cut there: twelve rows, not a thirteenth holding the first
    # hours of the next year in UTC (#514).
    from tests.conftest import _build_open_meteo_weather

    zone = "America/Punta_Arenas"
    index = pd.date_range("2013-01-01", "2014-01-01", freq="h", inclusive="left", tz=zone)
    _build_open_meteo_weather(index).to_csv(tmp_path / "local.csv")
    monkeypatch.chdir(tmp_path)
    app = App(
        {
            "location": {"latitude": -53.1638, "longitude": -70.9171, "timezone": zone},
            "n_modules": 6,
            "annual_consumption_kwh": 4000,
            "projection_years": 1,
            "start_date": "2013-01-01",
            "weather_file": "local.csv",
            "solar_position": "interval-start",
        }
    )
    app.simulate()
    result = app.result()
    months = result["monthly"]

    assert [row["month"] for row in months] == list(pd.date_range("2013-01", periods=12, freq="MS").strftime("%b"))
    assert sum(row["consumption_kwh"] for row in months) == pytest.approx(result["consumption_kwh"], abs=0.1)
    january_kwh = app.timeseries().set_index("Datetime").tz_convert(zone).loc["2013-01", "Houseload"].sum() / 1000
    assert months[0]["consumption_kwh"] == pytest.approx(january_kwh, abs=0.01)
