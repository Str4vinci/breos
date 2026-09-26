"""An App run across both DST transitions, and timezone spellings of one instant (#184)."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import App
from breos.load_profiles import load_profile
from breos.runners.app import run_app_simulation

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
        artifacts = run_app_simulation(app._cfg, app._resolved, app._runtime_dependencies())
        app.simulate()
    return app.result(), artifacts.first_year_results_df


def _civil_days(frame):
    local = pd.DatetimeIndex(frame["Datetime"]).tz_convert("Europe/Berlin")
    return frame.set_index(local), pd.Series(local.date).value_counts()


def test_berlin_15min_run_keeps_civil_days_across_both_transitions(synthetic_weather):
    _, results = _run(synthetic_weather)
    local, steps_per_day = _civil_days(results)
    profile = load_profile("1", 4000, start_date="2025-01-01", freq="15min", timezone="Europe/Berlin").iloc[:, 0]

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


@pytest.mark.xfail(
    strict=True,
    reason="#180: monthly rows group on the weather index's own clock, so a UTC year spelled in "
    "Berlin time ends in a 13th row holding the last UTC hour",
)
def test_monthly_rows_do_not_depend_on_the_weather_timezone_spelling(synthetic_weather):
    reference, _ = _run(synthetic_weather, "UTC")
    respelled, _ = _run(synthetic_weather, "Europe/Berlin")

    assert respelled["monthly"] == reference["monthly"]
