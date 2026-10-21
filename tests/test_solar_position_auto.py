"""``solar_position = "auto"``, the default since 0.8.0 (#418).

``auto`` is ``weather`` when the weather metadata states its
``radiation_time_basis`` and ``interval-start`` when it does not, and then
warns once per run. Explicit ``weather`` stays strict, and the explicit
methods are unchanged.
"""

import warnings
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos import App, cli
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.pv.model_options import (
    DEFAULT_SOLAR_POSITION,
    SOLAR_POSITION_METHODS,
    resolve_solar_position_for_weather,
    solar_position_record,
    solar_position_warning_scope,
)
from breos.pv_modules import get_module
from breos.solar import calculate_multi_array_production, calculate_pv_production_dc
from breos.weather import WEATHER_METADATA_KEY
from tests.conftest import _build_synthetic_weather

AUTO_WARNING = "solar_position = 'auto'"
PORTO = Location(41.15, -8.61, tz="UTC")

# Weather metadata that states a time basis, one case per supported convention.
STATED = {
    "instant_with_offset": {"radiation_time_basis": "instant", "irradiance_time_offset_hours": 0.1714},
    "interval_mean_left": {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "left"},
    "interval_mean_right": {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "right"},
}
UNSTATED = {"source": "user_csv", "horizon": {"status": "not_applied", "provider": None, "profile": None}}


def _weather(metadata, year=2023):
    """The synthetic year, with ``metadata`` as its only weather metadata."""
    weather = _build_synthetic_weather(year)
    weather.attrs = {WEATHER_METADATA_KEY: dict(metadata)}
    return weather


@contextmanager
def _auto_warnings():
    """Record every warning; the yielded list holds the auto fallback ones."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matching: list = []
        yield matching
    matching.extend(w for w in caught if AUTO_WARNING in str(w.message))


def _dc(weather, **kwargs):
    return calculate_pv_production_dc(
        weather_data=weather,
        location=PORTO,
        tilt=35,
        surface_azimuth=180,
        n_modules=4,
        pv_params=get_module("Suntech_STP550S_STC"),
        freq="h",
        **kwargs,
    )


def test_auto_is_the_default_and_a_choice():
    assert DEFAULT_SOLAR_POSITION == "auto"
    assert set(SOLAR_POSITION_METHODS) == {"auto", "interval-start", "mid-interval", "weather"}


@pytest.mark.parametrize(
    ("metadata", "resolved"),
    [*((stated, "weather") for stated in STATED.values()), (UNSTATED, "interval-start"), ({}, "interval-start")],
)
def test_auto_resolves_on_the_stated_time_basis(metadata, resolved):
    assert resolve_solar_position_for_weather(" AUTO ", _weather(metadata)) == resolved
    for explicit in ("interval-start", "mid-interval", "weather"):
        assert resolve_solar_position_for_weather(explicit, _weather(metadata)) == explicit


@pytest.mark.parametrize("case", sorted(STATED))
def test_auto_with_a_stated_time_basis_equals_weather_bit_for_bit(case):
    weather = _weather(STATED[case])

    with _auto_warnings() as caught:
        auto = _dc(weather)
    weather_method = _dc(weather, solar_position="weather")

    np.testing.assert_array_equal(auto.to_numpy(), weather_method.to_numpy())
    assert not caught
    # The metadata moves the sun, so the case would catch an auto that ignored it.
    assert not np.array_equal(auto.to_numpy(), _dc(weather, solar_position="interval-start").to_numpy())


def test_auto_without_a_time_basis_equals_interval_start_and_warns_once():
    weather = _weather(UNSTATED)

    with _auto_warnings() as caught:
        auto = _dc(weather)
    interval_start = _dc(weather, solar_position="interval-start")

    np.testing.assert_array_equal(auto.to_numpy(), interval_start.to_numpy())
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "does not state its time basis" in message
    assert "step labels" in message
    assert "set solar_position explicitly" in message
    assert "metadata sidecar" in message
    assert issubclass(caught[0].category, UserWarning)


@pytest.mark.parametrize("method", ["interval-start", "mid-interval"])
def test_explicit_label_methods_never_warn(method):
    with _auto_warnings() as caught:
        _dc(_weather(UNSTATED), solar_position=method)
    assert not caught


def test_malformed_metadata_raises_under_auto_as_under_weather():
    weather = _weather({"radiation_time_basis": "interval_mean"})
    for method in ("auto", "weather"):
        with pytest.raises(ValueError, match="timestamp_label_basis='left' or 'right'"):
            _dc(weather, solar_position=method)


def test_explicit_weather_without_metadata_still_raises():
    with pytest.raises(ValueError, match="radiation_time_basis"):
        _dc(_weather(UNSTATED), solar_position="weather")


def test_a_warning_scope_warns_once_and_a_scope_inside_it_is_part_of_it():
    weather = _weather(UNSTATED)
    with _auto_warnings() as caught:
        with solar_position_warning_scope():
            _dc(weather)
            with solar_position_warning_scope():
                _dc(weather)
            _dc(weather)
    assert len(caught) == 1

    # Outside a scope every computation warns.
    with _auto_warnings() as caught:
        _dc(weather)
        _dc(weather)
    assert len(caught) == 2


def test_solar_position_record_names_the_requested_and_resolved_method():
    assert solar_position_record("Auto", _weather(UNSTATED), "h") == {
        "requested": "auto",
        "resolved": "interval-start",
        "offset_minutes": 0.0,
    }
    assert solar_position_record("auto", _weather(STATED["instant_with_offset"]), "h") == {
        "requested": "auto",
        "resolved": "weather",
        "offset_minutes": pytest.approx(0.1714 * 60),
    }
    assert solar_position_record("auto", _weather(STATED["interval_mean_left"]), "15min")["offset_minutes"] == 7.5
    assert solar_position_record("mid-interval", _weather(UNSTATED), "h") == {
        "requested": "mid-interval",
        "resolved": "mid-interval",
        "offset_minutes": 30.0,
    }


def test_a_multi_array_call_warns_once_for_all_its_arrays():
    weather = _weather(UNSTATED)
    module = "Suntech_STP550S_STC"
    arrays = [
        {"modules": 2, "module": module, "tilt": 30, "azimuth": 90},
        {"modules": 2, "module": module, "tilt": 30, "azimuth": 270},
    ]
    with _auto_warnings() as caught:
        auto = calculate_multi_array_production(weather, PORTO, arrays, freq="h")
    explicit = calculate_multi_array_production(weather, PORTO, arrays, freq="h", solar_position="interval-start")
    pd.testing.assert_series_equal(auto, explicit, check_exact=True)
    assert len(caught) == 1


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

APP_BASE = {
    "location": "porto",
    "annual_consumption_kwh": 3000,
    "projection_years": 1,
    "start_date": "2023-01-01",
    "battery_kwh": 0.0,
    "pv_arrays": [{"modules": 3, "tilt": 30, "azimuth": 120}, {"modules": 3, "tilt": 30, "azimuth": 240}],
    # Terrain shading computes the sun as well, so one run computes it three times.
    "horizon_profile": [[0, 5], [180, 5]],
}


def _app(config, metadata):
    weather = _weather(metadata)

    def fetch(*_args, **_kwargs):
        return weather.copy(), {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}

    with (
        mock.patch("breos.app.fetch_tmy_weather_data", fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App(config)
        app.simulate()
    return app


def _series(app):
    return app._artifacts.first_year_results_df["PV_DC"].to_numpy()


def test_app_default_without_a_time_basis_warns_once_and_keeps_the_label_results():
    with _auto_warnings() as caught:
        auto = _app(APP_BASE, UNSTATED)
    assert len(caught) == 1

    explicit = _app({**APP_BASE, "solar_position": "interval-start"}, UNSTATED)
    np.testing.assert_array_equal(_series(auto), _series(explicit))
    assert auto.result()["npv_savings"] == explicit.result()["npv_savings"]

    provenance = auto.result()["provenance"]
    assert provenance["resolved_config"]["solar_position"] == "auto"
    assert provenance["pv_model"]["solar_position"] == {
        "requested": "auto",
        "resolved": "interval-start",
        "offset_minutes": 0.0,
    }
    assert provenance["weather"]["horizon"]["profile"]["solar_position"] == "interval-start"
    assert explicit.result()["provenance"]["pv_model"]["solar_position"] == {
        "requested": "interval-start",
        "resolved": "interval-start",
        "offset_minutes": 0.0,
    }


def test_app_default_with_a_time_basis_equals_weather_without_a_warning():
    metadata = {**UNSTATED, **STATED["instant_with_offset"]}
    with _auto_warnings() as caught:
        auto = _app(APP_BASE, metadata)
    assert not caught

    explicit = _app({**APP_BASE, "solar_position": "weather"}, metadata)
    np.testing.assert_array_equal(_series(auto), _series(explicit))
    assert auto.result()["pv_dc_generation_kwh"] == explicit.result()["pv_dc_generation_kwh"]
    assert auto.result()["provenance"]["pv_model"]["solar_position"] == {
        "requested": "auto",
        "resolved": "weather",
        "offset_minutes": pytest.approx(0.1714 * 60),
    }
    assert explicit.result()["provenance"]["pv_model"]["solar_position"]["requested"] == "weather"


def test_app_warns_once_per_run():
    with _auto_warnings() as caught:
        _app(APP_BASE, UNSTATED)
        _app(APP_BASE, UNSTATED)
    assert len(caught) == 2


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

MC_BASE = {
    "location": "porto",
    "n_modules": 4,
    "annual_consumption_kwh": 3000,
    "battery_kwh": 0.0,
    "projection_years": 1,
    "start_date": "2023-01-01",
    "execution_backend": "python",
}


def _mc_weather(tmp_path, write_multiyear_weather, *, sidecar):
    path = write_multiyear_weather(tmp_path / "multi.csv", years=(2021, 2022, 2023))
    if not sidecar:
        (tmp_path / "multi.csv.metadata.json").unlink()
    return MonteCarloSettings(weather_file=str(path), n_runs=2, years_per_run=1, seed=1)


def test_montecarlo_warns_once_across_its_weather_years(tmp_path, write_multiyear_weather):
    settings = _mc_weather(tmp_path, write_multiyear_weather, sidecar=False)
    with _auto_warnings() as caught:
        auto = run_montecarlo(MC_BASE, settings)
    assert len(caught) == 1

    explicit = run_montecarlo({**MC_BASE, "solar_position": "interval-start"}, settings)
    pd.testing.assert_frame_equal(auto.runs, explicit.runs, check_exact=True)
    runtime = auto.provenance["runtime_weather"]
    assert runtime["solar_position_requested"] == "auto"
    assert runtime["solar_position_method"] == "interval-start"
    assert runtime["solar_position_offset_minutes"] == 0.0


def test_montecarlo_with_a_sidecar_resolves_auto_to_weather(tmp_path, write_multiyear_weather):
    settings = _mc_weather(tmp_path, write_multiyear_weather, sidecar=True)
    with _auto_warnings() as caught:
        result = run_montecarlo(MC_BASE, settings)
    assert not caught
    runtime = result.provenance["runtime_weather"]
    assert (runtime["solar_position_requested"], runtime["solar_position_method"]) == ("auto", "weather")


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

OPT_CONFIG = {
    "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
    "simulation": {"resolution": "h", "years_projection": 2},
    "constraints": {"budget": 100000, "max_area_m2": 100.0, "max_modules": 5},
    "mode": {"fixed_azimuth": 180},
    "pv": {"module": "Suntech_STP550S_STC"},
    "battery": {"temperature": 20.0, "indoor_model": {"enabled": False}},
}


def _load(index):
    return pd.DataFrame({"Load": 400.0 + 300.0 * (index.hour >= 17)}, index=index)


def test_the_optimizer_resolves_auto_once_on_its_weather():
    pytest.importorskip("pymoo")
    from breos.optimization import SolarDesignProblem

    weather = _weather(UNSTATED)
    with _auto_warnings() as caught:
        problem = SolarDesignProblem(weather, _load(weather.index), OPT_CONFIG)
        out: dict = {}
        problem._evaluate(np.array([3.0, 0.0, 30.0], dtype=float), out)
        problem._evaluate(np.array([4.0, 0.0, 35.0], dtype=float), out)
    assert len(caught) == 1
    # Candidates, in this process or a worker, get the resolved method.
    assert problem.model_options["solar_position"] == "interval-start"
    assert problem.solar_position == {"requested": "auto", "resolved": "interval-start", "offset_minutes": 0.0}

    stated = _weather(STATED["interval_mean_left"])
    with _auto_warnings() as caught:
        problem = SolarDesignProblem(stated, _load(stated.index), OPT_CONFIG)
    assert not caught
    assert problem.model_options["solar_position"] == "weather"
    assert problem.solar_position["offset_minutes"] == 30.0

    with pytest.raises(ValueError, match="radiation_time_basis"):
        SolarDesignProblem(weather, _load(weather.index), {**OPT_CONFIG, "solar_position": "weather"})


def test_a_fixed_design_warns_once_across_its_weather_years():
    from breos.optimization import evaluate_projected_design

    tmy = _weather(UNSTATED)
    frames = [_weather(UNSTATED, 2021), _weather(UNSTATED, 2022)]
    kwargs = {"n_modules": 4, "battery_kwh": 0.0, "tilt": 30.0, "azimuth": 180.0, "weather_by_year": frames}
    load = _load(frames[0].index)
    with _auto_warnings() as caught:
        auto = evaluate_projected_design(tmy, load, OPT_CONFIG, **kwargs)
    assert len(caught) == 1
    explicit = evaluate_projected_design(tmy, load, {**OPT_CONFIG, "solar_position": "interval-start"}, **kwargs)
    pd.testing.assert_frame_equal(auto.yearly, explicit.yearly, check_exact=True)
    assert auto.provenance["solar_position"] == {
        "requested": "auto",
        "resolved": "interval-start",
        "offset_minutes": 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_the_cli_offers_auto():
    parser = cli.build_parser()
    assert cli._build_config(parser.parse_args(["run", "--solar-position", "auto"]))["solar_position"] == "auto"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--solar-position", "noon"])
