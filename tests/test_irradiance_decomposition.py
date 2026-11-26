"""GHI-only weather: DNI and DHI estimated by a configured pvlib decomposition model."""

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pytest
from pvlib.location import Location

from breos import App, decompose_ghi, optimization
from breos.app_config import DEFAULTS, resolve_app_config
from breos.cli import build_parser
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.optimization_config import resolve_optimization_config
from breos.pv.decomposition import DECOMPOSITION_MODELS, DEFAULT_DECOMPOSITION_MODEL
from breos.weather import WEATHER_METADATA_KEY

PORTO = Location(41.1579, -8.6291, tz="Europe/Lisbon")
APP_BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 0,
    "tilt": 35,
    "azimuth": 180,
    "projection_years": 1,
    "start_date": "2023-01-01",
}


def _clear_sky_year(year=2023, freq="h"):
    """A year of Ineichen clear-sky weather: full split plus air temperature and wind."""
    index = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=freq, inclusive="left", tz="UTC")
    sky = PORTO.get_clearsky(index)
    # Some cloud: scale every third day so the decomposition sees more than clear sky.
    scale = np.where(index.dayofyear % 3 == 0, 0.4, 1.0)
    weather = pd.DataFrame({"ghi": sky.ghi * scale, "dni": sky.dni * scale, "dhi": sky.dhi * scale}, index=index)
    weather["temp_air"] = 15.0
    weather["wind_speed"] = 2.0
    return weather


def _ghi_only(weather):
    return weather.drop(columns=["dni", "dhi"])


def _closure_residual(weather, solar_position="interval-start"):
    offset = pd.Timedelta(minutes=30) if solar_position == "mid-interval" else pd.Timedelta(0)
    zenith = PORTO.get_solarposition(weather.index + offset).zenith.to_numpy()
    cos_zenith = np.clip(np.cos(np.radians(zenith)), 0.0, None)
    return np.abs(weather.ghi.to_numpy() - weather.dni.to_numpy() * cos_zenith - weather.dhi.to_numpy())


# --------------------------------------------------------------------------- the kernel


@pytest.mark.parametrize("model", DECOMPOSITION_MODELS)
def test_every_model_gives_a_closed_non_negative_split(model):
    result = decompose_ghi(_ghi_only(_clear_sky_year()), PORTO, model)

    assert (result[["dni", "dhi"]] >= 0).all().all()
    assert np.isfinite(result[["dni", "dhi"]].to_numpy()).all()
    np.testing.assert_allclose(_closure_residual(result), 0.0, atol=1e-9)
    # Some beam on clear days, and none at night.
    assert result.dni.max() > 300
    night = result.ghi == 0
    assert (result.dni[night] == 0).all() and (result.dhi[night] == 0).all()


def test_the_split_is_pvlibs_model():
    weather = _ghi_only(_clear_sky_year())
    result = decompose_ghi(weather, PORTO, "erbs_driesse")

    zenith = PORTO.get_solarposition(weather.index).zenith
    expected = pvlib.irradiance.erbs_driesse(weather.ghi, zenith, weather.index)["dni"].fillna(0.0)
    # Away from the horizon, where BREOS caps nothing, the DNI is pvlib's.
    high_sun = (zenith < 80).to_numpy()
    np.testing.assert_allclose(result.dni.to_numpy()[high_sun], expected.to_numpy()[high_sun], rtol=1e-12)


def test_the_default_model_is_erbs_driesse():
    assert DEFAULT_DECOMPOSITION_MODEL == "erbs_driesse"
    weather = _ghi_only(_clear_sky_year())
    pd.testing.assert_frame_equal(decompose_ghi(weather, PORTO), decompose_ghi(weather, PORTO, "erbs_driesse"))


def test_a_supplied_split_is_never_replaced():
    weather = _clear_sky_year()
    assert decompose_ghi(weather, PORTO, "disc") is weather

    open_meteo = weather.rename(
        columns={"ghi": "shortwave_radiation", "dni": "direct_normal_irradiance", "dhi": "diffuse_radiation"}
    )
    assert decompose_ghi(open_meteo, PORTO) is open_meteo


@pytest.mark.parametrize("keep", ["dni", "dhi"])
def test_half_a_split_is_refused(keep):
    weather = _clear_sky_year().drop(columns=[{"dni": "dhi", "dhi": "dni"}[keep]])
    with pytest.raises(ValueError, match=f"GHI and {keep.upper()} but not the other"):
        decompose_ghi(weather, PORTO)


def test_weather_without_ghi_is_refused():
    weather = _clear_sky_year()[["temp_air", "wind_speed"]]
    with pytest.raises(ValueError, match="needs a GHI column"):
        decompose_ghi(weather, PORTO)


def test_an_unknown_model_lists_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown irradiance decomposition model 'nope'.*erbs_driesse"):
        decompose_ghi(_ghi_only(_clear_sky_year()), PORTO, "nope")


def test_model_names_are_normalised():
    weather = _ghi_only(_clear_sky_year())
    pd.testing.assert_frame_equal(decompose_ghi(weather, PORTO, " DIRINT "), decompose_ghi(weather, PORTO, "dirint"))


def test_open_meteo_ghi_name_is_read():
    weather = _ghi_only(_clear_sky_year()).rename(columns={"ghi": "shortwave_radiation"})
    result = decompose_ghi(weather, PORTO)
    assert result.attrs[WEATHER_METADATA_KEY]["irradiance_decomposition"]["source_variable"] == "shortwave_radiation"
    assert result.dni.max() > 300


def test_the_input_is_not_mutated_and_metadata_is_recorded():
    weather = _ghi_only(_clear_sky_year())
    weather.attrs[WEATHER_METADATA_KEY] = {"source": "logger"}
    before = weather.copy()

    result = decompose_ghi(weather, PORTO, "dirint", solar_position="mid-interval")

    pd.testing.assert_frame_equal(weather, before)
    assert weather.attrs[WEATHER_METADATA_KEY] == {"source": "logger"}
    metadata = result.attrs[WEATHER_METADATA_KEY]
    assert metadata["source"] == "logger"
    assert metadata["irradiance_decomposition"] == {
        "model": "dirint",
        "derived_variables": ["dni", "dhi"],
        "source_variable": "ghi",
        "solar_position": "mid-interval",
        "pvlib_version": pvlib.__version__,
    }


def test_the_sun_is_evaluated_where_the_pv_model_evaluates_it():
    weather = _ghi_only(_clear_sky_year())
    start = decompose_ghi(weather, PORTO, solar_position="interval-start")
    middle = decompose_ghi(weather, PORTO, solar_position="mid-interval")

    assert not np.allclose(start.dni, middle.dni)
    np.testing.assert_allclose(_closure_residual(middle, "mid-interval"), 0.0, atol=1e-9)


def test_missing_ghi_stays_missing():
    weather = _ghi_only(_clear_sky_year())
    weather.iloc[4000, weather.columns.get_loc("ghi")] = np.nan
    result = decompose_ghi(weather, PORTO)
    assert np.isnan(result.dni.iloc[4000]) and np.isnan(result.dhi.iloc[4000])
    assert result.dni.drop(result.index[4000]).notna().all()


def test_fifteen_minute_weather_is_decomposed_at_its_own_step():
    weather = _ghi_only(_clear_sky_year(freq="15min"))
    result = decompose_ghi(weather, PORTO, solar_position="mid-interval")
    # Mid-interval at 15 minutes is 7.5 minutes, not half an hour.
    zenith = PORTO.get_solarposition(weather.index + pd.Timedelta(minutes=7.5)).zenith.to_numpy()
    residual = result.ghi - result.dni * np.clip(np.cos(np.radians(zenith)), 0, None) - result.dhi
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


# --------------------------------------------------------------------------- App


def test_config_key_defaults_to_unset_and_validates():
    assert DEFAULTS["irradiance_decomposition"] is None
    assert resolve_app_config(APP_BASE).cfg["irradiance_decomposition"] is None
    resolved = resolve_app_config({**APP_BASE, "irradiance_decomposition": "Erbs_Driesse"})
    assert resolved.cfg["irradiance_decomposition"] == "erbs_driesse"
    with pytest.raises(ValueError, match="'irradiance_decomposition' must be one of: erbs_driesse"):
        resolve_app_config({**APP_BASE, "irradiance_decomposition": "nope"})


def test_cli_flags_set_the_key():
    parser = build_parser()
    run = parser.parse_args(["run", "--config", "c.toml", "--irradiance-decomposition", "dirint"])
    assert run.irradiance_decomposition == "dirint"
    mc = parser.parse_args(["montecarlo", "--config", "c.toml", "--irradiance-decomposition", "disc"])
    assert mc.irradiance_decomposition == "disc"


def _app_result(config):
    app = App(config)
    app.simulate()
    return app.result()


# The repository's own PVGIS-SARAH3 TMY for Porto (validation/data/weather).
PORTO_TMY = next((Path(__file__).resolve().parents[1] / "validation" / "data" / "weather").glob("porto_tmy_*.csv.gz"))


@pytest.mark.parametrize(("resolution", "resampling"), [("h", "auto"), ("15min", "auto"), ("15min", "quadrature")])
def test_app_runs_ghi_only_weather(tmp_path, resolution, resampling):
    full = pd.read_csv(PORTO_TMY, index_col=0)
    full.to_csv(tmp_path / "full.csv")
    _ghi_only(full).to_csv(tmp_path / "ghi.csv")
    config = {
        **APP_BASE,
        "resolution": resolution,
        "irradiance_resampling": resampling,
        "transposition_model": "perez",
    }

    reference = _app_result({**config, "weather_file": str(tmp_path / "full.csv")})
    result = _app_result(
        {**config, "weather_file": str(tmp_path / "ghi.csv"), "irradiance_decomposition": "erbs_driesse"}
    )

    weather = result["provenance"]["weather"]
    assert weather["irradiance_decomposition"]["model"] == "erbs_driesse"
    assert result["provenance"]["resolved_config"]["irradiance_decomposition"] == "erbs_driesse"
    # On real satellite weather Erbs-Driesse's split of the same GHI gives
    # nearly the same yield as the satellite's own split (about -0.01 %).
    assert result["pv_dc_generation_kwh"] == pytest.approx(reference["pv_dc_generation_kwh"], rel=0.01)


def test_app_ignores_the_key_when_the_weather_has_a_split(tmp_path):
    _clear_sky_year().to_csv(tmp_path / "full.csv", index_label="time")
    config = {**APP_BASE, "weather_file": str(tmp_path / "full.csv")}

    plain = _app_result(config)
    with_key = _app_result({**config, "irradiance_decomposition": "boland"})

    assert with_key["pv_dc_generation_kwh"] == plain["pv_dc_generation_kwh"]
    assert "irradiance_decomposition" not in with_key["provenance"]["weather"]


def test_app_names_the_key_when_ghi_only_weather_is_not_decomposed(tmp_path):
    _ghi_only(_clear_sky_year()).to_csv(tmp_path / "ghi.csv", index_label="time")
    with pytest.raises(KeyError, match="set 'irradiance_decomposition'"):
        _app_result({**APP_BASE, "weather_file": str(tmp_path / "ghi.csv")})


def test_quadrature_resampling_names_the_key_for_ghi_only_weather(tmp_path):
    _ghi_only(_clear_sky_year()).to_csv(tmp_path / "ghi.csv", index_label="time")
    config = {
        **APP_BASE,
        "weather_file": str(tmp_path / "ghi.csv"),
        "resolution": "15min",
        "irradiance_resampling": "quadrature",
    }
    with pytest.raises(ValueError, match="missing: dni, dhi. The weather has GHI only: set 'irradiance_decomposition'"):
        _app_result(config)


def test_a_partial_split_gets_no_decomposition_hint(tmp_path):
    _clear_sky_year().drop(columns=["dhi"]).to_csv(tmp_path / "partial.csv", index_label="time")
    with pytest.raises(KeyError) as error:
        _app_result({**APP_BASE, "weather_file": str(tmp_path / "partial.csv")})
    assert "irradiance_decomposition" not in str(error.value)
    with pytest.raises(ValueError, match="GHI and DNI but not the other"):
        _app_result(
            {**APP_BASE, "weather_file": str(tmp_path / "partial.csv"), "irradiance_decomposition": "erbs_driesse"}
        )


def test_the_terrain_horizon_names_the_key_and_takes_the_decomposed_split():
    from breos.pv.horizon import apply_terrain_horizon_profile

    weather = _ghi_only(_clear_sky_year())
    weather.attrs[WEATHER_METADATA_KEY] = {"horizon": {"status": "not_applied"}}
    horizon = [[0, 5], [90, 5], [180, 5], [270, 5]]
    with pytest.raises(ValueError, match="set 'irradiance_decomposition'"):
        apply_terrain_horizon_profile(weather, PORTO, horizon, freq="h", solar_position="interval-start")

    split = decompose_ghi(weather, PORTO, solar_position="interval-start")
    shaded = apply_terrain_horizon_profile(split, PORTO, horizon, freq="h", solar_position="interval-start")
    assert shaded.dni.sum() < split.dni.sum()


# --------------------------------------------------------------------------- Monte Carlo and optimizer


def test_montecarlo_decomposes_each_weather_year(tmp_path, write_multiyear_weather):
    path = write_multiyear_weather(tmp_path / "years.csv")
    frame = pd.read_csv(path).drop(columns=["direct_normal_irradiance", "diffuse_radiation"])
    frame.to_csv(path, index=False)
    config = {**APP_BASE, "projection_years": 2}
    settings = MonteCarloSettings(weather_file=str(path), n_runs=2, years_per_run=2, seed=1)

    with pytest.raises(KeyError, match="set 'irradiance_decomposition'"):
        run_montecarlo(config, settings)
    result = run_montecarlo({**config, "irradiance_decomposition": "erbs_driesse"}, settings)
    assert len(result.runs) == 2


def test_montecarlo_weather_cache_key_includes_the_model():
    from breos.montecarlo import _weather_cache_key

    settings = MonteCarloSettings(weather_file=__file__)
    keys = []
    for model in (None, "erbs", "disc"):
        resolved = resolve_app_config({**APP_BASE, "irradiance_decomposition": model})
        keys.append(_weather_cache_key(resolved.cfg, resolved, settings)["irradiance_decomposition"])
    assert keys == [None, "erbs", "disc"]


def test_optimizer_fills_a_ghi_only_split():
    config = resolve_optimization_config(
        {"location": {"latitude": 41.1579, "longitude": -8.6291}, "simulation": {"irradiance_decomposition": "erbs"}}
    )
    assert (
        resolve_optimization_config({"location": {"latitude": 41.1579, "longitude": -8.6291}})["simulation"][
            "irradiance_decomposition"
        ]
        is None
    )

    weather = _ghi_only(_clear_sky_year())
    prepared = optimization._prepare_study_weather(weather, config, optimization._site_location(config["location"]))

    assert prepared.attrs[WEATHER_METADATA_KEY]["irradiance_decomposition"]["model"] == "erbs"
    assert prepared.dni.max() > 300


def test_optimizer_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="irradiance_decomposition"):
        resolve_optimization_config(
            {"location": {"latitude": 41.15, "longitude": -8.61}, "simulation": {"irradiance_decomposition": "nope"}}
        )


# --------------------------------------------------------------------------- physical limits and input shapes


def _overcast_low_sun_year():
    """GHI-only weather with diffuse light at every daylight step, low sun included."""
    weather = _ghi_only(_clear_sky_year())
    zenith = PORTO.get_solarposition(weather.index).zenith
    weather["ghi"] = np.where(zenith < 90, 120.0, 0.0)
    return weather


@pytest.mark.parametrize("model", DECOMPOSITION_MODELS)
def test_dni_never_exceeds_the_extraterrestrial_irradiance(model):
    for weather in (_clear_sky_year(), _overcast_low_sun_year()):
        result = decompose_ghi(_ghi_only(weather) if "dni" in weather else weather, PORTO, model)
        dni_extra = pvlib.irradiance.get_extra_radiation(result.index).to_numpy()
        assert np.all(result.dni.to_numpy() <= dni_extra + 1e-9)
        np.testing.assert_allclose(_closure_residual(result), 0.0, atol=1e-9)


def test_negative_ghi_is_zeroed_and_the_split_still_closes():
    weather = _ghi_only(_clear_sky_year())
    night = weather.index[weather.ghi.to_numpy() == 0][:5]
    weather.loc[night, "ghi"] = -4.0
    result = decompose_ghi(weather, PORTO)

    assert (result.loc[night, ["ghi", "dni", "dhi"]].to_numpy() == 0.0).all()
    assert (weather.loc[night, "ghi"] == -4.0).all()  # the input is not mutated
    np.testing.assert_allclose(_closure_residual(result), 0.0, atol=1e-9)


@pytest.mark.parametrize("model", ["erbs_driesse", "dirint", "dirindex"])
def test_rows_in_any_order_give_the_same_split(model):
    # DIRINT and DIRINDEX read the neighbouring rows, so order matters to them.
    weather = _ghi_only(_clear_sky_year())
    shuffled = weather.sample(frac=1.0, random_state=7)
    expected = decompose_ghi(weather, PORTO, model, solar_position="mid-interval")
    result = decompose_ghi(shuffled, PORTO, model, solar_position="mid-interval")

    pd.testing.assert_index_equal(result.index, shuffled.index)
    pd.testing.assert_frame_equal(result.sort_index(), expected, check_freq=False)


def test_a_gap_does_not_change_the_step():
    weather = _ghi_only(_clear_sky_year())
    gapped = weather.drop(weather.index[1:48])
    result = decompose_ghi(gapped, PORTO, solar_position="mid-interval")
    np.testing.assert_allclose(_closure_residual(result, "mid-interval"), 0.0, atol=1e-9)


def test_non_finite_ghi_is_left_for_the_input_checks():
    weather = _ghi_only(_clear_sky_year())
    weather.iloc[100, weather.columns.get_loc("ghi")] = -np.inf
    result = decompose_ghi(weather, PORTO)
    assert result.ghi.iloc[100] == -np.inf
    assert np.isnan(result.dni.iloc[100]) and np.isnan(result.dhi.iloc[100])


def test_dirindex_falls_back_to_dirint_where_its_clear_sky_reference_vanishes():
    weather = _overcast_low_sun_year()
    sun = PORTO.get_solarposition(weather.index)
    clearsky = PORTO.get_clearsky(weather.index, model="ineichen", solar_position=sun)
    pressure = pvlib.atmosphere.alt2pres(PORTO.altitude)
    raw = pvlib.irradiance.dirindex(
        weather.ghi, clearsky.ghi, clearsky.dni, sun.zenith, weather.index, pressure=pressure
    )
    fallback = ~np.isfinite(raw) & (sun.zenith < 90)
    assert fallback.any()

    result = decompose_ghi(weather, PORTO, "dirindex", solar_position="interval-start")
    dirint = decompose_ghi(weather, PORTO, "dirint", solar_position="interval-start")
    np.testing.assert_allclose(result.dni[fallback], dirint.dni[fallback])
    assert result.dni[fallback].gt(0).any()


def test_stamps_off_the_hour_use_their_own_sun():
    weather = _ghi_only(_clear_sky_year())
    weather.index = weather.index + pd.Timedelta(minutes=10)
    result = decompose_ghi(weather, PORTO, solar_position="interval-start")
    np.testing.assert_allclose(_closure_residual(result), 0.0, atol=1e-9)


def test_optimizer_decomposes_right_labelled_weather_on_interval_starts():
    config = resolve_optimization_config(
        {
            "location": {"latitude": 41.1579, "longitude": -8.6291},
            "simulation": {"irradiance_decomposition": "erbs", "resolution": "15min"},
            "solar_position": "mid-interval",
        }
    )
    weather = _ghi_only(_clear_sky_year())
    weather.index = weather.index + pd.Timedelta(hours=1)
    weather.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "right"}
    location = optimization._site_location(config["location"])

    prepared = optimization._prepare_study_weather(weather, config, location)

    starts = weather.copy()
    starts.index = starts.index - pd.Timedelta(hours=1)
    starts.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "left"}
    expected = optimization._prepare_study_weather(starts, config, location)
    np.testing.assert_allclose(prepared.dni.to_numpy(), expected.dni.to_numpy())
    np.testing.assert_allclose(prepared.dhi.to_numpy(), expected.dhi.to_numpy())


@pytest.mark.parametrize("resolution", ["h", "15min"])
def test_optimizer_leaves_a_supplied_split_as_it_was(resolution):
    location_config = {"latitude": 41.1579, "longitude": -8.6291}
    with_key = resolve_optimization_config(
        {"location": location_config, "simulation": {"irradiance_decomposition": "erbs", "resolution": resolution}}
    )
    without_key = resolve_optimization_config({"location": location_config, "simulation": {"resolution": resolution}})
    weather = _clear_sky_year()
    weather.index = weather.index + pd.Timedelta(hours=1)
    weather.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "right"}
    location = optimization._site_location(with_key["location"])

    pd.testing.assert_frame_equal(
        optimization._prepare_study_weather(weather, with_key, location),
        optimization._prepare_study_weather(weather, without_key, location),
    )
