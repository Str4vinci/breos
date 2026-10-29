"""PV-side weather input is refused instead of repaired (#172)."""

import logging
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from breos.solar import (
    TRANSPOSITION_MODELS,
    _extract_irradiance,
    calculate_multi_array_production,
    calculate_pv_production_dc,
    calculate_pv_production_dc_tracking,
)
from breos.weather import _complete_weather_years, build_battery_temperature_series, validate_weather_irradiance


def _dc(weather, location, pv_params, freq="h", **kwargs):
    return calculate_pv_production_dc(
        weather_data=weather,
        location=location,
        tilt=35,
        surface_azimuth=180,
        n_modules=1,
        pv_params=pv_params,
        freq=freq,
        **kwargs,
    )


def test_weather_with_a_missing_month_raises(synthetic_weather, porto_location, pv_params):
    # Nearest-row reindexing used to give June zero PV (the nearest rows are
    # night-time 31 May and 1 July) without a message.
    no_june = synthetic_weather[synthetic_weather.index.month != 6]

    with pytest.raises(ValueError, match=r"must step evenly at freq='h': row 3624 is 2023-07-01 00:00"):
        _dc(no_june, porto_location, pv_params)


def test_weather_at_another_resolution_raises(synthetic_weather_15min, porto_location, pv_params):
    with pytest.raises(ValueError, match=r"must step evenly at freq='h': row 1 .* 0 days 00:15:00 after"):
        _dc(synthetic_weather_15min, porto_location, pv_params, freq="h")


def test_tracking_path_checks_the_weather_grid(synthetic_weather, porto_location, pv_params):
    with pytest.raises(ValueError, match="must step evenly"):
        calculate_pv_production_dc_tracking(
            weather_data=synthetic_weather.drop(synthetic_weather.index[100]),
            location=porto_location,
            n_modules=1,
            pv_params=pv_params,
            freq="h",
        )


@pytest.mark.parametrize(
    ("column", "match"),
    [("temp_air", "no air temperature column"), ("wind_speed", "no wind speed column")],
)
def test_weather_without_a_met_column_raises(synthetic_weather, porto_location, pv_params, column, match):
    # These used to become 25 °C and 1 m/s without a message.
    with pytest.raises(ValueError, match=match):
        _dc(synthetic_weather.drop(columns=column), porto_location, pv_params)


def test_explicit_constant_met_columns_run(synthetic_weather, porto_location, pv_params):
    weather = synthetic_weather.drop(columns=["temp_air", "wind_speed"])
    weather["temp_air"] = 25.0
    weather["wind_speed"] = 1.0

    assert _dc(weather, porto_location, pv_params).sum() > 0


def test_non_finite_air_temperature_raises(synthetic_weather, porto_location, pv_params):
    # A NaN air temperature used to give a NaN cell temperature, replaced by 25 °C.
    weather = synthetic_weather.copy()
    weather.iloc[4000, weather.columns.get_loc("temp_air")] = np.nan

    with pytest.raises(ValueError, match=r"'temp_air' has 1 values that are not finite .* 2023-06-16 16:00"):
        _dc(weather, porto_location, pv_params)


def test_battery_weather_temperature_without_a_column_raises(synthetic_weather):
    weather = synthetic_weather.drop(columns="temp_air")

    with pytest.raises(ValueError, match="needs an air temperature column"):
        build_battery_temperature_series("weather", synthetic_weather.index, weather_df=weather)
    # The explicit fixed temperature is the documented alternative.
    fixed = build_battery_temperature_series(25.0, synthetic_weather.index, indoor_model={"enabled": False})
    assert (fixed == 25.0).all()


def _write_year(path, stamps):
    frame = pd.DataFrame({"date": stamps, "ghi": 0.0, "dni": 0.0, "dhi": 0.0, "temperature_2m": 10.0})
    frame.to_csv(path, index=False)


def test_naive_weather_file_timestamps_are_warned_about(tmp_path, caplog):
    path = tmp_path / "naive.csv"
    _write_year(path, pd.date_range("2023-01-01", periods=8760, freq="h").strftime("%Y-%m-%d %H:%M"))

    with caplog.at_level(logging.WARNING, logger="breos.weather"):
        years, _ = _complete_weather_years(str(path))

    assert list(years) == [2023]
    assert "timestamps without a timezone; BREOS reads them as UTC" in caplog.text


def test_weather_file_with_utc_offsets_is_not_warned_about(tmp_path, caplog):
    path = tmp_path / "aware.csv"
    _write_year(path, pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M%z"))

    with caplog.at_level(logging.WARNING, logger="breos.weather"):
        _complete_weather_years(str(path))

    assert "without a timezone" not in caplog.text


@pytest.fixture
def tiny_weather(synthetic_weather):
    return synthetic_weather.loc["2023-06-16"].copy()


def test_dkasc_klucher_inconsistency_reports_weather_error(tiny_weather, porto_location, pv_params):
    tiny_weather.index = pd.date_range("2012-12-14 09:00", periods=24, freq="15min", tz="Australia/Darwin")
    stamp = tiny_weather.index[11]
    tiny_weather.loc[stamp, ["ghi", "dhi"]] = [0, 867.347]
    with pytest.raises(ValueError) as exc:
        _dc(tiny_weather, porto_location, pv_params, freq="15min", transposition_model="klucher")
    assert str(exc.value) == (
        "weather columns 'ghi' and 'dhi' have 1 rows with inconsistent irradiance "
        "(first at 2012-12-14 11:45:00+09:30: GHI=0, DHI=867.347 W/m²). "
        "DHI must not exceed GHI by more than max(50 W/m², 10% of GHI), or exceed 20 W/m² when GHI is "
        "at most 1 W/m². Repair these values explicitly before computing PV, preserving a regular time grid."
    )


@pytest.mark.parametrize("model", TRANSPOSITION_MODELS)
def test_irradiance_consistency_is_model_independent(tiny_weather, porto_location, pv_params, model):
    tiny_weather.loc[tiny_weather.index[12], ["ghi", "dhi"]] = [0, 867.347]
    with pytest.raises(ValueError, match="inconsistent irradiance"):
        _dc(tiny_weather, porto_location, pv_params, transposition_model=model)


@pytest.mark.parametrize("ghi,dhi", [(200, 250), (1000, 1100), (1, 20), (0, 20), (-1, 20)])
def test_irradiance_tolerance_boundaries(tiny_weather, ghi, dhi):
    tiny_weather[["ghi", "dhi"]] = [float(ghi), float(dhi)]
    validate_weather_irradiance(tiny_weather)
    tiny_weather.loc[tiny_weather.index[12], "dhi"] = np.nextafter(float(dhi), np.inf)
    with pytest.raises(ValueError, match="have 1 rows with inconsistent irradiance"):
        validate_weather_irradiance(tiny_weather)
    if ghi == 1:
        tiny_weather[["ghi", "dhi"]] = [np.nextafter(1.0, np.inf), 21]
        validate_weather_irradiance(tiny_weather)


@pytest.mark.parametrize("ghi,dhi", [(0, 5), (0.07, 5.027), (200, 225), (600, 608)])
def test_measured_irradiance_noise_is_preserved(tiny_weather, ghi, dhi):
    tiny_weather[["ghi", "dhi"]] = [float(ghi), float(dhi)]
    original = tiny_weather.copy(deep=True)
    dni_values, ghi_values, dhi_values = _extract_irradiance(tiny_weather)
    np.testing.assert_array_equal(ghi_values, original.ghi)
    np.testing.assert_array_equal(dhi_values, original.dhi)
    np.testing.assert_array_equal(dni_values, original.dni)
    pd.testing.assert_frame_equal(tiny_weather, original)


@pytest.mark.parametrize("column", ["ghi", "dni", "dhi"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "missing", pd.NA])
@pytest.mark.parametrize("row", [0, 12], ids=["night", "day"])
def test_nonfinite_irradiance_reports_column_count_and_timestamp(
    tiny_weather, porto_location, pv_params, column, bad, row
):
    tiny_weather[column] = tiny_weather[column].astype(object)
    tiny_weather.loc[tiny_weather.index[row], column] = bad
    tiny_weather.loc[tiny_weather.index[row + 1], column] = bad
    original = tiny_weather.copy(deep=True)
    with pytest.raises(ValueError) as exc:
        _dc(tiny_weather, porto_location, pv_params)
    assert str(exc.value) == (
        f"weather column '{column}' has 2 values that are not finite numbers "
        f"(first at {tiny_weather.index[row]}). Fill them explicitly before computing PV."
    )
    pd.testing.assert_frame_equal(tiny_weather, original)


@pytest.mark.parametrize("model", TRANSPOSITION_MODELS)
def test_dark_zero_irradiance_is_valid(tiny_weather, porto_location, pv_params, model):
    tiny_weather[["ghi", "dni", "dhi"]] = 0.0
    power = _dc(tiny_weather, porto_location, pv_params, transposition_model=model)
    assert (power == 0).all()


def test_night_inconsistency_is_not_exempt(tiny_weather, porto_location, pv_params):
    assert porto_location.get_solarposition(tiny_weather.index[:1]).apparent_elevation.iloc[0] < 0
    tiny_weather.loc[tiny_weather.index[0], ["ghi", "dhi"]] = [0, 100]
    with pytest.raises(ValueError, match="inconsistent irradiance"):
        _dc(tiny_weather, porto_location, pv_params)


@pytest.mark.parametrize("row", [0, 12], ids=["night", "day"])
def test_klucher_zero_ghi_small_positive_dhi_reports_domain_error(tiny_weather, porto_location, pv_params, row):
    tiny_weather.loc[tiny_weather.index[row], ["ghi", "dhi"]] = [0, 0.1]
    with pytest.raises(ValueError) as exc:
        _dc(tiny_weather, porto_location, pv_params, transposition_model="klucher")
    assert str(exc.value) == (
        f"weather has 1 rows with GHI=0 and DHI>0 (first at {tiny_weather.index[row]}). "
        "Klucher transposition divides DHI by GHI. Repair these values explicitly before computing PV."
    )
    assert np.isfinite(_dc(tiny_weather, porto_location, pv_params, transposition_model="isotropic")).all()


@pytest.mark.parametrize(
    "aliases",
    [
        {"ghi": "GHI", "dni": "DNI", "dhi": "DHI"},
        {"ghi": "shortwave_radiation", "dni": "direct_normal_irradiance", "dhi": "diffuse_radiation"},
        {
            "ghi": "global_horizontal_irradiance",
            "dni": "direct_normal_irradiance",
            "dhi": "diffuse_horizontal_irradiance",
        },
    ],
)
def test_irradiance_aliases_and_input_immutability(tiny_weather, porto_location, pv_params, aliases):
    weather = tiny_weather.rename(columns=aliases)
    original = weather.copy(deep=True)
    assert _dc(weather, porto_location, pv_params).sum() > 0
    pd.testing.assert_frame_equal(weather, original)
    weather.loc[weather.index[12], [aliases["ghi"], aliases["dhi"]]] = [0, 867.347]
    original = weather.copy(deep=True)
    with pytest.raises(ValueError, match=re.escape(f"weather columns '{aliases['ghi']}' and '{aliases['dhi']}'")):
        _dc(weather, porto_location, pv_params)
    pd.testing.assert_frame_equal(weather, original)


@pytest.mark.parametrize(
    "entrypoint", ["fixed", "single_axis", "dual_axis", "multi_array", "app", "montecarlo", "optimizer", "forecast"]
)
@pytest.mark.parametrize("invalid", ["inconsistent", "nonfinite"])
def test_shared_weather_validation_reaches_all_pv_entrypoints(
    tiny_weather, porto_location, pv_params, monkeypatch, entrypoint, invalid
):
    import breos.montecarlo as mc
    from breos import App
    from breos.app_config import resolve_app_config
    from breos.app_inputs import prepare_planner_forecast
    from breos.optimization import evaluate_projected_design
    from tests.offline_app import serve_app_inputs

    if invalid == "inconsistent":
        tiny_weather.loc[tiny_weather.index[12], ["ghi", "dhi"]] = [0, 867.347]
        message = "inconsistent irradiance"
    else:
        tiny_weather.loc[tiny_weather.index[12], "dni"] = np.nan
        message = "not finite numbers"
    load = pd.DataFrame({"Load": 500.0}, index=tiny_weather.index)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "n_modules": 1,
        "annual_consumption_kwh": 4000,
        "start_date": "2023-01-01",
        "projection_years": 1,
        "resolution": "h",
        "period": {"start": "2023-06-16", "end": "2023-06-17"},
        "execution_backend": "python",
    }
    with pytest.raises(ValueError, match=message):
        if entrypoint == "fixed":
            _dc(tiny_weather, porto_location, pv_params)
        elif entrypoint in ("single_axis", "dual_axis"):
            calculate_pv_production_dc_tracking(
                tiny_weather, porto_location, n_modules=1, pv_params=pv_params, tracking=entrypoint
            )
        elif entrypoint == "multi_array":
            calculate_multi_array_production(tiny_weather, porto_location, [{"modules": 1, "tilt": 35, "azimuth": 180}])
        elif entrypoint == "app":
            serve_app_inputs(monkeypatch, tiny_weather, load)
            config.pop("projection_years")
            App(config).simulate()
        elif entrypoint == "montecarlo":
            config.pop("period")
            monkeypatch.setattr(mc, "_weather_years", lambda *a, **k: {2023: tiny_weather})
            monkeypatch.setattr(mc, "_weather_file_sha256", lambda path: "in-memory")
            mc.build_year_cache(config, mc.MonteCarloSettings(weather_file="in-memory.csv", n_runs=1, years_per_run=1))
        elif entrypoint == "optimizer":
            evaluate_projected_design(
                tiny_weather,
                load,
                {
                    "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
                    "simulation": {"resolution": "h"},
                },
                n_modules=1,
                battery_kwh=0,
                tilt=35,
                azimuth=180,
            )
        else:
            # Keep forecast loading in memory; run its real preparation/PV path.
            resolved = resolve_app_config(config)
            resolved = replace(
                resolved,
                smart_charging=SimpleNamespace(forecast="weather", forecast_weather_files=("lead.csv",)),
            )
            monkeypatch.setattr("breos.weather.read_weather_csv", lambda path: tiny_weather.copy())
            monkeypatch.setattr("breos.app_inputs.load_weather_for_simulation", lambda *a, **k: tiny_weather.copy())
            prepare_planner_forecast(
                resolved.cfg,
                resolved,
                None,
                None,
                tiny_weather.index,
            )


@pytest.mark.parametrize("shaded", [False, True])
@pytest.mark.parametrize("diffuse_iam", ["none", "marion"])
def test_finite_measured_noise_still_needs_poa_nan_cleanup(
    tiny_weather, porto_location, monkeypatch, shaded, diffuse_iam
):
    import pvlib

    original = pvlib.irradiance.get_total_irradiance
    raw_poa = []

    def capture(*args, **kwargs):
        poa = original(*args, **kwargs)
        raw_poa.append(poa.copy())
        return poa

    monkeypatch.setattr(pvlib.irradiance, "get_total_irradiance", capture)
    # Small negative sensor readings pass the weather tolerances. Reindl's
    # sqrt(direct_horizontal / GHI) still produces NaN diffuse POA here.
    stamp = tiny_weather.index[12]
    tiny_weather.loc[stamp, ["ghi", "dni", "dhi"]] = [-0.1, 0.1, 0.1]
    array = {"modules": 1, "tilt": 35, "azimuth": 180}
    if shaded:
        array["shading"] = {"azimuth_deg": [0], "elevation_deg": [0], "beam_shaded_fraction": [[0.1]]}
    with np.errstate(invalid="ignore"):
        power = calculate_multi_array_production(
            tiny_weather, porto_location, [array], transposition_model="reindl", diffuse_iam=diffuse_iam
        )
    assert np.isfinite(tiny_weather.to_numpy()).all()
    assert raw_poa[0].loc[stamp, ["poa_global", "poa_diffuse", "poa_sky_diffuse"]].isna().all()
    assert np.isfinite(power).all()
    assert power.sum() > 0
