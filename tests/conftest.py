"""Shared fixtures for BREOS test suite."""

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos.battery import BatteryConfig, apply_indoor_temperature_model
from breos.economics import CostParams
from breos.emissions import EmissionsParams
from breos.load_profiles import load_profile
from breos.pv_modules import get_module
from breos.solar import PVModuleParams, calculate_pv_production_dc
from breos.weather import extract_ambient_temperature

# ---------------------------------------------------------------------------
# Synthetic weather (1 year, hourly, no API call)
# ---------------------------------------------------------------------------


def _build_synthetic_weather(year: int = 2023, freq: str = "h") -> pd.DataFrame:
    """Build a realistic-ish 1-year weather DataFrame with sinusoidal patterns."""
    steps_per_hour = 1 if freq == "h" else 4
    n_steps = 8760 * steps_per_hour
    index = pd.date_range(
        start=f"{year}-01-01",
        periods=n_steps,
        freq="h" if freq == "h" else "15min",
        tz="UTC",
    )

    hour_of_year = np.arange(n_steps, dtype=float) / steps_per_hour
    day_of_year = hour_of_year / 24.0
    hour_of_day = hour_of_year % 24

    # Solar pattern: bell curve during daylight, zero at night
    solar_angle = np.clip(np.sin((hour_of_day - 6) / 12 * np.pi), 0, 1)
    # Seasonal modulation: summer has more sun
    seasonal = 0.6 + 0.4 * np.sin((day_of_year - 80) / 365 * 2 * np.pi)
    ghi = solar_angle * seasonal * 800  # W/m2 peak
    dni = ghi * 0.7
    dhi = ghi * 0.3

    # Temperature: seasonal + diurnal
    temp_air = 15 + 8 * np.sin((day_of_year - 80) / 365 * 2 * np.pi) + 4 * np.sin((hour_of_day - 14) / 24 * 2 * np.pi)
    wind_speed = np.full(n_steps, 3.0)

    return pd.DataFrame(
        {"ghi": ghi, "dni": dni, "dhi": dhi, "temp_air": temp_air, "wind_speed": wind_speed},
        index=index,
    )


@pytest.fixture
def synthetic_weather():
    return _build_synthetic_weather()


@pytest.fixture
def synthetic_weather_15min():
    return _build_synthetic_weather(freq="15min")


# ---------------------------------------------------------------------------
# Synthetic multi-year historical weather (Open-Meteo columns, no API call)
# ---------------------------------------------------------------------------


def _build_open_meteo_weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a daytime-bell weather frame on *index* with Open-Meteo column names.

    breos.solar recognizes these names (shortwave_radiation /
    direct_normal_irradiance / diffuse_radiation + temperature_2m /
    wind_speed_10m), as it does for a historical Open-Meteo download.
    """
    hour = index.hour.to_numpy()
    # Simple daytime bell centered at noon.
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    ghi = 700.0 * daylight
    return pd.DataFrame(
        {
            "temperature_2m": 15.0 + 8.0 * daylight,
            "wind_speed_10m": 2.0,
            "shortwave_radiation": ghi,
            "direct_normal_irradiance": 0.8 * ghi,
            "diffuse_radiation": 0.2 * ghi,
        },
        index=index,
    )


def _write_multiyear_weather(path, years=(2021, 2022)):
    """Write a small synthetic multi-year hourly weather CSV with a ``date`` column."""
    frames = []
    for year in years:
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        idx = idx[~((idx.month == 2) & (idx.day == 29))]  # keep 8760 rows/year
        weather = _build_open_meteo_weather(idx)
        weather.insert(0, "date", idx)
        frames.append(weather)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


@pytest.fixture
def open_meteo_weather():
    """Return the builder: ``open_meteo_weather(index) -> DataFrame``."""
    return _build_open_meteo_weather


@pytest.fixture
def write_multiyear_weather():
    """Return the writer: ``write_multiyear_weather(path, years=(2021, 2022)) -> path``."""
    return _write_multiyear_weather


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


@pytest.fixture
def porto_location():
    return Location(41.1579, -8.6291, tz="Europe/Lisbon")


# ---------------------------------------------------------------------------
# PV module
# ---------------------------------------------------------------------------


@pytest.fixture
def pv_params():
    return get_module("Suntech_STP550S_STC")


# ---------------------------------------------------------------------------
# Load profile
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_load():
    return load_profile(
        profile_type="1",
        annual_consumption_kwh=3000,
        start_date="2023-01-01",
        freq="h",
        num_years=1,
        timezone="UTC",
    )


# ---------------------------------------------------------------------------
# PV DC production (1 module, hourly, synthetic weather)
# ---------------------------------------------------------------------------


@pytest.fixture
def dc_production(synthetic_weather, porto_location, pv_params):
    return calculate_pv_production_dc(
        weather_data=synthetic_weather,
        location=porto_location,
        tilt=35,
        surface_azimuth=180,
        n_modules=1,
        pv_params=pv_params,
        freq="h",
    )


# ---------------------------------------------------------------------------
# Battery config
# ---------------------------------------------------------------------------


@pytest.fixture
def battery_config():
    return BatteryConfig(nominal_energy_wh=5000, battery_type="lfp")


# ---------------------------------------------------------------------------
# Temperature series
# ---------------------------------------------------------------------------


@pytest.fixture
def temperature_series(synthetic_weather):
    ambient = extract_ambient_temperature(synthetic_weather)
    return apply_indoor_temperature_model(ambient)


# ---------------------------------------------------------------------------
# Cost params (residential_pt style)
# ---------------------------------------------------------------------------


@pytest.fixture
def cost_params():
    return CostParams(
        electricity_cost=0.2582,
        electricity_sold_cost=0.04,
        module_cost_per_w=0.125,
        battery_cost_per_kwh=500.0,
        dc_ac_ratio=1.25,
        inverter_cost_per_kw=102.58,
        inverter_cost_per_kw_nobatt=48.37,
        installation_cost_per_module=350.0,
        battery_installation_cost=350.0,
        maintenance_cost_per_panel=10.0,
        other_cost_per_module=50.0,
        inflation_rate=0.02,
        discount_rate=0.03,
        pv_degradation_rate=0.005,
    )


# ---------------------------------------------------------------------------
# Emissions params
# ---------------------------------------------------------------------------


@pytest.fixture
def emissions_params():
    return EmissionsParams(
        average_grid_carbon_intensity_gco2_kwh=127.91,
        year=2025,
        country="Portugal",
    )


# ---------------------------------------------------------------------------
# Helper: monkeypatch weather for App tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _patch_weather(monkeypatch, synthetic_weather):
    """Monkeypatch fetch_tmy_weather_data so App tests never hit the network."""

    def _fake_fetch(*args, **kwargs):
        weather = synthetic_weather.copy()
        use_horizon = kwargs.get("use_horizon", True)
        weather.attrs["breos_weather_metadata"] = {
            "source": "PVGIS_TMY",
            "horizon": {
                "status": "applied" if use_horizon else "not_applied",
                "provider": "pvgis",
                "profile": "provider_default" if use_horizon else None,
            },
        }
        return weather, {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _fake_fetch)
    monkeypatch.setattr("breos.app.load_weather", lambda **kw: None)  # force fetch path
