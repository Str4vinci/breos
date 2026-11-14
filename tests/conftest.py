"""Shared fixtures for BREOS test suite."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos import tariffs
from breos.battery import BatteryConfig
from breos.economics import CostParams
from breos.emissions import EmissionsParams
from breos.load_profiles import load_profile
from breos.pv_modules import get_module
from breos.solar import calculate_pv_production_dc
from breos.weather import WEATHER_METADATA_KEY, apply_indoor_temperature_model, extract_ambient_temperature


@pytest.fixture(autouse=True)
def _fresh_tariff_classifications():
    """Classify afresh in every test, so a test that patches a tariff check sees it run."""
    tariffs._CLASSIFICATIONS.clear()


# ---------------------------------------------------------------------------
# Synthetic weather (1 year, hourly, no API call)
# ---------------------------------------------------------------------------

# The time basis the synthetic weather declares: instant samples at their
# labels. With it, solar_position = "auto" (the default) resolves to
# "weather" with a zero offset, so the sun is where "interval-start" puts it,
# bit for bit, and auto does not warn; the 15-minute resampler reads it as
# it reads weather that declares nothing.
INSTANT_AT_LABELS = {
    "radiation_time_basis": "instant",
    "timestamp_label_basis": "instant",
    "irradiance_time_offset_hours": 0.0,
}


def declare_instant_at_labels(weather: pd.DataFrame) -> pd.DataFrame:
    """Add :data:`INSTANT_AT_LABELS` to ``weather``'s metadata, in place, and return it."""
    metadata = dict(weather.attrs.get(WEATHER_METADATA_KEY) or {})
    weather.attrs[WEATHER_METADATA_KEY] = {**metadata, **INSTANT_AT_LABELS}
    return weather


def write_instant_at_labels_sidecar(path) -> None:
    """Write a valid metadata sidecar declaring :data:`INSTANT_AT_LABELS` for the weather CSV at ``path``."""
    payload = {
        "schema_version": 1,
        "weather_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        WEATHER_METADATA_KEY: dict(INSTANT_AT_LABELS),
    }
    Path(f"{path}.metadata.json").write_text(json.dumps(payload))


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

    return declare_instant_at_labels(
        pd.DataFrame(
            {"ghi": ghi, "dni": dni, "dhi": dhi, "temp_air": temp_air, "wind_speed": wind_speed},
            index=index,
        )
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
    return declare_instant_at_labels(
        pd.DataFrame(
            {
                "temperature_2m": 15.0 + 8.0 * daylight,
                "wind_speed_10m": 2.0,
                "shortwave_radiation": ghi,
                "direct_normal_irradiance": 0.8 * ghi,
                "diffuse_radiation": 0.2 * ghi,
            },
            index=index,
        )
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
    write_instant_at_labels_sidecar(path)
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
# Stubbed projection year (optimizer wiring tests)
# ---------------------------------------------------------------------------


def _stub_projection_balance(monkeypatch, index, captured=None, **columns):
    """Replace the projection loop's dispatch with a fixed per-step ledger.

    Every column the year row sums is zero on ``index`` unless ``columns``
    sets it, in W. Each call's keyword arguments are stored in ``captured``
    when given, so a test can check what the optimizer asked the loop to run.
    """
    from breos.projection import _ROW_SUM_COLUMNS

    frame = pd.DataFrame(
        0.0,
        index=index,
        columns=[
            *_ROW_SUM_COLUMNS,
            "Battery_Energy_End",
            "Battery_PV_Origin_Energy_End",
            "Battery_Grid_Origin_Energy_End",
            "Battery_Replaced",
            "Battery_Replaced_Capacity_Wh",
        ],
    )
    for name, values in columns.items():
        frame[name] = values

    def fake_balance(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return frame, 0.0, pd.DataFrame(), 0, pd.DataFrame(), None

    monkeypatch.setattr("breos.projection.simulate_energy_balance", fake_balance)
    return frame


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
        profile_type="demandlib_h0",
        annual_consumption_kwh=3000,
        start_date="2023-01-01",
        freq="h",
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
    return BatteryConfig(nominal_energy_wh=5000)


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
            **INSTANT_AT_LABELS,
            "horizon": {
                "status": "applied" if use_horizon else "not_applied",
                "provider": "pvgis",
                "profile": "provider_default" if use_horizon else None,
            },
        }
        return weather, {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _fake_fetch)
    monkeypatch.setattr("breos.app.load_weather", lambda **kw: None)  # force fetch path


# ---------------------------------------------------------------------------
# Guard: App must not fetch weather
# ---------------------------------------------------------------------------


@pytest.fixture
def no_fetch(monkeypatch):
    """Fail if App falls back to a PVGIS fetch, for tests that serve weather from local files."""

    def _fail(*args, **kwargs):
        raise AssertionError("App fetched PVGIS weather instead of reading a local weather file")

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _fail)


# ---------------------------------------------------------------------------
# Shared optimizer and Monte Carlo cases
# ---------------------------------------------------------------------------


@pytest.fixture
def tou_optimizer_case(monkeypatch):
    """Two Lisbon winter days for the projected optimizer: stubbed PV, 1 kW load, daily bi-hourly prices."""
    from breos import optimization

    index = pd.date_range("2026-01-05", periods=48, freq="h", tz="Europe/Lisbon")
    weather = declare_instant_at_labels(pd.DataFrame({"temp_air": 20.0}, index=index))
    load = pd.DataFrame({"Load": 1000.0}, index=index)
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 16), 1800.0, 0.0), index=index)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "Europe/Lisbon", "altitude": 100},
        "simulation": {"resolution": "h", "years_projection": 2},
        "financials": {"inflation_rate": 0.02, "discount_rate": 0.05},
        "battery": {"temperature": 20.0},
        "mode": {"fixed_azimuth": 180.0},
        "constraints": {"budget": 100000, "max_area_m2": 100, "max_tilt_deg": 60},
        "tariff": {
            "schedule": "pt_mainland_2026_daily_bi",
            "currency": "EUR",
            "import_prices": {"peak": 0.50, "off_peak": 0.10},
            "export_prices": {"all": 0.03},
            "fixed_charge_per_day": 0.40,
        },
    }
    return weather, load, config


@pytest.fixture
def small_mc_settings(tmp_path):
    """Two Monte Carlo runs of two years over a two-year weather file, with yearly rows."""
    from breos.montecarlo import MonteCarloSettings

    weather = _write_multiyear_weather(tmp_path / "multi.csv")
    return MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3, collect_yearly=True)


# ---------------------------------------------------------------------------
# Sourced module datasheets in the catalog
# ---------------------------------------------------------------------------


@pytest.fixture
def trina(monkeypatch):
    """Register the sourced Trina 450 W module in the catalog and return its key."""
    from breos.pv_modules import MODULES
    from tests.sourced_modules import TRINA_450, TRINA_KEY

    monkeypatch.setitem(MODULES, TRINA_KEY, TRINA_450)
    return TRINA_KEY


@pytest.fixture
def jinko(monkeypatch):
    """Register the sourced Jinko 440 W module in the catalog and return its key."""
    from breos.pv_modules import MODULES
    from tests.sourced_modules import JINKO_440, JINKO_KEY

    monkeypatch.setitem(MODULES, JINKO_KEY, JINKO_440)
    return JINKO_KEY


@pytest.fixture(autouse=True)
def _fresh_solar_positions():
    """Each test computes the sun afresh, so no test depends on which ran before it in the process."""
    from breos.pv import model_options

    model_options._SOLAR_POSITIONS.clear()
