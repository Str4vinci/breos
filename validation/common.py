"""Shared paths and helpers for the validation suite.

Used by the validation scripts and by ``tests/test_validation_drift.py`` so
that the baseline generator and the drift test load weather identically.
"""

import json
from pathlib import Path

import pandas as pd
import pvlib
from pvlib.location import Location

from breos.pv_modules import get_module
from breos.solar import calculate_pv_production_ac

VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parent
WEATHER_DIR = VALIDATION_DIR / "data" / "weather"
REFERENCES_DIR = VALIDATION_DIR / "data" / "references"
BASELINE_PATH = VALIDATION_DIR / "baselines" / "breos_baseline.json"
RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "breos_results.json"
REPORT_PATH = VALIDATION_DIR / "REPORT.md"

MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Model configs the suite runs BREOS with. isotropic = shipped default;
# perez = what the external references effectively use; perez_mid = perez plus
# mid-interval solar position (the full PVWatts/SAM convention); perez_diffuse
# = perez plus Marion diffuse IAM; perez_physical and perez_martin_ruiz select
# alternative beam/diffuse optics; perez_roof = perez with the PVsyst
# semi-integrated (close roof mount) thermal preset; perez_sapm_roof is the
# comparable Sandia close-mount glass/glass preset. Both measure rooftop yield
# deltas rather than matching the free-standing references. Keys are the
# model names stored in results/baseline JSONs; values are
# calculate_pv_production_ac keyword overrides. ``pv_module`` and ``n_modules``
# are consumed by the harness before calling that function, allowing
# same-weather and equal-nameplate comparisons. The drift test recomputes from
# this same mapping.
MODEL_CONFIGS = {
    "isotropic": {"transposition_model": "isotropic"},
    "perez": {"transposition_model": "perez"},
    "perez_mid": {"transposition_model": "perez", "solar_position": "mid-interval"},
    "perez_diffuse": {"transposition_model": "perez", "diffuse_iam": "marion"},
    "perez_physical": {"transposition_model": "perez", "iam_model": "physical", "diffuse_iam": "marion"},
    "perez_martin_ruiz": {
        "transposition_model": "perez",
        "iam_model": "martin_ruiz",
        "diffuse_iam": "marion",
    },
    "perez_roof": {"transposition_model": "perez", "temperature_model": "pvsyst-semi-integrated"},
    "perez_sapm_roof": {"transposition_model": "perez", "temperature_model": "sapm-close-mount-glass-glass"},
    "bifacial_front": {"pv_module": "Generic_600W_Bifacial", "transposition_model": "perez"},
    "bifacial_rear": {
        "pv_module": "Generic_600W_Bifacial",
        "transposition_model": "perez",
        "bifacial_model": "infinite_sheds",
        "gcr": 0.35,
        "pvrow_height": 1.5,
        "pvrow_pitch": 6.0,
    },
    # Recommended common front-side settings at equal 1.2 kWp STC nameplate.
    # The front-only bifacial row separates module-model differences from the
    # rear-gain delta of the otherwise identical infinite-sheds row.
    "nameplate_mono_1200": {
        "pv_module": "Generic_400W",
        "n_modules": 3,
        "transposition_model": "perez",
        "solar_position": "mid-interval",
        "iam_model": "physical",
        "diffuse_iam": "marion",
    },
    "nameplate_bifacial_front_1200": {
        "pv_module": "Generic_600W_Bifacial",
        "n_modules": 2,
        "transposition_model": "perez",
        "solar_position": "mid-interval",
        "iam_model": "physical",
        "diffuse_iam": "marion",
    },
    "nameplate_bifacial_rear_1200": {
        "pv_module": "Generic_600W_Bifacial",
        "n_modules": 2,
        "transposition_model": "perez",
        "solar_position": "mid-interval",
        "iam_model": "physical",
        "diffuse_iam": "marion",
        "bifacial_model": "infinite_sheds",
        "gcr": 0.35,
        "pvrow_height": 1.5,
        "pvrow_pitch": 6.0,
    },
}

# The baseline and REPORT.md put the sun at the step labels, so every model
# without its own solar_position pins "interval-start" rather than following
# the solar_position default, which reads each weather file's metadata
# sidecar and places the sun at the recorded irradiance instant.
MODEL_CONFIGS = {name: {"solar_position": "interval-start", **overrides} for name, overrides in MODEL_CONFIGS.items()}


def compute_annual_poa(weather: pd.DataFrame, loc: dict, system: dict) -> float:
    """Independent pvlib Perez front-side irradiation (kWh/m²), once per site.

    Use the weather timestamps without a shift, and the same Perez POA for
    every BREOS variant so transposition changes cannot cancel out of PR.
    """
    solarpos = pvlib.solarposition.get_solarposition(weather.index, loc["latitude"], loc["longitude"])
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=loc["tilt"],
        surface_azimuth=loc["azimuth"],
        solar_zenith=solarpos["apparent_zenith"],
        solar_azimuth=solarpos["azimuth"],
        ghi=weather["ghi"],
        dni=weather["dni"],
        dhi=weather["dhi"],
        dni_extra=pvlib.irradiance.get_extra_radiation(weather.index),
        airmass=pvlib.atmosphere.get_relative_airmass(solarpos["apparent_zenith"]),
        model="perez",
        albedo=system["albedo"],
    )
    return float(poa["poa_global"].fillna(0.0).sum()) / 1000.0


def compute_model_result(
    weather: pd.DataFrame, loc: dict, system: dict, model_kwargs: dict, annual_poa_kwh_m2: float
) -> dict:
    """Run the public AC path with independent pvlib Perez POA supplied per site."""
    model_kwargs = dict(model_kwargs)
    pv_params = get_module(model_kwargs.pop("pv_module", system["module"]))
    n_modules = model_kwargs.pop("n_modules", system["n_modules"])
    nameplate_w = n_modules * pv_params.Mpp
    ac = calculate_pv_production_ac(
        weather_data=weather,
        location=Location(loc["latitude"], loc["longitude"], tz=loc["timezone"]),
        tilt=loc["tilt"],
        surface_azimuth=loc["azimuth"],
        n_modules=n_modules,
        pv_params=pv_params,
        freq="h",
        inverter_loading_ratio=system["dc_ac_ratio"],
        inverter_efficiency=system["inverter_efficiency"],
        albedo=system["albedo"],
        **model_kwargs,
    )
    monthly = ac.groupby(ac.index.month).sum() / 1000.0
    return {
        "monthly_kwh": [float(monthly.get(m, 0.0)) for m in range(1, 13)],
        "annual_kwh": float(ac.sum()) / 1000.0,
        "annual_poa_kwh_m2": annual_poa_kwh_m2,
        "dc_nameplate_kw": nameplate_w / 1000.0,
    }


def load_spec():
    """Return (system_spec, locations) from validation/locations.json."""
    with open(VALIDATION_DIR / "locations.json") as f:
        cfg = json.load(f)
    return cfg["system"], cfg["locations"]


def find_weather_file(location_key: str):
    """Return the checked-in TMY weather file for a location, or None."""
    if not WEATHER_DIR.is_dir():
        return None
    matches = sorted(WEATHER_DIR.glob(f"{location_key}_tmy_*.csv*"))
    return matches[0] if matches else None


def load_validation_weather(filepath) -> pd.DataFrame:
    """Load a checked-in validation weather CSV with a tz-aware index.

    Timestamps are converted to UTC (same instants, so solar position is
    unaffected); monthly aggregation therefore uses UTC month boundaries,
    which only moves zero-production night hours between months.
    """
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_reference(location_key: str):
    """Return the reference JSON for a location, or None if not fetched."""
    path = REFERENCES_DIR / f"{location_key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
