"""Drift tests for the standing validation suite (see validation/README.md).

Two layers:

1. **Regression** — BREOS run on the checked-in validation weather must match
   the committed baseline within 0.1%. This catches unintended changes to the
   PV chain anywhere between irradiance transposition and the inverter curve.
   If a change is *supposed* to move results, regenerate the baseline with
   ``uv run python validation/run_breos.py --write-baseline`` and say so in
   the changelog.
2. **Validation band** — BREOS (perez) annual yield must stay within ±10% of
   the independent PVGIS PVcalc reference for every location. The band is
   loose because the references use different underlying weather (TMY vs the
   full multi-year record); it exists to catch gross model errors, not to
   prove accuracy. Same idea for PVWatts v8 (±15%) where a trusted reference
   has been fetched.
"""

import json
import sys
from pathlib import Path

import pytest
from pvlib.location import Location

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "validation"
BASELINE_PATH = VALIDATION_DIR / "baselines" / "breos_baseline.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.skipif(
    not BASELINE_PATH.exists(), reason="validation baseline not present (see validation/README.md)"
)

REGRESSION_RTOL = 1e-3  # 0.1% — cross-platform numerical headroom, catches real drift
MONTHLY_ABS_FLOOR_KWH = 0.5  # winter months are small; don't turn noise into failures
PVGIS_BAND = 0.10
PVWATTS_BAND = 0.15


def _load_json(path):
    with open(path) as f:
        return json.load(f)


if BASELINE_PATH.exists():
    _BASELINE = _load_json(BASELINE_PATH)
    _CASES = [
        (key, model)
        for key, entry in _BASELINE["locations"].items()
        for model in entry["models"]
        if (VALIDATION_DIR / "data" / "weather" / entry["weather_file"]).exists()
    ]
    _KEYS = sorted({key for key, _ in _CASES})
else:  # pragma: no cover - guarded by pytestmark
    _BASELINE, _CASES, _KEYS = None, [], []

_RESULT_CACHE = {}


def _compute(key, model):
    """Recompute BREOS production for a baseline case (cached per session)."""
    if (key, model) in _RESULT_CACHE:
        return _RESULT_CACHE[(key, model)]

    from breos.pv_modules import get_module
    from breos.solar import calculate_pv_production_ac
    from validation.common import MODEL_CONFIGS, load_spec, load_validation_weather

    system, locations = load_spec()
    loc = locations[key]
    entry = _BASELINE["locations"][key]
    weather = load_validation_weather(VALIDATION_DIR / "data" / "weather" / entry["weather_file"])

    ac = calculate_pv_production_ac(
        weather_data=weather,
        location=Location(loc["latitude"], loc["longitude"], tz=loc["timezone"]),
        tilt=loc["tilt"],
        surface_azimuth=loc["azimuth"],
        n_modules=system["n_modules"],
        pv_params=get_module(system["module"]),
        freq="h",
        inverter_loading_ratio=system["dc_ac_ratio"],
        inverter_efficiency=system["inverter_efficiency"],
        albedo=system["albedo"],
        **MODEL_CONFIGS[model],
    )
    monthly = ac.groupby(ac.index.month).sum() / 1000.0
    result = {
        "monthly_kwh": [float(monthly.get(m, 0.0)) for m in range(1, 13)],
        "annual_kwh": float(ac.sum()) / 1000.0,
    }
    _RESULT_CACHE[(key, model)] = result
    return result


@pytest.mark.parametrize("key,model", _CASES)
def test_baseline_regression(key, model):
    computed = _compute(key, model)
    expected = _BASELINE["locations"][key]["models"][model]

    assert computed["annual_kwh"] == pytest.approx(expected["annual_kwh"], rel=REGRESSION_RTOL), (
        f"{key}/{model}: annual yield drifted from the committed baseline. If this change is "
        f"intentional, regenerate it with 'validation/run_breos.py --write-baseline'."
    )
    for month_idx, (got, want) in enumerate(zip(computed["monthly_kwh"], expected["monthly_kwh"]), start=1):
        tolerance = max(MONTHLY_ABS_FLOOR_KWH, abs(want) * REGRESSION_RTOL)
        assert got == pytest.approx(want, abs=tolerance), f"{key}/{model}: month {month_idx} drifted"


def _reference(key):
    path = VALIDATION_DIR / "data" / "references" / f"{key}.json"
    return _load_json(path) if path.exists() else {}


@pytest.mark.parametrize("key", _KEYS)
def test_pvgis_validation_band(key):
    pvgis = _reference(key).get("pvgis_pvcalc") or {}
    if not pvgis.get("annual_kwh"):
        pytest.skip(f"no PVGIS reference fetched for {key}")

    annual = _compute(key, "perez")["annual_kwh"]
    deviation = annual / pvgis["annual_kwh"] - 1.0
    assert abs(deviation) <= PVGIS_BAND, (
        f"{key}: BREOS (perez) annual {annual:.0f} kWh deviates {deviation:+.1%} from "
        f"PVGIS PVcalc {pvgis['annual_kwh']:.0f} kWh — beyond the ±{PVGIS_BAND:.0%} gross-error band."
    )


@pytest.mark.parametrize("key", _KEYS)
def test_pvwatts_validation_band(key):
    pvwatts = _reference(key).get("pvwatts_v8") or {}
    if not pvwatts.get("annual_kwh") or not pvwatts.get("trusted"):
        pytest.skip(f"no trusted PVWatts reference fetched for {key}")

    annual = _compute(key, "perez")["annual_kwh"]
    deviation = annual / pvwatts["annual_kwh"] - 1.0
    assert abs(deviation) <= PVWATTS_BAND, (
        f"{key}: BREOS (perez) annual {annual:.0f} kWh deviates {deviation:+.1%} from "
        f"PVWatts v8 {pvwatts['annual_kwh']:.0f} kWh — beyond the ±{PVWATTS_BAND:.0%} gross-error band."
    )
