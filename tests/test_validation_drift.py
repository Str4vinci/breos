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
   prove accuracy. For trusted PVWatts v8 references, compare performance
   ratios within ±8%: AC / (front-side POA irradiation × DC nameplate).
   BREOS AC is normalized by an independent pvlib Perez transposition of
   its PVGIS TMY, while PVWatts uses its NSRDB POA. This takes irradiance
   differences out of the comparison without masking BREOS transposition errors.
"""

import json
import sys
from pathlib import Path

import pytest

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
PVWATTS_PR_BAND = 0.08


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
_POA_CACHE = {}


def test_every_baseline_location_has_its_weather_file():
    # Cases are built only for weather files on disk, so a missing file would
    # silently drop that location from the drift suite instead of failing it.
    missing = sorted(
        entry["weather_file"]
        for entry in _BASELINE["locations"].values()
        if not (VALIDATION_DIR / "data" / "weather" / entry["weather_file"]).exists()
    )
    assert not missing, f"validation weather files missing from validation/data/weather: {missing}"


def test_equal_nameplate_comparison_contract():
    from breos.pv_modules import get_module
    from validation.common import MODEL_CONFIGS

    mono = MODEL_CONFIGS["nameplate_mono_1200"]
    front = MODEL_CONFIGS["nameplate_bifacial_front_1200"]
    rear = MODEL_CONFIGS["nameplate_bifacial_rear_1200"]

    assert mono["n_modules"] * get_module(mono["pv_module"]).Mpp == pytest.approx(1200)
    assert front["n_modules"] * get_module(front["pv_module"]).Mpp == pytest.approx(1200)
    assert rear["n_modules"] * get_module(rear["pv_module"]).Mpp == pytest.approx(1200)

    common_keys = ("pv_module", "n_modules", "transposition_model", "solar_position", "iam_model", "diffuse_iam")
    assert {key: front[key] for key in common_keys} == {key: rear[key] for key in common_keys}
    assert rear["bifacial_model"] == "infinite_sheds"


def _compute(key, model):
    """Recompute BREOS production for a baseline case (cached per session)."""
    if (key, model) in _RESULT_CACHE:
        return _RESULT_CACHE[(key, model)]

    from validation.common import (
        MODEL_CONFIGS,
        compute_annual_poa,
        compute_model_result,
        load_spec,
        load_validation_weather,
    )

    system, locations = load_spec()
    loc = locations[key]
    entry = _BASELINE["locations"][key]
    weather = load_validation_weather(VALIDATION_DIR / "data" / "weather" / entry["weather_file"])
    if key not in _POA_CACHE:
        _POA_CACHE[key] = compute_annual_poa(weather, loc, system)
    result = compute_model_result(weather, loc, system, MODEL_CONFIGS[model], _POA_CACHE[key])
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

    computed = _compute(key, "perez")
    reference_capacity = pvwatts["params"]["system_capacity"]
    assert computed["dc_nameplate_kw"] == pytest.approx(reference_capacity), (
        f"{key}: BREOS DC nameplate {computed['dc_nameplate_kw']} kW differs from PVWatts {reference_capacity} kW"
    )
    pr_breos = computed["annual_kwh"] / (computed["annual_poa_kwh_m2"] * computed["dc_nameplate_kw"])
    pr_pvwatts = pvwatts["annual_kwh"] / (pvwatts["solrad_annual_kwh_m2_day"] * 365 * reference_capacity)
    deviation = pr_breos / pr_pvwatts - 1.0
    print(f"{key}: PR BREOS={pr_breos:.6f}, PR PVWatts={pr_pvwatts:.6f}, deviation={deviation:+.4%}")
    assert abs(deviation) <= PVWATTS_PR_BAND, (
        f"{key}: BREOS (perez) PR {pr_breos:.4f} vs PVWatts v8 PR {pr_pvwatts:.4f}, "
        f"deviation {deviation:+.1%} — beyond the ±{PVWATTS_PR_BAND:.0%} PR band. "
        f"Annual AC: BREOS {computed['annual_kwh']:.0f} kWh, PVWatts {pvwatts['annual_kwh']:.0f} kWh."
    )
