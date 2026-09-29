"""Reusing the Monte Carlo year cache across a design sweep (#165).

``YEAR_CACHE_INDEPENDENT_KEYS`` lists the config keys the PV layer of the
year cache never reads. Each is pinned here by changing it and comparing the
year cache, so a key cannot join the list, or stay on it, unless it truly
leaves the cache unchanged. A study run with a reused cache must match the
same study run without one, every float compared bit for bit.
"""

import json
import math
import pickle
import shutil
from pathlib import Path

import pandas as pd
import pytest

import breos.montecarlo as montecarlo_module
from breos.app_config import APP_CONFIG_FIELDS, resolve_app_config
from breos.montecarlo import (
    YEAR_CACHE_INDEPENDENT_KEYS,
    MonteCarloSettings,
    _pv_cache_key,
    build_year_cache,
    run_montecarlo,
)
from breos.weather import _weather_file_sha256 as _file_sha256
from tests.test_input_cache import CHANGES as INPUT_INDEPENDENT_CHANGES

BASE = {
    "location": "porto",
    "n_modules": 8,
    "annual_consumption_kwh": 4000,
    "start_date": "2025-01-01",
    "battery_kwh": 5.0,
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
    "resolution": "h",
    "projection_years": 3,
}

# A custom profile the year cache never reads, so the file need not exist.
CUSTOM_LOAD = {"load_profile": "custom", "load_profile_file": "profile.csv", "load_profile_unit": "kW"}

# One change per key the PV layer does not read: the App sweep's
# input-independent keys, pinned by the same changes, and the demand keys.
# The companions are year-cache-independent keys too.
CHANGES = {
    **INPUT_INDEPENDENT_CHANGES,
    "annual_consumption_kwh": {"annual_consumption_kwh": 3000},
    "load_profile": {"load_profile": "bdew_h0"},
    "load_profile_file": CUSTOM_LOAD,
    "load_profile_column": {**CUSTOM_LOAD, "load_profile_column": "load"},
    "load_profile_unit": {**CUSTOM_LOAD, "load_profile_unit": "kWh"},
    "rlp_directory": {"rlp_directory": "rlps"},
    "start_date": {"start_date": "2024-01-01"},
    "montecarlo": {"montecarlo": {"n_runs": 7, "seed": 3}},
}


def _hex(value):
    """A float as ``float.hex``, so -0.0, 0.0 and every NaN payload stay apart."""
    if isinstance(value, float):
        return "nan" if math.isnan(value) else value.hex()
    return repr(value)


def _frame_bits(frame):
    if frame is None:
        return None
    return (
        list(frame.columns),
        [str(dtype) for dtype in frame.dtypes],
        [[_hex(value) for value in row] for row in frame.itertuples(index=False)],
    )


def _result_bits(result):
    return (
        _frame_bits(result.runs),
        _frame_bits(result.yearly),
        json.dumps(
            {name: {key: _hex(value) for key, value in stats.items()} for name, stats in result.summary.items()},
            sort_keys=True,
        ),
        result.available_years,
        json.dumps(result.provenance, sort_keys=True, default=str),
    )


@pytest.fixture
def weather(tmp_path, write_multiyear_weather):
    return write_multiyear_weather(tmp_path / "multi.csv")


def _settings(weather, **overrides):
    return MonteCarloSettings(
        weather_file=str(weather),
        n_runs=3,
        years_per_run=2,
        seed=11,
        collect_yearly=True,
        **overrides,
    )


@pytest.fixture
def builds(monkeypatch):
    """Count how often each layer of the year cache is built."""
    counts = {"weather": 0, "pv": 0}
    load_weather_years = montecarlo_module._load_weather_years
    build_pv_years = montecarlo_module._build_pv_years

    def counting_weather(*args, **kwargs):
        counts["weather"] += 1
        return load_weather_years(*args, **kwargs)

    def counting_pv(*args, **kwargs):
        counts["pv"] += 1
        return build_pv_years(*args, **kwargs)

    monkeypatch.setattr(montecarlo_module, "_load_weather_years", counting_weather)
    monkeypatch.setattr(montecarlo_module, "_build_pv_years", counting_pv)
    return counts


def test_a_battery_sweep_reuses_both_layers_and_moves_no_number(weather, builds):
    settings = _settings(weather)
    designs = [
        {"battery_kwh": 2.0},
        {"battery_kwh": 10.0, "battery_max_charge_power_w": 2000.0},
        {"battery_kwh": 5.0, "battery_min_soc": 0.2, "inverter_ac_rating_kw": 2.5},
    ]
    fresh = [_result_bits(run_montecarlo({**BASE, **design}, settings)) for design in designs]

    cache = build_year_cache(BASE, settings)
    builds.update(weather=0, pv=0)
    reused = [_result_bits(run_montecarlo({**BASE, **design}, settings, year_cache=cache)) for design in designs]

    assert builds == {"weather": 0, "pv": 0}
    assert reused == fresh


def test_a_pv_only_design_reuses_a_cache_built_for_a_battery(weather, builds):
    settings = _settings(weather)
    config = {**BASE, "battery_kwh": 0.0}
    fresh = run_montecarlo(config, settings)

    cache = build_year_cache(BASE, settings)
    builds.update(weather=0, pv=0)
    reused = run_montecarlo(config, settings, year_cache=cache)

    assert builds == {"weather": 0, "pv": 0}
    assert reused.provenance["execution"]["dispatch_path"] == "pv_only_vectorized"
    assert _result_bits(reused) == _result_bits(fresh)


def test_a_module_count_sweep_rebuilds_the_pv_layer_from_the_cached_weather(weather, builds):
    settings = _settings(weather)
    fresh = {n: _result_bits(run_montecarlo({**BASE, "n_modules": n}, settings)) for n in (8, 10)}

    cache = build_year_cache(BASE, settings)
    key_8 = cache.pv_key
    builds.update(weather=0, pv=0)
    reused_10 = _result_bits(run_montecarlo({**BASE, "n_modules": 10}, settings, year_cache=cache))
    assert builds == {"weather": 0, "pv": 1}
    assert cache.pv_key != key_8

    # The cache holds the latest PV layer; going back rebuilds it, from
    # weather frames the rebuild above must not have written to.
    reused_8 = _result_bits(run_montecarlo({**BASE, "n_modules": 8}, settings, year_cache=cache))
    assert builds == {"weather": 0, "pv": 2}
    assert cache.pv_key == key_8

    assert reused_10 == fresh[10]
    assert reused_8 == fresh[8]


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"weather_start_year": 2022}, "weather_start_year"),
        ({"weather_end_year": 2021}, "weather_end_year"),
        ({"target_year": 2024}, "target_year"),
        ({"preserve_irradiance_energy": True}, "preserve_irradiance_energy"),
    ],
)
def test_a_cache_for_other_weather_settings_is_refused(weather, change, field):
    cache = build_year_cache(BASE, _settings(weather))
    with pytest.raises(ValueError, match=f"year_cache was built for other weather inputs \\({field} differ"):
        run_montecarlo(BASE, _settings(weather, **change), year_cache=cache)


@pytest.mark.parametrize(
    ("change", "fields"),
    [
        ({"resolution": "15min"}, "resolution"),
        ({"location": "lisbon"}, "latitude, longitude"),
        ({"solar_position": "mid-interval"}, "solar_position"),
    ],
)
def test_a_cache_for_another_site_or_resolution_is_refused(weather, change, fields):
    cache = build_year_cache(BASE, _settings(weather))
    with pytest.raises(ValueError, match=f"\\({fields} differ"):
        run_montecarlo({**BASE, **change}, _settings(weather), year_cache=cache)


def test_a_cache_is_refused_for_another_weather_file_or_once_its_file_changes(tmp_path, write_multiyear_weather):
    path = write_multiyear_weather(tmp_path / "multi.csv")
    cache = build_year_cache(BASE, _settings(path))

    # The same contents at another path are still another file.
    copy = tmp_path / "copy.csv"
    shutil.copyfile(path, copy)
    with pytest.raises(ValueError, match="\\(weather_file differ"):
        run_montecarlo(BASE, _settings(copy), year_cache=cache)

    write_multiyear_weather(path, years=(2021, 2022, 2023))
    with pytest.raises(ValueError, match="\\(weather_file_sha256 differ"):
        run_montecarlo(BASE, _settings(path), year_cache=cache)


def _write_sidecar(weather, offset_hours):
    """A valid metadata sidecar that moves the irradiance timestamps by ``offset_hours``."""
    metadata = {"radiation_time_basis": "instant", "irradiance_time_offset_hours": offset_hours}
    payload = {"schema_version": 1, "weather_sha256": _file_sha256(weather), "breos_weather_metadata": metadata}
    Path(f"{weather}.metadata.json").write_text(json.dumps(payload))


def test_a_cache_is_refused_once_the_weather_metadata_sidecar_changes(weather):
    config = {**BASE, "solar_position": "weather"}
    settings = _settings(weather)
    _write_sidecar(weather, 0.0)
    cache = build_year_cache(config, settings)
    before = _result_bits(run_montecarlo(config, settings))

    # Same CSV bytes, other timing metadata: the sun moves, so the numbers do.
    _write_sidecar(weather, 0.5)
    assert _result_bits(run_montecarlo(config, settings)) != before
    with pytest.raises(ValueError, match="\\(weather_metadata_sidecar_sha256 differ"):
        run_montecarlo(config, settings, year_cache=cache)

    Path(f"{weather}.metadata.json").unlink()
    with pytest.raises(ValueError, match="\\(weather_metadata_sidecar_sha256 differ"):
        run_montecarlo(config, settings, year_cache=cache)


def test_a_battery_temperature_file_rewritten_in_place_rebuilds_the_pv_layer(tmp_path, weather, builds):
    index = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    temperatures = tmp_path / "battery_temperature.csv"
    pd.DataFrame({"date": index, "temperature": 20.0}).to_csv(temperatures, index=False)
    config = {**BASE, "battery_temperature": str(temperatures)}
    settings = _settings(weather)
    cache = build_year_cache(config, settings)

    pd.DataFrame({"date": index, "temperature": 40.0}).to_csv(temperatures, index=False)
    fresh = _result_bits(run_montecarlo(config, settings))
    builds.update(weather=0, pv=0)
    reused = _result_bits(run_montecarlo(config, settings, year_cache=cache))

    assert builds == {"weather": 0, "pv": 1}
    assert reused == fresh


def test_a_cache_built_through_a_symlink_is_refused_for_the_real_file(tmp_path, weather):
    link = tmp_path / "link.csv"
    link.symlink_to(weather)
    cache = build_year_cache(BASE, _settings(link))
    # Provenance records the path as given, so the link and its target are
    # two weather files, and reusing one for the other would record the wrong one.
    with pytest.raises(ValueError, match="\\(weather_file differ"):
        run_montecarlo(BASE, _settings(weather), year_cache=cache)

    fresh = run_montecarlo(BASE, _settings(link))
    reused = run_montecarlo(BASE, _settings(link), year_cache=cache)
    assert reused.provenance["runtime_weather"]["metadata"]["path"] == str(link)
    assert _result_bits(reused) == _result_bits(fresh)


def test_every_year_cache_independent_key_is_a_registry_key_with_a_pinning_change():
    assert YEAR_CACHE_INDEPENDENT_KEYS <= set(APP_CONFIG_FIELDS)
    assert set(CHANGES) == YEAR_CACHE_INDEPENDENT_KEYS


def _year_layers(config, settings):
    """The PV layer and weather record a fresh study builds, pickled."""
    resolved = resolve_app_config(config)
    runtime_weather = {}
    dc_by_year, temp_by_year = montecarlo_module._precompute_year_caches(
        resolved.cfg, resolved, settings, runtime_weather=runtime_weather
    )
    return pickle.dumps((dc_by_year, temp_by_year, runtime_weather))


@pytest.fixture(scope="module")
def one_year_weather(tmp_path_factory):
    from tests.conftest import _write_multiyear_weather

    return _write_multiyear_weather(tmp_path_factory.mktemp("weather") / "one.csv", years=(2021,))


@pytest.fixture(scope="module")
def base_layers(one_year_weather):
    return _year_layers(BASE, MonteCarloSettings(weather_file=str(one_year_weather)))


@pytest.mark.parametrize("key", sorted(CHANGES))
def test_a_year_cache_independent_key_leaves_the_year_cache_unchanged(key, one_year_weather, base_layers):
    changed = {**BASE, **CHANGES[key]}
    assert _pv_cache_key(resolve_app_config(changed).cfg) == _pv_cache_key(resolve_app_config(BASE).cfg)
    assert _year_layers(changed, MonteCarloSettings(weather_file=str(one_year_weather))) == base_layers


@pytest.mark.parametrize(
    "change",
    [
        {"n_modules": 10},
        {"tilt": 20.0},
        {"pv_module": "Generic_600W_Bifacial"},
        {"battery_temperature": 20.0},
        {"battery_indoor_model": {"enabled": False}},
    ],
)
def test_a_pv_key_changes_the_cache_key_and_the_year_cache(change, one_year_weather, base_layers):
    changed = {**BASE, **change}
    assert _pv_cache_key(resolve_app_config(changed).cfg) != _pv_cache_key(resolve_app_config(BASE).cfg)
    assert _year_layers(changed, MonteCarloSettings(weather_file=str(one_year_weather))) != base_layers


def test_a_value_json_cannot_hold_is_never_reused():
    assert _pv_cache_key({"battery_temperature": pd.Series([20.0])}) is None
    assert _pv_cache_key({"battery_temperature": 20.0}) is not None
