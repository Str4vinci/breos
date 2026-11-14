"""Vehicle inputs fail before any weather, load or scoring work (ADR 0005 PR 1)."""

from copy import deepcopy

import pandas as pd
import pytest

import breos
from breos.app_config import DEFAULTS, merge_defaults, resolve_app_config, validate_config
from breos.degradation.profiles import ENABLED_BLAST_MODEL_KEYS
from breos.optimization_config import resolve_optimization_config
from breos.smart_charging import resolve_instructions
from breos.vehicle import AwayChargingConfig, VehicleConfig, VehicleScheduleSpec

BASE = {"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000}
VEHICLE = {
    "mode": "v2h",
    "battery_kwh": 60,
    "departure_soc": 0.8,
    "charger_ac_rating_kw": 7.4,
    "charger_efficiency": 0.94,
    "charger_cost": 2500,
    "battery_cost_per_kwh": 120,
    "schedule": {
        "weekday": {"arrive": "18:30", "depart": "07:45", "trip_kwh": 9},
        "weekend": {"arrive": "14:00", "depart": "10:00", "trip_kwh": 12},
    },
}
AWAY = {"target_soc": 0.7, "power_kw": 3.7, "efficiency": 0.9, "price": 0}


def checked(vehicle=None, **extra):
    return validate_config(
        merge_defaults({**BASE, "vehicle": deepcopy(VEHICLE if vehicle is None else vehicle), **extra})
    )


def test_valid_vehicle_builds_typed_specs_without_mutating_inputs():
    value = {
        **VEHICLE,
        "away_charging": AWAY,
        "battery_replacement_cost": 6000,
        "enable_replacement": False,
        "replacement_min_remaining_years": 2,
        "skipped_replacement_action": "retire",
    }
    original = deepcopy(value)
    config = checked(value).vehicle
    assert value == original
    assert isinstance(config, VehicleConfig)
    assert isinstance(config.schedule, VehicleScheduleSpec)
    assert config.away_charging == AwayChargingConfig(**AWAY)
    assert config.reference_mode == "v1g" and config.v2h_min_soc == config.min_soc
    assert not config.enable_replacement and config.battery_replacement_cost == 6000
    assert config.replacement_min_remaining_years == 2 and config.skipped_replacement_action == "retire"


@pytest.mark.parametrize(
    "key",
    [
        "mode",
        "battery_kwh",
        "departure_soc",
        "charger_ac_rating_kw",
        "charger_efficiency",
        "charger_cost",
        "battery_cost_per_kwh",
        "schedule",
    ],
)
def test_missing_required_vehicle_facts_are_reported_before_rollout_refusal(key):
    value = deepcopy(VEHICLE)
    del value[key]
    with pytest.raises(ValueError, match=rf"vehicle\.{key}"):
        resolve_app_config({**BASE, "vehicle": value})
    if key == "mode":
        with pytest.raises(TypeError, match="required positional argument: 'mode'"):
            VehicleConfig(**value)


@pytest.mark.parametrize("key", AWAY)
def test_all_away_charger_facts_are_required_even_for_free_charging(key):
    away = dict(AWAY)
    del away[key]
    with pytest.raises(ValueError, match=rf"vehicle.away_charging\.{key}"):
        checked({**VEHICLE, "away_charging": away})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("battery_kwh", 0),
        ("battery_kwh", float("inf")),
        ("departure_soc", 1.1),
        ("departure_soc", 0.01),
        ("min_soc", 0.9),
        ("max_soc", 0.7),
        ("v2h_min_soc", 0),
        ("v2h_min_soc", 1.1),
        ("charge_efficiency", 0),
        ("discharge_efficiency", 1.1),
        ("charger_ac_rating_kw", -1),
        ("charger_efficiency", 0),
        ("charger_efficiency", float("nan")),
        ("charger_cost", -1),
        ("battery_cost_per_kwh", -1),
        ("battery_replacement_cost", -1),
        ("outside_charging_price", -1),
        ("replacement_min_remaining_years", -1),
        ("eol_percentage", 1.1),
        ("enable_replacement", "false"),
        ("allow_terminal_replacement", "false"),
        ("skipped_replacement_action", "sell"),
        ("mode", "v2g"),
        ("reference_mode", "v2h"),
        ("calendar_model", "unknown"),
        ("blast_model", "unknown"),
        ("degradation_engine", "unknown"),
        ("battery_temperature", True),
    ],
)
def test_bad_vehicle_values_fail_by_key(key, value):
    with pytest.raises((ValueError, TypeError), match="vehicle"):
        checked({**VEHICLE, key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("target_soc", 0.05),
        ("target_soc", 1.1),
        ("power_kw", 0),
        ("efficiency", 0),
        ("efficiency", 1.1),
        ("price", -1),
        ("price", True),
    ],
)
def test_bad_away_charger_values(key, value):
    with pytest.raises((TypeError, ValueError), match=rf"vehicle.away_charging.{key}"):
        checked({**VEHICLE, "away_charging": {**AWAY, key: value}})


@pytest.mark.parametrize(("mode", "reference"), [("uncontrolled", "none"), ("v1g", "uncontrolled"), ("v2h", "v1g")])
def test_reference_mode_defaults_and_zero_prices(mode, reference):
    value = {**VEHICLE, "mode": mode, "battery_cost_per_kwh": 0, "outside_charging_price": 0}
    if mode != "v2h":
        del value["charger_cost"]
    config = checked(value).vehicle
    assert config.reference_mode == reference
    assert config.charger_cost == (2500 if mode == "v2h" else 0)
    assert checked({**value, "reference_mode": "none"}).vehicle.reference_mode == "none"


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "uncontrolled", "v2h_min_soc": 0.4},
        {"mode": "v1g", "v2h_min_soc": 0.4},
        {"mode": "uncontrolled", "charge_periods": ["P1"]},
        {"mode": "uncontrolled", "discharge_periods": ["P1"]},
        {"mode": "v1g", "discharge_periods": ["P1"]},
        {"mode": "uncontrolled", "reference_mode": "v1g"},
        {"charge_periods": ["P1"], "discharge_periods": ["P1"]},
    ],
)
def test_incompatible_modes_are_refused(overrides):
    with pytest.raises(ValueError, match="vehicle"):
        checked({**VEHICLE, **overrides})


def test_vehicle_tariff_periods_need_a_tariff_and_must_exist():
    value = {**VEHICLE, "charge_periods": ["peak"]}
    with pytest.raises(ValueError, match=r"vehicle.charge_periods.*needs a \[tariff\]"):
        checked(value)
    tariff = {
        "schedule": "pt_mainland_2026_daily_tri",
        "currency": "EUR",
        "import_prices": {"peak": 0.3, "mid_peak": 0.2, "off_peak": 0.1},
        "export_prices": {"all": 0.05},
    }
    assert checked(value, tariff=tariff, resolution="15min").vehicle.charge_periods == ("peak",)
    with pytest.raises(ValueError, match="unknown tariff period"):
        checked({**value, "discharge_periods": ["missing"]}, tariff=tariff, resolution="15min")


@pytest.mark.parametrize(
    "schedule",
    [
        {},
        {"weekday": VEHICLE["schedule"]["weekday"]},
        {**VEHICLE["schedule"], "file": "trips.csv"},
        {"weekday": {"arrive": "24:00", "depart": "08:00", "trip_kwh": 1}, "weekend": VEHICLE["schedule"]["weekend"]},
        {"weekday": {"arrive": "18:00", "depart": "08:00", "trip_kwh": -1}, "weekend": VEHICLE["schedule"]["weekend"]},
    ],
)
def test_invalid_schedule_shapes(schedule):
    with pytest.raises(ValueError, match="vehicle.schedule"):
        checked({**VEHICLE, "schedule": schedule})


def test_unknown_nested_keys_are_never_silently_dropped():
    with pytest.raises(ValueError, match="vehicle.arrival_soc"):
        checked({**VEHICLE, "arrival_soc": 0.6})
    with pytest.raises(ValueError, match="vehicle.away_charging.rating"):
        checked({**VEHICLE, "away_charging": {**AWAY, "rating": 7}})


def test_app_accepts_blast_vehicle_and_both_montecarlo_entries_refuse_before_input_preparation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Vehicle rollout refusal must happen before inputs are read")

    monkeypatch.setattr("breos.montecarlo._weather_years", unexpected)
    monkeypatch.setattr("breos.app_inputs.load_consumption_profile", unexpected)
    config = {**BASE, "vehicle": {**VEHICLE, "degradation_engine": "blast", "blast_model": "nmc_gr_50ah_b1"}}
    assert breos.App(config)._resolved.vehicle is not None
    settings = breos.MonteCarloSettings(weather_file="nonexistent.csv", n_runs=1)
    for call in (
        lambda: breos.run_montecarlo(config, settings),
        lambda: breos.build_year_cache(config, settings),
    ):
        with pytest.raises(ValueError, match="vehicle.*Monte Carlo"):
            call()


@pytest.mark.parametrize("value", [VEHICLE, {}, None])
def test_optimizer_explicitly_refuses_any_vehicle_table(value):
    with pytest.raises(ValueError, match=r"^'\[vehicle\]' is not supported by the optimizer$"):
        resolve_optimization_config({"location": "porto", "vehicle": value})


def test_vehicle_only_site_limit_preserves_the_stationary_spec():
    spec = checked(smart_charging={"mode": "disabled", "grid_import_limit_w": 7400}).smart_charging
    baseline = validate_config(merge_defaults({**BASE, "smart_charging": {"mode": "disabled"}})).smart_charging
    assert spec == baseline
    assert spec.grid_import_limit_w is None
    resolved = resolve_app_config(
        {**BASE, "vehicle": VEHICLE, "smart_charging": {"mode": "disabled", "grid_import_limit_w": 7400}}
    )
    assert resolved.cfg["smart_charging"]["grid_import_limit_w"] == 7400
    # Without a vehicle the existing disabled-mode restriction remains.
    with pytest.raises(ValueError, match="takes no other keys"):
        resolve_app_config({**BASE, "smart_charging": {"mode": "disabled", "grid_import_limit_w": 7400}})


@pytest.mark.parametrize("limit", [0, -1, True, float("nan")])
def test_vehicle_only_site_limit_still_validates_its_value(limit):
    with pytest.raises((ValueError, TypeError), match="smart_charging.grid_import_limit_w"):
        checked(smart_charging={"mode": "disabled", "grid_import_limit_w": limit})


def test_vehicle_does_not_enable_other_stationary_controls():
    with pytest.raises(ValueError, match="takes no other keys"):
        checked(smart_charging={"mode": "disabled", "grid_import_limit_w": 7400, "target_usable_fraction": 0.5})


def test_vehicle_blast_montecarlo_is_explicitly_refused():
    value = {**VEHICLE, "degradation_engine": "blast", "blast_model": next(iter(ENABLED_BLAST_MODEL_KEYS))}
    assert checked(value).vehicle.degradation_engine == "blast"
    with pytest.raises(ValueError, match="vehicle.degradation_engine=blast.*Monte Carlo"):
        checked(value, montecarlo={"weather_file": "missing.csv"})


def test_no_vehicle_has_no_new_default_or_serialized_key():
    assert "vehicle" not in DEFAULTS
    resolved = resolve_app_config(BASE)
    assert resolved.vehicle is None and "vehicle" not in resolved.cfg
    for name in ("VehicleConfig", "VehicleSchedule", "resolve_vehicle_schedule"):
        assert name not in breos.__all__ and not hasattr(breos, name)


@pytest.mark.parametrize("coupling", ["dc", "ac"])
def test_vehicle_keeps_stationary_instruction_hashes_unchanged(coupling):
    tariff = {
        "schedule": "pt_mainland_2026_daily_bi",
        "currency": "EUR",
        "import_prices": {"peak": 0.3, "off_peak": 0.1},
        "export_prices": {"all": 0.05},
    }
    smart = {
        "mode": "fixed_target",
        "target_usable_fraction": 0.5,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
        "grid_import_limit_w": 7400,
    }
    battery = {"battery_kwh": 5, "battery_coupling": coupling}
    if coupling == "dc":
        smart["grid_charge_efficiency"] = 0.95
    else:
        battery.update(battery_inverter_efficiency=0.94, battery_inverter_ac_rating_kw=3)
    config = {**BASE, **battery, "tariff": tariff, "smart_charging": smart}
    baseline = validate_config(merge_defaults(config))
    vehicle = validate_config(merge_defaults({**config, "vehicle": VEHICLE}))
    index = pd.date_range("2026-01-05", periods=24, freq="h", tz="UTC")
    resolved_tariff = baseline.tariff.resolve(index, "Europe/Lisbon")
    before = resolve_instructions(baseline.smart_charging, resolved_tariff)
    after = resolve_instructions(vehicle.smart_charging, resolved_tariff)
    assert before.instruction_hash() == after.instruction_hash()
    assert before.grid_charge_efficiency == after.grid_charge_efficiency
    if coupling == "ac":
        assert after.grid_charge_efficiency is None


@pytest.mark.parametrize("forecast", ["persistence", "weather", "perfect"])
@pytest.mark.parametrize("boundary", ["civil_day", "charge_window_start"])
def test_daily_persistence_is_refused_for_every_vehicle_forecast_and_boundary(forecast, boundary):
    with pytest.raises(ValueError, match="daily_persistence.*not supported with"):
        checked(smart_charging={"mode": "daily_persistence", "forecast": forecast, "decision_boundary": boundary})


@pytest.mark.parametrize(
    ("key", "value"), [("mode", None), ("schedule", None), ("schedule", "trips.csv"), ("charger_efficiency", None)]
)
def test_direct_config_rejects_invalid_types_by_key(key, value):
    with pytest.raises((ValueError, TypeError), match=f"vehicle.{key}"):
        VehicleConfig(**{**VEHICLE, key: value})
