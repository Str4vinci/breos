"""The optimizer with an AC-coupled battery (ADR 0004 C2, C9, C13, C14)."""

import pytest

from breos.economics import default_amount_cost_keys
from breos.optimization import (
    _battery_inverter_ac_capacity_w,
    _battery_replacement_treatment,
    _build_battery_config_from_spec,
)
from breos.optimization_config import resolve_optimization_config

LOCATION = {"location": {"latitude": 41.15, "longitude": -8.61}}
AC_BATTERY = {"coupling": "ac", "inverter_kw_per_kwh": 0.5, "inverter_efficiency": 0.95}
AC = {**LOCATION, "battery": AC_BATTERY, "costs": {"battery_inverter_cost_per_kw": 120.0}}
FIXED = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
}
TARIFF = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.50, "off_peak": 0.10},
    "export_prices": {"all": 0.03},
}


def test_dc_is_the_default_and_resolving_twice_changes_nothing():
    resolved = resolve_optimization_config(LOCATION)
    battery = resolved["battery"]
    assert battery["coupling"] == "dc"
    assert (battery["inverter_ac_rating_kw"], battery["inverter_kw_per_kwh"], battery["inverter_efficiency"]) == (
        None,
        None,
        None,
    )
    ac = resolve_optimization_config(AC)
    assert resolve_optimization_config(ac) == ac


@pytest.mark.parametrize(
    "config",
    [
        {**LOCATION, "battery": {"inverter_kw_per_kwh": 0.5}},
        {**LOCATION, "battery": {"inverter_efficiency": 0.95}},
        {**LOCATION, "costs": {"battery_inverter_cost_per_kw": 120.0}},
    ],
)
def test_dc_refuses_every_battery_inverter_setting(config):
    with pytest.raises(ValueError, match="set battery.coupling = 'ac' or remove them"):
        resolve_optimization_config(config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"battery": {**AC_BATTERY, "inverter_kw_per_kwh": None}}, "needs its inverter's rating"),
        ({"battery": {**AC_BATTERY, "inverter_efficiency": None}}, "needs battery.inverter_efficiency"),
        ({"costs": {}}, "needs costs.battery_inverter_cost_per_kw"),
        ({"battery": {**AC_BATTERY, "inverter_ac_rating_kw": 3.0}}, "both size the battery inverter"),
    ],
)
def test_an_ac_study_needs_its_rating_efficiency_and_price_before_any_candidate(change, message):
    with pytest.raises(ValueError, match=message):
        resolve_optimization_config({**AC, **change})


def test_an_ac_study_without_a_battery_needs_none_of_them():
    resolve_optimization_config({**LOCATION, "battery": {"coupling": "ac"}, "constraints": {"max_battery_kwh": 0}})


@pytest.mark.parametrize("value", [0.0, -1.0, 1.5])
def test_the_battery_inverter_efficiency_is_a_fraction(value):
    with pytest.raises(ValueError, match="inverter_efficiency"):
        resolve_optimization_config({**AC, "battery": {**AC_BATTERY, "inverter_efficiency": value}})


def test_smart_charging_checks_grid_charge_efficiency_against_the_coupling_at_resolution():
    with pytest.raises(ValueError, match="grid_charge_efficiency' is for a DC-coupled battery"):
        resolve_optimization_config(
            {**AC, "tariff": TARIFF, "smart_charging": {**FIXED, "grid_charge_efficiency": 0.95}}
        )
    resolve_optimization_config({**AC, "tariff": TARIFF, "smart_charging": FIXED})
    with pytest.raises(ValueError, match="needs smart_charging.grid_charge_efficiency"):
        resolve_optimization_config({**LOCATION, "tariff": TARIFF, "smart_charging": FIXED})


def test_a_non_eur_ac_study_needs_the_simple_pv_inverter_price():
    needed = default_amount_cost_keys(tariff=False, battery=None, coupling="ac")
    costs = {key: 1.0 for key in needed}
    base = {**AC, "currency": "USD", "constraints": {"budget": 20000.0}}
    resolve_optimization_config({**base, "costs": {**costs, "battery_inverter_cost_per_kw": 120.0}})
    del costs["inverter_cost_per_kw_simple"]
    with pytest.raises(ValueError, match="inverter_cost_per_kw_simple would come from the built-in EUR defaults"):
        resolve_optimization_config({**base, "costs": {**costs, "battery_inverter_cost_per_kw": 120.0}})


def test_the_rating_follows_each_candidates_capacity_or_stays_fixed():
    battery = resolve_optimization_config(AC)["battery"]
    assert _battery_inverter_ac_capacity_w(battery, 4.0) == 2000.0
    assert _battery_inverter_ac_capacity_w(battery, 10.0) == 5000.0
    assert _battery_inverter_ac_capacity_w(battery, 0.0) is None
    fixed = {**battery, "inverter_kw_per_kwh": None, "inverter_ac_rating_kw": 3.0}
    assert _battery_inverter_ac_capacity_w(fixed, 4.0) == _battery_inverter_ac_capacity_w(fixed, 10.0) == 3000.0
    assert _battery_inverter_ac_capacity_w(resolve_optimization_config(LOCATION)["battery"], 10.0) is None


def test_each_candidate_runs_its_battery_inverter_and_the_treatment_records_the_coupling():
    battery = resolve_optimization_config(AC)["battery"]
    config = _build_battery_config_from_spec(battery, nominal_energy_wh=8000.0)
    assert (config.coupling, config.battery_inverter_ac_capacity_w, config.battery_inverter_efficiency) == (
        "ac",
        4000.0,
        0.95,
    )
    pv_only = _build_battery_config_from_spec(battery, nominal_energy_wh=0.0)
    assert pv_only.coupling == "dc"
    assert _battery_replacement_treatment(battery)["coupling"] == "ac"
    assert _battery_replacement_treatment(resolve_optimization_config(LOCATION)["battery"])["coupling"] == "dc"


@pytest.mark.parametrize("spelling", ["AC", " ac "])
def test_a_normalised_coupling_spelling_selects_the_ac_smart_charging_table(spelling):
    config = {**AC, "battery": {**AC_BATTERY, "coupling": spelling}, "tariff": TARIFF, "smart_charging": FIXED}
    assert resolve_optimization_config(config)["battery"]["coupling"] == "ac"


def test_no_battery_inverter_below_the_one_wh_cutoff():
    battery = resolve_optimization_config(
        {**AC, "battery": {**AC_BATTERY, "inverter_kw_per_kwh": None, "inverter_ac_rating_kw": 3.0}}
    )["battery"]
    for tiny in (0.0005, 0.001):
        assert _battery_inverter_ac_capacity_w(battery, tiny) is None
        assert _build_battery_config_from_spec(battery, nominal_energy_wh=tiny * 1000.0).coupling == "dc"
    assert _battery_inverter_ac_capacity_w(battery, 0.0011) == 3000.0
