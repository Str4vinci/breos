"""The battery and inverter App, Monte Carlo and the optimizer build are built once (#179)."""

import dataclasses
import math

import pytest

import breos.montecarlo as montecarlo_module
import breos.projection as projection_module
import breos.runners.app as app_runner
from breos.app_config import resolve_app_config
from breos.inverter import inverter_ac_capacity_w
from breos.projection import build_battery_config, build_pv_only_battery_config

BASE = {
    "location": "porto",
    "n_modules": 7,
    "annual_consumption_kwh": 3500,
    "battery_kwh": 6.5,
    "cost_preset": "residential_pt",
    "inverter_loading_ratio": 1.4,
    "battery_rte": 0.9,
    "battery_min_soc": 0.15,
    "battery_max_soc": 0.95,
    "battery_power_limit_c_rate": 0.5,
    "enable_resistance_fade": True,
}


@pytest.mark.parametrize(
    ("peak_w", "ratio", "expected"),
    [(4400.0, 1.25, 3520.0), (4400.0, 1.0, 4400.0), (4400.0, 0.0, None), (4400.0, -1.0, None), (4400.0, None, None)],
)
def test_inverter_ac_capacity_is_the_dc_peak_over_the_loading_ratio(peak_w, ratio, expected):
    assert inverter_ac_capacity_w(peak_w, ratio) == expected


def test_resolved_config_carries_the_inverter_ac_capacity():
    resolved = resolve_app_config(BASE)

    assert resolved.inverter_ac_capacity_w == 7 * resolved.avg_module_power_w / 1.4


def test_battery_config_comes_from_the_config():
    resolved = resolve_app_config(BASE)

    battery = build_battery_config(resolved.cfg, resolved, initial_soh=93.5)

    assert battery.nominal_energy_wh == 6500
    assert battery.initial_soh == 93.5
    assert battery.charge_efficiency == battery.discharge_efficiency == math.sqrt(0.9)
    assert (battery.min_soc, battery.max_soc) == (0.15, 0.95)
    assert battery.inverter_ac_capacity_w == resolved.inverter_ac_capacity_w
    assert battery.replacement_cost == resolved.cost_params.battery_cost_per_kwh * 6.5
    assert battery.enable_replacement and battery.enable_resistance_fade


def test_pv_only_config_keeps_the_inverter():
    resolved = resolve_app_config({**BASE, "battery_kwh": 0})

    battery = build_pv_only_battery_config(resolved.cfg, resolved)

    assert battery.nominal_energy_wh == 0
    assert battery.inverter_efficiency == resolved.cfg["inverter_efficiency"]
    assert battery.inverter_ac_capacity_w == resolved.inverter_ac_capacity_w


def test_app_and_montecarlo_run_the_shared_projection():
    # Both year loops are the one in breos.projection, not copies of it.
    for module in (app_runner, montecarlo_module):
        assert module.run_projection is projection_module.run_projection
        assert module.value_projection is projection_module.value_projection


def test_battery_config_without_rte_keeps_the_dataclass_efficiencies():
    resolved = resolve_app_config({**BASE, "battery_rte": None})
    defaults = {
        f.name: f.default for f in dataclasses.fields(build_battery_config(resolved.cfg, resolved, initial_soh=100))
    }

    battery = build_battery_config(resolved.cfg, resolved, initial_soh=100)

    assert battery.charge_efficiency == defaults["charge_efficiency"]
    assert battery.discharge_efficiency == defaults["discharge_efficiency"]
