"""The [smart_charging] table, the fixed-target controller and its instruction contract (ADR 0002, #178)."""

import json
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from breos.app import App
from breos.app_config import resolve_tariff_spec
from breos.smart_charging import SmartChargingSpec, resolve_instructions, smart_charging_provenance

LISBON = "Europe/Lisbon"
BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 3}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
}
FIXED = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 5000,
}
TRI = {
    "schedule": "pt_mainland_2026_daily_tri",
    "currency": "EUR",
    "import_prices": {"off_peak": 0.10, "mid_peak": 0.18, "peak": 0.30},
    "export_prices": {"all": 0.04},
}
TRI_SPEC = SmartChargingSpec(
    mode="fixed_target",
    target_usable_fraction=0.6,
    charge_periods=("off_peak",),
    discharge_periods=("peak",),
    grid_charge_efficiency=0.9,
)


def _app(smart_charging, **extra):
    return App({**BASE, "tariff": TOU, **extra, "smart_charging": smart_charging})


def _without(table, *keys):
    return {key: value for key, value in table.items() if key not in keys}


def _tri_tariff(index):
    spec = resolve_tariff_spec({"tariff": TRI, "costs": None, "resolution": "15min"}, LISBON)
    return spec.resolve(index, LISBON)


# --- [smart_charging] validation --------------------------------------------


@pytest.mark.parametrize(
    ("table", "extra", "error", "message"),
    [
        ([], {}, TypeError, r"'smart_charging' must be a table/dict"),
        (_without(FIXED, "mode"), {}, ValueError, r"'smart_charging' needs smart_charging\.mode$"),
        (
            {**FIXED, "mode": "daily_persistence"},
            {},
            ValueError,
            r"'smart_charging\.mode' must be one of: disabled, fixed_target",
        ),
        (
            {**FIXED, "target_soc": 0.5},
            {},
            ValueError,
            r"Unknown key 'smart_charging\.target_soc'\. Available: smart_charging\.charge_periods",
        ),
        (
            {**FIXED, "target_usable_fraction": 1.2},
            {},
            ValueError,
            r"'smart_charging\.target_usable_fraction' must be between 0 and 1$",
        ),
        (
            {**FIXED, "target_usable_fraction": -0.1},
            {},
            ValueError,
            r"'smart_charging\.target_usable_fraction' must be between 0 and 1$",
        ),
        (
            {**FIXED, "target_usable_fraction": "half"},
            {},
            TypeError,
            r"'smart_charging\.target_usable_fraction' must be a finite number",
        ),
        (
            {**FIXED, "charge_periods": []},
            {},
            ValueError,
            r"'smart_charging\.charge_periods' needs at least 1 entry",
        ),
        (
            {**FIXED, "discharge_periods": []},
            {},
            ValueError,
            r"'smart_charging\.discharge_periods' needs at least 1 entry",
        ),
        (
            {**FIXED, "charge_periods": "off_peak"},
            {},
            TypeError,
            r"'smart_charging\.charge_periods' must be a list",
        ),
        (
            {**FIXED, "discharge_periods": [3]},
            {},
            TypeError,
            r"'smart_charging\.discharge_periods\[0\]' must be a non-empty string",
        ),
        (
            {**FIXED, "charge_periods": ["night"]},
            {},
            ValueError,
            r"'smart_charging\.charge_periods' has period\(s\) night that schedule 'pt_mainland_2026_daily_bi' "
            r"does not have\. Its periods: off_peak, peak\.",
        ),
        (
            {**FIXED, "discharge_periods": ["peak", "mid_peak"]},
            {},
            ValueError,
            r"'smart_charging\.discharge_periods' has period\(s\) mid_peak that schedule",
        ),
        (
            {**FIXED, "discharge_periods": ["peak", "off_peak"]},
            {},
            ValueError,
            r"'smart_charging\.charge_periods' and 'smart_charging\.discharge_periods' share off_peak; "
            r"every step either charges or discharges \(ADR 0002 A8\)",
        ),
        (
            _without(FIXED, "grid_charge_efficiency"),
            {},
            ValueError,
            r"'smart_charging' needs smart_charging\.grid_charge_efficiency for mode = 'fixed_target'",
        ),
        (
            _without(FIXED, "target_usable_fraction", "charge_periods"),
            {},
            ValueError,
            r"'smart_charging' needs smart_charging\.target_usable_fraction, smart_charging\.charge_periods for",
        ),
        (
            {**FIXED, "grid_charge_efficiency": 0.0},
            {},
            ValueError,
            r"'smart_charging\.grid_charge_efficiency' must be between 0 and 1 \(exclusive of the lower bound\)",
        ),
        (
            {**FIXED, "grid_charge_efficiency": 1.05},
            {},
            ValueError,
            r"'smart_charging\.grid_charge_efficiency' must be between 0 and 1 \(exclusive of the lower bound\)",
        ),
        (
            {**FIXED, "grid_import_limit_w": 0},
            {},
            ValueError,
            r"'smart_charging\.grid_import_limit_w' must be > 0",
        ),
        (
            {**FIXED, "grid_import_limit_w": math.inf},
            {},
            ValueError,
            r"'smart_charging\.grid_import_limit_w' must be a finite number",
        ),
        (
            {"mode": "disabled", "charge_periods": ["off_peak"], "grid_charge_efficiency": 0.9},
            {},
            ValueError,
            r"'smart_charging\.mode' = 'disabled' takes no other keys; remove smart_charging\.charge_periods, "
            r"smart_charging\.grid_charge_efficiency",
        ),
        (
            FIXED,
            {"tariff": None},
            ValueError,
            r"'smart_charging\.mode' = 'fixed_target' needs a \[tariff\]",
        ),
        (
            FIXED,
            {"battery_kwh": 0.0},
            ValueError,
            r"'smart_charging\.mode' = 'fixed_target' needs a battery; set battery_kwh > 0",
        ),
    ],
)
def test_smart_charging_config_is_checked_at_construction(table, extra, error, message):
    with pytest.raises(error, match=message):
        _app(table, **extra)


def test_a_valid_fixed_target_table_resolves_to_a_spec():
    table = {**FIXED, "mode": "Fixed_Target", "charge_periods": ["off_peak", "off_peak"]}

    resolved = _app(table)._resolved

    assert resolved.smart_charging == SmartChargingSpec(
        mode="fixed_target",
        target_usable_fraction=0.5,
        charge_periods=("off_peak",),
        discharge_periods=("peak",),
        grid_charge_efficiency=0.95,
        grid_import_limit_w=5000.0,
    )
    assert _app(_without(FIXED, "grid_import_limit_w"))._resolved.smart_charging.grid_import_limit_w is None
    assert _app({**FIXED, "grid_import_limit_w": None})._resolved.smart_charging.grid_import_limit_w is None


def test_disabled_needs_neither_a_tariff_nor_a_battery():
    resolved = App({**BASE, "battery_kwh": 0.0, "smart_charging": {"mode": "disabled"}})._resolved

    assert resolved.smart_charging == SmartChargingSpec(mode="disabled")
    assert App(BASE)._resolved.smart_charging is None


# --- App runs ----------------------------------------------------------------


def _record_app_artifacts(monkeypatch):
    import breos.app as app_module

    artifacts = []
    run_app = app_module.run_app_simulation

    def record(*args):
        result = run_app(*args)
        artifacts.append(result)
        return result

    monkeypatch.setattr(app_module, "run_app_simulation", record)
    return artifacts


@pytest.mark.usefixtures("_patch_weather")
def test_fixed_target_app_run_charges_from_the_grid_and_reports_it(monkeypatch):
    from tests.energy_conservation import assert_energy_conservation, assert_origin_reconciliation

    artifacts = _record_app_artifacts(monkeypatch)
    greedy = App({**BASE, "tariff": TOU, "emissions_country": "PT"})
    smart = _app(FIXED, emissions_country="PT")
    greedy.simulate()
    smart.simulate()
    plain, charged = greedy.result(), smart.result()
    frame = artifacts[1].first_year_results_df

    # The step ledger balances and every origin reconciles under grid charging.
    config = smart._resolved
    from breos.projection import build_battery_config

    assert_energy_conservation(frame, build_battery_config(smart._cfg, config, initial_soh=100.0))
    assert_origin_reconciliation(frame, 1.0)
    assert frame["Grid_AC_To_Battery"].sum() > 0.0
    # No step both grid-charges and exports.
    assert not ((frame["Grid_AC_To_Battery"] > 0.0) & (frame["PV_AC_Export"] > 0.0)).any()

    block = charged["smart_charging"]
    year1 = block["yearly"][0]
    assert block["mode"] == "fixed_target" and block["terminal_convention"] == "physical_carry"
    assert year1["grid_charge_ac_kwh"] > 0.0
    assert year1["grid_charge_conversion_loss_kwh"] == pytest.approx(0.05 * year1["grid_charge_ac_kwh"], abs=0.02)
    assert year1["battery_ac_to_load_kwh"]["grid_origin"] > 0.0
    # Grid charging happens only off peak, at 0.11 per kWh.
    assert year1["grid_charge_cost_year1_prices"] == pytest.approx(
        frame["Grid_AC_To_Battery"].sum() / 1000 * 0.11, abs=0.006
    )
    for key in ("initial_stored_energy", "final_stored_energy"):
        state = block[key]
        assert state["total_wh"] == pytest.approx(
            state["pv_origin_wh"] + state["grid_origin_wh"] + state["unattributed_wh"], abs=0.02
        )
    assert block["initial_stored_energy"]["unattributed_wh"] == block["initial_stored_energy"]["total_wh"]
    assert len(block["yearly"]) == BASE["projection_years"]

    # A rounded residue is reported as 0.0, never -0.0.
    def values(node):
        if isinstance(node, dict):
            return [v for child in node.values() for v in values(child)]
        if isinstance(node, list):
            return [v for child in node for v in values(child)]
        return [node] if isinstance(node, float) else []

    assert not any(value == 0.0 and math.copysign(1.0, value) < 0 for value in values(block))

    record = charged["provenance"]["smart_charging"]
    assert record["mode"] == "fixed_target"
    assert record["charge_periods"] == ["off_peak"]
    assert record["schedule_hash"] == charged["provenance"]["tariff"]["schedule_hash"]
    assert len(record["instruction_hash"]) == 64
    assert "smart_charging" not in plain and "smart_charging" not in plain["provenance"]

    # Grid energy shifted through the battery is imported, so it earns no
    # avoided emissions, and its round-trip loss counts against the system
    # (A10): self-consumed CO2 falls by the grid charge net of its delivery.
    assert charged["grid_import_kwh"] > plain["grid_import_kwh"]
    assert charged["co2_avoided_self_consumption_year1_kg"] < plain["co2_avoided_self_consumption_year1_kg"]
    ci = smart._resolved.emissions_params.avoided_intensity_gco2_kwh
    shift = year1["battery_ac_to_load_kwh"]["grid_origin"] - year1["grid_charge_ac_kwh"]
    assert charged["co2_avoided_self_consumption_year1_kg"] == pytest.approx(
        (charged["self_consumption_kwh"] + shift) * ci / 1000, abs=0.05
    )


@pytest.mark.usefixtures("_patch_weather")
def test_fixed_target_reports_its_grid_charge_cost_at_the_top_level():
    app = _app(FIXED)
    app.simulate()
    result = app.result()

    for key in (
        "grid_import_cost_year1_prices",
        "grid_export_revenue_year1_prices",
        "fixed_charge_year1_prices",
        "no_system_import_cost_year1_prices",
    ):
        assert key in result
    grid_charge = result["grid_charge_cost_year1_prices"]
    assert grid_charge == result["smart_charging"]["yearly"][0]["grid_charge_cost_year1_prices"]
    # The grid charge is part of the grid import cost, not added to it.
    assert 0.0 < grid_charge <= result["grid_import_cost_year1_prices"]


@pytest.mark.usefixtures("_patch_weather")
def test_disabled_smart_charging_matches_no_table():
    omitted = App({**BASE, "tariff": TOU})
    disabled = _app({"mode": "disabled"})
    omitted.simulate()
    disabled.simulate()

    first, second = deepcopy(omitted.result()), deepcopy(disabled.result())
    # The configuration echo records what was given; everything else is identical.
    assert first["provenance"]["resolved_config"].pop("smart_charging") is None
    assert second["provenance"]["resolved_config"].pop("smart_charging") == {"mode": "disabled"}
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)


# --- The fixed-target controller ---------------------------------------------


def test_fixed_target_follows_the_resolved_tri_hourly_periods():
    # 12 January 2026, a winter weekday, in 15-minute steps.
    index = pd.date_range("2026-01-12", periods=96, freq="15min", tz=LISBON)
    tariff = _tri_tariff(index)
    labels = np.asarray(tariff.period_labels)

    instructions = resolve_instructions(TRI_SPEC, tariff)

    assert len(instructions) == 96
    assert set(labels) == {"off_peak", "mid_peak", "peak"}
    np.testing.assert_array_equal(instructions.discharge_allowed, labels == "peak")
    np.testing.assert_array_equal(instructions.reserve_fraction, np.zeros(96))
    np.testing.assert_array_equal(instructions.grid_target_fraction, np.where(labels == "off_peak", 0.6, np.nan))
    # 08:45 is mid-peak and 09:00 peak (Diretiva ERSE n.º 1/2026); mid-peak
    # neither discharges nor grid-charges.
    assert labels[35] == "mid_peak" and not instructions.discharge_allowed[35]
    assert np.isnan(instructions.grid_target_fraction[35])
    assert labels[36] == "peak" and instructions.discharge_allowed[36]
    assert labels[8] == "off_peak" and instructions.grid_target_fraction[8] == 0.6
    assert instructions.grid_charge_efficiency == 0.9
    assert instructions.grid_import_limit_w == math.inf


def test_fixed_target_stays_aligned_across_a_dst_day():
    # 29 March 2026 has 23 local hours; a UTC index keeps every instant.
    index = pd.date_range("2026-03-28", "2026-03-31", freq="15min", tz="UTC", inclusive="left")
    tariff = _tri_tariff(index)
    spec = SmartChargingSpec(
        mode="fixed_target",
        target_usable_fraction=1.0,
        charge_periods=("off_peak",),
        discharge_periods=("mid_peak", "peak"),
        grid_charge_efficiency=1.0,
        grid_import_limit_w=3000.0,
    )

    instructions = resolve_instructions(spec, tariff)

    labels = np.asarray(tariff.period_labels)
    assert len(instructions) == len(index)
    np.testing.assert_array_equal(instructions.discharge_allowed, labels != "off_peak")
    np.testing.assert_array_equal(np.isnan(instructions.grid_target_fraction), labels != "off_peak")
    assert instructions.grid_import_limit_w == 3000.0


def test_disabled_gives_no_instructions():
    assert resolve_instructions(SmartChargingSpec(mode="disabled"), None) is None


def test_the_controller_checks_its_tariff():
    index = pd.date_range("2026-01-12", periods=4, freq="15min", tz=LISBON)
    with pytest.raises(ValueError, match="needs a resolved tariff"):
        resolve_instructions(TRI_SPEC, None)
    spec = SmartChargingSpec(
        mode="fixed_target",
        target_usable_fraction=0.5,
        charge_periods=("night",),
        discharge_periods=("peak",),
        grid_charge_efficiency=0.9,
    )
    with pytest.raises(ValueError, match=r"'smart_charging\.charge_periods' has period\(s\) night"):
        resolve_instructions(spec, _tri_tariff(index))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"mode": "greedy"}, r"'smart_charging\.mode' must be one of"),
        ({"mode": "disabled", "target_usable_fraction": 0.5}, "takes no other settings"),
        ({"mode": "fixed_target", "charge_periods": ("off_peak",)}, r"needs smart_charging\.target_usable_fraction"),
        (
            {
                "mode": "fixed_target",
                "target_usable_fraction": 0.5,
                "charge_periods": ("peak",),
                "discharge_periods": ("peak",),
                "grid_charge_efficiency": 0.9,
            },
            "share peak",
        ),
    ],
)
def test_a_spec_built_directly_stays_coherent(fields, message):
    with pytest.raises(ValueError, match=message):
        SmartChargingSpec(**fields)


def test_provenance_records_parameters_and_hashes():
    index = pd.date_range("2026-01-12", periods=96, freq="15min", tz=LISBON)
    tariff = _tri_tariff(index)
    instructions = resolve_instructions(TRI_SPEC, tariff)

    record = smart_charging_provenance(TRI_SPEC, instructions, tariff)

    assert record == {
        "mode": "fixed_target",
        "target_usable_fraction": 0.6,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
        "grid_charge_efficiency": 0.9,
        "grid_import_limit_w": None,
        "instruction_hash": instructions.instruction_hash(),
        "schedule_hash": tariff.schedule_hash,
        "terminal_convention": "physical_carry",
    }
    json.dumps(record, allow_nan=False)
    with pytest.raises(ValueError, match="96 steps but the tariff has 4"):
        smart_charging_provenance(TRI_SPEC, instructions, _tri_tariff(index[:4]))


def test_monte_carlo_runs_fixed_target_charging(tmp_path, write_multiyear_weather):
    from breos.montecarlo import MonteCarloSettings, run_montecarlo

    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3, collect_yearly=True)
    greedy = run_montecarlo({**BASE, "tariff": TOU}, settings)
    charged = run_montecarlo({**BASE, "tariff": TOU, "smart_charging": FIXED}, settings)

    assert (greedy.yearly["Grid_AC_To_Battery_kWh"] == 0.0).all()
    assert (charged.yearly["Grid_AC_To_Battery_kWh"] > 0.0).all()
    assert (charged.yearly["Import_kWh"] > greedy.yearly["Import_kWh"]).all()
    record = charged.provenance["smart_charging"]
    assert record["mode"] == "fixed_target"
    assert record["schedule_hash"] == charged.provenance["tariff"]["schedule_hash"]
    assert "smart_charging" not in greedy.provenance


def test_the_smart_charging_block_never_reports_negative_zero():
    from breos.app_results import _round2

    assert math.copysign(1.0, _round2(-1e-13)) == 1.0
    assert _round2(-0.004) == 0.0 and math.copysign(1.0, _round2(-0.004)) == 1.0
    assert _round2(12.345678) == 12.35
