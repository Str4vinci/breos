"""Discharge-only smart charging: discharge in chosen tariff periods, never grid-charge (#338).

``mode = "discharge_only"`` takes ``discharge_periods`` alone. It runs on
the same dispatch instructions as ``fixed_target``, so these tests pin the
instructions, the physical result in every entry point (App, Monte Carlo,
projected optimization), backend parity, provenance and sweep keys, and that
discharging in every period is greedy dispatch bit for bit.
"""

import csv
import json
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from breos import cli, optimization
from breos.app import App
from breos.app_config import resolve_tariff_spec
from breos.dispatch_instructions import DispatchInstructions
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.runners.app import run_app_simulation
from breos.smart_charging import (
    FixedTargetDayController,
    SmartChargingSpec,
    resolve_instructions,
    smart_charging_provenance,
)

LISBON = "Europe/Lisbon"
BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 2}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
TRI = {
    "schedule": "pt_mainland_2026_daily_tri",
    "currency": "EUR",
    "import_prices": {"off_peak": 0.10, "mid_peak": 0.18, "peak": 0.30},
    "export_prices": {"all": 0.05},
}
PEAK_ONLY = {"mode": "discharge_only", "discharge_periods": ["peak"]}


def _config(smart_charging=PEAK_ONLY, **extra):
    config = {**BASE, "tariff": TOU, **extra}
    if smart_charging is not None:
        config["smart_charging"] = smart_charging
    return config


def _artifacts(config):
    app = App(config)
    return app, run_app_simulation(app._resolved, app._runtime_dependencies())


# --- configuration ----------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "extra", "error", "message"),
    [
        (
            {"mode": "discharge_only"},
            {},
            ValueError,
            r"needs smart_charging\.discharge_periods for mode = 'discharge_only'",
        ),
        (
            {**PEAK_ONLY, "discharge_periods": []},
            {},
            ValueError,
            r"'smart_charging\.discharge_periods' needs at least 1 entry",
        ),
        *(
            (
                {**PEAK_ONLY, key: value},
                {},
                ValueError,
                rf"'discharge_only' does not take smart_charging\.{key}: it never charges from the grid",
            )
            for key, value in (
                ("charge_periods", ["off_peak"]),
                ("target_usable_fraction", 0.0),
                ("grid_charge_efficiency", 0.95),
                ("grid_import_limit_w", 5000),
                ("grid_import_limit_w", None),
                ("forecast_horizon_days", 2),
                ("target_levels", 3),
                ("soc_states", 5),
            )
        ),
        (
            {**PEAK_ONLY, "discharge_periods": ["evening"]},
            {},
            ValueError,
            r"'smart_charging\.discharge_periods' has period\(s\) evening that schedule",
        ),
        (PEAK_ONLY, {"tariff": None}, ValueError, r"'discharge_only' needs a \[tariff\]"),
        (PEAK_ONLY, {"battery_kwh": 0.0}, ValueError, r"'discharge_only' needs a battery; set battery_kwh > 0"),
    ],
)
def test_discharge_only_takes_its_discharge_periods_alone(table, extra, error, message):
    config = _config(table)
    config.update(extra)
    if config.get("tariff") is None:
        config.pop("tariff")
    with pytest.raises(error, match=message):
        App(config)


def test_a_discharge_only_table_resolves_to_a_spec_without_grid_settings():
    app = App(_config({**PEAK_ONLY, "discharge_periods": ["peak", "peak"]}))
    spec = app._resolved.smart_charging
    assert spec == SmartChargingSpec(mode="discharge_only", discharge_periods=("peak",))
    assert spec.charge_periods == ()
    assert spec.target_usable_fraction is None
    assert spec.grid_charge_efficiency is None
    assert spec.grid_import_limit_w is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({}, r"needs smart_charging\.discharge_periods"),
        ({"discharge_periods": ("peak",), "charge_periods": ("off_peak",)}, r"remove smart_charging\.charge_periods"),
        ({"discharge_periods": ("peak",), "grid_charge_efficiency": 0.9}, r"smart_charging\.grid_charge_efficiency"),
        ({"discharge_periods": ("peak",), "soc_states": 5}, r"smart_charging\.soc_states"),
    ],
)
def test_a_discharge_only_spec_built_directly_stays_coherent(fields, message):
    with pytest.raises(ValueError, match=message):
        SmartChargingSpec(mode="discharge_only", **fields)


# --- instructions -----------------------------------------------------------


def _tri_tariff(index):
    spec = resolve_tariff_spec({"tariff": TRI, "costs": None, "resolution": "15min"}, LISBON)
    return spec.resolve(index, LISBON)


def test_discharge_only_instructions_gate_discharge_and_never_grid_charge():
    index = pd.date_range("2026-01-05", periods=96 * 3, freq="15min", tz=LISBON)
    tariff = _tri_tariff(index)
    spec = SmartChargingSpec(mode="discharge_only", discharge_periods=("mid_peak", "peak"))
    instructions = resolve_instructions(spec, tariff)
    labels = np.asarray(tariff.period_labels)

    np.testing.assert_array_equal(instructions.discharge_allowed, labels != "off_peak")
    assert np.isnan(instructions.grid_target_fraction).all()
    assert (instructions.reserve_fraction == 0.0).all()
    assert instructions.grid_charge_efficiency == 1.0 and math.isinf(instructions.grid_import_limit_w)

    every_period = SmartChargingSpec(mode="discharge_only", discharge_periods=tuple(tariff.schedule.periods))
    assert resolve_instructions(every_period, tariff) == DispatchInstructions.noop(len(index))


def test_discharge_only_provenance_records_the_grid_settings_as_unset():
    index = pd.date_range("2026-01-05", periods=96, freq="15min", tz=LISBON)
    tariff = _tri_tariff(index)
    spec = SmartChargingSpec(mode="discharge_only", discharge_periods=("peak",))
    instructions = resolve_instructions(spec, tariff)
    record = smart_charging_provenance(spec, instructions, tariff)
    assert record == {
        "mode": "discharge_only",
        "target_usable_fraction": None,
        "charge_periods": [],
        "discharge_periods": ["peak"],
        "grid_charge_efficiency": None,
        "grid_import_limit_w": None,
        "instruction_hash": instructions.instruction_hash(),
        "schedule_hash": tariff.schedule_hash,
        "terminal_convention": "physical_carry",
    }
    json.dumps(record, allow_nan=False)


# --- App --------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_weather")
@pytest.mark.parametrize(
    ("resolution", "tariff", "backend"),
    [("h", TOU, "python"), ("h", TOU, "numba"), ("15min", TRI, "python"), ("15min", TRI, "numba")],
)
def test_app_discharges_only_in_the_listed_periods(resolution, tariff, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    config = _config(resolution=resolution, tariff=tariff, execution_backend=backend)
    app, artifacts = _artifacts(config)
    frame = artifacts.first_year_results_df
    local = pd.DatetimeIndex(frame["Datetime"]).tz_convert(LISBON)
    labels = np.asarray(app._resolved.tariff.resolve(pd.DatetimeIndex(frame["Datetime"]), LISBON).period_labels)
    peak = labels == "peak"
    assert len(local) == len(labels)

    assert (frame["Grid_AC_To_Battery"] == 0.0).all()
    assert (frame["Grid_DC_To_Battery"] == 0.0).all()
    assert (frame["Battery_Discharge_DC"][~peak] == 0.0).all()
    assert frame["Battery_Discharge_DC"][peak].sum() > 0.0
    # PV still charges the battery, including outside the discharge periods
    # where they have daylight (the tri-hourly mid_peak).
    assert frame["PV_DC_To_Battery"].sum() > 0.0
    if tariff is TRI:
        assert frame["PV_DC_To_Battery"][~peak].sum() > 0.0
    # Stored energy never has a grid origin.
    assert (frame["Battery_Grid_Origin_Energy_End"] == 0.0).all()

    assert isinstance(artifacts.instructions, DispatchInstructions)
    assert artifacts.instructions.instruction_hash() == artifacts.smart_charging["instruction_hash"]

    app.simulate()
    result = app.result()
    assert result["result_schema_version"] == "2.6"
    record = result["provenance"]["smart_charging"]
    assert record["mode"] == "discharge_only"
    assert record["discharge_periods"] == ["peak"]
    assert record["charge_periods"] == [] and record["grid_charge_efficiency"] is None
    assert record["schedule_hash"] == result["provenance"]["tariff"]["schedule_hash"]
    assert result["provenance"]["resolved_config"]["smart_charging"] == PEAK_ONLY
    block = result["smart_charging"]
    assert block["mode"] == "discharge_only"
    assert all(row["grid_charge_ac_kwh"] == 0.0 for row in block["yearly"])
    assert all(row["battery_ac_to_load_kwh"]["grid_origin"] == 0.0 for row in block["yearly"])
    assert result["grid_charge_cost_year1_prices"] == 0.0


@pytest.mark.usefixtures("_patch_weather")
def test_app_runs_discharge_only_through_the_daily_controller(monkeypatch):
    import breos.runners.app as runner

    controllers = []
    run_projection = runner.run_projection

    def spy(*args, **kwargs):
        controllers.append(kwargs.get("day_controller"))
        return run_projection(*args, **kwargs)

    monkeypatch.setattr(runner, "run_projection", spy)
    _artifacts(_config())
    assert len(controllers) == 1 and isinstance(controllers[0], FixedTargetDayController)


@pytest.mark.usefixtures("_patch_weather")
@pytest.mark.parametrize(("resolution", "tariff"), [("h", TOU), ("15min", TRI)])
def test_python_and_numba_give_the_same_discharge_only_ledger(resolution, tariff):
    pytest.importorskip("numba")
    runs = {}
    for backend in ("python", "numba"):
        _app, artifacts = _artifacts(_config(resolution=resolution, tariff=tariff, execution_backend=backend))
        runs[backend] = artifacts
    pd.testing.assert_frame_equal(
        runs["python"].first_year_results_df, runs["numba"].first_year_results_df, check_exact=True
    )
    columns = [column for column in runs["python"].yearly_df.columns if column != "JIT_Cache_State"]
    pd.testing.assert_frame_equal(runs["python"].yearly_df[columns], runs["numba"].yearly_df[columns], check_exact=True)


def _without_smart_charging_echo(result):
    copy = deepcopy(result)
    copy["provenance"]["resolved_config"].pop("smart_charging")
    copy["provenance"].pop("smart_charging", None)
    copy.pop("smart_charging", None)
    copy.pop("grid_charge_cost_year1_prices", None)
    copy["provenance"].pop("execution", None)
    return copy


@pytest.mark.usefixtures("_patch_weather")
@pytest.mark.parametrize(("resolution", "tariff"), [("h", TOU), ("15min", TRI)])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_discharging_in_every_period_is_greedy_dispatch_bit_for_bit(resolution, tariff, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    periods = sorted(tariff["import_prices"])
    every = {"mode": "discharge_only", "discharge_periods": periods}
    greedy_app, greedy = _artifacts(_config(None, resolution=resolution, tariff=tariff, execution_backend=backend))
    every_app, gated = _artifacts(_config(every, resolution=resolution, tariff=tariff, execution_backend=backend))

    pd.testing.assert_frame_equal(greedy.first_year_results_df, gated.first_year_results_df, check_exact=True)
    pd.testing.assert_frame_equal(greedy.yearly_df, gated.yearly_df, check_exact=True)

    greedy_app.simulate()
    every_app.simulate()
    plain, listed = greedy_app.result(), every_app.result()
    assert json.dumps(_without_smart_charging_echo(plain), sort_keys=True, default=str) == json.dumps(
        _without_smart_charging_echo(listed), sort_keys=True, default=str
    )


@pytest.mark.usefixtures("_patch_weather")
def test_peak_only_discharge_keeps_energy_for_the_peak():
    greedy_app, greedy = _artifacts(_config(None))
    _app, gated = _artifacts(_config())
    greedy_frame, gated_frame = greedy.first_year_results_df, gated.first_year_results_df
    labels = np.asarray(
        greedy_app._resolved.tariff.resolve(pd.DatetimeIndex(greedy_frame["Datetime"]), LISBON).period_labels
    )
    off_peak = labels == "off_peak"
    # Greedy dispatch discharges off peak too; the gate moves that to the peak.
    assert greedy_frame["Battery_Discharge_DC"][off_peak].sum() > 0.0
    assert gated_frame["Battery_Discharge_DC"][off_peak].sum() == 0.0
    assert gated_frame["Battery_AC_To_Load"][~off_peak].sum() > greedy_frame["Battery_AC_To_Load"][~off_peak].sum()


@pytest.mark.usefixtures("_patch_weather")
def test_revalue_reprices_a_discharge_only_run():
    app = App(_config())
    app.simulate()
    changes = {"tariff": {"import_prices": {"peak": 0.35, "off_peak": 0.09}}}
    revalued = app.revalue(changes)
    fresh = App(_config(tariff={**TOU, "import_prices": {"peak": 0.35, "off_peak": 0.09}}))
    fresh.simulate()
    expected = fresh.result()

    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert revalued["provenance"]["smart_charging"] == expected["provenance"]["smart_charging"]
    assert revalued["npv_savings"] == pytest.approx(expected["npv_savings"], abs=0.011)
    assert revalued["grid_import_kwh"] == expected["grid_import_kwh"]
    with pytest.raises(ValueError, match="smart_charging is not a price key"):
        app.revalue({"smart_charging": {**PEAK_ONLY, "discharge_periods": ["off_peak"]}})


# --- Monte Carlo and projected optimization ----------------------------------


def test_montecarlo_runs_discharge_only(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3, collect_yearly=True)
    greedy = run_montecarlo({**BASE, "tariff": TOU}, settings)
    gated = run_montecarlo({**BASE, "tariff": TOU, "smart_charging": PEAK_ONLY}, settings)
    every = run_montecarlo(
        {
            **BASE,
            "tariff": TOU,
            "smart_charging": {"mode": "discharge_only", "discharge_periods": ["off_peak", "peak"]},
        },
        settings,
    )

    assert (gated.yearly["Grid_AC_To_Battery_kWh"] == 0.0).all()
    assert not np.array_equal(gated.yearly["Import_kWh"], greedy.yearly["Import_kWh"])
    record = gated.provenance["smart_charging"]
    assert record["mode"] == "discharge_only" and record["discharge_periods"] == ["peak"]
    assert record["schedule_hash"] == gated.provenance["tariff"]["schedule_hash"]
    assert gated.provenance["result_schema_version"] == "2.6"
    pd.testing.assert_frame_equal(every.runs, greedy.runs, check_exact=True)


TARIFF_CASE_INDEX = pd.date_range("2026-01-05", periods=48, freq="h", tz=LISBON)


@pytest.fixture
def tariff_case(monkeypatch):
    weather = pd.DataFrame({"temp_air": 20.0}, index=TARIFF_CASE_INDEX)
    load = pd.DataFrame({"Load": 1000.0}, index=TARIFF_CASE_INDEX)
    pv = pd.Series(
        np.where((TARIFF_CASE_INDEX.hour >= 10) & (TARIFF_CASE_INDEX.hour < 16), 1800.0, 0.0), index=TARIFF_CASE_INDEX
    )
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": LISBON, "altitude": 100},
        "simulation": {"resolution": "h", "years_projection": 2},
        "battery": {"temperature": 20.0},
        "mode": {"fixed_azimuth": 180.0},
        "constraints": {"budget": 100000, "max_area_m2": 100, "max_tilt_deg": 60},
        "tariff": deepcopy(TOU),
        "smart_charging": deepcopy(PEAK_ONLY),
    }
    return weather, load, config


def _evaluate(case, **overrides):
    weather, load, config = case
    config = {**config, **overrides}
    return optimization.evaluate_projected_design(
        weather, load, config, n_modules=4, battery_kwh=5.0, tilt=30.0, azimuth=180.0
    )


def test_optimizer_scores_a_discharge_only_design(tariff_case):
    gated = _evaluate(tariff_case)
    greedy = _evaluate(tariff_case, smart_charging={"mode": "disabled"})
    every = _evaluate(tariff_case, smart_charging={"mode": "discharge_only", "discharge_periods": ["off_peak", "peak"]})

    assert gated.provenance["smart_charging"]["mode"] == "discharge_only"
    assert gated.provenance["result_schema_version"] == "2.6"
    assert (gated.yearly["Grid_AC_To_Battery_kWh"] == 0.0).all()
    assert not gated.yearly["Import_kWh"].equals(greedy.yearly["Import_kWh"])
    pd.testing.assert_frame_equal(every.yearly, greedy.yearly, check_exact=True)
    assert every.metrics["Projected_NPV"] == greedy.metrics["Projected_NPV"]


def test_optimizer_search_shares_discharge_only_scoring(tariff_case):
    pytest.importorskip("pymoo")
    weather, load, config = tariff_case
    problem = optimization.SolarDesignProblem(weather, load, config)
    assert problem.pricing.smart_charging["mode"] == "discharge_only"
    assert problem.pricing.instructions == resolve_instructions(
        SmartChargingSpec(mode="discharge_only", discharge_periods=("peak",)), problem.tariff
    )


# --- sweep ------------------------------------------------------------------

SWEEP_BASE = """
location = "porto"
n_modules = 8
annual_consumption_kwh = 4000
battery_kwh = 5.0

[tariff]
schedule = "pt_mainland_2026_daily_bi"
currency = "EUR"
import_prices = { peak = 0.28, off_peak = 0.11 }
export_prices = { all = 0.05 }

[smart_charging]
mode = "discharge_only"
discharge_periods = ["peak"]
"""


def test_sweep_varies_the_discharge_periods(monkeypatch, tmp_path):
    config_path = tmp_path / "discharge-sweep.toml"
    config_path.write_text(
        SWEEP_BASE + '\n[sweep]\n"smart_charging.discharge_periods" = [["peak"], ["off_peak", "peak"]]\n',
        encoding="utf-8",
    )
    assert cli.main(["validate-config", str(config_path)]) == 0
    seen = []

    class SweepFakeApp:
        def __init__(self, config):
            seen.append(config)

        def simulate(self):
            return None

        def result(self):
            return {}

    monkeypatch.setattr(cli, "App", SweepFakeApp)
    output_path = tmp_path / "discharge-sweep.csv"
    assert cli.main(["sweep", "--config", str(config_path), "--output", str(output_path)]) == 0
    assert [c["smart_charging"] for c in seen] == [
        {"mode": "discharge_only", "discharge_periods": ["peak"]},
        {"mode": "discharge_only", "discharge_periods": ["off_peak", "peak"]},
    ]
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert [row["param_smart_charging.discharge_periods"] for row in rows] == ['["peak"]', '["off_peak", "peak"]']


def test_sweep_compares_greedy_with_discharge_only_as_whole_tables(monkeypatch, tmp_path):
    # A mode sweep alone cannot change the keys a mode takes, so the two
    # policies are swept as whole tables.
    config_path = tmp_path / "policy-sweep.toml"
    config_path.write_text(
        SWEEP_BASE
        + '\n[sweep]\nsmart_charging = [{ mode = "disabled" }, '
        + '{ mode = "discharge_only", discharge_periods = ["peak"] }]\n',
        encoding="utf-8",
    )
    assert cli.main(["validate-config", str(config_path)]) == 0
    seen = []

    class SweepFakeApp:
        def __init__(self, config):
            seen.append(config)

        def simulate(self):
            return None

        def result(self):
            return {}

    monkeypatch.setattr(cli, "App", SweepFakeApp)
    output_path = tmp_path / "policy-sweep.csv"
    assert cli.main(["sweep", "--config", str(config_path), "--output", str(output_path)]) == 0
    assert [c["smart_charging"] for c in seen] == [
        {"mode": "disabled"},
        {"mode": "discharge_only", "discharge_periods": ["peak"]},
    ]


def test_sweep_refuses_a_grid_charge_key_for_discharge_only(tmp_path, capsys):
    config_path = tmp_path / "discharge-sweep.toml"
    config_path.write_text(
        SWEEP_BASE + '\n[sweep]\n"smart_charging.target_usable_fraction" = [0.0, 0.5]\n', encoding="utf-8"
    )
    assert cli.main(["validate-config", str(config_path)]) == 1
    assert "'discharge_only' does not take smart_charging.target_usable_fraction" in capsys.readouterr().err
