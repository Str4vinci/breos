"""Time-of-use valuation through App, Monte Carlo and the shared projection loop (ADR 0002, 0003 E7)."""

import dataclasses
import json
from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from breos.app import App
from breos.app_config import resolve_app_config
from breos.battery import align_simulation_inputs
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.projection import ProjectionYear, run_projection
from breos.runners.app import run_app_simulation
from breos.smart_charging import FixedTargetDayController, SmartChargingSpec, resolve_instructions
from breos.tariffs import ScheduleDefinition

BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 3}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
CUSTOM_SCHEDULE = {
    "identifier": "example_supplier_2023",
    "version": "2023-01",
    "timezone": "Europe/Lisbon",
    "cycle": "weekly",
    "periods": ["peak", "off_peak"],
    "source": "Illustrative supplier tariff sheet",
    "effective_from": date(2023, 1, 1),
    "effective_to": date(2023, 12, 31),
    "rules": [
        {
            "days": "weekday",
            "season": "all",
            "intervals": {"off_peak": [["00:00", "08:00"], ["22:00", "24:00"]], "peak": [["08:00", "22:00"]]},
        },
        {"days": "saturday", "season": "all", "intervals": {"off_peak": [["00:00", "24:00"]]}},
        {"days": "sunday", "season": "all", "intervals": {"off_peak": [["00:00", "24:00"]]}},
    ],
    "holidays": {
        "day_type": "sunday",
        "source": "Illustrative supplier holiday calendar",
        "dates": {"2023": [date(2023, 1, 2)]},
    },
}
CUSTOM_TARIFF = {
    "custom_schedule": CUSTOM_SCHEDULE,
    "currency": "EUR",
    "import_prices": {"peak": 0.31, "off_peak": 0.12},
    "export_prices": {"all": 0.05},
}


@pytest.mark.parametrize(
    ("tariff", "extra", "error", "message"),
    [
        ({**TOU, "schedule": "pt_2026"}, {}, ValueError, r"'tariff\.schedule' must be one of"),
        ({**TOU, "currency": "USD"}, {}, ValueError, r"'tariff\.currency' must be one of: EUR"),
        ({**TOU, "import_prices": {"peak": 0.28}}, {}, ValueError, r"has no price for off_peak"),
        (
            {**TOU, "import_prices": {"peak": 0.3, "off_peak": 0.1, "super_off_peak": 0.05}},
            {},
            ValueError,
            "super_off_peak",
        ),
        ({**TOU, "import_prices": {"all": -0.1}}, {}, ValueError, r"'tariff\.import_prices\.all' must be >= 0"),
        ({k: v for k, v in TOU.items() if k != "currency"}, {}, ValueError, r"'tariff' needs tariff\.currency"),
        (TOU, {"costs": {"electricity_cost": 0.2}}, ValueError, r"costs\.electricity_cost would price them twice"),
        (
            {**TOU, "schedule": "pt_mainland_2026_daily_tri", "import_prices": {"all": 0.2}},
            {},
            ValueError,
            r"needs steps that divide 30 minutes, which 'h' steps do not; use resolution = \"15min\"",
        ),
        (
            TOU,
            {"location": {"latitude": 52.5, "longitude": 13.4, "timezone": "Europe/Berlin"}},
            ValueError,
            "does not move a schedule",
        ),
        ({**TOU, "study_date": "July 2027"}, {}, ValueError, "ISO date"),
    ],
)
def test_tariff_config_is_checked_at_construction(tariff, extra, error, message):
    with pytest.raises(error, match=message):
        App({**BASE, **extra, "tariff": tariff})


def test_a_schedule_finer_than_every_app_step_names_what_it_needs(monkeypatch):
    # A 10-minute boundary, as a custom schedule may have: no App resolution fits.
    monkeypatch.setattr("breos.app_config.schedule_resolution_minutes", lambda schedule, years=None: 10)
    with pytest.raises(ValueError, match="needs steps that divide 10 minutes.*App offers no step that divides 10"):
        App({**BASE, "resolution": "15min", "tariff": TOU})


def _artifacts(config):
    app = App(config)
    return app, run_app_simulation(app._resolved, app._runtime_dependencies())


@pytest.mark.usefixtures("_patch_weather")
def test_a_flat_tariff_reproduces_flat_pricing():
    flat_app = App(BASE)
    flat_app.simulate()
    params = flat_app._resolved.cost_params
    tariff = {
        "schedule": "pt_mainland_2026_daily_bi",
        "currency": "EUR",
        "import_prices": {"all": params.electricity_cost},
        "export_prices": {"all": params.electricity_sold_cost},
        "fixed_charge_per_day": params.daily_power_cost,
    }
    tou_app = App({**BASE, "tariff": tariff})
    tou_app.simulate()

    flat, tou = flat_app.result(), tou_app.result()
    # sum(energy * price) against price * sum(energy): the same to rounding.
    assert tou["npv_savings"] == pytest.approx(flat["npv_savings"], abs=0.01)
    assert tou["lcoe_per_kwh"] == flat["lcoe_per_kwh"]
    assert tou["grid_import_kwh"] == flat["grid_import_kwh"]
    assert "tariff" not in flat["provenance"]
    assert tou["provenance"]["tariff"]["calendar_policy"] == "replay_start_year"


@pytest.mark.usefixtures("_patch_weather")
@pytest.mark.parametrize(("resolution", "hours"), [("h", 1.0), ("15min", 0.25)])
def test_tou_money_reconciles_with_the_step_ledger(resolution, hours):
    app, artifacts = _artifacts({**BASE, "resolution": resolution, "tariff": TOU})
    frame = artifacts.first_year_results_df
    local = pd.DatetimeIndex(frame["Datetime"]).tz_convert("Europe/Lisbon")
    import_price = np.where((local.hour >= 8) & (local.hour < 22), 0.28, 0.11)
    year1 = artifacts.yearly_df.iloc[0]

    def priced(column, price):
        return float((frame[column] * price).sum() * hours / 1000)

    assert year1["Import_Cost"] == pytest.approx(priced("Import_From_Grid", import_price), rel=1e-12)
    assert year1["Baseline_Import_Cost"] == pytest.approx(priced("Houseload", import_price), rel=1e-12)
    assert year1["Export_Revenue"] == pytest.approx(priced("PV_AC_Export", 0.05), rel=1e-12)
    # The DST days are priced over their real 23 and 25 hours: no step is
    # invented, dropped, or counted twice.
    steps_per_day = pd.Series(local.date).value_counts()
    assert steps_per_day[pd.Timestamp("2023-03-26").date()] == 23 / hours
    assert steps_per_day[pd.Timestamp("2023-10-29").date()] == 25 / hours
    assert len(frame) == 8760 / hours
    assert year1["Fixed_Charge"] == pytest.approx(365 * 0.25)
    projection = artifacts.cost_projection
    assert projection["Cost_Import"].iloc[0] == year1["Import_Cost"]
    # Energy is unchanged: pricing does not touch dispatch.
    flat = _artifacts({**BASE, "resolution": resolution})[1]
    pd.testing.assert_series_equal(artifacts.yearly_df["Import_kWh"], flat.yearly_df["Import_kWh"])


@pytest.mark.usefixtures("_patch_weather")
def test_a_tou_run_reports_its_year1_money_at_the_top_level(monkeypatch):
    import breos.app as app_module

    artifacts = []
    run_app = app_module.run_app_simulation

    def record(*args):
        artifacts.append(run_app(*args))
        return artifacts[-1]

    monkeypatch.setattr(app_module, "run_app_simulation", record)
    app = App({**BASE, "tariff": TOU})
    app.simulate()
    result = app.result()
    year1 = artifacts[0].yearly_df.iloc[0]

    assert result["grid_import_cost_year1_prices"] == round(year1["Import_Cost"], 2)
    assert result["grid_export_revenue_year1_prices"] == round(year1["Export_Revenue"], 2)
    assert result["no_system_import_cost_year1_prices"] == round(year1["Baseline_Import_Cost"], 2)
    assert result["fixed_charge_year1_prices"] == 91.25  # 365 days at 0.25
    assert result["grid_import_cost_year1_prices"] == result["financial"][1]["cost_import"]
    # Without smart charging the battery charges from PV only.
    assert "grid_charge_cost_year1_prices" not in result


@pytest.mark.usefixtures("_patch_weather")
def test_cheap_nights_lower_the_bill():
    night = {**TOU, "import_prices": {"peak": 0.40, "off_peak": 0.05}}
    day = {**TOU, "import_prices": {"peak": 0.05, "off_peak": 0.40}}

    night_run = _artifacts({**BASE, "tariff": night})[1].yearly_df
    day_run = _artifacts({**BASE, "tariff": day})[1].yearly_df

    # The household imports mostly outside PV hours, and more of it at night.
    assert night_run["Import_Cost"].iloc[0] != day_run["Import_Cost"].iloc[0]
    assert (night_run["Import_Cost"] <= night_run["Baseline_Import_Cost"]).all()


def test_frames_and_summaries_price_a_year_identically():
    resolved = resolve_app_config({**BASE, "tariff": TOU})
    idx = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    pv = pd.Series(np.where((hour >= 9) & (hour < 16), 2600.0, 0.0), index=idx)
    load = pd.DataFrame({"Load": np.where((hour >= 19) | (hour < 3), 700.0, 250.0)}, index=idx)
    temperature = pd.Series(20.0, index=idx)
    aligned = align_simulation_inputs(pv, load, temperature, freq="h")
    tariff = resolved.tariff.resolve(aligned.index, resolved.timezone)

    def run(summary):
        def year_inputs(year_idx):
            if summary:
                return ProjectionYear(pv_degradation_factor=1.0, aligned=aligned)
            return ProjectionYear(pv_degradation_factor=1.0, pv_dc=pv, houseload=load, temperature_series=temperature)

        return run_projection(
            resolved.cfg, resolved, 2, year_inputs, has_battery=True, execution_backend="python", tariff=tariff
        ).yearly_df

    frames, summaries = run(False), run(True)
    columns = ["Import_Cost", "Export_Revenue", "Baseline_Import_Cost", "Fixed_Charge"]
    pd.testing.assert_frame_equal(frames[columns], summaries[columns], check_exact=True)


def test_a_tariff_on_another_calendar_is_refused():
    resolved = resolve_app_config({**BASE, "tariff": TOU})
    idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
    tariff = resolved.tariff.resolve(idx[:24], resolved.timezone)
    aligned = align_simulation_inputs(
        pd.Series(0.0, index=idx), pd.DataFrame({"Load": 100.0}, index=idx), None, freq="h"
    )

    with pytest.raises(ValueError, match="different calendar"):
        run_projection(
            resolved.cfg,
            resolved,
            1,
            lambda _: ProjectionYear(pv_degradation_factor=1.0, aligned=aligned),
            has_battery=True,
            execution_backend="python",
            tariff=tariff,
        )


def test_a_smart_charging_year_on_another_calendar_names_the_calendar():
    # The detailed path checks the calendar before it simulates, so a year off
    # the tariff's calendar is refused for that reason, not for the
    # instructions' step count.
    resolved = resolve_app_config({**BASE, "tariff": TOU})
    idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
    tariff = resolved.tariff.resolve(idx[:24], resolved.timezone)
    spec = SmartChargingSpec(
        mode="fixed_target",
        target_usable_fraction=0.5,
        charge_periods=("off_peak",),
        discharge_periods=("peak",),
        grid_charge_efficiency=0.95,
    )

    with pytest.raises(ValueError, match="different calendar"):
        run_projection(
            resolved.cfg,
            resolved,
            1,
            lambda _: ProjectionYear(
                pv_degradation_factor=1.0,
                pv_dc=pd.Series(0.0, index=idx),
                houseload=pd.DataFrame({"Load": 100.0}, index=idx),
                temperature_series=pd.Series(20.0, index=idx),
            ),
            has_battery=True,
            execution_backend="python",
            tariff=tariff,
            instructions=resolve_instructions(spec, tariff),
        )


SMART = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.6,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 5000,
}


def _spy_on_projection(monkeypatch):
    import breos.runners.app as runner

    calls = []
    run_projection_ = runner.run_projection

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return run_projection_(*args, **kwargs)

    monkeypatch.setattr(runner, "run_projection", spy)
    return calls


@pytest.mark.usefixtures("_patch_weather")
@pytest.mark.parametrize("resolution", ["h", "15min"])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_app_fixed_target_runs_through_the_daily_controller_unchanged(monkeypatch, resolution, backend):
    # ADR 0002 A11: App's configured fixed target runs through the civil-day
    # controller seam; a replayed schedule stays on the static path. The two
    # must give the same ledger, year rows and carried state bit for bit.
    if backend == "numba":
        pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    calls = _spy_on_projection(monkeypatch)
    app = App({**BASE, "resolution": resolution, "execution_backend": backend, "tariff": TOU, "smart_charging": SMART})
    deps = app._runtime_dependencies()
    controlled = run_app_simulation(app._resolved, deps)
    static = run_app_simulation(app._resolved, deps, instructions=controlled.instructions)

    assert isinstance(calls[0]["day_controller"], FixedTargetDayController) and calls[0]["instructions"] is None
    assert calls[0]["replay_seam"] is True
    assert calls[1]["day_controller"] is None and calls[1]["instructions"] is controlled.instructions
    pd.testing.assert_frame_equal(controlled.first_year_results_df, static.first_year_results_df, check_exact=True)
    pd.testing.assert_frame_equal(controlled.yearly_df, static.yearly_df, check_exact=True)
    assert dataclasses.replace(controlled.projection.carry, controller_carry=None) == static.projection.carry
    assert controlled.smart_charging["instruction_hash"] == controlled.instructions.instruction_hash()
    assert controlled.first_year_results_df["Grid_AC_To_Battery"].sum() > 0
    carry = controlled.projection.carry.controller_carry
    assert carry.next_project_step_ordinal == 3 * len(controlled.first_year_results_df)
    assert carry.complete_days_observed == 3 * 365


def test_montecarlo_prices_every_trajectory_with_the_tariff(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3, collect_yearly=True)

    flat = run_montecarlo(BASE, settings)
    tou = run_montecarlo({**BASE, "tariff": TOU}, settings)

    assert tou.provenance["tariff"]["schedule"] == "pt_mainland_2026_daily_bi"
    assert "tariff" not in flat.provenance
    pd.testing.assert_series_equal(tou.yearly["Import_kWh"], flat.yearly["Import_kWh"])
    assert not np.allclose(tou.runs["npv_savings"], flat.runs["npv_savings"])


@pytest.mark.usefixtures("_patch_weather")
def test_spanish_2_0td_in_2026():
    config = {
        **BASE,
        "location": {"latitude": 40.42, "longitude": -3.70, "timezone": "Europe/Madrid"},
        "start_date": "2026-01-01",
        "tariff": {
            "schedule": "es_2_0td",
            "currency": "EUR",
            "import_prices": {"peak": 0.25, "mid_peak": 0.17, "off_peak": 0.11},
            "export_prices": {"all": 0.06},
        },
    }
    app = App(config)
    app.simulate()
    record = app.result()["provenance"]["tariff"]

    assert (record["schedule"], record["timezone"], record["calendar_year"]) == ("es_2_0td", "Europe/Madrid", 2026)
    assert record["fixed_charge_per_day"] == 0.0


def test_study_date_is_accepted_as_a_date():
    from datetime import date

    resolved = resolve_app_config(
        {
            **BASE,
            "resolution": "15min",
            "tariff": {**TOU, "schedule": "pt_mainland_2027_daily_bi", "study_date": date(2027, 7, 1)},
        }
    )
    assert resolved.tariff.study_date == date(2027, 7, 1)


@pytest.mark.parametrize(
    ("tariff", "message"),
    [
        ({key: value for key, value in CUSTOM_TARIFF.items() if key != "custom_schedule"}, "exactly one"),
        ({**CUSTOM_TARIFF, "schedule": "pt_mainland_2026_daily_bi"}, "exactly one"),
        ({**CUSTOM_TARIFF, "import_prices": {"peak": 0.3}}, "has no price for off_peak"),
        ({**CUSTOM_TARIFF, "export_prices": {"all": 0.05, "night": 0.01}}, "night"),
    ],
)
def test_custom_schedule_selection_and_prices_are_checked(tariff, message):
    with pytest.raises(ValueError, match=message):
        App({**BASE, "tariff": tariff})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update(unexpected=True), r"custom_schedule\.unexpected"),
        (lambda raw: raw["rules"][0].update(unexpected=True), r"rules\[0\]\.unexpected"),
        (lambda raw: raw["holidays"].update(observed=True), r"holidays\.observed"),
        (lambda raw: raw["holidays"]["dates"].update({"all": []}), "four-digit years"),
        (
            lambda raw: raw["rules"].pop(),
            "exactly one rule for sunday/standard",
        ),
        (
            lambda raw: raw["rules"][0]["intervals"].update(
                {"peak": [["07:00", "22:00"]], "off_peak": [["00:00", "08:00"], ["22:00", "24:00"]]}
            ),
            "overlap at minute 420",
        ),
        (
            lambda raw: raw["rules"][0]["intervals"].update(
                {"peak": [["08:00", "22:00"]], "off_peak": [["00:00", "07:00"], ["22:00", "24:00"]]}
            ),
            "gap at minute 420",
        ),
    ],
    ids=["schedule-key", "rule-key", "holiday-key", "holiday-year", "coverage", "overlap", "gap"],
)
def test_custom_schedule_rejects_unknown_and_malformed_nested_values(mutate, message):
    custom = deepcopy(CUSTOM_SCHEDULE)
    mutate(custom)
    with pytest.raises((TypeError, ValueError), match=message):
        App({**BASE, "tariff": {**CUSTOM_TARIFF, "custom_schedule": custom}})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        *[
            (lambda raw, key=key: raw.pop(key), rf"custom_schedule.*{key}")
            for key in ("identifier", "version", "timezone", "cycle", "periods", "rules")
        ],
        *[
            (lambda raw, key=key: raw["rules"][0].pop(key), rf"rules\[0\].*{key}")
            for key in ("days", "season", "intervals")
        ],
        (lambda raw: raw.update(timezone="Europe/Atlantis"), "Unknown IANA tariff timezone"),
        (lambda raw: raw.update(cycle="monthly"), r"custom_schedule\.cycle"),
        (lambda raw: raw.update(periods=["peak", "peak", "off_peak"]), "must not contain duplicates"),
        (lambda raw: raw.update(periods=["Peak", "off_peak"]), "lowercase letters"),
        (lambda raw: raw["rules"][0]["intervals"].update(shoulder=[]), "shoulder"),
        (lambda raw: raw["rules"][1]["intervals"].update(off_peak=[["24:00", "24:00"]]), "start before end"),
        (lambda raw: raw["holidays"]["dates"].update({"2023": [date(2024, 1, 1)]}), "dates in 2023"),
    ],
)
def test_custom_schedule_requires_its_fields_and_valid_metadata(mutate, message):
    custom = deepcopy(CUSTOM_SCHEDULE)
    mutate(custom)
    with pytest.raises((TypeError, ValueError), match=message):
        App({**BASE, "tariff": {**CUSTOM_TARIFF, "custom_schedule": custom}})


def test_custom_schedule_is_a_table_and_is_parsed_once(monkeypatch):
    # A parsed definition would be recorded as its repr and could not be fed back.
    definition = resolve_app_config({**BASE, "tariff": CUSTOM_TARIFF}).tariff.schedule
    with pytest.raises(TypeError, match=r"'tariff\.custom_schedule' must be a table"):
        App({**BASE, "tariff": {**CUSTOM_TARIFF, "custom_schedule": definition}})

    from breos import app_config

    calls = []
    parse = app_config.parse_schedule_definition
    monkeypatch.setattr(
        app_config, "parse_schedule_definition", lambda *args, **kwargs: calls.append(1) or parse(*args, **kwargs)
    )
    App({**BASE, "tariff": CUSTOM_TARIFF})
    assert len(calls) == 1


def test_custom_schedule_must_match_location_before_pv_metadata_resolution(monkeypatch):
    custom = deepcopy(CUSTOM_TARIFF)
    custom["custom_schedule"]["timezone"] = "Europe/Berlin"
    monkeypatch.setattr(
        "breos.app_config.resolve_pv_system", lambda *args, **kwargs: pytest.fail("PV metadata resolved first")
    )

    with pytest.raises(ValueError, match="does not move a schedule"):
        App({**BASE, "tariff": custom})


def test_custom_schedule_resolution_keeps_definition_holidays_and_provenance():
    app_config = {**BASE, "projection_years": 1, "tariff": CUSTOM_TARIFF}
    resolved = resolve_app_config(app_config)
    assert isinstance(resolved.tariff.schedule, ScheduleDefinition)
    assert resolved.tariff.definition.schedule.identifier == "example_supplier_2023"
    assert resolved.tariff.definition.schedule.effective_to == date(2023, 12, 31)

    index = pd.date_range("2023-01-02 07:00", "2023-01-03 10:00", freq="h", tz="Europe/Lisbon")
    classified = resolved.tariff.resolve(index, resolved.timezone)
    assert classified.period_labels[index.get_loc(pd.Timestamp("2023-01-02 09:00", tz="Europe/Lisbon"))] == "off_peak"
    assert classified.period_labels[index.get_loc(pd.Timestamp("2023-01-03 09:00", tz="Europe/Lisbon"))] == "peak"
    without_effective_window = deepcopy(CUSTOM_TARIFF)
    without_effective_window["custom_schedule"].pop("effective_from")
    without_effective_window["custom_schedule"].pop("effective_to")
    unrestricted = resolve_app_config({**BASE, "tariff": without_effective_window})
    # A year covered for a civil day or more needs its explicit calendar. (A
    # shorter graze, as at a UTC year end, is exempt by the classifier.)
    with pytest.raises(ValueError, match="no holiday calendar for 2024"):
        unrestricted.tariff.resolve(
            pd.date_range("2024-01-01", periods=24, freq="h", tz="Europe/Lisbon"), "Europe/Lisbon"
        )


@pytest.mark.usefixtures("_patch_weather")
def test_custom_schedule_run_records_its_config_and_tariff_provenance():
    app = App({**BASE, "projection_years": 1, "tariff": CUSTOM_TARIFF})
    app.simulate()
    result = app.result()

    assert result["result_schema_version"] == "2.4"
    assert result["provenance"]["tariff"]["schedule"] == "example_supplier_2023"
    assert result["provenance"]["tariff"]["schedule_version"] == "2023-01"
    custom = result["provenance"]["resolved_config"]["tariff"]["custom_schedule"]
    assert custom["identifier"] == "example_supplier_2023"
    assert custom["holidays"]["dates"]["2023"] == ["2023-01-02"]

    # The recorded config reproduces the schedule after a JSON round trip.
    recorded = json.loads(json.dumps(result["provenance"]["resolved_config"]))
    replay = App(recorded)
    replay.simulate()
    assert replay.result()["provenance"]["tariff"]["schedule_hash"] == result["provenance"]["tariff"]["schedule_hash"]


def test_custom_schedule_period_names_drive_smart_charging_validation():
    smart_charging = {
        "mode": "fixed_target",
        "target_usable_fraction": 0.5,
        "charge_periods": ["off_peak"],
        "discharge_periods": ["peak"],
        "grid_charge_efficiency": 0.95,
    }
    resolved = resolve_app_config({**BASE, "tariff": CUSTOM_TARIFF, "smart_charging": smart_charging})
    assert resolved.smart_charging.charge_periods == ("off_peak",)

    invalid = {**smart_charging, "charge_periods": ["night"]}
    with pytest.raises(ValueError, match="smart_charging.charge_periods.*night.*example_supplier_2023"):
        App({**BASE, "tariff": CUSTOM_TARIFF, "smart_charging": invalid})


def test_custom_schedule_effective_window_is_enforced_when_resolved():
    custom = deepcopy(CUSTOM_TARIFF)
    custom["custom_schedule"]["effective_to"] = date(2022, 12, 31)
    with pytest.raises(ValueError, match="effective_from.*on or before.*effective_to"):
        App({**BASE, "tariff": custom})

    custom = deepcopy(CUSTOM_TARIFF)
    custom["custom_schedule"]["effective_from"] = date(2024, 1, 1)
    custom["custom_schedule"].pop("effective_to")
    resolved = resolve_app_config({**BASE, "tariff": custom})
    with pytest.raises(ValueError, match="not effective across the index date range"):
        resolved.tariff.resolve(pd.date_range("2023-01-01", periods=2, freq="h", tz="Europe/Lisbon"), "Europe/Lisbon")


def test_custom_schedule_with_ten_minute_boundaries_rejects_app_steps():
    custom = deepcopy(CUSTOM_TARIFF)
    custom["custom_schedule"]["rules"][0]["intervals"] = {
        "off_peak": [["00:00", "07:10"]],
        "peak": [["07:10", "24:00"]],
    }
    with pytest.raises(ValueError, match="needs steps that divide 10 minutes.*App offers no step that divides 10"):
        App({**BASE, "resolution": "15min", "tariff": custom})


def test_montecarlo_accepts_custom_schedule_with_explicit_calendar(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "weather.csv")
    settings = MonteCarloSettings(
        weather_file=str(weather), n_runs=1, years_per_run=1, target_year=2023, seed=2, collect_yearly=True
    )

    result = run_montecarlo({**BASE, "projection_years": 1, "tariff": CUSTOM_TARIFF}, settings)

    assert result.provenance["tariff"]["schedule"] == "example_supplier_2023"
    assert result.provenance["result_schema_version"] == "2.4"
    recorded = result.provenance["resolved_config"]["tariff"]["custom_schedule"]
    assert recorded["identifier"] == "example_supplier_2023"
