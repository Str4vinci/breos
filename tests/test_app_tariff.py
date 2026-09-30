"""Time-of-use valuation through App, Monte Carlo and the shared projection loop (ADR 0002, 0003 E7)."""

import numpy as np
import pandas as pd
import pytest

from breos.app import App
from breos.app_config import resolve_app_config
from breos.battery import align_simulation_inputs
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.projection import ProjectionYear, run_projection
from breos.runners.app import run_app_simulation
from breos.smart_charging import SmartChargingSpec, resolve_instructions

BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0, "projection_years": 3}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
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
