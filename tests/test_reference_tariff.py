"""The no-system reference tariff, `[reference_tariff]` (#339, ADR 0002 A13).

The household without the system pays the reference: its import prices on
the whole household load and its own fixed charge, escalated at its own
rate. It never drives the dispatch, so the costs with the system and every
energy flow are those of the same run without it.
"""

import math
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import App, cli, optimization
from breos.app import _revalued_config
from breos.app_config import resolve_app_config, resolve_reference_tariff_spec
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.optimization_config import resolve_optimization_config
from breos.runners.app import run_app_simulation
from tools.generate_app_golden import EXCLUDED_PREFIXES, SCENARIOS, _fake_fetch, flatten

BASE = {**SCENARIOS["native_h_replacement"], "execution_backend": "python"}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
FLAT_REFERENCE = {"currency": "EUR", "import_prices": {"all": 0.30}, "fixed_charge_per_day": 0.40}
TOU_REFERENCE = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.35, "off_peak": 0.09},
    "fixed_charge_per_day": 0.20,
}
LISBON = "Europe/Lisbon"
CUSTOM_SCHEDULE = {
    "identifier": "reference_custom_daily",
    "version": "1",
    "timezone": LISBON,
    "cycle": "daily",
    "periods": ["peak", "off_peak"],
    "rules": [
        {
            "days": "all",
            "season": "all",
            "intervals": {"off_peak": [["00:00", "08:00"], ["22:00", "24:00"]], "peak": [["08:00", "22:00"]]},
        }
    ],
}
QUARTERS = {"q1": [1, 2, 3], "q2": [4, 5, 6], "q3": [7, 8, 9], "q4": [10, 11, 12]}
MONTH_REFERENCE = {
    "currency": "EUR",
    "custom_schedule": {**CUSTOM_SCHEDULE, "identifier": "reference_quarterly", "seasons": QUARTERS},
    "import_prices": {
        "q1": {"peak": 0.35, "off_peak": 0.09},
        "q2": {"peak": 0.30, "off_peak": 0.12},
        "q3": {"peak": 0.25, "off_peak": 0.08},
        "q4": {"peak": 0.40, "off_peak": 0.15},
    },
    "fixed_charge_per_day": 0.20,
}


def _offline():
    return (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    )


def _simulated(config):
    fetch, weather = _offline()
    with fetch, weather:
        app = App(config)
        app.simulate()
    return app


def _artifacts(config):
    fetch, weather = _offline()
    with fetch, weather:
        app = App(config)
        return run_app_simulation(app._resolved, app._runtime_dependencies())


def _revalued(app, changes):
    fetch, weather = _offline()
    with fetch, weather:
        return app.revalue(changes)


def _fields(result):
    return {
        key: value
        for key, value in flatten(result).items()
        if not key.startswith(EXCLUDED_PREFIXES) and not key.startswith("provenance.revaluation")
    }


def _assert_close(revalued, fresh, rel=1e-12):
    assert revalued.keys() == fresh.keys()
    for key, value in fresh.items():
        if isinstance(value, float) and math.isfinite(value):
            assert revalued[key] == pytest.approx(value, rel=rel, abs=1e-9), key
        else:
            assert revalued[key] == value, key


def _peak(frame):
    local = pd.DatetimeIndex(frame["Datetime"]).tz_convert(LISBON)
    return (local.hour >= 8) & (local.hour < 22)


@pytest.fixture(scope="module")
def flat_run():
    return _artifacts(BASE)


@pytest.fixture(scope="module")
def flat_app():
    return _simulated(BASE)


@pytest.fixture(scope="module")
def tou_app():
    return _simulated({**BASE, "tariff": TOU})


# --- Default -----------------------------------------------------------------


def test_without_a_reference_the_household_pays_the_system_prices(flat_app, tou_app):
    for app in (flat_app, tou_app):
        result = app.result()
        assert "reference_tariff" not in result["provenance"]
        assert result["provenance"]["resolved_config"]["reference_tariff"] is None
        assert result["no_system_fixed_charge_year1_prices"] == result["fixed_charge_year1_prices"]
        year1 = result["financial"][1]
        assert year1["no_system_cost_fixed_charge"] == year1["cost_fixed_charge"]
        assert year1["no_system_cost_import"] == result["no_system_import_cost_year1_prices"]
    # Today's arithmetic: the system's prices and fixed charge, summed and
    # then escalated at the import escalation.
    run = _artifacts(BASE)
    inflation = resolve_app_config(BASE).cfg["inflation_rate"]
    expected = (run.yearly_df["Baseline_Import_Cost"] + run.yearly_df["Fixed_Charge"]) * (1 + inflation) ** (
        run.yearly_df["Year"] - 1
    )
    assert run.cost_projection["Cost_No_Sys_Annual"].tolist() == expected.tolist()


def test_an_explicit_none_is_the_default(flat_app):
    assert _fields(_simulated({**BASE, "reference_tariff": None}).result()) == _fields(flat_app.result())


# --- Flat and scheduled references ---------------------------------------------


@pytest.mark.parametrize(
    ("reference", "previous_npv"),
    [(FLAT_REFERENCE, -9725.03), (TOU_REFERENCE, -9945.27), (MONTH_REFERENCE, -10042.23)],
    ids=["flat", "bundled", "month"],
)
def test_app_accepts_zero_reference_fixed_charge_with_previous_results(reference, previous_npv):
    app = _simulated({**BASE, "reference_tariff": {**reference, "fixed_charge_per_day": 0}})
    result = app.result()
    # Pin the results from when an omitted reference fixed charge defaulted to zero.
    assert result["npv_savings"] == previous_npv
    assert result["no_system_fixed_charge_year1_prices"] == 0
    assert result["fixed_charge_year1_prices"] > 0
    assert result["provenance"]["reference_tariff"]["fixed_charge_per_day"] == 0
    assert (app._artifacts.cost_projection["Cost_No_Sys_Fixed_Charge"] == 0).all()


def test_a_flat_reference_prices_the_whole_load_and_its_own_fixed_charge(flat_run):
    run = _artifacts({**BASE, "reference_tariff": FLAT_REFERENCE})
    frame = run.first_year_results_df
    year1 = run.yearly_df.iloc[0]
    assert year1["Baseline_Import_Cost"] == pytest.approx(frame["Houseload"].sum() / 1000 * 0.30, rel=1e-12)
    assert year1["Baseline_Fixed_Charge"] == pytest.approx(365 * 0.40)
    # The system's own money and every energy flow are untouched.
    for column in ("Import_Cost", "Export_Revenue", "Fixed_Charge", "Import_kWh", "Export_kWh", "Load_kWh"):
        pd.testing.assert_series_equal(run.yearly_df[column], flat_run.yearly_df[column])
    pd.testing.assert_series_equal(
        run.cost_projection["Cost_System_Cumulative_NPV"], flat_run.cost_projection["Cost_System_Cumulative_NPV"]
    )
    assert run.reference_tariff["schedule"] == "flat"
    assert run.reference_tariff["import_prices"] == {"all": 0.30}
    assert "export_prices" not in run.reference_tariff


def test_a_scheduled_reference_reconciles_with_the_step_ledger(flat_run):
    run = _artifacts({**BASE, "reference_tariff": TOU_REFERENCE})
    frame = run.first_year_results_df
    price = np.where(_peak(frame), 0.35, 0.09)
    year1 = run.yearly_df.iloc[0]
    assert year1["Baseline_Import_Cost"] == pytest.approx(float((frame["Houseload"] * price).sum() / 1000), rel=1e-12)
    assert year1["Baseline_Fixed_Charge"] == pytest.approx(365 * 0.20)
    pd.testing.assert_series_equal(run.yearly_df["Import_kWh"], flat_run.yearly_df["Import_kWh"])
    assert run.reference_tariff["calendar_policy"] == "replay_start_year"
    assert run.reference_tariff["schedule"] == "pt_mainland_2026_daily_bi"


def test_a_custom_schedule_reference_prices_as_the_bundled_one():
    bundled = _artifacts({**BASE, "reference_tariff": TOU_REFERENCE})
    custom = {k: v for k, v in TOU_REFERENCE.items() if k != "schedule"}
    run = _artifacts({**BASE, "reference_tariff": {**custom, "custom_schedule": CUSTOM_SCHEDULE}})
    # Same periods, so the same per-step prices and the same floats.
    pd.testing.assert_series_equal(run.yearly_df["Baseline_Import_Cost"], bundled.yearly_df["Baseline_Import_Cost"])
    assert run.reference_tariff["schedule"] == "reference_custom_daily"


def test_a_reference_beside_a_system_tariff_prices_only_the_baseline(tou_app):
    run = _artifacts({**BASE, "tariff": TOU, "reference_tariff": FLAT_REFERENCE})
    plain = _artifacts({**BASE, "tariff": TOU})
    for column in ("Import_Cost", "Export_Revenue", "Fixed_Charge", "Import_kWh"):
        pd.testing.assert_series_equal(run.yearly_df[column], plain.yearly_df[column])
    assert run.yearly_df["Baseline_Fixed_Charge"].iloc[0] == pytest.approx(365 * 0.40)
    assert run.yearly_df["Baseline_Import_Cost"].iloc[0] == pytest.approx(
        run.first_year_results_df["Houseload"].sum() / 1000 * 0.30, rel=1e-12
    )
    assert run.tariff["schedule"] == "pt_mainland_2026_daily_bi"


def test_the_app_result_reconciles_with_the_ledger():
    app = _simulated({**BASE, "reference_tariff": TOU_REFERENCE})
    run = app._artifacts
    result = app.result()
    projection = run.cost_projection
    np.testing.assert_allclose(
        projection["Cost_No_Sys_Annual"],
        projection["Cost_No_Sys_Import"] + projection["Cost_No_Sys_Fixed_Charge"],
        rtol=1e-14,
    )
    year1 = run.yearly_df.iloc[0]
    assert result["no_system_import_cost_year1_prices"] == round(year1["Baseline_Import_Cost"], 2)
    assert result["no_system_fixed_charge_year1_prices"] == round(365 * 0.20, 2)
    for row, (_, projected) in zip(result["financial"][1:], projection.iterrows(), strict=True):
        assert row["no_system_cost_import"] == round(projected["Cost_No_Sys_Import"], 2)
        assert row["no_system_cost_fixed_charge"] == round(projected["Cost_No_Sys_Fixed_Charge"], 2)
        assert row["cost_without_system"] == round(projected["Cost_No_Sys_Cumulative_NPV"], 2)
    record = result["provenance"]["reference_tariff"]
    assert record["import_prices"] == TOU_REFERENCE["import_prices"]
    assert record["fixed_charge_per_day"] == 0.20
    assert record["calendar_year"] == 2025


# --- Escalation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "reference_escalation", "expected"),
    [
        ({}, None, "inflation"),
        ({"import_price_escalation": 0.04}, None, 0.04),
        ({"import_price_escalation": 0.04}, 0.0, 0.0),
        ({}, 0.07, 0.07),
    ],
    ids=["default-inflation", "default-system-escalation", "explicit-zero", "explicit"],
)
def test_the_reference_escalates_energy_and_fixed_charge(extra, reference_escalation, expected):
    reference = dict(FLAT_REFERENCE)
    if reference_escalation is not None:
        reference["import_price_escalation"] = reference_escalation
    app = _simulated({**BASE, **extra, "reference_tariff": reference})
    run = app._artifacts
    rate = resolve_app_config(BASE).cfg["inflation_rate"] if expected == "inflation" else expected
    years = run.yearly_df["Year"].to_numpy()
    factors = (1 + rate) ** (years - 1)
    np.testing.assert_allclose(
        run.cost_projection["Cost_No_Sys_Import"], run.yearly_df["Baseline_Import_Cost"] * factors, rtol=1e-14
    )
    np.testing.assert_allclose(
        run.cost_projection["Cost_No_Sys_Fixed_Charge"], run.yearly_df["Baseline_Fixed_Charge"] * factors, rtol=1e-14
    )
    assert app.result()["provenance"]["reference_tariff"]["import_price_escalation"] == rate
    if expected == 0.0:
        assert run.cost_projection["Cost_No_Sys_Import"].nunique() == 1
    # The system's import escalation is not the reference's.
    system_rate = extra.get("import_price_escalation", resolve_app_config(BASE).cfg["inflation_rate"])
    np.testing.assert_allclose(
        run.cost_projection["Cost_Daily"], run.yearly_df["Fixed_Charge"] * (1 + system_rate) ** (years - 1)
    )


# --- [period] ------------------------------------------------------------------


def test_a_period_bills_the_reference_fixed_charge_on_its_civil_days():
    config = {key: value for key, value in BASE.items() if key != "projection_years"}
    # 31 March days include the spring-forward day of 23 hours.
    config = {**config, "period": {"start": "2025-03-01", "end": "2025-04-01"}, "reference_tariff": FLAT_REFERENCE}
    app = _simulated(config)
    assert app._artifacts.yearly_df["Billed_Days"].iloc[0] == 31
    assert app.result()["no_system_fixed_charge_year1_prices"] == round(31 * 0.40, 2)
    revalued = _revalued(app, {"reference_tariff": {"fixed_charge_per_day": 0.5}})
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert revalued["no_system_fixed_charge_year1_prices"] == 15.5


# --- Validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "reference", [FLAT_REFERENCE, TOU_REFERENCE, MONTH_REFERENCE], ids=["flat", "bundled", "month"]
)
@pytest.mark.parametrize("entry_point", ["app", "app_config", "reference_spec", "montecarlo"])
def test_reference_fixed_charge_is_required(reference, entry_point):
    reference = {key: value for key, value in reference.items() if key != "fixed_charge_per_day"}
    config = {**BASE, "reference_tariff": reference}
    with pytest.raises(ValueError, match=r"reference_tariff\.fixed_charge_per_day"):
        if entry_point == "app":
            App(config)
        elif entry_point == "app_config":
            resolve_app_config(config)
        elif entry_point == "reference_spec":
            resolve_reference_tariff_spec(config, LISBON, None)
        else:
            run_montecarlo(config, MonteCarloSettings(weather_file="unused.csv", n_runs=1))


def test_revalue_requires_fixed_charge_when_adding_a_reference(flat_app):
    reference = {key: value for key, value in FLAT_REFERENCE.items() if key != "fixed_charge_per_day"}
    with pytest.raises(ValueError, match=r"reference_tariff\.fixed_charge_per_day"):
        flat_app.revalue({"reference_tariff": reference})


@pytest.mark.parametrize(
    ("reference", "extra", "message"),
    [
        ({**FLAT_REFERENCE, "currency": "USD"}, {}, r"'reference_tariff\.currency' must be one of: EUR"),
        ({k: v for k, v in FLAT_REFERENCE.items() if k != "currency"}, {}, r"reference_tariff\.currency"),
        (
            {**FLAT_REFERENCE, "fixed_charge_per_day": -0.1},
            {},
            r"'reference_tariff\.fixed_charge_per_day' must be >= 0",
        ),
        ({**FLAT_REFERENCE, "import_prices": {"peak": 0.3}}, {}, r"one flat price.*all = <price>"),
        ({**FLAT_REFERENCE, "study_date": "2025-06-01"}, {}, r"reference_tariff\.study_date applies to a schedule"),
        ({**FLAT_REFERENCE, "boundary_policy": "strict"}, {}, r"boundary_policy applies to a schedule"),
        ({**FLAT_REFERENCE, "export_prices": {"all": 0.05}}, {}, "export_prices"),
        (
            {**FLAT_REFERENCE, "import_prices": {"all": -0.1}},
            {},
            r"'reference_tariff\.import_prices\.all' must be >= 0",
        ),
        ({**FLAT_REFERENCE, "import_price_escalation": -1}, {}, r"reference_tariff\.import_price_escalation"),
        ({**TOU_REFERENCE, "import_prices": {"peak": 0.3}}, {}, "has no price for off_peak"),
        ({**TOU_REFERENCE, "import_prices": {"all": 0.2, "night": 0.1}}, {}, "night"),
        (
            {**TOU_REFERENCE, "custom_schedule": CUSTOM_SCHEDULE},
            {},
            "sets both 'schedule' and 'custom_schedule'",
        ),
        (
            {**TOU_REFERENCE, "schedule": "pt_mainland_2026_daily_tri", "import_prices": {"all": 0.2}},
            {},
            r"needs steps that divide 30 minutes, which 'h' steps do not",
        ),
        (
            TOU_REFERENCE,
            {"location": {"latitude": 52.5, "longitude": 13.4, "timezone": "Europe/Berlin"}},
            "does not move a schedule",
        ),
    ],
)
def test_reference_config_is_checked_at_construction(reference, extra, message):
    with pytest.raises(ValueError, match=message):
        App({**BASE, **extra, "reference_tariff": reference})


@pytest.mark.parametrize("tariff", [None, SimpleNamespace(prices=SimpleNamespace(currency="GBP"))])
def test_the_reference_must_be_in_the_result_currency(tariff):
    cfg = {"reference_tariff": {**FLAT_REFERENCE, "currency": "EUR"}, "resolution": "h"}
    if tariff is None:
        assert resolve_reference_tariff_spec(cfg, LISBON, None).prices.currency == "EUR"
        return
    with pytest.raises(ValueError, match=r"'reference_tariff\.currency' is EUR, but the result is in GBP"):
        resolve_reference_tariff_spec(cfg, LISBON, tariff)


# --- App.revalue -----------------------------------------------------------------


@pytest.mark.parametrize("system", ["flat", "tou"])
@pytest.mark.parametrize("step", ["add", "change", "remove"])
def test_revalue_reprices_a_reference_to_the_values_of_a_fresh_run(system, step, flat_app, tou_app):
    base = BASE if system == "flat" else {**BASE, "tariff": TOU}
    if step == "add":
        start = flat_app if system == "flat" else tou_app
        changes = {"reference_tariff": TOU_REFERENCE}
    elif step == "change":
        start = _simulated({**base, "reference_tariff": TOU_REFERENCE})
        changes = {"reference_tariff": {"import_prices": {"all": 0.25}, "import_price_escalation": 0.01}}
    else:
        start = _simulated({**base, "reference_tariff": FLAT_REFERENCE})
        changes = {"reference_tariff": None}
    revalued = _revalued(start, changes)
    assert revalued["provenance"]["revaluation"] == {"method": "repriced", "changed_keys": ["reference_tariff"]}
    config = _revalued_config(start._config, changes)
    fresh = _simulated(config).result()
    # Removing the reference beside a tariff re-prices the system's baseline
    # from the energy by period, which agrees to rounding.
    _assert_close(_fields(revalued), _fields(fresh))
    if step == "change":
        # The price list is replaced whole: "all" does not sit beside peak.
        assert revalued["provenance"]["reference_tariff"]["import_prices"] == {"all": 0.25}
    assert ("reference_tariff" in revalued["provenance"]) == (step != "remove")


def test_revalue_of_a_flat_reference_on_flat_prices_gives_the_floats_of_a_fresh_run(flat_app):
    revalued = _revalued(flat_app, {"reference_tariff": FLAT_REFERENCE})
    assert _fields(revalued) == _fields(_simulated({**BASE, "reference_tariff": FLAT_REFERENCE}).result())


def test_revalue_reprices_a_reference_under_daily_persistence():
    window = {key: value for key, value in BASE.items() if key != "projection_years"}
    config = {
        **window,
        "battery_kwh": 5.0,
        "tariff": TOU,
        "period": {"start": "2025-01-10", "end": "2025-01-14"},
        "smart_charging": {
            "mode": "daily_persistence",
            "charge_periods": ["off_peak"],
            "discharge_periods": ["peak"],
            "grid_charge_efficiency": 0.95,
            "target_levels": 3,
            "soc_states": 3,
        },
    }
    app = _simulated(config)
    for changes in (
        {"reference_tariff": TOU_REFERENCE},
        {"reference_tariff": FLAT_REFERENCE},
        {"reference_tariff": MONTH_REFERENCE},
    ):
        revalued = _revalued(app, changes)
        assert revalued["provenance"]["revaluation"]["method"] == "repriced"
        assert revalued["provenance"]["smart_charging"] == app.result()["provenance"]["smart_charging"]
        _assert_close(_fields(revalued), _fields(_simulated(_revalued_config(config, changes)).result()))
    with_reference = _simulated({**config, "reference_tariff": TOU_REFERENCE})
    removed = _revalued(with_reference, {"reference_tariff": None})
    assert removed["provenance"]["revaluation"]["method"] == "repriced"
    _assert_close(_fields(removed), _fields(app.result()))


# --- Monte Carlo -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "previous_npv"),
    [
        (FLAT_REFERENCE, [-5234.194955228494, -5201.517082178313]),
        (TOU_REFERENCE, [-5379.259727246049, -5351.478510011397]),
        (MONTH_REFERENCE, [-5443.119039206365, -5417.493390597623]),
    ],
    ids=["flat", "bundled", "month"],
)
def test_montecarlo_accepts_zero_reference_fixed_charge_with_previous_results(
    reference, previous_npv, tmp_path, write_multiyear_weather
):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3, collect_yearly=True)
    config = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0}
    result = run_montecarlo({**config, "reference_tariff": {**reference, "fixed_charge_per_day": 0}}, settings)
    np.testing.assert_allclose(result.runs["npv_savings"], previous_npv, rtol=1e-12)
    assert (result.yearly["Cost_No_Sys_Fixed_Charge"] == 0).all()
    assert result.provenance["reference_tariff"]["fixed_charge_per_day"] == 0


def test_montecarlo_prices_each_trajectory_load_at_the_reference(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3, collect_yearly=True)
    config = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5.0}
    reference = {**FLAT_REFERENCE, "import_price_escalation": 0.0}

    plain = run_montecarlo(config, settings)
    priced = run_montecarlo({**config, "reference_tariff": reference}, settings)

    assert "reference_tariff" not in plain.provenance
    record = priced.provenance["reference_tariff"]
    assert record["import_prices"] == {"all": 0.30}
    assert record["import_price_escalation"] == 0.0
    assert record["calendar_year"] == settings.target_year
    yearly, base = priced.yearly, plain.yearly
    # The same sampled weather and load, and the same costs with the system.
    for column in ("Load_kWh", "Load_Scale", "Weather_Year", "Import_kWh", "Cost_System_Cumulative_NPV"):
        pd.testing.assert_series_equal(yearly[column], base[column])
    # Each trajectory year's own load at the reference, not escalated.
    np.testing.assert_allclose(yearly["Cost_No_Sys_Import"], yearly["Load_kWh"] * 0.30, rtol=1e-12)
    assert yearly["Load_kWh"].nunique() > 1
    np.testing.assert_allclose(yearly["Cost_No_Sys_Fixed_Charge"], 365 * 0.40)
    # Paired: the change in each run's saving is its change in no-system cost.
    last = yearly.groupby("run").tail(1).set_index("run")
    last_base = base.groupby("run").tail(1).set_index("run")
    saving_change = priced.runs.set_index("run")["npv_savings"] - plain.runs.set_index("run")["npv_savings"]
    np.testing.assert_allclose(
        saving_change, last["Cost_No_Sys_Cumulative_NPV"] - last_base["Cost_No_Sys_Cumulative_NPV"], rtol=1e-12
    )


# --- Projected optimization ----------------------------------------------------


@pytest.fixture
def optimizer_case(monkeypatch):
    index = pd.date_range("2026-01-05", periods=48, freq="h", tz=LISBON)
    weather = pd.DataFrame({"temp_air": 20.0}, index=index)
    load = pd.DataFrame({"Load": 1000.0}, index=index)
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 16), 1800.0, 0.0), index=index)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": LISBON, "altitude": 100},
        "simulation": {"resolution": "h", "years_projection": 2},
        "financials": {"inflation_rate": 0.02, "discount_rate": 0.05},
        "battery": {"temperature": 20.0},
        "mode": {"fixed_azimuth": 180.0},
        "constraints": {"budget": 100000, "max_area_m2": 100, "max_tilt_deg": 60},
        "tariff": {
            "schedule": "pt_mainland_2026_daily_bi",
            "currency": "EUR",
            "import_prices": {"peak": 0.50, "off_peak": 0.10},
            "export_prices": {"all": 0.03},
            "fixed_charge_per_day": 0.40,
        },
    }
    return weather, load, config


def _evaluate(weather, load, config):
    return optimization.evaluate_projected_design(
        weather, load, config, n_modules=4, battery_kwh=0.0, tilt=30.0, azimuth=180.0
    )


@pytest.mark.parametrize(
    "reference", [FLAT_REFERENCE, TOU_REFERENCE, MONTH_REFERENCE], ids=["flat", "bundled", "month"]
)
@pytest.mark.parametrize("entry_point", ["config", "evaluation"])
def test_optimizer_requires_reference_fixed_charge(reference, entry_point, optimizer_case, monkeypatch):
    weather, load, config = optimizer_case
    config["reference_tariff"] = {key: value for key, value in reference.items() if key != "fixed_charge_per_day"}
    monkeypatch.setattr(
        optimization, "calculate_pv_production_dc", lambda **kwargs: pytest.fail("PV ran before validation")
    )
    with pytest.raises(ValueError, match=r"reference_tariff\.fixed_charge_per_day"):
        if entry_point == "config":
            resolve_optimization_config(config)
        else:
            _evaluate(weather, load, config)


@pytest.mark.parametrize(
    ("reference", "previous_npv"),
    [
        (FLAT_REFERENCE, -2027.9893134920192),
        (TOU_REFERENCE, -2033.2464563491621),
        (MONTH_REFERENCE, -2033.2464563491621),
    ],
    ids=["flat", "bundled", "month"],
)
def test_optimizer_accepts_zero_reference_fixed_charge_with_previous_results(reference, previous_npv, optimizer_case):
    weather, load, config = optimizer_case
    config["reference_tariff"] = {**reference, "fixed_charge_per_day": 0}
    resolved = resolve_optimization_config(config)
    assert resolved["reference_tariff"]["fixed_charge_per_day"] == 0
    result = _evaluate(weather, load, resolved)
    assert result.metrics["Projected_NPV"] == pytest.approx(previous_npv, rel=1e-12)
    assert (result.financial["Cost_No_Sys_Fixed_Charge"] == 0).all()
    assert result.provenance["reference_tariff"]["fixed_charge_per_day"] == 0


def test_the_optimizer_npv_is_the_saving_against_the_reference(optimizer_case):
    weather, load, config = optimizer_case
    plain = _evaluate(weather, load, config)
    with_reference = deepcopy(config)
    with_reference["reference_tariff"] = {**FLAT_REFERENCE, "import_price_escalation": 0.0}
    priced = _evaluate(weather, load, with_reference)

    # 48 kWh of load at 0.30, and two days of the reference fixed charge.
    np.testing.assert_allclose(priced.yearly["Baseline_Import_Cost"], 48 * 0.30)
    np.testing.assert_allclose(priced.yearly["Baseline_Fixed_Charge"], 2 * 0.40)
    np.testing.assert_allclose(priced.financial["Cost_No_Sys_Annual"], 48 * 0.30 + 2 * 0.40)
    pd.testing.assert_series_equal(
        priced.financial["Cost_System_Cumulative_NPV"], plain.financial["Cost_System_Cumulative_NPV"]
    )
    assert priced.metrics["Projected_NPV"] == pytest.approx(
        plain.metrics["Projected_NPV"]
        + priced.financial["Cost_No_Sys_Cumulative_NPV"].iloc[-1]
        - plain.financial["Cost_No_Sys_Cumulative_NPV"].iloc[-1]
    )
    assert priced.metrics["Projected_NPV"] != plain.metrics["Projected_NPV"]
    assert priced.provenance["reference_tariff"]["import_price_escalation"] == 0.0
    assert "reference_tariff" not in plain.provenance


def test_the_optimizer_reference_escalates_at_the_financials_import_escalation(optimizer_case):
    weather, load, config = optimizer_case
    config["financials"]["import_price_escalation"] = 0.06
    config["reference_tariff"] = deepcopy(TOU_REFERENCE)
    result = _evaluate(weather, load, config)
    assert result.provenance["reference_tariff"]["import_price_escalation"] == 0.06
    np.testing.assert_allclose(
        result.financial["Cost_No_Sys_Import"].to_numpy(),
        result.yearly["Baseline_Import_Cost"].to_numpy() * 1.06 ** np.arange(2),
        rtol=1e-14,
    )


def test_the_optimizer_search_scores_against_the_reference(optimizer_case):
    pytest.importorskip("pymoo")
    import pickle

    weather, load, config = optimizer_case
    config["reference_tariff"] = deepcopy(TOU_REFERENCE)
    fixed = _evaluate(weather, load, config)
    problem = pickle.loads(pickle.dumps(optimization.SolarDesignProblem(weather, load, config)))
    out = {}
    problem._evaluate(np.array([4, 0.0, 30.0]), out)
    assert out["F"][1] == -fixed.metrics["Projected_NPV"]
    assert problem.pricing.provenance()["reference_tariff"] == fixed.provenance["reference_tariff"]


def test_the_optimizer_checks_the_reference_before_pv(optimizer_case, monkeypatch):
    weather, load, config = optimizer_case
    config["reference_tariff"] = {**FLAT_REFERENCE, "import_prices": {"peak": 0.3}}
    monkeypatch.setattr(
        optimization, "calculate_pv_production_dc", lambda **kwargs: pytest.fail("PV ran before validation")
    )
    with pytest.raises(ValueError, match="one flat price"):
        _evaluate(weather, load, config)


# --- Sweeps ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        ["run", "--config"],
        ["run", "--dry-run", "--config"],
        ["validate-config"],
        ["sweep", "--config"],
        ["montecarlo", "--config"],
    ],
)
def test_cli_requires_reference_fixed_charge(command, tmp_path, capsys):
    path = tmp_path / "reference.toml"
    config = (
        'location = "porto"\nn_modules = 8\nannual_consumption_kwh = 4000\n'
        '[reference_tariff]\ncurrency = "EUR"\nimport_prices = { all = 0.30 }\n'
    )
    if command[0] == "sweep":
        config += "[sweep]\nbattery_kwh = [0]\n"
    elif command[0] == "montecarlo":
        config += '[montecarlo]\nweather_file = "unused.csv"\n'
    path.write_text(config, encoding="utf-8")
    assert cli.main([*command, str(path)]) == 1
    assert "reference_tariff.fixed_charge_per_day" in capsys.readouterr().err


@pytest.mark.parametrize("command", [["run", "--dry-run", "--config"], ["validate-config"]])
def test_cli_accepts_zero_reference_fixed_charge(command, tmp_path, capsys):
    path = tmp_path / "reference.toml"
    path.write_text(
        'location = "porto"\nn_modules = 8\nannual_consumption_kwh = 4000\n'
        '[reference_tariff]\ncurrency = "EUR"\nimport_prices = { all = 0.30 }\nfixed_charge_per_day = 0\n',
        encoding="utf-8",
    )
    assert cli.main([*command, str(path)]) == 0
    assert not capsys.readouterr().err


def test_sweeps_vary_reference_prices_by_dotted_key():
    for key in ("reference_tariff.import_prices.all", "reference_tariff.fixed_charge_per_day", "reference_tariff"):
        cli._check_sweep_key(key)
    with pytest.raises(ValueError, match="Unknown sweep key 'reference_tariff.export_prices'"):
        cli._check_sweep_key("reference_tariff.export_prices")
    runs = cli._sweep_run_configs(
        {**BASE, "reference_tariff": FLAT_REFERENCE}, {"reference_tariff.import_prices.all": [0.2, 0.3]}
    )
    prices = [resolve_app_config(config).reference_tariff.prices.import_prices["all"] for _, config in runs]
    assert prices == [0.2, 0.3]
    assert [config["reference_tariff"]["fixed_charge_per_day"] for _, config in runs] == [0.40, 0.40]


# --- Integration with month seasons and discharge_only -----------------------------


@pytest.fixture(scope="module")
def month_reference_app():
    return _simulated({**BASE, "reference_tariff": MONTH_REFERENCE})


def test_month_reference_reconciles_with_the_step_ledger_and_records_the_partition(month_reference_app):
    app = month_reference_app
    run, result = app._artifacts, app.result()
    ledger = run.first_year_results_df
    local = pd.DatetimeIndex(ledger["Datetime"]).tz_convert(LISBON)
    seasons = [f"q{(month - 1) // 3 + 1}" for month in local.month]
    periods = np.where(_peak(ledger), "peak", "off_peak")
    prices = np.array([MONTH_REFERENCE["import_prices"][s][p] for s, p in zip(seasons, periods, strict=True)])
    expected = (ledger["Houseload"].to_numpy() * prices).sum() / 1000
    np.testing.assert_allclose(run.yearly_df["Baseline_Import_Cost"], expected, rtol=1e-14)
    assert result["no_system_import_cost_year1_prices"] == round(expected, 2)
    assert result["no_system_fixed_charge_year1_prices"] == 365 * 0.20
    for row, (_, projected) in zip(result["financial"][1:], run.cost_projection.iterrows(), strict=True):
        assert row["no_system_cost_import"] == round(projected["Cost_No_Sys_Import"], 2)
        assert row["no_system_cost_fixed_charge"] == round(projected["Cost_No_Sys_Fixed_Charge"], 2)
        assert row["cost_without_system"] == round(projected["Cost_No_Sys_Cumulative_NPV"], 2)
    reference = run.resolved_reference_tariff
    assert reference.season_labels == tuple(seasons)
    record = result["provenance"]["reference_tariff"]
    assert record["seasons"] == QUARTERS
    assert record["import_prices"] == MONTH_REFERENCE["import_prices"]
    assert record["schedule_hash"] == reference.schedule_hash
    assert record["price_hash"] == reference.price_hash
    assert result["provenance"]["resolved_config"]["reference_tariff"]["custom_schedule"]["seasons"] == QUARTERS


@pytest.mark.parametrize("system", ["flat", "month"])
@pytest.mark.parametrize("step", ["add", "change"])
def test_revalue_month_reference_is_bit_identical_to_a_fresh_run(system, step, flat_app, month_reference_app):
    base = deepcopy(BASE)
    if system == "month":
        base["tariff"] = {**MONTH_REFERENCE, "export_prices": {"all": 0.05}}
    if step == "add":
        app = flat_app if system == "flat" else _simulated(base)
        changes = {"reference_tariff": MONTH_REFERENCE}
    else:
        app = month_reference_app if system == "flat" else _simulated({**base, "reference_tariff": MONTH_REFERENCE})
        # Replace the nested map whole, including each season's old periods.
        changes = {"reference_tariff": {"import_prices": {s: {"all": 0.2 + i * 0.05} for i, s in enumerate(QUARTERS)}}}
    with mock.patch("breos.runners.app.run_app_simulation", side_effect=AssertionError("reference drove dispatch")):
        revalued = _revalued(app, changes)
    assert revalued["provenance"]["revaluation"] == {"method": "repriced", "changed_keys": ["reference_tariff"]}
    fresh = _simulated(_revalued_config(app._config, changes))
    assert _fields(revalued) == _fields(fresh.result())
    record = revalued["provenance"]["reference_tariff"]
    assert record["import_prices"] == changes["reference_tariff"]["import_prices"]
    assert record["seasons"] == QUARTERS
    if step == "change":
        old = app.result()["provenance"]["reference_tariff"]
        assert record["schedule_hash"] == old["schedule_hash"]
        assert record["price_hash"] != old["price_hash"]


@pytest.mark.parametrize("kind", ["bundled", "dst", "flat", "flat_all"])
def test_seasonal_reference_prices_need_a_month_schedule(kind):
    reference = deepcopy(MONTH_REFERENCE)
    reference.pop("custom_schedule")
    if kind == "bundled":
        reference["schedule"] = TOU_REFERENCE["schedule"]
    elif kind == "dst":
        reference["custom_schedule"] = deepcopy(CUSTOM_SCHEDULE)
        rule = reference["custom_schedule"]["rules"][0]
        reference["custom_schedule"]["rules"] = [{**rule, "season": s} for s in ("standard", "dst")]
    elif kind == "flat_all":
        reference["import_prices"] = {"all": {"peak": 0.3}}
    message = "has no month seasons" if kind in ("bundled", "dst") else "one flat price"
    with pytest.raises(ValueError, match=message):
        App({**BASE, "reference_tariff": reference})


def test_reference_season_cannot_price_a_period_it_never_uses():
    reference = deepcopy(MONTH_REFERENCE)
    rule = reference["custom_schedule"]["rules"][0]
    reference["custom_schedule"]["rules"] = [
        {**rule, "season": "q1"},
        {**rule, "season": "q2"},
        {**rule, "season": "q3", "intervals": {"off_peak": [["00:00", "24:00"]]}},
        {**rule, "season": "q4"},
    ]
    with pytest.raises(ValueError, match=r"reference_tariff.import_prices.q3.*peak.*never uses"):
        App({**BASE, "reference_tariff": reference})


def test_sweep_accepts_a_reference_season_and_period():
    key = "reference_tariff.import_prices.q1.peak"
    cli._check_sweep_key(key)
    runs = cli._sweep_run_configs({**BASE, "reference_tariff": MONTH_REFERENCE}, {key: [0.2, 0.3]})
    for expected, (varied, config) in zip((0.2, 0.3), runs, strict=True):
        spec = resolve_app_config(config).reference_tariff
        assert spec.prices.import_prices["q1"]["peak"] == varied[key] == expected
        assert config["reference_tariff"]["import_prices"]["q2"] == MONTH_REFERENCE["import_prices"]["q2"]
    assert MONTH_REFERENCE["import_prices"]["q1"]["peak"] == 0.35


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_month_reference_never_drives_discharge_only_dispatch(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    config = {
        **BASE,
        "execution_backend": backend,
        "tariff": TOU,
        "smart_charging": {"mode": "discharge_only", "discharge_periods": ["peak"]},
    }
    plain = _simulated(config)
    priced = _simulated({**config, "reference_tariff": MONTH_REFERENCE})
    pd.testing.assert_frame_equal(plain._artifacts.first_year_results_df, priced._artifacts.first_year_results_df)
    assert plain.result()["provenance"]["smart_charging"] == priced.result()["provenance"]["smart_charging"]
    assert priced._artifacts.first_year_results_df["Grid_AC_To_Battery"].sum() == 0
    assert plain.result()["no_system_import_cost_year1_prices"] != priced.result()["no_system_import_cost_year1_prices"]
    changes = {"reference_tariff": {"import_prices": {s: {"all": 0.1} for s in QUARTERS}}}
    revalued = _revalued(priced, changes)
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert _fields(revalued) == _fields(_simulated(_revalued_config(priced._config, changes)).result())


def test_montecarlo_month_reference_matches_equivalent_flat_prices(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3, collect_yearly=True)
    config = {
        "location": "porto",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "tariff": TOU,
        "smart_charging": {"mode": "discharge_only", "discharge_periods": ["peak"]},
    }
    flat = run_montecarlo({**config, "reference_tariff": FLAT_REFERENCE}, settings)
    reference = {**MONTH_REFERENCE, "import_prices": {s: {"all": 0.30} for s in QUARTERS}, "fixed_charge_per_day": 0.40}
    month = run_montecarlo({**config, "reference_tariff": reference}, settings)
    pd.testing.assert_frame_equal(month.yearly, flat.yearly, check_exact=True)
    pd.testing.assert_frame_equal(month.runs, flat.runs, check_exact=True)
    assert month.provenance["reference_tariff"]["seasons"] == QUARTERS
    assert month.provenance["reference_tariff"]["import_prices"] == reference["import_prices"]
    assert month.provenance["smart_charging"] == flat.provenance["smart_charging"]


def test_optimizer_month_reference_prices_both_sides_of_a_quarter_boundary(optimizer_case, monkeypatch):
    weather, load, config = optimizer_case
    index = pd.date_range("2026-03-31", periods=48, freq="h", tz=LISBON)
    weather.index = load.index = index
    pv = pd.Series(np.where((index.hour >= 10) & (index.hour < 16), 1800.0, 0.0), index=index)
    monkeypatch.setattr(optimization, "calculate_pv_production_dc", lambda **kwargs: pv.copy())
    plain = _evaluate(weather, load, config)
    priced = _evaluate(weather, load, {**config, "reference_tariff": MONTH_REFERENCE})
    # One kWh each hour, 14 peak and 10 off-peak hours on each side.
    expected = 14 * 0.35 + 10 * 0.09 + 14 * 0.30 + 10 * 0.12
    np.testing.assert_allclose(priced.yearly["Baseline_Import_Cost"], expected, rtol=1e-14)
    np.testing.assert_allclose(priced.yearly["Baseline_Fixed_Charge"], 2 * 0.20)
    pd.testing.assert_series_equal(priced.yearly["Import_kWh"], plain.yearly["Import_kWh"], check_exact=True)
    pd.testing.assert_series_equal(
        priced.financial["Cost_System_Cumulative_NPV"], plain.financial["Cost_System_Cumulative_NPV"], check_exact=True
    )
    assert priced.provenance["reference_tariff"]["seasons"] == QUARTERS
    assert priced.provenance["reference_tariff"]["import_prices"] == MONTH_REFERENCE["import_prices"]
    assert priced.metrics["Projected_NPV"] == pytest.approx(
        plain.metrics["Projected_NPV"]
        + priced.financial["Cost_No_Sys_Cumulative_NPV"].iloc[-1]
        - plain.financial["Cost_No_Sys_Cumulative_NPV"].iloc[-1]
    )
