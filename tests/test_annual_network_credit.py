"""The annual network credit, `annual_network_credit` (#375, ADR 0002 A15).

Some network tariffs reduce a household's network charges by an annual
amount capped at what it paid for the network that year. Each household gets
its own credit, capped at its own eligible network charges: the network part
of its grid import plus the network fixed amount. The network parts are parts
of the tariff's own gross prices, so they set the cap and are never added to
the bill.
"""

import math
from copy import deepcopy
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from breos import App, optimization
from breos.app import _revalued_config
from breos.battery import BatteryConfig
from breos.economics import cost_analysis_projection
from breos.montecarlo import MonteCarloSettings, _pv_cache_key, run_montecarlo
from breos.optimization_config import resolve_optimization_config
from breos.projection import ProjectionYear, project_years
from breos.tariffs import FLAT_SCHEDULE, AnnualNetworkCredit, TariffPrices, resolve_tariff
from tools.generate_app_golden import EXCLUDED_PREFIXES, SCENARIOS, _fake_fetch, flatten

BERLIN = "Europe/Berlin"
LISBON = "Europe/Lisbon"
BASE = {**SCENARIOS["native_h_replacement"], "execution_backend": "python"}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
CREDIT = {
    "amount_per_year": 146.58,
    "network_fixed_per_year": 39.70,
    "network_import_prices": {"peak": 0.0888, "off_peak": 0.0311},
}
FLAT_REFERENCE = {"currency": "EUR", "import_prices": {"all": 0.30}, "fixed_charge_per_day": 0.40}
FLAT_CREDIT = {**CREDIT, "network_import_prices": {"all": 0.0888}}

# Stromnetz Berlin's Modul 1: 146.58 EUR a year, capped at the network
# charges, with a 39.70 EUR network fixed amount and a 0.4405 EUR daily
# standing charge (the legacy study's values).
MODULE1 = AnnualNetworkCredit(
    amount_per_year=146.58, network_fixed_per_year=39.70, network_import_prices={"all": 0.0888}
)


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


def _flat_tariff(index, *, price=0.355096, fixed_per_day=0.4405, credit=MODULE1):
    prices = TariffPrices("EUR", {"all": price}, {"all": 0.0}, fixed_per_day, annual_network_credit=credit)
    return resolve_tariff(index, ("all",) * len(index), FLAT_SCHEDULE, prices, timezone=BERLIN)


def _pv_equals_load(index, *, tariff, extra=None, reference=None):
    """No battery, PV and load at 1000 W on every step, through a lossless inverter: nothing is imported."""
    pv = pd.Series(1000.0, index=index)
    load = pd.DataFrame({"Load": 1000.0}, index=index)
    return project_years(
        1,
        lambda _year: ProjectionYear(1.0, pv_dc=pv, houseload=load, extra=extra or {}),
        battery_config=lambda _soh: BatteryConfig(nominal_energy_wh=0, inverter_efficiency=1.0),
        freq="h",
        has_battery=False,
        execution_backend="python",
        tariff=tariff,
        reference_tariff=reference,
    ).yearly_df


@pytest.fixture(scope="module")
def tou_app():
    return _simulated({**BASE, "tariff": TOU})


@pytest.fixture(scope="module")
def credit_app():
    return _simulated({**BASE, "tariff": {**TOU, "annual_network_credit": CREDIT}})


# --- The cap -------------------------------------------------------------------


def test_the_credit_is_capped_below_the_amount_when_imports_are_low():
    index = pd.date_range("2026-01-01", periods=8760, freq="h", tz=BERLIN)
    rows = _pv_equals_load(index, tariff=_flat_tariff(index))
    year = rows.iloc[0]
    assert year["Import_kWh"] == 0
    assert year["Fixed_Charge"] == pytest.approx(365 * 0.4405) == pytest.approx(160.7825)
    # No volumetric network charge: the cap is the network fixed amount.
    assert year["Network_Charge"] == pytest.approx(39.70, rel=1e-15)
    assert year["Network_Credit"] == pytest.approx(39.70, rel=1e-15)
    projection = cost_analysis_projection(
        rows,
        {"total_initial_cost": 0.0, "annual_operation_cost": 0.0, "electricity_cost": 0.0},
        num_years=1,
        inflation_rate=0.0,
    )
    flows = projection.iloc[0]
    assert flows["Cost_Daily"] - flows["Cost_Network_Credit"] == pytest.approx(121.0825, rel=1e-14)
    assert flows["Cost_System_Annual"] == pytest.approx(121.0825, rel=1e-14)


def test_system_and_no_system_credits_differ_on_the_same_day():
    # The legacy one-day case: 24 steps of an hour, PV = load = 1000 W, no
    # battery. A one-day [period] gets 1/365 of each annual amount in 2026.
    index = pd.date_range("2026-03-10", periods=24, freq="h", tz=BERLIN)
    year = _pv_equals_load(index, tariff=_flat_tariff(index), extra={"Billed_Days": 1.0}).iloc[0]
    assert year["Import_kWh"] == 0 and year["Load_kWh"] == 24
    # The system household imports nothing: its cap is the day's fixed amount.
    assert year["Network_Charge"] == pytest.approx(39.70 / 365, rel=1e-15)
    assert year["Network_Credit"] == pytest.approx(0.1087671, abs=5e-8)
    # The household without the system buys 24 kWh: the full day's amount.
    assert year["Baseline_Network_Charge"] == pytest.approx(24 * 0.0888 + 39.70 / 365, rel=1e-15)
    assert year["Baseline_Network_Charge"] == pytest.approx(2.2399671, abs=5e-8)
    assert year["Baseline_Network_Credit"] == pytest.approx(146.58 / 365, rel=1e-15)
    assert year["Baseline_Network_Credit"] == pytest.approx(0.4015890, abs=5e-8)


def test_a_leap_year_gets_the_annual_amount_and_a_window_its_share_of_the_year():
    index = pd.date_range("2028-01-01", periods=8784, freq="h", tz=BERLIN)
    whole = _pv_equals_load(index, tariff=_flat_tariff(index)).iloc[0]
    assert whole["Network_Credit"] == pytest.approx(39.70, rel=1e-15)
    day = pd.date_range("2028-03-10", periods=24, freq="h", tz=BERLIN)
    one_day = _pv_equals_load(day, tariff=_flat_tariff(day), extra={"Billed_Days": 1.0}).iloc[0]
    assert one_day["Network_Credit"] == pytest.approx(39.70 / 366, rel=1e-15)


# --- Gross prices, given once ----------------------------------------------------


def test_vat_is_applied_once_and_the_network_parts_are_never_added():
    # Full prices are built once from net components times 19% VAT; the
    # network part of each is given gross, as part of that full price.
    net = {"low": 0.2499, "standard": 0.2984, "high": 0.3632}
    gross = {period: round(price * 1.19, 6) for period, price in net.items()}
    assert gross == {"low": 0.297381, "standard": 0.355096, "high": 0.432208}
    network = {"low": 0.0311, "standard": 0.0888, "high": 0.1659}
    tariff = {
        "custom_schedule": {
            "identifier": "three_windows",
            "version": "1",
            "timezone": LISBON,
            "cycle": "daily",
            "periods": ["low", "standard", "high"],
            "rules": [
                {
                    "days": "all",
                    "season": "all",
                    "intervals": {
                        "low": [["00:00", "06:00"]],
                        "standard": [["06:00", "17:00"], ["21:00", "24:00"]],
                        "high": [["17:00", "21:00"]],
                    },
                }
            ],
        },
        "currency": "EUR",
        "import_prices": gross,
        "export_prices": {"all": 0.08},
        "fixed_charge_per_day": 0.4405,
    }
    credit = {"amount_per_year": 146.58, "network_fixed_per_year": 39.70, "network_import_prices": network}
    plain = _simulated({**BASE, "tariff": tariff})
    credited = _simulated({**BASE, "tariff": {**tariff, "annual_network_credit": credit}})

    plain_rows, rows = plain._artifacts.yearly_df, credited._artifacts.yearly_df
    # The bill is the same: the network parts are not added to it, and the
    # fixed charge does not gain the network fixed amount.
    for column in ("Import_Cost", "Export_Revenue", "Fixed_Charge", "Baseline_Import_Cost", "Baseline_Fixed_Charge"):
        pd.testing.assert_series_equal(rows[column], plain_rows[column])
    frame = credited._artifacts.first_year_results_df
    local_hour = pd.DatetimeIndex(frame["Datetime"]).tz_convert(LISBON).hour
    period = np.where(local_hour < 6, "low", np.where((local_hour >= 17) & (local_hour < 21), "high", "standard"))
    imported = frame["Import_From_Grid"].to_numpy() / 1000
    load = frame["Houseload"].to_numpy() / 1000
    # Energy is priced at the gross price once, not times 1.19 again.
    assert rows["Import_Cost"].iloc[0] == pytest.approx(sum(imported[period == p].sum() * gross[p] for p in gross))
    # The cap basis: the gross network part of each import, plus the fixed amount once.
    network_import = sum(imported[period == p].sum() * network[p] for p in network)
    assert rows["Network_Charge"].iloc[0] == pytest.approx(network_import + 39.70, rel=1e-12)
    network_load = sum(load[period == p].sum() * network[p] for p in network)
    assert rows["Baseline_Network_Charge"].iloc[0] == pytest.approx(network_load + 39.70, rel=1e-12)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"network_import_prices": {"peak": 0.30, "off_peak": 0.0311}}, "more than its import price"),
        ({"network_fixed_per_year": 0.25 * 365 + 1}, "more than 365 days of the fixed charge"),
        ({"network_import_prices": {"peak": 0.08}}, "no price for off_peak"),
        ({"network_import_prices": {"shoulder": 0.01, "all": 0.02}}, "shoulder"),
        ({"amount_per_year": -1.0}, "must be >= 0"),
    ],
)
def test_network_parts_are_checked_at_construction(change, message):
    with pytest.raises(ValueError, match=message):
        App({**BASE, "tariff": {**TOU, "annual_network_credit": {**CREDIT, **change}}})


@pytest.mark.parametrize("key", sorted(CREDIT))
def test_every_credit_key_is_required(key):
    credit = {name: value for name, value in CREDIT.items() if name != key}
    with pytest.raises(ValueError, match=rf"annual_network_credit\.{key}"):
        App({**BASE, "tariff": {**TOU, "annual_network_credit": credit}})


def test_a_flat_reference_has_one_flat_network_price():
    reference = {**FLAT_REFERENCE, "annual_network_credit": CREDIT}
    with pytest.raises(ValueError, match="network price is flat too"):
        App({**BASE, "reference_tariff": reference})


# --- Results without the table ---------------------------------------------------


def test_without_the_table_results_carry_no_credit(tou_app):
    result = tou_app.result()
    assert not any("network" in key for key in flatten(result))
    assert "Network_Credit" not in tou_app._artifacts.yearly_df


def test_a_zero_credit_changes_no_reported_value(tou_app):
    zero = {"amount_per_year": 0.0, "network_fixed_per_year": 0.0, "network_import_prices": {"all": 0.0}}
    credited = _simulated({**BASE, "tariff": {**TOU, "annual_network_credit": zero}}).result()
    plain = _fields(tou_app.result())
    fields = _fields(credited)
    for key, value in plain.items():
        if key.startswith(("provenance.tariff.price_hash", "resolved_config.tariff")):
            continue
        assert fields[key] == value, key
    added = sorted(set(fields) - set(plain))
    assert "network_credit_year1_prices" in added and "financial[1].no_system_network_credit" in added
    assert credited["network_credit_year1_prices"] == 0.0


# --- App ------------------------------------------------------------------------------


def test_the_app_books_each_credit_in_its_household_cashflow(tou_app, credit_app):
    plain, credited = tou_app._artifacts.cost_projection, credit_app._artifacts.cost_projection
    rows = credit_app._artifacts.yearly_df
    years = rows["Year"].to_numpy()
    escalation = credit_app._resolved.cfg["inflation_rate"]
    factors = (1 + escalation) ** (years - 1)
    # The system household imports less than its load, so its cap binds.
    assert (rows["Network_Credit"] < 146.58).all()
    assert (rows["Network_Credit"] == rows["Network_Charge"]).all()
    assert (rows["Baseline_Network_Credit"] == 146.58).all()
    np.testing.assert_allclose(credited["Cost_Network_Credit"], rows["Network_Credit"] * factors, rtol=1e-14)
    np.testing.assert_allclose(
        credited["Cost_System_Annual"], plain["Cost_System_Annual"] - credited["Cost_Network_Credit"], rtol=1e-14
    )
    np.testing.assert_allclose(
        credited["Cost_No_Sys_Annual"], plain["Cost_No_Sys_Annual"] - credited["Cost_No_Sys_Network_Credit"], rtol=1e-14
    )
    result, before = credit_app.result(), tou_app.result()
    discount = (1 + credit_app._resolved.cfg["discount_rate"]) ** years
    saving = float(np.sum((credited["Cost_Network_Credit"] - credited["Cost_No_Sys_Network_Credit"]) / discount))
    assert result["npv_savings"] == pytest.approx(before["npv_savings"] + saving, abs=0.01)
    # The credit is a tariff outcome, which LCOE leaves out.
    assert result["lcoe_per_kwh"] == before["lcoe_per_kwh"]
    first = result["financial"][1]
    assert first["network_credit"] == round(float(credited["Cost_Network_Credit"].iloc[0]), 2)
    assert first["no_system_network_credit"] == 146.58
    assert result["network_charge_year1_prices"] == round(float(rows["Network_Charge"].iloc[0]), 2)
    record = result["provenance"]["tariff"]["annual_network_credit"]
    assert record["amount_per_year"] == 146.58
    assert record["network_import_prices"] == CREDIT["network_import_prices"]
    assert result["provenance"]["tariff"]["price_hash"] != before["provenance"]["tariff"]["price_hash"]


def test_a_reference_gives_the_no_system_household_its_own_credit_or_none():
    tariff = {**TOU, "annual_network_credit": CREDIT}
    without = _simulated({**BASE, "tariff": tariff, "reference_tariff": FLAT_REFERENCE})
    rows = without._artifacts.yearly_df
    assert "Network_Credit" in rows and "Baseline_Network_Credit" not in rows
    assert "no_system_network_credit" not in without.result()["financial"][1]

    reference = {**FLAT_REFERENCE, "import_price_escalation": 0.0, "annual_network_credit": FLAT_CREDIT}
    own = _simulated({**BASE, "reference_tariff": reference})
    rows, flows = own._artifacts.yearly_df, own._artifacts.cost_projection
    np.testing.assert_allclose(rows["Baseline_Network_Charge"], rows["Load_kWh"] * 0.0888 + 39.70, rtol=1e-12)
    assert (rows["Baseline_Network_Credit"] == 146.58).all()
    # Escalated at the reference's own rate, here none.
    assert (flows["Cost_No_Sys_Network_Credit"] == 146.58).all()
    # Flat system prices have no credit.
    assert "Network_Credit" not in rows
    assert own.result()["provenance"]["reference_tariff"]["annual_network_credit"]["amount_per_year"] == 146.58


def test_a_period_gets_its_share_of_the_annual_amounts():
    config = {key: value for key, value in BASE.items() if key != "projection_years"}
    config = {
        **config,
        "period": {"start": "2025-03-01", "end": "2025-04-01"},
        "tariff": {**TOU, "annual_network_credit": CREDIT},
    }
    app = _simulated(config)
    row = app._artifacts.yearly_df.iloc[0]
    assert row["Baseline_Network_Credit"] == pytest.approx(146.58 * 31 / 365, rel=1e-15)
    assert row["Network_Charge"] - row["Network_Credit"] >= 0


# --- App.revalue ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"tariff": {"annual_network_credit": CREDIT}},
        {"tariff": {"annual_network_credit": {"amount_per_year": 50.0}}},
        {"tariff": {"annual_network_credit": None}},
        {"tariff": {"annual_network_credit": CREDIT, "import_prices": {"peak": 0.30, "off_peak": 0.12}}},
        {"reference_tariff": {**FLAT_REFERENCE, "annual_network_credit": FLAT_CREDIT}},
        {"reference_tariff": FLAT_REFERENCE},
    ],
    ids=["add", "change", "remove", "with_prices", "reference_credit", "reference_without_credit"],
)
def test_revalue_reprices_the_credit_to_the_values_of_a_fresh_run(changes, tou_app, credit_app):
    start = tou_app if changes == {"tariff": {"annual_network_credit": CREDIT}} else credit_app
    revalued = _revalued(start, changes)
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    fresh = _simulated(_revalued_config(start._config, changes)).result()
    # Re-priced from the energy by period, which agrees with the per-step sums to rounding.
    _assert_close(_fields(revalued), _fields(fresh))


def test_revalue_of_the_credit_alone_keeps_the_energy_money(credit_app):
    revalued = _revalued(credit_app, {"tariff": {"annual_network_credit": {"amount_per_year": 50.0}}})
    before = credit_app.result()
    for key in ("grid_import_cost_year1_prices", "fixed_charge_year1_prices", "no_system_import_cost_year1_prices"):
        assert revalued[key] == before[key]
    assert revalued["no_system_network_credit_year1_prices"] == 50.0


def test_revalue_never_resimulates_for_the_credit_under_daily_persistence():
    window = {key: value for key, value in BASE.items() if key != "projection_years"}
    config = {
        **window,
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
    changes = {"tariff": {"annual_network_credit": CREDIT}}
    revalued = _revalued(app, changes)
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert revalued["provenance"]["smart_charging"] == app.result()["provenance"]["smart_charging"]
    _assert_close(_fields(revalued), _fields(_simulated(_revalued_config(config, changes)).result()))


# --- Monte Carlo -----------------------------------------------------------------------


def test_montecarlo_caps_each_trajectory_at_its_own_import(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=3, collect_yearly=True)
    config = {
        "location": "porto",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "import_price_escalation": 0.0,
        "tariff": TOU,
    }
    credit = {**CREDIT, "amount_per_year": 1000.0, "network_import_prices": {"all": 0.05}}
    credited_config = {**config, "tariff": {**TOU, "annual_network_credit": credit}}
    # Pure pricing: the year cache is shared.
    assert _pv_cache_key(App(config)._resolved.cfg) == _pv_cache_key(App(credited_config)._resolved.cfg)

    plain = run_montecarlo(config, settings)
    credited = run_montecarlo(credited_config, settings)
    yearly, base = credited.yearly, plain.yearly
    for column in ("Load_kWh", "Import_kWh", "Cost_Import", "Cost_Daily"):
        pd.testing.assert_series_equal(yearly[column], base[column])
    # A cap above every year's charges: each year's credit is its own charges.
    np.testing.assert_allclose(yearly["Cost_Network_Credit"], yearly["Import_kWh"] * 0.05 + 39.70, rtol=1e-9)
    np.testing.assert_allclose(yearly["Cost_No_Sys_Network_Credit"], yearly["Load_kWh"] * 0.05 + 39.70, rtol=1e-9)
    assert yearly["Load_kWh"].nunique() > 1
    np.testing.assert_allclose(
        yearly["Cost_System_Annual"], base["Cost_System_Annual"] - yearly["Cost_Network_Credit"], rtol=1e-12
    )
    assert credited.provenance["tariff"]["annual_network_credit"]["amount_per_year"] == 1000.0
    assert "annual_network_credit" not in plain.provenance["tariff"]


# --- Projected optimization -------------------------------------------------------------


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


def test_the_optimizer_npv_includes_both_credits(optimizer_case):
    weather, load, config = optimizer_case
    plain = _evaluate(weather, load, config)
    credited_config = deepcopy(config)
    credited_config["tariff"]["annual_network_credit"] = {
        "amount_per_year": 146.58,
        "network_fixed_per_year": 39.70,
        "network_import_prices": {"all": 0.0888},
    }
    resolved = resolve_optimization_config(credited_config)
    assert resolved["tariff"]["annual_network_credit"]["amount_per_year"] == 146.58
    credited = _evaluate(weather, load, resolved)

    # Two days of the 48-step year (no [period]) are one simulated year each.
    rows, flows = credited.yearly, credited.financial
    np.testing.assert_allclose(rows["Baseline_Network_Charge"], 48 * 0.0888 + 39.70, rtol=1e-12)
    np.testing.assert_allclose(rows["Network_Charge"], rows["Import_kWh"] * 0.0888 + 39.70, rtol=1e-12)
    discount = 1.05 ** flows["Year"]
    saving = float(np.sum((flows["Cost_Network_Credit"] - flows["Cost_No_Sys_Network_Credit"]) / discount))
    assert credited.metrics["Projected_NPV"] == pytest.approx(plain.metrics["Projected_NPV"] + saving, rel=1e-12)
    assert credited.provenance["tariff"]["annual_network_credit"]["network_fixed_per_year"] == 39.70


def test_the_optimizer_search_scores_with_the_credit(optimizer_case):
    pytest.importorskip("pymoo")
    import pickle

    weather, load, config = optimizer_case
    config["tariff"]["annual_network_credit"] = {
        "amount_per_year": 146.58,
        "network_fixed_per_year": 39.70,
        "network_import_prices": {"peak": 0.0888, "off_peak": 0.0311},
    }
    fixed = _evaluate(weather, load, config)
    problem = pickle.loads(pickle.dumps(optimization.SolarDesignProblem(weather, load, config)))
    out = {}
    problem._evaluate(np.array([4, 0.0, 30.0]), out)
    assert out["F"][1] == -fixed.metrics["Projected_NPV"]
