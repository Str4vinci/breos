"""App.revalue re-prices a finished run, or re-simulates when prices change the dispatch (#183, #393)."""

import math
from unittest import mock

import pytest

from breos import App
from breos.app import REVALUATION_KEYS
from tests.test_tariff_month_seasons import SEASONAL_TARIFF
from tools.generate_app_golden import EXCLUDED_PREFIXES, SCENARIOS, _fake_fetch, flatten

BASE = {**SCENARIOS["native_h_replacement"], "execution_backend": "python"}
TOU = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.28, "off_peak": 0.11},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}
SMART = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.6,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
    "grid_import_limit_w": 5000,
}
FLAT_CHANGES = {
    "costs": {"storage_cost_per_kwh": 250.0, "electricity_cost": 0.31},
    "discount_rate": 0.05,
    "inflation_rate": 0.025,
    "replacement_cost_learning": 0.02,
}
TOU_PRICES = {"tariff": {"import_prices": {"peak": 0.33, "off_peak": 0.09}, "fixed_charge_per_day": 0.4}}
# Bundled Portuguese schedules at 15 minutes; the weekly ones have half-hour
# boundaries. The tri-hourly prices are illustrative.
TRI_PRICES = {"peak": 0.31, "mid_peak": 0.17, "off_peak": 0.10}
DAILY_TRI = {**TOU, "schedule": "pt_mainland_2026_daily_tri", "import_prices": TRI_PRICES}
WEEKLY_BI = {**TOU, "schedule": "pt_mainland_2026_weekly_bi", "import_prices": {"peak": 0.26, "off_peak": 0.12}}
WEEKLY_TRI = {**TOU, "schedule": "pt_mainland_2026_weekly_tri", "import_prices": TRI_PRICES}
REFERENCE = {"currency": "EUR", "import_prices": {"all": 0.30}, "fixed_charge_per_day": 0.40}
# Native and BLAST aging, each with a pack replacement inside the three years.
GREEDY_15MIN = {
    "native": {**SCENARIOS["native_15min_replacement"], "execution_backend": "python"},
    "blast": {**SCENARIOS["blast_15min_replacement"], "execution_backend": "python"},
}
# From the stored run's prices to the new ones: a first schedule, another
# schedule, a Module 3-style custom schedule with month seasons, and a
# reference tariff added, kept and removed.
SCHEDULE_CHANGES = {
    "flat_to_daily_bi": ({}, {"tariff": TOU}),
    "daily_bi_to_daily_tri": ({"tariff": TOU}, {"tariff": DAILY_TRI}),
    "daily_bi_to_weekly_bi_with_a_reference": ({"tariff": TOU}, {"tariff": WEEKLY_BI, "reference_tariff": REFERENCE}),
    "weekly_bi_to_weekly_tri_under_a_reference": (
        {"tariff": WEEKLY_BI, "reference_tariff": REFERENCE},
        {"tariff": WEEKLY_TRI},
    ),
    "weekly_tri_to_quarterly_without_the_reference": (
        {"tariff": WEEKLY_TRI, "reference_tariff": REFERENCE},
        # A table merges key by key, so the bundled schedule is removed.
        {"tariff": {**SEASONAL_TARIFF, "schedule": None}, "reference_tariff": None},
    ),
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


def _revalued(app, changes):
    fetch, weather = _offline()
    with fetch, weather:
        return app.revalue(changes)


def _fields(result):
    fields = flatten(result)
    return {
        key: value
        for key, value in fields.items()
        if not key.startswith(EXCLUDED_PREFIXES) and not key.startswith("provenance.revaluation")
    }


def _merge(base, changes):
    from breos.app import _revalued_config

    return _revalued_config(base, changes)


def _assert_close(revalued, fresh, rel):
    assert revalued.keys() == fresh.keys()
    for key, value in fresh.items():
        if isinstance(value, float) and math.isfinite(value):
            assert revalued[key] == pytest.approx(value, rel=rel, abs=1e-9), key
        else:
            assert revalued[key] == value, key


@pytest.fixture(scope="module")
def flat_app():
    return _simulated(BASE)


@pytest.fixture(scope="module")
def tou_app():
    return _simulated({**BASE, "tariff": TOU})


@pytest.fixture(scope="module")
def smart_app():
    return _simulated({**BASE, "tariff": TOU, "smart_charging": SMART})


def test_no_change_gives_the_run(flat_app):
    revalued = _revalued(flat_app, {})
    assert revalued["provenance"]["revaluation"] == {"method": "repriced", "changed_keys": []}
    assert _fields(revalued) == _fields(flat_app.result())


def test_flat_prices_are_repriced_to_the_floats_of_a_new_run(flat_app):
    revalued = _revalued(flat_app, FLAT_CHANGES)
    assert revalued["provenance"]["revaluation"] == {"method": "repriced", "changed_keys": sorted(FLAT_CHANGES)}
    fresh = _simulated(_merge(BASE, FLAT_CHANGES)).result()
    assert _fields(revalued) == _fields(fresh)
    assert revalued["npv_savings"] != flat_app.result()["npv_savings"]
    # The original run is left as it was.
    assert flat_app.result()["provenance"].get("revaluation") is None


def test_tou_prices_are_repriced_by_period(tou_app):
    revalued = _revalued(tou_app, TOU_PRICES)
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    fresh = _simulated(_merge({**BASE, "tariff": TOU}, TOU_PRICES)).result()
    _assert_close(_fields(revalued), _fields(fresh), rel=1e-12)
    assert revalued["grid_import_cost_year1_prices"] != tou_app.result()["grid_import_cost_year1_prices"]


def test_smart_charging_on_an_unchanged_schedule_is_repriced(smart_app):
    # Fixed-target instructions follow the periods, not the prices, so the
    # dispatch cannot change.
    revalued = _revalued(smart_app, TOU_PRICES)
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    fresh = _simulated(_merge({**BASE, "tariff": TOU, "smart_charging": SMART}, TOU_PRICES)).result()
    _assert_close(_fields(revalued), _fields(fresh), rel=1e-12)


@pytest.fixture(scope="module", params=sorted(GREEDY_15MIN))
def greedy_runs(request):
    """Stored greedy runs for one degradation model, simulated once per starting price set."""
    return GREEDY_15MIN[request.param], {}


@pytest.mark.parametrize("case", sorted(SCHEDULE_CHANGES))
def test_a_greedy_run_is_priced_on_a_new_schedule_from_its_step_flows(greedy_runs, case):
    # Greedy dispatch never reads the tariff, so the stored per-step import
    # and export of every year price on any schedule.
    base, apps = greedy_runs
    start, changes = SCHEDULE_CHANGES[case]
    config = _merge(base, start)
    key = repr(sorted(start))
    if key not in apps:
        apps[key] = _simulated(config)
    revalued = _revalued(apps[key], changes)
    assert revalued["provenance"]["revaluation"] == {"method": "repriced_by_step", "changed_keys": sorted(changes)}
    # The year loop's own sums over the same flows: the floats of a new run.
    fresh = _simulated(_merge(config, changes)).result()
    assert _fields(revalued) == _fields(fresh)


def test_a_pv_only_run_is_priced_on_a_new_schedule_from_its_step_flows():
    config = {**GREEDY_15MIN["native"], "battery_kwh": 0, "tariff": TOU}
    revalued = _revalued(_simulated(config), {"tariff": WEEKLY_TRI})
    assert revalued["provenance"]["revaluation"]["method"] == "repriced_by_step"
    assert _fields(revalued) == _fields(_simulated({**config, "tariff": WEEKLY_TRI}).result())


def test_a_greedy_run_keeps_step_flows_for_every_year_and_each_unchanging_flow_once():
    app = _simulated({**GREEDY_15MIN["native"], "tariff": TOU})
    flows = app._artifacts.projection.priced_flows
    assert len(flows) == 3
    steps = len(app.timeseries())
    assert {len(values) for year in flows for values in year.values()} == {steps}
    # The load and the zero grid charge are one array across the years.
    for column in ("Houseload", "Grid_AC_To_Battery"):
        assert flows[0][column] is flows[1][column] is flows[2][column]
    assert not flows[0]["Grid_AC_To_Battery"].any()
    assert flows[0]["Import_From_Grid"] is not flows[1]["Import_From_Grid"]


@pytest.mark.parametrize(
    "smart", [SMART, {"mode": "discharge_only", "discharge_periods": ["peak"]}], ids=["fixed_target", "discharge_only"]
)
def test_a_new_schedule_simulates_a_smart_charging_run_again(smart):
    # Instructions follow the tariff's periods, so a new calendar moves the
    # dispatch, and the run keeps no step flows to price it from.
    config = {**GREEDY_15MIN["native"], "projection_years": 1, "tariff": TOU, "smart_charging": smart}
    app = _simulated(config)
    assert app._artifacts.projection.priced_flows is None
    changes = {"tariff": {"schedule": "pt_mainland_2026_weekly_bi"}}
    revalued = _revalued(app, changes)
    assert revalued["provenance"]["revaluation"]["method"] == "resimulated"
    assert _fields(revalued) == _fields(_simulated(_merge(config, changes)).result())


def test_a_tariff_removed_is_repriced_at_flat_prices(tou_app, flat_app):
    revalued = _revalued(tou_app, {"tariff": None})
    assert revalued["provenance"]["revaluation"]["method"] == "repriced"
    assert _fields(revalued) == _fields(_simulated({**BASE, "tariff": None}).result())


@pytest.mark.parametrize("changes", [{"battery_kwh": 3.0}, {"projection_years": 5}, {"location": "lisbon"}])
def test_keys_that_change_the_simulation_are_refused(flat_app, changes):
    with pytest.raises(ValueError, match=f"changes prices only, and {next(iter(changes))} is not a price key"):
        _revalued(flat_app, changes)


def test_revaluation_keys_are_the_prices():
    assert REVALUATION_KEYS == {
        "cost_preset",
        "costs",
        "discount_rate",
        "import_price_escalation",
        "inflation_rate",
        "om_escalation",
        "reference_tariff",
        "terminal_value",
        "replacement_cost_learning",
        "sell_price_inflation",
        "tariff",
    }


def test_revalue_needs_a_run():
    with pytest.raises(RuntimeError, match="simulate"):
        App(BASE).revalue({"discount_rate": 0.05})


def test_an_unknown_key_is_named(flat_app):
    with pytest.raises(ValueError, match="Unknown config key\\(s\\) for revalue\\(\\): discount"):
        _revalued(flat_app, {"discount": 0.05})


def test_a_price_list_is_replaced_whole(tou_app):
    # Merged key by key, "all" would sit beside peak and off_peak and never apply.
    changes = {"tariff": {"import_prices": {"all": 0.2}}}
    revalued = _revalued(tou_app, changes)
    fresh = _simulated({**BASE, "tariff": {**TOU, "import_prices": {"all": 0.2}}}).result()
    _assert_close(_fields(revalued), _fields(fresh), rel=1e-12)
    assert revalued["npv_savings"] != tou_app.result()["npv_savings"]


def test_none_removes_a_key_so_flat_prices_can_give_way_to_a_tariff():
    flat = _simulated({**BASE, "costs": {"electricity_cost": 0.24}})
    revalued = _revalued(flat, {"costs": {"electricity_cost": None}, "tariff": TOU})
    assert revalued["provenance"]["revaluation"]["method"] == "repriced_by_step"
    assert _fields(revalued) == _fields(_simulated({**BASE, "costs": {}, "tariff": TOU}).result())


def test_a_revalued_result_shares_nothing_with_the_run():
    app = _simulated(BASE)
    before = _fields(app.result())
    revalued = _revalued(app, {"discount_rate": 0.06})
    revalued["degradation"]["engine"] = "edited"
    revalued["pv_loss_waterfall"]["stages"].clear()
    revalued["provenance"]["execution"]["backend"] = "edited"
    app.result()["degradation"]["engine"] = "edited too"
    assert _fields(_revalued(app, {})) == before


def test_none_in_a_new_table_removes_nothing(flat_app):
    revalued = _revalued(flat_app, {"costs": {"storage_cost_per_kwh": None}})
    assert _fields(revalued) == _fields(flat_app.result())
