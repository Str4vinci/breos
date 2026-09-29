"""App.revalue re-prices a finished run, or re-simulates when prices change the dispatch (#183)."""

import math
from unittest import mock

import pytest

from breos import App
from breos.app import REVALUATION_KEYS
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


def test_a_new_schedule_is_simulated_again():
    # The same periods on another calendar: the stored energy by period no
    # longer applies. The weekly schedule has half-hour boundaries.
    config = {**BASE, "resolution": "15min", "projection_years": 1, "tariff": TOU}
    changes = {"tariff": {"schedule": "pt_mainland_2026_weekly_bi"}}
    revalued = _revalued(_simulated(config), changes)
    assert revalued["provenance"]["revaluation"]["method"] == "resimulated"
    assert _fields(revalued) == _fields(_simulated(_merge(config, changes)).result())


def test_a_tariff_added_is_simulated_again(flat_app):
    revalued = _revalued(flat_app, {"tariff": TOU})
    assert revalued["provenance"]["revaluation"] == {"method": "resimulated", "changed_keys": ["tariff"]}
    assert _fields(revalued) == _fields(_simulated({**BASE, "tariff": TOU}).result())


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
    assert revalued["provenance"]["revaluation"]["method"] == "resimulated"
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
