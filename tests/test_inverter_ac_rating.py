"""An absolute inverter AC rating, resolved once and read by every consumer (#181)."""

from unittest import mock

import pytest

from breos import App
from breos.app_config import build_costs_dict, resolve_app_config
from tools.generate_app_golden import SCENARIOS, _fake_fetch

BASE = {**SCENARIOS["native_h_battery"], "execution_backend": "python"}


def test_the_rating_sets_the_resolved_capacity():
    resolved = resolve_app_config({**BASE, "inverter_ac_rating_kw": 3.2})
    assert resolved.inverter_ac_capacity_w == 3200.0
    # The ratio no longer sizes anything, so it is reported unset.
    assert resolved.cfg["inverter_loading_ratio"] is None


def test_without_the_rating_the_ratio_sizes_the_inverter():
    resolved = resolve_app_config(BASE)
    assert resolved.cfg["inverter_ac_rating_kw"] is None
    assert resolved.inverter_ac_capacity_w == BASE["n_modules"] * resolved.avg_module_power_w / 1.25


def test_rating_and_ratio_are_exclusive():
    with pytest.raises(ValueError, match="both size the inverter; set one of them"):
        resolve_app_config({**BASE, "inverter_ac_rating_kw": 3.2, "inverter_loading_ratio": 1.2})


@pytest.mark.parametrize("rating", [0, -1.0, float("nan")])
def test_a_bad_rating_is_refused(rating):
    with pytest.raises(ValueError, match="inverter_ac_rating_kw"):
        resolve_app_config({**BASE, "inverter_ac_rating_kw": rating})


@pytest.mark.parametrize("config", [{"inverter_ac_rating_kw": 3.2}, {"inverter_loading_ratio": 1.4}])
def test_capex_prices_the_rating_that_clips(config):
    resolved = resolve_app_config({**BASE, **config})
    costs = build_costs_dict(resolved.cfg, resolved)
    kw = resolved.inverter_ac_capacity_w / 1000
    assert costs["inverter_cost"] == resolved.cost_params.inverter_cost_per_kw * kw


def test_the_run_clips_and_reports_at_the_rating():
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        low = App({**BASE, "inverter_ac_rating_kw": 1.5})
        low.simulate()
        high = App({**BASE, "inverter_ac_rating_kw": 6.0})
        high.simulate()
    low_result, high_result = low.result(), high.result()
    assert low_result["pv_loss_waterfall"]["inverter"]["ac_capacity_kw"] == 1.5
    assert high_result["pv_loss_waterfall"]["inverter"]["ac_capacity_kw"] == 6.0
    # A smaller inverter clips more and costs less.
    assert low_result["total_investment"] < high_result["total_investment"]
    assert low_result["usable_ac_system_production_kwh"] < high_result["usable_ac_system_production_kwh"]
    assert low_result["provenance"]["resolved_config"]["inverter_ac_rating_kw"] == 1.5
