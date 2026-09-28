"""Currency-neutral result names and the result schema version (ADR 0003 E8, E9)."""

from unittest import mock

import pandas as pd
import pytest

from breos.cli import _load_options
from breos.io import _economics_summary_metrics
from breos.optimization import SolarDesignProblem
from breos.plotting import _currency, plot_breakeven_comparison
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.tariffs import DEFAULT_CURRENCY, result_currency
from tools.generate_app_golden import SCENARIOS, _fake_fetch


@pytest.fixture(scope="module")
def replacement_result():
    from breos import App

    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App({**SCENARIOS["native_h_replacement"], "execution_backend": "python"})
        app.simulate()
        return app.result()


def _keys(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{prefix}{key}"
            yield from _keys(item, f"{prefix}{key}.")
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item, prefix)


def test_the_first_result_schema_version_is_1_0():
    assert RESULT_SCHEMA_VERSION == "1.0"


def test_app_result_records_the_schema_version_and_currency(replacement_result):
    assert replacement_result["result_schema_version"] == "1.0"
    assert replacement_result["provenance"]["currency"] == "EUR"


def test_app_result_keys_name_no_currency_and_no_exact_payback(replacement_result):
    # The resolved config echoes input keys, which are not result names.
    keys = [key for key in _keys(replacement_result) if not key.startswith("provenance.resolved_config.")]

    assert not [key for key in keys if "eur" in key.lower() or "exact" in key.lower()]
    for renamed in ("total_investment", "npv_savings", "lcoe_per_kwh", "battery_replacement_cost_t0_prices"):
        assert renamed in replacement_result


def test_replacement_npv_discounts_each_outlay_from_its_swap_instant(replacement_result):
    discount_rate = replacement_result["provenance"]["resolved_config"]["discount_rate"]
    rows = [row for row in replacement_result["financial"] if row.get("cost_replacement")]
    expected = sum(row["cost_replacement"] / (1 + discount_rate) ** row["replacement_time_years"] for row in rows)

    assert replacement_result["battery_replacements"] == 2
    assert len(rows) == 2
    assert replacement_result["battery_replacement_cost_npv"] == pytest.approx(expected, abs=0.02)
    # t = 0 prices, neither inflated nor discounted, beside the discounted total.
    assert replacement_result["battery_replacement_cost_t0_prices"] == 5000.0
    assert replacement_result["battery_replacement_cost_npv"] < replacement_result["battery_replacement_cost_t0_prices"]


def test_a_run_without_a_tariff_is_in_the_default_currency():
    assert DEFAULT_CURRENCY == "EUR"
    assert result_currency(None) == "EUR"


def test_optimizer_rejects_the_removed_budget_key():
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0], "ghi": [0.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "constraints": {"budget_eur": 5000.0},
        "mode": {"fixed_azimuth": 180},
    }

    with pytest.raises(ValueError, match=r"constraints\.budget_eur was renamed to constraints\.budget"):
        SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_budget_eur")

    # Setting the new key as well does not let the old one through.
    config["constraints"] = {"budget_eur": 5000.0, "budget": 5000.0}
    with pytest.raises(ValueError, match=r"constraints\.budget_eur was renamed to constraints\.budget"):
        SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_budget_both")

    config["constraints"] = {"budget": 5000.0}
    assert SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/problem_budget").budget_limit == 5000.0


def test_summary_labels_name_the_projection_currency():
    projection = pd.DataFrame({"Year": [1]})
    projection.attrs.update({"lcoe_per_kwh": 0.1327, "total_investment": 1000.0, "final_npv_savings": 50.0})

    assert set(_economics_summary_metrics(projection)) == {
        "LCOE [EUR/kWh]",
        "Total Investment [EUR]",
        "NPV Savings [EUR]",
    }

    projection.attrs["currency"] = "CHF"
    assert set(_economics_summary_metrics(projection)) == {
        "LCOE [CHF/kWh]",
        "Total Investment [CHF]",
        "NPV Savings [CHF]",
    }


def test_plot_labels_read_the_frame_currency():
    frame = pd.DataFrame()
    assert _currency(frame) == "EUR"
    frame.attrs["currency"] = "CHF"
    assert _currency(frame) == "CHF"


def test_breakeven_comparison_draws_an_empty_list(tmp_path):
    pytest.importorskip("matplotlib")
    plot_breakeven_comparison([], [], [], str(tmp_path), "empty.png")

    assert (tmp_path / "empty.png").exists()


def test_cost_presets_state_their_currency():
    rows = _load_options("cost-presets")

    assert rows
    for row in rows:
        assert row["currency"] == "EUR"
        assert {"electricity_cost_per_kwh", "export_price_per_kwh", "storage_cost_per_kwh"} <= set(row)
        assert not [key for key in row if "eur" in key]
