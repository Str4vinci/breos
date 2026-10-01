"""Currency-neutral result names and the result schema version (ADR 0003 E8, E9)."""

from unittest import mock

import pandas as pd
import pytest

from breos.cli import _load_options
from breos.optimization import SolarDesignProblem
from breos.plotting import _currency, plot_breakeven_comparison
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.tariffs import DEFAULT_CURRENCY, result_currency
from tools.generate_app_golden import SCENARIOS, _fake_fetch


@pytest.fixture(scope="module")
def replacement_app():
    from breos import App

    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App({**SCENARIOS["native_h_replacement"], "execution_backend": "python"})
        app.simulate()
        return app


@pytest.fixture(scope="module")
def replacement_result(replacement_app):
    return replacement_app.result()


def _keys(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{prefix}{key}"
            yield from _keys(item, f"{prefix}{key}.")
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item, prefix)


def test_the_result_schema_version_is_2_6():
    assert RESULT_SCHEMA_VERSION == "2.7"


def test_app_result_records_the_schema_version_and_currency(replacement_result):
    assert replacement_result["result_schema_version"] == "2.7"
    assert replacement_result["provenance"]["currency"] == "EUR"


def test_a_flat_run_reports_its_year1_money_at_the_top_level(replacement_app, replacement_result):
    result = replacement_result
    year1 = result["financial"][1]
    assert year1["year"] == 1

    # Year 1 is escalated by (1 + i)^0, so its projected cashflows are the year-1 prices.
    assert result["grid_import_cost_year1_prices"] == year1["cost_import"]
    assert result["grid_export_revenue_year1_prices"] == year1["revenue_export"]
    assert result["fixed_charge_year1_prices"] == year1["cost_fixed_charge"]
    # The no-system household buys its whole load at the flat price.
    price = replacement_app._resolved.cost_params.electricity_cost
    assert result["no_system_import_cost_year1_prices"] == pytest.approx(result["consumption_kwh"] * price, abs=0.01)
    assert result["no_system_import_cost_year1_prices"] > result["grid_import_cost_year1_prices"] > 0.0
    # Only smart charging buys energy for the battery.
    assert "grid_charge_cost_year1_prices" not in result


def test_app_result_keys_name_no_currency_and_no_exact_payback(replacement_result):
    # The resolved config echoes input keys, which are not result names.
    keys = [key for key in _keys(replacement_result) if not key.startswith("provenance.resolved_config.")]

    assert not [key for key in keys if "eur" in key.lower() or "exact" in key.lower()]
    for renamed in ("total_investment", "npv_savings", "lcoe_per_kwh", "battery_replacement_cost_t0_prices"):
        assert renamed in replacement_result


def test_schema_2_removes_legacy_and_duplicate_result_keys(replacement_result):
    keys = set(_keys(replacement_result))
    removed = {
        "pv_production_kwh",
        "pv_kwh",
        "co2_avoided_year1_kg",
        "co2_avoided_total_kg",
        "provenance.resolved_config.dc_coupled",
    }
    assert not removed & keys
    assert replacement_result["usable_ac_system_production_kwh"] > 0
    assert replacement_result["usable_ac_system_production_kwh"] == pytest.approx(
        replacement_result["yearly"][0]["usable_ac_system_production_kwh"], abs=0.01
    )
    assert replacement_result["usable_ac_system_production_kwh"] == pytest.approx(
        replacement_result["self_consumption_kwh"] + replacement_result["grid_export_kwh"], abs=0.02
    )
    assert replacement_result["co2_avoided_total_year1_kg"] > 0
    assert replacement_result["co2_avoided_total_lifetime_kg"] > 0
    for row in [*replacement_result["monthly"], *replacement_result["yearly"]]:
        assert row["usable_ac_system_production_kwh"] == pytest.approx(
            row["direct_pv_ac_load_kwh"] + row["pv_origin_battery_ac_load_kwh"] + row["grid_export_kwh"],
            abs=0.02,
        )
        assert "grid_import_kwh" in row
        assert "grid_export_kwh" in row
        assert "pv_kwh" not in row
        assert "import_kwh" not in row
        assert "export_kwh" not in row


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
        SolarDesignProblem(tmy_data, houseload, config)

    # Setting the new key as well does not let the old one through.
    config["constraints"] = {"budget_eur": 5000.0, "budget": 5000.0}
    with pytest.raises(ValueError, match=r"constraints\.budget_eur was renamed to constraints\.budget"):
        SolarDesignProblem(tmy_data, houseload, config)

    config["constraints"] = {"budget": 5000.0}
    assert SolarDesignProblem(tmy_data, houseload, config).budget_limit == 5000.0


def test_plot_labels_read_the_frame_currency():
    frame = pd.DataFrame()
    assert _currency(frame) == "EUR"
    frame.attrs["currency"] = "CHF"
    assert _currency(frame) == "CHF"


def test_breakeven_comparison_draws_an_empty_list(tmp_path):
    pytest.importorskip("matplotlib")
    plot_breakeven_comparison([], [], str(tmp_path), filename="empty.png")

    assert (tmp_path / "empty.png").exists()


def test_cost_presets_state_their_currency():
    rows = _load_options("cost-presets")

    assert rows
    for row in rows:
        assert row["currency"] == "EUR"
        assert {"electricity_cost_per_kwh", "export_price_per_kwh", "storage_cost_per_kwh"} <= set(row)
        assert not [key for key in row if "eur" in key]
