"""App integration tests for static time-of-use tariff valuation."""

import json

import pandas as pd
import pytest

from breos.app import App
from breos.runners.app import _tariff_year_values
from breos.tariffs import TariffPrices, resolve_named_tariff

BASE_CONFIG = {
    "location": "porto",
    "n_modules": 6,
    "annual_consumption_kwh": 3000,
    "cost_preset": "residential_pt",
    "projection_years": 1,
}

DAILY_BI_TARIFF = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "eur",
    "import_prices": {"off_peak": 0.10, "peak": 0.40},
    "export_prices": {"all": 0.05},
    "fixed_charge_per_day": 0.25,
}


@pytest.mark.parametrize("value", [None, "pt_mainland_2026_daily_bi", 2])
def test_tariff_config_must_be_a_table(value):
    with pytest.raises(TypeError, match="'tariff' must be a table"):
        App({**BASE_CONFIG, "tariff": value})


def test_tariff_config_requires_schedule_currency_and_prices():
    with pytest.raises(ValueError, match="tariff.currency.*tariff.import_prices.*tariff.export_prices"):
        App({**BASE_CONFIG, "tariff": {"schedule": "pt_mainland_2026_daily_bi"}})


def test_tariff_config_rejects_unknown_nested_keys_and_schedules():
    with pytest.raises(ValueError, match="Unknown tariff key.*tariff.prices"):
        App({**BASE_CONFIG, "tariff": {**DAILY_BI_TARIFF, "prices": {}}})

    with pytest.raises(ValueError, match="Unknown tariff schedule"):
        App({**BASE_CONFIG, "tariff": {**DAILY_BI_TARIFF, "schedule": "pt"}})


def test_tariff_config_rejects_unknown_or_missing_period_prices():
    with pytest.raises(ValueError, match="Unknown import_prices period.*typo"):
        App(
            {
                **BASE_CONFIG,
                "tariff": {**DAILY_BI_TARIFF, "import_prices": {"all": 0.20, "typo": 0.30}},
            }
        )

    with pytest.raises(ValueError, match="Missing import price.*peak"):
        App(
            {
                **BASE_CONFIG,
                "tariff": {**DAILY_BI_TARIFF, "import_prices": {"off_peak": 0.10}},
            }
        )


def test_tariff_config_rejects_an_incompatible_hourly_schedule_before_simulation():
    with pytest.raises(ValueError, match="requires 30-minute resolution"):
        App(
            {
                **BASE_CONFIG,
                "tariff": {
                    **DAILY_BI_TARIFF,
                    "schedule": "pt_mainland_2026_daily_tri",
                    "import_prices": {"all": 0.20},
                },
            }
        )


def test_tariff_config_normalizes_public_values():
    app = App({**BASE_CONFIG, "tariff": DAILY_BI_TARIFF})

    assert app._cfg["tariff"] == {
        **DAILY_BI_TARIFF,
        "currency": "EUR",
        "boundary_policy": "strict",
    }


def test_future_schedule_requires_a_valid_study_date_for_reference_year_weather():
    tariff = {
        "schedule": "pt_mainland_2027_daily_tri",
        "currency": "EUR",
        "import_prices": {"off_peak": 0.10, "mid_peak": 0.20, "peak": 0.30},
        "export_prices": {"all": 0.05},
    }

    with pytest.raises(ValueError, match="not effective across the index date range.*study_date"):
        App({**BASE_CONFIG, "resolution": "15min", "tariff": tariff})

    app = App(
        {
            **BASE_CONFIG,
            "resolution": "15min",
            "tariff": {**tariff, "study_date": "2027-07-01"},
        }
    )
    assert app._cfg["tariff"]["study_date"] == "2027-07-01"


def test_timestep_valuation_uses_resolved_prices_for_system_and_baseline():
    index = pd.date_range("2026-01-05 07:00", periods=4, freq="h", tz="Europe/Lisbon")
    prices = TariffPrices(
        currency="EUR",
        import_prices={"off_peak": 0.10, "peak": 0.40},
        export_prices={"all": 0.05},
        fixed_charge_per_day=0.25,
    )
    tariff = resolve_named_tariff(index, "pt_mainland_2026_daily_bi", prices)
    results = pd.DataFrame(
        {
            "Datetime": index,
            "Import_From_Grid": [1000.0, 2000.0, 3000.0, 4000.0],
            "Sell_To_Grid": [400.0, 300.0, 200.0, 100.0],
            "Houseload": [5000.0, 5000.0, 5000.0, 5000.0],
        }
    )

    values = _tariff_year_values(results, tariff, "h")

    assert tariff.period_labels == ("off_peak", "peak", "peak", "peak")
    assert values["Tariff_Import_Cost_Base"] == pytest.approx(0.1 + 0.8 + 1.2 + 1.6)
    assert values["Tariff_Export_Revenue_Base"] == pytest.approx(0.05)
    assert values["Tariff_Baseline_Import_Cost_Base"] == pytest.approx(0.5 + 2.0 + 2.0 + 2.0)
    assert values["Tariff_Fixed_Charge_Base"] == pytest.approx(0.25)


def test_tou_valuation_is_additive_to_physical_results(_patch_weather):
    flat = App(BASE_CONFIG)
    tou = App({**BASE_CONFIG, "tariff": DAILY_BI_TARIFF})

    flat.simulate()
    tou.simulate()
    flat_result = flat.result()
    result = tou.result()

    for key in (
        "pv_dc_generation_kwh",
        "usable_ac_system_production_kwh",
        "consumption_kwh",
        "self_consumption_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "grid_independence_pct",
    ):
        assert result[key] == flat_result[key]

    assert "tariff" not in flat_result
    assert "currency" not in flat_result
    assert result["currency"] == "EUR"
    assert result["total_investment"] == result["total_investment_eur"]
    assert result["npv_savings"] == result["npv_savings_eur"]
    assert result["lcoe_per_kwh"] == result["lcoe_eur_kwh"]

    tariff = result["tariff"]
    assert tariff["schedule"]["identifier"] == "pt_mainland_2026_daily_bi"
    assert tariff["study_date"] is None
    assert tariff["schedule"]["hash"] == result["provenance"]["tariff"]["schedule"]["hash"]
    assert tariff["prices"]["hash"] == result["provenance"]["tariff"]["prices"]["hash"]
    assert tariff["resolved_steps"] == 8760
    assert len(tariff["schedule"]["hash"]) == 64
    assert len(tariff["prices"]["hash"]) == 64

    year = tariff["yearly"][0]
    periods = year["periods"]
    assert sum(period["import_kwh"] for period in periods.values()) == pytest.approx(
        result["grid_import_kwh"], abs=0.01
    )
    assert sum(period["export_kwh"] for period in periods.values()) == pytest.approx(
        result["grid_export_kwh"], abs=0.01
    )
    assert sum(period["baseline_import_kwh"] for period in periods.values()) == pytest.approx(
        result["consumption_kwh"], abs=0.01
    )
    assert sum(period["import_cost"] for period in periods.values()) == pytest.approx(year["import_cost"])
    assert sum(period["export_revenue"] for period in periods.values()) == pytest.approx(year["export_revenue"])
    assert sum(period["baseline_import_cost"] for period in periods.values()) == pytest.approx(
        year["baseline_import_cost"]
    )
    assert year["fixed_charge"] == pytest.approx(365 * 0.25)
    assert year["baseline_total_cost"] == pytest.approx(year["baseline_import_cost"] + year["fixed_charge"])
    assert year["net_grid_cost"] == pytest.approx(year["import_cost"] - year["export_revenue"] + year["fixed_charge"])
    json.dumps(result)
