"""Tests for the public API facade (breos.App)."""

import json

import pytest

import breos
from breos.app import App

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestAppValidation:
    def test_missing_location(self):
        with pytest.raises(ValueError, match="location"):
            App({"n_modules": 10, "annual_consumption_kwh": 4000})

    def test_missing_n_modules(self):
        with pytest.raises(ValueError, match="n_modules"):
            App({"location": "porto", "annual_consumption_kwh": 4000})

    def test_missing_consumption(self):
        with pytest.raises(ValueError, match="annual_consumption_kwh"):
            App({"location": "porto", "n_modules": 10})

    def test_invalid_location_key(self):
        with pytest.raises(ValueError, match="Unknown location"):
            App({"location": "atlantis", "n_modules": 10, "annual_consumption_kwh": 4000})

    def test_invalid_n_modules_zero(self):
        with pytest.raises(ValueError, match="n_modules"):
            App({"location": "porto", "n_modules": 0, "annual_consumption_kwh": 4000})

    def test_invalid_consumption_negative(self):
        with pytest.raises(ValueError, match="annual_consumption_kwh"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": -100})

    def test_invalid_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "resolution": "30min"})

    def test_invalid_cost_preset(self):
        with pytest.raises(ValueError, match="Unknown cost preset"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "cost_preset": "fake_preset"})

    def test_invalid_emissions_country(self):
        with pytest.raises(ValueError, match="Unknown emissions country"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "emissions_country": "XX"})

    def test_custom_location_missing_fields(self):
        with pytest.raises(ValueError, match="latitude"):
            App({"location": {"longitude": -8.6}, "n_modules": 10, "annual_consumption_kwh": 4000})

    def test_custom_location_valid(self):
        app = App(
            {
                "location": {"latitude": 41.15, "longitude": -8.63, "timezone": "Europe/Lisbon"},
                "n_modules": 6,
                "annual_consumption_kwh": 3000,
            }
        )
        assert app._lat == 41.15

    def test_pv_arrays_allow_missing_n_modules(self):
        app = App(
            {
                "location": "porto",
                "annual_consumption_kwh": 3000,
                "pv_arrays": [
                    {"modules": 3, "module": "Erlangen_445W", "tilt": 10, "azimuth": 90},
                    {"modules": 3, "module": "Erlangen_445W", "tilt": 10, "azimuth": 270},
                ],
            }
        )
        assert app._cfg["n_modules"] == 6

    def test_result_before_simulate(self):
        app = App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000})
        with pytest.raises(RuntimeError, match="simulate"):
            app.result()


# ---------------------------------------------------------------------------
# Simulation (with monkeypatched weather)
# ---------------------------------------------------------------------------


class TestAppSimulateNoBattery:
    @pytest.fixture(autouse=True)
    def _setup(self, _patch_weather):
        self.app = App(
            {
                "location": "porto",
                "n_modules": 6,
                "annual_consumption_kwh": 3000,
                "cost_preset": "residential_pt",
                "emissions_country": "PT",
                "projection_years": 5,
            }
        )
        self.app.simulate()
        self.result = self.app.result()

    def test_result_is_dict(self):
        assert isinstance(self.result, dict)

    def test_json_serializable(self):
        json.dumps(self.result)

    def test_no_battery_keys(self):
        assert "battery_soh_end_pct" not in self.result
        assert "battery_replacements" not in self.result

    def test_expected_keys_present(self):
        expected = {
            "n_modules",
            "pv_kwp",
            "battery_kwh",
            "pv_production_kwh",
            "consumption_kwh",
            "self_consumption_kwh",
            "grid_import_kwh",
            "grid_export_kwh",
            "grid_independence_pct",
            "self_consumption_pct",
            "total_investment_eur",
            "payback_year",
            "npv_savings_eur",
            "lcoe_eur_kwh",
            "co2_avoided_year1_kg",
            "co2_avoided_total_kg",
            "yearly",
            "monthly",
            "financial",
        }
        assert expected.issubset(self.result.keys())

    def test_yearly_length(self):
        assert len(self.result["yearly"]) == 5

    def test_monthly_and_financial_lengths(self):
        assert len(self.result["monthly"]) == 12
        assert len(self.result["financial"]) == 6

    def test_system_echo(self):
        assert self.result["n_modules"] == 6
        assert self.result["battery_kwh"] == 0.0

    def test_pv_production_positive(self):
        assert self.result["pv_production_kwh"] > 0

    def test_grid_independence_range(self):
        gi = self.result["grid_independence_pct"]
        assert 0 <= gi <= 100

    def test_energy_conservation(self):
        r = self.result
        # self_consumption + export should approximately equal PV production
        assert abs(r["self_consumption_kwh"] + r["grid_export_kwh"] - r["pv_production_kwh"]) < 1.0

    def test_investment_positive(self):
        assert self.result["total_investment_eur"] > 0

    def test_lcoe_positive(self):
        assert self.result["lcoe_eur_kwh"] > 0


class TestAppSimulateMultiArray:
    @pytest.fixture(autouse=True)
    def _setup(self, _patch_weather):
        self.app = App(
            {
                "location": "porto",
                "annual_consumption_kwh": 3000,
                "cost_preset": "residential_pt",
                "projection_years": 5,
                "pv_arrays": [
                    {"modules": 3, "module": "Erlangen_445W", "slope": 10, "azimuth": 90},
                    {"modules": 3, "module": "Erlangen_445W", "slope": 10, "azimuth": 270},
                ],
            }
        )
        self.app.simulate()
        self.result = self.app.result()

    def test_arrays_are_echoed(self):
        assert self.result["n_modules"] == 6
        assert len(self.result["pv_arrays"]) == 2
        assert {arr["azimuth"] for arr in self.result["pv_arrays"]} == {90.0, 270.0}

    def test_multi_array_result_has_chart_data(self):
        assert len(self.result["monthly"]) == 12
        assert self.result["financial"][0]["year"] == 0
        assert self.result["financial"][-1]["year"] == 5

    def test_multi_array_energy_positive(self):
        assert self.result["pv_production_kwh"] > 0


class TestAppSimulateWithBattery:
    @pytest.fixture(autouse=True)
    def _setup(self, _patch_weather):
        self.app = App(
            {
                "location": "porto",
                "n_modules": 6,
                "annual_consumption_kwh": 3000,
                "battery_kwh": 5.0,
                "cost_preset": "residential_pt",
                "emissions_country": "PT",
                "projection_years": 5,
            }
        )
        self.app.simulate()
        self.result = self.app.result()

    def test_battery_keys_present(self):
        assert "battery_soh_end_pct" in self.result
        assert "battery_replacements" in self.result
        assert "battery_replacement_cost_eur" in self.result

    def test_battery_soh_range(self):
        soh = self.result["battery_soh_end_pct"]
        assert 0 < soh <= 100

    def test_battery_improves_grid_independence(self, _patch_weather):
        # Same system without battery should have lower grid independence
        app_no_batt = App(
            {
                "location": "porto",
                "n_modules": 6,
                "annual_consumption_kwh": 3000,
                "cost_preset": "residential_pt",
                "projection_years": 5,
            }
        )
        app_no_batt.simulate()
        gi_no_batt = app_no_batt.result()["grid_independence_pct"]
        gi_with_batt = self.result["grid_independence_pct"]
        assert gi_with_batt >= gi_no_batt

    def test_yearly_has_soh(self):
        for year in self.result["yearly"]:
            assert "soh_pct" in year

    def test_json_serializable(self):
        json.dumps(self.result)
