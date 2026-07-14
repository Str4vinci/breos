"""Tests for the public API facade (breos.App)."""

import json

import pytest

import breos
import breos.app as app_module
from breos.app import App
from breos.load_profiles import load_profile as real_load_profile

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

    def test_invalid_negative_battery_kwh(self):
        # A negative battery must not silently reduce CAPEX
        with pytest.raises(ValueError, match="battery_kwh"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "battery_kwh": -5})

    def test_invalid_top_level_tilt(self):
        with pytest.raises(ValueError, match="tilt"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "tilt": 120})

    def test_invalid_top_level_azimuth(self):
        with pytest.raises(ValueError, match="azimuth"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "azimuth": 400})

    def test_invalid_transposition_model(self):
        with pytest.raises(ValueError, match="transposition_model"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "transposition_model": "not_a_model",
                }
            )

    def test_invalid_per_array_transposition_model(self):
        with pytest.raises(ValueError, match=r"pv_arrays\[0\].transposition_model"):
            App(
                {
                    "location": "porto",
                    "annual_consumption_kwh": 4000,
                    "pv_arrays": [{"modules": 5, "transposition_model": "not_a_model"}],
                }
            )

    def test_invalid_albedo(self):
        with pytest.raises(ValueError, match="albedo"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "albedo": 1.5})

    def test_invalid_surface_type(self):
        with pytest.raises(ValueError, match="surface_type"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "surface_type": "lava"})

    def test_albedo_and_surface_type_conflict(self):
        with pytest.raises(ValueError, match="either"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "albedo": 0.3,
                    "surface_type": "snow",
                }
            )

    def test_invalid_model_perez(self):
        with pytest.raises(ValueError, match="model_perez"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "model_perez": "nope"})

    def test_invalid_per_array_albedo(self):
        with pytest.raises(ValueError, match=r"pv_arrays\[0\].albedo"):
            App(
                {
                    "location": "porto",
                    "annual_consumption_kwh": 4000,
                    "pv_arrays": [{"modules": 5, "albedo": 9}],
                }
            )

    def test_invalid_inverter_efficiency(self):
        with pytest.raises(ValueError, match="inverter_efficiency"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "inverter_efficiency": 1.5})

    def test_invalid_inverter_loading_ratio(self):
        with pytest.raises(ValueError, match="inverter_loading_ratio"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "inverter_loading_ratio": 0})

    def test_invalid_projection_years(self):
        # Must fail at config load, not with a late RuntimeError mid-simulation
        with pytest.raises(ValueError, match="projection_years"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "projection_years": 0})

    def test_invalid_pv_degradation_rate(self):
        with pytest.raises(ValueError, match="pv_degradation_rate"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "pv_degradation_rate": 1.5})

    def test_invalid_battery_soc_window(self):
        with pytest.raises(ValueError, match="battery_min_soc"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "battery_min_soc": 0.9,
                    "battery_max_soc": 0.2,
                }
            )

    def test_invalid_battery_rte(self):
        with pytest.raises(ValueError, match="battery_rte"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "battery_rte": 1.5,
                }
            )

    def test_invalid_battery_eol(self):
        with pytest.raises(ValueError, match="battery_eol_percentage"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "battery_eol_percentage": 0.0,
                }
            )

    def test_invalid_sell_price_inflation(self):
        with pytest.raises(ValueError, match="sell_price_inflation"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "sell_price_inflation": 1.0,
                }
            )

    def test_invalid_pv_loss_overrides_value(self):
        with pytest.raises(ValueError, match="pv_loss_overrides"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "pv_loss_overrides": {"shading": 200},
                }
            )

    def test_invalid_pv_loss_overrides_type(self):
        with pytest.raises(TypeError, match="pv_loss_overrides"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "pv_loss_overrides": 5.0,
                }
            )

    def test_unknown_config_key_rejected(self):
        # A typo such as `batery_kwh` must fail loudly instead of being silently
        # dropped by merge_defaults (which would default the battery to 0).
        with pytest.raises(ValueError, match="Unknown config key"):
            App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "batery_kwh": 5.0,
                }
            )

    def test_unknown_config_key_lists_the_offending_key(self):
        with pytest.raises(ValueError, match="batery_kwh"):
            App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "batery_kwh": 5.0})

    def test_montecarlo_section_is_allowed(self):
        # MC configs carry a [montecarlo] section and validate through the same
        # path; it must not be flagged as an unknown key.
        app = App(
            {
                "location": "porto",
                "n_modules": 10,
                "annual_consumption_kwh": 4000,
                "montecarlo": {"n_runs": 10, "weather_file": "weather.csv"},
            }
        )
        assert app._cfg["n_modules"] == 10

    def test_custom_location_valid(self):
        app = App(
            {
                "location": {"latitude": 41.15, "longitude": -8.63, "timezone": "Europe/Lisbon"},
                "n_modules": 6,
                "annual_consumption_kwh": 3000,
            }
        )
        assert app._resolved.lat == 41.15

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

    def test_resolution_does_not_mutate_input_config(self):
        # Resolving the derived module count must not write back into the
        # caller's dict (the frozen ResolvedAppConfig owns its own copy).
        user_config = {
            "location": "porto",
            "annual_consumption_kwh": 3000,
            "pv_arrays": [
                {"modules": 3, "module": "Erlangen_445W", "tilt": 10, "azimuth": 90},
                {"modules": 4, "module": "Erlangen_445W", "tilt": 10, "azimuth": 270},
            ],
        }
        app = App(user_config)
        assert "n_modules" not in user_config
        assert app._cfg["n_modules"] == 7

    def test_result_before_simulate(self):
        app = App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000})
        with pytest.raises(RuntimeError, match="simulate"):
            app.result()

    def test_simulate_passes_external_rlp_directory(self, _patch_weather, monkeypatch, tmp_path):
        seen = {}

        def _fake_load_profile(**kwargs):
            seen["rlp_directory"] = kwargs["rlp_directory"]
            return real_load_profile(
                profile_type="1",
                annual_consumption_kwh=kwargs["annual_consumption_kwh"],
                start_date=kwargs["start_date"],
                freq=kwargs["freq"],
                num_years=kwargs["num_years"],
                timezone=kwargs["timezone"],
            )

        monkeypatch.setattr(app_module, "load_profile", _fake_load_profile)

        app = App(
            {
                "location": "porto",
                "n_modules": 1,
                "annual_consumption_kwh": 1000,
                "projection_years": 1,
                "rlp_directory": str(tmp_path),
            }
        )
        app.simulate()

        assert seen["rlp_directory"] == str(tmp_path)

    def test_battery_soc_window_reaches_simulation(self, _patch_weather):
        def _run(**extra):
            app = App(
                {
                    "location": "porto",
                    "n_modules": 6,
                    "annual_consumption_kwh": 3000,
                    "battery_kwh": 5.0,
                    "projection_years": 1,
                    **extra,
                }
            )
            app.simulate()
            return app

        default_gi = _run().result()["grid_independence_pct"]
        narrow_gi = _run(battery_min_soc=0.45, battery_max_soc=0.55).result()["grid_independence_pct"]

        # A 10% SOC window stores a tenth of the energy of the default
        # 10-90% window, so grid independence must drop
        assert narrow_gi < default_gi

    def test_sell_price_inflation_reaches_projection(self, _patch_weather):
        def _run(**extra):
            app = App(
                {
                    "location": "porto",
                    "n_modules": 6,
                    "annual_consumption_kwh": 3000,
                    "projection_years": 2,
                    **extra,
                }
            )
            app.simulate()
            return app.result()["npv_savings_eur"]

        # Inflating the export price raises later-year export revenue, so
        # cumulative NPV savings must grow. The key used to exist only on
        # CostParams and never reached cost_analysis_projection from a config.
        assert _run(sell_price_inflation=0.5) > _run()

    def test_transposition_model_reaches_simulation(self, _patch_weather):
        def _run(model):
            app = App(
                {
                    "location": "porto",
                    "n_modules": 6,
                    "annual_consumption_kwh": 3000,
                    "projection_years": 1,
                    "transposition_model": model,
                }
            )
            app.simulate()
            return app.result()["pv_production_kwh"]

        # The model must flow all the way through App.simulate(); an
        # anisotropic model yields a different PV total than isotropic.
        assert _run("perez") != pytest.approx(_run("isotropic"))

    def test_albedo_reaches_simulation(self, _patch_weather):
        def _run(**extra):
            app = App(
                {
                    "location": "porto",
                    "n_modules": 6,
                    "annual_consumption_kwh": 3000,
                    "projection_years": 1,
                    **extra,
                }
            )
            app.simulate()
            return app.result()["pv_production_kwh"]

        # A higher ground reflectance must raise PV production end-to-end.
        assert _run(albedo=0.65) > _run()
        # surface_type is an equivalent way to set the same albedo.
        assert _run(surface_type="snow") == pytest.approx(_run(albedo=0.65))

    def test_pv_loss_overrides_increase_production(self, _patch_weather):
        def _run(overrides):
            app = App(
                {
                    "location": "porto",
                    "n_modules": 6,
                    "annual_consumption_kwh": 3000,
                    "projection_years": 1,
                    "pv_loss_overrides": overrides,
                }
            )
            app.simulate()
            return app.result()["pv_production_kwh"]

        base = _run(None)
        no_shading = _run({"shading": 0.0})

        # Both operands are rounded to 2 decimals by build_result, so the
        # tightest honest tolerance is the rounding granularity (±0.005 each).
        assert no_shading == pytest.approx(base / 0.97, abs=0.011)

    def test_smaller_inverter_clips_app_production(self, _patch_weather):
        def _run(loading_ratio):
            app = App(
                {
                    "location": "porto",
                    "n_modules": 6,
                    "annual_consumption_kwh": 3000,
                    "projection_years": 1,
                    "inverter_loading_ratio": loading_ratio,
                }
            )
            app.simulate()
            return app.result()["pv_production_kwh"]

        # A heavily undersized inverter (high DC/AC ratio) must clip yield;
        # before 0.3.0 the App paid clipping-sized inverter CAPEX while
        # production was never clipped.
        assert _run(3.0) < _run(1.0)

    def test_simulate_localizes_load_to_location_timezone(self, _patch_weather, monkeypatch):
        seen = {}

        def _fake_load_profile(**kwargs):
            seen["timezone"] = kwargs["timezone"]
            return real_load_profile(
                profile_type="1",
                annual_consumption_kwh=kwargs["annual_consumption_kwh"],
                start_date=kwargs["start_date"],
                freq=kwargs["freq"],
                num_years=kwargs["num_years"],
                timezone=kwargs["timezone"],
            )

        monkeypatch.setattr(app_module, "load_profile", _fake_load_profile)

        app = App(
            {
                "location": "porto",
                "n_modules": 1,
                "annual_consumption_kwh": 1000,
                "projection_years": 1,
            }
        )
        app.simulate()

        assert seen["timezone"] == "Europe/Lisbon"

    def test_simulate_passes_location_timezone_to_tmy_fetch(self, monkeypatch, synthetic_weather):
        seen = {}

        def _fake_fetch(**kwargs):
            seen["timezone"] = kwargs["timezone"]
            return synthetic_weather, {"inputs": {"location": {"latitude": 41.15, "longitude": -8.63, "elevation": 0}}}

        monkeypatch.setattr(app_module, "load_weather", lambda **kw: None)
        monkeypatch.setattr(app_module, "fetch_tmy_weather_data", _fake_fetch)

        app = App(
            {
                "location": "porto",
                "n_modules": 1,
                "annual_consumption_kwh": 1000,
                "projection_years": 1,
            }
        )
        app.simulate()

        assert seen["timezone"] == "Europe/Lisbon"


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
                    {"modules": 3, "module": "Erlangen_445W", "tilt": 10, "azimuth": 90},
                    {"modules": 3, "module": "Erlangen_445W", "tilt": 10, "azimuth": 270},
                ],
            }
        )
        self.app.simulate()
        self.result = self.app.result()

    def test_arrays_are_echoed(self):
        assert self.result["n_modules"] == 6
        assert len(self.result["pv_arrays"]) == 2
        assert {arr["azimuth"] for arr in self.result["pv_arrays"]} == {90.0, 270.0}
        assert {arr["tilt"] for arr in self.result["pv_arrays"]} == {10.0}

    def test_multi_array_result_has_chart_data(self):
        assert len(self.result["monthly"]) == 12
        assert self.result["financial"][0]["year"] == 0
        assert self.result["financial"][-1]["year"] == 5

    def test_multi_array_energy_positive(self):
        assert self.result["pv_production_kwh"] > 0


class TestAppSimulateTracking:
    def test_invalid_tracking(self, _patch_weather):
        with pytest.raises(ValueError, match="tracking must be"):
            App(
                {
                    "location": "porto",
                    "n_modules": 6,
                    "annual_consumption_kwh": 3000,
                    "tracking": "trinity_axis",
                }
            )

    def test_single_axis_runs(self, _patch_weather):
        app = App(
            {
                "location": "porto",
                "n_modules": 6,
                "annual_consumption_kwh": 3000,
                "cost_preset": "residential_pt",
                "projection_years": 3,
                "tracking": "single_axis",
                "max_angle": 60.0,
                "gcr": 0.35,
            }
        )
        app.simulate()
        result = app.result()
        assert result["pv_production_kwh"] > 0
        json.dumps(result)

    def test_dual_axis_runs(self, _patch_weather):
        app = App(
            {
                "location": "porto",
                "n_modules": 6,
                "annual_consumption_kwh": 3000,
                "cost_preset": "residential_pt",
                "projection_years": 3,
                "tracking": "dual_axis",
            }
        )
        app.simulate()
        result = app.result()
        assert result["pv_production_kwh"] > 0

    def test_tracking_beats_fixed_via_app(self, _patch_weather):
        """At the App level, single-axis (no backtrack, ±90°) should beat optimal fixed tilt."""
        common = {
            "location": "porto",
            "n_modules": 6,
            "annual_consumption_kwh": 3000,
            "cost_preset": "residential_pt",
            "projection_years": 1,
        }
        fixed = App(common)
        fixed.simulate()
        tracked = App({**common, "tracking": "single_axis", "backtrack": False, "max_angle": 90.0})
        tracked.simulate()
        assert tracked.result()["pv_production_kwh"] > fixed.result()["pv_production_kwh"]

    def test_per_array_tracking_flows_through(self, _patch_weather):
        """Tracking keys on pv_arrays entries must reach calculate_multi_array_production."""
        fixed_app = App(
            {
                "location": "porto",
                "annual_consumption_kwh": 3000,
                "cost_preset": "residential_pt",
                "projection_years": 1,
                "pv_arrays": [
                    {"modules": 6, "module": "Erlangen_445W", "tilt": 30, "azimuth": 180},
                ],
            }
        )
        fixed_app.simulate()
        tracked_app = App(
            {
                "location": "porto",
                "annual_consumption_kwh": 3000,
                "cost_preset": "residential_pt",
                "projection_years": 1,
                "pv_arrays": [
                    {
                        "modules": 6,
                        "module": "Erlangen_445W",
                        "tracking": "single_axis",
                        "axis_azimuth": 180,
                        "max_angle": 90.0,
                        "backtrack": False,
                    },
                ],
            }
        )
        tracked_app.simulate()
        # Tracking key must echo into result
        assert tracked_app.result()["pv_arrays"][0].get("tracking") == "single_axis"
        # And actually change production vs fixed
        assert tracked_app.result()["pv_production_kwh"] != fixed_app.result()["pv_production_kwh"]


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
