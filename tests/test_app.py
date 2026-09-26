"""Tests for the public API facade (breos.App)."""

import json
import math
from datetime import date

import pandas as pd
import pytest

import breos
import breos.app as app_module
from breos.app import App
from breos.app_config import merge_defaults, validate_config
from breos.load_profiles import load_profile as real_load_profile


def test_app_rejects_invalid_profile_losses_temperature_and_montecarlo_keys():
    base = {"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000}

    with pytest.raises(ValueError, match="Unknown load_profile 'nonexistent'"):
        App({**base, "load_profile": "nonexistent"})
    with pytest.raises(ValueError, match="Unknown loss component"):
        App({**base, "pv_loss_overrides": {"soiling_typo": 2.0}})
    with pytest.raises(FileNotFoundError, match="battery_temperature file not found: wether"):
        App({**base, "battery_temperature": "wether"})
    with pytest.raises(ValueError, match="Unknown Monte Carlo config key.*montecarlo.nruns"):
        App({**base, "montecarlo": {"nruns": 10, "weather_file": "weather.csv"}})


def test_app_resolves_profile_alias_and_native_date_during_construction():
    app = App(
        {
            "location": "porto",
            "n_modules": 10,
            "annual_consumption_kwh": 4000,
            "load_profile": "BDEW_H0",
            "start_date": date(2023, 1, 1),
        }
    )

    assert app._cfg["load_profile"] == "1"
    assert app._cfg["start_date"] == "2023-01-01"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestAppValidation:
    BASE = {"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000}

    @pytest.mark.parametrize("key", ["location", "n_modules", "annual_consumption_kwh"])
    def test_missing_required_key(self, key):
        with pytest.raises(ValueError, match=key):
            App({k: v for k, v in self.BASE.items() if k != key})

    @pytest.mark.parametrize(
        ("overrides", "error", "match"),
        [
            pytest.param({"location": "atlantis"}, ValueError, "Unknown location", id="location-key"),
            pytest.param({"location": {"longitude": -8.6}}, ValueError, "latitude", id="location-missing-latitude"),
            pytest.param(
                {"location": {"latitude": 91, "longitude": 0, "timezone": "UTC"}},
                (TypeError, ValueError),
                "location.latitude",
                id="location-latitude",
            ),
            pytest.param(
                {"location": {"latitude": 0, "longitude": -181, "timezone": "UTC"}},
                (TypeError, ValueError),
                "location.longitude",
                id="location-longitude",
            ),
            pytest.param(
                {"location": {"latitude": 0, "longitude": 0, "timezone": "Mars/Olympus"}},
                (TypeError, ValueError),
                "timezone",
                id="location-timezone",
            ),
            pytest.param({"n_modules": 0}, ValueError, "n_modules", id="n_modules"),
            pytest.param({"annual_consumption_kwh": -100}, ValueError, "annual_consumption_kwh", id="consumption"),
            pytest.param(
                {"annual_consumption_kwh": math.nan},
                (TypeError, ValueError),
                "annual_consumption_kwh",
                id="consumption-nan",
            ),
            pytest.param({"resolution": "30min"}, ValueError, "resolution", id="resolution"),
            pytest.param({"start_date": "2024-02-30"}, ValueError, "start_date", id="start_date"),
            pytest.param({"calendar_model": "invented"}, ValueError, "calendar_model", id="calendar_model"),
            pytest.param(
                {"pv_module": "Suntech_STP550S_STC", "temperature_model": "noct-sam"},
                ValueError,
                "NOCT metadata",
                id="noct-sam-metadata",
            ),
            pytest.param({"pv_module": "Imaginary_900W"}, ValueError, "Unknown PV module", id="pv_module"),
            pytest.param(
                {"pv_arrays": [{"modules": 4, "module": "Imaginary_900W"}]},
                ValueError,
                r"pv_arrays\[0\]",
                id="pv_arrays-module",
            ),
            pytest.param({"cost_preset": "fake_preset"}, ValueError, "Unknown cost preset", id="cost_preset"),
            pytest.param({"costs": 0.25}, TypeError, "'costs' must be a table/dict", id="costs-type"),
            pytest.param(
                {"costs": {"electricty_cost": 0.25}},
                ValueError,
                r"Unknown key 'costs\.electricty_cost'.*costs\.electricity_cost",
                id="costs-unknown-key",
            ),
            *[
                pytest.param(
                    {"costs": {"electricity_cost": value}},
                    (TypeError, ValueError),
                    r"costs\.electricity_cost",
                    id=f"costs-electricity_cost-{value}",
                )
                for value in (-0.01, float("inf"), True)
            ],
            pytest.param({"emissions_country": "XX"}, ValueError, "Unknown emissions country", id="emissions_country"),
            pytest.param(
                {"export_emissions_factor_gco2_kwh": -1.0},
                (TypeError, ValueError),
                "export_emissions_factor_gco2_kwh",
                id="export_emissions_factor_gco2_kwh",
            ),
            # A negative battery must not silently reduce CAPEX
            pytest.param({"battery_kwh": -5}, ValueError, "battery_kwh", id="battery_kwh"),
            pytest.param({"battery_kwh": math.inf}, (TypeError, ValueError), "battery_kwh", id="battery_kwh-inf"),
            pytest.param({"tilt": 120}, ValueError, "tilt", id="tilt"),
            pytest.param({"azimuth": 400}, ValueError, "azimuth", id="azimuth"),
            pytest.param({"transposition_model": "not_a_model"}, ValueError, "transposition_model", id="transposition"),
            pytest.param({"iam_model": "not_a_model"}, ValueError, "iam_model", id="iam_model"),
            pytest.param(
                {"pv_arrays": [{"modules": 5, "transposition_model": "not_a_model"}]},
                ValueError,
                r"pv_arrays\[0\].transposition_model",
                id="pv_arrays-transposition",
            ),
            pytest.param({"albedo": 1.5}, ValueError, "albedo", id="albedo"),
            pytest.param({"surface_type": "lava"}, ValueError, "surface_type", id="surface_type"),
            pytest.param({"albedo": 0.3, "surface_type": "snow"}, ValueError, "either", id="albedo-surface-conflict"),
            pytest.param({"model_perez": "nope"}, ValueError, "model_perez", id="model_perez"),
            pytest.param(
                {"pv_arrays": [{"modules": 5, "albedo": 9}]},
                ValueError,
                r"pv_arrays\[0\].albedo",
                id="pv_arrays-albedo",
            ),
            pytest.param({"inverter_efficiency": 1.5}, ValueError, "inverter_efficiency", id="inverter_efficiency"),
            pytest.param(
                {"inverter_efficiency": "0.96"},
                (TypeError, ValueError),
                "inverter_efficiency",
                id="inverter_efficiency-str",
            ),
            pytest.param({"inverter_loading_ratio": 0}, ValueError, "inverter_loading_ratio", id="loading_ratio"),
            pytest.param(
                {"inverter_loading_ratio": True},
                (TypeError, ValueError),
                "inverter_loading_ratio",
                id="loading_ratio-bool",
            ),
            pytest.param({"dc_coupled": False}, NotImplementedError, "DC-coupled", id="ac-coupled"),
            pytest.param({"dc_coupled": "true"}, TypeError, "dc_coupled", id="dc_coupled-type"),
            # Must fail at config load, not with a late RuntimeError mid-simulation
            pytest.param({"projection_years": 0}, ValueError, "projection_years", id="projection_years"),
            pytest.param(
                {"projection_years": 2.5},
                (TypeError, ValueError),
                "projection_years",
                id="projection_years-fraction",
            ),
            pytest.param({"inflation_rate": -1.0}, (TypeError, ValueError), "inflation_rate", id="inflation_rate"),
            pytest.param({"discount_rate": -1.0}, (TypeError, ValueError), "discount_rate", id="discount_rate"),
            pytest.param({"sell_price_inflation": 1.0}, ValueError, "sell_price_inflation", id="sell_price_inflation"),
            pytest.param({"pv_degradation_rate": 1.5}, ValueError, "pv_degradation_rate", id="pv_degradation_rate"),
            pytest.param({"pv_loss_overrides": {"shading": 200}}, ValueError, "pv_loss_overrides", id="loss-value"),
            pytest.param({"pv_loss_overrides": 5.0}, TypeError, "pv_loss_overrides", id="loss-type"),
            pytest.param(
                {"battery_min_soc": 0.9, "battery_max_soc": 0.2},
                ValueError,
                "battery_min_soc",
                id="battery-soc-window",
            ),
            pytest.param({"battery_rte": 1.5}, ValueError, "battery_rte", id="battery_rte"),
            pytest.param({"battery_eol_percentage": 0.0}, ValueError, "battery_eol_percentage", id="battery_eol"),
            pytest.param(
                {"battery_max_charge_power_w": -1.0},
                (TypeError, ValueError),
                "battery_max_charge_power_w",
                id="battery_max_charge_power_w",
            ),
            pytest.param(
                {"battery_max_discharge_power_w": math.inf},
                (TypeError, ValueError),
                "battery_max_discharge_power_w",
                id="battery_max_discharge_power_w",
            ),
            pytest.param(
                {"battery_temperature": math.inf},
                (TypeError, ValueError),
                "battery_temperature",
                id="battery_temperature",
            ),
            pytest.param({"battery_indoor_model": False}, TypeError, "battery_indoor_model", id="indoor-model-type"),
            pytest.param(
                {"battery_indoor_model": {"coupling_alpha": 1.1}},
                ValueError,
                "coupling_alpha",
                id="indoor-model-coupling_alpha",
            ),
            # A typo such as `batery_kwh` must fail loudly instead of being silently
            # dropped by merge_defaults (which would default the battery to 0).
            pytest.param({"batery_kwh": 5.0}, ValueError, "Unknown config key.*batery_kwh", id="unknown-key"),
            pytest.param({"degradation_engine": "magic"}, ValueError, "degradation_engine", id="degradation_engine"),
            *[
                pytest.param(
                    {"battery_kwh": 5.0, "blast_model": "lfp_gr_250ah_prismatic", **engine},
                    ValueError,
                    "blast_model.*requires.*degradation_engine=blast",
                    id=f"blast_model-without-blast-engine-{name}",
                )
                for name, engine in (("unset", {}), ("native", {"degradation_engine": "native"}))
            ],
            pytest.param(
                {
                    "battery_kwh": 5.0,
                    "degradation_engine": "blast",
                    "blast_model": "lfp_gr_250ah_prismatic",
                    "montecarlo": {"n_runs": 10, "weather_file": "weather.csv"},
                },
                ValueError,
                "Monte Carlo",
                id="blast-montecarlo",
            ),
            pytest.param(
                {
                    "battery_kwh": 5.0,
                    "degradation_engine": "blast",
                    "blast_model": "lfp_gr_250ah_prismatic",
                    "enable_resistance_fade": True,
                },
                ValueError,
                "enable_resistance_fade",
                id="blast-resistance-fade",
            ),
        ],
    )
    def test_invalid_config_fails_at_app_boundary(self, overrides, error, match):
        with pytest.raises(error, match=match):
            App({**self.BASE, **overrides})

    @pytest.mark.parametrize("start_date", ["2025-07-01", "2025-01-15", "2025-12-31"])
    def test_start_date_other_than_1_january_fails_early(self, start_date):
        """Weather starts on 1 January, so a later start used to misplace the load.

        The load profile's first row is 1 January. A July start stamped January's
        winter demand onto July, against weather that still began in January.
        """
        with pytest.raises(ValueError, match=r"'start_date' must be 1 January .* use '2025-01-01'"):
            App({**self.BASE, "start_date": start_date})

    @pytest.mark.parametrize(
        ("overrides", "error_type", "message"),
        [
            (
                {"location": {"latitude": 91, "longitude": 0, "timezone": "UTC"}},
                ValueError,
                "'location.latitude' must be between -90 and 90",
            ),
            ({"pv_arrays": [{"modules": 0}]}, ValueError, "'pv_arrays[0].modules' must be >= 1"),
            (
                {"inverter_efficiency": 0},
                ValueError,
                "'inverter_efficiency' must be between 0 (exclusive) and 1 (inclusive)",
            ),
            ({"resolution": "30min"}, ValueError, "'resolution' must be 'h' or '15min'"),
            ({"discount_rate": -1}, ValueError, "'discount_rate' must be greater than -1"),
            (
                {"battery_min_soc": 0.9, "battery_max_soc": 0.2},
                ValueError,
                "'battery_min_soc' and 'battery_max_soc' must satisfy 0 <= min < max <= 1",
            ),
        ],
    )
    def test_validation_subsystems_preserve_public_error_contract(self, overrides, error_type, message):
        """Characterize exact public errors at each validation subsystem boundary."""
        cfg = merge_defaults({**self.BASE, **overrides})

        with pytest.raises(error_type) as exc_info:
            validate_config(cfg)

        assert str(exc_info.value) == message

    def test_validation_preserves_blast_normalization_and_conflict_errors(self):
        cfg = merge_defaults(
            {**self.BASE, "degradation_engine": " BLAST ", "blast_model": "lfp_gr_250ah_prismatic", "battery_kwh": 5.0}
        )

        validate_config(cfg)

        assert cfg["degradation_engine"] == "blast"

        invalid = merge_defaults(
            {
                **self.BASE,
                "degradation_engine": "blast",
                "blast_model": "lfp_gr_250ah_prismatic",
                "battery_kwh": 5.0,
                "montecarlo": {},
            }
        )
        with pytest.raises(ValueError) as exc_info:
            validate_config(invalid)
        assert str(exc_info.value) == "'degradation_engine=blast' is not supported with Monte Carlo yet"

    def test_battery_temperature_and_indoor_model_are_accepted(self):
        app = App({**self.BASE, "battery_temperature": 25.0, "battery_indoor_model": {"enabled": False}})

        assert app._cfg["battery_temperature"] == 25.0
        assert app._cfg["battery_indoor_model"] == {"enabled": False}

    def test_cost_overrides_resolve_through_app_facade(self):
        app = App(
            {
                "location": "porto",
                "n_modules": 10,
                "annual_consumption_kwh": 4000,
                "cost_preset": "residential_pt",
                "costs": {
                    "electricity_cost": 0.22,
                    "storage_cost_per_kwh": 425.0,
                    "land_cost": 1000.0,
                },
            }
        )

        assert app._resolved.cost_params.electricity_cost == 0.22
        assert app._resolved.cost_params.battery_cost_per_kwh == 425.0
        assert app._resolved.cost_params.land_cost == 1000.0
        assert app._resolved.cost_params.module_cost_per_w == 0.125

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

    def test_blast_config_accepts_enabled_p1_models(self):
        for blast_model in ("lfp_gr_250ah_prismatic", "nca_gr_panasonic_3ah"):
            app = App(
                {
                    "location": "porto",
                    "n_modules": 10,
                    "annual_consumption_kwh": 4000,
                    "battery_kwh": 5.0,
                    "degradation_engine": "blast",
                    "blast_model": blast_model,
                }
            )
            assert app._cfg["degradation_engine"] == "blast"
            assert app._cfg["blast_model"] == blast_model

    def test_blast_config_enables_phase3_models(self):
        app = App(
            {
                "location": "porto",
                "n_modules": 10,
                "annual_consumption_kwh": 4000,
                "battery_kwh": 5.0,
                "degradation_engine": "blast",
                "blast_model": "nmc811_grsi_lgm50_5ah",
            }
        )
        assert app._cfg["blast_model"] == "nmc811_grsi_lgm50_5ah"

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

    def test_iam_model_reaches_simulation(self, _patch_weather):
        def _run(model):
            app = App(
                {
                    "location": "porto",
                    "n_modules": 6,
                    "annual_consumption_kwh": 3000,
                    "projection_years": 1,
                    "iam_model": model,
                }
            )
            app.simulate()
            return app.result()["pv_production_kwh"]

        assert _run("physical") != pytest.approx(_run("ashrae"))

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

        # Removing a DC loss increases production. The AC increase is not a
        # fixed 1 / 0.97 ratio because inverter efficiency varies with load.
        assert no_shading > base

    def test_horizon_profile_reduces_generation_and_is_serialized(self, _patch_weather):
        common = {
            "location": "porto",
            "n_modules": 6,
            "annual_consumption_kwh": 3000,
            "projection_years": 1,
        }
        baseline = App(common)
        baseline.simulate()
        horizon = App({**common, "horizon_profile": [[0, 90], [180, 90]]})
        horizon.simulate()

        result = horizon.result()
        assert result["pv_dc_generation_kwh"] < baseline.result()["pv_dc_generation_kwh"]
        horizon_provenance = result["provenance"]["weather"]["horizon"]
        assert horizon_provenance["status"] == "applied"
        assert horizon_provenance["provider"] == "breos"
        assert horizon_provenance["profile"]["points"] == [[0.0, 90.0], [180.0, 90.0]]
        assert horizon_provenance["profile"]["shaded_timesteps"] > 0
        json.dumps(result)

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


class TestGcrValidation:
    """``gcr`` is checked on every path, not only under a bifacial model.

    It drives pvlib's backtracking rotation as well as ``infinite_sheds`` rear
    view factors, and pvlib silently computes a different rotation rather than
    rejecting a nonsensical ratio — a mistyped ``3.5`` used to return roughly
    half the annual energy with no error anywhere.
    """

    BASE = {"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000}

    @pytest.mark.parametrize("gcr", [0, 0.0, -0.5, 1.2, 3.5, 5.0])
    @pytest.mark.parametrize("tracking", ["fixed", "single_axis", "dual_axis"])
    def test_out_of_range_gcr_rejected_without_a_bifacial_model(self, gcr, tracking):
        with pytest.raises(ValueError, match="'gcr' must be between 0 .exclusive. and 1 .inclusive."):
            App({**self.BASE, "gcr": gcr, "tracking": tracking})

    @pytest.mark.parametrize("gcr", ["x", None, float("nan"), float("inf")])
    def test_non_finite_gcr_rejected(self, gcr):
        with pytest.raises((ValueError, TypeError), match="'gcr' must be a finite number"):
            App({**self.BASE, "gcr": gcr})

    @pytest.mark.parametrize("gcr", [0.01, 0.35, 1, 1.0])
    def test_valid_gcr_accepted(self, gcr):
        App({**self.BASE, "gcr": gcr})

    def test_per_array_gcr_override_is_reported_against_its_own_key(self):
        with pytest.raises(ValueError, match=r"'pv_arrays\[1\].gcr' must be between"):
            App(
                {
                    "location": "porto",
                    "annual_consumption_kwh": 4000,
                    "pv_arrays": [{"modules": 4}, {"modules": 3, "gcr": 2.0}],
                }
            )

    def test_top_level_gcr_checked_even_when_every_array_overrides_it(self):
        """It is still the function-level default handed to the multi-array path."""
        with pytest.raises(ValueError, match="'gcr' must be between"):
            App(
                {
                    "location": "porto",
                    "annual_consumption_kwh": 4000,
                    "gcr": 4.0,
                    "pv_arrays": [{"modules": 4, "gcr": 0.4}],
                }
            )

    @pytest.mark.parametrize(
        ("extra", "expected"),
        [
            ({"resolution": "weekly"}, "resolution"),
            ({"inverter_efficiency": 2.0}, "inverter_efficiency"),
            ({"transposition_model": "nope"}, "transposition_model"),
            ({"tilt": 120}, "tilt"),
            ({"bifacial_model": "nope"}, "bifacial_model"),
        ],
    )
    def test_gcr_check_never_steals_a_pre_existing_error(self, extra, expected):
        """The check runs last, so it is purely additive.

        A config that was already rejected for some other key must keep
        reporting that key rather than switching to gcr.
        """
        with pytest.raises((ValueError, TypeError), match=expected):
            App({**self.BASE, **extra, "gcr": 3.5})


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
            "pv_loss_waterfall",
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

    def test_provenance_records_the_pv_only_dispatch_path(self):
        assert self.result["provenance"]["execution"]["dispatch_path"] == "pv_only_vectorized"

    def test_pv_production_positive(self):
        assert self.result["pv_production_kwh"] > 0

    def test_pv_loss_waterfall_reconciles_to_reported_pv(self):
        waterfall = self.result["pv_loss_waterfall"]
        assert waterfall["basis"] == "year_1"
        assert waterfall["unit"] == "kWh"
        balance = waterfall["energy_balance"]
        assert balance["pv_dc"]["generation_kwh"] == pytest.approx(self.result["pv_dc_generation_kwh"], abs=0.02)
        assert balance["pv_dc"]["residual_kwh"] == pytest.approx(0.0, abs=1e-5)
        assert balance["ac_delivery"]["usable_system_production_kwh"] == pytest.approx(
            self.result["usable_ac_system_production_kwh"], abs=0.02
        )
        assert waterfall["pvwatts"]["components_pct"]["shading"] == 3.0
        assert waterfall["pvwatts"]["combined_kwh"] > 0
        assert waterfall["inverter"]["conversion_loss_kwh"] > 0
        assert waterfall["bifacial"]["enabled"] is False
        assert waterfall["bifacial"]["model"] == "none"
        assert waterfall["bifacial"]["rear_gain_effective_dc_kwh"] == 0.0
        rear_stage = next(stage for stage in waterfall["stages"] if stage["key"] == "bifacial_rear_gain")
        assert rear_stage["delta_kwh"] == 0.0

    def test_grid_independence_range(self):
        gi = self.result["grid_independence_pct"]
        assert 0 <= gi <= 100

    def test_energy_conservation(self):
        r = self.result
        # self_consumption + export should approximately equal PV production
        assert abs(r["self_consumption_kwh"] + r["grid_export_kwh"] - r["usable_ac_system_production_kwh"]) < 1.0

    def test_explicit_production_schema_and_provenance(self):
        r = self.result
        assert r["self_consumption_kwh"] == pytest.approx(
            r["direct_pv_ac_load_kwh"] + r["pv_origin_battery_ac_load_kwh"], abs=0.02
        )
        assert r["usable_ac_system_production_kwh"] == pytest.approx(
            r["self_consumption_kwh"] + r["grid_export_kwh"], abs=0.02
        )
        assert r["provenance"]["ledger_schema_version"] == "1.1"
        assert r["provenance"]["timezone"] == "Europe/Lisbon"
        json.dumps(r["provenance"])

    def test_blast_result_reports_model_identity_and_state_provenance(self):
        app = App(
            {
                "location": "porto",
                "n_modules": 6,
                "annual_consumption_kwh": 3000,
                "battery_kwh": 5.0,
                "projection_years": 1,
                "degradation_engine": "blast",
                "blast_model": "lfp_gr_250ah_prismatic",
            }
        )
        app.simulate()
        result = app.result()

        degradation = result["degradation"]
        assert degradation["engine"] == "blast"
        assert degradation["model_key"] == "lfp_gr_250ah_prismatic"
        assert degradation["model_profile"]["key"] == degradation["model_key"]
        assert degradation["model_profile"]["upstream"]["commit"] == "d789e00bca60f628de640745c18eb724b07358bd"
        assert degradation["model_profile"]["calibration_basis"] == "cell-model"
        assert degradation["pack_calibrated"] is False
        assert degradation["initial_soh_pct"] == 100.0
        assert degradation["final_soh_pct"] == round(result["battery_soh_end_pct"], 1)
        assert degradation["state_schema_version"] == "1.0"
        range_warnings = degradation["experimental_range_warnings"]
        assert range_warnings
        assert len({warning["code"] for warning in range_warnings}) == len(range_warnings)
        assert degradation["aging_horizon_extrapolation_warnings"] == []
        assert result["provenance"]["degradation"] == degradation
        assert result["provenance"]["degradation"] is degradation

    def test_investment_positive(self):
        assert self.result["total_investment_eur"] > 0

    def test_lcoe_positive(self):
        assert self.result["lcoe_eur_kwh"] > 0


def test_monthly_rows_follow_the_local_year_of_fixed_offset_weather(monkeypatch, synthetic_weather):
    # PVGIS weather for a zone east of UTC arrives as Etc/GMT-N, so the local
    # year starts on 31 December in UTC. Monthly rows must still be the twelve
    # local months and add up to the first simulated year.
    weather = synthetic_weather.copy()
    weather.index = pd.date_range("2023-01-01", periods=len(weather), freq="h", tz="Etc/GMT-1")

    def _fake_fetch(*args, **kwargs):
        return weather.copy(), {"inputs": {"location": {"latitude": 52.52, "longitude": 13.40, "elevation": 0}}}

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", _fake_fetch)
    monkeypatch.setattr("breos.app.load_weather", lambda **kw: None)
    app = App(
        {
            "location": {"latitude": 52.52, "longitude": 13.40, "timezone": "Europe/Berlin"},
            "n_modules": 6,
            "annual_consumption_kwh": 3000,
            "projection_years": 2,
        }
    )
    app.simulate()
    result = app.result()

    monthly = result["monthly"]
    assert [row["month"] for row in monthly] == [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]  # fmt: skip
    year_one = result["yearly"][0]
    for key in ("pv_kwh", "consumption_kwh", "import_kwh", "export_kwh"):
        assert sum(row[key] for row in monthly) == pytest.approx(year_one[key], abs=0.1)


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


class TestAppBifacialConfig:
    def _config(self, **overrides):
        return {
            "location": "porto",
            "n_modules": 6,
            "annual_consumption_kwh": 3000,
            "pv_module": "Generic_600W_Bifacial",
            **overrides,
        }

    def test_front_only_default_needs_no_row_geometry(self):
        app = App(self._config())

        assert app._cfg["bifacial_model"] == "none"
        assert app._cfg["pvrow_height"] is None
        assert app._cfg["pvrow_pitch"] is None

    def test_infinite_sheds_requires_bifacial_module(self):
        with pytest.raises(ValueError, match="requires bifaciality metadata"):
            App(
                self._config(
                    pv_module="Generic_400W",
                    bifacial_model="infinite_sheds",
                    pvrow_height=1.5,
                    pvrow_pitch=6.0,
                )
            )

    @pytest.mark.parametrize("missing", ["pvrow_height", "pvrow_pitch"])
    def test_infinite_sheds_requires_complete_geometry(self, missing):
        geometry = {"pvrow_height": 1.5, "pvrow_pitch": 6.0}
        geometry.pop(missing)

        with pytest.raises(ValueError, match=missing):
            App(self._config(bifacial_model="infinite_sheds", **geometry))

    def test_per_array_bifacial_override_is_normalized(self):
        app = App(
            {
                "location": "porto",
                "annual_consumption_kwh": 3000,
                "pv_arrays": [
                    {
                        "modules": 6,
                        "module": "Generic_600W_Bifacial",
                        "tilt": 25,
                        "azimuth": 180,
                        "bifacial_model": "infinite_sheds",
                        "gcr": 0.35,
                        "pvrow_height": 1.5,
                        "pvrow_pitch": 6.0,
                    }
                ],
            }
        )

        assert app._cfg["pv_arrays"][0]["bifacial_model"] == "infinite_sheds"
        assert app._cfg["pv_arrays"][0]["pvrow_height"] == 1.5

    def test_infinite_sheds_runs_through_app_and_adds_generation(self, _patch_weather):
        common = self._config(projection_years=1, albedo=0.3)
        front_only = App(common)
        front_only.simulate()
        bifacial = App(
            {
                **common,
                "bifacial_model": "infinite_sheds",
                "gcr": 0.35,
                "pvrow_height": 1.5,
                "pvrow_pitch": 6.0,
            }
        )
        bifacial.simulate()

        result = bifacial.result()
        assert result["pv_dc_generation_kwh"] > front_only.result()["pv_dc_generation_kwh"]
        summary = result["pv_loss_waterfall"]["bifacial"]
        assert summary["enabled"] is True
        assert summary["model"] == "infinite_sheds"
        assert summary["rear_gain_effective_dc_kwh"] > 0
        assert summary["rear_gain_pct_of_front_effective"] > 0
        assert summary["arrays"] == [
            {
                "array_index": 0,
                "modules": 6,
                "module": "Generic_600W_Bifacial",
                "model": "infinite_sheds",
                "bifaciality": 0.7,
                "gcr": 0.35,
                "pvrow_height": 1.5,
                "pvrow_pitch": 6.0,
            }
        ]
        rear_stage = next(
            stage for stage in result["pv_loss_waterfall"]["stages"] if stage["key"] == "bifacial_rear_gain"
        )
        assert rear_stage["delta_kwh"] == pytest.approx(summary["rear_gain_effective_dc_kwh"], abs=0.01)
        assert result["provenance"]["pv_model"]["bifacial"] == summary


class TestAppSimulateTracking:
    def test_invalid_tracking(self, _patch_weather):
        with pytest.raises(ValueError, match="'tracking' must be"):
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

    def test_loss_waterfall_reports_battery_dispatch_losses(self):
        dispatch = self.result["pv_loss_waterfall"]["dispatch"]
        inverter = self.result["pv_loss_waterfall"]["inverter"]
        assert dispatch["battery_round_trip_loss_kwh"] >= 0
        assert dispatch["battery_round_trip_loss_kwh"] == pytest.approx(
            dispatch["battery_charge_loss_kwh"] + dispatch["battery_discharge_loss_kwh"],
            abs=0.01,
        )
        stored = self.result["pv_loss_waterfall"]["energy_balance"]["battery_stored_energy"]
        assert stored["residual_kwh"] == pytest.approx(0.0, abs=1e-5)
        assert inverter["conversion_loss_kwh"] == pytest.approx(
            inverter["direct_pv_conversion_loss_kwh"] + inverter["battery_discharge_conversion_loss_kwh"],
            abs=0.02,
        )
        assert dispatch["stored_energy_report"] == "energy_balance.battery_stored_energy"

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


def test_multiyear_battery_inventory_and_pv_origin_cross_year_boundary(_patch_weather, monkeypatch):
    import breos.runners.app as runner_module

    original = runner_module.simulate_energy_balance
    calls = []

    def _capture(*args, **kwargs):
        output = original(*args, **kwargs)
        results = output[0]
        calls.append(
            {
                "initial_energy_wh": kwargs.get("initial_energy_wh"),
                "initial_pv_origin_energy_wh": kwargs.get("initial_pv_origin_energy_wh"),
                "ending_energy_wh": float(results["Battery_Energy_End"].iloc[-1]),
                "ending_pv_origin_energy_wh": float(results["Battery_PV_Origin_Energy_End"].iloc[-1]),
                "beginning_energy_wh": float(results["Battery_Energy_Beginning"].iloc[0]),
                "beginning_pv_origin_energy_wh": float(results["Battery_PV_Origin_Energy_Beginning"].iloc[0]),
            }
        )
        return output

    monkeypatch.setattr(runner_module, "simulate_energy_balance", _capture)
    app = App(
        {
            "location": "porto",
            "n_modules": 6,
            "annual_consumption_kwh": 3000,
            "battery_kwh": 5.0,
            "emissions_country": "PT",
            "projection_years": 2,
        }
    )
    app.simulate()
    result = app.result()

    assert calls[0]["initial_energy_wh"] is None
    assert calls[0]["initial_pv_origin_energy_wh"] is None
    assert calls[0]["ending_pv_origin_energy_wh"] > 0
    assert calls[1]["initial_energy_wh"] == pytest.approx(calls[0]["ending_energy_wh"])
    assert calls[1]["initial_pv_origin_energy_wh"] == pytest.approx(calls[0]["ending_pv_origin_energy_wh"])
    assert calls[1]["beginning_energy_wh"] == pytest.approx(calls[0]["ending_energy_wh"])
    assert calls[1]["beginning_pv_origin_energy_wh"] == pytest.approx(calls[0]["ending_pv_origin_energy_wh"])
    year2 = result["yearly"][1]
    assert year2["pv_origin_battery_ac_load_kwh"] > 0
    assert year2["self_consumption_kwh"] == pytest.approx(
        year2["direct_pv_ac_load_kwh"] + year2["pv_origin_battery_ac_load_kwh"], abs=0.02
    )
    assert year2["usable_ac_system_production_kwh"] == pytest.approx(
        year2["self_consumption_kwh"] + year2["export_kwh"], abs=0.02
    )
