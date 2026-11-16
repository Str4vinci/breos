"""ADR 0005 PR 5: paired draws, unchanged streams and App valuation."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import breos.montecarlo as mc
import breos.projection as projection
from breos import App
from breos.app_config import resolve_app_config
from tests.backend_parity import BACKENDS
from tests.offline_app import CREDIT, TOU, TOU_REFERENCE
from tests.test_montecarlo import _base_config
from tests.test_multi_inverter import TWO_INVERTERS
from tests.test_vehicle_config import AWAY, VEHICLE


def config(**extra):
    return {
        **_base_config(),
        "start_date": "2025-01-01",
        "projection_years": 2,
        "vehicle": {
            **VEHICLE,
            "battery_kwh": 40,
            "eol_percentage": 0.98,
            "allow_terminal_replacement": False,
            "outside_charging_price": 0.6,
            "away_charging": {**AWAY, "price": 0.3},
        },
        **extra,
    }


@pytest.fixture
def weather(tmp_path, write_multiyear_weather):
    return write_multiyear_weather(tmp_path / "one.csv", years=(2021,))


def settings(weather, **extra):
    return mc.MonteCarloSettings(
        weather_file=str(weather), n_runs=1, years_per_run=2, load_uncertainty=0, seed=42, collect_yearly=True, **extra
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("pack", ["none", "dc", "ac", "flat"])
def test_no_noise_trajectory_matches_all_app_households(weather, monkeypatch, backend, freq, pack):
    cfg = config(
        resolution=freq,
        execution_backend=backend,
        battery_kwh=0 if pack == "none" else 5,
        battery_temperature=35.0,
        replacement_cost_learning=0.01,
        tariff={**TOU, "annual_network_credit": CREDIT},
        reference_tariff={**TOU_REFERENCE, "annual_network_credit": CREDIT},
    )
    if pack == "ac":
        cfg.update(
            battery_coupling="ac",
            battery_inverter_ac_rating_kw=2.5,
            battery_inverter_efficiency=0.94,
            costs={"battery_inverter_cost_per_kw": 120},
        )
    if pack not in {"none", "flat"}:
        cfg["smart_charging"] = {
            "mode": "fixed_target",
            "charge_periods": ["off_peak"],
            "discharge_periods": ["peak"],
            "target_usable_fraction": 0.6,
            "grid_import_limit_w": 5000,
        }
        if pack == "dc":
            cfg["smart_charging"]["grid_charge_efficiency"] = 0.95
    cfg["vehicle"].update(charge_periods=["off_peak"], discharge_periods=["peak"])
    del cfg["vehicle"]["outside_charging_price"]  # Exercise per-absence comparability in both entry points.
    if pack == "flat":
        del cfg["tariff"], cfg["reference_tariff"]
        del cfg["vehicle"]["charge_periods"], cfg["vehicle"]["discharge_periods"]
    captured = []
    original = mc.value_vehicle_households

    def value(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result[1])
        return result

    monkeypatch.setattr(mc, "value_vehicle_households", value)
    resolved = resolve_app_config(cfg)
    series = mc._load_weather_years(resolved.cfg, resolved, settings(weather))[2021]
    monkeypatch.setattr("breos.app.load_weather", lambda **kwargs: series.copy())
    app = App(cfg)
    app.simulate()
    result = mc.run_montecarlo(cfg, settings(weather))
    assert captured[0] == app.result()["vehicle"]
    row = result.runs.iloc[0]
    assert row["npv_savings"] == app._artifacts.cost_projection.attrs["final_npv_savings"]
    assert row["npv_savings_household_adjusted"] == captured[0]["household_adjusted_npv"]["npv"]
    assert row["vehicle_benefit_npv"] == captured[0]["vehicle_benefit"]["npv"]
    for key, household in captured[0]["households"].items():
        prefix = f"vehicle_{key.lower()}_"
        for name, amount in household["totals"].items():
            assert row[prefix + name] == amount
        assert row[prefix + "replacement_outlays_npv"] == household["replacement_outlays_npv"]
        assert row[prefix + "terminal_pack_credit_npv"] == household["terminal_pack_credit_npv"]
    assert result.provenance["vehicle"]["schedule_hash"] == app.result()["provenance"]["vehicle"]["schedule_hash"]


def test_sampling_once_per_year_and_pairing_across_households(tmp_path, write_multiyear_weather, monkeypatch):
    path = write_multiyear_weather(tmp_path / "multi.csv")
    options = mc.MonteCarloSettings(
        weather_file=str(path),
        n_runs=2,
        years_per_run=3,
        load_uncertainty=0.2,
        model_uncertainty_pct=5,
        seed=7,
        collect_yearly=True,
    )
    seen, scales, factors = [], [], []
    original = projection._run_frame_year

    def frame(year, year_idx, common, *args, **kwargs):
        seen.append((common["vehicle_config"].mode, year_idx, year))
        return original(year, year_idx, common, *args, **kwargs)

    sample_scale, sample_factor = mc._sample_load_scale, mc._sample_pv_model_factor

    def scale(*args, **kwargs):
        result = sample_scale(*args, **kwargs)
        scales.append(result)
        return result

    def factor(*args, **kwargs):
        result = sample_factor(*args, **kwargs)
        factors.append(result)
        return result

    monkeypatch.setattr(projection, "_run_frame_year", frame)
    monkeypatch.setattr(mc, "_sample_load_scale", scale)
    monkeypatch.setattr(mc, "_sample_pv_model_factor", factor)
    result = mc.run_montecarlo(config(), options)
    assert len(scales) == 6 and len(factors) == 2
    assert len(seen) == 18
    for run in range(2):
        chunk = seen[run * 9 : (run + 1) * 9]
        for year_idx in range(3):
            years = {mode: year for mode, idx, year in chunk if idx == year_idx}
            assert set(years) == {"v2h", "v1g", "uncontrolled"}
            study = years["v2h"]
            for year in years.values():
                assert year.houseload is study.houseload
                assert year.vehicle_schedule is study.vehicle_schedule
                assert year.vehicle_temperature_series is study.vehicle_temperature_series
                assert year.extra is study.extra
            assert years["v1g"].pv_dc is study.pv_dc
            assert (years["uncontrolled"].pv_dc == 0.0).all()
    assert result.runs["pv_model_factor"].tolist() == factors


@pytest.mark.parametrize("seed", [7, 8])
def test_vehicle_leaves_system_sampling_and_stationary_state_unchanged(tmp_path, write_multiyear_weather, seed):
    path = write_multiyear_weather(tmp_path / "multi.csv")
    options = mc.MonteCarloSettings(
        weather_file=str(path),
        n_runs=2,
        years_per_run=2,
        load_uncertainty=0.2,
        model_uncertainty_pct=5,
        seed=seed,
        collect_yearly=True,
    )
    cfg = config()
    plain = mc.run_montecarlo({k: v for k, v in cfg.items() if k != "vehicle"}, options)
    vehicle = mc.run_montecarlo(cfg, options)
    columns = [
        "run",
        "Year",
        "Weather_Year",
        "Load_Scale",
        "Load_kWh",
        "PV_DC_Generation_kWh",
        "Battery_SOH_%",
        "Replacements",
        "Grid_AC_To_Battery_kWh",
    ]
    pd.testing.assert_frame_equal(plain.yearly[columns], vehicle.yearly[columns], check_exact=True)
    np.testing.assert_array_equal(plain.runs["pv_model_factor"], vehicle.runs["pv_model_factor"])
    for run_idx in range(options.n_runs):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(run_idx,)))
        rows = vehicle.yearly[vehicle.yearly["run"] == run_idx + 1]
        for _, row in rows.iterrows():
            assert row["Weather_Year"] == vehicle.available_years[rng.integers(2)]
            assert row["Load_Scale"] == mc._sample_load_scale(rng, 0.2, 0.0, None)


def test_house_load_scale_never_scales_driving(weather):
    cfg = config(battery_kwh=0)
    results = [
        mc.run_montecarlo(cfg, settings(weather, min_load_scale=scale, max_load_scale=scale)) for scale in [0.5, 1.5]
    ]
    assert results[1].yearly["Load_kWh"].sum() == 3 * results[0].yearly["Load_kWh"].sum()
    for key in ("r0", "r1", "r2"):
        prefix = f"vehicle_{key}_"
        requested = [
            r.runs.iloc[0][prefix + "drive_served_kwh"] + r.runs.iloc[0][prefix + "drive_unserved_kwh"] for r in results
        ]
        assert requested[0] == pytest.approx(requested[1], abs=1e-9)


@pytest.mark.parametrize("reference", ["v1g", "none"])
def test_summary_metrics_and_optional_reference(weather, reference):
    result = mc.run_montecarlo(config(vehicle={**VEHICLE, "reference_mode": reference}), settings(weather))
    required = {
        "npv_savings_household_adjusted",
        "household_adjusted_comparable",
        "vehicle_benefit_npv",
        "vehicle_benefit_comparable",
        "vehicle_r0_departures_short",
        "vehicle_r2_drive_unserved_kwh",
        "vehicle_r2_away_purchased_kwh",
        "vehicle_r2_replacements",
        "vehicle_wear_r2_r0_npv",
    }
    assert required <= set(result.runs)
    assert "v2h_benefit_npv" not in result.runs
    assert not any(column.startswith("vehicle_v2h_") for column in result.runs)
    for column in result.runs:
        if column.startswith("vehicle_") or column in required:
            if pd.isna(result.runs.iloc[0][column]):
                assert column not in result.summary
            else:
                assert result.summary[column]["count"] == 1
                assert result.summary[column]["mean"] == result.runs.iloc[0][column]
    if reference == "none":
        assert pd.isna(result.runs.iloc[0]["vehicle_benefit_npv"])
        assert result.runs.iloc[0]["vehicle_benefit_comparable"] is None


def test_flag_summary_mean_is_probability():
    result = mc._summarize(
        pd.DataFrame({"household_adjusted_comparable": [True, False], "vehicle_benefit_comparable": [None, None]})
    )
    assert result["household_adjusted_comparable"]["mean"] == 0.5
    assert result["household_adjusted_comparable"]["count"] == 2
    assert "vehicle_benefit_comparable" not in result


@pytest.mark.parametrize("entry", [mc.run_montecarlo, mc.build_year_cache])
@pytest.mark.parametrize("reason", ["blast", "inverters"])
def test_kept_refusals_before_weather(entry, reason, tmp_path, monkeypatch, trina):
    cfg = config()
    if reason == "blast":
        cfg["vehicle"].update(degradation_engine="blast", blast_model="nmc_gr_50ah_b1")
        match = "vehicle.degradation_engine=blast.*Monte Carlo"
    else:
        cfg = {**TWO_INVERTERS, "vehicle": VEHICLE}
        match = "several.*inverters|inverters.*Monte Carlo"
    monkeypatch.setattr(mc, "_load_weather_years", lambda *a, **kw: pytest.fail("weather read before refusal"))
    with pytest.raises(ValueError, match=match):
        entry(cfg, settings(tmp_path / "missing.csv"))


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_backend_worker_and_year_cache_parity(weather, freq):
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    cfg = config(resolution=freq, battery_kwh=0)
    options = settings(weather)
    cache = mc.build_year_cache(cfg, options)
    python = mc.run_montecarlo(cfg, options)
    for backend, workers in [("python", 2), ("numba", 1)]:
        cached = mc.run_montecarlo(cfg, replace(options, execution_backend=backend, n_procs=workers), year_cache=cache)
        pd.testing.assert_frame_equal(python.runs, cached.runs, check_exact=True)
        pd.testing.assert_frame_equal(python.yearly, cached.yearly, check_exact=True)
        assert python.summary == cached.summary
    resolved = resolve_app_config(cfg)
    raw = mc._load_weather_years(resolved.cfg, resolved, options)
    dc, temp, inverter = mc._build_pv_years(resolved.cfg, resolved, raw)
    aligned = mc._align_years(
        resolved.cfg, pd.DataFrame({"Load": 100.0}, index=dc[2021].index), dc, temp, inverter, has_battery=False
    )
    assert mc._prepare_pv_chains(resolved.cfg, resolved, aligned, replace(options, n_runs=100), 2) is None
    for temperature in [5.0, 35.0]:
        varied = deepcopy(cfg)
        varied["vehicle"]["battery_temperature"] = temperature
        cached = mc.run_montecarlo(varied, options, year_cache=cache)
        fresh = mc.run_montecarlo(varied, options)
        pd.testing.assert_frame_equal(cached.runs, fresh.runs, check_exact=True)
