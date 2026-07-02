"""Optimizer/App parity guardrails (2026-07 audit follow-up).

The NSGA-II optimizer must score candidate designs with the same model the
App reports for the winning design:

- financials: ``calculate_financials`` mirrors the year-1-estimation formulas
  of ``economics.cost_analysis_projection`` — enforced here by direct
  numerical comparison, so the two cannot drift apart silently;
- inverter: candidates are simulated with the AC nameplate their CAPEX pays
  for (``pv_peak / dc_ac_ratio``), i.e. clipping applies during scoring;
- load alignment: the raw load frame reaches ``simulate_energy_balance``
  unmangled, so its timezone-aware alignment applies (no positional
  re-stamping).
"""

import numpy as np
import pandas as pd
import pytest

from breos.economics import calculate_costs, cost_analysis_projection, cost_params_from_config
from breos.optimization import calculate_financials

COSTS_CONFIG = {
    "panel_wp": 400,
    "electricity_cost": 0.25,
    "electricity_sold_cost": 0.07,
    "daily_power_cost": 0.50,  # cancels out of savings; nonzero to prove it
    "module_cost_per_w": 0.15,
    "storage_cost_per_kwh": 400.0,
    "dc_ac_ratio": 1.25,
    "installation_cost_per_module": 300.0,
    "maintenance_cost_per_panel": 12.0,
    "maintenance_cost": 30.0,
}
FINANCIALS_CONFIG = {
    "inflation_rate": 0.03,
    "sell_price_inflation": 0.015,
    "discount_rate": 0.04,
    "pv_degradation_rate": 0.007,
    "project_lifespan": 20,
}


def _first_year_results_df():
    """Synthetic constant-power first-year results (hourly, 2023, UTC)."""
    idx = pd.date_range("2023-01-01 00:00", periods=8760, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "PV_Production": 1000.0,  # W -> 8760 kWh/yr
            "Houseload": 500.0,
            "Import_From_Grid": 200.0,
            "Sell_To_Grid": 300.0,
        },
        index=idx,
    )


def test_calculate_financials_matches_projection_engine():
    results_df = _first_year_results_df()
    pv_kwh = 8760.0
    load_kwh = 500.0 * 8760 / 1000
    import_kwh = 200.0 * 8760 / 1000
    export_kwh = 300.0 * 8760 / 1000
    n_modules, battery_kwh = 10, 5.0

    cost_params = cost_params_from_config(COSTS_CONFIG, FINANCIALS_CONFIG)
    costs = calculate_costs(
        n_modules=n_modules,
        module_power_w=COSTS_CONFIG["panel_wp"],
        battery_capacity_wh=battery_kwh * 1000,
        cost_params=cost_params,
    )
    projection = cost_analysis_projection(
        results_df=results_df,
        costs=costs,
        num_years=FINANCIALS_CONFIG["project_lifespan"],
        inflation_rate=FINANCIALS_CONFIG["inflation_rate"],
        sell_price_inflation=FINANCIALS_CONFIG["sell_price_inflation"],
        discount_rate=FINANCIALS_CONFIG["discount_rate"],
        degradation_rate=FINANCIALS_CONFIG["pv_degradation_rate"],
        freq="h",
    )
    expected_npv = float(projection["Savings_Cumulative_NPV"].iloc[-1])

    capex, npv = calculate_financials(
        n_modules,
        battery_kwh,
        import_kwh,
        export_kwh,
        load_kwh,
        costs_config=COSTS_CONFIG,
        financials_config=FINANCIALS_CONFIG,
        annual_pv_kwh=pv_kwh,
    )

    assert capex == pytest.approx(costs["total_initial_cost"])
    assert npv == pytest.approx(expected_npv, rel=1e-9)


def test_calculate_financials_flat_fallback_without_pv():
    # Without annual_pv_kwh degradation cannot be apportioned; year-1 flows
    # are held flat (documented pre-0.3.4 behaviour), which yields a higher
    # NPV than the degradation-aware estimate.
    kwargs = dict(costs_config=COSTS_CONFIG, financials_config=FINANCIALS_CONFIG)
    _, npv_flat = calculate_financials(10, 5.0, 1752.0, 2628.0, 4380.0, **kwargs)
    _, npv_degraded = calculate_financials(10, 5.0, 1752.0, 2628.0, 4380.0, annual_pv_kwh=8760.0, **kwargs)
    assert npv_flat > npv_degraded


# ---------------------------------------------------------------------------
# SolarDesignProblem wiring (requires pymoo)
# ---------------------------------------------------------------------------


def _problem_config(dc_ac_ratio: float = 1.6):
    return {
        "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
        "simulation": {"resolution": "h"},
        "constraints": {"budget_eur": 100000, "max_area_m2": 100.0, "max_modules": 5},
        "mode": {"fixed_azimuth": 180},
        "pv": {"module": "Suntech_STP550S_STC"},
        "battery": {"temperature": 20.0, "indoor_model": {"enabled": False}},
        "costs": dict(COSTS_CONFIG, dc_ac_ratio=dc_ac_ratio),
        "financials": FINANCIALS_CONFIG,
    }


def _run_evaluate(monkeypatch, config, houseload, tmy_index):
    pytest.importorskip("pymoo")
    from breos.optimization import SolarDesignProblem

    tmy_data = pd.DataFrame({"temp_air": 15.0, "ghi": 0.0}, index=tmy_index)
    dc = pd.Series(0.0, index=tmy_index)
    summary = pd.DataFrame({"Import [kWh]": [1.0], "Sell [kWh]": [0.0]})
    captured: dict = {}

    def fake_balance(**kwargs):
        captured["battery_config"] = kwargs["battery_config"]
        captured["houseload"] = kwargs["houseload"]
        return pd.DataFrame(), 0.0, summary, 0.0, 0, pd.DataFrame()

    def fake_financials(*args, **kwargs):
        captured["financials_kwargs"] = kwargs
        return 0.0, 0.0

    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **kwargs: dc)
    monkeypatch.setattr("breos.optimization.simulate_energy_balance", fake_balance)
    monkeypatch.setattr("breos.optimization.calculate_financials", fake_financials)

    problem = SolarDesignProblem(tmy_data, houseload, config, "results/_test_run/parity")
    out: dict = {}
    problem._evaluate(np.array([2.0, 1.0, 10.0], dtype=float), out)
    return captured


def test_optimizer_applies_capex_matched_ac_clipping(monkeypatch):
    idx = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
    captured = _run_evaluate(monkeypatch, _problem_config(dc_ac_ratio=1.6), houseload, idx)

    # 2 modules x 550 Wp / 1.6 — the same nameplate the CAPEX pays for
    assert captured["battery_config"].inverter_ac_capacity_w == pytest.approx(2 * 550 / 1.6)
    # and the financials receive the PV energy for degradation apportioning
    assert "annual_pv_kwh" in captured["financials_kwargs"]


def test_optimizer_passes_load_with_original_timestamps(monkeypatch):
    tmy_index = pd.date_range("2025-01-01 00:00", periods=2, freq="h", tz="UTC")
    load_index = pd.date_range("2024-01-01 00:00", periods=2, freq="h", tz="Europe/Lisbon")
    houseload = pd.DataFrame({"Load": [500.0, 500.0]}, index=load_index)
    captured = _run_evaluate(monkeypatch, _problem_config(), houseload, tmy_index)

    # The load must reach simulate_energy_balance with its real timestamps —
    # its internal alignment (UTC instants, year remap) does the rest.
    assert captured["houseload"].index.equals(load_index)
