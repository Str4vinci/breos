"""Self-contained dispatch-parity scenarios and a bit-exact result dump.

The scenarios are synthetic so this file can run unchanged against an older
BREOS checkout: it touches only ``BatteryConfig`` and
``simulate_energy_balance``, which are stable across the revisions being
compared. Every scenario is deterministic given its name.

Run it against two trees and compare the ``.npz`` files with
``compare.py``; any differing bit in any exported column shows up as a
non-zero maximum absolute difference for that column.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

FREQ = "15min"
STEPS_PER_DAY = 96


def _index(days: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=days * STEPS_PER_DAY, freq=FREQ, tz="UTC")


def _profiles(index: pd.DatetimeIndex, seed: int, pv_peak_w: float, load_base_w: float):
    """Build a deterministic PV/load/temperature triple over *index*."""
    rng = np.random.default_rng(seed)
    n = len(index)
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    doy = index.dayofyear.to_numpy()

    # Diurnal PV with a seasonal envelope and per-day cloud variability.
    daylight = np.clip(np.sin((hour - 6.0) / 12.0 * np.pi), 0.0, None)
    seasonal = 0.65 + 0.35 * np.cos((doy - 172) / 365.0 * 2.0 * np.pi)
    cloud = np.repeat(rng.uniform(0.15, 1.0, size=n // STEPS_PER_DAY + 1), STEPS_PER_DAY)[:n]
    pv = pv_peak_w * daylight * seasonal * cloud
    pv[rng.random(n) < 0.004] = 0.0

    # Load with a morning and an evening peak plus noise.
    shape = 0.55 + 0.45 * np.exp(-(((hour - 7.5) / 1.6) ** 2)) + 0.9 * np.exp(-(((hour - 19.5) / 2.0) ** 2))
    load = load_base_w * shape * rng.uniform(0.8, 1.2, size=n)

    # Temperature sweeps below 0 C and above 25 C so every branch of the
    # capacity-derating curve is exercised.
    temp = 12.0 + 16.0 * np.cos((doy - 200) / 365.0 * 2.0 * np.pi) - 6.0 * np.cos(hour / 24.0 * 2.0 * np.pi)
    temp += rng.normal(0.0, 1.5, size=n)

    return (
        pd.Series(pv, index=index),
        pd.DataFrame({"Load": load}, index=index),
        pd.Series(temp, index=index),
    )


def build(name: str):
    """Return ``(pv_dc, houseload, temperature, battery_kwargs, sim_kwargs)``."""
    from breos.battery import BatteryConfig

    days = {"one_day": 1, "partial_day": 2}.get(name, 365)
    index = _index(days)
    if name == "partial_day":
        index = index[: STEPS_PER_DAY + 37]  # a full day plus a trailing stub

    common = dict(
        max_soc=0.95,
        min_soc=0.10,
        dc_coupled=True,
        inverter_efficiency=0.96,
        enable_replacement=True,
        calendar_model="naumann_lam",
        enable_resistance_fade=True,
    )

    if name == "no_battery":
        pv, load, temp = _profiles(index, 11, 8000.0, 1200.0)
        cfg = dict(nominal_energy_wh=0, inverter_efficiency=0.96, inverter_ac_capacity_w=6400.0)
        return pv, load, None, cfg, {}

    if name == "discharge_limited":
        # A hard AC discharge cap that binds most evenings, which is the one
        # dispatch branch no Article case reaches.
        pv, load, temp = _profiles(index, 12, 7000.0, 2200.0)
        cfg = dict(
            nominal_energy_wh=14000.0,
            inverter_ac_capacity_w=5600.0,
            max_charge_power_w=5000.0,
            max_discharge_power_w=900.0,
            **common,
        )
        return pv, load, temp, cfg, {}

    if name == "charge_limited":
        pv, load, temp = _profiles(index, 13, 12000.0, 900.0)
        cfg = dict(
            nominal_energy_wh=6000.0,
            inverter_ac_capacity_w=4000.0,
            max_charge_power_w=700.0,
            max_discharge_power_w=2500.0,
            **common,
        )
        return pv, load, temp, cfg, {}

    if name == "saturating":
        # Small pack against a large array: rides the upper and lower SOC
        # bounds every day and clips the inverter at midday.
        pv, load, temp = _profiles(index, 14, 15000.0, 700.0)
        cfg = dict(
            nominal_energy_wh=2500.0,
            inverter_ac_capacity_w=3600.0,
            max_charge_power_w=None,
            max_discharge_power_w=None,
            **common,
        )
        return pv, load, temp, cfg, {}

    if name == "replacement":
        # Aggressive cycling and a high end-of-life threshold so replacement
        # fires inside a single simulated year.
        pv, load, temp = _profiles(index, 15, 11000.0, 3500.0)
        cfg = dict(
            nominal_energy_wh=5000.0,
            inverter_ac_capacity_w=5000.0,
            max_charge_power_w=None,
            max_discharge_power_w=None,
            **{**common, "eol_percentage": 0.995},
        )
        return pv, load, temp, cfg, {}

    if name == "carried_state":
        pv, load, temp = _profiles(index, 16, 9000.0, 2000.0)
        cfg = dict(
            nominal_energy_wh=10000.0,
            inverter_ac_capacity_w=5000.0,
            max_charge_power_w=4000.0,
            max_discharge_power_w=3000.0,
            **common,
        )
        sim = dict(
            initial_energy_wh=6234.5678,
            initial_pv_origin_energy_wh=1234.5678,
            initial_fec=87.25,
            initial_calendar_seconds=1.5e7,
            initial_resistance_growth=0.031,
            initial_cumulative_cycle_deg=0.012,
            initial_cumulative_cal_deg=0.008,
        )
        return pv, load, temp, cfg, sim

    if name == "no_inverter_cap":
        # inverter_ac_capacity_w=None selects the flat-efficiency compatibility
        # formula for PV_Production, a separate arithmetic path.
        pv, load, temp = _profiles(index, 17, 9000.0, 2000.0)
        cfg = dict(
            nominal_energy_wh=8000.0,
            inverter_ac_capacity_w=None,
            max_charge_power_w=3000.0,
            max_discharge_power_w=3000.0,
            **common,
        )
        return pv, load, temp, cfg, {}

    # "baseline", "one_day", "partial_day"
    pv, load, temp = _profiles(index, 18, 9000.0, 2100.0)
    cfg = dict(
        nominal_energy_wh=10000.0,
        inverter_ac_capacity_w=5400.0,
        max_charge_power_w=4500.0,
        max_discharge_power_w=3500.0,
        **common,
    )
    return pv, load, temp, cfg, {}


SCENARIOS = (
    "baseline",
    "one_day",
    "partial_day",
    "no_battery",
    "discharge_limited",
    "charge_limited",
    "saturating",
    "replacement",
    "carried_state",
    "no_inverter_cap",
)


def run(name: str, backend: str = "python"):
    from breos.battery import BatteryConfig, simulate_energy_balance

    pv, load, temp, cfg, sim_kwargs = build(name)
    battery_config = BatteryConfig(**cfg)
    kwargs = dict(sim_kwargs)
    if backend != "python":
        kwargs["execution_backend"] = backend
    results_df, total_pv, summary_df, rep_cost, n_rep, deg_df = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=battery_config,
        freq=FREQ,
        temperature_series=temp,
        **kwargs,
    )
    return results_df, total_pv, summary_df, rep_cost, n_rep, deg_df


def dump(path: str, backend: str = "python") -> None:
    payload: dict[str, np.ndarray] = {}
    for name in SCENARIOS:
        results_df, total_pv, summary_df, rep_cost, n_rep, deg_df = run(name, backend)
        for col in results_df.columns:
            if col == "Datetime":
                continue
            payload[f"{name}::results::{col}"] = results_df[col].to_numpy(dtype=np.float64)
        for col in deg_df.columns:
            if col == "Datetime":
                continue
            payload[f"{name}::degradation::{col}"] = deg_df[col].to_numpy(dtype=np.float64)
        for col in summary_df.columns:
            payload[f"{name}::summary::{col}"] = summary_df[col].to_numpy(dtype=np.float64)
        payload[f"{name}::scalar::total_pv"] = np.array([total_pv], dtype=np.float64)
        payload[f"{name}::scalar::replacement_cost"] = np.array([rep_cost], dtype=np.float64)
        payload[f"{name}::scalar::n_replacements"] = np.array([n_rep], dtype=np.float64)
        print(f"  {name}: {len(results_df)} steps, {n_rep} replacement(s)", flush=True)
    np.savez(path, **payload)
    print(f"wrote {path} ({len(payload)} arrays)")


if __name__ == "__main__":
    dump(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "python")
