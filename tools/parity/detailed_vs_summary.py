"""Validation step 1: the summary path must reduce to the detailed path exactly.

For every scenario, run both entry points and compare each summary field
against the same quantity derived from the detailed results frame. Exact
equality is required; any difference is printed with its magnitude.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from harness import FREQ, SCENARIOS, build  # noqa: E402


def check(name: str, backend: str = "python") -> list[str]:
    from breos.battery import BatteryConfig, simulate_energy_balance, simulate_energy_balance_summary

    pv, load, temp, cfg, sim_kwargs = build(name)
    common = dict(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**cfg),
        freq=FREQ,
        temperature_series=temp,
        return_degradation_state=True,
        **sim_kwargs,
    )
    detailed = simulate_energy_balance(**common)
    summary = simulate_energy_balance_summary(**common, execution_backend=backend)

    results_df, total_pv, summary_df, rep_cost, n_rep, deg_df = detailed[:6]
    detailed_state = detailed[6]

    failures: list[str] = []

    def same(label, a, b):
        if a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
            failures.append(f"{name}::{label}: detailed={a!r} summary={b!r} delta={abs(a - b):.6e}")

    for col in results_df.columns:
        if col == "Datetime":
            continue
        same(f"sum::{col}", float(results_df[col].sum()), summary.column_sums[col])

    same("total_pv", float(total_pv), float(summary.total_pv_wh))
    same("replacement_cost", float(rep_cost), float(summary.total_replacement_cost))
    same("n_replacements", int(n_rep), int(summary.n_replacements))
    for col in summary_df.columns:
        same(f"summary_row::{col}", float(summary_df[col].iloc[0]), float(summary.summary_row[col]))

    same("carried_energy", float(results_df["Battery_Energy_End"].iloc[-1]), summary.carried_energy_wh)
    same(
        "carried_pv_origin",
        float(results_df["Battery_PV_Origin_Energy_End"].iloc[-1]),
        summary.carried_pv_origin_energy_wh,
    )
    same("has_degradation_rows", not deg_df.empty, summary.has_degradation_rows)
    if not deg_df.empty:
        same("fec", float(deg_df["Cumulative_FEC"].iloc[-1]), summary.fec_cum)
        same(
            "calendar_seconds",
            float(deg_df["Cumulative_Calendar_Seconds"].iloc[-1]),
            summary.cumulative_calendar_seconds,
        )
        same(
            "cycle_deg",
            float(deg_df["Cumulative_Cycle_Degradation"].iloc[-1]),
            summary.cumulative_cycle_degradation,
        )
        same(
            "calendar_deg",
            float(deg_df["Cumulative_Calendar_Degradation"].iloc[-1]),
            summary.cumulative_calendar_degradation,
        )
        same("final_soh", float(deg_df["SOH"].iloc[-1]), summary.final_soh_percent)
        if "Resistance_Growth" in deg_df.columns:
            same("resistance_growth", float(deg_df["Resistance_Growth"].iloc[-1]), summary.resistance_growth)

    expected_steps = tuple(int(i) for i in np.flatnonzero(results_df["Battery_Replaced"].to_numpy()))
    same("replacement_steps", expected_steps, summary.replacement_steps)

    for key, value in detailed_state.items():
        got = summary.final_degradation_state[key]
        if isinstance(value, float):
            same(f"state::{key}", value, float(got))
        elif value != got:
            failures.append(f"{name}::state::{key}: detailed={value!r} summary={got!r}")

    return failures


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "python"
    all_failures: list[str] = []
    for name in SCENARIOS:
        failures = check(name, backend)
        status = "EXACT" if not failures else f"{len(failures)} DIFF"
        print(f"  {name:<20} {status}", flush=True)
        all_failures.extend(failures)
    if all_failures:
        print()
        for line in all_failures:
            print(line)
        print(f"\n{len(all_failures)} field(s) differ")
        return 1
    print(f"\nsummary path is bit-identical to the detailed path (backend={backend})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
