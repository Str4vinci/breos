#!/usr/bin/env python3
"""
Calendar Aging Sensitivity Analysis

Runs the singleyear Porto 15-min simulation across a range of calendar k0
scaling factors to evaluate how calendar aging rate affects battery lifetime.

Usage:
    python tools/sensitivity_calendar_aging.py
    python tools/sensitivity_calendar_aging.py --config configs/scenarios/singleyear_porto_15min.json
    python tools/sensitivity_calendar_aging.py --factors 0.25 0.50 0.75 1.0
    python tools/sensitivity_calendar_aging.py --output results/sensitivity_cal

Output:
    - SOH trajectory plot across k0 scaling factors
    - Console summary with years-to-replacement for each factor
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import from run_simulation for config resolution and helpers
sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Calendar aging k0 sensitivity analysis")
    parser.add_argument(
        '--config', type=Path,
        default=PROJECT_ROOT / 'configs' / 'scenarios' / 'singleyear_porto_15min.json',
        help='Base scenario config (default: singleyear_porto_15min.json)',
    )
    parser.add_argument(
        '--factors', type=float, nargs='+',
        default=[0.25, 0.50, 0.75, 1.0],
        help='k0 scaling factors (default: 0.25 0.50 0.75 1.0)',
    )
    parser.add_argument(
        '--output', type=str,
        default='results/sensitivity_calendar_aging',
        help='Output directory',
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Import after path setup
    from run_simulation import resolve_extends, load_tmy_weather, get_temperature_series
    from breos import (
        fetch_tmy_weather_data,
        calculate_pv_production_dc,
        get_module,
        load_profile,
        simulate_energy_balance,
        BatteryConfig,
        default_azimuth,
    )
    from breos.constants import (
        LAM_CAL_K0_FRAC, LAM_CAL_EA_J_MOL,
        LAM_CAL_EXPONENT_B, LAM_CAL_SOC_EXPONENT_N,
        DEFAULT_MAX_SOC, DEFAULT_MIN_SOC,
    )
    from breos.battery import update_battery_soh_calendar, update_battery_soh_cyclewise
    from breos.utils import get_hours_per_step
    from breos.plotting import plot_calendar_aging_sensitivity
    from pvlib.location import Location
    import math

    DEFAULT_BATTERY_EFFICIENCY = math.sqrt(0.95)

    # Load and resolve config
    raw_config = load_json(args.config)
    config_dir = PROJECT_ROOT / 'configs'
    config = resolve_extends(raw_config, config_dir)

    loc = config['location']
    pv_cfg = config['pv']
    batt_cfg = config.get('battery', {})
    inv_cfg = config.get('inverter', {})
    load_cfg = config.get('load', {})
    sim_cfg = config.get('simulation', {})

    freq = sim_cfg.get('resolution', 'h')
    hours_per_step = get_hours_per_step(freq)
    years_projection = sim_cfg.get('years_projection', 20)
    degradation_rate = pv_cfg.get('degradation_rate', 0.005)

    # Setup location
    location = Location(
        latitude=loc['latitude'],
        longitude=loc['longitude'],
        tz=loc.get('timezone', 'UTC'),
        name=loc.get('name', 'Unknown'),
    )

    # Load weather
    tmy_data, tmy_metadata = load_tmy_weather(loc, sim_cfg, config)

    # PV setup
    pv_params = get_module(pv_cfg['module'])
    n_modules = pv_cfg.get('n_modules', 5)
    slope = pv_cfg.get('slope', 35)
    azimuth = pv_cfg.get('azimuth', default_azimuth(loc['latitude']))

    # Load profile
    load_data = load_profile(
        profile_type=load_cfg.get('source', 'crest'),
        annual_consumption_kwh=load_cfg.get('annual_consumption_kwh', 4000),
        freq=freq,
    )

    # Battery base config
    rep_cost_input = batt_cfg.get('replacement_cost')
    if rep_cost_input is not None and str(rep_cost_input).lower() != 'calculate':
        rep_cost_val = float(rep_cost_input) or None
    else:
        rep_cost_val = None

    batt_temp_cfg = batt_cfg.get('temperature', 'weather')
    indoor_model = batt_cfg.get('indoor_model', None)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  Calendar Aging k₀ Sensitivity Analysis")
    print("=" * 60)
    print(f"  Config:     {args.config.name}")
    print(f"  Location:   {location.name}")
    print(f"  Battery:    {batt_cfg.get('nominal_kwh', 7)} kWh")
    print(f"  Projection: {years_projection} years")
    print(f"  Factors:    {args.factors}")
    print(f"  Output:     {args.output}")
    print()

    # Base k0 from lam_calibrated
    base_k0 = LAM_CAL_K0_FRAC

    soh_trajectories = {}
    summary_rows = []

    for factor in sorted(args.factors):
        label = f"k\u2080 \u00d7 {factor:.2f}"
        scaled_k0 = base_k0 * factor
        print(f"--- Running factor {factor:.2f} (k0 = {scaled_k0:.4e}) ---")

        # Run multi-year propagation
        cumulative_fec = 0.0
        cumulative_cal_seconds = 0.0
        cumulative_resistance_growth = 0.0
        cumulative_cycle_deg_val = 0.0
        cumulative_cal_deg_val = 0.0
        current_soh = batt_cfg.get('initial_soh', 100.0)
        total_replacements = 0
        first_replacement_year = None
        yearly_soh = []

        for year_idx in range(years_projection):
            year_num = year_idx + 1
            pv_degradation_factor = (1 - degradation_rate) ** year_idx

            dc_power = calculate_pv_production_dc(
                weather_data=tmy_data,
                location=location,
                slope=slope,
                surface_azimuth=azimuth,
                n_modules=n_modules,
                pv_params=pv_params,
                freq=freq,
            ) * pv_degradation_factor

            T_series_C = get_temperature_series(
                batt_temp_cfg,
                dc_power.index[0],
                dc_power.index[-1],
                freq=freq,
                weather_df=tmy_data,
                indoor_model=indoor_model,
            )

            # Create battery config with custom calendar_model that won't be used
            # directly — we override via the k0 scaling approach.
            # We use 'lam_calibrated' as base and manually patch the k0 value
            # by creating a temporary constants override approach.
            # Actually, the cleanest way is to just use 'lam_calibrated' and
            # temporarily patch the constant. But that's not clean.
            # Instead, run simulate_energy_balance which uses _get_degradation_params.
            # We need to pass a custom k0. The simplest approach: temporarily
            # monkey-patch the constant in the battery module for this run.
            import breos.battery as battery_mod
            import breos.constants as constants_mod

            # Save originals
            orig_k0 = constants_mod.LAM_CAL_K0_FRAC

            # Patch
            constants_mod.LAM_CAL_K0_FRAC = scaled_k0
            battery_mod.LAM_CAL_K0_FRAC = scaled_k0

            year_battery_config = BatteryConfig(
                nominal_energy_wh=batt_cfg.get('nominal_kwh', 7) * 1000,
                initial_soh=current_soh,
                eol_percentage=batt_cfg.get('eol_percentage', 0.80),
                max_soc=batt_cfg.get('max_soc', DEFAULT_MAX_SOC),
                min_soc=batt_cfg.get('min_soc', DEFAULT_MIN_SOC),
                charge_efficiency=batt_cfg.get('charge_efficiency', DEFAULT_BATTERY_EFFICIENCY),
                discharge_efficiency=batt_cfg.get('discharge_efficiency', DEFAULT_BATTERY_EFFICIENCY),
                dc_coupled=batt_cfg.get('dc_coupled', True),
                inverter_efficiency=inv_cfg.get('efficiency', 0.96),
                enable_replacement=batt_cfg.get('enable_replacement', True),
                replacement_cost=rep_cost_val,
                calendar_model='lam_calibrated',
                enable_resistance_fade=batt_cfg.get('enable_resistance_fade', False),
            )

            results_df, total_pv, summary_df, year_rep_cost, year_n_rep, degradation_df = simulate_energy_balance(
                pv_dc=dc_power,
                houseload=load_data,
                battery_config=year_battery_config,
                freq=freq,
                temperature_series=T_series_C,
                initial_fec=cumulative_fec,
                initial_calendar_seconds=cumulative_cal_seconds,
                initial_resistance_growth=cumulative_resistance_growth,
                initial_cumulative_cycle_deg=cumulative_cycle_deg_val,
                initial_cumulative_cal_deg=cumulative_cal_deg_val,
            )

            # Restore originals
            constants_mod.LAM_CAL_K0_FRAC = orig_k0
            battery_mod.LAM_CAL_K0_FRAC = orig_k0

            # Update carryover state
            if not degradation_df.empty:
                cumulative_fec = degradation_df['Cumulative_FEC'].iloc[-1]
                cumulative_cal_seconds = degradation_df['Cumulative_Calendar_Seconds'].iloc[-1]
                cumulative_cycle_deg_val = degradation_df['Cumulative_Cycle_Degradation'].iloc[-1]
                cumulative_cal_deg_val = degradation_df['Cumulative_Calendar_Degradation'].iloc[-1]
                current_soh = degradation_df['SOH'].iloc[-1]
                if 'Resistance_Growth' in degradation_df.columns:
                    cumulative_resistance_growth = degradation_df['Resistance_Growth'].iloc[-1]

            total_replacements += year_n_rep
            if year_n_rep > 0 and first_replacement_year is None:
                first_replacement_year = year_num

            yearly_soh.append(current_soh)

            if year_num % 5 == 0:
                print(f"   Year {year_num:2d}: SOH = {current_soh:.1f}%")

        soh_trajectories[label] = yearly_soh

        summary_rows.append({
            'Factor': factor,
            'First_Replacement_Year': first_replacement_year,
            'Total_Replacements': total_replacements,
            'Final_SOH': current_soh,
        })

        repl_str = f"year {first_replacement_year}" if first_replacement_year else "none"
        print(f"   Result: {total_replacements} replacement(s), first at {repl_str}, "
              f"final SOH = {current_soh:.1f}%\n")

    # Summary table
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Factor':>8s}  {'1st Repl':>10s}  {'Total Repl':>10s}  {'Final SOH':>10s}")
    print(f"  {'':->8s}  {'':->10s}  {'':->10s}  {'':->10s}")
    for row in summary_rows:
        repl_str = f"Year {row['First_Replacement_Year']}" if row['First_Replacement_Year'] else "N/A"
        print(f"  {row['Factor']:>8.2f}  {repl_str:>10s}  {row['Total_Replacements']:>10d}  "
              f"{row['Final_SOH']:>9.1f}%")

    # Generate plot
    eol_pct = batt_cfg.get('eol_percentage', 0.80) * 100
    plot_calendar_aging_sensitivity(
        soh_trajectories=soh_trajectories,
        eol_threshold=eol_pct,
        results_dir=args.output,
    )
    print(f"\nPlot saved to: {os.path.join(args.output, 'calendar_aging_sensitivity.png')}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output, 'sensitivity_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
