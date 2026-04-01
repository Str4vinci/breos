#!/usr/bin/env python3
"""
Batch location comparison for PV+battery systems.

Systematically compares grid independence across a range of system sizes
(PV panels × battery capacities) for multiple locations. Each location
uses its own real costs for economic metrics.

Usage:
    uv run python tools/batch_compare_locations.py                                # default: porto vs berlin
    uv run python tools/batch_compare_locations.py --locations porto berlin lisbon
    uv run python tools/batch_compare_locations.py --panels 4 15 --batteries 4 6 8
    uv run python tools/batch_compare_locations.py --years 1                      # quick test
    uv run python tools/batch_compare_locations.py --workers 4                    # parallel with 4 workers
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pvlib.location import Location

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from breos.weather import load_weather, resample_to_15min, extract_ambient_temperature
from breos.solar import calculate_pv_production_dc, PVModuleParams
from breos.battery import (
    simulate_energy_balance,
    BatteryConfig,
    apply_indoor_temperature_model,
)
from breos.load_profiles import load_profile
from breos.economics import cost_analysis_projection, find_payback_year
from breos.pv_modules import get_module
from breos.inverter import InverterConfig
from breos.utils import get_hours_per_step
from breos.plotting import (
    plot_grid_independence_heatmap,
    plot_location_comparison_delta,
    plot_breakeven,
    plot_breakeven_comparison,
    _BREAKEVEN_COLORS,
)

# ---------------------------------------------------------------------------
# Location registry
# ---------------------------------------------------------------------------
LOCATION_REGISTRY = {
    'porto':     {'costs': 'residential_pt', 'slope': 35, 'azimuth': 180, 'load_profile': '6'},
    'berlin':    {'costs': 'residential_de', 'slope': 40, 'azimuth': 160, 'load_profile': '1'},
    'lisbon':    {'costs': 'residential_pt', 'slope': 35, 'azimuth': 180, 'load_profile': '6'},
    'erlangen':  {'costs': 'residential_de', 'slope': 40, 'azimuth': 160, 'load_profile': '1'},
    'esposende': {'costs': 'residential_pt', 'slope': 35, 'azimuth': 180, 'load_profile': '6'},
}

# Fixed system parameters
PV_MODULE_NAME = 'Suntech_STP550S_STC'
PV_DEGRADATION_RATE = 0.005
ANNUAL_CONSUMPTION_KWH = 5000
BATTERY_TYPE = 'lfp'
CALENDAR_MODEL = 'lam_calibrated'
EOL_PERCENTAGE = 0.70
MAX_SOC = 0.90
MIN_SOC = 0.10
CHARGE_EFF = 0.974679434
DISCHARGE_EFF = 0.974679434
INVERTER_EFF = 0.96
DC_AC_RATIO = 1.25
INDOOR_SETPOINT = 22.0
INDOOR_ALPHA = 0.3
FREQ = '15min'
START_DATE = '2023-01-01'


def load_configs():
    """Load location and cost configs from JSON."""
    configs_dir = os.path.join(PROJECT_ROOT, 'configs', 'base')
    with open(os.path.join(configs_dir, 'locations.json')) as f:
        locations = json.load(f)
    with open(os.path.join(configs_dir, 'costs.json')) as f:
        costs = json.load(f)
    return locations, costs


def _remap_tmy_year(df, target_year):
    """Remap TMY index to target year."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return df
    was_tz = idx.tz
    if was_tz is not None:
        idx_utc = idx.tz_convert('UTC')
    else:
        idx_utc = idx.tz_localize('UTC')
    dominant_year = idx_utc.year.value_counts().idxmax()
    year_offset = target_year - dominant_year
    if year_offset == 0:
        return df
    new_idx = idx_utc.map(lambda dt: dt.replace(year=dt.year + year_offset))
    if was_tz is not None:
        new_idx = new_idx.tz_convert(was_tz)
    else:
        new_idx = new_idx.tz_localize(None)
    df = df.copy()
    df.index = new_idx
    return df


def load_and_prepare_weather(loc_key, loc_cfg):
    """Load TMY weather, resample to 15min, remap to 2023."""
    from breos import fetch_tmy_weather_data

    weather_dir = os.path.join(PROJECT_ROOT, 'weather')
    tmy_data = load_weather(location=loc_key, data_type='tmy', weather_dir=weather_dir)

    if tmy_data is not None:
        if tmy_data.index.tz is None:
            tmy_data.index = tmy_data.index.tz_localize('UTC')
        tmy_data = _remap_tmy_year(tmy_data, 2023)
        metadata = {
            'inputs': {
                'location': {
                    'latitude': loc_cfg['latitude'],
                    'longitude': loc_cfg['longitude'],
                    'elevation': loc_cfg.get('altitude', 0),
                }
            }
        }
    else:
        print(f"  No local TMY file found for {loc_key}, fetching from PVGIS...")
        tmy_data, metadata = fetch_tmy_weather_data(
            latitude=loc_cfg['latitude'],
            longitude=loc_cfg['longitude'],
            sample_year=2023,
            freq='h',
        )

    # Resample to 15min if hourly
    freq_inferred = pd.infer_freq(tmy_data.index[:10])
    if freq_inferred and 'h' in freq_inferred.lower() and '15' not in freq_inferred:
        print(f"  Resampling {loc_key} TMY to 15-minute resolution...")
        tmy_data = resample_to_15min(
            tmy_data,
            latitude=loc_cfg['latitude'],
            longitude=loc_cfg['longitude'],
        )

    return tmy_data


def precompute_location(loc_key, loc_cfg, reg, pv_params):
    """Precompute 1-module DC production and temperature for a location."""
    print(f"  Precomputing {loc_key}...")
    tmy_data = load_and_prepare_weather(loc_key, loc_cfg)

    location = Location(
        loc_cfg['latitude'],
        loc_cfg['longitude'],
        tz=loc_cfg['timezone'],
    )

    # 1-module DC production
    dc_1mod = calculate_pv_production_dc(
        weather_data=tmy_data,
        location=location,
        slope=reg['slope'],
        surface_azimuth=reg['azimuth'],
        n_modules=1,
        pv_params=pv_params,
        freq=FREQ,
    )

    # Temperature series with indoor model
    ambient_temp = extract_ambient_temperature(tmy_data)
    if ambient_temp is not None:
        temp_series = apply_indoor_temperature_model(
            ambient_temp,
            setpoint_c=INDOOR_SETPOINT,
            coupling_alpha=INDOOR_ALPHA,
        )
    else:
        temp_series = pd.Series(25.0, index=dc_1mod.index)

    return dc_1mod, temp_series


def build_costs_dict(n_modules, battery_kwh, pv_params, costs_cfg):
    """Build the costs dict the same way run_simulation.py does."""
    module_power_w = pv_params.Mpp
    total_power_w = module_power_w * n_modules
    has_battery = battery_kwh > 0

    inv_config = InverterConfig(is_hybrid=has_battery, dc_ac_ratio=DC_AC_RATIO)

    pv_cost = costs_cfg['module_cost_per_w'] * module_power_w * n_modules
    inverter_cost = inv_config.get_cost(total_power_w)
    installation_cost = costs_cfg.get('installation_cost_per_module', 350) * n_modules
    if has_battery:
        installation_cost += costs_cfg.get('installation_cost_battery', 350)
    battery_cost = costs_cfg['storage_cost_per_kwh'] * battery_kwh
    other_costs = costs_cfg.get('other_costs', 50)
    total_initial = pv_cost + inverter_cost + installation_cost + battery_cost + other_costs

    return {
        'electricity_cost': costs_cfg['electricity_cost'],
        'electricity_sold_cost': costs_cfg['electricity_sold_cost'],
        'total_initial_cost': total_initial,
        'annual_operation_cost': costs_cfg.get('maintenance_cost', 50),
        'daily_power_cost': costs_cfg.get('daily_power_cost', 0.57),
        'pv_cost': pv_cost,
        'inverter_cost': inverter_cost,
        'battery_cost': battery_cost,
        'installation_cost': installation_cost,
        'other_costs': other_costs,
    }


def _run_single_sim(args_tuple):
    """
    Run a single (location, n_modules, battery_kwh) simulation.

    Top-level function so it can be pickled by multiprocessing.
    Returns a dict with the result row.
    """
    (loc_key, n_modules, battery_kwh, dc_1mod, temp_series,
     load_data, costs_cfg, pv_params, years_projection) = args_tuple

    hours_per_step = get_hours_per_step(FREQ)
    battery_wh = battery_kwh * 1000
    replacement_cost = costs_cfg['storage_cost_per_kwh'] * battery_kwh

    # State for multi-year propagation
    cumulative_fec = 0.0
    cumulative_cal_seconds = 0.0
    cumulative_resistance_growth = 0.0
    cumulative_cycle_deg = 0.0
    cumulative_cal_deg = 0.0
    current_soh = 100.0
    total_replacements = 0
    total_replacement_cost = 0.0
    yearly_summaries = []
    year1_fec = 0.0

    for year_idx in range(years_projection):
        pv_degradation_factor = (1 - PV_DEGRADATION_RATE) ** year_idx
        dc_power = dc_1mod * n_modules * pv_degradation_factor

        year_battery_config = BatteryConfig(
            nominal_energy_wh=battery_wh,
            battery_type=BATTERY_TYPE,
            initial_soh=current_soh,
            eol_percentage=EOL_PERCENTAGE,
            max_soc=MAX_SOC,
            min_soc=MIN_SOC,
            charge_efficiency=CHARGE_EFF,
            discharge_efficiency=DISCHARGE_EFF,
            dc_coupled=True,
            inverter_efficiency=INVERTER_EFF,
            enable_replacement=True,
            replacement_cost=replacement_cost,
            calendar_model=CALENDAR_MODEL,
        )

        results_df, total_pv, summary_df, year_rep_cost, year_n_rep, degradation_df = (
            simulate_energy_balance(
                pv_dc=dc_power,
                houseload=load_data,
                battery_config=year_battery_config,
                freq=FREQ,
                temperature_series=temp_series,
                initial_fec=cumulative_fec,
                initial_calendar_seconds=cumulative_cal_seconds,
                initial_resistance_growth=cumulative_resistance_growth,
                initial_cumulative_cycle_deg=cumulative_cycle_deg,
                initial_cumulative_cal_deg=cumulative_cal_deg,
            )
        )

        # Update carryover state
        if not degradation_df.empty:
            cumulative_fec = degradation_df['Cumulative_FEC'].iloc[-1]
            cumulative_cal_seconds = degradation_df['Cumulative_Calendar_Seconds'].iloc[-1]
            cumulative_cycle_deg = degradation_df['Cumulative_Cycle_Degradation'].iloc[-1]
            cumulative_cal_deg = degradation_df['Cumulative_Calendar_Degradation'].iloc[-1]
            current_soh = degradation_df['SOH'].iloc[-1]
            if 'Resistance_Growth' in degradation_df.columns:
                cumulative_resistance_growth = degradation_df['Resistance_Growth'].iloc[-1]

        # Capture year-1 FEC
        if year_idx == 0 and not degradation_df.empty:
            year1_fec = degradation_df['Cumulative_FEC'].iloc[-1]

        total_replacements += year_n_rep
        total_replacement_cost += year_rep_cost

        # Yearly summary
        total_pv_kwh = total_pv / 1000
        total_load = (results_df['Houseload'].sum() / 1000) * hours_per_step
        total_import = (results_df['Import_From_Grid'].sum() / 1000) * hours_per_step
        total_export = (results_df['Sell_To_Grid'].sum() / 1000) * hours_per_step
        grid_indep = (1 - total_import / total_load) * 100 if total_load > 0 else 0

        yearly_summaries.append({
            'Year': year_idx + 1,
            'PV_Production_kWh': total_pv_kwh,
            'Load_kWh': total_load,
            'Import_kWh': total_import,
            'Export_kWh': total_export,
            'Grid_Independence_%': grid_indep,
            'Battery_SOH_%': current_soh,
            'Replacements': year_n_rep,
            'Replacement_Cost': year_rep_cost,
            'PV_Degradation_Factor': pv_degradation_factor,
        })

    yearly_df = pd.DataFrame(yearly_summaries)
    year1 = yearly_df.iloc[0]
    avg_gi = yearly_df['Grid_Independence_%'].mean()

    # Cost analysis
    costs_dict = build_costs_dict(n_modules, battery_kwh, pv_params, costs_cfg)
    cost_proj = cost_analysis_projection(
        results_df=results_df,
        costs=costs_dict,
        num_years=years_projection,
        inflation_rate=costs_cfg.get('inflation_rate', 0.02),
        discount_rate=costs_cfg.get('discount_rate', 0.0),
        freq=FREQ,
        yearly_summary_df=yearly_df,
        total_replacement_cost=total_replacement_cost,
    )
    payback = find_payback_year(cost_proj)

    # --- Derived metrics ---
    # Economic efficiency
    npv_savings = cost_proj['Savings_Cumulative_NPV'].iloc[-1] if 'Savings_Cumulative_NPV' in cost_proj.columns else 0.0
    total_initial = costs_dict['total_initial_cost']
    system_kwp = n_modules * pv_params.Mpp / 1000

    roi_percent = (npv_savings / total_initial * 100) if total_initial > 0 else 0.0
    savings_per_kwp = (npv_savings / system_kwp) if system_kwp > 0 else 0.0
    savings_per_euro = (npv_savings / total_initial) if total_initial > 0 else 0.0

    # System utilization (year 1)
    yr1_pv = year1['PV_Production_kWh']
    yr1_export = year1['Export_kWh']
    self_consumption_pct = ((yr1_pv - yr1_export) / yr1_pv * 100) if yr1_pv > 0 else 0.0
    export_ratio_pct = (yr1_export / yr1_pv * 100) if yr1_pv > 0 else 0.0

    row = {
        'location': loc_key,
        'n_modules': n_modules,
        'battery_kwh': battery_kwh,
        'year1_grid_independence': year1['Grid_Independence_%'],
        'avg_grid_independence': avg_gi,
        'year1_pv_kwh': yr1_pv,
        'payback_year': payback,
        'final_soh': current_soh,
        'total_replacements': total_replacements,
        'total_initial_cost': total_initial,
        'total_replacement_cost': total_replacement_cost,
        # Economic efficiency
        'npv_savings_20yr': npv_savings,
        'roi_percent': roi_percent,
        'savings_per_kwp': savings_per_kwp,
        'savings_per_euro': savings_per_euro,
        # System utilization (year 1)
        'self_consumption_pct': self_consumption_pct,
        'export_ratio_pct': export_ratio_pct,
        'year1_fec': year1_fec,
    }
    return row, cost_proj


def run_sweep(
    locations_data,
    panel_range,
    battery_sizes,
    years_projection,
    pv_params,
    all_costs,
    max_workers=1,
):
    """Run the parameter sweep across all locations and system sizes."""
    # Build job list
    jobs = []
    for loc_key, loc_info in locations_data.items():
        costs_cfg = all_costs[LOCATION_REGISTRY[loc_key]['costs']]
        for n_modules in panel_range:
            for battery_kwh in battery_sizes:
                jobs.append((
                    loc_key, n_modules, battery_kwh,
                    loc_info['dc_1mod'], loc_info['temp_series'],
                    loc_info['load_data'], costs_cfg, pv_params, years_projection,
                ))

    total_sims = len(jobs)
    cost_projections = {}  # (loc_key, n_modules, battery_kwh) -> cost_proj DataFrame

    if max_workers == 1:
        # Sequential mode
        results = []
        for idx, job in enumerate(jobs, 1):
            t0 = time.time()
            row, cost_proj = _run_single_sim(job)
            elapsed = time.time() - t0
            _print_sim_progress(idx, total_sims, row, elapsed)
            results.append(row)
            cost_projections[(row['location'], row['n_modules'], row['battery_kwh'])] = cost_proj
    else:
        # Parallel mode
        results = [None] * total_sims
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_run_single_sim, job): idx
                for idx, job in enumerate(jobs)
            }
            t_batch_start = time.time()
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                row, cost_proj = future.result()
                results[idx] = row
                cost_projections[(row['location'], row['n_modules'], row['battery_kwh'])] = cost_proj
                completed += 1
                elapsed_total = time.time() - t_batch_start
                avg_per_sim = elapsed_total / completed
                remaining = (total_sims - completed) * avg_per_sim
                _print_sim_progress(
                    completed, total_sims, row, avg_per_sim,
                    eta_min=remaining / 60,
                )

    return pd.DataFrame(results), cost_projections


def _print_sim_progress(idx, total, row, elapsed, eta_min=None):
    """Print a single simulation progress line."""
    eta_str = f" | ETA {eta_min:.1f}m" if eta_min is not None else ""
    payback = row['payback_year']
    print(
        f"  [{idx:3d}/{total}] {row['location']:>10s} | "
        f"{row['n_modules']:2d} panels | {row['battery_kwh']} kWh | "
        f"GI_y1={row['year1_grid_independence']:5.1f}% | "
        f"GI_avg={row['avg_grid_independence']:5.1f}% | "
        f"payback={'N/A' if payback is None else f'{payback:2d}y'} | "
        f"{elapsed:.1f}s{eta_str}"
    )


def _interpolate_breakeven(df):
    """Interpolate break-even year from cost projection DataFrame."""
    if 'Savings_Cumulative_NPV' not in df.columns:
        return None
    savings = df['Savings_Cumulative_NPV'].values
    years = df['Year'].values
    for i in range(1, len(savings)):
        if savings[i] >= 0 and savings[i - 1] < 0:
            frac = -savings[i - 1] / (savings[i] - savings[i - 1])
            return years[i - 1] + frac
    return None


def _compute_marginal_returns(results_df, locations):
    """Compute marginal GI per additional panel and per additional kWh battery."""
    results_df['marginal_gi_per_panel'] = np.nan
    results_df['marginal_gi_per_kwh_batt'] = np.nan

    for loc in locations:
        # Marginal GI per panel (for each battery size)
        battery_sizes = sorted(results_df[results_df['location'] == loc]['battery_kwh'].unique())
        for batt in battery_sizes:
            mask = (results_df['location'] == loc) & (results_df['battery_kwh'] == batt)
            subset = results_df.loc[mask].sort_values('n_modules')
            results_df.loc[subset.index, 'marginal_gi_per_panel'] = (
                subset['year1_grid_independence'].diff().values
            )

        # Marginal GI per kWh battery (for each panel count)
        panel_counts = sorted(results_df[results_df['location'] == loc]['n_modules'].unique())
        for n_mod in panel_counts:
            mask = (results_df['location'] == loc) & (results_df['n_modules'] == n_mod)
            subset = results_df.loc[mask].sort_values('battery_kwh')
            batt_diff = subset['battery_kwh'].diff()
            gi_diff = subset['year1_grid_independence'].diff()
            marginal = gi_diff / batt_diff.replace(0, np.nan)
            results_df.loc[subset.index, 'marginal_gi_per_kwh_batt'] = marginal.values

    return results_df


def generate_outputs(results_df, output_dir, locations, cost_projections=None):
    """Generate CSV, console tables, and plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute marginal returns
    results_df = _compute_marginal_returns(results_df, locations)

    # Save CSV
    csv_path = os.path.join(output_dir, 'batch_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV: {csv_path}")

    # Console table per location
    for loc in locations:
        loc_df = results_df[results_df['location'] == loc]
        print(f"\n{'='*60}")
        print(f"  {loc.upper()} — Grid Independence (%) Year 1")
        print(f"{'='*60}")
        pivot = loc_df.pivot_table(
            index='battery_kwh', columns='n_modules',
            values='year1_grid_independence',
        )
        print(pivot.to_string(float_format='{:.1f}'.format))

    # Per-location heatmaps (year-1 grid independence)
    all_gi_values = results_df['year1_grid_independence'].values
    vmin = np.floor(all_gi_values.min())
    vmax = np.ceil(all_gi_values.max())

    for loc in locations:
        loc_df = results_df[results_df['location'] == loc]
        pivot = loc_df.pivot_table(
            index='battery_kwh', columns='n_modules',
            values='year1_grid_independence',
        )
        plot_grid_independence_heatmap(
            pivot, output_dir, loc,
            filename=f'grid_independence_{loc}.png',
            vmin=vmin, vmax=vmax,
        )
        print(f"Saved: grid_independence_{loc}.png")

    # Per-location heatmaps (20yr avg grid independence)
    all_avg_gi = results_df['avg_grid_independence'].values
    vmin_avg = np.floor(all_avg_gi.min())
    vmax_avg = np.ceil(all_avg_gi.max())

    for loc in locations:
        loc_df = results_df[results_df['location'] == loc]
        pivot = loc_df.pivot_table(
            index='battery_kwh', columns='n_modules',
            values='avg_grid_independence',
        )
        plot_grid_independence_heatmap(
            pivot, output_dir, loc,
            filename=f'grid_independence_avg_{loc}.png',
            metric_label='Avg Grid Independence (%)',
            vmin=vmin_avg, vmax=vmax_avg,
        )
        print(f"Saved: grid_independence_avg_{loc}.png")

    # Payback heatmaps
    for loc in locations:
        loc_df = results_df[results_df['location'] == loc]
        pivot = loc_df.pivot_table(
            index='battery_kwh', columns='n_modules',
            values='payback_year',
        )
        plot_grid_independence_heatmap(
            pivot, output_dir, loc,
            filename=f'payback_{loc}.png',
            metric_label='Payback Year',
            cmap='YlOrRd',
        )
        print(f"Saved: payback_{loc}.png")

    # Self-consumption heatmaps
    for loc in locations:
        loc_df = results_df[results_df['location'] == loc]
        pivot = loc_df.pivot_table(
            index='battery_kwh', columns='n_modules',
            values='self_consumption_pct',
        )
        plot_grid_independence_heatmap(
            pivot, output_dir, loc,
            filename=f'self_consumption_{loc}.png',
            metric_label='Self-Consumption (%)',
            cmap='YlGnBu',
        )
        print(f"Saved: self_consumption_{loc}.png")

    # ROI % heatmaps
    for loc in locations:
        loc_df = results_df[results_df['location'] == loc]
        pivot = loc_df.pivot_table(
            index='battery_kwh', columns='n_modules',
            values='roi_percent',
        )
        plot_grid_independence_heatmap(
            pivot, output_dir, loc,
            filename=f'roi_{loc}.png',
            metric_label='ROI (%)',
            cmap='RdYlGn',
        )
        print(f"Saved: roi_{loc}.png")

    # Savings per € invested heatmaps
    for loc in locations:
        loc_df = results_df[results_df['location'] == loc]
        pivot = loc_df.pivot_table(
            index='battery_kwh', columns='n_modules',
            values='savings_per_euro',
        )
        plot_grid_independence_heatmap(
            pivot, output_dir, loc,
            filename=f'savings_per_euro_{loc}.png',
            metric_label='Savings per € Invested',
            cmap='RdYlGn',
        )
        print(f"Saved: savings_per_euro_{loc}.png")

    # Marginal GI per panel heatmaps
    for loc in locations:
        loc_df = results_df[results_df['location'] == loc]
        pivot = loc_df.pivot_table(
            index='battery_kwh', columns='n_modules',
            values='marginal_gi_per_panel',
        )
        plot_grid_independence_heatmap(
            pivot, output_dir, loc,
            filename=f'marginal_gi_per_panel_{loc}.png',
            metric_label='Marginal GI per Panel (pp)',
            cmap='YlOrRd_r',
        )
        print(f"Saved: marginal_gi_per_panel_{loc}.png")

    # Delta heatmaps (pairwise between first two locations)
    if len(locations) >= 2:
        loc_a, loc_b = locations[0], locations[1]

        # Grid independence delta
        pivot_a = results_df[results_df['location'] == loc_a].pivot_table(
            index='battery_kwh', columns='n_modules',
            values='year1_grid_independence',
        )
        pivot_b = results_df[results_df['location'] == loc_b].pivot_table(
            index='battery_kwh', columns='n_modules',
            values='year1_grid_independence',
        )
        common_idx = pivot_a.index.intersection(pivot_b.index)
        common_cols = pivot_a.columns.intersection(pivot_b.columns)
        delta = pivot_a.loc[common_idx, common_cols] - pivot_b.loc[common_idx, common_cols]

        plot_location_comparison_delta(
            delta, output_dir, loc_a, loc_b,
            filename='grid_independence_delta.png',
        )
        print(f"Saved: grid_independence_delta.png ({loc_a} - {loc_b})")

        # ROI delta
        roi_a = results_df[results_df['location'] == loc_a].pivot_table(
            index='battery_kwh', columns='n_modules',
            values='roi_percent',
        )
        roi_b = results_df[results_df['location'] == loc_b].pivot_table(
            index='battery_kwh', columns='n_modules',
            values='roi_percent',
        )
        roi_delta = roi_a.loc[common_idx, common_cols] - roi_b.loc[common_idx, common_cols]

        plot_location_comparison_delta(
            roi_delta, output_dir, loc_a, loc_b,
            filename='roi_delta.png',
            metric_label='ROI Delta (pp)',
        )
        print(f"Saved: roi_delta.png ({loc_a} - {loc_b})")

    # Breakeven plots
    if cost_projections:
        breakeven_dir = os.path.join(output_dir, 'breakeven')
        os.makedirs(breakeven_dir, exist_ok=True)

        # Per-location breakeven cumulative plots for each system size
        for loc in locations:
            loc_df = results_df[results_df['location'] == loc]
            for _, row in loc_df.iterrows():
                key = (row['location'], row['n_modules'], row['battery_kwh'])
                if key in cost_projections:
                    scenario = f"{loc}_{int(row['n_modules'])}p_{int(row['battery_kwh'])}kWh"
                    plot_breakeven(cost_projections[key], breakeven_dir, scenario_name=scenario)
            print(f"Saved: breakeven plots for {loc}")

        # Cross-location comparison for each (n_modules, battery_kwh) combo
        if len(locations) >= 2:
            system_sizes = results_df[['n_modules', 'battery_kwh']].drop_duplicates()
            for _, sz in system_sizes.iterrows():
                n_mod = sz['n_modules']
                batt = sz['battery_kwh']
                cost_dfs = []
                labels = []
                colors = []
                for i, loc in enumerate(locations):
                    key = (loc, n_mod, batt)
                    if key in cost_projections:
                        cost_dfs.append(cost_projections[key])
                        labels.append(f"{loc.capitalize()} ({int(n_mod)}p/{int(batt)}kWh)")
                        colors.append(_BREAKEVEN_COLORS[i % len(_BREAKEVEN_COLORS)])

                if len(cost_dfs) >= 2:
                    fname = f"breakeven_comparison_{int(n_mod)}p_{int(batt)}kWh.png"
                    plot_breakeven_comparison(cost_dfs, labels, colors, breakeven_dir, fname)

            print(f"Saved: cross-location breakeven comparisons")


def main():
    parser = argparse.ArgumentParser(
        description='Batch compare PV+battery grid independence across locations'
    )
    parser.add_argument(
        '--locations', nargs='+', default=['porto', 'berlin'],
        help='Location keys (default: porto berlin)',
    )
    parser.add_argument(
        '--panels', nargs='+', type=int, default=None,
        help='Panel counts: min max or explicit list (default: 4..15)',
    )
    parser.add_argument(
        '--batteries', nargs='+', type=float, default=[4, 6, 8],
        help='Battery sizes in kWh (default: 4 6 8)',
    )
    parser.add_argument(
        '--years', type=int, default=20,
        help='Projection years (default: 20)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory (default: results/batch_comparison)',
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of parallel workers (default: cpu_count - 1, use 1 for sequential)',
    )
    args = parser.parse_args()

    # Resolve panel range
    if args.panels is None:
        panel_range = list(range(4, 16))
    elif len(args.panels) == 2 and args.panels[1] > args.panels[0]:
        panel_range = list(range(args.panels[0], args.panels[1] + 1))
    else:
        panel_range = args.panels

    output_dir = args.output or os.path.join(PROJECT_ROOT, 'results', 'batch_comparison')

    # Validate locations
    for loc in args.locations:
        if loc not in LOCATION_REGISTRY:
            available = ', '.join(LOCATION_REGISTRY.keys())
            print(f"Error: Unknown location '{loc}'. Available: {available}")
            sys.exit(1)

    # Resolve worker count
    if args.workers is not None:
        max_workers = max(1, args.workers)
    else:
        cpu_count = os.cpu_count() or 2
        max_workers = max(1, cpu_count - 1)

    locations_cfg, all_costs = load_configs()
    pv_params = get_module(PV_MODULE_NAME)

    print("="*60)
    print("  Batch Location Comparison")
    print("="*60)
    print(f"  Locations:  {', '.join(args.locations)}")
    print(f"  Panels:     {panel_range}")
    print(f"  Batteries:  {args.batteries} kWh")
    print(f"  Years:      {args.years}")
    print(f"  Resolution: {FREQ}")
    total_sims = len(args.locations) * len(panel_range) * len(args.batteries)
    print(f"  Total sims: {total_sims}")
    mode = "sequential" if max_workers == 1 else f"{max_workers} workers"
    print(f"  Execution:  {mode}")
    print()

    # Precompute per-location data and load profiles
    print("--- Precomputing per-location data ---")
    locations_data = {}
    for loc_key in args.locations:
        loc_cfg = locations_cfg[loc_key]
        reg = LOCATION_REGISTRY[loc_key]
        dc_1mod, temp_series = precompute_location(loc_key, loc_cfg, reg, pv_params)

        profile_type = reg['load_profile']
        loc_load_data = load_profile(
            profile_type=profile_type,
            annual_consumption_kwh=ANNUAL_CONSUMPTION_KWH,
            start_date=START_DATE,
            freq=FREQ,
            num_years=1,
            timezone='UTC',
        )
        print(f"  {loc_key} load profile: type={profile_type}, {ANNUAL_CONSUMPTION_KWH} kWh/yr, {len(loc_load_data)} steps")

        locations_data[loc_key] = {
            'dc_1mod': dc_1mod,
            'temp_series': temp_series,
            'load_data': loc_load_data,
        }

    # Run sweep
    print(f"\n--- Running {total_sims} simulations ({args.years}-year propagation each) ---")
    t_start = time.time()
    results_df, cost_projections = run_sweep(
        locations_data=locations_data,
        panel_range=panel_range,
        battery_sizes=args.batteries,
        years_projection=args.years,
        pv_params=pv_params,
        all_costs=all_costs,
        max_workers=max_workers,
    )
    elapsed = time.time() - t_start
    print(f"\nTotal sweep time: {elapsed/60:.1f} minutes")

    # Generate outputs
    print("\n--- Generating outputs ---")
    generate_outputs(results_df, output_dir, args.locations, cost_projections)

    print(f"\nDone! Results in: {output_dir}")


if __name__ == '__main__':
    main()
