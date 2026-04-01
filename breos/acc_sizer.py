
"""
ACC Sizer Module
Optimizes the PV system size for a Renewable Energy Community (ACC).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from breos.acc import calculate_variable_coefficients, calculate_acc_metrics
from breos.solar import calculate_multi_array_production, calculate_pv_production_dc
from breos.economics import calculate_costs
from breos import fetch_tmy_weather_data, load_profile
from pvlib.location import Location

def run_acc_sizer_logic(config: Dict[str, Any], verbose: bool = True) -> pd.DataFrame:
    """
    Core sizing logic. Sweeps PV sizes and calculates financial/energy metrics.
    """
    
    # 1. Setup Inputs ---------------------------------------------------------
    loc_cfg = config['location']
    location = Location(loc_cfg['latitude'], loc_cfg['longitude'], 
                       tz=loc_cfg.get('timezone', 'UTC'), name=loc_cfg.get('name', ''))
    
    # Weather
    if verbose: print("Fetching TMY Data...")
    tmy_data, _ = fetch_tmy_weather_data(loc_cfg['latitude'], loc_cfg['longitude'], freq='h')
    
    # Households (Load)
    households = config.get('households', [])
    if verbose: print(f"Loading profiles for {len(households)} households...")
    load_matrix = []
    for hh in households:
        # Load profile
        p = load_profile(str(hh.get('profile_type', '6')), hh.get('consumption', 3000), freq='h')
        load_matrix.append(p.values)
    load_matrix_np = np.column_stack(load_matrix)
    
    # Annual Total Load (for reference)
    total_community_load = np.sum(load_matrix_np) / 1000.0 # kWh
    
    # PV Arrays (Base Configuration for Geometry)
    base_arrays = config.get('pv_arrays', [])
    if not base_arrays:
        raise ValueError("No 'pv_arrays' defined in config for sizing geometry.")
    
    # 2. Pre-Calculate Unit PV Production (1 Module PER ARRAY) ----------------
    # We will assume we scale all arrays proportionally.
    # Eg. if Array1 has 1 mod, Array2 has 1 mod.
    # If user wants different ratios, they should specificy distinct arrays.
    # For sizing, we'll define "System Size" as Total Modules, distributed evenly or as per config ratios?
    # Simple approach: Keep the RATIO of modules in 'pv_arrays' constant, and scale the multiplier.
    
    # Let's calculate the DC output of the "Base Config" defined in JSON
    # And then sweep multipliers [0.1, 0.2 ... 5.0]
    
    if verbose: print("Calculating Base PV Production...")
    base_pv_curve = calculate_multi_array_production(
        weather_data=tmy_data,
        location=location,
        arrays=base_arrays,
        freq='h',
        verbose=False
    )
    # Ensure fillna (handled in recent fix, but good practice)
    base_pv_curve = base_pv_curve.fillna(0)
    
    # Base Module Count
    base_modules = sum([a.get('modules', 0) for a in base_arrays])
    if base_modules == 0:
        raise ValueError("Base configuration has 0 modules.")
        
    # 3. Sweep Sizes ----------------------------------------------------------
    # Defined in config or default
    sizer_cfg = config.get('sizer', {})
    min_scale = sizer_cfg.get('min_scale', 0.1)
    max_scale = sizer_cfg.get('max_scale', 3.0)
    step_scale = sizer_cfg.get('step_scale', 0.1)
    
    multipliers = np.arange(min_scale, max_scale + step_scale, step_scale)
    
    results = []
    
    # Cost Params
    costs_cfg = config.get('costs', {})
    financials_cfg = config.get('financials', {})
    elec_price_grid = financials_cfg.get('elec_price_grid', 0.16)
    elec_price_export = financials_cfg.get('elec_price_export', 0.05)
    
    # Baseline Cost (No Solar)
    # Simple assumption: Cost = Load * GridPrice
    baseline_bill = total_community_load * elec_price_grid
    
    if verbose: print(f"Sweeping {len(multipliers)} sizes (Scale {min_scale}x to {max_scale}x)...")
    
    idx = base_pv_curve.index
    # Align load length
    min_len = min(len(base_pv_curve), len(load_matrix_np))
    base_pv = base_pv_curve.values[:min_len]
    load_np = load_matrix_np[:min_len, :]
    
    # Pre-calc Dynamic Shares (Load dependent only, so constant across PV sizes!)
    # Dynamic strategy shares rely only on Load proportions.
    # share_i = load_i / total_load
    sharing_matrix = calculate_variable_coefficients(load_np)
    
    for m in multipliers:
        # Scale PV
        # Note: Inverter clipping is ignored here for speed. 
        # For a "Sizer", this is usually acceptable approximation until detailed design.
        current_pv = base_pv * m
        total_pv_kwh = np.sum(current_pv) / 1000.0
        n_modules_total = int(base_modules * m)
        
        # Calculate ACC Metrics (Energy Flows)
        # We reuse the logic from acc.py but optimized
        # Since shares are constant (Dynamic), we just scale allocated energy?
        # allocated = PV * shares. 
        # net = load - allocated.
        
        # Fast calc using acc.py
        metrics = calculate_acc_metrics(current_pv, load_np, sharing_matrix)
        
        total_import = metrics['total_import'] / 1000.0
        total_export = metrics['total_export'] / 1000.0
        
        # Financials
        # CAPEX: rough estimate
        # Assuming simple cost model: Fixed + Per_Module
        # We can use breos.economics if available or simple:
        # Cost = N_Modules * Cost_Per_Mod + Inverter + Balance
        # Let's use simple linear for sizer:
        cost_per_kwp = costs_cfg.get('cost_per_kwp', 800) # EUR
        # Assume 550W modules
        kwp = n_modules_total * 0.550
        capex = kwp * cost_per_kwp
        
        # Bill
        bill_grid = total_import * elec_price_grid
        bill_export_rev = total_export * elec_price_export
        net_bill = bill_grid - bill_export_rev
        
        annual_savings = baseline_bill - net_bill
        
        # Simple Payback
        payback = capex / annual_savings if annual_savings > 0 else 999
        
        results.append({
            'Scale': m,
            'Modules': n_modules_total,
            'System_kWp': kwp,
            'Capex_Eur': capex,
            'Total_Generation_MWh': total_pv_kwh / 1000.0,
            'Import_MWh': total_import / 1000.0,
            'Export_MWh': total_export / 1000.0,
            'Self_Sufficiency_Pct': metrics['self_sufficiency'],
            'Self_Consumption_Pct': metrics['self_consumption'],
            'Annual_Bill_Eur': net_bill,
            'Annual_Savings_Eur': annual_savings,
            'Payback_Years': payback
        })
        
    return pd.DataFrame(results)

