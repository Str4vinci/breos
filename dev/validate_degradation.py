"""
Battery Degradation Model Validation Tool.

Validates PVBAT degradation model against experimental datasets.

Usage:
    python tools/validate_degradation.py <dataset_path> --dataset-type zenodo_home --cell-id 01
    python tools/validate_degradation.py <dataset_path> --dataset-type zenodo_home --multi-system --calibrate
    python tools/validate_degradation.py <dataset_path> --dataset-type custom_csv --calibrate

Dataset types:
    zenodo_home  — Zenodo 21 home storage systems (12091223)
    zenodo_lfp   — Zenodo 28 LFP field systems
    hust         — HUST 77 LFP cells (lab cycling data, 1C-3C)
    calce        — CALCE A123 LFP cells (temperature characterization, NOT degradation)
    fastcharge   — Severson et al. 2019, 140 LFP cells (4-6C charge)
    custom_csv   — Generic CSV with columns: cycle_or_time, soh
"""

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breos.battery import (
    update_battery_soh_cyclewise,
    update_battery_soh_calendar,
    detect_cycles_rainflow,
    _get_degradation_params,
    k_c_rate_Q,
    k_doc_Q,
)
from breos.constants import Z_Q, R_GAS, T_REF_K
from breos.plotting import (
    plot_validation_soh_comparison,
    plot_validation_residuals,
    plot_validation_parity,
    plot_validation_multi_system,
    plot_validation_degradation_split,
    plot_loo_cv_summary,
    plot_loo_param_stability,
    plot_loo_predictions,
)


# =========================================================================
# Validation metrics
# =========================================================================

def compute_validation_metrics(
    predicted: np.ndarray,
    measured: np.ndarray,
) -> Dict[str, float]:
    """
    Compute validation metrics between predicted and measured SOH.

    Args:
        predicted: Predicted SOH values (fraction, 0-1)
        measured: Measured SOH values (fraction, 0-1)

    Returns:
        Dict with RMSE, MAE, MAPE, R2, Max_Error
    """
    residuals = predicted - measured
    n = len(residuals)

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    nonzero = measured != 0
    mape = np.mean(np.abs(residuals[nonzero] / measured[nonzero])) * 100 if nonzero.any() else float('inf')

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    max_error = np.max(np.abs(residuals))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Max_Error': max_error,
        'N_points': n,
    }


# =========================================================================
# Zenodo Home Storage — Metadata loaders
# =========================================================================

def _load_home_storage_metadata(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load system specs and capacity test timestamps from Zenodo home storage dataset.

    Returns:
        (systems_df, capacity_tests_df):
            systems_df has columns: ID, nominal_ah, nominal_v, nominal_kwh, chemistry, ...
            capacity_tests_df has columns: Start Time, ID
    """
    metadata_dir = os.path.join(dataset_path, 'Metadata_and_Code', '00_Data', '00_Metadata')

    systems_df = pd.read_excel(os.path.join(metadata_dir, 'Metadata_Systems.xlsx'))
    capacity_tests_df = pd.read_excel(os.path.join(metadata_dir, 'Capacity_Tests.xlsx'))

    # Convert unix timestamp dates to datetime
    for col in ['Date_storage_system_installation', 'Date_measurement_system_installation']:
        if col in systems_df.columns:
            systems_df[col] = pd.to_datetime(systems_df[col], unit='s', errors='coerce')

    return systems_df, capacity_tests_df


def _get_system_info(systems_df: pd.DataFrame, system_id: int) -> Dict:
    """Extract system info for a given ID."""
    row = systems_df[systems_df['ID'] == system_id]
    if row.empty:
        raise ValueError(f"System ID {system_id} not found in metadata")
    row = row.iloc[0]
    return {
        'id': system_id,
        'nominal_ah': row['Capacity_nominal_in_Ah'],
        'nominal_v': row['Voltage_nominal_in_V'],
        'nominal_kwh': row['Energy_nominal_in_kWh'],
        'usable_kwh': row['Energy_usable_datasheet_in_kWh'],
        'chemistry': row['Chemistry'],
        'chemistry_detail': row.get('Chemistry_detail', ''),
        'cells_series': row['Cell_number_in_series'],
        'cells_parallel': row['Cell_number_in_parallel'],
        'inverter_power_kw': row.get('Inverter_nominal_power', 0),
        'install_date': row.get('Date_storage_system_installation'),
        'measurement_start': row.get('Date_measurement_system_installation'),
    }


def _extract_capacity_from_test(
    dataset_path: str,
    system_id: int,
    test_start: pd.Timestamp,
    nominal_ah: float,
) -> Optional[float]:
    """
    Extract measured capacity (Ah) from raw data around a capacity test timestamp.

    The capacity test protocol is a controlled discharge at constant current
    starting near the test_start timestamp. We find the discharge segment,
    integrate |I| × dt to get Ah, and return SOH as fraction.

    Returns:
        Measured SOH as fraction (0-1), or None if extraction fails.
    """
    # Find the monthly CSV containing the test date
    sys_id_str = f"{system_id:02d}"
    year = test_start.year
    month = test_start.month
    csv_pattern = os.path.join(
        dataset_path, f'Data_ID_{sys_id_str}', sys_id_str,
        f'{year}_{month:02d}_System_ID_{sys_id_str}.csv'
    )
    matches = glob.glob(csv_pattern)
    if not matches:
        return None

    filepath = matches[0]

    # Read data around the test window (±6 hours)
    window_start = test_start - pd.Timedelta(hours=2)
    window_end = test_start + pd.Timedelta(hours=6)

    rows = []
    for chunk in pd.read_csv(filepath, chunksize=500_000,
                             parse_dates=['Time'], dayfirst=True,
                             usecols=['Time', 'I_in_A', 'V_in_V', 'P_in_W']):
        mask = (chunk['Time'] >= window_start) & (chunk['Time'] <= window_end)
        if mask.any():
            rows.append(chunk[mask])
    if not rows:
        return None

    df = pd.concat(rows).sort_values('Time').reset_index(drop=True)

    # Find the controlled discharge: starts near test_start, constant high current
    # Look for discharge (I < -1A) starting within ±15 min of test_start
    search_start = test_start - pd.Timedelta(minutes=15)
    search_end = test_start + pd.Timedelta(minutes=15)

    # Find first discharge sample near test start
    discharge_mask = (df['Time'] >= search_start) & (df['Time'] <= search_end) & (df['I_in_A'] < -1.0)
    if not discharge_mask.any():
        # Broaden search to ±30 min
        search_start = test_start - pd.Timedelta(minutes=30)
        search_end = test_start + pd.Timedelta(minutes=30)
        discharge_mask = (df['Time'] >= search_start) & (df['Time'] <= search_end) & (df['I_in_A'] < -1.0)
        if not discharge_mask.any():
            return None

    dis_start_idx = discharge_mask.idxmax()
    dis_start_time = df.loc[dis_start_idx, 'Time']

    # Find end of discharge: first time current goes > -0.5A after discharge start
    # (allowing brief interruptions up to 60 seconds)
    after_start = df[df.index >= dis_start_idx].copy()

    # Find the main discharge block: continuous segment where most seconds have I < -0.5A
    # Use a rolling window to detect end of sustained discharge
    idle_count = 0
    dis_end_idx = after_start.index[-1]
    for idx, row in after_start.iterrows():
        if row['I_in_A'] > -0.5:
            idle_count += 1
            if idle_count > 120:  # 2 minutes of non-discharge = test ended
                dis_end_idx = idx - idle_count
                break
        else:
            idle_count = 0

    # Extract discharge segment
    discharge = df.loc[dis_start_idx:dis_end_idx]
    discharge = discharge[discharge['I_in_A'] < -0.1]

    if len(discharge) < 60:  # Need at least 60 seconds of data
        return None

    # Integrate |I| × dt (1-second sampling assumed)
    # Use actual time deltas for robustness
    times = discharge['Time'].values
    currents = np.abs(discharge['I_in_A'].values)

    dt_seconds = np.diff(times).astype('timedelta64[s]').astype(float)
    # Trapezoidal integration
    measured_ah = np.sum((currents[:-1] + currents[1:]) / 2 * dt_seconds) / 3600.0

    soh = measured_ah / nominal_ah
    # Sanity check: SOH should be between 0.5 and 1.2
    if soh < 0.5 or soh > 1.2:
        return None

    return soh


def _get_capacity_test_soh(
    dataset_path: str,
    system_id: int,
    capacity_tests_df: pd.DataFrame,
    nominal_ah: float,
) -> pd.DataFrame:
    """
    Extract SOH measurements from all capacity tests for a system.

    Returns:
        DataFrame with columns: date, measured_soh
    """
    tests = capacity_tests_df[capacity_tests_df['ID'] == system_id].copy()
    tests = tests.sort_values('Start Time')

    results = []
    for _, row in tests.iterrows():
        test_time = row['Start Time']
        soh = _extract_capacity_from_test(dataset_path, system_id, test_time, nominal_ah)
        if soh is not None:
            results.append({
                'date': test_time,
                'measured_soh': soh,
            })

    return pd.DataFrame(results)


# =========================================================================
# Zenodo Home Storage — Aggregation with caching
# =========================================================================

VALID_RESOLUTIONS = ('h', '15min')

def _aggregate_home_storage_system(
    dataset_path: str,
    system_id: int,
    nominal_kwh: float,
    cells_series: int = 16,
    nominal_v: float = 51.2,
    cache_dir: str = 'results/validation_cache',
    resolution: str = '15min',
) -> pd.DataFrame:
    """
    Aggregate 1-second home storage data to a target resolution.

    Reads monthly CSVs in chunks, resamples to the target resolution, and
    caches as parquet. SOC is reconstructed using cumulative Ah integration
    with voltage-based resets when the system voltage hits near-empty or
    near-full thresholds.

    Args:
        dataset_path: Path to dataset root
        system_id: System ID number
        nominal_kwh: Nominal energy capacity in kWh
        cells_series: Number of cells in series (for per-cell voltage)
        nominal_v: Nominal system voltage (V)
        cache_dir: Directory for parquet cache files
        resolution: Resample frequency ('h' for hourly, '15min' for 15-minute)

    Returns:
        DataFrame indexed by datetime with columns:
            mean_temp_C, mean_room_temp_C, power_w, energy_throughput_wh,
            charge_ah, discharge_ah, v_min, v_max, v_mean, soc_estimated
    """
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {VALID_RESOLUTIONS}, got '{resolution}'")

    res_label = resolution.replace('min', 'min')  # 'h' or '15min'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{res_label}_system_{system_id:02d}.parquet')

    if os.path.exists(cache_path):
        print(f"   Loading cached {res_label} data: {cache_path}")
        return pd.read_parquet(cache_path)

    sys_id_str = f"{system_id:02d}"
    data_dir = os.path.join(dataset_path, f'Data_ID_{sys_id_str}', sys_id_str)

    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"   Processing {len(csv_files)} monthly files for System {system_id} "
          f"(resolution: {res_label})...")

    resampled_frames = []
    for i, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"      [{i+1}/{len(csv_files)}] {filename}")

        monthly_resampled = []
        for chunk in pd.read_csv(
            filepath, chunksize=200_000,
            parse_dates=['Time'], dayfirst=True,
            usecols=['Time', 'P_in_W', 'I_in_A', 'V_in_V', 'T_Bat_in_C', 'T_Room_in_C'],
            na_values=['NaN', 'nan', ''],
        ):
            chunk = chunk.dropna(subset=['Time'])
            chunk = chunk.set_index('Time')
            # Fill NaN values for temperature with forward fill
            chunk['T_Bat_in_C'] = chunk['T_Bat_in_C'].ffill()
            chunk['T_Room_in_C'] = chunk['T_Room_in_C'].ffill()
            chunk['P_in_W'] = chunk['P_in_W'].fillna(0.0)
            chunk['I_in_A'] = chunk['I_in_A'].fillna(0.0)
            chunk['V_in_V'] = chunk['V_in_V'].ffill()

            # Resample to target resolution within this chunk
            resampled = pd.DataFrame()
            resampled['mean_temp_C'] = chunk['T_Bat_in_C'].resample(resolution).mean()
            resampled['mean_room_temp_C'] = chunk['T_Room_in_C'].resample(resolution).mean()
            resampled['power_w'] = chunk['P_in_W'].resample(resolution).mean()
            # Energy throughput: sum of |P| × dt, where dt = 1 second, convert to Wh
            resampled['energy_throughput_wh'] = chunk['P_in_W'].abs().resample(resolution).sum() / 3600.0
            # Charge and discharge Ah for SOC tracking
            resampled['charge_ah'] = chunk['I_in_A'].clip(lower=0).resample(resolution).sum() / 3600.0
            resampled['discharge_ah'] = chunk['I_in_A'].clip(upper=0).abs().resample(resolution).sum() / 3600.0
            # Voltage statistics for SOC resets
            resampled['v_min'] = chunk['V_in_V'].resample(resolution).min()
            resampled['v_max'] = chunk['V_in_V'].resample(resolution).max()
            resampled['v_mean'] = chunk['V_in_V'].resample(resolution).mean()

            monthly_resampled.append(resampled)

        if monthly_resampled:
            month_df = pd.concat(monthly_resampled)
            # Re-aggregate in case chunk boundaries split intervals
            month_agg = month_df.groupby(month_df.index).agg({
                'mean_temp_C': 'mean',
                'mean_room_temp_C': 'mean',
                'power_w': 'mean',
                'energy_throughput_wh': 'sum',
                'charge_ah': 'sum',
                'discharge_ah': 'sum',
                'v_min': 'min',
                'v_max': 'max',
                'v_mean': 'mean',
            })
            resampled_frames.append(month_agg)

    if not resampled_frames:
        raise ValueError(f"No data loaded for System {system_id}")

    result = pd.concat(resampled_frames).sort_index()

    # Remove duplicate indices (overlap between monthly files)
    result = result[~result.index.duplicated(keep='first')]

    # SOC reconstruction using cumulative Ah + voltage-based resets
    # LFP cell voltage thresholds (per-cell):
    #   ~2.8 V/cell → near empty (~5% SOC)
    #   ~3.5 V/cell → near full (~95% SOC)
    v_empty_system = cells_series * 2.8   # system-level empty threshold
    v_full_system = cells_series * 3.5    # system-level full threshold
    nominal_ah = nominal_kwh * 1000.0 / nominal_v

    result['net_ah'] = result['charge_ah'] - result['discharge_ah']
    soc_list = []
    current_soc = 0.5
    for idx, row in result.iterrows():
        # Voltage-based SOC resets
        if row['v_min'] <= v_empty_system:
            current_soc = 0.05
        elif row['v_max'] >= v_full_system:
            current_soc = 0.95

        delta_soc = row['net_ah'] / nominal_ah
        current_soc = np.clip(current_soc + delta_soc, 0.0, 1.0)
        soc_list.append(current_soc)
    result['soc_estimated'] = soc_list

    # Cache to parquet
    result.to_parquet(cache_path)
    print(f"   Cached to: {cache_path} ({len(result)} {res_label} rows)")

    return result


# =========================================================================
# Field data simulation engine
# =========================================================================

def simulate_on_field_data(
    timeseries_df: pd.DataFrame,
    calendar_model: str = 'lam_calibrated',
    nominal_energy_wh: float = 5000.0,
    custom_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Simulate degradation using measured field data.

    Processes data day-by-day:
    - Extracts daily SOC profile for rainflow cycle counting
    - Computes cycle aging from rainflow cycles
    - Computes calendar aging from daily mean temperature and SOC

    Works with any sub-daily resolution (hourly, 15-minute, etc.).
    Higher resolution provides more SOC data points per day for
    rainflow cycle counting, potentially capturing more sub-hourly cycles.

    Args:
        timeseries_df: Aggregated DataFrame (hourly or 15-min) with columns:
            mean_temp_C, energy_throughput_wh, soc_estimated
        calendar_model: 'naumann', 'lam', or 'lam_calibrated'
        nominal_energy_wh: Nominal system capacity (Wh)
        custom_params: Optional dict with 'k0_frac', 'Ea', 'cal_b', 'n'

    Returns:
        DataFrame with columns: date, predicted_soh, fec_cumulative, cal_loss, cycle_loss
    """
    if custom_params:
        k0_frac = custom_params['k0_frac']
        Ea = custom_params['Ea']
        cal_b = custom_params['cal_b']
        n_soc = custom_params['n']
    else:
        k0_frac, Ea, cal_b, n_soc = _get_degradation_params(calendar_model)

    soh = 1.0
    fec_cum = 0.0
    cal_seconds = 0.0
    total_cal_loss = 0.0
    total_cycle_loss = 0.0

    results = []

    # Group by day
    timeseries_df = timeseries_df.copy()
    timeseries_df['date'] = timeseries_df.index.date

    for date, day_data in timeseries_df.groupby('date'):
        # --- Cycle aging ---
        # Use rainflow on daily SOC profile
        soc_series = day_data['soc_estimated']
        if len(soc_series) >= 2 and soc_series.std() > 0.001:
            time_idx = pd.DatetimeIndex(day_data.index)
            cycles = detect_cycles_rainflow(soc_series, time_idx, min_doc_fraction=0.01)

            for cyc in cycles:
                doc = cyc['doc']
                c_rate = cyc['mean_c_rate']
                fec_delta = doc * cyc['count']
                fec_new = fec_cum + fec_delta

                kC = k_c_rate_Q(c_rate)
                kDOC = k_doc_Q(doc)
                dq_pct = kC * kDOC * (fec_new ** Z_Q - fec_cum ** Z_Q)
                d_cycle = dq_pct / 100.0

                soh -= d_cycle
                total_cycle_loss += d_cycle
                fec_cum = fec_new

        # --- Calendar aging ---
        mean_temp = day_data['mean_temp_C'].mean()
        mean_soc = day_data['soc_estimated'].mean()
        if np.isnan(mean_temp):
            mean_temp = 25.0
        if np.isnan(mean_soc):
            mean_soc = 0.5

        soh_after, d_cal, cal_seconds = update_battery_soh_calendar(
            soh, k0_frac, Ea, n_soc, cal_b,
            T_cell_C=mean_temp,
            cumulative_cal_seconds=cal_seconds,
            dt_days=1.0,
            mean_soc_absolute=mean_soc,
        )
        total_cal_loss += d_cal
        soh = soh_after

        soh = max(0.0, soh)

        results.append({
            'date': pd.Timestamp(date),
            'predicted_soh': soh,
            'fec_cumulative': fec_cum,
            'cal_loss': total_cal_loss,
            'cycle_loss': total_cycle_loss,
            'mean_temp_C': mean_temp,
        })

    return pd.DataFrame(results)


# =========================================================================
# Dataset loaders
# =========================================================================

def load_zenodo_home_dataset(
    path: str,
    system_id: Optional[str] = None,
    cache_dir: str = 'results/validation_cache',
    chemistry_filter: str = 'all',
    resolution: str = '15min',
) -> Dict:
    """
    Load Zenodo home storage dataset for validation.

    Args:
        path: Path to 12091223 dataset directory
        system_id: System ID (1-21) as string. If None, lists available systems.
        cache_dir: Directory for caching aggregated data
        chemistry_filter: Filter by chemistry ('LFP', 'NMC', 'LMO', 'all')
        resolution: Resample frequency ('h' for hourly, '15min' for 15-minute)

    Returns:
        Dict with keys:
            'system_info': system specs dict
            'timeseries_df': aggregated DataFrame at the requested resolution
            'soh_ground_truth': DataFrame with date, measured_soh
            'available_systems': list of system IDs (always present)
    """
    systems_df, capacity_tests_df = _load_home_storage_metadata(path)

    if chemistry_filter != 'all':
        systems_df = systems_df[systems_df['Chemistry'] == chemistry_filter]

    available = sorted(systems_df['ID'].unique())

    if system_id is None:
        filter_label = f" ({chemistry_filter})" if chemistry_filter != 'all' else ""
        print(f"Available systems{filter_label}: {available}")
        print("\nSystem details:")
        for _, row in systems_df.iterrows():
            sid = row['ID']
            n_tests = len(capacity_tests_df[capacity_tests_df['ID'] == sid])
            print(f"  ID {sid:2d}: {row['Energy_nominal_in_kWh']:.1f} kWh, "
                  f"{row['Chemistry']}/{row.get('Chemistry_detail', '?')}, "
                  f"{n_tests} capacity tests")
        return {'available_systems': available}

    sid = int(system_id)
    sys_info = _get_system_info(systems_df, sid)
    print(f"\n--- System {sid}: {sys_info['nominal_kwh']:.1f} kWh "
          f"{sys_info['chemistry']}/{sys_info['chemistry_detail']} ---")

    # Extract SOH from capacity tests
    print(f"   Extracting SOH from capacity tests...")
    soh_truth = _get_capacity_test_soh(path, sid, capacity_tests_df, sys_info['nominal_ah'])
    if soh_truth.empty:
        print(f"   WARNING: No valid capacity tests extracted for System {sid}")
    else:
        for _, row in soh_truth.iterrows():
            print(f"      {row['date'].strftime('%Y-%m-%d')}: SOH = {row['measured_soh']*100:.1f}%")

    # Aggregate to target resolution
    print(f"   Aggregating data (resolution: {resolution})...")
    timeseries_df = _aggregate_home_storage_system(
        path, sid, sys_info['nominal_kwh'],
        cells_series=sys_info['cells_series'],
        nominal_v=sys_info['nominal_v'],
        cache_dir=cache_dir,
        resolution=resolution,
    )

    return {
        'system_info': sys_info,
        'timeseries_df': timeseries_df,
        'soh_ground_truth': soh_truth,
        'available_systems': available,
    }


def load_zenodo_lfp_field(path: str, system_id: Optional[str] = None) -> pd.DataFrame:
    """Load Zenodo 28 LFP field systems dataset."""
    raise NotImplementedError("Zenodo LFP field dataset loader not yet implemented.")


def _parse_hust_readme(path: str) -> Dict[int, Dict]:
    """Parse HUST Readme.txt for per-cell charge/discharge rates."""
    readme_path = os.path.join(path, 'Readme.txt')
    cell_info = {}
    if not os.path.exists(readme_path):
        return cell_info
    with open(readme_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Battery'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                cell_num = int(parts[0].replace('#', ''))
                charge_rate = parts[1].strip()
                discharge_rate = parts[2].strip()
                # Convert C-rate strings to float
                dis_c = 3.0  # default
                if discharge_rate.endswith('C'):
                    try:
                        dis_c = float(discharge_rate.replace('C', ''))
                    except ValueError:
                        pass
                chg_c = None
                if charge_rate == 'Random':
                    chg_c = 2.0  # mean of 1C/2C/3C
                elif charge_rate.endswith('C'):
                    try:
                        chg_c = float(charge_rate.replace('C', ''))
                    except ValueError:
                        chg_c = 2.0
                cell_info[cell_num] = {
                    'charge_rate_str': charge_rate,
                    'discharge_rate_str': discharge_rate,
                    'c_rate_discharge': dis_c,
                    'c_rate_charge': chg_c or 2.0,
                }
    return cell_info


def load_hust_dataset(
    path: str,
    cell_id: Optional[str] = None,
) -> Dict:
    """
    Load HUST 77 LFP cells dataset.

    Each cell has cycling Excel files with per-second data. Extracts discharge
    capacity per cycle to build SOH trajectory.

    Args:
        path: Path to kw34hhw7xg-3 directory
        cell_id: Cell number as string (e.g., '1'). If None, lists available cells.

    Returns:
        Dict with:
            'cell_id': str
            'cycles_df': DataFrame with cycle, discharge_capacity_ah, temperature_C
            'c_rate_discharge': float
            'c_rate_charge': float
            'nominal_ah': 1.1
            'chemistry': 'LFP'
            'profile_type': 'fixed' or 'arbitrary'
            'available_cells': list of cell IDs (always present)
    """
    # LR1865SZ cells — nominal capacity ~2 Ah (varies by cell).
    # We use the first few cycling cycles (2-5) as the nominal reference,
    # since cycle 1 is often a characterization discharge at different rate.
    NOMINAL_AH_FALLBACK = 2.0

    cell_info = _parse_hust_readme(path)

    # Scan both subdirectories for cell folders
    subdirs = {
        'Cycled with Fixed Current Profiles': 'fixed',
        'Cycled with Arbitrary Uses Profiles': 'arbitrary',
    }
    cell_paths = {}  # cell_num -> (full_path, profile_type)
    for subdir, ptype in subdirs.items():
        subdir_path = os.path.join(path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for entry in os.listdir(subdir_path):
            if entry.startswith('#'):
                cell_num = int(entry.replace('#', ''))
                cell_dir = os.path.join(subdir_path, entry)
                # Check if has cycling data (not just characterization)
                files = os.listdir(cell_dir)
                has_cycling = any(
                    f.endswith('.xlsx') and 'first20' not in f.lower()
                    for f in files
                )
                if has_cycling:
                    cell_paths[cell_num] = (cell_dir, ptype)

    available = sorted(cell_paths.keys())

    if cell_id is None:
        print(f"Available HUST cells with cycling data: {len(available)}")
        for cn in available:
            info = cell_info.get(cn, {})
            _, ptype = cell_paths[cn]
            print(f"  #{cn}: {ptype}, charge={info.get('charge_rate_str', '?')}, "
                  f"discharge={info.get('discharge_rate_str', '?')}")
        return {'available_cells': available}

    cn = int(cell_id)
    if cn not in cell_paths:
        raise ValueError(f"Cell #{cn} not found or has no cycling data. "
                         f"Available: {available}")

    cell_dir, profile_type = cell_paths[cn]
    info = cell_info.get(cn, {'c_rate_discharge': 3.0, 'c_rate_charge': 2.0})

    # Find cycling Excel files (exclude first20cycle characterization)
    xlsx_files = sorted([
        os.path.join(cell_dir, f) for f in os.listdir(cell_dir)
        if f.endswith('.xlsx') and 'first20' not in f.lower()
    ])

    print(f"   Loading HUST cell #{cn}: {len(xlsx_files)} cycling file(s), "
          f"{profile_type} profile")

    all_cycle_data = []
    for filepath in xlsx_files:
        print(f"      Reading: {os.path.basename(filepath)}")
        df = pd.read_excel(filepath)

        # Group by Cycle_Index, get max Capacity (which is discharge capacity)
        for cyc_idx, cyc_data in df.groupby('Cycle_Index'):
            # Max Capacity(Ah) in a cycle = total discharge capacity
            dis_cap = cyc_data['Capacity(Ah)'].max()

            # Mean temperature
            temp = cyc_data['Temperature(℃)'].mean() if 'Temperature(℃)' in cyc_data.columns else 30.0

            # Duration: last - first Test_Time
            if 'Date_Time' in cyc_data.columns:
                times = pd.to_datetime(cyc_data['Date_Time'])
                duration_s = (times.max() - times.min()).total_seconds()
            else:
                duration_s = 0

            if dis_cap > 0.01:  # Skip near-zero capacity cycles
                all_cycle_data.append({
                    'cycle': int(cyc_idx),
                    'discharge_capacity_ah': dis_cap,
                    'temperature_C': temp,
                    'duration_s': duration_s,
                    'c_rate_discharge': info['c_rate_discharge'],
                    'c_rate_charge': info['c_rate_charge'],
                })

    if not all_cycle_data:
        raise ValueError(f"No valid cycle data found for cell #{cn}")

    cycles_df = pd.DataFrame(all_cycle_data).sort_values('cycle').reset_index(drop=True)
    # Re-number cycles sequentially (some files may overlap or have gaps)
    cycles_df['cycle'] = range(1, len(cycles_df) + 1)

    # Derive nominal capacity from early cycling cycles (2-5), skipping
    # cycle 1 which is often a characterization discharge at different rate
    if len(cycles_df) >= 5:
        nominal_ah = cycles_df.iloc[1:5]['discharge_capacity_ah'].median()
    elif len(cycles_df) >= 2:
        nominal_ah = cycles_df.iloc[1]['discharge_capacity_ah']
    else:
        nominal_ah = cycles_df.iloc[0]['discharge_capacity_ah']
    # Sanity bound
    nominal_ah = max(nominal_ah, NOMINAL_AH_FALLBACK * 0.5)

    print(f"      {len(cycles_df)} cycles loaded, "
          f"cap range: {cycles_df['discharge_capacity_ah'].min():.3f} - "
          f"{cycles_df['discharge_capacity_ah'].max():.3f} Ah, "
          f"nominal={nominal_ah:.3f} Ah")

    return {
        'cell_id': str(cn),
        'cycles_df': cycles_df,
        'c_rate_discharge': info['c_rate_discharge'],
        'c_rate_charge': info['c_rate_charge'],
        'nominal_ah': nominal_ah,
        'chemistry': 'LFP',
        'profile_type': profile_type,
        'available_cells': available,
    }


def load_calce_lfp(
    path: str,
    cell_id: Optional[str] = None,
) -> Dict:
    """
    Load CALCE A123 LFP characterization dataset.

    NOTE: This is a characterization dataset (DST/FUDS drive cycles at different
    temperatures), NOT a cycling degradation dataset. Each file contains only
    a few drive cycles at a single temperature. Useful for validating
    temperature-dependent capacity but NOT for degradation trajectory validation.

    Args:
        path: Path to CAUCE_A123 directory
        cell_id: Cell identifier (e.g., 'A1-007'). If None, lists available cells.

    Returns:
        Dict with:
            'cell_id': str
            'temperature_capacity': DataFrame with temperature_C, discharge_capacity_ah
            'nominal_ah': 1.1
            'chemistry': 'LFP'
            'available_cells': list of cell IDs
            'is_characterization': True (flag that this is NOT degradation data)
    """
    NOMINAL_AH = 1.1

    # Scan DST-FUDS temperature folders
    temp_folders = {}
    for entry in sorted(os.listdir(path)):
        if not entry.startswith('A123_DST-US06-FUDS-'):
            continue
        # Parse temperature from folder name
        suffix = entry.replace('A123_DST-US06-FUDS-', '')
        if suffix.startswith('N'):
            temp_c = -int(suffix[1:])
        else:
            try:
                temp_c = int(suffix)
            except ValueError:
                continue
        inner_dir = os.path.join(path, entry)
        # Find the inner subfolder
        for sub in os.listdir(inner_dir):
            sub_path = os.path.join(inner_dir, sub)
            if os.path.isdir(sub_path):
                temp_folders[temp_c] = sub_path
                break

    # Discover all cell IDs
    all_cells = set()
    for temp_c, folder in temp_folders.items():
        for f in os.listdir(folder):
            if f.endswith('.xlsx') and not f.startswith('~$'):
                # Extract cell ID: A1-007 from A1-007-DST-US06-FUDS-25-20120827.xlsx
                parts = f.split('-DST-')
                if parts:
                    all_cells.add(parts[0])

    available = sorted(all_cells)

    if cell_id is None:
        print(f"Available CALCE cells: {available}")
        print(f"Temperatures: {sorted(temp_folders.keys())} °C")
        print("NOTE: This is a characterization dataset, NOT cycling degradation.")
        return {'available_cells': available, 'is_characterization': True}

    if cell_id not in available:
        raise ValueError(f"Cell {cell_id} not found. Available: {available}")

    print(f"   Loading CALCE cell {cell_id} across {len(temp_folders)} temperatures")
    print(f"   NOTE: Characterization data (temperature-capacity), not degradation.")

    temp_cap_data = []
    for temp_c in sorted(temp_folders.keys()):
        folder = temp_folders[temp_c]
        # Find Excel file for this cell
        xlsx_files = [
            f for f in os.listdir(folder)
            if f.startswith(cell_id) and f.endswith('.xlsx') and not f.startswith('~$')
            and 'newprofile' not in f.lower()  # skip alternate profiles
        ]
        if not xlsx_files:
            continue

        filepath = os.path.join(folder, xlsx_files[0])
        print(f"      {temp_c}°C: {xlsx_files[0]}")

        try:
            df = pd.read_excel(filepath, sheet_name=1)
        except Exception as e:
            print(f"         Error reading: {e}")
            continue

        if 'Discharge_Capacity(Ah)' in df.columns:
            # Discharge_Capacity is cumulative across the entire file.
            # Extract per-block capacity: find steps with significant discharge
            # (DST/FUDS blocks) by looking for discharge capacity jumps > 0.5 Ah.
            block_caps = []
            for si in sorted(df['Step_Index'].unique()):
                step = df[df['Step_Index'] == si]
                dis_start = step['Discharge_Capacity(Ah)'].iloc[0]
                dis_end = step['Discharge_Capacity(Ah)'].iloc[-1]
                delta = dis_end - dis_start
                if delta > 0.5:  # significant discharge block
                    block_caps.append(delta)
            # Use the first discharge block capacity (most reliable, fresh cell)
            if block_caps:
                max_dis_cap = block_caps[0]
            else:
                max_dis_cap = 0.0
        else:
            # Integrate |Current| for discharge segments
            dis_mask = df['Current(A)'] < -0.05
            if dis_mask.any():
                dis_data = df[dis_mask]
                times = dis_data['Test_Time(s)'].values
                currents = np.abs(dis_data['Current(A)'].values)
                if len(times) > 1:
                    dt = np.diff(times)
                    max_dis_cap = np.sum((currents[:-1] + currents[1:]) / 2 * dt) / 3600.0
                else:
                    max_dis_cap = 0.0
            else:
                max_dis_cap = 0.0

        measured_temp = temp_c
        if 'Temperature (C)_1' in df.columns:
            measured_temp = df['Temperature (C)_1'].mean()

        if max_dis_cap > 0.01:
            temp_cap_data.append({
                'temperature_C': temp_c,
                'measured_temperature_C': measured_temp,
                'discharge_capacity_ah': max_dis_cap,
                'soh': max_dis_cap / NOMINAL_AH,
            })

    temp_cap_df = pd.DataFrame(temp_cap_data)
    print(f"      Loaded {len(temp_cap_df)} temperature points")

    return {
        'cell_id': cell_id,
        'temperature_capacity': temp_cap_df,
        'nominal_ah': NOMINAL_AH,
        'chemistry': 'LFP',
        'available_cells': available,
        'is_characterization': True,
    }


def load_fastcharge_dataset(
    path: str,
    cell_id: Optional[str] = None,
) -> Dict:
    """
    Load FastCharge dataset (Severson et al. 2019) — 140 LFP cells.

    JSON structure files with per-cycle summary data including discharge capacity
    and temperature. High C-rate (4-6C charge), tests model extrapolation.

    Args:
        path: Path to FastCharge directory containing JSON files
        cell_id: Cell index as string (0-139). If None, lists available cells.

    Returns:
        Dict with:
            'cell_id': str
            'cycles_df': DataFrame with cycle, discharge_capacity_ah, temperature_C
            'nominal_ah': float (from cycle 2 capacity)
            'chemistry': 'LFP'
            'barcode': str
            'protocol': str
            'available_cells': list of cell indices
    """
    # Discover all JSON files
    json_files = sorted([
        f for f in os.listdir(path)
        if f.endswith('_structure.json')
    ])

    if not json_files:
        raise FileNotFoundError(f"No FastCharge JSON files found in {path}")

    available = [str(i) for i in range(len(json_files))]

    if cell_id is None:
        print(f"Available FastCharge cells: {len(json_files)}")
        # Show first few
        for i, fname in enumerate(json_files[:10]):
            print(f"  {i}: {fname}")
        if len(json_files) > 10:
            print(f"  ... ({len(json_files) - 10} more)")
        return {'available_cells': available}

    idx = int(cell_id)
    if idx < 0 or idx >= len(json_files):
        raise ValueError(f"Cell index {idx} out of range [0, {len(json_files)-1}]")

    filepath = os.path.join(path, json_files[idx])
    print(f"   Loading FastCharge cell {idx}: {json_files[idx]}")

    with open(filepath) as f:
        data = json.load(f)

    summary = data['summary']
    cycle_indices = summary['cycle_index']
    discharge_caps = summary['discharge_capacity']
    temps = summary['temperature_average']

    # Skip cycle 0 (formation cycle with anomalous capacity)
    start_idx = 1 if cycle_indices[0] == 0 else 0

    # Nominal capacity from cycle ~2 (first normal cycle)
    nominal_ah = discharge_caps[start_idx]

    # Build per-cycle DataFrame
    cycles_data = []
    for i in range(start_idx, len(cycle_indices)):
        cap = discharge_caps[i]
        if cap <= 0:
            continue
        cycles_data.append({
            'cycle': cycle_indices[i],
            'discharge_capacity_ah': cap,
            'temperature_C': temps[i] if i < len(temps) else 30.0,
        })

    cycles_df = pd.DataFrame(cycles_data)

    # Extract charge duration for C-rate estimation
    if 'charge_duration' in summary:
        durations = summary['charge_duration']
        charge_durations = [durations[i] for i in range(start_idx, len(durations))]
        if charge_durations and len(charge_durations) == len(cycles_df):
            cycles_df['charge_duration_s'] = charge_durations
            # C-rate = capacity / (duration_h * nominal)
            cycles_df['c_rate_charge'] = (
                cycles_df['discharge_capacity_ah'] /
                (cycles_df['charge_duration_s'] / 3600.0 * nominal_ah)
            ).clip(upper=10.0)

    print(f"      {len(cycles_df)} cycles, nominal={nominal_ah:.3f} Ah, "
          f"SOH range: {cycles_df['discharge_capacity_ah'].min()/nominal_ah*100:.1f}% - "
          f"{cycles_df['discharge_capacity_ah'].max()/nominal_ah*100:.1f}%")
    print(f"      Barcode: {data.get('barcode', 'N/A')}, "
          f"Protocol: {data.get('protocol', 'N/A')}")

    return {
        'cell_id': str(idx),
        'cycles_df': cycles_df,
        'nominal_ah': nominal_ah,
        'chemistry': 'LFP',
        'barcode': data.get('barcode', ''),
        'protocol': data.get('protocol', ''),
        'available_cells': available,
    }


def load_custom_csv(path: str) -> pd.DataFrame:
    """
    Load a generic CSV with degradation measurements.
    Expected: cycle_or_time, soh columns. SOH as fraction (0-1).
    """
    df = pd.read_csv(path)

    required = {'cycle_or_time', 'soh'}
    missing = required - set(df.columns)
    if missing:
        renames = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ('cycle', 'cycles', 'time', 'day', 'days', 'hours'):
                renames[col] = 'cycle_or_time'
            elif cl in ('soh', 'capacity_fraction', 'relative_capacity'):
                renames[col] = 'soh'
        df = df.rename(columns=renames)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV must contain columns {required}. "
            f"Found: {list(df.columns)}. Missing: {missing}"
        )
    return df


# =========================================================================
# Simulation runners
# =========================================================================

def simulate_on_lab_protocol(
    measured_df: pd.DataFrame,
    calendar_model: str = 'lam_calibrated',
    nominal_energy_wh: float = 1000.0,
    temperature: float = 25.0,
    c_rate: float = 1.0,
    doc: float = 1.0,
    custom_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Simulate degradation on a lab cycling protocol.
    """
    if custom_params:
        k0_frac = custom_params.get('k0_frac')
        Ea = custom_params.get('Ea')
        cal_b = custom_params.get('cal_b')
        n_soc = custom_params.get('n')
    else:
        k0_frac, Ea, cal_b, n_soc = _get_degradation_params(calendar_model)

    x_values = measured_df['cycle_or_time'].values
    measured_soh = measured_df['soh'].values

    predicted = []
    soh_frac = 1.0
    fec_cum = 0.0
    cal_seconds = 0.0

    if 'temperature' in measured_df.columns:
        temperature = measured_df['temperature'].iloc[0]

    prev_x = 0.0
    for x_val in x_values:
        delta_x = x_val - prev_x
        if delta_x <= 0:
            predicted.append(soh_frac)
            prev_x = x_val
            continue

        dFEC = delta_x * doc
        fec_new = fec_cum + dFEC

        kC = k_c_rate_Q(c_rate)
        kDOC = k_doc_Q(doc)
        dq_percent = kC * kDOC * (fec_new ** Z_Q - fec_cum ** Z_Q)
        dq_frac = dq_percent / 100.0

        fec_cum = fec_new
        soh_frac -= dq_frac

        hours_per_cycle = 2.0 * doc / c_rate if c_rate > 0 else 2.0
        dt_seconds = delta_x * hours_per_cycle * 3600.0
        cal_seconds += dt_seconds

        T_K = temperature + 273.15
        arr = math.exp(-Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF_K))
        t_old_b = math.pow(cal_seconds - dt_seconds, cal_b) if (cal_seconds - dt_seconds) > 0 else 0.0
        t_new_b = math.pow(cal_seconds, cal_b)
        soc_stress = 0.5 ** n_soc
        d_cal = k0_frac * arr * (t_new_b - t_old_b) * soc_stress
        soh_frac -= d_cal

        soh_frac = max(0.0, soh_frac)
        predicted.append(soh_frac)
        prev_x = x_val

    return pd.DataFrame({
        'cycle_or_time': x_values,
        'predicted_soh': predicted,
        'measured_soh': measured_soh,
    })


# =========================================================================
# Lab cycle simulation engine
# =========================================================================

def simulate_on_lab_cycles(
    cycles_df: pd.DataFrame,
    calendar_model: str = 'lam_calibrated',
    nominal_ah: float = 1.1,
    temperature_C: float = 25.0,
    custom_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Simulate degradation using per-cycle lab data.

    For each cycle: apply one cycle of cycle aging + proportional calendar aging
    based on the cycle duration.

    Args:
        cycles_df: DataFrame with columns:
            - cycle: cycle number
            - discharge_capacity_ah: measured discharge capacity
            - temperature_C (optional): temperature per cycle (overrides temperature_C arg)
            - charge_capacity_ah (optional): charge capacity
            - duration_s (optional): total cycle duration in seconds
            - c_rate_charge (optional): charge C-rate
            - c_rate_discharge (optional): discharge C-rate
        calendar_model: Degradation model name
        nominal_ah: Nominal cell capacity (Ah)
        temperature_C: Default temperature if not in DataFrame
        custom_params: Optional dict with k0_frac, Ea, cal_b, n

    Returns:
        DataFrame with cycle, predicted_soh, measured_soh, fec_cumulative,
        cal_loss, cycle_loss
    """
    if custom_params:
        k0_frac = custom_params['k0_frac']
        Ea = custom_params['Ea']
        cal_b = custom_params['cal_b']
        n_soc = custom_params['n']
    else:
        k0_frac, Ea, cal_b, n_soc = _get_degradation_params(calendar_model)

    soh = 1.0
    fec_cum = 0.0
    cal_seconds = 0.0
    total_cal_loss = 0.0
    total_cycle_loss = 0.0

    has_temp = 'temperature_C' in cycles_df.columns
    has_duration = 'duration_s' in cycles_df.columns
    has_c_rate_dis = 'c_rate_discharge' in cycles_df.columns
    has_c_rate_chg = 'c_rate_charge' in cycles_df.columns

    results = []
    prev_cycle = 0

    for _, row in cycles_df.iterrows():
        cycle_num = row['cycle']
        dis_cap = row['discharge_capacity_ah']

        # SOH from data
        measured_soh = dis_cap / nominal_ah

        # DOC = discharge capacity / (nominal * current SOH)
        effective_capacity = nominal_ah * soh
        doc = min(dis_cap / effective_capacity, 1.0) if effective_capacity > 0 else 1.0

        # Temperature
        temp = row['temperature_C'] if has_temp and not np.isnan(row['temperature_C']) else temperature_C

        # C-rate: use provided values or estimate from capacity/duration
        if has_c_rate_dis:
            c_rate = row['c_rate_discharge']
        elif has_duration and row['duration_s'] > 0:
            # Estimate from discharge capacity and duration
            dis_time_h = row['duration_s'] / 3600.0 / 2.0  # assume ~half the cycle is discharge
            c_rate = dis_cap / (nominal_ah * dis_time_h) if dis_time_h > 0 else 1.0
        else:
            c_rate = 1.0

        # Cycle aging
        delta_cycles = cycle_num - prev_cycle
        if delta_cycles <= 0:
            delta_cycles = 1
        dFEC = doc * delta_cycles
        fec_new = fec_cum + dFEC

        kC = k_c_rate_Q(c_rate)
        kDOC = k_doc_Q(doc)
        dq_percent = kC * kDOC * (fec_new ** Z_Q - fec_cum ** Z_Q)
        d_cycle = dq_percent / 100.0

        soh -= d_cycle
        total_cycle_loss += d_cycle
        fec_cum = fec_new

        # Calendar aging — proportional to cycle duration
        if has_duration and row['duration_s'] > 0:
            dt_seconds = row['duration_s'] * delta_cycles
        else:
            # Default: estimate cycle time from C-rate and DOC
            hours_per_cycle = 2.0 * doc / c_rate if c_rate > 0 else 2.0
            dt_seconds = delta_cycles * hours_per_cycle * 3600.0

        T_K = temp + 273.15
        arr = math.exp(-Ea / R_GAS * (1.0 / T_K - 1.0 / T_REF_K))
        t_old_b = math.pow(cal_seconds, cal_b) if cal_seconds > 0 else 0.0
        cal_seconds += dt_seconds
        t_new_b = math.pow(cal_seconds, cal_b)
        soc_stress = 0.5 ** n_soc  # Assume mid-SOC for lab cycling
        d_cal = k0_frac * arr * (t_new_b - t_old_b) * soc_stress
        soh -= d_cal
        total_cal_loss += d_cal

        soh = max(0.0, soh)
        prev_cycle = cycle_num

        results.append({
            'cycle': cycle_num,
            'predicted_soh': soh,
            'measured_soh': measured_soh,
            'fec_cumulative': fec_cum,
            'cal_loss': total_cal_loss,
            'cycle_loss': total_cycle_loss,
        })

    return pd.DataFrame(results)


# =========================================================================
# Calibration
# =========================================================================

def calibrate_parameters(
    measured_df: pd.DataFrame,
    calendar_model: str = 'lam_calibrated',
    nominal_energy_wh: float = 1000.0,
    temperature: float = 25.0,
    c_rate: float = 1.0,
    doc: float = 1.0,
) -> Dict:
    """Calibrate degradation parameters for lab protocol data."""
    from scipy.optimize import differential_evolution

    k0_default, Ea_default, cal_b_default, n_default = _get_degradation_params(calendar_model)

    bounds = [
        (k0_default * 0.01, k0_default * 100),
        (Ea_default * 0.3, Ea_default * 3.0),
        (0.3, 1.0),
        (0.1, 2.0),
    ]

    def objective(params):
        k0_frac, Ea, cal_b, n = params
        custom = {'k0_frac': k0_frac, 'Ea': Ea, 'cal_b': cal_b, 'n': n}
        try:
            result_df = simulate_on_lab_protocol(
                measured_df, custom_params=custom,
                nominal_energy_wh=nominal_energy_wh,
                temperature=temperature, c_rate=c_rate, doc=doc,
            )
            residuals = result_df['predicted_soh'].values - result_df['measured_soh'].values
            return np.sqrt(np.mean(residuals ** 2))
        except Exception:
            return 1e6

    print("Calibrating parameters with differential evolution...")
    result = differential_evolution(objective, bounds=bounds, maxiter=100,
                                    popsize=20, seed=42, tol=1e-5, polish=True)

    k0_frac, Ea, cal_b, n = result.x
    fitted_params = {'k0_frac': k0_frac, 'Ea': Ea, 'cal_b': cal_b, 'n': n}

    final_df = simulate_on_lab_protocol(
        measured_df, custom_params=fitted_params,
        nominal_energy_wh=nominal_energy_wh,
        temperature=temperature, c_rate=c_rate, doc=doc,
    )
    metrics = compute_validation_metrics(
        final_df['predicted_soh'].values, final_df['measured_soh'].values,
    )

    print(f"\nCalibration complete:")
    for k, v in fitted_params.items():
        print(f"  {k:10s} = {v:.6e}")
    print(f"  RMSE      = {metrics['RMSE']:.6f}")
    print(f"  R^2       = {metrics['R2']:.6f}")

    return {
        'params': fitted_params, 'rmse': result.fun,
        'metrics': metrics, 'result': result, 'simulation_df': final_df,
    }


def calibrate_field_parameters(
    systems_data: List[Dict],
    calendar_model: str = 'lam_calibrated',
) -> Dict:
    """
    Calibrate degradation parameters across multiple field systems.

    Args:
        systems_data: List of dicts, each with:
            'timeseries_df': aggregated DataFrame at requested resolution
            'soh_ground_truth': DataFrame with date, measured_soh
            'system_info': system info dict
        calendar_model: Base model for parameter bounds

    Returns:
        Dict with fitted params, per-system metrics, aggregate metrics
    """
    from scipy.optimize import differential_evolution

    k0_default, Ea_default, cal_b_default, n_default = _get_degradation_params(calendar_model)

    bounds = [
        (k0_default * 0.01, k0_default * 100),
        (Ea_default * 0.3, Ea_default * 3.0),
        (0.3, 1.0),
        (0.1, 2.0),
    ]

    def objective(params):
        k0_frac, Ea, cal_b, n = params
        custom = {'k0_frac': k0_frac, 'Ea': Ea, 'cal_b': cal_b, 'n': n}
        total_rmse = 0.0
        n_valid = 0

        for sd in systems_data:
            truth = sd['soh_ground_truth']
            if truth.empty or len(truth) < 2:
                continue
            try:
                sim = simulate_on_field_data(
                    sd['timeseries_df'],
                    nominal_energy_wh=sd['system_info']['nominal_kwh'] * 1000,
                    custom_params=custom,
                )
                # Interpolate predictions at ground truth timestamps
                pred_soh = np.interp(
                    truth['date'].astype(np.int64),
                    sim['date'].astype(np.int64),
                    sim['predicted_soh'],
                )
                rmse = np.sqrt(np.mean((pred_soh - truth['measured_soh'].values) ** 2))
                total_rmse += rmse
                n_valid += 1
            except Exception:
                continue

        return total_rmse / n_valid if n_valid > 0 else 1e6

    print(f"Calibrating across {len(systems_data)} systems...")
    result = differential_evolution(objective, bounds=bounds, maxiter=100,
                                    popsize=20, seed=42, tol=1e-5, polish=True)

    k0_frac, Ea, cal_b, n = result.x
    fitted_params = {'k0_frac': k0_frac, 'Ea': Ea, 'cal_b': cal_b, 'n': n}

    # Compute per-system metrics with fitted params
    per_system = {}
    for sd in systems_data:
        sid = sd['system_info']['id']
        truth = sd['soh_ground_truth']
        if truth.empty or len(truth) < 2:
            continue
        try:
            sim = simulate_on_field_data(
                sd['timeseries_df'],
                nominal_energy_wh=sd['system_info']['nominal_kwh'] * 1000,
                custom_params=fitted_params,
            )
            pred_soh = np.interp(
                truth['date'].astype(np.int64),
                sim['date'].astype(np.int64),
                sim['predicted_soh'],
            )
            metrics = compute_validation_metrics(pred_soh, truth['measured_soh'].values)
            per_system[sid] = metrics
        except Exception as e:
            per_system[sid] = {'error': str(e)}

    # Aggregate
    rmses = [m['RMSE'] for m in per_system.values() if 'RMSE' in m]
    aggregate = {
        'mean_RMSE': np.mean(rmses) if rmses else float('inf'),
        'max_RMSE': np.max(rmses) if rmses else float('inf'),
        'n_systems': len(rmses),
    }

    print(f"\nMulti-system calibration complete:")
    for k, v in fitted_params.items():
        print(f"  {k:10s} = {v:.6e}")
    print(f"  Mean RMSE = {aggregate['mean_RMSE']:.6f} ({aggregate['n_systems']} systems)")

    return {
        'params': fitted_params,
        'per_system_metrics': per_system,
        'aggregate_metrics': aggregate,
        'result': result,
    }


def loo_cross_validation(systems_data, calendar_model='lam', output_dir='results/validation_loo'):
    """Leave-one-out cross-validation for field calibration.

    For each system with >=2 SOH tests:
      - Calibrate on all OTHER systems
      - Predict the held-out system with the fitted params
      - Record held-out RMSE

    Reports mean cross-validated RMSE (honest generalization error).
    If significantly worse than in-sample RMSE → overfitting warning.
    """
    from scipy.optimize import differential_evolution

    os.makedirs(output_dir, exist_ok=True)

    # Filter to systems with enough SOH data for evaluation
    valid_systems = [sd for sd in systems_data
                     if not sd['soh_ground_truth'].empty and len(sd['soh_ground_truth']) >= 2]

    if len(valid_systems) < 3:
        print(f"Need at least 3 systems for LOO, got {len(valid_systems)}")
        return None

    print(f"\nLeave-One-Out Cross-Validation ({len(valid_systems)} systems)")
    print("=" * 60)

    k0_default, Ea_default, cal_b_default, n_default = _get_degradation_params(calendar_model)
    bounds = [
        (k0_default * 0.01, k0_default * 100),
        (Ea_default * 0.3, Ea_default * 3.0),
        (0.3, 1.0),
        (0.1, 2.0),
    ]

    loo_results = []

    for i, held_out in enumerate(valid_systems):
        held_out_id = held_out['system_info']['id']
        train_systems = [s for j, s in enumerate(valid_systems) if j != i]

        print(f"\n--- Fold {i+1}/{len(valid_systems)}: Hold out System {held_out_id}, "
              f"train on {len(train_systems)} systems ---")

        # Calibrate on training systems
        def objective(params):
            k0_frac, Ea, cal_b, n = params
            custom = {'k0_frac': k0_frac, 'Ea': Ea, 'cal_b': cal_b, 'n': n}
            total_rmse = 0.0
            n_valid = 0
            for sd in train_systems:
                truth = sd['soh_ground_truth']
                if truth.empty or len(truth) < 2:
                    continue
                try:
                    sim = simulate_on_field_data(
                        sd['timeseries_df'],
                        nominal_energy_wh=sd['system_info']['nominal_kwh'] * 1000,
                        custom_params=custom,
                    )
                    pred_soh = np.interp(
                        truth['date'].astype(np.int64),
                        sim['date'].astype(np.int64),
                        sim['predicted_soh'],
                    )
                    rmse = np.sqrt(np.mean((pred_soh - truth['measured_soh'].values) ** 2))
                    total_rmse += rmse
                    n_valid += 1
                except Exception:
                    continue
            return total_rmse / n_valid if n_valid > 0 else 1e6

        result = differential_evolution(objective, bounds=bounds, maxiter=100,
                                        popsize=20, seed=42, tol=1e-5, polish=True)

        k0_frac, Ea, cal_b, n = result.x
        fold_params = {'k0_frac': k0_frac, 'Ea': Ea, 'cal_b': cal_b, 'n': n}

        # Predict held-out system
        truth = held_out['soh_ground_truth']
        sim = simulate_on_field_data(
            held_out['timeseries_df'],
            nominal_energy_wh=held_out['system_info']['nominal_kwh'] * 1000,
            custom_params=fold_params,
        )
        pred_soh = np.interp(
            truth['date'].astype(np.int64),
            sim['date'].astype(np.int64),
            sim['predicted_soh'],
        )
        held_out_rmse = np.sqrt(np.mean((pred_soh - truth['measured_soh'].values) ** 2))
        held_out_metrics = compute_validation_metrics(pred_soh, truth['measured_soh'].values)

        # Also compute in-sample RMSE for this fold
        train_rmses = []
        for sd in train_systems:
            t = sd['soh_ground_truth']
            if t.empty or len(t) < 2:
                continue
            s = simulate_on_field_data(
                sd['timeseries_df'],
                nominal_energy_wh=sd['system_info']['nominal_kwh'] * 1000,
                custom_params=fold_params,
            )
            p = np.interp(t['date'].astype(np.int64), s['date'].astype(np.int64), s['predicted_soh'])
            train_rmses.append(np.sqrt(np.mean((p - t['measured_soh'].values) ** 2)))

        fold_result = {
            'held_out_system': held_out_id,
            'held_out_rmse': held_out_rmse,
            'held_out_metrics': held_out_metrics,
            'train_mean_rmse': np.mean(train_rmses) if train_rmses else float('inf'),
            'params': fold_params,
        }
        loo_results.append(fold_result)

        print(f"  Fitted params: k0={k0_frac:.3e}, Ea={Ea:.1f}, cal_b={cal_b:.3f}, n={n:.3f}")
        print(f"  Train RMSE: {fold_result['train_mean_rmse']:.4f}")
        print(f"  Held-out System {held_out_id} RMSE: {held_out_rmse:.4f}, "
              f"R²={held_out_metrics['R2']:.4f}")

    # Summary
    cv_rmses = [r['held_out_rmse'] for r in loo_results]
    train_rmses_all = [r['train_mean_rmse'] for r in loo_results]
    mean_cv_rmse = np.mean(cv_rmses)
    mean_train_rmse = np.mean(train_rmses_all)

    print(f"\n{'=' * 60}")
    print(f"LOO Cross-Validation Summary")
    print(f"{'=' * 60}")
    print(f"  Mean CV RMSE (held-out):  {mean_cv_rmse:.4f}")
    print(f"  Mean train RMSE:          {mean_train_rmse:.4f}")
    print(f"  Overfitting gap:          {mean_cv_rmse - mean_train_rmse:.4f}")

    if mean_cv_rmse > mean_train_rmse * 1.5:
        print(f"  WARNING: CV RMSE is {mean_cv_rmse/mean_train_rmse:.1f}x train RMSE — "
              f"possible overfitting")
    else:
        print(f"  Ratio CV/train: {mean_cv_rmse/mean_train_rmse:.2f} — "
              f"{'good generalization' if mean_cv_rmse/mean_train_rmse < 1.3 else 'moderate overfitting'}")

    print(f"\nPer-fold results:")
    for r in loo_results:
        print(f"  System {r['held_out_system']:2d}: "
              f"held-out RMSE={r['held_out_rmse']:.4f}, "
              f"train RMSE={r['train_mean_rmse']:.4f}")

    # Save results
    summary = {
        'mean_cv_rmse': mean_cv_rmse,
        'mean_train_rmse': mean_train_rmse,
        'overfitting_gap': mean_cv_rmse - mean_train_rmse,
        'folds': [{
            'held_out_system': int(r['held_out_system']),
            'held_out_rmse': r['held_out_rmse'],
            'train_mean_rmse': r['train_mean_rmse'],
            'held_out_metrics': r['held_out_metrics'],
            'params': r['params'],
        } for r in loo_results],
    }
    with open(os.path.join(output_dir, 'loo_cross_validation.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/loo_cross_validation.json")
    return summary


# =========================================================================
# Dataset loader dispatch
# =========================================================================

def generate_loo_plots(
    loo_json_path: str,
    systems_data: List[Dict],
    output_dir: str,
    full_cal_params: Optional[Dict] = None,
) -> None:
    """
    Generate LOO cross-validation plots from cached results.

    Args:
        loo_json_path: Path to loo_cross_validation.json
        systems_data: List of loaded system dicts (for re-simulating predictions)
        output_dir: Directory to save plots
        full_cal_params: Full-calibration params (for parameter stability reference line).
            If None, uses lam_calibrated defaults from constants.
    """
    with open(loo_json_path) as f:
        loo_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Default full-cal params from constants
    if full_cal_params is None:
        from breos.constants import (
            LAM_CAL_K0_FRAC, LAM_CAL_EA_J_MOL,
            LAM_CAL_EXPONENT_B, LAM_CAL_SOC_EXPONENT_N,
        )
        full_cal_params = {
            'k0_frac': LAM_CAL_K0_FRAC,
            'Ea': LAM_CAL_EA_J_MOL,
            'cal_b': LAM_CAL_EXPONENT_B,
            'n': LAM_CAL_SOC_EXPONENT_N,
        }

    # Plot 1: CV summary bars
    print("  Generating LOO CV summary plot...")
    plot_loo_cv_summary(loo_data, output_dir)

    # Plot 2: Parameter stability
    print("  Generating parameter stability plots...")
    plot_loo_param_stability(loo_data, full_cal_params, output_dir)

    # Plot 3: Held-out predictions (requires re-simulation)
    print("  Generating held-out prediction plots...")
    systems_by_id = {sd['system_info']['id']: sd for sd in systems_data}
    predictions = []

    for fold in loo_data['folds']:
        sid = fold['held_out_system']
        fold_params = fold['params']
        sd = systems_by_id.get(sid)
        if sd is None:
            continue

        truth = sd['soh_ground_truth']
        if truth.empty:
            continue

        sim = simulate_on_field_data(
            sd['timeseries_df'],
            nominal_energy_wh=sd['system_info']['nominal_kwh'] * 1000,
            custom_params=fold_params,
        )

        pred_at_tests = np.interp(
            truth['date'].astype(np.int64),
            sim['date'].astype(np.int64),
            sim['predicted_soh'],
        )

        predictions.append({
            'system_id': sid,
            'dates_measured': truth['date'].values,
            'soh_measured': truth['measured_soh'].values,
            'dates_predicted': sim['date'].values,
            'soh_predicted': sim['predicted_soh'].values,
            'rmse': fold['held_out_rmse'],
        })

    if predictions:
        plot_loo_predictions(predictions, output_dir)

    print(f"  LOO plots saved to {output_dir}/")


DATASET_LOADERS = {
    'hust': load_hust_dataset,
    'zenodo_home': None,  # Handled specially in main()
    'zenodo_lfp': load_zenodo_lfp_field,
    'calce': load_calce_lfp,
    'fastcharge': load_fastcharge_dataset,
    'custom_csv': load_custom_csv,
}


# =========================================================================
# Main CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate PVBAT battery degradation model against experimental data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('dataset_path', help='Path to dataset file or directory')
    parser.add_argument('--dataset-type', required=True,
                        choices=sorted(DATASET_LOADERS.keys()),
                        help='Type of dataset to load')
    parser.add_argument('--cell-id', default=None,
                        help='Specific cell/system ID within dataset')
    parser.add_argument('--calendar-model', default='lam_calibrated',
                        choices=['naumann', 'lam', 'lam_calibrated', 'lam_calibrated_hourly'],
                        help='Calendar aging model (default: lam_calibrated)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibrate model parameters to fit measured data')
    parser.add_argument('--multi-system', action='store_true',
                        help='Run across all systems (zenodo_home only)')
    parser.add_argument('--loo', action='store_true',
                        help='Run leave-one-out cross-validation (requires --multi-system)')
    parser.add_argument('--loo-plots', action='store_true',
                        help='Generate LOO validation plots from cached results')
    parser.add_argument('--chemistry', default='all',
                        choices=['LFP', 'NMC', 'LMO', 'all'],
                        help='Filter systems by chemistry (zenodo_home only, default: all)')
    parser.add_argument('--output', default='results/validation',
                        help='Output directory for results and plots')
    parser.add_argument('--cache-dir', default='results/validation_cache',
                        help='Cache directory for intermediate data')
    parser.add_argument('--resolution', default='15min',
                        choices=['h', '15min'],
                        help='Temporal resolution for field data aggregation (default: 15min)')
    parser.add_argument('--temperature', type=float, default=25.0,
                        help='Cell temperature for lab protocols (C)')
    parser.add_argument('--c-rate', type=float, default=1.0,
                        help='C-rate for lab protocols')
    parser.add_argument('--doc', type=float, default=1.0,
                        help='Depth of cycle for lab protocols (fraction)')
    parser.add_argument('--nominal-wh', type=float, default=1000.0,
                        help='Nominal cell/system capacity (Wh)')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading {args.dataset_type} dataset from: {args.dataset_path}")

    # ---------------------------------------------------------------
    # Zenodo Home Storage — specialized flow
    # ---------------------------------------------------------------
    if args.dataset_type == 'zenodo_home':
        data = load_zenodo_home_dataset(args.dataset_path, args.cell_id, args.cache_dir,
                                        chemistry_filter=args.chemistry,
                                        resolution=args.resolution)

        if args.cell_id is None and not args.multi_system:
            print("\nUse --cell-id <ID> to select a system, or --multi-system for all.")
            sys.exit(0)

        if args.multi_system:
            # Multi-system mode
            systems_data = []
            for sid in data.get('available_systems', []):
                try:
                    sd = load_zenodo_home_dataset(args.dataset_path, str(sid), args.cache_dir,
                                                  resolution=args.resolution)
                    if sd.get('soh_ground_truth') is not None and not sd['soh_ground_truth'].empty:
                        systems_data.append(sd)
                except Exception as e:
                    print(f"   Skipping System {sid}: {e}")

            print(f"\n{len(systems_data)} systems with valid SOH data")

            if args.loo_plots:
                loo_json = os.path.join(args.output, 'loo_cross_validation.json')
                if not os.path.exists(loo_json):
                    # Try default LOO output dir
                    loo_json = 'results/validation_loo/loo_cross_validation.json'
                if os.path.exists(loo_json):
                    generate_loo_plots(loo_json, systems_data, args.output)
                else:
                    print(f"ERROR: LOO results not found. Run --loo first.")
                    sys.exit(1)
                print(f"\nLOO plots complete. Saved to: {args.output}")
                sys.exit(0)

            if args.loo:
                loo_result = loo_cross_validation(
                    systems_data, calendar_model=args.calendar_model,
                    output_dir=args.output,
                )
                if loo_result and args.calibrate:
                    # Also run full calibration after LOO
                    print("\nRunning full calibration on all systems...")
                    cal_result = calibrate_field_parameters(systems_data, args.calendar_model)
                    with open(os.path.join(args.output, 'calibrated_params.json'), 'w') as f:
                        json.dump(cal_result['params'], f, indent=2)
                    with open(os.path.join(args.output, 'per_system_metrics.json'), 'w') as f:
                        json.dump({str(k): v for k, v in cal_result['per_system_metrics'].items()}, f, indent=2)
                    print(f"\nSaved calibrated parameters to {args.output}/")
            elif args.calibrate:
                cal_result = calibrate_field_parameters(systems_data, args.calendar_model)
                with open(os.path.join(args.output, 'calibrated_params.json'), 'w') as f:
                    json.dump(cal_result['params'], f, indent=2)
                with open(os.path.join(args.output, 'per_system_metrics.json'), 'w') as f:
                    json.dump({str(k): v for k, v in cal_result['per_system_metrics'].items()}, f, indent=2)
                print(f"\nSaved calibrated parameters to {args.output}/")
            else:
                # Run default model on all systems
                print(f"\nRunning {args.calendar_model} model on all systems...")
                all_results = {}
                for sd in systems_data:
                    sid = sd['system_info']['id']
                    sim = simulate_on_field_data(
                        sd['timeseries_df'],
                        calendar_model=args.calendar_model,
                        nominal_energy_wh=sd['system_info']['nominal_kwh'] * 1000,
                    )
                    truth = sd['soh_ground_truth']
                    if not truth.empty and len(truth) >= 2:
                        pred_at_tests = np.interp(
                            truth['date'].astype(np.int64),
                            sim['date'].astype(np.int64),
                            sim['predicted_soh'],
                        )
                        metrics = compute_validation_metrics(pred_at_tests, truth['measured_soh'].values)
                        all_results[sid] = {
                            'metrics': metrics,
                            'simulation': sim,
                            'truth': truth,
                        }
                        print(f"  System {sid:2d}: RMSE={metrics['RMSE']:.4f}, "
                              f"R²={metrics['R2']:.4f}, N={metrics['N_points']}")

                # Summary
                rmses = [r['metrics']['RMSE'] for r in all_results.values()]
                if rmses:
                    print(f"\n  Mean RMSE: {np.mean(rmses):.4f}")
                    print(f"  Max RMSE:  {np.max(rmses):.4f}")

                # Multi-system plot
                if all_results:
                    print("  Generating multi-system plot...")
                    plot_validation_multi_system(all_results, args.output)

            print(f"\nValidation complete. Results saved to: {args.output}")
            sys.exit(0)

        # Single system mode
        timeseries_df = data['timeseries_df']
        soh_truth = data['soh_ground_truth']
        sys_info = data['system_info']
        nominal_wh = sys_info['nominal_kwh'] * 1000

        # Run simulation
        custom_params = None
        if args.calibrate and not soh_truth.empty and len(soh_truth) >= 2:
            # TODO: single-system field calibration
            print("Single-system field calibration not yet implemented. Running default model.")

        print(f"\n   Running {args.calendar_model} model simulation...")
        sim_df = simulate_on_field_data(
            timeseries_df,
            calendar_model=args.calendar_model,
            nominal_energy_wh=nominal_wh,
            custom_params=custom_params,
        )

        # Save simulation results
        sim_df.to_csv(os.path.join(args.output, f'simulation_system_{args.cell_id}.csv'), index=False)

        if not soh_truth.empty and len(soh_truth) >= 2:
            # Interpolate predictions at ground truth timestamps
            pred_at_tests = np.interp(
                soh_truth['date'].astype(np.int64),
                sim_df['date'].astype(np.int64),
                sim_df['predicted_soh'],
            )
            metrics = compute_validation_metrics(pred_at_tests, soh_truth['measured_soh'].values)

            print(f"\n   Validation Metrics (System {args.cell_id}):")
            print(f"      RMSE      = {metrics['RMSE']:.6f}")
            print(f"      MAE       = {metrics['MAE']:.6f}")
            print(f"      R²        = {metrics['R2']:.6f}")
            print(f"      Max Error = {metrics['Max_Error']:.6f}")

            with open(os.path.join(args.output, f'metrics_system_{args.cell_id}.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            # Generate plots
            print("   Generating validation plots...")
            measured_series = pd.Series(soh_truth['measured_soh'].values, index=soh_truth['date'].values)
            predicted_series = pd.Series(pred_at_tests, index=soh_truth['date'].values)

            plot_validation_soh_comparison(measured_series, predicted_series, args.output,
                                          x_label='Date', metrics=metrics)
            plot_validation_residuals(measured_series, predicted_series, args.output, x_label='Date')
            plot_validation_parity(measured_series, predicted_series, args.output, metrics=metrics)

            # Degradation split plot
            plot_validation_degradation_split(sim_df, args.output, system_label=args.cell_id)
        else:
            print(f"\n   No SOH ground truth available. Simulation saved without validation.")
            # Plot predicted SOH timeline
            print(f"   Final predicted SOH: {sim_df['predicted_soh'].iloc[-1]*100:.1f}%")
            print(f"   Total FEC: {sim_df['fec_cumulative'].iloc[-1]:.1f}")
            print(f"   Calendar loss: {sim_df['cal_loss'].iloc[-1]*100:.2f}%")
            print(f"   Cycle loss: {sim_df['cycle_loss'].iloc[-1]*100:.2f}%")

        print(f"\n   Validation complete. Results saved to: {args.output}")
        sys.exit(0)

    # ---------------------------------------------------------------
    # Lab cycling datasets (hust, fastcharge) — use simulate_on_lab_cycles
    # ---------------------------------------------------------------
    if args.dataset_type in ('hust', 'fastcharge'):
        loader = DATASET_LOADERS[args.dataset_type]
        data = loader(args.dataset_path, cell_id=args.cell_id)

        if args.cell_id is None:
            print("\nUse --cell-id <ID> to select a cell.")
            sys.exit(0)

        cycles_df = data['cycles_df']
        nominal_ah = data['nominal_ah']

        if args.multi_system:
            # Batch mode: run all cells
            all_cells = data['available_cells']
            all_metrics = {}

            print(f"\nRunning {args.calendar_model} model on all {len(all_cells)} cells...")
            for cid in all_cells:
                try:
                    cell_data = loader(args.dataset_path, cell_id=str(cid))
                    sim_df = simulate_on_lab_cycles(
                        cell_data['cycles_df'],
                        calendar_model=args.calendar_model,
                        nominal_ah=cell_data['nominal_ah'],
                        temperature_C=args.temperature,
                    )
                    if len(sim_df) >= 2:
                        metrics = compute_validation_metrics(
                            sim_df['predicted_soh'].values,
                            sim_df['measured_soh'].values,
                        )
                        all_metrics[str(cid)] = metrics
                        print(f"  Cell {cid}: RMSE={metrics['RMSE']:.4f}, "
                              f"N={len(sim_df)} cycles")
                except Exception as e:
                    print(f"  Cell {cid}: ERROR — {e}")

            if all_metrics:
                rmses = [m['RMSE'] for m in all_metrics.values()]
                print(f"\n  Mean RMSE: {np.mean(rmses):.4f} ({len(rmses)} cells)")
                print(f"  Median RMSE: {np.median(rmses):.4f}")
                print(f"  Max RMSE: {np.max(rmses):.4f}")

                with open(os.path.join(args.output, 'batch_metrics.json'), 'w') as f:
                    json.dump(all_metrics, f, indent=2)

            print(f"\nBatch validation complete. Results saved to: {args.output}")
            sys.exit(0)

        # Single cell mode
        print(f"\n   Running {args.calendar_model} model on cell {args.cell_id}...")
        sim_df = simulate_on_lab_cycles(
            cycles_df,
            calendar_model=args.calendar_model,
            nominal_ah=nominal_ah,
            temperature_C=args.temperature,
        )

        metrics = compute_validation_metrics(
            sim_df['predicted_soh'].values,
            sim_df['measured_soh'].values,
        )

        print(f"\n   Validation Metrics (Cell {args.cell_id}):")
        print(f"      RMSE      = {metrics['RMSE']:.6f}")
        print(f"      MAE       = {metrics['MAE']:.6f}")
        print(f"      MAPE      = {metrics['MAPE']:.2f}%")
        print(f"      R²        = {metrics['R2']:.6f}")
        print(f"      Max Error = {metrics['Max_Error']:.6f}")

        sim_df.to_csv(os.path.join(args.output, f'validation_cell_{args.cell_id}.csv'), index=False)
        with open(os.path.join(args.output, f'metrics_cell_{args.cell_id}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Plots
        print("   Generating validation plots...")
        measured_series = pd.Series(sim_df['measured_soh'].values, index=sim_df['cycle'].values)
        predicted_series = pd.Series(sim_df['predicted_soh'].values, index=sim_df['cycle'].values)

        plot_validation_soh_comparison(measured_series, predicted_series, args.output,
                                      x_label='Cycle', metrics=metrics)
        plot_validation_residuals(measured_series, predicted_series, args.output, x_label='Cycle')
        plot_validation_parity(measured_series, predicted_series, args.output, metrics=metrics)

        print(f"\n   Validation complete. Results saved to: {args.output}")
        sys.exit(0)

    # ---------------------------------------------------------------
    # CALCE characterization dataset
    # ---------------------------------------------------------------
    if args.dataset_type == 'calce':
        data = load_calce_lfp(args.dataset_path, cell_id=args.cell_id)
        if args.cell_id is None:
            sys.exit(0)
        if data.get('is_characterization'):
            temp_cap = data['temperature_capacity']
            print(f"\n   Temperature-capacity data for {args.cell_id}:")
            for _, row in temp_cap.iterrows():
                print(f"      {row['temperature_C']:6.1f}°C: "
                      f"{row['discharge_capacity_ah']:.3f} Ah "
                      f"(SOH={row['soh']*100:.1f}%)")
            temp_cap.to_csv(os.path.join(args.output, f'calce_{args.cell_id}_temp_capacity.csv'),
                            index=False)
            print(f"\n   Saved to: {args.output}")
        sys.exit(0)

    # ---------------------------------------------------------------
    # Other dataset types (custom CSV, lab protocols)
    # ---------------------------------------------------------------
    loader = DATASET_LOADERS[args.dataset_type]
    try:
        if args.dataset_type == 'custom_csv':
            measured_df = loader(args.dataset_path)
        else:
            measured_df = loader(args.dataset_path, cell_id=args.cell_id)
    except NotImplementedError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"  Loaded {len(measured_df)} data points")

    custom_params = None
    if args.calibrate:
        cal_result = calibrate_parameters(
            measured_df, calendar_model=args.calendar_model,
            nominal_energy_wh=args.nominal_wh, temperature=args.temperature,
            c_rate=args.c_rate, doc=args.doc,
        )
        custom_params = cal_result['params']
        sim_df = cal_result['simulation_df']
        metrics = cal_result['metrics']

        with open(os.path.join(args.output, 'calibrated_params.json'), 'w') as f:
            json.dump(cal_result['params'], f, indent=2)
    else:
        print(f"Running simulation with {args.calendar_model} model parameters...")
        sim_df = simulate_on_lab_protocol(
            measured_df, calendar_model=args.calendar_model,
            nominal_energy_wh=args.nominal_wh, temperature=args.temperature,
            c_rate=args.c_rate, doc=args.doc, custom_params=custom_params,
        )
        metrics = compute_validation_metrics(
            sim_df['predicted_soh'].values, sim_df['measured_soh'].values,
        )

    print(f"\nValidation Metrics:")
    print(f"  RMSE      = {metrics['RMSE']:.6f}")
    print(f"  MAE       = {metrics['MAE']:.6f}")
    print(f"  MAPE      = {metrics['MAPE']:.2f}%")
    print(f"  R²        = {metrics['R2']:.6f}")
    print(f"  Max Error = {metrics['Max_Error']:.6f}")

    sim_df.to_csv(os.path.join(args.output, 'validation_results.csv'), index=False)

    with open(os.path.join(args.output, 'validation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Generating validation plots...")
    measured_series = pd.Series(sim_df['measured_soh'].values, index=sim_df['cycle_or_time'].values)
    predicted_series = pd.Series(sim_df['predicted_soh'].values, index=sim_df['cycle_or_time'].values)

    x_label = 'Cycle' if 'cycle' in measured_df.columns.str.lower().str.cat() else 'Time'

    plot_validation_soh_comparison(measured_series, predicted_series, args.output,
                                  x_label=x_label, metrics=metrics)
    plot_validation_residuals(measured_series, predicted_series, args.output, x_label=x_label)
    plot_validation_parity(measured_series, predicted_series, args.output, metrics=metrics)

    print(f"\nValidation complete. Results saved to: {args.output}")


if __name__ == '__main__':
    main()
