"""
Batch Battery Degradation Validation Script.

Runs all validation tasks sequentially and saves everything to results/.
Designed to be launched on a machine with access to all datasets and left running.

Usage:
    uv run python tools/run_full_validation.py --data-dir /path/to/Battery_deg_datasets --output results/validation_full

Dataset paths auto-discovered from --data-dir:
    {data-dir}/12091223/                    → Zenodo home storage (LOO plots)
    {data-dir}/hust/kw34hhw7xg-3/           → HUST 77 LFP cells (lab cycling)
    {data-dir}/FastCharge/                  → FastCharge 140 cells (high C-rate)
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.validate_degradation import (
    load_zenodo_home_dataset,
    load_hust_dataset,
    load_fastcharge_dataset,
    simulate_on_field_data,
    simulate_on_lab_cycles,
    compute_validation_metrics,
    generate_loo_plots,
)


def _elapsed(start: float) -> str:
    """Format elapsed time as MM:SS."""
    dt = time.time() - start
    return f"{int(dt // 60)}:{int(dt % 60):02d}"


def run_loo_plots(loo_json_path: str, zenodo_path: str, output_dir: str, cache_dir: str,
                  resolution: str = '15min'):
    """Task 1: Generate LOO cross-validation plots from cached results."""
    print("\n" + "=" * 70)
    print("TASK 1: LOO Cross-Validation Plots")
    print("=" * 70)

    if not os.path.exists(loo_json_path):
        print(f"  SKIP: LOO results not found at {loo_json_path}")
        return {'status': 'skipped', 'reason': 'no LOO results file'}

    # Load systems data for re-simulation
    print("  Loading Zenodo LFP systems for prediction re-simulation...")
    lfp_systems = [14, 15, 17, 20, 21]
    systems_data = []
    for sid in lfp_systems:
        try:
            sd = load_zenodo_home_dataset(zenodo_path, str(sid), cache_dir,
                                          resolution=resolution)
            if sd.get('soh_ground_truth') is not None and not sd['soh_ground_truth'].empty:
                systems_data.append(sd)
        except Exception as e:
            print(f"  Warning: Could not load System {sid}: {e}")

    if not systems_data:
        print("  SKIP: No systems data loaded for LOO plots")
        return {'status': 'skipped', 'reason': 'no systems data'}

    loo_output = os.path.join(output_dir, 'loo_plots')
    generate_loo_plots(loo_json_path, systems_data, loo_output)

    return {'status': 'complete', 'output_dir': loo_output}


def run_hust_validation(hust_path: str, output_dir: str, calendar_model: str = 'lam_calibrated'):
    """Task 2: Validate on HUST 77 LFP cells."""
    print("\n" + "=" * 70)
    print("TASK 2: HUST Dataset Validation (77 LFP cells)")
    print("=" * 70)

    if not os.path.exists(hust_path):
        print(f"  SKIP: HUST path not found: {hust_path}")
        return {'status': 'skipped', 'reason': 'path not found'}

    hust_output = os.path.join(output_dir, 'hust')
    os.makedirs(hust_output, exist_ok=True)

    # Discover available cells
    data = load_hust_dataset(hust_path)
    available = data['available_cells']
    print(f"\n  Found {len(available)} cells with cycling data")

    all_metrics = {}
    errors = []
    t0 = time.time()

    for i, cid in enumerate(available):
        try:
            cell_data = load_hust_dataset(hust_path, cell_id=str(cid))
            sim_df = simulate_on_lab_cycles(
                cell_data['cycles_df'],
                calendar_model=calendar_model,
                nominal_ah=cell_data['nominal_ah'],
            )
            if len(sim_df) >= 2:
                metrics = compute_validation_metrics(
                    sim_df['predicted_soh'].values,
                    sim_df['measured_soh'].values,
                )
                all_metrics[str(cid)] = metrics
                all_metrics[str(cid)]['n_cycles'] = len(sim_df)
                all_metrics[str(cid)]['c_rate_discharge'] = cell_data['c_rate_discharge']
                all_metrics[str(cid)]['profile_type'] = cell_data['profile_type']

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{len(available)}] Cell #{cid}: "
                          f"RMSE={metrics['RMSE']:.4f}, {len(sim_df)} cycles "
                          f"[{_elapsed(t0)}]")
        except Exception as e:
            errors.append({'cell_id': cid, 'error': str(e)})
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(available)}] Cell #{cid}: ERROR — {e}")

    # Summary
    if all_metrics:
        rmses = [m['RMSE'] for m in all_metrics.values()]
        summary = {
            'n_cells': len(all_metrics),
            'n_errors': len(errors),
            'mean_rmse': float(np.mean(rmses)),
            'median_rmse': float(np.median(rmses)),
            'std_rmse': float(np.std(rmses)),
            'max_rmse': float(np.max(rmses)),
            'min_rmse': float(np.min(rmses)),
            'calendar_model': calendar_model,
        }

        print(f"\n  HUST Summary ({len(all_metrics)} cells):")
        print(f"    Mean RMSE:   {summary['mean_rmse']:.4f}")
        print(f"    Median RMSE: {summary['median_rmse']:.4f}")
        print(f"    Std RMSE:    {summary['std_rmse']:.4f}")
        print(f"    Range:       {summary['min_rmse']:.4f} — {summary['max_rmse']:.4f}")
        print(f"    Errors:      {len(errors)}")

        with open(os.path.join(hust_output, 'per_cell_metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)
        with open(os.path.join(hust_output, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        if errors:
            with open(os.path.join(hust_output, 'errors.json'), 'w') as f:
                json.dump(errors, f, indent=2)

        return {'status': 'complete', **summary}

    return {'status': 'failed', 'n_errors': len(errors)}


def run_fastcharge_validation(
    fastcharge_path: str, output_dir: str, calendar_model: str = 'lam_calibrated',
):
    """Task 4: Validate on FastCharge 140 LFP cells."""
    print("\n" + "=" * 70)
    print("TASK 4: FastCharge Dataset Validation (140 LFP cells, high C-rate)")
    print("=" * 70)

    if not os.path.exists(fastcharge_path):
        print(f"  SKIP: FastCharge path not found: {fastcharge_path}")
        return {'status': 'skipped', 'reason': 'path not found'}

    fc_output = os.path.join(output_dir, 'fastcharge')
    os.makedirs(fc_output, exist_ok=True)

    # Discover available cells
    data = load_fastcharge_dataset(fastcharge_path)
    available = data['available_cells']
    print(f"\n  Found {len(available)} cells")

    all_metrics = {}
    errors = []
    t0 = time.time()

    for i, cid in enumerate(available):
        try:
            cell_data = load_fastcharge_dataset(fastcharge_path, cell_id=str(cid))
            sim_df = simulate_on_lab_cycles(
                cell_data['cycles_df'],
                calendar_model=calendar_model,
                nominal_ah=cell_data['nominal_ah'],
            )
            if len(sim_df) >= 2:
                metrics = compute_validation_metrics(
                    sim_df['predicted_soh'].values,
                    sim_df['measured_soh'].values,
                )
                all_metrics[str(cid)] = metrics
                all_metrics[str(cid)]['n_cycles'] = len(sim_df)
                all_metrics[str(cid)]['barcode'] = cell_data.get('barcode', '')

                if (i + 1) % 20 == 0 or i == 0:
                    print(f"  [{i+1}/{len(available)}] Cell {cid}: "
                          f"RMSE={metrics['RMSE']:.4f}, {len(sim_df)} cycles "
                          f"[{_elapsed(t0)}]")
        except Exception as e:
            errors.append({'cell_id': cid, 'error': str(e)})

    # Summary
    if all_metrics:
        rmses = [m['RMSE'] for m in all_metrics.values()]
        summary = {
            'n_cells': len(all_metrics),
            'n_errors': len(errors),
            'mean_rmse': float(np.mean(rmses)),
            'median_rmse': float(np.median(rmses)),
            'std_rmse': float(np.std(rmses)),
            'max_rmse': float(np.max(rmses)),
            'min_rmse': float(np.min(rmses)),
            'calendar_model': calendar_model,
        }

        print(f"\n  FastCharge Summary ({len(all_metrics)} cells):")
        print(f"    Mean RMSE:   {summary['mean_rmse']:.4f}")
        print(f"    Median RMSE: {summary['median_rmse']:.4f}")
        print(f"    Std RMSE:    {summary['std_rmse']:.4f}")
        print(f"    Range:       {summary['min_rmse']:.4f} — {summary['max_rmse']:.4f}")
        print(f"    Errors:      {len(errors)}")

        with open(os.path.join(fc_output, 'per_cell_metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)
        with open(os.path.join(fc_output, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        if errors:
            with open(os.path.join(fc_output, 'errors.json'), 'w') as f:
                json.dump(errors, f, indent=2)

        return {'status': 'complete', **summary}

    return {'status': 'failed', 'n_errors': len(errors)}


def main():
    parser = argparse.ArgumentParser(
        description='Run full battery degradation validation across all datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data-dir', required=True,
                        help='Root directory containing all battery datasets')
    parser.add_argument('--output', default='results/validation_full',
                        help='Output directory for all results')
    parser.add_argument('--calendar-model', default='lam_calibrated',
                        choices=['naumann', 'lam', 'lam_calibrated', 'lam_calibrated_hourly'],
                        help='Calendar aging model (default: lam_calibrated)')
    parser.add_argument('--cache-dir', default='results/validation_cache',
                        help='Cache directory for Zenodo aggregated data')
    parser.add_argument('--resolution', default='15min',
                        choices=['h', '15min'],
                        help='Temporal resolution for field data aggregation (default: 15min)')
    parser.add_argument('--loo-json', default='results/validation_loo/loo_cross_validation.json',
                        help='Path to LOO cross-validation results JSON')
    parser.add_argument('--skip', nargs='*', default=[],
                        choices=['loo', 'hust', 'fastcharge'],
                        help='Skip specific validation tasks')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Auto-discover dataset paths
    data_dir = args.data_dir
    zenodo_path = os.path.join(data_dir, '12091223')
    hust_path = os.path.join(data_dir, 'hust', 'kw34hhw7xg-3')
    fastcharge_path = os.path.join(data_dir, 'FastCharge')

    print("=" * 70)
    print("  PVBAT Battery Degradation — Full Validation Suite")
    print("=" * 70)
    print(f"  Data directory:  {data_dir}")
    print(f"  Output:          {args.output}")
    print(f"  Calendar model:  {args.calendar_model}")
    print(f"  Resolution:      {args.resolution}")
    print(f"  Skip:            {args.skip or 'none'}")
    print()
    print(f"  Zenodo:     {'FOUND' if os.path.isdir(zenodo_path) else 'NOT FOUND'} — {zenodo_path}")
    print(f"  HUST:       {'FOUND' if os.path.isdir(hust_path) else 'NOT FOUND'} — {hust_path}")
    print(f"  FastCharge: {'FOUND' if os.path.isdir(fastcharge_path) else 'NOT FOUND'} — {fastcharge_path}")

    results = {}
    t_total = time.time()

    # Task 1: LOO plots
    if 'loo' not in args.skip:
        try:
            t0 = time.time()
            results['loo_plots'] = run_loo_plots(
                args.loo_json, zenodo_path, args.output, args.cache_dir,
                resolution=args.resolution,
            )
            results['loo_plots']['time_s'] = time.time() - t0
        except Exception as e:
            print(f"  ERROR in LOO plots: {e}")
            traceback.print_exc()
            results['loo_plots'] = {'status': 'error', 'error': str(e)}

    # Task 2: HUST
    if 'hust' not in args.skip:
        try:
            t0 = time.time()
            results['hust'] = run_hust_validation(
                hust_path, args.output, args.calendar_model,
            )
            results['hust']['time_s'] = time.time() - t0
        except Exception as e:
            print(f"  ERROR in HUST validation: {e}")
            traceback.print_exc()
            results['hust'] = {'status': 'error', 'error': str(e)}

    # Task 3: FastCharge
    if 'fastcharge' not in args.skip:
        try:
            t0 = time.time()
            results['fastcharge'] = run_fastcharge_validation(
                fastcharge_path, args.output, args.calendar_model,
            )
            results['fastcharge']['time_s'] = time.time() - t0
        except Exception as e:
            print(f"  ERROR in FastCharge validation: {e}")
            traceback.print_exc()
            results['fastcharge'] = {'status': 'error', 'error': str(e)}

    # Cross-dataset summary
    total_time = time.time() - t_total
    results['total_time_s'] = total_time
    results['calendar_model'] = args.calendar_model
    results['resolution'] = args.resolution

    print("\n" + "=" * 70)
    print("  VALIDATION SUITE COMPLETE")
    print("=" * 70)
    for task, res in results.items():
        if isinstance(res, dict) and 'status' in res:
            status = res['status']
            extra = ''
            if 'mean_rmse' in res:
                extra = f", mean RMSE={res['mean_rmse']:.4f}"
            if 'time_s' in res:
                extra += f", {res['time_s']:.0f}s"
            print(f"  {task:20s}: {status}{extra}")
    print(f"\n  Total time: {_elapsed(t_total)}")
    print(f"  Results saved to: {args.output}")

    with open(os.path.join(args.output, 'validation_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
