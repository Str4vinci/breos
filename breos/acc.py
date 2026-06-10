"""
Collective Self-Consumption (ACC) Module.

This module implements algorithms for Energy Sharing in Renewable Energy Communities (REC)
and Collective Self-Consumption (ACC) schemes.

Supported Sharing Coefficients:
1. Fixed (Simples): Static shares for each participant.
2. Time-of-Use (Diferenciados): Different shares for Weekdays vs Weekends.
3. Variable (Dinâmicos): Proportional to real-time consumption.

Includes optimization capabilities to find ideal coefficients.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def calculate_variable_coefficients(load_matrix: np.ndarray) -> np.ndarray:
    """
    Implements 'Coeficientes Variáveis' (Dynamic).
    The Coefficient (alpha) for each user is proportional to their consumption
    at that specific timestamp.
    alpha_i = Load_i / Total_Load

    Args:
        load_matrix: (N_Timesteps, N_Households) array of loads

    Returns:
        shares_matrix: (N_Timesteps, N_Households) array of coefficients (0-1)
    """
    # Sum across households (axis 1) to get total community load per hour
    total_load = np.sum(load_matrix, axis=1)

    # Avoid division by zero
    safe_total_load = total_load.copy()
    safe_total_load[safe_total_load == 0] = 1.0

    # Calculate shares for every hour
    # Broadcasting: (T, N) / (T, 1)
    shares_matrix = load_matrix / safe_total_load[:, np.newaxis]

    return shares_matrix


def calculate_fixed_time_of_use(
    index: pd.DatetimeIndex, n_households: int, weekday_shares: List[float], weekend_shares: List[float]
) -> np.ndarray:
    """
    Implements 'Coeficientes Fixos Diferenciados'.
    Applies one set of fixed % for weekdays, another for weekends.

    Args:
        index: DatetimeIndex of the simulation
        n_households: Number of participants
        weekday_shares: List of percentages (0-100) for weekdays
        weekend_shares: List of percentages (0-100) for weekends

    Returns:
        shares_matrix: (N_Timesteps, N_Households) array of coefficients (0-1)
    """
    shares_matrix = np.zeros((len(index), n_households))

    # Create boolean mask for weekends (Saturday=5, Sunday=6)
    is_weekend = index.dayofweek >= 5

    # Apply Weekday Shares
    shares_matrix[~is_weekend] = np.array(weekday_shares) / 100.0

    # Apply Weekend Shares
    shares_matrix[is_weekend] = np.array(weekend_shares) / 100.0

    return shares_matrix


def optimize_fixed_shares(
    pv_curve: np.ndarray, load_matrix: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Optimize fixed shares to minimize grid import + small penalty on export.
    If mask is provided, only optimize for those timesteps.

    Args:
        pv_curve: (N_Timesteps,) array of Total PV generation
        load_matrix: (N_Timesteps, N_Households) array of loads
        mask: Optional boolean mask to select specific timesteps (e.g., weekdays)

    Returns:
        Optimal shares as percentages (0-100)
    """
    n_households = load_matrix.shape[1]

    if mask is not None:
        pv_subset = pv_curve[mask]
        load_subset = load_matrix[mask, :]
    else:
        pv_subset = pv_curve
        load_subset = load_matrix

    pv_reshaped = pv_subset[:, np.newaxis]

    def objective(shares):
        # shares are 0-100
        allocated_pv = pv_reshaped * (shares / 100.0)
        net_load = load_subset - allocated_pv
        imports = np.sum(np.maximum(0, net_load))
        exports = np.sum(np.maximum(0, -net_load))
        return imports + 0.0001 * exports

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 100}
    bounds = tuple((0, 100) for _ in range(n_households))
    initial_guess = np.full(n_households, 100.0 / n_households)

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=1e-3,
        options={"maxiter": 150},
    )

    return result.x if result.success else initial_guess


@dataclass
class ACCResult:
    shares_matrix: np.ndarray
    allocated_pv: np.ndarray
    imports: np.ndarray
    exports: np.ndarray
    self_sufficiency_global: float
    self_consumption_global: float
    household_results: List[Dict[str, float]]


def calculate_acc_metrics(pv_curve: np.ndarray, load_matrix: np.ndarray, shares_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Calculate full results for an ACC scenario.
    """
    # Ensure PV matches Load length
    min_len = min(len(pv_curve), len(load_matrix))
    pv_curve = pv_curve[:min_len]
    load_matrix = load_matrix[:min_len]
    shares_matrix = shares_matrix[:min_len]

    allocated_pv = pv_curve[:, np.newaxis] * shares_matrix
    net_load = load_matrix - allocated_pv

    imports = np.maximum(0, net_load)
    exports = np.maximum(0, -net_load)

    total_load = np.sum(load_matrix)
    total_import = np.sum(imports)
    total_export = np.sum(exports)
    total_pv = np.sum(pv_curve)

    ss = (1 - total_import / total_load) * 100 if total_load > 0 else 0
    scr = ((total_pv - total_export) / total_pv) * 100 if total_pv > 0 else 0

    households = []
    for i in range(load_matrix.shape[1]):
        h_load = np.sum(load_matrix[:, i])
        h_pv = np.sum(allocated_pv[:, i])
        h_imp = np.sum(imports[:, i])
        h_exp = np.sum(exports[:, i])
        h_ss = (1 - h_imp / h_load) * 100 if h_load > 0 else 0
        h_scr = ((h_pv - h_exp) / h_pv) * 100 if h_pv > 0 else 0

        households.append(
            {
                "id": i,
                "load": h_load,
                "allocated_pv": h_pv,
                "import": h_imp,
                "export": h_exp,
                "self_sufficiency": h_ss,
                "self_consumption": h_scr,
            }
        )

    return {
        "total_load": total_load,
        "total_pv": total_pv,
        "total_import": total_import,
        "total_export": total_export,
        "self_sufficiency": ss,
        "self_consumption": scr,
        "allocated_pv": allocated_pv,
        "imports": imports,
        "exports": exports,
        "households": households,
    }


def format_energy(wh_value: float) -> str:
    """Auto-scale energy values to appropriate units (Wh, kWh, MWh, GWh)"""
    val = abs(wh_value)
    if val < 1000:
        return f"{wh_value:.2f} Wh"
    elif val < 1_000_000:
        return f"{wh_value / 1000:.2f} kWh"
    elif val < 1_000_000_000:
        return f"{wh_value / 1_000_000:.2f} MWh"
    else:
        return f"{wh_value / 1_000_000_000:.2f} GWh"


def generate_acc_report(
    results_dir: str, results_list: List[Dict[str, Any]], baseline_ss: float, share_details: Dict[str, Any]
) -> None:
    """
    Generate a detailed text report comparing ACC strategies.

    Args:
        results_dir: Directory to save the report
        results_list: List of result dictionaries containing 'Mode', 'Self-Sufficiency', etc.
        baseline_ss: Baseline self-sufficiency percentage for gain calculation
        share_details: Dictionary containing share arrays for reporting (Equal, Optimized, etc.)
    """
    import os

    report_path = os.path.join(results_dir, "Comprehensive_Comparison_Report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ACC COMPREHENSIVE STRATEGY COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Fixed share details
        f.write("FIXED SHARE CONFIGURATIONS:\n")
        f.write("-" * 40 + "\n")

        if "equal_shares" in share_details:
            f.write(f"Equal Shares:     {[f'{s:.2f}%' for s in share_details['equal_shares']]}\n")
        if "optimized_shares" in share_details:
            f.write(f"Optimized Shares: {[f'{s:.2f}%' for s in share_details['optimized_shares']]}\n")
        if "weekday_shares" in share_details:
            f.write(f"Weekday Shares:   {[f'{s:.2f}%' for s in share_details['weekday_shares']]}\n")
        if "weekend_shares" in share_details:
            f.write(f"Weekend Shares:   {[f'{s:.2f}%' for s in share_details['weekend_shares']]}\n")

        f.write("\n")

        f.write("PERFORMANCE RESULTS:\n")
        f.write("=" * 70 + "\n")
        for r in results_list:
            f.write(f"\nStrategy: {r['Strategy']}\n")
            f.write("-" * 40 + "\n")
            total_pv = r.get("Total_PV_Wh", 0)
            total_export = r.get("Total_Export_Wh", 0)
            used_energy = total_pv - total_export
            export_pct = (total_export / total_pv * 100) if total_pv > 0 else 0

            f.write(f"  Community Load:    {format_energy(r.get('Total_Load_Wh', 0)):>12}\n")
            f.write(f"  Self-Sufficiency:  {r['Self_Sufficiency_%']:>6.2f}% (Load covered by PV)\n")
            f.write(f"  Grid Import:       {format_energy(r.get('Total_Import_Wh', 0)):>12}\n")
            f.write("-" * 40 + "\n")
            f.write("  PV Array Statistics:\n")
            f.write(f"    Total Generation:    {format_energy(total_pv):>12}\n")
            f.write(f"    Used by Community:   {format_energy(used_energy):>12} ({r['Self_Consumption_%']:.2f}%)\n")
            f.write(f"    Exported to Grid:    {format_energy(total_export):>12} ({export_pct:.2f}%)\n")

            if "Households" in r:
                f.write("\n  Household Breakdown:\n")
                f.write(f"    {'ID':<5} | {'Load':<10} | {'Alloc. PV':<10} | {'Self-Suff.':<10} | {'Self-Cons.':<10}\n")
                f.write("    " + "-" * 56 + "\n")
                for hh in r["Households"]:
                    # id is int usually
                    hid = f"HH{hh['id'] + 1}"
                    f.write(
                        f"    {hid:<5} | {format_energy(hh['load']):<10} | {format_energy(hh['allocated_pv']):<10} | {hh['self_sufficiency']:>9.1f}% | {hh['self_consumption']:>9.1f}%\n"
                    )

        f.write("\n" + "=" * 70 + "\n")
        f.write("PERFORMANCE GAINS (vs Baseline):\n")
        f.write("-" * 40 + "\n")

        # Assume first result is baseline if not specified differently?
        # Or compare against baseline_ss arg.

        for r in results_list:
            if r["Strategy"] == "Fixed Equal":
                continue  # Skip baseline comparison with itself? Or show 0.

            gain = r["Self_Sufficiency_%"] - baseline_ss
            f.write(f"  {r['Strategy']:>30}: {gain:+.2f}%\n")

    logger.info("Detailed report saved to: %s", report_path)
