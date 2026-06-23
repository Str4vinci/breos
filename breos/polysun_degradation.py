"""
Polysun-style battery degradation model.

Implements the Wöhler curve + Miner's linear damage accumulation methodology
used by Polysun (Vela Solaris) for battery lifetime estimation. This serves
as a comparison baseline against BREOS's Naumann-based continuous degradation.

Polysun methodology:
    1. Cycle counting: 20 DOD histogram bins (equal width)
    2. Cycle life: Wöhler curve N(DOD) = a * DOD^(-b)
    3. Damage: Miner's rule D = sum(n_i / N_i)
    4. Calendar life: Fixed (20 years for Li-ion)
    5. Total life: min(calendar_life, 1/D_annual)
    6. No temperature effects, no continuous SOH tracking

References:
    - Polysun User Manual, Section "Battery Lifetime" (Vela Solaris AG)
    - Weniger et al., "Performance Model for PV-Battery Systems (PerMod)",
      HTW Berlin, 2023
    - Palmgren-Miner linear damage hypothesis (Miner, 1945)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from breos.constants import (
    POLYSUN_CALENDAR_LIFE_LEAD,
    POLYSUN_CALENDAR_LIFE_LION,
    WOEHLER_LFP_CONSERVATIVE_A,
    WOEHLER_LFP_CONSERVATIVE_B,
    WOEHLER_LFP_OPTIMISTIC_A,
    WOEHLER_LFP_OPTIMISTIC_B,
    WOEHLER_LFP_TYPICAL_A,
    WOEHLER_LFP_TYPICAL_B,
)


@dataclass
class PolysunDegradationConfig:
    """Configuration for Polysun-style degradation model.

    Attributes:
        woehler_a: Scale parameter for Wöhler curve N(DOD) = a * DOD^(-b).
            Equals cycles to failure at 100% DOD.
        woehler_b: Shape parameter for Wöhler curve. Higher b means deeper
            cycles are disproportionately more damaging.
        calendar_life_years: Fixed calendar lifetime in years.
        n_bins: Number of DOD histogram bins (Polysun uses 20).
        min_doc: Minimum DOD to count as a cycle (fraction, 0-1).
        deep_cycle_threshold: DOD above which a cycle is classified as "deep".
    """

    woehler_a: float = WOEHLER_LFP_TYPICAL_A
    woehler_b: float = WOEHLER_LFP_TYPICAL_B
    calendar_life_years: float = POLYSUN_CALENDAR_LIFE_LION
    n_bins: int = 20
    min_doc: float = 0.01
    deep_cycle_threshold: float = 0.50


def woehler_cycles_to_failure(dod: float, a: float, b: float) -> float:
    """Cycles to failure from Wöhler curve: N(DOD) = a * DOD^(-b).

    Args:
        dod: Depth of discharge (fraction, 0-1). Must be > 0.
        a: Scale parameter (cycles to failure at DOD=1.0).
        b: Shape exponent (steepness of DOD sensitivity).

    Returns:
        Number of cycles to failure at the given DOD.
    """
    if dod <= 0:
        return float("inf")
    return a * dod ** (-b)


def compute_dod_histogram(
    soc_series: np.ndarray,
    n_bins: int = 20,
    min_doc: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Count cycles per DOD range from an SOC timeseries using peak detection.

    Uses simple local-extrema-based half-cycle detection to match Polysun's
    approach (not rainflow counting). Half-cycles are paired and binned by DOD.

    Args:
        soc_series: SOC timeseries (0-1 range), typically one year of data.
        n_bins: Number of equal-width DOD bins (Polysun uses 20).
        min_doc: Minimum DOD to include a cycle (fraction, 0-1).

    Returns:
        Tuple of (bin_centers, cycle_counts, total_cycles, deep_cycles) where:
            bin_centers: DOD value at center of each bin (length n_bins).
            cycle_counts: Number of full cycles in each bin (length n_bins).
            total_cycles: Total number of cycles detected.
            deep_cycles: Number of deep cycles (DOD > deep_cycle_threshold).
    """
    # Find local extrema (direction changes in SOC)
    diffs = np.diff(soc_series)
    # Remove zero-diffs to find actual direction changes
    nonzero_mask = diffs != 0
    if not np.any(nonzero_mask):
        return (
            np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins),
            np.zeros(n_bins),
            0,
            0,
        )

    nonzero_diffs = diffs[nonzero_mask]
    nonzero_indices = np.where(nonzero_mask)[0]

    # Direction changes indicate extrema
    sign_changes = np.diff(np.sign(nonzero_diffs))
    extrema_positions = nonzero_indices[1:][sign_changes != 0]

    # Include start and end
    extrema_values = np.concatenate(
        [
            [soc_series[0]],
            soc_series[extrema_positions],
            [soc_series[-1]],
        ]
    )

    # Compute half-cycle DODs
    half_cycle_dods = np.abs(np.diff(extrema_values))

    # Pair half-cycles into full cycles (two half-cycles = one full cycle)
    # Each half-cycle with DOD > min_doc counts as 0.5 cycles
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cycle_counts = np.zeros(n_bins)

    total_cycles = 0
    deep_cycles = 0

    for dod in half_cycle_dods:
        if dod < min_doc:
            continue
        bin_idx = min(int(dod * n_bins), n_bins - 1)
        cycle_counts[bin_idx] += 0.5  # Half-cycle
        total_cycles += 1  # Count half-cycles for total

    # Convert total to full-cycle equivalent
    total_cycles = int(np.sum(cycle_counts))
    deep_cycles = int(np.sum(cycle_counts[bin_centers >= 0.50]))

    return bin_centers, cycle_counts, total_cycles, deep_cycles


def compute_miner_damage(
    cycle_counts: np.ndarray,
    bin_centers: np.ndarray,
    woehler_a: float,
    woehler_b: float,
) -> float:
    """Compute cumulative damage using Miner's linear damage rule.

    D = sum(n_i / N_i) where n_i is the number of cycles at DOD_i and
    N_i = a * DOD_i^(-b) is the cycles to failure at that DOD.

    Args:
        cycle_counts: Number of cycles per DOD bin.
        bin_centers: DOD value at center of each bin.
        woehler_a: Wöhler curve scale parameter.
        woehler_b: Wöhler curve shape exponent.

    Returns:
        Cumulative damage fraction. D >= 1 means end of cycle life.
    """
    damage = 0.0
    for n_i, dod_i in zip(cycle_counts, bin_centers):
        if n_i <= 0 or dod_i <= 0:
            continue
        n_fail = woehler_cycles_to_failure(dod_i, woehler_a, woehler_b)
        damage += n_i / n_fail
    return damage


def predict_polysun_lifetime(
    annual_damage: float,
    calendar_life_years: float,
) -> Tuple[float, float, float]:
    """Predict battery lifetime following Polysun methodology.

    Total life = min(calendar_life, cycle_life) where
    cycle_life = 1 / annual_damage.

    Args:
        annual_damage: Miner's damage per year (D_annual).
        calendar_life_years: Fixed calendar lifetime in years.

    Returns:
        Tuple of (total_life, cycle_life, calendar_life) in years.
    """
    if annual_damage > 0:
        cycle_life = 1.0 / annual_damage
    else:
        cycle_life = float("inf")
    total_life = min(calendar_life_years, cycle_life)
    return total_life, cycle_life, calendar_life_years


def simulate_polysun_degradation(
    soc_series: np.ndarray,
    config: PolysunDegradationConfig,
    n_years: int = 20,
) -> pd.DataFrame:
    """Run Polysun-style degradation over multiple years.

    Polysun assumes constant annual usage (same SOC profile each year) and
    does not feed degradation back into the energy balance. Damage accumulates
    linearly; when cumulative damage >= 1, the battery is replaced.

    Args:
        soc_series: One year of SOC data (0-1 range). Reused each year.
        config: Polysun degradation configuration.
        n_years: Number of years to simulate.

    Returns:
        DataFrame with columns: Year, Damage_Annual, Damage_Cumulative,
        Cycle_Life_Years, Total_Life_Years, Replacement, SOH_Equivalent,
        Total_Cycles, Deep_Cycles.
    """
    # Compute annual cycle histogram once (same profile every year)
    bin_centers, cycle_counts, total_cycles, deep_cycles = compute_dod_histogram(
        soc_series,
        n_bins=config.n_bins,
        min_doc=config.min_doc,
    )

    # Annual Miner's damage
    annual_damage = compute_miner_damage(cycle_counts, bin_centers, config.woehler_a, config.woehler_b)

    # Predicted lifetime
    total_life, cycle_life, calendar_life = predict_polysun_lifetime(annual_damage, config.calendar_life_years)

    rows = []
    cumulative_damage = 0.0
    n_replacements = 0
    last_replacement_year = 0

    for year in range(1, n_years + 1):
        cumulative_damage += annual_damage
        replaced = False

        # Calendar check: has the battery exceeded calendar life since last replacement?
        # Track the actual replacement year instead of deriving it from n_replacements ×
        # int(total_life) — that approximation drifts once total_life is fractional or
        # once a cycle-driven replacement happens before the expected calendar point.
        years_since_replacement = year - last_replacement_year

        # Check both cycle and calendar end-of-life
        if cumulative_damage >= 1.0:
            replaced = True
            n_replacements += 1
            cumulative_damage = 0.0  # Reset after replacement
            last_replacement_year = year

        # Calendar replacement: check if we hit calendar life
        if not replaced and years_since_replacement >= config.calendar_life_years:
            replaced = True
            n_replacements += 1
            cumulative_damage = 0.0
            last_replacement_year = year

        # SOH equivalent: Polysun is binary, but for comparison we map
        # damage to an equivalent SOH assuming linear capacity fade
        # SOH = 1 - (damage * 0.20) maps D=1 to SOH=80% (typical EOL)
        soh_equivalent = max(0, 1.0 - cumulative_damage * 0.20) * 100.0

        rows.append(
            {
                "Year": year,
                "Damage_Annual": annual_damage,
                "Damage_Cumulative": cumulative_damage,
                "Cycle_Life_Years": cycle_life,
                "Calendar_Life_Years": calendar_life,
                "Total_Life_Years": total_life,
                "Replacement": replaced,
                "N_Replacements": n_replacements,
                "SOH_Equivalent": soh_equivalent,
                "Total_Cycles": total_cycles,
                "Deep_Cycles": deep_cycles,
            }
        )

    return pd.DataFrame(rows)
