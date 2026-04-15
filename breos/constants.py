"""
Physical and model constants for battery degradation models.
"""

import math

# Physical constant
R_GAS = 8.31446261815324  # J/(mol*K)

# Reference temperature for calendar aging (25°C)
T_REF_K = 25.0 + 273.15

# === Naumann 2020 parameters (Table 6) ===
# Capacity fade
NAUMANN_K0_PERCENT = 0.0012571  # % per sqrt(second) at 25°C, SOC=1.0
NAUMANN_EA_J_MOL = 17126.0  # Activation Energy in J/mol
NAUMANN_EXPONENT_B = 0.5  # Time exponent (Square-root time)
NAUMANN_SOC_EXPONENT_N = 0.75  # SOC exponent

# Cycle aging parameters (Naumann Eq. 5-10)
A_Q = 0.0630
B_Q = 0.0971
C_DOC_Q = 4.0253
D_DOC_Q = 1.0923
Z_Q = 0.5

A_R = 0.0020
B_R = 0.0021
C_DOC_R = 6.8477
D_DOC_R = 0.91882
Z_R = 1.0

# Calendar resistance fade parameters (Naumann 2020 Table 6)
NAUMANN_K0_R_PERCENT = 0.0010483  # % per sqrt(second) at 25°C, SOC=1.0
NAUMANN_EA_R_J_MOL = 14406.0  # Activation Energy for resistance (J/mol)
NAUMANN_EXPONENT_B_R = 0.5  # Time exponent (same sqrt-time as capacity)
NAUMANN_SOC_EXPONENT_N_R = 0.75  # SOC exponent for resistance

# === Lam 2025 Parameters (Derived for LFP, lab conditions) ===
# k0 derived from Lam Fig 3A (0.135% loss per week^0.75) and converted to seconds.
# (0.00135 / (604800^0.75))
LAM_K0_FRAC = 6.17e-8  # Fraction loss per second^0.75 at 25°C, SOC=1.0
LAM_EA_J_MOL = 38500.0  # Activation Energy (0.4 eV converted to J/mol)
LAM_EXPONENT_B = 0.75  # Time exponent (Based on Lam Fig 5A LFP cluster)
LAM_SOC_EXPONENT_N = 0.75  # Assume same SOC exponent as Naumann/general LFP

# === Lam + Naumann field-calibrated parameters (current default, 15-min) ===
# Re-run on 2026-04-12 against the effective 5-system Zenodo LFP field set
# (systems 14, 15, 17, 20, 21) using the finalized validation pipeline.
# Mean RMSE = 4.40pp on the full calibration fit; LOO mean CV RMSE = 6.0pp.
LAM_NAUMANN_FIELD_CALIBRATED_K0_FRAC = 8.019530e-08
LAM_NAUMANN_FIELD_CALIBRATED_EA_J_MOL = 11876.09
LAM_NAUMANN_FIELD_CALIBRATED_EXPONENT_B = 0.7701374
LAM_NAUMANN_FIELD_CALIBRATED_SOC_EXPONENT_N = 0.1010299

# Backward-compatible aliases for the primary field-calibrated model.
LAM_CAL_K0_FRAC = LAM_NAUMANN_FIELD_CALIBRATED_K0_FRAC
LAM_CAL_EA_J_MOL = LAM_NAUMANN_FIELD_CALIBRATED_EA_J_MOL
LAM_CAL_EXPONENT_B = LAM_NAUMANN_FIELD_CALIBRATED_EXPONENT_B
LAM_CAL_SOC_EXPONENT_N = LAM_NAUMANN_FIELD_CALIBRATED_SOC_EXPONENT_N

# === Lam + Naumann field-calibrated parameters (previous 15-min fit) ===
# Retained for reproducibility under the model name
# `lam_naumann_field_calibrated_old`.
LAM_NAUMANN_FIELD_CALIBRATED_OLD_K0_FRAC = 5.7841386975994235e-08
LAM_NAUMANN_FIELD_CALIBRATED_OLD_EA_J_MOL = 3748.510470404377
LAM_NAUMANN_FIELD_CALIBRATED_OLD_EXPONENT_B = 0.788762656479986
LAM_NAUMANN_FIELD_CALIBRATED_OLD_SOC_EXPONENT_N = 0.10230418425202481

# Optional aliases for code that wants to reference the old fit directly.
LAM_CAL_OLD_K0_FRAC = LAM_NAUMANN_FIELD_CALIBRATED_OLD_K0_FRAC
LAM_CAL_OLD_EA_J_MOL = LAM_NAUMANN_FIELD_CALIBRATED_OLD_EA_J_MOL
LAM_CAL_OLD_EXPONENT_B = LAM_NAUMANN_FIELD_CALIBRATED_OLD_EXPONENT_B
LAM_CAL_OLD_SOC_EXPONENT_N = LAM_NAUMANN_FIELD_CALIBRATED_OLD_SOC_EXPONENT_N

# === Lam Calibrated Parameters — Hourly resolution (legacy) ===
# Same calibration as above but fitted at hourly (1h) resolution.
# Mean CV RMSE = 6.0pp, train RMSE = 4.3pp, overfitting gap 1.7pp.
# Superseded by 15-min calibration; retained for reproducibility.
LAM_CAL_HOURLY_K0_FRAC = 4.375846e-07  # 7.1x higher than lab
LAM_CAL_HOURLY_EA_J_MOL = 11655.12  # 0.30x lab
LAM_CAL_HOURLY_EXPONENT_B = 0.6782  # 0.90x lab
LAM_CAL_HOURLY_SOC_EXPONENT_N = 0.1144  # 0.15x lab

# === Lam Calibrated Parameters — Relaxed Calendar Aging (1.5× Ea) ===
# Sensitivity variant: 1.5× activation energy reduces calendar degradation
# at typical residential temperatures (below 25°C reference).  Higher Ea
# means the Arrhenius rate drops more steeply below T_ref, producing
# slower calendar aging in mild climates like Porto (~15°C annual mean).
LAM_CAL_RELAXED_K0_FRAC = LAM_CAL_K0_FRAC  # Same rate constant
LAM_CAL_RELAXED_EA_J_MOL = LAM_CAL_EA_J_MOL * 1.5  # 1.5× activation energy
LAM_CAL_RELAXED_EXPONENT_B = LAM_CAL_EXPONENT_B  # Same time exponent
LAM_CAL_RELAXED_SOC_EXPONENT_N = LAM_CAL_SOC_EXPONENT_N  # Same SOC sensitivity

# === Modern LFP Parameters (projected for 2020+ cells) ===
# The lam_naumann_field_calibrated model was fitted to 2015-era residential LFP systems.
# Modern cells (CATL/BYD, 2020+) use improved electrolyte formulations and
# BMS designs that are expected to reduce calendar aging. No field calibration
# data is available for these newer cells yet.
# Conservative estimate: halve the calendar rate constant k0 while keeping
# the same physical mechanisms (activation energy, time exponent, SOC sensitivity).
MODERN_LFP_K0_FRAC = LAM_CAL_K0_FRAC * 0.5  # 50% reduction in calendar rate constant
MODERN_LFP_EA_J_MOL = LAM_CAL_EA_J_MOL  # Same activation energy (same mechanism)
MODERN_LFP_EXPONENT_B = LAM_CAL_EXPONENT_B  # Same time exponent
MODERN_LFP_SOC_EXPONENT_N = LAM_CAL_SOC_EXPONENT_N  # Same SOC sensitivity

# Default battery parameters
# For 95% round-trip efficiency, use sqrt(0.95) ≈ 0.9747 for each direction
_ROUND_TRIP_EFFICIENCY = 0.95
DEFAULT_CHARGE_EFFICIENCY = math.sqrt(_ROUND_TRIP_EFFICIENCY)  # ~0.9747
DEFAULT_DISCHARGE_EFFICIENCY = math.sqrt(_ROUND_TRIP_EFFICIENCY)  # ~0.9747
DEFAULT_STANDBY_LOSS_WH = 5.0
DEFAULT_MAX_SOC = 0.90  # LFP long-life window (10-90%); use 1.0 for max performance
DEFAULT_MIN_SOC = 0.10  # LFP long-life window; legacy NMC-style was 0.20

# === LFP Temperature-Capacity Derating ===
# Piecewise-linear approximation for LFP usable capacity vs. temperature.
# Based on typical LFP characterisation data (Hesse et al. 2017; manufacturer datasheets).
# At residential discharge rates (0.1–0.3C), LFP is relatively temperature-stable:
#   25°C → 100% (reference), 0°C → ~95%, -10°C → ~85%
LFP_CAP_DERATE_PER_C_MODERATE = 0.002  # fraction/°C below 25°C (down to 0°C)
LFP_CAP_DERATE_PER_C_COLD = 0.010  # fraction/°C below 0°C  (steeper derating)

# === Lumped Thermal Model ===
# Quasi-steady-state thermal resistance from pack to ambient.
# For a residential home battery pack (3-10 kWh), the effective
# pack-to-ambient resistance is much lower than a single cell due
# to large surface area and (often) forced ventilation.
# Range: 0.01-0.10 K/W.  0.05 K/W gives ~1-3°C rise above ambient
# during typical 0.3C charging of a 5 kWh pack.
DEFAULT_THERMAL_RESISTANCE_KW = 0.05  # K per W of heat dissipation

# === Indoor Temperature Model ===
# Residential batteries are installed indoors (garage, utility room) where
# building thermal mass buffers outdoor temperature extremes.
# T_indoor = clamp(alpha * T_outdoor + (1 - alpha) * T_setpoint, floor, ceiling)
DEFAULT_INDOOR_MODEL_ENABLED = True  # Apply indoor buffering by default
DEFAULT_INDOOR_SETPOINT_C = 22.0  # Indoor comfort midpoint (°C)
DEFAULT_INDOOR_COUPLING_ALPHA = 0.3  # Outdoor influence (0=fully insulated, 1=outdoor)
DEFAULT_INDOOR_FLOOR_C = 15.0  # Min indoor temp — even unheated garage in mild climate
DEFAULT_INDOOR_CEILING_C = 35.0  # Max indoor temp — summer heat buildup

# === Polysun Wöhler Curve Parameters (Cycle Life vs DOD) ===
# Polysun models cycle life using a Wöhler (S-N) curve: N(DOD) = a * DOD^(-b)
# where N = cycles to failure, DOD = depth of discharge (0-1).
#
# LFP parameters derived from published cycle life data:
#   - Wang et al. 2011: ~3000 cycles at 100% DOD, ~7500 at 50% DOD for LFP
#   - Xu et al. 2018: ~2000-5000 at 100% DOD depending on temperature
#   - Safari & Delacourt 2011: semi-empirical LFP cycle life model
#
# Conservative: shorter life, e.g., early-generation LFP or harsh conditions
# Typical: mid-range LFP (most residential home storage systems)
# Optimistic: modern high-quality LFP (CATL, BYD 2020+ cells)
# References: Xu et al. 2018 (IEEE Trans. Smart Grid), Wang et al. 2011 (J. Power Sources)
WOEHLER_LFP_CONSERVATIVE_A = 3500.0  # Cycles at 100% DOD
WOEHLER_LFP_CONSERVATIVE_B = 1.5  # DOD exponent
WOEHLER_LFP_TYPICAL_A = 5000.0  # Cycles at 100% DOD
WOEHLER_LFP_TYPICAL_B = 1.6  # DOD exponent
WOEHLER_LFP_OPTIMISTIC_A = 6000.0  # Cycles at 100% DOD
WOEHLER_LFP_OPTIMISTIC_B = 1.7  # DOD exponent
# Reference data points for validation (typical, a=5000, b=1.6):
#   DOD=1.0 → 5000,  DOD=0.8 → 7440,  DOD=0.5 → 15157,  DOD=0.2 → 67860

# Polysun default calendar lifetimes (fixed, no temperature dependence)
POLYSUN_CALENDAR_LIFE_LION = 20.0  # Li-ion (LFP)
POLYSUN_CALENDAR_LIFE_LEAD = 10.0  # Lead-acid
