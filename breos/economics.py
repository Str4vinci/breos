"""
Economics module for cost analysis and projections.

This module handles:
- CAPEX calculations (PV, battery, installation)
- OPEX calculations (maintenance, grid costs)
- Multi-year cost projections with inflation and degradation
- Payback period analysis
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from breos.tariffs import DEFAULT_CURRENCY

# Default battery and replacement cost per kWh of battery capacity (currency/kWh)
BATTERY_REPLACEMENT_COST_PER_KWH: float = 500.0

# Where a replacement year carries no recorded instant, the outlay is booked
# at the middle of that year. The simulation records the real instant and the
# runners carry it through, so this covers summaries built before the instant
# was carried and callers that supply none; mid-year is the least biased
# choice for an unknown day. The resolved time is reported back in the
# projection's ``Replacement_Time_Years`` column, so the assumption is visible
# rather than buried here.
DEFAULT_REPLACEMENT_YEAR_FRACTION: float = 0.5

# Canonical translation from the public cost-catalogue/config vocabulary to
# CostParams attributes. App presets, App ``[costs]`` overrides, and the
# lower-level ``cost_params_from_config`` helper all share this mapping.
COST_CONFIG_KEY_TO_PARAM: dict[str, str] = {
    "electricity_cost": "electricity_cost",
    "electricity_sold_cost": "electricity_sold_cost",
    "daily_power_cost": "daily_power_cost",
    "module_cost_per_w": "module_cost_per_w",
    "storage_cost_per_kwh": "battery_cost_per_kwh",
    "inverter_cost_per_kw_hybrid": "inverter_cost_per_kw",
    "inverter_cost_per_kw_simple": "inverter_cost_per_kw_nobatt",
    "installation_cost_per_module": "installation_cost_per_module",
    "installation_cost_battery": "battery_installation_cost",
    "other_cost_per_module": "other_cost_per_module",
    "other_costs": "other_cost_fixed",
    "land_cost": "land_cost",
    "maintenance_cost_per_panel": "maintenance_cost_per_panel",
    "maintenance_cost": "maintenance_cost_fixed",
    "operation_cost": "operation_cost",
}


# The flat energy prices, which a time-of-use tariff replaces.
FLAT_PRICE_COST_KEYS: tuple[str, ...] = ("electricity_cost", "electricity_sold_cost", "daily_power_cost")
# The cost keys :func:`calculate_costs` prices only with a battery, and only without one.
BATTERY_COST_KEYS: tuple[str, ...] = (
    "storage_cost_per_kwh",
    "installation_cost_battery",
    "inverter_cost_per_kw_hybrid",
)
PV_ONLY_COST_KEYS: tuple[str, ...] = ("inverter_cost_per_kw_simple",)


def default_amount_cost_keys(*, tariff: bool, battery: bool | None) -> tuple[str, ...]:
    """The cost keys a run prices with whose :class:`CostParams` default is an amount of money.

    Those defaults are amounts in the default currency, EUR; a zero default
    is zero in any currency and is left out. ``tariff`` drops the flat energy
    prices a tariff replaces. ``battery`` keeps the keys priced with a
    battery (True), without one (False), or both (None, for a search that
    may install either).
    """
    defaults = CostParams()
    skipped = set(FLAT_PRICE_COST_KEYS if tariff else ())
    if battery is True:
        skipped.update(PV_ONLY_COST_KEYS)
    elif battery is False:
        skipped.update(BATTERY_COST_KEYS)
    return tuple(
        key for key, param in COST_CONFIG_KEY_TO_PARAM.items() if key not in skipped and getattr(defaults, param) != 0
    )


# The one default set for every projection entry point (ADR 0003 E6). Both are
# nominal annual rates; an explicit 0.0 is valid and used as given, so callers
# test for an absent key, never a falsy value.
DEFAULT_DISCOUNT_RATE = 0.03
DEFAULT_INFLATION_RATE = 0.02


@dataclass
class CostParams:
    """Cost parameters for economic analysis."""

    # Electricity prices
    electricity_cost: float = 0.27  # currency/kWh purchased
    electricity_sold_cost: float = 0.06  # currency/kWh sold to grid
    daily_power_cost: float = 0.30  # currency per day connection fee

    # Equipment costs
    module_cost_per_w: float = 0.125  # currency/W
    battery_cost_per_kwh: float = BATTERY_REPLACEMENT_COST_PER_KWH  # currency/kWh
    dc_ac_ratio: float = 1.25  # DC/AC sizing ratio for inverter CAPEX
    inverter_cost_per_kw: float = 102.58  # currency/kW (with battery)
    inverter_cost_per_kw_nobatt: float = 48.37  # currency/kW (without battery)
    installation_cost_per_module: float = 350.0  # currency/module
    battery_installation_cost: float = 350.0  # currency, fixed
    other_cost_per_module: float = 50.0  # currency/module: cables, etc.
    other_cost_fixed: float = 0.0  # currency, fixed misc. costs
    land_cost: float = 0.0

    # Operations
    maintenance_cost_per_panel: float = 10.0  # currency/panel/year
    maintenance_cost_fixed: float = 0.0  # currency/year, fixed
    operation_cost: float = 0.0  # currency/year, additional

    # Analysis parameters
    inflation_rate: float = DEFAULT_INFLATION_RATE
    sell_price_inflation: float = 0.0
    discount_rate: float = DEFAULT_DISCOUNT_RATE
    pv_degradation_rate: float = 0.005


def cost_params_from_config(
    costs_config: Optional[Dict[str, Any]] = None,
    financials_config: Optional[Dict[str, Any]] = None,
) -> CostParams:
    """Build :class:`CostParams` from BREOS cost and financial config keys.

    Missing keys fall back to the :class:`CostParams` dataclass defaults, so
    the config and direct-construction paths cannot diverge.
    """
    costs_config = costs_config or {}
    financials_config = financials_config or {}
    if "panel_wp" in costs_config:
        raise ValueError(
            "costs.panel_wp was removed in 0.7.0: CAPEX is priced at the selected module's own "
            "rating. Remove the key, or select a module with the wattage you meant."
        )
    defaults = CostParams()

    params = {
        param_key: costs_config.get(config_key, getattr(defaults, param_key))
        for config_key, param_key in COST_CONFIG_KEY_TO_PARAM.items()
    }
    # Historical lower-level callers can supply the two energy prices through
    # ``financials_config``; an explicit costs value still takes precedence.
    for key in ("electricity_cost", "electricity_sold_cost"):
        if key not in costs_config:
            params[key] = financials_config.get(key, getattr(defaults, key))
    params.update(
        dc_ac_ratio=costs_config.get("dc_ac_ratio", defaults.dc_ac_ratio),
        inflation_rate=financials_config.get("inflation_rate", defaults.inflation_rate),
        sell_price_inflation=financials_config.get("sell_price_inflation", defaults.sell_price_inflation),
        discount_rate=financials_config.get("discount_rate", defaults.discount_rate),
        pv_degradation_rate=financials_config.get("pv_degradation_rate", defaults.pv_degradation_rate),
    )
    return CostParams(**params)


def replacement_event_cost(battery_kwh: float, cost_per_kwh: float, configured: Any = None) -> float:
    """The t = 0 price of one battery replacement.

    The physics reports when a pack is swapped and its nominal capacity; the
    economics prices each swap here, so App, Monte Carlo and the optimizer
    price replacements the same way. ``configured`` is an explicit price per
    replacement, as the optimizer's ``battery.replacement_cost`` gives it;
    ``None``, ``"auto"`` or ``"calculate"`` prices the pack at
    ``cost_per_kwh``.

    Raises:
        ValueError: If ``configured`` is not a finite non-negative number or
            one of the words above.
    """
    if configured is None or (isinstance(configured, str) and configured.strip().lower() in {"auto", "calculate"}):
        return float(battery_kwh) * float(cost_per_kwh)
    if isinstance(configured, bool):
        raise ValueError("battery.replacement_cost must be a non-negative number or 'calculate'")
    try:
        price = float(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError("battery.replacement_cost must be a non-negative number or 'calculate'") from exc
    if not np.isfinite(price) or price < 0.0:
        raise ValueError("battery.replacement_cost must be a non-negative number or 'calculate'")
    return price


def calculate_costs(
    n_modules: int,
    module_power_w: float,
    battery_capacity_wh: float = 0.0,
    cost_params: Optional[CostParams] = None,
    replacement_cost_each: Optional[float] = None,
    inverter_ac_capacity_w: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate system costs (CAPEX) and return cost dictionary.

    Args:
        n_modules: Number of PV modules
        module_power_w: Power per module in Watts (STC)
        battery_capacity_wh: Battery capacity in Wh (0 for no battery)
        cost_params: Cost parameters
        replacement_cost_each: The t = 0 price of one battery replacement,
            from :func:`replacement_event_cost`. ``None`` prices the pack at
            ``cost_params.battery_cost_per_kwh``.
        inverter_ac_capacity_w: The inverter AC rating to price, in W, the one
            the dispatch clips at. ``None`` sizes it as the DC peak over
            ``cost_params.dc_ac_ratio``.

    Returns:
        Dictionary with cost breakdown and totals. ``replacement_cost_each``
        is what the projection prices each simulated replacement at.
    """
    if cost_params is None:
        cost_params = CostParams()

    total_power_kw = n_modules * module_power_w / 1000
    if inverter_ac_capacity_w is not None:
        inverter_power_kw = inverter_ac_capacity_w / 1000
    elif cost_params.dc_ac_ratio > 0:
        inverter_power_kw = total_power_kw / cost_params.dc_ac_ratio
    else:
        inverter_power_kw = total_power_kw
    has_battery = battery_capacity_wh > 1

    # PV module costs
    pv_cost = cost_params.module_cost_per_w * module_power_w * n_modules

    # Installation costs
    installation_cost = cost_params.installation_cost_per_module * n_modules
    if has_battery:
        installation_cost += cost_params.battery_installation_cost

    # Battery costs
    if has_battery:
        battery_cost = (battery_capacity_wh / 1000) * cost_params.battery_cost_per_kwh
        inverter_cost = cost_params.inverter_cost_per_kw * inverter_power_kw
    else:
        battery_cost = 0.0
        inverter_cost = cost_params.inverter_cost_per_kw_nobatt * inverter_power_kw

    # Other costs
    other_costs = (cost_params.other_cost_per_module * n_modules) + cost_params.other_cost_fixed

    # Maintenance
    annual_operation_cost = (
        cost_params.maintenance_cost_per_panel * n_modules
        + cost_params.maintenance_cost_fixed
        + cost_params.operation_cost
    )

    # Total CAPEX
    total_initial_cost = (
        pv_cost + inverter_cost + battery_cost + installation_cost + cost_params.land_cost + other_costs
    )

    return {
        "electricity_cost": cost_params.electricity_cost,
        "electricity_sold_cost": cost_params.electricity_sold_cost,
        "daily_power_cost": cost_params.daily_power_cost,
        "total_initial_cost": total_initial_cost,
        "annual_operation_cost": annual_operation_cost,
        "pv_cost": pv_cost,
        "inverter_cost": inverter_cost,
        "battery_cost": battery_cost,
        "installation_cost": installation_cost,
        "other_costs": other_costs,
        "replacement_cost_each": (
            replacement_event_cost(battery_capacity_wh / 1000, cost_params.battery_cost_per_kwh)
            if replacement_cost_each is None
            else float(replacement_cost_each)
        ),
    }


def replacement_booking_time(
    relative_years,
    year_fractions=None,
    has_replacement=None,
) -> np.ndarray:
    """Project time, in years from commissioning, at which replacements are booked.

    A battery replacement is a single transaction on the day the pack is
    swapped, not a flow spread over the year it falls in. Booking it at a
    calendar-year granularity is what makes the inflation and the discount
    disagree: inflating by ``(1 + i) ** (year - 1)`` values the outlay at the
    start of the year while discounting by ``(1 + d) ** year`` values it at
    the end, so neither matches the swap. This returns the instant both should
    use.

    Args:
        relative_years: 1-based projection years.
        year_fractions: Position of the swap within its year, in ``[0, 1]``.
            ``None`` or a non-finite entry falls back to
            :data:`DEFAULT_REPLACEMENT_YEAR_FRACTION`. Where a year holds more
            than one replacement, the caller supplies the cost-weighted mean
            instant; discounting is not linear in time, so that aggregate is
            exact only when the events share a year-fraction, which is
            adequate for the sub-annual pack lives it would take to reach it.
        has_replacement: Mask of years that carry a replacement. Years outside
            it return NaN.

    Returns:
        Array of project times. A swap 109 days into year 11 gives 10.299.
    """
    years = np.asarray(relative_years, dtype=float)
    if year_fractions is None:
        fractions = np.full(years.shape, np.nan, dtype=float)
    else:
        fractions = np.asarray(year_fractions, dtype=float)
    fractions = np.where(np.isfinite(fractions), fractions, DEFAULT_REPLACEMENT_YEAR_FRACTION)
    booked = years - 1.0 + fractions
    if has_replacement is None:
        return booked
    return np.where(np.asarray(has_replacement, dtype=bool), booked, np.nan)


def _booking_exponents(replacement_time: np.ndarray, relative_years) -> np.ndarray:
    """Fill the NaN of non-replacement years so the power is finite.

    Those years carry a zero outlay, so the exponent cannot reach the result;
    it only has to not be NaN.
    """
    years = np.asarray(relative_years, dtype=float)
    return np.where(np.isfinite(replacement_time), replacement_time, years)


def replacement_fraction_from_steps(replacement_steps, n_steps: int) -> float:
    """Mean within-year position of the interval ends where a pack was swapped.

    ``Battery_Replaced`` flags the last interval run by the old pack. The swap
    occurs at that interval's end, so the zero-based step index advances by
    one before it is divided by the year's step count. On an hourly year, the
    interval at index 2616 ends at step boundary 2617 of 8760.

    Returns:
        Fraction in ``(0, 1]``, or NaN when the year holds no replacement.
    """
    steps = np.asarray(replacement_steps, dtype=float)
    if steps.size == 0 or n_steps <= 0:
        return float("nan")
    return float(np.mean((steps + 1.0) / float(n_steps)))


def _discount_annual_with_replacement(
    annual: pd.Series,
    replacement: pd.Series,
    discount_factors: pd.Series,
    replacement_exponents: np.ndarray,
    discount_rate: float,
) -> pd.Series:
    """Discount the annual system cost, taking the replacement from its own instant.

    Every other component is a flow spread over its year and keeps the
    year-end convention it has always had; the replacement is a single dated
    transaction, so it is pulled out, discounted from the swap, and added
    back.
    """
    exponents = np.asarray(replacement_exponents, dtype=float)
    replacement_discount = 1.0 / ((1.0 + discount_rate) ** exponents)
    outlay = pd.Series(np.asarray(replacement, dtype=float), index=annual.index)
    return (annual - outlay) * discount_factors + outlay * replacement_discount


def _replacement_npv(replacement: pd.Series, replacement_exponents: np.ndarray, discount_rate: float) -> float:
    """The replacement outlays discounted from their swap instants, as the system NPV counts them."""
    outlay = np.asarray(replacement, dtype=float)
    discount = 1.0 / ((1.0 + discount_rate) ** np.asarray(replacement_exponents, dtype=float))
    return float(np.sum(outlay * discount))


# Year-row money columns at year-1 prices (ADR 0003 E7). The projection
# escalates, times and discounts them; TOU valuation fills them from per-step
# energy and prices in the year loop.
YEAR_ROW_MONEY_COLUMNS = (
    "Import_Cost",
    "Export_Revenue",
    "Fixed_Charge",
    "Baseline_Import_Cost",
    "Baseline_Fixed_Charge",
)


def _replacement_outlays_t0(counts: Any, each: float) -> np.ndarray:
    """``each`` summed once per replacement, for each entry of ``counts``.

    Summed one event at a time from 0.0, as the physics layer once added its
    per-event cost, so the totals are the same floats (ADR 0003 E4).
    """
    totals = []
    for count in np.asarray(counts, dtype=float):
        if count < 0 or count != np.floor(count):
            raise ValueError(f"a year's Replacements must be a whole non-negative count, got {count!r}")
        total = 0.0
        for _ in range(int(count)):
            total += each
        totals.append(total)
    return np.asarray(totals, dtype=float)


def replacement_total_t0(replacement_cost: Any) -> float:
    """The replacements at t = 0 prices, summed year by year as the projection loop once did.

    An explicit ``+=`` loop, not ``sum()``: from Python 3.12 the built-in sums
    floats with compensation, so its total would depend on the Python version.
    """
    total = 0.0
    for value in replacement_cost:
        total += float(value)
    return total


def _replacement_cost_each(costs: Dict[str, float], n_events: float) -> float:
    if "replacement_cost_each" in costs:
        return float(costs["replacement_cost_each"])
    if n_events > 0:
        raise ValueError(
            "costs has no 'replacement_cost_each', so the simulated battery replacements cannot be priced; "
            "build costs with calculate_costs, or set it from replacement_event_cost"
        )
    return 0.0


def price_year_rows(yearly_summary_df: pd.DataFrame, costs: Dict[str, float]) -> pd.DataFrame:
    """Add the year-1-price money columns to flat-priced year rows.

    ``Import_Cost`` is ``Import_kWh`` times the import price, ``Export_Revenue``
    ``Export_kWh`` times the export price, ``Baseline_Import_Cost`` the load
    bought without a system, and ``Fixed_Charge`` the daily charge for the
    simulated duration, ``Simulated_Hours / 24`` days.
    ``Baseline_Fixed_Charge``, the fixed charge without a system, is
    ``Fixed_Charge`` unless a reference tariff set it. A row with
    ``Billed_Days`` (a [period] window) is billed on those civil days instead,
    and a row without either is billed as a 365-day year. ``Replacement_Cost`` is
    the year's ``Replacements`` at ``costs["replacement_cost_each"]``, t = 0
    prices. Columns already present (TOU valuation sets them) are kept.
    The operation order is the one the projection used before these columns
    existed, so flat results are the same floats.
    """
    priced = yearly_summary_df.copy()
    if "Replacement_Cost" not in priced.columns:
        if "Replacements" in priced.columns:
            counts = pd.to_numeric(priced["Replacements"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            # Right after the count, where the year rows carried the money
            # before ledger schema 3.0, so the column order is unchanged.
            position = list(priced.columns).index("Replacements") + 1
        else:
            counts, position = np.zeros(len(priced)), len(priced.columns)
        each = _replacement_cost_each(costs, float(np.sum(counts)))
        priced.insert(position, "Replacement_Cost", _replacement_outlays_t0(counts, each))
    days: Any
    if "Billed_Days" in priced.columns:
        # A [period] window bills its civil days, whatever DST does to its hours.
        days = priced["Billed_Days"]
    else:
        days = priced["Simulated_Hours"] / 24 if "Simulated_Hours" in priced.columns else 365
    computed = {
        "Import_Cost": lambda: priced["Import_kWh"] * costs["electricity_cost"],
        "Export_Revenue": lambda: priced["Export_kWh"] * costs["electricity_sold_cost"],
        "Fixed_Charge": lambda: days * costs["daily_power_cost"],
        "Baseline_Import_Cost": lambda: priced["Load_kWh"] * costs["electricity_cost"],
        # After Fixed_Charge: without a reference tariff the household pays
        # the same fixed charge with or without the system.
        "Baseline_Fixed_Charge": lambda: priced["Fixed_Charge"],
    }
    for column, value in computed.items():
        if column not in priced.columns:
            priced[column] = value()
    return priced


def _grid_shift_kwh(yearly: pd.DataFrame) -> Optional[np.ndarray]:
    """Per year, grid-origin battery delivery minus grid-charge import (ADR 0002 A10).

    None when the year rows carry no grid-charge columns, so emissions keep
    their old arithmetic exactly.
    """
    columns = ("Grid_Origin_Battery_AC_Load_kWh", "Grid_AC_To_Battery_kWh")
    if not all(column in yearly.columns for column in columns):
        return None
    return (yearly[columns[0]] - yearly[columns[1]]).to_numpy(dtype=float)


def resolve_escalation_rates(
    inflation_rate: float, import_price_escalation: Optional[float] = None, om_escalation: Optional[float] = None
) -> Dict[str, float]:
    """The escalation rate of each flow (ADR 0003 E2).

    Import energy and the fixed charge escalate at ``import_price_escalation``
    and O&M at ``om_escalation``; either left as None inherits
    ``inflation_rate``, today's single escalator.
    """
    return {
        "import_price_escalation": inflation_rate if import_price_escalation is None else import_price_escalation,
        "om_escalation": inflation_rate if om_escalation is None else om_escalation,
    }


def projection_rates_record(cfg: Dict[str, Any]) -> Dict[str, float]:
    """The rates a projection used, for provenance (ADR 0003 E1, E2).

    All are nominal annual rates. ``real_discount_rate`` is the discount rate
    net of general inflation, ``(1 + d) / (1 + inflation) - 1``; BREOS records
    the rates it used, not whether the user meant a nominal or real study.
    """
    rates = resolve_escalation_rates(
        cfg["inflation_rate"], cfg.get("import_price_escalation"), cfg.get("om_escalation")
    )
    return {
        "basis": "nominal",
        "inflation_rate": float(cfg["inflation_rate"]),
        "import_price_escalation": float(rates["import_price_escalation"]),
        "export_price_escalation": float(cfg["sell_price_inflation"]),
        "om_escalation": float(rates["om_escalation"]),
        "replacement_cost_learning": float(cfg.get("replacement_cost_learning", 0.0)),
        "discount_rate": float(cfg["discount_rate"]),
        "real_discount_rate": (1 + cfg["discount_rate"]) / (1 + cfg["inflation_rate"]) - 1,
    }


def _replacement_outlay(base: np.ndarray, exponents: np.ndarray, inflation_rate: float, learning: float) -> np.ndarray:
    """``C0 × (1 + inflation)^t × (1 − learning)^t`` at each swap instant ``t`` (ADR 0003 E2)."""
    outlay = base * (1 + inflation_rate) ** exponents
    # Without learning the result is the pre-E2 expression, the same floats.
    return outlay if learning == 0.0 else outlay * (1 - learning) ** exponents


@dataclass(frozen=True)
class TerminalHealthCredit:
    """An accounting sensitivity beside the unadjusted NPV (ADR 0003 E10)."""

    nominal: float
    npv: float
    adjusted_npv: float
    provenance: Dict[str, Any]


def terminal_health_credit(
    *,
    final_soh_fraction: Optional[float],
    threshold: float,
    replacement_cost_each: float,
    inflation_rate: float,
    replacement_cost_learning: float,
    discount_rate: float,
    horizon_years: int,
    npv_savings: float,
    allow_terminal_replacement: bool,
    replacement_min_remaining_years: float = 0.0,
    skipped_replacement_action: str = "keep",
) -> TerminalHealthCredit:
    """Credit the installed pack's capacity health above its physical EOL threshold.

    ``None`` health denotes no battery. The full pack is priced as a replacement
    booked at exactly t = T, with the outlay routine's t = 0 price convention,
    and discounted from that instant. No cashflow or replacement is changed.
    Capacity health omits resistance-related limits; this is not resale value.
    A pack kept or retired below its threshold (ADR 0003 E11, E12) has no
    health above it, so it earns no credit.
    """
    inputs = {
        "threshold": threshold,
        "replacement_cost_each": replacement_cost_each,
        "inflation_rate": inflation_rate,
        "replacement_cost_learning": replacement_cost_learning,
        "discount_rate": discount_rate,
        "horizon_years": horizon_years,
        "npv_savings": npv_savings,
    }
    if final_soh_fraction is not None:
        inputs["final_soh_fraction"] = final_soh_fraction
    for name, value in inputs.items():
        if not np.isfinite(value):
            raise ValueError(f"terminal health credit: {name} must be finite")
    if not 0 <= threshold < 1:
        raise ValueError("terminal health credit: threshold must be at least 0 and below 1")
    if replacement_cost_each < 0 or horizon_years < 1:
        raise ValueError("terminal health credit: price must be non-negative and horizon at least 1")
    if inflation_rate <= -1 or discount_rate <= -1 or not 0 <= replacement_cost_learning < 1:
        raise ValueError("terminal health credit: invalid escalation, learning or discount rate")
    fraction = (
        float(np.clip((final_soh_fraction - threshold) / (1 - threshold), 0, 1))
        if final_soh_fraction is not None
        else 0.0
    )
    full_price = float(
        _replacement_outlay(
            np.array([replacement_cost_each]),
            np.array([horizon_years], dtype=float),
            inflation_rate,
            replacement_cost_learning,
        )[0]
    )
    nominal = full_price * fraction
    # Reuse replacement discounting at t = T, also the year-T convention.
    npv = _replacement_npv(pd.Series([nominal]), np.array([horizon_years]), discount_rate)
    adjusted = npv_savings + npv
    if not all(np.isfinite(value) for value in (full_price, nominal, npv, adjusted)):
        raise ValueError("terminal health credit: valuation must be finite")
    return TerminalHealthCredit(
        nominal=nominal,
        npv=npv,
        adjusted_npv=adjusted,
        provenance={
            "basis": "battery_health_fraction",
            "formula_version": "1.0",
            "formula": "clip((h - h*) / (1 - h*), 0, 1)",
            "final_soh_fraction": final_soh_fraction,
            "threshold": threshold,
            "health_fraction": fraction,
            "price_basis": "full_replacement_pack_t0_prices",
            "replacement_cost_each_t0_prices": replacement_cost_each,
            "replacement_pack_price_year_t": full_price,
            "inflation_rate": inflation_rate,
            "replacement_cost_learning": replacement_cost_learning,
            "discount_rate": discount_rate,
            "horizon_years": horizon_years,
            "booking_time_years": horizon_years,
            "timing": "end_of_horizon_after_degradation_and_replacement",
            "replacement_policy": {
                "enable_replacement": True,
                "allow_terminal_replacement": allow_terminal_replacement,
                "replacement_min_remaining_years": replacement_min_remaining_years,
                "skipped_replacement_action": skipped_replacement_action,
            },
        },
    )


# Column order of a cost projection. Columns a stage does not produce (CO2
# without emissions) are left out; any other column a caller added to the
# cashflows follows, in its own order.
COST_PROJECTION_COLUMNS = (
    "Year",
    "Load_kWh",
    "Cost_No_Sys_Annual",
    "Cost_No_Sys_Cumulative",
    "Cost_No_Sys_Import",
    "Cost_No_Sys_Fixed_Charge",
    "Cost_No_Sys_Network_Charge",
    "Cost_No_Sys_Network_Credit",
    "PV_Production_kWh",
    "Export_kWh",
    "Degradation_Factor",
    "Cost_Import",
    "Revenue_Export",
    "Cost_Operation",
    "Cost_Daily",
    "Cost_Network_Charge",
    "Cost_Network_Credit",
    "Replacement_Time_Years",
    "Cost_Replacement",
    "Cost_System_Annual",
    "Cost_System_Cumulative",
    "Cost_No_Sys_Annual_NPV",
    "Cost_System_Annual_NPV",
    "Cost_No_Sys_Cumulative_NPV",
    "Cost_System_Cumulative_NPV",
    "Savings_Cumulative",
    "Savings_Cumulative_NPV",
)


def _validated_year_rows(yearly_summary_df: pd.DataFrame, num_years: int) -> pd.DataFrame:
    """The year rows ordered by their ``Year`` labels, which must be exactly 1 through ``num_years``."""
    expected_years = pd.Index(range(1, num_years + 1), name="Year")
    numeric_years = pd.to_numeric(yearly_summary_df["Year"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric_years).all() or not np.equal(numeric_years, np.floor(numeric_years)).all():
        raise ValueError("yearly_summary_df Year values must be finite integers")

    yearly_index = pd.Index(numeric_years.astype(int), name="Year")
    if yearly_index.has_duplicates:
        raise ValueError("yearly_summary_df Year values must be unique")
    if not yearly_index.difference(expected_years).empty or not expected_years.difference(yearly_index).empty:
        raise ValueError(f"yearly_summary_df Year values must cover exactly 1 through {num_years}")

    # Reorder the rows by their validated year labels so each financial row
    # stays attached to the simulation year it describes.
    rows = yearly_summary_df.drop(columns="Year")
    rows.index = yearly_index
    rows = rows.reindex(expected_years)
    rows.insert(0, "Year", expected_years)
    return rows.reset_index(drop=True)


def value_year_rows(
    year_rows: pd.DataFrame,
    costs: Dict[str, float],
    *,
    inflation_rate: float = DEFAULT_INFLATION_RATE,
    sell_price_inflation: float = 0.0,
    import_price_escalation: Optional[float] = None,
    om_escalation: Optional[float] = None,
    replacement_cost_learning: float = 0.0,
    baseline_import_price_escalation: Optional[float] = None,
) -> pd.DataFrame:
    """Turn priced year rows into each year's component cashflows (the valuation stage).

    ``year_rows`` are rows 1 through N, in order, carrying the year-1-price
    money of :func:`price_year_rows`. Energy, fixed-charge and O&M flows are
    escalated from year-1 prices; each replacement is priced at t = 0 and
    inflated to its swap instant, ``Replacement_Time_Years``. The no-system cost, ``Baseline_Import_Cost`` plus
    ``Baseline_Fixed_Charge``, escalates at ``baseline_import_price_escalation``,
    a reference tariff's; None uses the import escalation. Nothing is
    discounted here. ``attrs["total_replacement_cost"]`` is the replacements
    at t = 0 prices.

    Rows with an annual network credit (ADR 0002 A15) carry each credit and
    its cap basis, the eligible network charges, at year-1 prices:
    ``Network_Credit`` and ``Network_Charge`` for the system household,
    ``Baseline_Network_Credit`` and ``Baseline_Network_Charge`` for the
    no-system one. Each escalates with its household's import prices and is
    subtracted from its yearly cost, ``Cost_System_Annual`` or
    ``Cost_No_Sys_Annual``. Rows without them value as before, bit for bit.
    """
    years = year_rows["Year"]
    rates = resolve_escalation_rates(inflation_rate, import_price_escalation, om_escalation)
    inflation_factors = (1 + rates["import_price_escalation"]) ** (years - 1)
    baseline_factors = (
        inflation_factors
        if baseline_import_price_escalation is None
        else (1 + baseline_import_price_escalation) ** (years - 1)
    )
    om_factors = (1 + rates["om_escalation"]) ** (years - 1)
    sell_inflation_factors = (1 + sell_price_inflation) ** (years - 1)

    flows = pd.DataFrame({"Year": years})
    if "Load_kWh" in year_rows.columns:
        flows["Load_kWh"] = year_rows["Load_kWh"]
    # Summed before escalating, as the no-system cost always was, so a run
    # without a reference tariff gives the same floats.
    flows["Cost_No_Sys_Annual"] = (
        year_rows["Baseline_Import_Cost"] + year_rows["Baseline_Fixed_Charge"]
    ) * baseline_factors
    flows["Cost_No_Sys_Import"] = year_rows["Baseline_Import_Cost"] * baseline_factors
    flows["Cost_No_Sys_Fixed_Charge"] = year_rows["Baseline_Fixed_Charge"] * baseline_factors
    if "Baseline_Network_Credit" in year_rows.columns:
        flows["Cost_No_Sys_Network_Charge"] = year_rows["Baseline_Network_Charge"] * baseline_factors
        flows["Cost_No_Sys_Network_Credit"] = year_rows["Baseline_Network_Credit"] * baseline_factors
        flows["Cost_No_Sys_Annual"] = flows["Cost_No_Sys_Annual"] - flows["Cost_No_Sys_Network_Credit"]
    flows["PV_Production_kWh"] = year_rows["PV_Production_kWh"]
    flows["Export_kWh"] = year_rows["Export_kWh"]
    flows["Degradation_Factor"] = year_rows["PV_Degradation_Factor"]
    flows["Cost_Import"] = year_rows["Import_Cost"] * inflation_factors
    flows["Revenue_Export"] = year_rows["Export_Revenue"] * sell_inflation_factors
    flows["Cost_Operation"] = costs["annual_operation_cost"] * om_factors
    flows["Cost_Daily"] = year_rows["Fixed_Charge"] * inflation_factors
    if "Network_Credit" in year_rows.columns:
        flows["Cost_Network_Charge"] = year_rows["Network_Charge"] * inflation_factors
        flows["Cost_Network_Credit"] = year_rows["Network_Credit"] * inflation_factors

    # The replacement outlay is booked at the instant the pack is swapped:
    # inflated to it here and discounted from it later. The annual flows
    # keep the year-end convention.
    replacement_base = pd.to_numeric(year_rows["Replacement_Cost"], errors="coerce").fillna(0.0).to_numpy()
    replacement_time = replacement_booking_time(
        years.to_numpy(dtype=float),
        (
            pd.to_numeric(year_rows["Replacement_Year_Fraction"], errors="coerce").to_numpy()
            if "Replacement_Year_Fraction" in year_rows.columns
            else None
        ),
        replacement_base > 0.0,
    )
    flows["Replacement_Time_Years"] = replacement_time
    flows["Cost_Replacement"] = _replacement_outlay(
        replacement_base,
        _booking_exponents(replacement_time, years.to_numpy(dtype=float)),
        inflation_rate,
        replacement_cost_learning,
    )
    flows["Cost_System_Annual"] = (
        flows["Cost_Import"]
        - flows["Revenue_Export"]
        + flows["Cost_Operation"]
        + flows["Cost_Daily"]
        + flows["Cost_Replacement"]
    )
    if "Cost_Network_Credit" in flows.columns:
        flows["Cost_System_Annual"] = flows["Cost_System_Annual"] - flows["Cost_Network_Credit"]
    flows.attrs["total_replacement_cost"] = replacement_total_t0(replacement_base)
    return flows


def discount_cashflows(
    flows: pd.DataFrame,
    *,
    total_investment: float,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    currency: str = DEFAULT_CURRENCY,
) -> pd.DataFrame:
    """Accumulate and discount component cashflows, and compute the metrics (the discounting stage).

    ``flows`` is the output of :func:`value_year_rows`. Annual flows are
    discounted at year end, a replacement from its swap instant. Adds the
    cumulative, NPV and savings columns and sets ``attrs``: ``currency``,
    ``total_investment``, ``payback_year``, ``final_npv_savings``,
    ``replacement_cost_npv``, ``total_replacement_cost`` and
    ``lcoe_per_kwh``, the one LCOE every runner reports.
    """
    proj = flows.copy()
    years = proj["Year"]
    discount_factors = 1 / ((1 + discount_rate) ** years)
    replacement_exponents = _booking_exponents(
        proj["Replacement_Time_Years"].to_numpy(dtype=float), years.to_numpy(dtype=float)
    )

    proj["Cost_No_Sys_Cumulative"] = proj["Cost_No_Sys_Annual"].cumsum()
    proj["Cost_System_Cumulative"] = total_investment + proj["Cost_System_Annual"].cumsum()
    proj["Cost_No_Sys_Annual_NPV"] = proj["Cost_No_Sys_Annual"] * discount_factors
    proj["Cost_System_Annual_NPV"] = _discount_annual_with_replacement(
        proj["Cost_System_Annual"],
        proj["Cost_Replacement"],
        discount_factors,
        replacement_exponents,
        discount_rate,
    )
    proj["Cost_No_Sys_Cumulative_NPV"] = proj["Cost_No_Sys_Annual_NPV"].cumsum()
    proj["Cost_System_Cumulative_NPV"] = total_investment + proj["Cost_System_Annual_NPV"].cumsum()
    proj["Savings_Cumulative"] = proj["Cost_No_Sys_Cumulative"] - proj["Cost_System_Cumulative"]
    proj["Savings_Cumulative_NPV"] = proj["Cost_No_Sys_Cumulative_NPV"] - proj["Cost_System_Cumulative_NPV"]
    proj = proj[
        [column for column in COST_PROJECTION_COLUMNS if column in proj.columns]
        + [column for column in proj.columns if column not in COST_PROJECTION_COLUMNS]
    ]

    proj.attrs["currency"] = currency
    proj.attrs["total_investment"] = total_investment
    proj.attrs["payback_year"] = find_payback_year(proj)
    proj.attrs["final_npv_savings"] = proj["Savings_Cumulative_NPV"].iloc[-1]
    proj.attrs["replacement_cost_npv"] = _replacement_npv(
        proj["Cost_Replacement"], replacement_exponents, discount_rate
    )
    proj.attrs["total_replacement_cost"] = flows.attrs["total_replacement_cost"]
    proj.attrs["lcoe_per_kwh"] = calculate_lcoe_from_projection(
        proj, total_investment=total_investment, discount_rate=discount_rate
    )
    return proj


def add_co2_projection(proj: pd.DataFrame, year_rows: pd.DataFrame, emissions_params) -> None:
    """Add each year's avoided CO2 to a cost projection, in place (the emissions stage).

    Self-consumption is ``PV_Production_kWh - Export_kWh`` plus any grid
    shift through the battery. Sets
    ``attrs["lifetime_co2_avoided_total_kg"]`` and
    ``attrs["lifetime_co2_avoided_self_consumed_kg"]``.
    """
    from breos.emissions import calculate_co2_projection

    co2_proj = calculate_co2_projection(
        proj["PV_Production_kWh"].to_numpy(),
        proj["Export_kWh"].to_numpy(),
        emissions_params,
        _grid_shift_kwh(year_rows),
    )
    for col in (
        "CO2_Avoided_Total_kg",
        "CO2_Avoided_SelfConsumed_kg",
        "CO2_Avoided_Total_Cumulative_kg",
        "CO2_Avoided_SelfConsumed_Cumulative_kg",
        "CO2_Avoided_CI_gCO2_kWh",
        "CO2_Avoided_CI_Type",
        "Average_Grid_CI_gCO2_kWh",
        "Marginal_Grid_CI_gCO2_kWh",
        "CO2_Avoided_Export_kg",
        "CO2_Avoided_Export_Cumulative_kg",
    ):
        proj[col] = co2_proj[col].values
    proj.attrs["lifetime_co2_avoided_total_kg"] = float(proj["CO2_Avoided_Total_Cumulative_kg"].iloc[-1])
    proj.attrs["lifetime_co2_avoided_self_consumed_kg"] = float(proj["CO2_Avoided_SelfConsumed_Cumulative_kg"].iloc[-1])
    proj.attrs["lifetime_co2_avoided_export_kg"] = float(proj["CO2_Avoided_Export_Cumulative_kg"].iloc[-1])


def _recorded_currency(frame: pd.DataFrame) -> Optional[str]:
    """The currency a frame records: ``attrs["currency"]``, else its ``currency`` column, else None.

    BREOS writes a ``currency`` column into the CSVs it saves, since a CSV
    keeps no attrs; a column that names more than one currency raises.
    """
    if frame.attrs.get("currency") is not None:
        return str(frame.attrs["currency"])
    if "currency" not in frame.columns:
        return None
    codes = sorted({str(code) for code in frame["currency"].dropna()})
    if len(codes) > 1:
        raise ValueError(f"The input's currency column names several currencies ({', '.join(codes)}); keep them apart")
    return codes[0] if codes else None


def write_cost_projection(proj: pd.DataFrame, results_directory: str, scenario_name: str = "") -> str:
    """Write a cost projection to ``cost_projection[_<scenario>].csv`` in ``results_directory``; return the path.

    The ``currency`` column names the currency the frame records
    (``attrs["currency"]``, else its own ``currency`` column); a frame that
    records none is written without one.
    """
    import os

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""
    path = f"{results_directory}/cost_projection{suffix}.csv"
    # A CSV keeps no attrs, so each row names the currency of its money.
    currency = _recorded_currency(proj)
    (proj if currency is None else proj.assign(currency=currency)).to_csv(path, index=False)
    return path


def cost_analysis_projection(
    yearly_summary_df: pd.DataFrame,
    costs: Dict[str, float],
    num_years: int = 20,
    inflation_rate: float = DEFAULT_INFLATION_RATE,
    sell_price_inflation: float = 0.0,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    *,
    results_directory: Optional[str] = None,
    scenario_name: str = "",
    emissions_params=None,
    currency: str = DEFAULT_CURRENCY,
    import_price_escalation: Optional[float] = None,
    om_escalation: Optional[float] = None,
    replacement_cost_learning: float = 0.0,
    baseline_import_price_escalation: Optional[float] = None,
) -> pd.DataFrame:
    """
    Perform multi-year cost projection analysis from simulated year rows.

    Includes inflation and the discount rate; PV degradation is already in
    the simulated rows. The work runs in four stages, each also public:
    :func:`price_year_rows` and :func:`value_year_rows` turn energy into
    component cashflows, :func:`discount_cashflows` discounts them and
    computes the metrics, :func:`add_co2_projection` adds avoided emissions,
    and :func:`write_cost_projection` writes the file.

    Args:
        yearly_summary_df: One row per simulated project year, with ``Year``
            1 through ``num_years``, ``PV_Production_kWh``, ``Import_kWh``,
            ``Export_kWh``, etc., as the App, Monte Carlo and the optimizer
            produce them. Each year's ``Replacements`` are priced at
            ``costs["replacement_cost_each"]`` unless the rows already carry
            ``Replacement_Cost``.
        costs: Dictionary with cost parameters (from calculate_costs())
        num_years: Number of years to project
        inflation_rate: General annual inflation. Import energy, the fixed
            charge and O&M escalate at it unless their own rate is given,
            and replacement prices always inflate at it.
        sell_price_inflation: Annual escalation of the export price
        discount_rate: Nominal discount rate for NPV calculations
        results_directory: Optional directory to save results
        scenario_name: Optional name suffix for saved files
        emissions_params: Optional :class:`~breos.emissions.EmissionsParams`;
            when given, the avoided-emissions columns are added.
        currency: The currency every money input is in. BREOS does not
            convert; it is recorded as ``attrs["currency"]`` for labels.
        baseline_import_price_escalation: Annual escalation of the
            no-system cost, a reference tariff's; None uses the import
            escalation.

    Returns:
        DataFrame with yearly cost projections

    Raises:
        ValueError: If ``yearly_summary_df`` is missing, empty or has no
            ``Year`` column, or its ``Year`` labels are not exactly 1 through
            ``num_years``.
    """
    if not isinstance(yearly_summary_df, pd.DataFrame) or yearly_summary_df.empty:
        raise ValueError("cost_analysis_projection needs yearly_summary_df, one row per simulated project year")
    if "Year" not in yearly_summary_df.columns:
        raise ValueError(
            "yearly_summary_df has no Year column: cost_analysis_projection takes one row per project year, "
            "not a per-step results frame"
        )
    year_rows = _validated_year_rows(yearly_summary_df, num_years)
    year_rows = price_year_rows(year_rows, costs)

    flows = value_year_rows(
        year_rows,
        costs,
        inflation_rate=inflation_rate,
        sell_price_inflation=sell_price_inflation,
        import_price_escalation=import_price_escalation,
        om_escalation=om_escalation,
        replacement_cost_learning=replacement_cost_learning,
        baseline_import_price_escalation=baseline_import_price_escalation,
    )
    proj = discount_cashflows(
        flows, total_investment=costs["total_initial_cost"], discount_rate=discount_rate, currency=currency
    )
    if emissions_params is not None:
        add_co2_projection(proj, year_rows, emissions_params)
    if results_directory:
        write_cost_projection(proj, results_directory, scenario_name)
    return proj


def _initial_investment(cost_projection: pd.DataFrame) -> Optional[float]:
    """Return the year-0 investment a projection starts from, if it records one.

    ``cost_analysis_projection`` stores it in ``attrs["total_investment"]``.
    A projection read back from CSV has no attrs, so the investment is
    recovered from the first row of the cumulative system cost, which is the
    investment plus that year's cost.
    """
    investment = cost_projection.attrs.get("total_investment")
    for cumulative, annual in (
        ("Cost_System_Cumulative_NPV", "Cost_System_Annual_NPV"),
        ("Cost_System_Cumulative", "Cost_System_Annual"),
    ):
        if investment is None and {cumulative, annual} <= set(cost_projection.columns):
            investment = cost_projection[cumulative].iloc[0] - cost_projection[annual].iloc[0]
    if investment is None:
        return None
    return float(investment)


def _payback_points(
    cost_projection: pd.DataFrame, initial_investment: Optional[float]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the years and cumulative discounted savings payback is read from.

    When the projection starts after year 0 and the investment is known, a
    year-0 point with savings of minus the investment is prepended.

    Raises:
        ValueError: If a year, a savings value or the investment is NaN or
            infinite. Payback is undefined on such a series, and reading it
            anyway would report NaN or a false crossing.
    """
    if (
        "Savings_Cumulative_NPV" not in cost_projection.columns
        or "Year" not in cost_projection.columns
        or cost_projection.empty
    ):
        return None

    years = cost_projection["Year"].to_numpy(dtype=float)
    savings = cost_projection["Savings_Cumulative_NPV"].to_numpy(dtype=float)
    if initial_investment is None:
        initial_investment = _initial_investment(cost_projection)
    if initial_investment is not None and years[0] > 0.0:
        years = np.concatenate(([0.0], years))
        savings = np.concatenate(([-float(initial_investment)], savings))
    if not (np.isfinite(years).all() and np.isfinite(savings).all()):
        raise ValueError(
            "Payback needs finite years, Savings_Cumulative_NPV and initial investment; "
            "the projection contains NaN or infinite values"
        )
    return years, savings


def _sustained_payback_index(savings: np.ndarray) -> Optional[int]:
    """Index of the first point from which savings stay >= 0 to the horizon.

    ``savings`` must be finite (see :func:`_payback_points`). Returns None
    when the last point is negative, that is when the savings never recover
    for good.
    """
    negative = np.flatnonzero(savings < 0.0)
    if negative.size == 0:
        return 0
    last_negative = int(negative[-1])
    if last_negative == len(savings) - 1:
        return None
    return last_negative + 1


def find_payback_year(cost_projection: pd.DataFrame, initial_investment: Optional[float] = None) -> Optional[int]:
    """Return the sustained discounted payback year, as a whole year.

    The integer counterpart of :func:`find_payback_year_interpolated`, under the same
    rule: the first year from which cumulative discounted savings
    (``Savings_Cumulative_NPV``) are zero or above and stay so to the end of
    the simulated period. The series starts at year 0 with minus the
    investment (see :func:`find_payback_year_interpolated`), so it is the whole year
    in which the savings cross zero for the last time, and a system that pays
    back within its first year reports 1. If a battery replacement turns the
    savings negative again, payback is the later recovery.

    Args:
        cost_projection: DataFrame from :func:`cost_analysis_projection`.
        initial_investment: Year-0 investment. Defaults to
            ``cost_projection.attrs["total_investment"]``, or to the value
            recovered from the system cost columns.

    Returns:
        The payback year, or None when the savings do not stay nonnegative
        through the last projected year, or the projection has no savings
        column.

    Raises:
        ValueError: If the years, the savings or the investment contain NaN
            or infinite values.
    """
    points = _payback_points(cost_projection, initial_investment)
    if points is None:
        return None
    years, savings = points
    index = _sustained_payback_index(savings)
    return None if index is None else int(years[index])


def find_payback_year_interpolated(
    cost_projection: pd.DataFrame, initial_investment: Optional[float] = None
) -> Optional[float]:
    """Return the sustained discounted payback, interpolated between years.

    Sustained discounted payback within the simulated period: the earliest
    time at which cumulative discounted savings (``Savings_Cumulative_NPV``)
    reach zero or above and remain nonnegative through the end of the
    projection. The series starts at year 0 with minus the initial
    investment, and the crossing is interpolated linearly between the annual
    points, including between years 0 and 1. If a battery replacement turns
    the savings negative again, payback is the later recovery.

    The result is fractional, not exact: a straight line between year-end
    points only approximates when within the year the savings cross zero.

    The year-0 point is taken from ``initial_investment`` when given, else
    from ``cost_projection.attrs["total_investment"]``, else from the first
    row of ``Cost_System_Cumulative_NPV`` minus ``Cost_System_Annual_NPV``
    (or their nominal counterparts), as in a projection read back from CSV.
    A projection that already has a year-0 row, or records no investment, is
    read as it is; its first row then pays back at that row's year if the
    savings stay nonnegative from it.

    Returns:
        The payback in years, or None when the savings do not stay
        nonnegative through the last projected year, or the projection has no
        savings column.

    Raises:
        ValueError: If the years, the savings or the investment contain NaN
            or infinite values.
    """
    points = _payback_points(cost_projection, initial_investment)
    if points is None:
        return None
    years, savings = points
    index = _sustained_payback_index(savings)
    if index is None:
        return None
    if index == 0:
        return float(years[0])
    # savings[index - 1] < 0 <= savings[index], so the fraction is in (0, 1].
    before, after = savings[index - 1], savings[index]
    fraction = -before / (after - before)
    return float(years[index - 1] + fraction * (years[index] - years[index - 1]))


def calculate_lcoe_from_projection(
    cost_projection: pd.DataFrame,
    total_investment: Optional[float] = None,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> float:
    """Calculate LCOE from a simulated multi-year projection.

    This variant is intended for simulation outputs that already contain
    year-by-year PV production and replacement costs. It uses system CAPEX,
    operation costs, and replacement costs as the cost basis; grid import
    charges, fixed grid charges, and export revenue are excluded because those
    are tariff outcomes rather than generation costs.

    Args:
        cost_projection: DataFrame from :func:`cost_analysis_projection`.
        total_investment: System CAPEX. If omitted, uses
            ``cost_projection.attrs["total_investment"]``, which
            :func:`cost_analysis_projection` records.
        discount_rate: Discount rate used for production and annual costs.

    Returns:
        LCOE per kWh, in the currency of the inputs.
    """
    if cost_projection.empty:
        return float("inf")
    if "PV_Production_kWh" not in cost_projection.columns:
        raise ValueError("cost_projection must include 'PV_Production_kWh'")

    if total_investment is None:
        total_investment = cost_projection.attrs.get("total_investment")
    if total_investment is None:
        raise ValueError("total_investment is required when cost_projection.attrs does not record it")

    years = (
        pd.to_numeric(cost_projection["Year"], errors="coerce")
        if "Year" in cost_projection.columns
        else pd.Series(range(1, len(cost_projection) + 1), index=cost_projection.index)
    )
    discount_factors = 1 / ((1 + discount_rate) ** years)

    production = pd.to_numeric(cost_projection["PV_Production_kWh"], errors="coerce").fillna(0.0)
    operation = (
        pd.to_numeric(cost_projection["Cost_Operation"], errors="coerce").fillna(0.0)
        if "Cost_Operation" in cost_projection.columns
        else pd.Series(0.0, index=cost_projection.index)
    )
    replacement = (
        pd.to_numeric(cost_projection["Cost_Replacement"], errors="coerce").fillna(0.0)
        if "Cost_Replacement" in cost_projection.columns
        else pd.Series(0.0, index=cost_projection.index)
    )

    # The replacement is discounted from the swap instant when the projection
    # carries it, matching how the NPV above books the same outlay.
    if "Replacement_Time_Years" in cost_projection.columns:
        replacement_time = pd.to_numeric(cost_projection["Replacement_Time_Years"], errors="coerce")
        replacement_discount_factors = 1 / ((1 + discount_rate) ** replacement_time.fillna(years))
    else:
        replacement_discount_factors = discount_factors

    npv_costs = float(total_investment) + float(
        ((operation * discount_factors) + (replacement * replacement_discount_factors)).sum()
    )
    npv_production = float((production * discount_factors).sum())

    return npv_costs / npv_production if npv_production > 0 else float("inf")
