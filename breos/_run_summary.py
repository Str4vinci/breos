"""Plain-text summary of a run result, printed by ``breos run`` on a terminal.

The summary reads only the result dict that ``App.result()`` returns, so it
shows the same numbers as the JSON. A line whose value is missing (no battery,
a period run without lifetime economics) is left out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_LABEL_WIDTH = 26


def _number(value: float, decimals: int = 0) -> str:
    """Format a number without commas: 5000, 12 345, -4675, 0.102."""
    text = f"{value:.{decimals}f}"
    if text.startswith("-") and float(text) == 0:
        text = text[1:]
    sign, digits = ("-", text[1:]) if text.startswith("-") else ("", text)
    whole, dot, fraction = digits.partition(".")
    if len(whole) >= 5:
        groups = [whole[max(i - 3, 0) : i] for i in range(len(whole), 0, -3)]
        whole = " ".join(reversed(groups))
    return f"{sign}{whole}{dot}{fraction}"


def _value(result: dict[str, Any], key: str) -> float | None:
    """A number from the result, or None when it is missing or not a number."""
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric_row(
    result: dict[str, Any], label: str, key: str, unit: str, decimals: int = 0, note: str = ""
) -> list[str]:
    """The row for one result key, or no row when the key has no value."""
    value = _value(result, key)
    return [] if value is None else [_row(label, _number(value, decimals), unit, note)]


def _percent_note(result: dict[str, Any], key: str, words: str) -> str:
    value = _value(result, key)
    return "" if value is None else f"{_number(value, 1)} % {words}"


def _row(label: str, value: str, unit: str = "", note: str = "") -> str:
    """One line: label, the number right-aligned, then its unit and a note."""
    line = f"  {label:<{_LABEL_WIDTH}}{value:>8} {unit:<8}"
    return f"{line}  {note}".rstrip() if note else line.rstrip()


def _text_row(label: str, text: str) -> str:
    """A line whose value is words, not a number: it starts at the number column."""
    return f"  {label:<{_LABEL_WIDTH}}{text}"


def _header(result: dict[str, Any], config_name: str | None) -> str:
    provenance = result.get("provenance") or {}
    resolved = provenance.get("resolved_config") or {}
    location = resolved.get("location") or {}
    parts = [f"BREOS {provenance.get('breos_version', '')}".rstrip()]
    if config_name:
        parts.append(config_name)
    name = location.get("preset")
    lat, lon = location.get("latitude"), location.get("longitude")
    if lat is not None and lon is not None:
        coords = f"({lat:.2f}, {lon:.2f})"
        parts.append(f"{str(name).title()} {coords}" if name else coords)
    source = (provenance.get("weather") or {}).get("source")
    if source:
        parts.append(str(source).replace("_", " "))
    return ", ".join(parts)


def _system(result: dict[str, Any]) -> str:
    provenance = result.get("provenance") or {}
    resolved = provenance.get("resolved_config") or {}
    inverter = (result.get("pv_loss_waterfall") or {}).get("inverter") or {}
    parts = []
    pv_kwp = _value(result, "pv_kwp")
    if pv_kwp is not None:
        modules = result.get("n_modules")
        parts.append(f"{_number(pv_kwp, 1)} kWp PV" + (f" ({modules} modules)" if modules else ""))
    ac_kw = inverter.get("ac_capacity_kw")
    if ac_kw:
        parts.append(f"{_number(ac_kw, 1)} kW inverter")
    battery_kwh = _value(result, "battery_kwh")
    if battery_kwh:
        coupling = str(resolved.get("battery_coupling") or "").upper()
        parts.append(
            f"{_number(battery_kwh, 1).removesuffix('.0')} kWh battery" + (f" ({coupling})" if coupling else "")
        )
    elif battery_kwh is not None:
        parts.append("no battery")
    return "System: " + ", ".join(parts) if parts else ""


def _energy_block(result: dict[str, Any]) -> list[str]:
    vehicle_year = _vehicle_study_year1(result)
    lines = _metric_row(result, "PV generation (DC)", "pv_dc_generation_kwh", "kWh")
    lines += _metric_row(result, "House load", "consumption_kwh", "kWh")
    if vehicle_year is not None:
        pv_charge = _value(vehicle_year, "pv_ac_charge_kwh") or 0.0
        charge = pv_charge + (_value(vehicle_year, "grid_ac_charge_kwh") or 0.0)
        lines.append(_row("Vehicle charging at home", _number(charge), "kWh", f"{_number(pv_charge)} kWh from PV"))
    self_note = _percent_note(result, "self_consumption_pct", "of PV used on site")
    lines += _metric_row(result, "Self-consumed PV", "self_consumption_kwh", "kWh", note=self_note)
    # Grid independence compares the import with the house load alone, so it
    # says nothing useful once the import also charges a vehicle.
    if vehicle_year is None:
        import_note = _percent_note(result, "grid_independence_pct", "grid independence")
    else:
        import_note = "house and vehicle"
    lines += _metric_row(result, "Grid import", "grid_import_kwh", "kWh", note=import_note)
    lines += _metric_row(result, "Grid export", "grid_export_kwh", "kWh")
    period = result.get("period") or {}
    if period:
        title = f"Energy, {period.get('start')} to {period.get('end')} (end exclusive, {period.get('days')} days)"
    else:
        title = "Year 1 energy"
    return [title, *lines] if lines else []


def _vehicle_study_year1(result: dict[str, Any]) -> dict[str, Any] | None:
    """Year 1 of the studied household (R2) in a vehicle run, else None."""
    households = (result.get("vehicle") or {}).get("households") or {}
    yearly = (households.get("R2") or {}).get("yearly") or []
    return yearly[0] if yearly else None


def _battery_block(result: dict[str, Any]) -> list[str]:
    degradation = result.get("degradation") or {}
    final_soh = _value(degradation, "final_soh_pct")
    if not _value(result, "battery_kwh") or final_soh is None:
        return []
    years = len(result.get("yearly") or [])
    replaced = [event for event in degradation.get("replacement_events") or [] if event.get("count")]
    note = ""
    if replaced:
        note = "replaced in year " + ", ".join(str(event.get("year")) for event in replaced)
    if result.get("period"):
        title = "Battery at the end of the period"
    else:
        title = f"Battery over {years} years" if years else "Battery"
    return [title, _row("Health at the end", _number(final_soh, 1), "%", note)]


def _economics_block(result: dict[str, Any]) -> list[str]:
    if _value(result, "npv_savings") is None:
        if (result.get("period") or {}).get("lifetime_economics") == "skipped":
            return ["Economics: skipped, a period shorter than a year has no project lifetime."]
        return []
    provenance = result.get("provenance") or {}
    currency = provenance.get("currency") or ""
    economics = provenance.get("economics") or {}
    years = len(result.get("yearly") or [])
    title = "Economics"
    details = [f"{years} years" if years else "", currency]
    discount_rate = _value(economics, "discount_rate")
    if discount_rate is not None:
        details.append(f"{_number(discount_rate * 100, 1).removesuffix('.0')} % discount rate")
    title += ", " + ", ".join(d for d in details if d)
    lines = [title]
    lines += _metric_row(result, "Investment", "total_investment", currency)
    lines += _metric_row(result, "Net present value", "npv_savings", currency)
    payback = result.get("payback_year")
    if payback is not None:
        lines.append(_row("Discounted payback", str(payback), "years"))
    else:
        lines.append(_text_row("Discounted payback", f"not within {years} years" if years else "not reached"))
    lines += _metric_row(result, "LCOE", "lcoe_per_kwh", f"{currency}/kWh", decimals=3)
    vehicle = result.get("vehicle") or {}
    note = "" if result.get("household_adjusted_comparable", True) else "not comparable"
    lines += _metric_row(result, "NPV with vehicle costs", "npv_savings_household_adjusted", currency, note=note)
    label = f"{str(vehicle.get('mode', '')).upper()} benefit vs {str(vehicle.get('reference_mode', '')).upper()}"
    note = "" if result.get("vehicle_benefit_comparable", True) else "not comparable"
    lines += _metric_row(result, label, "vehicle_benefit_npv", currency, note=note)
    return lines


def format_run_summary(result: dict[str, Any], config_name: str | None = None, output: Path | None = None) -> str:
    """Return the terminal summary of ``result``."""
    system = _system(result)
    blocks = [
        [_header(result, config_name)],
        [system] if system else [],
        _energy_block(result),
        _battery_block(result),
        _economics_block(result),
    ]
    if output is not None:
        footer = f"Full results written to {output} (JSON, result schema {result.get('result_schema_version')})."
    else:
        footer = "Full results: add --output result.json, or pipe the command to get JSON."
    blocks.append([footer])
    return "\n\n".join("\n".join(block) for block in blocks if block)
