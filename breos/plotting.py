"""
Plotting and visualization module.

This module provides visualization functions for:
- Cost projections
- Energy balance results
- Monthly/yearly/weekly analysis
- Battery degradation

Every plot function returns what it draws, and one rule decides whether it
saves and closes it:

- With an output directory (``results_directory``, or ``output_path`` for
  :func:`plot_pv_loss_waterfall`), the function saves the figure there,
  closes it, and returns it. A closed figure can still be saved again with
  ``fig.savefig``, but ``plt.show()`` and tools that collect the open
  figures no longer see it.
- With ``results_directory=None``, the default, nothing is written, and the
  figure is returned open, for ``plt.show()``, a notebook or a
  sphinx-gallery example. Close it with ``plt.close(fig)`` when done.

A function that draws one figure returns a matplotlib ``Figure``, or None
when it has nothing to draw. A function that draws several returns a dict of
them, keyed by the file name each is saved as, without the scenario suffix
and the ``.png`` extension, for example ``"breakeven_cumulative"``.
"""

import math
import os
import warnings
from typing import Any, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from breos.constants import DEFAULT_EOL_PERCENTAGE
from breos.economics import _from_year_zero, _initial_investment, _recorded_currency, find_payback_year_interpolated
from breos.tariffs import DEFAULT_CURRENCY
from breos.utils import find_irradiance_column, local_datetime_index
from breos.weather import _weather_years, extract_ambient_temperature

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# matplotlib is the optional ``plots`` extra. Importing this module validates it.
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath
    from matplotlib.ticker import MaxNLocator
except ImportError as exc:
    raise ModuleNotFoundError('breos.plotting needs matplotlib. Install it with: pip install "breos[plots]"') from exc


def _finish(
    fig: Figure,
    results_directory: Optional[str],
    filename: str,
    dpi: int = 300,
    bbox_inches: Optional[str] = None,
) -> Figure:
    """Save ``fig`` as ``filename`` in ``results_directory`` and close it; with None, leave it open.

    Returns ``fig`` either way. See the module docstring for the rule.
    """
    if results_directory is None:
        return fig
    os.makedirs(results_directory, exist_ok=True)
    fig.savefig(os.path.join(results_directory, filename), dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    return fig


# The four energy-flow columns of the yearly and monthly plots, in their
# drawing order, with their legend labels.
_ENERGY_LABELS = {
    "PV_Production": "PV Production",
    "Houseload": "Load",
    "Import_From_Grid": "Grid Import",
    "PV_AC_Export": "Grid Export",
}


def _scenario_suffix(scenario_name: str) -> str:
    """The file-name suffix of a scenario: ``_<name>``, or nothing without one."""
    return f"_{scenario_name}" if scenario_name else ""


def _currency(frame: pd.DataFrame) -> str:
    """The currency a frame's money is in, for labels: the recorded one, else the default."""
    return _recorded_currency(frame) or DEFAULT_CURRENCY


def _label_currency(frames: Sequence[pd.DataFrame], currency: Optional[str]) -> Optional[str]:
    """The currency of the money in ``frames``, or None if nothing records it.

    A frame records its currency in ``attrs["currency"]`` or, read back from
    a CSV that BREOS wrote, in a ``currency`` column; another table records
    none. ``currency`` names it for those. Two different recorded
    currencies, or a recorded one that ``currency`` contradicts, raise
    ``ValueError``.
    """
    known = sorted({code for code in map(_recorded_currency, frames) if code is not None})
    if len(known) > 1:
        raise ValueError(f"The inputs are in different currencies ({', '.join(known)}); plot them apart")
    if currency is not None and known and known[0] != currency:
        raise ValueError(f"currency={currency!r}, but the input records its currency as {known[0]!r}")
    if currency is not None:
        return currency
    return known[0] if known else None


def _result_instants(results_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the absolute time of each results row.

    Read from a ``Datetime`` column, else the ``Datetime_UTC`` column that
    :func:`breos.io.load_results` adds to a DST-zone CSV, else the index. A
    CSV of a run in a DST zone mixes UTC offsets; those rows are read as UTC
    instants, which step evenly where their wall-clock labels do not.
    """
    if "Datetime" in results_df.columns:
        values = results_df["Datetime"]
    elif "Datetime_UTC" in results_df.columns:
        values = results_df["Datetime_UTC"]
    else:
        values = results_df.index
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.DatetimeIndex(values)
    try:
        return pd.DatetimeIndex(pd.to_datetime(values))
    except ValueError:
        return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


_Frame = TypeVar("_Frame", pd.DataFrame, pd.Series)


def _on_wall_clock(frame: _Frame) -> _Frame:
    """``frame`` with its index on the wall clock, for calendar grouping.

    Resampling a zone-aware index normalises its first step to local
    midnight, which does not exist where the clocks move forward at
    midnight (Chile, for one).
    """
    index = frame.index
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return frame.set_axis(index.tz_localize(None))
    return frame


def _local_time_indexed(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``results_df`` indexed by its ``Datetime`` column on the local clock, if it has one."""
    df = results_df.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = local_datetime_index(df["Datetime"])
        df = df.set_index("Datetime")
    return df


def _power_frame_to_energy_kwh(frame: pd.DataFrame, instants: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """Convert regularly sampled power columns in watts to interval energy.

    ``instants`` gives each row's absolute time when the frame's index is a
    wall-clock calendar, which has a gap and a repeat at the DST transitions.
    """
    index = frame.index if instants is None else instants
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Power-to-energy plotting requires a DatetimeIndex")
    if len(index) < 2:
        raise ValueError("Power-to-energy plotting requires at least two timestamps")
    intervals = np.diff(index.asi8)
    if np.any(intervals <= 0) or not np.all(intervals == intervals[0]):
        raise ValueError("Power-to-energy plotting requires a regular increasing time index")
    hours_per_step = (index[1] - index[0]).total_seconds() / 3600.0
    return frame * (hours_per_step / 1000.0)


def set_presentation_mode(enabled: bool = True, scale: float = 1.5):
    """
    Enable presentation mode with larger fonts for all plots.

    Args:
        enabled: True to enable, False to reset to defaults
        scale: Font size multiplier (default 1.5x)

    Usage:
        from breos.plotting import set_presentation_mode
        set_presentation_mode(True)  # Enable before generating plots
        set_presentation_mode(False) # Reset to defaults
    """
    if enabled:
        plt.rcParams.update(
            {
                "font.size": 14 * scale,
                "axes.titlesize": 16 * scale,
                "axes.labelsize": 14 * scale,
                "xtick.labelsize": 12 * scale,
                "ytick.labelsize": 12 * scale,
                "legend.fontsize": 12 * scale,
                "figure.titlesize": 18 * scale,
            }
        )
    else:
        plt.rcdefaults()


def _grouped(value: float, decimals: int = 0) -> str:
    """``value`` with ``decimals`` places, digits grouped by a space from five digits up: 5000, 12 345."""
    text = f"{value:,.{decimals}f}"
    whole = text.split(".")[0].lstrip("-")
    return text.replace(",", " " if len(whole) > 5 else "")


def _is_monofacial_rear_stage(stage: Mapping[str, Any], waterfall: Mapping[str, Any]) -> bool:
    """True for the bifacial rear-gain stage of a system without bifacial modelling."""
    if stage.get("key") != "bifacial_rear_gain":
        return False
    bifacial = waterfall.get("bifacial")
    if isinstance(bifacial, Mapping) and "enabled" in bifacial:
        return not bifacial["enabled"]
    return float(stage.get("delta_kwh", 0.0)) == 0.0


# Colours of the PV energy-flow diagram. Text uses the two inks only.
_FLOW_INK, _FLOW_INK2, _FLOW_MUTED = "#0b0b0b", "#52514e", "#898781"
_FLOW_GRID, _FLOW_BASE, _FLOW_SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
_FLOW_BAND, _FLOW_TOTAL, _FLOW_GAIN, _FLOW_LOSS = "#cde2fb", "#184f95", "#2a78d6", "#e34948"
# Where the usable AC energy goes: (ac_delivery key, label, colour).
_FLOW_END_USES = (
    ("direct_pv_to_load_kwh", "Direct to load", "#1baf7a"),
    ("pv_origin_battery_to_load_kwh", "Via battery to load", "#eda100"),
    ("pv_origin_vehicle_to_load_kwh", "Via vehicle to load", "#e9a3c6"),
    ("export_kwh", "Grid export", "#6250d6"),
)
_FLOW_VEHICLE_CHARGING = ("Vehicle charging, net of energy returned", "#c2387f")
_FLOW_STAGE_LABELS = {
    "horizontal_reference_dc": "Horizontal irradiance reference",
    "transposition": "Tilt and orientation (transposition)",
    "iam": "Incidence angle (front side)",
    "pvwatts_static": "Static losses",
}
_FLOW_STATIC_LABELS = {"lid": "Light-induced degradation"}
# Tiny static components share the residual row; their energy is never discarded.
_FLOW_MIN_KWH = 0.05


def _flow_number(value: float, decimals: int = 0) -> str:
    """``value`` as 5000, 12 345 (narrow space) or −150: never a comma, a real minus sign."""
    text = f"{abs(value):,.{decimals}f}"
    text = text.replace(",", " " if len(text.split(".")[0]) > 5 else "")
    return ("−" if round(value, decimals) < 0 else "") + text


def _flow_kwh(value: Any, name: str, *, signed: bool = False) -> float:
    """Validate energies before allocating a figure, with an actionable field name."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number) or (number < 0 and not signed):
        raise ValueError(f"{name} must be finite" + ("" if signed else " and nonnegative"))
    return number


def _flow_rows(waterfall: Mapping[str, Any], stages: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str, float]]:
    """Rows as (kind, label, kWh); every step participates in the energy balance."""
    energies = [_flow_kwh(stage.get("energy_kwh"), f"stage {i} energy_kwh") for i, stage in enumerate(stages)]
    first = stages[0]
    rows = [("total", _FLOW_STAGE_LABELS.get(first.get("key", ""), first.get("label", "PV reference")), energies[0])]
    components = (waterfall.get("pvwatts") or {}).get("components_kwh") or {}
    for stage, previous, energy in zip(stages[1:], energies[:-1], energies[1:], strict=True):
        key = stage.get("key", "")
        # Stage totals define the geometry; validate reported deltas as well.
        if "delta_kwh" in stage:
            _flow_kwh(stage["delta_kwh"], f"stage {key!r} delta_kwh", signed=True)
        delta = energy - previous
        if key == "pvwatts_static" and components:
            drawn = 0.0
            for name, value in components.items():
                loss = _flow_kwh(value, f"pvwatts component {name!r}", signed=True)
                if abs(loss) >= _FLOW_MIN_KWH:
                    label = _FLOW_STATIC_LABELS.get(name, name.replace("_", " ").capitalize())
                    rows.append(("step", label, -loss))
                    drawn += loss
            rest = delta + drawn
            if abs(rest) > 1e-9:
                rows.append(("step", "Other static losses", rest))
        elif delta != 0 or key == "bifacial_rear_gain":
            rows.append(("step", _FLOW_STAGE_LABELS.get(key, stage.get("label", key)), delta))
    pv_dc = energies[-1]
    rows.append(("total", "PV DC output", pv_dc))

    balance = waterfall.get("energy_balance") or {}
    ac = balance.get("ac_delivery") or {}
    if not ac:
        return rows
    _flow_end_uses(ac)  # Validate every delivery field before drawing.
    vehicle = _flow_vehicle_charging(ac)
    usable = _flow_kwh(
        ac.get("usable_system_production_kwh", sum(_flow_kwh(ac.get(k, 0), k) for k, _, _ in _FLOW_END_USES)),
        "usable_system_production_kwh",
    )
    pv = balance.get("pv_dc") or {}
    curtailed = _flow_kwh(pv.get("curtailed_kwh", 0), "curtailed_kwh")
    inverter = waterfall.get("inverter") or {}
    direct_loss = inverter.get("direct_pv_conversion_loss_kwh")
    remainder = pv_dc - curtailed - usable - (vehicle or 0.0)
    if curtailed:
        rows.append(("step", "Curtailment", -curtailed))
    if direct_loss is not None:
        direct_loss = _flow_kwh(direct_loss, "direct_pv_conversion_loss_kwh")
        remainder -= direct_loss
        if direct_loss:
            ac_coupled = "battery_inverter" in waterfall or "pv_to_battery_kwh" in ac
            rows.append(
                ("step", "PV inverter conversion" if ac_coupled else "Inverter conversion, direct PV", -direct_loss)
            )
    night = inverter.get("night_consumption") or {}
    if night and len(inverter.get("inverters") or []) > 1 and direct_loss is not None and "to_inverter_kwh" in pv:
        # The direct-PV AC balance isolates PV used by idle units. Total night
        # consumption also includes grid energy, so it cannot be subtracted as a loss.
        delivered = sum(
            _flow_kwh(ac.get(key, 0), key)
            for key in (
                "direct_pv_to_load_kwh",
                "export_kwh",
                "pv_to_battery_kwh",
                "pv_to_vehicle_kwh",
            )
        )
        idle = max(0.0, _flow_kwh(pv["to_inverter_kwh"], "to_inverter_kwh") - direct_loss - delivered)
        idle = min(idle, _flow_kwh(night.get("energy_kwh", 0), "night_consumption.energy_kwh"))
        if idle >= _FLOW_MIN_KWH:
            rows.append(("step", "PV to inverter idle consumption", -idle))
            remainder -= idle
    battery = any(
        _flow_kwh(value, name) > 0
        for name, value in (
            ("to_battery_kwh", pv.get("to_battery_kwh", 0)),
            ("pv_to_battery_kwh", ac.get("pv_to_battery_kwh", 0)),
            ("pv_origin_battery_to_load_kwh", ac.get("pv_origin_battery_to_load_kwh", 0)),
        )
    )
    if abs(remainder) > 1e-9:
        if battery:
            label = (
                "Battery losses and stored energy" if direct_loss is not None else "Conversion losses and stored energy"
            )
        else:
            label = "Other conversion and energy balance"
        rows.append(("step", label, -remainder))
    if vehicle is not None and vehicle < 0:
        rows.append(("step", "Vehicle energy carried into the period", -vehicle))
    if vehicle is not None and vehicle > 0:
        rows.append(("total", "Usable AC energy and vehicle charging", usable + vehicle))
    else:
        rows.append(("total", "Usable AC energy", usable))
    return rows


def _flow_vehicle_charging(ac: Mapping[str, Any]) -> Optional[float]:
    """PV AC sent to the charger less PV-origin energy returned to the load, counted once."""
    if "pv_to_vehicle_kwh" not in ac:
        return None
    return _flow_kwh(ac["pv_to_vehicle_kwh"], "pv_to_vehicle_kwh") - _flow_kwh(
        ac.get("pv_origin_vehicle_to_load_kwh", 0), "pv_origin_vehicle_to_load_kwh"
    )


def _flow_end_uses(ac: Mapping[str, Any]) -> List[Tuple[str, str, float]]:
    """Positive bar segments; vehicle carryover joins the band as an incoming flow."""
    end_uses = [(label, color, _flow_kwh(ac.get(key, 0), key)) for key, label, color in _FLOW_END_USES]
    vehicle = _flow_vehicle_charging(ac)
    if vehicle is not None and vehicle > 0:
        end_uses.insert(3, (*_FLOW_VEHICLE_CHARGING, vehicle))
    return [item for item in end_uses if item[2] > 0]


def plot_pv_loss_waterfall(
    waterfall: dict,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    yearly: Optional[Sequence[Mapping[str, Any]]] = None,
    replacement_years: Sequence[float] = (),
):
    """
    Plot the PV energy flow from ``pv_loss_waterfall``, with an optional lifetime strip.

    A band whose width is the remaining energy runs from the horizontal
    irradiance reference down to the PV DC output and, when the waterfall has
    an ``energy_balance`` block, on to the usable AC energy. Each loss leaves
    the band as a red strip and each gain joins it as a blue strip, all on
    one scale, labelled with its kWh and its percent of the energy before it.
    The static PVWatts losses get one row each from ``pvwatts.components_kwh``.
    After the PV DC output come curtailment, the inverter's conversion loss on
    direct PV, PV supplied to idle inverters when identifiable, and a remainder
    named "Battery losses and stored energy" when there is battery flow.
    This neutral-coloured remainder includes PV energy still stored at the
    end of the period; it is not a measurement of battery losses alone. With an AC-coupled battery (a
    ``battery_inverter`` block) all PV passes through the PV inverter, so its
    row reads "PV inverter conversion", and the remainder includes the
    battery inverter. A bar under the last band splits the usable AC energy
    into direct use, use through the battery or vehicle and grid export.
    Small static components share the residual row, preserving continuity.
    A waterfall without the
    ``pvwatts``, ``inverter`` or ``energy_balance`` block draws the stages it
    has.

    With a vehicle (``ac_delivery.pv_to_vehicle_kwh``), charging it from PV is
    a use, not a loss. The bar gains a "Vehicle charging" segment:
    ``pv_to_vehicle_kwh`` less ``pv_origin_vehicle_to_load_kwh``, the PV
    energy the vehicle gave back to the home, which is already in the usable
    AC energy. The waterfall does not split the vehicle's own losses by
    origin, so the segment holds the charger and vehicle losses on the
    returned energy, and the energy driven or still on board. The last band is
    then the usable AC energy and vehicle charging together. If returns
    exceed charging, the net energy carried into the period joins the band
    as an incoming flow, and the final band equals usable AC energy.

    A monofacial system has no ``Bifacial rear gain`` row: the stage is left
    out when ``waterfall["bifacial"]["enabled"]`` is false, or, for a
    waterfall without that block, when the stage changes nothing.

    The diagram shows year 1, or the simulated window of a ``[period]`` run
    (``waterfall["basis"] == "period"``), which the default title then says.

    Args:
        waterfall: ``App.result()["pv_loss_waterfall"]`` dictionary.
        output_path: Optional PNG/PDF/SVG path. When provided, parent
            directories are created, the figure is saved and then closed.
            Without it, the figure is returned open.
        title: Figure title. None names the basis: ``"PV energy flow, year 1"``
            or ``"PV energy flow, simulated period"``.
        figsize: Matplotlib figure size. None, the default, sizes the figure
            to the diagram. A given size scales the drawing and its text to
            fit, keeping the proportions.
        yearly: Optional ``App.result()["yearly"]`` list. With more than one
            year, a strip under the diagram plots ``pv_dc_generation_kwh`` and
            ``usable_ac_system_production_kwh`` over the project years and
            tabulates year 1, the last year, the lifetime total and the
            lifetime mean.
        replacement_years: Project years with a battery replacement, marked
            on the lifetime strip, for example
            ``[event["year"] for event in result["degradation"]["replacement_events"]]``.
            Used only with ``yearly``.

    Returns:
        The matplotlib ``Figure``.
    """
    for i, stage in enumerate(waterfall.get("stages", [])):
        _flow_kwh(stage.get("energy_kwh"), f"stage {i} energy_kwh")
    stages = [stage for stage in waterfall.get("stages", []) if not _is_monofacial_rear_stage(stage, waterfall)]
    if len(stages) < 2:
        raise ValueError("waterfall must contain at least two stages")
    rows = _flow_rows(waterfall, stages)
    ac = (waterfall.get("energy_balance") or {}).get("ac_delivery") or {}
    end_uses = _flow_end_uses(ac)
    has_vehicle = _flow_vehicle_charging(ac) is not None
    years = list(yearly or [])
    for i, row in enumerate(years):
        for key in ("year", "pv_dc_generation_kwh", "usable_ac_system_production_kwh"):
            value = _flow_kwh(row.get(key), f"yearly[{i}].{key}")
            if key == "year" and (value < 1 or value != int(value)):
                raise ValueError(f"yearly[{i}].year must be a positive integer")
    if any(float(b["year"]) <= float(a["year"]) for a, b in zip(years, years[1:], strict=False)):
        raise ValueError("yearly entries must have strictly increasing year values")
    replacement_years = [_flow_kwh(y, "replacement_years") for y in replacement_years]
    lifetime = len(years) > 1
    period = waterfall.get("basis") == "period"

    # Layout in inches, s along the flow (down), c across it (right).
    energy, peak = 0.0, 1e-9
    for kind, _, value in rows:
        energy = value if kind == "total" else energy + value
        peak = max(peak, energy)
    ipk = 4.2 / peak  # inches per kWh
    step_h, total_h, radius = 0.42, 0.36, 0.28
    c_end = peak * ipk + 0.55
    x0, y0 = 0.6, 1.55
    labels = [label for kind, label, _ in rows if kind == "step"] + [label for label, _, _ in end_uses]
    change_x = max(3.45, 0.07 * max((len(label) for label in labels), default=0) + 1.1)
    pct_x = change_x + 0.8
    label_x = x0 + c_end + 0.12
    end_h = max(0.5, 0.24 * len(end_uses))
    # A gain after a loss starts lower, so the two labels keep apart.
    gain_pad = [
        0.1 if kind == "step" and v > 0 and i and rows[i - 1][0] == "step" and rows[i - 1][2] < 0 else 0.0
        for i, (kind, _, v) in enumerate(rows)
    ]
    heights = [
        total_h + 0.03 if kind == "total" else max(step_h, abs(v) * ipk + 0.2) + pad
        for (kind, _, v), pad in zip(rows, gain_pad, strict=True)
    ]
    length = sum(heights)
    flow_end = y0 + length
    height = flow_end + (0.1 + end_h + 0.45 if end_uses else 0.0)
    strip_top = height + 0.55
    if lifetime:
        height = strip_top + 0.45 + 2.2 + 0.55
    size = (label_x + pct_x + 0.35, height + 0.35)

    fig = plt.figure(figsize=figsize or size)
    # A given figsize scales the drawing, and the text with it.
    scale = min(figsize[0] / size[0], figsize[1] / size[1]) if figsize else 1.0
    fig.patch.set_facecolor(_FLOW_SURFACE)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.set_aspect("equal", anchor="N")
    ax.axis("off")
    ax.apply_aspect()
    if title is None:
        title = "PV energy flow, simulated period" if period else "PV energy flow, year 1"
    basis = "kWh over the simulated period, which is not a full year." if period else "kWh in year 1 of the project."
    ax.text(x0 - 0.15, 0.5, title, fontsize=15 * scale, weight="bold", color=_FLOW_INK, va="center")
    ax.text(
        x0 - 0.15,
        0.88,
        "The band width is the energy that remains. Grey includes energy still stored at the period end.",
        fontsize=9.5 * scale,
        color=_FLOW_INK2,
        va="center",
    )
    ax.text(x0 - 0.15, 1.12, basis, fontsize=9.5 * scale, color=_FLOW_INK2, va="center")

    def rect(s: float, c: float, ds: float, dc: float, color: str) -> None:
        ax.add_patch(Rectangle((x0 + c, y0 + s), dc, ds, facecolor=color, edgecolor="none"))

    def columns(y: float, label: str, change: str, pct: str, x: float = label_x) -> None:
        ax.text(x, y, label, ha="left", va="center", fontsize=9 * scale, color=_FLOW_INK)
        ax.text(label_x + change_x, y, change, ha="right", va="center", fontsize=9 * scale, color=_FLOW_INK)
        ax.text(label_x + pct_x, y, pct, ha="right", va="center", fontsize=9 * scale, color=_FLOW_INK2)

    s, energy = 0.0, 0.0
    for (kind, label, value), h, pad in zip(rows, heights, gain_pad, strict=True):
        if kind == "total":
            energy = value
            width = max(value, 0.0) * ipk
            rect(s, 0, total_h, width, _FLOW_TOTAL)
            text = f"{label}   {_flow_number(value)} kWh"
            mark = ax.text(
                x0 + width / 2,
                y0 + s + total_h / 2,
                text,
                ha="center",
                va="center",
                fontsize=9 * scale,
                color="white",
                weight="bold",
            )
            # A band too narrow for its label gets it beside the band, in ink.
            text_px = mark.get_window_extent(renderer=fig.canvas.get_renderer()).width
            band_px = ax.transData.transform((width, 0))[0] - ax.transData.transform((0, 0))[0]
            if text_px + 0.2 * scale * fig.dpi > band_px:
                mark.set(x=x0 + width + 0.12, ha="left", color=_FLOW_INK)
            s += h
            continue
        thickness = abs(value) * ipk
        after = energy + value
        cb, ca = max(energy, 0.0) * ipk, max(after, 0.0) * ipk
        moveto, curve, lineto = MplPath.MOVETO, MplPath.CURVE3, MplPath.LINETO
        if value < 0:  # leaves at the start of the row, then turns outward
            rect(s, 0, h, ca, _FLOW_BAND)
            lo = s + h - 0.06
            hi = lo - thickness
            pts = [
                (s, ca),
                (s, cb),
                (hi, cb),
                (hi, cb + radius),
                (hi, c_end),
                (lo, c_end),
                (lo, cb + radius),
                (lo, ca),
                (s, ca),
            ]
        else:  # arrives from outside and joins at the end of the row
            rect(s, 0, h, cb, _FLOW_BAND)
            hi = s + 0.06 + pad
            lo = hi + thickness
            pts = [
                (s + h, cb),
                (s + h, ca),
                (lo, ca),
                (lo, ca + radius),
                (lo, c_end),
                (hi, c_end),
                (hi, ca + radius),
                (hi, cb),
                (s + h, cb),
            ]
        # Storage is an energy destination as well as a loss; give the aggregate a neutral colour.
        color = _FLOW_MUTED if "stored energy" in label else (_FLOW_LOSS if value < 0 else _FLOW_GAIN)
        codes = [moveto, lineto, curve, curve, lineto, lineto, lineto, curve, curve]
        path = MplPath([(x0 + c, y0 + v) for v, c in pts], codes)
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor=color, linewidth=0.8))
        sign = "+" if value > 0 else ""
        pct = f"{sign}{_flow_number(100.0 * value / energy, 2)} %" if energy else ""
        change = f"{sign}{_flow_number(value, 0 if abs(value) >= 10 else 1)} kWh"
        columns(y0 + (hi + lo) / 2, label, change, pct)
        ax.text(
            x0 + 0.1, y0 + s + h / 2, _flow_number(after), ha="left", va="center", fontsize=8 * scale, color=_FLOW_INK2
        )
        energy = after
        s += h

    if end_uses:
        top = flow_end + 0.1
        total = sum(value for _, _, value in end_uses)
        c = 0.0
        for i, (label, color, value) in enumerate(end_uses):
            width = max(value, 0.0) * ipk
            # Each segment keeps a 0.01 in margin on both sides; one too thin for it is listed, not drawn.
            if width > 0.02:
                ax.add_patch(Rectangle((x0 + c + 0.01, top), width - 0.02, end_h, facecolor=color, edgecolor="none"))
            c += width
            y = top + end_h / 2 + (i - (len(end_uses) - 1) / 2) * 0.24
            ax.add_patch(Rectangle((label_x, y - 0.07), 0.14, 0.14, facecolor=color, edgecolor="none"))
            share = f"{_flow_number(100.0 * value / total, 1)} %" if total else ""
            columns(y, label, f"{_flow_number(value)} kWh", share, x=label_x + 0.24)
        ax.text(
            x0,
            top + end_h + 0.2,
            "Where the PV energy goes" if has_vehicle else "Where the usable AC energy goes",
            ha="left",
            va="center",
            fontsize=8.5 * scale,
            color=_FLOW_INK2,
        )

    if lifetime:
        _draw_lifetime_strip(
            fig, ax, years, replacement_years, x0, strip_top, c_end, label_x, pct_x, scale, has_vehicle
        )

    if output_path:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def _draw_lifetime_strip(
    fig: Figure,
    ax: Any,
    yearly: Sequence[Mapping[str, Any]],
    replacement_years: Sequence[float],
    x0: float,
    top: float,
    c_end: float,
    label_x: float,
    pct_x: float,
    scale: float,
    has_vehicle: bool = False,
) -> None:
    """Draw the yearly PV DC and usable AC lines with their table under the flow diagram; positions in inches."""
    years = [int(row["year"]) for row in yearly]
    dc = [float(row["pv_dc_generation_kwh"]) for row in yearly]
    usable = [float(row["usable_ac_system_production_kwh"]) for row in yearly]
    n = len(years)
    ax.text(
        x0 - 0.15,
        top,
        "Project lifetime, kWh per year",
        fontsize=10 * scale,
        weight="bold",
        color=_FLOW_INK,
        va="center",
    )

    # Place the chart in inches through the main axes, which may be scaled to a given figsize.
    ax.apply_aspect()
    to_figure = ax.transData + fig.transFigure.inverted()
    left, chart_top, chart_h = x0 + 0.55, top + 0.45, 2.2
    right = x0 + c_end - 1.35
    (fx0, fy1), (fx1, fy0) = to_figure.transform([(left, chart_top), (right, chart_top + chart_h)])
    chart = fig.add_axes((fx0, fy0, fx1 - fx0, fy1 - fy0))
    chart.set_facecolor(_FLOW_SURFACE)
    for side in ("top", "right", "left"):
        chart.spines[side].set_visible(False)
    chart.spines["bottom"].set_color(_FLOW_BASE)
    chart.tick_params(axis="both", colors=_FLOW_MUTED, labelsize=8 * scale, length=0)
    chart.yaxis.grid(True, color=_FLOW_GRID, linewidth=0.6)
    chart.set_axisbelow(True)
    chart.plot(years, dc, color=_FLOW_INK2, linewidth=2)
    chart.plot(years, usable, color=_FLOW_GAIN, linewidth=2)
    low, high = min(usable + dc), max(usable + dc)
    pad = 0.08 * (high - low or high or 1.0)
    chart.set_ylim(low - pad, high + pad)
    pad_years = max(0.15, 0.03 * (years[-1] - years[0]))
    chart.set_xlim(years[0] - pad_years, years[-1] + pad_years)
    ticks = sorted({years[0], years[-1], *range(5, years[-1], 5)}) if n > 10 else years
    chart.set_xticks([year for year in ticks if years[0] <= year <= years[-1]])
    chart.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: _flow_number(value)))
    chart.set_xlabel("Project year", fontsize=8.5 * scale, color=_FLOW_INK2)

    # Direct labels at the line ends, pushed apart when the lines end close together.
    span = chart.get_ylim()[1] - chart.get_ylim()[0]
    gap = 0.2 / chart_h * span
    y_dc, y_ac = dc[-1], usable[-1]
    if abs(y_dc - y_ac) < gap:
        middle = (y_dc + y_ac) / 2
        y_dc, y_ac = (
            (middle + gap / 2, middle - gap / 2) if dc[-1] >= usable[-1] else (middle - gap / 2, middle + gap / 2)
        )
    for y, label in ((y_dc, "PV DC output"), (y_ac, "Usable AC energy")):
        chart.annotate(
            label,
            (years[-1], y),
            xytext=(6 * scale, 0),
            textcoords="offset points",
            fontsize=8.5 * scale,
            color=_FLOW_INK2,
            va="center",
            annotation_clip=False,
        )
    marked = False
    for year in sorted(set(replacement_years)):
        if years[0] <= year <= years[-1]:
            chart.axvline(year, color=_FLOW_BASE, linewidth=0.8, linestyle=(0, (3, 3)))
            if not marked:
                chart.text(
                    0.5,
                    1.02,
                    "Battery replacement",
                    transform=chart.transAxes,
                    fontsize=8 * scale,
                    color=_FLOW_INK2,
                    ha="center",
                    va="bottom",
                )
                marked = True

    # The table beside the chart.
    right_edges = (label_x + pct_x - 2.0, label_x + pct_x - 1.0, label_x + pct_x)
    header = ("Year 1", f"Year {years[-1]}", f"{n}-year total")
    table = (("PV DC output", dc), ("Usable AC energy", usable))
    y = chart_top + 0.15
    for edge, text in zip(right_edges, header, strict=True):
        ax.text(edge, y, text, ha="right", va="center", fontsize=8.5 * scale, color=_FLOW_MUTED)
    for name, values in table:
        y += 0.32
        ax.text(label_x, y, name, ha="left", va="center", fontsize=8.5 * scale, color=_FLOW_INK, weight="bold")
        cells = (values[0], values[-1], sum(values))
        for edge, value in zip(right_edges, cells, strict=True):
            ax.text(edge, y, _flow_number(value), ha="right", va="center", fontsize=8.5 * scale, color=_FLOW_INK)
    mean = sum(usable) / n
    change = (
        f"{'+' if mean > usable[0] else ''}{_flow_number(100.0 * (mean / usable[0] - 1.0), 1)} %" if usable[0] else ""
    )
    ax.text(
        label_x,
        y + 0.55,
        f"Lifetime mean usable AC energy: {_flow_number(mean)} kWh per year,\n{change} against year 1."
        + ("\nUsable AC energy here leaves out vehicle charging." if has_vehicle else ""),
        ha="left",
        va="top",
        fontsize=8.5 * scale,
        color=_FLOW_INK2,
        linespacing=1.4,
    )


def yearly_graphs(results_df: pd.DataFrame, results_directory: Optional[str] = None) -> Figure:
    """
    Create yearly aggregated summary.

    Args:
        results_df: Energy balance results DataFrame
        results_directory: Directory to save ``yearly_energy.png`` in. None
            returns the figure open without saving it.

    Returns:
        The matplotlib ``Figure``.
    """
    df = _local_time_indexed(results_df)

    columns = [c for c in _ENERGY_LABELS if c in df.columns]

    yearly = _on_wall_clock(_power_frame_to_energy_kwh(df[columns], _result_instants(results_df))).resample("YE").sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    yearly.plot(kind="bar", ax=ax, width=0.8, alpha=0.8)

    ax.set_xticklabels([d.strftime("%Y") for d in yearly.index], rotation=0)
    ax.set_ylabel("Energy (kWh)")
    ax.legend([_ENERGY_LABELS[column] for column in columns])
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return _finish(fig, results_directory, "yearly_energy.png")


def weekly_graphs(
    results_df: pd.DataFrame, week_number: int, results_directory: Optional[str] = None
) -> Optional[Figure]:
    """
    Create detailed weekly time series plot.

    Draws PV production and load in kW, and the battery's stored energy in
    kWh on a second axis when the results have it, with one legend for all
    three and a title naming the ISO week and its dates.

    Args:
        results_df: Energy balance results DataFrame
        week_number: ISO week of the year to plot (1-53)
        results_directory: Directory to save ``week_<n>_profile.png`` in.
            None returns the figure open without saving it.

    Returns:
        The matplotlib ``Figure``, or None when the results have no row in
        that week.
    """
    df = _local_time_indexed(results_df)

    df["Week"] = df.index.isocalendar().week
    week_data = df[df["Week"] == week_number]

    if week_data.empty:
        warnings.warn(f"No data found for week {week_number}", UserWarning, stacklevel=2)
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    # The legend goes on the top axes, so the battery line does not hide it.
    legend_ax = ax
    handles = []

    if "PV_Production" in week_data.columns:
        handles.append(
            ax.fill_between(
                week_data.index, 0, week_data["PV_Production"] / 1000, alpha=0.3, color="gold", label="PV production"
            )
        )
    if "Houseload" in week_data.columns:
        handles += ax.plot(week_data.index, week_data["Houseload"] / 1000, "b-", label="Load", linewidth=1.5)
    if "Battery_Energy" in week_data.columns:
        ax2 = legend_ax = ax.twinx()
        handles += ax2.plot(
            week_data.index, week_data["Battery_Energy"] / 1000, "g--", label="Battery energy (kWh)", linewidth=1.5
        )
        ax2.set_ylabel("Battery Energy (kWh)", color="green")
        ax2.set_ylim(bottom=0)
        ax2.grid(False)

    ax.set_xlabel("Date")
    ax.set_ylabel("Power (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    first, last = week_data.index.min(), week_data.index.max()
    ax.set_title(f"Week {week_number}: {first:%d %b %Y} to {last:%d %b %Y}", loc="left")
    if handles:
        # Above the axes, right of the title, where it hides no data.
        legend_ax.legend(
            handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles), frameon=False
        )

    fig.tight_layout()
    return _finish(fig, results_directory, f"week_{week_number}_profile.png")


def _line_figure(x: Any, y: Any, style: str, ylabel: str, results_directory: Optional[str], filename: str) -> Figure:
    """One line over time on a 10 x 6 figure, as the degradation plots draw each series."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, style, linewidth=2)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _finish(fig, results_directory, filename)


def _degradation_x(degradation_df: pd.DataFrame) -> pd.Index:
    """The x values of a daily degradation frame: its ``Datetime`` on the local clock, else its index."""
    if "Datetime" in degradation_df.columns:
        return local_datetime_index(degradation_df["Datetime"])
    return degradation_df.index


def degradation_plots(degradation_df: pd.DataFrame, results_directory: Optional[str] = None) -> "dict[str, Figure]":
    """
    Create battery degradation visualization.
    Generates separate plots for SOH, degradation components and FEC, plus
    resistance growth and round-trip efficiency when the frame has them.

    Args:
        degradation_df: Degradation tracking DataFrame
        results_directory: Directory to save plots. None returns the figures
            open without saving them.

    Returns:
        The figures, keyed by file name: ``battery_degradation_soh``,
        ``battery_degradation_components_per_battery``,
        ``battery_degradation_fec``, ``battery_resistance_growth`` and
        ``battery_effective_rte``, each when the frame has its columns.
        Empty for an empty frame.
    """
    figures: "dict[str, Figure]" = {}
    if degradation_df.empty:
        return figures

    x = _degradation_x(degradation_df)

    figures["battery_degradation_soh"] = _line_figure(
        x, degradation_df["SOH"], "b-", "SOH (%)", results_directory, "battery_degradation_soh.png"
    )

    def _plot_degradation_components(cycle_data, calendar_data, filename, ylabel):
        if cycle_data is None and calendar_data is None:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if cycle_data is not None:
            ax.fill_between(x, 0, cycle_data, alpha=0.5, label="Cycle")

        if calendar_data is not None and not isinstance(calendar_data, (int, float)):
            base = cycle_data if cycle_data is not None else 0
            ax.fill_between(x, base, base + calendar_data, alpha=0.5, label="Calendar")

        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figures[filename.removesuffix(".png")] = _finish(fig, results_directory, filename)

    # These production columns reset when the battery is replaced.
    if "Cumulative_Cycle_Degradation" in degradation_df.columns:
        cum_cyc = degradation_df["Cumulative_Cycle_Degradation"] * 100
        cum_cal = degradation_df.get("Cumulative_Calendar_Degradation", 0) * 100
        _plot_degradation_components(
            cum_cyc, cum_cal, "battery_degradation_components_per_battery.png", "Per-Battery Cumulative Degradation (%)"
        )

    if "Cumulative_FEC" in degradation_df.columns:
        figures["battery_degradation_fec"] = _line_figure(
            x,
            degradation_df["Cumulative_FEC"],
            "g-",
            "Full Equivalent Cycles",
            results_directory,
            "battery_degradation_fec.png",
        )

    if "Resistance_Growth" in degradation_df.columns:
        figures.update(plot_resistance_and_efficiency(degradation_df, results_directory))
    return figures


def plot_resistance_and_efficiency(
    degradation_df: pd.DataFrame, results_directory: Optional[str] = None
) -> "dict[str, Figure]":
    """
    Plot battery resistance growth and effective round-trip efficiency.

    Args:
        degradation_df: Degradation tracking DataFrame with Resistance_Growth and Effective_RTE columns
        results_directory: Directory to save plots. None returns the figures
            open without saving them.

    Returns:
        The figures, keyed by file name: ``battery_resistance_growth``, and
        ``battery_effective_rte`` when the frame has ``Effective_RTE``. Empty
        when the frame has no ``Resistance_Growth``.
    """
    figures: "dict[str, Figure]" = {}
    if degradation_df.empty or "Resistance_Growth" not in degradation_df.columns:
        return figures

    x = _degradation_x(degradation_df)

    figures["battery_resistance_growth"] = _line_figure(
        x,
        degradation_df["Resistance_Growth"] * 100,
        "r-",
        "Resistance Growth (%)",
        results_directory,
        "battery_resistance_growth.png",
    )

    if "Effective_RTE" in degradation_df.columns:
        figures["battery_effective_rte"] = _line_figure(
            x,
            degradation_df["Effective_RTE"] * 100,
            "m-",
            "Round-Trip Efficiency (%)",
            results_directory,
            "battery_effective_rte.png",
        )
    return figures


def plot_cell_temperature(
    results_df: pd.DataFrame,
    results_directory: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot monthly battery cell temperature statistics (min, mean, max).

    Shows the seasonal trend of cell temperature with a shaded min-max band
    and mean line, aggregated by month.

    Args:
        results_df: Hourly results DataFrame with 'Datetime' and 'T_cell' columns.
        results_directory: Directory to save ``battery_cell_temperature.png``
            in. None returns the figure open without saving it.

    Returns:
        The matplotlib ``Figure``, or None when the results have no
        ``T_cell`` column.
    """
    if "T_cell" not in results_df.columns:
        return None

    df = _local_time_indexed(results_df)

    t_cell = _on_wall_clock(df["T_cell"])
    monthly_mean = t_cell.resample("ME").mean()
    monthly_min = t_cell.resample("ME").min()
    monthly_max = t_cell.resample("ME").max()

    # Group by month number (handles multi-year data)
    mean_by_month = monthly_mean.groupby(monthly_mean.index.month).mean()
    min_by_month = monthly_min.groupby(monthly_min.index.month).min()
    max_by_month = monthly_max.groupby(monthly_max.index.month).max()

    # Ensure all 12 months; a month without data is a gap, not 0 °C.
    months = np.arange(1, 13)
    mean_by_month = mean_by_month.reindex(months)
    min_by_month = min_by_month.reindex(months)
    max_by_month = max_by_month.reindex(months)

    month_names = MONTH_LABELS

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(months, min_by_month.values, max_by_month.values, alpha=0.25, color="red", label="Min–Max range")
    ax.plot(months, mean_by_month.values, "r-o", linewidth=1.5, markersize=5, label="Mean")
    ax.plot(months, min_by_month.values, "b--", linewidth=1, alpha=0.7, label="Min")
    ax.plot(months, max_by_month.values, "r--", linewidth=1, alpha=0.7, label="Max")

    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Cell Temperature (\u00b0C)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return _finish(fig, results_directory, "battery_cell_temperature.png")


def plot_timeseries(
    df: pd.DataFrame,
    columns: List[str],
    results_directory: Optional[str] = None,
    filename: str = "timeseries.png",
) -> Figure:
    """
    Plot multiple columns as time series.

    Args:
        df: DataFrame with datetime index
        columns: Column names to plot
        results_directory: Directory to save plot. None returns the figure
            open without saving it.
        filename: Output filename

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for col in columns:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _finish(fig, results_directory, filename)


def plot_breakeven(
    cost_projection: pd.DataFrame, results_directory: Optional[str] = None, scenario_name: str = ""
) -> "dict[str, Figure]":
    """
    Plot break-even analysis: PV system vs no system accumulated costs.

    Creates TWO separate graph files:
    1. breakeven_cumulative_{scenario}.png - Cumulative costs comparison
    2. breakeven_annual_{scenario}.png - Annual savings

    Break-even point calculated with month precision using linear interpolation.

    Args:
        cost_projection: DataFrame from cost_analysis_projection()
        results_directory: Directory to save plots. None returns the figures
            open without saving them.
        scenario_name: Optional suffix for filenames

    Returns:
        The figures, keyed ``breakeven_cumulative`` and ``breakeven_annual``.
    """
    suffix = _scenario_suffix(scenario_name)

    years = cost_projection["Year"]

    if "Cost_No_Sys_Cumulative_NPV" in cost_projection.columns:
        no_sys = cost_projection["Cost_No_Sys_Cumulative_NPV"]
        with_sys = cost_projection["Cost_System_Cumulative_NPV"]
        label_suffix = " (NPV)"
    else:
        no_sys = cost_projection["Cost_No_Sys_Cumulative"]
        with_sys = cost_projection["Cost_System_Cumulative"]
        label_suffix = ""

    # The year-0 investment anchors the break-even curve, as in the library.
    initial_investment = _initial_investment(cost_projection)

    # Break-even with month precision, by the same interpolation Monte Carlo
    # and the optimizer report.
    savings = pd.DataFrame({"Year": years.to_numpy(), "Savings_Cumulative_NPV": no_sys.values - with_sys.values})
    be_year_interpolated = find_payback_year_interpolated(savings, initial_investment=initial_investment)
    be_text = "Not reached"
    if be_year_interpolated is not None:
        be_years = int(be_year_interpolated)
        be_months = int((be_year_interpolated - be_years) * 12)
        be_text = f"{be_years} years {be_months} months"

    # First panel: cumulative costs
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(years, no_sys, "r-", linewidth=2.5, marker="o", markersize=4, label=f"No System{label_suffix}")
    ax1.plot(years, with_sys, "g-", linewidth=2.5, marker="s", markersize=4, label=f"PV System{label_suffix}")

    if be_year_interpolated is not None:
        # Interpolate the cost at break-even, from the year-0 investment when
        # the break-even falls inside the first year.
        curve_years, curve_cost = years.to_numpy(dtype=float), with_sys.to_numpy(dtype=float)
        if initial_investment is not None:
            curve_years, curve_cost = _from_year_zero(curve_years, curve_cost, initial_investment)
        be_cost = np.interp(be_year_interpolated, curve_years, curve_cost)
        ax1.axvline(x=be_year_interpolated, color="blue", linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.scatter([be_year_interpolated], [be_cost], s=120, c="blue", zorder=5, edgecolors="white", linewidth=2)
        ax1.annotate(
            f"Break-even\n{be_text}",
            xy=(be_year_interpolated, be_cost),
            xytext=(be_year_interpolated + 1.5, be_cost * 0.85),
            fontsize=11,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        )

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel(f"Cumulative Cost ({_currency(cost_projection)})", fontsize=12)
    ax1.set_xticks(years)  # Show every year
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    _finish(fig1, results_directory, f"breakeven_cumulative{suffix}.png")

    # Second panel: annual savings
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    annual_savings = no_sys.diff().fillna(no_sys.iloc[0]) - with_sys.diff().fillna(with_sys.iloc[0])

    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in annual_savings]
    ax2.bar(years, annual_savings, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel(f"Annual Savings ({_currency(cost_projection)})", fontsize=12)
    ax2.set_xticks(years)  # Show every year
    ax2.grid(True, alpha=0.3, axis="y")

    fig2.tight_layout()
    _finish(fig2, results_directory, f"breakeven_annual{suffix}.png")
    return {"breakeven_cumulative": fig1, "breakeven_annual": fig2}


def _on_index_clock(value, index: pd.Index) -> pd.Timestamp:
    """Return ``value`` as a timestamp comparable with ``index``: naive dates take its zone."""
    stamp = pd.Timestamp(value)
    zone = getattr(index, "tz", None)
    if zone is not None and stamp.tz is None:
        return stamp.tz_localize(zone, nonexistent="shift_forward", ambiguous=True)
    if zone is None and stamp.tz is not None:
        return stamp.tz_localize(None)
    return stamp


def plot_battery_soh_timeseries(
    results_df: pd.DataFrame,
    results_directory: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    scenario_name: str = "",
    eol_percentage: float = DEFAULT_EOL_PERCENTAGE,
) -> Optional[Figure]:
    """
    Time series plot of battery State of Health (SOH) over time.

    Args:
        results_df: Energy balance results DataFrame with Battery_SOH column
        results_directory: Directory to save plots. None returns the figure
            open without saving it.
        start_date: Optional start date filter (e.g., '2025-01-01')
        end_date: Optional end date filter (e.g., '2025-12-31')
        scenario_name: Optional suffix for filenames
        eol_percentage: The end-of-life threshold the run replaced the
            battery at, as a fraction like the ``battery_eol_percentage``
            config key. Drawn as the "End of Life" line. Defaults to that
            key's default, 0.70.

    Returns:
        The matplotlib ``Figure``, or None when the results have no
        ``Battery_SOH`` column.

    Raises:
        ValueError: If ``eol_percentage`` is not between 0 and 1.
    """
    eol_percentage = float(eol_percentage)
    if not 0.0 < eol_percentage < 1.0:
        raise ValueError(f"eol_percentage is a fraction between 0 and 1, not {eol_percentage!r}")
    eol_pct = 100.0 * eol_percentage
    suffix = _scenario_suffix(scenario_name)

    df = _local_time_indexed(results_df)

    # Filter date range if specified. The bounds are civil dates, read on the
    # results' own clock, so a naive date compares with a tz-aware index.
    if start_date:
        df = df[df.index >= _on_index_clock(start_date, df.index)]
    if end_date:
        df = df[df.index <= _on_index_clock(end_date, df.index)]

    if "Battery_SOH" not in df.columns:
        warnings.warn("Battery_SOH column not found in results", UserWarning, stacklevel=2)
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df["Battery_SOH"], "b-", linewidth=1.5, label="SOH")

    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="Initial (100%)")
    ax.axhline(y=eol_pct, color="red", linestyle="--", alpha=0.5, label=f"End of Life ({eol_pct:g}%)")

    ax.fill_between(df.index, df["Battery_SOH"], 100, alpha=0.2, color="red")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("State of Health (%)", fontsize=12)
    # Between the degradation fill at the top and the end-of-life line.
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(eol_pct - 5, df["Battery_SOH"].min() - 5), 102])

    fig.tight_layout()
    return _finish(fig, results_directory, f"battery_soh_timeseries{suffix}.png")


def plot_monthly_comparison(
    results_df: pd.DataFrame, results_directory: Optional[str] = None, scenario_name: str = ""
) -> Figure:
    """
    Compare PV production, load, import, and export by month.

    Creates a stacked/grouped bar chart showing energy flows for each month.

    Args:
        results_df: Energy balance results DataFrame
        results_directory: Directory to save plots. None returns the figure
            open without saving it.
        scenario_name: Optional suffix for filenames

    Returns:
        The matplotlib ``Figure``.
    """
    suffix = _scenario_suffix(scenario_name)

    df = _local_time_indexed(results_df)

    columns = [c for c in _ENERGY_LABELS if c in df.columns]

    monthly = _on_wall_clock(_power_frame_to_energy_kwh(df[columns], _result_instants(results_df))).resample("ME").sum()
    monthly["Month"] = monthly.index.strftime("%b")

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(monthly))
    width = 0.2

    colors = {
        "PV_Production": "#FFD700",
        "Houseload": "#4169E1",
        "Import_From_Grid": "#FF6347",
        "PV_AC_Export": "#32CD32",
    }
    for i, col in enumerate(columns):
        ax.bar(
            x + i * width,
            monthly[col],
            width,
            label=_ENERGY_LABELS[col],
            color=colors.get(col, "gray"),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x + width * (len(columns) - 1) / 2)
    ax.set_xticklabels(monthly["Month"], fontsize=11)
    ax.set_ylabel("Energy (kWh)", fontsize=12)
    ax.set_xlabel("Month", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for i, col in enumerate(columns):
        for j, val in enumerate(monthly[col]):
            if val > 0:
                ax.text(
                    j + i * width,
                    val + monthly[col].max() * 0.01,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    fig.tight_layout()
    return _finish(fig, results_directory, f"monthly_comparison{suffix}.png")


def plot_monthly_balance(results_df: pd.DataFrame, results_directory: Optional[str] = None) -> Figure:
    """
    Plot monthly energy balance with positive (PV, Export) and negative (Load, Import) bars.
    X-axis shows only month names (1-12).

    Args:
        results_df: Simulation results DataFrame
        results_directory: Directory to save ``monthly_balance.png`` in.
            None returns the figure open without saving it.

    Returns:
        The matplotlib ``Figure``.
    """
    df = _local_time_indexed(results_df)

    energy_columns = list(_ENERGY_LABELS)
    missing = [column for column in energy_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing energy-balance column(s): {', '.join(missing)}")

    # Convert power to interval energy before monthly aggregation.
    monthly = (
        _on_wall_clock(_power_frame_to_energy_kwh(df[energy_columns], _result_instants(results_df)))
        .resample("ME")
        .sum()
    )

    # Group by month (1-12) to aggregate multi-year data
    monthly_avg = monthly.groupby(monthly.index.month).mean()

    monthly_avg = monthly_avg.reindex(np.arange(1, 13), fill_value=0.0)

    months = np.arange(1, 13)
    month_names = MONTH_LABELS

    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.35

    # Positives
    labels = _ENERGY_LABELS
    ax.bar(
        months - width / 2, monthly_avg["PV_Production"], width, label=labels["PV_Production"], color="gold", alpha=0.9
    )
    ax.bar(
        months + width / 2, monthly_avg["PV_AC_Export"], width, label=labels["PV_AC_Export"], color="green", alpha=0.9
    )

    # Negatives (Load and Import)
    ax.bar(
        months - width / 2, -monthly_avg["Houseload"], width, label=labels["Houseload"], color="steelblue", alpha=0.9
    )
    ax.bar(
        months + width / 2,
        -monthly_avg["Import_From_Grid"],
        width,
        label=labels["Import_From_Grid"],
        color="red",
        alpha=0.9,
    )

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Energy (kWh)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    return _finish(fig, results_directory, "monthly_balance.png")


def _finite_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric values from a DataFrame column."""
    if column not in df.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(df[column], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _is_breos_montecarlo_summary(df: pd.DataFrame) -> bool:
    """Detect the one-row-per-run schema written by ``breos montecarlo``."""
    return "npv_savings" in df.columns and ("run" in df.columns or "payback_year" in df.columns)


def _zero_is_near(values: pd.Series) -> bool:
    """True when zero lies inside the values' range, or within half its width of either end."""
    low, high = float(values.min()), float(values.max())
    if low <= 0.0 <= high:
        return True
    return min(abs(low), abs(high)) <= 0.5 * (high - low)


def _plot_montecarlo_distribution(
    values: pd.Series,
    results_directory: Optional[str],
    filename: str,
    xlabel: str,
    color: str,
    suffix: str = "",
    zero_line: Optional[bool] = False,
) -> Optional[Figure]:
    """Shared histogram with P5/P50/P95 markers for MC summary metrics.

    ``zero_line`` draws the break-even line at zero: True always, False
    never, None when zero is inside or near the values' range. When None
    leaves the line out, a note says that every run is on one side of it,
    so the axis keeps the distribution's scale.
    """
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None

    p5 = float(values.quantile(0.05))
    p50 = float(values.quantile(0.50))
    p95 = float(values.quantile(0.95))

    fig, ax = plt.subplots(figsize=(12, 8))
    bins = min(80, max(8, int(np.sqrt(len(values)) * 2)))
    ax.hist(values, bins=bins, color=color, alpha=0.65, edgecolor="white", linewidth=0.5)

    for val, label, linestyle, linewidth in (
        (p5, "P5", "--", 1.5),
        (p50, "P50", "-", 2.5),
        (p95, "P95", "--", 1.5),
    ):
        ax.axvline(val, color="tab:red", linestyle=linestyle, linewidth=linewidth, label=f"{label}: {_grouped(val, 2)}")

    legend_loc = "best"
    if zero_line or (zero_line is None and _zero_is_near(values)):
        ax.axvline(0, color="black", linewidth=1.2, alpha=0.7, label="Break-even (0)")
    elif zero_line is None:
        side = "above" if float(values.min()) > 0.0 else "below"
        legend_loc = "upper left"
        ax.text(
            0.98,
            0.98,
            f"All {len(values)} runs are {side} break-even (0),\nwhich is off the axis",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Runs", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc=legend_loc)

    fig.tight_layout()
    return _finish(fig, results_directory, f"{filename}{suffix}.png")


def _plot_montecarlo_payback_summary(
    df: pd.DataFrame, results_directory: Optional[str], suffix: str = ""
) -> "dict[str, Figure]":
    """Plot payback distribution, CDF, and achieved/not-achieved summary."""
    total_runs = len(df)
    # The fractional year, which the distribution's 0.1-year bins and mean need;
    # older run tables carry only the integer year.
    payback = _finite_numeric_series(
        df, "payback_year_interpolated" if "payback_year_interpolated" in df.columns else "payback_year"
    )
    achieved_count = len(payback)

    figures: "dict[str, Figure]" = {}
    if achieved_count:
        payback_values = payback.tolist()
        histogram = plot_breakeven_distribution(payback_values, total_runs, results_directory, suffix)
        cdf = plot_breakeven_cdf(payback_values, results_directory, suffix, total_runs=total_runs)
        if histogram is not None:
            figures["breakeven_histogram"] = histogram
        if cdf is not None:
            figures["breakeven_cdf"] = cdf
    figures["breakeven_summary_bar"] = plot_breakeven_summary_bar(achieved_count, total_runs, results_directory, suffix)
    return figures


def plot_montecarlo_simulation(
    runs: pd.DataFrame,
    results_directory: Optional[str] = None,
    scenario_name: str = "",
    verbose: bool = True,
) -> "dict[str, Figure]":
    """
    Generate all plots for Monte Carlo simulation results.

    Args:
        runs: The one-row-per-run table written by ``breos montecarlo``.
        results_directory: Directory to save plots; they go in its ``plots``
            subdirectory. None returns the figures open without saving them.
        scenario_name: Optional suffix for filenames
        verbose: Print the output directory when plots are generated.

    Returns:
        The figures, keyed by file name: ``montecarlo_npv_distribution``,
        ``montecarlo_grid_independence_distribution``,
        ``montecarlo_final_soh_distribution``,
        ``montecarlo_lcoe_distribution``, ``breakeven_histogram``,
        ``breakeven_cdf`` and ``breakeven_summary_bar``, each when the table
        has values for it.
    """
    if not _is_breos_montecarlo_summary(runs):
        raise ValueError(
            "plot_montecarlo_simulation needs the one-row-per-run table written by breos montecarlo "
            "(an npv_savings column and a run or payback_year column)"
        )

    suffix = _scenario_suffix(scenario_name)
    plots_folder = None if results_directory is None else os.path.join(results_directory, "plots")
    if plots_folder is not None:
        os.makedirs(plots_folder, exist_ok=True)

    drawn = {
        "montecarlo_npv_distribution": plot_montecarlo_npv_distribution(runs, plots_folder, suffix),
        "montecarlo_grid_independence_distribution": plot_montecarlo_grid_independence_distribution(
            runs, plots_folder, suffix
        ),
        "montecarlo_final_soh_distribution": plot_montecarlo_final_soh_distribution(runs, plots_folder, suffix),
        "montecarlo_lcoe_distribution": _plot_montecarlo_distribution(
            _finite_numeric_series(runs, "lcoe_per_kwh"),
            plots_folder,
            "montecarlo_lcoe_distribution",
            f"LCOE ({_currency(runs)}/kWh)",
            "tab:purple",
            suffix,
        ),
    }
    figures = {name: fig for name, fig in drawn.items() if fig is not None}
    figures.update(_plot_montecarlo_payback_summary(runs, plots_folder, suffix))
    if verbose and plots_folder is not None:
        print(f"Monte Carlo plots saved to: {plots_folder}")
    return figures


def _finite_payback_years(breakeven_steps: Sequence[Optional[float]]) -> np.ndarray:
    """The payback years that are numbers; NaN, None and inf are runs that never pay back."""
    years = pd.to_numeric(pd.Series(list(breakeven_steps), dtype=object), errors="coerce").to_numpy(dtype=float)
    return years[np.isfinite(years)]


def plot_breakeven_distribution(
    breakeven_steps: List[float], total_runs: int, results_directory: Optional[str] = None, suffix: str = ""
) -> Optional[Figure]:
    """
    Histogram of the time to break-even of the runs that pay back.

    Args:
        breakeven_steps: The fractional payback year of each run, such as
            the ``payback_year_interpolated`` column of a Monte Carlo run
            table. NaN, None and inf entries, runs that never pay back, are
            not drawn.
        total_runs: The number of runs, those that never pay back included.
        results_directory: Directory to save ``breakeven_histogram.png`` in.
            None returns the figure open without saving it.
        suffix: Suffix for the file name.

    Returns:
        The matplotlib ``Figure``, or None when no run pays back.
    """
    paid_back = _finite_payback_years(breakeven_steps)
    if not paid_back.size:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    bin_width = 0.1
    min_be = float(paid_back.min())
    max_be = float(paid_back.max())
    bins = np.arange(min_be - 0.05, max_be + 0.15, bin_width)

    n, bins, patches = ax.hist(paid_back, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, linewidth=1)

    achieved = len(paid_back)
    mean_val = np.mean(paid_back)
    median_val = np.median(paid_back)
    std_val = np.std(paid_back)

    stats_text = (
        f"Total Runs: {total_runs}\n"
        f"Achieved: {achieved} ({achieved / total_runs:.1%})\n"
        f"Mean: {mean_val:.2f} yrs\n"
        f"Median: {median_val:.2f} yrs\n"
        f"Std Dev: {std_val:.2f} yrs"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
        fontfamily="monospace",
    )

    ax.set_xlabel("Years to break-even", fontsize=12)
    ax.set_ylabel("Number of Runs", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    for i in range(len(n)):
        if n[i] > 0:
            ax.text((bins[i] + bins[i + 1]) / 2, n[i] + 0.1, int(n[i]), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return _finish(fig, results_directory, f"breakeven_histogram{suffix}.png")


def plot_breakeven_cdf(
    breakeven_steps: List[float],
    results_directory: Optional[str] = None,
    suffix: str = "",
    total_runs: Optional[int] = None,
) -> Optional[Figure]:
    """
    Plot the cumulative share of runs that have paid back, against the years to break-even.

    The curve is normalised by every run, so its plateau is the share of
    runs that ever pay back; a dotted line marks it. The 2.5, 25, 50, 75 and
    97.5% points are marked where the curve reaches them; a level above the
    plateau has no point, since that share of runs never pays back.

    Args:
        breakeven_steps: The fractional payback year of each run, such as
            the ``payback_year_interpolated`` column of a Monte Carlo run
            table. NaN, None and inf entries are runs that never pay back.
        results_directory: Directory to save ``breakeven_cdf.png`` in. None
            returns the figure open without saving it.
        suffix: Suffix for the file name.
        total_runs: The number of runs. Defaults to the length of
            ``breakeven_steps``; give it when that list holds only the runs
            that pay back.

    Returns:
        The matplotlib ``Figure``, or None when no run pays back.

    Raises:
        ValueError: If ``total_runs`` is less than the number of runs that
            pay back.
    """
    x = np.sort(_finite_payback_years(breakeven_steps))
    n = len(x)
    if not n:
        return None
    total = len(breakeven_steps) if total_runs is None else int(total_runs)
    if total < n:
        raise ValueError(f"total_runs is {total}, but {n} runs pay back")
    y = np.arange(1, n + 1) / total
    share = n / total

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.step(np.r_[x[0], x], np.r_[0.0, y], where="post", color="blue", linewidth=2, label="Runs paid back")
    ax.axhline(share, color="blue", linestyle=":", alpha=0.6, linewidth=1, label=f"Ever pay back: {share:.1%}")

    # Quantiles of every run; a run that never pays back sits at infinity.
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    colors = ["red", "gray", "black", "gray", "red"]
    every_run = np.concatenate([x, np.full(total - n, np.inf)])

    for q, color in zip(quantiles, colors, strict=True):
        val = float(np.quantile(every_run, q, method="inverted_cdf"))
        if not np.isfinite(val):
            continue
        ax.axvline(val, color=color, linestyle="--", alpha=0.6, linewidth=1)
        ax.scatter([val], [q], color=color, zorder=5)
        ax.text(val, q, f" {q:.1%} ({val:.1f}y)", color=color, ha="left", va="bottom", fontsize=9)

    ax.set_xlabel("Years to break-even", fontsize=12)
    ax.set_ylabel(f"Share of runs paid back (of {total})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    fig.tight_layout()
    return _finish(fig, results_directory, f"breakeven_cdf{suffix}.png")


def plot_breakeven_summary_bar(
    achieved_count: int, total_runs: int, results_directory: Optional[str] = None, suffix: str = ""
) -> Figure:
    """
    Bar chart of Success vs Failure for break-even.

    Args:
        achieved_count: The number of runs that pay back.
        total_runs: The number of runs.
        results_directory: Directory to save ``breakeven_summary_bar.png``
            in. None returns the figure open without saving it.
        suffix: Suffix for the file name.

    Returns:
        The matplotlib ``Figure``.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Break-even\nAchieved", "No Break-even"]
    not_achieved = total_runs - achieved_count
    counts = [achieved_count, not_achieved]
    colors = ["green", "red"]

    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor="black")

    for bar, count in zip(bars, counts, strict=True):
        if total_runs > 0:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count}\n({count / total_runs:.1%})",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    ax.set_ylabel("Number of Runs")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(counts) * 1.2)

    fig.tight_layout()
    return _finish(fig, results_directory, f"breakeven_summary_bar{suffix}.png")


def plot_montecarlo_npv_distribution(
    all_results_df: pd.DataFrame,
    results_directory: Optional[str] = None,
    suffix: str = "",
    zero_line: Optional[bool] = None,
) -> Optional[Figure]:
    """
    Histogram of NPV savings (``npv_savings``) across one-row-per-run MC results.

    Args:
        all_results_df: The one-row-per-run table written by ``breos montecarlo``.
        results_directory: Directory to save
            ``montecarlo_npv_distribution.png`` in. None returns the figure
            open without saving it.
        suffix: Suffix for the file name.
        zero_line: Whether to draw the break-even line at zero NPV. True
            always draws it, False never does. None, the default, draws it
            when zero is inside the range of the runs' NPV, or within half
            that range's width of it, so a distribution far from zero keeps
            its own scale; a note then says that every run is above (or
            below) break-even.

    Returns:
        The matplotlib ``Figure``, or None when no run has a finite NPV.
    """
    return _plot_montecarlo_distribution(
        _finite_numeric_series(all_results_df, "npv_savings"),
        results_directory,
        "montecarlo_npv_distribution",
        f"NPV Savings ({_currency(all_results_df)})",
        "tab:blue",
        suffix,
        zero_line=zero_line,
    )


def plot_montecarlo_grid_independence_distribution(
    all_results_df: pd.DataFrame, results_directory: Optional[str] = None, suffix: str = ""
) -> Optional[Figure]:
    """
    Histogram of mean grid independence (``mean_grid_independence_pct``) across one-row-per-run MC results.

    Args:
        all_results_df: The one-row-per-run table written by ``breos montecarlo``.
        results_directory: Directory to save the plot in. None returns the
            figure open without saving it.
        suffix: Suffix for the file name.

    Returns:
        The matplotlib ``Figure``, or None when no run has a finite value.
    """
    return _plot_montecarlo_distribution(
        _finite_numeric_series(all_results_df, "mean_grid_independence_pct"),
        results_directory,
        "montecarlo_grid_independence_distribution",
        "Mean Grid Independence (%)",
        "tab:green",
        suffix,
    )


def plot_montecarlo_final_soh_distribution(
    all_results_df: pd.DataFrame, results_directory: Optional[str] = None, suffix: str = ""
) -> Optional[Figure]:
    """
    Histogram of final battery state-of-health across one-row-per-run MC results.

    Args:
        all_results_df: The one-row-per-run table written by ``breos montecarlo``.
        results_directory: Directory to save the plot in. None returns the
            figure open without saving it.
        suffix: Suffix for the file name.

    Returns:
        The matplotlib ``Figure``, or None when no run has a finite value.
    """
    return _plot_montecarlo_distribution(
        _finite_numeric_series(all_results_df, "final_soh_pct"),
        results_directory,
        "montecarlo_final_soh_distribution",
        "Final Battery SOH (%)",
        "tab:cyan",
        suffix,
    )


# Name and unit of each weather variable the comparison plots draw. Irradiance
# is summed to an energy total, temperature averaged.
_WEATHER_VARIABLES = {
    "ghi": ("GHI", "kWh/m²"),
    "dni": ("DNI", "kWh/m²"),
    "dhi": ("DHI", "kWh/m²"),
    "temp_air": ("Air temperature", "°C"),
}


def _weather_values(weather: pd.DataFrame, variable: str) -> pd.Series:
    """One weather variable as a series indexed by time.

    Time is the ``date`` column of a :func:`~breos.weather.preload_weather_by_year`
    year, else the index. Irradiance is found under any of its aliases
    (``shortwave_radiation`` is GHI), air temperature as
    :func:`~breos.weather.extract_ambient_temperature` finds it.
    """
    if variable not in _WEATHER_VARIABLES:
        raise ValueError(f"variable must be one of {', '.join(_WEATHER_VARIABLES)}, not {variable!r}")
    if variable == "temp_air":
        temperature = extract_ambient_temperature(weather)
        column = None if temperature is None else temperature.name
    else:
        column = find_irradiance_column(weather.columns, variable)
    if column is None:
        raise ValueError(f"The weather has no {variable} column; it has {', '.join(map(str, weather.columns))}")
    index = pd.DatetimeIndex(weather["date"] if "date" in weather.columns else weather.index)
    if len(index) < 2:
        raise ValueError("The weather needs at least two timestamps")
    return pd.Series(pd.to_numeric(weather[column], errors="coerce").to_numpy(dtype=float), index=index)


def _weather_monthly(weather: pd.DataFrame, variable: str, source: str = "The weather") -> pd.Series:
    """Monthly irradiance totals in kWh/m², or monthly mean temperature, indexed 1 to 12.

    Missing values are skipped with a warning that names ``source``. A month
    without any value is NaN.
    """
    values = _weather_values(weather, variable)
    missing = int(values.isna().sum())
    if missing:
        warnings.warn(
            f"{source} has {missing} missing {variable} values; its monthly figures skip them",
            UserWarning,
            stacklevel=3,
        )
    by_month = values.groupby(values.index.month)
    if variable == "temp_air":
        monthly = by_month.mean()
    else:
        hours_per_step = values.index.to_series().diff().median().total_seconds() / 3600.0
        monthly = by_month.sum(min_count=1) * hours_per_step / 1000.0
    return monthly.reindex(range(1, 13))


def _annual_ghi(weather: pd.DataFrame, source: str) -> float:
    """Annual GHI in kWh/m², the sum of the twelve monthly totals."""
    monthly = _weather_monthly(weather, "ghi", source)
    absent = [MONTH_LABELS[month - 1] for month in monthly.index[monthly.isna()]]
    if absent:
        raise ValueError(f"{source} has no GHI for {', '.join(absent)}, so it has no annual total")
    return float(monthly.sum())


def _historical_weather_years(
    historical: Union[str, "os.PathLike[str]", Mapping[int, pd.DataFrame]],
) -> "dict[int, pd.DataFrame]":
    """The complete years of a multi-year weather file, as a Monte Carlo study reads them."""
    if isinstance(historical, Mapping):
        years = dict(historical)
    else:
        years = _weather_years(os.fspath(historical), target_year=2025)
    if not years:
        raise ValueError("The historical weather has no complete year")
    return years


def _weather_monthly_stats(historical_years: Mapping[int, pd.DataFrame], variable: str) -> pd.DataFrame:
    """Per month: the historical mean, its 95% confidence interval, and the lowest and highest value.

    The minimum and maximum of each month are taken over the years
    separately, so they can come from different years.
    """
    table = pd.DataFrame(
        {year: _weather_monthly(frame, variable, f"Historical year {year}") for year, frame in historical_years.items()}
    )
    count = table.count(axis=1)
    mean = table.mean(axis=1)
    half_width = student_t.ppf(0.975, count - 1) * table.std(axis=1) / np.sqrt(count)
    return pd.DataFrame(
        {
            "mean": mean,
            "ci_low": mean - half_width,
            "ci_high": mean + half_width,
            "min": table.min(axis=1),
            "max": table.max(axis=1),
        }
    )


def plot_weather_monthly_comparison(
    tmy: pd.DataFrame,
    historical: Union[str, "os.PathLike[str]", Mapping[int, pd.DataFrame]],
    results_directory: Optional[str] = None,
    variable: str = "ghi",
    tmy_label: str = "TMY",
    filename: Optional[str] = None,
) -> Figure:
    """
    Compare a TMY with historical weather years, month by month.

    Draws the TMY, the historical mean with its 95% confidence interval, and
    the monthly minimum and maximum: each month's lowest and highest value
    over the historical years, which can come from different years.
    Irradiance is the monthly total in kWh/m²; temperature is the monthly
    mean in °C. A month with missing values warns, and its total skips them.

    Args:
        tmy: One weather year as a DataFrame: the frame
            :func:`breos.weather.load_weather` returns, or the first item of
            the ``(frame, metadata)`` tuple
            :func:`breos.weather.fetch_tmy_weather_data` returns.
        historical: The multi-year weather CSV that a Monte Carlo study
            samples (``MonteCarloSettings.weather_file``), or the per-year
            frames :func:`breos.weather.preload_weather_by_year` splits it
            into. Only complete years count, as in the study.
        results_directory: Directory to save the plot. None returns the
            figure open without saving it.
        variable: ``"ghi"``, ``"dni"``, ``"dhi"`` or ``"temp_air"``.
        tmy_label: Legend label of the TMY line, for example its source.
        filename: Output filename. Defaults to ``weather_monthly_<variable>.png``.

    Returns:
        The matplotlib ``Figure``.
    """
    name, unit = _WEATHER_VARIABLES.get(variable, (variable, ""))
    tmy_vals = _weather_monthly(tmy, variable, "The TMY").to_numpy()
    historical_years = _historical_weather_years(historical)
    monthly_stats = _weather_monthly_stats(historical_years, variable)

    x = np.arange(12)
    hist_mean = monthly_stats["mean"].values
    hist_ci_low = monthly_stats["ci_low"].values
    hist_ci_high = monthly_stats["ci_high"].values
    hist_min = monthly_stats["min"].values
    hist_max = monthly_stats["max"].values

    fig, ax = plt.subplots(figsize=(14, 7))

    # Monthly minimum / maximum over the historical years
    ax.plot(
        x,
        hist_min,
        color="tomato",
        linewidth=1.5,
        linestyle="--",
        marker="v",
        markersize=5,
        zorder=3,
        label="Monthly minimum",
    )
    ax.plot(
        x,
        hist_max,
        color="seagreen",
        linewidth=1.5,
        linestyle="--",
        marker="^",
        markersize=5,
        zorder=3,
        label="Monthly maximum",
    )

    ax.fill_between(x, hist_ci_low, hist_ci_high, color="steelblue", alpha=0.25, label="95% CI of the mean")

    ax.plot(
        x,
        hist_mean,
        color="steelblue",
        linewidth=2.5,
        marker="o",
        markersize=7,
        zorder=4,
        label=f"Historical mean ({len(historical_years)} years)",
    )

    ax.plot(x, tmy_vals, color="darkorange", linewidth=2.5, marker="D", markersize=6, zorder=6, label=tmy_label)

    ax.set_xlabel("Month")
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.5, 11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return _finish(fig, results_directory, filename or f"weather_monthly_{variable}.png", bbox_inches="tight")


def plot_weather_annual_ghi_distribution(
    tmy: pd.DataFrame,
    historical: Union[str, "os.PathLike[str]", Mapping[int, pd.DataFrame]],
    results_directory: Optional[str] = None,
    tmy_label: str = "TMY",
    filename: str = "annual_ghi_distribution.png",
) -> Figure:
    """
    Histogram of the annual GHI of historical weather years, with the TMY and the historical mean.

    Args:
        tmy: One weather year as a DataFrame: the frame
            :func:`breos.weather.load_weather` returns, or the first item of
            the ``(frame, metadata)`` tuple
            :func:`breos.weather.fetch_tmy_weather_data` returns.
        historical: The multi-year weather CSV that a Monte Carlo study
            samples (``MonteCarloSettings.weather_file``), or the per-year
            frames :func:`breos.weather.preload_weather_by_year` splits it
            into. Only complete years count, as in the study.
        results_directory: Directory to save the plot. None returns the
            figure open without saving it.
        tmy_label: Legend label of the TMY line, for example its source.
        filename: Output filename.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If the TMY or a historical year has no GHI for a whole
            month. Missing values inside a month warn, and the total skips them.
    """
    tmy_annual_ghi = _annual_ghi(tmy, "The TMY")
    annual_ghi_per_year = pd.Series(
        {
            year: _annual_ghi(frame, f"Historical year {year}")
            for year, frame in _historical_weather_years(historical).items()
        }
    )
    hist_annual_ghi_mean = float(annual_ghi_per_year.mean())

    n_years = len(annual_ghi_per_year)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        annual_ghi_per_year.values,
        bins=10,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
        label=f"Historical ({n_years} years)",
    )
    ax.axvline(
        tmy_annual_ghi,
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        label=f"{tmy_label} ({tmy_annual_ghi:.0f} kWh/m²)",
    )
    ax.axvline(
        hist_annual_ghi_mean,
        color="navy",
        linewidth=2,
        linestyle="-",
        label=f"Historical mean ({hist_annual_ghi_mean:.0f} kWh/m²)",
    )
    ax.set_xlabel("Annual GHI (kWh/m²)")
    ax.set_ylabel("Count (years)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=True)

    # Ensure TMY and mean lines are always visible with padding
    data_min = annual_ghi_per_year.values.min()
    data_max = annual_ghi_per_year.values.max()
    x_min = min(data_min, tmy_annual_ghi, hist_annual_ghi_mean)
    x_max = max(data_max, tmy_annual_ghi, hist_annual_ghi_mean)
    span = (x_max - x_min) or 1.0
    ax.set_xlim(x_min - 0.05 * span, x_max + 0.05 * span)

    fig.tight_layout()
    return _finish(fig, results_directory, filename, bbox_inches="tight")


def _breakeven_projection(projection: Union[pd.DataFrame, Mapping[str, Any]]) -> pd.DataFrame:
    """A cost projection frame: the frame itself, or one built from an App result's ``financial`` rows."""
    if isinstance(projection, pd.DataFrame):
        return projection
    rows = projection.get("financial")
    if not rows:
        raise ValueError("The App result has no 'financial' projection; a [period] run has no break-even")
    years = [row for row in rows if row["year"] > 0]
    frame = pd.DataFrame(
        {
            "Year": [row["year"] for row in years],
            "Cost_No_Sys_Cumulative_NPV": [row["cost_without_system"] for row in years],
            "Cost_System_Cumulative_NPV": [row["cost_with_system"] for row in years],
            "Savings_Cumulative_NPV": [row["balance"] for row in years],
        }
    )
    # Year 0 holds minus the investment, where payback is read from.
    year_zero = [row for row in rows if row["year"] == 0]
    if year_zero:
        frame.attrs["total_investment"] = -float(year_zero[0]["balance"])
    currency = (projection.get("provenance") or {}).get("currency")
    if currency is not None:
        frame.attrs["currency"] = currency
    return frame


def plot_breakeven_comparison(
    projections: Sequence[Union[pd.DataFrame, Mapping[str, Any]]],
    labels: Sequence[str],
    results_directory: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    currency: Optional[str] = None,
    filename: str = "breakeven_comparison.png",
) -> Figure:
    """
    Multi-scenario break-even comparison: N cumulative cost curves vs No-System baseline.

    Each curve starts at year 0 with the investment, and a dotted line,
    labelled with its year, marks its payback by
    :func:`breos.economics.find_payback_year_interpolated`. Scenarios with the
    same no-system cost share one baseline: "No system" when every scenario
    shares it, else "No system (<labels>)" for each group.

    Args:
        projections: One per scenario: a :meth:`breos.App.result` dict, whose
            ``financial`` rows are read, or a cost projection frame from
            :func:`breos.economics.cost_analysis_projection` or its CSV,
            with ``Year``, ``Cost_No_Sys_Cumulative_NPV``,
            ``Cost_System_Cumulative_NPV`` and ``Savings_Cumulative_NPV``.
        labels: Display label for each scenario.
        results_directory: Output directory. None returns the figure open
            without saving it.
        colors: Line colour for each scenario. Defaults to the colour cycle.
        currency: Currency code for the money axis, for projections that
            do not record it. An App result, a projection frame from
            :func:`~breos.economics.cost_analysis_projection` and a
            projection CSV that BREOS wrote, with its ``currency`` column,
            record their own. When no projection records it and ``currency``
            is None, the axis shows the amounts without a currency code.
        filename: Output filename.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If the labels or colours do not match the projections,
            the projections record different currencies, ``currency``
            contradicts the recorded one, or an App result has no
            ``financial`` projection.
    """
    if len(labels) != len(projections):
        raise ValueError(f"{len(projections)} projections need {len(projections)} labels, not {len(labels)}")
    if colors is None:
        colors = [f"C{index % 10}" for index in range(len(projections))]
    elif len(colors) != len(projections):
        raise ValueError(f"{len(projections)} projections need {len(projections)} colors, not {len(colors)}")
    cost_dfs = [_breakeven_projection(projection) for projection in projections]
    label_currency = _label_currency(cost_dfs, currency)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Year 0 costs the investment with a system and nothing without one.
    curves = []
    for df in cost_dfs:
        years = df["Year"].to_numpy(dtype=float)
        no_sys = df["Cost_No_Sys_Cumulative_NPV"].to_numpy(dtype=float)
        with_sys = df["Cost_System_Cumulative_NPV"].to_numpy(dtype=float)
        investment = _initial_investment(df)
        if investment is not None:
            _, no_sys = _from_year_zero(years, no_sys, 0.0)
            years, with_sys = _from_year_zero(years, with_sys, investment)
        curves.append((years, no_sys, with_sys))

    # One no-system baseline per group of scenarios that share it.
    baselines: "dict[Tuple[Any, ...], List[int]]" = {}
    for index, (years, no_sys, _) in enumerate(curves):
        baselines.setdefault((tuple(years), tuple(np.round(no_sys, 0))), []).append(index)
    for members in baselines.values():
        years, no_sys, _ = curves[members[0]]
        if len(baselines) == 1:
            no_sys_label, no_sys_color = "No system", "black"
        else:
            no_sys_label = f"No system ({', '.join(str(labels[index]) for index in members)})"
            no_sys_color = colors[members[0]]
        ax.plot(years, no_sys, color=no_sys_color, linestyle="--", label=no_sys_label, linewidth=2.5, alpha=0.7)

    max_year = 0
    for df, (years, _, with_sys), label, color in zip(cost_dfs, curves, labels, colors, strict=True):
        ax.plot(years, with_sys, color=color, label=label, linewidth=2)
        max_year = max(max_year, int(df["Year"].max()))

        # Break-even dotted line, labelled with its year at the foot of the axes
        be = find_payback_year_interpolated(df)
        if be is not None:
            ax.axvline(x=be, color=color, linestyle=":", alpha=0.5, linewidth=1)
            ax.text(
                be,
                0.02,
                f" {be:.1f} years",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="bottom",
                color=color,
                fontsize=10,
            )

    unit = f" {label_currency}" if label_currency else ""
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative Cost ({label_currency})" if label_currency else "Cumulative Cost")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{_grouped(x)}{unit}"))
    ax.set_xticks(range(0, max_year + 1))
    ax.set_xlim(0, max_year + 0.5)

    fig.tight_layout()
    return _finish(fig, results_directory, filename, bbox_inches="tight")


# Axis labels of the sweep and optimizer columns BREOS writes, without their
# ``param_`` or ``resolved_`` prefix; an optimizer ``Projected_`` or
# ``Objective_`` column is looked up by its full name, then without the
# prefix. Other columns get a label made from their name.
_COLUMN_LABELS = {
    "n_modules": "Number of PV modules",
    "battery_kwh": "Battery capacity (kWh)",
    "pv_kwp": "PV size (kWp)",
    "tilt": "Tilt (°)",
    "azimuth": "Azimuth (°)",
    "grid_independence_pct": "Grid independence (%)",
    "self_consumption_pct": "Self-consumption (%)",
    "usable_ac_system_production_kwh": "Usable AC system production (kWh)",
    "total_investment": "Investment ({currency})",
    "npv_savings": "NPV savings ({currency})",
    "lcoe_per_kwh": "LCOE ({currency}/kWh)",
    "payback_year": "Payback (years)",
    "Modules": "Number of PV modules",
    "Battery_kWh": "Battery capacity (kWh)",
    "Tilt": "Tilt (°)",
    "Azimuth": "Azimuth (°)",
    "Grid_Independence_%": "Grid independence (%)",
    "NPV": "NPV savings ({currency})",
    "ZEB_Ratio": "ZEB ratio",
    "Projected_Initial_Cost": "Investment ({currency})",
    "LCOE_per_kWh": "LCOE ({currency}/kWh)",
    "Payback_Year": "Payback (years)",
    "Payback_Year_Interpolated": "Payback (years)",
    "Final_SOH_%": "Final battery SOH (%)",
    "Total_Replacements": "Battery replacements",
    "Replacement_Cost_T0_Prices": "Replacement cost at today's prices ({currency})",
    "CO2_Avoided_Total_kg": "CO$_2$ avoided over the projection (kg CO$_2$eq)",
    "CO2_Avoided_SelfConsumed_kg": "CO$_2$ avoided by self-consumed PV over the projection (kg CO$_2$eq)",
}

# Trailing unit tokens of a column name, as the label's unit.
_NAME_UNITS = {"kg": "kg", "kWh": "kWh", "kWp": "kWp", "pct": "%", "%": "%", "kW": "kW", "W": "W", "years": "years"}

_COMPASS_POINTS = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}


def _label_template(column: str) -> Optional[str]:
    """The ``_COLUMN_LABELS`` entry of a column, with ``{currency}`` unfilled, or None."""
    name = str(column)
    for prefix in ("param_", "resolved_"):
        name = name.removeprefix(prefix)
    if name in _COLUMN_LABELS:
        return _COLUMN_LABELS[name]
    for prefix in ("Projected_", "Objective_"):
        if name.startswith(prefix):
            return _COLUMN_LABELS.get(name.removeprefix(prefix))
    return None


def _name_label(column: str) -> str:
    """A label made from a column name: words for the underscores, a trailing unit in parentheses.

    ``Projected_PV_DC_Year1_kWh`` is "PV DC Year1 (kWh)".
    """
    name = str(column)
    for prefix in ("param_", "resolved_", "Projected_", "Objective_"):
        name = name.removeprefix(prefix)
    words = [word for word in name.split("_") if word]
    if len(words) > 1 and words[-1] in _NAME_UNITS:
        unit = _NAME_UNITS[words.pop()]
        return f"{' '.join(words)} ({unit})"
    return " ".join(words) or str(column)


def _column_label(column: str, currency: Optional[str]) -> str:
    """Axis label of a sweep or optimizer column.

    Money is labelled in ``currency``; with None, the label names no currency
    ("NPV savings", "LCOE (per kWh)").
    """
    label = _label_template(column)
    if label is None:
        return _name_label(column)
    if currency is None:
        return label.replace(" ({currency})", "").replace("{currency}/kWh", "per kWh")
    return label.format(currency=currency)


def _money_currency(frames: Sequence[pd.DataFrame], currency: Optional[str], columns: Sequence[str]) -> Optional[str]:
    """:func:`_label_currency` when a label of ``columns`` shows money, else None.

    Only a money label needs the inputs to agree on a currency, so a
    grid-independence difference between a CHF and a EUR sweep still plots.
    """
    if any("{currency}" in (_label_template(column) or "") for column in columns):
        return _label_currency(frames, currency)
    return None


def _difference_label(metric: str, label: str, labels: Optional[Tuple[str, str]]) -> str:
    """Colour-bar label of a difference: a percentage becomes percentage points."""
    name, unit = label, ""
    if label.endswith(")") and " (" in label:
        name, unit = label[:-1].rsplit(" (", 1)
    if unit == "%" or (not unit and str(metric).endswith(("_pct", "_%"))):
        unit = "percentage points"
    compared = f", {labels[0]} − {labels[1]}" if labels else " difference"
    return f"{name}{compared} ({unit})" if unit else f"{name}{compared}"


def _read_table(data: Union[pd.DataFrame, str, "os.PathLike[str]"]) -> pd.DataFrame:
    """A result table: the frame itself, or the CSV at ``data``."""
    return data if isinstance(data, pd.DataFrame) else pd.read_csv(data)


def _sweep_column(frame: pd.DataFrame, name: str) -> str:
    """The ``param_<name>`` column ``breos sweep`` writes for a swept key, else the column ``name``.

    The swept column comes first: a sweep CSV also has result columns such as
    ``n_modules`` and ``battery_kwh``, which hold the App's resolved values.
    """
    for column in (f"param_{name}", name):
        if column in frame.columns:
            return column
    raise ValueError(
        f"The table has no 'param_{name}' or {name!r} column; its columns are {', '.join(map(str, frame.columns))}"
    )


def _metric_column(frame: pd.DataFrame, metric: str) -> str:
    """``metric``, checked to be a column of ``frame``."""
    if metric not in frame.columns:
        numeric = ", ".join(str(column) for column in frame.select_dtypes("number").columns)
        raise ValueError(f"The table has no {metric!r} column; its numeric columns are {numeric}")
    return metric


def _swept_parameters(frame: pd.DataFrame) -> List[str]:
    """The ``param_`` columns of a ``breos sweep`` table, in sweep order."""
    return [str(column) for column in frame.columns if str(column).startswith("param_")]


def _sweep_grid(frame: pd.DataFrame, x: str, y: str, metric: str) -> pd.DataFrame:
    """``metric`` with one row per ``y`` value and one column per ``x`` value, both ascending."""
    if frame[[y, x]].duplicated().any():
        axes = {x.removeprefix("param_"), y.removeprefix("param_")}
        others = [
            column
            for column in _swept_parameters(frame)
            if column.removeprefix("param_") not in axes and frame[column].nunique() > 1
        ]
        hint = f"; it also varies {', '.join(others)}, so select one value of each" if others else ""
        raise ValueError(f"The sweep has more than one row for some {x} and {y} pair{hint}")
    grid = frame.pivot(index=y, columns=x, values=metric).sort_index().sort_index(axis=1)
    return grid.astype(float)


def _tick_text(value: Any) -> str:
    """A grid value as a tick label: 10.0 is "10", 2500000.0 is "2500000", 0.25 is "0.25"."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        return str(value)
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}" if abs(number) < 1e6 else f"{number:.0f}"


def _cell_format(values: np.ndarray) -> str:
    """One number format for every cell of a grid, from its largest magnitude."""
    finite = np.abs(values[np.isfinite(values)])
    largest = float(finite.max()) if finite.size else 0.0
    if largest >= 100:
        return ",.0f"
    if largest >= 10:
        return ".1f"
    if largest >= 1:
        return ".2f"
    return ".3f"


def plot_sweep_heatmap(
    sweep: Union[pd.DataFrame, str, "os.PathLike[str]"],
    metric: str,
    results_directory: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    diff: Union[pd.DataFrame, str, "os.PathLike[str]", None] = None,
    labels: Optional[Tuple[str, str]] = None,
    metric_label: Optional[str] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    currency: Optional[str] = None,
    filename: Optional[str] = None,
) -> Figure:
    """
    Heatmap of one metric of a two-parameter ``breos sweep`` result.

    Each cell is one run, annotated with its value when the grid has at most
    15 rows and 15 columns. With ``diff``, each cell is ``sweep`` minus
    ``diff``, for example the same sizing grid at two locations or two
    tariffs. Cells in only one of the two sweeps stay blank. A difference of
    a percentage, such as grid independence, is labelled in percentage points.

    A difference, and a metric with a negative value such as a loss in
    ``npv_savings``, are drawn on a diverging scale centred on zero and
    symmetric about it, unless ``vmin`` or ``vmax`` is given. Other metrics
    get a sequential scale.

    Args:
        sweep: The CSV ``breos sweep`` writes, or its DataFrame.
        metric: The result column to draw, such as ``grid_independence_pct``
            or ``npv_savings``.
        results_directory: Directory to save the plot. None returns the
            figure open without saving it.
        x: Parameter on the horizontal axis, as its sweep key (``n_modules``)
            or its column (``param_n_modules``). A sweep key names the swept
            ``param_`` column, not the result column of the same name.
            ``x`` and ``y`` default to the two swept parameters, in sweep order.
        y: Parameter on the vertical axis.
        diff: A second sweep over the same parameters, subtracted from ``sweep``.
        labels: Names of ``sweep`` and ``diff`` for the colour-bar label.
            Only used with ``diff``.
        metric_label: Colour-bar label. Defaults to a label for the metric.
        cmap: Matplotlib colormap name. Defaults to ``RdBu`` on a diverging
            scale, else ``YlGnBu``.
        vmin: Colour-scale minimum (auto if None). Giving ``vmin`` or
            ``vmax`` turns a metric with a negative value to the sequential
            scale.
            With ``diff`` the scale is always symmetric about zero, and
            ``vmin`` and ``vmax`` are not read.
        vmax: Colour-scale maximum (auto if None).
        currency: Currency code for money labels, for a table that does not
            record it. A sweep CSV records it in its ``currency`` column, and
            a DataFrame can in ``attrs["currency"]``. When nothing names it,
            money labels show no currency code.
        filename: Output filename. Defaults to ``sweep_<metric>.png``, or
            ``sweep_<metric>_diff.png`` with ``diff``.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If ``x`` and ``y`` are not given and the sweep does not
            vary exactly two parameters, if it has more than one run per cell
            (it varies a third parameter), if a column is missing, if the two
            sweeps share no cell or record different currencies, or if
            ``labels`` is given without ``diff``.
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm

    if labels is not None and diff is None:
        raise ValueError("labels names the two sweeps of a difference; pass diff= as well")
    frame = _read_table(sweep)
    if x is None and y is None:
        swept = _swept_parameters(frame)
        if len(swept) != 2:
            raise ValueError(f"Name x and y: the sweep varies {len(swept)} parameters ({', '.join(swept)}), not 2")
        x_column, y_column = swept
    elif x is None or y is None:
        raise ValueError("Give both x and y, or neither")
    else:
        x_column, y_column = _sweep_column(frame, x), _sweep_column(frame, y)
    grid = _sweep_grid(frame, x_column, y_column, _metric_column(frame, metric))

    if diff is None:
        values = grid
        label_currency = _money_currency([frame], currency, [metric, x_column, y_column])
    else:
        other = _read_table(diff)
        label_currency = _money_currency([frame, other], currency, [metric, x_column, y_column])
        # The difference table is resolved on its own, so either may name a key bare or as param_.
        other_x = _sweep_column(other, x_column.removeprefix("param_"))
        other_y = _sweep_column(other, y_column.removeprefix("param_"))
        values = grid.sub(_sweep_grid(other, other_x, other_y, _metric_column(other, metric)))
        if not np.isfinite(values.to_numpy()).any():
            raise ValueError(f"The two sweeps share no {x_column} and {y_column} cell with a {metric} value")
    data = values.to_numpy()
    finite = data[np.isfinite(data)]
    if not finite.size:
        raise ValueError(f"The sweep has no finite {metric} value")

    label = metric_label or _column_label(metric, label_currency)
    if diff is not None:
        if metric_label is None:
            label = _difference_label(metric, label, labels)
        else:
            label = f"{label} ({labels[0]} − {labels[1]})" if labels else f"{label} difference"
    # Zero-centred when the cells are differences or include a loss.
    signed = finite.min() < 0 and vmin is None and vmax is None
    if diff is not None or signed:
        abs_max = float(np.abs(finite).max()) or 1.0
        norm: Normalize = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        cmap = cmap or "RdBu"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        norm.autoscale_None(finite)
        cmap = cmap or "YlGnBu"

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, origin="lower")

    ax.set_xticks(range(len(values.columns)))
    ax.set_xticklabels([_tick_text(c) for c in values.columns])
    ax.set_yticks(range(len(values.index)))
    ax.set_yticklabels([_tick_text(i) for i in values.index])
    ax.set_xlabel(_column_label(x_column, label_currency))
    ax.set_ylabel(_column_label(y_column, label_currency))

    # Annotate cells, in white on dark colours
    if data.shape[0] <= 15 and data.shape[1] <= 15:
        number_format = ("+" if diff is not None else "") + _cell_format(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    red, green, blue, _ = im.cmap(norm(val))
                    text_color = "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        format(val, number_format).replace("-", "\u2212"),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                        fontweight="bold",
                    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(label)

    default_name = f"sweep_{metric}_diff.png" if diff is not None else f"sweep_{metric}.png"
    fig.tight_layout()
    return _finish(fig, results_directory, filename or default_name, bbox_inches="tight")


def _set_azimuth_ticks(ax, azimuths: np.ndarray) -> None:
    """Label azimuth ticks with their compass point, where the range spans three of them.

    Ticks are the multiples of 45° in the range, negative ones included, as a
    southern-hemisphere sweep from −90° to 90° has them.
    """
    low, high = float(np.min(azimuths)), float(np.max(azimuths))
    ticks = list(range(int(np.ceil(low / 45.0)) * 45, int(np.floor(high / 45.0)) * 45 + 1, 45))
    if len(ticks) >= 3:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{angle}° {_COMPASS_POINTS[angle % 360]}".replace("-", "−") for angle in ticks])


def plot_orientation_landscape(
    sweep: Union[pd.DataFrame, str, "os.PathLike[str]"],
    metric: str,
    results_directory: Optional[str] = None,
    tilt: str = "tilt",
    azimuth: str = "azimuth",
    maximize: bool = True,
    metric_label: Optional[str] = None,
    currency: Optional[str] = None,
    filename: str = "orientation_landscape.png",
) -> Figure:
    """
    Plot a metric over panel orientation from a ``breos sweep`` result.

    A tilt × azimuth sweep gets two panels. The left one maps the metric over
    both angles and marks the best orientation. The right one is the
    east-west profile: the metric against azimuth at the best tilt. Azimuth
    ticks carry compass points, negative azimuths included. A flat optimum
    (0° tilt) is labelled without an azimuth, as every azimuth ties there.

    A sweep of tilt alone gets one panel, the metric against tilt, with the
    best tilt marked. This is the plot for an east-west roof whose
    ``[[pv_arrays]]`` set their azimuths and take the swept top-level ``tilt``.

    Args:
        sweep: The CSV ``breos sweep`` writes, or its DataFrame. It must have
            one run per orientation.
        metric: The result column to draw, such as ``usable_ac_system_production_kwh`` or
            ``npv_savings``.
        results_directory: Directory to save the plot. None returns the
            figure open without saving it.
        tilt: The tilt parameter, as its sweep key or column.
        azimuth: The azimuth parameter, as its sweep key or column. A table
            without it, or with one azimuth only, is a tilt sweep.
        maximize: True if a higher metric is better; False for a metric such
            as ``lcoe_per_kwh``.
        metric_label: Axis and colour-bar label. Defaults to a label for the metric.
        currency: Currency code for a money metric, for a table that does
            not record it in ``attrs["currency"]`` or, as a sweep CSV does, in
            a ``currency`` column. When nothing names it, the label shows no
            currency code.
        filename: Output filename.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If the sweep has more than one run for an orientation
            (it varies another parameter too), if a column is missing, or if
            the metric has no finite value.
    """
    frame = _read_table(sweep)
    tilt_column = _sweep_column(frame, tilt)
    try:
        azimuth_column: Optional[str] = _sweep_column(frame, azimuth)
    except ValueError:
        azimuth_column = None
    if azimuth_column is not None and frame[azimuth_column].nunique() < 2:
        azimuth_column = None
    metric = _metric_column(frame, metric)
    if not np.isfinite(pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)).any():
        raise ValueError(f"The sweep has no finite {metric} value")
    label = metric_label or _column_label(metric, _money_currency([frame], currency, [metric]))
    pick = np.nanargmax if maximize else np.nanargmin

    if azimuth_column is None:
        if frame[tilt_column].duplicated().any():
            raise ValueError(f"The sweep has more than one row for some {tilt_column}; select one value of the others")
        profile = frame.set_index(tilt_column)[metric].astype(float).sort_index()
        best = int(pick(profile.to_numpy()))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(profile.index, profile.values, "b-", marker="o", linewidth=2)
        ax.plot(
            profile.index[best],
            profile.iloc[best],
            "rx",
            markersize=12,
            mew=3,
            label=f"Optimum: {profile.index[best]:g}° tilt",
        )
        ax.set_xlabel("Tilt (°)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return _finish(fig, results_directory, filename, bbox_inches="tight")

    grid = _sweep_grid(frame, azimuth_column, tilt_column, metric)
    row, col = np.unravel_index(int(pick(grid.to_numpy())), grid.shape)
    best_tilt, best_azimuth = float(grid.index[row]), float(grid.columns[col])
    azimuths = grid.columns.to_numpy(dtype=float)
    optimum = (
        f"Optimum: {best_tilt:g}° tilt"
        if best_tilt == 0
        else f"Optimum: {best_tilt:g}° tilt, {best_azimuth:g}° azimuth"
    )

    fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.25, 1]})
    mesh = ax_map.pcolormesh(
        azimuths, grid.index.to_numpy(dtype=float), grid.to_numpy(), cmap="viridis", shading="nearest"
    )
    ax_map.scatter([best_azimuth], [best_tilt], color="red", marker="x", s=200, linewidth=3, label=optimum)
    fig.colorbar(mesh, ax=ax_map, label=label)
    ax_map.set_xlabel("Azimuth (°)")
    ax_map.set_ylabel("Tilt (°)")
    _set_azimuth_ticks(ax_map, azimuths)
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12))

    # East-west profile through the optimum
    ax_profile.plot(azimuths, grid.iloc[row].to_numpy(), "b-", marker="o", linewidth=2, label=f"{best_tilt:g}° tilt")
    ax_profile.plot(best_azimuth, grid.iloc[row, col], "rx", markersize=12, mew=3, label="Optimum")
    ax_profile.set_xlabel("Azimuth (°)")
    ax_profile.set_ylabel(label)
    _set_azimuth_ticks(ax_profile, azimuths)
    ax_profile.grid(True, alpha=0.3)
    ax_profile.legend()

    fig.tight_layout()
    return _finish(fig, results_directory, filename, bbox_inches="tight")


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask of the points no other point dominates, when both coordinates are maximised.

    A point is dominated when another is at least as good in both and better
    in one. Equal points on the front all stay on it.
    """
    front = np.zeros(len(x), dtype=bool)
    best_y = -np.inf
    last: Optional[Tuple[float, float]] = None
    # Best x first, and the best y among equal x; a point is on the front if
    # its y beats every point before it.
    for index in np.lexsort((-y, -x)):
        point = (float(x[index]), float(y[index]))
        if point[1] > best_y or point == last:
            front[index] = True
            best_y, last = point[1], point
    return front


def plot_pareto_front(
    designs: Any,
    results_directory: Optional[str] = None,
    x: str = "Grid_Independence_%",
    y: str = "NPV",
    maximize: Tuple[bool, bool] = (True, True),
    color_by: Optional[str] = None,
    currency: Optional[str] = None,
    filename: str = "pareto_front.png",
) -> Figure:
    """
    Scatter two objectives of a set of designs and draw their Pareto front.

    The designs no other design beats in both objectives are drawn large and
    joined by a line; the dominated ones are drawn small and grey. On the
    optimizer's front with its own two objectives, every design is on it.
    Axes are labelled with the quantity and its unit; the optimizer's
    ``Projected_`` columns are labelled like the ones they project, and other
    columns by their name, with a trailing unit such as ``_kg`` in
    parentheses. With ``color_by``, the legend draws the front's marker
    without fill, since the colour bar gives its colours.

    Args:
        designs: The :class:`~breos.optimization.OptimizationResult` of
            :func:`~breos.optimization.optimize_system_multi_objective`, its
            ``details["pareto"]`` frame or that frame's CSV, or any table of
            designs, such as a ``breos sweep`` CSV.
        results_directory: Directory to save the plot. None returns the
            figure open without saving it.
        x: Column of the objective on the horizontal axis, or a swept key.
        y: Column of the objective on the vertical axis, or a swept key.
            The defaults are the optimizer's two objectives.
        maximize: For ``x`` and ``y``, True if higher is better; False for a
            cost such as ``Projected_Initial_Cost`` or ``lcoe_per_kwh``.
        color_by: Column that colours the front, such as ``Battery_kWh``,
            with a colour bar. None draws it in one colour.
        currency: Currency code for money labels, for a table that does not
            record it. The optimizer's frame, a DataFrame with
            ``attrs["currency"]`` and a CSV with a ``currency`` column record
            their own. When nothing names it, money labels show no currency
            code.
        filename: Output filename.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If a column is missing, ``currency`` contradicts the
            recorded one, or no design has finite values of both objectives.
    """
    details = getattr(designs, "details", None)
    if isinstance(details, Mapping) and "pareto" in details:
        designs = details["pareto"]
    frame = _read_table(designs)
    x_column, y_column = _sweep_column(frame, x), _sweep_column(frame, y)
    color_column = None if color_by is None else _sweep_column(frame, color_by)
    label_currency = _money_currency(
        [frame], currency, [x_column, y_column] + ([] if color_column is None else [color_column])
    )
    x_values = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not finite.any():
        raise ValueError(f"No design has finite {x_column} and {y_column} values")
    frame, x_values, y_values = frame[finite], x_values[finite], y_values[finite]
    signs = [1.0 if higher else -1.0 for higher in maximize]
    front = _pareto_mask(signs[0] * x_values, signs[1] * y_values)

    fig, ax = plt.subplots(figsize=(11, 7))
    if not front.all():
        ax.scatter(
            x_values[~front], y_values[~front], s=25, color="grey", alpha=0.4, label=f"Dominated ({(~front).sum()})"
        )
    order = np.argsort(x_values[front], kind="stable")
    ax.plot(x_values[front][order], y_values[front][order], color="black", linewidth=1, alpha=0.6, zorder=2)
    front_label = f"Pareto front ({front.sum()})"
    if color_column is None:
        ax.scatter(
            x_values[front], y_values[front], s=70, color="tab:blue", edgecolor="black", zorder=3, label=front_label
        )
    else:
        points = ax.scatter(
            x_values[front],
            y_values[front],
            s=70,
            c=pd.to_numeric(frame[color_column], errors="coerce").to_numpy(dtype=float)[front],
            cmap="viridis",
            edgecolor="black",
            zorder=3,
        )
        fig.colorbar(points, ax=ax, label=_column_label(color_column, label_currency))
    handles, _ = ax.get_legend_handles_labels()
    if color_column is not None:
        # A legend marker in one of the colour-map colours would match only
        # some of the points; an open marker on the joining line matches all.
        handles.append(
            Line2D(
                [],
                [],
                color="dimgray",
                linewidth=1,
                marker="o",
                markersize=np.sqrt(70),
                markerfacecolor="none",
                markeredgecolor="black",
                label=front_label,
            )
        )
    ax.set_xlabel(_column_label(x_column, label_currency))
    ax.set_ylabel(_column_label(y_column, label_currency))
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles)

    fig.tight_layout()
    return _finish(fig, results_directory, filename, bbox_inches="tight")


def plot_co2_savings(
    cost_projection: pd.DataFrame,
    results_directory: Optional[str] = None,
    scenario_name: str = "",
) -> "dict[str, Figure]":
    """
    Plot CO2 emissions avoided over system lifetime.

    Creates two figures: co2_avoided_yearly_{scenario}.png, with yearly CO2
    avoided (total and self-consumed) as bars, and
    co2_avoided_cumulative_{scenario}.png, with the cumulative totals as lines.

    Args:
        cost_projection: DataFrame from cost_analysis_projection() with CO2 columns
        results_directory: Directory to save plots. None returns the figures
            open without saving them.
        scenario_name: Optional suffix for filenames

    Returns:
        The figures, keyed ``co2_avoided_yearly`` and
        ``co2_avoided_cumulative``. Empty when the projection has no CO2
        columns.
    """
    if "CO2_Avoided_Total_kg" not in cost_projection.columns:
        return {}

    suffix = _scenario_suffix(scenario_name)

    years = cost_projection["Year"]
    co2_total = cost_projection["CO2_Avoided_Total_kg"]
    co2_self = cost_projection["CO2_Avoided_SelfConsumed_kg"]
    co2_total_cum = cost_projection["CO2_Avoided_Total_Cumulative_kg"]
    co2_self_cum = cost_projection["CO2_Avoided_SelfConsumed_Cumulative_kg"]

    bar_width = 0.35

    # First panel: yearly CO2 avoided, as bars
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(years))
    ax.bar(x - bar_width / 2, co2_total, bar_width, label="Total PV Production", color="#2196F3", alpha=0.85)
    ax.bar(x + bar_width / 2, co2_self, bar_width, label="Self-Consumed PV", color="#4CAF50", alpha=0.85)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("CO$_2$ Avoided (kg CO$_2$eq)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in years])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    yearly_fig = _finish(fig, results_directory, f"co2_avoided_yearly{suffix}.png", bbox_inches="tight")

    # Second panel: cumulative CO2 avoided, as lines
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(years, co2_total_cum / 1000, "b-", linewidth=2.5, marker="o", markersize=4, label="Total PV Production")
    ax.plot(years, co2_self_cum / 1000, "g-", linewidth=2.5, marker="s", markersize=4, label="Self-Consumed PV")
    ax.fill_between(years, 0, co2_self_cum / 1000, alpha=0.15, color="green")
    ax.fill_between(years, co2_self_cum / 1000, co2_total_cum / 1000, alpha=0.10, color="blue")

    # Annotate final values right of the last point, in room added to the
    # axes, so the labels cross neither the lines nor the frame.
    final_total = co2_total_cum.iloc[-1] / 1000
    final_self = co2_self_cum.iloc[-1] / 1000
    first_year, last_year = float(years.iloc[0]), float(years.iloc[-1])
    ax.set_xlim(first_year - 0.5, last_year + max(1.0, 0.12 * (last_year - first_year)))
    for value, offset, va, color in ((final_total, 3, "bottom", "#1565C0"), (final_self, -3, "top", "#2E7D32")):
        ax.annotate(
            f"{_grouped(value, 1)} t",
            xy=(last_year, value),
            xytext=(8, offset),
            textcoords="offset points",
            ha="left",
            va=va,
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative CO$_2$ Avoided (t CO$_2$eq)", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    cumulative_fig = _finish(fig, results_directory, f"co2_avoided_cumulative{suffix}.png", bbox_inches="tight")
    return {"co2_avoided_yearly": yearly_fig, "co2_avoided_cumulative": cumulative_fig}
