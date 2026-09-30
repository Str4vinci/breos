"""
Plotting and visualization module.

This module provides visualization functions for:
- Cost projections
- Energy balance results
- Monthly/yearly/weekly analysis
- Battery degradation
"""

import os
import warnings
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from breos.economics import _initial_investment, find_payback_year_interpolated
from breos.tariffs import DEFAULT_CURRENCY
from breos.utils import find_irradiance_column, local_datetime_index
from breos.weather import extract_ambient_temperature, preload_weather_by_year

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# matplotlib is the optional ``plots`` extra. ``import breos`` loads this
# module only when a plotting name is first used, so the error surfaces there.
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
except ImportError as exc:
    raise ModuleNotFoundError('breos.plotting needs matplotlib. Install it with: pip install "breos[plots]"') from exc


def _currency(frame: pd.DataFrame) -> str:
    """The currency a frame's money is in, for labels: ``attrs["currency"]``, else the default."""
    return str(frame.attrs.get("currency", DEFAULT_CURRENCY))


def _label_currency(frames: Sequence[pd.DataFrame], currency: Optional[str]) -> Optional[str]:
    """The currency of the money in ``frames``, or None if nothing records it.

    A frame records its currency in ``attrs["currency"]``; a table read back
    from CSV records none. ``currency`` names it for those. Two different
    recorded currencies, or a recorded one that ``currency`` contradicts,
    raise ``ValueError``.
    """
    known = sorted({str(frame.attrs["currency"]) for frame in frames if frame.attrs.get("currency") is not None})
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


def _format_loss_energy(value_kwh: float) -> str:
    """Format kWh values compactly for plot annotations."""
    if abs(value_kwh) >= 1000:
        return f"{value_kwh / 1000:.2f} MWh"
    return f"{value_kwh:.1f} kWh"


def _format_loss_delta(stage: dict) -> str:
    """Format a stage delta using the plot's sign convention."""
    delta_pct = float(stage.get("delta_pct_of_previous", 0.0))
    delta_kwh = float(stage.get("delta_kwh", 0.0))
    sign = "+" if delta_pct > 0 else ""
    return f"{sign}{delta_pct:.2f}% ({sign}{_format_loss_energy(delta_kwh)})"


def plot_pv_loss_waterfall(
    waterfall: dict,
    output_path: Optional[str] = None,
    title: str = "PV Loss Diagram - Year 1",
    figsize: Tuple[float, float] = (12.5, 10.0),
):
    """
    Plot an annual PV loss diagram from ``pv_loss_waterfall``.

    Args:
        waterfall: ``App.result()["pv_loss_waterfall"]`` dictionary.
        output_path: Optional PNG/PDF/SVG path. When provided, parent
            directories are created and the figure is saved.
        title: Figure title.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib ``Figure``.
    """
    stages = list(waterfall.get("stages", []))
    if len(stages) < 2:
        raise ValueError("waterfall must contain at least two stages")

    energies = np.array([float(stage["energy_kwh"]) for stage in stages], dtype=float)
    max_energy = max(float(np.nanmax(energies)), 1.0)
    reference_energy = energies[0] if energies[0] > 0 else 1.0
    n_stages = len(stages)

    # Funnel occupies the left/centre; stage names and step losses sit to the
    # right of each band (PVsyst-style) so the diagram is self-describing.
    center_x = 0.32
    max_width = 0.46
    top_y, bottom_y = 0.86, 0.30
    y_positions = np.linspace(top_y, bottom_y, n_stages)
    widths = np.maximum(0.05, max_width * energies / max_energy)
    left = center_x - widths / 2.0
    right = center_x + widths / 2.0
    label_x = center_x + max_width / 2.0 + 0.035

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.965, title, ha="left", va="center", fontsize=16, weight="bold", color="#111827")
    ax.text(
        0.04,
        0.93,
        "Annual PV conversion stages to the dispatch-ready DC bus",
        ha="left",
        va="center",
        fontsize=9,
        color="#4b5563",
    )

    flow_points = [(left[0], y_positions[0]), *zip(left[1:], y_positions[1:], strict=True)]
    flow_points += [(right[-1], y_positions[-1]), *zip(right[-2::-1], y_positions[-2::-1], strict=True)]
    ax.add_patch(Polygon(flow_points, closed=True, facecolor="#eef6ff", edgecolor="none", zorder=1))

    # Hatch (rather than solid-fill) the area gained/lost between consecutive
    # stages so the wedges read as loss annotations, not as a render artefact.
    for idx in range(1, n_stages):
        delta = float(stages[idx].get("delta_kwh", 0.0))
        if delta == 0.0:
            continue
        if delta > 0:
            face, hatch_color = "#f0fdf4", "#4ade80"
        else:
            face, hatch_color = "#fef2f2", "#f87171"
        y0, y1 = y_positions[idx - 1], y_positions[idx]
        for prev_edge, edge in ((left[idx - 1], left[idx]), (right[idx - 1], right[idx])):
            ax.add_patch(
                Polygon(
                    [(prev_edge, y0), (edge, y1), (prev_edge, y1)],
                    closed=True,
                    facecolor=face,
                    edgecolor=hatch_color,
                    hatch="////",
                    linewidth=0.0,
                    zorder=2,
                )
            )

    ax.add_patch(Polygon(flow_points, closed=True, facecolor="none", edgecolor="#1f2937", linewidth=1.2, zorder=4))

    for idx, (stage, y) in enumerate(zip(stages, y_positions, strict=True)):
        is_edge = idx in (0, n_stages - 1)
        ax.plot([left[idx], right[idx]], [y, y], color="#1f2937", linewidth=0.7, alpha=0.35, zorder=3)

        cumulative_pct = 100.0 * energies[idx] / reference_energy
        ax.text(
            center_x,
            y + 0.011,
            f"{_format_loss_energy(float(stage['energy_kwh']))}  ·  {cumulative_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.6,
            weight="bold" if is_edge else "normal",
            color="#111827",
        )

        # Dotted leader tying the band to its stage label on the right.
        ax.plot(
            [right[idx] + 0.008, label_x - 0.008],
            [y, y],
            color="#9ca3af",
            linewidth=0.6,
            linestyle=(0, (2, 3)),
            zorder=0,
        )
        if idx == 0:
            ax.text(label_x, y, stage["label"], ha="left", va="center", fontsize=9.2, weight="bold", color="#111827")
            continue
        delta_pct = float(stage.get("delta_pct_of_previous", 0.0))
        step_color = "#047857" if delta_pct > 0 else "#b91c1c" if delta_pct < 0 else "#9ca3af"
        ax.text(
            label_x,
            y + 0.009,
            stage["label"],
            ha="left",
            va="bottom",
            fontsize=9.2,
            weight="bold" if is_edge else "normal",
            color="#111827",
        )
        ax.text(
            label_x,
            y - 0.009,
            _format_loss_delta(stage) if delta_pct else "0.00% (0.0 kWh)",
            ha="left",
            va="top",
            fontsize=8.4,
            color=step_color,
        )

    ax.text(
        center_x,
        bottom_y - 0.05,
        f"Dispatch-ready PV DC: {_format_loss_energy(float(stages[-1]['energy_kwh']))}",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color="#111827",
    )

    detail_y0 = 0.035
    detail_h = 0.165
    ax.add_patch(
        Rectangle(
            (0.04, detail_y0),
            0.92,
            detail_h,
            facecolor="#f9fafb",
            edgecolor="#e5e7eb",
            linewidth=0.8,
        )
    )
    component_items = list(waterfall.get("pvwatts", {}).get("components_kwh", {}).items())
    left_components = component_items[:5]
    right_components = component_items[5:]
    component_pct = waterfall.get("pvwatts", {}).get("components_pct", {})

    def _component_line(name: str, value: float) -> str:
        pct = component_pct.get(name)
        pct_text = f"{pct:.1f}% · " if isinstance(pct, (int, float)) else ""
        return f"{name.replace('_', ' ').title()}: {pct_text}{_format_loss_energy(float(value))}"

    heading_y = detail_y0 + detail_h - 0.022
    row0_y = detail_y0 + detail_h - 0.046
    row_step = 0.021
    ax.text(0.06, heading_y, "Static PVWatts components", ha="left", va="center", fontsize=8.7, weight="bold")
    for row, (name, value) in enumerate(left_components):
        ax.text(
            0.06,
            row0_y - row * row_step,
            _component_line(name, value),
            ha="left",
            va="center",
            fontsize=7.5,
            color="#374151",
        )
    for row, (name, value) in enumerate(right_components):
        ax.text(
            0.35,
            row0_y - row * row_step,
            _component_line(name, value),
            ha="left",
            va="center",
            fontsize=7.5,
            color="#374151",
        )

    ax.text(0.66, heading_y, "Inverter and dispatch", ha="left", va="center", fontsize=8.7, weight="bold")
    balance = waterfall.get("energy_balance", {})
    pv_dc = balance.get("pv_dc", {})
    ac = balance.get("ac_delivery", {})
    extra_rows = [
        ("Direct PV to load (AC)", ac.get("direct_pv_to_load_kwh", 0.0)),
        ("PV via battery to load (AC)", ac.get("pv_origin_battery_to_load_kwh", 0.0)),
        ("Grid export (AC)", ac.get("export_kwh", 0.0)),
        ("Inverter conversion loss", waterfall.get("inverter", {}).get("conversion_loss_kwh", 0.0)),
        ("Curtailment (DC)", pv_dc.get("curtailed_kwh", 0.0)),
    ]
    for row, (label, value) in enumerate(extra_rows):
        ax.text(
            0.66,
            row0_y - row * row_step,
            f"{label}: {_format_loss_energy(float(value))}",
            ha="left",
            va="center",
            fontsize=7.5,
            color="#374151",
        )

    # Keep this summary near the detail box rather than as a stage row because
    # the component losses are sequential and therefore do not sum arithmetically
    # to the combined percentage.
    combined = waterfall.get("pvwatts", {}).get("combined_kwh")
    combined_pct = waterfall.get("pvwatts", {}).get("combined_pct")
    if isinstance(combined, (int, float)) and isinstance(combined_pct, (int, float)):
        ax.text(
            0.35,
            detail_y0 + 0.012,
            f"PVWatts combined: {combined_pct:.2f}% · {_format_loss_energy(float(combined))}",
            ha="left",
            va="center",
            fontsize=7.7,
            color="#111827",
            weight="bold",
        )

    if output_path:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def yearly_graphs(results_df: pd.DataFrame, results_directory: str) -> None:
    """
    Create yearly aggregated summary.

    Args:
        results_df: Energy balance results DataFrame
        results_directory: Directory to save plots
    """
    os.makedirs(results_directory, exist_ok=True)

    df = _local_time_indexed(results_df)

    columns = ["PV_Production", "Houseload", "Import_From_Grid", "PV_AC_Export"]
    columns = [c for c in columns if c in df.columns]

    yearly = _power_frame_to_energy_kwh(df[columns], _result_instants(results_df)).resample("YE").sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    yearly.plot(kind="bar", ax=ax, width=0.8, alpha=0.8)

    ax.set_xticklabels([d.strftime("%Y") for d in yearly.index], rotation=0)
    ax.set_ylabel("Energy (kWh)")
    labels = {
        "PV_Production": "PV Production",
        "Houseload": "Load",
        "Import_From_Grid": "Grid Import",
        "PV_AC_Export": "Grid Export",
    }
    ax.legend([labels[column] for column in columns])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{results_directory}/yearly_energy.png", dpi=300)
    plt.close()


def weekly_graphs(results_df: pd.DataFrame, week_number: int, results_directory: str) -> None:
    """
    Create detailed weekly time series plot.

    Args:
        results_df: Energy balance results DataFrame
        week_number: Week of year to plot (1-52)
        results_directory: Directory to save plots
    """
    os.makedirs(results_directory, exist_ok=True)

    df = _local_time_indexed(results_df)

    # Filter to specific week
    df["Week"] = df.index.isocalendar().week
    week_data = df[df["Week"] == week_number]

    if week_data.empty:
        print(f"No data found for week {week_number}")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    if "PV_Production" in week_data.columns:
        ax.fill_between(
            week_data.index, 0, week_data["PV_Production"] / 1000, alpha=0.3, color="gold", label="PV Production"
        )
    if "Houseload" in week_data.columns:
        ax.plot(week_data.index, week_data["Houseload"] / 1000, "b-", label="Load", linewidth=1.5)
    if "Battery_Energy" in week_data.columns:
        ax2 = ax.twinx()
        ax2.plot(week_data.index, week_data["Battery_Energy"] / 1000, "g--", label="Battery (kWh)", linewidth=1.5)
        ax2.set_ylabel("Battery Energy (kWh)", color="green")

    ax.set_xlabel("Date")
    ax.set_ylabel("Power (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))

    plt.tight_layout()
    plt.savefig(f"{results_directory}/week_{week_number}_profile.png", dpi=300)
    plt.close()


def _degradation_x(degradation_df: pd.DataFrame) -> pd.Index:
    """The x values of a daily degradation frame: its ``Datetime`` on the local clock, else its index."""
    if "Datetime" in degradation_df.columns:
        return local_datetime_index(degradation_df["Datetime"])
    return degradation_df.index


def degradation_plots(degradation_df: pd.DataFrame, results_directory: str) -> None:
    """
    Create battery degradation visualization.
    Generates separate plots for SOH, degradation components and FEC, plus
    resistance growth and round-trip efficiency when the frame has them.

    Args:
        degradation_df: Degradation tracking DataFrame
        results_directory: Directory to save plots
    """
    if degradation_df.empty:
        print("No degradation data to plot")
        return

    os.makedirs(results_directory, exist_ok=True)

    x = _degradation_x(degradation_df)

    # 1. SOH over time
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, degradation_df["SOH"], "b-", linewidth=2)
    ax1.set_ylabel("SOH (%)")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_directory}/battery_degradation_soh.png", dpi=300)
    plt.close()

    # 2. Degradation components for the active battery inventory.
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
        plt.tight_layout()
        plt.savefig(os.path.join(results_directory, filename), dpi=300)
        plt.close()

    # These production columns reset when the battery is replaced.
    if "Cumulative_Cycle_Degradation" in degradation_df.columns:
        cum_cyc = degradation_df["Cumulative_Cycle_Degradation"] * 100
        cum_cal = degradation_df.get("Cumulative_Calendar_Degradation", 0) * 100
        _plot_degradation_components(
            cum_cyc, cum_cal, "battery_degradation_components_per_battery.png", "Per-Battery Cumulative Degradation (%)"
        )

    # 3. FEC
    if "Cumulative_FEC" in degradation_df.columns:
        fig, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(x, degradation_df["Cumulative_FEC"], "g-", linewidth=2)
        ax3.set_ylabel("Full Equivalent Cycles")
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_directory}/battery_degradation_fec.png", dpi=300)
        plt.close()

    # 4. Resistance growth and RTE (if available)
    if "Resistance_Growth" in degradation_df.columns:
        plot_resistance_and_efficiency(degradation_df, results_directory)


def plot_resistance_and_efficiency(degradation_df: pd.DataFrame, results_directory: str) -> None:
    """
    Plot battery resistance growth and effective round-trip efficiency.

    Args:
        degradation_df: Degradation tracking DataFrame with Resistance_Growth and Effective_RTE columns
        results_directory: Directory to save plots
    """
    if degradation_df.empty or "Resistance_Growth" not in degradation_df.columns:
        return

    os.makedirs(results_directory, exist_ok=True)

    x = _degradation_x(degradation_df)

    # Resistance growth plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, degradation_df["Resistance_Growth"] * 100, "r-", linewidth=2)
    ax.set_ylabel("Resistance Growth (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_directory}/battery_resistance_growth.png", dpi=300)
    plt.close()

    # Effective RTE plot
    if "Effective_RTE" in degradation_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, degradation_df["Effective_RTE"] * 100, "m-", linewidth=2)
        ax.set_ylabel("Round-Trip Efficiency (%)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_directory}/battery_effective_rte.png", dpi=300)
        plt.close()


def plot_cell_temperature(
    results_df: pd.DataFrame,
    results_directory: str,
) -> None:
    """
    Plot monthly battery cell temperature statistics (min, mean, max).

    Shows the seasonal trend of cell temperature with a shaded min-max band
    and mean line, aggregated by month.

    Args:
        results_df: Hourly results DataFrame with 'Datetime' and 'T_cell' columns.
        results_directory: Directory to save plots
    """
    if "T_cell" not in results_df.columns:
        return

    os.makedirs(results_directory, exist_ok=True)

    df = _local_time_indexed(results_df)

    # Monthly aggregation
    monthly_mean = df["T_cell"].resample("ME").mean()
    monthly_min = df["T_cell"].resample("ME").min()
    monthly_max = df["T_cell"].resample("ME").max()

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
    plt.tight_layout()
    plt.savefig(f"{results_directory}/battery_cell_temperature.png", dpi=300)
    plt.close()


def plot_timeseries(
    df: pd.DataFrame,
    columns: List[str],
    results_directory: str,
    filename: str = "timeseries.png",
) -> None:
    """
    Plot multiple columns as time series.

    Args:
        df: DataFrame with datetime index
        columns: Column names to plot
        results_directory: Directory to save plot
        filename: Output filename
    """
    os.makedirs(results_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    for col in columns:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_directory}/{filename}", dpi=300)
    plt.close()


def plot_breakeven(cost_projection: pd.DataFrame, results_directory: str, scenario_name: str = "") -> None:
    """
    Plot break-even analysis: PV system vs no system accumulated costs.

    Creates TWO separate graph files:
    1. breakeven_cumulative_{scenario}.png - Cumulative costs comparison
    2. breakeven_annual_{scenario}.png - Annual savings

    Break-even point calculated with month precision using linear interpolation.

    Args:
        cost_projection: DataFrame from cost_analysis_projection()
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
    """
    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    years = cost_projection["Year"]

    # Try NPV columns first, fall back to nominal
    if "Cost_No_Sys_Cumulative_NPV" in cost_projection.columns:
        no_sys = cost_projection["Cost_No_Sys_Cumulative_NPV"]
        with_sys = cost_projection["Cost_System_Cumulative_NPV"]
        annual_column = "Cost_System_Annual_NPV"
        label_suffix = " (NPV)"
    else:
        no_sys = cost_projection["Cost_No_Sys_Cumulative"]
        with_sys = cost_projection["Cost_System_Cumulative"]
        annual_column = "Cost_System_Annual"
        label_suffix = ""

    # The year-0 investment anchors the break-even curve, as in the library.
    initial_investment = cost_projection.attrs.get("total_investment")
    if initial_investment is None and annual_column in cost_projection.columns:
        initial_investment = float(with_sys.iloc[0] - cost_projection[annual_column].iloc[0])

    # Break-even with month precision, by the same interpolation Monte Carlo
    # and the optimizer report.
    savings = pd.DataFrame({"Year": years.to_numpy(), "Savings_Cumulative_NPV": no_sys.values - with_sys.values})
    be_year_interpolated = find_payback_year_interpolated(savings, initial_investment=initial_investment)
    be_text = "Not reached"
    if be_year_interpolated is not None:
        be_years = int(be_year_interpolated)
        be_months = int((be_year_interpolated - be_years) * 12)
        be_text = f"{be_years} years {be_months} months"

    # =========================================================================
    # GRAPH 1: Cumulative costs comparison
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(years, no_sys, "r-", linewidth=2.5, marker="o", markersize=4, label=f"No System{label_suffix}")
    ax1.plot(years, with_sys, "g-", linewidth=2.5, marker="s", markersize=4, label=f"PV System{label_suffix}")

    # Mark break-even point
    if be_year_interpolated is not None:
        # Interpolate the cost at break-even, from the year-0 investment when
        # the break-even falls inside the first year.
        curve_years, curve_cost = years.to_numpy(dtype=float), with_sys.to_numpy(dtype=float)
        if initial_investment is not None and curve_years[0] > 0.0:
            curve_years = np.concatenate(([0.0], curve_years))
            curve_cost = np.concatenate(([float(initial_investment)], curve_cost))
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

    plt.tight_layout()
    plt.savefig(f"{results_directory}/breakeven_cumulative{suffix}.png", dpi=300)
    plt.close()

    # =========================================================================
    # GRAPH 2: Annual savings
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    annual_savings = no_sys.diff().fillna(no_sys.iloc[0]) - with_sys.diff().fillna(with_sys.iloc[0])

    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in annual_savings]
    ax2.bar(years, annual_savings, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel(f"Annual Savings ({_currency(cost_projection)})", fontsize=12)
    ax2.set_xticks(years)  # Show every year
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{results_directory}/breakeven_annual{suffix}.png", dpi=300)
    plt.close()

    # Print BEP to console
    print(f"   Break-even point: {be_text}")


def _on_index_clock(value, index: pd.Index) -> pd.Timestamp:
    """Return ``value`` as a timestamp comparable with ``index``: naive dates take its zone."""
    stamp = pd.Timestamp(value)
    zone = getattr(index, "tz", None)
    if zone is not None and stamp.tz is None:
        return stamp.tz_localize(zone)
    if zone is None and stamp.tz is not None:
        return stamp.tz_localize(None)
    return stamp


def plot_battery_soh_timeseries(
    results_df: pd.DataFrame,
    results_directory: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    scenario_name: str = "",
) -> None:
    """
    Time series plot of battery State of Health (SOH) over time.

    Args:
        results_df: Energy balance results DataFrame with Battery_SOH column
        results_directory: Directory to save plots
        start_date: Optional start date filter (e.g., '2025-01-01')
        end_date: Optional end date filter (e.g., '2025-12-31')
        scenario_name: Optional suffix for filenames
    """
    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    df = _local_time_indexed(results_df)

    # Filter date range if specified. The bounds are civil dates, read on the
    # results' own clock, so a naive date compares with a tz-aware index.
    if start_date:
        df = df[df.index >= _on_index_clock(start_date, df.index)]
    if end_date:
        df = df[df.index <= _on_index_clock(end_date, df.index)]

    if "Battery_SOH" not in df.columns:
        print("Warning: Battery_SOH column not found in results")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df["Battery_SOH"], "b-", linewidth=1.5, label="SOH")

    # Add reference lines
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="Initial (100%)")
    ax.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="End of Life (80%)")

    # Fill degradation region
    ax.fill_between(df.index, df["Battery_SOH"], 100, alpha=0.2, color="red")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("State of Health (%)", fontsize=12)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(75, df["Battery_SOH"].min() - 5), 102])

    plt.tight_layout()
    plt.savefig(f"{results_directory}/battery_soh_timeseries{suffix}.png", dpi=300)
    plt.close()


def plot_monthly_comparison(results_df: pd.DataFrame, results_directory: str, scenario_name: str = "") -> None:
    """
    Compare PV production, load, import, and export by month.

    Creates a stacked/grouped bar chart showing energy flows for each month.

    Args:
        results_df: Energy balance results DataFrame
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
    """
    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    df = _local_time_indexed(results_df)

    # Monthly aggregation
    columns = ["PV_Production", "Houseload", "Import_From_Grid", "PV_AC_Export"]
    columns = [c for c in columns if c in df.columns]

    monthly = _power_frame_to_energy_kwh(df[columns], _result_instants(results_df)).resample("ME").sum()
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
    labels = {
        "PV_Production": "PV Generation",
        "Houseload": "Load Demand",
        "Import_From_Grid": "Grid Import",
        "PV_AC_Export": "Grid Export",
    }

    for i, col in enumerate(columns):
        ax.bar(
            x + i * width,
            monthly[col],
            width,
            label=labels.get(col, col),
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

    # Add value labels on top of bars
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

    plt.tight_layout()
    plt.savefig(f"{results_directory}/monthly_comparison{suffix}.png", dpi=300)
    plt.close()


def plot_monthly_balance(results_df: pd.DataFrame, results_directory: str) -> None:
    """
    Plot monthly energy balance with positive (PV, Export) and negative (Load, Import) bars.
    X-axis shows only month names (1-12).

    Args:
        results_df: Simulation results DataFrame
        results_directory: Directory to save plots
    """
    df = _local_time_indexed(results_df)

    energy_columns = ["PV_Production", "Houseload", "Import_From_Grid", "PV_AC_Export"]
    missing = [column for column in energy_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing energy-balance column(s): {', '.join(missing)}")

    # Convert power to interval energy before monthly aggregation.
    monthly = _power_frame_to_energy_kwh(df[energy_columns], _result_instants(results_df)).resample("ME").sum()

    # Group by month (1-12) to aggregate multi-year data
    monthly_avg = monthly.groupby(monthly.index.month).mean()

    # Ensure all 12 months present
    monthly_avg = monthly_avg.reindex(np.arange(1, 13), fill_value=0.0)

    months = np.arange(1, 13)
    month_names = MONTH_LABELS

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars
    width = 0.35

    # Positives
    ax.bar(months - width / 2, monthly_avg["PV_Production"], width, label="PV Production", color="gold", alpha=0.9)
    ax.bar(months + width / 2, monthly_avg["PV_AC_Export"], width, label="Grid Export", color="green", alpha=0.9)

    # Negatives (Load and Import)
    ax.bar(months - width / 2, -monthly_avg["Houseload"], width, label="Load", color="steelblue", alpha=0.9)
    ax.bar(months + width / 2, -monthly_avg["Import_From_Grid"], width, label="Grid Import", color="red", alpha=0.9)

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Energy (kWh)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", ncol=2)

    # Reduce margins
    plt.tight_layout()

    # Save
    plt.savefig(f"{results_directory}/monthly_balance.png", dpi=300)
    plt.close()


def _finite_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric values from a DataFrame column."""
    if column not in df.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(df[column], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _is_breos_montecarlo_summary(df: pd.DataFrame) -> bool:
    """Detect the one-row-per-run schema written by ``breos montecarlo``."""
    return "npv_savings" in df.columns and ("run" in df.columns or "payback_year" in df.columns)


def _plot_montecarlo_distribution(
    values: pd.Series,
    results_directory: str,
    filename: str,
    xlabel: str,
    color: str,
    suffix: str = "",
    include_zero: bool = False,
) -> bool:
    """Shared histogram with P5/P50/P95 markers for MC summary metrics."""
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return False

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
        ax.axvline(val, color="tab:red", linestyle=linestyle, linewidth=linewidth, label=f"{label}: {val:,.2f}")

    if include_zero:
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Runs", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_directory}/{filename}{suffix}.png", dpi=300)
    plt.close()
    return True


def _plot_montecarlo_payback_summary(df: pd.DataFrame, results_directory: str, suffix: str = "") -> None:
    """Plot payback distribution, CDF, and achieved/not-achieved summary."""
    total_runs = len(df)
    # The fractional year, which the distribution's 0.1-year bins and mean need;
    # older run tables carry only the integer year.
    payback = _finite_numeric_series(
        df, "payback_year_interpolated" if "payback_year_interpolated" in df.columns else "payback_year"
    )
    achieved_count = len(payback)

    if achieved_count:
        payback_values = payback.tolist()
        plot_breakeven_distribution(payback_values, total_runs, results_directory, suffix)
        plot_breakeven_cdf(payback_values, results_directory, suffix)
    plot_breakeven_summary_bar(achieved_count, total_runs, results_directory, suffix)


def plot_montecarlo_simulation(
    runs: pd.DataFrame,
    results_directory: str,
    scenario_name: str = "",
    verbose: bool = True,
) -> None:
    """
    Generate all plots for Monte Carlo simulation results.

    Args:
        runs: The one-row-per-run table written by ``breos montecarlo``.
        results_directory: Directory to save plots; they go in its ``plots``
            subdirectory.
        scenario_name: Optional suffix for filenames
        verbose: Print the output directory when plots are generated.
    """
    if not _is_breos_montecarlo_summary(runs):
        raise ValueError(
            "plot_montecarlo_simulation needs the one-row-per-run table written by breos montecarlo "
            "(an npv_savings column and a run or payback_year column)"
        )

    suffix = f"_{scenario_name}" if scenario_name else ""
    plots_folder = os.path.join(results_directory, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    plot_montecarlo_npv_distribution(runs, plots_folder, suffix)
    plot_montecarlo_grid_independence_distribution(runs, plots_folder, suffix)
    plot_montecarlo_final_soh_distribution(runs, plots_folder, suffix)
    _plot_montecarlo_distribution(
        _finite_numeric_series(runs, "lcoe_per_kwh"),
        plots_folder,
        "montecarlo_lcoe_distribution",
        f"LCOE ({_currency(runs)}/kWh)",
        "tab:purple",
        suffix,
    )
    _plot_montecarlo_payback_summary(runs, plots_folder, suffix)
    if verbose:
        print(f"Monte Carlo plots saved to: {plots_folder}")


def plot_breakeven_distribution(
    breakeven_steps: List[float], total_runs: int, results_directory: str, suffix: str = ""
) -> None:
    """
    Create histogram availability of break-even years.
    """
    if not breakeven_steps:
        print("No break-even points to plot histogram.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create histogram with 0.1 year bins
    bin_width = 0.1
    min_be = min(breakeven_steps)
    max_be = max(breakeven_steps)
    bins = np.arange(min_be - 0.05, max_be + 0.15, bin_width)

    n, bins, patches = ax.hist(breakeven_steps, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, linewidth=1)

    # Stats box
    achieved = len(breakeven_steps)
    mean_val = np.mean(breakeven_steps)
    median_val = np.median(breakeven_steps)
    std_val = np.std(breakeven_steps)

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

    ax.set_xlabel("Break-even Year", fontsize=12)
    ax.set_ylabel("Number of Runs", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i in range(len(n)):
        if n[i] > 0:
            ax.text((bins[i] + bins[i + 1]) / 2, n[i] + 0.1, int(n[i]), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{results_directory}/breakeven_histogram{suffix}.png", dpi=300)
    plt.close()


def plot_breakeven_cdf(breakeven_steps: List[float], results_directory: str, suffix: str = "") -> None:
    """
    Plot Cumulative Distribution Function of break-even years.
    """
    if not breakeven_steps:
        return

    x = np.sort(breakeven_steps)
    n = len(x)
    y = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.step(x, y, where="post", color="blue", linewidth=2, label="CDF")

    # Quantiles
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    colors = ["red", "gray", "black", "gray", "red"]

    for q, color in zip(quantiles, colors, strict=True):
        val = np.quantile(x, q)
        ax.axvline(val, color=color, linestyle="--", alpha=0.6, linewidth=1)
        ax.scatter([val], [q], color=color, zorder=5)
        ax.text(val, q, f" {q:.1%} ({val:.1f}y)", color=color, ha="left", va="bottom", fontsize=9)

    ax.set_xlabel("Break-even Year", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"{results_directory}/breakeven_cdf{suffix}.png", dpi=300)
    plt.close()


def plot_breakeven_summary_bar(achieved_count: int, total_runs: int, results_directory: str, suffix: str = "") -> None:
    """
    Bar chart of Success vs Failure for break-even.
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

    plt.tight_layout()
    plt.savefig(f"{results_directory}/breakeven_summary_bar{suffix}.png", dpi=300)
    plt.close()


def plot_montecarlo_npv_distribution(all_results_df: pd.DataFrame, results_directory: str, suffix: str = "") -> None:
    """
    Histogram of NPV savings (``npv_savings``) across one-row-per-run MC results.
    """
    _plot_montecarlo_distribution(
        _finite_numeric_series(all_results_df, "npv_savings"),
        results_directory,
        "montecarlo_npv_distribution",
        f"NPV Savings ({_currency(all_results_df)})",
        "tab:blue",
        suffix,
        include_zero=True,
    )


def plot_montecarlo_grid_independence_distribution(
    all_results_df: pd.DataFrame, results_directory: str, suffix: str = ""
) -> None:
    """
    Histogram of mean grid independence (``mean_grid_independence_pct``) across one-row-per-run MC results.
    """
    _plot_montecarlo_distribution(
        _finite_numeric_series(all_results_df, "mean_grid_independence_pct"),
        results_directory,
        "montecarlo_grid_independence_distribution",
        "Mean Grid Independence (%)",
        "tab:green",
        suffix,
    )


def plot_montecarlo_final_soh_distribution(
    all_results_df: pd.DataFrame, results_directory: str, suffix: str = ""
) -> None:
    """
    Histogram of final battery state-of-health across one-row-per-run MC results.
    """
    _plot_montecarlo_distribution(
        _finite_numeric_series(all_results_df, "final_soh_pct"),
        results_directory,
        "montecarlo_final_soh_distribution",
        "Final Battery SOH (%)",
        "tab:cyan",
        suffix,
    )


# =========================================================================
# TMY VS HISTORICAL WEATHER
# =========================================================================

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
        years = preload_weather_by_year(os.fspath(historical))
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
    results_directory: str,
    variable: str = "ghi",
    tmy_label: str = "TMY",
    filename: Optional[str] = None,
) -> None:
    """
    Compare a TMY with historical weather years, month by month.

    Draws the TMY, the historical mean with its 95% confidence interval, and
    the monthly minimum and maximum: each month's lowest and highest value
    over the historical years, which can come from different years.
    Irradiance is the monthly total in kWh/m²; temperature is the monthly
    mean in °C. A month with missing values warns, and its total skips them.

    Args:
        tmy: One weather year, as :func:`breos.weather.load_weather` or
            :func:`breos.weather.fetch_tmy_weather_data` return it.
        historical: The multi-year weather CSV that a Monte Carlo study
            samples (``MonteCarloSettings.weather_file``), or the per-year
            frames :func:`breos.weather.preload_weather_by_year` splits it
            into. Only complete years count, as in the study.
        results_directory: Directory to save the plot.
        variable: ``"ghi"``, ``"dni"``, ``"dhi"`` or ``"temp_air"``.
        tmy_label: Legend label of the TMY line, for example its source.
        filename: Output filename. Defaults to ``weather_monthly_<variable>.png``.
    """
    name, unit = _WEATHER_VARIABLES.get(variable, (variable, ""))
    tmy_vals = _weather_monthly(tmy, variable, "The TMY").to_numpy()
    historical_years = _historical_weather_years(historical)
    monthly_stats = _weather_monthly_stats(historical_years, variable)
    os.makedirs(results_directory, exist_ok=True)

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

    # 95% CI shaded band
    ax.fill_between(x, hist_ci_low, hist_ci_high, color="steelblue", alpha=0.25, label="95% CI of the mean")

    # Historical mean line
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

    # TMY line (on top)
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
    fig.savefig(
        os.path.join(results_directory, filename or f"weather_monthly_{variable}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def plot_weather_annual_ghi_distribution(
    tmy: pd.DataFrame,
    historical: Union[str, "os.PathLike[str]", Mapping[int, pd.DataFrame]],
    results_directory: str,
    tmy_label: str = "TMY",
    filename: str = "annual_ghi_distribution.png",
) -> None:
    """
    Histogram of the annual GHI of historical weather years, with the TMY and the historical mean.

    Args:
        tmy: One weather year, as :func:`breos.weather.load_weather` or
            :func:`breos.weather.fetch_tmy_weather_data` return it.
        historical: The multi-year weather CSV that a Monte Carlo study
            samples (``MonteCarloSettings.weather_file``), or the per-year
            frames :func:`breos.weather.preload_weather_by_year` splits it
            into. Only complete years count, as in the study.
        results_directory: Directory to save the plot.
        tmy_label: Legend label of the TMY line, for example its source.
        filename: Output filename.

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
    os.makedirs(results_directory, exist_ok=True)

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
    fig.savefig(os.path.join(results_directory, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    results_directory: str,
    colors: Optional[Sequence[str]] = None,
    currency: Optional[str] = None,
    filename: str = "breakeven_comparison.png",
) -> None:
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
        results_directory: Output directory.
        colors: Line colour for each scenario. Defaults to the colour cycle.
        currency: Currency code for the money axis, for projections read
            from CSV, which do not record it. An App result and a projection
            frame from :func:`~breos.economics.cost_analysis_projection`
            record their own. When no projection records it and ``currency``
            is None, the axis shows the amounts without a currency code.
        filename: Output filename.

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
    os.makedirs(results_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Year 0 costs the investment with a system and nothing without one.
    curves = []
    for df in cost_dfs:
        years = df["Year"].to_numpy(dtype=float)
        no_sys = df["Cost_No_Sys_Cumulative_NPV"].to_numpy(dtype=float)
        with_sys = df["Cost_System_Cumulative_NPV"].to_numpy(dtype=float)
        investment = _initial_investment(df)
        if investment is not None and len(years) and years[0] > 0.0:
            years = np.concatenate(([0.0], years))
            no_sys = np.concatenate(([0.0], no_sys))
            with_sys = np.concatenate(([investment], with_sys))
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
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}{unit}"))
    ax.set_xticks(range(0, max_year + 1))
    ax.set_xlim(0, max_year + 0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# SWEEP RESULTS
# =========================================================================

# Axis labels of the sweep and optimizer columns BREOS writes, without their
# ``param_`` or ``resolved_`` prefix. Other columns are labelled by name.
_COLUMN_LABELS = {
    "n_modules": "Number of PV modules",
    "battery_kwh": "Battery capacity (kWh)",
    "pv_kwp": "PV size (kWp)",
    "tilt": "Tilt (°)",
    "azimuth": "Azimuth (°)",
    "grid_independence_pct": "Grid independence (%)",
    "self_consumption_pct": "Self-consumption (%)",
    "pv_production_kwh": "PV production (kWh)",
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
}

_COMPASS_POINTS = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}


def _column_label(column: str, currency: Optional[str]) -> str:
    """Axis label of a sweep or optimizer column.

    Money is labelled in ``currency``; with None, the label names no currency
    ("NPV savings", "LCOE (per kWh)").
    """
    name = str(column)
    for prefix in ("param_", "resolved_"):
        name = name.removeprefix(prefix)
    label = _COLUMN_LABELS.get(name)
    if label is None:
        return name
    if currency is None:
        return label.replace(" ({currency})", "").replace("{currency}/kWh", "per kWh")
    return label.format(currency=currency)


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
    results_directory: str,
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
) -> None:
    """
    Heatmap of one metric of a two-parameter ``breos sweep`` result.

    Each cell is one run, annotated with its value when the grid has at most
    15 rows and 15 columns. With ``diff``, each cell is ``sweep`` minus
    ``diff``, for example the same sizing grid at two locations or two
    tariffs. Cells in only one of the two sweeps stay blank. A difference of
    a percentage, such as grid independence, is labelled in percentage points.

    A difference, and a metric with both gains and losses such as
    ``npv_savings``, are drawn on a diverging scale centred on zero and
    symmetric about it. Other metrics get a sequential scale.

    Args:
        sweep: The CSV ``breos sweep`` writes, or its DataFrame.
        metric: The result column to draw, such as ``grid_independence_pct``
            or ``npv_savings``.
        results_directory: Directory to save the plot.
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
            ``vmax`` turns a metric with both signs to the sequential scale.
            With ``diff`` the scale is always symmetric about zero, and
            ``vmin`` and ``vmax`` are not read.
        vmax: Colour-scale maximum (auto if None).
        currency: Currency code for money labels, for a sweep CSV, which does
            not record it. A DataFrame can record it in ``attrs["currency"]``.
            When neither names it, money labels show no currency code.
        filename: Output filename. Defaults to ``sweep_<metric>.png``, or
            ``sweep_<metric>_diff.png`` with ``diff``.

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
        label_currency = _label_currency([frame], currency)
    else:
        other = _read_table(diff)
        label_currency = _label_currency([frame, other], currency)
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
    # Zero-centred when the cells are differences, or gains and losses.
    signed = finite.min() < 0 < finite.max() and vmin is None and vmax is None
    if diff is not None or signed:
        abs_max = float(np.abs(finite).max()) or 1.0
        norm: Normalize = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        cmap = cmap or "RdBu"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        norm.autoscale_None(finite)
        cmap = cmap or "YlGnBu"
    os.makedirs(results_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, origin="lower")

    # Axis labels from pivot index/columns
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
                        format(val, number_format),
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
    fig.savefig(os.path.join(results_directory, filename or default_name), dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    results_directory: str,
    tilt: str = "tilt",
    azimuth: str = "azimuth",
    maximize: bool = True,
    metric_label: Optional[str] = None,
    currency: Optional[str] = None,
    filename: str = "orientation_landscape.png",
) -> None:
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
        metric: The result column to draw, such as ``pv_production_kwh`` or
            ``npv_savings``.
        results_directory: Directory to save the plot.
        tilt: The tilt parameter, as its sweep key or column.
        azimuth: The azimuth parameter, as its sweep key or column. A table
            without it, or with one azimuth only, is a tilt sweep.
        maximize: True if a higher metric is better; False for a metric such
            as ``lcoe_per_kwh``.
        metric_label: Axis and colour-bar label. Defaults to a label for the metric.
        currency: Currency code for a money metric, for a sweep CSV, which
            does not record it. When neither it nor ``attrs["currency"]``
            names it, the label shows no currency code.
        filename: Output filename.

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
    label = metric_label or _column_label(metric, _label_currency([frame], currency))
    pick = np.nanargmax if maximize else np.nanargmin

    if azimuth_column is None:
        if frame[tilt_column].duplicated().any():
            raise ValueError(f"The sweep has more than one row for some {tilt_column}; select one value of the others")
        profile = frame.set_index(tilt_column)[metric].astype(float).sort_index()
        best = int(pick(profile.to_numpy()))
        os.makedirs(results_directory, exist_ok=True)

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
        fig.savefig(os.path.join(results_directory, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    grid = _sweep_grid(frame, azimuth_column, tilt_column, metric)
    row, col = np.unravel_index(int(pick(grid.to_numpy())), grid.shape)
    best_tilt, best_azimuth = float(grid.index[row]), float(grid.columns[col])
    azimuths = grid.columns.to_numpy(dtype=float)
    optimum = (
        f"Optimum: {best_tilt:g}° tilt"
        if best_tilt == 0
        else f"Optimum: {best_tilt:g}° tilt, {best_azimuth:g}° azimuth"
    )
    os.makedirs(results_directory, exist_ok=True)

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
    fig.savefig(os.path.join(results_directory, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# PARETO ANALYSIS
# =========================================================================


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
    results_directory: str,
    x: str = "Grid_Independence_%",
    y: str = "NPV",
    maximize: Tuple[bool, bool] = (True, True),
    color_by: Optional[str] = None,
    currency: Optional[str] = None,
    filename: str = "pareto_front.png",
) -> None:
    """
    Scatter two objectives of a set of designs and draw their Pareto front.

    The designs no other design beats in both objectives are drawn large and
    joined by a line; the dominated ones are drawn small and grey. On the
    optimizer's front with its own two objectives, every design is on it.

    Args:
        designs: The :class:`~breos.optimization.OptimizationResult` of
            :func:`~breos.optimization.optimize_system_multi_objective`, its
            ``details["pareto"]`` frame or that frame's CSV, or any table of
            designs, such as a ``breos sweep`` CSV.
        results_directory: Directory to save the plot.
        x: Column of the objective on the horizontal axis, or a swept key.
        y: Column of the objective on the vertical axis, or a swept key.
            The defaults are the optimizer's two objectives.
        maximize: For ``x`` and ``y``, True if higher is better; False for a
            cost such as ``Projected_Initial_Cost`` or ``lcoe_per_kwh``.
        color_by: Column that colours the front, such as ``Battery_kWh``,
            with a colour bar. None draws it in one colour.
        currency: Currency code for money labels, for a CSV, which does not
            record it. The optimizer's frame and a DataFrame with
            ``attrs["currency"]`` record their own. When neither names it,
            money labels show no currency code.
        filename: Output filename.

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
    label_currency = _label_currency([frame], currency)
    x_values = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not finite.any():
        raise ValueError(f"No design has finite {x_column} and {y_column} values")
    frame, x_values, y_values = frame[finite], x_values[finite], y_values[finite]
    signs = [1.0 if higher else -1.0 for higher in maximize]
    front = _pareto_mask(signs[0] * x_values, signs[1] * y_values)
    os.makedirs(results_directory, exist_ok=True)

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
            label=front_label,
        )
        fig.colorbar(points, ax=ax, label=_column_label(color_column, label_currency))
    ax.set_xlabel(_column_label(x_column, label_currency))
    ax.set_ylabel(_column_label(y_column, label_currency))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_co2_savings(
    cost_projection: pd.DataFrame,
    results_directory: str,
    scenario_name: str = "",
) -> None:
    """
    Plot CO2 emissions avoided over system lifetime.

    Creates two figures: co2_avoided_yearly_{scenario}.png, with yearly CO2
    avoided (total and self-consumed) as bars, and
    co2_avoided_cumulative_{scenario}.png, with the cumulative totals as lines.

    Args:
        cost_projection: DataFrame from cost_analysis_projection() with CO2 columns
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
    """
    if "CO2_Avoided_Total_kg" not in cost_projection.columns:
        return

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    years = cost_projection["Year"]
    co2_total = cost_projection["CO2_Avoided_Total_kg"]
    co2_self = cost_projection["CO2_Avoided_SelfConsumed_kg"]
    co2_total_cum = cost_projection["CO2_Avoided_Total_Cumulative_kg"]
    co2_self_cum = cost_projection["CO2_Avoided_SelfConsumed_Cumulative_kg"]

    bar_width = 0.35

    # =========================================================================
    # GRAPH 1: Yearly CO2 avoided (bars)
    # =========================================================================
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
    fig.savefig(os.path.join(results_directory, f"co2_avoided_yearly{suffix}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # GRAPH 2: Cumulative CO2 avoided (lines)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(years, co2_total_cum / 1000, "b-", linewidth=2.5, marker="o", markersize=4, label="Total PV Production")
    ax.plot(years, co2_self_cum / 1000, "g-", linewidth=2.5, marker="s", markersize=4, label="Self-Consumed PV")
    ax.fill_between(years, 0, co2_self_cum / 1000, alpha=0.15, color="green")
    ax.fill_between(years, co2_self_cum / 1000, co2_total_cum / 1000, alpha=0.10, color="blue")

    # Annotate final values
    final_total = co2_total_cum.iloc[-1] / 1000
    final_self = co2_self_cum.iloc[-1] / 1000
    ax.annotate(
        f"{final_total:,.1f} t",
        xy=(years.iloc[-1], final_total),
        xytext=(-50, 10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="#1565C0",
    )
    ax.annotate(
        f"{final_self:,.1f} t",
        xy=(years.iloc[-1], final_self),
        xytext=(-50, -20),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="#2E7D32",
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative CO$_2$ Avoided (t CO$_2$eq)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, f"co2_avoided_cumulative{suffix}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
