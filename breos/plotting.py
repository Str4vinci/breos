"""
Plotting and visualization module.

This module provides visualization functions for:
- Cost projections
- Energy balance results
- Monthly/yearly/weekly analysis
- Battery degradation
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Plotting imports with backend handling
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: uv add matplotlib")


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
    _check_matplotlib()

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
        matplotlib.use("Agg")


def create_cost_plots(
    cost_projection: pd.DataFrame, total_initial_cost: float, results_directory: str, scenario_name: str = ""
) -> None:
    """
    Create cost projection visualization.

    Args:
        cost_projection: DataFrame from cost_analysis_projection()
        total_initial_cost: Total investment cost
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cumulative costs
    if "Cost_No_Sys_Cumulative_NPV" in cost_projection.columns:
        ax.plot(
            cost_projection["Year"],
            cost_projection["Cost_No_Sys_Cumulative_NPV"],
            "r--",
            label="No System (NPV)",
            linewidth=2,
        )
        ax.plot(
            cost_projection["Year"],
            cost_projection["Cost_System_Cumulative_NPV"],
            "g-",
            label="With PV System (NPV)",
            linewidth=2,
        )
    else:
        ax.plot(
            cost_projection["Year"], cost_projection["Cost_No_Sys_Cumulative"], "r--", label="No System", linewidth=2
        )
        ax.plot(
            cost_projection["Year"],
            cost_projection["Cost_System_Cumulative"],
            "g-",
            label="With PV System",
            linewidth=2,
        )

    # Find and mark payback
    if "Savings_Cumulative_NPV" in cost_projection.columns:
        payback = cost_projection[cost_projection["Savings_Cumulative_NPV"] > 0]
        if not payback.empty:
            payback_year = payback["Year"].iloc[0]
            ax.axvline(x=payback_year, color="blue", linestyle=":", alpha=0.7)
            ax.annotate(
                f"Payback: Year {payback_year}", xy=(payback_year, ax.get_ylim()[1] * 0.9), fontsize=10, color="blue"
            )

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Cost (€)")
    # ax.set_title('Cost Comparison: With vs Without PV System')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_directory}/cost_projection{suffix}.png", dpi=300)
    plt.close()


def monthly_graphs(results_df: pd.DataFrame, results_directory: str, columns: Optional[List[str]] = None) -> None:
    """
    Create monthly aggregated bar charts.

    Args:
        results_df: Energy balance results DataFrame
        results_directory: Directory to save plots
        columns: Columns to plot (default: PV, Load, Import, Sell)
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)

    df = results_df.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    if columns is None:
        columns = ["PV_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid"]

    # Filter to available columns
    columns = [c for c in columns if c in df.columns]

    # Monthly aggregation
    monthly = df[columns].resample("ME").sum() / 1000  # Convert to kWh

    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(monthly))
    width = 0.2

    colors = ["gold", "steelblue", "coral", "lightgreen"]
    labels = ["PV Production", "Load", "Grid Import", "Grid Export"]

    for i, (col, color, label) in enumerate(zip(columns, colors, labels)):
        if col in monthly.columns:
            ax.bar([xi + i * width for xi in x], monthly[col], width, label=label, color=color, alpha=0.8)

    ax.set_xticks([xi + width * (len(columns) - 1) / 2 for xi in x])
    ax.set_xticklabels([d.strftime("%b %Y") for d in monthly.index], rotation=45)
    ax.set_ylabel("Energy (kWh)")
    # ax.set_title('Monthly Energy Summary')
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{results_directory}/monthly_energy.png", dpi=300)
    plt.close()


def yearly_graphs(results_df: pd.DataFrame, results_directory: str) -> None:
    """
    Create yearly aggregated summary.

    Args:
        results_df: Energy balance results DataFrame
        results_directory: Directory to save plots
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)

    df = results_df.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    columns = ["PV_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid"]
    columns = [c for c in columns if c in df.columns]

    yearly = df[columns].resample("Y").sum() / 1000

    fig, ax = plt.subplots(figsize=(10, 6))

    yearly.plot(kind="bar", ax=ax, width=0.8, alpha=0.8)

    ax.set_xticklabels([d.strftime("%Y") for d in yearly.index], rotation=0)
    ax.set_ylabel("Energy (kWh)")
    # ax.set_title('Yearly Energy Summary')
    ax.legend(["PV Production", "Load", "Grid Import", "Grid Export"])
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
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)

    df = results_df.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

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


def degradation_plots(degradation_df: pd.DataFrame, results_directory: str) -> None:
    """
    Create battery degradation visualization.
    Generates separate plots for SOH, Components, FEC, and SOC.

    Args:
        degradation_df: Degradation tracking DataFrame
        results_directory: Directory to save plots
    """
    _check_matplotlib()

    if degradation_df.empty:
        print("No degradation data to plot")
        return

    os.makedirs(results_directory, exist_ok=True)

    # Determine x-axis: use sequential days for multi-year propagation
    # (TMY data repeats the same dates each year, so we need to adjust)
    if "Year" in degradation_df.columns and degradation_df["Year"].nunique() > 1:
        # Multi-year: use day index (sequential)
        x = np.arange(len(degradation_df))  # Day index (0, 1, 2, ...)
        x_label = "Days"
        x_years = x / 365.0  # Convert to years for tick labels
        use_years_axis = True
    elif "Datetime" in degradation_df.columns:
        x = pd.to_datetime(degradation_df["Datetime"])
        x_label = None  # Use default date formatting
        use_years_axis = False
    else:
        x = degradation_df.index
        x_label = None
        use_years_axis = False

    # 1. SOH over time
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, degradation_df["SOH"], "b-", linewidth=2)
    ax1.set_ylabel("SOH (%)")
    if use_years_axis:
        ax1.set_xlabel("Year")
        # Set ticks at each year
        max_years = int(x_years.max()) + 1
        ax1.set_xticks([y * 365 for y in range(max_years + 1)])
        ax1.set_xticklabels([str(y) for y in range(max_years + 1)])
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_directory}/battery_degradation_soh.png", dpi=300)
    plt.close()

    # 2. Degradation Components (Global / Total Lifespan and Per-Battery / Resetting)
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
        if use_years_axis:
            ax.set_xlabel("Year")
            max_years = int(x_years.max()) + 1
            ax.set_xticks([y * 365 for y in range(max_years + 1)])
            ax.set_xticklabels([str(y) for y in range(max_years + 1)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_directory, filename), dpi=300)
        plt.close()

    # Generate Global (accumulated) plot
    if "Global_Cycle_Degradation" in degradation_df.columns:
        glob_cyc = degradation_df["Global_Cycle_Degradation"] * 100
        glob_cal = degradation_df.get("Global_Calendar_Degradation", 0) * 100
        _plot_degradation_components(
            glob_cyc, glob_cal, "battery_degradation_components_global.png", "Global Cumulative Degradation (%)"
        )

    # Generate Per-Battery (resetting) plot
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
        if use_years_axis:
            ax3.set_xlabel("Year")
            max_years = int(x_years.max()) + 1
            ax3.set_xticks([y * 365 for y in range(max_years + 1)])
            ax3.set_xticklabels([str(y) for y in range(max_years + 1)])
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
    _check_matplotlib()

    if degradation_df.empty or "Resistance_Growth" not in degradation_df.columns:
        return

    os.makedirs(results_directory, exist_ok=True)

    # Determine x-axis
    if "Year" in degradation_df.columns and degradation_df["Year"].nunique() > 1:
        x = np.arange(len(degradation_df))
        x_years = x / 365.0
        use_years_axis = True
    elif "Datetime" in degradation_df.columns:
        x = pd.to_datetime(degradation_df["Datetime"])
        use_years_axis = False
    else:
        x = degradation_df.index
        use_years_axis = False

    def _set_year_ticks(ax, x_years):
        max_years = int(x_years.max()) + 1
        ax.set_xticks([y * 365 for y in range(max_years + 1)])
        ax.set_xticklabels([str(y) for y in range(max_years + 1)])
        ax.set_xlabel("Year")

    # Resistance growth plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, degradation_df["Resistance_Growth"] * 100, "r-", linewidth=2)
    ax.set_ylabel("Resistance Growth (%)")
    if use_years_axis:
        _set_year_ticks(ax, x_years)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_directory}/battery_resistance_growth.png", dpi=300)
    plt.close()

    # Effective RTE plot
    if "Effective_RTE" in degradation_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, degradation_df["Effective_RTE"] * 100, "m-", linewidth=2)
        ax.set_ylabel("Round-Trip Efficiency (%)")
        if use_years_axis:
            _set_year_ticks(ax, x_years)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_directory}/battery_effective_rte.png", dpi=300)
        plt.close()


def plot_validation_soh_comparison(
    measured_soh: "pd.Series",
    predicted_soh: "pd.Series",
    results_directory: str,
    x_label: str = "Time",
    metrics: Optional[dict] = None,
) -> None:
    """
    Plot measured vs predicted SOH for degradation model validation.

    Measured data is shown as scatter points, predicted as a line.
    Optionally annotates RMSE on the plot.

    Args:
        measured_soh: Series indexed by x-axis values (cycles, days, etc.) with measured SOH
        predicted_soh: Series indexed by same x-axis values with predicted SOH
        results_directory: Directory to save plot
        x_label: Label for x-axis
        metrics: Optional dict with 'RMSE', 'MAE', etc. to annotate
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(measured_soh.index, measured_soh.values, s=15, alpha=0.6, color="b", label="Measured", zorder=5)
    ax.plot(predicted_soh.index, predicted_soh.values, "r-", linewidth=2, label="Predicted")

    ax.set_xlabel(x_label)
    ax.set_ylabel("SOH")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if metrics:
        text_parts = []
        if "RMSE" in metrics:
            text_parts.append(f"RMSE = {metrics['RMSE']:.4f}")
        if "MAE" in metrics:
            text_parts.append(f"MAE = {metrics['MAE']:.4f}")
        if "R2" in metrics:
            text_parts.append(f"R\u00b2 = {metrics['R2']:.4f}")
        if text_parts:
            ax.text(
                0.02,
                0.02,
                "\n".join(text_parts),
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    plt.tight_layout()
    plt.savefig(f"{results_directory}/validation_soh_comparison.png", dpi=300)
    plt.close()


def plot_validation_residuals(
    measured_soh: "pd.Series",
    predicted_soh: "pd.Series",
    results_directory: str,
    x_label: str = "Time",
) -> None:
    """
    Plot residuals (measured - predicted) over time.

    Args:
        measured_soh: Measured SOH series
        predicted_soh: Predicted SOH series (must share same index)
        results_directory: Directory to save plot
        x_label: Label for x-axis
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    residuals = (measured_soh - predicted_soh) * 100  # to percentage points

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(residuals.index, residuals.values, s=10, alpha=0.5, color="steelblue")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Residual (measured - predicted) [pp]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_directory}/validation_residuals.png", dpi=300)
    plt.close()


def plot_validation_parity(
    measured_soh: "pd.Series",
    predicted_soh: "pd.Series",
    results_directory: str,
    metrics: Optional[dict] = None,
) -> None:
    """
    Plot parity (predicted vs measured) with 1:1 line.

    Args:
        measured_soh: Measured SOH values
        predicted_soh: Predicted SOH values (same length)
        results_directory: Directory to save plot
        metrics: Optional dict with R2 etc. to annotate
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(measured_soh.values, predicted_soh.values, s=15, alpha=0.5, color="steelblue")

    # 1:1 line
    lims = [
        min(measured_soh.min(), predicted_soh.min()) - 0.01,
        max(measured_soh.max(), predicted_soh.max()) + 0.01,
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Measured SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if metrics and "R2" in metrics:
        ax.text(
            0.05,
            0.95,
            f"R\u00b2 = {metrics['R2']:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(f"{results_directory}/validation_parity.png", dpi=300)
    plt.close()


def plot_validation_multi_system(
    systems_results: dict,
    results_directory: str,
) -> None:
    """
    Plot measured vs predicted SOH for multiple systems on one figure.

    Each system gets a unique color. Measured SOH shown as markers,
    predicted SOH as lines.

    Args:
        systems_results: Dict keyed by system_id, each value a dict with:
            'simulation': DataFrame with 'date' and 'predicted_soh'
            'truth': DataFrame with 'date' and 'measured_soh'
            'metrics': dict with RMSE, R2, etc.
        results_directory: Directory to save plot
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    n_systems = len(systems_results)
    if n_systems == 0:
        return

    cmap = plt.cm.get_cmap("tab20", max(n_systems, 2))

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (sid, res) in enumerate(sorted(systems_results.items())):
        color = cmap(i)
        sim = res["simulation"]
        truth = res["truth"]
        label = f"System {sid}"

        # Predicted as line
        ax.plot(sim["date"], sim["predicted_soh"], "-", color=color, linewidth=1.2, alpha=0.7)

        # Measured as markers
        if not truth.empty:
            ax.scatter(
                truth["date"],
                truth["measured_soh"],
                color=color,
                s=40,
                marker="o",
                edgecolors="k",
                linewidths=0.5,
                label=label,
                zorder=5,
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("SOH")
    ax.set_ylim(0.6, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=9)

    # Format x-axis as years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "validation_multi_system.png"), dpi=300)
    plt.close()


def plot_validation_degradation_split(
    simulation_df: "pd.DataFrame",
    results_directory: str,
    system_label: str = "",
) -> None:
    """
    Plot calendar vs cycle aging contribution over time as stacked area.

    Args:
        simulation_df: DataFrame with columns 'date', 'cal_loss', 'cycle_loss'
        results_directory: Directory to save plot
        system_label: Optional label for filename suffix
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    dates = simulation_df["date"]
    cal_loss = simulation_df["cal_loss"] * 100  # to percentage points
    cycle_loss = simulation_df["cycle_loss"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(dates, 0, cal_loss, alpha=0.6, color="#2196F3", label="Calendar aging")
    ax.fill_between(dates, cal_loss, cal_loss + cycle_loss, alpha=0.6, color="#FF5722", label="Cycle aging")

    ax.set_xlabel("Date")
    ax.set_ylabel("Capacity loss [pp]")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    plt.tight_layout()
    suffix = f"_{system_label}" if system_label else ""
    plt.savefig(os.path.join(results_directory, f"validation_degradation_split{suffix}.png"), dpi=300)
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
    _check_matplotlib()

    if "T_cell" not in results_df.columns:
        return

    os.makedirs(results_directory, exist_ok=True)

    df = results_df.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")

    # Monthly aggregation
    monthly_mean = df["T_cell"].resample("ME").mean()
    monthly_min = df["T_cell"].resample("ME").min()
    monthly_max = df["T_cell"].resample("ME").max()

    # Group by month number (handles multi-year data)
    mean_by_month = monthly_mean.groupby(monthly_mean.index.month).mean()
    min_by_month = monthly_min.groupby(monthly_min.index.month).min()
    max_by_month = monthly_max.groupby(monthly_max.index.month).max()

    # Ensure all 12 months
    months = np.arange(1, 13)
    mean_by_month = mean_by_month.reindex(months, fill_value=0.0)
    min_by_month = min_by_month.reindex(months, fill_value=0.0)
    max_by_month = max_by_month.reindex(months, fill_value=0.0)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

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
    title: str = "Time Series",
) -> None:
    """
    Plot multiple columns as time series.

    Args:
        df: DataFrame with datetime index
        columns: Column names to plot
        results_directory: Directory to save plot
        filename: Output filename
        title: Plot title
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    for col in columns:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    # ax.set_title(title)
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
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    years = cost_projection["Year"]

    # Try NPV columns first, fall back to nominal
    if "Cost_No_Sys_Cumulative_NPV" in cost_projection.columns:
        no_sys = cost_projection["Cost_No_Sys_Cumulative_NPV"]
        with_sys = cost_projection["Cost_System_Cumulative_NPV"]
        label_suffix = " (NPV)"
    else:
        no_sys = cost_projection["Cost_No_Sys_Cumulative"]
        with_sys = cost_projection["Cost_System_Cumulative"]
        label_suffix = ""

    # Calculate break-even point with MONTH precision using linear interpolation
    savings = no_sys.values - with_sys.values
    breakeven_idx = np.where(savings > 0)[0]

    be_years = None
    be_months = None
    be_text = "Not reached"
    be_year_exact = None

    if len(breakeven_idx) > 0:
        idx = breakeven_idx[0]
        if idx > 0:
            # Linear interpolation between year (idx-1) and year (idx)
            y0, y1 = savings[idx - 1], savings[idx]
            x0, x1 = years.iloc[idx - 1], years.iloc[idx]
            # Find where savings crosses zero
            be_year_exact = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
            be_years = int(be_year_exact)
            be_months = int((be_year_exact - be_years) * 12)
            be_text = f"{be_years} years {be_months} months"
        else:
            be_year_exact = years.iloc[0]
            be_years = int(be_year_exact)
            be_months = 0
            be_text = f"{be_years} years 0 months"

    # =========================================================================
    # GRAPH 1: Cumulative costs comparison
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(years, no_sys, "r-", linewidth=2.5, marker="o", markersize=4, label=f"No System{label_suffix}")
    ax1.plot(years, with_sys, "g-", linewidth=2.5, marker="s", markersize=4, label=f"PV System{label_suffix}")

    # Mark break-even point
    if be_year_exact is not None:
        # Interpolate the cost at break-even
        be_cost = np.interp(be_year_exact, years, with_sys)
        ax1.axvline(x=be_year_exact, color="blue", linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.scatter([be_year_exact], [be_cost], s=120, c="blue", zorder=5, edgecolors="white", linewidth=2)
        ax1.annotate(
            f"Break-even\n{be_text}",
            xy=(be_year_exact, be_cost),
            xytext=(be_year_exact + 1.5, be_cost * 0.85),
            fontsize=11,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        )

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Cumulative Cost (€)", fontsize=12)
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

    if "Savings_Annual_NPV" in cost_projection.columns:
        annual_savings = cost_projection["Savings_Annual_NPV"]
    elif "Savings_Annual" in cost_projection.columns:
        annual_savings = cost_projection["Savings_Annual"]
    else:
        annual_savings = no_sys.diff().fillna(no_sys.iloc[0]) - with_sys.diff().fillna(with_sys.iloc[0])

    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in annual_savings]
    ax2.bar(years, annual_savings, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Annual Savings (€)", fontsize=12)
    ax2.set_xticks(years)  # Show every year
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{results_directory}/breakeven_annual{suffix}.png", dpi=300)
    plt.close()

    # Print BEP to console
    print(f"   Break-even point: {be_text}")


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
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    df = results_df.copy()

    # Ensure datetime index
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    # Filter date range if specified
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

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
    # ax.set_title('Battery State of Health Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(75, df["Battery_SOH"].min() - 5), 102])

    plt.tight_layout()
    plt.savefig(f"{results_directory}/battery_soh_timeseries{suffix}.png", dpi=300)
    plt.close()


def plot_slope_optimization(
    slope_results: pd.DataFrame,
    results_directory: str,
    scenario_name: str = "",
    x_col: str = "Slope",
    y_col: str = "Total_PV_Production_kWh",
    optimal_marker: bool = True,
) -> None:
    """
    Scatter plot of slope optimization results.

    Args:
        slope_results: DataFrame with slope optimization results
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
        x_col: Column for x-axis (default: 'Slope')
        y_col: Column for y-axis (default: 'Total_PV_Production_kWh')
        optimal_marker: Whether to highlight optimal point
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    if x_col not in slope_results.columns or y_col not in slope_results.columns:
        print(f"Warning: Required columns {x_col} or {y_col} not found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = slope_results[x_col]
    y = slope_results[y_col]

    # Scatter plot with color based on y value
    scatter = ax.scatter(x, y, c=y, cmap="viridis", s=80, alpha=0.8, edgecolor="black")
    ax.plot(x, y, "k-", alpha=0.3, linewidth=1)

    # Mark optimal point
    if optimal_marker:
        optimal_idx = y.idxmax()
        optimal_x = x.loc[optimal_idx]
        optimal_y = y.loc[optimal_idx]
        ax.scatter(
            [optimal_x],
            [optimal_y],
            s=200,
            c="red",
            marker="*",
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
            label=f"Optimal: {optimal_x}°",
        )
        ax.annotate(
            f"{optimal_y:.1f} kWh",
            xy=(optimal_x, optimal_y),
            xytext=(optimal_x + 2, optimal_y + optimal_y * 0.02),
            fontsize=10,
            fontweight="bold",
        )

    plt.colorbar(scatter, ax=ax, label=y_col)

    ax.set_xlabel(f"{x_col} (°)", fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    # ax.set_title('Slope Optimization Results', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_directory}/slope_optimization{suffix}.png", dpi=300)
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
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    df = results_df.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    # Monthly aggregation
    columns = ["PV_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid"]
    columns = [c for c in columns if c in df.columns]

    monthly = df[columns].resample("ME").sum() / 1000  # kWh
    monthly["Month"] = monthly.index.strftime("%b")

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(monthly))
    width = 0.2

    colors = {
        "PV_Production": "#FFD700",
        "Houseload": "#4169E1",
        "Import_From_Grid": "#FF6347",
        "Sell_To_Grid": "#32CD32",
    }
    labels = {
        "PV_Production": "PV Generation",
        "Houseload": "Load Demand",
        "Import_From_Grid": "Grid Import",
        "Sell_To_Grid": "Grid Export",
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
    # ax.set_title('Monthly Energy Comparison', fontsize=14, fontweight='bold')
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
    _check_matplotlib()

    # Ensure Datetime index
    if "Datetime" in results_df.columns:
        df = results_df.set_index("Datetime")
    else:
        df = results_df.copy()

    # Resample to monthly sums
    monthly = df.resample("ME").sum()

    # Group by month (1-12) to aggregate multi-year data
    monthly_avg = monthly.groupby(monthly.index.month).mean() / 1000.0  # Convert to kWh

    # Ensure all 12 months present
    monthly_avg = monthly_avg.reindex(np.arange(1, 13), fill_value=0.0)

    months = np.arange(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars
    width = 0.35

    # Positives
    ax.bar(months - width / 2, monthly_avg["PV_Production"], width, label="PV Production", color="gold", alpha=0.9)
    ax.bar(months + width / 2, monthly_avg["Sell_To_Grid"], width, label="Grid Export", color="green", alpha=0.9)

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


def plot_montecarlo_simulation(
    all_data: List[dict], results_directory: str, scenario_name: str = "", full_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Generate all plots for Monte Carlo simulation results.

    Args:
        all_data: List of result dictionaries from simulation
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
        full_df: Optional DataFrame with full time-series results (for overlays)
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""
    plots_folder = os.path.join(results_directory, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    # Use provided DF or try to load
    if full_df is not None:
        pass  # use provided full_df
    else:
        # Fallback to loading from disk
        csv_path = os.path.join(results_directory, "monte_carlo_results.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(results_directory, "combined_results.csv")  # Legacy name check

        if os.path.exists(csv_path):
            full_df = pd.read_csv(csv_path)

    # Try to load detailed degradation data
    details_df = None
    details_path = os.path.join(results_directory, "monte_carlo_degradation_details.csv")
    if os.path.exists(details_path):
        details_df = pd.read_csv(details_path)

    if full_df is not None:
        # 4. Cost Overlay
        plot_montecarlo_cost_overlay(full_df, plots_folder, suffix)

        # 5. SOH Overlay
        if details_df is not None:
            plot_montecarlo_soh_traces(details_df, plots_folder, suffix)
        else:
            plot_montecarlo_soh_overlay(full_df, plots_folder, suffix)

        # 6. NPV Savings Distribution (P10/P50/P90/P99)
        plot_montecarlo_npv_distribution(full_df, plots_folder, suffix)

        # 7. Grid Independence Distribution (P10/P50/P90/P99)
        plot_montecarlo_grid_independence_distribution(full_df, plots_folder, suffix)

    # Process Data for Break-even Stats (Unique Runs only)
    if full_df is not None:
        df = full_df
    else:
        df = pd.DataFrame(all_data)

    if not df.empty and "run_number" in df.columns:
        # Total unique runs
        total_runs = df["run_number"].nunique()

        # Filter for runs that achieved break even
        # Since break_even_achieved is boolean and persists, we can just check if ANY row for a run is True
        # Or more simply, take the unique break_even_year for rows where it is not null

        # Filter for rows that actually have the break-even flag set to True
        # This ensures we get a row where break_even_year is populated
        successful_rows = df[df["break_even_achieved"] == True]

        # Get unique runs from these rows
        success_df = successful_rows.drop_duplicates("run_number")

        # If break_even_year is in columns
        if "break_even_year" in success_df.columns:
            breakeven_steps = success_df["break_even_year"].dropna().tolist()
        else:
            breakeven_steps = []

    else:
        # Fallback (should not happen with correct data)
        total_runs = len(all_data) if isinstance(all_data, list) else 0
        breakeven_steps = []

    # 1. Break-even Histogram
    plot_breakeven_distribution(breakeven_steps, total_runs, plots_folder, suffix)

    # 2. Break-even CDF
    plot_breakeven_cdf(breakeven_steps, plots_folder, suffix)

    # 3. Summary Bar (Success Rate)
    plot_breakeven_summary_bar(len(breakeven_steps), total_runs, plots_folder, suffix)

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
    not_achieved = total_runs - achieved
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

    for q, color in zip(quantiles, colors):
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

    for bar, count in zip(bars, counts):
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


def plot_montecarlo_cost_overlay(all_results_df: pd.DataFrame, results_directory: str, suffix: str = "") -> None:
    """
    Overlay plot of Cumulative System Cost vs No System Cost for all runs.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 8))

    runs = all_results_df["run_number"].unique()

    # Plot each run with high transparency
    for run in runs:
        run_data = all_results_df[all_results_df["run_number"] == run]
        x = run_data["year"]
        y_sys = run_data["cumulative_system_cost"]
        y_nosys = run_data["cumulative_nosys_cost"]

        ax.plot(x, y_sys, color="blue", alpha=0.1, linewidth=1)
        ax.plot(x, y_nosys, color="red", alpha=0.1, linewidth=1)

    # Dummy lines for legend
    ax.plot([], [], color="blue", label="System Cost (All Runs)")
    ax.plot([], [], color="red", label="No System Cost (All Runs)")

    # Plot Mean Lines
    mean_sys = all_results_df.groupby("year")["cumulative_system_cost"].mean()
    mean_nosys = all_results_df.groupby("year")["cumulative_nosys_cost"].mean()

    ax.plot(mean_sys.index, mean_sys.values, color="darkblue", linewidth=2.5, linestyle="-", label="Mean System Cost")
    ax.plot(
        mean_nosys.index, mean_nosys.values, color="darkred", linewidth=2.5, linestyle="--", label="Mean No System Cost"
    )

    # Force integer ticks on x-axis
    max_year = int(all_results_df["year"].max())
    if max_year > 0:
        ax.set_xticks(range(1, max_year + 1))

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative Cost (EUR)", fontsize=12)
    # ax.set_title('Financial Projection Uncertainty', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_directory}/montecarlo_cost_overlay{suffix}.png", dpi=300)
    plt.close()


def plot_montecarlo_soh_overlay(all_results_df: pd.DataFrame, results_directory: str, suffix: str = "") -> None:
    """
    Overlay plot of Battery SOH degradation for all runs.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 8))

    runs = all_results_df["run_number"].unique()

    # Plot each run
    for run in runs:
        run_data = all_results_df[all_results_df["run_number"] == run]
        x = run_data["year"]
        y = run_data["battery_soh"]

        ax.plot(x, y, color="green", alpha=0.1, linewidth=1)

    # Mean line
    mean_soh = all_results_df.groupby("year")["battery_soh"].mean()
    ax.plot(mean_soh.index, mean_soh.values, color="darkgreen", linewidth=2.5, label="Mean SOH")

    # Force integer ticks on x-axis
    max_year = int(all_results_df["year"].max())
    if max_year > 0:
        ax.set_xticks(range(1, max_year + 1))

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("State of Health (%)", fontsize=12)
    # ax.set_title('Battery Degradation Uncertainty', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_directory}/montecarlo_soh_overlay{suffix}.png", dpi=300)
    plt.close()


def plot_montecarlo_npv_distribution(all_results_df: pd.DataFrame, results_directory: str, suffix: str = "") -> None:
    """
    Histogram of final-year NPV savings across all MC runs, with P10/P50/P90/P99 lines.

    NPV savings = cumulative_nosys_cost - cumulative_system_cost
    """
    _check_matplotlib()

    df = all_results_df.copy()
    df["npv_savings"] = df["cumulative_nosys_cost"] - df["cumulative_system_cost"]

    final_year = df["year"].max()
    final = df[df["year"] == final_year]["npv_savings"]

    if final.empty:
        print("No final-year data for NPV distribution.")
        return

    p10 = float(final.quantile(0.10))
    p50 = float(final.quantile(0.50))
    p90 = float(final.quantile(0.90))
    p99 = float(final.quantile(0.99))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.hist(final, bins=80, color="tab:blue", alpha=0.6, edgecolor="white", linewidth=0.5)

    # Percentile lines
    line_cfg = [
        (p10, "P10", "--", 1.5),
        (p50, "P50", "-", 2.5),
        (p90, "P90", "--", 1.5),
        (p99, "P99", ":", 1.0),
    ]
    for val, label, ls, lw in line_cfg:
        ax.axvline(x=val, color="tab:red", linestyle=ls, linewidth=lw, label=f"{label}: {val:,.0f} EUR")

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

    ax.set_xlabel(f"NPV Savings at Year {int(final_year)} (EUR)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_directory}/montecarlo_npv_distribution{suffix}.png", dpi=300)
    plt.close()


def plot_montecarlo_grid_independence_distribution(
    all_results_df: pd.DataFrame, results_directory: str, suffix: str = ""
) -> None:
    """
    Histogram of final-year grid independence across all MC runs, with P10/P50/P90/P99 lines.
    """
    _check_matplotlib()

    if "grid_independence_pct" not in all_results_df.columns:
        print("No grid_independence_pct column found, skipping distribution plot.")
        return

    final_year = all_results_df["year"].max()
    final = all_results_df[all_results_df["year"] == final_year]["grid_independence_pct"]

    if final.empty:
        print("No final-year data for grid independence distribution.")
        return

    p10 = float(final.quantile(0.10))
    p50 = float(final.quantile(0.50))
    p90 = float(final.quantile(0.90))
    p99 = float(final.quantile(0.99))

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.hist(final, bins=80, color="tab:green", alpha=0.6, edgecolor="white", linewidth=0.5)

    line_cfg = [
        (p10, "P10", "--", 1.5),
        (p50, "P50", "-", 2.5),
        (p90, "P90", "--", 1.5),
        (p99, "P99", ":", 1.0),
    ]
    for val, label, ls, lw in line_cfg:
        ax.axvline(x=val, color="tab:red", linestyle=ls, linewidth=lw, label=f"{label}: {val:.1f}%")

    ax.set_xlabel(f"Grid Independence at Year {int(final_year)} (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_directory}/montecarlo_grid_independence_distribution{suffix}.png", dpi=300)
    plt.close()


def plot_tariff_comparison(
    results_df: pd.DataFrame,
    tou_periods: "TOUPeriods",
    prices_list: List[Tuple[str, "TOUPrices"]],
    results_directory: str,
    scenario_name: str = "",
) -> None:
    """
    Plot comparison of yearly net costs under different tariff schemes.

    Args:
        results_df: Simulation results
        tou_periods: TOUPeriods object to determine peak/off-peak
        prices_list: List of (name, TOUPrices object) tuples
        results_directory: Output directory
        scenario_name: Optional suffix
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    if "Datetime" not in results_df.columns and not isinstance(results_df.index, pd.DatetimeIndex):
        print("Error: results_df needs Datetime index or column")
        return

    df = results_df.copy()
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

    # Calculate costs for each tariff
    costs = []
    names = []

    # Needs flow columns
    if "Import_From_Grid" not in df.columns or "Sell_To_Grid" not in df.columns:
        print("Error: Import/Sell columns missing")
        return

    # Calculate hours per step (safe approximation from frequency if not 1h)
    # Better: get step size from index
    # Assuming constant step size
    dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600.0 if len(df) > 1 else 1.0

    # Pre-calculate period names for the index to speed up
    # We can use the cycle from tou_periods
    period_names = [tou_periods.get_period(t) for t in df.index]
    df["TOU_Period"] = period_names

    for name, price_obj in prices_list:
        total_import_cost = 0.0
        total_revenue = 0.0

        # Vectorized calculation
        if hasattr(price_obj, "simple_tariff") and name.lower() == "simple":
            # Simple tariff: one price for import
            import_cost = df["Import_From_Grid"].sum() * dt_hours * price_obj.simple_tariff
        else:
            # TOU calculation
            # Map period names to prices in price_obj
            # period_names are 'peak', 'mid_peak', ...
            # We create a series of prices aligned with index

            # Create a mapping dict
            p_map = {
                "peak": price_obj.peak,
                "mid_peak": price_obj.mid_peak,
                "off_peak": price_obj.off_peak,
                "super_off_peak": price_obj.super_off_peak,
            }

            price_series = df["TOU_Period"].map(p_map)

            # Calculate import cost
            import_cost = (df["Import_From_Grid"] * dt_hours * price_series).sum()

        # Export (Fixed sell price usually, or could be indexed too?)
        # Assuming simple sell_price for now
        revenue = df["Sell_To_Grid"].sum() * dt_hours * price_obj.sell_price

        net_cost = (import_cost - revenue) / 1000.0  # Convert to k€? No, keep in €
        costs.append(net_cost)
        names.append(name)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(names, costs, color="steelblue", alpha=0.8, edgecolor="black")

    ax.set_ylabel("Yearly Net Electricity Cost (€)", fontsize=12)
    # ax.set_title('Tariff Regime Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"€{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f"{results_directory}/tariff_comparison{suffix}.png", dpi=300)
    plt.close()


def plot_optimization_results_3d(
    df: pd.DataFrame,
    results_dir: str,
    x_col: str = "Grid_Independence_%",
    y_col: str = "NPV_Eur",
    z_col: str = "ZEB_Ratio",
    filename: str = "pareto_front_3d.png",
) -> None:
    """
    Create a 3D scatter plot of optimization results.

    Args:
        df: DataFrame containing the results
        results_dir: Directory to save the plot
        x_col: Column name for X axis (Grid Independence)
        y_col: Column name for Y axis (NPV)
        z_col: Column name for Z axis (ZEB Ratio)
        filename: Output filename
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    os.makedirs(results_dir, exist_ok=True)

    # Check columns exist
    missing = [col for col in [x_col, y_col, z_col] if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns for 3D plot: {missing}")
        return

    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Scatte plot
        # Color by Z axis (ZEB Ratio) for extra clarity
        sc = ax.scatter(
            df[x_col],
            df[y_col],
            df[z_col],
            c=df[z_col],
            cmap="viridis",
            s=80,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " "))
        ax.set_zlabel(z_col.replace("_", " "))

        # Add colorbar
        plt.colorbar(sc, ax=ax, label=z_col.replace("_", " "), shrink=0.6)

        # Title
        # ax.set_title('Multi-Objective Optimization Surface')

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename), dpi=300)
        plt.close()
        print(f"3D Plot saved to {os.path.join(results_dir, filename)}")

    except Exception as e:
        print(f"Error creating 3D plot: {e}")


def plot_optimization_results_2d(
    df: pd.DataFrame,
    results_dir: str,
    x_col: str = "Grid_Independence_%",
    y_col: str = "NPV_Eur",
    z_col: str = "ZEB_Ratio",
    filename: str = "pareto_front.png",
) -> None:
    """
    Create a 2D scatter plot of optimization results.
    """
    _check_matplotlib()

    os.makedirs(results_dir, exist_ok=True)

    try:
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(df[x_col], df[y_col], c=df[z_col], cmap="viridis", s=100, alpha=0.8)
        plt.colorbar(sc, label=z_col.replace("_", " "))
        plt.xlabel(x_col.replace("_", " ").replace("%", "(%)"))
        plt.ylabel("Net Present Value (€)")
        # plt.title('Pareto Front: Financials vs Independence') # Removed per user request
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"2D Plot saved to {filepath}")

    except Exception as e:
        print(f"Error creating 2D plot: {e}")


def plot_tariff_comparison_manual(
    regimes: List[str],
    vals_sys: List[float],
    vals_nosys: List[float],
    results_dir: str,
    filename: str = "tariff_comparison_bar.png",
) -> None:
    """
    Create a grouped bar chart comparing tariffs.
    """
    _check_matplotlib()
    import numpy as np

    os.makedirs(results_dir, exist_ok=True)

    x = np.arange(len(regimes))
    width = 0.35

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        rects1 = ax.bar(
            x - width / 2, vals_nosys, width, label="No System", color="indianred", alpha=0.8, edgecolor="black"
        )
        rects2 = ax.bar(
            x + width / 2, vals_sys, width, label="With System", color="steelblue", alpha=0.8, edgecolor="black"
        )

        ax.set_ylabel("Net Electricity Cost (Year 1) [€]")
        # ax.set_title('Tariff Comparison: No System vs With PV+Batt')
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"€{height:.0f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Tariff plot saved to {filepath}")

    except Exception as e:
        print(f"Error creating tariff plot: {e}")


def plot_tariff_comparison(results_df: pd.DataFrame, results_directory: str, scenario_name: str = "") -> None:
    """
    Plot bar charts comparing costs across different tariff schemes.
    Generates two plots:
    1. Net Annual Cost (With System)
    2. No System Cost (Pure Load) - if column exists

    Args:
        results_df: DataFrame with 'Tariff', 'Net Cost (€)' and optional 'No System Cost (€)'
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    if "Tariff" not in results_df.columns:
        print("Warning: Missing 'Tariff' column for comparison plot")
        return

    # Define a helper for plotting
    def _create_bar_plot(data_col, filename_part, title_metric):
        fig, ax = plt.subplots(figsize=(12, 6))

        tariffs = results_df["Tariff"]
        values = results_df[data_col]

        # Color palette (Set2 or similar for distinct colors)
        # Or just use a colormap
        cmap = plt.get_cmap("Set3")
        colors = [cmap(i) for i in np.linspace(0, 1, len(tariffs))]

        bars = ax.bar(tariffs, values, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)

        # Add value labels (rounded to cents)
        min_val = values.min()
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label_text = f"€{val:.2f}"

            # Make the lowest cost bold
            # fontweight = 'bold' if val == min_val else 'normal'
            fontweight = "bold"

            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight=fontweight,
                fontsize=10,
            )

        ax.set_ylabel(f"{title_metric} (€)")
        # ax.set_title(f'Tariff Comparison: {title_metric}')
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"{results_directory}/tariff_comparison_{filename_part}{suffix}.png", dpi=300)
        plt.close()

    # Plot 1: Net Cost (With System) - Single Bar
    if "Net Cost (€)" in results_df.columns:
        _create_bar_plot("Net Cost (€)", "net_cost", "Net Annual Cost")

    # Plot 2: Side-by-Side Comparison (No System vs With System)
    if "No System Cost (€)" in results_df.columns and "Net Cost (€)" in results_df.columns:
        fig, ax = plt.subplots(figsize=(14, 7))

        tariffs = results_df["Tariff"]
        no_sys = results_df["No System Cost (€)"]
        with_sys = results_df["Net Cost (€)"]

        x = np.arange(len(tariffs))
        width = 0.35

        # Color mapping for With System (Distinct colors for each tariff)
        cmap = plt.get_cmap("tab10")  # or Set2
        sys_colors = [cmap(i) for i in np.linspace(0, 1, len(tariffs))]

        # Plot Bars
        # No System: Gray, no hatch
        rects1 = ax.bar(
            x - width / 2,
            no_sys,
            width,
            label="No System",
            color="lightgray",
            alpha=1.0,
            edgecolor="gray",
            linewidth=0.5,
        )

        # With System: Distinct Colors, Hatch pattern
        rects2 = ax.bar(
            x + width / 2,
            with_sys,
            width,
            label="With System",
            color=sys_colors,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
        )

        # Add values
        def autolabel(rects, is_gray=False):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"€{height:.0f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="gray" if is_gray else "black",
                    fontweight="normal" if is_gray else "bold",
                )

        autolabel(rects1, is_gray=True)
        autolabel(rects2, is_gray=False)

        ax.set_ylabel("Annual Cost (€)")
        # ax.set_title('Cost Savings Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(tariffs, rotation=15)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(f"{results_directory}/tariff_comparison_savings{suffix}.png", dpi=300)
        plt.close()


def plot_smart_charging_sweep(
    results_df: pd.DataFrame, optimal_pct: float, results_directory: str, scenario_name: str = ""
) -> None:
    """
    Plot smart charging parameter sweep results.

    Args:
        results_df: DataFrame with 'Percentage' and 'Net Cost' columns
        optimal_pct: The optimal percentage found
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    fig, ax = plt.subplots(figsize=(10, 6))

    x = results_df["Percentage"] * 100
    y = results_df["Net Cost"]

    ax.plot(x, y, marker="o", linestyle="-", linewidth=2, markersize=4, label="Annual Cost")

    # Mark optimal
    optimal_row = results_df.loc[results_df["Percentage"] == optimal_pct]
    if not optimal_row.empty:
        opt_cost = optimal_row["Net Cost"].values[0]
        ax.axvline(
            x=optimal_pct * 100, color="r", linestyle="--", alpha=0.7, label=f"Optimal: {optimal_pct * 100:.0f}%"
        )
        ax.scatter([optimal_pct * 100], [opt_cost], color="red", s=100, zorder=5)

    ax.set_xlabel("Target SOC in Off-Peak (Vazio) [%]")
    ax.set_ylabel("Annual Net Cost (€)")
    # ax.set_title('Smart Charging Optimization')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_directory}/smart_charging_sweep{suffix}.png", dpi=300)
    plt.close()


def plot_optimization_results_3d(results_df: pd.DataFrame, results_directory: str, scenario_name: str = "") -> None:
    """
    Create 3D scatter plot for 3-objective optimization results.
    Axes: Modules, Battery Size, NPV (Color mapped to Grid Independence or ZEB)

    Args:
        results_df: DataFrame with 'Modules', 'Battery_kWh', 'NPV_Eur', 'Grid_Independence_%'
        results_directory: Directory to save plots
        scenario_name: Optional suffix
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Data
    x = results_df["Modules"]
    y = results_df["Battery_kWh"]
    z = results_df["NPV_Eur"]
    c = results_df["Grid_Independence_%"]

    img = ax.scatter(x, y, z, c=c, cmap="viridis", s=60, edgecolors="black", alpha=0.9)

    ax.set_xlabel("PV Modules")
    ax.set_ylabel("Battery (kWh)")
    ax.set_zlabel("NPV (€)")

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, pad=0.1)
    cbar.set_label("Grid Independence (%)")

    # ax.set_title('Pareto Front Application')

    plt.tight_layout()
    plt.savefig(f"{results_directory}/pareto_front_3d{suffix}.png", dpi=300)
    plt.close()


def plot_optimization_results_2d(results_df: pd.DataFrame, results_directory: str, scenario_name: str = "") -> None:
    """
    Create 2D scatter plot for optimization results (Pareto Front).
    Axes: Grid Independence vs NPV, Color: ZEB Ratio or Battery Size.

    Args:
        results_df: DataFrame with results
        results_directory: Directory to save plots
        scenario_name: Optional suffix
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    # 2D Projection (Grid Independence vs NPV, color=ZEB_Ratio)
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    if "Grid_Independence_%" in results_df.columns:
        x_2d = results_df["Grid_Independence_%"]
    else:
        # Fallback if column missing (legacy results?)
        x_2d = results_df.get("Modules", [])  # Fallback placeholder

    y_2d = results_df["NPV_Eur"]

    if "ZEB_Ratio" in results_df.columns:
        c_2d = results_df["ZEB_Ratio"]
        c_label = "ZEB Ratio"
    else:
        c_2d = results_df["Battery_kWh"]
        c_label = "Battery (kWh)"

    scatter2 = ax2.scatter(x_2d, y_2d, c=c_2d, cmap="viridis", s=100, edgecolors=(0, 0, 0, 0.5))

    ax2.set_xlabel("Grid Independence (%)")
    ax2.set_ylabel("Net Present Value (€)")
    cbar2 = fig2.colorbar(scatter2, ax=ax2)
    cbar2.set_label(c_label)

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_directory}/pareto_front{suffix}.png", dpi=300)
    plt.close()


def plot_acc_sizing_sweep(results_df: pd.DataFrame, results_directory: str, scenario_name: str = "") -> None:
    """
    Plot ACC Sizer results: Dual-axis plot of Self-Sufficiency vs Payback Period.

    Args:
        results_df: DataFrame with sizing results (from acc_sizer_logic)
        results_directory: Directory to save plots
        scenario_name: Optional suffix
    """
    _check_matplotlib()

    os.makedirs(results_directory, exist_ok=True)
    suffix = f"_{scenario_name}" if scenario_name else ""

    # Sort by Scale/Size to ensure lines plot correctly
    results_df = results_df.sort_values("Modules")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # X-Axis: System Size (kWp)
    x = results_df["System_kWp"]

    # Y-Axis 1: Self-Sufficiency (Benefit)
    color1 = "tab:blue"
    ax1.set_xlabel("System Size (kWp)", fontsize=12)
    ax1.set_ylabel("Self-Sufficiency (%)", color=color1, fontsize=12)
    ax1.plot(x, results_df["Self_Sufficiency_Pct"], color=color1, marker="o", label="Self-Sufficiency")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Y-Axis 2: Payback Period (Cost Metric)
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Payback Period (Years)", color=color2, fontsize=12)
    # Filter out infinite payback for plotting if necessary, but typically sizer limits range
    # Clamp large values for valid display? Or log scale?
    # Let's plot raw but perhaps limit Y to reasonable max if distinct outlier exists
    y2 = results_df["Payback_Years"]
    ax2.plot(x, y2, color=color2, marker="s", linestyle="--", label="Payback")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Mark Optima
    # Financial Optimum (Min Payback)
    opt_fin_idx = y2.idxmin()
    opt_fin_x = x.loc[opt_fin_idx]
    opt_fin_y = y2.loc[opt_fin_idx]
    ax2.scatter(
        [opt_fin_x], [opt_fin_y], s=150, c="red", marker="*", edgecolors="black", zorder=10, label="Financial Opt."
    )

    # Technical Optimum (Max SS)
    opt_tech_idx = results_df["Self_Sufficiency_Pct"].idxmax()
    opt_tech_x = x.loc[opt_tech_idx]
    opt_tech_y = results_df["Self_Sufficiency_Pct"].loc[opt_tech_idx]
    ax1.scatter(
        [opt_tech_x], [opt_tech_y], s=150, c="blue", marker="*", edgecolors="black", zorder=10, label="Technical Opt."
    )

    # Combine Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    title = f"ACC Sizing Sweep{': ' + scenario_name if scenario_name else ''}"
    # ax1.set_title(title)

    plt.tight_layout()
    plt.savefig(f"{results_directory}/acc_sizing_sweep{suffix}.png", dpi=300)
    plt.close()

    # Additional Plot: Trade-off (ROI vs Grid Independence -> Payback vs SS)
    # User asked for ROI vs SS specifically.
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Scatter Payback vs SS
    # Ideal: High SS (Right), Low Payback (Bottom) -> Bottom-Right corner
    sc = ax.scatter(
        results_df["Self_Sufficiency_Pct"],
        results_df["Payback_Years"],
        c=results_df["System_kWp"],
        cmap="viridis",
        s=100,
        edgecolor="black",
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("System Size (kWp)")

    ax.set_xlabel("Self-Sufficiency (%)", fontsize=12)
    ax.set_ylabel("Payback Period (Years)", fontsize=12)
    # ax.set_title("Sizing Trade-off: Payback vs Self-Sufficiency")
    ax.grid(True, alpha=0.3)

    # Annotate Pareto points
    for idx in [opt_fin_idx, opt_tech_idx]:
        txt = f"{results_df.loc[idx, 'System_kWp']:.1f} kWp"
        ax.annotate(
            txt,
            (results_df.loc[idx, "Self_Sufficiency_Pct"], results_df.loc[idx, "Payback_Years"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(f"{results_directory}/acc_sizing_tradeoff{suffix}.png", dpi=300)
    plt.close()


def plot_montecarlo_soh_traces(details_df: pd.DataFrame, results_directory: str, suffix: str = "") -> None:
    """
    Plot detailed SOH traces for sample runs (daily resolution).
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 8))

    runs = details_df["run_number"].unique()

    for run in runs:
        run_data = details_df[details_df["run_number"] == run].copy()

        # Use simple index-based years
        # Assuming daily data
        x = np.arange(len(run_data)) / 365.0
        y = run_data["SOH"]

        ax.plot(x, y, linewidth=1.5, alpha=0.6, label=f"Run {run}")

    ax.set_xlabel("Simulation Year", fontsize=12)
    ax.set_ylabel("State of Health (%)", fontsize=12)
    # ax.set_title('Detailed Degradation Traces (Sample Runs)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower left")

    # Add reference lines
    ax.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="EOL (80%)")

    plt.tight_layout()
    plt.savefig(f"{results_directory}/montecarlo_soh_traces{suffix}.png", dpi=300)
    plt.close()


# =========================================================================
# FUTURE-PROOFING PLOTS
# =========================================================================


def plot_weather_monthly_comparison(
    tmy_vals: np.ndarray,
    stats: "pd.DataFrame",
    ylabel: str,
    tmy_source: str,
    results_dir: str,
    filename: str,
) -> None:
    """
    Scatter+line monthly comparison: TMY vs historical mean with 95% CI band,
    red min-year line, and green max-year line.

    Args:
        tmy_vals:    Array of 12 monthly TMY values.
        stats:       DataFrame with columns mean, ci_low, ci_high, min, max
                     (one row per month, 12 rows).
        ylabel:      Y-axis label (e.g. "GHI (kWh/m²)").
        tmy_source:  Source label for the TMY line (e.g. "pvgis-sarah3").
        results_dir: Directory to save the plot.
        filename:    Output filename (e.g. "monthly_ghi_comparison.png").
    """
    _check_matplotlib()
    os.makedirs(results_dir, exist_ok=True)

    x = np.arange(12)
    hist_mean = stats["mean"].values
    hist_ci_low = stats["ci_low"].values
    hist_ci_high = stats["ci_high"].values
    hist_min = stats["min"].values
    hist_max = stats["max"].values

    fig, ax = plt.subplots(figsize=(14, 7))

    # Min / max envelope lines
    ax.plot(
        x, hist_min, color="tomato", linewidth=1.5, linestyle="--", marker="v", markersize=5, zorder=3, label="Min year"
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
        label="Max year",
    )

    # 95% CI shaded band
    ax.fill_between(x, hist_ci_low, hist_ci_high, color="steelblue", alpha=0.25, label="95% CI")

    # Historical mean line
    ax.plot(x, hist_mean, color="steelblue", linewidth=2.5, marker="o", markersize=7, zorder=4, label="Historical mean")

    # TMY line (on top)
    ax.plot(
        x, tmy_vals, color="darkorange", linewidth=2.5, marker="D", markersize=6, zorder=6, label=f"TMY ({tmy_source})"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.5, 11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weather_annual_ghi_distribution(
    annual_ghi_per_year: "pd.Series",
    tmy_annual_ghi: float,
    hist_annual_ghi_mean: float,
    results_dir: str,
    filename: str = "annual_ghi_distribution.png",
) -> None:
    """
    Histogram of annual GHI values across historical years with TMY and mean lines.

    Args:
        annual_ghi_per_year:  Series of annual GHI totals, one per historical year.
        tmy_annual_ghi:       TMY annual GHI total.
        hist_annual_ghi_mean: Mean of historical annual GHI totals.
        results_dir:          Directory to save the plot.
        filename:             Output filename.
    """
    _check_matplotlib()
    os.makedirs(results_dir, exist_ok=True)

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
        tmy_annual_ghi, color="darkorange", linewidth=2.5, linestyle="--", label=f"TMY ({tmy_annual_ghi:.0f} kWh/m²)"
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
    span = x_max - x_min
    ax.set_xlim(x_min - 0.05 * span, x_max + 0.05 * span)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fmt_years_months(years_decimal) -> str:
    """Convert decimal years to 'Xy Ym' label."""
    if years_decimal is None:
        return "N/A"
    y = int(years_decimal)
    m = int((years_decimal - y) * 12)
    return f"{y}y" if m == 0 else f"{y}y {m}m"


def plot_breakeven_comparison(
    cost_dfs: "List[pd.DataFrame]",
    labels: "List[str]",
    colors: "List[str]",
    results_dir: str,
    filename: str = "breakeven_comparison.png",
) -> None:
    """
    Multi-scenario break-even comparison: N cumulative cost curves vs No-System baseline.

    Args:
        cost_dfs: List of DataFrames, each with columns 'Year',
                  'Cost_No_Sys_Cumulative_NPV', 'Cost_System_Cumulative_NPV',
                  'Savings_Cumulative_NPV'.
        labels:   Display label for each scenario.
        colors:   Line colour for each scenario.
        results_dir: Output directory.
        filename: Output filename.
    """
    _check_matplotlib()
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot No-System baseline for each scenario
    # Track unique baselines to avoid duplicate lines when scenarios share the same baseline
    seen_baselines = {}
    for df, label, color in zip(cost_dfs, labels, colors):
        no_sys_values = tuple(df["Cost_No_Sys_Cumulative_NPV"].round(0).values)
        if no_sys_values not in seen_baselines:
            seen_baselines[no_sys_values] = label
            no_sys_label = "No System" if len(cost_dfs) == 1 else f"No System ({label})"
            ax.plot(
                df["Year"],
                df["Cost_No_Sys_Cumulative_NPV"],
                color=color,
                linestyle="--",
                label=no_sys_label,
                linewidth=2.5,
                alpha=0.7,
            )

    max_year = 20
    for df, label, color in zip(cost_dfs, labels, colors):
        ax.plot(df["Year"], df["Cost_System_Cumulative_NPV"], color=color, label=label, linewidth=2)
        max_year = int(df["Year"].max())

        # Break-even dotted line
        savings = df["Savings_Cumulative_NPV"].values
        years = df["Year"].values
        for i in range(1, len(savings)):
            if savings[i] >= 0 and savings[i - 1] < 0:
                be = years[i - 1] + (-savings[i - 1] / (savings[i] - savings[i - 1]))
                ax.axvline(x=be, color=color, linestyle=":", alpha=0.5, linewidth=1)
                break

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Cost (€)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}€"))
    ax.set_xticks(range(1, max_year + 1))
    ax.set_xlim(0.5, max_year + 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.close()


def plot_breakeven_two(
    df1: "pd.DataFrame",
    label1: str,
    be1: "Optional[float]",
    df2: "pd.DataFrame",
    label2: str,
    be2: "Optional[float]",
    results_dir: str,
    filename: str = "breakeven_two.png",
) -> None:
    """
    Two-scenario break-even comparison with annotated crossover markers.

    Args:
        df1, df2:  DataFrames with 'Year', 'Cost_No_Sys_Cumulative_NPV',
                   'Cost_System_Cumulative_NPV'.
        label1/2:  Display labels.
        be1/be2:   Pre-computed break-even years (decimal), or None.
        results_dir: Output directory.
        filename:  Output filename.
    """
    _check_matplotlib()
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(df1["Year"], df1["Cost_No_Sys_Cumulative_NPV"], "r--", label="No System", linewidth=2.5)
    ax.plot(df1["Year"], df1["Cost_System_Cumulative_NPV"], "b-", label=label1, linewidth=2)
    ax.plot(df2["Year"], df2["Cost_System_Cumulative_NPV"], "g-", label=label2, linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Cost (€)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}€"))

    max_year = int(df1["Year"].max())
    ax.set_xticks(range(1, max_year + 1))
    ax.set_xlim(0.5, max_year + 0.5)

    ylim = ax.get_ylim()
    y_pos = ylim[1] * 0.85
    if be1:
        ax.axvline(x=be1, color="blue", linestyle=":", alpha=0.7, linewidth=1.5)
        ax.annotate(f"{label1}: {_fmt_years_months(be1)}", xy=(be1 + 0.3, y_pos), fontsize=10, color="blue")
    if be2:
        ax.axvline(x=be2, color="green", linestyle=":", alpha=0.7, linewidth=1.5)
        ax.annotate(f"{label2}: {_fmt_years_months(be2)}", xy=(be2 + 0.3, y_pos * 0.92), fontsize=10, color="green")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.close()


def plot_azislope_landscape_2d(
    df_grid: "pd.DataFrame",
    opt_azimuth: float,
    opt_slope: float,
    results_dir: str,
    filename: str = "optimization_landscape_2d.png",
) -> None:
    """
    2-D scatter plot of the azimuth/slope optimisation landscape (colour = metric).

    Args:
        df_grid:     DataFrame with columns 'Azimuth', 'Slope', 'Metric'.
        opt_azimuth: Optimal azimuth to mark with a red cross.
        opt_slope:   Optimal slope to mark with a red cross.
        results_dir: Output directory.
        filename:    Output filename.
    """
    _check_matplotlib()
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(df_grid["Azimuth"], df_grid["Slope"], c=df_grid["Metric"], cmap="viridis", s=50)
    ax.scatter([opt_azimuth], [opt_slope], color="red", marker="x", s=200, linewidth=3, label="Optimum")
    plt.colorbar(sc, ax=ax, label="Metric")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Slope (deg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.close()


def plot_azislope_landscape_3d(
    df_grid: "pd.DataFrame",
    opt_azimuth: float,
    opt_slope: float,
    opt_metric: float,
    results_dir: str,
    filename: str = "optimization_landscape_3d.png",
) -> None:
    """
    3-D surface plot of the azimuth/slope optimisation landscape.

    Args:
        df_grid:     DataFrame with columns 'Azimuth', 'Slope', 'Metric'.
        opt_azimuth: Optimal azimuth to mark.
        opt_slope:   Optimal slope to mark.
        opt_metric:  Metric value at optimum.
        results_dir: Output directory.
        filename:    Output filename.
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    os.makedirs(results_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    pivot = df_grid.pivot(index="Slope", columns="Azimuth", values="Metric")
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.scatter([opt_azimuth], [opt_slope], [opt_metric], color="red", s=100, label="Optimum", zorder=10)
    ax.set_xlabel("Azimuth")
    ax.set_ylabel("Slope")
    ax.set_zlabel("Metric")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.close()


def plot_azislope_ew_1d(
    slope_vals: "np.ndarray",
    metrics: "List[float]",
    opt_slope: float,
    opt_metric: float,
    results_dir: str,
    filename: str = "optimization_1d_slope_ew.png",
) -> None:
    """
    1-D line plot of PV metric vs slope for the East-West configuration.

    Args:
        slope_vals:  Array of slope angles evaluated.
        metrics:     Corresponding metric values (positive = better).
        opt_slope:   Optimal slope to mark.
        opt_metric:  Metric value at optimal slope.
        results_dir: Output directory.
        filename:    Output filename.
    """
    _check_matplotlib()
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(slope_vals, metrics, "b-", linewidth=2)
    ax.plot(opt_slope, opt_metric, "rx", markersize=10, mew=2, label="Optimum")
    ax.set_xlabel("Slope (deg)")
    ax.set_ylabel("Metric")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.close()


# =========================================================================
# PARETO ANALYSIS  (used by tools/analyze_pareto.py)
# =========================================================================


def plot_pareto_front_analysis(
    df: "pd.DataFrame",
    consumptions: "List[float]",
    results_dir: str,
    filename: str = "pareto_front_refined.png",
) -> None:
    """
    2×2 grid of Pareto-front scatter plots, one panel per consumption level.

    Background points show the full solution set (faint); foreground points
    highlight Pareto-optimal configurations (coloured by tariff, shaped by
    strategy).

    Args:
        df:           Full results DataFrame with columns 'Consumption_kWh',
                      'Tariff', 'Detailed_Strategy', 'Net_Cost_Eur',
                      'Grid_Independence_%'.
        consumptions: List of consumption levels to plot (one subplot each).
        results_dir:  Output directory.
        filename:     Output filename.
    """
    _check_matplotlib()
    from matplotlib.lines import Line2D

    os.makedirs(results_dir, exist_ok=True)

    def _is_pareto_efficient(costs, independence):
        is_efficient = [True] * len(costs)
        for i in range(len(costs)):
            for j in range(len(costs)):
                if i == j:
                    continue
                if (costs[j] <= costs[i] and independence[j] >= independence[i]) and (
                    costs[j] < costs[i] or independence[j] > independence[i]
                ):
                    is_efficient[i] = False
                    break
        return is_efficient

    strategies = df["Detailed_Strategy"].unique()
    tariffs = df["Tariff"].unique()

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(tariffs)}
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(strategies)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, cons in enumerate(consumptions):
        ax = axes[i]
        subset = df[df["Consumption_kWh"] == cons].copy()
        if subset.empty:
            continue

        mask = _is_pareto_efficient(subset["Net_Cost_Eur"].values, subset["Grid_Independence_%"].values)
        pareto_subset = subset[mask].copy()
        pareto_subset.to_csv(os.path.join(results_dir, f"pareto_front_{cons}.csv"), index=False)

        for tariff in tariffs:
            for strategy in strategies:
                m = (subset["Tariff"] == tariff) & (subset["Detailed_Strategy"] == strategy)
                if m.any():
                    ax.scatter(
                        subset.loc[m, "Grid_Independence_%"],
                        subset.loc[m, "Net_Cost_Eur"],
                        c=color_map[tariff],
                        marker=marker_map[strategy],
                        alpha=0.2,
                        label=None,
                    )

        for tariff in tariffs:
            for strategy in strategies:
                m = (pareto_subset["Tariff"] == tariff) & (pareto_subset["Detailed_Strategy"] == strategy)
                if m.any():
                    ax.scatter(
                        pareto_subset.loc[m, "Grid_Independence_%"],
                        pareto_subset.loc[m, "Net_Cost_Eur"],
                        c=color_map[tariff],
                        marker=marker_map[strategy],
                        s=100,
                        edgecolor="black",
                        zorder=10,
                        label=f"{tariff} - {strategy}",
                    )

        ax.set_title(f"Pareto Front - {cons} kWh Annual Consumption")
        ax.set_xlabel("Grid Independence (%)")
        ax.set_ylabel("Net Cost (€)")
        ax.grid(True, alpha=0.3)

    legend_elements = [Line2D([0], [0], marker="o", color="w", label="Tariffs:", markersize=0)]
    for tariff, color in color_map.items():
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=tariff, markersize=10)
        )
    legend_elements.append(Line2D([0], [0], marker="o", color="w", label="Strategies:", markersize=0))
    for strategy, marker in marker_map.items():
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markeredgecolor="black",
                markerfacecolor="gray",
                label=strategy,
                markersize=10,
            )
        )

    fig.legend(handles=legend_elements, loc="center right", title="Configuration")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.close()


def plot_loo_cv_summary(
    loo_data: dict,
    results_directory: str,
) -> None:
    """
    Plot LOO cross-validation summary: train vs held-out RMSE per fold.

    Grouped bars showing train RMSE (blue) and held-out RMSE (red) for each
    fold, with a horizontal dashed line at the mean CV RMSE.

    Args:
        loo_data: Dict from loo_cross_validation.json with 'folds' list
        results_directory: Directory to save plot
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    folds = loo_data["folds"]
    system_ids = [f"Sys {f['held_out_system']}" for f in folds]
    train_rmses = [f["train_mean_rmse"] * 100 for f in folds]
    held_out_rmses = [f["held_out_rmse"] * 100 for f in folds]
    mean_cv = loo_data["mean_cv_rmse"] * 100

    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_train = ax.bar(x - width / 2, train_rmses, width, color="#2196F3", alpha=0.85, label="Train RMSE")
    bars_held = ax.bar(x + width / 2, held_out_rmses, width, color="#F44336", alpha=0.85, label="Held-out RMSE")

    ax.axhline(
        y=mean_cv, color="#F44336", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Mean CV RMSE ({mean_cv:.1f} pp)"
    )

    ax.set_xlabel("Held-out system")
    ax.set_ylabel("RMSE (percentage points)")
    ax.set_xticks(x)
    ax.set_xticklabels(system_ids)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "loo_cv_summary.png"), dpi=300)
    plt.close()


def plot_loo_param_stability(
    loo_data: dict,
    full_cal_params: dict,
    results_directory: str,
) -> None:
    """
    Plot parameter stability across LOO folds (one figure per parameter).

    Each figure shows the fitted parameter value per fold as scatter/line,
    with a horizontal reference line for the full-calibration value.

    Args:
        loo_data: Dict from loo_cross_validation.json with 'folds' list
        full_cal_params: Dict with full-calibration values (k0_frac, Ea, cal_b, n)
        results_directory: Directory to save plots
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    folds = loo_data["folds"]
    system_ids = [f"Sys {f['held_out_system']}" for f in folds]
    x = np.arange(len(folds))

    param_configs = [
        ("k0_frac", "k\u2080 (frac/s^b)", True),
        ("Ea", "E_a (J/mol)", True),
        ("cal_b", "b (time exponent)", False),
        ("n", "n (SOC exponent)", False),
    ]

    for param_key, ylabel, use_log in param_configs:
        values = [f["params"][param_key] for f in folds]
        ref_value = full_cal_params[param_key]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, values, "o-", color="#2196F3", markersize=8, linewidth=1.5)
        ax.axhline(
            y=ref_value,
            color="#F44336",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Full calibration ({ref_value:.3e})",
        )

        if use_log:
            ax.set_yscale("log")

        ax.set_xlabel("Held-out system")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(system_ids)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(results_directory, f"loo_param_{param_key}.png"), dpi=300)
        plt.close()


def plot_loo_predictions(
    systems_predictions: list,
    results_directory: str,
) -> None:
    """
    Plot held-out SOH predictions for all LOO folds on one figure.

    Each system shows predicted SOH (line) and measured SOH (markers),
    color-coded by system with RMSE annotation.

    Args:
        systems_predictions: List of dicts, each with:
            'system_id': int
            'dates_measured': array of datetime/timestamps
            'soh_measured': array of measured SOH
            'dates_predicted': array of datetime/timestamps for prediction line
            'soh_predicted': array of predicted SOH
            'rmse': float (RMSE for this system)
        results_directory: Directory to save plot
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(systems_predictions), 10)))

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, sp in enumerate(systems_predictions):
        color = colors[i]
        label = f"Sys {sp['system_id']} (RMSE={sp['rmse'] * 100:.1f} pp)"

        ax.plot(sp["dates_predicted"], np.array(sp["soh_predicted"]) * 100, "-", color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(
            sp["dates_measured"],
            np.array(sp["soh_measured"]) * 100,
            color=color,
            s=60,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label=label,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("SOH (%)")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, "loo_predictions.png"), dpi=300)
    plt.close()


def plot_calendar_aging_sensitivity(
    soh_trajectories: dict,
    eol_threshold: float,
    results_dir: str,
    filename: str = "calendar_aging_sensitivity.png",
) -> None:
    """
    Plot SOH trajectories for different calendar aging k0 scaling factors.

    Args:
        soh_trajectories: Dict mapping label strings (e.g. "k₀ × 0.25") to lists
            of yearly SOH values (length = number of projection years).
        eol_threshold: End-of-life SOH threshold as percentage (e.g. 80.0).
        results_dir: Directory to save the plot.
        filename: Output filename.
    """
    _check_matplotlib()
    os.makedirs(results_dir, exist_ok=True)

    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (label, soh_values) in enumerate(soh_trajectories.items()):
        years = list(range(1, len(soh_values) + 1))
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(
            years,
            soh_values,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            markevery=max(1, len(years) // 10),
            label=label,
            zorder=3,
        )

    # EOL threshold line
    ax.axhline(
        eol_threshold,
        color="grey",
        linewidth=1.5,
        linestyle="--",
        label=f"EOL threshold ({eol_threshold:.0f}%)",
        zorder=2,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("State of Health (%)")
    ax.set_xticks(range(1, len(next(iter(soh_trajectories.values()))) + 1))
    ax.set_ylim(None, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grid_independence_heatmap(
    pivot_data: pd.DataFrame,
    results_directory: str,
    location_name: str,
    filename: str = "grid_independence_heatmap.png",
    metric_label: str = "Grid Independence (%)",
    cmap: str = "YlGnBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot a heatmap of grid independence (or other metric) vs system size.

    Args:
        pivot_data: DataFrame with battery_kwh as index, n_modules as columns,
                    values are the metric (e.g. grid independence %)
        results_directory: Directory to save the plot
        location_name: Location name for labelling (used in axis/legend, not title)
        filename: Output filename
        metric_label: Colorbar label
        cmap: Matplotlib colormap name
        vmin: Colorbar minimum (auto if None)
        vmax: Colorbar maximum (auto if None)
    """
    _check_matplotlib()
    from matplotlib.colors import Normalize

    os.makedirs(results_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(
        pivot_data.values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="lower",
    )

    # Axis labels from pivot index/columns
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([str(c) for c in pivot_data.columns])
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([str(i) for i in pivot_data.index])

    ax.set_xlabel("Number of PV Modules")
    ax.set_ylabel("Battery Capacity (kWh)")

    # Annotate cells
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > (norm.vmax + norm.vmin) / 2 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(metric_label)

    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_location_comparison_delta(
    delta_data: pd.DataFrame,
    results_directory: str,
    loc_a: str,
    loc_b: str,
    filename: str = "grid_independence_delta.png",
    metric_label: str = "Grid Independence Delta (pp)",
) -> None:
    """
    Plot a diverging heatmap of the difference in a metric between two locations.

    Args:
        delta_data: DataFrame with battery_kwh as index, n_modules as columns,
                    values are (loc_a - loc_b) in percentage points
        results_directory: Directory to save the plot
        loc_a: Name of location A
        loc_b: Name of location B
        filename: Output filename
        metric_label: Colorbar label
    """
    _check_matplotlib()
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(results_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    abs_max = max(abs(np.nanmin(delta_data.values)), abs(np.nanmax(delta_data.values)))
    if abs_max == 0:
        abs_max = 1.0
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    im = ax.imshow(
        delta_data.values,
        aspect="auto",
        cmap="RdBu",
        norm=norm,
        origin="lower",
    )

    ax.set_xticks(range(len(delta_data.columns)))
    ax.set_xticklabels([str(c) for c in delta_data.columns])
    ax.set_yticks(range(len(delta_data.index)))
    ax.set_yticklabels([str(i) for i in delta_data.index])

    ax.set_xlabel("Number of PV Modules")
    ax.set_ylabel("Battery Capacity (kWh)")

    # Annotate cells
    for i in range(len(delta_data.index)):
        for j in range(len(delta_data.columns)):
            val = delta_data.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > abs_max * 0.6 else "black"
                sign = "+" if val > 0 else ""
                ax.text(
                    j, i, f"{sign}{val:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold"
                )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"{metric_label} ({loc_a} \u2212 {loc_b})")

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

    Creates co2_savings_{scenario}.png showing yearly CO2 avoided
    (total and self-consumed) as bars with a cumulative line.

    Args:
        cost_projection: DataFrame from cost_analysis_projection() with CO2 columns
        results_directory: Directory to save plots
        scenario_name: Optional suffix for filenames
    """
    _check_matplotlib()

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


# =========================================================================
# Polysun vs PVBAT degradation comparison plots
# =========================================================================


def plot_degradation_methodology_comparison(
    pvbat_soh: "pd.DataFrame",
    polysun_df: "pd.DataFrame",
    results_directory: str,
    scenario_label: str = "",
    suffix: str = "",
) -> None:
    """
    Compare PVBAT continuous SOH vs Polysun Miner's damage accumulation.

    Produces two separate figures:
      1. SOH over time: PVBAT's declining SOH curve vs Polysun's equivalent SOH
      2. Polysun damage accumulation (D) with replacement threshold at D=1

    Args:
        pvbat_soh: PVBAT degradation DataFrame with 'SOH' column (%) indexed by year
            or containing a 'Year' column.
        polysun_df: Output of simulate_polysun_degradation().
        results_directory: Directory to save plots.
        scenario_label: Label for annotation (e.g., "Porto 5kWp/5kWh").
        suffix: Filename suffix.
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    years_pvbat = pvbat_soh["Year"].values if "Year" in pvbat_soh.columns else np.arange(1, len(pvbat_soh) + 1)
    soh_pvbat = pvbat_soh["SOH"].values if "SOH" in pvbat_soh.columns else pvbat_soh.iloc[:, 0].values
    years_polysun = polysun_df["Year"].values
    soh_polysun = polysun_df["SOH_Equivalent"].values

    # --- Figure 1: SOH comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years_pvbat, soh_pvbat, "b-", linewidth=2.5, marker="o", markersize=3, label="PVBAT (Naumann)")
    ax.plot(years_polysun, soh_polysun, "r--", linewidth=2.5, marker="s", markersize=3, label="Polysun (Miner/Wöhler)")
    ax.axhline(80, color="grey", linestyle=":", linewidth=1.5, alpha=0.7, label="EOL threshold (80%)")

    # Mark replacements
    replacements = polysun_df[polysun_df["Replacement"]]
    for _, row in replacements.iterrows():
        ax.axvline(row["Year"], color="red", linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("State of Health (%)", fontsize=12)
    ax.set_ylim(60, 102)
    ax.set_xticks(range(int(min(years_pvbat[0], years_polysun[0])), int(max(years_pvbat[-1], years_polysun[-1])) + 1))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    if scenario_label:
        ax.text(
            0.98,
            0.02,
            scenario_label,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            style="italic",
            alpha=0.7,
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_directory, f"polysun_pvbat_soh_comparison{suffix}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # --- Figure 2: Polysun damage accumulation ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        years_polysun,
        polysun_df["Damage_Cumulative"].values,
        "r-",
        linewidth=2.5,
        marker="s",
        markersize=3,
        label="Miner's cumulative damage",
    )
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.5, alpha=0.7, label="Cycle EOL (D = 1)")

    # Mark replacements
    for _, row in replacements.iterrows():
        ax.axvline(row["Year"], color="red", linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative Damage D", fontsize=12)
    ax.set_xticks(range(int(years_polysun[0]), int(years_polysun[-1]) + 1))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    if scenario_label:
        ax.text(
            0.98,
            0.02,
            scenario_label,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            style="italic",
            alpha=0.7,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(results_directory, f"polysun_miner_damage{suffix}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lifetime_prediction_comparison(
    scenarios: dict,
    results_directory: str,
    suffix: str = "",
) -> None:
    """
    Grouped bar chart: predicted lifetime per methodology per scenario.

    Args:
        scenarios: Dict mapping scenario label to dict with keys:
            'pvbat_eol_year': Year PVBAT hits 80% SOH (float or int).
            'polysun_total_life': Polysun predicted total life (years).
            'polysun_cycle_life': Polysun cycle life component (years).
            'polysun_calendar_life': Polysun calendar life component (years).
        results_directory: Directory to save plot.
        suffix: Filename suffix.
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    labels = list(scenarios.keys())
    pvbat_years = [scenarios[s]["pvbat_eol_year"] for s in labels]
    polysun_years = [scenarios[s]["polysun_total_life"] for s in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, pvbat_years, width, label="PVBAT (Naumann)", color="#1976D2", alpha=0.85)
    bars2 = ax.bar(x + width / 2, polysun_years, width, label="Polysun (Miner/Wöhler)", color="#D32F2F", alpha=0.85)

    # Annotate bar values
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Predicted Battery Lifetime (years)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_directory, f"lifetime_prediction_comparison{suffix}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def plot_temperature_sensitivity_comparison(
    locations: dict,
    results_directory: str,
    suffix: str = "",
) -> None:
    """
    Show how PVBAT lifetime varies across locations (temperature-dependent)
    while Polysun predicts the same lifetime everywhere (temperature-blind).

    Args:
        locations: Dict mapping location name to dict with keys:
            'pvbat_eol_year': PVBAT predicted EOL year.
            'polysun_total_life': Polysun total life (same for all if same cycling).
            'mean_temp_c': Annual mean ambient temperature (°C).
        results_directory: Directory to save plot.
        suffix: Filename suffix.
    """
    _check_matplotlib()
    os.makedirs(results_directory, exist_ok=True)

    labels = list(locations.keys())
    pvbat_years = [locations[s]["pvbat_eol_year"] for s in labels]
    polysun_years = [locations[s]["polysun_total_life"] for s in labels]
    temps = [locations[s]["mean_temp_c"] for s in labels]

    # Sort by temperature
    sort_idx = np.argsort(temps)
    labels = [labels[i] for i in sort_idx]
    pvbat_years = [pvbat_years[i] for i in sort_idx]
    polysun_years = [polysun_years[i] for i in sort_idx]
    temps = [temps[i] for i in sort_idx]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, pvbat_years, width, label="PVBAT (Naumann)", color="#1976D2", alpha=0.85)
    bars2 = ax.bar(x + width / 2, polysun_years, width, label="Polysun (Miner/Wöhler)", color="#D32F2F", alpha=0.85)

    # Annotate with temperature
    for i, (bar, temp) in enumerate(zip(bars1, temps)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}y",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}y",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add temperature as secondary labels
    ax2_labels = [f"{l}\n({t:.0f}°C)" for l, t in zip(labels, temps)]
    ax.set_xticks(x)
    ax.set_xticklabels(ax2_labels, fontsize=11)

    ax.set_ylabel("Predicted Battery Lifetime (years)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_directory, f"temperature_sensitivity_comparison{suffix}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
