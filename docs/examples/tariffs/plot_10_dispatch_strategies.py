"""
Battery dispatch strategies on a bi-hourly tariff
=================================================

On a time-of-use tariff a battery can do more than store surplus PV: it can
hold its charge for the expensive hours, or charge from the grid off peak.
This page compares, for the quickstart home on the Portuguese bi-hourly
schedule, four ``[smart_charging]`` settings:

- **greedy**: no ``[smart_charging]`` table. The battery stores PV surplus
  and discharges whenever the house needs power.
- **discharge_only**: ``mode = "discharge_only"`` with
  ``discharge_periods = ["peak"]``. The battery stores PV surplus but
  discharges only at peak, and never charges from the grid.
- **fixed_target**: the packaged file. Off peak the grid charges the battery
  towards half its usable window; at peak it discharges to the load.
- **hold_target**: as fixed_target, but the battery may also discharge off
  peak (``discharge_periods = ["peak", "off_peak"]`` with
  ``overlap_policy = "hold_target"``). The grid target becomes a floor: off
  peak the battery discharges above it and the grid charges below it.

.. literalinclude:: /../configs/examples/smart-charging-portugal.toml
   :language: toml
   :caption: configs/examples/smart-charging-portugal.toml

The prices and power limits are illustrative. See
`Smart charging <../../getting-started/configuration.html#smart-charging>`__
for every key.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
from gallery_results import load_case, money, say, table

case = load_case("dispatch_strategies")
strategies = case.csv("strategies.csv").set_index("strategy")
week = case.week("week_winter.csv", case.manifest["timezone"])
currency = case.manifest["currency"]
names = list(strategies.index)
case.stamp()

# %%
# A winter week
# -------------
# State of charge in the usable window and the power the grid sends into the
# battery, for one February week. Off peak on this schedule is the night.

fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.1 * len(names)), sharex=True, sharey=True)
for ax, name in zip(axes, names, strict=True):
    ax.fill_between(week["Datetime"], week[f"{name}:Battery_SOC_Normalized"] * 100, color="#4477AA", alpha=0.35,
                    label="State of charge (%)")  # fmt: skip
    grid = ax.twinx()
    grid.plot(week["Datetime"], week[f"{name}:Grid_AC_To_Battery"] / 1000, color="#EE6677", lw=1.2,
              label="Grid to battery (kW)")  # fmt: skip
    grid.set_ylim(0, max(1.0, week.filter(like="Grid_AC_To_Battery").max().max() / 1000 * 1.1))
    grid.set_ylabel("kW", color="#EE6677")
    ax.set_ylim(0, 105)
    ax.set_ylabel("SOC (%)")
    ax.set_title(name, loc="left", fontsize=10)
    ax.grid(alpha=0.3)
handles = axes[0].get_legend_handles_labels()[0] + grid.get_legend_handles_labels()[0]
fig.legend(handles, ["State of charge (%)", "Grid to battery (kW)"], loc="upper right", ncol=2, frameon=False)
fig.autofmt_xdate()
fig.tight_layout(rect=(0, 0, 1, 0.97))

# %%
# The year and the money
# ----------------------

shown = strategies.reset_index()[["strategy", "grid_independence_pct", "self_consumption_pct", "grid_import_kwh",
                                  "grid_export_kwh", "bill_year1", "npv_savings", "payback_year"]]  # fmt: skip
table(
    shown.rename(
        columns={
            "strategy": "Strategy",
            "grid_independence_pct": "Grid independence (%)",
            "self_consumption_pct": "Self-consumption (%)",
            "grid_import_kwh": "Import (kWh)",
            "grid_export_kwh": "Export (kWh)",
            "bill_year1": "Year-1 bill",
            "npv_savings": "NPV savings",
            "payback_year": "Payback year",
        }
    ),
    **{"Import (kWh)": ".0f", "Export (kWh)": ".0f", "Year-1 bill": ".2f", "NPV savings": ".0f"},
)

# %%

# sphinx_gallery_start_ignore
hours = (week["Datetime"].iloc[1] - week["Datetime"].iloc[0]).total_seconds() / 3600
grid_charge = {name: week[f"{name}:Grid_AC_To_Battery"].sum() * hours / 1000 for name in names}
best = strategies["npv_savings"].idxmax()
ranking = " > ".join(strategies["npv_savings"].sort_values(ascending=False).index)
say(
    f"Ranked by NPV savings: {ranking}. The best, {best}, saves "
    f"{money(strategies.loc[best, 'npv_savings'], currency)}; "
    + "; ".join(
        f"{name} saves {money(strategies.loc[best, 'npv_savings'] - strategies.loc[name, 'npv_savings'], currency)} less"
        for name in names
        if name != best
    )
    + ".",
    "In the winter week the grid charges the battery with "
    + ", ".join(f"{grid_charge[name]:.1f} kWh under {name}" for name in names)
    + ". Over the year, "
    + ", ".join(
        f"{name} exports {strategies.loc[name, 'grid_export_kwh'] - strategies.loc['greedy', 'grid_export_kwh']:+,.0f} kWh"
        for name in names
        if name != "greedy"
    )
    + " against greedy: a battery held for the peak, or filled from the grid at night, has less room for PV on a "
    "sunny day, and that PV is exported at the low export price instead."
    + (
        " On these prices the gap between peak and off peak does not pay for it."
        if best == "greedy"
        else f" On these prices the gap between peak and off peak pays for it, and {best} wins."
    ),
    f"Grid independence is {strategies.loc['greedy', 'grid_independence_pct']:.1f} % under greedy and "
    f"{strategies.loc['fixed_target', 'grid_independence_pct']:.1f} % under fixed_target. Grid charging lowers "
    "it by definition: energy stored from the grid counts as imported.",
)
# sphinx_gallery_end_ignore

# %%
# When a strategy can win
# -----------------------
# A grid-charging strategy pays when the peak and off-peak prices are far
# apart, when PV is small next to the load (so the battery is rarely full of
# PV), or in winter. Use ``App.revalue`` to test other prices on a stored run
# (:doc:`plot_12_price_scenarios`): fixed-target instructions follow the
# periods, not the prices, so new prices on the same schedule are re-priced
# without simulating again.
