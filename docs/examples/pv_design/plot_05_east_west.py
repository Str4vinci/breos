"""
East–West or South?
===================

An east–west roof spreads production over the morning and the afternoon
instead of one midday peak, so more of it lands when the house is using
power. Does that make it the better layout? This page compares the packaged
east–west roof with the same twelve modules facing south, at the same low
tilt and at the latitude tilt BREOS picks by default, each without a battery
and with the file's 5 kWh battery.

``[[pv_arrays]]`` simulates each face separately and adds their DC output
before the energy balance, so an east–west roof is not collapsed into one
average orientation:

.. literalinclude:: /../configs/examples/east-west-roof.toml
   :language: toml
   :caption: configs/examples/east-west-roof.toml

The south designs replace ``[[pv_arrays]]`` with ``n_modules`` and, for the
low tilt, ``tilt`` and ``azimuth = 180``. Each array may also set its own
``module``.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
from gallery_results import load_case, money, say, table

case = load_case("east_west")
designs = case.csv("designs.csv")
names = list(case.manifest["variants"])
currency = case.manifest["currency"]
summer = case.week("week_summer.csv", case.manifest["timezone"])
winter = case.week("week_winter.csv", case.manifest["timezone"])
case.stamp()

# %%
# A summer day, averaged over a week
# ----------------------------------
# The average of each hour of the stored July week. The east–west roof
# starts earlier and ends later, but its peak is lower and its day total
# smaller.

colors = dict(zip(names, ["#c0392b", "#e6a700", "#3a6ea5"], strict=True))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, week, title in ((axes[0], summer, "July week"), (axes[1], winter, "January week")):
    hour = week["Datetime"].dt.hour
    profile = week.groupby(hour)[[*names, "Houseload"]].mean() / 1000
    for name in names:
        ax.plot(profile.index, profile[name], label=name, color=colors[name])
    ax.plot(profile.index, profile["Houseload"], "k--", label="Load")
    ax.set_title(f"Average day, {title}")
    ax.set_xlabel("Hour of day (local time)")
    ax.set_xticks(range(0, 25, 3))
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Mean AC power (kW)")
axes[1].legend()
fig.tight_layout()

# %%
# Year-1 energy and the money
# ---------------------------

shown = designs[["design", "battery_kwh", "usable_ac_system_production_kwh", "self_consumption_pct",
                 "grid_independence_pct", "grid_export_kwh", "npv_savings", "payback_year"]]  # fmt: skip
shown = shown.rename(
    columns={
        "design": "Design",
        "battery_kwh": "Battery (kWh)",
        "usable_ac_system_production_kwh": "Usable PV (kWh)",
        "self_consumption_pct": "Self-consumption (%)",
        "grid_independence_pct": "Grid independence (%)",
        "grid_export_kwh": "Export (kWh)",
        "npv_savings": "NPV savings",
        "payback_year": "Payback year",
    }
)
table(shown, **{"Battery (kWh)": "g", "Usable PV (kWh)": ",.0f", "Export (kWh)": ",.0f", "NPV savings": ",.0f"})

# %%

pv_only = designs[designs["battery_kwh"] == 0].set_index("design")
battery = designs[designs["battery_kwh"] > 0].set_index("design")
east_west, south_low, south = names
say(
    f"Without a battery the east–west roof uses {pv_only.loc[east_west, 'self_consumption_pct']:.1f} % of its "
    f"production in the house, against {pv_only.loc[south, 'self_consumption_pct']:.1f} % facing south at the "
    f"latitude tilt. But it produces "
    f"{pv_only.loc[south, 'usable_ac_system_production_kwh'] - pv_only.loc[east_west, 'usable_ac_system_production_kwh']:,.0f} "
    f"kWh less a year, so grid independence moves by only "
    f"{pv_only.loc[east_west, 'grid_independence_pct'] - pv_only.loc[south, 'grid_independence_pct']:+.1f} points "
    f"and NPV savings change by {money(pv_only.loc[east_west, 'npv_savings'] - pv_only.loc[south, 'npv_savings'], currency)}.",
    f"With the {battery['battery_kwh'].iloc[0]:g} kWh battery the battery does the time shifting, so the "
    f"east–west advantage in grid independence disappears "
    f"({battery.loc[east_west, 'grid_independence_pct']:.1f} % against "
    f"{battery.loc[south, 'grid_independence_pct']:.1f} %) and NPV savings differ by "
    f"{money(battery.loc[east_west, 'npv_savings'] - battery.loc[south, 'npv_savings'], currency)}.",
    "A higher self-consumption share flatters the east–west roof: it is a share of a smaller production. "
    "Where the roof allows a choice, compare the absolute energy and the money. Where it does not, an "
    "east–west roof is a reasonable use of the space; the low south-facing tilt shows how much of the gap "
    "comes from the tilt alone: "
    f"{pv_only.loc[south_low, 'usable_ac_system_production_kwh']:,.0f} kWh against "
    f"{pv_only.loc[south, 'usable_ac_system_production_kwh']:,.0f} kWh.",
)
