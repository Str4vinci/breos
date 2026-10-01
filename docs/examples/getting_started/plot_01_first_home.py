"""
Your first home: PV and a battery in Porto
==========================================

What does a rooftop PV system with a home battery do for a household in Porto
over a year, and does it pay for itself? This page reports the packaged
quickstart configuration: a 10-module array, a 5 kWh battery, the bundled H0
household load profile and the Portuguese residential cost preset.

.. literalinclude:: /../configs/examples/quickstart.toml
   :language: toml
   :caption: configs/examples/quickstart.toml

To run it yourself::

    breos run --config configs/examples/quickstart.toml --output result.json

``result.json`` is the dict :meth:`breos.App.result` returns;
:doc:`/getting-started/interpreting-results` explains every field.
"""

# %%
# The stored run
# --------------
# The result below was computed once and stored; this page only reads it.

# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import pandas as pd
from gallery_results import load_case, money, say, table

from breos.plotting import plot_pv_loss_waterfall, weekly_graphs

case = load_case("first_home")
result = case.json("result.json")
currency = result["provenance"]["currency"]
timezone = result["provenance"]["timezone"]
case.stamp()

# %%
# Headline numbers
# ----------------
# Grid independence is the share of the load served without the grid, and
# self-consumption the share of the PV production used in the house. NPV
# savings compare the discounted cost with the system against the household's
# cost without it over the project.

say(
    f"The {result['pv_kwp']:g} kWp array delivers {result['usable_ac_system_production_kwh']:,.0f} kWh of usable AC "
    f"energy in the first year against a load of {result['consumption_kwh']:,.0f} kWh. The house uses "
    f"{result['self_consumption_pct']:.1f} % of it and exports the rest "
    f"({result['grid_export_kwh']:,.0f} kWh), and the grid supplies only {result['grid_import_kwh']:,.0f} kWh: "
    f"grid independence is {result['grid_independence_pct']:.1f} %.",
    f"The system costs {money(result['total_investment'], currency)}, pays back in year {result['payback_year']} "
    f"and saves {money(result['npv_savings'], currency)} in present value over "
    f"{len(result['yearly'])} years. The battery ends the project at {result['battery_soh_end_pct']:.1f} % state "
    f"of health after {result['battery_replacements']} replacement.",
)

# %%

headline = pd.DataFrame(
    [
        ("PV size", f"{result['pv_kwp']:g} kWp"),
        ("Battery", f"{result['battery_kwh']:g} kWh"),
        ("Usable AC production, year 1", f"{result['usable_ac_system_production_kwh']:,.0f} kWh"),
        ("Self-consumption", f"{result['self_consumption_pct']:.1f} %"),
        ("Grid independence", f"{result['grid_independence_pct']:.1f} %"),
        ("Investment", money(result["total_investment"], currency)),
        ("Payback year", f"{result['payback_year']}"),
        ("NPV savings", money(result["npv_savings"], currency)),
        ("LCOE", f"{result['lcoe_per_kwh']:.3f} {currency}/kWh"),
        ("CO2 avoided over the project", f"{result['co2_avoided_total_lifetime_kg'] / 1000:,.1f} t"),
    ],
    columns=["Metric", "Value"],
)
table(headline)

# %%
# Month by month
# --------------
# Summer production is far above the load, and most of it is exported. In
# winter the array still covers part of the load, but the grid fills the gap.

monthly = pd.DataFrame(result["monthly"]).set_index("month")
fig, ax = plt.subplots(figsize=(9, 4.5))
positions = range(len(monthly))
width = 0.2
for offset, (column, label, color) in enumerate(
    [
        ("usable_ac_system_production_kwh", "Usable PV production", "#e6a700"),
        ("consumption_kwh", "Load", "#3a6ea5"),
        ("grid_import_kwh", "Grid import", "#c0392b"),
        ("grid_export_kwh", "Grid export", "#2e8b57"),
    ]
):
    ax.bar([p + (offset - 1.5) * width for p in positions], monthly[column], width, label=label, color=color)
ax.set_xticks(list(positions), monthly.index)
ax.set_ylabel("Energy (kWh)")
ax.set_title("Monthly energy balance, year 1")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

# %%

winter_month = monthly["grid_independence_pct"].idxmin()
summer_month = monthly["grid_independence_pct"].idxmax()
say(
    f"Grid independence ranges from {monthly.loc[winter_month, 'grid_independence_pct']:.0f} % in {winter_month} "
    f"to {monthly.loc[summer_month, 'grid_independence_pct']:.0f} % in {summer_month}."
)

# %%
# A week in winter and a week in summer
# -------------------------------------
# The stored weeks are the first simulated year step by step, from
# :meth:`breos.App.timeseries`: mean power over each step, in W.

winter = case.week("week_winter.csv", timezone)
summer = case.week("week_summer.csv", timezone)


def describe(week, label):
    days = week.groupby(week["Datetime"].dt.date)
    full_days = int((days["Battery_SOC_Normalized"].max() >= 0.99).sum())
    hours = (week["Datetime"].iloc[1] - week["Datetime"].iloc[0]).total_seconds() / 3600
    load, from_battery, imported = (
        week[column].sum() * hours / 1000 for column in ("Houseload", "Battery_AC_To_Load", "Import_From_Grid")
    )
    return (
        f"In the {label} week the battery fills on {full_days} of {days.ngroups} days and serves "
        f"{from_battery:.0f} of the {load:.0f} kWh load; the grid supplies {imported:.0f} kWh."
    )


say(describe(winter, "winter"), describe(summer, "summer"))

# %%

weekly_graphs(winter, case.manifest["iso_weeks"]["winter"])

# %%

weekly_graphs(summer, case.manifest["iso_weeks"]["summer"])

# %%
# Where the PV energy goes
# ------------------------
# The loss diagram follows the year-1 energy from the irradiance on a
# horizontal surface, through transposition onto the tilted array and the
# optical, thermal and static losses, to the DC energy ready for dispatch. The
# box below it splits that energy into what reached the load directly or
# through the battery, what was exported and what the inverter lost.

fig = plot_pv_loss_waterfall(result["pv_loss_waterfall"], figsize=(10, 8))

# %%
# The battery over the project
# ----------------------------
# The yearly ledger holds the state of health at the end of each year. The
# pack is replaced when its usable capacity reaches the end-of-life threshold.

yearly = pd.DataFrame(result["yearly"]).set_index("year")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(yearly.index, yearly["soh_pct"], marker="o", color="#3a6ea5")
ax.axhline(100 * result["provenance"]["resolved_config"]["battery_eol_percentage"], color="0.4", ls=":")
ax.set_xlabel("Project year")
ax.set_ylabel("State of health (%)")
ax.set_title("Battery state of health at the end of each year")
ax.grid(alpha=0.3)
fig.tight_layout()

# %%

events = result["degradation"]["replacement_events"]
say(
    "The pack is replaced in year "
    + ", ".join(str(event["year"]) for event in events)
    + f", and the replacement costs {money(result['battery_replacement_cost_npv'], currency)} in present value."
    if events
    else "The pack lasts the whole project."
)
