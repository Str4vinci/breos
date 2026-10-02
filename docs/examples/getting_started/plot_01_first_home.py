"""
PV and battery system in Porto: energy balance and economics
=============================================================

The energy flows, battery ageing and project economics of a residential
rooftop PV system with a battery in Porto. This page reports the packaged
quickstart configuration: a 10-module array, a 5 kWh battery, the bundled H0
household load profile and the Portuguese residential cost preset.

The Portuguese profiles are the E-REDES BTN classes (``eredes_btn_a``,
``_b``, ``_c``). Their terms give no clear grant to redistribute them, so
BREOS does not bundle them and this page uses H0, the profile every install
has. :doc:`/how-to/external-load-profile` shows how to point BREOS at your own
copy of the E-REDES files.

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

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
import pandas as pd
from gallery_results import load_case, money, number, say, table

from breos.plotting import plot_pv_loss_waterfall, weekly_graphs

case = load_case("first_home")
result = case.json("result.json")
currency = result["provenance"]["currency"]
timezone = result["provenance"]["timezone"]
case.stamp()

# %%
# Summary
# -------
# Grid independence is the share of the load served without the grid, and
# self-consumption the share of the PV production used in the house. NPV
# savings compare the discounted cost with the system against the household's
# cost without it over the project.

summary = pd.DataFrame(
    [
        ("PV size", f"{result['pv_kwp']:g} kWp"),
        ("Battery", f"{result['battery_kwh']:g} kWh"),
        ("Load, year 1", f"{number(result['consumption_kwh'])} kWh"),
        ("Usable AC production, year 1", f"{number(result['usable_ac_system_production_kwh'])} kWh"),
        ("Grid import, year 1", f"{number(result['grid_import_kwh'])} kWh"),
        ("Grid export, year 1", f"{number(result['grid_export_kwh'])} kWh"),
        ("Self-consumption", f"{result['self_consumption_pct']:.1f} %"),
        ("Grid independence", f"{result['grid_independence_pct']:.1f} %"),
        ("Investment", money(result["total_investment"], currency)),
        ("Payback year", f"{result['payback_year']}"),
        (f"NPV savings over {len(result['yearly'])} years", money(result["npv_savings"], currency)),
        ("LCOE", f"{result['lcoe_per_kwh']:.3f} {currency}/kWh"),
        ("Battery replacements", f"{result['battery_replacements']}"),
        ("Battery state of health at the end", f"{result['battery_soh_end_pct']:.1f} %"),
        ("CO2 avoided over the project", f"{number(result['co2_avoided_total_lifetime_kg'] / 1000, 1)} t"),
    ],
    columns=["Metric", "Value"],
)
table(summary)

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
        ("usable_ac_system_production_kwh", "Usable PV production", "#CCBB44"),
        ("consumption_kwh", "Load", "#4477AA"),
        ("grid_import_kwh", "Grid import", "#EE6677"),
        ("grid_export_kwh", "Grid export", "#228833"),
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

# sphinx_gallery_start_ignore
winter_month = monthly["grid_independence_pct"].idxmin()
summer_month = monthly["grid_independence_pct"].idxmax()
say(
    f"Grid independence ranges from {monthly.loc[winter_month, 'grid_independence_pct']:.0f} % in {winter_month} "
    f"to {monthly.loc[summer_month, 'grid_independence_pct']:.0f} % in {summer_month}."
)
# sphinx_gallery_end_ignore

# %%
# A week in winter and a week in summer
# -------------------------------------
# The stored weeks are the first simulated year step by step, from
# :meth:`breos.App.timeseries`: mean power over each step, in W.

winter = case.week("week_winter.csv", timezone)
summer = case.week("week_summer.csv", timezone)

# sphinx_gallery_start_ignore


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
# sphinx_gallery_end_ignore

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
# The yearly ledger holds the state of health at the end of each year; the
# new pack starts at 100 %. The pack is replaced when its usable capacity
# reaches the end-of-life threshold, at the time the financial ledger records.

yearly = pd.DataFrame(result["yearly"]).set_index("year")
eol = 100 * result["provenance"]["resolved_config"]["battery_eol_percentage"]
swaps = [row["replacement_time_years"] for row in result["financial"] if row.get("replacement_time_years")]
years, health = [0.0], [100.0]
for year, soh in yearly["soh_pct"].items():
    for swap in (swap for swap in swaps if years[-1] < swap <= year):
        years += [swap, swap]
        health += [eol, 100.0]
    years.append(year)
    health.append(soh)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(years, health, color="C0")
ax.plot([0, *yearly.index], [100.0, *yearly["soh_pct"]], "o", color="C0", label="End of year")
ax.axhline(eol, color="0.4", ls=":", label="End-of-life threshold")
ax.set_xticks(range(0, len(yearly) + 1))
ax.set_xlim(-0.3, len(yearly) + 0.3)
ax.set_xlabel("Project year")
ax.set_ylabel("State of health (%)")
ax.set_title("Battery state of health over the project")
ax.legend(loc="lower left")
fig.tight_layout()

# %%

# sphinx_gallery_start_ignore
events = result["degradation"]["replacement_events"]
say(
    "The pack is replaced in year "
    + ", ".join(str(event["year"]) for event in events)
    + f", and the replacement costs {money(result['battery_replacement_cost_npv'], currency)} in present value."
    if events
    else "The pack lasts the whole project."
)
# sphinx_gallery_end_ignore

# %%
# .. sphinx-gallery drops a final code block whose code is all hidden, output
#    included; this closing text block keeps the summary above.
