"""
DC/AC ratio, inverter clipping and part-load efficiency
=======================================================

An inverter smaller than the array's DC rating costs less and, for most of
the year, runs closer to its rated power, where it converts more
efficiently. It loses the energy above its rating in the few hours when the
array would deliver more than it can pass. This page runs the quickstart
home at 15-minute resolution, where short peaks are not averaged away, with
``inverter_loading_ratio`` from 1.0 to 2.0, without and with the battery, and
shows how the two effects add up.

.. literalinclude:: /../configs/examples/quickstart.toml
   :language: toml
   :caption: configs/examples/quickstart.toml

The runs set ``resolution = "15min"`` and ``inverter_loading_ratio``. The
ratio sizes the inverter's AC rating from the array's DC rating and prices
it; ``inverter_ac_rating_kw`` sets the rating directly instead.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
import numpy as np
from gallery_results import load_case, money, number, say, table

from breos.inverter import calculate_dc_ac_power

case = load_case("clipping")
ratios = case.csv("ratios.csv")
distribution = case.csv("dc_power_distribution.csv")
week = case.week("week_summer.csv", case.manifest["timezone"])
currency = case.manifest["currency"]
efficiency = case.manifest["inverter_efficiency"]
pv_only = ratios[ratios["battery_kwh"] == 0].set_index("inverter_loading_ratio")
battery = ratios[ratios["battery_kwh"] > 0].set_index("inverter_loading_ratio")
case.stamp()

# %%
# Clipped energy and NPV against the ratio
# ----------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for kwh, group in ratios.groupby("battery_kwh"):
    label = "PV only" if kwh == 0 else f"{kwh:g} kWh battery"
    axes[0].plot(group["inverter_loading_ratio"], group["curtailment_dc_kwh"], "o-", label=label)
    axes[1].plot(group["inverter_loading_ratio"], group["npv_savings"], "o-", label=label)
axes[0].set_ylabel("Clipped DC energy, year 1 (kWh)")
axes[1].set_ylabel(f"NPV savings ({currency})")
for ax in axes:
    ax.set_xlabel("DC/AC ratio")
    ax.legend()
fig.tight_layout()

# %%
# The inverter's part-load efficiency
# -----------------------------------
# BREOS converts DC to AC with the PVWatts part-load curve, scaled so that the
# inverter reaches its nominal efficiency at its rated power. The curve falls
# steeply below about a fifth of the rating, is nearly flat above it and
# peaks near 60 %. The bars show where the array's year-1 DC energy falls on
# that curve for two ratios. A smaller inverter moves the same energy to the
# right: less of it is converted at the inefficient low loads, which outweighs
# the slightly lower efficiency near full load. Input above the rating
# (right of the dotted line) is clipped.

load = np.linspace(0.005, 1.0, 400)
curve = [calculate_dc_ac_power(x * 1000 / efficiency, 1000, efficiency).ac_power_w / (x * 1000 / efficiency)
         for x in load]  # fmt: skip
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.twinx()
width = distribution["dc_fraction_high"] - distribution["dc_fraction_low"]
for ratio, color in ((1.0, "C0"), (1.25, "C1")):
    pdc0 = pv_only.loc[ratio, "inverter_ac_kw"] / efficiency
    share = pv_only.loc[ratio, "pv_kwp"] / pdc0
    bars.bar(distribution["dc_fraction_low"] * share, distribution["energy_kwh"], width * share, align="edge",
             color=color, alpha=0.35, label=f"DC energy, ratio {ratio:g}")  # fmt: skip
ax.plot(load, 100 * np.array(curve), color="k", label="Conversion efficiency")
ax.axvline(1.0, color="0.4", ls=":")
ax.set_zorder(bars.get_zorder() + 1)
ax.patch.set_visible(False)
ax.set_xlim(0, 1.05)
ax.set_ylim(80, 100)
ax.set_xlabel("DC input as a fraction of the inverter's rated DC input")
ax.set_ylabel("Conversion efficiency (%)")
bars.set_ylabel("Year-1 DC energy per bin (kWh)")
bars.grid(False)
handles = ax.get_legend_handles_labels()[0] + bars.get_legend_handles_labels()[0]
ax.legend(handles=handles, loc="lower right")
fig.tight_layout()

# %%

table(
    ratios[["battery_kwh", "inverter_loading_ratio", "inverter_ac_kw", "curtailment_dc_kwh",
            "inverter_conversion_loss_kwh", "usable_ac_system_production_kwh", "total_investment",
            "npv_savings"]].rename(
        columns={
            "battery_kwh": "Battery (kWh)",
            "inverter_loading_ratio": "DC/AC",
            "inverter_ac_kw": "Inverter (kW AC)",
            "curtailment_dc_kwh": "Clipped (kWh)",
            "inverter_conversion_loss_kwh": "Conversion loss (kWh)",
            "usable_ac_system_production_kwh": "Usable AC (kWh)",
            "total_investment": "Investment",
            "npv_savings": "NPV savings",
        }
    ),
    **{"Battery (kWh)": "g", "DC/AC": "g", "Inverter (kW AC)": ".2f", "Clipped (kWh)": ".1f",
       "Conversion loss (kWh)": ".0f", "Usable AC (kWh)": ".0f", "Investment": ".0f", "NPV savings": ".0f"},
)  # fmt: skip

# %%

# sphinx_gallery_start_ignore
low, step = pv_only.index.min(), pv_only.index[1]
top = pv_only.index.max()
best = pv_only["npv_savings"].idxmax()
say(
    f"From a ratio of {low:g} to {step:g}, without a battery, the inverter shrinks from "
    f"{pv_only.loc[low, 'inverter_ac_kw']:.2f} to {pv_only.loc[step, 'inverter_ac_kw']:.2f} kW AC. It clips "
    f"{pv_only.loc[step, 'curtailment_dc_kwh']:.1f} kWh in the first year, but its conversion loss falls by "
    f"{number(pv_only.loc[low, 'inverter_conversion_loss_kwh'] - pv_only.loc[step, 'inverter_conversion_loss_kwh'])} "
    f"kWh, so the usable AC energy changes by "
    f"{pv_only.loc[step, 'usable_ac_system_production_kwh'] - pv_only.loc[low, 'usable_ac_system_production_kwh']:+.0f} "
    f"kWh. The investment falls by "
    f"{money(pv_only.loc[low, 'total_investment'] - pv_only.loc[step, 'total_investment'], currency)}. Together "
    f"they raise NPV savings by {money(pv_only.loc[step, 'npv_savings'] - pv_only.loc[low, 'npv_savings'], currency)}.",
    f"Beyond that, clipping grows faster than the savings: a ratio of {top:g} clips "
    f"{number(pv_only.loc[top, 'curtailment_dc_kwh'])} kWh of {number(pv_only.loc[top, 'pv_dc_generation_kwh'])} "
    f"kWh DC ({pv_only.loc[top, 'curtailment_dc_kwh'] / pv_only.loc[top, 'pv_dc_generation_kwh'] * 100:.1f} %). "
    f"Without a battery NPV savings are highest at a ratio of {best:g}; with the battery at "
    f"{battery['npv_savings'].idxmax():g}.",
    f"With the battery, a ratio of {top:g} clips {number(battery.loc[top, 'curtailment_dc_kwh'])} kWh"
    + (
        f": the battery takes {number(saved)} kWh of the clipped energy on the DC side, when it has room at the "
        "clipping hours."
        if (saved := pv_only.loc[top, "curtailment_dc_kwh"] - battery.loc[top, "curtailment_dc_kwh"]) > 0
        else ": the battery does not reduce clipping here."
    ),
)
# sphinx_gallery_end_ignore

# %%
# A summer week at the highest ratio
# ----------------------------------
# DC power the array produces, PV-only, in 15-minute steps. The dashed line
# is the inverter's rated DC input; the hatched area above it is the clipped
# energy.

limit = pv_only.loc[case.manifest["week_ratio"], "inverter_ac_kw"] / efficiency
dc = week["PV_DC"] / 1000
used = dc - week["PV_DC_Curtailed"] / 1000
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(week["Datetime"], used, color="C3", alpha=0.4, lw=0, label="PV DC converted")
ax.fill_between(week["Datetime"], used, dc, where=dc > used, facecolor="none", edgecolor="C2", hatch="////", lw=0,
                label="Clipped")  # fmt: skip
ax.axhline(limit, color="C2", ls="--", lw=1, label="Inverter rated DC input")
ax.plot(week["Datetime"], week["PV_Production"] / 1000, color="C0", lw=1, label="PV AC")
ax.set_ylim(bottom=0)
ax.set_ylabel("Power (kW)")
ax.set_title(f"DC/AC ratio {case.manifest['week_ratio']:g}, ISO week {case.manifest['iso_week']}")
ax.legend(loc="upper left", ncol=4)
fig.autofmt_xdate()
fig.tight_layout()
