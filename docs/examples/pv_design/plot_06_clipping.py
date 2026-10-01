"""
DC/AC ratio and clipping
========================

An inverter smaller than the array's DC rating is cheaper, and it loses only
the few hours when the array would deliver more than the inverter can pass.
How far can the DC/AC ratio go before clipping costs real energy, and does a
battery on the DC side catch what the inverter clips? This page runs the
quickstart home at 15-minute resolution, where short peaks are not averaged
away, with ``inverter_loading_ratio`` from 1.0 to 2.0, without and with the
battery.

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
from gallery_results import load_case, money, say, table

case = load_case("clipping")
ratios = case.csv("ratios.csv")
week = case.week("week_summer.csv", case.manifest["timezone"])
currency = case.manifest["currency"]
case.stamp()

# %%
# Clipped energy and NPV against the ratio
# ----------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for battery, group in ratios.groupby("battery_kwh"):
    label = "PV only" if battery == 0 else f"{battery:g} kWh battery"
    axes[0].plot(group["inverter_loading_ratio"], group["curtailment_dc_kwh"], "o-", label=label)
    axes[1].plot(group["inverter_loading_ratio"], group["npv_savings"], "o-", label=label)
axes[0].set_ylabel("Clipped DC energy, year 1 (kWh)")
axes[1].set_ylabel(f"NPV savings ({currency})")
for ax in axes:
    ax.set_xlabel("DC/AC ratio")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()

# %%

table(
    ratios[["battery_kwh", "inverter_loading_ratio", "pv_dc_generation_kwh", "curtailment_dc_kwh",
            "usable_ac_system_production_kwh", "total_investment", "npv_savings"]].rename(
        columns={
            "battery_kwh": "Battery (kWh)",
            "inverter_loading_ratio": "DC/AC",
            "pv_dc_generation_kwh": "PV DC (kWh)",
            "curtailment_dc_kwh": "Clipped (kWh)",
            "usable_ac_system_production_kwh": "Usable AC (kWh)",
            "total_investment": "Investment",
            "npv_savings": "NPV savings",
        }
    ),
    **{"Battery (kWh)": "g", "DC/AC": "g", "PV DC (kWh)": ",.0f", "Clipped (kWh)": ",.1f",
       "Usable AC (kWh)": ",.0f", "Investment": ",.0f", "NPV savings": ",.0f"},
)  # fmt: skip

# %%

pv_only = ratios[ratios["battery_kwh"] == 0].set_index("inverter_loading_ratio")
battery = ratios[ratios["battery_kwh"] > 0].set_index("inverter_loading_ratio")
top = pv_only.index.max()
best = pv_only["npv_savings"].idxmax()
say(
    f"Without a battery, a ratio of {top:g} clips {pv_only.loc[top, 'curtailment_dc_kwh']:,.0f} kWh of "
    f"{pv_only.loc[top, 'pv_dc_generation_kwh']:,.0f} kWh DC in the first year "
    f"({(pv_only.loc[top, 'curtailment_dc_kwh'] / pv_only.loc[top, 'pv_dc_generation_kwh']) * 100:.1f} %). NPV savings are "
    f"highest at a ratio of {best:g}, and stay within {money(pv_only['npv_savings'].max() - pv_only['npv_savings'].min(), currency)} "
    "across the range.",
    f"With the battery, the same ratio clips {battery.loc[top, 'curtailment_dc_kwh']:,.0f} kWh"
    + (
        f": the battery takes {saved:,.0f} kWh of the clipped energy on the DC side, when it has room at the "
        "clipping hours."
        if (saved := pv_only.loc[top, "curtailment_dc_kwh"] - battery.loc[top, "curtailment_dc_kwh"]) > 0
        else ": the battery does not reduce clipping here."
    ),
)

# %%
# A summer week at the highest ratio
# ----------------------------------
# DC power the array produces and the part the inverter clips, PV-only, in
# 15-minute steps.

fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(week["Datetime"], week["PV_DC"] / 1000, color="#e6a700", alpha=0.4, label="PV DC")
ax.fill_between(week["Datetime"], week["PV_DC_Curtailed"] / 1000, color="#c0392b", alpha=0.8, label="Clipped")
ax.plot(week["Datetime"], week["PV_Production"] / 1000, color="#3a6ea5", lw=1, label="PV AC")
ax.set_ylabel("Power (kW)")
ax.set_title(f"DC/AC ratio {case.manifest['week_ratio']:g}, ISO week {case.manifest['iso_week']}")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
