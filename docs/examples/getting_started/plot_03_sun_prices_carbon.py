"""
Sun, prices or grid carbon?
===========================

Move the quickstart home from Porto to Berlin. Berlin gets much less sun, so
does the same system do worse there? This page separates the three things a
site changes: the solar resource, the electricity prices and the carbon
intensity of the grid the system displaces.

Three runs of the quickstart file (:doc:`plot_01_first_home`):

- **Porto**: the file as it is.
- **Berlin, Porto prices and grid**: ``location = "berlin"`` only, so just
  the weather and the tilt change.
- **Berlin**: also ``cost_preset = "residential_de"`` and
  ``emissions_country = "DE"``.

.. literalinclude:: /../configs/examples/quickstart.toml
   :language: toml
   :caption: configs/examples/quickstart.toml

Tilt is not set, so BREOS estimates it from the latitude for each site. The
cost presets are packaged examples, not current offers.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gallery_results import load_case, money, say, table

from breos.plotting import plot_breakeven_comparison

case = load_case("sun_prices_carbon")
sites = case.json("sites.json")
porto, berlin_sun, berlin = sites
currency = porto["provenance"]["currency"]
case.stamp()

# %%
# Side by side
# ------------

table(
    pd.DataFrame(
        {
            "Run": [s["site"] for s in sites],
            "Tilt (°)": [s["tilt"] for s in sites],
            "Usable PV (kWh/yr)": [s["usable_ac_system_production_kwh"] for s in sites],
            "Grid independence (%)": [s["grid_independence_pct"] for s in sites],
            f"Import price ({currency}/kWh)": [s["electricity_cost"] for s in sites],
            f"Export price ({currency}/kWh)": [s["electricity_sold_cost"] for s in sites],
            "Investment": [s["total_investment"] for s in sites],
            "NPV savings": [s["npv_savings"] for s in sites],
            "Payback year": [s["payback_year"] for s in sites],
            "Grid carbon (gCO2/kWh)": [s["grid_carbon"]["average_grid_carbon_intensity_gco2_kwh"] for s in sites],
            "CO2 avoided (t)": [s["co2_avoided_total_lifetime_kg"] / 1000 for s in sites],
        }
    ),
    **{"Tilt (°)": ".0f", "Usable PV (kWh/yr)": ",.0f", "Investment": ",.0f", "NPV savings": ",.0f",
       "CO2 avoided (t)": ".1f"},
)  # fmt: skip

# %%
# Less sun
# --------

drop = 1 - berlin_sun["usable_ac_system_production_kwh"] / porto["usable_ac_system_production_kwh"]
say(
    f"The same array delivers {drop * 100:.0f} % less usable energy in Berlin "
    f"({berlin_sun['usable_ac_system_production_kwh']:,.0f} against "
    f"{porto['usable_ac_system_production_kwh']:,.0f} kWh). At Porto's prices, NPV savings fall from "
    f"{money(porto['npv_savings'], currency)} to {money(berlin_sun['npv_savings'], currency)} and payback moves "
    f"from year {porto['payback_year']} to year {berlin_sun['payback_year']}. Self-consumption rises, from "
    f"{porto['self_consumption_pct']:.0f} % to {berlin_sun['self_consumption_pct']:.0f} %, only because there is "
    "less surplus to export."
)

# %%

months = np.arange(1, 13)
labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(months - 0.2, porto["monthly_production_kwh"], 0.4, label="Porto", color="#e6a700")
ax.bar(months + 0.2, berlin_sun["monthly_production_kwh"], 0.4, label="Berlin", color="#3a6ea5")
ax.plot(months, porto["monthly_consumption_kwh"], "k_", ms=18, mew=2, label="Load")
ax.set_xticks(months, labels)
ax.set_ylabel("Usable PV energy (kWh)")
ax.set_title("Monthly usable PV energy, year 1")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

# %%
# Higher prices
# -------------
# With the German preset the same Berlin energy is worth more: the import
# price is higher, so every self-consumed kWh saves more, and exported energy
# earns more.

say(
    f"The German preset prices imports at {berlin['electricity_cost']:.4f} {currency}/kWh against "
    f"{porto['electricity_cost']:.4f}, and pays {berlin['electricity_sold_cost']:.3f} against "
    f"{porto['electricity_sold_cost']:.3f} for exports. The investment also rises, to "
    f"{money(berlin['total_investment'], currency)}. Even so, NPV savings in Berlin reach "
    f"{money(berlin['npv_savings'], currency)}, "
    + (
        f"more than Porto's {money(porto['npv_savings'], currency)}"
        if berlin["npv_savings"] > porto["npv_savings"]
        else f"against Porto's {money(porto['npv_savings'], currency)}"
    )
    + f", and payback comes in year {berlin['payback_year']}. Prices, not sunshine, decide this comparison."
    if berlin["npv_savings"] > porto["npv_savings"]
    else "Prices do not make up for the lower yield here."
)

# %%

fig = plot_breakeven_comparison(sites, [s["site"] for s in sites])

# %%
# A dirtier grid
# --------------
# Avoided CO2 is the PV energy used or exported times the grid's average
# carbon intensity. A kWh of PV avoids more where the grid burns more fossil
# fuel.

fig, ax = plt.subplots(figsize=(7, 3.5))
names = [s["site"] for s in sites]
values = [s["co2_avoided_total_lifetime_kg"] / 1000 for s in sites]
bars = ax.barh(names, values, color=["#e6a700", "#7f9fc4", "#3a6ea5"])
ax.bar_label(bars, fmt="%.1f t", padding=3)
ax.invert_yaxis()
ax.set_xlabel("CO2 avoided over the project (t)")
ax.set_xlim(0, max(values) * 1.2)
fig.tight_layout()

# %%

say(
    f"On Porto's grid ({porto['grid_carbon']['average_grid_carbon_intensity_gco2_kwh']:.0f} gCO2/kWh) the "
    f"Berlin system would avoid {berlin_sun['co2_avoided_total_lifetime_kg'] / 1000:.1f} t of CO2 over the "
    f"project, less than Porto's {porto['co2_avoided_total_lifetime_kg'] / 1000:.1f} t. On the German grid "
    f"({berlin['grid_carbon']['average_grid_carbon_intensity_gco2_kwh']:.0f} gCO2/kWh, "
    f"{berlin['grid_carbon']['source']} {berlin['grid_carbon']['year']}) it avoids "
    f"{berlin['co2_avoided_total_lifetime_kg'] / 1000:.1f} t: less sun, but each kWh displaces dirtier power."
)
