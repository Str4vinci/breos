"""
Vehicle-to-home on a bi-hourly tariff in Porto
============================================================

A Porto household has PV, a 5 kWh stationary battery and a 40 kWh vehicle.
The house uses the bundled H0 profile at 4000 kWh/year. The vehicle leaves
on weekday and weekend trips. This page compares uncontrolled charging
without a system (R0), V1G with the system (R1), and V2H with the same
system (R2).

This is a short **three-year study**, at hourly resolution, to keep the
stored case quick to regenerate. It is not a full vehicle-lifetime or
hardware-payback study. The Portuguese bi-hourly schedule is bundled;
energy prices, the **2500 EUR installed charger premium** and the
**120 EUR/kWh pack-only price** are explicit illustrative inputs, not
BREOS presets, market quotes or product data. The charger and pack
efficiencies are also example inputs. Replace them for a real study.

.. literalinclude:: /../configs/examples/vehicle-to-home-portugal.toml
   :language: toml
   :caption: configs/examples/vehicle-to-home-portugal.toml

At hourly steps, the 07:45 weekday departure rounds down to 07:00, and
the 18:30 arrival rounds up to 19:00. The template uses half the gross
trip energy in each end leg and parks between them. It is an approximation
of a commute. This case has no away charging or outside-charging penalty.
The weather is the committed Porto PVGIS TMY; the load and tariff calendar
replay over the projection. No weather is fetched and no simulation runs
during the docs build.
"""

# %%
# The stored households
# ---------------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gallery_results import load_case, money, number, say, table

case = load_case("vehicle_to_home")
households = case.csv("households.csv").set_index("household")
comparisons = case.json("comparisons.json")
currency = case.manifest["currency"]
case.stamp()

# %%
# Home energy in year 1
# ----------------------
# The stationary battery dispatches first. The vehicle uses only the
# remaining PV surplus or house deficit. V1G never discharges; V2H serves
# the house during peak periods while its departure target stays reachable.
# R0 has no PV equipment or inverter night draw.

positions = np.arange(len(households))
labels = [f"{name}\n{row['mode']}" for name, row in households.iterrows()]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, columns, title in (
    (axes[0], [("grid_import_year1_kwh", "Home import"), ("grid_export_year1_kwh", "Home export")], "Site meter"),
    (
        axes[1],
        [
            ("pv_ac_charge_kwh", "PV to vehicle"),
            ("grid_ac_charge_kwh", "Grid to vehicle"),
            ("ac_to_load_kwh", "Vehicle to house"),
        ],
        "Vehicle home AC flows",
    ),
):
    width = 0.8 / len(columns)
    for i, (column, label) in enumerate(columns):
        ax.bar(positions + (i - (len(columns) - 1) / 2) * width, households[column], width, label=label)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Energy, year 1 (kWh)")
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

# %%

table(
    pd.DataFrame(
        {
            "Household": households.index,
            "Home bill, year 1 (EUR)": households["home_bill_year1"].to_numpy(),
            "Driving served (kWh)": households["drive_served_kwh"].to_numpy(),
            "Driving unserved (kWh)": households["drive_unserved_kwh"].to_numpy(),
            "Short departures": households["departures_short"].to_numpy(),
        }
    ),
    **{
        "Home bill, year 1 (EUR)": ".2f",
        "Driving served (kWh)": ".0f",
        "Driving unserved (kWh)": ".0f",
        "Short departures": ".0f",
    },
)

# %%
# The bill includes import and fixed charges, less export revenue and any
# network credit. Driving columns and departure counts above also cover
# year 1; they describe transport service, not house demand.

# sphinx_gallery_start_ignore
r0, r1, r2 = (households.loc[key] for key in ("R0", "R1", "R2"))
say(
    f"In year 1, V2H delivers {number(r2['ac_to_load_kwh'])} kWh to the house. "
    f"Home imports are {number(r0['grid_import_year1_kwh'])} kWh for R0, "
    f"{number(r1['grid_import_year1_kwh'])} kWh for R1 and {number(r2['grid_import_year1_kwh'])} kWh for R2. "
    "These imports include home vehicle charging.",
    f"R2 minus R1 changes home import by {number(r2['grid_import_year1_kwh'] - r1['grid_import_year1_kwh'])} kWh "
    f"and the home bill by {number(r2['home_bill_year1'] - r1['home_bill_year1'], 2)} {currency}. "
    "More imported energy can still cost less when charging moves to a cheaper period; conversion losses also matter.",
)
# sphinx_gallery_end_ignore

# %%
# Project costs and pack wear
# ---------------------------
# The ordinary system NPV uses R0's actual charging bill. Household-adjusted
# NPV includes R2/R0 vehicle wear and away/unserved cost differences. V2H
# benefit compares R2 with R1 on the same system, charging the installed
# premium once. It does not charge PV and stationary storage again.

adjusted, benefit = comparisons["household_adjusted_npv"], comparisons["v2h_benefit"]
values = [comparisons["system_npv"], adjusted["npv"], benefit["npv"]]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(["System NPV\nR2 against R0", "Household-adjusted NPV\nR2 against R0", "V2H benefit\nR2 against R1"], values)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel(f"Present value over three years ({currency})")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

# %%

table(
    pd.DataFrame(
        {
            "Household": households.index,
            "Final health (%)": households["final_soh_pct"].to_numpy(),
            "Pack swaps": households["replacements"].to_numpy(),
            "Replacement outlays P (EUR NPV)": households["replacement_outlays_npv"].to_numpy(),
            "Terminal pack credit H (EUR NPV)": households["terminal_pack_credit_npv"].to_numpy(),
        }
    ),
    **{
        "Final health (%)": ".2f",
        "Pack swaps": ".0f",
        "Replacement outlays P (EUR NPV)": ".2f",
        "Terminal pack credit H (EUR NPV)": ".2f",
    },
)

# %%
# Wear is P2 − P1 + H1 − H2. Credits use pack-only value above the study's
# replacement threshold; they are clipped individually before differences
# are formed. The outlays-only difference is reported separately. A
# same-SOC pack swap can occur at home or away. The threshold and the
# keep/retire rules are study policy, not a prediction of owner behaviour.

# sphinx_gallery_start_ignore
assert r0["pv_ac_charge_kwh"] == r0["grid_export_year1_kwh"] == 0
assert r0["ac_to_load_kwh"] == r1["ac_to_load_kwh"] == 0 < r2["ac_to_load_kwh"]
assert (households[["departures_short", "drive_unserved_kwh", "away_purchased_kwh"]] == 0).all().all()
assert np.allclose(households["drive_served_kwh"], r0["drive_served_kwh"])
assert adjusted["comparable"] and benefit["comparable"]
wear = (
    r2["replacement_outlays_npv"]
    - r1["replacement_outlays_npv"]
    + r1["terminal_pack_credit_npv"]
    - r2["terminal_pack_credit_npv"]
)
assert np.isclose(wear, benefit["wear_npv"])
assert np.isclose(
    benefit["npv"], r1["operating_cost_npv"] - r2["operating_cost_npv"] - benefit["charger_premium"] - wear
)
say(
    f"Over three years, system NPV is {money(comparisons['system_npv'], currency)}; "
    f"household-adjusted NPV is {money(adjusted['npv'], currency)}. "
    f"V2H benefit against V1G is {money(benefit['npv'], currency)}, including "
    f"{money(benefit['wear_npv'], currency)} of additional vehicle wear and the "
    f"{money(benefit['charger_premium'], currency)} charger premium. "
    f"The outlays-only difference is {money(benefit['replacement_outlays_difference_npv'], currency)}.",
    "Both comparisons are comparable: no scheduled driving is left unserved. "
    "The short horizon carries the whole installed premium up front; these figures do not establish lifetime payback.",
)
# sphinx_gallery_end_ignore

# %%
# Model limits
# -------------
# Charger conversion uses the PVWatts part-load curve in both directions;
# using its discharge curve for charging is an assumption. Ambient weather
# is the vehicle's temperature proxy, without a thermal-management model.
# The native LFP cell ageing fit does not validate this vehicle pack. Trip
# timing, equipment efficiencies, pack prices and replacement policy all
# need scrutiny before applying the comparison to a real household.
# See :doc:`/getting-started/configuration` and
# :doc:`/getting-started/interpreting-results` for the model and result fields.
