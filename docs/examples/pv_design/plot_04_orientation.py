"""
Array tilt and azimuth: annual yield and NPV
============================================

The annual usable energy and the NPV savings of the PV-only example over
tilt and azimuth, under two sky and optics models. The first is BREOS's
compatible default: isotropic sky transposition, the ASHRAE incidence-angle
modifier on beam irradiance only and the sun position at the start of each
interval. The second is the PV-model starting point recommended in
:doc:`/getting-started/configuration`: the Perez sky model, the physical
incidence-angle modifier, Marion's diffuse modifier and the sun position the
weather file's timing metadata gives.

.. literalinclude:: /../configs/examples/pv-only.toml
   :language: toml
   :caption: configs/examples/pv-only.toml

The second model adds these keys:

.. code-block:: toml

    transposition_model = "perez"
    iam_model = "physical"
    diffuse_iam = "marion"
    solar_position = "weather"

Each run sets ``tilt`` and ``azimuth`` (180° is south, 90° east, 270° west).
The map is a 5° grid; a 1° grid around the optimum finds the best orientation
more precisely. From the command line the same map is a sweep:

.. code-block:: toml

    [sweep]
    tilt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    azimuth = [
        90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
        155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215,
        220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270,
    ]

and ``breos sweep --config config.toml --output orientation.csv``, which
:func:`breos.plotting.plot_orientation_landscape` reads directly.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 2
import pandas as pd
from gallery_results import load_case, money, number, say, table

from breos.plotting import plot_orientation_landscape

case = load_case("orientation")
grid = case.csv("orientations.csv")
fine = case.csv("fine.csv")
defaults = case.json("default.json")
currency = case.manifest["currency"]
labels = {"default": "Default model (isotropic)", "perez_marion": "Perez sky, Marion diffuse IAM"}
case.stamp()

# %%
# Annual usable energy, default model
# -----------------------------------

fig = plot_orientation_landscape(grid[grid["model"] == "default"], "usable_ac_system_production_kwh")
fig.suptitle(labels["default"], y=1.02)

# %%
# Annual usable energy, Perez sky and Marion diffuse IAM
# ------------------------------------------------------
# The Perez model keeps part of the diffuse light concentrated around the sun
# and near the horizon, where the isotropic model spreads it evenly over the
# sky. A tilted array facing the sun collects more of it, so the yield rises
# and the best tilt moves up.

fig = plot_orientation_landscape(grid[grid["model"] == "perez_marion"], "usable_ac_system_production_kwh")
fig.suptitle(labels["perez_marion"], y=1.02)

# %%
# NPV savings, Perez sky and Marion diffuse IAM
# ---------------------------------------------

fig = plot_orientation_landscape(grid[grid["model"] == "perez_marion"], "npv_savings", currency=currency)
fig.suptitle(labels["perez_marion"], y=1.02)

# %%
# The best orientation and a practical one
# ----------------------------------------
# Installers set a roof's tilt and azimuth to round values, not to the exact
# optimum. The table compares the best orientation on the 1° grid with the
# nearest 5° step, 35° facing due south, and with BREOS's default
# orientation for the site (the latitude tilt facing the equator).

rows = []
for model, label in labels.items():
    runs = fine[fine["model"] == model]
    best = runs.loc[runs["usable_ac_system_production_kwh"].idxmax()]
    rounded = runs[(runs["tilt"] == 35) & (runs["azimuth"] == 180)].iloc[0]
    default = defaults[model]
    for name, run in (
        ("Best on the 1° grid", best),
        ("35°, due south", rounded),
        (f"Default, {default['tilt']:.1f}° due south", default),
    ):
        rows.append(
            {
                "Model": label,
                "Orientation": name,
                "Tilt (°)": run["tilt"],
                "Azimuth (°)": run["azimuth"],
                "Usable AC (kWh/yr)": run["usable_ac_system_production_kwh"],
                "Of the best (%)": 100 * run["usable_ac_system_production_kwh"]
                / best["usable_ac_system_production_kwh"],
                "NPV savings": run["npv_savings"],
            }
        )
orientations = pd.DataFrame(rows)
table(orientations, **{"Tilt (°)": "g", "Azimuth (°)": "g", "Usable AC (kWh/yr)": ".0f", "Of the best (%)": ".2f",
                       "NPV savings": ".0f"})  # fmt: skip

# %%

# sphinx_gallery_start_ignore
lines = []
for label, group in orientations.groupby("Model", sort=False):
    best, rounded = group.iloc[0], group.iloc[1]
    lines.append(
        f"{label}: the best orientation is {best['Tilt (°)']:g}° tilt at {best['Azimuth (°)']:g}° azimuth, "
        f"{number(best['Usable AC (kWh/yr)'])} kWh a year. Rounding it to 35° due south costs "
        f"{number(best['Usable AC (kWh/yr)'] - rounded['Usable AC (kWh/yr)'])} kWh "
        f"({100 - rounded['Of the best (%)']:.2f} %) and changes NPV savings by "
        f"{money(rounded['NPV savings'] - best['NPV savings'], currency)}."
    )
default_best, perez_best = (orientations[orientations["Model"] == label].iloc[0] for label in labels.values())
lines.append(
    f"The Perez sky and Marion diffuse modifier give "
    f"{number(perez_best['Usable AC (kWh/yr)'] - default_best['Usable AC (kWh/yr)'])} kWh "
    f"({(perez_best['Usable AC (kWh/yr)'] / default_best['Usable AC (kWh/yr)'] - 1) * 100:.1f} %) more at their "
    "optimum than the default model at its own. The choice of model moves the yield more than the choice "
    "between the exact optimum and a rounded installation."
)
say(*lines)
# sphinx_gallery_end_ignore

# %%
# How broad the optimum is
# ------------------------

# sphinx_gallery_start_ignore
perez = grid[grid["model"] == "perez_marion"]
top = perez["usable_ac_system_production_kwh"].max()
near = perez[perez["usable_ac_system_production_kwh"] >= 0.95 * top]
flat = perez[perez["tilt"] == 0].iloc[0]
east = perez[(perez["azimuth"] == 90) & (perez["tilt"] == 30)]
say(
    f"With the Perez model, {len(near)} of the {len(perez)} orientations on the 5° map reach at least 95 % of "
    f"the best, spanning tilts {near['tilt'].min():.0f}–{near['tilt'].max():.0f}° and azimuths "
    f"{near['azimuth'].min():.0f}–{near['azimuth'].max():.0f}°. A flat array gives "
    f"{flat['usable_ac_system_production_kwh'] / top * 100:.0f} % of the best"
    + (f", and a 30° east-facing roof {east['usable_ac_system_production_kwh'].iloc[0] / top * 100:.0f} %." if len(east) else "."),
    f"NPV savings range from {money(perez['npv_savings'].min(), currency)} to "
    f"{money(perez['npv_savings'].max(), currency)} over the map, and self-consumption from "
    f"{perez['self_consumption_pct'].min():.1f} % to {perez['self_consumption_pct'].max():.1f} %: a lower yield "
    "is used a little better, because less of it is surplus.",
)
# sphinx_gallery_end_ignore

# %%
# .. sphinx-gallery drops a final code block whose code is all hidden, output
#    included; this closing text block keeps the summary above.
