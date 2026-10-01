"""
How much does orientation matter?
=================================

A roof rarely faces due south at the ideal tilt. How much yield and money
does a less than ideal orientation cost? This page sweeps the PV-only
example over tilt and azimuth and maps the annual usable energy and the NPV
savings.

.. literalinclude:: /../configs/examples/pv-only.toml
   :language: toml
   :caption: configs/examples/pv-only.toml

Each run sets ``tilt`` and ``azimuth`` (180° is south, 90° east, 270° west).
From the command line the same grid is a sweep:

.. code-block:: toml

    [sweep]
    tilt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    azimuth = [90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270]

and ``breos sweep --config config.toml --output orientation.csv``, which
:func:`breos.plotting.plot_orientation_landscape` reads directly.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
from gallery_results import load_case, money, say

from breos.plotting import plot_orientation_landscape

case = load_case("orientation")
grid = case.csv("orientations.csv")
default = case.json("default.json")
currency = case.manifest["currency"]
case.stamp()

# %%
# Annual usable energy
# --------------------

fig = plot_orientation_landscape(grid, "usable_ac_system_production_kwh")

# %%
# NPV savings
# -----------

fig = plot_orientation_landscape(grid, "npv_savings", currency=currency)

# %%

best = grid.loc[grid["usable_ac_system_production_kwh"].idxmax()]
share = grid["usable_ac_system_production_kwh"] / best["usable_ac_system_production_kwh"]
near = grid[share >= 0.95]
east = grid[(grid["azimuth"] == 90) & (grid["tilt"] == 30)]
flat = grid[grid["tilt"] == 0].iloc[0]
say(
    f"The most productive orientation is {best['tilt']:.0f}° tilt at {best['azimuth']:.0f}° azimuth, with "
    f"{best['usable_ac_system_production_kwh']:,.0f} kWh a year. BREOS's default for this site, "
    f"{default['tilt']:.1f}° facing {default['azimuth']:.0f}°, gives "
    f"{default['usable_ac_system_production_kwh']:,.0f} kWh.",
    f"The optimum is broad: {len(near)} of the {len(grid)} orientations reach at least 95 % of the best, "
    f"spanning tilts {near['tilt'].min():.0f}–{near['tilt'].max():.0f}° and azimuths "
    f"{near['azimuth'].min():.0f}–{near['azimuth'].max():.0f}°. A flat array gives "
    f"{(flat['usable_ac_system_production_kwh'] / best['usable_ac_system_production_kwh']) * 100:.0f} % of the best"
    + (
        f", and a 30° east-facing roof {(east['usable_ac_system_production_kwh'].iloc[0] / best['usable_ac_system_production_kwh']) * 100:.0f} %."
        if len(east)
        else "."
    ),
    f"NPV savings range from {money(grid['npv_savings'].min(), currency)} to "
    f"{money(grid['npv_savings'].max(), currency)} over the grid, and self-consumption from "
    f"{grid['self_consumption_pct'].min():.1f} % to {grid['self_consumption_pct'].max():.1f} %: a lower yield "
    "is used a little better, because less of it is surplus.",
)
