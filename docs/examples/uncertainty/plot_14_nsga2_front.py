"""
The NSGA-II sizing front
========================

Instead of simulating a design you chose, let the optimizer search for
designs. :func:`breos.optimization.optimize_system_multi_objective` varies
module count, battery capacity, tilt and azimuth with NSGA-II and scores
every candidate over the whole project, with PV degradation, battery ageing
and replacements. It returns the trade-off front between two objectives:
projected lifetime grid independence and projected NPV savings. No design on
the front beats another in both.

The optimizer takes a nested config, not the flat ``App`` config, and the
weather and the load as DataFrames:

.. literalinclude:: /../configs/optimization/projected-optimization.toml
   :language: toml
   :caption: configs/optimization/projected-optimization.toml

The stored front used the committed Porto PVGIS TMY and the bundled H0 load
profile scaled to the annual consumption in the manifest. See
:doc:`/getting-started/optimization` for loading both and running the search.
"""

# %%
# The stored front
# ----------------

# sphinx_gallery_thumbnail_number = 1
from gallery_results import load_case, money, say, table

from breos.plotting import plot_pareto_front

case = load_case("nsga2_front")
front = case.csv("pareto.csv")
manifest = case.manifest
currency = manifest.get("currency", "EUR")
constraints = manifest["constraints"]
case.stamp()

# %%

say(
    f"{len(front)} designs on the front after {manifest['generations']} generations, for a household using "
    f"{manifest['annual_consumption_kwh']:,.0f} kWh a year. Constraints: a budget of "
    f"{money(constraints['budget'], currency)}, at most {constraints['max_area_m2']:g} m² of roof, "
    f"{constraints['max_modules']} modules and {constraints['max_battery_kwh']:g} kWh of storage."
)

# %%
# Independence against money
# --------------------------
# Each point is a design, coloured by its battery size.

fig = plot_pareto_front(front, x="Projected_Grid_Independence_%", y="Projected_NPV", color_by="Battery_kWh",
                        currency=currency)  # fmt: skip

# %%

best_npv = front.loc[front["Projected_NPV"].idxmax()]
most_independent = front.loc[front["Projected_Grid_Independence_%"].idxmax()]
positive = front[front["Projected_NPV"] > 0]
best_positive = positive.loc[positive["Projected_Grid_Independence_%"].idxmax()] if len(positive) else None
max_by_area = int(constraints["max_area_m2"] // manifest["module_area_m2"])
say(
    f"The best NPV, {money(best_npv['Projected_NPV'], currency)}, comes from {best_npv['Modules']:.0f} modules and "
    f"{best_npv['Battery_kWh']:g} kWh at {best_npv['Tilt']:g}° tilt, with "
    f"{best_npv['Projected_Grid_Independence_%']:.1f} % grid independence over the project.",
    f"Every kWh of storage buys independence at a price: the most independent design, "
    f"{most_independent['Modules']:.0f} modules and {most_independent['Battery_kWh']:g} kWh, reaches "
    f"{most_independent['Projected_Grid_Independence_%']:.1f} % and NPV savings of "
    f"{money(most_independent['Projected_NPV'], currency)}."
    + (
        f" The most independent design that still saves money has {best_positive['Battery_kWh']:g} kWh "
        f"({best_positive['Projected_Grid_Independence_%']:.1f} %, "
        f"{money(best_positive['Projected_NPV'], currency)})."
        if best_positive is not None
        else ""
    ),
    f"No design has more than {front['Modules'].max():.0f} modules, and {constraints['max_area_m2']:g} m² of roof "
    f"holds {max_by_area} modules of {manifest['module_area_m2']:.2f} m²"
    + (
        ": the roof, not the budget, limits the PV here."
        if front["Modules"].max() == max_by_area and front["Projected_Initial_Cost"].max() < constraints["budget"]
        else "."
    ),
)

# %%
# Money against CO2
# -----------------
# The front optimizes NPV and independence only, but every design also
# carries its projected avoided CO2.

fig = plot_pareto_front(front, x="Projected_CO2_Avoided_Total_kg", y="Projected_NPV", color_by="Battery_kWh",
                        currency=currency)  # fmt: skip

# %%
# The designs
# -----------

columns = {
    "Modules": "Modules",
    "Battery_kWh": "Battery (kWh)",
    "Tilt": "Tilt (°)",
    "Azimuth": "Azimuth (°)",
    "Projected_Grid_Independence_%": "Grid independence (%)",
    "Projected_NPV": "NPV savings",
    "Projected_Initial_Cost": "Investment",
    "Projected_Payback_Year": "Payback year",
}
table(
    front[list(columns)].rename(columns=columns),
    **{"Battery (kWh)": "g", "Tilt (°)": "g", "Azimuth (°)": "g", "Grid independence (%)": ".1f",
       "NPV savings": ",.0f", "Investment": ",.0f", "Payback year": ".0f"},
)  # fmt: skip
