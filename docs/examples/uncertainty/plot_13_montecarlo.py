"""
Monte Carlo over weather years and demand
=========================================

A single run on a typical meteorological year (TMY) gives one number. How
much could the outcome move with real weather and a household that uses more
or less than planned? ``breos montecarlo`` runs many 20-year trajectories.
For each project year it draws a historical weather year at random, and for
each trajectory it scales the demand by a random factor.

.. literalinclude:: /../configs/examples/montecarlo.toml
   :language: toml
   :caption: configs/examples/montecarlo.toml

The weather is Open-Meteo's historical record for Porto, fetched when the
results were generated and identified by its SHA-256 in the manifest; it is
not stored in the repository. To run the study yourself, fetch the history
(``python tools/fetch_weather.py historical --location porto --start 2005
--end 2024``) and run::

    breos montecarlo --config configs/examples/montecarlo.toml --plots

The Python API is :func:`breos.montecarlo.run_montecarlo`; see
:doc:`/getting-started/monte-carlo`.
"""

# %%
# The stored study
# ----------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
from gallery_results import load_case, money, say

from breos.plotting import (
    plot_breakeven_cdf,
    plot_montecarlo_final_soh_distribution,
    plot_montecarlo_npv_distribution,
)

case = load_case("montecarlo")
runs = case.csv("runs.csv")
stored = case.json("summary.json")
summary, settings, tmy = stored["summary"], stored["settings"], stored["tmy_result"]
weather_years = case.csv("weather_years.csv")
currency = case.manifest["currency"]
case.stamp()

# %%

npv = summary["npv_savings"]
say(
    f"{settings['n_runs']} trajectories of {settings['years_per_run']} years, drawing from "
    f"{len(stored['available_years'])} complete weather years ({stored['available_years'][0]}–"
    f"{stored['available_years'][-1]}), with demand scaled by a {settings['load_distribution']} factor of standard "
    f"deviation {settings['load_uncertainty']:.0%} and seed {settings['seed']}.",
    f"Median NPV savings are {money(npv['p50'], currency)}; 90 % of the runs fall between "
    f"{money(npv['p5'], currency)} and {money(npv['p95'], currency)}. The same system on the PVGIS TMY saves "
    f"{money(tmy['npv_savings'], currency)}, "
    + (
        "above every Monte Carlo run."
        if tmy["npv_savings"] > npv["max"]
        else f"which sits at the {(runs['npv_savings'] < tmy['npv_savings']).mean():.0%} quantile of the runs."
    ),
)

# %%
# NPV savings
# -----------

fig = plot_montecarlo_npv_distribution(runs)

# %%
# Payback
# -------
# The share of runs that have paid back by each year. Runs that never pay
# back within the horizon keep the curve below 100 %.

fig = plot_breakeven_cdf(runs["payback_year_interpolated"].tolist(), total_runs=len(runs))

# %%
# Battery health at the end
# -------------------------

fig = plot_montecarlo_final_soh_distribution(runs)

# %%
# Why the TMY run looks better
# ----------------------------
# A TMY is assembled from typical months of a longer record, not from one
# real year, and its source differs from the historical record here (PVGIS
# satellite data against Open-Meteo reanalysis). Annual global horizontal
# irradiation of each historical year against the TMY:

used = weather_years[weather_years["year"].isin(stored["available_years"])]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(used["year"], used["ghi_kwh_m2"], color="#7f9fc4", label="Open-Meteo historical year")
ax.axhline(stored["tmy_ghi_kwh_m2"], color="#e6a700", lw=2, label="PVGIS TMY")
ax.set_ylim(used["ghi_kwh_m2"].min() * 0.9, max(used["ghi_kwh_m2"].max(), stored["tmy_ghi_kwh_m2"]) * 1.03)
ax.set_ylabel("GHI (kWh/m²/yr)")
ax.set_title("Annual global horizontal irradiation, Porto")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

# %%

sunnier = int((used["ghi_kwh_m2"] < stored["tmy_ghi_kwh_m2"]).sum())
say(
    f"The TMY receives {stored['tmy_ghi_kwh_m2']:,.0f} kWh/m², more than {sunnier} of the {len(used)} historical "
    f"years (mean {used['ghi_kwh_m2'].mean():,.0f} kWh/m²). The spread across weather years and demand is "
    f"{money(npv['p95'] - npv['p5'], currency)} between the 5th and 95th percentiles, and the TMY run is "
    f"{money(tmy['npv_savings'] - npv['p50'], currency)} above the median. Before trusting the level of a result, "
    "check the weather source "
    "against local measurements; the Monte Carlo spread says how much the year-to-year weather and the demand "
    "move it."
)
