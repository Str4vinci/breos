"""
Battery size and storage price against PV only
==============================================

A battery raises how much of its own PV a household uses, but it also costs
money and wears out. This page runs the quickstart home
(:doc:`plot_01_first_home`) with no battery, a 5 kWh battery and a 10 kWh
battery, then finds the storage price below which each battery beats PV only,
using :meth:`breos.App.revalue` to re-price each battery run without
simulating it again.

The configuration is the quickstart file with only ``battery_kwh`` changed:

.. literalinclude:: /../configs/examples/quickstart.toml
   :language: toml
   :caption: configs/examples/quickstart.toml

A run with ``battery_kwh = 0`` is a PV-only system: the investment, payback and
NPV are those of the PV alone, and the battery keys are left out of the
result. To compare sizes from the command line, sweep ``battery_kwh``:

.. code-block:: toml

    [sweep]
    battery_kwh = [0.0, 5.0, 10.0]

and run ``breos sweep --config config.toml --output sweep.csv``.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gallery_results import load_case, money, number, say, table

from breos.plotting import plot_breakeven_comparison

case = load_case("battery_worth")
designs = case.json("designs.json")
prices = case.csv("storage_prices.csv")
currency = designs[0]["provenance"]["currency"]
preset_price = case.manifest["preset_storage_cost_per_kwh"]
case.stamp()

# %%
# Three battery sizes
# -------------------

rows = pd.DataFrame(
    {
        "Battery (kWh)": [d["battery_kwh"] for d in designs],
        "Investment": [d["total_investment"] for d in designs],
        "Grid independence (%)": [d["grid_independence_pct"] for d in designs],
        "Self-consumption (%)": [d["self_consumption_pct"] for d in designs],
        "Payback year": [d["payback_year"] for d in designs],
        "Replacements": [d.get("battery_replacements", 0) for d in designs],
        "NPV savings": [d["npv_savings"] for d in designs],
    }
)
table(rows, **{"Battery (kWh)": "g", "Investment": ".0f", "NPV savings": ".0f", "Grid independence (%)": ".1f",
               "Self-consumption (%)": ".1f"})  # fmt: skip

# %%

pv_only, *batteries = designs
# sphinx_gallery_start_ignore
best = max(designs, key=lambda d: d["npv_savings"])
say(
    f"Each battery size raises grid independence, from {pv_only['grid_independence_pct']:.0f} % without a battery "
    + " and ".join(f"to {d['grid_independence_pct']:.0f} % with {d['battery_kwh']:g} kWh" for d in batteries)
    + (". The money goes the other way" if best is pv_only else ". The money follows")
    + ": at the preset storage price of "
    f"{number(preset_price)} {currency}/kWh, the highest NPV savings come from "
    + ("the PV-only system" if best is pv_only else f"the {best['battery_kwh']:g} kWh battery")
    + f", {money(best['npv_savings'], currency)}. "
    + " ".join(
        f"The {d['battery_kwh']:g} kWh battery adds {money(d['total_investment'] - pv_only['total_investment'], currency)} "
        f"to the investment and changes NPV savings by {money(d['npv_savings'] - pv_only['npv_savings'], currency)}."
        for d in batteries
    )
)
# sphinx_gallery_end_ignore

# %%
# Cumulative cost with and without the system
# -------------------------------------------
# Each curve starts at the investment in year 0 and falls below the no-system
# line at payback. A battery replacement shows as a step up.

fig = plot_breakeven_comparison(designs, [f"{d['battery_kwh']:g} kWh" for d in designs])

# %%
# How cheap must storage be?
# --------------------------
# ``App.revalue`` re-prices a finished run at another storage price. A price
# change cannot change the dispatch, so each scenario reuses the stored
# simulation (``provenance.revaluation.method`` is ``"repriced"``) and gives
# the same numbers as a new run. The storage price also sets the replacement
# pack's price.

fig, ax = plt.subplots(figsize=(8, 4.5))
for size, group in prices.groupby("battery_kwh"):
    ax.plot(group["storage_cost_per_kwh"], group["npv_savings"], marker="o", ms=3, label=f"{size:g} kWh battery")
ax.axhline(pv_only["npv_savings"], color="0.3", ls="--", label="PV only")
ax.axvline(preset_price, color="0.6", ls=":", label="Preset storage price")
ax.set_xlabel(f"Storage price ({currency}/kWh)")
ax.set_ylabel(f"NPV savings ({currency})")
ax.set_title("NPV savings against the storage price")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

# %%


# sphinx_gallery_start_ignore
def break_even_price(group: pd.DataFrame) -> float | None:
    """The storage price at which the battery's NPV equals the PV-only NPV."""
    gap = group["npv_savings"].to_numpy() - pv_only["npv_savings"]
    price = group["storage_cost_per_kwh"].to_numpy()
    crossings = np.flatnonzero(np.sign(gap[:-1]) != np.sign(gap[1:]))
    if not len(crossings):
        return None
    i = crossings[0]
    return float(price[i] + (price[i + 1] - price[i]) * gap[i] / (gap[i] - gap[i + 1]))


lines = []
for size, group in prices.groupby("battery_kwh"):
    crossing = break_even_price(group)
    lines.append(
        f"The {size:g} kWh battery beats PV only below about {number(crossing)} {currency}/kWh."
        if crossing is not None
        else f"The {size:g} kWh battery does not beat PV only between {number(group['storage_cost_per_kwh'].min())} "
        f"and {number(group['storage_cost_per_kwh'].max())} {currency}/kWh."
    )
resimulated = int((prices["method"] != "repriced").sum())
say(*lines, f"Of the {len(prices)} storage-price scenarios, {len(prices) - resimulated} were re-priced from the "
    f"stored runs and {resimulated} simulated again.")  # fmt: skip
# sphinx_gallery_end_ignore

# %%
# .. sphinx-gallery drops a final code block whose code is all hidden, output
#    included; this closing text block keeps the summary above.
