"""
Price scenarios with App.revalue
================================

Electricity prices over a 20-year project are a guess. How much does the
answer depend on that guess? :meth:`breos.App.revalue` returns the result a
finished run would give at other prices, and when the new prices cannot
change the dispatch it re-prices the stored simulation instead of running
it again. That makes a grid of price scenarios cheap.

This page simulates the time-of-use example once and re-prices it over the
annual import-price escalation and the peak import price.

.. literalinclude:: /../configs/examples/time-of-use-portugal.toml
   :language: toml
   :caption: configs/examples/time-of-use-portugal.toml

The scenarios change two keys:

.. code-block:: python

    result = app.revalue(
        {
            "import_price_escalation": escalation,
            "tariff": {"import_prices": {"peak": peak, "off_peak": off_peak}},
        }
    )

``revalue`` accepts the economics keys only: ``costs``, ``cost_preset``,
``tariff``, ``reference_tariff``, ``discount_rate``, ``inflation_rate``, the
escalators and ``terminal_value``. A nested table changes only the keys it
sets, a key set to ``None`` in it is removed, and ``{"tariff": None}``
removes the tariff. A price list (``tariff.import_prices``,
``tariff.export_prices``, ``reference_tariff.import_prices``) replaces the
old one whole, which is why the call above repeats the off-peak price. A key
that changes the simulation, such as ``battery_kwh`` or
``projection_years``, raises ``ValueError``; build a new ``App`` for it.
``revalue`` leaves the App and its ``result()`` unchanged.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
from gallery_results import load_case, money, say

case = load_case("price_scenarios")
scenarios = case.csv("scenarios.csv")
parity = case.json("parity.json")
currency = case.manifest["currency"]
case.stamp()

# %%
# NPV savings over the price grid
# -------------------------------

grid = scenarios.pivot(index="import_price_escalation", columns="peak_price", values="npv_savings")
fig, ax = plt.subplots(figsize=(8, 4.5))
image = ax.imshow(grid.to_numpy(), cmap="RdYlGn", aspect="auto", origin="lower")
ax.set_xticks(range(len(grid.columns)), [f"{value:.3f}" for value in grid.columns])
ax.set_yticks(range(len(grid.index)), [f"{value:.0%}" for value in grid.index])
for row in range(grid.shape[0]):
    for column in range(grid.shape[1]):
        ax.text(column, row, f"{grid.iat[row, column]:,.0f}", ha="center", va="center", fontsize=8)
ax.set_xlabel(f"Peak import price ({currency}/kWh)")
ax.set_ylabel("Import price escalation per year")
ax.set_title(f"NPV savings ({currency})")
fig.colorbar(image, ax=ax, label=currency)
fig.tight_layout()

# %%

low, high = scenarios["npv_savings"].min(), scenarios["npv_savings"].max()
base_peak = case.manifest["base_peak_price"]
base_escalation = case.manifest["base_import_price_escalation"]
by_escalation = grid[base_peak].max() - grid[base_peak].min()
by_peak = grid.loc[base_escalation].max() - grid.loc[base_escalation].min()
negative = int((scenarios["npv_savings"] < 0).sum())
methods = scenarios["method"].value_counts()
say(
    f"Across the {len(scenarios)} scenarios NPV savings range from {money(low, currency)} to "
    f"{money(high, currency)}, and {negative} of them lose money. At the file's peak price "
    f"({base_peak}), the escalation alone moves them by {money(by_escalation, currency)}; at the default escalation "
    f"({base_escalation:.0%}), the peak price alone moves them by {money(by_peak, currency)}. The escalation "
    "compounds over every project year.",
    f"{methods.get('repriced', 0)} scenarios were re-priced from the stored simulation and "
    f"{methods.get('resimulated', 0)} simulated again. The simulation took {case.manifest['simulate_s']:.1f} s; "
    f"all {len(scenarios)} revaluations together took {case.manifest['revalue_all_s']:.1f} s.",
)

# %%
# Same answer as a new run
# ------------------------
# One scenario was also simulated from scratch with the changed keys in the
# config.

say(
    f"With `import_price_escalation = {parity['changes']['import_price_escalation']}` and a peak price of "
    f"{parity['changes']['tariff']['import_prices']['peak']}, revaluation gives "
    f"{money(parity['revalued_npv_savings'], currency)} and a new simulation "
    f"{money(parity['simulated_npv_savings'], currency)}. A re-priced tariff sums each year's energy by period "
    "instead of by step, so it agrees with a new run to rounding."
)

# %%
# What ``provenance.revaluation.method`` says
# -------------------------------------------
# ``"repriced"``: the stored run was priced again. That is the case for flat
# prices, for a tariff removed, for a ``reference_tariff`` change, and for new
# prices on the same tariff schedule when the smart-charging instructions do
# not change (fixed-target instructions follow the periods, not the prices).
#
# ``"resimulated"``: the run was simulated again, because a tariff was added,
# the schedule changed, or the instructions would change. Under the
# experimental ``daily_persistence`` smart charging, whose planner reads the
# prices, any change to the import or export prices simulates again; a change
# to the fixed charge alone is re-priced.
