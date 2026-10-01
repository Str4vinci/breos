"""
Which tariff after PV?
======================

A household installing PV usually also picks an electricity offer. Is the
offer under which the system shows the highest NPV also the cheapest one?
Not necessarily. ``npv_savings`` measures the system against *no system
under the same offer*, and each offer has its own no-system bill. An offer
with an expensive peak makes PV look valuable because the household without
PV would pay a lot on it.

This page compares a simple, a bi-hourly and a tri-hourly offer on the
Portuguese mainland schedules, with and without a 5 kWh battery, two ways:
against each offer's own no-system bill (the default), and against one
fixed no-system bill, the household's current simple offer, set with
``[reference_tariff]``.

.. literalinclude:: /../configs/examples/tariff-comparison.toml
   :language: toml
   :caption: configs/examples/tariff-comparison.toml

The run takes 15-minute steps because the tri-hourly schedule changes on the
half hour. The prices are illustrative, not a supplier's offer. The
dispatch does not depend on prices unless ``[smart_charging]`` is set, so
the energy flows are the same under every offer and only the money differs.
"""

# %%
# The stored runs
# ---------------
# The six runs of the sweep, each re-priced once more with
# ``App.revalue({"reference_tariff": ...})``. A reference tariff prices only
# the no-system household, so revaluation never simulates again for it.

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
import numpy as np
from gallery_results import load_case, money, say, table

case = load_case("which_tariff")
offers = case.csv("offers.csv")
reference = case.manifest["reference_tariff"]
currency = case.manifest["currency"]
case.stamp()

# %%

say(
    "The reference is today's simple offer: "
    f"`import_prices = {{ all = {reference['import_prices']['all']} }}` and "
    f"`fixed_charge_per_day = {reference['fixed_charge_per_day']}`, in {reference['currency']}. As TOML it is a "
    "`[reference_tariff]` table with those keys and `currency`."
)

# %%
# Two baselines, two rankings
# ---------------------------

baselines = list(dict.fromkeys(offers["baseline"]))
batteries = sorted(offers["battery_kwh"].unique())
names = list(dict.fromkeys(offers["offer"]))
fig, axes = plt.subplots(1, len(batteries), figsize=(11, 4), sharey=True)
for ax, battery in zip(np.atleast_1d(axes), batteries, strict=True):
    subset = offers[offers["battery_kwh"] == battery]
    x = np.arange(len(names))
    for i, (baseline, color) in enumerate(zip(baselines, ["#9db4d0", "#3a6ea5"], strict=True)):
        values = subset[subset["baseline"] == baseline].set_index("offer").loc[names, "npv_savings"]
        ax.bar(x + (i - 0.5) * 0.38, values, 0.38, label=f"against {baseline}", color=color)
    ax.set_xticks(x, names)
    ax.set_title("PV only" if battery == 0 else f"PV + {battery:g} kWh battery")
    ax.grid(axis="y", alpha=0.3)
np.atleast_1d(axes)[0].set_ylabel(f"NPV savings ({currency})")
np.atleast_1d(axes)[-1].legend()
fig.tight_layout()

# %%

shown = offers.pivot_table(
    index=["battery_kwh", "offer"], columns="baseline", values="npv_savings", sort=False
).reset_index()
costs = offers[offers["baseline"] == baselines[0]].set_index(["battery_kwh", "offer"])
shown["Year-1 bill"] = [
    costs.loc[(b, o), "bill_year1"] for b, o in zip(shown["battery_kwh"], shown["offer"], strict=True)
]
shown["Project cost"] = [
    costs.loc[(b, o), "project_cost"] for b, o in zip(shown["battery_kwh"], shown["offer"], strict=True)
]
shown = shown.rename(
    columns={"battery_kwh": "Battery (kWh)", "offer": "Offer", **{b: f"NPV vs {b}" for b in baselines}}
)
table(shown, **{"Battery (kWh)": "g", **{f"NPV vs {b}": ",.0f" for b in baselines}, "Year-1 bill": ",.2f",
                "Project cost": ",.0f"})  # fmt: skip

# %%
# What the numbers say
# --------------------
# ``Project cost`` is the cumulative discounted cost with the system over the
# project (``financial[-1]["cost_with_system"]``): investment, energy, fixed
# charge, O&M and replacements. It ranks the offers by what the household
# pays. Against one shared reference, NPV savings rank them the same way.

lines = []
for battery in batteries:
    subset = offers[offers["battery_kwh"] == battery]
    label = "Without a battery" if battery == 0 else f"With the {battery:g} kWh battery"
    own = subset[subset["baseline"] == baselines[0]].set_index("offer")
    ref = subset[subset["baseline"] == baselines[1]].set_index("offer")
    cheapest = own["project_cost"].idxmin()
    best_own = own["npv_savings"].idxmax()
    best_ref = ref["npv_savings"].idxmax()
    order_own = " > ".join(own["npv_savings"].sort_values(ascending=False).index)
    order_ref = " > ".join(ref["npv_savings"].sort_values(ascending=False).index)
    overstated = {name: own.loc[name, "npv_savings"] - ref.loc[name, "npv_savings"] for name in own.index}
    biggest = max(overstated, key=lambda name: abs(overstated[name]))
    lines.append(
        f"{label}, the cheapest offer over the project is {cheapest}. NPV savings rank the offers "
        f"{order_own} against each offer's own no-system bill, and {order_ref} against today's simple offer"
        + (" — the ranking flips." if order_own != order_ref else ".")
        + f" The own-offer baseline changes the {biggest} offer's NPV savings by "
        f"{money(overstated[biggest], currency)}, because its no-system bill differs from today's."
        + (
            f" The highest NPV against the own-offer baseline, {best_own}, is not the cheapest offer."
            if best_own != cheapest
            else ""
        )
        + (
            f" Against the shared reference the best NPV is {best_ref}, which is also the cheapest."
            if best_ref == cheapest
            else ""
        )
    )
say(*lines)

# %%
# In Python
# ---------
# The same comparison with :class:`breos.App`: run each offer once, and price
# it against the shared reference with :meth:`breos.App.revalue`.
#
# .. code-block:: python
#
#     import tomllib
#     from breos import App
#
#     with open("configs/examples/tariff-comparison.toml", "rb") as handle:
#         config = tomllib.load(handle)
#     sweep = config.pop("sweep")
#     reference = {"currency": "EUR", "import_prices": {"all": 0.1950}, "fixed_charge_per_day": 0.30}
#
#     for tariff in sweep["tariff"]:
#         app = App({**config, "battery_kwh": 5.0, "tariff": tariff})
#         app.simulate()
#         own = app.result()
#         shared = app.revalue({"reference_tariff": reference})
#         bill = (
#             own["grid_import_cost_year1_prices"]
#             + own["fixed_charge_year1_prices"]
#             - own["grid_export_revenue_year1_prices"]
#         )
#         print(
#             tariff["schedule"],
#             f"year-1 bill {bill:.2f}",
#             f"project cost {own['financial'][-1]['cost_with_system']:.0f}",
#             f"NPV vs own offer {own['npv_savings']:.0f}",
#             f"NPV vs today's offer {shared['npv_savings']:.0f}",
#         )
#
# The year-1 bill leaves out O&M, which does not depend on the offer. A flat
# run, without ``[tariff]``, uses the cost preset's ``electricity_cost``,
# ``electricity_sold_cost`` and ``daily_power_cost``; a tariff run must not
# set them. ``breos sweep`` shares one weather and PV preparation between the
# runs; this loop prepares them again for each offer. See
# `No-system reference tariff <../../getting-started/configuration.html#no-system-reference-tariff>`__
# for the reference table's keys.
