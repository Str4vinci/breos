"""
Discharge-only dispatch on a quarterly tariff in Berlin
=======================================================

Some tariffs change their time windows with the season. This example uses a
custom schedule in the style of a German §14a EnWG "Module 3" network tariff:
low, standard and high windows in the first and fourth quarters, and the
standard price all day in the second and third. The page compares greedy
dispatch with a battery that discharges only in the dearer windows.

.. literalinclude:: /../configs/examples/quarterly-tariff-berlin.toml
   :language: toml
   :caption: configs/examples/quarterly-tariff-berlin.toml

The windows and prices are illustrative, not a network operator's published
values. Three dispatch settings are compared: greedy (no
``[smart_charging]``), ``mode = "discharge_only"`` with
``discharge_periods = ["high"]``, and the same with
``discharge_periods = ["standard", "high"]``.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gallery_results import load_case, money, say, table

case = load_case("quarterly_berlin")
strategies = case.json("strategies.json")
currency = case.manifest["currency"]
case.stamp()

# %%
# Battery discharge month by month
# --------------------------------

months = np.arange(1, 13)
labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize=(9, 4))
width = 0.8 / len(strategies)
for i, row in enumerate(strategies):
    ax.bar(months + (i - (len(strategies) - 1) / 2) * width, row["monthly_battery_to_load_kwh"], width,
           label=row["strategy"])  # fmt: skip
ax.set_xticks(months, labels)
ax.set_ylabel("Battery to load (kWh)")
ax.set_title("Energy the battery delivers to the house, year 1")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

# %%

table(
    pd.DataFrame(
        {
            "Strategy": [row["strategy"] for row in strategies],
            "Grid independence (%)": [row["grid_independence_pct"] for row in strategies],
            "Import (kWh)": [row["grid_import_kwh"] for row in strategies],
            "Year-1 bill": [row["bill_year1"] for row in strategies],
            "NPV savings": [row["npv_savings"] for row in strategies],
        }
    ),
    **{"Grid independence (%)": ".1f", "Import (kWh)": ".0f", "Year-1 bill": ".2f", "NPV savings": ".0f"},
)

# %%

# sphinx_gallery_start_ignore
by_name = {row["strategy"]: row for row in strategies}
greedy, high_only = by_name["greedy"], by_name["discharge_only high"]
summer = [m - 1 for m in range(4, 10)]
idle = sum(high_only["monthly_battery_to_load_kwh"][m] for m in summer)
say(
    f'With `discharge_periods = ["high"]` the battery delivers {idle:.0f} kWh from April to September: the high '
    "window does not exist in those quarters, so the battery fills with PV and then holds it. NPV savings fall "
    f"from {money(greedy['npv_savings'], currency)} under greedy dispatch to "
    f"{money(high_only['npv_savings'], currency)}.",
    "Allowing the standard window as well restores summer discharge: NPV savings are "
    f"{money(by_name['discharge_only standard and high']['npv_savings'], currency)}. A discharge restriction "
    "names periods, and on a seasonal schedule a period can be absent for months.",
)
# sphinx_gallery_end_ignore

# %%
# .. sphinx-gallery drops a final code block whose code is all hidden, output
#    included; this closing text block keeps the summary above.
