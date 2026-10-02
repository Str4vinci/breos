"""
Battery replacement timing near the end of the project
======================================================

BREOS replaces a battery pack when its usable capacity falls to the
end-of-life threshold (``battery_eol_percentage``, 70 % by default), and the
economics pay the new pack's full price at that moment. If the replacement lands
just before the project horizon, the household pays for a nearly new pack
and the project ends before it is used. NPV savings then drop by almost the
price of a pack, for a design that is no worse than its neighbours.

This page shows that drop twice, by moving the horizon and by adding
modules, and shows how valuing the battery's residual value at the horizon
removes it. The runs are the quickstart home
(:doc:`/gallery/getting_started/plot_01_first_home`) with the residual value
switched on:

.. literalinclude:: /../configs/examples/quickstart.toml
   :language: toml
   :caption: configs/examples/quickstart.toml

.. code-block:: toml

    [terminal_value]
    basis = "battery_health_fraction"

The residual value is the pack present at the end, valued by its remaining
health above the end-of-life threshold at the replacement-pack price and
discounted from the horizon. It is an accounting sensitivity, not a resale
price. The result keeps the plain ``npv_savings`` and adds the residual value
as ``terminal_health_credit`` (``terminal_health_credit_npv`` discounted) and
``npv_savings_terminal_adjusted``, the NPV including it. See
`Terminal-health credit <../../getting-started/configuration.html#terminal-health-credit>`__
for the formula.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
from gallery_results import load_case, money, say, table

case = load_case("replacement_timing")
horizons = case.csv("horizons.csv")
modules = case.csv("modules.csv")
currency = case.manifest["currency"]
case.stamp()

# %%
# Moving the horizon
# ------------------
# The same system, valued over 6 to 25 years. Every other input is the same,
# so the yearly energy and degradation are identical up to each horizon.

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(horizons["projection_years"], horizons["npv_savings"], "o-", color="#EE6677", label="NPV savings")
ax.plot(
    horizons["projection_years"],
    horizons["npv_savings_terminal_adjusted"],
    "s-",
    color="#228833",
    label="NPV savings including the battery's residual value",
)
for years in horizons.loc[horizons["battery_replacements"].diff() > 0, "projection_years"]:
    ax.axvline(years, color="0.6", ls=":")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(horizons["projection_years"])
ax.set_xlabel("Project horizon (years)")
ax.set_ylabel(f"{currency}")
ax.set_title("NPV savings against the project horizon")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

# %%

# sphinx_gallery_start_ignore
by_years = horizons.set_index("projection_years")
jumps = by_years.index[by_years["battery_replacements"].diff() > 0]
lines = []
for years in jumps:
    before, after = by_years.loc[years - 1], by_years.loc[years]
    lines.append(
        f"From {years - 1} to {years} years a replacement enters the project (at "
        f"{after['last_replacement_years']:.2f} years): NPV savings fall from {money(before['npv_savings'], currency)} "
        f"to {money(after['npv_savings'], currency)}, although the longer project earns one more year of savings. "
        f"The pack ends at {after['battery_soh_end_pct']:.1f} % health, and its residual value, "
        f"{money(after['terminal_health_credit_npv'], currency)} discounted, keeps the NPV including it rising: "
        f"{money(before['npv_savings_terminal_adjusted'], currency)} to "
        f"{money(after['npv_savings_terminal_adjusted'], currency)}."
    )
say(*lines)
# sphinx_gallery_end_ignore

# %%
# Adding modules
# --------------
# A bigger array charges the battery harder, so it ages slightly faster and
# its first replacement comes a little earlier. Near the horizon that can
# move the second replacement from just after the end to just before it.

counts = modules[modules["terminal_replacement"]].set_index("n_modules")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(counts.index, counts["npv_savings"], "o-", color="#EE6677", label="NPV savings")
ax.plot(counts.index, counts["npv_savings_terminal_adjusted"], "s-", color="#228833",
        label="NPV savings including the battery's residual value")  # fmt: skip
for count, row in counts.iterrows():
    ax.annotate(
        f"{row['battery_replacements']:.0f} replacement" + ("s" if row["battery_replacements"] != 1 else ""),
        (count, row["npv_savings"]),
        textcoords="offset points",
        xytext=(0, -16),
        ha="center",
        fontsize=8,
    )
ax.set_xlabel("PV modules")
ax.set_ylabel(currency)
ax.set_title(f"NPV savings against the array size, {modules['projection_years'].iloc[0]}-year horizon")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

# %%

table(
    counts.reset_index()[["n_modules", "pv_kwp", "replacement_times_years", "battery_soh_end_pct", "npv_savings",
                          "terminal_health_credit_npv", "npv_savings_terminal_adjusted"]].rename(
        columns={
            "n_modules": "Modules",
            "pv_kwp": "kWp",
            "replacement_times_years": "Replacements at (years)",
            "battery_soh_end_pct": "Final health (%)",
            "npv_savings": "NPV savings",
            "terminal_health_credit_npv": "Residual value (discounted)",
            "npv_savings_terminal_adjusted": "NPV with residual value",
        }
    ),
    **{"NPV savings": ".0f", "Residual value (discounted)": ".0f", "NPV with residual value": ".0f", "kWp": "g"},
)  # fmt: skip

# %%

# sphinx_gallery_start_ignore
horizon = int(modules["projection_years"].iloc[0])
cliff = counts["battery_replacements"].diff().fillna(0).gt(0)
if cliff.any():
    count = int(counts.index[cliff][0])
    before, after = counts.loc[count - 1], counts.loc[count]
    days = (horizon - after["last_replacement_years"]) * 365
    say(
        f"Going from {count - 1} to {count} modules, the second replacement lands at "
        f"{after['last_replacement_years']:.4f} years, about {days:.0f} days before the {horizon}-year horizon. "
        f"NPV savings fall from {money(before['npv_savings'], currency)} to {money(after['npv_savings'], currency)}, "
        f"a drop of {money(before['npv_savings'] - after['npv_savings'], currency)} for one more module. With the "
        f"residual value, the new pack's {after['battery_soh_end_pct']:.1f} % health is worth "
        f"{money(after['terminal_health_credit_npv'], currency)}, and the NPV including it moves smoothly: "
        f"{money(before['npv_savings_terminal_adjusted'], currency)} to "
        f"{money(after['npv_savings_terminal_adjusted'], currency)}."
    )
# sphinx_gallery_end_ignore

# %%
# Not buying the last pack
# ------------------------
# ``battery_allow_terminal_replacement = false`` skips a replacement only in
# the horizon's final degradation period, which is the last day of an hourly
# whole-year run. It does not move a replacement that falls a few days earlier.

# sphinx_gallery_start_ignore
pairs = modules.pivot(index="n_modules", columns="terminal_replacement", values="npv_savings")
unchanged = (pairs[True] == pairs[False]).all()
say(
    "Here every design gives the same NPV savings with and without terminal replacement, because no replacement "
    "falls on the last day: the policy does not remove the drop, and the residual value is the tool for it."
    if unchanged
    else "Here the policy changes the result for "
    + ", ".join(f"{count} modules" for count in pairs.index[pairs[True] != pairs[False]])
    + "."
)
# sphinx_gallery_end_ignore

# %%
# Which number to use
# -------------------
# The plain NPV is what the household pays and saves within the horizon,
# and it stays the headline. Read it with the replacement times next to it: a
# design that buys a pack in the last months of the project looks worse than
# it is. The NPV including the residual value is a sensitivity that says how
# much of that is timing. When designs are ranked by NPV, for example in a
# sweep, check both.
