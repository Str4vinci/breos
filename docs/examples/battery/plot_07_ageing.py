"""
Battery ageing with different degradation models
================================================

How fast a home battery loses capacity depends on the degradation model as
much as on how it is used. This page follows one battery pack for 20 years
under five models: the native engine's two fits, which pair a calendar model
fitted to field data with a laboratory cycle model, and three BLAST cell
models. Replacement is switched off (``battery_enable_replacement = false``),
so one pack serves the whole projection and its state of health can be
followed past the 70 % end-of-life threshold. See
`Running without replacement <../../getting-started/configuration.html#running-without-replacement>`__.

The runs are the quickstart home (:doc:`/gallery/getting_started/plot_01_first_home`):

.. literalinclude:: /../configs/examples/quickstart.toml
   :language: toml
   :caption: configs/examples/quickstart.toml

with replacement off, PV module ageing off and one model selected per run:

.. code-block:: toml

    battery_enable_replacement = false
    pv_degradation_rate = 0.0

    calendar_model = "naumann_lam_field_calibrated_v1"  # native, v1 field fit
    calendar_model = "naumann_lam_field_calibrated_v2"  # native, v2 field fit
    degradation_engine = "blast"                         # or a BLAST cell model
    blast_model = "lfp_gr_250ah_prismatic"               # also lfp_gr_sonymurata_3ah, nmc_gr_50ah_b1

The PV array is held at its first-year output, so the only feedback is the
battery's own fade. The models are compared in two ways, which answer
different questions:

- **Identical imposed stress history.** Every model is given the same state
  of charge and cell temperature, step by step: the first simulated year of
  the native v1 run, repeated every year. The use does not respond to the
  fade, so any difference between the curves comes from the models alone.
- **Full simulation with feedback into dispatch.** Each model runs in the
  App. As the pack loses capacity its usable energy shrinks: the state of
  charge stays within the same 10–90 % window, but of a smaller capacity.
  The dispatch moves less energy and the state of charge it holds changes,
  and that use is what ages the pack next. This is what the household would
  see under each model.

.. note::

   None of the BLAST model files states how long its tests ran, and BREOS
   records no ageing horizon for these models: read the late years as model
   projections, not measured behaviour. The native engine combines
   Naumann's laboratory LFP cycle-ageing model with calendar parameters
   fitted to field data from LFP home storage systems. The BLAST models are
   fitted to laboratory tests of single cells; BREOS applies a cell's
   relative fade to the whole pack, without cell-to-cell spread, temperature
   differences within the pack (BREOS models one lumped cell temperature,
   heated by the pack's own losses) or battery-management effects. Each model
   is a separate
   empirical fit. The native cycle model and the BLAST LFP-Gr 3 Ah model both
   relate to Naumann's laboratory data on Sony/Murata 3 Ah LFP cells, which
   makes that pair the closest native-to-BLAST comparison; the native
   calendar parameters come from field data, and the other two BLAST models
   from other cells (see the
   `battery degradation policy <https://github.com/Str4vinci/breos/blob/develop/design/architecture/battery-degradation-policy.md>`__).
   A difference between the models is not a difference between chemistries
   in general.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
import pandas as pd
from gallery_results import load_case, number, say, table

case = load_case("ageing")
models = pd.DataFrame(case.json("models.json")).set_index("model")
imposed = case.csv("imposed.csv")
simulated = case.csv("simulated.csv")
threshold = 100 * case.manifest["battery_eol_percentage"]
stress = case.manifest["imposed_stress"]
labels = models["label"].to_dict()
years = int(imposed["year"].max())
native = [m for m in models.index if models.loc[m, "engine"] == "native"]
blast = [m for m in models.index if models.loc[m, "engine"] == "blast"]
# Native fits solid, BLAST cell models dashed, one colour per model.
styles = {
    model: {"color": f"C{i}", "ls": "-" if models.loc[model, "engine"] == "native" else "--"}
    for i, model in enumerate(models.index)
}


def from_installation(group: pd.DataFrame, column: str, start: float) -> tuple[list, list]:
    """Year-end values of ``column`` with the installation (year 0, ``start``) first."""
    return [0, *group["year"]], [start, *group[column]]


def at_year(frame: pd.DataFrame, model: str, year: int, column: str) -> float:
    return float(frame.loc[(frame["model"] == model) & (frame["year"] == year), column].iloc[0])


# sphinx_gallery_start_ignore
# The fixed text above holds for the stored runs.
assert case.manifest["overrides"]["pv_degradation_rate"] == 0
assert case.manifest["soc_window"] == [0.1, 0.9]
assert models.loc[blast, "aging_horizon_days"].isna().all()
# sphinx_gallery_end_ignore
case.stamp()

# %%
# Identical imposed stress history
# --------------------------------
# One year of state of charge and cell temperature, repeated every project
# year, drives each model. The imposed year is the native v1 run's first year.
# Its cycle count is the native engine's for that run; each BLAST model counts
# full equivalent cycles itself from the same state of charge, against the
# capacity it has left, so its count differs as it fades.

table(
    pd.DataFrame(
        {
            "Mean state of charge (%)": [stress["mean_soc_pct"]],
            "Mean cell temperature (°C)": [stress["mean_cell_temperature_c"]],
            "Full equivalent cycles in the year, native count": [stress["fec_per_year"]],
        }
    ),
    **{"Mean state of charge (%)": ".1f", "Mean cell temperature (°C)": ".1f",
       "Full equivalent cycles in the year, native count": ".0f"},
)  # fmt: skip

# %%

fig, ax = plt.subplots(figsize=(9, 4.8))
for model, group in imposed.groupby("model", sort=False):
    ax.plot(*from_installation(group, "soh_pct", 100.0), marker="o", ms=3, label=labels[model], **styles[model])
ax.axhline(threshold, color="k", lw=0.8, ls=":", label=f"End-of-life threshold ({threshold:.0f} %)")
ax.set_xlabel("Project year")
ax.set_ylabel("State of health at year end (%)")
ax.set_title("Model projections under the same imposed use")
ax.set_xticks(range(0, years + 1, 2))
ax.legend()
fig.tight_layout()

# %%

table(
    pd.DataFrame(
        {
            "Model": [labels[m] for m in models.index],
            "Chemistry": models["chemistry"].to_list(),
            "Basis": models["basis"].to_list(),
            "Health after 10 years (%)": [at_year(imposed, m, 10, "soh_pct") for m in models.index],
            f"Health after {years} years (%)": [at_year(imposed, m, years, "soh_pct") for m in models.index],
            "First end of life (years)": models["imposed_first_end_of_life_years"].to_list(),
        }
    ),
    **{"Health after 10 years (%)": ".1f", f"Health after {years} years (%)": ".1f",
       "First end of life (years)": ".2f"},
)  # fmt: skip

# %%
# "First end of life" is the time from installation to the end of the first
# day that closes at or below the threshold. The full simulation records the
# same instant as ``battery_first_end_of_life_years``, so the tables below
# compare like with like. An empty cell means the model stays above the
# threshold for the whole projection.
#
# Cycle and calendar ageing in the native fits
# --------------------------------------------
# The native engine reports how much health each pack lost to cycling and
# how much to calendar ageing. The BLAST models report a state of health
# without this split, so the result leaves it empty (None) for them rather
# than reporting zero.

# sphinx_gallery_start_ignore
cycling = imposed[imposed["model"].isin(native)].pivot(index="year", columns="model", values="cycle_loss_pct")
# The v1 and v2 fits share the cycle model and, here, the use, so they lose the same health to cycling.
assert cycling.nunique(axis=1).eq(1).all()
say(
    "The v1 and v2 fits share the cycle model and, here, the same imposed use, so their cycle losses are "
    f"identical: {at_year(imposed, native[0], years, 'cycle_loss_pct'):.1f} percentage points after {years} years. "
    "They differ only in the calendar parameters: "
    + " against ".join(
        f"{at_year(imposed, m, years, 'calendar_loss_pct'):.1f} percentage points of calendar loss under the "
        f"{labels[m].split(', ')[-1]}"
        for m in native
    )
    + "."
)
# sphinx_gallery_end_ignore

# %%

fig, ax = plt.subplots(figsize=(9, 4.2))
for model in native:
    group = imposed[imposed["model"] == model]
    ax.plot(*from_installation(group, "calendar_loss_pct", 0.0), label=f"{labels[model]}: calendar", **styles[model])
ax.plot(*from_installation(imposed[imposed["model"] == native[0]], "cycle_loss_pct", 0.0), color="0.3", ls="-.",
        label="Both native fits: cycling")  # fmt: skip
ax.set_xlabel("Project year")
ax.set_ylabel("Health lost (percentage points)")
ax.set_title("Calendar and cycle loss, native fits, imposed use")
ax.set_xticks(range(0, years + 1, 2))
ax.legend()
fig.tight_layout()

# %%
# Full simulation with feedback into dispatch
# -------------------------------------------
# The same five models, each in its own App run. The left panel is the state
# of health; the right panel is the energy the cells deliver each year,
# which falls as the usable energy shrinks.

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained")
for model, group in simulated.groupby("model", sort=False):
    left.plot(*from_installation(group, "soh_pct", 100.0), marker="o", ms=3, label=labels[model], **styles[model])
    right.plot(group["year"], group["discharge_throughput_kwh"], marker="o", ms=3, **styles[model])
left.axhline(threshold, color="k", lw=0.8, ls=":", label=f"End-of-life threshold ({threshold:.0f} %)")
left.set_ylabel("State of health at year end (%)")
right.set_ylabel("kWh per year")
for ax in (left, right):
    ax.set_xlabel("Project year")
    ax.set_xticks(range(0, years + 1, 4))
left.set_title("State of health")
right.set_title("Energy discharged from the cells")
fig.legend(loc="outside lower center", ncol=3)

# %%
# Each model's full simulation against the common imposed reference:

table(
    pd.DataFrame(
        {
            "Model": [labels[m] for m in models.index],
            f"Health after {years} years, imposed use (%)": [at_year(imposed, m, years, "soh_pct") for m in models.index],
            "Same, full simulation (%)": [at_year(simulated, m, years, "soh_pct") for m in models.index],
            "First end of life, imposed use (years)": models["imposed_first_end_of_life_years"].to_list(),
            "Same, full simulation (years)": models["first_end_of_life_years"].to_list(),
        }
    ),
    **{f"Health after {years} years, imposed use (%)": ".1f", "Same, full simulation (%)": ".1f",
       "First end of life, imposed use (years)": ".2f", "Same, full simulation (years)": ".2f"},
)  # fmt: skip

# %%

# sphinx_gallery_start_ignore
gaps = {m: at_year(simulated, m, years, "soh_pct") - at_year(imposed, m, years, "soh_pct") for m in models.index}
widest = max(gaps, key=lambda m: abs(gaps[m]))
split = {
    m: {
        arm: {column: at_year(frame, m, years, column) for column in ("calendar_loss_pct", "cycle_loss_pct", "cumulative_fec")}
        for arm, frame in (("imposed", imposed), ("simulated", simulated))
    }
    for m in native
}
soc_first = simulated.groupby("model", sort=False)["mean_soc_pct"].first()
soc_last = simulated.groupby("model", sort=False)["mean_soc_pct"].last()
# Every pack holds a lower mean state of charge as it fades. In the full simulations the native
# packs lose less to calendar ageing and more to cycling, over more cycles, and end healthier and later.
assert (soc_last < soc_first).all()
for m in native:
    assert split[m]["simulated"]["calendar_loss_pct"] < split[m]["imposed"]["calendar_loss_pct"]
    assert split[m]["simulated"]["cycle_loss_pct"] > split[m]["imposed"]["cycle_loss_pct"]
    assert split[m]["simulated"]["cumulative_fec"] > split[m]["imposed"]["cumulative_fec"]
    assert gaps[m] > 0
    assert models.loc[m, "first_end_of_life_years"] > models.loc[m, "imposed_first_end_of_life_years"]
say(
    f"After {years} years each model's full simulation is within {abs(gaps[widest]):.1f} percentage points of "
    f"the imposed reference (the widest gap: {labels[widest]}). The gap is not the feedback alone: the reference "
    "repeats the native v1 run's first year for every model, while each full simulation starts from its own "
    "first year. In the full simulations, as the pack fades, the dispatch holds it at a lower mean state of charge: "
    f"{soc_first.min():.1f}–{soc_first.max():.1f} % in year 1, {soc_last.min():.1f}–{soc_last.max():.1f} % in "
    f"year {years}. A lower state of charge slows calendar ageing, and a smaller pack makes more full equivalent "
    "cycles for the same energy. The native split, against the imposed reference, shows both. "
    + " ".join(
        f"Under the {labels[m].split(', ')[-1]} the pack loses {split[m]['simulated']['calendar_loss_pct']:.1f} "
        "percentage points to calendar ageing "
        f"instead of {split[m]['imposed']['calendar_loss_pct']:.1f}, and "
        f"{split[m]['simulated']['cycle_loss_pct']:.2f} to cycling instead of "
        f"{split[m]['imposed']['cycle_loss_pct']:.2f}, over {number(split[m]['simulated']['cumulative_fec'])} full "
        f"equivalent cycles instead of {number(split[m]['imposed']['cumulative_fec'])}."
        for m in native
    )
    + " The slower calendar ageing outweighs the extra cycling, so both native packs end their full simulation "
    "with more health and a later end of life than under the imposed reference.",
)
# sphinx_gallery_end_ignore

# %%
# What the fade costs the household
# ---------------------------------
# Only the full simulation shows this; the imposed history has no dispatch.
# The energy the battery discharges falls with its capacity:

table(
    pd.DataFrame(
        {
            "Model": [labels[m] for m in models.index],
            "Discharged in year 1 (kWh)": [at_year(simulated, m, 1, "discharge_throughput_kwh") for m in models.index],
            f"Discharged in year {years} (kWh)": [
                at_year(simulated, m, years, "discharge_throughput_kwh") for m in models.index
            ],
            "Mean SOC, year 1 (%)": [at_year(simulated, m, 1, "mean_soc_pct") for m in models.index],
            f"Mean SOC, year {years} (%)": [at_year(simulated, m, years, "mean_soc_pct") for m in models.index],
        }
    ),
    **{"Discharged in year 1 (kWh)": ".0f", f"Discharged in year {years} (kWh)": ".0f", "Mean SOC, year 1 (%)": ".1f",
       f"Mean SOC, year {years} (%)": ".1f"},
)  # fmt: skip

# %%
# With replacement off the pack is kept after its end-of-life crossing:
# ``battery_end_of_life_events`` records the crossing with the reason
# ``"replacement_disabled"``, at the time a replacement would have been
# booked.
#
# What the models were fitted to
# -------------------------------
# Each BLAST model file gives an ``experimental_range``: in its own words,
# the range of conditions the model is expected to be valid in. BREOS copies
# it into its model registry (``breos.get_battery_model_profile(key)``), and
# its range check (``BlastWarningCollector``) warns when an input leaves it
# (``degradation.experimental_range_warnings`` in the result). These are the
# limits below. They are not the test conditions themselves; the model files
# describe those in their notes, summarised after the table. The result
# records the first day each input leaves its limits; the table counts every
# day of the stored runs.

INPUTS = {
    "c_rate_charge": ("Charge C-rate (per hour, of aged capacity)", 1.0, 2),
    "dod": ("Depth of discharge (%)", 100.0, 0),
    "temperature_c": ("Cell temperature (°C)", 1.0, 1),
}
ranges = pd.DataFrame([{"model": m, **row} for m in blast for row in models.loc[m, "input_ranges"]])


def span(low: float | None, high: float, scale: float, decimals: int | None = None) -> str:
    """``low–high``, or ``up to high`` without a lower bound, in the page's units; registry limits as given."""
    shown = (lambda v: f"{v * scale:g}") if decimals is None else (lambda v: f"{v * scale:.{decimals}f}")
    return f"up to {shown(high)}" if pd.isna(low) else f"{shown(low)}–{shown(high)}"


table(
    pd.DataFrame(
        {
            "Model": [labels[row.model] for row in ranges.itertuples()],
            "Input": [INPUTS[row.input][0] for row in ranges.itertuples()],
            "Range-check limits": [
                span(row.limit_min, row.limit_max, INPUTS[row.input][1]) for row in ranges.itertuples()
            ],
            "In these runs": [span(row.run_min, row.run_max, *INPUTS[row.input][1:]) for row in ranges.itertuples()],
            "Days outside the limits": [
                f"{number(row.periods_outside)} of {number(row.periods)}" for row in ranges.itertuples()
            ],
        }
    )
)  # fmt: skip

# %%

# sphinx_gallery_start_ignore
low, high = case.manifest["soc_window"]
dod = ranges[ranges["input"] == "dod"]
# One degradation period is one day; the window's width is the lower edge of every
# depth limit, no day swings deeper than the window by more than a rounding, so a
# day outside the depth limits is a shallower one.
assert (ranges["periods"] == 365 * years).all()
assert (dod["limit_min"] - (high - low)).abs().lt(1e-9).all() and dod["run_max"].lt(high - low + 0.01).all()
assert dod["run_max"].lt(dod["limit_max"]).all()
share = 100 * dod["periods_outside"] / dod["periods"]
lines = [
    f"The depth-of-discharge flag is structural. The {100 * low:.0f}–{100 * high:.0f} % SOC window caps the daily "
    f"swing at about {100 * (high - low):.0f} %, the lower edge of every model's depth limits. A day can at best "
    f"reach that edge, and on {share.min():.0f}–{share.max():.0f} % of days the swing is shallower still."
]
for row in ranges[(ranges["input"] == "c_rate_charge") & (ranges["periods_outside"] > 0)].itertuples():
    nominal = (
        f"it stays below {row.limit_max:g} C on every day (peak {row.run_max_nominal:.2f} C)"
        if row.periods_outside_nominal == 0
        else f"it passes {row.limit_max:g} C on {number(row.periods_outside_nominal)} days "
        f"(peak {row.run_max_nominal:.2f} C)"
    )
    lines.append(
        f"The {labels[row.model]} model's charge-rate limit is {row.limit_max:g} C. The model file does not say "
        "whether that rate refers to the nominal or the aged capacity, so this page reports both. Against the "
        "aged capacity, which the range check "
        "uses, the same charging power is a higher C-rate as the pack fades: the charge rate passes "
        f"{row.limit_max:g} C on {number(row.periods_outside)} of {number(row.periods)} days, the first in year "
        f"{row.first_year_outside:.0f}, and reaches {row.run_max:.2f} C. Against the nominal capacity {nominal}."
    )
for row in ranges[(ranges["input"] == "temperature_c") & (ranges["periods_outside"] > 0)].itertuples():
    below = row.run_max <= row.limit_max
    lines.append(
        f"The {labels[row.model]} model's temperature limits are {row.limit_min:g}–{row.limit_max:g} °C. The cell "
        f"temperature here ranges from {row.run_min:.1f} to {row.run_max:.1f} °C, and on "
        f"{number(row.periods_outside)} of {number(row.periods)} days it leaves those limits"
        + (f", always below {row.limit_min:g} °C." if below else ".")
    )
say(*lines)
# sphinx_gallery_end_ignore

# %%
# What each vendored model file (``breos/degradation/blast/models/``) says
# about its ageing tests, where that bears on these limits:

# sphinx_gallery_start_ignore
# The notes below compare with these limits.
limits = ranges.set_index(["model", "input"])
assert (limits.loc[("lfp_gr_250ah_prismatic", "c_rate_charge"), "limit_max"] == 0.65
        and limits.loc[("lfp_gr_250ah_prismatic", "temperature_c"), ["limit_min", "limit_max"]].tolist() == [10, 45])
assert limits.loc[("lfp_gr_sonymurata_3ah", "temperature_c"), ["limit_min", "limit_max"]].tolist() == [20, 40]
# sphinx_gallery_end_ignore
TEST_NOTES = {
    "lfp_gr_250ah_prismatic": "Cycling at 10–45 °C; charging limited to 0.16 C at 10 °C and 0.65 C at higher "
    "temperatures, so the model is fitted to low-rate charging only. The temperature and charge-rate limits "
    "follow these conditions. Cycle ageing varied depth of discharge, average SOC and C-rate, with no values "
    "given, so the depth-of-discharge limit is not stated in the test note.",
    "lfp_gr_sonymurata_3ah": "Cycling only at 25 °C and 45 °C, with no low-temperature cycling data; calendar "
    "ageing varied temperature and SOC. The 20–40 °C limit is not the range of the cycling tests. The model's "
    "cycle fade does not depend on temperature (the file calls this not physically realistic), so the days "
    "below 20 °C affect only its calendar ageing. Cycle ageing varied depth of discharge, average SOC and C-rate, "
    "with no values given, so the depth-of-discharge limit is not stated in the test note.",
    "nmc_gr_50ah_b1": "Calendar ageing varied temperature and SOC, cycle ageing depth of discharge, average SOC and "
    "C-rate; no temperature range or depths are given. Charging at 10 °C was limited to 0.3 C.",
}
table(pd.DataFrame({"Model": [labels[m] for m in blast], "Model file note on the tests": [TEST_NOTES[m] for m in blast]}))

# %%
# The native engine has no such check. Its calendar parameters are fitted to
# field data from LFP home storage systems, and its cycle model to
# laboratory LFP cells; the v1 and v2 fits are two calibrations of the same
# calendar law to that field data.
#
# Which comparison to use
# -----------------------
# Use the imposed history to compare models: it holds the use fixed, so the
# curves differ only by the model. Use the full simulation to project a
# household's battery: the fade changes the dispatch, and the dispatch the
# fade. Either way, read the late years as model projections, and prefer a
# model fitted to a cell and an operating range close to the system studied.
