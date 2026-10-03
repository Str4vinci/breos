"""
Battery ageing with different degradation models
================================================

How fast a home battery loses capacity depends on the degradation model as
much as on how it is used. This page follows one battery pack for 20 years
under five models: the two field-calibrated fits of the native engine and
three BLAST cell models. Replacement is switched off
(``battery_enable_replacement = false``), so one pack serves the whole
projection and its state of health can be followed past the 70 % end-of-life
threshold. See
`Running without replacement <../../getting-started/configuration.html#running-without-replacement>`__.

The runs are the quickstart home (:doc:`/gallery/getting_started/plot_01_first_home`):

.. literalinclude:: /../configs/examples/quickstart.toml
   :language: toml
   :caption: configs/examples/quickstart.toml

with replacement off and one model selected per run:

.. code-block:: toml

    battery_enable_replacement = false

    calendar_model = "naumann_lam_field_calibrated_v1"  # native, v1 field fit
    calendar_model = "naumann_lam_field_calibrated_v2"  # native, v2 field fit
    degradation_engine = "blast"                         # or a BLAST cell model
    blast_model = "lfp_gr_250ah_prismatic"               # also lfp_gr_sonymurata_3ah, nmc_gr_50ah_b1

The models are compared in two ways, which answer different questions:

- **Identical imposed stress history.** Every model is given the same state
  of charge and cell temperature, step by step: the first simulated year of
  the native v1 run, repeated every year. The use does not respond to the
  fade, so any difference between the curves comes from the models alone.
- **Full simulation with feedback into dispatch.** Each model runs in the
  App. As the pack loses capacity its SOC window shrinks, the dispatch moves
  less energy and the state of charge it holds changes, and that use is what
  ages the pack next. This is what the household would see under each model.

.. note::

   Twenty years is longer than the ageing data behind any of these models,
   so every curve beyond the first years is a model projection, not a
   measured behaviour, and the further below the end-of-life threshold, the
   further it extrapolates. The native engine combines Naumann's laboratory
   LFP cycle-ageing model with calendar parameters fitted to field data from
   LFP home storage systems. The BLAST models are fitted to laboratory tests
   of single cells; BREOS applies a cell's relative fade to the whole pack,
   without cell-to-cell spread, pack thermal behaviour or battery-management
   effects. Each model is a separate empirical fit for its own cell and
   chemistry: the models do not share physics, and a difference between them
   is not a difference between chemistries in general.
"""

# %%
# The stored runs
# ---------------

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
import pandas as pd
from gallery_results import load_case, table

case = load_case("ageing")
models = pd.DataFrame(case.json("models.json")).set_index("model")
imposed = case.csv("imposed.csv")
simulated = case.csv("simulated.csv")
threshold = 100 * case.manifest["battery_eol_percentage"]
stress = case.manifest["imposed_stress"]
labels = models["label"].to_dict()
# Native fits solid, BLAST cell models dashed, one colour per model.
styles = {
    model: {"color": f"C{i}", "ls": "-" if models.loc[model, "engine"] == "native" else "--"}
    for i, model in enumerate(models.index)
}


def first_year_at_or_below(frame: pd.DataFrame, model: str) -> float | None:
    """The first project year that ends at or below the end-of-life threshold."""
    years = frame.loc[(frame["model"] == model) & (frame["soh_pct"] <= threshold), "year"]
    return float(years.min()) if len(years) else None


def from_installation(group: pd.DataFrame, column: str, start: float) -> tuple[list, list]:
    """Year-end values of ``column`` with the installation (year 0, ``start``) first."""
    return [0, *group["year"]], [start, *group[column]]


def at_year(frame: pd.DataFrame, model: str, year: int, column: str) -> float:
    return float(frame.loc[(frame["model"] == model) & (frame["year"] == year), column].iloc[0])


case.stamp()

# %%
# Identical imposed stress history
# --------------------------------
# One year of state of charge and cell temperature, repeated every project
# year, drives each model. The imposed year is the native v1 run's first year:

table(
    pd.DataFrame(
        {
            "Mean state of charge (%)": [stress["mean_soc_pct"]],
            "Mean cell temperature (°C)": [stress["mean_cell_temperature_c"]],
            "Full equivalent cycles per year": [stress["fec_per_year"]],
        }
    ),
    **{"Mean state of charge (%)": ".1f", "Mean cell temperature (°C)": ".1f",
       "Full equivalent cycles per year": ".0f"},
)  # fmt: skip

# %%

fig, ax = plt.subplots(figsize=(9, 4.8))
for model, group in imposed.groupby("model", sort=False):
    ax.plot(*from_installation(group, "soh_pct", 100.0), marker="o", ms=3, label=labels[model], **styles[model])
ax.axhline(threshold, color="k", lw=0.8, ls=":", label=f"End-of-life threshold ({threshold:.0f} %)")
ax.set_xlabel("Project year")
ax.set_ylabel("State of health at year end (%)")
ax.set_title("Model projections under the same imposed use")
ax.set_xticks(range(0, int(imposed["year"].max()) + 1, 2))
ax.legend(fontsize=9)
fig.tight_layout()

# %%

years = int(imposed["year"].max())
table(
    pd.DataFrame(
        {
            "Model": [labels[m] for m in models.index],
            "Chemistry": models["chemistry"].to_list(),
            "Basis": models["basis"].to_list(),
            "Health after 10 years (%)": [at_year(imposed, m, 10, "soh_pct") for m in models.index],
            f"Health after {years} years (%)": [at_year(imposed, m, years, "soh_pct") for m in models.index],
            f"First year at or below {threshold:.0f} %": [first_year_at_or_below(imposed, m) for m in models.index],
        }
    ),
    **{"Health after 10 years (%)": ".1f", f"Health after {years} years (%)": ".1f",
       f"First year at or below {threshold:.0f} %": ".0f"},
)  # fmt: skip

# %%
# An empty cell in the last column means the model stays above the threshold
# for the whole projection.
#
# Cycle and calendar ageing in the native fits
# --------------------------------------------
# The native engine reports how much health each pack lost to cycling and
# how much to calendar ageing. The v1 and v2 fits share the cycle model and,
# here, the same imposed use, so their cycle losses are identical; they
# differ only in the calendar parameters. The BLAST models report a state of
# health without this split, so the result leaves it empty (None) for them
# rather than reporting zero.

native = [m for m in models.index if models.loc[m, "engine"] == "native"]
fig, ax = plt.subplots(figsize=(9, 4.2))
for model in native:
    group = imposed[imposed["model"] == model]
    ax.plot(*from_installation(group, "calendar_loss_pct", 0.0), label=f"{labels[model]}: calendar", **styles[model])
    ax.plot(*from_installation(group, "cycle_loss_pct", 0.0), label=f"{labels[model]}: cycling",
            color=styles[model]["color"], ls="-.")  # fmt: skip
ax.set_xlabel("Project year")
ax.set_ylabel("Health lost since installation (percentage points)")
ax.set_title("Calendar and cycle loss, native fits, imposed use")
ax.set_xticks(range(0, years + 1, 2))
ax.legend(fontsize=9)
fig.tight_layout()

# %%
# Full simulation with feedback into dispatch
# -------------------------------------------
# The same five models, each in its own App run. The left panel is the state
# of health; the right panel is the energy the cells deliver each year,
# which falls as the usable capacity shrinks.

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
for model, group in simulated.groupby("model", sort=False):
    left.plot(*from_installation(group, "soh_pct", 100.0), marker="o", ms=3, label=labels[model], **styles[model])
    right.plot(group["year"], group["discharge_throughput_kwh"], marker="o", ms=3, label=labels[model],
               **styles[model])  # fmt: skip
left.axhline(threshold, color="k", lw=0.8, ls=":")
left.set_ylabel("State of health at year end (%)")
right.set_ylabel("Energy discharged from the cells (kWh per year)")
for ax in (left, right):
    ax.set_xlabel("Project year")
    ax.set_xticks(range(0, years + 1, 4))
left.set_title("State of health")
right.set_title("Battery use")
right.legend(fontsize=9)
fig.tight_layout()

# %%


def simulated_row(model: str) -> dict:
    group = simulated[simulated["model"] == model]
    first, last = group.iloc[0], group.iloc[-1]
    return {
        "Model": labels[model],
        f"Health after {years} years (%)": last["soh_pct"],
        "Same, imposed use (%)": at_year(imposed, model, years, "soh_pct"),
        "First end of life (years)": models.loc[model, "first_end_of_life_years"],
        "Discharged in year 1 (kWh)": first["discharge_throughput_kwh"],
        f"Discharged in year {years} (kWh)": last["discharge_throughput_kwh"],
        "Mean SOC, year 1 (%)": first["mean_soc_pct"],
        f"Mean SOC, year {years} (%)": last["mean_soc_pct"],
    }


table(
    pd.DataFrame([simulated_row(m) for m in models.index]),
    **{f"Health after {years} years (%)": ".1f", "Same, imposed use (%)": ".1f", "First end of life (years)": ".1f",
       "Discharged in year 1 (kWh)": ".0f", f"Discharged in year {years} (kWh)": ".0f", "Mean SOC, year 1 (%)": ".1f",
       f"Mean SOC, year {years} (%)": ".1f"},
)  # fmt: skip

# %%
# "First end of life" is the time of the first end-of-life crossing, the
# instant a replacement would have been booked; with replacement off the
# pack is kept (``battery_end_of_life_events`` records it with the reason
# ``"replacement_disabled"``). Empty means no crossing within the projection.
#
# In this home the feedback changes the health curves little: the dispatch
# works in shares of the aged capacity, so a smaller pack is cycled about as
# deeply as a new one. It changes the household's outcome much more. The
# energy the battery shifts falls with its capacity, and a pack that empties
# earlier each evening also sits at a lower state of charge, which slows its
# calendar ageing.
#
# What the models were fitted to
# -------------------------------
# A BLAST model reports when its input leaves the range of the laboratory
# tests it was fitted to (``degradation.experimental_range_warnings`` in the
# result). Home use at hourly resolution often does: shallow daily cycles lie
# below the tested depths of discharge, for example. The model then
# extrapolates, and so does every number above.

blast = [m for m in models.index if models.loc[m, "engine"] == "blast"]
table(
    pd.DataFrame(
        {
            "Model": [labels[m] for m in blast],
            "Cell": [f"{models.loc[m, 'cell_capacity_ah']:g} Ah, {models.loc[m, 'chemistry']}" for m in blast],
            "Tested depth of discharge": [
                "–".join(f"{v:g}" for v in models.loc[m, "experimental_range"]["dod"]) for m in blast
            ],
            "Tested temperature (°C)": [
                "–".join(f"{v:g}" for v in models.loc[m, "experimental_range"]["cycling_temperature_c"]) for m in blast
            ],
            "Outside the tested range here": [
                ", ".join(models.loc[m, "experimental_range_warnings"]) or "none" for m in blast
            ],
        }
    )
)

# %%
# The native engine has no such check. Its calendar parameters are fitted to
# field records of five LFP home storage systems, and its cycle model to
# laboratory LFP cells; the v1 and v2 fits are two calibrations of the same
# calendar law to those records.
#
# Which comparison to use
# -----------------------
# Use the imposed history to compare models: it holds the use fixed, so the
# curves differ only by the model. Use the full simulation to project a
# household's battery: the fade changes the dispatch, and the dispatch the
# fade. Either way, read the late years as model projections, and prefer a
# model fitted to a cell and an operating range close to the system studied.
