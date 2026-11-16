"""Each per-step project year as the projection loop runs it, for tests that read a year's steps.

The loop builds a per-step frame for its first project year only: through
``simulate_energy_balance`` (greedy) or ``_simulate_detailed_run`` (a
controller, a year planner or a vehicle). Every later year runs
``_simulate_core`` without frames and is reduced from the core's arrays by
``_build_simulation_summary``. :func:`record_project_years` spies on all four
names in ``breos.projection`` and keeps one :class:`ProjectYearRun` per year,
in project order, read from what that path produced: a later year's results
frame is the core's own step buffers, and its health, replacements and
end-of-life events are the summary the loop carried forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

import breos.projection as projection_module
from breos.battery import END_OF_LIFE_EVENTS_ATTR


@dataclass(frozen=True)
class ProjectYearRun:
    """One project year's keyword arguments and what its run produced."""

    kwargs: dict[str, Any]
    results_df: pd.DataFrame
    n_replacements: int
    end_of_life_records: list[dict[str, Any]]
    degradation_state: dict[str, Any]
    controller_carry: Any
    controller_instructions: Any
    # The year's degradation-tracking rows; None on a year reduced from the
    # core, which keeps none.
    degradation_df: pd.DataFrame | None


def _step_frame(core: Any) -> pd.DataFrame:
    """The core's step buffers as a results frame; a PV-only summary's buffers have no ``to_frame``."""
    if hasattr(core.buffers, "to_frame"):
        return core.buffers.to_frame(core.rng)
    return pd.DataFrame({"Datetime": core.rng, **core.buffers.columns})


def record_project_years(patch: pytest.MonkeyPatch) -> list[ProjectYearRun]:
    """Spy on the projection loop's per-step year paths; returns the list each year is appended to."""
    runs: list[ProjectYearRun] = []
    balance = projection_module.simulate_energy_balance
    detailed = projection_module._simulate_detailed_run
    core_run = projection_module._simulate_core
    summarize = projection_module._build_simulation_summary
    pending: dict[int, dict[str, Any]] = {}

    def recording_balance(**kwargs):
        output = balance(**kwargs)
        results_df, _total_pv, _summary_df, n_replacements, degradation_df, state = output
        runs.append(
            ProjectYearRun(
                kwargs=kwargs,
                results_df=results_df,
                n_replacements=n_replacements,
                end_of_life_records=list(degradation_df.attrs[END_OF_LIFE_EVENTS_ATTR]),
                degradation_state=state,
                controller_carry=None,
                controller_instructions=None,
                degradation_df=degradation_df,
            )
        )
        return output

    def recording_detailed(**kwargs):
        run = detailed(**kwargs)
        runs.append(
            ProjectYearRun(
                kwargs=kwargs,
                results_df=run.results_df,
                n_replacements=run.n_replacements,
                end_of_life_records=list(run.degradation_df.attrs[END_OF_LIFE_EVENTS_ATTR]),
                degradation_state=run.degradation_state,
                controller_carry=run.controller_carry,
                controller_instructions=run.controller_instructions,
                degradation_df=run.degradation_df,
            )
        )
        return run

    def recording_core(**kwargs):
        core = core_run(**kwargs)
        pending[id(core)] = kwargs
        return core

    def recording_summary(core, **kwargs):
        summary = summarize(core, **kwargs)
        runs.append(
            ProjectYearRun(
                kwargs=pending.pop(id(core)),
                results_df=_step_frame(core),
                n_replacements=summary.n_replacements,
                end_of_life_records=[event.to_record() for event in summary.end_of_life_events],
                degradation_state=summary.final_degradation_state,
                controller_carry=core.controller_carry,
                controller_instructions=core.controller_instructions,
                degradation_df=None,
            )
        )
        return summary

    patch.setattr(projection_module, "simulate_energy_balance", recording_balance)
    patch.setattr(projection_module, "_simulate_detailed_run", recording_detailed)
    patch.setattr(projection_module, "_simulate_core", recording_core)
    patch.setattr(projection_module, "_build_simulation_summary", recording_summary)
    return runs


def keep_every_frame(patch: pytest.MonkeyPatch) -> None:
    """Make the projection loop build every year's per-step frames, as it did for all years before.

    Test-only: production keeps the first year's frame only. Use it for a
    check that needs a later year's degradation-tracking rows, which a year
    reduced from the core does not keep, and compare that run with the real
    one so the check still speaks for the production path.
    """
    run_frame_year = projection_module._run_frame_year

    def framed(*args, **kwargs):
        return run_frame_year(*args, **{**kwargs, "keep_frame": True})

    patch.setattr(projection_module, "_run_frame_year", framed)
