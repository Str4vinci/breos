"""The opt-in guard against a battery replacement in the horizon's final degradation period (#304).

A pack bought when the horizon ends serves no simulated step but costs a full
purchase. ``BatteryConfig.allow_terminal_replacement``, the App and Monte Carlo
key ``battery_allow_terminal_replacement`` and the optimizer's
``[battery] allow_terminal_replacement`` skip that one purchase when false.
The final period is the positional degradation window that ends on the span's
last step, whole or partial. It is still aged, finalized and recorded; only
its swap is skipped. All three default to true, which changes nothing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import breos.optimization as optimization_module
import breos.projection as projection_module
from breos import App
from breos.app_config import resolve_app_config
from breos.battery import (
    BatteryConfig,
    align_simulation_inputs,
    simulate_energy_balance,
    simulate_energy_balance_summary,
)
from breos.economics import cost_analysis_projection, price_year_rows, replacement_fraction_from_steps
from breos.montecarlo import MonteCarloSettings, run_montecarlo
from breos.optimization import _evaluate_projected_design_metrics, evaluate_projected_design
from breos.optimization_config import resolve_optimization_config
from breos.projection import ProjectionYear, project_years, value_projection
from breos.pv_modules import get_module
from breos.result_schema import RESULT_SCHEMA_VERSION
from breos.smart_charging import FixedTargetDayController
from tests.test_controller_seam import _battery, _core, _Recording, _scenario
from tests.test_daily_persistence import _small_grid
from tools.generate_app_golden import _fake_fetch

# A threshold this close to full health is crossed when any period closes,
# including the first period of a fresh pack, so every close replaces.
EVERY_PERIOD_EOL = 0.99999
BLAST = {"degradation_engine": "blast", "blast_model": "lfp_gr_250ah_prismatic"}
FREQS = ["h", "15min"]
# The columns a swap writes. Without one, each is zero or False.
SWAP_COLUMNS = (
    "Battery_Replaced",
    "Battery_Replaced_Capacity_Wh",
    "Battery_Replacement_Energy_Removed",
    "Battery_Replacement_Energy_Added",
    "PV_Origin_Replacement_Energy_Removed",
    "Grid_Origin_Replacement_Energy_Removed",
)
# The rates are nonzero so a surviving swap's timing and price show in the money.
PRICING = {
    "inflation_rate": 0.02,
    "discount_rate": 0.05,
    "replacement_cost_learning": 0.03,
    "costs": {"storage_cost_per_kwh": 400.0},
}

pytestmark = pytest.mark.filterwarnings("ignore::breos.degradation.validation.BlastExperimentalRangeWarning")


def _require_numba() -> None:
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")


def _steps_per_day(freq: str) -> int:
    return 24 if freq == "h" else 96


def _inputs(n_steps: int, freq: str = "h"):
    """A hard-cycled series: PV by day, load morning and evening."""
    idx = pd.date_range("2025-01-01 00:00", periods=n_steps, freq=freq, tz="UTC")
    hour = idx.hour.to_numpy()
    pv = pd.Series(np.where((hour >= 8) & (hour < 16), 1800.0, 0.0), index=idx)
    load = pd.DataFrame({"Load": np.where(hour < 12, 600.0, 1000.0)}, index=idx)
    return pv, load, pd.Series(22.0, index=idx)


def _run(n_steps: int, freq: str = "h", *, backend: str = "python", engine=None, finalize=True, **config):
    pv, load, temperature = _inputs(n_steps, freq)
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(nominal_energy_wh=5000.0, **config),
        freq=freq,
        temperature_series=temperature,
        execution_backend=backend,
        return_degradation_state=True,
        finalize_degradation=finalize,
        **(engine or {}),
    )


def _replaced_steps(run) -> list[int]:
    return np.flatnonzero(run[0]["Battery_Replaced"].to_numpy()).tolist()


def _assert_same_run(left, right) -> None:
    left_df, left_pv, left_summary, left_n, left_deg, left_state = left
    right_df, right_pv, right_summary, right_n, right_deg, right_state = right
    pd.testing.assert_frame_equal(left_df, right_df, check_exact=True)
    pd.testing.assert_frame_equal(left_summary, right_summary, check_exact=True)
    pd.testing.assert_frame_equal(left_deg, right_deg, check_exact=True)
    assert (left_pv, left_n) == (right_pv, right_n)
    # NaN-aware: a BLAST state holds NaN stressors.
    np.testing.assert_equal(left_state, right_state)


def _final_period_eol(n_steps: int, freq: str, engine=None, **config) -> float:
    """An end-of-life threshold that only the span's final period crosses."""
    soh = _run(n_steps, freq, engine=engine, enable_replacement=False, **config)[4]["SOH"].to_numpy()
    assert soh[-2] > soh[-1]
    return float((soh[-2] + soh[-1]) / 2.0) / 100.0


# -- the setting ------------------------------------------------------------------


class TestBatteryConfigFlag:
    def test_default_allows_a_terminal_replacement(self):
        assert BatteryConfig(nominal_energy_wh=5000.0).allow_terminal_replacement is True

    def test_a_numpy_bool_is_stored_as_a_bool(self):
        config = BatteryConfig(nominal_energy_wh=5000.0, allow_terminal_replacement=np.bool_(False))
        assert config.allow_terminal_replacement is False

    @pytest.mark.parametrize("value", ["false", "true", 0, 1, 1.0, None])
    def test_a_value_that_is_not_a_bool_is_refused(self, value):
        with pytest.raises(ValueError, match="allow_terminal_replacement must be a bool"):
            BatteryConfig(nominal_energy_wh=5000.0, allow_terminal_replacement=value)


class TestAppAndMonteCarloKey:
    BASE = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "battery_kwh": 5}

    def test_the_key_resolves_true_by_default_and_keeps_false(self):
        assert resolve_app_config(self.BASE).cfg["battery_allow_terminal_replacement"] is True
        resolved = resolve_app_config({**self.BASE, "battery_allow_terminal_replacement": False})
        assert resolved.cfg["battery_allow_terminal_replacement"] is False

    @pytest.mark.parametrize("value", ["false", 0, 1, None])
    def test_a_value_that_is_not_a_bool_is_refused(self, value):
        with pytest.raises(TypeError, match="'battery_allow_terminal_replacement' must be a boolean"):
            resolve_app_config({**self.BASE, "battery_allow_terminal_replacement": value})

    @pytest.mark.parametrize("value", [True, False])
    def test_the_projection_battery_carries_the_policy(self, value):
        resolved = resolve_app_config({**self.BASE, "battery_allow_terminal_replacement": value})
        battery = projection_module.build_battery_config(resolved.cfg, resolved, initial_soh=100.0)
        assert battery.allow_terminal_replacement is value


class TestOptimizerKey:
    LOCATION = {"location": {"latitude": 41.15, "longitude": -8.61}}

    def test_the_key_resolves_true_by_default_and_keeps_false(self):
        assert resolve_optimization_config(self.LOCATION)["battery"]["allow_terminal_replacement"] is True
        config = {**self.LOCATION, "battery": {"allow_terminal_replacement": False}}
        assert resolve_optimization_config(config)["battery"]["allow_terminal_replacement"] is False

    @pytest.mark.parametrize("value", ["false", 0, None])
    def test_a_value_that_is_not_a_bool_is_refused(self, value):
        with pytest.raises(TypeError, match="'battery.allow_terminal_replacement' must be true or false"):
            resolve_optimization_config({**self.LOCATION, "battery": {"allow_terminal_replacement": value}})

    @pytest.mark.parametrize("value", [True, False])
    def test_the_spec_forwards_the_policy(self, value):
        config = optimization_module._build_battery_config_from_spec(
            {"allow_terminal_replacement": value}, nominal_energy_wh=5000.0
        )
        assert config.allow_terminal_replacement is value


# -- one span ---------------------------------------------------------------------


@pytest.mark.parametrize("freq", FREQS)
@pytest.mark.parametrize("engine", [None, BLAST], ids=["native", "blast"])
def test_a_final_period_crossing_skips_exactly_that_swap(freq, engine):
    n_steps = 3 * _steps_per_day(freq)
    eol = _final_period_eol(n_steps, freq, engine)

    default = _run(n_steps, freq, engine=engine, eol_percentage=eol)
    guarded = _run(n_steps, freq, engine=engine, eol_percentage=eol, allow_terminal_replacement=False)
    unreplaced = _run(n_steps, freq, engine=engine, eol_percentage=eol, enable_replacement=False)

    assert default[3] == 1 and _replaced_steps(default) == [n_steps - 1]
    # Only the terminal swap existed, so skipping it is the run without any.
    assert guarded[3] == 0
    _assert_same_run(guarded, unreplaced)
    guarded_df, default_df = guarded[0], default[0]
    for column in SWAP_COLUMNS:
        assert not guarded_df[column].any(), column
    # The final period was still aged and recorded, and the old pack ends it.
    assert len(guarded[4]) == len(default[4]) == 3
    assert guarded[4]["SOH"].iloc[-1] < 100.0 * eol
    # The old pack keeps its stored energy and its origins; the default's
    # fresh pack starts full and unattributed.
    assert default_df["Battery_Energy_End"].iloc[-1] == 5000.0 * 0.9
    assert default_df["Battery_PV_Origin_Energy_End"].iloc[-1] == 0.0
    assert guarded_df["Battery_Energy_End"].iloc[-1] != 5000.0 * 0.9
    assert guarded_df["Battery_PV_Origin_Energy_End"].iloc[-1] > 0.0
    # Everything before the swap's closing step is the same run.
    pd.testing.assert_frame_equal(default_df.iloc[:-1], guarded_df.iloc[:-1], check_exact=True)


def test_the_final_period_still_counts_its_rainflow_residue():
    n_steps = 72
    eol = _final_period_eol(n_steps, "h")
    guarded = _run(n_steps, eol_percentage=eol, allow_terminal_replacement=False)
    open_residue = _run(n_steps, eol_percentage=eol, allow_terminal_replacement=False, finalize=False)

    # The residue is settled on the existing schedule whatever the permission.
    assert guarded[4]["Cumulative_FEC"].iloc[-1] > open_residue[4]["Cumulative_FEC"].iloc[-1]
    assert guarded[4]["Cumulative_FEC_All_Packs"].iloc[-1] == guarded[4]["Cumulative_FEC"].iloc[-1]


def test_resistance_fade_keeps_the_old_pack_resistance():
    n_steps = 72
    config = {"enable_resistance_fade": True}
    eol = _final_period_eol(n_steps, "h", **config)
    default = _run(n_steps, eol_percentage=eol, **config)
    guarded = _run(n_steps, eol_percentage=eol, allow_terminal_replacement=False, **config)
    unreplaced = _run(n_steps, eol_percentage=eol, enable_replacement=False, **config)

    assert default[4]["Resistance_Growth"].iloc[-1] == 0.0
    assert guarded[4]["Resistance_Growth"].iloc[-1] > 0.0
    _assert_same_run(guarded, unreplaced)


def test_without_a_final_period_crossing_the_setting_changes_nothing():
    _assert_same_run(_run(72), _run(72, allow_terminal_replacement=False))


def test_an_explicit_true_is_the_default():
    _assert_same_run(
        _run(72, eol_percentage=EVERY_PERIOD_EOL),
        _run(72, eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=True),
    )


@pytest.mark.parametrize("freq", FREQS)
def test_earlier_periods_still_replace(freq):
    spd = _steps_per_day(freq)
    n_steps = 3 * spd
    default = _run(n_steps, freq, eol_percentage=EVERY_PERIOD_EOL)
    guarded = _run(n_steps, freq, eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=False)

    assert _replaced_steps(default) == [spd - 1, 2 * spd - 1, 3 * spd - 1]
    assert _replaced_steps(guarded) == [spd - 1, 2 * spd - 1]
    assert guarded[3] == 2


# -- partial spans ----------------------------------------------------------------


@pytest.mark.parametrize("freq", FREQS)
def test_a_trailing_partial_period_is_the_final_one(freq):
    spd = _steps_per_day(freq)
    tail = 5 * spd // 24
    n_steps = 2 * spd + tail
    default = _run(n_steps, freq, eol_percentage=EVERY_PERIOD_EOL)
    guarded = _run(n_steps, freq, eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=False)

    # The whole day before the partial one keeps its swap: its pack serves the tail.
    assert _replaced_steps(default) == [spd - 1, 2 * spd - 1, n_steps - 1]
    assert _replaced_steps(guarded) == [spd - 1, 2 * spd - 1]
    # The partial period is still aged and recorded, on the pack that serves it.
    assert len(guarded[4]) == len(default[4]) == 3
    step_seconds = 3600.0 * 24 / spd
    assert guarded[4]["Cumulative_Calendar_Seconds"].iloc[-1] == tail * step_seconds
    assert guarded[4]["Calendar_Degradation"].iloc[-1] > 0.0
    assert guarded[4]["SOH"].iloc[-1] < 100.0


@pytest.mark.parametrize("freq", FREQS)
def test_the_whole_day_before_a_partial_period_keeps_its_end_of_life_swap(freq):
    spd = _steps_per_day(freq)
    n_steps = 2 * spd + 5 * spd // 24
    soh = _run(n_steps, freq, enable_replacement=False)[4]["SOH"].to_numpy()
    # Crossed at the close of the second, whole day, which is not the final period.
    eol = float((soh[0] + soh[1]) / 2.0) / 100.0

    default = _run(n_steps, freq, eol_percentage=eol)
    guarded = _run(n_steps, freq, eol_percentage=eol, allow_terminal_replacement=False)

    assert _replaced_steps(default) == [2 * spd - 1]
    _assert_same_run(default, guarded)


@pytest.mark.parametrize("freq", FREQS)
def test_a_span_shorter_than_a_day_has_one_final_period(freq):
    n_steps = 10 * _steps_per_day(freq) // 24
    default = _run(n_steps, freq, eol_percentage=EVERY_PERIOD_EOL)
    guarded = _run(n_steps, freq, eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=False)

    assert _replaced_steps(default) == [n_steps - 1]
    assert guarded[3] == 0 and not guarded[0]["Battery_Replaced"].any()
    assert len(guarded[4]) == len(default[4]) == 1
    assert guarded[4]["SOH"].iloc[-1] < 100.0


def test_a_run_without_a_battery_is_unchanged():
    pv, load, _temperature = _inputs(60)

    def run(allow):
        return simulate_energy_balance(
            pv_dc=pv,
            houseload=load,
            battery_config=BatteryConfig(nominal_energy_wh=0.0, allow_terminal_replacement=allow),
            freq="h",
        )

    default, guarded = run(True), run(False)
    pd.testing.assert_frame_equal(default[0], guarded[0], check_exact=True)
    assert default[3] == guarded[3] == 0


def test_disabled_replacement_stays_disabled():
    _assert_same_run(
        _run(60, eol_percentage=EVERY_PERIOD_EOL, enable_replacement=False),
        _run(60, eol_percentage=EVERY_PERIOD_EOL, enable_replacement=False, allow_terminal_replacement=False),
    )


@pytest.mark.parametrize("freq", FREQS)
def test_the_summary_path_places_the_final_period_as_the_frames_do(freq):
    spd = _steps_per_day(freq)
    n_steps = 2 * spd + 5 * spd // 24
    pv, load, temperature = _inputs(n_steps, freq)
    aligned = align_simulation_inputs(pv, load, temperature, freq=freq)
    config = BatteryConfig(nominal_energy_wh=5000.0, eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=False)

    summary = simulate_energy_balance_summary(aligned=aligned, battery_config=config, freq=freq)
    detailed = _run(n_steps, freq, eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=False)

    assert list(summary.replacement_steps) == _replaced_steps(detailed)
    assert summary.n_replacements == detailed[3] == 2
    assert summary.replaced_capacity_wh == detailed[0]["Battery_Replaced_Capacity_Wh"].sum()
    assert summary.final_soh_percent == detailed[4]["SOH"].iloc[-1]
    assert summary.carried_energy_wh == detailed[0]["Battery_Energy_End"].iloc[-1]
    assert summary.carried_pv_origin_energy_wh == detailed[0]["Battery_PV_Origin_Energy_End"].iloc[-1]


# -- Python and Numba ---------------------------------------------------------------


@pytest.mark.parametrize("freq", FREQS)
@pytest.mark.parametrize("tail", [0, 5])
@pytest.mark.parametrize("engine", [None, BLAST], ids=["native", "blast"])
def test_python_and_numba_skip_the_same_swap(freq, tail, engine):
    _require_numba()
    spd = _steps_per_day(freq)
    n_steps = 3 * spd + tail * spd // 24
    config = {"eol_percentage": EVERY_PERIOD_EOL, "allow_terminal_replacement": False}
    python = _run(n_steps, freq, engine=engine, **config)
    numba = _run(n_steps, freq, backend="numba", engine=engine, **config)

    _assert_same_run(python, numba)
    assert n_steps - 1 not in _replaced_steps(numba)


def test_both_settings_share_one_compiled_dispatch_signature():
    _require_numba()
    from breos._numba_dispatch import _kernel

    _run(48, backend="numba", eol_percentage=EVERY_PERIOD_EOL)
    signatures = list(_kernel().signatures)
    for allow in (False, True, False):
        _run(48, backend="numba", eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=allow)
        _run(48, "15min", backend="numba", eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=allow)
    assert list(_kernel().signatures) == signatures


# -- the shared projection -----------------------------------------------------------


def _project(allow: bool, *, aligned: bool = False, years: int = 3, eol: float = EVERY_PERIOD_EOL, record=None):
    pv, load, temperature = _inputs(72)
    year_aligned = align_simulation_inputs(pv, load, temperature, freq="h") if aligned else None

    def year_inputs(_year_idx):
        if year_aligned is not None:
            return ProjectionYear(1.0, aligned=year_aligned)
        return ProjectionYear(1.0, pv_dc=pv, houseload=load, temperature_series=temperature)

    def battery(soh_pct):
        config = BatteryConfig(
            nominal_energy_wh=5000.0,
            initial_soh=soh_pct,
            eol_percentage=eol,
            allow_terminal_replacement=allow,
        )
        if record is not None:
            record.append(config)
        return config

    return project_years(
        years, year_inputs, battery_config=battery, freq="h", has_battery=True, execution_backend="python"
    )


def _seen_permissions(monkeypatch, aligned: bool) -> list[bool]:
    seen: list[bool] = []
    name = "simulate_energy_balance_summary" if aligned else "simulate_energy_balance"
    real = getattr(projection_module, name)

    def recording(**kwargs):
        seen.append(kwargs["battery_config"].allow_terminal_replacement)
        return real(**kwargs)

    monkeypatch.setattr(projection_module, name, recording)
    return seen


@pytest.mark.parametrize("aligned", [False, True], ids=["frames", "summary"])
def test_the_projection_applies_the_policy_to_its_final_year_only(monkeypatch, aligned):
    seen = _seen_permissions(monkeypatch, aligned)
    built: list[BatteryConfig] = []
    default = _project(True, aligned=aligned)
    assert seen == [True, True, True]
    seen.clear()
    guarded = _project(False, aligned=aligned, record=built)

    assert seen == [True, True, False]
    # The earlier years ran on copies; the caller's configs are untouched.
    assert [config.allow_terminal_replacement for config in built] == [False, False, False]
    assert default.yearly_df["Replacements"].tolist() == [3, 3, 3]
    assert guarded.yearly_df["Replacements"].tolist() == [3, 3, 2]
    assert guarded.yearly_df["Replacement_Steps"].tolist() == ["23;47;71", "23;47;71", "23;47"]
    # Each earlier year's close still swaps, and the next year inherits that pack.
    pd.testing.assert_frame_equal(default.yearly_df.iloc[:2], guarded.yearly_df.iloc[:2], check_exact=True)
    assert default.carry.soh_pct == 100.0
    assert guarded.carry.soh_pct < 100.0
    assert guarded.total_replacements == default.total_replacements - 1


def test_frames_and_summaries_agree_on_the_guarded_projection():
    frames = _project(False)
    summary = _project(False, aligned=True)
    pd.testing.assert_frame_equal(frames.yearly_df, summary.yearly_df, check_exact=True)
    assert frames.carry.soh_pct == summary.carry.soh_pct
    assert frames.carry.energy_wh == summary.carry.energy_wh


def test_a_final_year_crossing_leaves_the_earlier_years_alone():
    # The unreplaced project's period closes, to place end of life between
    # the last two of them.
    soh: list[float] = []
    pv, load, temperature = _inputs(72)
    carry = projection_module.CarryState()
    for year in range(3):
        run = simulate_energy_balance(
            pv_dc=pv,
            houseload=load,
            temperature_series=temperature,
            battery_config=BatteryConfig(nominal_energy_wh=5000.0, initial_soh=carry.soh_pct, enable_replacement=False),
            freq="h",
            return_degradation_state=True,
            finalize_degradation=year == 2,
            **carry.simulation_kwargs(),
        )
        carry = carry.after_frames(run[0], run[4], run[5], has_battery=True)
        soh.extend(run[4]["SOH"].tolist())
    eol = (soh[-2] + soh[-1]) / 200.0

    default = _project(True, eol=eol)
    guarded = _project(False, eol=eol)

    assert default.yearly_df["Replacements"].tolist() == [0, 0, 1]
    assert guarded.yearly_df["Replacements"].tolist() == [0, 0, 0]
    pd.testing.assert_frame_equal(default.yearly_df.iloc[:2], guarded.yearly_df.iloc[:2], check_exact=True)
    assert guarded.carry.soh_pct == soh[-1]
    assert np.isnan(guarded.yearly_df["Replacement_Year_Fraction"].iloc[-1])


# -- priced results -------------------------------------------------------------------


def _without_final_swap(rows: pd.DataFrame, hours_per_step: float = 1.0) -> pd.DataFrame:
    """The year rows with the final year's last swap removed, as a ledger edit."""
    rows = rows.copy()
    last = rows.index[-1]
    steps = [int(step) for step in str(rows.at[last, "Replacement_Steps"]).split(";") if step]
    n_steps = round(rows.at[last, "Simulated_Hours"] / hours_per_step)
    rows.at[last, "Replacements"] = rows.at[last, "Replacements"] - 1
    rows.at[last, "Replacement_Steps"] = ";".join(str(step) for step in steps[:-1])
    rows.at[last, "Replacement_Year_Fraction"] = replacement_fraction_from_steps(steps[:-1], n_steps)
    return rows


def _app(**overrides) -> App:
    config = {
        "location": "porto",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5,
        "start_date": "2025-01-01",
        "projection_years": 2,
        "cost_preset": "residential_pt",
        "resolution": "h",
        "battery_eol_percentage": EVERY_PERIOD_EOL,
        **PRICING,
        **overrides,
    }
    if "period" in config:
        # A [period] window runs once and takes no projection_years.
        del config["projection_years"]
    with (
        mock.patch("breos.app.fetch_tmy_weather_data", _fake_fetch),
        mock.patch("breos.app.load_weather", lambda **_kwargs: None),
    ):
        app = App(config)
        app.simulate()
    return app


def test_app_prices_every_swap_but_the_terminal_one():
    default_app, guarded_app = _app(), _app(battery_allow_terminal_replacement=False)
    default, guarded = default_app.result(), guarded_app.result()
    default_art, guarded_art = default_app._artifacts, guarded_app._artifacts

    assert default["provenance"]["resolved_config"]["battery_allow_terminal_replacement"] is True
    assert guarded["provenance"]["resolved_config"]["battery_allow_terminal_replacement"] is False
    assert guarded["result_schema_version"] == RESULT_SCHEMA_VERSION == "2.4"
    assert default_art.yearly_df["Replacements"].tolist() == [365, 365]
    assert guarded_art.yearly_df["Replacements"].tolist() == [365, 364]
    assert guarded["battery_replacements"] == default["battery_replacements"] - 1
    pack = default_art.costs["replacement_cost_each"]
    assert pack == 2000.0
    assert guarded_art.yearly_df["Replacement_Cost"].tolist() == [730000.0, 728000.0]
    assert guarded["battery_replacement_cost_t0_prices"] == default["battery_replacement_cost_t0_prices"] - pack
    assert guarded["battery_soh_end_pct"] < 100.0 == default["battery_soh_end_pct"]
    assert guarded["provenance"]["degradation"]["replacement_events"] == [
        {"year": 1, "count": 365},
        {"year": 2, "count": 364},
    ]

    # The guarded money is the existing event-time pricing of the default
    # ledger with its terminal swap removed: the remaining swaps keep their
    # instants, inflation, learning and discounting.
    resolved = resolve_app_config(default_app._config)
    assert default_art.projection is not None and guarded_art.projection is not None
    ledger = _without_final_swap(default_art.projection.yearly_df)
    event_columns = ["Replacements", "Replacement_Steps", "Replacement_Year_Fraction"]
    pd.testing.assert_frame_equal(guarded_art.projection.yearly_df[event_columns], ledger[event_columns])
    assert guarded_art.yearly_df["Replaced_Capacity_kWh"].tolist() == [365 * 5.0, 364 * 5.0]
    expected = value_projection(resolved.cfg, resolved, replace(default_art.projection, yearly_df=ledger))
    pd.testing.assert_frame_equal(guarded_art.cost_projection, expected.cost_projection, check_exact=True)
    assert guarded["battery_replacement_cost_npv"] == round(expected.cost_projection.attrs["replacement_cost_npv"], 2)
    assert guarded["npv_savings"] > default["npv_savings"]
    # Year 1, whose close still swaps, is untouched.
    default_financial = {row["year"]: row for row in default["financial"]}
    guarded_financial = {row["year"]: row for row in guarded["financial"]}
    assert guarded_financial[1] == default_financial[1]


def test_app_without_a_terminal_crossing_is_unchanged():
    default = _app(battery_eol_percentage=0.7).result()
    guarded = _app(battery_eol_percentage=0.7, battery_allow_terminal_replacement=False).result()
    default["provenance"]["resolved_config"].pop("battery_allow_terminal_replacement")
    guarded["provenance"]["resolved_config"].pop("battery_allow_terminal_replacement")
    for result in (default, guarded):
        result["provenance"].pop("execution", None)
    assert default == guarded


def _mc_config(**overrides):
    return {
        "location": "porto",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "cost_preset": "residential_pt",
        "resolution": "h",
        "projection_years": 2,
        **PRICING,
        **overrides,
    }


def test_monte_carlo_skips_the_terminal_swap_of_each_trajectory(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=7)

    default = run_montecarlo(_mc_config(battery_eol_percentage=EVERY_PERIOD_EOL), settings)
    guarded = run_montecarlo(
        _mc_config(battery_eol_percentage=EVERY_PERIOD_EOL, battery_allow_terminal_replacement=False), settings
    )

    assert default.provenance["resolved_config"]["battery_allow_terminal_replacement"] is True
    assert guarded.provenance["resolved_config"]["battery_allow_terminal_replacement"] is False
    assert (default.runs["total_replacements"] == 730).all()
    assert (guarded.runs["total_replacements"] == 729).all()
    np.testing.assert_array_equal(
        guarded.runs["total_replacement_cost_t0_prices"], default.runs["total_replacement_cost_t0_prices"] - 2000.0
    )


def test_monte_carlo_without_a_terminal_crossing_is_unchanged(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv")
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=2, years_per_run=2, seed=7)

    default = run_montecarlo(_mc_config(), settings)
    guarded = run_montecarlo(_mc_config(battery_allow_terminal_replacement=False), settings)

    pd.testing.assert_frame_equal(default.runs, guarded.runs, check_exact=True)


def _projected_metrics(monkeypatch, **batt_spec):
    """Project a two-"year" design of three hourly days each, with real dispatch and economics."""
    pv, load, temperature = _inputs(72)
    captured: dict = {"permissions": []}

    def recording(**kwargs):
        captured.update(kwargs)
        return cost_analysis_projection(**kwargs)

    def recording_balance(**kwargs):
        captured["permissions"].append(kwargs["battery_config"].allow_terminal_replacement)
        return simulate_energy_balance(**kwargs)

    monkeypatch.setattr(optimization_module, "cost_analysis_projection", recording)
    monkeypatch.setattr(projection_module, "simulate_energy_balance", recording_balance)
    metrics = _evaluate_projected_design_metrics(
        base_dc_power=pv,
        houseload=load,
        temperature_series=temperature,
        pv_params=get_module("Suntech_STP550S_STC"),
        batt_spec={"enable_replacement": True, "eol_percentage": EVERY_PERIOD_EOL, **batt_spec},
        costs_cfg={"storage_cost_per_kwh": 400.0},
        fin_cfg={"inflation_rate": 0.02, "sell_price_inflation": 0.0, "discount_rate": 0.05},
        freq="h",
        years_projection=2,
        degradation_rate=0.0,
        n_modules=4,
        battery_kwh=5.0,
        inverter_efficiency=0.96,
        inverter_ac_capacity_w=None,
        return_tables=True,
    )
    return metrics, captured


def test_projected_optimizer_skips_the_terminal_swap_and_its_money(monkeypatch):
    default, default_inputs = _projected_metrics(monkeypatch)
    guarded, guarded_inputs = _projected_metrics(monkeypatch, allow_terminal_replacement=False)

    assert default["_yearly_summary_df"]["Replacements"].tolist() == [3, 3]
    assert guarded["_yearly_summary_df"]["Replacements"].tolist() == [3, 2]
    assert guarded["Projected_Total_Replacements"] == default["Projected_Total_Replacements"] - 1
    assert guarded["Projected_Replacement_Cost_T0_Prices"] == default["Projected_Replacement_Cost_T0_Prices"] - 2000.0
    assert guarded["Projected_Final_SOH_%"] < 100.0 == default["Projected_Final_SOH_%"]

    # The default ledger without its terminal swap, priced by the existing
    # event-time economics, is the guarded run's money.
    rows = _without_final_swap(default_inputs.pop("yearly_summary_df")).drop(columns="Replacement_Cost")
    default_inputs.pop("permissions")
    expected = cost_analysis_projection(
        **default_inputs, yearly_summary_df=price_year_rows(rows, default_inputs["costs"])
    )
    pd.testing.assert_frame_equal(guarded["_cost_projection_df"], expected, check_exact=True)
    assert guarded["Projected_NPV"] == float(expected["Savings_Cumulative_NPV"].iloc[-1])
    assert guarded_inputs["costs"] == default_inputs["costs"]


def test_projected_optimizer_without_a_terminal_crossing_is_unchanged(monkeypatch):
    default, default_inputs = _projected_metrics(monkeypatch, eol_percentage=0.7)
    guarded, guarded_inputs = _projected_metrics(monkeypatch, eol_percentage=0.7, allow_terminal_replacement=False)
    assert default_inputs["permissions"] == [True, True]
    assert guarded_inputs["permissions"] == [True, False]
    pd.testing.assert_frame_equal(default.pop("_yearly_summary_df"), guarded.pop("_yearly_summary_df"))
    pd.testing.assert_frame_equal(default.pop("_cost_projection_df"), guarded.pop("_cost_projection_df"))
    assert default == guarded


# -- provenance -----------------------------------------------------------------------


def _design_inputs():
    pv, _load, _temperature = _inputs(72)
    weather = pd.DataFrame({"temp_air": np.full(72, 20.0)}, index=pv.index)
    load = pd.DataFrame({"Load": np.full(72, 500.0)}, index=pv.index)
    return weather, load, pv


@pytest.mark.parametrize("value", [None, True, False])
def test_a_projected_design_records_the_policy(monkeypatch, value):
    weather, load, pv = _design_inputs()
    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **_kwargs: pv)
    battery = {"eol_percentage": EVERY_PERIOD_EOL}
    if value is not None:
        battery["allow_terminal_replacement"] = value
    result = evaluate_projected_design(
        weather,
        load,
        {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "simulation": {"resolution": "h", "years_projection": 2},
            "financials": {"project_lifespan": 2},
            "battery": battery,
        },
        n_modules=4,
        battery_kwh=5.0,
        tilt=35.0,
        azimuth=180.0,
    )

    treatment = result.provenance["battery_replacement_treatment"]
    expected = True if value is None else value
    assert treatment["allow_terminal_replacement"] is expected
    assert treatment["method"] == "simulated_yearly_state_propagation"
    assert "ends on the horizon's last step, whole or partial" in treatment["terminal_period"]
    assert result.provenance["result_schema_version"] == "2.4"
    assert result.yearly["Replacements"].tolist() == ([3, 3] if expected else [3, 2])


@pytest.mark.parametrize("value", [True, False])
def test_an_optimizer_search_records_the_policy(monkeypatch, value):
    pytest.importorskip("pymoo")
    from breos.optimization import optimize_system_multi_objective

    idx = pd.date_range("2025-01-01 00:00", periods=4, freq="h", tz="UTC")
    tmy_data = pd.DataFrame({"temp_air": [15.0, 16.0, 17.0, 18.0], "ghi": [0.0, 500.0, 500.0, 0.0]}, index=idx)
    houseload = pd.DataFrame({"Load": [500.0] * 4}, index=idx)
    monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **_kwargs: pd.Series(0.0, index=idx))
    seen: list[bool] = []

    def stub(**kwargs):
        seen.append(kwargs["batt_spec"]["allow_terminal_replacement"])
        return {
            "Projected_Grid_Independence_%": 50.0,
            "Projected_ZEB_Ratio": 0.5,
            "Projected_NPV": 1000.0,
            "Projected_Initial_Cost": 500.0,
        }

    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", stub)
    result = optimize_system_multi_objective(
        tmy_data,
        houseload,
        {
            "location": {"latitude": 41.15, "longitude": -8.61, "timezone": "UTC"},
            "simulation": {"resolution": "h"},
            "constraints": {"max_modules": 4, "max_battery_kwh": 3.0},
            "mode": {"fixed_azimuth": 180},
            "battery": {"temperature": 20.0, "allow_terminal_replacement": value},
        },
        pop_size=4,
        n_gen=1,
        seed=1,
        verbose=False,
    )

    recorded = result.details["provenance"]["battery_replacement_treatment"]
    assert recorded == result.details["battery_replacement_treatment"]
    assert recorded["allow_terminal_replacement"] is value
    assert recorded["method"] == "simulated_yearly_state_propagation"
    assert seen and set(seen) == {value}


# -- smart charging -------------------------------------------------------------------


def _controller_run(scenario, controller, allow):
    battery = _battery(eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=allow)
    return _core(scenario, "python", controller=controller, battery=battery)


@pytest.mark.parametrize("freq", FREQS)
@pytest.mark.parametrize("policy", ["fixed_target", "daily_persistence"])
def test_controllers_on_offset_civil_days_leave_the_guard_on_the_aging_windows(freq, policy):
    spd = _steps_per_day(freq)
    # Starts mid-day in Berlin and ends with a partial window, so civil days
    # and degradation windows never share a boundary.
    scenario = _scenario("2024-03-28T10:00Z", 3 * spd + 7 * spd // 24, freq, "Europe/Berlin")

    def controller():
        if policy == "fixed_target":
            return _Recording(FixedTargetDayController(scenario.instructions))
        return _Recording(_small_grid(freq, "python"))

    default_controller, guarded_controller = controller(), controller()
    default = _controller_run(scenario, default_controller, True)
    guarded = _controller_run(scenario, guarded_controller, False)
    n_steps = scenario.n_steps

    default_steps = np.flatnonzero(default.buffers.replaced).tolist()
    guarded_steps = np.flatnonzero(guarded.buffers.replaced).tolist()
    assert default_steps == [spd - 1, 2 * spd - 1, 3 * spd - 1, n_steps - 1]
    assert guarded_steps == default_steps[:-1]
    # Actual execution only: every decision, and so every executed step, is
    # the same, including the planner's terminal energy target.
    assert guarded_controller.decisions == default_controller.decisions
    assert guarded.controller_instructions == default.controller_instructions
    assert np.array_equal(guarded.buffers.matrix[:, :-1], default.buffers.matrix[:, :-1])
    # A decision after an interior swap sees the new pack.
    after_swap = [day for day in guarded_controller.days if day.project_step_ordinal > spd - 1]
    assert after_swap and all(day.battery_state.soh_fraction == 1.0 for day in after_swap)
    assert guarded.aging.soh_fraction < 1.0 == default.aging.soh_fraction


@pytest.mark.parametrize("freq", FREQS)
def test_static_smart_charging_instructions_follow_the_same_guard(freq):
    spd = _steps_per_day(freq)
    scenario = _scenario("2024-03-28T10:00Z", 3 * spd + 7 * spd // 24, freq, "Europe/Berlin")

    def run(allow):
        return _core(
            scenario, "python", battery=_battery(eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=allow)
        )

    default, guarded = run(True), run(False)
    assert np.flatnonzero(guarded.buffers.replaced).tolist() == np.flatnonzero(default.buffers.replaced).tolist()[:-1]
    assert guarded.buffers.columns["Grid_AC_To_Battery"].sum() == default.buffers.columns["Grid_AC_To_Battery"].sum()


def test_a_controller_projection_keeps_its_year_seam():
    freq, spd = "h", 24
    scenario = _scenario("2023-01-01T00:00Z", 3 * spd, freq, "Europe/Lisbon")

    def project(allow):
        recording = _Recording(FixedTargetDayController(scenario.instructions))
        run = project_years(
            2,
            lambda _year: ProjectionYear(
                1.0, pv_dc=scenario.pv, houseload=scenario.load, temperature_series=scenario.temperature
            ),
            battery_config=lambda soh: _battery(
                initial_soh=soh, eol_percentage=EVERY_PERIOD_EOL, allow_terminal_replacement=allow
            ),
            freq=freq,
            has_battery=True,
            execution_backend="python",
            tariff=scenario.tariff,
            day_controller=recording,
        )
        return run, recording

    (default, default_days), (guarded, guarded_days) = project(True), project(False)

    assert default.yearly_df["Replacement_Steps"].tolist() == ["23;47;71", "23;47;71"]
    assert guarded.yearly_df["Replacement_Steps"].tolist() == ["23;47;71", "23;47"]
    pd.testing.assert_frame_equal(default.yearly_df.iloc[:1], guarded.yearly_df.iloc[:1], check_exact=True)
    assert guarded_days.decisions == default_days.decisions
    assert guarded.controller_instructions == default.controller_instructions
    # The second year's first decision sees the pack the first year's close installed.
    second_year = [day for day in guarded_days.days if day.projection_year == 1]
    assert second_year[0].battery_state.soh_fraction == 1.0


def test_app_daily_persistence_honours_the_guard_with_unchanged_decisions():
    from tests.test_daily_persistence import COARSE, DAILY, JANUARY, TOU

    config = {
        "location": "porto",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "start_date": "2025-01-01",
        "battery_kwh": 5.0,
        "execution_backend": "python",
        "tariff": TOU,
        "smart_charging": {**DAILY, **COARSE},
        "period": JANUARY,
        "battery_eol_percentage": EVERY_PERIOD_EOL,
    }
    default = _app(**config).result()
    guarded = _app(**config, battery_allow_terminal_replacement=False).result()

    assert guarded["battery_replacements"] == default["battery_replacements"] - 1 == 3
    default_record = default["provenance"]["smart_charging"]
    guarded_record = guarded["provenance"]["smart_charging"]
    assert guarded_record["instruction_hash"] == default_record["instruction_hash"]
    # The default ends on a fresh, full, unattributed pack; the guarded run
    # carries the old pack's energy and origins.
    assert default_record["final_stored_energy"]["unattributed_wh"] == default_record["final_stored_energy"]["total_wh"]
    assert default_record["final_stored_energy"]["total_wh"] == 5000.0 * 0.9
    assert guarded_record["final_stored_energy"] != default_record["final_stored_energy"]


def test_the_key_reference_lists_the_setting():
    reference = (Path(__file__).resolve().parents[1] / "docs" / "getting-started" / "config-reference.md").read_text()
    assert "| `battery_allow_terminal_replacement` | `True` | — |" in reference
