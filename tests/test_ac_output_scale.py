"""The post-inverter AC output scale and the per-year projected weather sequence.

Both were added for the upcoming publication's PV-bias scenarios. Two properties
carry every result that depends on them, and both are asserted here rather than
argued: the default reproduces prior behaviour exactly, and the compiled
backend reproduces the Python reference exactly when the scale is active.

The scale is applied *after* the part-load curve and every inverter limit, so
it corrects modelled AC delivery without moving the clipping threshold or the
part-load ratio. Those invariances are asserted too, because folding the factor
into ``inverter_efficiency`` instead would silently break them.

The factor is an in-dispatch derate rather than a post-processing multiplier:
it is applied inside the conversion the dispatcher calls, so the reachable AC
ceiling and the battery discharge decisions respond to it. The ceiling test
below asserts exactly that, and it is the reason the factor is not simply
applied to a finished result series.

Landing after the nameplate limit is also what bounds the factor to ``(0, 1]``.
Above 1 the inverter would deliver more than its nameplate and more AC than the
DC entering it, so the reported conversion loss would pin at zero and the AC
ledger would stop balancing. An under-predicting model is corrected on the DC
side with ``dc_output_scale``, which stays unbounded above because clipping and
the part-load ratio respond to it. Both invariants are asserted below.

While the derate is active the reported conversion loss covers the whole
DC-to-AC shortfall, not the converter's own loss alone. The invariant asserted
here is that it stays non-negative and equals the real difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breos.battery import BatteryConfig, simulate_energy_balance
from breos.inverter import (
    _calculate_dc_ac_power_arrays,
    calculate_dc_ac_power,
    dc_power_for_ac_output,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "parity"))

from harness import FREQ, build  # noqa: E402

AC_RATING = 3520.0
EFFICIENCY = 0.96
# The upcoming-publication AC-side factor, a deeper trim, plus a no-op.
SCALES = (1.0, 1.0 / 1.0857, 0.5)
DC_GRID = (0.0, 1e-9, 50.0, 500.0, 1800.0, 3520.0, 3666.6666666666665, 9000.0)


@pytest.mark.parametrize("dc", DC_GRID)
def test_default_scale_is_a_no_op(dc):
    """An explicit 1.0 must be bit-identical to omitting the argument."""
    without = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY)
    with_one = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY, 1.0)
    assert without.ac_power_w == with_one.ac_power_w
    assert without.conversion_loss_w == with_one.conversion_loss_w
    assert without.clipping_loss_dc_w == with_one.clipping_loss_dc_w
    assert without.clipping_loss_ac_equivalent_w == with_one.clipping_loss_ac_equivalent_w


def test_default_scale_is_a_no_op_on_the_array_path():
    dc = np.array(DC_GRID)
    without = _calculate_dc_ac_power_arrays(dc, AC_RATING, EFFICIENCY)
    with_one = _calculate_dc_ac_power_arrays(dc, AC_RATING, EFFICIENCY, 1.0)
    for left, right in zip(without, with_one):
        assert np.array_equal(left, right)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dc", DC_GRID)
def test_scaling_is_exactly_multiplicative_on_ac(scale, dc):
    reference = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY).ac_power_w
    scaled = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY, scale).ac_power_w
    assert scaled == pytest.approx(reference * scale, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dc", DC_GRID)
def test_scaling_does_not_move_dc_side_clipping(scale, dc):
    """Clipping is a DC-side inverter behaviour and must be untouched."""
    reference = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY)
    scaled = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY, scale)
    assert scaled.clipping_loss_dc_w == reference.clipping_loss_dc_w


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("fraction", (0.0005, 0.05, 0.4, 0.85, 0.999))
def test_inverse_is_the_exact_inverse_at_the_same_scale(scale, fraction):
    """Dispatch sizes DC against the inverse and delivers against the forward.

    The contract holds up to the scaled nameplate. Targets are expressed as a
    fraction of it so the achievable range is tested at every scale rather
    than only at the unscaled one.
    """
    target = fraction * AC_RATING * scale
    dc = dc_power_for_ac_output(target, AC_RATING, EFFICIENCY, scale)
    assert calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY, scale).ac_power_w == pytest.approx(
        target, rel=0.0, abs=1e-9
    )


@pytest.mark.parametrize("scale", SCALES)
def test_requests_above_the_scaled_nameplate_clamp_to_it(scale):
    """Scaling moves the deliverable AC ceiling, and the inverse must agree."""
    ceiling = AC_RATING * scale
    dc = dc_power_for_ac_output(ceiling * 1.5, AC_RATING, EFFICIENCY, scale)
    delivered = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY, scale).ac_power_w
    assert delivered == pytest.approx(ceiling, rel=0.0, abs=1e-9)
    # The DC request is the unscaled saturation input; the scale is an AC-side
    # correction and must not change how much DC the inverter can accept.
    assert dc == pytest.approx(AC_RATING / EFFICIENCY, rel=0.0, abs=1e-9)


@pytest.mark.parametrize("scale", SCALES)
def test_scaling_is_multiplicative_on_the_array_path(scale):
    dc = np.array(DC_GRID)
    ac_reference, _, clip_reference = _calculate_dc_ac_power_arrays(dc, AC_RATING, EFFICIENCY)
    ac_scaled, _, clip_scaled = _calculate_dc_ac_power_arrays(dc, AC_RATING, EFFICIENCY, scale)
    assert np.allclose(ac_scaled, ac_reference * scale, rtol=0.0, atol=1e-12)
    assert np.array_equal(clip_scaled, clip_reference)


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_battery_config_rejects_a_non_positive_scale(value):
    with pytest.raises(ValueError):
        BatteryConfig(nominal_energy_wh=5000.0, ac_output_scale=value)


@pytest.mark.parametrize("value", (1.0857, 1.5, 2.0, 1.0 + 1e-9))
def test_battery_config_rejects_a_scale_above_one(value):
    """Above 1 the factor would push the inverter past its own nameplate."""
    with pytest.raises(ValueError, match="at most 1"):
        BatteryConfig(nominal_energy_wh=5000.0, ac_output_scale=value)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("dc", DC_GRID)
def test_scaled_ac_never_exceeds_the_nameplate_or_its_own_dc(scale, dc):
    """The two invariants the (0, 1] bound exists to protect.

    Delivered AC stays at or below the inverter rating, and at or below the DC
    entering the converter, so the conversion loss the ledger reports is a
    real non-negative quantity rather than a clamp hiding created energy.
    """
    result = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY, scale)
    assert result.ac_power_w <= AC_RATING + 1e-12
    dc_into_inverter = dc - result.clipping_loss_dc_w
    assert result.ac_power_w <= dc_into_inverter + 1e-12
    assert result.conversion_loss_w >= 0.0
    assert result.conversion_loss_w == pytest.approx(dc_into_inverter - result.ac_power_w, rel=0.0, abs=1e-9)


@pytest.mark.parametrize("dc", DC_GRID)
def test_helpers_clamp_a_scale_above_one_to_one(dc):
    """A direct helper call cannot produce an unphysical result either."""
    bounded = calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY, 1.0).ac_power_w
    assert calculate_dc_ac_power(dc, AC_RATING, EFFICIENCY, 1.5).ac_power_w == bounded
    array_scaled, _, _ = _calculate_dc_ac_power_arrays(np.array([dc]), AC_RATING, EFFICIENCY, 1.5)
    assert array_scaled[0] == bounded
    assert dc_power_for_ac_output(1000.0, AC_RATING, EFFICIENCY, 1.5) == dc_power_for_ac_output(
        1000.0, AC_RATING, EFFICIENCY, 1.0
    )


def test_battery_config_defaults_to_one():
    assert BatteryConfig(nominal_energy_wh=5000.0).ac_output_scale == 1.0


def _simulate(scenario: str, backend: str, scale: float):
    pv, load, temp, cfg, sim_kwargs = build(scenario)
    config = dict(cfg)
    config["ac_output_scale"] = scale
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**config),
        freq=FREQ,
        temperature_series=temp,
        return_degradation_state=True,
        execution_backend=backend,
        **sim_kwargs,
    )


@pytest.mark.parametrize("scenario", ("one_day", "baseline", "saturating", "discharge_limited"))
def test_dispatch_default_scale_reproduces_the_unscaled_run(scenario):
    """Passing 1.0 through the whole dispatch must change nothing."""
    reference = _simulate(scenario, "python", 1.0)
    pv, load, temp, cfg, sim_kwargs = build(scenario)
    plain = simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=BatteryConfig(**cfg),
        freq=FREQ,
        temperature_series=temp,
        return_degradation_state=True,
        execution_backend="python",
        **sim_kwargs,
    )
    for column in plain[0].columns:
        if column == "Datetime":
            continue
        assert np.array_equal(plain[0][column].to_numpy(), reference[0][column].to_numpy()), (
            f"{scenario}: {column} moved under an explicit 1.0"
        )
    assert plain[1] == reference[1]


@pytest.mark.parametrize("scenario", ("one_day", "saturating", "discharge_limited"))
def test_scaling_reduces_delivered_ac(scenario):
    full = _simulate(scenario, "python", 1.0)[0]
    reduced = _simulate(scenario, "python", 1.0 / 1.0857)[0]
    assert reduced["Import_From_Grid"].sum() > full["Import_From_Grid"].sum()


class TestCompiledBackendParity:
    """The compiled kernel must mirror the reference with the scale active."""

    @pytest.mark.parametrize("scenario", ("one_day", "baseline", "saturating", "discharge_limited"))
    @pytest.mark.parametrize("scale", (1.0 / 1.0857, 0.5))
    def test_numba_matches_python_exactly(self, scenario, scale):
        pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
        python_out = _simulate(scenario, "python", scale)
        numba_out = _simulate(scenario, "numba", scale)
        for column in python_out[0].columns:
            if column == "Datetime":
                continue
            left = python_out[0][column].to_numpy()
            right = numba_out[0][column].to_numpy()
            assert np.array_equal(left, right), (
                f"{scenario} at scale {scale}: column {column} differs at {int(np.flatnonzero(left != right)[0])}"
            )
        assert python_out[1] == numba_out[1]
        assert python_out[3] == numba_out[3]
        assert python_out[4] == numba_out[4]


def _scaled_battery_ceiling_run(backend: str):
    """Run one over-load hour against a scaled 1 kW shared inverter."""
    index = pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC")
    pv = pd.Series([0.0], index=index)
    load = pd.DataFrame({"Load": [1200.0]}, index=index)
    temperature = pd.Series([25.0], index=index)
    config = BatteryConfig(
        nominal_energy_wh=2000.0,
        min_soc=0.0,
        max_soc=1.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        standby_loss_wh=0.0,
        enable_replacement=False,
        inverter_efficiency=1.0,
        inverter_ac_capacity_w=1000.0,
        ac_output_scale=0.8,
    )
    return simulate_energy_balance(
        pv_dc=pv,
        houseload=load,
        battery_config=config,
        freq="h",
        temperature_series=temperature,
        initial_energy_wh=2000.0,
        execution_backend=backend,
    )


def test_scaled_inverter_ceiling_binds_and_is_backend_identical():
    """A 0.8 AC correction caps battery delivery below the 1.2 kW load.

    The 1 kW inverter scaled by 0.8 can reach 800 W, so the remaining 400 W
    is imported. The ceiling moves down with the factor and never above the
    nameplate, which is the behaviour the (0, 1] bound guarantees.
    """
    pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
    python_out = _scaled_battery_ceiling_run("python")
    numba_out = _scaled_battery_ceiling_run("numba")

    assert python_out[0]["Battery_AC_To_Load"].iloc[0] == pytest.approx(800.0)
    assert python_out[0]["Import_From_Grid"].iloc[0] == pytest.approx(400.0)
    for column in python_out[0].columns:
        if column != "Datetime":
            assert np.array_equal(python_out[0][column].to_numpy(), numba_out[0][column].to_numpy()), column
    assert python_out[1] == numba_out[1]


class TestProjectedWeatherSequence:
    """A per-year weather sequence, used by the historical-weather scenarios."""

    @staticmethod
    def _inputs():
        import tomllib

        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / "validation/article1/revision-0.6.1/article1-power-1c.toml").read_text())
        config["simulation"]["years_projection"] = 2
        return config

    def test_repeating_one_year_reproduces_the_repeated_tmy_run(self):
        """A sequence of identical years must be bit-identical to the default."""
        from breos.optimization import _evaluate_projected_design_metrics

        index = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        dc = pd.Series(np.linspace(0.0, 3000.0, 48), index=index)
        load = pd.DataFrame({"Load": np.full(48, 400.0)}, index=index)
        temperature = pd.Series(np.full(48, 20.0), index=index)
        from breos.pv_modules import get_module

        shared = dict(
            tmy_data=pd.DataFrame(index=index),
            houseload=load,
            temperature_series=temperature,
            pv_params=get_module("Suntech_STP550S_STC"),
            batt_spec={"calendar_model": "naumann_lam", "enable_replacement": False},
            costs_cfg={},
            fin_cfg={},
            freq="h",
            years_projection=2,
            degradation_rate=0.005,
            n_modules=6,
            battery_kwh=0.0,
            inverter_efficiency=0.96,
            inverter_ac_capacity_w=2640.0,
        )
        repeated = _evaluate_projected_design_metrics(base_dc_power=dc, **shared)
        sequenced = _evaluate_projected_design_metrics(base_dc_power=[dc, dc], **shared)
        assert repeated == sequenced

    def test_wrong_sequence_length_raises(self):
        from breos.optimization import _evaluate_projected_design_metrics

        index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        dc = pd.Series(np.zeros(24), index=index)
        from breos.pv_modules import get_module

        with pytest.raises(ValueError, match="expected 3"):
            _evaluate_projected_design_metrics(
                base_dc_power=[dc, dc],
                tmy_data=pd.DataFrame(index=index),
                houseload=pd.DataFrame({"Load": np.zeros(24)}, index=index),
                temperature_series=pd.Series(np.full(24, 20.0), index=index),
                pv_params=get_module("Suntech_STP550S_STC"),
                batt_spec={},
                costs_cfg={},
                fin_cfg={},
                freq="h",
                years_projection=3,
                degradation_rate=0.0,
                n_modules=1,
                battery_kwh=0.0,
                inverter_efficiency=0.96,
                inverter_ac_capacity_w=440.0,
            )

    def test_public_evaluator_accepts_a_weather_sequence(self, monkeypatch):
        """evaluate_projected_design must align on the reference year's index.

        Regression test for a defect the sequence tests above could not reach:
        they call _evaluate_projected_design_metrics directly, so the public
        wrapper that builds the per-year list was never exercised. It read
        base_dc_power.index, which on a list is the builtin list.index method
        rather than a DatetimeIndex, and every weather_by_year call died in
        pd.DatetimeIndex(...).
        """
        from breos.optimization import evaluate_projected_design

        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        weather = pd.DataFrame({"temp_air": np.full(24, 20.0)}, index=idx)
        load = pd.DataFrame({"Load": np.full(24, 500.0)}, index=idx)
        seen: list = []

        monkeypatch.setattr(
            "breos.optimization.calculate_pv_production_dc",
            lambda **_kwargs: pd.Series(np.linspace(0.0, 3000.0, 24), index=idx),
        )

        def _capture(_mode, index, **_kwargs):
            seen.append(index)
            return pd.Series(np.full(24, 20.0), index=idx)

        monkeypatch.setattr("breos.optimization._temperature_series_from_config", _capture)
        monkeypatch.setattr(
            "breos.optimization._evaluate_projected_design_metrics",
            lambda **kwargs: {
                "Projected_Grid_Independence_%": 60.0,
                "Projected_NPV_Eur": 100.0,
                "sequence_length": (
                    1 if isinstance(kwargs["base_dc_power"], pd.Series) else len(kwargs["base_dc_power"])
                ),
                "_yearly_summary_df": pd.DataFrame({"Year": [1, 2]}),
                "_cost_projection_df": pd.DataFrame({"Year": [1, 2]}),
            },
        )

        config = {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "pv": {"module": "Suntech_STP550S_STC"},
            "simulation": {"resolution": "h", "years_projection": 2},
            "financials": {"project_lifespan": 2},
            "costs": {"dc_ac_ratio": 1.25},
        }
        kwargs = dict(n_modules=9, battery_kwh=5.0, tilt=35.0, azimuth=200.0)

        result = evaluate_projected_design(weather, load, config, weather_by_year=[weather, weather], **kwargs)

        assert result.metrics["sequence_length"] == 2
        assert isinstance(seen[0], pd.DatetimeIndex)
        pd.testing.assert_index_equal(seen[0], idx)

        repeated = evaluate_projected_design(weather, load, config, **kwargs)
        assert repeated.metrics["sequence_length"] == 1
        pd.testing.assert_index_equal(seen[1], idx)


class TestUnlimitedInverterAndOptimizer:
    """The two paths the original patch left the scale out of."""

    @pytest.mark.parametrize("scale", [0.5, 0.9210647508519848, 1.0])
    def test_vectorized_infinite_rating_matches_the_scalar_reference(self, scale):
        """An unlimited inverter rating must scale like every other rating.

        The vectorized helper returned before applying the scale on the
        not isfinite branch, so the default inverter_ac_capacity_w=None path
        silently ignored the bias correction and disagreed with both the
        scalar reference and the compiled kernel.
        """
        from breos.inverter import _calculate_dc_ac_power_arrays, calculate_dc_ac_power

        dc = np.array([0.0, 500.0, 1000.0, 2000.0, 7500.0])
        ac, loss, clip = _calculate_dc_ac_power_arrays(dc, float("inf"), 0.96, scale)
        reference = np.array(
            [calculate_dc_ac_power(float(d), float("inf"), 0.96, ac_output_scale=scale).ac_power_w for d in dc]
        )
        assert np.array_equal(ac, reference)
        assert np.array_equal(ac, dc * 0.96 * scale)
        assert np.all(loss >= 0.0)
        assert np.array_equal(clip, np.zeros_like(dc))

    @staticmethod
    def _infinite_public_run(backend: str, *, with_battery: bool, scale: float):
        index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        pv = pd.Series([500.0, 1000.0, 2000.0], index=index)
        load = pd.DataFrame({"Load": np.zeros(3)}, index=index)
        temperature = pd.Series(np.full(3, 25.0), index=index)
        config = BatteryConfig(
            nominal_energy_wh=2000.0 if with_battery else 0.0,
            min_soc=0.0,
            max_soc=0.8 if with_battery else 1.0,
            inverter_efficiency=0.96,
            inverter_ac_capacity_w=None,
            ac_output_scale=scale,
            enable_replacement=False,
            standby_loss_wh=0.0,
        )
        return simulate_energy_balance(
            pv_dc=pv,
            houseload=load,
            battery_config=config,
            freq="h",
            temperature_series=temperature,
            initial_energy_wh=1600.0 if with_battery else None,
            execution_backend=backend,
        )

    @pytest.mark.parametrize("scale", [0.5, 0.8, 1.0])
    def test_infinite_public_total_pv_and_export_follow_scaled_ac(self, scale):
        """The vectorized PV-only path reports the same AC as its export."""
        result = self._infinite_public_run("python", with_battery=False, scale=scale)
        expected = np.array([500.0, 1000.0, 2000.0]) * 0.96 * scale
        assert np.array_equal(result[0]["PV_Production"].to_numpy(), expected)
        assert np.array_equal(result[0]["Sell_To_Grid"].to_numpy(), expected)
        assert result[1] == expected.sum()

    @pytest.mark.parametrize("scale", [0.5, 0.8])
    def test_infinite_public_scalar_and_numba_paths_match_scaled_ac(self, scale):
        """The scalar and Numba battery paths preserve the infinite-cap result."""
        pytest.importorskip("numba", reason="the compiled backend needs the breos[fast] extra")
        python_result = self._infinite_public_run("python", with_battery=True, scale=scale)
        numba_result = self._infinite_public_run("numba", with_battery=True, scale=scale)
        expected = np.array([500.0, 1000.0, 2000.0]) * 0.96 * scale
        assert np.array_equal(python_result[0]["PV_Production"].to_numpy(), expected)
        assert np.array_equal(python_result[0]["Sell_To_Grid"].to_numpy(), expected)
        assert python_result[1] == expected.sum()
        for column in python_result[0].columns:
            if column != "Datetime":
                assert np.array_equal(python_result[0][column].to_numpy(), numba_result[0][column].to_numpy()), column
        assert python_result[1] == numba_result[1]

    def test_optimizer_problem_carries_the_configured_scale(self):
        """SolarDesignProblem must not silently evaluate at scale 1.0."""
        from breos.optimization import SolarDesignProblem

        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        weather = pd.DataFrame({"temp_air": np.full(24, 20.0)}, index=idx)
        load = pd.DataFrame({"Load": np.full(24, 500.0)}, index=idx)
        config = {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "pv": {"module": "Suntech_STP550S_STC"},
            "simulation": {"resolution": "h", "years_projection": 1},
            "financials": {"project_lifespan": 1},
            "costs": {"dc_ac_ratio": 1.25},
            "ac_output_scale": 0.9210647508519848,
            "dc_output_scale": 0.75,
        }
        problem = SolarDesignProblem(weather, load, config, None)
        assert problem.ac_output_scale == pytest.approx(0.9210647508519848)
        assert problem.dc_output_scale == pytest.approx(0.75)

        del config["ac_output_scale"]
        del config["dc_output_scale"]
        assert SolarDesignProblem(weather, load, config, None).ac_output_scale == 1.0
        assert SolarDesignProblem(weather, load, config, None).dc_output_scale == 1.0

    @pytest.mark.parametrize("scale", [0.0, -1.0, 1.0857, 2.0, float("nan"), float("inf")])
    def test_optimizer_problem_rejects_an_out_of_range_ac_scale(self, scale):
        """The AC factor is rejected at the config boundary, not deep in dispatch."""
        from breos.optimization import SolarDesignProblem

        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        weather = pd.DataFrame({"temp_air": [20.0, 20.0]}, index=idx)
        load = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
        config = {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "pv": {"module": "Suntech_STP550S_STC"},
            "simulation": {"resolution": "h", "years_projection": 1},
            "financials": {"project_lifespan": 1},
            "costs": {"dc_ac_ratio": 1.25},
            "ac_output_scale": scale,
        }
        with pytest.raises(ValueError, match="ac_output_scale must be finite"):
            SolarDesignProblem(weather, load, config, None)

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_optimizer_problem_rejects_invalid_dc_scale(self, scale):
        from breos.optimization import SolarDesignProblem

        idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        weather = pd.DataFrame({"temp_air": [20.0, 20.0]}, index=idx)
        load = pd.DataFrame({"Load": [500.0, 500.0]}, index=idx)
        config = {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "pv": {"module": "Suntech_STP550S_STC"},
            "simulation": {"resolution": "h", "years_projection": 1},
            "financials": {"project_lifespan": 1},
            "costs": {"dc_ac_ratio": 1.25},
            "dc_output_scale": scale,
        }
        with pytest.raises(ValueError, match="dc_output_scale must be finite and greater than 0"):
            SolarDesignProblem(weather, load, config, None)

    def test_optimizer_problem_scales_dc_for_steady_and_projected_paths(self, monkeypatch):
        """NSGA candidate scoring sends corrected raw DC to both evaluators."""
        from breos.optimization import SolarDesignProblem

        idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        weather = pd.DataFrame({"temp_air": np.full(3, 20.0)}, index=idx)
        load = pd.DataFrame({"Load": np.full(3, 100.0)}, index=idx)
        raw = pd.Series([100.0, 200.0, 300.0], index=idx)
        seen = {}

        monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **_kwargs: raw.copy())
        monkeypatch.setattr(
            "breos.optimization._temperature_series_from_config",
            lambda *_args, **_kwargs: pd.Series(np.full(3, 20.0), index=idx),
        )

        def fake_simulate(*, pv_dc, houseload, **_kwargs):
            seen["steady"] = pv_dc.copy()
            frame = pd.DataFrame(
                {
                    "Houseload": houseload.iloc[:, 0].to_numpy(),
                    "PV_Production": pv_dc.to_numpy(),
                    "Battery_SOH": np.full(len(pv_dc), 100.0),
                },
                index=pv_dc.index,
            )
            summary = pd.DataFrame(
                {
                    "Import [kWh]": [0.0],
                    "Sell [kWh]": [float(pv_dc.sum() / 1000.0)],
                    "Total Load [kWh]": [float(houseload.iloc[:, 0].sum() / 1000.0)],
                }
            )
            return frame, float(pv_dc.sum()), summary, 0.0, 0, pd.DataFrame()

        monkeypatch.setattr("breos.optimization.simulate_energy_balance", fake_simulate)
        monkeypatch.setattr("breos.optimization.calculate_financials", lambda *_args, **_kwargs: (0.0, 0.0))

        def fake_projected(**kwargs):
            seen["projected"] = kwargs["base_dc_power"].copy()
            return {
                "Projected_Grid_Independence_%": 100.0,
                "Projected_NPV_Eur": 0.0,
                "Projected_ZEB_Ratio": 1.0,
            }

        monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", fake_projected)
        config = {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "pv": {"module": "Suntech_STP550S_STC"},
            "simulation": {"resolution": "h", "years_projection": 1},
            "financials": {"project_lifespan": 1},
            "costs": {"dc_ac_ratio": 1.25},
            "optimization": {"objective_basis": "projected"},
            "dc_output_scale": 0.5,
        }
        problem = SolarDesignProblem(weather, load, config, None)
        out = {}
        problem._evaluate(np.array([1.0, 1.0, 35.0, 180.0]), out)

        expected = raw * 0.5
        assert np.array_equal(seen["steady"].to_numpy(), expected.to_numpy())
        assert np.array_equal(seen["projected"].to_numpy(), expected.to_numpy())


class TestDcOutputScale:
    """The DC-side reading of the same yield correction."""

    @staticmethod
    def _config(**extra):
        cfg = {
            "location": {"latitude": 41.15, "longitude": -8.63},
            "pv": {"module": "Suntech_STP550S_STC"},
            "simulation": {"resolution": "h", "years_projection": 1},
            "financials": {"project_lifespan": 1},
            "costs": {"dc_ac_ratio": 1.25},
        }
        cfg.update(extra)
        return cfg

    def _run(self, monkeypatch, config):
        from breos.optimization import evaluate_projected_design

        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        weather = pd.DataFrame({"temp_air": np.full(24, 20.0)}, index=idx)
        load = pd.DataFrame({"Load": np.full(24, 500.0)}, index=idx)
        raw = pd.Series(np.linspace(0.0, 4000.0, 24), index=idx)
        seen: dict = {}

        monkeypatch.setattr("breos.optimization.calculate_pv_production_dc", lambda **_k: raw.copy())
        monkeypatch.setattr(
            "breos.optimization._temperature_series_from_config",
            lambda *_a, **_k: pd.Series(np.full(24, 20.0), index=idx),
        )

        def _capture(**kwargs):
            seen["dc"] = kwargs["base_dc_power"]
            return {
                "Projected_Grid_Independence_%": 0.0,
                "Projected_NPV_Eur": 0.0,
                "_yearly_summary_df": pd.DataFrame({"Year": [1]}),
                "_cost_projection_df": pd.DataFrame({"Year": [1]}),
            }

        monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", _capture)
        evaluate_projected_design(weather, load, config, n_modules=9, battery_kwh=5.0, tilt=35.0, azimuth=200.0)
        return raw, seen["dc"]

    def test_default_leaves_the_dc_series_untouched(self, monkeypatch):
        raw, used = self._run(monkeypatch, self._config())
        assert np.array_equal(used.to_numpy(), raw.to_numpy())

    def test_explicit_one_is_the_same_object_value(self, monkeypatch):
        raw, used = self._run(monkeypatch, self._config(dc_output_scale=1.0))
        assert np.array_equal(used.to_numpy(), raw.to_numpy())

    @pytest.mark.parametrize("scale", [0.5, 0.9210647508519848, 1.0857])
    def test_dc_series_is_scaled_before_dispatch(self, monkeypatch, scale):
        raw, used = self._run(monkeypatch, self._config(dc_output_scale=scale))
        assert np.allclose(used.to_numpy(), raw.to_numpy() * scale, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_dc_scale_must_be_positive_and_finite(self, monkeypatch, scale):
        with pytest.raises(ValueError, match="dc_output_scale must be finite and greater than 0"):
            self._run(monkeypatch, self._config(dc_output_scale=scale))
