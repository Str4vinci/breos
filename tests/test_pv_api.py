"""PV production API gaps: empty multi-array systems and AC loss overrides (#220)."""

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos.solar import (
    calculate_multi_array_production_breakdown,
    calculate_pv_production_ac,
    calculate_pv_production_dc,
    dc_to_ac,
)

_LOCATION = Location(41.15, -8.61, tz="UTC")


@pytest.mark.parametrize("freq", ["h", "15min"])
def test_empty_multi_array_system_is_zero_on_the_same_grid(synthetic_weather, synthetic_weather_15min, freq):
    weather = synthetic_weather if freq == "h" else synthetic_weather_15min
    empty = calculate_multi_array_production_breakdown(weather, _LOCATION, [{"modules": 0}], freq=freq)
    one = calculate_multi_array_production_breakdown(weather, _LOCATION, [{"modules": 2}], freq=freq)

    assert empty.dc_after_losses.index.equals(one.dc_after_losses.index)
    assert (empty.dc_after_losses == 0.0).all()


def _grid_or_error(weather, arrays):
    try:
        return calculate_multi_array_production_breakdown(
            weather, _LOCATION, arrays, freq="15min"
        ).dc_after_losses.index
    except ValueError as exc:
        return type(exc)


def test_empty_multi_array_system_treats_a_resolution_mismatch_like_a_non_empty_one(synthetic_weather):
    # The empty result used to keep the weather's own index, so hourly weather
    # at freq="15min" gave hourly zeros next to quarter-hour production. The
    # empty path now prepares the weather the same way, so it returns the same
    # grid, or raises the same error once mismatched weather is refused.
    empty = _grid_or_error(synthetic_weather, [{"modules": 0}])
    one = _grid_or_error(synthetic_weather, [{"modules": 2}])

    if isinstance(one, pd.DatetimeIndex):
        assert empty.equals(one)
    else:
        assert empty is one


def test_ac_production_keeps_its_positional_parameters(synthetic_weather, pv_params):
    # loss_overrides is appended, so a positional transposition model still
    # binds to transposition_model.
    args = (synthetic_weather, _LOCATION, 35, 180, 4, pv_params, "h", 0.0, None, None, 1.25, 0.96, False)
    positional = calculate_pv_production_ac(*args, "isotropic")
    keyword = calculate_pv_production_ac(*args, transposition_model="isotropic")

    pd.testing.assert_series_equal(positional, keyword)


def test_zero_module_array_contributes_nothing(synthetic_weather):
    arrays = [{"modules": 2, "tilt": 30, "azimuth": 180}]
    with_empty = calculate_multi_array_production_breakdown(synthetic_weather, _LOCATION, [*arrays, {"modules": 0}])
    alone = calculate_multi_array_production_breakdown(synthetic_weather, _LOCATION, arrays)

    pd.testing.assert_series_equal(with_empty.dc_after_losses, alone.dc_after_losses)


def test_negative_module_count_raises(synthetic_weather):
    with pytest.raises(ValueError, match="Array 2: modules must be zero or positive, got -1"):
        calculate_multi_array_production_breakdown(synthetic_weather, _LOCATION, [{"modules": 2}, {"modules": -1}])


def test_ac_production_forwards_loss_overrides(synthetic_weather, pv_params):
    # The AC wrapper was the one entry point without loss_overrides.
    kwargs = dict(weather_data=synthetic_weather, location=_LOCATION, tilt=35, surface_azimuth=180, n_modules=4)
    overrides = {"soiling": 0.0, "shading": 0.0}

    ac = calculate_pv_production_ac(**kwargs, pv_params=pv_params, loss_overrides=overrides)
    dc = calculate_pv_production_dc(**kwargs, pv_params=pv_params, loss_overrides=overrides)
    default_ac = calculate_pv_production_ac(**kwargs, pv_params=pv_params)

    np.testing.assert_array_equal(ac.to_numpy(), dc_to_ac(dc, 4 * pv_params.Mpp, 1.25, 0.96).to_numpy())
    assert ac.sum() > default_ac.sum()
