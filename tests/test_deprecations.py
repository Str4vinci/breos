"""Names deprecated in 0.8.0 still work, warn, and are removed in 0.9.0."""

import math

import pandas as pd
import pytest

from breos.optimization import OptimizationResult
from breos.optimization_config import resolve_optimization_config


def test_format_years_months_warns_and_still_formats():
    from breos.utils import format_years_months

    with pytest.warns(DeprecationWarning, match=r"format_years_months .*0\.9\.0"):
        assert format_years_months(4.25) == "4y 3m"


def test_update_battery_soh_cyclewise_warns_and_still_ages():
    from breos.battery import update_battery_soh_cyclewise

    soc = pd.Series([0.1, 0.9, 0.1], index=pd.date_range("2025-01-01", periods=3, freq="h"))
    with pytest.warns(DeprecationWarning, match=r"update_battery_soh_cyclewise .*0\.9\.0"):
        soh, _, fec = update_battery_soh_cyclewise(1.0, soc)
    assert soh < 1.0
    assert fec > 0.0


def test_lfp_capacity_factor_from_breos_battery_warns_and_is_the_dispatch_function():
    import breos.battery
    from breos._dispatch import lfp_capacity_factor

    with pytest.warns(DeprecationWarning, match=r"breos\.battery\.lfp_capacity_factor .*0\.9\.0"):
        assert breos.battery.lfp_capacity_factor is lfp_capacity_factor
    with pytest.raises(AttributeError, match="no_such_name"):
        breos.battery.no_such_name  # noqa: B018


def test_apply_indoor_temperature_model_from_breos_battery_warns_and_is_the_weather_function():
    import breos.battery
    from breos.weather import apply_indoor_temperature_model

    with pytest.warns(DeprecationWarning, match=r"breos\.battery\.apply_indoor_temperature_model .*0\.9\.0"):
        assert breos.battery.apply_indoor_temperature_model is apply_indoor_temperature_model


@pytest.mark.parametrize("name", ["optimal_value", "objective_value"])
def test_optimization_result_nan_fields_warn(name):
    result = OptimizationResult(iterations=1, details={})

    with pytest.warns(DeprecationWarning, match=rf"OptimizationResult\.{name} .*0\.9\.0"):
        assert math.isnan(getattr(result, name))


def test_objective_basis_warns_when_set_and_is_silent_when_omitted(recwarn):
    location = {"latitude": 41.15, "longitude": -8.61}

    with pytest.warns(DeprecationWarning, match=r"objective_basis .*0\.9\.0"):
        resolved = resolve_optimization_config({"location": location, "optimization": {"objective_basis": "projected"}})
    assert resolved["optimization"]["objective_basis"] == "projected"

    recwarn.clear()
    resolved = resolve_optimization_config({"location": location})
    assert resolved["optimization"]["objective_basis"] == "projected"
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
