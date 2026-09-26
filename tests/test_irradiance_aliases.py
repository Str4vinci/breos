"""One irradiance alias map for the resampler, the PV model and horizon shading (#211)."""

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos.solar import calculate_pv_production_dc
from breos.utils import IRRADIANCE_COLUMN_ALIASES, find_irradiance_column, irradiance_component


@pytest.mark.parametrize(
    ("column", "component"),
    [
        ("ghi", "ghi"),
        ("GHI", "ghi"),
        ("Shortwave_Radiation", "ghi"),
        ("direct_normal_irradiance", "dni"),
        ("temp_air", None),
    ],
)
def test_irradiance_component_is_case_insensitive(column, component):
    assert irradiance_component(column) == component


def test_find_irradiance_column_prefers_exact_then_earlier_aliases():
    assert find_irradiance_column(["GHI", "ghi"], "ghi") == "ghi"
    assert find_irradiance_column(["shortwave_radiation", "GHI"], "ghi") == "GHI"
    assert find_irradiance_column(["temp_air"], "dni") is None
    assert set(IRRADIANCE_COLUMN_ALIASES) == {"ghi", "dni", "dhi"}


def test_pv_model_reads_open_meteo_names(synthetic_weather, pv_params):
    location = Location(41.15, -8.61, tz="UTC")
    kwargs = dict(location=location, tilt=35, surface_azimuth=180, n_modules=2, pv_params=pv_params, freq="h")
    renamed = synthetic_weather.rename(
        columns={"ghi": "shortwave_radiation", "dni": "direct_normal_irradiance", "dhi": "diffuse_radiation"}
    )

    np.testing.assert_array_equal(
        calculate_pv_production_dc(weather_data=renamed, **kwargs).to_numpy(),
        calculate_pv_production_dc(weather_data=synthetic_weather, **kwargs).to_numpy(),
    )
