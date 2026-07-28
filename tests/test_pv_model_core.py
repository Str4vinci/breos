"""Focused tests for the internal PV-model option and kernel modules."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pvlib
import pytest

from breos.pv.iam import calculate_front_effective_irradiance
from breos.pv.model_options import PVModelOptions, resolve_pv_model_options
from breos.pv.temperature import calculate_cell_temperature


def test_model_options_are_normalised_validated_and_immutable():
    options = resolve_pv_model_options(
        transposition_model=" PEREZ ",
        diffuse_iam="MARION",
        temperature_model="PVsyst-Insulated",
        surface_type="urban",
    )

    assert options == PVModelOptions(
        transposition_model="perez",
        albedo=None,
        surface_type="urban",
        model_perez="allsitescomposite1990",
        diffuse_iam="marion",
        temperature_model="pvsyst-insulated",
        bifacial_model="none",
        bifaciality=None,
        gcr=0.35,
        pvrow_height=None,
        pvrow_pitch=None,
    )
    with pytest.raises(FrozenInstanceError):
        options.temperature_model = "faiman"


def test_front_irradiance_kernel_preserves_beam_only_ashrae_path():
    poa = pd.DataFrame(
        {
            "poa_direct": [800.0, 200.0, np.nan],
            "poa_diffuse": [100.0, 80.0, 20.0],
            "poa_sky_diffuse": [75.0, 60.0, 15.0],
            "poa_ground_diffuse": [25.0, 20.0, 5.0],
        }
    )
    aoi = np.array([10.0, 70.0, 95.0])
    expected = (
        np.nan_to_num(poa["poa_direct"].to_numpy(), nan=0.0) * np.nan_to_num(pvlib.iam.ashrae(aoi), nan=0.0)
        + poa["poa_diffuse"].to_numpy()
    )

    actual = calculate_front_effective_irradiance(poa, aoi, 30.0, "none")

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("model", "pvlib_function", "parameters"),
    [
        ("faiman", pvlib.temperature.faiman, {}),
        (
            "pvsyst-semi-integrated",
            pvlib.temperature.pvsyst_cell,
            pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"]["semi_integrated"],
        ),
    ],
)
def test_temperature_kernel_preserves_pvlib_dispatch(model, pvlib_function, parameters):
    poa_global = np.array([0.0, 400.0, 900.0])
    temp_air = np.array([15.0, 20.0, 28.0])
    wind_speed = np.array([1.0, 2.0, 4.0])
    expected = pvlib_function(poa_global, temp_air, wind_speed, **parameters)

    actual = calculate_cell_temperature(poa_global, temp_air, wind_speed, model)

    np.testing.assert_allclose(actual, expected)
