"""Focused tests for the internal PV-model option and kernel modules."""

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pvlib
import pytest

from breos import solar
from breos.pv.iam import calculate_front_effective_irradiance
from breos.pv.model_options import PVModelOptions, resolve_pv_model_options
from breos.pv.temperature import DEFAULT_MODULE_EFFICIENCY, calculate_cell_temperature

# Every public entry point that accepts the shared PV model-option block.
MODEL_OPTION_ENTRY_POINTS = (
    solar.calculate_pv_production_breakdown,
    solar.calculate_pv_production_dc,
    solar.calculate_pv_production_tracking_breakdown,
    solar.calculate_pv_production_dc_tracking,
    solar.calculate_pv_production_ac,
    solar.calculate_multi_array_production_breakdown,
    solar.calculate_multi_array_production,
)


def test_model_options_are_normalised_validated_and_immutable():
    options = resolve_pv_model_options(
        transposition_model=" PEREZ ",
        iam_model=" Martin_Ruiz ",
        diffuse_iam="MARION",
        temperature_model="PVsyst-Insulated",
        surface_type="urban",
    )

    assert options == PVModelOptions(
        transposition_model="perez",
        albedo=None,
        surface_type="urban",
        model_perez="allsitescomposite1990",
        iam_model="martin_ruiz",
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


@pytest.mark.parametrize("iam_model", ("physical", "martin_ruiz"))
def test_front_irradiance_kernel_dispatches_selected_beam_and_marion_models(iam_model):
    poa = pd.DataFrame(
        {
            "poa_direct": [800.0, 200.0],
            "poa_diffuse": [100.0, 80.0],
            "poa_sky_diffuse": [75.0, 60.0],
            "poa_ground_diffuse": [25.0, 20.0],
        }
    )
    aoi = np.array([10.0, 70.0])
    marion = pvlib.iam.marion_diffuse(iam_model, 30.0)
    expected = (
        poa["poa_direct"].to_numpy() * getattr(pvlib.iam, iam_model)(aoi)
        + poa["poa_sky_diffuse"].to_numpy() * marion["sky"]
        + poa["poa_ground_diffuse"].to_numpy() * marion["ground"]
    )

    actual = calculate_front_effective_irradiance(poa, aoi, 30.0, "marion", iam_model)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("model", "pvlib_function", "parameters"),
    [
        ("faiman", pvlib.temperature.faiman, {}),
        (
            "pvsyst-semi-integrated",
            pvlib.temperature.pvsyst_cell,
            {
                **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"]["semi_integrated"],
                # Metadata-less modules take BREOS's representative c-Si value,
                # not pvlib's legacy 0.1 default.
                "module_efficiency": DEFAULT_MODULE_EFFICIENCY,
            },
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


def test_pvsyst_kernel_uses_sourced_module_efficiency():
    poa_global = np.array([0.0, 400.0, 900.0])
    temp_air = np.array([15.0, 20.0, 28.0])
    wind_speed = np.array([1.0, 2.0, 4.0])
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"]["semi_integrated"]

    actual = calculate_cell_temperature(
        poa_global,
        temp_air,
        wind_speed,
        "pvsyst-semi-integrated",
        module_efficiency=0.213,
    )
    expected = pvlib.temperature.pvsyst_cell(
        poa_global,
        temp_air,
        wind_speed,
        module_efficiency=0.213,
        **params,
    )

    np.testing.assert_allclose(actual, expected)


def test_pvsyst_kernel_falls_back_to_breos_default_efficiency():
    """A metadata-less module must not silently inherit pvlib's legacy 0.1."""
    poa_global = np.array([0.0, 400.0, 900.0])
    temp_air = np.array([15.0, 20.0, 28.0])
    wind_speed = np.array([1.0, 2.0, 4.0])
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"]["freestanding"]

    actual = calculate_cell_temperature(poa_global, temp_air, wind_speed, "pvsyst-freestanding")

    expected = pvlib.temperature.pvsyst_cell(
        poa_global,
        temp_air,
        wind_speed,
        module_efficiency=DEFAULT_MODULE_EFFICIENCY,
        **params,
    )
    np.testing.assert_allclose(actual, expected)

    pvlib_legacy_default = pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind_speed, **params)
    assert actual[-1] < pvlib_legacy_default[-1]


def test_breos_default_efficiency_is_representative_of_modern_silicon():
    """Guards the default against drifting back toward pvlib's 0.1 placeholder."""
    assert 0.15 < DEFAULT_MODULE_EFFICIENCY < 0.25


def test_pvsyst_rejects_nonphysical_supplied_module_efficiency():
    with pytest.raises(ValueError, match="Module_Efficiency"):
        calculate_cell_temperature(
            np.array([900.0]),
            np.array([28.0]),
            np.array([2.0]),
            "pvsyst-semi-integrated",
            module_efficiency=1.1,
        )


@pytest.mark.parametrize(
    ("model", "parameter_key"),
    [
        ("sapm-open-rack-glass-glass", "open_rack_glass_glass"),
        ("sapm-close-mount-glass-glass", "close_mount_glass_glass"),
        ("sapm-open-rack-glass-polymer", "open_rack_glass_polymer"),
        ("sapm-insulated-back-glass-polymer", "insulated_back_glass_polymer"),
    ],
)
def test_sapm_temperature_presets_dispatch_to_matching_pvlib_parameters(model, parameter_key):
    poa_global = np.array([0.0, 400.0, 900.0])
    temp_air = np.array([15.0, 20.0, 28.0])
    wind_speed = np.array([1.0, 2.0, 4.0])
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][parameter_key]

    actual = calculate_cell_temperature(poa_global, temp_air, wind_speed, model)
    expected = pvlib.temperature.sapm_cell(poa_global, temp_air, wind_speed, **params)

    np.testing.assert_allclose(actual, expected)


def test_noct_sam_requires_complete_sourced_module_metadata():
    poa_global = np.array([0.0, 400.0, 900.0])
    temp_air = np.array([15.0, 20.0, 28.0])
    wind_speed = np.array([1.0, 2.0, 4.0])

    with pytest.raises(ValueError, match="Module_Efficiency"):
        calculate_cell_temperature(poa_global, temp_air, wind_speed, "noct-sam")
    with pytest.raises(ValueError, match="NOCT metadata"):
        calculate_cell_temperature(poa_global, temp_air, wind_speed, "noct-sam", module_efficiency=0.213)

    actual = calculate_cell_temperature(
        poa_global,
        temp_air,
        wind_speed,
        "noct-sam",
        module_efficiency=0.213,
        noct=45.0,
    )
    expected = pvlib.temperature.noct_sam(poa_global, temp_air, wind_speed, noct=45.0, module_efficiency=0.213)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("function", MODEL_OPTION_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_every_entry_point_declares_the_whole_model_option_block(function):
    """Each public entry point must accept every shared model option by keyword.

    ``solar._MODEL_OPTION_KEYS`` drives the wrappers' dict forwarding, so a new
    option added to the tuple without being added to a signature would raise a
    ``TypeError`` only when that path happened to run. Assert it up front, and
    keep the options keyword-addressable — ``breos.App``, ``cli.py`` and
    ``validation/`` all pass them by name.
    """
    parameters = inspect.signature(function).parameters
    missing = [key for key in solar._MODEL_OPTION_KEYS if key not in parameters]
    assert not missing, f"{function.__name__} is missing model options: {missing}"

    positional_only = [
        key for key in solar._MODEL_OPTION_KEYS if parameters[key].kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    assert not positional_only, f"{function.__name__} made model options positional-only: {positional_only}"


def test_model_option_keys_partition_into_per_array_and_function_level():
    """The multi-array override asymmetry is intentional — pin it exactly.

    ``iam_model``/``diffuse_iam``/``temperature_model``/``solar_position`` are function-level
    for every array while the sky and ground geometry is per-array
    overridable. A new option must land in exactly one of the two tuples, so
    the partition (not just the union) is what gets asserted.
    """
    per_array = set(solar._PER_ARRAY_MODEL_OPTION_KEYS)
    function_level = set(solar._FUNCTION_LEVEL_MODEL_OPTION_KEYS)

    assert per_array | function_level == set(solar._MODEL_OPTION_KEYS)
    assert per_array & function_level == set()
    assert function_level == {"solar_position", "iam_model", "diffuse_iam", "temperature_model"}
    # gcr is model geometry here but tracker geometry on the tracking path.
    assert set(solar._TRACKING_MODEL_OPTION_KEYS) == set(solar._MODEL_OPTION_KEYS) - {"gcr"}


def test_no_public_entry_point_accepts_var_keywords():
    """``**kwargs`` on these would silently swallow a misspelled option."""
    for function in MODEL_OPTION_ENTRY_POINTS:
        kinds = [p.kind for p in inspect.signature(function).parameters.values()]
        assert inspect.Parameter.VAR_KEYWORD not in kinds, f"{function.__name__} accepts **kwargs"
