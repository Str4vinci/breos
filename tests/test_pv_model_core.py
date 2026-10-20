"""Focused tests for the internal PV-model option and kernel modules."""

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pvlib
import pytest

from breos import solar
from breos.pv.iam import calculate_front_effective_irradiance
from breos.pv.model_options import (
    PER_ARRAY_MODEL_OPTION_KEYS,
    PV_MODEL_CONFIG_KEYS,
    PV_MODEL_DEFAULTS,
    PVModelOptions,
    resolve_pv_model_options,
)
from breos.pv.temperature import DEFAULT_MODULE_EFFICIENCY, calculate_cell_temperature

# Every public entry point that accepts the shared PV model-option block.
MODEL_OPTION_ENTRY_POINTS = (
    solar.calculate_pv_production_breakdown,
    solar.calculate_pv_production_dc,
    solar.calculate_pv_production_dc_tracking,
    solar.calculate_pv_production_ac,
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


# Shared cell-temperature kernel inputs: night, part load and full sun.
POA_GLOBAL = np.array([0.0, 400.0, 900.0])
TEMP_AIR = np.array([15.0, 20.0, 28.0])
WIND_SPEED = np.array([1.0, 2.0, 4.0])
_PVSYST = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["pvsyst"]
_SAPM = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]


@pytest.mark.parametrize(
    ("model", "kwargs", "pvlib_function", "parameters"),
    [
        pytest.param("faiman", {}, pvlib.temperature.faiman, {}, id="faiman"),
        # Metadata-less modules take BREOS's representative c-Si value, not
        # pvlib's legacy 0.1 default.
        pytest.param(
            "pvsyst-semi-integrated",
            {},
            pvlib.temperature.pvsyst_cell,
            {**_PVSYST["semi_integrated"], "module_efficiency": DEFAULT_MODULE_EFFICIENCY},
            id="pvsyst-semi-integrated",
        ),
        pytest.param(
            "pvsyst-freestanding",
            {},
            pvlib.temperature.pvsyst_cell,
            {**_PVSYST["freestanding"], "module_efficiency": DEFAULT_MODULE_EFFICIENCY},
            id="pvsyst-freestanding",
        ),
        # A sourced module efficiency replaces the default.
        pytest.param(
            "pvsyst-semi-integrated",
            {"module_efficiency": 0.213},
            pvlib.temperature.pvsyst_cell,
            {**_PVSYST["semi_integrated"], "module_efficiency": 0.213},
            id="pvsyst-semi-integrated-sourced-efficiency",
        ),
        *(
            pytest.param(f"sapm-{key.replace('_', '-')}", {}, pvlib.temperature.sapm_cell, _SAPM[key], id=f"sapm-{key}")
            for key in (
                "open_rack_glass_glass",
                "close_mount_glass_glass",
                "open_rack_glass_polymer",
                "insulated_back_glass_polymer",
            )
        ),
    ],
)
def test_temperature_kernel_preserves_pvlib_dispatch(model, kwargs, pvlib_function, parameters):
    expected = pvlib_function(POA_GLOBAL, TEMP_AIR, WIND_SPEED, **parameters)

    actual = calculate_cell_temperature(POA_GLOBAL, TEMP_AIR, WIND_SPEED, model, **kwargs)

    np.testing.assert_allclose(actual, expected)


def test_breos_default_efficiency_is_representative_of_modern_silicon():
    """Guards the default against drifting back toward pvlib's 0.1 placeholder."""
    assert 0.15 < DEFAULT_MODULE_EFFICIENCY < 0.25


def test_noct_sam_requires_complete_sourced_module_metadata():
    poa_global, temp_air, wind_speed = POA_GLOBAL, TEMP_AIR, WIND_SPEED

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

    ``PV_MODEL_CONFIG_KEYS`` drives the wrappers' dict forwarding, so a new
    option added to the tuple without being added to a signature would raise a
    ``TypeError`` only when that path happened to run. Assert it up front, and
    keep the options keyword-addressable — ``breos.App``, ``cli.py`` and
    ``validation/`` all pass them by name.
    """
    parameters = inspect.signature(function).parameters
    missing = [key for key in PV_MODEL_CONFIG_KEYS if key not in parameters]
    assert not missing, f"{function.__name__} is missing model options: {missing}"

    positional_only = [key for key in PV_MODEL_CONFIG_KEYS if parameters[key].kind is inspect.Parameter.POSITIONAL_ONLY]
    assert not positional_only, f"{function.__name__} made model options positional-only: {positional_only}"

    # The App and the optimizer fill unset options from PV_MODEL_DEFAULTS.
    defaults = {key: parameters[key].default for key in PV_MODEL_CONFIG_KEYS}
    assert defaults == dict(PV_MODEL_DEFAULTS), f"{function.__name__} defaults differ from PV_MODEL_DEFAULTS"


def test_model_option_keys_partition_into_per_array_and_function_level():
    """The multi-array override asymmetry is intentional — pin it exactly.

    ``iam_model``/``diffuse_iam``/``solar_position`` are function-level for
    every array while the sky and ground geometry and the thermal model are
    per-array overridable. The function-level set is pinned exactly.
    """
    per_array = set(PER_ARRAY_MODEL_OPTION_KEYS)

    assert per_array < set(PV_MODEL_CONFIG_KEYS)
    assert set(PV_MODEL_CONFIG_KEYS) - per_array == {"solar_position", "iam_model", "diffuse_iam"}


def test_no_public_entry_point_accepts_var_keywords():
    """``**kwargs`` on these would silently swallow a misspelled option."""
    for function in MODEL_OPTION_ENTRY_POINTS:
        kinds = [p.kind for p in inspect.signature(function).parameters.values()]
        assert inspect.Parameter.VAR_KEYWORD not in kinds, f"{function.__name__} accepts **kwargs"


def test_a_repeated_sun_position_is_computed_once_and_returned_as_a_copy(monkeypatch):
    from pvlib.location import Location

    from breos.pv.model_options import solar_position_at_labels

    calls = []
    real = pvlib.solarposition.get_solarposition

    def counting(times, *args, **kwargs):
        calls.append(times)
        return real(times, *args, **kwargs)

    monkeypatch.setattr(pvlib.solarposition, "get_solarposition", counting)
    times = pd.date_range("2031-06-01", periods=48, freq="h", tz="UTC")
    weather = pd.DataFrame(index=times)

    first, _ = solar_position_at_labels(Location(41.15, -8.61), times, weather, "h", "interval-start")
    first.loc[:, "azimuth"] = -1.0
    second, _ = solar_position_at_labels(Location(41.15, -8.61), times, weather, "h", "interval-start")
    assert len(calls) == 1
    assert (second["azimuth"] >= 0.0).all()

    # Another offset, site or calendar is computed afresh.
    mid, _ = solar_position_at_labels(Location(41.15, -8.61), times, weather, "h", "mid-interval")
    solar_position_at_labels(Location(41.15, -8.62), times, weather, "h", "interval-start")
    solar_position_at_labels(Location(41.15, -8.61), times.tz_convert("Europe/Lisbon"), weather, "h", "interval-start")
    assert len(calls) == 4
    assert mid.index.equals(times)
    assert not np.array_equal(mid["azimuth"], second["azimuth"])
