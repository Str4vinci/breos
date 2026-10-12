"""Near shading from a per-array table of beam shaded fraction against sun position."""

import json

import numpy as np
import pandas as pd
import pvlib
import pytest
from pvlib.location import Location

from breos.app_config import resolve_app_config
from breos.pv.model_options import resolve_pv_model_options
from breos.pv.shading import ShadingTable, as_shading_table
from breos.pv_modules import get_module
from breos.solar import (
    _compute_irradiance_and_cell_temp_detail,
    _prepare_solarpos_and_weather,
    calculate_pv_production_breakdown,
)

PORTO = Location(41.15, -8.63, tz="Europe/Lisbon")
UNSHADED = {"azimuth_deg": [0], "elevation_deg": [0], "beam_shaded_fraction": [[0.0]]}
# A wall to the east: the morning beam is shaded below 30 degrees of elevation.
EAST_WALL = {
    "azimuth_deg": [0, 90, 180, 270],
    "elevation_deg": [0, 30, 40],
    "beam_shaded_fraction": [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    "sky_view_factor": 0.9,
}


def test_the_table_is_read_bilinearly_and_closes_the_circle():
    table = ShadingTable(
        azimuth_deg=tuple(range(0, 360, 90)),
        elevation_deg=(10.0, 50.0),
        beam_shaded_fraction=((0.0, 0.4, 0.8, 0.2), (0.0, 0.0, 0.4, 0.0)),
    )
    assert table.periodic
    values = table.beam_fraction([90, 135, 315, 360, 225, 135], [10, 10, 10, 10, 30, 0])
    np.testing.assert_allclose(values, [0.4, 0.6, 0.1, 0.0, 0.35, 0.6])
    # Above the top row and below the bottom one, the nearest row holds.
    np.testing.assert_allclose(table.beam_fraction([180, 180], [80, 2]), [0.4, 0.8])


def test_a_table_that_does_not_close_the_circle_holds_its_edges():
    table = ShadingTable(azimuth_deg=(90, 180, 270), elevation_deg=(10,), beam_shaded_fraction=((1.0, 0.0, 0.5),))
    assert not table.periodic
    np.testing.assert_allclose(table.beam_fraction([0, 45, 135, 300, 359], [20] * 5), [1.0, 1.0, 0.5, 0.5, 0.5])
    full = ShadingTable(azimuth_deg=(0, 180, 360), elevation_deg=(10,), beam_shaded_fraction=((1.0, 0.0, 1.0),))
    assert full.periodic
    np.testing.assert_allclose(full.beam_fraction([0, 90, 270, 359.9], [5] * 4), [1.0, 0.5, 0.5, 1.0 - 0.1 / 180])


@pytest.mark.parametrize(
    "change, message",
    [
        ({"azimuth_deg": [90, 60]}, "must increase"),
        ({"azimuth_deg": [-10]}, "between 0 and 360"),
        ({"elevation_deg": [0, 95]}, "between 0 and 90"),
        ({"beam_shaded_fraction": [[1.2]]}, "from 0 to 1"),
        ({"beam_shaded_fraction": [[0.0], [0.0]]}, "one row per elevation_deg"),
        ({"beam_shaded_fraction": [[0.0, 0.0]]}, "one value per azimuth_deg"),
        ({"sky_view_factor": 1.5}, "sky_view_factor"),
        ({"horizon": 3}, "unknown shading key"),
    ],
)
def test_a_bad_table_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        ShadingTable.from_mapping({**UNSHADED, **change})


def test_a_full_circle_needs_matching_end_columns():
    with pytest.raises(ValueError, match="first and last columns must match"):
        ShadingTable(azimuth_deg=(0, 180, 360), elevation_deg=(10,), beam_shaded_fraction=((1.0, 0.0, 0.5),))
    with pytest.raises(ValueError, match="needs beam_shaded_fraction"):
        ShadingTable.from_mapping({"azimuth_deg": [0], "elevation_deg": [0]})


def _detail(weather, shading=None, transposition_model="isotropic"):
    _, solarpos, aligned = _prepare_solarpos_and_weather(weather, PORTO, "h")
    options = resolve_pv_model_options(transposition_model=transposition_model)
    detail = _compute_irradiance_and_cell_temp_detail(
        aligned, solarpos, 30.0, 180.0, get_module("Generic_400W"), options, shading=as_shading_table(shading)
    )
    return detail, solarpos, aligned


def test_an_all_zero_table_changes_nothing(synthetic_weather):
    plain = calculate_pv_production_breakdown(synthetic_weather, PORTO, 30, 180, 10)
    shaded = calculate_pv_production_breakdown(synthetic_weather, PORTO, 30, 180, 10, shading=UNSHADED)
    np.testing.assert_allclose(shaded.dc_after_losses, plain.dc_after_losses, rtol=1e-12)
    np.testing.assert_allclose(shaded.poa_shaded_dc, plain.poa_global_dc, rtol=1e-12)
    np.testing.assert_allclose(shaded.poa_global_dc, plain.poa_global_dc, rtol=1e-12)
    assert plain.poa_shaded_dc is None


def test_the_beam_is_shaded_at_the_suns_position_and_the_sky_by_the_view_factor(synthetic_weather):
    plain, solarpos, aligned = _detail(synthetic_weather)
    shaded, _, _ = _detail(synthetic_weather, EAST_WALL)
    np.testing.assert_array_equal(shaded.poa_global_unshaded, plain.poa_global)
    dni, ghi, dhi = (aligned[c].to_numpy() for c in ("dni", "ghi", "dhi"))
    poa = pvlib.irradiance.get_total_irradiance(
        30, 180, solarpos.apparent_zenith, solarpos.azimuth, dni, ghi, dhi, model="isotropic"
    )
    f = as_shading_table(EAST_WALL).beam_fraction(solarpos.azimuth, 90 - solarpos.apparent_zenith)
    expected = (
        np.nan_to_num(poa["poa_direct"].to_numpy()) * (1 - f)
        + np.nan_to_num(poa["poa_sky_diffuse"].to_numpy()) * 0.9
        + np.nan_to_num(poa["poa_ground_diffuse"].to_numpy())
    )
    np.testing.assert_allclose(shaded.poa_global, expected, rtol=1e-12, atol=1e-9)
    azimuth, zenith = solarpos.azimuth.to_numpy(), solarpos.apparent_zenith.to_numpy()
    behind_the_wall = (azimuth > 80) & (azimuth < 100) & (zenith > 60) & (zenith < 90)
    assert behind_the_wall.any() and (f[behind_the_wall] > 0.85).all()
    # Less irradiance runs the cells cooler.
    assert (shaded.temp_cell <= plain.temp_cell + 1e-9).all()


@pytest.mark.parametrize("model", ["haydavies", "perez", "perez-driesse"])
def test_the_circumsolar_diffuse_is_shaded_with_the_beam(synthetic_weather, model):
    half_beam = {**UNSHADED, "beam_shaded_fraction": [[0.5]]}
    plain, solarpos, aligned = _detail(synthetic_weather, UNSHADED, model)
    shaded, _, _ = _detail(synthetic_weather, half_beam, model)
    dni, dhi = aligned["dni"].to_numpy(), aligned["dhi"].to_numpy()
    extra = pvlib.irradiance.get_extra_radiation(solarpos.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solarpos.apparent_zenith)
    common = (30, 180, dhi, dni, extra, solarpos.apparent_zenith, solarpos.azimuth)
    if model == "haydavies":
        parts = pvlib.irradiance.haydavies(*common, return_components=True)
    elif model == "perez":
        parts = pvlib.irradiance.perez(*common, airmass, return_components=True)
    else:
        parts = pvlib.irradiance.perez_driesse(*common, airmass, return_components=True)
    poa = pvlib.irradiance.get_total_irradiance(
        30, 180, solarpos.apparent_zenith, solarpos.azimuth, dni, aligned["ghi"].to_numpy(), dhi,
        dni_extra=extra, airmass=airmass, model=model,
    )  # fmt: skip
    # Signed components: the circumsolar part is not capped at the sky total.
    circumsolar = np.clip(np.nan_to_num(np.asarray(parts["poa_circumsolar"], dtype=float)), 0, None)
    direct = np.nan_to_num(poa["poa_direct"].to_numpy())
    np.testing.assert_allclose(plain.poa_global - shaded.poa_global, 0.5 * (direct + circumsolar), rtol=1e-9, atol=1e-9)
    assert circumsolar.sum() > 0


def test_a_fully_shaded_array_keeps_its_unshaded_reference(synthetic_weather):
    dark = {**UNSHADED, "beam_shaded_fraction": [[1.0]], "sky_view_factor": 0.0}
    plain = calculate_pv_production_breakdown(synthetic_weather, PORTO, 30, 180, 10, albedo=0.0)
    shaded = calculate_pv_production_breakdown(synthetic_weather, PORTO, 30, 180, 10, albedo=0.0, shading=dark)
    assert shaded.dc_after_losses.abs().max() == pytest.approx(0.0, abs=1e-6)
    assert shaded.poa_shaded_dc.abs().max() == pytest.approx(0.0, abs=1e-6)
    # The stages before near shading still show what the shading removed;
    # only the cell temperature, which shading lowers, sets them apart.
    assert shaded.poa_global_dc.sum() == pytest.approx(plain.poa_global_dc.sum(), rel=1e-2)
    assert shaded.horizontal_reference_dc.sum() == pytest.approx(plain.horizontal_reference_dc.sum(), rel=1e-2)


def test_partial_shading_leaves_the_unshaded_stages_close(synthetic_weather):
    plain = calculate_pv_production_breakdown(synthetic_weather, PORTO, 30, 180, 10)
    shaded = calculate_pv_production_breakdown(synthetic_weather, PORTO, 30, 180, 10, shading=EAST_WALL)
    # Only the cell temperature differs: the shaded cells run cooler.
    assert shaded.poa_global_dc.sum() == pytest.approx(plain.poa_global_dc.sum(), rel=5e-3)
    assert shaded.poa_shaded_dc.sum() < shaded.poa_global_dc.sum()


def test_the_rear_side_is_not_shaded(synthetic_weather):
    _, solarpos, aligned = _prepare_solarpos_and_weather(synthetic_weather, PORTO, "h")
    options = resolve_pv_model_options(
        bifacial_model="infinite_sheds", bifaciality=0.7, gcr=0.4, pvrow_height=1.0, pvrow_pitch=5.0
    )
    module = get_module("Generic_600W_Bifacial")
    plain = _compute_irradiance_and_cell_temp_detail(aligned, solarpos, 30.0, 180.0, module, options)
    shaded = _compute_irradiance_and_cell_temp_detail(
        aligned, solarpos, 30.0, 180.0, module, options, shading=as_shading_table(EAST_WALL)
    )
    np.testing.assert_array_equal(shaded.rear_effective_irradiance, plain.rear_effective_irradiance)
    assert shaded.front_effective_irradiance.sum() < plain.front_effective_irradiance.sum()


def test_a_tracker_and_a_15_minute_run_take_the_table(synthetic_weather, synthetic_weather_15min):
    from breos.solar import calculate_pv_production_tracking_breakdown

    plain = calculate_pv_production_tracking_breakdown(synthetic_weather, PORTO, 10)
    shaded = calculate_pv_production_tracking_breakdown(synthetic_weather, PORTO, 10, shading=EAST_WALL)
    assert shaded.dc_after_losses.sum() < plain.dc_after_losses.sum()
    hourly = calculate_pv_production_breakdown(synthetic_weather, PORTO, 30, 180, 10, shading=EAST_WALL)
    quarter = calculate_pv_production_breakdown(
        synthetic_weather_15min, PORTO, 30, 180, 10, freq="15min", shading=EAST_WALL
    )
    loss = {name: 1 - b.poa_shaded_dc.sum() / b.poa_global_dc.sum() for name, b in (("h", hourly), ("15min", quarter))}
    assert 0 < loss["15min"] and loss["15min"] == pytest.approx(loss["h"], abs=0.02)


@pytest.fixture
def shaded_config():
    return {
        "location": "porto",
        "annual_consumption_kwh": 4000,
        "pv_arrays": [
            {"modules": 8, "tilt": 30, "azimuth": 180, "shading": EAST_WALL},
            {"modules": 4, "tilt": 30, "azimuth": 270, "shading": UNSHADED},
        ],
    }


def test_every_array_or_none(shaded_config):
    del shaded_config["pv_arrays"][1]["shading"]
    with pytest.raises(ValueError, match=r"'pv_arrays\[1\]' has no 'shading' table"):
        resolve_app_config(shaded_config)
    shaded_config["pv_arrays"][0]["shading"] = {**EAST_WALL, "sky_view_factor": -1}
    with pytest.raises(ValueError, match=r"'pv_arrays\[0\].shading': 'sky_view_factor'"):
        resolve_app_config(shaded_config)


def test_the_flat_shading_loss_steps_aside_unless_stated(shaded_config):
    assert resolve_app_config(shaded_config).cfg["pv_loss_overrides"]["shading"] == 0.0
    stated = {**shaded_config, "pv_loss_overrides": {"shading": 1.5}}
    assert resolve_app_config(stated).cfg["pv_loss_overrides"]["shading"] == 1.5
    plain = {
        **shaded_config,
        "pv_arrays": [{k: v for k, v in a.items() if k != "shading"} for a in shaded_config["pv_arrays"]],
    }
    assert (resolve_app_config(plain).cfg["pv_loss_overrides"] or {}).get("shading") is None


def test_app_reports_the_near_shading_stage(shaded_config, _patch_weather):
    from breos import App

    app = App(shaded_config)
    app.simulate()
    result = app.result()
    waterfall = result["pv_loss_waterfall"]
    keys = [stage["key"] for stage in waterfall["stages"]]
    assert keys.index("transposition") + 1 == keys.index("near_shading") == keys.index("iam") - 1
    stage = waterfall["stages"][keys.index("near_shading")]
    summary = waterfall["near_shading"]
    assert stage["delta_kwh"] < 0 and summary["loss_kwh"] == pytest.approx(-stage["delta_kwh"], abs=0.02)
    assert summary["flat_shading_pct"] == 0.0
    assert summary["arrays"][0] == {
        "array": 0,
        "form": "table",
        "azimuth_points": 4,
        "elevation_points": 3,
        "azimuth_periodic": True,
        "sky_view_factor": 0.9,
        "sky_view_factor_stated": True,
        "circumsolar_shaded_with_beam": False,
    }
    assert summary["arrays"][1]["sky_view_factor_stated"] is False
    json.dumps(result)

    reference = App(
        {
            **shaded_config,
            "pv_arrays": [{k: v for k, v in a.items() if k != "shading"} for a in shaded_config["pv_arrays"]],
            "pv_loss_overrides": {"shading": 0.0},
        }
    )
    reference.simulate()
    assert reference.result()["pv_loss_waterfall"]["near_shading"] is None
    assert result["usable_ac_system_production_kwh"] < reference.result()["usable_ac_system_production_kwh"]


def test_string_level_energy_takes_the_shaded_irradiance(shaded_config, _patch_weather):
    from breos import App
    from breos.pv_modules import MODULES, PVModuleParams

    # Trina Solar TSM-450NEG9R.28, datasheet TSM_EN_2024_C, PDF page 2.
    MODULES["Trina_TSM_450NEG9R28"] = PVModuleParams(
        Mpp=450, Vmp=44.6, Imp=10.09, Voc=52.9, Isc=10.74,
        T_Pmax_pct=-0.29, T_Voc_pct=-0.24, T_Isc_pct=0.04, N_Cells=144, Max_System_Voltage=1500,
    )  # fmt: skip
    try:
        config = {
            **shaded_config,
            "pv_arrays": [{**shaded_config["pv_arrays"][0], "module": "Trina_TSM_450NEG9R28"}],
            "inverters": [
                {
                    "nominal_power_w": 3600,
                    "max_dc_voltage_v": 600,
                    "min_mppt_voltage_v": 80,
                    "max_mppt_voltage_v": 550,
                    "startup_voltage_v": 80,
                    "mppts": [{"max_input_current_a": 15, "max_short_circuit_current_a": 20, "max_strings": 1}],
                }
            ],
            "pv_strings": [{"inverter": 0, "mppt": 0, "array": 0, "modules": 8}],
            "electrical_environment": {"min_design_temp_c": -5, "max_ambient_temp_c": 40},
            "electrical_fidelity": "string",
        }
        results = {}
        for name, shading in (("shaded", EAST_WALL), ("unshaded", UNSHADED)):
            config["pv_arrays"][0]["shading"] = shading
            app = App(config)
            app.simulate()
            results[name] = app.result()
        shaded_dc = results["shaded"]["pv_loss_waterfall"]["energy_balance"]["pv_dc"]["generation_kwh"]
        unshaded_dc = results["unshaded"]["pv_loss_waterfall"]["energy_balance"]["pv_dc"]["generation_kwh"]
        assert shaded_dc < unshaded_dc
    finally:
        del MODULES["Trina_TSM_450NEG9R28"]


def test_the_shading_table_is_echoed_as_plain_lists(shaded_config):
    resolved = resolve_app_config(shaded_config)
    echoed = resolved.pv_arrays[0]["shading"]
    assert echoed["beam_shaded_fraction"][0] == [0.0, 1.0, 0.0, 0.0]
    assert "sky_view_factor" not in resolved.pv_arrays[1]["shading"]
    assert isinstance(pd.Series(echoed["azimuth_deg"]).sum(), float)


def test_the_provenance_echo_round_trips(shaded_config, _patch_weather):
    from breos import App

    shaded_config["pv_arrays"][0]["shading"] = {
        **EAST_WALL,
        "beam_shaded_fraction": [[np.float32(v) for v in row] for row in EAST_WALL["beam_shaded_fraction"]],
    }
    app = App(shaded_config)
    app.simulate()
    echoed = app.result()["provenance"]["resolved_config"]
    assert echoed["pv_arrays"][0]["shading"]["beam_shaded_fraction"][0] == [0.0, 1.0, 0.0, 0.0]
    again = {key: echoed[key] for key in ("location", "annual_consumption_kwh", "pv_arrays")}
    assert (
        resolve_app_config(again).pv_arrays[0]["shading"] == resolve_app_config(shaded_config).pv_arrays[0]["shading"]
    )


def test_the_circumsolar_flag_reads_any_spelling(shaded_config, _patch_weather):
    from breos import App

    shaded_config["pv_arrays"][0]["transposition_model"] = " Perez "
    app = App(shaded_config)
    app.simulate()
    arrays = app.result()["pv_loss_waterfall"]["near_shading"]["arrays"]
    assert arrays[0]["circumsolar_shaded_with_beam"] is True
    assert arrays[1]["circumsolar_shaded_with_beam"] is False


def test_monte_carlo_years_take_the_tables(shaded_config, synthetic_weather):
    from breos.montecarlo import _build_pv_years

    plain_config = {
        **shaded_config,
        "pv_arrays": [{k: v for k, v in a.items() if k != "shading"} for a in shaded_config["pv_arrays"]],
        "pv_loss_overrides": {"shading": 0.0},
    }
    years = {}
    for name, config in (("shaded", shaded_config), ("plain", plain_config)):
        resolved = resolve_app_config(config)
        dc, _, _ = _build_pv_years(resolved.cfg, resolved, {2023: synthetic_weather})
        years[name] = dc[2023].sum()
    assert years["shaded"] < years["plain"]
