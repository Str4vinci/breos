"""Regression cases for the energy-flow diagram's accounting and input boundary."""

import copy

import pytest

from breos import plotting
from tests.test_plotting_figures import _full_waterfall, _vehicle_waterfall, plt


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _assert_continuous(rows):
    energy = rows[0][2]
    for kind, label, value in rows[1:]:
        if kind == "total":
            assert energy == pytest.approx(value, abs=1e-9), label
        else:
            energy += value


@pytest.mark.parametrize("gap", [0.2, 0.02])
def test_static_remainder_never_leaves_a_gap(gap):
    w = _full_waterfall()
    w["pvwatts"]["components_kwh"]["soiling"] -= gap
    _assert_continuous(plotting._flow_rows(w, w["stages"]))


def test_battery_remainder_names_stored_energy():
    w = _full_waterfall()
    rows = plotting._flow_rows(w, w["stages"])
    assert ("step", "Battery losses and stored energy", -20.0) in rows


def test_idle_consumption_without_battery():
    w = _full_waterfall(battery=False)
    ac = w["energy_balance"]["ac_delivery"]
    ac["export_kwh"] -= 4
    ac["usable_system_production_kwh"] -= 4
    w["inverter"]["night_consumption"] = {"energy_kwh": 4}
    w["inverter"]["inverters"] = [{}, {}]
    w["energy_balance"]["pv_dc"]["to_inverter_kwh"] = 999.6
    rows = plotting._flow_rows(w, w["stages"])
    assert ("step", "PV to inverter idle consumption", pytest.approx(-4.0)) in rows
    assert not any("battery" in label.lower() for _, label, _ in rows)
    _assert_continuous(rows)


def test_vehicle_carryover_is_an_incoming_flow():
    w = _vehicle_waterfall()
    ac = w["energy_balance"]["ac_delivery"]
    ac["pv_origin_vehicle_to_load_kwh"] = 150
    ac["usable_system_production_kwh"] += 142
    rows = plotting._flow_rows(w, w["stages"])
    assert ("step", "Vehicle energy carried into the period", 30.0) in rows
    assert rows[-1][2] == pytest.approx(sum(v for _, _, v in plotting._flow_end_uses(ac)))
    assert all(v >= 0 for _, _, v in plotting._flow_end_uses(ac))
    _assert_continuous(rows)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), None])
def test_invalid_stage_energy_is_explained(value):
    w = _full_waterfall()
    w["stages"][1]["energy_kwh"] = value
    with pytest.raises(ValueError, match="energy_kwh"):
        plotting.plot_pv_loss_waterfall(w)


@pytest.mark.parametrize("key", ["year", "pv_dc_generation_kwh", "usable_ac_system_production_kwh"])
def test_incomplete_year_is_explained(key):
    row = dict(year=1, pv_dc_generation_kwh=1000, usable_ac_system_production_kwh=950)
    other = copy.copy(row)
    del other[key]
    with pytest.raises(ValueError, match=key):
        plotting.plot_pv_loss_waterfall(_full_waterfall(), yearly=[row, other])


def test_close_replacements_have_one_shared_label():
    years = [dict(year=y, pv_dc_generation_kwh=1000 - y, usable_ac_system_production_kwh=950 - y) for y in range(1, 21)]
    fig = plotting.plot_pv_loss_waterfall(_full_waterfall(), yearly=years, replacement_years=[10, 10.2, 11])
    fig.canvas.draw()
    labels = [t for t in fig.axes[1].texts if "Battery replacement" in t.get_text()]
    assert len(labels) == 1


@pytest.mark.parametrize("battery", [False, True])
@pytest.mark.parametrize("pv_idle", [0.0, 4.0])
def test_idle_row_excludes_grid_draw_and_battery_remainder(battery, pv_idle):
    w = _full_waterfall(battery=battery)
    ac = w["energy_balance"]["ac_delivery"]
    ac["export_kwh"] -= pv_idle
    ac["usable_system_production_kwh"] -= pv_idle
    w["inverter"].update(night_consumption={"energy_kwh": 10}, inverters=[{}, {}])
    # With a DC battery, 120 kWh took the storage route before the inverter.
    w["energy_balance"]["pv_dc"]["to_inverter_kwh"] = 1000 - 0.4 - (120 if battery else 0)
    rows = plotting._flow_rows(w, w["stages"])
    idle = [v for _, label, v in rows if label == "PV to inverter idle consumption"]
    assert sum(idle) == pytest.approx(-pv_idle)
    if battery:
        assert ("step", "Battery losses and stored energy", pytest.approx(-20)) in rows
    _assert_continuous(rows)


def test_actual_multi_inverter_result_labels_only_pv_supplied_idle_draw(trina, _patch_weather):
    from tests.test_multi_inverter import SMALL, TWO_INVERTERS, timeseries
    from tests.test_string_energy import FRONIUS_GEN24_6

    frame, result = timeseries({**TWO_INVERTERS, "inverters": [{**SMALL, "startup_voltage_v": 500.0}, FRONIUS_GEN24_6]})
    waterfall = result["pv_loss_waterfall"]
    rows = plotting._flow_rows(waterfall, waterfall["stages"])
    expected = (frame["PV_DC_To_Inverter"] - frame["PV_Direct_Inverter_Loss"] - frame["PV_AC_To_Load"]).clip(lower=0)
    expected = expected.clip(upper=frame["Inverter_Night_Consumption"]).sum() / 1000
    idle = next(-v for _, label, v in rows if label == "PV to inverter idle consumption")
    assert idle == pytest.approx(expected, abs=0.02)  # Result fields are rounded to 0.01 kWh.
    assert not any("battery" in label.lower() for _, label, _ in rows)
    _assert_continuous(rows)


@pytest.mark.parametrize("field", ["pv_dc_generation_kwh", "usable_ac_system_production_kwh"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_yearly_energy_is_explained(field, value):
    row = dict(year=1, pv_dc_generation_kwh=1000, usable_ac_system_production_kwh=950)
    row[field] = value
    with pytest.raises(ValueError, match=field):
        plotting.plot_pv_loss_waterfall(_full_waterfall(), yearly=[row])


def test_old_four_argument_call(tmp_path):
    target = tmp_path / "flow.png"
    fig = plotting.plot_pv_loss_waterfall(_full_waterfall(), str(target), "Custom title", (10, 12))
    assert target.is_file()
    assert fig.axes[0].texts[0].get_text() == "Custom title"
