"""Tests for plotting helpers."""

import pytest


def test_plot_pv_loss_waterfall_writes_png(tmp_path):
    pytest.importorskip("matplotlib")

    from breos.plotting import plot_pv_loss_waterfall

    waterfall = {
        "basis": "year_1",
        "unit": "kWh",
        "stages": [
            {"key": "horizontal_reference_dc", "label": "Horizontal reference", "energy_kwh": 1200.0},
            {
                "key": "transposition",
                "label": "Plane-of-array transposition",
                "energy_kwh": 1300.0,
                "delta_kwh": 100.0,
                "delta_pct_of_previous": 8.33,
            },
            {
                "key": "iam",
                "label": "Incidence-angle modifier",
                "energy_kwh": 1280.0,
                "delta_kwh": -20.0,
                "delta_pct_of_previous": -1.54,
            },
            {
                "key": "temperature",
                "label": "Cell temperature",
                "energy_kwh": 1230.0,
                "delta_kwh": -50.0,
                "delta_pct_of_previous": -3.91,
            },
            {
                "key": "inverter_conversion",
                "label": "Inverter conversion",
                "energy_kwh": 1100.0,
                "delta_kwh": -130.0,
                "delta_pct_of_previous": -10.57,
            },
        ],
        "pvwatts": {
            "components_pct": {"soiling": 2.0, "shading": 3.0},
            "components_kwh": {"soiling": 24.0, "shading": 35.0},
        },
        "inverter": {"clipping_kwh": 0.0, "conversion_loss_kwh": 44.0},
        "dispatch": {
            "battery_round_trip_loss_kwh": 12.0,
            "battery_standby_loss_kwh": 5.0,
            "curtailment_kwh": 0.0,
        },
        "energy_balance": {
            "pv_dc": {"curtailed_kwh": 10.0},
            "ac_delivery": {
                "direct_pv_to_load_kwh": 700.0,
                "pv_origin_battery_to_load_kwh": 100.0,
                "export_kwh": 300.0,
            },
        },
    }
    output = tmp_path / "pv_loss_waterfall.png"

    fig = plot_pv_loss_waterfall(waterfall, output_path=str(output))

    assert fig is not None
    assert output.exists()
    assert output.stat().st_size > 0
