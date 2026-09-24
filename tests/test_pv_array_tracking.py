"""pv_arrays inherit and validate tracker settings like the top level (#167)."""

import pytest

from breos.app import App

BASE = {"location": "porto", "annual_consumption_kwh": 3000, "projection_years": 1}
SINGLE_AXIS = {"tracking": "single_axis", "max_angle": 20.0}


def _dc_kwh(config: dict) -> float:
    app = App({**BASE, **config})
    app.simulate()
    return app.result()["pv_dc_generation_kwh"]


def test_arrays_inherit_top_level_tracking(_patch_weather):
    top_level = _dc_kwh({**SINGLE_AXIS, "n_modules": 10})
    one_array = _dc_kwh({**SINGLE_AXIS, "pv_arrays": [{"modules": 10}]})
    fixed = _dc_kwh({"n_modules": 10})

    # On develop the array ran fixed-tilt, matching `fixed`.
    assert one_array == pytest.approx(top_level, rel=1e-12)
    assert one_array != pytest.approx(fixed, rel=1e-3)


def test_arrays_inherit_top_level_tracker_geometry(_patch_weather):
    narrow = _dc_kwh({**SINGLE_AXIS, "pv_arrays": [{"modules": 10, "tracking": "single_axis"}]})
    default_angle = _dc_kwh({"pv_arrays": [{"modules": 10, "tracking": "single_axis"}]})

    # On develop the array ignored the top-level 20 degree limit and used 60.
    assert narrow == pytest.approx(_dc_kwh({**SINGLE_AXIS, "n_modules": 10}), rel=1e-12)
    assert narrow != pytest.approx(default_angle, rel=1e-3)


def test_resolved_array_reports_its_tracking_mode(_patch_weather):
    app = App({**BASE, **SINGLE_AXIS, "pv_arrays": [{"modules": 6}, {"modules": 4, "tracking": "fixed"}]})
    app.simulate()
    tracker, fixed = app.result()["pv_arrays"]

    assert tracker["tracking"] == "single_axis"
    assert tracker["max_angle"] == 20.0
    assert tracker["backtrack"] is True
    assert fixed["tracking"] == "fixed"
    assert "max_angle" not in fixed


@pytest.mark.parametrize(
    "settings,error,match",
    [
        ({"max_angle": "60"}, TypeError, "'max_angle' must be a finite number"),
        ({"max_angle": -500}, ValueError, "'max_angle' must be between 0 and 90"),
        ({"axis_tilt": 400}, ValueError, "'axis_tilt' must be between 0 and 90"),
        ({"axis_azimuth": 400}, ValueError, "'axis_azimuth' must be between 0 and 360"),
        ({"cross_axis_tilt": 95}, ValueError, "'cross_axis_tilt' must be between -90 and 90"),
        ({"dual_axis_max_tilt": 120}, ValueError, "'dual_axis_max_tilt' must be between 0 and 90"),
        ({"backtrack": "no"}, TypeError, "'backtrack' must be true or false"),
    ],
)
def test_top_level_tracker_settings_are_validated(settings, error, match):
    with pytest.raises(error, match=match):
        App({**BASE, "n_modules": 10, "tracking": "single_axis", **settings})


@pytest.mark.parametrize(
    "array,error,match",
    [
        ({"tracking": "singleaxis"}, ValueError, r"'pv_arrays\[0\]\.tracking' must be 'fixed'"),
        ({"backtrack": "no"}, TypeError, r"'pv_arrays\[0\]\.backtrack' must be true or false"),
        ({"max_angle": -500}, ValueError, r"'pv_arrays\[0\]\.max_angle' must be between"),
        ({"tlt": 10}, ValueError, r"Unknown key\(s\) in pv_arrays\[0\]: tlt"),
    ],
)
def test_array_entries_are_validated_before_the_weather_fetch(array, error, match):
    with pytest.raises(error, match=match):
        App({**BASE, "pv_arrays": [{"modules": 5, **array}]})
