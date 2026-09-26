"""PV module age is counted at the start of each simulated year (#175).

Year 1 (age 0) has no degradation and year n is degraded by n - 1 full years
of compound ``degradation_rate``. ``breos.solar`` used to add half a year, so
a direct call at age N disagreed with App, Monte Carlo, the optimizer and the
economics projection, which all use ``(1 - r) ** (n - 1)``.
"""

import pytest

import breos.app_inputs as app_inputs
from breos import App
from breos.solar import (
    calculate_multi_array_production,
    calculate_pv_production_breakdown,
    calculate_pv_production_dc,
    calculate_pv_production_dc_tracking,
)

RATE = 0.02
START = 2023


@pytest.fixture
def fixed_kwargs(synthetic_weather, porto_location, pv_params):
    return dict(
        weather_data=synthetic_weather.iloc[: 24 * 14],
        location=porto_location,
        tilt=35,
        surface_azimuth=180,
        n_modules=2,
        pv_params=pv_params,
        freq="h",
        degradation_rate=RATE,
    )


def test_installation_year_has_no_age_loss(fixed_kwargs):
    new = calculate_pv_production_breakdown(**fixed_kwargs, current_year=START, start_year=START)
    no_age = calculate_pv_production_breakdown(**fixed_kwargs)

    assert new.age_degradation_pct == 0.0
    assert new.age_degradation_loss.abs().max() == 0.0
    assert (new.dc_after_losses == new.dc_after_static_losses).all()
    assert (new.dc_after_losses == no_age.dc_after_losses).all()


@pytest.mark.parametrize("age", [1, 4, 19])
def test_age_n_is_n_full_years_of_compound_degradation(fixed_kwargs, age):
    breakdown = calculate_pv_production_breakdown(**fixed_kwargs, current_year=START + age, start_year=START)
    factor = (1 - RATE) ** age

    assert breakdown.age_degradation_pct == pytest.approx(100 * (1 - factor), rel=1e-12)
    ratio = breakdown.dc_after_losses.sum() / breakdown.dc_after_static_losses.sum()
    assert ratio == pytest.approx(factor, rel=1e-12)


def test_every_public_entry_point_uses_the_same_age(fixed_kwargs):
    age = 3
    factor = (1 - RATE) ** age
    aged = dict(current_year=START + age, start_year=START)
    new = dict(current_year=START, start_year=START)

    def ratio(fn, kwargs):
        return fn(**kwargs, **aged).sum() / fn(**kwargs, **new).sum()

    tracking_kwargs = {k: v for k, v in fixed_kwargs.items() if k not in ("tilt", "surface_azimuth")}
    multi_kwargs = {k: v for k, v in tracking_kwargs.items() if k not in ("n_modules", "pv_params")}
    multi_kwargs["arrays"] = [{"modules": 2, "module": "Suntech_STP550S_STC", "tilt": 35, "azimuth": 180}]

    assert ratio(calculate_pv_production_dc, fixed_kwargs) == pytest.approx(factor, rel=1e-12)
    assert ratio(calculate_pv_production_dc_tracking, tracking_kwargs) == pytest.approx(factor, rel=1e-12)
    assert ratio(calculate_multi_array_production, multi_kwargs) == pytest.approx(factor, rel=1e-12)


def test_a_current_year_before_installation_is_rejected(fixed_kwargs):
    with pytest.raises(ValueError, match="must not be earlier than start_year"):
        calculate_pv_production_dc(**fixed_kwargs, current_year=START - 1, start_year=START)


def test_app_year_k_pv_matches_solar_at_age_k_minus_1(_patch_weather, monkeypatch):
    # Capture the exact undegraded call App makes, then repeat it through
    # breos.solar with an explicit age for each projected year.
    calls = []
    real = app_inputs.calculate_pv_production_breakdown

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(app_inputs, "calculate_pv_production_breakdown", spy)
    app = App(
        {
            "location": "porto",
            "n_modules": 6,
            "annual_consumption_kwh": 3000,
            "projection_years": 4,
            "pv_degradation_rate": RATE,
        }
    )
    app.simulate()
    yearly = app.result()["yearly"]
    assert len(calls) == 1

    for k, row in enumerate(yearly, start=1):
        dc = calculate_pv_production_dc(**calls[0], degradation_rate=RATE, current_year=START + k - 1, start_year=START)
        solar_kwh = dc.sum() / 1000.0
        assert row["year"] == k
        assert row["pv_dc_generation_kwh"] == pytest.approx(solar_kwh, abs=0.006)
