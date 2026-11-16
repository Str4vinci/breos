"""P50/P90 yield from weather variability and a stated model uncertainty."""

import math
from statistics import NormalDist

import numpy as np
import pytest

from breos.exceedance import yield_exceedance
from breos.montecarlo import MonteCarloSettings, _sample_pv_model_factor


def test_exceedance_combines_variability_and_model_uncertainty():
    yields = {2019: 9800.0, 2020: 10200.0, 2021: 10000.0, 2022: 9600.0, 2023: 10400.0}
    result = yield_exceedance(yields, model_uncertainty_pct=3.0)
    mean = 10000.0
    variability = 100 * np.std(list(yields.values()), ddof=1) / mean
    assert result.p50_kwh == mean
    assert result.variability_pct == pytest.approx(variability)
    z90 = NormalDist().inv_cdf(0.9)
    one_year = math.sqrt(variability**2 + 9.0)
    ten_years = math.sqrt(variability**2 / 10 + 9.0)
    assert result.value_kwh(90) == pytest.approx(mean * (1 - z90 * one_year / 100))
    assert result.value_kwh(90, 10) == pytest.approx(mean * (1 - z90 * ten_years / 100))
    assert result.value_kwh(50) == pytest.approx(mean)
    assert result.value_kwh(99) < result.value_kwh(90) < result.value_kwh(75) < mean
    record = result.as_dict(horizons=(1, 10))
    assert set(record["horizons"]) == {"1", "10"}
    assert record["horizons"]["10"]["P90_kwh"] > record["horizons"]["1"]["P90_kwh"]
    # The model uncertainty does not average out.
    assert yield_exceedance(yields).sigma_pct(10_000) == pytest.approx(variability / 100)
    assert result.sigma_pct(10_000) == pytest.approx(3.0, rel=1e-3)


def test_exceedance_refuses_what_it_cannot_estimate():
    with pytest.raises(ValueError, match="at least two years"):
        yield_exceedance({2020: 1.0})
    with pytest.raises(ValueError, match="model_uncertainty_pct"):
        yield_exceedance({2020: 1.0, 2021: 1.1}, model_uncertainty_pct=-1)
    with pytest.raises(ValueError, match="level"):
        yield_exceedance({2020: 1.0, 2021: 1.1}).value_kwh(100)


def test_each_run_draws_its_model_factor_from_its_own_stream():
    settings = MonteCarloSettings(weather_file="unused.csv", seed=7, model_uncertainty_pct=4.0)
    factors = [_sample_pv_model_factor(settings, run) for run in range(2000)]
    assert factors[3] == _sample_pv_model_factor(settings, 3)
    assert np.std(factors) == pytest.approx(0.04, rel=0.1)
    assert _sample_pv_model_factor(MonteCarloSettings(weather_file="unused.csv", seed=7), 3) == 1.0
