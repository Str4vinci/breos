"""One payback rule for the library, the plots and the tools (#218, #175).

Payback is the sustained discounted payback within the simulated period: the
earliest time cumulative discounted savings reach zero or above and stay
nonnegative to the horizon, from a year-0 point at minus the investment.
"""

import math

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from breos import plotting  # noqa: E402
from breos.economics import cost_analysis_projection, find_payback_year, find_payback_year_exact  # noqa: E402
from breos.utils import format_years_months  # noqa: E402


def _projection(savings, first_year=1, investment=None):
    years = list(range(first_year, first_year + len(savings)))
    no_sys = [1000.0 * (i + 1) for i in range(len(savings))]
    projection = pd.DataFrame(
        {
            "Year": years,
            "Savings_Cumulative_NPV": savings,
            "Cost_No_Sys_Cumulative_NPV": no_sys,
            "Cost_System_Cumulative_NPV": [n - s for n, s in zip(no_sys, savings)],
        }
    )
    if investment is not None:
        projection.attrs["total_investment"] = investment
    return projection


def _assert_payback(projection, exact, integer):
    assert find_payback_year_exact(projection) == (pytest.approx(exact) if exact is not None else None)
    assert find_payback_year(projection) == integer


@pytest.mark.parametrize(
    ("savings", "exact", "integer"),
    [
        ([-300.0, -100.0, 200.0], 2 + 1 / 3, 3),
        # Without a recorded investment there is no year-0 point, so a first
        # row that is already nonnegative pays back at its own year.
        ([50.0, 100.0], 1.0, 1),
        ([-300.0, -200.0], None, None),
        # Savings that fall back below zero have not paid back.
        ([-100.0, 0.0, -10.0], None, None),
        ([0.0, -10.0], None, None),
        # Zero savings count as paid back when they hold to the horizon.
        ([-100.0, 0.0, 0.0, 50.0], 2.0, 2),
    ],
)
def test_exact_and_integer_payback_agree_on_the_crossing(savings, exact, integer):
    projection = _projection(savings)

    _assert_payback(projection, exact, integer)
    assert find_payback_year_exact(projection.drop(columns="Savings_Cumulative_NPV")) is None
    assert find_payback_year(projection.drop(columns="Savings_Cumulative_NPV")) is None


def test_payback_within_the_first_year_interpolates_from_the_investment():
    # With no year-0 anchor a true 0.21-year payback reported 1.0 (#175).
    _assert_payback(_projection([79.0, 200.0], investment=21.0), 0.21, 1)


def test_replacement_dip_pays_back_at_the_later_recovery():
    # Positive in year 3, negative again after a replacement in year 5.
    projection = _projection([-600.0, -200.0, 100.0, 300.0, -50.0, 150.0, 400.0], investment=1000.0)

    _assert_payback(projection, 5.25, 6)


def test_a_dip_at_the_horizon_is_not_payback():
    _assert_payback(_projection([-500.0, 100.0, 200.0, -10.0], investment=1000.0), None, None)


def test_a_zero_plateau_at_the_horizon_is_payback():
    _assert_payback(_projection([-500.0, -100.0, 0.0, 0.0], investment=1000.0), 3.0, 3)


def test_no_payback():
    _assert_payback(_projection([-800.0, -500.0, -300.0], investment=1000.0), None, None)


def test_a_zero_investment_that_never_loses_pays_back_at_year_zero():
    _assert_payback(_projection([0.0, 10.0], investment=0.0), 0.0, 0)


def test_explicit_investment_overrides_the_attrs():
    projection = _projection([79.0, 200.0], investment=1000.0)

    assert find_payback_year_exact(projection, initial_investment=21.0) == pytest.approx(0.21)
    assert find_payback_year(projection, initial_investment=21.0) == 1


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("investment", [None, 100.0])
def test_non_finite_savings_are_rejected(bad, investment):
    # NaN used to propagate into the interpolated payback, and +inf passed
    # the >= 0 test and reported a false crossing.
    projection = _projection([-100.0, bad, 10.0], investment=investment)

    for find in (find_payback_year_exact, find_payback_year):
        with pytest.raises(ValueError, match="NaN or infinite"):
            find(projection)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_investment_is_rejected(bad):
    projection = _projection([-50.0, 10.0])

    for find in (find_payback_year_exact, find_payback_year):
        with pytest.raises(ValueError, match="NaN or infinite"):
            find(projection, initial_investment=bad)


def test_a_year_zero_row_is_not_anchored_twice():
    projection = _projection([-21.0, 79.0, 200.0], first_year=0, investment=21.0)

    _assert_payback(projection, 0.21, 1)


def _cost_projection(investment, electricity_cost=0.30):
    costs = {
        "electricity_cost": electricity_cost,
        "electricity_sold_cost": 0.05,
        "daily_power_cost": 0.0,
        "total_initial_cost": investment,
        "annual_operation_cost": 0.0,
    }
    yearly = pd.DataFrame(
        {
            "Year": [1, 2, 3],
            "Load_kWh": 1000.0,
            "PV_Production_kWh": 800.0,
            "Import_kWh": 400.0,
            "Export_kWh": 200.0,
            "PV_Degradation_Factor": 1.0,
            "Replacement_Cost": 0.0,
        }
    )
    return cost_analysis_projection(
        None, costs, num_years=3, inflation_rate=0.0, discount_rate=0.0, yearly_summary_df=yearly
    )


def test_cost_projection_anchors_payback_at_its_investment():
    # Each year saves 600 kWh * 0.30 + 200 kWh * 0.05 = 190 against a
    # 40 investment, so payback is 40 / 190 of a year.
    projection = _cost_projection(investment=40.0)

    assert projection.attrs["payback_year"] == 1
    assert find_payback_year_exact(projection) == pytest.approx(40.0 / 190.0)


def test_payback_of_a_projection_read_back_without_attrs():
    # A projection saved to CSV loses its attrs; the investment is recovered
    # from the first row of the system cost.
    projection = _cost_projection(investment=40.0)
    projection.attrs.clear()

    assert find_payback_year_exact(projection) == pytest.approx(40.0 / 190.0)
    assert find_payback_year(projection) == 1


def test_exact_payback_scales_the_crossing_by_the_year_spacing():
    projection = pd.DataFrame({"Year": [1, 3], "Savings_Cumulative_NPV": [-100.0, 100.0]})

    assert find_payback_year_exact(projection) == pytest.approx(2.0)
    assert find_payback_year(projection) == 3


def _captured_vlines(monkeypatch):
    lines = []
    real_axvline = plotting.plt.Axes.axvline

    def axvline(self, x=0, *args, **kwargs):
        lines.append(float(x))
        return real_axvline(self, x, *args, **kwargs)

    monkeypatch.setattr(plotting.plt.Axes, "axvline", axvline)
    return lines


def test_breakeven_comparison_marks_a_crossing_before_the_second_year(tmp_path, monkeypatch):
    # The comparison plot looked only for a crossing between two rows, so a
    # design that paid back in its first row had no break-even line.
    lines = _captured_vlines(monkeypatch)
    early, late = _projection([50.0, 100.0]), _projection([-300.0, -100.0, 200.0])
    first_year = _projection([79.0, 200.0], investment=21.0)

    plotting.plot_breakeven_comparison(
        [early, late, first_year], ["early", "late", "first year"], ["C0", "C1", "C2"], str(tmp_path)
    )

    assert lines == [pytest.approx(1.0), pytest.approx(2 + 1 / 3), pytest.approx(0.21)]


def test_breakeven_plot_uses_the_shared_interpolation(tmp_path, monkeypatch, capsys):
    plotting.plot_breakeven(_projection([-300.0, -100.0, 200.0]), str(tmp_path), scenario_name="x")

    assert "Break-even point: 2 years 4 months" in capsys.readouterr().out


def test_breakeven_plot_anchors_at_the_investment(tmp_path, capsys):
    plotting.plot_breakeven(_cost_projection(investment=40.0), str(tmp_path), scenario_name="x")

    # 40 / 190 of a year is 2.5 months.
    assert "Break-even point: 0 years 2 months" in capsys.readouterr().out


def test_montecarlo_payback_plots_use_the_exact_year(tmp_path, monkeypatch):
    # The distribution's 0.1-year bins and mean used the integer year (#218).
    received = {}
    monkeypatch.setattr(
        plotting, "plot_breakeven_distribution", lambda values, *args, **kwargs: received.setdefault("dist", values)
    )
    monkeypatch.setattr(
        plotting, "plot_breakeven_cdf", lambda values, *args, **kwargs: received.setdefault("cdf", values)
    )
    monkeypatch.setattr(plotting, "plot_breakeven_summary_bar", lambda *args, **kwargs: None)
    runs = pd.DataFrame(
        {"run": [1, 2], "npv_savings_eur": [1.0, -1.0], "payback_year": [5, None], "payback_year_exact": [4.2, None]}
    )

    plotting._plot_montecarlo_payback_summary(runs, str(tmp_path))

    assert received["dist"] == [4.2]
    assert received["cdf"] == [4.2]


@pytest.mark.parametrize(("years", "text"), [(None, "N/A"), (4.0, "4y"), (4.25, "4y 3m")])
def test_format_years_months(years, text):
    assert format_years_months(years) == text
