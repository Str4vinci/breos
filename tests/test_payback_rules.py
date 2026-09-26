"""One payback rule for the library, the plots and the tools (#218)."""

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from breos import plotting  # noqa: E402
from breos.economics import find_payback_year, find_payback_year_exact  # noqa: E402
from breos.utils import format_years_months  # noqa: E402


def _projection(savings, first_year=1):
    years = list(range(first_year, first_year + len(savings)))
    no_sys = [1000.0 * (i + 1) for i in range(len(savings))]
    return pd.DataFrame(
        {
            "Year": years,
            "Savings_Cumulative_NPV": savings,
            "Cost_No_Sys_Cumulative_NPV": no_sys,
            "Cost_System_Cumulative_NPV": [n - s for n, s in zip(no_sys, savings)],
        }
    )


@pytest.mark.parametrize(
    ("savings", "exact", "integer"),
    [
        ([-300.0, -100.0, 200.0], 2 + 1 / 3, 3),
        ([50.0, 100.0], 1.0, 1),
        ([-300.0, -200.0], None, None),
        # Zero savings is not payback, under either rule.
        ([-100.0, 0.0, -10.0], None, None),
        ([0.0, -10.0], None, None),
        ([-100.0, 0.0, 0.0, 50.0], 3.0, 4),
    ],
)
def test_exact_and_integer_payback_agree_on_the_crossing(savings, exact, integer):
    projection = _projection(savings)

    assert find_payback_year_exact(projection) == (pytest.approx(exact) if exact is not None else None)
    assert find_payback_year(projection) == integer
    assert find_payback_year_exact(projection.drop(columns="Savings_Cumulative_NPV")) is None


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

    plotting.plot_breakeven_comparison([early, late], ["early", "late"], ["C0", "C1"], str(tmp_path))

    assert lines == [pytest.approx(1.0), pytest.approx(2 + 1 / 3)]


def test_breakeven_plot_uses_the_shared_interpolation(tmp_path, monkeypatch, capsys):
    plotting.plot_breakeven(_projection([-300.0, -100.0, 200.0]), str(tmp_path), scenario_name="x")

    assert "Break-even point: 2 years 4 months" in capsys.readouterr().out


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
