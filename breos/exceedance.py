"""P50/P90 exceedance values of annual energy yield.

A P90 yield is exceeded with 90 % probability: it is the 10th percentile of
the yield's distribution. Bankable estimates combine two spreads, as PVsyst's
P50/P90 evaluation does. Year-to-year weather variability comes from the
annual yields of historical weather years. Model uncertainty covers the
irradiance data, the transposition and PV models and the loss assumptions;
the caller states it, because it depends on the data and the study. Both are
taken as normal and independent. Over an average of ``N`` years the
variability shrinks by ``sqrt(N)`` and the model uncertainty does not, since
a biased model is biased every year:

    sigma_N = sqrt(sigma_variability**2 / N + sigma_model**2)
    P_x = P50 * (1 - z_x * sigma_N)

with ``z_x`` the standard normal quantile of ``x`` %, 1.2816 for P90.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

#: The exceedance levels reported, in percent.
EXCEEDANCE_LEVELS = (50, 75, 90, 99)


@dataclass(frozen=True)
class YieldExceedance:
    """Exceedance values of annual yield from historical years and a stated model uncertainty.

    ``p50_kwh`` is the mean annual yield of ``years``, and
    ``variability_pct`` their sample standard deviation over that mean, in
    percent. ``model_uncertainty_pct`` is the caller's one-sigma model
    uncertainty in percent, or None when none was stated; the values then
    cover weather variability alone.
    """

    p50_kwh: float
    variability_pct: float
    model_uncertainty_pct: float | None
    years: tuple[int, ...]

    def sigma_pct(self, horizon_years: int) -> float:
        """The combined one-sigma uncertainty of the mean yield over ``horizon_years``, in percent."""
        if horizon_years < 1:
            raise ValueError("horizon_years must be at least 1")
        model = self.model_uncertainty_pct or 0.0
        return math.sqrt(self.variability_pct**2 / horizon_years + model**2)

    def value_kwh(self, level: float, horizon_years: int = 1) -> float:
        """The mean annual yield over ``horizon_years`` exceeded with ``level`` % probability, in kWh."""
        if not 0 < level < 100:
            raise ValueError("level must be between 0 and 100")
        z = NormalDist().inv_cdf(level / 100.0)
        return self.p50_kwh * (1.0 - z * self.sigma_pct(horizon_years) / 100.0)

    def as_dict(self, horizons: Sequence[int] = (1,)) -> dict[str, Any]:
        """A JSON-ready record: the inputs and, per horizon, the sigma and each level's yield."""
        return {
            "basis": "annual AC yield, first year, undegraded",
            "p50_kwh": self.p50_kwh,
            "variability_pct": self.variability_pct,
            "model_uncertainty_pct": self.model_uncertainty_pct,
            "years": list(self.years),
            "horizons": {
                str(horizon): {
                    "sigma_pct": self.sigma_pct(horizon),
                    **{f"P{level}_kwh": self.value_kwh(level, horizon) for level in EXCEEDANCE_LEVELS},
                }
                for horizon in sorted(set(horizons))
            },
        }


def yield_exceedance(
    annual_yields_kwh: Mapping[int, float], *, model_uncertainty_pct: float | None = None
) -> YieldExceedance:
    """Exceedance values from each historical year's annual yield, keyed by year.

    Needs at least two years with a positive mean.
    """
    if len(annual_yields_kwh) < 2:
        raise ValueError("yield_exceedance needs the annual yields of at least two years")
    values = [float(v) for v in annual_yields_kwh.values()]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("annual yields must be finite")
    mean = sum(values) / len(values)
    if mean <= 0:
        raise ValueError("the mean annual yield must be above 0")
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    if model_uncertainty_pct is not None and (not math.isfinite(model_uncertainty_pct) or model_uncertainty_pct < 0):
        raise ValueError("model_uncertainty_pct must be a finite number of at least 0")
    return YieldExceedance(
        p50_kwh=mean,
        variability_pct=100.0 * std / mean,
        model_uncertainty_pct=model_uncertainty_pct,
        years=tuple(sorted(int(year) for year in annual_yields_kwh)),
    )
