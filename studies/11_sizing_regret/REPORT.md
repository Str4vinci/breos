# Study 11 — Sizing on a TMY vs actual years

**Question.** A BREOS user picks a size on one PVGIS TMY. If the actual weather is one of the
19 SARAH3 years 2005–2023 instead, how much NPV does that choice lose, and does it still meet the
user's target?

**Answer.** For the NPV-best size, the TMY is safe: the loss is at most 14 € in any site-year.
For a size chosen to reach a self-sufficiency target, the TMY pick misses the target in 30–50 % of
actual years, but only by about 1 point (at most 4.5). The real risk is elsewhere. In 5 % of
site-years the TMY pick loses over 1,000 € (up to 5,300 €), because in those years the battery needs
one more replacement than on the TMY. BREOS's `replacement_min_remaining_years = 2` removes most of
that tail (1 % of site-years).

## Setup

Seven sites (Porto, Lisbon, Madrid, Athens, Erlangen, Berlin, Stockholm), the PVGIS TMY and 19
actual SARAH3 years each (data from long-run study 3). Household: the shipped
`projected-optimization.toml` (Portuguese costs at every site, so the sites differ by weather only),
4,000 kWh/yr H0 load, 20-year projection, hourly. Design grid: 2–20 modules × 0–15 kWh, south facing
at 30°. Three battery variants: the default field-calibrated v1 calendar model, the same with
`replacement_min_remaining_years = 2`, and the v2 fit. Each weather year is repeated over the 20
years, as a user running BREOS on that one year would get.

Decisions, each made on the TMY and then scored in every actual year:
- **max NPV:** the NPV-best design.
- **target XX:** the NPV-best design with projected grid independence ≥ XX % (50–90).

## NPV-best size: robust

The NPV-best design has no battery at every site (4–6 modules), so it doesn't depend on the
ageing model. The TMY picks the same design as the actual year in 81 % of site-years. Where it
picks a different one, the difference is at most 14 €. Sizing for NPV alone does not need more
than a TMY.

## Sizing to a self-sufficiency target

| Target | TMY pick = year's pick | TMY pick misses target | Shortfall when missed (median / max) | Median regret when met | Site-years with regret > 1 k€ |
|---|---|---|---|---|---|
| 50 % | 42 % | 33 % | 0.6 / 1.8 pp | 0 € | 0 % |
| 70 % | 25 % | 41 % | 1.1 / 4.5 pp | 120 € | 16 % |
| 90 % | 16 % | 51 % | 1.1 / 2.9 pp | 172 € | 0 % |

(Default model, all sites.) A target reached on the TMY is a coin toss in a real year, but the miss
is small. Users should read a TMY result as "about XX %", not "at least XX %". Meeting the target in a
weak year would have cost a median 150–570 € of NPV.

## The tail: replacement count

The largest regrets (up to 4.9 k€ with the default model, 5.3 k€ with v2) all have the same cause.
In the TMY, the pick's battery reaches end of life once more in the last years of the horizon, or
it does in the actual year and the year's own pick avoids it. Example: Stockholm, 70 % target,
default model. The TMY pick (19 modules, 10 kWh) needs one replacement in 12 of 19 years and two in
7 of them. The second one costs about 4.8 k€. This is study 3's replacement cliff, now showing up in
the sizing decision rather than in one design's NPV.

| Variant | Site-years with regret > 1 k€ | Worst front HV ratio (TMY front in an actual year) |
|---|---|---|
| default (v1) | 5.0 % | 0.962 (Berlin, Stockholm) |
| min remaining 2 y | 1.0 % | 0.997 |
| v2 | 5.4 % | 0.990 |

With `replacement_min_remaining_years = 2`, the whole TMY front stays within 0.3 % of each actual
year's own front at every site. Without it, Berlin, Erlangen and Stockholm lose up to 4 %.

## Robust picks

Choosing the design with the best mean over the 19 years, rather than the TMY's best, changes the
pick for targets of 60 % and above in about half the cases. It gains up to 1.7 k€ of mean NPV with
the default model, and at most 0.35 k€ with min remaining 2 y.

## Takeaways for BREOS

1. For NPV-only sizing a TMY is enough. No change needed.
2. Self-sufficiency targets met on a TMY are missed in 30–50 % of real years by about 1 point. The
   docs could say this, or the optimizer could take a margin.
3. The large sizing regrets come from the end-of-horizon replacement rule, not from weather.
   Making `replacement_min_remaining_years` (or an economic rule, #403, see study 13) the
   recommended setting for sizing would remove most of them.
