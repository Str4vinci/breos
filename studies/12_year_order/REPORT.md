# Study 12 — Does the order of weather years matter?

**Question.** BREOS can run a projection on a real sequence of weather years (`weather_by_year`).
Battery ageing depends on what came before, and discounting weighs early years more. So does
the order of the same years change the result?

**Answer.** Hardly. Across 305 orders of the same 19 years, NPV moves by 50–140 € (standard
deviation about 10 €) for every design, site and ageing model. The replacement count never changed
with the order. Most of the spread is plain discounting: a PV-only design spreads by 56–94 €, about
as much as a battery design. Putting the worst years first costs at most 0.1 k€. The order of years
is not a risk worth modelling.

## Setup

Seven sites, 19 SARAH3 years 2005–2023 (long-run study 3 data), 19-year horizon so that every
order holds each year exactly once. Household and costs as in study 11. Orders: chronological,
reversed, worst-first and best-first (ranked by the year's DC yield), 300 seeded random shuffles,
and the TMY repeated 19 times as a reference. Designs: 10 modules with 0, 5 and 10 kWh, 6 × 5 kWh,
16 × 10 kWh. Calendar models v1 (default) and v2.

## Results

| | NPV range across all orders (median over sites) | Std over 300 shuffles |
|---|---|---|
| 10 modules, no battery | 79 € | 9 € |
| 10 × 5 kWh, v1 / v2 | 86 / 68 € | 10 / 9 € |
| 10 × 10 kWh, v1 / v2 | 93 / 53 € | 11 / 9 € |
| 16 × 10 kWh, v1 / v2 | 122 / 97 € | 14 / 12 € |

Best-first always gives the highest NPV and worst-first the lowest, as discounting predicts. The
largest range is 140 € (Berlin, 16 × 10 kWh). The ageing path does depend on the order, but over
19 years the differences don't move any end-of-life crossing across a year boundary that changes a
replacement.

## The repeated TMY vs a real sequence

The repeated TMY differs from the median real order by a median of 114 € (up to 365 €). That
difference comes from the TMY not being an average year, which study 3 covered. One case stands
out: Porto, 6 modules × 5 kWh, default model. The real sequence ends with the battery at 70.4 %
health, just above the 70 % end-of-life threshold, so one replacement is enough. On the repeated
TMY, it crosses the threshold just before the end and BREOS buys a second pack (final health
99.6 %), costing 1,171 €. It's the replacement cliff of studies 3 and 11 again.

## Takeaways for BREOS

1. Year order adds well under 1 % of NPV uncertainty. Monte Carlo can keep drawing years
   independently, and users don't need to worry about which years come first.
2. The only big differences between weather inputs come from the end-of-horizon replacement
   rule (see studies 11 and 13).
