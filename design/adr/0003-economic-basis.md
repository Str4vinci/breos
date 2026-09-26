# 0003 — Economic basis, escalators, and currency-neutral results

- **Status:** Proposed
- **Date:** 2026-09-26

## Context

The 0.7 plan calls for a BREOS economic-basis ADR before tariff valuation
(`design/architecture/0.7x-tariffs-and-smart-charging-plan.md`, source-to-target
map). The legacy research ADR it replaces is design evidence only. #183
records where the current code resists currency, TOU valuation and component
cashflows. This record settles the conventions; it changes no code.

What the code does today, in `cost_analysis_projection` (`breos/economics.py`)
unless stated:

- One rate, `inflation_rate`, escalates the import price, O&M, the daily
  charge and replacement cost. The CLI help calls it "annual electricity price
  inflation". Only export has its own rate, `sell_price_inflation`.
- Annual energy, O&M and daily-charge flows are priced at year-1 prices
  escalated by `(1 + i)^(n − 1)` and discounted at the end of year `n` by
  `(1 + d)^n`. The entered price is therefore the first project year's price,
  and a year of discounting remains even when `i = d`.
- Replacements are priced at t = 0, inflated to the swap instant and
  discounted from it.
- The daily charge assumes 365 days (`first_year_days = 365`), ignoring leap
  and partial years.
- Nothing says whether `discount_rate` is nominal or real.
- Defaults disagree. The App registry uses discount 0.03 and inflation 0.02;
  `CostParams` and optimization (`DEFAULT_DISCOUNT_RATE`,
  `DEFAULT_INFLATION_ELEC`) use 0.0 and 0.02; the `cost_analysis_projection`
  signature uses 0.02 and 0.03. `breos/data/configs/financials.json` says
  0.05 and 0.02 but is never read at runtime.
- Replacement cost is money inside the physics layer: the App sets
  `BatteryConfig.replacement_cost` from the storage cost per kWh, and only
  optimization can override it (`_replacement_event_cost`).
- Money names carry the currency: 19 distinct `*_eur` identifiers in
  `breos/`, including `total_investment_eur`, `npv_savings_eur`,
  `lcoe_eur_kwh`, `battery_replacement_cost_eur`, Monte Carlo
  `total_replacement_cost_eur` and optimization `*_NPV_Eur` columns. The
  projection columns (`Cost_Import`, `Revenue_Export`, ...) are already
  neutral.
- `battery_replacement_cost_eur` sums t = 0 prices, neither inflated nor
  discounted, beside discounted neighbours.
- There is no result schema version. `LEDGER_SCHEMA_VERSION` covers the
  ledger and loss waterfall only.

## Decision

Every item is **Proposed**.

### E1. Nominal basis, stated

All rates are nominal annual rates, and `discount_rate` is a nominal discount
rate. This is the arithmetic BREOS already performs and how tariff and
financing inputs are usually quoted. Provenance records the basis and the
implied real discount rate, `(1 + d) / (1 + inflation_rate) − 1`. A
real-terms study enters real rates and zero inflation; the arithmetic is the
same, and the recorded basis is how a reader tells the two apart. Real-terms
outputs are not added in 0.7.0.

### E2. Separate escalators, defaulting to today's behaviour

`inflation_rate` becomes general inflation, the default for every component
without its own rate. New keys (spelling settled in the config registry):

| Component | Key | Default |
|---|---|---|
| Import energy and fixed charge | `import_price_escalation` | `inflation_rate` |
| Export energy | `sell_price_inflation` (existing) | 0.0 |
| O&M | `om_escalation` | `inflation_rate` |
| Replacement price | `replacement_cost_learning` | 0.0 |

A replacement at time `t` years costs `C0 × (1 + inflation_rate)^t ×
(1 − learning)^t`. With every new key omitted the arithmetic is today's, so
the App golden baseline does not move. The CLI help for `inflation_rate`
changes to match.

### E3. Timing conventions kept and documented

Energy, fixed-charge and O&M flows stay at year-1 prices, booked at year end.
Replacements stay at t = 0 prices, inflated to and discounted from the swap
instant. Initial CAPEX is at t = 0. Changing the energy timing would move
every NPV for no gain in correctness; the user documentation states it,
including the year of discounting that remains when escalation equals the
discount rate.

### E4. Economics prices replacements

The physics layer reports replacement events (instant and replaced kWh);
economics prices them. `BatteryConfig.replacement_cost` leaves the physics
path. Learning rates and price revaluation then need no re-simulation, and
App and optimization price replacements the same way.

### E5. Fixed charge by simulated duration

The daily charge is `fixed_charge_per_day × simulated hours / 24`. A whole
non-leap year is still exactly 365 days, so those results do not change; leap
years and partial runs (#242) are charged for their actual length. The count
does not depend on the index timezone. Per-day charges that differ by day
type use the civil-day array of ADR 0002 A1.

### E6. One default set

Defaults are discount 0.03 and inflation 0.02, the values App users already
get. They are defined once, beside `CostParams`, and read by the App registry,
optimization and Monte Carlo. The rate parameters of
`cost_analysis_projection` lose their own defaults. `financials.json` is
deleted. Configurations that relied on the optimization default of 0.0 change
and get a release note.

### E7. Year rows carry money at year-1 prices

The year loop adds import cost, export revenue, fixed charge and no-system
import cost, at year-1 prices, to each year row. Economics applies
escalators, timing and discounting. The flat case computes kWh × price in the
current operation order and stays bit-identical; TOU fills the same columns
from `sum(energy × price)` over the steps. Storing each year's energy by
tariff period, for revaluation without re-simulation, is a 0.7.x follow-up.
The App `financial` rows gain the component cashflows the projection already
computes: `Cost_Import`, `Revenue_Export`, `Cost_Operation`, `Cost_Daily`,
`Cost_Replacement` and `Replacement_Time_Years`.

### E8. Currency-neutral names, no aliases

Money keys drop the currency token: `total_investment`, `npv_savings`,
`lcoe_per_kwh`, `Projected_NPV`, and so on. The resolved currency is recorded
once in result metadata and provenance, and plot labels read it. The
replacement total's name states its basis
(`battery_replacement_cost_t0_prices`), with the discounted total beside it.

The `*_eur` keys are renamed in 0.7.0 without aliases. This amends ADR 0002's
"compatibility aliases only while the resolved currency is EUR". Aliases would
give EUR and non-EUR runs different key sets, double every money field in the
App result, the golden fixture and the Monte Carlo and optimization frames,
and still need a removal release later. BREOS removes superseded APIs rather
than deprecating them (#164), and ledger schema 2.0 already drops duplicate
columns the same way. The cost is that scripts reading `*_eur` keys break
once, loudly; the changelog carries an old-to-new table.

### E9. Result schema version

`App.result()`, Monte Carlo summaries and optimization provenance gain a
top-level `result_schema_version`, independent of the ledger schema. It
starts at `"1.0"` with the E8 names. A rename or removal bumps the major
version; an added field bumps the minor.

## Consequences

- With no new keys, flat App results match the golden baseline except for
  leap-year and partial runs (E5). Optimization without explicit financials
  changes with the default discount rate (E6).
- TOU valuation, escalator scenarios and replacement learning share one
  valuation step and need no re-simulation for price-blind dispatch.
- Downstream code must move to the neutral names in one release.
- Splitting `cost_analysis_projection` into valuation, discounting and
  metrics, emissions, and file output, and computing LCOE and lifetime CO2
  once rather than again in each runner, are implementation work under #183.
