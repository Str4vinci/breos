# Tariffs and smart charging

**Status:** Implemented in 0.7.0.

This note gives contributors an overview of how time-of-use (TOU) tariffs and
smart charging fit into BREOS. The accepted decisions and their reasons are in
[ADR 0002](../adr/0002-tariffs-and-smart-charging.md) (tariffs and smart
charging) and [ADR 0003](../adr/0003-economic-basis.md) (economic basis and
currency-neutral results). User-facing behavior is documented in the
[Time-of-use tariffs](../../docs/getting-started/configuration.md#time-of-use-tariffs)
and [Smart charging](../../docs/getting-started/configuration.md#smart-charging)
guides and in the [Tariffs API reference](../../docs/api/tariffs.md).

## Principles

- **One physics engine.** The canonical dispatch step in `breos/_dispatch.py`
  is the only code that moves energy. It owns conservation, conversion
  losses, power and SOC limits, origin accounting and the ledger.
  Degradation and replacement stay in the Python year loop around it.
  Tariffs and controllers prepare inputs for that step. They never simulate
  the battery themselves.
- **Tariffs are values, not a simulation mode.** A tariff changes what energy
  costs. It changes the energy flows only when a smart-charging mode reads
  its periods.
- **Opt-in, with exact defaults.** Without `[tariff]` the run uses the flat
  cost-preset prices. Without `[smart_charging]`, or with
  `mode = "disabled"`, dispatch is greedy self-consumption. The App golden
  fixture (`tests/fixtures/app_golden`) pins both defaults.
- **Static, versioned inputs.** Schedules are dated regulatory data. Prices
  are user inputs. BREOS fetches no live prices and converts no currencies.

## Data flow

```text
[tariff] / [reference_tariff]          [smart_charging]
          |                                    |
          v                                    v
  schedule + prices  --resolve-->  ResolvedTariff  -->  DispatchInstructions
  (breos/tariffs.py)               (periods, prices,     (breos/smart_charging.py,
                                    civil days, hashes)   breos/dispatch_instructions.py)
                                          |                        |
                                          |                        v
                                          |       canonical dispatch step, per civil day
                                          |       (breos/_dispatch.py, breos/_controller.py)
                                          |                        |
                                          v                        v
                         shared projection year loop (breos/projection.py):
                         year rows of energy plus money at year-1 prices
                                          |
                                          v
                         escalation, discounting, NPV, LCOE, payback
                         (breos/economics.py)
```

## Tariff domain

`breos/tariffs.py` keeps three immutable concepts apart (ADR 0002):

1. A **schedule definition** assigns local civil time to named periods and
   records its regulatory source and effective dates. The bundled schedules
   are in `breos/data/configs/tariffs.json`: eight Portuguese mainland
   schedules (2026 and 2027, daily and weekly, bi- and tri-hourly) and the
   Spanish 2.0TD schedule. `[tariff.custom_schedule]` defines a schedule
   inline, optionally with calendar-month seasons (a quarter is a season of
   three months) that switch rules and prices by month.
2. **Prices** for import and export per period, or per season and period,
   plus a daily fixed charge, in one currency. 0.7.0 supports EUR only.
3. A **resolved tariff** is both of these aligned to one simulation index. It
   holds the period labels, the price arrays, the civil-day boundaries and
   separate schedule and price hashes.

Resolution classifies each instant in the configured timezone and never reads
the index's own timezone (A1). Every project year replays the start-year
calendar (A2). A schedule boundary that the step cannot represent is rejected
under `boundary_policy = "strict"`, the only policy. Half-hour boundaries
therefore need 15-minute input (A3). An optional `[reference_tariff]` prices
the no-system baseline separately. It never drives dispatch (A13).

### Schedule sources

Schedules are checked against primary sources, not against earlier code:

- The Portuguese 2026 periods cite Diretiva ERSE n.º 1/2026. BTN tri-hourly
  has three periods; there is no fourth `super_off_peak` period.
- The 2027 periods cite
  [Diretiva n.º 3/2026, de 19 de agosto](https://diariodarepublica.pt/dr/detalhe/diretiva/3-2026-1159784501),
  Article 2. The directive with the same number dated 26 June concerns gas
  tariffs. Article 3 phases BTN meter changes in from 1 July through
  31 December 2027. A user selects a 2027 schedule explicitly, with a
  `study_date` when the simulated year is outside its effective window. The
  calendar never switches schedules on its own.
- Spain's 2.0TD cites CNMC Circular 3/2020, not Royal Decree 446/2023, which
  concerns PVPC price calculation.
- There is no German preset. Under the
  [Bundesnetzagentur §14a framework](https://www.bundesnetzagentur.de/DE/Vportal/Energie/SteuerbareVBE/artikel.html?nn=877500),
  each network operator sets its own Module 3 windows and prices. Users
  describe them with a custom schedule. The example
  `configs/examples/quarterly-tariff-berlin.toml` uses invented windows and
  prices.

Supplier prices always come from the user, so a schedule never implies a
retail price.

## Smart charging

A mode resolves to per-step `DispatchInstructions`: whether discharge is
allowed, the reserve kept before discharge, and a grid-charge target or none.
The canonical step applies them under every physical limit. All-permissive
instructions reproduce greedy dispatch bit for bit.

| Mode | What it does | App | Monte Carlo | Projected optimization |
|---|---|---|---|---|
| `disabled` (or no table) | Greedy self-consumption | Yes | Yes | Yes |
| `fixed_target` | Grid-charges toward `target_usable_fraction` in `charge_periods`; discharges in `discharge_periods` | Yes | Yes | Yes |
| `discharge_only` | Discharges only in `discharge_periods`; never grid-charges | Yes | Yes | Yes |
| `daily_persistence` (experimental) | Plans one target per civil day from the last complete observed day (A12) | Yes | Refused | Refused |

The amendments to ADR 0002 settle these conventions:

- The target is a fraction of each step's current usable window, so it moves
  with temperature and state of health (A7).
- Grid charging is an AC-to-DC path with its own `grid_charge_efficiency`,
  followed by the battery's charge efficiency. PV keeps priority on every
  shared limit. PV AC output and grid-charge AC input together count against
  the inverter AC rating (A6).
- Stored energy has PV, grid and unattributed origins. Only PV-origin
  discharge counts as self-consumption. Avoided emissions use net exchange,
  so grid energy that the battery time-shifts earns no credit (A8, A10).
- Charge and discharge periods are disjoint by default. With
  `overlap_policy = "hold_target"` (`fixed_target` and `daily_persistence`),
  a period in both sets holds the target as the discharge floor (A14). Under
  `daily_persistence` and the daily-target oracle the floor follows each
  day's target (A18).
- Controllers decide at configured-timezone civil-day boundaries.
  Degradation windows stay positional (A11).
- Normal runs carry stored energy, origins and degradation state from one
  project year to the next (`physical_carry`).

The Python and Numba backends run the same day-loop source; see
[Numba dispatch backend](numba-dispatch-backend.md).

## Valuation and revaluation

App, Monte Carlo and the projected optimizer share one projection year loop
(A5). Each year row has the energy totals and the money at year-1 prices:
import cost, export revenue, fixed charge, grid-charge cost and the no-system
baseline (ADR 0003 E7). Economics then applies the escalators, timing and
discounting (E2, E3), prices replacements (E4) and computes the
currency-neutral results (E8). The optional `[terminal_value]` credit for
battery health is reported beside the unadjusted NPV (E10).

`App.revalue` re-prices a stored run without simulating again when the new
prices cannot change the dispatch. For this, the run records its energy by
tariff period (and season). A change that can move the dispatch, for example
a new schedule or new prices under `daily_persistence`, causes a new
simulation. `provenance["revaluation"]` records which of the two happened.

## Validation tooling

`tools/oracles/` holds offline tools that check smart-charging schedules.
They are not public API and not App strategies. Each one replays its schedule
through the production App run (`tools/oracles/replay.py`) before it reports
a cost. The two tools answer different questions and are not
interchangeable:

- `tools/oracles/daily_target_dp.py` gives **perfect-information daily
  targets** on a grid. It plans a project year with perfect foresight, with
  one grid-charge target per civil day from a grid of usable fractions
  (`--target-levels`) and a grid of stored energy (`--soc-states`). It
  holds state of health and efficiencies at the year's opening values and
  keeps the configured instruction layout. By default it plans the first
  year, and every year replays that plan. With `--planning yearly` (ADR
  0002 A16) it plans each year as the year begins: on that year's degraded
  PV, load and temperature, and at the stored energy, health and
  efficiencies that the production replay of the earlier years reached. The
  report records the planning mode and each year's inputs, opening state,
  plan and replayed cost. `--wear-cost-per-kwh` gives every solve the
  planner's wear weight (ADR 0002 A17, default 0); the report records it
  and each plan's wear cost apart from its stage cost. The result is the
  best schedule within that policy class under the planner's model. It is
  not a bound, and not a bound on lifetime NPV: other policies can do
  better, the replayed cost under production physics differs from the plan
  as health moves, and a year's choice ignores what its cycling costs later
  years.
- `tools/oracles/lp_bound.py` is a **conditional lower bound** on the first
  project year's import cost less export revenue, without the fixed charge.
  The linear program relaxes the dispatch rules. The bound holds only for a
  dispatch that stays at or above the program's floor state of health, has
  no replacement in that year, and grid-charges at the program's converter
  efficiency and within its site limit. Terminal energy is left free,
  because a cyclic end state is not a bound on one year's bill. The tool
  checks each reported run against the bound and reports `bound_is_strict`.
  Outside this scope it claims nothing.

Neither tool gives a forecast, a deployable controller or a bound on lifetime NPV.

## Outside 0.7.0

These items are not implemented:

- Live tariffs, market prices, FX conversion and currencies other than EUR.
- 30-minute input, and boundary policies that approximate a boundary.
- Calendars that advance across project years (A2).
- Inverter netting of PV and grid-charge AC flows (A6).
- Thermal storage, heat pumps, electro-thermal dispatch and vehicle-to-home.
- Price-aware predictive dispatch beyond the experimental daily planner.
- Collective self-consumption (ACC). Sharing coefficients are a pure
  transformation of a community production series and a per-participant
  load matrix, but the App models one site and one load. A community model
  needs stable participant identities and an explicit settlement ledger that
  reconciles every 15-minute interval. It also needs a defined order for
  shared PV, community batteries and behind-the-meter batteries, and
  participant tariffs and ownership, which the physical coefficient does not
  give. Fixed and proportional allocation would come first. Dynamic
  allocation would come later, as an experimental mode with a plain NumPy
  reference.
