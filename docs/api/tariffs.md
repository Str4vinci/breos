# Tariffs

A time-of-use tariff has three parts:

- a **schedule**, which assigns every instant to a named period (`off_peak`,
  `mid_peak`, `peak`) in local civil time and carries its regulatory source;
- **prices**, per kWh for import and export in each period, plus a daily fixed
  charge, in one currency; and
- a **resolved tariff**: both aligned to one simulation index, with period
  labels, price arrays, civil-day boundaries, and hashes for provenance.

BREOS bundles schedules only, never supplier prices: a regulated schedule
stays valid while prices change. Periods are classified in the location's
timezone, which is passed explicitly; the index's own timezone is never read,
so a UTC, fixed-offset, or local index resolves to the same periods.

## Bundled schedules

| Identifier | Periods | Boundary step | Source |
|---|---|---|---|
| `pt_mainland_2026_daily_bi` | off_peak, peak | 60 min | Diretiva ERSE n.º 1/2026, Art. 45.º |
| `pt_mainland_2026_daily_tri` | off_peak, mid_peak, peak | 30 min | Diretiva ERSE n.º 1/2026, Art. 45.º |
| `pt_mainland_2026_weekly_bi` | off_peak, peak | 30 min | Diretiva ERSE n.º 1/2026, Art. 45.º |
| `pt_mainland_2026_weekly_tri` | off_peak, mid_peak, peak | 15 min | Diretiva ERSE n.º 1/2026, Art. 45.º |
| `pt_mainland_2027_daily_bi` | off_peak, peak | 30 min | Diretiva ERSE n.º 3/2026, de 19 de agosto, Art. 2.º |
| `pt_mainland_2027_daily_tri` | off_peak, mid_peak, peak | 30 min | Diretiva ERSE n.º 3/2026, de 19 de agosto, Art. 2.º |
| `pt_mainland_2027_weekly_bi` | off_peak, peak | 30 min | Diretiva ERSE n.º 3/2026, de 19 de agosto, Art. 2.º |
| `pt_mainland_2027_weekly_tri` | off_peak, mid_peak, peak | 30 min | Diretiva ERSE n.º 3/2026, de 19 de agosto, Art. 2.º |
| `es_2_0td` | off_peak (P3), mid_peak (P2), peak (P1) | 60 min | CNMC Circular 3/2020, Art. 7.3 |

The Portuguese schedules are the low-voltage (BTN) access cycles for mainland
Portugal: `vazio` is `off_peak`, `cheias` `mid_peak`, `ponta` `peak`, and the
bi-hourly `fora de vazio` is `peak`. The 2027 schedules replace the 2026 ones
for each installation between 1 July and 31 December 2027, as its meter is
reparametrised. They are never chosen by date: select one explicitly, and pass
a `study_date` in its effective window when the simulated year is earlier.
`es_2_0td` is the Spanish Peninsula 2.0TD access tariff (up to 15 kW); its
weekends and national holidays are `off_peak` all day. BREOS carries the
national holidays for 2026 only, so a simulated year without a holiday
calendar raises; use a custom schedule with your own `holidays` for other
years.

The boundary step is the coarsest step that lands on every period boundary
and on every change of the zone's UTC offset in the simulated years, since a
regular index moves on the local clock when the clocks change. Lisbon and
Madrid change theirs by an hour every year, so no schedule there takes steps
longer than 60 minutes. Input steps must divide the boundary step, and every
step must start on the local step grid: a schedule whose boundaries fall on
the half hour or quarter hour needs 15-minute input, and hourly input is
rejected rather than approximated.

## Custom App schedules

App, Monte Carlo, and projected optimization accept an inline definition as
`[tariff.custom_schedule]` instead of the bundled `schedule` key. The supplied
identifier and version are retained in tariff provenance, while the complete
input table is recorded in `provenance.resolved_config`:

```toml
[tariff]
currency = "EUR"
import_prices = { peak = 0.31, off_peak = 0.12 }
export_prices = { all = 0.05 }

[tariff.custom_schedule]
identifier = "my_supplier_2026"
version = "2026-01"
timezone = "Europe/Lisbon"
cycle = "weekly"
periods = ["peak", "off_peak"]
source = "Supplier tariff sheet"
effective_from = 2026-01-01
effective_to = 2026-12-31

[[tariff.custom_schedule.rules]]
days = "weekday"
season = "all"
intervals = { off_peak = [["00:00", "08:00"], ["22:00", "24:00"]], peak = [["08:00", "22:00"]] }

[[tariff.custom_schedule.rules]]
days = "saturday"
season = "all"
intervals = { off_peak = [["00:00", "24:00"]] }

[[tariff.custom_schedule.rules]]
days = "sunday"
season = "all"
intervals = { off_peak = [["00:00", "24:00"]] }

[tariff.custom_schedule.holidays]
day_type = "sunday"
source = "Supplier holiday calendar"
dates = { "2026" = [2026-01-01, 2026-12-25] }
```

The schedule table requires `identifier`, `version`, `timezone`, `cycle`,
`periods`, and `rules`; optional metadata is `source`, `source_url`, `note`,
`effective_from`, `effective_to`, `holidays`, and `seasons`. Unknown keys are rejected
throughout the nested definition. Use a valid IANA timezone and make it match
the configured location exactly. Prices at `[tariff]` must cover every listed
period, or use `all`; with [month seasons](#month-seasons), each season's
table prices the periods that season's rules use, and no others.

Rules use the same semantics as bundled schedules: `days` is `weekday`,
`saturday`, `sunday`, or `all`; `season` is `standard`, `dst`, or `all`, or
with [month seasons](#month-seasons) a season name or `all`. Every
day-type/season combination must resolve to exactly one rule. Period intervals
are inclusive at the start and exclusive at the end, and must tile the local
day from `00:00` through `24:00` without gaps or overlaps. `24:00` is valid
only as an interval end.

Holiday dates are never inferred or fetched. `holidays.dates` maps years to
explicit date lists, and each date must belong to its mapped year. Include the
complete calendar you intend for every simulated tariff year; resolving a
year absent from the map raises an error. Omit the `holidays` table when the
schedule has no separate holiday classification.

## Month seasons

A custom schedule can vary its periods and prices by calendar month instead
of by DST. `seasons` names each season and lists its months, 1 through 12; a
quarter is a season of three months. The seasons must hold every month
exactly once. Each rule's `season` then selects a season name or `all`, and
`standard` and `dst` are refused: a schedule uses DST seasons or month
seasons, never both. Season names follow the period-name rules, must differ
from the period names, and cannot be `all`, `standard` or `dst`.

A step's season is the month of its civil date in the location's timezone,
so the season changes at local midnight on the first of the month, whatever
the index's own timezone. Day types, holidays and effective dates apply as
they do without seasons, and every project year replays the start-year
calendar.

Germany's §14a EnWG Module 3 network charges are one such tariff: the
network operator publishes low, standard and high windows that apply in some
quarters only, with the standard price all day in the others. BREOS bundles
no German schedule, because each network operator sets its own windows and
prices. Enter the operator's published values as a custom schedule. The
windows and prices below are illustrative, not any operator's:

```toml
[tariff]
currency = "EUR"
export_prices = { all = 0.08 }

[tariff.import_prices]
q1 = { low = 0.25, standard = 0.33, high = 0.42 }
q2 = { standard = 0.33 }
q3 = { standard = 0.33 }
q4 = { low = 0.25, standard = 0.33, high = 0.42 }

[tariff.custom_schedule]
identifier = "illustrative_module3"
version = "1"
timezone = "Europe/Berlin"
cycle = "custom"
periods = ["low", "standard", "high"]
seasons = { q1 = [1, 2, 3], q2 = [4, 5, 6], q3 = [7, 8, 9], q4 = [10, 11, 12] }

[[tariff.custom_schedule.rules]]
days = "all"
season = "q1"
intervals = { low = [["00:00", "06:00"]], standard = [["06:00", "17:00"], ["21:00", "24:00"]], high = [["17:00", "21:00"]] }

[[tariff.custom_schedule.rules]]
days = "all"
season = "q2"
intervals = { standard = [["00:00", "24:00"]] }

[[tariff.custom_schedule.rules]]
days = "all"
season = "q3"
intervals = { standard = [["00:00", "24:00"]] }

[[tariff.custom_schedule.rules]]
days = "all"
season = "q4"
intervals = { low = [["00:00", "06:00"]], standard = [["06:00", "17:00"], ["21:00", "24:00"]], high = [["17:00", "21:00"]] }
```

`import_prices` and `export_prices` are each given one of two ways:

- per period, as without seasons: every period of the schedule, or `all`;
  or
- per season and then per period, as above. Every season needs a table. A
  table prices exactly the periods that season's rules use, or gives `all`.
  A period the season never uses is an error, since its price could never
  apply.

A price list cannot mix the two forms, and prices by season need a schedule
with month seasons. `breos sweep` can vary one season's price with a key
such as `tariff.import_prices.q1.high`.

The month partition is part of the schedule: it is recorded in
`provenance.tariff.seasons`, and each step's season joins the schedule hash.
Two partitions that give the same periods still have different hashes, so
`App.revalue` re-simulates when the seasons change and re-prices when only
the prices change. Schedules without month seasons keep the hashes they had
before seasons existed.

## Schedule definitions

A {class}`~breos.tariffs.ScheduleDefinition` is a complete schedule: its
{class}`~breos.tariffs.TariffSchedule` metadata, its
{class}`~breos.tariffs.ScheduleRule` values, which give every day type and
season exactly one set of intervals, an optional
{class}`~breos.tariffs.HolidayCalendar`, and optional
{class}`~breos.tariffs.MonthSeasons`. Every bundled schedule is one, built
from `tariffs.json` by {func}`~breos.tariffs.parse_schedule_definition`. The
classification and resolution functions and `TariffSpec` take either a
bundled identifier or a definition. A definition is immutable and pickles, so
it reaches optimizer worker processes unchanged.

## No-system reference

A {class}`~breos.tariffs.ReferenceTariffSpec` is the App's
[`[reference_tariff]`](../getting-started/configuration.md#no-system-reference-tariff):
the tariff the household would pay without the system. It holds import
prices and a fixed charge on a bundled or custom schedule, or one flat price
without a schedule, and an optional escalation. It prices only the
no-system cost, the whole household load plus its fixed charge, and never
the system's grid flows or the dispatch. Its `resolve` method gives a
{class}`~breos.tariffs.ResolvedTariff` on the simulation index, as
`TariffSpec` does, with an export price of 0: the household without a system
exports nothing.

## Annual network credit

An {class}`~breos.tariffs.AnnualNetworkCredit` is the App's
[`annual_network_credit`](../getting-started/configuration.md#annual-network-credit)
table of `[tariff]` or `[reference_tariff]`, held as
`TariffPrices.annual_network_credit`. It gives an annual amount, the network
part of the fixed charge, and the network part of each import price, given
gross. Resolving the tariff adds each step's network price as
`ResolvedTariff.network_price_per_kwh`. The projection caps each household's
credit at that year's eligible network charges.

## Resolution

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.tariffs.TariffSchedule
   breos.tariffs.ScheduleDefinition
   breos.tariffs.ScheduleRule
   breos.tariffs.HolidayCalendar
   breos.tariffs.MonthSeasons
   breos.tariffs.TariffPrices
   breos.tariffs.AnnualNetworkCredit
   breos.tariffs.TariffSpec
   breos.tariffs.ReferenceTariffSpec
   breos.tariffs.ResolvedTariff
   breos.tariffs.available_tariff_schedules
   breos.tariffs.get_tariff_schedule
   breos.tariffs.get_schedule_definition
   breos.tariffs.parse_schedule_definition
   breos.tariffs.schedule_resolution_minutes
   breos.tariffs.classify_tariff_periods
   breos.tariffs.classify_tariff_seasons
   breos.tariffs.resolve_named_tariff
   breos.tariffs.resolve_tariff
   breos.tariffs.resolve_flat_tariff
   breos.tariffs.civil_day_starts
```
