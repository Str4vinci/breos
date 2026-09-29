# Tariffs

A time-of-use tariff has three parts (ADR 0002):

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
weekends and national holidays are `off_peak` all day, using the holiday
calendar BREOS carries for each year. A year without one raises.

The boundary step is the coarsest step that lands on every period boundary
and on every change of the zone's UTC offset, since a regular index moves on
the local clock when the clocks change; it follows from the schedule's
intervals and zone. Input steps must divide it, and every step must start on
the local step grid: a schedule whose boundaries fall on the half hour or
quarter hour needs 15-minute input, and hourly input is rejected rather than
approximated.

## Schedule definitions

A {class}`~breos.tariffs.ScheduleDefinition` is a complete schedule: its
{class}`~breos.tariffs.TariffSchedule` metadata, its
{class}`~breos.tariffs.ScheduleRule` values, which give every day type and
season exactly one set of intervals, and an optional
{class}`~breos.tariffs.HolidayCalendar`. Every bundled schedule is one, built
from `tariffs.json` by {func}`~breos.tariffs.parse_schedule_definition`. The
classification and resolution functions and `TariffSpec` take either a
bundled identifier or a definition. A definition is immutable and pickles, so
it reaches optimizer worker processes unchanged.

## Resolution

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.tariffs.TariffSchedule
   breos.tariffs.ScheduleDefinition
   breos.tariffs.ScheduleRule
   breos.tariffs.HolidayCalendar
   breos.tariffs.TariffPrices
   breos.tariffs.ResolvedTariff
   breos.tariffs.available_tariff_schedules
   breos.tariffs.get_tariff_schedule
   breos.tariffs.get_schedule_definition
   breos.tariffs.parse_schedule_definition
   breos.tariffs.schedule_resolution_minutes
   breos.tariffs.classify_tariff_periods
   breos.tariffs.resolve_named_tariff
   breos.tariffs.resolve_tariff
   breos.tariffs.resolve_flat_tariff
   breos.tariffs.civil_day_starts
```
