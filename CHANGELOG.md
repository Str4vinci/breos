# Changelog

All notable changes to BREOS are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- A `[reference_tariff]` table prices the household without the system on
  its own tariff, independent of the system's `[tariff]` or flat prices
  ([#339](https://github.com/Str4vinci/breos/issues/339)). It takes import
  prices on a bundled or custom schedule, or one flat price
  (`import_prices = { all = <price> }`) without a schedule, a
  `fixed_charge_per_day` (default 0) and an optional
  `import_price_escalation`, which escalates the reference energy and fixed
  charge and defaults to the system's import escalation; an explicit 0 is
  kept. It has no export prices. The no-system cost of each year is then the
  whole household load at the reference prices plus the reference fixed
  charge. The reference must be in the result's currency and passes the
  system tariff's timezone and resolution checks. It never changes the
  dispatch or the costs with the system. App, Monte Carlo (each trajectory on
  its own sampled load) and projected optimization (its NPV objective) accept
  it. `App.revalue` re-prices a reference that is added, changed or removed
  without simulating again, also under `daily_persistence`. Sweeps accept
  dotted keys such as `reference_tariff.import_prices.all`. Result schema 2.4
  adds the no-system cost components with or without a reference
  (`no_system_fixed_charge_year1_prices`, the `financial` rows'
  `no_system_cost_import` and `no_system_cost_fixed_charge`, and the
  `Cost_No_Sys_Import` and `Cost_No_Sys_Fixed_Charge` projection columns),
  `reference_tariff` in `resolved_config`, and `provenance.reference_tariff`
  when one is configured. Without a reference every existing value is
  unchanged.
- `battery_allow_terminal_replacement` (App and Monte Carlo), the
  optimizer's `[battery] allow_terminal_replacement` and
  `BatteryConfig.allow_terminal_replacement` can skip buying a battery that
  would serve no step of the horizon
  ([#304](https://github.com/Str4vinci/breos/issues/304)). A pack that
  reaches end of life in the horizon's final degradation period was replaced
  and priced although it delivers no service. With `false`, only that
  replacement is skipped. The final period is the one that ends on the last
  simulated step: the last whole day, or a trailing partial day, or the whole
  span when it is shorter than a day. A whole day followed by a partial day
  is not final and keeps its replacement, because the new pack serves the
  remaining steps; this is how the "last whole day" of the original proposal
  reads with partial-period aging. The final period is still dispatched,
  aged, recorded and finalized, and the result reports the old pack's state.
  Earlier periods replace as usual, including the close of each earlier
  project year, whose pack the next year inherits, and Monte Carlo applies
  the key to the final year of each trajectory. Smart-charging decisions do
  not change. The default `true` changes no result; only an opted-out
  horizon whose final period reaches end of life changes its final battery
  state, replacement count and replacement costs. Result schema 2.3 records
  the value in `provenance.resolved_config` and, for a projected design and
  an optimizer search, in `battery_replacement_treatment`.
- `tools/convert_eredes_profiles.py` converts the E-REDES consumption-profile
  publication (`Perfil_Consumo_Injecao_E-REDES_<year>.csv`) to the
  `EREDES_<year>_BTN_1000kwh_15min.csv` and
  `EREDES_<year>_BTN_1000kwh_hourly.csv` files that `eredes_btn_a`,
  `eredes_btn_b` and `eredes_btn_c` read
  ([#303](https://github.com/Str4vinci/breos/issues/303)). It reads the
  Latin-1 file with its four header rows, finds the BTN A/B/C columns by their
  labels and drops blank rows. It refuses a file whose dates do not cover one
  complete calendar year, whose weekday does not match its date, whose values
  are not finite and non-negative, or whose quarter-hours have a gap or a
  repeat. The publication stamps each quarter-hour at its end in Portuguese
  legal time; the converter writes interval starts on the civil clock, 96 per
  date. `24:00` is the next midnight, and each end moves back 15 minutes. The
  repeated fall-back hour keeps its standard-time occurrence, and the skipped
  spring-forward hour is interpolated. These two rules are BREOS's own, not an
  E-REDES method: the shape of the first, summer-time fall-back occurrence is
  discarded, and the interpolated hour stays in the load when day-class
  alignment moves its date into a study year where that hour exists. For the
  2026 publication they changed each BTN profile's annual energy by less than
  5 Wh in 1,000 kWh; the load is scaled to `annual_consumption_kwh` exactly.
  kWh become Wh, and each hourly value is the sum of its four quarter-hours,
  so both files have the same phase. The output year comes from the dates, not
  the filename. The converter makes no network access, and BREOS still bundles
  no E-REDES data.
- Experimental App smart charging with `mode = "daily_persistence"` chooses
  a grid-charge target each local day from the last complete day's PV DC,
  load and input temperature and the known tariff. It starts without grid
  charging until a complete local day has been observed, carries observations
  and physical battery state across project years, and prices forecast-terminal
  energy shortfalls instead of treating depletion as free. The planner horizon
  and target/state grids are configurable; Monte Carlo and optimization refuse
  the mode. `App.revalue` simulates again when import or export prices change.
  Result schema 2.2 records the experimental policy, effective settings,
  executed instruction hash and initial/final stored energy by origin in
  `provenance.smart_charging`; existing modes change only their schema version.
- `[tariff.custom_schedule]` lets App, Monte Carlo and projected optimization
  use a strict inline schedule definition. It accepts the same periods, rules,
  effective dates and explicit year-keyed holidays as the tariff schedule
  parser, and records the definition in `provenance.resolved_config` so the
  run can be reproduced. Result schema 2.1 adds that config field.
- `breos.tariffs.ScheduleDefinition` holds a complete tariff schedule: its
  `TariffSchedule` metadata, `ScheduleRule` intervals per day type and
  season, and an optional `HolidayCalendar`. It is frozen and pickles.
  `parse_schedule_definition` builds one from the mapping form the bundled
  `tariffs.json` uses, and the catalogue now goes through it;
  `get_schedule_definition` returns a bundled one. `classify_tariff_periods`,
  `resolve_named_tariff`, `schedule_resolution_minutes` and `TariffSpec`
  take a bundled identifier or a definition. The resolution a schedule needs
  now follows from its interval boundaries and the changes of its zone's UTC
  offset in the simulated years (`schedule_resolution_minutes(schedule,
  years)`), and `tariffs.json` no longer declares it; every bundled schedule
  derives the value it declared. Classification checks that every step, not
  only the first, starts on the local step grid. Every bundled schedule
  gives the same period labels and schedule hashes as before, and no result
  changes. This is the first step toward custom tariff schedules in `App`.
- `breos.montecarlo.build_year_cache(config, settings)` prepares a Monte
  Carlo study's per-year weather, PV production and battery temperature
  once, and `run_montecarlo(..., year_cache=cache)` reuses them across a
  sweep over designs ([#165](https://github.com/Str4vinci/breos/issues/165)).
  The weather layer is keyed on the weather file's absolute path, its
  SHA-256 and that of its metadata sidecar, the year window, `target_year`,
  resolution, coordinates, `preserve_irradiance_energy` and the
  solar-position method; a study with other weather inputs raises
  `ValueError`. The PV layer is keyed on the resolved config without
  `YEAR_CACHE_INDEPENDENT_KEYS` (the App sweep's `INPUT_INDEPENDENT_KEYS`,
  the demand keys and `[montecarlo]`) and on the contents of a
  `battery_temperature` CSV, so a change to any other key, such as
  `n_modules` or `battery_temperature`, rebuilds it from the cached
  weather. Results match a study run without the cache bit for bit. On a
  15-minute, 10-year file with 8 runs of 20 years on the Numba backend, a
  battery-size design went from 3.5 s to 0.7 s and a module-count design
  from 3.7 s to 2.1 s.
- A `[period]` table simulates a window shorter than a year in `App`
  ([#242](https://github.com/Str4vinci/breos/issues/242)), for example
  `{"start": "2025-06-01", "end": "2025-06-08"}`. `start` and `end` are
  civil dates in the location's timezone, in the year of `start_date`; the
  window runs from local midnight of `start` to local midnight of `end`, so
  `end` is exclusive and may be 1 January of the next year. The full-year
  load is built and scaled to `annual_consumption_kwh` as before, then
  weather, PV, load and battery temperature are cut to the window, which
  runs once from the battery's initial state; `projection_years` is not
  used, and setting it alongside `[period]` gives a warning. The weather
  must cover the window; missing leading or trailing rows raise, so UTC-year
  weather cannot serve a window at a civil year edge away from UTC, and
  weather stamped at half past the hour cannot place a window. Energy
  results, the year-1-price money, the year-1 CO2 and the PV loss waterfall
  cover the window, and the fixed charge is billed on its civil days, so a
  window over a DST change bills whole days. The lifetime economics
  (`npv_savings`, `payback_year`, `lcoe_per_kwh`, `financial`, the
  replacement costs and the lifetime CO2) are `None`, and a new `period`
  block, also in `provenance.period`, records the window,
  `projection_years_used = 1` and why. `breos validate-config` lists the
  window only when it is set. The one `yearly` row carries `period_start` and `period_end`, and `monthly` groups the window by
  civil month. A PV-only window equals the same steps of a full-year run
  exactly. `App.revalue` re-prices a window; `breos sweep` can vary
  `period.start` and `period.end`; Monte Carlo and the optimizer reject
  `period`. Result schema 1.7. Runs without `period` are unchanged apart
  from the schema version.
- `inverter_ac_rating_kw` sets the inverter AC rating in kW, instead of
  `inverter_loading_ratio` ([#181](https://github.com/Str4vinci/breos/issues/181));
  setting both in one config raises. `--inverter-ac-rating-kw` sets it from
  the CLI, and a flag or a `breos sweep` value for one of the two replaces
  the other from the config file, as any flag replaces the file's value.
  `ResolvedAppConfig.inverter_ac_capacity_w` is resolved once, from either
  key, and the dispatch, CAPEX and the reports read it: `calculate_costs`
  gains `inverter_ac_capacity_w`, the rating to price, instead of re-deriving
  it from the ratio. With the rating set, `resolved_config` reports
  `inverter_loading_ratio` as unset. Result schema 1.6. CAPEX now divides the
  rating in W by 1000 where it divided the kWp by the ratio, so
  `inverter_cost` can differ in the last bit: over 480 module, count, ratio
  and battery combinations it did in 86, `total_initial_cost` in 2 (by at
  most 1.3e-16 of its value), and no value changed at two decimals. The
  optimizer keeps `costs.dc_ac_ratio`, since each candidate's rating follows
  its module count.
- The optimization config is checked and defaulted once, by
  `breos.optimization_config.resolve_optimization_config`
  ([#181](https://github.com/Str4vinci/breos/issues/181),
  [#162](https://github.com/Str4vinci/breos/issues/162)).
  `optimize_system_multi_objective`, `SolarDesignProblem` and
  `evaluate_projected_design` resolve their config before any candidate is
  scored. Every table takes a fixed set of keys, and an unknown key raises
  instead of being ignored. The resolved config carries every default the
  search reads: `constraints.budget` 10,000, `max_area_m2` 20,
  `max_modules` 60, `max_battery_kwh` 30, the tilt floor, now the key
  `constraints.min_tilt_deg` (10°), the horizon, PV degradation, the
  location's timezone and the financial rates. The early-stop tuning keeps
  its documented defaults. The settings the App shares (the tariff,
  smart-charging, cost and indoor-model tables, the projection horizon, PV
  degradation, inverter efficiency and the default module) default as the
  App's do. The search records its resolved `constraints` and
  `run_settings` in `details["provenance"]`. `optimize_system_multi_objective`
  now reads `pop_size`, `n_gen`, `n_offsprings` and `seed` from
  `[optimization]` when they are not passed; an argument that disagrees with
  its key raises. `[emissions]` now reaches the search, so every Pareto row
  of a search with emissions carries `Projected_CO2_*`. Result schema 1.5.

  Numbers change in one case: a config whose `[optimization]` sets
  `pop_size`, `n_gen`, `n_offsprings` or `seed`, run without passing them.
  Those keys used to be ignored, so the search ran at 40, 100, pymoo's
  default and seed 1; it now runs at the configured values. The example
  config, called as the optimization guide now shows, runs 40 generations
  with 20 offspring instead of 100 generations with pymoo's default. Any
  other config the optimizer still accepts gives the same numbers, bit for
  bit.
- `App.revalue(changes)` values a finished run at other prices
  ([#183](https://github.com/Str4vinci/breos/issues/183)). It accepts the
  economics keys only (`costs`, `cost_preset`, `tariff`, `discount_rate`,
  `inflation_rate` and the escalators; `breos.app.REVALUATION_KEYS`) and
  raises `ValueError` for any other key. A nested table changes only the
  keys it sets, `None` removes a key, and a tariff price list replaces the
  old one whole. When the prices cannot change the
  dispatch it re-prices the stored run: flat prices, a tariff removed, or new
  prices on the same schedule with unchanged smart-charging instructions.
  Otherwise it simulates again. `provenance.revaluation` records
  `method` (`"repriced"` or `"resimulated"`) and `changed_keys`. A flat
  revaluation gives the same floats as a new run; a re-priced tariff sums
  each year's energy by tariff period, which projections on a tariff now
  record (`ProjectionRun.period_energy`), and agrees with a new run to
  rounding. The App and its `result()` are unchanged. Result schema 1.4.
- `cost_analysis_projection` is split into public stages:
  `price_year_rows` and the new `value_year_rows` (energy to component
  cashflows), `discount_cashflows` (cumulative and discounted cashflows,
  payback, NPV and LCOE), `add_co2_projection` and `write_cost_projection`.
  Cost projections, and Monte Carlo
  trajectories with them, gain `CO2_Avoided_Export_kg`,
  `CO2_Avoided_Export_Cumulative_kg` and
  `attrs["lifetime_co2_avoided_export_kg"]`.
- Grid charging and dispatch instructions in the battery step
  ([#178](https://github.com/Str4vinci/breos/issues/178), ADR 0002 A6–A8).
  `simulate_energy_balance` and `simulate_energy_balance_summary` accept
  `dispatch_instructions`, a `breos.dispatch_instructions.DispatchInstructions`
  of per-step arrays:
  - whether the battery may discharge;
  - the usable fraction it keeps before discharging;
  - a grid-charge target as a usable fraction, or NaN for none;
  - the grid-charge AC-to-DC efficiency and a site import limit.

  A fraction applies to each step's capacity window, so the energy it names
  moves with temperature and SOH. Grid charging runs after PV allocation in
  the same step. Its DC input passes through the cell charge efficiency and
  is booked as `Battery_Charge_Input`, so cell losses, self-heating and both
  aging engines see it. PV keeps priority on the charge-power, stored-power,
  inverter-rating and site-import limits the two share. A step that exports
  PV or discharges does not grid-charge. New results columns:
  `Grid_AC_To_Battery` (part of `Import_From_Grid`), `Grid_DC_To_Battery`,
  `Grid_Charge_Conversion_Loss` and `Grid_Origin_Battery_Charge_Stored`. Year
  rows gain `Grid_AC_To_Battery_kWh` and `Grid_Charge_Conversion_Loss_kWh`.
  Both backends run the same step and match bit for bit under instructions.
  Without instructions, results are unchanged on both backends. No App or
  CLI setting takes the instructions directly; `[smart_charging]` builds them. A year's dispatch is
  about 8–12% slower on the Python backend and about 3% slower on Numba.
- Added the external validation page to the documentation. It collects the
  measured-data checks against NIST Gaithersburg, the DKA Solar Centre, IEA PVPS
  Task 13, the UCY PHAETHON test-bed, and a Reunion Island microgrid, with the
  measurement boundary each dataset supports. It replaces
  `validation/external/README.md`.
- `resample_to_15min` accepts `altitude` for the clear-sky model. When it is
  omitted, pvlib still looks the elevation up from the coordinates, as before.
- The package ships a `py.typed` marker, so mypy, pyright and IDEs now use
  BREOS's own annotations when checking code that calls it, instead of
  treating the package as untyped. Downstream type checks may report new
  errors where calls did not match the annotated signatures.
- A baseline mypy configuration in `pyproject.toml`; `uv run mypy breos` passes
  with the `dev` extra, which now includes mypy and pandas-stubs
  ([#185](https://github.com/Str4vinci/breos/issues/185)). Six modules are
  still excluded from error reporting until their annotations are fixed.
- `breos.io.repair_series`, an explicit repair step for measured load and PV
  power series, run before a simulation
  ([#194](https://github.com/Str4vinci/breos/issues/194)). The simulation
  still rejects gaps, non-finite values, and negative load (#151); this is the
  opt-in way to clean such data with a record. Negative readings within a
  repair tolerance are clipped to zero: by default down to −10 W
  (`negative_clip_w=10.0`), in stretches lasting at most one hour of interval
  duration (`max_negative_run="1h"`). The tolerance is not evidence that the
  readings are noise. More negative or longer stretches raise, since for load
  they usually mean a net-meter reading; the caller investigates the data or
  raises the limits explicitly. Gaps (missing timestamps, NaN, ±inf) raise by
  default; `gap_fill="nearby_days"` fills each step with the mean of the same
  time of day on up to two nearest valid days, preferring the same day type
  for load, and raises if none is in reach. `window_days=7` bounds that donor
  search either side of each step, not the gap length: a gap of up to 14 days
  with valid data on both sides is filled, its middle from a single day.
  Duplicate timestamps and an irregular index raise. It returns the repaired
  series and an `InputRepairReport` listing every repaired run, its method,
  the values written, and the energy added, with a strict-JSON `to_dict()`.
  Pass the reports as `App(config, input_repairs=[report])` to record them
  under `provenance.input_repairs`. Runs without them are unchanged and have
  no such key. There is no config key or CLI option for repair yet.
- The `weather_source` App config key, also `--weather-source` on the CLI,
  picks one cached TMY file when `weather/` holds several for a location
  preset. It names the filename's source part, as in
  `porto_tmy_2005_2023_pvgis-sarah3.csv`. The default, `None`, uses the only
  matching file as before. App rejects a malformed value, or one set with a
  coordinate-dict location, at construction. A source with no matching file
  raises instead of fetching PVGIS weather. The file used is recorded under
  `provenance.weather`, as before.
- A test that pins the timezone of the App result index for each weather
  source ([#180](https://github.com/Str4vinci/breos/issues/180)). The index
  takes the weather's clock, not the configured timezone: a PVGIS fetch runs
  on the fixed offset of 1 January all year (`Etc/GMT-1` for Berlin,
  `Etc/GMT-11` for Melbourne); a cached weather CSV keeps a single fixed
  offset written in its stamps and is read as UTC otherwise; Monte Carlo
  weather is UTC. No behaviour changes.

- **Load-profile registry** ([#182](https://github.com/Str4vinci/breos/issues/182)).
  `breos.load_profiles.PROFILES` holds one `ProfileSpec` per profile family:
  its filename patterns per resolution, the accepted columns and their units,
  and whether it is bundled. `resolve_profile_key()` makes a key canonical,
  and `resolve_profile_file()` finds the CSV. App config validation,
  `load_profile` and the CLI all go through them. The canonical keys are
  `demandlib_h0` (bundled, the default), `eredes_btn_a`, `eredes_btn_b`,
  `eredes_btn_c`, `bdew_h0`, `ree_2.0td` and `custom`; keys are
  case-insensitive.
  - External files are found by pattern in `rlp_directory`, with the year as
    `*` (`EREDES_*_BTN_1000kwh_15min.csv`, `bdew_h0_*_15min.csv`,
    `REE_*_2.0TD_1000kwh_hourly.csv`, ...), so one key covers every vintage.
    The pattern must match exactly one file: when several match, the error
    lists them instead of picking one. The new `load_profile_file` key (CLI
    `--load-profile-file`) names the file explicitly; a relative path is
    taken inside `rlp_directory`.
  - `load_profile = "custom"` reads any CSV named by `load_profile_file`, with
    `load_profile_unit` (`W` or `kW` mean power, `Wh` or `kWh` energy per row)
    and, when the file has several value columns, `load_profile_column`. Its
    resolution comes from its row count. The column and unit keys are
    rejected for the other profiles, which fix their own.
  - `result()["provenance"]["load_profile"]` records the canonical key, the
    file read (packaged filename or absolute path), its SHA-256, its native
    resolution, and the column and unit read. Monte Carlo provenance has the
    same block. `breos validate-config` prints the file it resolved, or says
    it is not there yet.
- Year rows carry money at year-1 prices (ADR 0003 E7): `Import_Cost`,
  `Export_Revenue`, `Fixed_Charge` and `Baseline_Import_Cost` (the load bought
  without a system), added by `breos.economics.price_year_rows`, plus the
  simulated duration, `Simulated_Hours`. `cost_analysis_projection`
  escalates, times and discounts these columns, and keeps any a caller
  supplies, which is how TOU valuation will fill them. The flat case
  multiplies in the same order as before, so every App golden number is the
  same float. The App result's `financial` rows gain each year's component
  cashflows, escalated and not discounted: `cost_import`, `revenue_export`,
  `cost_operation`, `cost_fixed_charge`, `cost_replacement` and
  `replacement_time_years`.
- `App.result()` reports the year-1 money components as top-level keys, at
  year-1 prices (before escalation and discounting):
  `grid_import_cost_year1_prices`, `grid_export_revenue_year1_prices`,
  `fixed_charge_year1_prices` and `no_system_import_cost_year1_prices` (the
  no-system household's import cost; its year-1 bill adds the same fixed
  charge). Flat and tariff runs report all four. With smart charging,
  `grid_charge_cost_year1_prices` is the part of the grid import cost bought
  to charge the battery; it is already included in
  `grid_import_cost_year1_prices`. The `breos sweep` CSV copies top-level
  scalars, so it now carries these columns
  ([#181](https://github.com/Str4vinci/breos/issues/181),
  [#183](https://github.com/Str4vinci/breos/issues/183)). The result schema
  version becomes `"1.1"`, since an added field bumps the minor. No reported
  number changes.
- **Tariff domain**, `breos.tariffs` (ADR 0002). A `TariffSchedule` assigns
  instants to named periods in local civil time and records its regulatory
  source; `TariffPrices` holds per-kWh import and export prices per period and
  a daily fixed charge in one currency (EUR in 0.7.0); a `ResolvedTariff`
  aligns both to a simulation index, with period labels and codes, price
  arrays, the civil-day boundaries (`day_starts`: 23-, 24- and 25-hour days),
  and separate schedule and price hashes. Periods are classified in the
  timezone passed in, never the index's own (A1), and a schedule is not moved
  to another zone. Nine schedules are bundled, checked against their primary
  sources: the Portuguese mainland BTN daily and weekly bi- and tri-hourly
  cycles of 2026 (Diretiva ERSE n.º 1/2026) and of the 2027 reform (Diretiva
  ERSE n.º 3/2026, de 19 de agosto), and the Spanish 2.0TD access tariff (CNMC
  Circular 3/2020) with its 2026 national holidays. A schedule whose
  boundaries hourly input cannot represent is rejected at that resolution
  (A3). Nothing in App, Monte Carlo or the optimizer uses tariffs yet, so no
  result changes; the `[tariff]` config table comes with TOU valuation.
- **Time-of-use valuation** through a `[tariff]` App table (ADR 0002): a
  bundled `schedule`, a `currency` (EUR), per-period `import_prices` and
  `export_prices` (or an `all` price), an optional `fixed_charge_per_day`,
  `boundary_policy = "strict"` and, for a 2027 schedule on an earlier year, a
  `study_date`. It is checked when App is built: unknown schedules,
  currencies and periods, unpriced periods, a schedule from another timezone,
  a resolution too coarse for the schedule's boundaries, and flat
  `costs.electricity_cost`, `electricity_sold_cost` or `daily_power_cost`
  set as well all raise. The tariff is resolved once on the simulated
  calendar and every project year replays it (A2). The shared projection
  loop prices each year as step energy times step price, from per-step
  frames (App) or weighted summary sums (Monte Carlo, through the new
  `weights` argument of `simulate_energy_balance_summary`), into the year
  rows' `Import_Cost`, `Export_Revenue`, `Baseline_Import_Cost` and
  `Fixed_Charge`. Dispatch does not change. `result()["provenance"]["tariff"]`
  and the Monte Carlo provenance record the schedule and its source, the
  prices, both hashes, the timezone and `calendar_policy =
  "replay_start_year"`. Projected optimization does not read a tariff yet.
  Flat-price runs are unchanged; their `resolved_config` gains `tariff: null`.
  New example: `configs/examples/time-of-use-portugal.toml`.
- **The `[smart_charging]` table: fixed-target grid charging** (ADR 0002,
  [#178](https://github.com/Str4vinci/breos/issues/178)). `mode = "fixed_target"` takes a
  `target_usable_fraction` of the usable SOC window, `charge_periods` and
  `discharge_periods` named from the tariff's schedule, a
  `grid_charge_efficiency` with no default (A6) and an optional
  `grid_import_limit_w`. It is checked when App is built: a missing
  `[tariff]` or battery, an unknown key or period, charge and discharge
  periods that overlap (A8) and out-of-range values all raise, naming the
  dotted key. The fixed-target controller, `breos.smart_charging`, turns the
  table and the resolved tariff into `DispatchInstructions`: in a charge
  period the grid may charge toward the target, in a discharge period the
  battery may discharge, and in neither it does neither. App, Monte Carlo and
  both optimizer entry points resolve the instructions once and apply them in
  every project year. `mode = "disabled"` runs with the same results as
  omitting the table. Results gain a `smart_charging` block: per year, the
  grid-charge energy, its conversion loss, its cost at year-1 prices, and
  battery delivery split by origin, plus the stored energy by origin at the
  start and end of the project. `provenance.smart_charging` (App, Monte Carlo,
  optimizer) records the parameters, the instruction hash, the schedule hash
  and the terminal convention, `physical_carry`. Tariff year rows gain
  `Grid_Charge_Cost`, the part of `Import_Cost` that charged the battery.
  Year rows gain `Grid_Origin_Battery_AC_Load_kWh`. Results'
  `resolved_config` gains `smart_charging: null`. New example:
  `configs/examples/smart-charging-portugal.toml`. On an example 5 kWh
  hourly system under the bi-hourly tariff (0.28/0.11 €/kWh, 25 years), a
  50% off-peak target raises NPV savings from 1,906 € to 2,346 € and imports
  from 1,431 to 1,823 kWh in year 1; 387 kWh of that is grid charge.
- **Separate escalators** (ADR 0003 E2): `import_price_escalation` (import
  energy and the fixed charge), `om_escalation` (O&M) and
  `replacement_cost_learning`, as App config keys and CLI flags, as
  `cost_analysis_projection` arguments and in the optimizer's `financials`
  table. The two escalators inherit `inflation_rate` when unset, which becomes
  general inflation, and learning defaults to 0, so a run that sets none of
  them prices exactly as before: the App golden baseline is bit-identical. A
  replacement at `t` years costs `C0 × (1 + inflation_rate)^t × (1 −
  learning)^t`. `result()["provenance"]["economics"]`, the Monte Carlo
  provenance and the optimizer's provenance record the rates used and the
  implied real discount rate (E1); the result schema version becomes `"1.2"`.
  The optimizer checks its `financials` rates before the search starts: a
  rate at or below −1, or a learning rate outside [0, 1), raises. `breos sweep` treats the
  three keys as input-independent, so sweeping them reuses the prepared
  inputs. The docs state the timing conventions (E3).
- **Plots of sweep and optimizer output**
  ([#186](https://github.com/Str4vinci/breos/issues/186)), in
  `breos.plotting` and the top-level `breos` namespace. They read the CSV
  `breos sweep` writes, or its DataFrame, and take a swept key as in the
  config (`n_modules`) or as its column (`param_n_modules`); both name the
  swept `param_` column, not the result column of the same name. A misspelt
  key or metric raises `ValueError` and lists the table's columns:
  - `plot_sweep_heatmap(sweep, metric, results_directory, x=None, y=None,
    diff=None, labels=None, ..., currency=None, filename=None)` draws one
    result column of a two-parameter sweep. With `diff=`, a second sweep over
    the same grid, it draws the difference, for example the same sizing grid
    at two locations; a difference of a percentage is labelled in percentage
    points. A difference, and a metric with a negative value such as a loss
    in `npv_savings`, use a diverging colour scale centred on zero, unless
    `vmin` or `vmax` is given. `labels` without `diff` raises `ValueError`.
  - `plot_orientation_landscape(sweep, metric, results_directory, tilt="tilt",
    azimuth="azimuth", maximize=True, ..., currency=None, ...)` maps a tilt ×
    azimuth sweep, marks the best orientation and draws the east-west profile
    at the best tilt. Azimuth ticks carry compass points, negative
    (southern-hemisphere) azimuths included. A sweep of tilt alone, such as
    an east-west roof, gets the tilt profile.
  - `plot_pareto_front(designs, results_directory, x="Grid_Independence_%",
    y="NPV", maximize=(True, True), color_by=None, currency=None, ...)` draws
    two objectives of the `OptimizationResult` of
    `optimize_system_multi_objective`, its `details["pareto"]` frame, or any
    table of designs, and marks the designs that no other one beats in both.

  A sweep CSV does not record its currency: `currency=` names it for the
  money labels. Without it, and without `attrs["currency"]` on a DataFrame,
  the labels name no currency ("NPV savings") rather than assume EUR.
- Two validation oracles for smart charging, under `tools/oracles/`. They
  are tools, not public API, and no result changes. Each replays its
  schedule through the production App run with `tools/oracles/replay.py`,
  prices it with the tariff, and compares it with App's own dispatch of the
  same configuration. Each has a command line
  (`python tools/oracles/<name>.py --config <file> --output <json> --csv <csv>`)
  that writes a JSON summary and a CSV, both tagged with a schema.
  - `tools/oracles/daily_target_dp.py` plans the first project year one
    grid-charge target per civil day, with perfect foresight, with the
    private daily-target dynamic program. It needs a fixed-target
    `[smart_charging]` table, whose layout it keeps. It reports the plan,
    the flows the plan expects at its fixed health against what production
    delivers, and the replayed cost next to the fixed-target run
    (schema `breos_daily_target_dp_oracle_v1`).
  - `tools/oracles/lp_bound.py` bounds the first project year's import cost
    less export revenue from below with a perfect-foresight linear program,
    solved with HiGHS through `scipy.optimize.linprog`. The program contains
    every flow the production dispatch can deliver under any instructions,
    as long as the battery's health stays at or above the program's floor
    health, no battery is replaced in the year, and grid charge passes
    through the program's converter: the `[smart_charging]` grid-charge
    efficiency and site limit, or no grid charge without that table. The
    bound holds for any such dispatch. It keeps the prices, the efficiencies, the power limits,
    the inverter rating and the temperature-dependent ceiling. It relaxes
    the rest: opening health for the ceiling and the efficiencies, a concave
    bound on the inverter's part-load curve, standby loss charged in part,
    the dispatch order, and a free end state. The module docstring lists
    each relaxation. `--floor-soh auto`, the default, solves once, replays
    App's dispatch and the program's own schedule, and solves again with
    the floor at the lowest health those replays reached. Each reported run
    is then checked: it counts as covered only if its ledger is a feasible
    point of the program that costs no more than the run did.
    `bound_is_strict` in the report, and a warning, say when a reported run
    is not covered, for example after a first-year replacement or a grid
    charge the program does not allow. The optimum at the opening health is reported
    only as `fixed_health_estimate`, which is not a bound. `--floor-soh`
    also takes `opening` or a fraction (schema `breos_lp_bound_v1`).
- `tools/benchmark_optimization.py` times the public
  `optimize_system_multi_objective` on the Python and Numba backends. It is
  a tool, not public API, and no result changes. The study is a
  three-year, fixed-target time-of-use search on the PT 2026 daily
  two-period tariff, hourly and at 15 minutes. Its weather is the local
  Porto TMY, read through `load_weather`, and its load is the bundled
  demandlib H0 profile. `--weather-file` takes one complete year of other
  weather. A weather metadata sidecar that does not match its file, or has
  another schema version, stops the run, since its timing fields reach the
  resampler and the PV model. Every case's inputs are checked before the
  first run. Before any timing, both backends must give the same Pareto
  designs, pymoo optimum, objectives, diagnostics and evaluation and
  generation counts at the same seed. A fixed 8-module, 5 kWh design must also give the same
  `evaluate_projected_design` tables and the same `App` step ledgers and
  degradation state on both, and that design must charge from the grid and
  replace its battery. Each measurement runs in a fresh process: a cold run
  per backend, with an empty `NUMBA_CACHE_DIR` for Numba, and at least
  three warm runs (`--warm-repeats`), each after an untimed warm-up at the
  study size. With one worker, a timed run whose counts differ from the
  parity run fails its case. `--smoke` allows fewer warm runs and marks the
  report. The report gives total time, the `SolarDesignProblem`
  construction time, the residual search and wrapper time, peak RSS (the
  child's own `VmHWM` on Linux, `ru_maxrss` elsewhere), and the speedup
  over the Python warm median. `--output` writes it as JSON (schema
  `breos_optimization_benchmark_v1`), also when a run fails part way.

### Changed
- Result schema 2.0 renames `monthly[].import_kwh` and
  `yearly[].import_kwh` to `grid_import_kwh`, with paired `export_kwh`
  fields renamed to `grid_export_kwh`; values are unchanged. Optimizer columns
  `Projected_Breakeven_Year` and `Projected_Breakeven_Year_Interpolated` are
  now `Projected_Payback_Year` and `Projected_Payback_Year_Interpolated`.
- The schema 2.0 migration guide maps removed CO2 aliases to the existing
  total-year and total-lifetime fields. Enum spelling in echoed provenance
  remains unchanged.
- **`plot_breakeven_comparison` reads App results**
  ([#186](https://github.com/Str4vinci/breos/issues/186)). It takes
  `App.result()` dicts, whose `financial` rows it reads, or cost projection
  frames: `plot_breakeven_comparison(projections, labels, results_directory,
  colors=None, currency=None, filename=...)`. To migrate, the first
  argument `cost_dfs` is now `projections`, `colors` moves after the
  directory and defaults to the colour cycle, and `results_dir` is now
  `results_directory`, as in the other plots. Each curve now starts at year 0
  with the investment, and the no-system baseline at 0, so the dotted payback
  line meets the curves where they cross; before, the curves began at year 1.
  Each payback line is labelled with its year. A baseline that every
  scenario shares is drawn once, in black, as "No system"; otherwise each
  group of scenarios that share one gets "No system (<labels>)". A
  projection read back from CSV records no currency: it takes the currency
  of the other projections, or `currency=`, or the axis shows no currency
  code. Labels or colours that do not match the projections, projections
  that record two currencies, a `currency=` that contradicts a recorded one,
  and a `[period]` result without `financial` rows raise `ValueError`. The
  payback line was already the shared `find_payback_year_interpolated` rule.
- **The TMY-versus-historical weather plots compute their own statistics**
  ([#186](https://github.com/Str4vinci/breos/issues/186)).
  `plot_weather_monthly_comparison(tmy, historical, results_directory,
  variable="ghi", tmy_label="TMY", filename=None)` and
  `plot_weather_annual_ghi_distribution(tmy, historical, results_directory,
  tmy_label="TMY", filename=...)` take a TMY weather frame and the historical
  weather a Monte Carlo study samples: its `weather_file` path, or the
  per-year frames of `preload_weather_by_year`. They read the same complete
  years as the study, and Open-Meteo column names. Before, they took monthly
  arrays and a statistics table that the caller had to build.
  `plot_weather_monthly_comparison` also draws `dni`, `dhi` and `temp_air`.
  To migrate `plot_weather_monthly_comparison`: `tmy_vals` and `stats` are
  replaced by `tmy` and `historical`; `ylabel` is gone, as the label comes
  from `variable`; `tmy_source` is now `tmy_label`, and the TMY legend entry
  is its text as given, not "TMY (<source>)"; `results_dir` is now
  `results_directory`; and `filename` is optional, defaulting to
  `weather_monthly_<variable>.png`. Its "Min year" and "Max year" legend
  entries are now "Monthly minimum" and "Monthly maximum": each month's
  lowest and highest value over the years, which can come from different
  years. To migrate `plot_weather_annual_ghi_distribution`:
  `annual_ghi_per_year`, `tmy_annual_ghi` and `hist_annual_ghi_mean` are
  replaced by `tmy` and `historical`, and `results_dir` is now
  `results_directory`. Missing weather values warn, and the monthly figures
  skip them; the annual distribution raises `ValueError` for a TMY or year
  without GHI for a whole month.
- `breos validate-config --json` and `breos run --dry-run` build their
  resolved-config summary from the config registry: each `AppConfigField`
  names its place (`summary = "section.key"`), so every App key is reported
  ([#181](https://github.com/Str4vinci/breos/issues/181)). Every field the
  summary reported before keeps its section, name and value; key order
  within a section now follows the registry. The summary gains the keys it
  left out: `pv.tracking`, `pv.axis_tilt`, `pv.axis_azimuth` (resolved from
  the latitude when unset), `pv.max_angle`, `pv.backtrack`,
  `pv.cross_axis_tilt`, `pv.dual_axis_max_tilt`, `pv.horizon_profile`,
  `pv.degradation_rate`, `load.load_profile_column`,
  `load.load_profile_unit`, `battery.calendar_model`,
  `battery.enable_resistance_fade`, `battery.temperature`,
  `battery.indoor_model`, `battery.smart_charging`, `economics.costs`,
  `economics.tariff`, and a new `simulation` section with `weather_source`
  and `execution_backend`. A TOML date in `[tariff]` is written as text.
- LCOE and lifetime CO2 are computed once per run, in the cost projection
  ([#183](https://github.com/Str4vinci/breos/issues/183)). App, Monte Carlo
  and the optimizer read `attrs["lcoe_per_kwh"]` instead of calling
  `calculate_lcoe_from_projection` again, which gave the same float. The App's
  CO2 fields read the projection's CO2 columns instead of looping
  `calculate_co2_savings` over the years. The projection counts
  self-consumption as production minus export, the loop used
  `Self_Consumption_kWh`, so the unrounded values differ by up to 4e-14 of
  their size. Over 69 configurations (flat, TOU and smart charging, three
  countries) no reported CO2 field changed at its two decimals.
- The economics prices battery replacements (ADR 0003 E4,
  [#183](https://github.com/Str4vinci/breos/issues/183)). The simulation
  reports each swap and its capacity; year rows carry `Replacements` and the
  new `Replaced_Capacity_kWh`. `price_year_rows` adds `Replacement_Cost` from
  `costs["replacement_cost_each"]`, which `calculate_costs` now returns and
  the new `breos.economics.replacement_event_cost` computes: the storage cost
  per kWh times the capacity, or the optimizer's `battery.replacement_cost`.
  App, Monte Carlo and the optimizer price replacements through this one
  function, so a replacement price, learning rate or revaluation no longer
  needs a re-simulation. Frames and year rows that already carry
  `Replacement_Cost` (ledger schema < 3.0) keep the stored money. Every
  number App, Monte Carlo and the optimizer report is unchanged: the App
  golden baseline and the dispatch parity harness match bit for bit, and the
  t = 0 totals are still added year by year, so they do not depend on the
  Python version. Year tables keep `Replacement_Cost` right after
  `Replacements`; `Replaced_Capacity_kWh` follows it, so the columns after it
  move one place. The optimizer's year tables also gain the year-1-price
  money columns. Result schema 1.3.

  Direct callers of the economics see one change: a hand-built `costs`
  dict without `replacement_cost_each` raises once the run has a
  replacement, instead of pricing it from the simulation. A `Replacements`
  count must be a whole number; a missing one counts as none.
- `breos sweep` prepares weather, PV, load and battery temperature once per
  distinct input configuration and reuses them across the runs that differ
  only in settings the input stage never reads, such as a tariff, a battery
  size or a price ([#181](https://github.com/Str4vinci/breos/issues/181)).
  `breos.app_inputs.INPUT_INDEPENDENT_KEYS` lists those settings, and a test
  changes each one and checks that the prepared inputs stay the same. Each run
  gets its own copy. A 24-run tariff and battery sweep on live PVGIS weather
  took 26 s instead of 59 s, with 2 weather fetches instead of 24, and wrote
  the same CSV bytes. The sweep runs its grid grouped by input
  configuration and writes the rows in grid order, and the cache holds one
  preparation at a time, so memory stays at one run's inputs. The cache lives
  only for one sweep; `App` itself still prepares its inputs afresh. No
  reported number changes.
- `breos sweep` takes its dotted keys from the config registry
  ([#181](https://github.com/Str4vinci/breos/issues/181)). It accepts any key
  of `[costs]`, `[battery_indoor_model]`, `[tariff]` and `[smart_charging]`,
  and a period of a tariff's price map (`"tariff.import_prices.off_peak"`).
  Before, only `costs.*` was accepted. Every combination is resolved before
  the first run, so an invalid one fails at once instead of after the runs
  before it, and `breos validate-config` checks every combination of a
  `[sweep]`. `breos run` merges its flags into the config file table by
  table, so a flag that sets one key of a table keeps the file's other keys
  (no flag sets a table key yet). No reported number changes.
- **Currency-neutral result names and result schema 1.0** (ADR 0003 E8 and
  E9, [#183](https://github.com/Str4vinci/breos/issues/183)). Money keys drop
  the currency, and the fractional payback is "interpolated" rather than
  "exact", since it is a straight line between year-end points. There are no
  aliases: scripts that read the old names break once and move to the table
  below. Every reported number is unchanged. A removed output key is absent.
  `constraints.budget_eur` raises a `ValueError` that names `budget`, rather
  than falling back to the default budget.

  | Surface | Old | New |
  |---|---|---|
  | `App.result()`, `breos sweep` CSV | `total_investment_eur` | `total_investment` |
  | `App.result()`, `breos sweep` CSV | `npv_savings_eur` | `npv_savings` |
  | `App.result()`, `breos sweep` CSV | `lcoe_eur_kwh` | `lcoe_per_kwh` |
  | `App.result()`, `breos sweep` CSV | `battery_replacement_cost_eur` | `battery_replacement_cost_t0_prices` |
  | `cost_analysis_projection` `attrs` | `lcoe_eur_kwh` | `lcoe_per_kwh` |
  | Monte Carlo `runs` column, `summary` key | `npv_savings_eur` | `npv_savings` |
  | Monte Carlo `runs` column, `summary` key | `lcoe_eur_kwh` | `lcoe_per_kwh` |
  | Monte Carlo `runs` column | `total_replacement_cost_eur` | `total_replacement_cost_t0_prices` |
  | Monte Carlo `runs` column, `summary` key | `payback_year_exact` | `payback_year_interpolated` |
  | Optimization Pareto column | `NPV_Eur` | `NPV` |
  | Optimization Pareto column | `Objective_NPV_Eur` | `Objective_NPV` |
  | Optimization Pareto column, `objective_names` | `Projected_NPV_Eur` | `Projected_NPV` |
  | Optimization Pareto column | `Projected_Initial_Cost_Eur` | `Projected_Initial_Cost` |
  | Optimization Pareto column | `Projected_Replacement_Cost_Eur` | `Projected_Replacement_Cost_T0_Prices` |
  | Optimization Pareto column | `Projected_LCOE_Eur_kWh` | `Projected_LCOE_per_kWh` |
  | Optimization Pareto column | `Projected_Breakeven_Year_Exact` | `Projected_Breakeven_Year_Interpolated` |
  | Optimization config, `constraints` | `budget_eur` | `budget` |
  | `breos.economics` function | `find_payback_year_exact` | `find_payback_year_interpolated` |
  | `breos list cost-presets --json` | `electricity_cost_eur_kwh` | `electricity_cost_per_kwh` |
  | `breos list cost-presets --json` | `export_price_eur_kwh` | `export_price_per_kwh` |
  | `breos list cost-presets --json` | `storage_cost_eur_kwh` | `storage_cost_per_kwh` |

  The ADR's table also lists `SteadyState_NPV_Eur` and
  `replacement_cost_eur_each`; both went earlier in this release with the
  steady-state objective basis. It lists two `plot_tariff_comparison` input
  columns and a `plot_pareto_front_analysis` one as well; both functions
  were removed in this release (see Removed). It also lists three
  `breos.io` summary labels, which went with `_economics_summary_metrics`
  (see Removed).
  `_t0_prices` marks a total at t = 0 prices, neither inflated nor
  discounted. `App.result()` gains
  `battery_replacement_cost_npv` beside it: the same replacements inflated to
  and discounted from each swap instant, as `npv_savings` counts them.

  The currency is recorded once per result: `provenance["currency"]` in
  `App.result()`, Monte Carlo and optimizer provenance (`details["provenance"]`
  and `evaluate_projected_design(...).provenance`, now present on flat prices
  too), and `attrs["currency"]` on cost projections, Monte Carlo `runs` and
  the Pareto frame. It is the tariff's `currency`, or `EUR`, the bundled
  catalogue's, without a tariff; `breos.tariffs.result_currency` resolves it.
  `cost_analysis_projection` takes `currency=` and stamps it. The summary
  labels above, plot axis and value labels, and `breos list cost-presets`
  (which gains a `currency` field) read it, so an EUR run writes the ISO code
  where plots used to write `€`. BREOS does not convert currencies.

  `App.result()`, Monte Carlo provenance and `--json` output, and optimizer
  provenance carry `result_schema_version`
  (`breos.result_schema.RESULT_SCHEMA_VERSION`), independent of the ledger
  schema: `"1.0"` for these names, `"1.1"` with the year-1 money keys above,
  `"1.2"` in 0.7.0 with `provenance.economics`. A renamed or removed field bumps the major version, an added
  field the minor. A result without it predates these names.
- **Avoided emissions use net exchange** (ADR 0002 A10,
  [#178](https://github.com/Str4vinci/breos/issues/178)). The self-consumed
  credit is `(Load − Import − B_u) × CI`, where `B_u` is unattributed battery
  energy delivered to load. Grid energy shifted through the battery is
  imported, so it earns nothing, and its round-trip loss counts against the
  system. `calculate_co2_projection` gains `yearly_grid_shift_kwh`: grid-origin
  battery delivery minus grid-charge import, zero or negative. Without grid
  charging the shift is zero and every emissions result is unchanged, bit for
  bit. In the smart-charging example above, year-1 avoided CO2 falls from
  479.4 to 476.1 kg.
- **Ledger schema 2.0: stored energy has three origins**
  ([#178](https://github.com/Str4vinci/breos/issues/178), ADR 0002 A8 and A9).
  Stored energy is split into PV, grid and an unattributed remainder. A fresh
  pack's initial energy and a replacement pack's energy are unattributed.
  Discharge, standby loss, capacity-window loss and replacement take from all
  three origins in proportion to their shares at the start of each operation.
  The results frame gains the grid-origin balances
  (`Battery_Grid_Origin_Energy_Beginning`/`_End`) and, for PV and grid, the
  per-origin columns `*_Origin_Battery_Discharge_DC`,
  `*_Origin_Battery_AC_To_Load`, `*_Origin_Standby_Loss`,
  `*_Origin_Capacity_Window_Loss` and `*_Origin_Replacement_Energy_Removed`,
  plus `PV_Origin_Battery_Charge_Stored`. Each origin now reconciles step by
  step from the frame alone. The unattributed share of a flow is its total
  minus the PV and grid shares. Nothing charges the grid origin yet: grid
  charging comes later in 0.7.0. It can already hold energy carried in
  through the new `initial_grid_origin_energy_wh` argument of
  `simulate_energy_balance` and `simulate_energy_balance_summary`. The App,
  Monte Carlo and optimizer year loops carry it from year to year, the year
  rows report `Battery_Carried_Grid_Origin_Energy_Wh`, and
  `SimulationSummary` gains `opening_grid_origin_energy_wh` and
  `carried_grid_origin_energy_wh`. When a BLAST run restores its SOH from a
  carried state without a carried energy, the carried origins are now checked
  against the energy the run starts with, which could open the unattributed
  origin below zero before. The dispatch step raises if a step both
  charges and discharges the battery, since one origin share per step relies
  on that. `ledger_schema_version` is now `2.0`. No reported number changes
  on either backend: the only App golden fields that move are the two
  `ledger_schema_version` strings. A year's dispatch is about 14% slower on
  the Python backend and about 3% slower on Numba.
- The per-step ledger is laid out in one place
  ([#178](https://github.com/Str4vinci/breos/issues/178)). Each buffer-matrix
  row has one name, its results-frame column, from the day loop through the
  buffers to the frame. The snake-case row aliases and the rename step
  between them are gone. The result buffers expose their columns as one
  read-only mapping, so a misspelt column raises instead of being created
  silently. `LEDGER_SCHEMA_VERSION` moves from `breos.runners.app` to sit
  beside the column tuples in `breos._dispatch`, and is still importable from
  `breos.battery`. `SimulationSummary.ledger_schema_version` and Monte Carlo
  `provenance.ledger_schema_version` now report it too, as App provenance
  already did. No reported number changes, and the results frame keeps its
  columns and their order.
- Cut CI runner time without dropping a check. Merges into `develop` no
  longer re-run the workflow, since branch protection already requires each PR
  to be tested up to date with `develop`. The macOS/Windows smoke suite runs
  nightly and at the release gates instead of on every PR commit. The
  full-suite jobs, including the coverage report, run under pytest-xdist
  (`-n auto`), now in the `dev` extra. A new push to a PR cancels that PR's
  run still in progress.
- **`PVModuleParams.gamma_pmp` now holds only what the user supplied.** It
  stays `None` when the power coefficient is left to default, instead of
  being overwritten with `T_Pmax_pct` on construction. Read the coefficient
  the models use from the new read-only `gamma_pmp_effective`, which is
  `gamma_pmp` when set and `T_Pmax_pct` otherwise. Code that reads
  `module.gamma_pmp` from a catalogue module, for example to pass it to
  `fit_cec_params`, needs to switch to `gamma_pmp_effective`. Setting
  `gamma_pmp` explicitly behaves as before. `alpha_sc` and `beta_voc` are now
  read-only properties; assign `alpha_sc_abs` or `beta_voc_abs` to override
  them.
- One frequency check, `breos.utils.normalise_frequency`, now serves the step
  helpers, the weather readers, `load_profile` and the CLI
  ([#175](https://github.com/Str4vinci/breos/issues/175)). It accepts `"h"`,
  `"1h"` and `"15min"`, returns `"h"` or `"15min"`, and raises `ValueError`
  naming those spellings for anything else. **The aliases `"H"`, `"1H"`,
  `"15T"` and `"15m"` are no longer accepted.** pandas 3 rejects the first
  three, and `"m"` is not a minute alias, so each one used to pass the step
  helpers and then fail later in `pd.date_range`. Use `"h"` or `"15min"`.
- E-REDES columns are read as what they are, Wh per interval, and converted
  to W, instead of being relabelled W
  ([#182](https://github.com/Str4vinci/breos/issues/182)). Every profile is
  scaled to `annual_consumption_kwh`, so results do not change.
- **`bdew_h0` now loads the BDEW H0 publication** (external file
  `bdew_h0_*_15min.csv`, formerly key `"7"`). It used to be an alias of the
  bundled demandlib profile. A config that set `load_profile = "bdew_h0"`
  without `rlp_directory` now raises; use `demandlib_h0` for the bundled
  profile, which is the same standard shape.
- The App golden baseline gains the `provenance.load_profile` block and the
  three new `resolved_config` keys, and `resolved_config.load_profile` reads
  `demandlib_h0` instead of `"1"`. No numbers change.
- App and Monte Carlo build the battery a projection year runs through one
  function, `breos.projection.build_battery_config`, and the PV-only
  configuration through `build_pv_only_battery_config`, instead of two copies
  each ([#179](https://github.com/Str4vinci/breos/issues/179)). The inverter
  AC nameplate is sized by one rule, `breos.inverter.inverter_ac_capacity_w`
  (DC peak over the loading ratio; none when the ratio is not positive),
  resolved once as `ResolvedAppConfig.inverter_ac_capacity_w` and read by App,
  Monte Carlo, the PV loss waterfall, `breos validate-config` and both
  optimizer paths. Results are unchanged. The optimizer's two paths no longer
  raise `TypeError` when `dc_ac_ratio` is `None`; they run without AC
  clipping, as App does.
- App and Monte Carlo run one multi-year projection loop,
  `breos.projection.run_projection`
  ([#179](https://github.com/Str4vinci/breos/issues/179)). The battery state
  a year hands the next is one `CarryState` (stored energy, PV-origin energy,
  throughput, calendar time, degradation, SOH, resistance, and the engine's
  native degradation payload), and every year row has one schema, built by
  `build_year_row` from column sums that are the same floats whether the year
  ran with per-step frames (App) or as a summary (Monte Carlo). Both price the
  rows through `value_projection`. App's `SimulationArtifacts.yearly_df`
  gains the columns Monte Carlo rows already had (cumulative battery state,
  loss diagnostics, carried energy, replacement steps); `App.result()` is
  unchanged, and so are all numbers: the App golden baseline matches bit for
  bit.
- Projected optimization (`evaluate_projected_design` and the NSGA-II
  projected scoring) runs the same projection loop,
  `breos.projection.project_years`, with its own battery and initial SOH
  ([#179](https://github.com/Str4vinci/breos/issues/179)).
  `_projected_year_summary` is gone, so the optimizer's yearly table
  (`ProjectedDesignResult.yearly`) has the shared year-row schema:
  **`PV_DC_kWh` is now `PV_DC_Generation_kWh` and `PV_DC_Curtailed_kWh` is
  now `Curtailment_DC_kWh`**, a PV-only design reports `Battery_SOH_%` as
  empty rather than 100, and the table gains the App and Monte Carlo columns.
  App and Monte Carlo rows gain the optimizer's `Inverter_Loss_kWh`,
  `Battery_Charge_Throughput_kWh`, `Battery_Discharge_Throughput_kWh`, the two
  SOC means and `Battery_Annual_FEC`, and `SimulationSummary` gains
  `fec_all_packs`. Optimizer results move by at most one unit in the last
  place (2e-16 relative in LCOE and the ZEB ratio), because delivered PV is
  now summed per column rather than per step; App and Monte Carlo results
  are unchanged.
- Nested config tables are checked by one schema, `breos.config_schema.TableSpec`
  ([#181](https://github.com/Str4vinci/breos/issues/181)): the keys a table
  allows, a checker per key, the keys it requires, and a hook for rules that
  span keys. `costs`, `battery_indoor_model` and each `pv_arrays` entry use
  it now, and the 0.7 `[tariff]` and `[smart_charging]` tables will. Every
  table reports an unknown key the same way, as `Unknown key
  'battery_indoor_model.setpoint'. Available: ...`; the indoor model and
  `pv_arrays` messages used to differ, and a non-boolean
  `battery_indoor_model.enabled` now says "must be true or false". Valid
  configs behave exactly as before.
- The daily fixed charge is billed on the simulated duration,
  `Simulated_Hours / 24` days, instead of 365 days (ADR 0003 E5). A common
  year is exactly 365 days, so its results do not change. A leap-year run is
  now billed 366 days: for a Porto run starting in 2024 (25 years,
  `residential_pt`, 0.30 €/day) the discounted cost with and without the
  system both rise by 6.49 €, and the NPV of savings is unchanged because the
  charge is paid either way. Year rows without `Simulated_Hours`, from direct
  callers, are billed as 365-day years, as before.
- **One default discount and inflation rate everywhere** (ADR 0003 E6): a
  discount rate of 0.03 and an inflation rate of 0.02, defined once as
  `breos.economics.DEFAULT_DISCOUNT_RATE` and `DEFAULT_INFLATION_RATE` and
  read by the App registry, `CostParams`, `cost_params_from_config`,
  optimization, `cost_analysis_projection` and
  `calculate_lcoe_from_projection`. **Callers that omit the discount rate get
  different results:** `CostParams`, `cost_params_from_config`, the optimizer
  and `calculate_lcoe_from_projection` used 0.0, and a direct
  `cost_analysis_projection` call used 0.02 (with inflation 0.03). On three projected-optimizer designs without a
  `financials.discount_rate`, NPV moves from 6430.77 to 3941.12 €, −1736.96 to
  −2690.35 € and −12091.27 to −11013.42 €, and LCOE rises by 0.017–0.021
  €/kWh; energy and battery results are unchanged. App results do not
  change, since App already used 0.03 and 0.02. An explicit 0.0 is used as
  given.
- The DST-day tariff test builds each civil day up to the next local
  midnight, so it passes on pandas 2.x too, where `pd.offsets.Day` is a fixed
  24 hours; the `floors` CI job failed on it. Test only.
- The greedy dispatch step is written once for both execution backends
  ([#177](https://github.com/Str4vinci/breos/issues/177)). The step, the day
  loop, the capacity window and the ledger layout live in `breos._dispatch`
  as plain scalar code; the Python backend calls it and the Numba backend
  compiles the same functions, instead of a hand-kept copy in
  `breos._numba_dispatch_kernels`. The inverter conversion has one scalar
  core behind `calculate_dc_ac_power` and `dc_power_for_ac_output`, and the
  unused no-battery branch of the day loop is gone (PV-only runs take the
  vectorised path). `lfp_capacity_factor` and `compute_cell_temperature`
  stay importable from `breos.battery`. Results are unchanged on both
  backends: the App golden baseline and Monte Carlo runs match develop bit
  for bit. A warm hourly battery year takes about 30% less time on the
  Python backend (80 to 56 ms on the parity harness) and about 15% less on
  Numba (13.0 to 11.1 ms), the latter because the aging step now reads
  timestamps from one precomputed tick array instead of slicing the
  `DatetimeIndex` every day.
- The energy loop keeps the only running totals of cycle and calendar
  degradation ([#186](https://github.com/Str4vinci/breos/issues/186)). The
  native and BLAST degradation adapters kept a second copy for the carried
  degradation state; they now report only each period's increment, and the
  state takes `cumulative_cycle_degradation` and
  `cumulative_calendar_degradation` from the energy loop, as it already took
  `resistance_growth`. The internal adapters lose their
  `initial_cumulative_*_degradation` arguments. The unused running totals of
  cycle and calendar resistance growth are gone
  ([#164](https://github.com/Str4vinci/breos/issues/164));
  `update_battery_resistance_cyclewise` and
  `update_battery_resistance_calendar` still return the increment. Results
  and carried states are unchanged bit for bit.
- `DEFAULTS` lists the App keys in config-registry order, and
  `AppConfigField.default_order` is gone
  ([#164](https://github.com/Str4vinci/breos/issues/164)). Nothing read the
  order: the input cache key sorts its keys, and configs are compared by
  value. The only visible effect is the key order of
  `provenance.resolved_config` in App and Monte Carlo results; every key and
  value is unchanged.
- `dc_to_ac`, and with it `calculate_pv_production_ac`, converts the whole
  series in one vectorised pass through the inverter curve the PV-only
  dispatch uses, instead of calling `calculate_dc_ac_power` once per timestep
  ([#186](https://github.com/Str4vinci/breos/issues/186)). A 15-minute year
  takes 0.2 ms instead of 46 ms. Missing DC still converts to 0 W, now also
  `None` in an object series, which raised `TypeError`, and an empty object
  series now returns a `float64` series. The scalar path squares the load
  ratio through libm `pow`, so on about 7 in a million random part-load inputs
  the two differ by 1 or 2 ULP (a relative change below 5e-16); on 525,600
  steps of synthetic Porto PV years, hourly and 15-minute, none differs. App,
  Monte Carlo and the optimizer do not call `dc_to_ac` and are unchanged bit
  for bit.
- Internal PV, weather and load cleanups from a code audit. The fixed-tilt
  and tracking breakdowns share one builder, and App passes the PV model
  options through `configured_pv_model_kwargs`, as the optimizer does. App
  and Monte Carlo decide whether to resample hourly weather to 15 minutes in
  one helper. Transposition and terrain shading compute the sun position in
  one function. `PVModuleParams` is defined in `breos.pv_modules`, so
  `breos.solar` imports the module catalog at load time instead of inside
  four functions; `breos.solar.PVModuleParams` and `breos.PVModuleParams`
  still work. The weather metadata key, the horizon-status defaults, the
  sidecar reading and the timestamp and air-temperature column names are
  each defined once. The unused `read_weather_csv`,
  `resolve_configured_pv_model_options` and `solar._get_column` are gone.
  Two edge cases change. App now takes the weather step from the first two
  rows when pandas cannot infer it, as Monte Carlo already did: hourly
  weather with fewer than 3 rows fails the full-year check instead of a
  frequency error, and hourly weather with a gap in its first ten rows is
  resampled to 15 minutes, which fills the gap, instead of failing the
  full-year check. When the first column of a weather CSV is not a
  timestamp, `load_weather` now takes the first column named `date`,
  `datetime` or `time` in any case, such as `Date` or `TIME`; it used to
  match only `date`, `time` and `Datetime`, in that order. Results are
  unchanged bit for bit.
- `breos.plotting` checks for matplotlib once, when it is imported, and the
  error says `pip install "breos[plots]"`. Each plot used to check on its
  own call, with a `uv add matplotlib` hint, and the three payback plots
  (`plot_breakeven_distribution`, `plot_breakeven_cdf`,
  `plot_breakeven_summary_bar`) did not check at all. `import breos` still
  loads no plotting code. Without matplotlib, `import breos.plotting` raises
  `ModuleNotFoundError`, and a top-level plotting name such as
  `breos.plot_co2_savings` raises `AttributeError` with the same hint, so
  `help(breos)` and `getattr(breos, name, default)` still work. The plotting
  module also drops code no BREOS output reaches: a multi-year `Year` axis
  in the degradation plots, which no degradation frame carries, and a
  `Savings_Annual` column in `plot_breakeven`, which no cost projection has.
  Every figure the plotting tests write is byte-identical to before.
- `BatteryConfig.thermal_resistance_kw` is now `thermal_resistance_k_per_w`,
  and `breos.constants.DEFAULT_THERMAL_RESISTANCE_KW` is
  `DEFAULT_THERMAL_RESISTANCE_K_PER_W`, with no alias. The value is the
  pack-to-ambient thermal resistance in K/W; the old name read as kilowatts.
  `compute_cell_temperature` takes the same new keyword. No App, optimizer or
  CLI key sets it. The old keyword raises `TypeError`. Results are unchanged
  bit for bit.

### Fixed
- `calendar_model` is stored as it is validated: trimmed, lower-case, with
  hyphens as underscores ([#186](https://github.com/Str4vinci/breos/issues/186)).
  For `"Naumann-Lam"`, `degradation.model_key` and
  `provenance.resolved_config.calendar_model` now read `"naumann_lam"`
  instead of the spelling given. A name with surrounding spaces, which
  validation accepted, no longer fails in the aging model with `ValueError`.
  The optimizer's `[battery] calendar_model` is checked and stored the same
  way when its config resolves: it accepted any value, so an unknown name
  failed only once a candidate was evaluated, and a name with surrounding
  spaces failed there too. No number changes.
- `breos --version`, `provenance.breos_version` in App and Monte Carlo
  results, and the `breos_version` column of `breos sweep` read
  `"0.0.0+unknown"` outside an installed package, as `breos.__version__`
  does ([#186](https://github.com/Str4vinci/breos/issues/186)). The CLI and
  the sweep table reported `"0.1.0"` and the results `"unknown"`. All five
  read `breos.utils.package_version()`.
- Bundled demandlib H0 profiles now put weekday, Saturday and Sunday shapes
  on the corresponding day types in the study year, instead of assigning the
  dated 2023 rows by position ([#298](https://github.com/Str4vinci/breos/issues/298)).
  The nearest matching source day stays within four calendar days of the
  target date; leap days use their actual day type, and annual energy is still
  normalized to the configured consumption. **Results can change when the
  study year's weekdays differ from 2023.** With 4,000 kWh of H0 load and
  illustrative prices of 0.11/0.18/0.28 EUR per kWh on the bundled 2026
  Portuguese weekly tri-hourly schedule, the load-only annual import bill
  moves from 694.90 EUR to 685.59 EUR (−1.34%). The default 2023 path is
  unchanged. On up to six days a year the nearest source day falls in the
  neighbouring demandlib season: 21 and 22 March 2026 get winter shapes, and
  14 May 2025 gets a summer shape. A `demandlib_h0` file supplied through
  `rlp_directory` must now have a dated first row at 1 January 00:00; an
  undated file, which previously loaded by position, raises `ValueError`.
- Dated E-REDES BTN A/B/C profiles now put working-day, Saturday and
  Sunday/holiday shapes on those day classes in the study year, instead of
  placing their rows by position from 1 January
  ([#303](https://github.com/Str4vinci/breos/issues/303)). They use the H0
  rule of [#298](https://github.com/Str4vinci/breos/issues/298): each target
  day takes the nearest source day of its own class, trying the same month and
  day first, then the day before, then the day after, and wrapping at New
  Year; 29 February starts from 28 February. A Sunday or one of Portugal's
  nationwide statutory holidays takes the Sunday/holiday class, in both the
  source and the study year, even when the holiday is a Saturday. These are
  the 13 holidays of Labour Code Article 234, without Corpus Christi, 5
  October, 1 November and 1 December in 2013–2015, when they were suspended.
  Carnival, municipal holidays and bridge days are not holidays for this rule.
  The holiday calendar starts in 2004, the first full year of the 2003 Labour
  Code: a file stamped before 2004, or a study year before 2004, raises
  `ValueError`, also when the file is in its own year. Later years are assumed
  to keep the current list. The source year now comes from the file's
  timestamps, not its filename: an hourly file named 2025 but stamped 2023
  aligns by its 2023 calendar, and the hourly and 15-minute files from one
  publication share a phase. A file in its own year loads unchanged. A dated
  E-REDES file must start at 1 January 00:00; one that starts elsewhere, for
  example with interval-end stamps from 00:15, raises `ValueError` and names
  the converter. Its timestamps must be naive civil times: a timestamp with a
  UTC offset, fixed or changing at DST, raises `ValueError`, because read as a
  UTC instant a summer row would move one hour from its civil time. Other
  profiles still read offset timestamps as UTC instants. Undated E-REDES
  files, `bdew_h0`, `ree_2.0td` and `custom` keep their positional placement,
  and the demandlib H0 alignment is unchanged bit for bit. Monte Carlo aligns
  to its `target_year` the same way. No result or provenance field is added;
  `provenance.load_profile` still records the file and its SHA-256. **Results
  change for E-REDES runs whose study year differs from the file's dated
  year.** This was measured on two BTN C files, a 15-minute file dated 2025
  and an hourly file dated 2023, for a 2026 study at 3,500 kWh with 8 modules
  on the Porto TMY. The tariffs were the bundled 2026 weekly bi-hourly and
  tri-hourly schedules with illustrative prices. Across PV only, PV with a 5
  kWh battery, and fixed-target smart charging, the change is:
  - year-1 bill without a system: −0.49 to +1.29 EUR (−0.08% to +0.19%);
  - year-1 import bill with the system: −0.27 to +1.33 EUR (−0.16% to +0.76%);
  - grid import: −1.7 to +6.9 kWh;
  - 25-year NPV of savings: −15.08 to +10.71 EUR.
- Monte Carlo now builds the load profile for `target_year`, the calendar its
  weather years and tariff already use, instead of for the year of
  `start_date` shifted onto `target_year` by position
  ([#302](https://github.com/Str4vinci/breos/issues/302)). With the defaults
  (`start_date` 2023, `target_year` 2025), H0 put a 2023 Sunday shape on
  Wednesday 8 January 2025; it now follows the H0 day-type alignment above,
  so a Monte Carlo study and an App run of the same year use the same load.
  Profiles placed by position now also follow the target year's
  daylight-saving dates, which moves 169 hourly steps in Lisbon for 2025.
  Monte Carlo no longer uses `start_date` for the load or weather; its
  provenance records the load's year as `load_profile.calendar_year`
  (result schema 1.8). A leap `target_year`, which raised an error at hourly
  and 15-minute resolution, now runs: each weather year gets a 29 February
  copied from its 28 February on the file's own clock, including a UTC offset
  written in its timestamps, as the App gives a TMY. Before, 29 February was
  missing from the weather, and at 15 minutes the resampler would have
  interpolated across it as one night. **Monte Carlo results change
  when `start_date` falls in another year than `target_year`.** On the
  shipped `configs/examples/montecarlo.toml` (100 runs, seed 42) with the
  Porto 2005–2024 Open-Meteo history, mean NPV savings fall from
  4,446.79 EUR to 4,433.51 EUR (−13.28 EUR, −0.30%), mean annual grid import
  rises from 1,029.63 kWh to 1,033.00 kWh (+3.36 kWh, +0.33%), mean annual
  self-consumption falls from 2,965.30 kWh to 2,961.93 kWh (−0.11%), mean
  lifetime grid independence falls from 74.25% to 74.16%, and the mean
  interpolated payback moves from 13.109 to 13.122 years. A study whose
  `start_date` is in `target_year` is unchanged bit for bit.
- An App run in a leap year now gives a TMY stamped at a fixed UTC offset its
  29 February ([#313](https://github.com/Str4vinci/breos/issues/313)).
  `remap_tmy_year` shifted the TMY to the study year in UTC, so for a PVGIS
  TMY saved at `+01:00` (Berlin) or `+11:00` (Melbourne) the first hours of
  local 1 March landed on 29 February, and the leap-day fill, which reads the
  index's own clock, saw that day as present and skipped it. The weather was
  8,760 hours on an 8,784-hour year, and 29 February had no irradiance
  (`+01:00`) or only part of 1 March's (`+11:00`); at 15 minutes the resampler
  interpolated across the gap. A fixed-offset index is now shifted on its own
  clock, so 29 February is a copy of 28 February, 1 March keeps its own
  hours, and the weather metadata records the fill as `leap_day`, as for a
  UTC TMY. **Results change for leap study years with a fixed-offset TMY.**
  This includes PVGIS TMYs saved by BREOS's own downloader, which stamps them
  at the location's winter UTC offset (for example `+01:00` for Berlin or
  Madrid). A leap-year fixed-offset TMY moved to a common study year also
  changes: it now drops its own local 29 February, where it previously
  dropped the UTC day and shifted part of it into 1 March. UTC-stamped files
  (such as Porto or Lisbon TMYs), naive timestamps (read as UTC), timestamps
  with mixed daylight-saving offsets (read as UTC), and a fixed-offset TMY
  moved between two common years or between two leap years are unchanged
  bit for bit.
- Fixed-design evaluation and multi-objective optimization now validate and
  apply the optional `tariff` table through the shared projection loop.
  Previously they silently ignored it and valued the design at flat prices.
  App's checks for prices, timezone, resolution and conflicting flat costs
  apply before PV calculation or worker startup. Results record tariff
  provenance, and immutable tariff prices can be pickled for worker processes.
  Configurations without either table keep their existing behavior.
- Corrected the Portuguese reform citation in ADR 0002 and the tariff plan:
  the electricity periods are set by Diretiva n.º 3/2026, de 19 de agosto; the
  directive with the same number dated 26 June sets gas prices. Docs only.
- App weather that does not cover the whole calendar year of `start_date`
  raises `ValueError` instead of simulating a shorter year
  ([#242](https://github.com/Str4vinci/breos/issues/242)). The simulation
  calendar ran from the first to the last weather row, so a cached TMY file
  starting on 8 January simulated about 51 weeks: consumption came out at
  3,908 kWh instead of the configured 4,000, and the economics still treated
  the run as a full year. A gap in the middle of the weather was already an
  error; missing leading or trailing rows now are too, and the message names
  the weather file or source and the missing span. The year may be complete on
  the weather's own clock, on UTC, or on the location's timezone, so PVGIS
  fixed-offset years, UTC-year CSV weather, and civil-year local weather all
  still run, as do leap years and 15-minute runs on resampled hourly weather.
  **Results change only for truncated weather, which now raises.** Monte Carlo
  already skips incomplete weather years, and the optimizer simulates the
  window its caller supplies.
- The optimizer now models the site at its real elevation, as App does.
  `evaluate_projected_design` and `SolarDesignProblem` built their pvlib
  location with `altitude=0` unless the config set one, while App leaves the
  altitude to pvlib's lookup (82 m for Porto). Elevation sets the air pressure
  in the refraction correction of the solar position, so the same design gave
  slightly different PV output in the two paths. An explicit
  `location.altitude` is still used as given. **Results change** slightly for
  optimizer runs without `location.altitude`: in the App parity case, year-1
  PV DC falls by 10 Wh of 2021 kWh and grid independence from 39.5956% to
  39.5947%, now equal to App.
- `fetch_weather_data`, `read_epw_file` and `load_profile` raise `ValueError`
  for a frequency other than hourly or 15-minute
  ([#175](https://github.com/Str4vinci/breos/issues/175)). They
  used to return hourly data for `freq="30min"`. The weather readers check the
  frequency before any request or file read.
- Removed the "Year 1 PV degradation" stage (`year_1_degradation`) from the
  App `pv_loss_waterfall` ([#175](https://github.com/Str4vinci/breos/issues/175)).
  Year 1 has no degradation, so the stage was always 0 kWh.
  `pvwatts_static` is now the last stage and equals
  `energy_balance.pv_dc.generation_kwh`. `provenance.ledger_schema_version`
  and `pv_loss_waterfall.ledger_schema_version` move to `1.2`. Consumers that
  look the stage up by key, or index `stages[6]`, need updating. These are
  the only changes to the App golden baseline.

### Fixed
- Config errors that surfaced only after the weather fetch are reported when
  the App is built, and CLI, TOML and Python config are normalised the same
  way ([#176](https://github.com/Str4vinci/breos/issues/176)). Load-profile
  aliases, PV loss component names and a missing battery temperature CSV fail
  during construction. Unknown `[montecarlo]` keys are rejected before the
  weather file is checked. Hyphenated keys in nested tables and native TOML
  `date` values are normalised, and conflicting spellings of one key in a
  table raise. `breos run`, `breos sweep` and `breos montecarlo` warn about a
  runner section they do not use.
- Monte Carlo selects its dispatch backend by one rule from the CLI and from
  Python: `--execution-backend`, then `[montecarlo].execution_backend` (or
  `MonteCarloSettings.execution_backend`), then the top-level
  `execution_backend`, then `"python"`. `MonteCarloSettings.execution_backend`
  now defaults to `None`, meaning inherit; `run_montecarlo` returns the
  resolved backend in `result.settings`. Before, the CLI honoured the
  top-level key but a Python `run_montecarlo` call ignored it.
- The 15-minute weather resamplers no longer depend on the timestamp resolution
  of the input index ([#150](https://github.com/Str4vinci/breos/issues/150)).
  Both divided the raw integers by `10**9`, which is only correct for
  nanosecond indexes. pandas 3, which the lockfile pins, parses CSV timestamps
  as microseconds, so the interpolation grid was jittered and some quarter-hours
  collapsed onto the same point. Second-resolution input failed with
  `ValueError`. **Results change for 15-minute runs under pandas 3** and now
  match pandas 2. On the bundled Porto PVGIS TMY, annual GHI falls by 0.05%,
  DNI by 0.04%, and DHI by 0.01%. Daytime GHI moves by 2.4 W/m² on average,
  and by up to 220 W/m² at individual steps. The same applies to a 15-minute
  load profile built from an external hourly file because its 15-minute file is
  missing. Hourly runs, nanosecond input, and profiles loaded from native
  15-minute files are unchanged.
- Hourly-to-15-minute resampling places hourly means at the middle of their
  hour ([#175](https://github.com/Str4vinci/breos/issues/175)).
  `resample_to_15min` evaluated only the clear-sky reference at the
  representative time; the clearness index, temperature and wind were
  interpolated at the labels, so interval-mean weather came out 22.5 minutes
  early. All columns are now interpolated at the representative times the
  weather metadata records. A load profile built from an hourly file was
  interpolated the same way and its hours kept their mean only in the annual
  total: the bundled H0 profile averaged to hours and resampled back ran about
  22.5 minutes early (RMSE 8.18 W against the native 15-minute profile, 3.08 W
  one step later) and missed each hour's mean by 5.95 W on average. It is now
  interpolated between hour midpoints and scaled per hour, so each hour keeps
  its mean exactly (RMSE 1.16 W, no lag). **Results change for 15-minute runs
  from interval-mean weather (EPW, Open-Meteo means, files whose metadata says
  `interval_mean`) and from external load profiles supplied only as hourly
  files.** PVGIS TMY weather (instant samples), weather without metadata, and
  native 15-minute load profiles are unchanged.
- The simulation boundary rejects invalid PV and load input instead of
  repairing it ([#151](https://github.com/Str4vinci/breos/issues/151)).
  `simulate_energy_balance`, `simulate_energy_balance_summary`, and
  `align_simulation_inputs` used to fill every PV or load step without a value
  with zero and accepted negative load. So a load covering half the window
  halved consumption, and a −100 W load made PV deliver negative energy to the
  load and export PV that did not exist. They now raise `ValueError` for PV or
  load that does not cover every step with a finite value, for load
  timestamps off the simulation interval, and for negative load. Fill gaps
  explicitly before simulating if that is the intent. A supplied temperature
  series gets the same check since
  [#153](https://github.com/Str4vinci/breos/issues/153).
- A civil-year load profile on a UTC-year weather calendar no longer gets
  zero-load steps at the year edge. This affects Monte Carlo and CSV weather
  outside UTC±0: 1 step for Berlin, 5 for New York, and 11 for Sydney at
  hourly resolution. Those steps now take the same instant of the repeating
  annual profile, one year earlier or later. **Results change slightly for
  those runs.** PVGIS runs use the location's fixed offset and are unchanged.
- A `start_date` other than 1 January is rejected instead of misplacing the
  load profile ([#152](https://github.com/Str4vinci/breos/issues/152)). App
  weather always starts on 1 January, but the load profile stamped its first
  row (1 January) onto the configured start. So `2025-07-01` put January's
  winter demand on July against January weather, and before #151 it also left
  half the year at zero load. `validate_config` and `load_profile` now raise
  `ValueError` and name the 1 January date to use. Arbitrary project start
  dates would need weather aligned to the requested window, which is a
  separate feature.
- Battery temperature input that cannot be used raises instead of becoming
  25 °C ([#153](https://github.com/Str4vinci/breos/issues/153)).
  `build_battery_temperature_series` used to substitute 25 °C, or 22.9 °C
  after the indoor model, for a missing or unreadable `battery_temperature`
  CSV, a CSV without recognised columns, and readings that did not line up
  with the simulation index. A CSV from another calendar year, or with naive
  timestamps on the UTC simulation index, matched no step, so the whole year
  ran at 25 °C. These now raise `FileNotFoundError` or `ValueError` and name
  the file, the missing columns, or the uncovered steps. Naive CSV timestamps
  are read as UTC, as naive weather timestamps are. Readings hold within their
  own sampling interval, so hourly readings still drive a 15-minute run, but a
  gap or a non-finite reading is an error. A bool, a non-finite number, or
  another unsupported `battery_temperature` value raises too, and so does a
  `temperature_series` passed to the simulation that leaves steps uncovered.
  Omitting the temperature still means 25 °C, as does weather without a
  temperature column; that case belongs to
  [#172](https://github.com/Str4vinci/breos/issues/172).
  `evaluate_projected_design` with `weather_by_year` now restamps the
  representative year's temperatures onto the sequence's calendar year, via
  the new `align_weather_year` keyword. **Results change for sequences in a
  different year from `tmy_data`**, which ran their battery at 22.9 °C with the
  indoor model on. App runs are unchanged: the default Porto run with a 5 kWh
  battery matches develop exactly at hourly and 15-minute resolution.
- The steady-state optimizer books battery replacements at the swap instant
  ([#173](https://github.com/Str4vinci/breos/issues/173)), as the App, Monte
  Carlo and projected paths have since the replacement-timing fix.
  `calculate_financials` rounded each estimated end of life up to a whole
  year, inflated the outlay to the start of that year and discounted it from
  the end. It now books each swap at its fractional time, the moment the
  repeated year-one SOH loss reaches EOL, and inflates and discounts from
  there. A swap at or after the end of the horizon is no longer booked. It
  also honours an explicit `battery.replacement_cost`, which it ignored in
  favour of the storage cost. A €4,000 swap at t = 9.375 with 2% inflation and
  5% discount now costs €3,048.15, as in the projection, instead of €2,934.73.
  **Results change for `objective_basis = "steady_state"` and for
  `SteadyState_NPV_Eur` in projected runs, for designs with a battery.** On the
  bundled Porto PVGIS TMY with the example config at 35° south, steady-state
  NPV moves from −€2,035.82 to −€2,238.85 for 8 modules and 5 kWh, and from
  −€21,697.74 to −€22,720.60 for 9 modules and 20 kWh. PV-only designs and
  projected objectives are unchanged.
- `power_limit_c_rate` limits the stored energy in both directions, as a cell
  current rating does ([#155](https://github.com/Str4vinci/breos/issues/155)).
  It used to set `max_charge_power_w` and `max_discharge_power_w` to the same
  wattage, but those apply at the DC charge input and at the AC discharge
  output. So 1 C on a 5 kWh pack stored at most 4,873 W (0.97 C) and drew up to
  5,553 W (1.11 C) from the cells, a 14% spread. Both directions now stop at
  5,000 W of stored-energy change. Charging accepts up to 5,130 W of DC, and
  discharge delivers up to about 4,500 W of AC, depending on the inverter's
  part-load efficiency. `BatteryConfig` no longer fills
  in the two absolute limits from the C-rate; the limit is
  `BatteryConfig.stored_power_limit_w`. To reproduce the old behaviour, set
  `max_charge_power_w` and `max_discharge_power_w` to the C-rate times the
  capacity. **Results change only where the limit binds.** For Porto with
  8,000 kWh/yr and 14 modules, grid import falls by 0.002% with 5 kWh at
  0.5 C and by 0.045% with 10 kWh at 0.25 C. It is unchanged with 3,500 kWh/yr,
  8 modules, and 5 kWh at 1 C or 0.5 C. Charging binds more often than
  discharge, so the gain in stored PV outweighs the lower discharge ceiling.
  The upcoming publication's C-rate never binds, so its results are unaffected.
- `pv_arrays` inherit the top-level tracker settings, and tracker keys and
  array entries are validated
  ([#167](https://github.com/Str4vinci/breos/issues/167)). An array inherited
  `module`, `tilt`, and `azimuth` from the top level but not `tracking`,
  `max_angle`, or the other tracker keys, so the PV model's own fallbacks
  applied: a top-level single-axis tracker ran fixed-tilt once `pv_arrays` was
  set, and a top-level `max_angle` did not reach a tracking array. Arrays now
  inherit every tracker key, and the result reports each array's resolved
  `tracking` and tracker geometry. Tracker keys are range-checked at the top
  level and per array, `backtrack` must be a bool (the string `"no"` is
  truthy, so it used to leave backtracking on), a misspelled array `tracking`
  fails before the weather fetch, and an unknown array key such as `tlt` is
  rejected instead of silently dropped. **Results change for arrays under a
  top-level tracker setting.** On the Porto PVGIS TMY with 10 modules,
  `tracking = "single_axis"` with `pv_arrays = [{modules = 10}]` gives
  9,700.6 kWh of DC instead of 8,672.0 (+11.9%), the same as without
  `pv_arrays`. A single-axis array under a top-level `max_angle = 20` gives
  8,983.4 kWh instead of 9,700.6 (−7.4%). Fixed-tilt arrays are unchanged.
- Open-Meteo interval-mean weather fetched for Monte Carlo keeps its last year
  ([#169](https://github.com/Str4vinci/breos/issues/169)). Those means are
  labelled at the end of their hour, and `fetch_weather_data` stopped at
  23:00 on `end_date`. Once BREOS moved each label to the start of its hour,
  the last year was one step short, and `preload_weather_by_year` dropped it
  without saying so: a 2022-2024 file gave Monte Carlo only 2022 and 2023.
  The fetch now runs through the midnight after `end_date` and starts at
  01:00 on `start_date`, so it covers exactly the requested hours.
  Instantaneous fetches are unchanged. `preload_weather_by_year` now warns
  about each year it skips, and its year-splitting helper takes the step size
  from the whole file. Files fetched before this fix still lose their last
  year, now with a warning; fetch them again to keep it. **Results change only for
  interval-mean files fetched from now on**, which give Monte Carlo one more
  weather year. Monte Carlo on existing files is unchanged: the per-year
  weather from both Porto 2005-2024 Open-Meteo files matches develop exactly.
- The optimizer resolves an unset battery setting to the App's default
  ([#156](https://github.com/Str4vinci/breos/issues/156)).
  `optimize_system_multi_objective` and `evaluate_projected_design` used their
  own fallbacks: a 20-80% SOC window instead of 10-90%, and 0.9795 each way
  (95.9% round trip) instead of 95%. So a battery section that omitted them
  gave a 5 kWh pack a 3 kWh window where the App gives it 4 kWh. Unset keys now
  take the `BatteryConfig` defaults, which the App's `battery_min_soc` and
  `battery_max_soc` defaults now reference. **Results change for optimizer
  configs that omit `min_soc`, `max_soc`, `charge_efficiency` or
  `discharge_efficiency`.** On the bundled Porto PVGIS TMY with the example
  projected config minus those four keys, 8 modules and 5 kWh at 35° south,
  projected grid independence rises from 67.35% to 73.08% and projected NPV
  from €733.87 to €1,426.86. Steady-state grid independence rises from 70.77%
  to 77.20%, but steady-state NPV falls from −€1,159.56 to −€2,037.31: the
  wider window raises the year-one SOH loss from 5.92 to 6.17 points, and the
  replacement estimate books four swaps instead of three. App runs and configs
  that set all four keys, such as the example config, are unchanged.
- Monthly result rows group on the local calendar of the result frame
  instead of UTC ([#166](https://github.com/Str4vinci/breos/issues/166)).
  They converted the `Datetime` column to UTC before grouping, so east of UTC
  the local year started with a stub of the previous December and every month
  boundary moved by the UTC offset. **The `monthly` result changes for App runs east of
  UTC.** A PVGIS run for Berlin or Melbourne returned 13 rows, the first a
  1-hour or 11-hour December stub. It now returns the 12 local months. With
  10 modules and a 5 kWh battery, Berlin monthly PV is unchanged because the
  moved hour is at night, and monthly consumption moves by up to 0.27 kWh.
  Melbourne monthly PV moves by up to 6.6 kWh and consumption by up to
  4.0 kWh. Yearly totals, NPV, payback and LCOE of App runs do not use this
  grouping and are unchanged, and Porto is unchanged. A `Datetime` column
  read back from a CSV of an IANA-zone run, which has two UTC offsets, groups
  on each row's own wall-clock time.
- The projected optimizer's budget constraint checks the CAPEX it reports
  ([#157](https://github.com/Str4vinci/breos/issues/157)). With
  `objective_basis = "projected"`, `budget_eur` was compared with the
  steady-state CAPEX, which prices modules at `costs.panel_wp` when that key is
  set, while `Projected_Initial_Cost_Eur` prices the selected module's `Mpp`.
  One 550 W module under a 400 W `panel_wp` passed a €480 budget at €465.48 and
  was reported at €490.03. The constraint now uses the projected CAPEX.
  **Pareto fronts change for projected runs that set `costs.panel_wp` below the
  module's `Mpp`**: designs over budget at the reported cost are now
  infeasible. Without `panel_wp` the two CAPEX figures are identical, so other
  runs are unchanged. On the bundled Porto TMY with the example projected
  config, 9 modules and 20 kWh give €14,794.97 on both paths.
- Terrain shading from `horizon_profile` honours `solar_position="weather"`
  ([#159](https://github.com/Str4vinci/breos/issues/159)). The shading step
  evaluated the sun at the timestamp itself, while transposition used the
  offset the weather metadata declares. So the terrain mask and the
  irradiance model could disagree about where the sun was, and beam that
  cleared the horizon at the interval midpoint was removed. Both now take the
  offset from one resolver, `solar_position_time_offset`, and shading with
  `"weather"` raises, as transposition does, when the metadata has no
  radiation time basis. **Results change for runs with a `horizon_profile`
  and `solar_position="weather"`.** For Porto with 10 modules and a
  six-point horizon of 4 to 12 degrees, annual AC production changes from
  8,260.44 to 8,253.78 kWh (−0.08%) on the PVGIS TMY, whose instants are
  offset by about 10 minutes, and from 7,983.21 to 8,003.96 kWh (+0.26%) on
  Open-Meteo 2023 interval means. On the Open-Meteo year the horizon loss
  falls from 96.9 to 76.2 kWh. Other `solar_position` methods and runs
  without a `horizon_profile` are unchanged.
- The optimizer's discrete repair keeps candidates inside the configured
  bounds ([#158](https://github.com/Str4vinci/breos/issues/158)). It rounded
  modules and battery kWh to integers and tilt and azimuth to 5° with no bound
  check, so 62.9° became 65° under a 63° `max_tilt_deg`, and 4.6 kWh became
  5 kWh under a 4.9 kWh `max_battery_kwh`. A value that rounds past a bound now
  takes the grid point just inside it. **Pareto fronts change for runs whose
  `max_modules`, `max_battery_kwh` or `max_tilt_deg` is off the grid.** On the
  bundled Porto PVGIS TMY with the example config on the steady-state basis,
  `max_tilt_deg = 24`, `max_battery_kwh = 4.9`, 20 candidates for 10
  generations and seed 1, develop evaluated 38 of 200 candidates at 5 kWh and
  1 at 25°, and returned 6 of 20 Pareto designs at 5 kWh. Now none leave the
  bounds. Bounds on the grid, including `max_tilt_deg = "adjust"`, are
  unchanged.
- The Monte Carlo summary says how many runs paid back
  ([#160](https://github.com/Str4vinci/breos/issues/160)). A run that never
  pays back within the horizon has no payback year, and the summary dropped it
  without saying so. Nine runs that never paid back and one that paid back in
  year five gave a payback summary of five years at every statistic. Each
  summary entry now gives `count`, the runs its statistics cover, and
  `n_runs`. `payback_year` and `payback_year_exact` also give
  `payback_probability`, and they are kept with only these counts when no run
  pays back; they used to be left out. The CLI table shows the run count per
  metric and prints the share that paid back. Reported statistics do not
  change. On the example `montecarlo.toml` config with Porto Open-Meteo
  weather, a 10-year horizon, seed 42 and 40 runs, the exact payback is still
  9.69-9.90 years from p5 to p95, and the summary now adds that 27 of the 40
  runs (67.5%) paid back. `run_montecarlo` also rejects load-scale bounds
  that would make demand negative: a negative or non-finite `min_load_scale`,
  a `max_load_scale` below `min_load_scale`, and a non-finite
  `load_uncertainty`. `max_load_scale=-1` used to run with negative demand.
- A leap-year `start_date` runs instead of failing after validation
  ([#170](https://github.com/Str4vinci/breos/issues/170)). The PVGIS fetch
  rejected leap sample years, because a TMY has 8,760 hours, so
  `App({"start_date": "2028-01-01", ...})` constructed and then failed in
  `simulate()`. `fetch_tmy_weather_data` now fetches a leap year on the
  preceding year's calendar and restamps it, and `remap_tmy_year` does the
  same for local TMY files. Both give the weather a 29 February copied from 28
  February, which is what the load profile already did, and record it in the
  weather metadata as `leap_day`. The new `fill_leap_day` does the copy.
  Non-leap years are unchanged. A 2028 Porto run with 8 modules and 5 kWh
  yields 19.4 kWh more PV than 2027, from the extra day.
- External load profiles of the wrong length raise instead of being repeated
  or cut ([#171](https://github.com/Str4vinci/breos/issues/171)).
  `load_profile` places rows on the calendar by position, and used to repeat a
  short file or truncate a long one without a message. The E-REDES 15-minute
  exports end with a blank `,,,` row, so a 2024 run saw 35,041 rows, skipped the
  leap-day insertion, and repeated the file: 29 February took 1 March's load,
  the rest of the year ran a day early, and 31 December took 1 January's load.
  Fully blank rows are now dropped, and a profile must then have exactly one
  common or leap year of rows at its resolution (8,760 or 8,784 hourly; 35,040
  or 35,136 at 15 minutes). A leap-year file on a common-year run drops its
  29 February instead of losing 31 December. Profiles are also checked when
  they load: values must be finite and non-negative, and a timestamp column,
  when present, must parse on every row and step evenly, so a local-clock file
  with a DST gap, or one malformed stamp, raises.
  E-REDES profiles 4, 5, and 6 select their own `BTN A/B/C - Wh` column by exact
  name; profile 6 used to fall back to the first `BTN` column, which is BTN A.
  **Results change for leap-year runs on the E-REDES 15-minute file.** On
  Porto 2024 with profile 6, 10 modules, and 4,000 kWh/yr, the load at each
  step moves by 2.6% on average (5.1% for BTN A). Grid import rises by 0.54 kWh
  (+0.02%) without a battery and falls by 0.45 kWh (−0.04%) with 5 kWh, and
  NPV moves by −€2.23 and +€0.71. Hourly E-REDES runs, common-year runs, and
  the bundled profiles are unchanged.

- PV weather input with gaps, at the wrong resolution, or without air
  temperature or wind speed raises instead of being repaired
  ([#172](https://github.com/Str4vinci/breos/issues/172)), the PV-side
  counterpart of #151 and #153. The PV model filled every simulation step from
  the nearest weather row, so weather with June removed gave June zero PV,
  a truncated file shortened the year, and 15-minute weather run at
  `freq="h"` was thinned to hourly, all without a message. Weather must now
  step evenly at `freq` from its first row to its last. A missing air
  temperature or wind speed column used to become 25 °C and 1 m/s, and a NaN
  air temperature a 25 °C cell; both now raise, and the message shows how to
  add a constant column explicitly. Battery temperature in `"weather"` mode
  raises the same way when the weather has no temperature column; a fixed
  `battery_temperature` is the explicit alternative. Weather files whose
  timestamps have no timezone are still read as UTC, but now log a warning,
  unless their metadata sidecar records the timezone, as files BREOS writes
  do. Results for complete, regular weather with both columns are unchanged.

- The optimizer's steady-state scoring uses the same horizon, PV degradation,
  and battery degradation engine as projected scoring
  ([#212](https://github.com/Str4vinci/breos/issues/212)). Steady-state NPV
  read `financials.project_lifespan` and `financials.pv_degradation_rate`,
  while projected scoring reads `simulation.years_projection` and
  `pv.degradation_rate` first, so the `SteadyState_*` diagnostics next to
  `Projected_*` could rest on another horizon. Both now resolve through one
  helper. The steady-state simulation also ignored `battery.degradation_engine`
  and `blast_model`, so it always ran native aging, and ran even with an
  invalid `blast_model`. It now uses the configured engine, and
  `SolarDesignProblem` validates both keys when it is built, on either basis.
  PV-only candidates run native aging on both bases: a projected BLAST
  optimization used to raise on every 0 kWh candidate. **Steady-state
  results change for configs whose key pairs disagree, including a
  `years_projection` set without `project_lifespan` (default 20), and for
  BLAST configs.** On synthetic test weather with 8 modules and 5 kWh, a
  config with 3 years / 10% for projected and 2 years / 2% under `financials`
  moves steady-state NPV from −€5,648.44 to −€5,172.08. With BLAST
  `nmc_gr_50ah_b1` over 20 years, steady-state grid independence moves from
  69.90% to 70.10% and NPV from −€1,547.61 to €1,374.77. Projected results
  are unchanged, apart from 0 kWh BLAST candidates, which now run. The example
  optimization config sets both pairs equal and uses native aging, so it is
  unaffected.

- EPW weather records its radiation time basis
  ([#213](https://github.com/Str4vinci/breos/issues/213)). EPW radiation is
  energy over the hour that ends at each record, and pvlib labels that hour at
  its start, but `read_epw_file` recorded neither fact. The 15-minute
  clear-sky resampling therefore evaluated each hour at its label rather than
  its midpoint, `solar_position = "weather"` raised on EPW input, and the
  metadata was attached after resampling, which lost the resampling
  provenance. `read_epw_file` now records `radiation_time_basis =
  "interval_mean"` and `timestamp_label_basis = "left"` before resampling.
  **Results change for 15-minute EPW weather.** On a synthetic clear-sky EPW,
  the mean error against the true 15-minute GHI falls from 27 to 1.6 W/m². On
  the Amsterdam IWEC file with 10 modules, annual 15-minute GHI rises by 0.64%
  and DC by 0.92% (4,425.96 to 4,466.35 kWh), with steps moving by up to
  71 W/m². Hourly EPW runs with the default solar position are unchanged.

- Aligned simulation inputs carry their own resolution, and `PV_Production`
  has one definition ([#214](https://github.com/Str4vinci/breos/issues/214)).
  `simulate_energy_balance_summary(aligned=...)` converted power to energy with
  its `freq` argument, default `"h"`, so an aligned 15-minute input reported
  96 kWh of load where it had 24 kWh. `AlignedSimulationInputs` now stores
  `freq`, runs on aligned inputs use it, and a `freq` that disagrees raises
  `ValueError`; `with_pv_only_chain` does the same. App and Monte Carlo passed
  `freq` explicitly and are unchanged. Without an inverter rating,
  `PV_Production` counted DC sent to the battery at the inverter efficiency;
  with one, at its DC value. All three copies (scalar, vectorised PV-only,
  and numba) now use `PV_DC − PV_DC_Curtailed − PV_Direct_Inverter_Loss`, which
  equals AC to load and export plus DC to the battery, and the kernel's
  `cap_wh_is_infinite` argument is gone. **`PV_Production`, `Total PV [kWh]`,
  and the returned total PV change for unrated runs with a battery.** On the
  48-hour golden fixture (3 kWh), total PV rises from 22.272 to 22.473 kWh
  (+0.90%). App, Monte Carlo, and `SolarDesignProblem` results are
  unchanged: the App always rates the inverter, and the optimizer scores
  from the AC ledger even with `dc_ac_ratio = 0`.

- `apply_terrain_horizon_profile` accepts `float32` and integer irradiance
  columns ([#210](https://github.com/Str4vinci/breos/issues/210)). Open-Meteo
  data arrives as `float32`, and pandas 3 raised `TypeError` when the shaded
  `float64` GHI was written back. A float column now keeps its dtype,
  including the nullable pandas `Float32` and `Float64`, and an integer
  column is widened to `float64` (nullable `Int64` to `Float64`). Missing
  nullable values count as zero irradiance, like `NaN`. Results for `float64`
  weather, which is what the App passes, are unchanged.

- Two plotting defects ([#219](https://github.com/Str4vinci/breos/issues/219)).
  `plot_validation_multi_system` called `plt.cm.get_cmap`, which matplotlib
  3.11 removed, and now uses the colormap registry. `plot_cell_temperature`
  drew months without data at 0 °C; they are now gaps.

- `load_results` and the plotting functions read result CSVs from a run in a
  DST zone, and `load_results` accepts a path-like
  ([#216](https://github.com/Str4vinci/breos/issues/216)). Such a CSV mixes UTC
  offsets, and pandas 3 refused to parse it as one column, so
  `load_results` and nine plotting call sites raised. They now parse it on
  the results' own wall clock through `utils.local_datetime_index`, and the
  energy plots take their step length from the rows' UTC instants. Because
  the wall-clock index repeats an hour in autumn, `load_results` also keeps
  those instants in a `Datetime_UTC` column for a file with mixed offsets,
  so its output can go straight to the plots. `plot_battery_soh_timeseries` reads `start_date` and `end_date` on the
  results' clock; with a timezone-aware index they used to raise. Results
  are unchanged.

- `apply_terrain_horizon_profile` finds irradiance under the same names as
  the resampler and the PV model
  ([#211](https://github.com/Str4vinci/breos/issues/211)), so BREOS's own
  Open-Meteo output (`shortwave_radiation`, `direct_normal_irradiance`,
  `diffuse_radiation`) no longer raises there. The three copies of the alias
  list are now one, `breos.utils.IRRADIANCE_COLUMN_ALIASES`, matched without
  regard to case. Results are unchanged.

- Provenance records what the model did
  ([#215](https://github.com/Str4vinci/breos/issues/215)). A 15-minute App
  run kept `input_resolution`, `output_resolution` and the resampling method
  out of `provenance.weather`, because App wrote the pre-resampling metadata
  back over the resampler's. With no `pv_module` set,
  `provenance.resolved_config.pv_module` and the CLI dry run recorded the
  module's display name, which App rejects; they now record its catalogue key,
  so the recorded config replays to the same result. Monte Carlo now records
  the solar-position method and offset the PV model applies: `"Mid-Interval"`
  used to be recorded with a 0-minute offset while 30 minutes were applied
  at hourly resolution. Simulated numbers are unchanged.

- An optimizer result run with early stopping pickles
  ([#217](https://github.com/Str4vinci/breos/issues/217)). The early-stopping
  termination was a local class, so `details["pymoo_result"]` could not be
  pickled.

- Two PV API gaps ([#220](https://github.com/Str4vinci/breos/issues/220)).
  When every array in `calculate_multi_array_production_breakdown` is empty,
  the zero result now uses the time grid a non-empty array uses; it kept the
  weather's own index, so hourly weather at `freq="15min"` gave hourly
  zeros. A negative module count raises instead of being skipped.
  `calculate_pv_production_ac` accepts and forwards `loss_overrides`, like
  every other production entry point. App results are unchanged.

- Payback has one rule across the library, the plots and the tools
  ([#218](https://github.com/Str4vinci/breos/issues/218)). The new
  `economics.find_payback_year_exact` interpolates where cumulative
  discounted savings first turn positive, the rule `find_payback_year`
  already used, scaled by the spacing between the two years; Monte Carlo
  and the optimizer each had a private copy of it, and `plot_breakeven`, `plot_breakeven_comparison`,
  `tools/compare_results.py` and `tools/batch_compare_locations.py` had
  three more rules. `create_cost_plots` uses `find_payback_year`. The Monte
  Carlo payback distribution and CDF plot `payback_year_exact` instead of
  the integer year, so a run that pays back at 4.2 years is binned at 4.2,
  not 5. `plot_breakeven_comparison` now marks a design that pays back in
  its first row. `format_years_months` in `breos.utils` replaces two copies
  of the "4y 2m" formatter. The interpolator Monte Carlo and the optimizer
  used counted zero savings as payback; it no longer does, which changes
  `payback_year_exact` only when cumulative savings land on exactly zero.
  Reported App, Monte Carlo and optimizer numbers are otherwise unchanged.

- Three small fixes from the 0.7 audit
  ([#175](https://github.com/Str4vinci/breos/issues/175)). The CLI reports a
  missing optional extra (such as Numba for `execution_backend = "numba"`) as
  an error line instead of a traceback. A carried resistance growth of 0.0,
  which a replaced pack starts from, is used as given: it was read as "not
  supplied" and replaced by `BatteryConfig.initial_resistance_growth`, so
  a direct API caller chaining years restarted the new pack at the old
  pack's resistance. App, Monte Carlo and the optimizer use a configured
  value of 0 and are unchanged. Monte Carlo summaries leave out infinite
  values as they do NaN, so an infinite LCOE no longer makes the mean
  infinite and the spread NaN; `count` shows how many runs remain.
- The CLI writes only standard JSON
  ([#175](https://github.com/Str4vinci/breos/issues/175)). Python's
  `json.dumps` writes NaN and infinity as `NaN` and `Infinity`, which are not
  JSON, so strict parsers (`jq`, JavaScript, most non-Python readers) rejected
  a `run` result with an infinite LCOE, a sweep summary, or a Monte Carlo
  summary and provenance file holding such a value. An undefined metric or
  summary statistic is now written as `null`, and every JSON output is
  written with `allow_nan=False`, so any other non-finite number fails with
  an error line instead of producing an invalid file. `breos.io` gains
  `nonfinite_to_none`, which does this conversion. The Monte Carlo
  `max_load_scale` must now be finite: `inf` passed the check and was
  written to the provenance file as `Infinity`. Leave it unset (`None`) for
  an unbounded load scale, as before. The sweep CSV still spells an
  undefined value `inf`, and numbers are otherwise unchanged.

- Four more findings from the 0.7 audit
  ([#175](https://github.com/Str4vinci/breos/issues/175)) are fixed.
  - Importing `breos.plotting`, and resetting it with
    `set_presentation_mode(False)`, no longer switches Matplotlib's
    process-wide backend to Agg.
  - A battery replacement is booked at the end of the interval flagged in
    `Battery_Replaced`, the last one the old pack ran, instead of at its
    start. The within-year replacement fraction is now `(step + 1) / n_steps`,
    in `(0, 1]`. Several reported financial values in the hourly App golden
    change by €0.01.
  - `PVModuleParams` rejects non-finite or nonphysical datasheet values,
    requires `T_Pmax_pct < 0`, and checks that `Mpp` matches `Vmp * Imp`
    within 2%. The checks run on construction and on every field assignment,
    and a rejected assignment leaves the module unchanged. To move the STC
    point further than 2% in one go, use
    `dataclasses.replace(module, Mpp=..., Vmp=..., Imp=...)`. `alpha_sc`,
    `beta_voc`, and the new `gamma_pmp_effective` are read-only properties
    computed from the current fields, so they stay current after in-place
    edits, `dataclasses.replace`, a `dataclasses.asdict` round trip, copies
    and pickles. Before, `replace(module, T_Pmax_pct=...)` kept the old
    power coefficient. See Changed for what `gamma_pmp` now holds.
  - The PV-only Monte Carlo cache records the PV array and inverter settings
    it was built for, and a mismatched inverter raises at simulation time. It
    keeps read-only copies of the PV input and the conversion arrays, so the
    caller's arrays stay writable. Load-only scaling keeps the cache and PV
    scaling clears it. Storing the PV input adds a fourth array per weather
    and project year pair, so a study now skips the cache from three quarters
    of the size it did before.
- PV module age is counted at the start of each simulated year everywhere
  ([#175](https://github.com/Str4vinci/breos/issues/175)). Year 1 has no
  degradation and year `n` is degraded by `n - 1` full years, compounded.
  App, Monte Carlo, the optimizer and the economics projection already did
  this, but the `breos.solar` production functions added half a year to
  `current_year - start_year`. A direct call at `current_year == start_year`
  lost 0.25% at the default 0.5%/year, and a module `N` years old was
  degraded for `N + 0.5` years where App uses `N`. **Results change only for
  direct `breos.solar` calls that pass both `current_year` and `start_year`
  with a nonzero `degradation_rate`.** At 0.5%/year their DC output rises by
  0.25% at every age: age 10 is scaled by 0.9511 instead of 0.9487. A
  `current_year` before `start_year` now raises `ValueError` instead of
  adding production. App, Monte Carlo and optimizer results are unchanged.
  The convention is documented under Module aging on the PV API page.
- Payback is the sustained discounted payback within the simulated period
  ([#175](https://github.com/Str4vinci/breos/issues/175)): the earliest time
  cumulative discounted savings reach zero or above and stay nonnegative to the
  end of the horizon. `find_payback_year` and `find_payback_year_exact` now
  start the savings at year 0 with minus the investment, taken from
  `attrs["total_investment"]` or, for a projection read back from CSV, from the
  first row of the system cost; both accept an optional `initial_investment`.
  The fractional value interpolates between years 0 and 1 too, so a system
  that pays back after 0.21 years reports 0.21 instead of 1.0. A system whose
  savings turn positive and then negative again after a battery replacement
  pays back at the later recovery, or not at all if the savings end negative;
  it used to report the first crossing. Savings of exactly zero that hold to
  the horizon count as payback. The integer year follows the same rule, and
  the projection's `attrs["payback_year"]` comes from `find_payback_year`.
  **Results change** for App, Monte Carlo and optimizer runs whose savings dip
  below zero after first turning positive, or reach exactly zero; in addition,
  `payback_year_exact` and `Projected_Breakeven_Year_Exact` change for runs
  that pay back within the first year. Other runs, and the App golden
  baseline, are unchanged. The break-even plots mark a first-year payback
  from the year-0 investment and widen the axis to show it. Both functions
  raise `ValueError` when the years, the savings or the investment contain
  NaN or infinite values, instead of reporting NaN or a false crossing.
- Three small fixes from the 0.6.2 audit
  ([#161](https://github.com/Str4vinci/breos/issues/161)).
  `cost_analysis_projection` matches yearly rows to projection years by their
  `Year` labels instead of by position, so rows in a different order give the
  same projection; labels that are not exactly 1 through the projection length
  raise. App results report an undefined LCOE, such as a run with 100% PV
  losses, as `null` instead of `Infinity`, so they pass
  `json.dumps(..., allow_nan=False)`. `load_weather` no longer falls back to a
  historical file that does not cover the requested years, and raises the new
  `AmbiguousWeatherError`, a `ValueError`, when several files match instead of
  taking whichever the directory listed first. **An App run whose `weather/`
  directory holds two TMY files for its location preset now stops** with the
  candidates and asks for `weather_source`. With one file, App results are
  unchanged.

### Removed
- The package root now exposes only the public non-module symbols listed in
  `breos.__all__`; import battery helpers, constants, repair events, catalogue
  values, and weather utilities from their owning modules (`breos.battery`,
  `breos.constants`, `breos.io`, `breos.pv_modules`, `breos.utils`, and
  `breos.weather`). Top-level plotting compatibility attributes are removed;
  import plotting functions from `breos.plotting`. `breos.App` and every
  declared `__all__` export remain available.
- App configuration no longer accepts `dc_coupled`, and
  `provenance.resolved_config` no longer echoes it. Dispatch remains the
  supported DC-coupled/hybrid model; supplying the old key now raises an
  unknown-config-key error.
- Remove `BatteryModelProfile.operating_defaults`, the matching discovery
  JSON field, and serialized `model_profile.operating_defaults` metadata.
  Profiles never supplied operational defaults and the field was always
  empty.
- Remove the exact-alias CO2 fields `co2_avoided_year1_kg` and
  `co2_avoided_total_kg`; read existing `co2_avoided_total_year1_kg` and
  `co2_avoided_total_lifetime_kg` respectively.
- Remove top-level `pv_production_kwh`, monthly/yearly `pv_kwh`, shared annual
  `Legacy_PV_Production_kWh`, and Monte Carlo `mean_pv_production_kwh`.
  Readers can use the existing `usable_ac_system_production_kwh`, annual
  `PV_Production_kWh`, and `mean_usable_ac_system_production_kwh`. The App PV
  values have different definitions; dispatch and retained usable-AC values
  do not change. The timestep ledger's `PV_Production` remains.
- Bump result schema from 1.8 directly to 2.0 for these removals and renames.
- `breos.optimization.DEFAULT_PROJECT_LIFESPAN`, with no deprecation period.
  Nothing read it: the optimizer's horizon comes from
  `simulation.years_projection` or `financials.project_lifespan`, and
  defaults to the same 20 years.
- **`breos.optimize_tilt` and `breos.optimize_battery_size`**, with no
  deprecation period. Nothing called them, and `OptimizationResult` stays. For
  a one-dimensional sweep over tilt or battery size, run `breos sweep` over
  the App key, which prices and projects every point; for a joint search, use
  `optimize_system_multi_objective`.
- **The `results_dir` argument of `optimize_system_multi_objective` and
  `SolarDesignProblem`**, with no deprecation period. It was stored and never
  read. It was the fourth positional argument, so every argument after
  `config` is now keyword-only: a call that passed `results_dir`, `pop_size`
  or a later argument by position raises `TypeError` instead of silently
  binding the next value, and must name them, as in
  `optimize_system_multi_objective(weather, load, config, pop_size=40)`.
- **`breos.calculate_lcoe`**, with no deprecation period. Nothing in BREOS
  called it. The `lcoe_per_kwh` that App, Monte Carlo and the optimizer
  report comes from `calculate_lcoe_from_projection`, which stays. For a
  real-terms LCOE, run the projection with `inflation_rate = 0`, no
  `om_escalation` and a real `discount_rate`; the two functions agreed there
  for a run without a battery replacement, which `calculate_lcoe` left out.
- **`breos.calculate_co2_savings`**, with no deprecation period. Nothing in
  BREOS called it after the projection took over the lifetime CO2.
  `calculate_co2_projection` gives the same kg values per year; pass
  one-element arrays for a single year, pass the export
  (`total_pv_kwh - self_consumed_kwh`) rather than the self-consumption, and
  pass `grid_shift_kwh` as `yearly_grid_shift_kwh`. Read the
  `CO2_Avoided_*_kg` columns and divide by 1000 for tonnes; the intensity is
  in `CO2_Avoided_CI_gCO2_kWh`, and its kind in `CO2_Avoided_CI_Type`. The
  projection's `Grid_CI_gCO2_kWh` column, a copy of
  `CO2_Avoided_CI_gCO2_kWh`, is gone too. No result carried either
  intensity copy: the cost projection, App, Monte Carlo and optimizer
  outputs never did.
- **The first-year input shape of `cost_analysis_projection`**, with no
  deprecation period. Given a per-step results frame and no year rows, it
  estimated later years from year 1 at a fixed self-consumption ratio; App,
  Monte Carlo and the optimizer all pass simulated year rows. The year rows
  are now the first argument, `cost_analysis_projection(yearly_summary_df,
  costs, num_years, ...)`, and `results_df`, `degradation_rate` and `freq`
  are gone. Every argument after `discount_rate` is keyword-only. A call
  such as `cost_analysis_projection(None, costs, yearly_summary_df=rows)`
  becomes `cost_analysis_projection(rows, costs)`; a per-step frame or an
  empty table raises `ValueError`. With the path go
  `breos.economics.replacement_fraction_by_year` and
  `breos.economics.system_ac_production_power` with its
  `SYSTEM_AC_PRODUCTION_COLUMNS`; for usable AC production, sum
  `PV_AC_To_Load`, `PV_Origin_Battery_AC_To_Load` and `PV_AC_Export`.
- **The `production_column` argument of `calculate_lcoe_from_projection`**,
  with no deprecation period, and its inference of the investment from the
  first projection row. It reads `PV_Production_kWh`, and takes the
  investment from `total_investment` or the projection's
  `attrs["total_investment"]`, which `cost_analysis_projection` records;
  without either it raises `ValueError`. A projection read back from CSV has
  no attrs, so pass `total_investment`.
- **The optimizer config aliases `T_Pmax`, `T_Voc` and `T_Isc` in
  `pv.params`, and the `optimization.algorithm` key**, with no deprecation
  period. Name the coefficients `T_Pmax_pct`, `T_Voc_pct` and `T_Isc_pct`,
  as `PVModuleParams` does. `algorithm` accepted only `"nsga2"`, which is
  the only search; remove the key. Both now raise as unknown keys, and the
  example `configs/optimization/projected-optimization.toml` drops
  `algorithm`.
- **`breos.io.export_cost_analysis`, the `extra_metrics` argument of
  `export_summary`, and `breos.utils.get_steps_per_day` and
  `get_steps_per_year`**, with no deprecation period. Nothing called them.
  Write a cost projection with `write_cost_projection`, which writes
  `cost_projection[_<scenario>].csv`, or with `DataFrame.to_csv` for another
  name, or `to_csv(path, sep="\t", index=False)` for the old `txt` format. Add a summary
  field as a column of the summary DataFrame before calling
  `export_summary`, and use `round(24 / get_hours_per_step(freq))` for steps
  per day. The private
  `breos.io._economics_summary_metrics` is gone with them.
- **`breos.resample_tmy_to_15min` and the `freq` argument of
  `fetch_tmy_weather_data`** ([#164](https://github.com/Str4vinci/breos/issues/164)),
  with no deprecation period. `fetch_tmy_weather_data` now always returns the
  hourly PVGIS data; for 15-minute steps, pass them to `resample_to_15min`, as
  App and Monte Carlo already do. The TMY wrapper also kept only the
  irradiance, temperature, humidity and wind columns, capped relative
  humidity at 100% and tagged its output's weather provenance with
  `irradiance_resampling_method = "makima_clear_sky"`; `resample_to_15min`
  keeps every numeric column and records the method it used, such as
  `"makima"`. A call to `fetch_tmy_weather_data` that passed `timezone`,
  `save_to_file` or `use_horizon` by position must name it. Results are
  unchanged.
- **`breos.select_random_year_and_replace_datetime`**, with no deprecation
  period. Nothing called it. `preload_weather_by_year` reads the same complete
  years, keyed by year; draw a key from a seeded `numpy.random.Generator` to
  pick one.
- **`breos.align_load_to_pv`** ([#164](https://github.com/Str4vinci/breos/issues/164)),
  with no deprecation period. It re-stamped the load onto the PV index by
  position and ignored timezones. `simulate_energy_balance` aligns load and PV
  by UTC instant itself, so pass both series to it unaligned. The optimizer
  stopped calling it in 0.3.4.
- **The `num_years` argument of `load_profile`**, with no deprecation period.
  Every caller passed 1, and `load_profile` now always returns one calendar
  year. A call that passed `rlp_directory` or `timezone` by position must name
  them.
- **`breos.InverterConfig`, `breos.INVERTER_PRESETS` and
  `breos.get_inverter_preset`**, with no deprecation period. Nothing in BREOS
  built an `InverterConfig` or read one. With the class go its methods
  `size_from_pv` and `get_cost`; its fields `cost_per_kw_simple` and
  `cost_per_kw_hybrid`, a third copy of the cost presets' inverter prices;
  and the datasheet fields `max_dc_voltage_v`, `max_dc_power_w`,
  `min_mppt_voltage_v`, `max_mppt_voltage_v`, `startup_voltage_v`,
  `max_strings_per_mppt`, `max_input_current_per_mppt_a` and
  `max_short_circuit_current_per_mppt_a`, added in 0.5.0 as groundwork for
  string-aware validation, with their checks. App sizes the inverter from
  `inverter_loading_ratio` or `inverter_ac_rating_kw` and prices it with the
  `inverter_cost_per_kw_hybrid` and `inverter_cost_per_kw_simple` cost keys.
  `calculate_dc_ac_power`, `dc_power_for_ac_output` and
  `InverterConversionResult` stay.
- **The `verbose` argument of the PV production functions**, with no
  deprecation period: `calculate_pv_production_dc`,
  `calculate_pv_production_breakdown`, `calculate_pv_production_dc_tracking`,
  `calculate_pv_production_tracking_breakdown`, `calculate_pv_production_ac`,
  `calculate_multi_array_production` and
  `calculate_multi_array_production_breakdown`. Nothing passed `True`, which
  printed the annual total; sum the returned series instead. A call that
  passed a later argument by position must name it.
- **Unused arguments of three functions**, with no deprecation period:
  `non_negative_cols` of `resample_to_15min`, `temp_ref` of `fit_cec_params`,
  and `start_time`, `end_time` and `freq` of
  `build_battery_temperature_series`. `resample_to_15min` still clips the
  solar and wind columns at zero; a call that passed a later argument by
  position must name it, or `resample_to_15min(df, "makima", None, 41.1, -8.6)`
  now reads `41.1` as the longitude. `fit_cec_params` fits at 25 °C, the
  reference temperature at which `calcparams_cec` reads the parameters and
  the fit normalises gamma, so another value gave inconsistent parameters.
  `build_battery_temperature_series` now requires `temp_config` and `index`,
  which every caller passed. The `fit_cec_params` docstring no longer says
  that `celltype` is unused: it selects the empirical starting guess.
- Optimization config keys that nothing read now raise
  ([#181](https://github.com/Str4vinci/breos/issues/181)): `[load]`,
  `simulation.weather_file`, `simulation.irradiance_resampling`, the
  top-level `name` and `execution_backend` (pass `execution_backend` to the
  function), and the undocumented `[pv_specs]` table, which duplicated
  `pv.params`. The example `configs/optimization/projected-optimization.toml`
  drops them. An inline module in `pv.params` must state `Mpp`, `Vmp`, `Imp`,
  `Voc` and `Isc`; they used to default to one 550 W module's values, and it
  can no longer be named by `pv.module` or `pv_module` too, which was ignored.
  Values the optimizer used to coerce or clamp now raise: a numeric string
  for `constraints.max_tilt_deg`, a whole-number key given as a fraction,
  `early_stop.period`, `min_gen` below 1 or `n_skip` below 0, a string
  `early_stop.ftol`, a negative cost, a non-boolean
  `battery.enable_replacement` or `constraints.enforce_zeb`, and a maximum
  tilt below `constraints.min_tilt_deg`. `evaluate_projected_design` now refuses
  `optimization.objective_basis = "steady_state"`, as the search does.
- Replacement money left the physics layer (ADR 0003 E4,
  [#183](https://github.com/Str4vinci/breos/issues/183)).
  `BatteryConfig.replacement_cost` is removed, and `BatteryConfig` no longer
  prices a pack at a built-in 500 per kWh. `simulate_energy_balance` returns
  a five-tuple, `(results_df, total_pv, summary_df, n_replacements,
  degradation_df)`, without the old fourth element `total_replacement_cost`;
  with `return_degradation_state=True` the state is the sixth. The per-step
  `Replacement_Cost` column is now `Battery_Replaced_Capacity_Wh`, the nominal
  capacity swapped in at that step, and the summary row's `Replacement_Cost`
  is `Replaced_Capacity_kWh` (ledger schema 3.0). `SimulationSummary`
  replaces `total_replacement_cost` with `replaced_capacity_wh`, and
  `cost_analysis_projection` drops its `total_replacement_cost` argument: it
  computes the total and always sets `attrs["total_replacement_cost"]`.
  The optimizer's private `_replacement_event_cost` is gone; use
  `breos.economics.replacement_event_cost`.
- Removed four per-step result columns that repeated another under a second
  name (ledger schema 2.0, ADR 0002 A9). Read `PV_AC_Export` for
  `Sell_To_Grid`, `PV_DC_Curtailed` for `PV_Curtailment`, `Standby_Loss` for
  `Battery_Standby_Loss`, and `PV_Origin_Battery_AC_To_Load` for
  `Battery_AC_To_Load_PV`. The year-row names, such as `Export_kWh` and
  `Battery_Standby_Loss_kWh`, are unchanged, and so is every value.
- Removed the optimizer's `costs.panel_wp` override. It priced the steady-state
  CAPEX at a nominal wattage instead of the selected module's rating, so the
  optimizer carried two CAPEX figures for one design: this is how the budget
  constraint passed designs over budget in
  [#157](https://github.com/Str4vinci/breos/issues/157). The App never accepted
  the key. CAPEX is now always priced at the selected module's `Mpp`, and a
  config that still sets `costs.panel_wp` raises `ValueError` rather than being
  silently ignored. Results are unchanged for configs without it, which
  includes every bundled config and the upcoming publication's.
- Removed the reproduction tooling for the upcoming publication:
  `validation/article1/`, `run-logs/`, `tools/revision/`,
  `tools/validation/recovery/`, the eight `tools/*article1*.py` drivers, and
  their tests. They are preserved in the
  [BREOS 0.6.2 archive](https://doi.org/10.5281/zenodo.22938914); reproduce the
  published numbers from that release. No engine behaviour changes.
- Removed the finished 0.4.x, 0.5.x, and 0.6.x delivery plans and the
  third-party wrapping proposal from `design/architecture/`. The wrapping
  idea stays open as [#11](https://github.com/Str4vinci/breos/issues/11).
- Stopped tracking the generated autosummary stubs under `docs/api/generated/`,
  which Sphinx rebuilds on every run.
- **Numeric load-profile keys and the aliases `h0`, `default` and `crest`**
  ([#182](https://github.com/Str4vinci/breos/issues/182)), with no
  deprecation period. `"1"` is `demandlib_h0`, `"4"`/`"5"`/`"6"` are
  `eredes_btn_a`/`_b`/`_c`, `"7"` is `bdew_h0` and `"8"` is `ree_2.0td`; the
  error names the replacement. `crest` loaded the demandlib H0 profile under
  the name of a different model; use `custom` for a CREST export. The 0.6.2
  reproduction bundle for the upcoming publication uses the numeric keys and
  stays valid on its own release.
- `breos.load_profiles.PROFILE_FILES`, `PROFILE_FILES_15MIN`,
  `PROFILE_FILE_NATIVE_FREQ`, `PROFILE_NAMES`, `PROFILE_ALIASES` and
  `EREDES_COLUMNS`. Read `PROFILES` instead. `breos list load-profiles` drops
  its `aliases` field and gains `files` and `requires_load_profile_file`.
- **The steady-state optimizer scoring basis and `calculate_financials`**
  ([#179](https://github.com/Str4vinci/breos/issues/179)), with no
  deprecation period. `optimization.objective_basis = "steady_state"` scored a
  candidate on one simulated year, with NPV from `calculate_financials` and
  battery replacements extrapolated from the year-one SOH loss. Candidates are
  now scored over the projected lifetime only, and a config that still sets
  `"steady_state"` raises `ValueError`; `"projected"` stays accepted. The
  default projected scoring also ran that year-one pass on every candidate for
  diagnostics, so the `SteadyState_Grid_Independence_%`,
  `SteadyState_NPV_Eur` and `SteadyState_ZEB_Ratio` values and Pareto columns
  are gone, along with the private helpers `_year_one_soh_loss_pct` and
  `_estimate_battery_replacement_treatment` and the constants
  `DEFAULT_PANEL_WP` and `DEFAULT_OBJECTIVE_BASIS`.
  `SolarDesignProblem.projected_objectives` is gone too;
  `objective_basis` remains and is always `"projected"`. Projected results are
  unchanged bit for bit, and each candidate evaluation skips one simulated
  year: 22% faster on a three-year horizon and 4% on twenty years.
- The packaged `breos/data/configs/financials.json`, which nothing loaded and
  which said discount 0.05 against the App's 0.03 (ADR 0003 E6,
  [#186](https://github.com/Str4vinci/breos/issues/186)).
- `breos.plotting.plot_tariff_comparison` and `plot_tariff_comparison_manual`
  ([#186](https://github.com/Str4vinci/breos/issues/186)), with no
  deprecation period. Nothing called them, and they expected hand-built
  `Tariff`/`Net Cost` tables that BREOS never produced. Compare tariffs with
  `breos sweep` over the `[tariff]` table and read its year-1 money columns,
  or loop over `App` runs. The "Compare tariffs" recipe shows both, and
  `configs/examples/tariff-comparison.toml` is the CLI version.
- Re-exports and test-only helpers in the battery and degradation modules
  ([#186](https://github.com/Str4vinci/breos/issues/186)), with no
  deprecation period. Import `EXECUTION_BACKENDS` from `breos.execution`,
  not `breos.battery`, and `BlastExperimentalRangeWarning` and
  `BlastAgingHorizonWarning` from `breos.degradation.validation`, not
  `breos.degradation.engine`. A pytest `filterwarnings` entry such as
  `ignore::breos.degradation.engine.BlastExperimentalRangeWarning` now stops
  the test with an `AttributeError`; name `breos.degradation.validation`
  instead. `breos.degradation.results.build_degradation_summary` is gone:
  `build_degradation_summary_from_state` returns the same dict and reads the
  BLAST warnings from the carried lifecycle state. These are gone without a
  replacement: `breos.degradation.protocol.DegradationProvenance`,
  `resolve_degradation_provenance` and `warning_records_from_snapshot`;
  `apply_battery_profile_defaults` (from `breos.degradation.profiles` and
  `breos.app_config`) and `merge_battery_config_layers`, which merged an
  always-empty model-profile layer;
  `breos.degradation.engine.P1_BLAST_MODEL_KEYS`;
  `BlastWarningCollector.from_snapshot`, which only called the constructor;
  and the `warnings()` and `provenance()` methods of `DegradationLifecycle`
  and its native and BLAST adapters. `DegradationLifecycle` is no longer
  `runtime_checkable`, so `isinstance(x, DegradationLifecycle)` raises
  `TypeError`. These re-exports are gone; import them from
  `breos.degradation.profiles`: `CORE_BLAST_MODEL_KEYS` from
  `breos.degradation.engine`, and `BLAST_STATE_SCHEMA_VERSION` and
  `get_battery_model_profile` from `breos.degradation.protocol`.
  Configuration still resolves as user values over the global defaults.
  Results are unchanged bit for bit.
- **Plotting functions that nothing called**
  ([#186](https://github.com/Str4vinci/breos/issues/186)), with no
  deprecation period, from `breos.plotting` and the top-level `breos`
  namespace: `plot_validation_soh_comparison`, `plot_validation_residuals`,
  `plot_validation_parity`, `plot_validation_multi_system`,
  `plot_validation_degradation_split`, `plot_calendar_aging_sensitivity`,
  `plot_breakeven_two`, `plot_tilt_optimization`, `create_cost_plots`,
  `plot_azitilt_landscape_3d` and `monthly_graphs`. Use
  `plot_monthly_comparison` for `monthly_graphs`: it draws the same monthly
  PV, load, import and export bars, while `monthly_graphs` could put the
  wrong legend label on a bar when a column was missing. Use
  `plot_breakeven_comparison` for `plot_breakeven_two`. `plot_timeseries`
  drops its `title` argument, which it never drew. Every remaining plot
  draws the same figure as before.
- **The run-year Monte Carlo plots**
  ([#186](https://github.com/Str4vinci/breos/issues/186)).
  `plot_montecarlo_cost_overlay`, `plot_montecarlo_soh_overlay` and
  `plot_montecarlo_soh_traces` read `run_number`/`year` tables that BREOS
  never writes. They are gone with the branch of `plot_montecarlo_simulation`
  that called them. `plot_montecarlo_simulation` now takes the
  one-row-per-run table as its first argument,
  `plot_montecarlo_simulation(result.runs, results_directory)`. The unused
  `all_data` positional and the `full_df` keyword are gone. It no longer
  looks for `monte_carlo_results.csv`, `combined_results.csv` or
  `monte_carlo_degradation_details.csv` in the directory, and any other
  table raises `ValueError`. `plot_montecarlo_npv_distribution` and
  `plot_montecarlo_grid_independence_distribution` read only `npv_savings`
  and `mean_grid_independence_pct`. `breos montecarlo --plots` writes the
  same seven figures as before.
- **Unmaintained tools** under `tools/`: `batch_compare_locations.py`,
  `recalculate_economics.py`, `compare_results.py`, `azitilt_optimizer.py`,
  `validate_cec_fit.py` and `parity/bundle_compare.py`, with their tests.
  They kept their own cost fallbacks, year loops and config formats apart
  from the App's, read files that no runner writes, or, for
  `validate_cec_fit.py`, needed `nrel-pysam`, which BREOS no longer
  installs. `parity/bundle_compare.py` compared Monte Carlo cases with
  reference bundles that are not in the repository. Compare designs and
  orientations with `breos sweep`. `fetch_historical_weather.py` is gone
  too: `python tools/fetch_weather.py historical --location <key> --start
  <year> --end <year>` fetches the same Open-Meteo years for one location
  preset; add a new site with `tools/add_location.py` first.
- **`configs/base/`**. Its `costs`, `emissions`, `locations` and
  `electricity` files were copies of the packaged presets that no run read,
  and its `financials.json` had lost its packaged counterpart. `breos list`
  and the Packaged options page show the presets. `tools/add_location.py`
  now adds to `breos/data/configs/locations.json` in the checkout, where the
  `location` key finds it when BREOS runs from that checkout; regenerate the
  options page with `tools/generate_option_docs.py` afterwards. For a
  one-off site, set `location` to a latitude/longitude/timezone table.
  `tools/fetch_weather.py` reads the packaged presets. The packaged
  `breos/data/configs/electricity.json`, which nothing loaded, is removed as
  well ([#186](https://github.com/Str4vinci/breos/issues/186)), and so is
  the unread `electricity_cost_excl_vat` field of the packaged cost presets.
- **The `validation` and `location-tools` extras.** `validation` installed
  nothing. `location-tools` only served `tools/add_location.py`, which is not
  in the wheel; install `geopy` and `timezonefinder` to run it.
- **`plot_grid_independence_heatmap` and `plot_location_comparison_delta`**
  ([#186](https://github.com/Str4vinci/breos/issues/186)), with no
  deprecation period. They took hand-built pivot tables, and only the removed
  `tools/batch_compare_locations.py` built them. Use `plot_sweep_heatmap` on
  a `breos sweep` result, with `diff=` for the difference between two sweeps.
- **`plot_azitilt_landscape_2d` and `plot_azitilt_ew_1d`**
  ([#186](https://github.com/Str4vinci/breos/issues/186)), with no
  deprecation period. Only the removed `tools/azitilt_optimizer.py` called
  them. Use `plot_orientation_landscape` on a tilt × azimuth sweep, or on a
  tilt sweep of an east-west roof.
- **`plot_pareto_front_analysis`**
  ([#186](https://github.com/Str4vinci/breos/issues/186)), with no
  deprecation period. It read `Tariff`, `Detailed_Strategy` and
  `Consumption_kWh` tables that BREOS never produced. Use
  `plot_pareto_front` on the optimizer result or on a sweep.
- **Unused energy-balance arguments**
  ([#186](https://github.com/Str4vinci/breos/issues/186)), with no
  deprecation period. `simulate_energy_balance` and
  `simulate_energy_balance_summary` drop `results_directory`, which they never
  read, and `debug`, which only printed; `update_battery_soh_cyclewise`,
  `update_battery_soh_calendar`, `update_battery_resistance_cyclewise` and
  `update_battery_resistance_calendar` drop `debug` too.
  `update_battery_soh_cyclewise` also drops `nominal_energy_Wh`, which it
  discarded on entry. Every argument after `temperature_series`, and after
  the SOC series of `update_battery_soh_cyclewise`, is now keyword-only, so
  an old positional call raises `TypeError` instead of passing a capacity as
  `fec_cum` or a directory as `initial_fec`.
- **The extrema cycle counter**
  ([#164](https://github.com/Str4vinci/breos/issues/164)), with no
  deprecation period: `update_battery_soh_cyclewise(use_rainflow=False)`,
  and `detect_half_cycles_from_soc_series` and `detect_cycles_rainflow` from
  `breos.battery` and the top-level namespace. Nothing passed `False`, and it
  counted a different quantity, not a second estimate: 1.2 FEC for an SOC
  excursion that rainflow counts as 0.6. The energy balance counts cycles
  with the incremental rainflow counter in `breos.degradation.protocol`, and
  `update_battery_soh_cyclewise` always uses rainflow. Cycle dicts passed to
  `update_battery_resistance_cyclewise` must carry `count`, as every rainflow
  cycle does; a missing one used to be read as a full cycle. BREOS has no
  public replacement for the two detectors. To count cycles outside a
  simulation, call `rainflow.extract_cycles` on the SOC series directly,
  which is what `detect_cycles_rainflow` wrapped. The wrapper also built the
  dicts that `update_battery_resistance_cyclewise` takes: from each
  `(range, mean, count, i_start, i_end)` tuple of an SOC series in percent,
  `doc` is `range / 100`, `count` is `count`, and `mean_c_rate` is `doc`
  divided by the hours from step `i_start` to step `i_end`. To age a pack
  from a whole SOC series, `update_battery_soh_cyclewise` still counts its
  cycles with rainflow.
- `breos.battery.k_c_rate_R` and `k_doc_R`, also in the top-level namespace
  ([#186](https://github.com/Str4vinci/breos/issues/186)). They were the
  resistance-growth factors of `update_battery_resistance_cyclewise`, which
  now computes them inline, as the capacity model does. The four
  `NAUMANN_LAM_FIELD_CALIBRATED_V1_*` constants repeated the
  `NAUMANN_LAM_FIELD_CALIBRATED_*` ones; the `"naumann_lam_field_calibrated_v1"`
  calendar model stays, with the same parameters.
- **Selectors with one legal value**
  ([#186](https://github.com/Str4vinci/breos/issues/186)), with no
  deprecation period. `BatteryConfig.battery_type` and
  `breos.battery.SUPPORTED_BATTERY_TYPES` accepted only `"lfp"`, the
  chemistry the native model always used; the lifecycle adapter drops its
  `battery_type` and `nominal_energy_wh` arguments with them. `App` no longer
  has a targeted error for `battery_type` and rejects it as an unknown key,
  and the config reference no longer lists it. `BatteryConfig.dc_coupled`
  and the `breos run --dc-coupled` flag could only say `True`: the flag was a
  `store_true` switch on a key that defaults to `True` and raises on `False`.
  The App key `dc_coupled` stays, so `resolved_config` keeps it, and
  `False` still raises, with an error that no longer names 0.3.x. The
  optimizer's `battery` table no longer accepts `battery_type` or
  `dc_coupled`.
- `list_battery_models(enabled_only=...)`, which filtered nothing because
  every registered model is enabled, and `SimulationSummary.has_battery`,
  which nothing read ([#186](https://github.com/Str4vinci/breos/issues/186)).
  Results are unchanged bit for bit for every removal above.
- Pass-through helpers in the App runner, configuration and execution
  modules, with no deprecation period. `breos.runners` no longer re-exports
  `run_app_simulation` and `SimulationArtifacts`; import them from
  `breos.runners.app`. `breos.app_config.load_json` only called
  `breos.resources.load_config_json`; call that instead.
  `breos.execution.backend_provenance` drops its `jit_cache_states`
  argument: a `numba` record's `jit_cache` is `"unknown"` until the caller
  sets it, for example to `aggregate_jit_cache_states(states)`. The App
  runner functions take the resolved configuration without the `cfg` dict it
  already holds: `run_app_simulation(resolved, deps)`,
  `revalue_app_simulation(resolved, artifacts, deps)` and
  `breos.app_results.build_result(resolved, artifacts)`. The orientation
  defaults are worked out once, by the new `resolve_orientation(cfg, lat)`:
  `resolve_pv_system(cfg, *, tilt, azimuth, axis_azimuth)` takes its result
  and returns six values instead of eight, without the tilt and azimuth;
  `resolve_tracking(cfg)` returns only the tracker mode; and
  `normalise_pv_arrays` takes the angles as keyword-only arguments.
  `load_consumption_profile` requires its `timezone` instead of falling back
  to UTC. Results are unchanged bit for bit.

### Documentation
- The release checklist records that `v0.5.0`, `v0.5.1` and `v0.6.0` are
  lightweight tags, while every other release tag is annotated, and says to
  leave them in place rather than re-push them
  ([#185](https://github.com/Str4vinci/breos/issues/185)). The release flow
  now creates annotated tags.
- A "Compare tariffs" recipe: the same system under several offers, from
  `breos sweep` over whole `[tariff]` tables or from a loop over `App` runs,
  read through the year-1 bill, the project-long cost with the system, and
  `npv_savings`, which ranks where a system pays most rather than which offer
  is cheapest. New example `configs/examples/tariff-comparison.toml` compares
  a simple, a bi-hourly and a tri-hourly offer with and without a battery. The
  configs README lists the three tariff examples.
- A generated configuration key reference,
  `docs/getting-started/config-reference.md`, lists every top-level App key
  and every key of `[costs]`, `[battery_indoor_model]`, `[[pv_arrays]]`,
  `[tariff]` and `[smart_charging]`, with its default, CLI flag and allowed
  values ([#181](https://github.com/Str4vinci/breos/issues/181)). Each
  registry entry (`AppConfigField.doc`, `TableSpec.docs`) now carries its
  description, and `tools/generate_config_docs.py --check`, run by the test
  suite, fails when the page and the registry disagree. It replaces the
  hand-written key table in Configuration, which had drifted: it lacked
  `enable_resistance_fade`, the three `load_profile_*` keys for custom
  profiles and the `montecarlo` and `sweep` sections, and still gave
  `load_profile` the removed default `"1"`.
- Removed `rlp/README.md` from the repository and the source archive. It
  still listed the numeric load-profile keys that 0.7.0 removes; the
  load-profile data page documents external profiles and `rlp_directory`.
  The plotting API page says which functions write files and which return
  the figure, and names the `plots` extra.
- `configs/examples/pv-plus-battery.toml` no longer claims to show every key:
  it sets 26 top-level keys, about a third of those BREOS accepts. It and the
  configs README now point to the configuration key reference for the full
  list.

## [0.6.2] - 2026-09-24

### Added
- Added `tools/revision/task3_eol_sweep.py`, the Task 3 end-of-life sweep. It
  quantifies how much of storage economics follows from the battery replacement
  assumption, by running three degradation models at their accepted knee
  orientations against 80%, 70% and replacement-disabled settings. The sweep
  previously existed only inside a result directory and could not be re-run from
  a later release. The pinned commit, lattice location and output directory are
  now options; the models, the 210-design grid, the candidate replays and the
  verification against the accepted lattice are unchanged, and the config stays
  pinned by hash. `--candidates` runs the sweep against the
  `reference_candidates` of an optimization config instead of the pinned
  five-configuration set; omitting it reproduces the pinned run exactly, and
  the run records which set it used in `candidate_set_source`.
- Added `tools/validation/recovery/dkasc/dkasc_ladder_chain.py`, which reruns
  the DKASC loss ladder under Hay-Davies, Perez, and Perez with the Marion
  diffuse IAM, so the shift between the published validation configuration and
  the case-study chain splits into its transposition and IAM parts.
  `run_chain` in `dkasc_validate.py` now accepts `diffuse_iam`.

### Changed
- The forthcoming publication's reproduction configurations now use the accepted
  representative candidate set rather than the superseded manuscript set, and
  that set moves from five configurations to six, each answering a distinct
  selection criterion verified against the domain-corrected exhaustive
  enumeration:

  | | design | orientation | criterion |
  | --- | --- | --- | --- |
  | C1 | 6 PV, no battery | 30/200 | maximum NPV |
  | C2 | 8 PV, 7 kWh | 35/200 | largest battery paying back before replacement |
  | C3 | 9 PV, 9 kWh | 35/195 | knee, nearest the utopia point |
  | C4 | 9 PV, 13 kWh | 45/190 | maximum grid independence with positive NPV |
  | C5 | 9 PV, 20 kWh | 50/185 | maximum grid independence |
  | C6 | 4 PV, no battery | 35/180 | off-front low-investment benchmark |

  C1 and C3 are unchanged. C2 gains a criterion the previous set did not express
  and drops from nine modules to eight. C4 is new, so the previous C4 and C5
  become C5 and C6. Each candidate's expected values now come from the accepted
  lattice, so reference columns compare against the accepted front. Results and
  write-ups that refer to C4 or C5 by index predate the renumbering.
- Both publication configurations replace the absolute 4352 W battery power cap
  with a symmetric 1.0 C capacity-proportional limit. The absolute cap did not
  scale with pack size and so under-limited small packs; 1 C is the BYD
  Battery-Box Premium HVS family rating at every size. Results generated before
  this change reproduce the superseded candidate set and need rerunning.
- Renamed two external-validation packages after the institutions that
  produced their data: `sandia_task13/` is now `iea_pvps_task13/` (IEA PVPS
  Task 13, measured at SUPSI PVLab) and `pcoe/` is now `ucy_phaethon/` (the
  PHAETHON Centre of Excellence, University of Cyprus). Archived output
  packages and recorded input manifests keep their original names so they
  still match the hash-verified archive.
- `CITATION.cff` and the README now cite the software itself instead of the
  SSRN preprint (`10.2139/ssrn.7032064`). The preprint's methods and results
  predate this release.

### Fixed
- Battery replacement outlays are booked at the instant the pack is swapped
  rather than at a calendar-year boundary. The events were aggregated into a
  replacement year and inflated by `(1 + inflation) ** (year - 1)`, valuing the
  outlay at the start of that year, then discounted as a year-N flow, valuing it
  at the end: inflated to one end of the year and discounted from the other,
  with neither matching the swap. The simulation already records the swap step,
  so the App, Monte Carlo and optimization year summaries now report
  `Replacement_Year_Fraction` and both projection paths apply the resulting
  project time to the inflation and the discount exponent alike. LCOE discounts
  the same outlay from the same instant, and the projection reports the resolved
  time in a new `Replacement_Time_Years` column. A summary carrying no fraction
  falls back to mid-year. Present value of each replacement rises by a factor
  between `1 + inflation_rate` and `1 + discount_rate`, so NPV and LCOE move for
  any run that replaces a pack; a run that never replaces is bit-identical.
  Results generated before this change carry the superseded booking.
- The test suite passes on numpy 2 and on macOS. The BLAST parity fixture
  check compares floats at `atol=1e-12` instead of bitwise, since numpy 2
  moves one fixture value by one ULP. The libm `pow` guard in the Numba
  dispatch tests finds a platform-specific discriminating input at run time
  instead of pinning a glibc-only literal.

## [0.6.1] - 2026-09-03

### Added
- Added `battery_power_limit_c_rate` (`--battery-power-limit-c-rate`), a
  capacity-proportional alternative to the absolute
  `battery_max_charge_power_w` and `battery_max_discharge_power_w` limits.
  Combining the C-rate limit with either absolute limit raises an error. The
  existing absolute limits remain the default.
- Added optional year-by-year weather sequences to
  `evaluate_projected_design`. Repeated-TMY evaluation remains the default.
- Added opt-in AC-side and DC-side PV-output scale factors for measured-bias
  sensitivity studies. Both factors default to no scaling.
- Added annual battery charge throughput, discharge throughput, mean state of
  charge, and all-pack full-equivalent-cycle fields to projected result
  ledgers. The installed-pack cumulative FEC field remains unchanged.

### Changed
- Projected fixed-design evaluation now accepts zero PV modules. This lets an
  exhaustive sizing grid evaluate its battery-only boundary.
- Source distributions now use an explicit include list. Files that happen to
  be present in a maintainer's checkout are no longer published.

### Fixed
- Applied optional AC-output scaling consistently in scalar, vectorized, and
  Numba inverter paths, including inverse conversion and optimization runs.
- Rejected an AC-output scale above 1. Delivered AC cannot exceed the inverter
  nameplate or the DC input.
- Fixed projected-design evaluation with a weather sequence.
- Removed duplicate annual energy columns from projected physical and
  financial ledgers. The join now rejects inconsistent duplicate values.

## [0.6.0] - 2026-08-31

### Added
- Added a projected multi-objective optimizer that evaluates each candidate
  over a repeated-TMY project lifetime, carries battery degradation and
  physical state between years, records actual replacements, and optimizes
  lifetime grid independence and NPV. This is now the default objective basis;
  projected ZEB is a diagnostic or optional feasibility constraint, not a third
  objective.
- Added a version-controlled configuration for the forthcoming publication, a
  deterministic fixed-candidate reproduction command, per-candidate yearly and
  financial source tables, an opt-in licensed-profile regression, input
  preflight, a complete result-bundle verifier, and a manuscript-to-source-data
  audit. Generated provenance records the resolved PV module and geometry as
  well as software, source, config, input, and output hashes.
- Added `evaluate_projected_design` for detailed evaluation of a fixed design.
  It returns the projected metrics, annual energy and degradation-state
  ledger, discounted financial ledger, LCOE, and configured lifetime
  avoided-emissions totals without requiring an NSGA-II run.
- Added reproducible Monte Carlo controls for normal or bounded-uniform demand
  sampling, historical weather-year bounds, energy-conserving 15-minute
  interpolation, and independent trajectory workers. Opt-in yearly output
  exposes the energy, degradation, and discounted-cost paths behind summary
  distributions, and the CLI writes a provenance report with input and output
  hashes. Existing normal sampling and aggregate-only output remain defaults.
- Added public battery-temperature and indoor-temperature-model configuration
  fields. Deterministic App and Monte Carlo runs now use the same resolved
  temperature input while retaining the ambient-weather default.
- Added a Monte Carlo configuration and BREOS orchestration command for the
  forthcoming publication's C1-C5 study. It pins the manuscript's uniform
  0.95-1.05 load multiplier and records that the archived research
  implementation instead used a normal draw.
- Added `tools/run_article1.py` as the single entry point for input checks,
  deterministic analyses, Monte Carlo runs, and final bundle verification.
  It discovers ignored local inputs under `dev/article1-inputs/` by default.
- Added opt-in hourly-energy conservation to `resample_to_15min`. It preserves
  each source hour's GHI, DNI, and DHI energy after clear-sky interpolation;
  energy conservation remains opt-in for general runs.
- Added a summary-only simulation path that returns annual energy totals,
  grid independence, degradation state, replacements, carried battery and
  PV-origin energy, and temperature diagnostics without materialising a
  per-timestep results frame. Monte Carlo runs through it.
- Added an optional Numba dispatch backend for Monte Carlo, selected with
  `[montecarlo].execution_backend = "numba"` and installed with
  `pip install "breos[fast]"`. It compiles one day of dispatch at a time at
  fixed state of health and resistance; rainflow counting, degradation,
  resistance growth, and replacement stay in the Python reference path.
  Selecting it without Numba installed fails before any trajectory starts.
  `"python"` remains the default and the numerical reference. Provenance
  records the resolved backend, whether the JIT cache was warm or cold for
  the run, and the installed Numba and llvmlite versions.
- Added `breos --execution-backend` to the `montecarlo` command and to the
  reproduction tool for the forthcoming publication study, so a run can
  select the accelerator without editing the pinned manuscript configuration.
- Added Monte Carlo yearly diagnostics needed to compare execution paths field
  by field: separate direct-PV and battery inverter losses, charge and
  discharge input and losses, standby loss, capacity-window loss, replacement
  energy removed and added, carried battery and PV-origin energy, and the
  within-year timestep indices at which the pack was replaced.

### Changed
- `optimization.objective_basis` now defaults to `"projected"`. Multi-objective
  sizing scores each candidate over the full project lifetime, with PV
  degradation, propagated battery state, and actual replacement events, and
  optimizes two objectives: lifetime grid independence and lifetime NPV. The
  previous single-year basis remains available as
  `optimization.objective_basis = "steady_state"`, which keeps the annual
  three-objective search with ZEB ratio as a third objective. Runs that relied
  on the implicit annual default now cost `years_projection` simulated years
  per candidate and return a two-objective front.

- Open-Meteo historical weather downloads now accept an explicit radiation
  time basis. Preceding-hour means remain the default, and callers can request
  the provider's instantaneous fields. Saved metadata records the selected
  fields, label convention, and time basis.
- Weather metadata can now drive solar-position timing. The forthcoming
  publication applies the exact PVGIS SARAH3 irradiance offset and relabels
  Open-Meteo's right-labelled preceding-hour means before energy-conserving
  disaggregation.
- Skip daily degradation work when battery capacity is zero, and keep the
  native degradation loop on NumPy arrays instead of rebuilding pandas objects
  for every simulated day. Battery simulations retain the same degradation
  calculations. PV-only simulations no longer return degradation rows for a
  battery that does not exist.
- Removed the manuscript-specific measured, PVsyst, and Polysun comparison
  CSVs from the public reproduction workflow. The input preflight and bundle
  verifier now cover only inputs and results needed to run BREOS; third-party
  comparison output remains in the private study archive.
- `remap_datetime_index_years` now shifts tz-naive and fixed-offset indices
  without a Python-level pass over the index, about 43x faster on a 15-minute
  year. Indices under a zone that can have offset transitions keep the
  element-wise path, which remains the authority; the fast path is bit-exact
  where it applies.
- Updated the Suntech STP550S-C72/Vmh catalogue temperature coefficients to
  the matching monofacial datasheet values of -0.34 %/°C for maximum power and
  -0.26 %/°C for open-circuit voltage. The configurations for the forthcoming
  publication also record the 1.134 m by 2.278 m module frame explicitly.
- Corrected power-to-energy aggregation in monthly and annual plotting helpers
  by applying the inferred timestep duration independently of pandas' internal
  datetime resolution. Hourly plots retain their prior values; 15-minute
  energy plots are no longer four times too large.
- The bundled hourly demandlib H0 profile now labels its watt-valued column as
  watts. The loader still accepts the historical kilowatt header for external
  compatibility.
- The configurations for the forthcoming publication study now model the
  battery's thermal environment
  rather than pinning it: battery temperature follows the weather and the
  indoor model buffers it, matching a pack installed indoors.
  `validation/article1/no-thermal-model/` keeps the flat 25 °C pair so the
  pinned assumption can still be read on its own.
- Withdrew the announced 0.6.0 removal of the `breos[fast]` extra. The extra
  is retained and undeprecated because it now installs the dependency for the
  optional dispatch backend.

### Fixed
- Kept PV module geometry metadata for the forthcoming publication out of the
  strict BREOS runtime configuration, and validated every Monte Carlo case
  before starting any trajectories.
- Filled the final three 15-minute slots produced by Makima weather
  interpolation by holding the last source-hour state instead of returning
  NaNs that downstream simulation silently treated as zeros.
- Made clear-sky-index interpolation use the same low-light stabilizer during
  division and reconstruction, avoiding systematic low-light attenuation.
- Reject fractional-hour timezone row rolls when coercing PVGIS TMY data to a
  sample year, because pvlib's interface accepts only whole-hour rolls. This
  replaces silent 15-30 minute irradiance misalignment with an actionable
  error.
- Normalize Perez coefficient-set names consistently with the other PV model
  selectors.
- Reject the unimplemented `max_self_consumption` tilt objective and raise when
  every tilt evaluation fails instead of silently returning the first angle.
- Warn when the CEC fit returns a best-residual physical solution that does not
  meet the requested gamma tolerance.
- Fixed plotting legend ordering and multi-scenario break-even x-axis limits.
- Removed an unreachable global-degradation plot branch that checked column
  names no BREOS simulation produces. The supported per-battery degradation
  plot continues to use the production cumulative-degradation columns.

### Removed
- Removed the APIs deprecated for 0.6.0: the standalone
  `breos.numba_kernels` screening module, the documentation-derived Polysun
  comparison baseline, nine unverified plotting helpers, and 17 orphaned
  helpers across `battery`, `io`, `optimization`, `solar`, `utils`, and
  `weather`. The `breos[fast]` extra remains available for the compiled dispatch
  backend. The archived
  [v0.5.2 migration guide](https://github.com/Str4vinci/breos/blob/v0.5.2/docs/deprecations.md)
  lists replacements for the removed helpers.

## [0.5.2] - 2026-08-19

### Added
- Add a validated `[costs]` override table with explicit user override → named
  preset → `CostParams` default precedence, plus dotted `[sweep]` keys such as
  `costs.electricity_cost` for economic sensitivity runs. Unknown cost keys and
  invalid values now fail with actionable errors; existing flat configurations
  retain their previous resolved costs.
- Added explicit terrain-horizon provenance to weather inputs. PVGIS TMY
  fetching now accepts `use_horizon=False` for explicitly unshaded irradiance
  while preserving the historical provider-horizon default. Automatically
  saved weather CSVs gain digest-bound `.csv.metadata.json` sidecars; missing,
  legacy, malformed, or stale sidecars load conservatively with an `unknown`
  horizon status rather than inferring treatment from the filename.
- Added opt-in App `horizon_profile` shading from inline azimuth/elevation
  pairs. BREOS circularly interpolates the terrain line, removes direct beam
  while the sun is on or below it, and records the normalized profile and
  shaded-timestep count in weather provenance. Fresh PVGIS requests
  automatically disable its provider horizon; already-shaded or unknown-
  provenance weather is rejected to prevent silent double-counting.

### Removed
- Removed the repository-only `tools/analyze_results.py` scratch script and
  the superseded two-folder `tools/compare_two_results.py` helper. The supported
  `tools/compare_results.py` command handles two or more result folders and no
  longer depends on a nonexistent private plotting palette.

## [0.5.1] - 2026-08-11

### Changed
- Centralized App configuration metadata in one declarative registry that now
  derives defaults, allowed top-level keys, CLI options, and CLI override
  handling. Historical ordering, aliases, normalization, config-file
  precedence, validation messages, and simulation results are preserved.
- Clarified that the deprecated article lifetime baseline is an independent
  approximation reconstructed from public documentation. Its generated plot
  legends now say "documentation-derived baseline" instead of presenting the
  series as Polysun output; calculations and compatibility APIs are unchanged.
- Updated branch protection on `develop` and `main` so the existing Python
  3.14 CI matrix job is required alongside Python 3.11–3.13. Python 3.14 had
  run successfully since June, but the manually maintained required-check list
  had not been updated when that matrix entry was added.

### Deprecated
- Deprecated the unused `breos.numba_kernels` module and `breos[fast]` extra;
  the standalone approximate kernels are not used by `App` or the supported
  simulation path and are scheduled for removal in 0.6.0. Both removals were
  later withdrawn: the extra is undeprecated and the module is retained while
  staying deprecated. See Unreleased.
- Deprecated the article-scoped, documentation-derived Wöhler/Miner comparison
  subsystem, its three plots, and its comparison-only constants for removal in
  0.6.0. Despite legacy API names, this is an independent BREOS approximation,
  not Polysun or PerMod source code or a validated reproduction of Polysun.
- Deprecated six uncalled, undocumented optimization/leave-one-out plots and
  orphaned helpers across `battery`, `io`, `optimization`, `solar`, `utils`,
  and `weather` for removal in 0.6.0. Functions keep their signatures and
  behaviour throughout 0.5.x and emit `DeprecationWarning` only when called;
  see the [v0.5.2 deprecation guide](https://github.com/Str4vinci/breos/blob/v0.5.2/docs/deprecations.md)
  for the complete inventory and migration paths.

### Fixed
- Accept the optional `[sweep]` config section in `ALLOWED_CONFIG_KEYS`, so
  `breos validate-config configs/examples/sweep.toml` no longer rejects a
  shipped example that `breos sweep` runs successfully. The documented
  behaviour was already that `[sweep]` and `[montecarlo]` are recognised; only
  `[montecarlo]` actually was. `validate-config` now also rejects an empty or
  malformed sweep grid, and every `configs/examples/*.toml` is covered by a
  regression test.
- Made the public `resistance_to_efficiency()` helper match the live
  resistance-fade path: both one-way efficiencies receive the same
  `sqrt(1 + growth)` derating, preserving configured charge/discharge
  asymmetry and removing the helper's former artificial floor. Initial and
  daily simulation paths now call the helper; simulation results are
  unchanged because they already used this mapping.

## [0.5.0] - 2026-08-05

### Added
- Added an explicit recommended rooftop PV example and a reproducible
  seven-site equal-nameplate comparison of 3×400 W monofacial, 2×600 W
  bifacial front-only, and 2×600 W bifacial rear-gain configurations. The
  report separates module-parameter differences from modeled rear gain and
  states the geometry, inverter loading, and front-shading limitation.
- Added selectable `iam_model` optics to the App config, `--iam-model` CLI
  flag, and every public solar-chain function. `"ashrae"` remains the
  bit-for-bit historical default; `"physical"` and `"martin_ruiz"` expose
  pvlib's respective defaults. When `diffuse_iam="marion"` is enabled, the
  same selected IAM model is integrated for the sky and ground components.
- Added named SAPM temperature presets for the four pvlib/Sandia construction
  and mounting combinations, plus strict opt-in `temperature_model="noct-sam"`
  support. SAM NOCT refuses to run without sourced NOCT and module-efficiency
  metadata; the bundled module catalog has no sourced NOCT yet, so no catalog
  entry is activated by default or by implication.
- Added optional, validated `PVModuleParams.bifaciality` metadata and a sourced
  maximum-power bifaciality value for the generic 600 W bifacial catalog entry.
  The metadata alone does not activate rear-gain modeling or change production.
- Added opt-in `bifacial_model="infinite_sheds"` rear-gain modeling for fixed,
  tracking, and mixed multi-array systems. The App/CLI path requires explicit
  row height and pitch plus sourced module bifaciality. Rear irradiance feeds
  both DC power and the cell-temperature model, following pvlib's
  `poa_front + poa_back * bifaciality` convention, so rear gain is not credited
  with power but no heat. With the default `bifacial_model="none"` the rear
  term is zero and front-side irradiance, cell temperature, and DC production
  are bit-for-bit unchanged. That guarantee covers the rear-gain model only,
  not the `gcr` forwarding change noted below.
- Added bifacial configuration and rear-gain diagnostics to the PV loss
  waterfall and result provenance, a runnable ground-mount example, and paired
  front/rear regression benchmarks across the seven-site validation matrix.
- Added optional `InverterConfig` datasheet limits: absolute maximum DC voltage
  and power, the MPPT operating window and startup voltage, per-MPPT operating
  and short-circuit current limits, and maximum parallel strings per MPPT. Each
  is validated on its own and against the others when supplied — the MPPT
  window must not be inverted and must sit within the DC voltage ceiling. Only
  that physical ceiling constrains startup voltage; it is deliberately allowed
  outside the MPPT window, because real datasheets quote a startup well below
  the MPP range minimum. The fields default to `None`, are not read by any
  model yet, and change no result.

### Changed
- Bumped `provenance.ledger_schema_version` to `1.1`. The result schema gains
  the `bifacial_rear_gain` PV loss-waterfall stage and the `provenance.pv_model`
  block, and the `iam` stage is relabelled to name the front side explicitly.
  The additions are backward compatible; consumers that pin the value need to
  accept `1.1`.
- Multi-array systems now honour the top-level `gcr` when placing per-array
  tracking geometry. `calculate_multi_array_production_breakdown` takes a
  function-level `gcr` and the App/CLI path forwards the configured `gcr`;
  previously per-array tracking fell back to a hardcoded `0.35` and a
  non-default top-level `gcr` never reached it. Existing multi-array tracking
  configurations that set a non-default top-level `gcr` without a per-array
  `gcr` now backtrack with different row geometry and produce slightly
  different production. Set `gcr` explicitly on each entry in `pv_arrays` to
  keep the previous geometry. Single-array systems and per-array `gcr`
  overrides are unaffected.
- `App` and CLI configuration now reject a `gcr` outside `(0, 1]`, including
  non-finite and null values and every explicit `pv_arrays[i].gcr` override,
  instead of passing it to pvlib. pvlib does not reject a nonsensical ratio on
  the tracking path; it quietly derives a different backtracking rotation, so a
  mistyped `3.5` previously returned roughly half the annual energy with no
  error. Configurations that were already invalid for another reason keep
  reporting that error. Tracking configurations with an out-of-range `gcr` that
  used to run and produce wrong numbers now fail at `App()` construction. This
  guards the config path only; callers going directly to `breos.solar` tracking
  functions are still responsible for their own `gcr`.
- **PVsyst temperature presets now model a realistic module conversion
  efficiency, which changes results for existing `pvsyst-*` configurations.**
  pvlib's heat balance treats module efficiency as the share of absorbed energy
  that leaves as electricity rather than heat, and its 0.1 default is a legacy
  placeholder no crystalline-silicon module has approached in decades. BREOS now
  passes a module's sourced `Module_Efficiency` when it has one and a
  representative 0.20 otherwise, so every module is modelled consistently
  instead of splitting on which catalog entry happens to carry metadata.
  Expect cell temperatures roughly 2.5–3 °C lower and annual yield 1.0–1.3%
  higher for `pvsyst-*` runs; the refreshed validation baseline moves only its
  `perez_roof` variant, by that margin, across all seven sites. The `faiman`
  default is untouched and all default results remain bit-for-bit unchanged.
- Sourced the 21.2% module efficiency for the generic 600 W bifacial catalog
  entry from the same Trina Vertex TSM-DEG20C.20 datasheet that already supplies
  its bifaciality and power temperature coefficient, and refreshed that
  citation's dead URL. The 445 W Erlangen and generic 400 W entries name no
  datasheet that quotes an efficiency, so they intentionally stay unset and use
  the representative default rather than a back-derived figure.
- Refactored PV model-option resolution and the IAM and temperature kernels
  into focused internal modules while preserving the `breos.App` facade,
  public solar-function signatures, defaults, and numerical paths. Config
  validation is preserved except for the `gcr` tightening noted above.
- `InverterConfig` now validates its pre-existing fields on construction rather
  than trusting callers: `nominal_power_w` and both cost fields must be finite
  and non-negative, `dc_ac_ratio` finite and positive, `inverter_efficiency`
  within `(0, 1]`, `mppt_channels` a positive integer, and `is_hybrid` a bool.
  Code that built a physically impossible inverter — a negative rating, an
  efficiency above 1, zero MPPT channels — used to construct successfully and
  produce meaningless numbers downstream; it now raises `ValueError` at
  construction. Configurations that were already valid are unaffected.
- Reorganized Read the Docs around task-oriented guides, model assumptions,
  and API reference; moved internal design plans, ADRs, and maintainer
  procedures to repository-only documentation; and replaced stale current
  capability references to the 0.3.x series with version-neutral wording.
- Expanded PyPI package metadata (classifiers and keywords) so the project is
  discoverable through scientific-computing and platform facets rather than the
  generic `Topic :: Scientific/Engineering` bucket alone. Packaging metadata
  only; no code, dependency, or public-API change.
- Pointed usage questions and feature ideas at GitHub Discussions from the
  README, contributing guide, and issue-template chooser, keeping the
  maintainer email for research collaboration and private enquiries.
- Credited pvlib as the PV modeling foundation in the README opening, added its
  recommended citation to the README, `ATTRIBUTIONS.md`, and a `references`
  entry in `CITATION.cff`, and added a README acknowledgements section covering
  pvlib, BLAST-Lite, demandlib, pymoo, and the weather data sources.
- Removed residential-only framing from the README description and feature
  list. The packaged presets remain residential-scale, but the engine itself
  carries no building assumption.
- Moved core-package coverage instrumentation out of the per-pull-request test
  matrix and into a separate `coverage-report` job on the nightly, manual, and
  release triggers. All four Python versions now run the suite uninstrumented,
  which takes a pull request from roughly 45 minutes to a few minutes. Coverage
  is unchanged in scope — still branch-aware, still excluding vendored
  BLAST-Lite — and remains a report rather than a gate, as it was before: no
  threshold is configured. CI also no longer filters the `pull_request` trigger
  by base branch, so pull requests stacked on other feature branches are built.

### Fixed
- Corrected the `fast` extra documentation: the current Numba kernels are
  approximate screening utilities and do not accelerate `breos.App`, Monte
  Carlo, or multi-objective optimization production paths.

## [0.4.2] - 2026-07-27

### Changed
- Deferred optional plotting imports until a plotting compatibility attribute
  is first accessed, keeping core imports and non-plotting CLI commands quiet
  while preserving existing top-level names.
- Improved first-run onboarding with a no-file/no-network dry run,
  troubleshooting guidance, corrected discovery commands, and synchronized
  support and release documentation.
- Refreshed locked Python dependencies and GitHub Actions, resolved reported
  dependency security advisories, kept automated dependency maintenance
  security-focused, added core-package coverage that excludes vendored
  BLAST-Lite, and added lightweight macOS/Windows public-entrypoint smoke tests.
- Removed the unused `openpyxl` and `pyarrow` development/validation
  dependencies; the now-empty `validation` extra remains as a compatibility
  alias because the validation harness uses committed CSV and JSON artifacts.
- Added reproducible BLAST parity-fixture tooling with strict clean-checkout and
  upstream-pin guards plus a reviewed provenance manifest, while keeping the
  committed scientific fixture unchanged during ordinary tests.
- Added a narrow internal degradation lifecycle protocol with native and BLAST
  adapters, centralizing daily stepping, reset, snapshot, warning, tracking,
  and provenance behavior without changing dispatch or public result schemas.

## [0.4.1] - 2026-07-23

### Changed
- Extracted BLAST experimental-range and aging-horizon validation into a focused
  internal warning collector while preserving warning records, snapshot
  continuation, replacement-reset behavior, and the `breos.App` result schema.
- Centralized native and BLAST degradation-result construction in a focused
  schema-aware builder while preserving public fields, engine-specific SOH
  precision, and shared result/provenance data.
- Decomposed `breos.App` configuration validation into focused subsystem
  validators while preserving validation order, public errors, normalization,
  defaults, and resolution precedence.

## [0.4.0] - 2026-07-20

### Added
- Vendored BLAST-Lite degradation models as an opt-in `degradation_engine="blast"`
  path with the `BlastEngine` adapter, daily endpoint-grid integration,
  cross-year snapshot threading, replacement reset handling, and validation that keeps
  BLAST disabled for Monte Carlo and resistance-fade runs until those paths are
  explicitly supported.
- A declarative 14-model BLAST registry with stable keys, chemistry and cell
  metadata, experimental ranges, study citations, output capabilities,
  upstream provenance, Python discovery (`list_battery_models()`), CLI
  discovery (`breos list battery-models`), explicit CLI model selection, and
  versioned JSON-safe engine snapshots. BLAST result/provenance blocks identify
  the cell model, calibration basis, initial/final SOH, replacements, warnings,
  and state schema.
- All 14 BLAST models enabled with multi-condition parameter and trajectory
  parity against the pinned upstream source, deduplicated experimental-range
  warnings, and sourced aging-horizon warnings carried through snapshots.

### Changed
- The App energy balance and public `dc_to_ac()` helper now share the same
  PVWatts part-load inverter curve. Dispatch therefore accounts for loading-
  dependent conversion losses when serving load, exporting PV, and
  discharging the battery, while preserving the shared AC nameplate,
  charge-before-export behavior, and explicit 0.3.4 energy ledger. Lower-
  level `BatteryConfig` callers that omit an inverter nameplate retain the
  legacy unbounded flat-efficiency fallback because part load is undefined
  without a rated power.
- App configuration now resolves user values over sourced battery-profile
  defaults over global defaults. Native Naumann/Lam remains the default;
  `blast_model` requires explicit `degradation_engine="blast"`. The ambiguous
  App-level `battery_type` selector, which strict App validation already
  rejected as unknown in 0.3.4, now raises targeted migration guidance instead
  of being repurposed for chemistry/model selection. The lower-level
  `BatteryConfig(battery_type="LFP")` API remains supported.

### Fixed
- BLAST time-varying state updates are guarded against trajectory-inversion
  domain overshoot. When day-varying stressors shrink a state's rate
  coefficient or sigmoid asymptote between updates, the accumulated state can
  fall outside the domain of the trajectory inversion; the next update
  previously returned NaN that silently corrupted SoH (`nmc_lto_10ah` around
  day 3 and `nca_grsi_sonymurata_2p5ah` around year 9 of real multi-year
  profiles). A saturated sigmoid state (`y0 >= y_inf` after `y_inf` shrank) now
  holds its accumulated loss by returning a zero increment instead of snapping
  it back down to `y_inf`; the earlier clamp produced a negative increment that
  decreased an accumulated degradation state and manufactured artificial
  capacity recovery. Increments for the supported positive sigmoid-loss
  trajectories are therefore never negative. Constant and periodic profiles
  never hit the guards, so the golden and multi-condition parity fixtures are
  unchanged.
- `BlastEngine.step` now raises `BlastNumericalError` — reporting the model key,
  elapsed days, and offending field names — whenever any newest state or output
  value is non-finite, not only when capacity `q` is, matching the documented
  fail-loud contract instead of propagating a corrupt state.

### Notes
- The Panasonic NCA model (`nca_gr_panasonic_3ah`) emits a
  `BlastAgingHorizonWarning` once a projection extends past its sourced 300-day
  aging-data horizon, so multi-year projections with that cell warn that they
  extrapolate beyond the calibration window.

### Acknowledgments
- Thanks to Paul Gasper at NLR for suggesting the BLAST-Lite integration.

## [0.3.4] - 2026-07-14

### Added
- A versioned, explicit DC/AC timestep energy ledger now reports PV routing,
  direct and battery inverter losses, cell conversion losses, standby,
  capacity-window adjustments, storage boundary state, and PV-origin battery
  delivery. Optional battery charge (DC input) and discharge (AC delivered)
  power limits are available through `BatteryConfig`, `App`, Monte Carlo, and
  CLI configuration.
- `App.result()` separates behind-the-meter, export, and total CO2 benefits
  and includes JSON-serializable model/configuration provenance.
- `App.result()` now includes `pv_loss_waterfall`, a year-1 diagnostic that
  reports the PV chain from horizontal irradiance reference through
  transposition, IAM, cell temperature, static PVWatts losses, year-1
  degradation, inverter clipping/conversion, surplus curtailment, and battery
  dispatch losses. `breos run` JSON output includes the same block, and
  `breos.plotting.plot_pv_loss_waterfall` renders it as a PV loss diagram.
- `temperature_model` App config key, `--temperature-model` CLI flag, and
  `temperature_model=` parameter on every solar-chain function. The
  `"pvsyst-freestanding"`, `"pvsyst-semi-integrated"`, and
  `"pvsyst-insulated"` presets use pvlib's PVsyst cell-temperature model
  with its documented mounting coefficient sets. The default `"faiman"`
  (open-rack Faiman coefficients, bit-for-bit unchanged) runs cool for
  roof-mounted systems — BREOS's primary audience — and systematically
  overestimates their yield; rooftop studies should pick a roof preset.
  The validation suite runs a `perez_roof` (semi-integrated) config so the
  rooftop yield delta is documented per site.
- `diffuse_iam` App config key, `--diffuse-iam` CLI flag, and `diffuse_iam=`
  parameter on every solar-chain function. `"marion"` applies the
  incidence-angle modifier to the sky- and ground-diffuse POA components via
  pvlib's view-factor-integrated `iam.marion_diffuse` (Marion 2017), using
  the same ashrae model as the beam IAM. Beam-only IAM (diffuse passing at
  1.0) was a known ~0.5–1.5% systematic overestimate; across the validation
  suite `"marion"` lowers annual yield 1.1–2.0% and moves BREOS toward the
  PVGIS reference at all seven sites (e.g. Porto +9.4% → +7.8%). The default
  `"none"` reproduces prior behaviour bit-for-bit; the validation suite now
  runs a `perez_diffuse` config so the effect is tracked per site.
- `solar_position` App config key, `--solar-position` CLI flag, and
  `solar_position=` parameter on every solar-chain function
  (`calculate_pv_production_dc`/`_dc_tracking`/`_ac`/`_tmy`,
  `calculate_multi_array_production`). `"mid-interval"` evaluates the sun
  half a timestep after each label — the PVWatts/SAM convention for
  interval-averaged irradiance (an hourly value labelled 07:00 representing
  the 07:00–08:00 average pairs with the 07:30 sun) — and also drives
  tracker rotation angles. The default `"interval-start"` reproduces prior
  behaviour bit-for-bit. The validation suite now runs a third
  `perez_mid` config so the effect is measured per site.
- A standing validation suite under `validation/` (repo-side, not shipped):
  seven sites on four continents with committed PVGIS TMY weather inputs
  (trimmed to the five columns BREOS reads and gzipped, ~90 KB per site),
  independent PVGIS PVcalc reference results (PVWatts v8 fetcher included,
  references pending network access to `developer.nrel.gov`), a comparison
  report generator, and `tests/test_validation_drift.py`, which fails CI when
  BREOS output drifts >0.1% from its committed baseline or falls outside a
  ±10% gross-error band around the PVGIS reference.

### Changed
- `PV_Production` remains as `(PV_DC - curtailed DC) × inverter efficiency`
  for compatibility. New self-consumption, emissions, loss reporting, and
  optimizer objectives use explicit AC ledger flows. Consumers should migrate
  to `pv_ac_system_kwh`, direct/load/export fields, and the versioned ledger.
- Real calendar-year load profiles now include leap day (8,784 hourly or
  35,136 quarter-hourly intervals) while preserving exact annual energy.
- `diffuse_iam="marion"` now keeps fixed-tilt arrays on pvlib's exact Marion
  integration path but uses a cached 0.5° tilt grid for tracking arrays,
  avoiding thousands of repeated sky/ground diffuse IAM integrations per run.
- POA transposition now receives the refraction-corrected apparent zenith,
  matching pvlib's `ModelChain`. Previously one call mixed zenith
  definitions: the AOI/IAM step used `apparent_zenith` while
  `get_total_irradiance` got the true `zenith`. Annual yields move by well
  under 0.1% (refraction only matters near the horizon); the validation
  baseline was regenerated accordingly.

### Fixed
- Unsupported AC-coupled dispatch (`dc_coupled=False`) now fails early instead
  of silently executing the DC-coupled model.
- The DC-coupled dispatcher now shares inverter headroom between PV and battery
  discharge, routes above-headroom PV to storage before curtailment, records
  battery-discharge inverter loss, closes with non-zero delta SOC, and reports
  temperature/SOH capacity-window changes explicitly.
- Resistance calendar aging now uses daily mean cell temperature rather than
  the final timestep's temperature.
- Multiyear App and Monte Carlo runs now carry stored energy and PV-origin
  inventory across year boundaries instead of silently resetting to a full,
  unknown-origin battery each January.
- Optimizer candidates now use the App's top-level inverter efficiency and
  the simulated/aligned load when computing objectives.
- The NSGA-II optimizer (`optimize_system_multi_objective`) scored candidate
  designs with a different model than the App reports, in three ways, all
  fixed:
  - candidates were simulated **without AC clipping** (no
    `inverter_ac_capacity_w`), biasing the Pareto front toward high DC/AC
    ratios whose clipping losses were never seen. Candidates now get the
    CAPEX-matched nameplate (`pv_peak / costs.dc_ac_ratio`) — the inverter a
    design pays for is the one that clips it;
  - `calculate_financials` ignored maintenance, PV degradation, and the
    separate export-price inflation. It now mirrors the year-1-estimation
    formulas of `cost_analysis_projection` exactly (equivalence enforced by
    `tests/test_optimization_parity.py`); the fixed daily grid fee cancels
    out of the savings NPV and remains omitted by construction. NPV values
    and Pareto fronts change (lower, more realistic NPVs). Candidate battery
    replacements use a documented year-1-SOH projection; App multiyear
    propagation remains the higher-fidelity basis;
  - the load was positionally re-stamped onto the PV index via
    `align_load_to_pv`, ignoring timezones (a UTC-offset shift of the whole
    profile against PV). The raw load now reaches
  `simulate_energy_balance`, whose internal alignment is timezone- and
  DST-aware — the same code path the App uses. `align_load_to_pv` keeps
  its behaviour for external callers but now carries a docstring warning.

## [0.3.3] - 2026-07-02

### Removed
- The `Suntech_STP550S_NOMT` catalog module. Its datasheet points were NMOT
  ratings (800 W/m², Mpp = 415 W), but the CEC single-diode fit interprets
  `Vmp`/`Imp`/`Voc`/`Isc` as STC values, so the entry produced silently wrong
  model parameters. Configs referencing it now fail with the standard
  "Module '...' not found. Available: ..." error; use `Suntech_STP550S_STC`
  (the same physical module at STC) instead.

### Added
- `breos sweep`, a serial parameter-grid CLI command that expands a `[sweep]`
  section in a normal App config and writes one combined CSV with varied
  parameters, resolved system sizing, BREOS version, and scalar result metrics.
- `configs/examples/sweep.toml` as a runnable sweep example over module count
  and battery size.
- Release-smoke tests for the README quickstart, the Monte Carlo example path,
  and the pymoo-backed multi-objective optimization helper.
- `breos.solar.resolve_pvwatts_losses`, used by dry-run/config inspection to
  report resolved PVWatts loss components and their combined percentage.
- `sell_price_inflation` App config key and `--sell-price-inflation` CLI flag
  (default `0.0`). `CostParams` and `cost_analysis_projection` already
  supported an annual export-price inflation, but no config key existed and
  neither the App runner nor the Monte Carlo runner passed it, so the public
  paths always projected with `0.0`. The value is validated in
  `validate_config`, threaded through both projection call sites, and shown
  in `breos run --dry-run` / `validate-config --json`. The `0.0` default
  reproduces existing results bit-for-bit.

### Changed
- `breos run --dry-run` and `breos validate-config --json` now include the
  fully resolved static PVWatts loss stack instead of only echoing
  `pv_loss_overrides`.
- `BatteryConfig.battery_type` is now explicit about the native degradation
  model being LFP-only: `"LFP"` normalizes to `"lfp"`, while unsupported
  chemistries raise instead of silently reusing LFP cycle-aging parameters.
- `BatteryConfig.eol_percentage` now defaults to `0.70`, aligning with the
  App config default `battery_eol_percentage = 0.70` and the optimizer's
  battery-spec fallback (previously `0.80` and `0.8` respectively — three
  surfaces, two values). App and CLI results are unchanged (they always pass
  the config value explicitly), but direct `BatteryConfig` users who relied
  on the implicit `0.80` will now see batteries replaced later, at 70% SOH;
  pass `eol_percentage=0.8` to keep the old threshold. The same applies to
  optimization battery specs without an explicit `eol_percentage`.

### Documentation
- Updated the README and `CITATION.cff` to cite the SSRN preprint DOI
  (`10.2139/ssrn.7032064`).
- Added the BLAST degradation-engine design note and refreshed roadmap
  priorities around model accuracy, validation, energy-loss accounting, and
  future default-model profiles.

### Fixed
- `dc_to_ac` (and therefore `calculate_pv_production_ac`) clipped ~4% below
  the intended inverter AC nameplate: it passed the nameplate
  (`pv_peak_power_w / inverter_loading_ratio`) as pvlib's `pdc0`, which is a
  DC-input limit whose AC nameplate is `eta_inv_nom * pdc0`. The DC limit is
  now derived as `nameplate / eta_inv_nom`, so clipping happens at the same
  AC rating used by `InverterConfig.size_from_pv`, the App energy balance,
  `economics.calculate_costs`, and the CLI's reported `ac_rating_kw`. This
  raises `dc_to_ac` / `calculate_pv_production_ac` outputs slightly at every
  operating point (most visibly during clipping hours); App simulation
  results are unchanged because the App path converts DC through
  `simulate_energy_balance`, not `dc_to_ac`.
- `PVModuleParams` no longer discards a user-supplied `gamma_pmp`: the
  constructor argument existed but `__post_init__` unconditionally overwrote
  it with `T_Pmax_pct`. It now only defaults to `T_Pmax_pct` when not given,
  matching the `alpha_sc_abs` / `beta_voc_abs` override pattern. Catalog
  modules and configs that never set `gamma_pmp` are unaffected.

## [0.3.2] - 2026-06-26

> **Upgrading:** config validation is now strict — a config with an unknown
> top-level key (e.g. a typo like `batery_kwh`) that silently defaulted in
> 0.3.1 now raises listing the offending key(s). Fix or remove stray keys
> before upgrading. All other changes preserve prior behaviour by default.

### Removed
- The `nrel-pysam` runtime dependency. It was only ever reached transitively,
  through pvlib's `fit_cec_sam`, to fit the CEC single-diode parameters on the
  default PV path. `nrel-pysam` publishes no Python 3.14 wheel or sdist and was
  the sole blocker to running BREOS on 3.14.

### Added
- `breos.cec_fit.fit_cec_params`: a pure-`scipy`/`pvlib` implementation of the
  CEC 6-parameter coefficient calculator (Dobos 2012, DOI:10.1115/1.4005759),
  a drop-in for `pvlib.ivtools.sdm.fit_cec_sam`. Across every bundled module it
  reproduces the SAM fit to within 0.03% on maximum power over a
  temperature x irradiance grid and 0.004% on annual energy, so model results
  are unchanged. Validated against the `nrel-pysam` oracle by
  `tools/validate_cec_fit.py`.
- Python 3.14 support: the `3.14` classifier and CI matrix entry, now that the
  `nrel-pysam` blocker is gone.
- Config validation now rejects unknown top-level keys. A typo such as
  `batery_kwh` previously slipped through `merge_defaults` and silently
  defaulted (e.g. the battery to `0`), producing plausible-but-wrong results;
  it now raises listing the offending key(s). The optional `montecarlo`
  section is recognised so Monte Carlo configs still validate.
- Configurable sky-diffusion (transposition) model via a `transposition_model`
  config key and `--transposition-model` / `--sky-model` CLI flag, threaded
  through `calculate_pv_production_dc`, the tracking and multi-array variants,
  and the `App` config surface. Supports `isotropic` (default), `klucher`,
  `haydavies`, `reindl`, `king`, `perez`, and `perez-driesse` via pvlib's
  `get_total_irradiance`; the extra inputs the anisotropic models need
  (extraterrestrial DNI, relative airmass) are derived internally. The default
  `isotropic` reproduces prior results bit-for-bit. Per-array overrides are
  supported in `pv_arrays`.
- Configurable ground reflectance and Perez coefficients to drive those models
  with real site information: `albedo` (0-1) or a named `surface_type`
  (`"snow"`, `"sea"`, `"grass"`, ...) sets the ground-diffuse reflectance for
  every model (previously fixed at pvlib's 0.25), and `model_perez` selects
  the Perez coefficient set. All three are App config keys with matching
  `--albedo` / `--surface-type` / `--perez-model` CLI flags and per-array
  overrides; not setting them leaves the previous defaults unchanged.

### Changed
- The default PV path fits CEC parameters via `breos.cec_fit.fit_cec_params`
  instead of `pvlib.ivtools.sdm.fit_cec_sam`; `breos/solar.py` and the public
  API are otherwise unchanged.
- The two placeholder `Generic_400W` and `Generic_600W_Bifacial` catalog
  modules now carry realistic mono-PERC datasheet specifications (their
  previous made-up values fit cleanly under SAM only via an internal
  short-circuit-current heuristic); their nameplate power and keys are
  unchanged.
- `resolve_pv_system` no longer mutates the merged config in place to record
  the derived `n_modules`; the resolved count is materialised into a fresh
  dict by `resolve_app_config`, so the dict wrapped by the frozen
  `ResolvedAppConfig` is built once and the caller's input dict is left
  untouched.

## [0.3.1] - 2026-06-25

### Changed
- Pinned `requires-python` to `>=3.11,<3.14`. The transitive `nrel-pysam`
  dependency (reached through pvlib's CEC fit) publishes no Python 3.14 wheel
  or sdist, so installs on 3.14 could not resolve. This is a stopgap; 0.3.2
  removes the `nrel-pysam` dependency and lifts the cap.

## [0.3.0] - 2026-06-24

### Fixed
- **TMY timezone misalignment (results-changing):** `fetch_tmy_weather_data`
  relabeled PVGIS's UTC-ordered rows with local-time labels, shifting
  irradiance against the computed solar position by the location's UTC offset
  (~1 h for Berlin, ~10 h for Melbourne; UTC+0 locations were unaffected).
  Rows are now rolled to start at local midnight while each timestamp keeps
  its correct UTC instant.
- **Battery phantom export (results-changing):** when temperature derating or
  daily SOH decline shrank `Emax` below the stored energy, the negative
  charge room silently drained the battery into `Sell_To_Grid`. Stored energy
  is now clamped into the derated window, mirroring the Numba kernel.
- Load profiles are pinned to the location's wall clock instead of the UTC
  clock, so H0 morning/evening peaks land at the correct local hours across
  DST (previously ~1 h off in Iberia during summer).
- `optimize_tilt`/`optimize_tilt_brent` reported "kWh" without accounting for
  the timestep (4x off at 15-minute resolution; ranking was unaffected).
- `BatteryConfig.initial_resistance_growth` was never read by
  `simulate_energy_balance`; it now seeds the resistance state when the
  continuation argument is not supplied.
- `get_module_info` printed the efficiency fraction as a percent and crashed
  on modules without efficiency metadata.
- Removed three dead, shadowed plotting functions and an undefined
  `MONTH_LABELS` reference that crashed the TMY-vs-historical monthly plot.

### Added
- Inverter AC clipping in the `App` energy pipeline: PV output, export, and
  battery discharge now saturate at the AC rating implied by
  `inverter_loading_ratio` — the same rating used for inverter CAPEX. DC
  surplus above the rating still charges a DC-coupled battery
  (`BatteryConfig.inverter_ac_capacity_w`, `None` = legacy uncapped model).
- Configurable PVWatts system losses: `breos.solar.DEFAULT_PVWATTS_LOSSES`
  (~14.1% combined) with a `loss_overrides` hook on the production functions
  and a `pv_loss_overrides` App config key.
- Battery operating parameters as App config keys: `battery_min_soc`,
  `battery_max_soc`, `battery_eol_percentage`, and `battery_rte` (previously
  hardcoded to 0.10/0.90/0.70/sqrt(0.95)).
- `enable_resistance_fade` now feeds the resistance-derated round-trip
  efficiency back into the energy loop (previously tracking-only).
- Battery degradation calibration variants are explicit for the 0.3.0
  release: `naumann_lam_field_calibrated` remains the default v1 field
  calibration, `naumann_lam_field_calibrated_v1` is an equivalent explicit
  alias, and `naumann_lam_field_calibrated_v2` exposes the v2
  field-calibrated fit with Lam `Ea`/`n` fixed and `k0`/`b` fitted to field
  data.
- Parity tests for the optional Numba kernels: the duplicated LFP derate
  constants against `battery.lfp_capacity_factor`, and the energy-balance
  kernel against the reference path under shared-model conditions.
- CLI discovery and inspection commands: `breos list
  {locations,modules,cost-presets,emissions,load-profiles}` prints packaged
  option keys, `breos validate-config <config>` checks a config file and
  summarizes the resolved choices, and `breos run --dry-run` writes the
  resolved configuration as JSON without running a simulation. `list` and
  `validate-config` accept `--json` for machine-readable output.
- PyPI distribution: 0.3.0 is the first release installable with
  `pip install breos`. Tagged `v*` releases on `main` now publish to PyPI
  through GitHub Actions trusted publishing (OIDC), running the release
  artifact verifier before upload, with a manually triggered TestPyPI
  dry-run path.
- Projection-based LCOE support:
  `breos.economics.calculate_lcoe_from_projection` computes LCOE from the
  simulated multi-year cost projection, and the batch location comparison
  tool now writes `lcoe_eur_kwh` plus LCOE heatmaps.

### Changed
- Renamed remaining pre-release "PVBAT" branding to BREOS in the Polysun
  comparison plots: the `plot_degradation_methodology_comparison` first
  argument is now `breos_soh`, the scenario/location dicts passed to
  `plot_lifetime_prediction_comparison` and
  `plot_temperature_sensitivity_comparison` use the `breos_eol_year` key,
  legend labels read "BREOS (Naumann)", and the SOH comparison figure is
  saved as `polysun_breos_soh_comparison*.png`.
- Cost defaults are single-sourced from the `CostParams` dataclass:
  `cost_params_from_config` and the App preset fallbacks no longer carry
  their own diverging literals (packaged presets are unaffected).
- Config validation rejects out-of-range values at load time: negative
  `battery_kwh`, top-level `tilt`/`azimuth`, `inverter_efficiency`,
  `inverter_loading_ratio`, `projection_years`, `pv_degradation_rate`, and
  the new battery keys.
- PV-only App runs construct an explicit inverter model, so a configured
  `inverter_efficiency` now applies without a battery (previously ignored).
- `App.result()["lcoe_eur_kwh"]` now uses the simulated projection, including
  O&M and battery replacement costs, instead of the simpler CAPEX + fixed
  annual O&M helper.
- Library progress messages (weather file discovery, saved files, CSV
  conversions) go through `logging` under `breos.*` logger names instead of
  unconditional `print()`. Functions with a `verbose` flag still print.
- Slimmed the default runtime dependency set to the BREOS core simulation
  stack and moved heavier workflow packages behind extras: `plots`,
  `optimization`, `weather`, `fast`, `validation`, and `location-tools`.
  NREL-PySAM stays in the core set because the default PV model fits CEC
  single-diode parameters at runtime via pvlib's `fit_cec_sam`. (Removed in
  0.3.2.)
- The `dev` extra now installs optional feature dependencies so contributor
  test runs continue to cover optional paths.

### Documentation
- Install snippets in the README and docs point at PyPI (`pip install breos`)
  instead of git tag installs, and the quickstart gained a "10-minute first
  run" walkthrough with a pip-friendly inline config, the matching
  `configs/examples/quickstart.toml` source-checkout example, the new
  option-discovery commands, and a representative output excerpt with
  plausibility ranges.
- New recipes page with validated copy-paste configs: PV-only home, PV plus
  battery, custom latitude/longitude/timezone, east-west roof with
  `pv_arrays`, 15-minute resolution, external E-REDES/BDEW/REE load
  profiles, and offline runs with cached weather.
- New generated "Packaged options" reference page listing locations, PV
  modules, cost presets, emissions factors, and load profiles. It is built
  by `tools/generate_option_docs.py` from the packaged data and source
  constants, and a test fails CI when the page drifts.
- README documents the fixed PVWatts loss components, the inverter clipping
  convention, the `weather/` working-directory override, the Open-Meteo
  `.cache.sqlite` file, logging configuration, and the new config keys.
- README describes the Numba kernels honestly as approximate standalone
  screening engines that `breos.App` does not use; the module docstring
  carries the same warning.
- Clarified that the `bdew_h0` alias maps to the bundled demandlib
  BDEW-H0-shaped profile `"1"`, distinct from the external BDEW H0 2025
  dataset (profile `"7"`).
- Replaced stream-of-consciousness working notes in `economics`,
  `optimization`, and `plotting` with factual comments, and
  fixed mislabeled docstrings (`total_pv` is post-inverter AC; the Suntech
  NOMT catalog entry documents its NMOT-condition rating).

## [0.2.3] - 2026-06-08

### Changed
- Lowered the minimum supported Python from 3.13 to 3.11 — the real floor, set by
  pandas, timezonefinder, and stdlib `tomllib`. CI now runs a 3.11/3.12/3.13 matrix.
- Relaxed the pvlib constraint from `==0.14.0` to `>=0.14.0,<0.16` after verifying
  the full API surface and the test suite against pvlib 0.15.1.
- `breos.__version__` is now resolved from installed package metadata
  (`importlib.metadata`) instead of a hardcoded literal, so it can no longer drift
  from `pyproject.toml`.

### Added
- `CITATION.cff`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` for open-source release
  readiness.

### Removed
- Duplicate top-level `rlp/*.csv` load-profile files (byte-identical to the
  packaged `breos/data/rlp/` copies that runtime actually uses). `rlp/README.md` is
  retained as external-RLP guidance.

### Documentation
- README badge and installation docs now state Python 3.11+.
- Trimmed `ATTRIBUTIONS.md` to reference only the packaged load-profile paths.

## [0.2.2] - 2026-06-07

### Documentation
- Expanded third-party notices with dependency credits, runtime data-source
  caveats, and scientific/model attribution guidance.

## [0.2.1] - 2026-06-03

### Documentation
- Updated installation guidance to use the stable GitHub tag until PyPI
  publishing is available.
- Added PyPI trusted publishing to the roadmap.
- Documented the full CI/release validation gates in the contributor guide.
- Standardized API documentation wording around domain areas.

## [0.2.0] - 2026-06-03

### Changed
- Narrowed the top-level `breos.__all__` release surface to the stable facade,
  key configuration/result objects, and core composition helpers. Lower-level
  module APIs remain importable from their modules.

## [0.1.0] - 2026-04-30

### Added
- Public API facade (`breos.App`) — single entry point for simulations: config dict in, plain dict out.
- Command line entry point (`breos run`) for running simulations from shell flags or TOML/JSON config files.
- Test suite — pytest coverage of the public API, battery, economics, emissions, and solar modules (all offline).
- GitHub Actions CI on every push/PR.
- `cost_params_from_config()` — config parser for `CostParams`.
- Marginal grid carbon intensity support in `EmissionsParams` for more accurate CO₂ avoidance accounting.

### Changed
- Renamed PV `slope` → `tilt` everywhere: function parameters, dataclass fields, docstrings, CLI/config keys, public API. Includes `optimize_slope()` → `optimize_tilt()` and the `tools/azitilt_optimizer.py` script.
- Calendar model name canonicalized to `naumann_lam_field_calibrated` (the legacy alias `naumann_lam_calibrated` has been removed).
- Constants renamed: `LAM_NAUMANN_FIELD_CALIBRATED_*` → `NAUMANN_LAM_FIELD_CALIBRATED_*`; alias indirections (`LAM_CAL_K0_FRAC` …) dropped.
- Configs modernized: per-unit cost keys (`maintenance_cost_per_panel`, `other_cost_per_module`); emissions schema renamed and country list expanded.
- Polysun degradation now tracks the actual `last_replacement_year` instead of approximating with `n_replacements × int(total_life)` — handles fractional lifetimes and cycle-driven replacements correctly.
- Numba degradation kernel now treats SOC reversals as half-cycles (rainflow-aligned) and applies LFP temperature derating per timestep, matching the Python reference path.

### Fixed
- NPV discount factor now uses `(1 + r) ** Year` (time-0 NPV) instead of `(1 + r) ** (Year - 1)`. Affects all `cost_analysis_projection` outputs.

### Removed
- Out-of-scope kernels (`combined_energy_balance_kernel`, `batch_combined_energy_balance_kernel`) and non-core energy-system code paths. BREOS focuses on PV + battery simulation.
