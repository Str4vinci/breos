"""Version of the public result schema (ADR 0003 E9).

``App.result()``, Monte Carlo provenance and summaries, and optimization
provenance carry it as ``result_schema_version``. It is independent of the
ledger schema (``LEDGER_SCHEMA_VERSION``). A renamed or removed field bumps
the major version; an added field bumps the minor. A result without the field
predates the currency-neutral names of 1.0. 1.1 adds the App result's
top-level year-1 money components (``grid_import_cost_year1_prices`` and its
neighbours). 1.2 adds ``provenance.economics``, the rates a projection used
(ADR 0003 E1, E2), to App, Monte Carlo and optimizer provenance, and the
escalator keys to the App's ``resolved_config``. 1.3 adds
``Replaced_Capacity_kWh``, the nominal capacity each year's replacements
swapped in, to the year rows of Monte Carlo trajectories and optimizer tables,
which also gain the year-1-price money columns (ADR 0003 E4). 1.4 adds
``provenance.revaluation`` to ``App.revalue`` results and the export CO2
columns to cost projections and Monte Carlo trajectories (#183). 1.5 adds
``constraints`` and ``run_settings`` to the optimizer's provenance, and the
``Projected_CO2_*`` columns to Pareto rows of a search with ``[emissions]``
(#181). 1.6 adds ``inverter_ac_rating_kw`` to the ``resolved_config`` of App and Monte
Carlo results
(#181). 1.7 adds the ``[period]`` window (#242): an App result of a period run
gains ``period``, ``provenance.period`` and ``period_start``/``period_end`` in
its year row, and reports its lifetime economics (``npv_savings``,
``payback_year``, ``lcoe_per_kwh``, ``financial``, the replacement costs and
the lifetime CO2) as None. Results of full-year runs are unchanged. 1.8 adds
``calendar_year`` to Monte Carlo's ``provenance.load_profile``: the
``target_year`` the load was built for (#302).

2.0 removes inert configuration and metadata, duplicate CO2 aliases and
legacy PV production fields; it renames monthly/yearly grid rows and optimizer
breakeven columns to use explicit grid-import, grid-export and payback names.
The migration table is in the result interpretation guide. It does not change
the timestep ledger or the annual ``PV_Production_kWh`` usable-AC field.

2.1 adds ``tariff.custom_schedule`` to the ``resolved_config`` of App and
Monte Carlo results when an inline schedule is configured.

2.2 adds the experimental ``daily_persistence`` record to an App result's
``provenance.smart_charging`` when that mode is configured: the
``experimental`` marker, controller and planner versions, planner settings,
forecast, warm-start and terminal policies, and the stored energy by origin
at the start and end of the project. Other results are unchanged.

2.3 adds ``battery_allow_terminal_replacement`` to the ``resolved_config`` of
App and Monte Carlo results, and ``battery_replacement_treatment``, with its
``allow_terminal_replacement`` policy and a ``terminal_period`` description,
to the provenance of a projected design and of an optimizer search. Default
results are otherwise unchanged.

2.4 adds calendar-month seasons to custom tariff schedules: ``seasons`` in
``resolved_config.tariff.custom_schedule`` and, as the month partition, in
``provenance.tariff`` of App, Monte Carlo and optimizer results that define
them. With month seasons, ``import_prices`` and ``export_prices`` in the
resolved config and in ``provenance.tariff`` may map each season to its
period prices instead of each period to a price. Results without month
seasons are unchanged.

2.5 adds ``mode = "discharge_only"`` to ``provenance.smart_charging`` of App,
Monte Carlo and optimizer results: a run in that mode records its discharge
periods, an empty ``charge_periods`` and None for
``target_usable_fraction``, ``grid_charge_efficiency`` and
``grid_import_limit_w``. The App's top-level ``smart_charging`` block reports
it as it reports the other modes. Other results are unchanged.

2.6 adds the no-system cost components (#339): ``no_system_cost_import`` and
``no_system_cost_fixed_charge`` in each App ``financial`` row from year 1,
the top-level ``no_system_fixed_charge_year1_prices``, the
``Cost_No_Sys_Import`` and ``Cost_No_Sys_Fixed_Charge`` cost-projection
columns and the ``Baseline_Fixed_Charge`` year-row column (and so Monte Carlo
trajectories and optimizer tables), and
``reference_tariff`` in the ``resolved_config`` of App and Monte Carlo
results. ``provenance.reference_tariff`` is present only when a
``[reference_tariff]`` is configured, in App, Monte Carlo and optimizer
provenance. Without one, every existing value is unchanged.

2.7 adds ``overlap_policy`` to ``provenance.smart_charging`` in App, Monte
Carlo and optimizer results, including the default ``reject``. ``hold_target``
is supported for ``fixed_target`` only (#347). The ledger is unchanged.

2.8 adds the optional terminal-health accounting sensitivity (#346):
``terminal_health_credit``, ``terminal_health_credit_npv`` and
``npv_savings_terminal_adjusted`` in App results and Monte Carlo runs and
statistics; ``terminal_value`` in resolved config; and, when enabled for a
lifetime run, ``provenance.terminal_value`` with formula and price inputs.
The three App scalars are None when disabled or on a [period] run. Monte
Carlo uses NaN for disabled values and omits their statistics. Projected
optimization ignores the table and continues ranking on unadjusted NPV.
"""

RESULT_SCHEMA_VERSION = "2.8"
