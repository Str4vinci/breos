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
columns to cost projections and Monte Carlo trajectories (#183).
"""

RESULT_SCHEMA_VERSION = "1.4"
