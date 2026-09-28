"""Version of the public result schema (ADR 0003 E9).

``App.result()``, Monte Carlo provenance and summaries, and optimization
provenance carry it as ``result_schema_version``. It is independent of the
ledger schema (``LEDGER_SCHEMA_VERSION``). A renamed or removed field bumps
the major version; an added field bumps the minor. A result without the field
predates the currency-neutral names of 1.0. 1.1 adds the App result's
top-level year-1 money components (``grid_import_cost_year1_prices`` and its
neighbours).
"""

RESULT_SCHEMA_VERSION = "1.1"
