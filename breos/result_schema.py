"""Format number of the public result schema (ADR 0003 E9).

``App.result()``, Monte Carlo provenance, optimizer provenance and the CLI's
``--json`` output record it as ``result_schema_version``. It is independent
of the ledger schema (``LEDGER_SCHEMA_VERSION``).

The format changes, to the next integer, only when a field is renamed or
removed, a change that breaks existing readers. Added fields do not change
it: the changelog of each release lists them, and ``breos_version`` identifies
the release that wrote a result. Format ``"1"`` is the first released format
(BREOS 0.7.0). Results from earlier BREOS versions carry no format number.
Field-level documentation is in the result interpretation guide
(``docs/getting-started/interpreting-results.md``).
"""

RESULT_SCHEMA_VERSION = "1"
