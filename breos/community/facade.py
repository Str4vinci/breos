"""Public collective self-consumption entry point beside App."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from breos.app import App
from breos.community.config import parse_community_config
from breos.community.money import price_community
from breos.community.results import CommunityResult, build_result
from breos.community.runner import run_community


class Community:
    """Simulate PV sharing between separately metered sites at one location.

    Parameters
    ----------
    config : Mapping
        TOML-shaped mapping with ``community``, ``participants`` and optional
        ``assets`` tables. Like App, this constructor takes a mapping; load a
        TOML file with ``tomllib.load`` first. Shared clock/economics keys
        belong in ``community``. Settlement requires ``resolution = '15min'``.

    Each site's equipment runs independently for the full project horizon.
    Sharing and pricing follow its metered balances; they never dispatch it.
    Only first-year interval frames and all annual totals are retained.
    Vehicles, community storage, revaluation, Monte Carlo and optimization
    are outside this facade's scope.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._resolved = parse_community_config(config)
        self._result: CommunityResult | None = None
        self._settlement: pd.DataFrame | None = None

    def simulate(self) -> None:
        """Run sites, settle and price the full horizon, replacing prior results."""
        # The same dependency seam as App also supports offline App providers.
        run = run_community(self._resolved, deps=App._runtime_dependencies())
        accounts = price_community(run)
        result = build_result(run, accounts)
        self._settlement = accounts.settlements[1]
        self._result = result
        # Do not retain SiteRun/projection arrays after pricing and provenance.

    def result(self) -> CommunityResult:
        """Return typed accounts and provenance; requires :meth:`simulate`."""
        if self._result is None:
            raise RuntimeError("Call simulate() before result().")
        return self._result

    def settlement(self) -> pd.DataFrame:
        """Return a copy of the first project year's 15-minute Wh ledger.

        Later years retain annual totals in ``result().settlement_totals``.
        Raises ``RuntimeError`` before :meth:`simulate` has been called.
        """
        if self._settlement is None:
            raise RuntimeError("Call simulate() before settlement().")
        return self._settlement.copy(deep=True)
