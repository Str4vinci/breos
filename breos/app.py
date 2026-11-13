"""
BREOS public facade - single entry point for PV + battery simulations.

Usage:
    import breos

    app = breos.App({
        "location": "porto",
        "n_modules": 10,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "cost_preset": "residential_pt",
        "emissions_country": "PT",
    })
    app.simulate()
    result = app.result()
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import pandas as pd

from breos.app_config import APP_CONFIG_FIELDS, ResolvedAppConfig, normalize_config_keys, resolve_app_config
from breos.app_inputs import AppRuntimeDependencies
from breos.app_results import build_result as build_app_result
from breos.load_profiles import load_profile
from breos.repair import input_repair_records
from breos.runners.app import SimulationArtifacts, revalue_app_simulation, run_app_simulation
from breos.weather import build_battery_temperature_series, fetch_tmy_weather_data, load_weather, resample_to_15min

# Nested tables App.revalue replaces whole: the entries of a price list
# belong together, so a change must not keep a period it leaves out.
_REPLACED_TABLES = frozenset(
    {("tariff", "import_prices"), ("tariff", "export_prices"), ("reference_tariff", "import_prices")}
)


def _revalued_config(config: dict[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """``config`` with ``changes`` applied table by table, as App.revalue documents."""
    merged = dict(config)
    for key, value in changes.items():
        current = merged.get(key)
        if not isinstance(value, Mapping):
            merged[key] = deepcopy(value)
            continue
        # A new table starts empty, so a key set to None has nothing to remove.
        table: dict[str, Any] = deepcopy(current) if isinstance(current, dict) else {}
        for name, item in value.items():
            if item is None:
                table.pop(name, None)
            elif (key, name) in _REPLACED_TABLES or not (
                isinstance(table.get(name), dict) and isinstance(item, Mapping)
            ):
                table[name] = deepcopy(item)
            else:
                table[name] = {**table[name], **deepcopy(dict(item))}
        merged[key] = table
    return merged


# The money keys of a [tariff] or [reference_tariff] table, and of its annual_network_credit.
_TARIFF_MONEY_KEYS = ("import_prices", "export_prices", "fixed_charge_per_day")
_CREDIT_MONEY_KEYS = ("amount_per_year", "network_fixed_per_year", "network_import_prices")


def _is_zero(value: Any) -> bool:
    """Whether an amount, or every price in a price list, is zero: zero in any currency."""
    if isinstance(value, Mapping):
        return all(_is_zero(item) for item in value.values())
    return bool(value == 0)


def _kept_amounts(table: Mapping[str, Any], change: Any, keys: Iterable[str], where: str) -> list[str]:
    """The non-zero amounts among ``keys`` in an old table that its revalue ``change`` does not restate."""
    restated = set(change or {})
    return [f"{where}.{key}" for key in keys if key in table and key not in restated and not _is_zero(table[key])]


def _check_currency_change(old: ResolvedAppConfig, currency: str, changes: Mapping[str, Any]) -> None:
    """Refuse a revaluation into ``currency`` that keeps an amount stated in the run's old currency.

    Tables merge key by key, so after a currency change every amount the old
    run set and the changes do not restate would be read in the new currency:
    each non-zero cost under ``[costs]``, and each non-zero price list, fixed
    charge and network-credit amount of a ``[tariff]`` or
    ``[reference_tariff]`` the run keeps. A key set to None, or a table
    removed, counts as restated. A cost preset carries its own currency and
    is checked when the new configuration resolves. The planner's wear
    weight is not a revaluation key, so a non-zero one cannot be restated.
    ``changes`` has normalised keys (:func:`normalize_config_keys`).
    """
    if currency == old.currency:
        return
    move = f"revalue() moves the run from {old.currency} to {currency}"
    kept = _kept_amounts(old.cfg.get("costs") or {}, changes.get("costs"), sorted(old.cfg.get("costs") or {}), "costs")
    for name in ("tariff", "reference_tariff"):
        table = old.cfg.get(name)
        if not isinstance(table, Mapping) or (name in changes and changes[name] is None):
            continue
        change = changes.get(name) or {}
        kept += _kept_amounts(table, change, _TARIFF_MONEY_KEYS, name)
        credit, credit_change = table.get("annual_network_credit"), change.get("annual_network_credit", {})
        if isinstance(credit, Mapping) and credit_change is not None:
            kept += _kept_amounts(credit, credit_change, _CREDIT_MONEY_KEYS, f"{name}.annual_network_credit")
    if kept:
        raise ValueError(
            f"{move}, but {', '.join(kept)} {'is an amount' if len(kept) == 1 else 'are amounts'} in "
            f"{old.currency}. BREOS does not convert currencies: restate each of them in {currency} in the "
            "changes, or set it to None to remove it."
        )
    wear = old.smart_charging.wear_cost_per_kwh if old.smart_charging is not None else None
    if wear:
        raise ValueError(
            f"{move}, but smart_charging.wear_cost_per_kwh ({wear}) is in {old.currency}, and revalue() cannot "
            f"change it. Build a new App with the wear cost in {currency}."
        )


# The keys App.revalue may change: the economics section of the resolved
# configuration, except the horizon, which sets how many years are simulated.
REVALUATION_KEYS = frozenset(
    key
    for key, field in APP_CONFIG_FIELDS.items()
    if field.summary is not None and field.summary.startswith("economics.") and key != "projection_years"
)


class App:
    """
    Single entry point for BREOS simulations.

    Parameters
    ----------
    config : dict
        Simulation configuration. Required keys:

        - ``location`` - preset key (``"porto"``) **or** dict with
          ``latitude``, ``longitude``, ``timezone``.
        - ``n_modules`` - number of PV modules (int, > 0), unless
          ``pv_arrays`` is provided.
        - ``annual_consumption_kwh`` - yearly electricity demand (float, > 0).

        Optional keys include battery size, PV arrays, module selection, load
        profile, tracking, sky-diffusion model (``transposition_model``),
        custom terrain shading (``horizon_profile``),
        opt-in bifacial rear gain (``bifacial_model`` plus row geometry),
        resolution, projection years, cost and emissions presets, degradation,
        and inverter assumptions.
    input_repairs : InputRepairReport, dict, or list of them, optional
        Reports from :func:`breos.io.repair_series <breos.repair.repair_series>` for input series repaired
        before this run, for example a measured load written to the
        ``rlp_directory`` file the config points at. They are recorded
        unchanged in ``result()["provenance"]["input_repairs"]``. App does not
        repair anything itself and does not check the reports against the
        input it loads. Omitted (None), provenance has no ``input_repairs``
        key, exactly as before.
    """

    def __init__(self, config: dict, *, input_repairs: Any = None) -> None:
        self._config = deepcopy(config)
        self._resolved = resolve_app_config(config)
        if self._resolved.period is not None and "projection_years" in normalize_config_keys(config):
            warnings.warn(
                f"'projection_years' ({self._resolved.cfg['projection_years']}) is ignored: a [period] window "
                "runs once and has no project lifetime. Remove it, or remove [period] to project whole years.",
                UserWarning,
                stacklevel=2,
            )
        self._input_repairs = input_repair_records(input_repairs)
        self._result: dict[str, Any] | None = None
        self._artifacts: SimulationArtifacts | None = None

    def simulate(self) -> None:
        """Run the full simulation pipeline."""
        artifacts = run_app_simulation(self._resolved, self._runtime_dependencies())
        self._result = build_app_result(self._resolved, artifacts, input_repairs=self._input_repairs)
        self._artifacts = artifacts

    def revalue(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        """Return the result this run would give at other prices, without changing this App.

        ``changes`` holds configuration keys. A nested table such as
        ``costs`` or ``tariff`` changes only the keys it sets, and a key set
        to ``None`` in it is removed; ``{"tariff": None}`` removes the table.
        A price list (``tariff.import_prices``, ``tariff.export_prices``,
        ``reference_tariff.import_prices``) replaces the old one whole. Only
        the economics keys in :data:`REVALUATION_KEYS` may change: ``costs``,
        ``cost_preset``, ``currency``, ``tariff``, ``reference_tariff``, the
        discount rate, the escalators and ``terminal_value``. The estimated battery residual
        value is recomputed from retained final health without simulating again.

        A change of the run's currency must restate every money input in the
        new currency, since tables merge key by key: each non-zero cost the
        old ``costs`` table set, and each non-zero price list, fixed charge
        and ``annual_network_credit`` amount of a ``tariff`` or
        ``reference_tariff`` the run keeps (a key set to None, or a table
        removed, counts), and a cost preset in the new currency or none. A
        run whose ``smart_charging.wear_cost_per_kwh`` is not zero cannot
        change currency; build a new App for it.

        When the new prices cannot change the dispatch, the stored simulation
        is re-priced (``"repriced"``): flat prices, a tariff removed, or a
        tariff on the same schedule whose smart-charging instructions stay
        the same. A tariff added, or a different schedule, is priced from the
        stored per-step flows of every year (``"repriced_by_step"``) when the
        run has no smart charging, whose greedy dispatch never sees the
        tariff. Otherwise (a tariff added or a different schedule under
        smart charging, or new import or export prices under the
        experimental ``daily_persistence`` smart charging, which plans on
        them) the run is simulated again (``"resimulated"``). A
        ``reference_tariff`` added, changed or removed is always re-priced:
        it prices only the no-system household, which the dispatch never
        sees. The result records the method in
        ``provenance["revaluation"]``, with the keys that changed. Flat
        prices and ``"repriced_by_step"`` give the same floats as a new
        simulation; a tariff re-priced on its own schedule sums energy by
        period instead of by step, so it agrees to rounding.

        A run without smart charging keeps each year's grid import and
        export step by step for this: about 11 MB for 20 years at 15-minute
        resolution, a quarter of that hourly.

        Raises:
            RuntimeError: If :meth:`simulate` has not been called.
            ValueError: If ``changes`` sets a key outside
                :data:`REVALUATION_KEYS`; build a new App for those. Or if it
                changes the currency and keeps an amount in the old one.
        """
        if self._artifacts is None:
            raise RuntimeError("Call simulate() before revalue().")
        # Hyphens and underscores name the same key, as in App, so a change
        # spelled either way meets the key it changes or removes.
        changes = normalize_config_keys(dict(changes))
        unknown = sorted(key for key in changes if key not in APP_CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"Unknown config key(s) for revalue(): {', '.join(unknown)}")
        old = normalize_config_keys(self._config)
        config = _revalued_config(old, changes)
        changed = sorted(key for key in config.keys() | old.keys() if config.get(key) != old.get(key))
        outside = [key for key in changed if key not in REVALUATION_KEYS]
        if outside:
            raise ValueError(
                f"revalue() changes prices only, and {', '.join(outside)} is not a price key. "
                f"Build a new App for it. Keys revalue() accepts: {', '.join(sorted(REVALUATION_KEYS))}."
            )
        resolved = resolve_app_config(config)
        _check_currency_change(self._resolved, resolved.currency, changes)
        artifacts, method = revalue_app_simulation(resolved, self._artifacts, self._runtime_dependencies())
        result = build_app_result(resolved, artifacts, input_repairs=self._input_repairs)
        result["provenance"]["revaluation"] = {"method": method, "changed_keys": changed}
        return result

    def result(self) -> dict[str, Any]:
        """
        Return simulation results as a plain dict.

        Raises ``RuntimeError`` if :meth:`simulate` has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call simulate() before result().")
        return self._result

    def timeseries(self) -> pd.DataFrame:
        """Return the first simulated year step by step, as a new DataFrame.

        The frame has one row per simulation step: 8,760 for an hourly year
        (8,784 in a leap year), and four times as many at
        ``resolution = "15min"``. A ``[period]`` run has the rows of its
        window only. Later project years are not kept step by step;
        ``result()["yearly"]`` holds their totals and state of health.

        The index is a ``RangeIndex``. The ``Datetime`` column is
        timezone-aware and on the clock of the weather, which for a PVGIS TMY
        is the fixed UTC offset of 1 January. These columns are stable:

        - ``PV_DC``, ``PV_Production``, ``Houseload``, ``Import_From_Grid``,
          ``PV_AC_To_Load``, ``PV_AC_Export``, ``PV_DC_Curtailed`` and
          ``Battery_AC_To_Load``: mean power over the step, in W. Multiply
          by the step length in hours and divide by 1,000 to get kWh.
        - ``Battery_Energy``, ``Battery_Energy_Beginning`` and
          ``Battery_Energy_End``: stored energy, in Wh.
        - ``Battery_SOC_Normalized``: state of charge in the usable window,
          from 0 to 1. ``Battery_SOH``: state of health, in %.

        The energy-balance ledger schema in the API reference defines the
        other columns. A later release can add columns. The frame is a copy:
        a change to it does not change this App or its results.

        Raises:
            RuntimeError: If :meth:`simulate` has not been called.
        """
        if self._artifacts is None:
            raise RuntimeError("Call simulate() before timeseries().")
        return self._artifacts.first_year_results_df.copy(deep=True)

    @staticmethod
    def _runtime_dependencies() -> AppRuntimeDependencies:
        return AppRuntimeDependencies(
            load_profile=load_profile,
            load_weather=load_weather,
            fetch_tmy_weather_data=fetch_tmy_weather_data,
            resample_to_15min=resample_to_15min,
            build_battery_temperature_series=build_battery_temperature_series,
        )
