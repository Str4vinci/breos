"""Command line interface for BREOS."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import shlex
import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

from breos.app import App
from breos.app_config import (
    ALLOWED_CONFIG_KEYS,
    APP_CONFIG_FIELDS,
    NESTED_TABLE_SPECS,
    ResolvedAppConfig,
    normalize_config_keys,
    override_config,
    resolve_app_config,
    validate_montecarlo_config,
)
from breos.app_inputs import _input_cache_key, reuse_prepared_inputs
from breos.config_schema import MappingOf
from breos.degradation import get_battery_model_profile, list_battery_models
from breos.execution import EXECUTION_BACKENDS
from breos.io import nonfinite_to_none
from breos.load_profiles import PROFILES, resolve_profile_file
from breos.pv_modules import MODULES
from breos.resources import load_config_json
from breos.solar import resolve_pvwatts_losses
from breos.tariffs import DEFAULT_CURRENCY
from breos.utils import package_version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as f:
        try:
            if suffix == ".toml":
                data = tomllib.load(f)
            elif suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError("Config file must be TOML or JSON")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Invalid TOML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config file must contain an object at the top level")
    return normalize_config_keys(data)


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config) if args.config else {}

    overrides: dict[str, Any] = {}
    for key, field in APP_CONFIG_FIELDS.items():
        if not field.cli_flags:
            continue
        value = getattr(args, key)
        if value is None:
            continue
        if field.normalizer is not None:
            value = field.normalizer(value)
        if value is None or value == "":
            continue
        overrides[key] = value

    return _deep_merge(override_config(config, overrides), overrides)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overrides`` applied, merging nested tables key by key.

    An override table replaces only the keys it sets, so a flag that sets one
    ``[tariff]`` key keeps the rest of the file's table. Neither input is
    changed.
    """
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _json_text(data: Any, what: str, **kwargs: Any) -> str:
    """Serialise user-facing output as strict JSON.

    ``NaN`` and ``Infinity`` are not JSON, and many parsers reject them. A
    payload whose metrics can be undefined goes through
    :func:`breos.io.nonfinite_to_none` first; any non-finite value left after
    that is a bug, so it fails here instead of writing an invalid file.
    """
    try:
        return json.dumps(data, allow_nan=False, **kwargs)
    except ValueError as exc:
        raise ValueError(f"Cannot write {what} as JSON: it contains a non-finite number (NaN or Infinity)") from exc


def _run(args: argparse.Namespace) -> int:
    config = _build_config(args)
    _ignore_unused_runner_sections(config, command="run")
    if args.dry_run:
        return _write_payload(_resolved_config_summary(config), args, "the resolved config")

    app = App(config)
    app.simulate()
    # An undefined metric (an LCOE with no production) is reported as null.
    return _write_payload(nonfinite_to_none(app.result()), args, "the run result")


def _write_payload(data: dict[str, Any], args: argparse.Namespace, what: str) -> int:
    indent = args.indent if args.indent > 0 else None
    payload = _json_text(data, what, indent=indent)

    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


# The sections of the resolved-config summary, in output order. Each App key
# names its place through ``AppConfigField.summary``.
_SUMMARY_SECTIONS = ("location", "pv", "inverter", "load", "battery", "economics", "emissions", "simulation")


def _resolved_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Summarise a raw App config: :func:`_config_summary` of its resolution."""
    return _config_summary(resolve_app_config(config))


def _config_summary(resolved: ResolvedAppConfig) -> dict[str, Any]:
    """Summarise a resolved App config without fetching weather or simulating.

    Every registered key appears at its ``AppConfigField.summary`` place, as
    configured after defaults and normalisation; ``None`` still means "not
    set" (an escalator that follows ``inflation_rate``, pvlib's albedo). Values
    the resolver derives (the location's coordinates, the tilt, azimuth and
    tracker axis, the module, the inverter AC rating, the load-profile file)
    then replace or join them. Key order within a section follows the
    registry.
    """
    cfg = resolved.cfg
    summary: dict[str, Any] = {"valid": True, **{section: {} for section in _SUMMARY_SECTIONS}}
    for name, field in APP_CONFIG_FIELDS.items():
        # An unset [period] is left out, so a full-year summary is unchanged.
        if name == "period" and cfg.get(name) is None:
            continue
        if field.summary is not None:
            section, key = field.summary.split(".")
            summary[section][key] = cfg.get(name)
    # A table can hold values JSON has no type for, such as a TOML date in
    # [tariff]; they are written as text, as in the result's provenance.
    summary = json.loads(json.dumps(summary, default=str))

    # A config is valid before its external profile file is in place, so a
    # missing file is reported rather than raised; several matches still raise.
    try:
        profile_file: str | None = resolve_profile_file(
            cfg["load_profile"], cfg["resolution"], cfg["rlp_directory"], cfg["load_profile_file"]
        ).label
        profile_file_error = None
    except FileNotFoundError as exc:
        profile_file, profile_file_error = None, str(exc)
    summary["location"].update(
        key=resolved.loc_key, latitude=resolved.lat, longitude=resolved.lon, timezone=resolved.timezone
    )
    summary["pv"].update(
        system_kwp=resolved.system_kwp,
        module=resolved.pv_module_key,
        arrays=resolved.pv_arrays or None,
        tilt=resolved.tilt,
        azimuth=resolved.azimuth,
        axis_azimuth=resolved.axis_azimuth,
        losses=resolve_pvwatts_losses(cfg["pv_loss_overrides"]),
    )
    summary["inverter"]["ac_rating_kw"] = (resolved.inverter_ac_capacity_w or 0.0) / 1000
    summary["load"].update(load_profile_file=profile_file, load_profile_file_error=profile_file_error)
    summary["battery"].update(
        round_trip_efficiency=cfg["battery_rte"] if cfg["battery_rte"] is not None else 0.95,
        model_profile=(
            get_battery_model_profile(cfg["blast_model"]).as_dict() if cfg["blast_model"] is not None else None
        ),
    )
    summary["emissions"]["enabled"] = resolved.emissions_params is not None
    summary["notes"] = [
        "This is a resolved configuration check only; no weather fetch or simulation was run.",
        "Packaged defaults are examples. Replace weather, load, PV, inverter, cost, and emissions inputs for real studies.",
    ]
    return summary


def _load_options(category: str) -> list[dict[str, Any]]:
    if category == "locations":
        locations = load_config_json("locations.json")
        return [
            {
                "key": key,
                "name": value.get("name", key),
                "latitude": value["latitude"],
                "longitude": value["longitude"],
                "timezone": value["timezone"],
            }
            for key, value in sorted(locations.items())
        ]

    if category == "modules":
        return [
            {
                "key": key,
                "power_w": module.Mpp,
                "name": module.Name or key,
                "celltype": module.celltype,
                "module_efficiency": module.Module_Efficiency,
                "bifaciality": module.bifaciality,
                "noct_c": module.NOCT,
            }
            for key, module in sorted(MODULES.items())
        ]

    if category == "cost-presets":
        presets = load_config_json("costs.json")
        return [
            {
                "key": key,
                # The bundled catalogue is in one currency; BREOS does not convert.
                "currency": DEFAULT_CURRENCY,
                "electricity_cost_per_kwh": value.get("electricity_cost"),
                "export_price_per_kwh": value.get("electricity_sold_cost"),
                "storage_cost_per_kwh": value.get("storage_cost_per_kwh"),
            }
            for key, value in sorted(presets.items())
        ]

    if category == "emissions":
        emissions = load_config_json("emissions.json")
        return [
            {
                "key": key,
                "country": value["country"],
                "grid_intensity_gco2_kwh": value["average_grid_carbon_intensity_gco2_kwh"],
                "year": value["year"],
            }
            for key, value in sorted(emissions.items())
        ]

    if category == "load-profiles":
        return [
            {
                "key": key,
                "name": spec.name,
                "bundled": spec.bundled,
                "files": sorted(set(spec.files.values())),
                "requires_rlp_directory": not spec.bundled and key != "custom",
                "requires_load_profile_file": key == "custom",
            }
            for key, spec in PROFILES.items()
        ]

    if category == "battery-models":
        return list_battery_models()

    raise ValueError(f"Unknown list category: {category}")


def _format_options(category: str, rows: list[dict[str, Any]]) -> str:
    if category == "locations":
        return "\n".join(
            f"{row['key']}: {row['name']} ({row['latitude']}, {row['longitude']}, {row['timezone']})" for row in rows
        )
    if category == "modules":
        lines = []
        for row in rows:
            bifaciality = row["bifaciality"]
            noct = row["noct_c"]
            suffix = f", bifaciality {bifaciality * 100:.1f}%" if bifaciality is not None else ""
            suffix += f", NOCT {noct:.1f}°C" if noct is not None else ""
            lines.append(f"{row['key']}: {row['power_w']} W, {row['name']}{suffix}")
        return "\n".join(lines)
    if category == "cost-presets":
        return "\n".join(
            f"{row['key']}: buy {row['electricity_cost_per_kwh']} {row['currency']}/kWh, "
            f"sell {row['export_price_per_kwh']} {row['currency']}/kWh, "
            f"battery {row['storage_cost_per_kwh']} {row['currency']}/kWh"
            for row in rows
        )
    if category == "emissions":
        return "\n".join(
            f"{row['key']}: {row['country']}, {row['grid_intensity_gco2_kwh']} gCO2/kWh ({row['year']})" for row in rows
        )
    if category == "load-profiles":
        lines = []
        for row in rows:
            if row["bundled"]:
                status = "bundled"
            elif row["requires_load_profile_file"]:
                status = "your CSV via load_profile_file"
            else:
                status = f"external CSV via rlp_directory: {' or '.join(row['files'])}"
            lines.append(f"{row['key']}: {row['name']} ({status})")
        return "\n".join(lines)
    if category == "battery-models":
        return "\n".join(
            f"{row['key']}: {row['name']} ({row['chemistry']}, {row['cell_format']}; {row['release_phase']})"
            for row in rows
        )
    raise ValueError(f"Unknown list category: {category}")


def _list_options_command(args: argparse.Namespace) -> int:
    rows = _load_options(args.category)
    if args.json:
        print(_json_text(rows, f"the {args.category} list", indent=2))
    else:
        print(_format_options(args.category, rows))
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if "sweep" in config:
        grid = _normalise_sweep_grid(config["sweep"])
        base = {key: value for key, value in config.items() if key != "sweep"}
        for _, run_config in _sweep_run_configs(base, grid):
            _resolved_config_summary(run_config)
    payload = _resolved_config_summary(config)
    if args.json:
        print(_json_text(payload, "the config summary", indent=2))
    else:
        print(f"Config OK: {args.config}")
        print(f"Location: {payload['location']['key'] or 'custom'} ({payload['location']['timezone']})")
        print(f"PV: {payload['pv']['n_modules']} modules, {payload['pv']['system_kwp']:.3f} kWp")
        print(f"Inverter AC rating: {payload['inverter']['ac_rating_kw']:.3f} kW")
        load = payload["load"]
        source = load["load_profile_file"] or f"file not found yet: {load['load_profile_file_error']}"
        print(f"Load profile: {load['load_profile']} at {load['resolution']}, from {source}")
        print(f"Battery: {payload['battery']['capacity_kwh']} kWh")
        print(f"Cost preset: {payload['economics']['cost_preset'] or 'none'}")
        print(f"Emissions: {payload['emissions']['country'] or 'disabled'}")
        period = payload["simulation"].get("period")
        if period is not None:
            print(f"Period: {period['start']} to {period['end']} (end exclusive); lifetime economics are skipped")
    return 0


def _ignore_unused_runner_sections(
    config: dict[str, Any],
    *,
    command: str,
    used_sections: frozenset[str] = frozenset(),
) -> None:
    """Warn about and remove runner tables that this command cannot use."""
    unused = sorted(({"montecarlo", "sweep"} - used_sections) & config.keys())
    if not unused:
        return
    section_names = ", ".join(f"[{name}]" for name in unused)
    warnings.warn(
        f"breos {command} does not use {section_names}; ignoring these runner sections",
        UserWarning,
        stacklevel=2,
    )
    for name in unused:
        config.pop(name, None)


def _normalise_sweep_grid(raw_grid: Any) -> dict[str, list[Any]]:
    """Validate and normalise a ``[sweep]`` section into parameter lists."""
    if not isinstance(raw_grid, dict):
        raise TypeError("Sweep config must contain a [sweep] table with parameter arrays.")

    grid: dict[str, list[Any]] = {}

    def add_entries(entries: dict[str, Any], prefix: str = "") -> None:
        for raw_key, values in entries.items():
            key = raw_key.replace("-", "_")
            dotted_key = f"{prefix}.{key}" if prefix else key
            if isinstance(values, dict):
                add_entries(values, dotted_key)
                continue
            if not isinstance(values, list) or not values:
                raise ValueError(f"sweep.{dotted_key} must be a non-empty array of values")
            if dotted_key in grid:
                raise ValueError(f"Duplicate sweep key '{dotted_key}'")
            grid[dotted_key] = values

    add_entries(raw_grid)

    if not grid:
        raise ValueError("Sweep config must define at least one parameter under [sweep].")

    for key in grid:
        _check_sweep_key(key)

    keys = set(grid)
    for key in keys:
        parts = key.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            if parent in keys:
                raise ValueError(f"Sweep keys '{parent}' and '{key}' conflict")
    return grid


def _sweep_run_configs(
    config: dict[str, Any], grid: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Each grid point's varied values and run config, in sweep order."""
    keys = list(grid)
    runs = []
    for values in itertools.product(*(grid[key] for key in keys)):
        varied = dict(zip(keys, values, strict=True))
        runs.append((varied, _apply_sweep_values(config, varied)))
    return runs


def _check_sweep_key(key: str) -> None:
    """Check one sweep key against the config registry.

    A top-level key must be registered. A dotted key must name a key of a
    nested table (``costs``, ``battery_indoor_model``, ``tariff``,
    ``smart_charging``), and may go one level further only into a free-form
    mapping such as ``tariff.import_prices.P1``. Values are checked later,
    when each run's config is resolved.
    """
    parts = key.split(".")
    top_level = parts[0]
    if top_level not in ALLOWED_CONFIG_KEYS:
        available = ", ".join(sorted(ALLOWED_CONFIG_KEYS))
        raise ValueError(f"Unknown sweep key '{key}'. Available: {available}")
    if top_level == "costs" and len(parts) == 1:
        # A whole [costs] table per run would drop the file's other overrides.
        available = ", ".join(f"costs.{name}" for name in sorted(NESTED_TABLE_SPECS["costs"].keys))
        raise ValueError(f"Unknown sweep key '{key}'. Available: {available}")
    if len(parts) == 1:
        return
    spec = NESTED_TABLE_SPECS.get(top_level)
    if spec is None:
        tables = ", ".join(f"'{name}'" for name in sorted(NESTED_TABLE_SPECS))
        raise ValueError(f"Unknown sweep key '{key}'. Dotted keys are supported only under {tables}.")
    table_key = parts[1]
    if table_key not in spec.keys:
        available = ", ".join(f"{top_level}.{name}" for name in sorted(spec.keys))
        raise ValueError(f"Unknown sweep key '{key}'. Available: {available}")
    if len(parts) == 2:
        return
    if not isinstance(spec.keys[table_key], MappingOf):
        raise ValueError(
            f"Unknown sweep key '{key}'. '{top_level}.{table_key}' is not a table of named entries; "
            f"sweep '{top_level}.{table_key}' itself."
        )
    if len(parts) > 3:
        raise ValueError(
            f"Unknown sweep key '{key}'. '{top_level}.{table_key}' takes one more level, the entry name, "
            f"as in '{top_level}.{table_key}.{parts[2]}'."
        )


def _apply_sweep_values(config: dict[str, Any], varied: dict[str, Any]) -> dict[str, Any]:
    """Return a run config with top-level or dotted sweep values applied.

    A value replaces the base's alternative key, as a CLI flag does
    (``inverter_ac_rating_kw`` over a base ``inverter_loading_ratio``).
    """
    result = override_config(copy.deepcopy(config), varied)
    for key, value in varied.items():
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            child = target.get(part)
            if child is None:
                child = {}
                target[part] = child
            if not isinstance(child, dict):
                raise TypeError(f"Cannot apply sweep key '{key}': '{part}' is not a table/dict")
            target = child
        target[parts[-1]] = value
    return result


def _csv_cell(value: Any) -> Any:
    """Return a stable scalar representation for CSV output."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _scalar_result_items(result: dict[str, Any]) -> dict[str, Any]:
    """Keep only top-level scalar result metrics for a sweep row."""
    scalars: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalars[key] = value
    return scalars


def _write_sweep_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(value) for key, value in row.items()})


def _sweep_row(
    run_idx: int, varied: dict[str, Any], resolved: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """One sweep CSV row: the run, its varied values, resolved sizing and scalar results."""
    row: dict[str, Any] = {
        "run": run_idx,
        "breos_version": package_version(),
    }
    row.update({f"param_{key}": value for key, value in varied.items()})
    row.update(
        {
            "resolved_location": resolved["location"]["key"] or "custom",
            "resolved_n_modules": resolved["pv"]["n_modules"],
            "resolved_battery_kwh": resolved["battery"]["capacity_kwh"],
            "resolved_pv_kwp": resolved["pv"]["system_kwp"],
            "resolved_inverter_ac_kw": resolved["inverter"]["ac_rating_kw"],
        }
    )
    row.update(_scalar_result_items(result))
    return row


def _sweep(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    _ignore_unused_runner_sections(config, command="sweep", used_sections=frozenset({"sweep"}))
    raw_grid = config.pop("sweep", None)
    if raw_grid is None:
        raise ValueError("Sweep config must include a [sweep] section.")

    grid = _normalise_sweep_grid(raw_grid)
    # Every grid point is resolved before the first one runs, so a bad
    # combination (a tariff period the schedule lacks) fails in seconds, not
    # after the runs before it. Each App is built only when it runs, so a
    # finished run's result is not held until the sweep ends.
    runs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    # Runs that differ only in keys the input stage never reads (a tariff, a
    # battery size) share one preparation of weather, PV and load (#181).
    # They run grouped by input configuration, so the one-entry cache serves
    # each group; the CSV keeps the grid order.
    input_keys: list[str | None] = []
    for varied, run_config in _sweep_run_configs(config, grid):
        resolved = resolve_app_config(run_config)
        runs.append((varied, run_config, _config_summary(resolved)))
        input_keys.append(_input_cache_key(resolved.cfg))
    first_seen: dict[str | None, int] = {}
    for index, key in enumerate(input_keys):
        first_seen.setdefault(key, index)
    order = sorted(range(len(runs)), key=lambda index: first_seen[input_keys[index]])
    by_index: dict[int, dict[str, Any]] = {}
    with reuse_prepared_inputs():
        for index in order:
            varied, run_config, summary = runs[index]
            app = App(run_config)
            app.simulate()
            by_index[index] = _sweep_row(index + 1, varied, summary, app.result())
    rows = [by_index[index] for index in range(len(runs))]

    _write_sweep_csv(rows, args.output)

    if args.json:
        payload = {"runs": len(rows), "results_csv": str(args.output), "rows": nonfinite_to_none(rows)}
        print(_json_text(payload, "the sweep summary", indent=2))
    else:
        print(f"Sweep: {len(rows)} runs written to {args.output}")
    return 0


def _montecarlo(args: argparse.Namespace) -> int:
    from breos.montecarlo import MonteCarloSettings, run_montecarlo

    config = _load_config(args.config)
    _ignore_unused_runner_sections(config, command="montecarlo", used_sections=frozenset({"montecarlo"}))
    if args.rlp_directory is not None:
        config["rlp_directory"] = str(args.rlp_directory)

    # Report a typo such as [montecarlo].weather_fille before a missing-file
    # error. The runner validates the full App config before weather access.
    validate_montecarlo_config(config)
    mc_cfg = config.get("montecarlo", {})

    weather_file = args.weather_file or mc_cfg.get("weather_file")
    if not weather_file:
        raise ValueError("Monte Carlo needs a weather file: set [montecarlo].weather_file or pass --weather-file.")

    # Each setting's flag and conversion; None for a setting with no flag or
    # no conversion. A flag beats [montecarlo], and a setting neither gives
    # keeps its MonteCarloSettings default. An unset execution_backend lets
    # run_montecarlo fall back to the top-level key, the same order a Python
    # caller gets.
    options: dict[str, tuple[Any, Callable[[Any], Any] | None]] = {
        "n_runs": (args.runs, int),
        "years_per_run": (args.years, None),
        "load_uncertainty": (args.load_uncertainty, float),
        "load_distribution": (args.load_distribution, str),
        "target_year": (args.target_year, int),
        "weather_start_year": (args.weather_start_year, None),
        "weather_end_year": (args.weather_end_year, None),
        "seed": (args.seed, None),
        "min_load_scale": (None, float),
        "max_load_scale": (None, None),
        "preserve_irradiance_energy": (args.preserve_irradiance_energy, bool),
        "collect_yearly": (args.collect_yearly, bool),
        "n_procs": (args.n_procs, int),
        "execution_backend": (args.execution_backend, None),
    }
    chosen: dict[str, Any] = {}
    for name, (cli_value, convert) in options.items():
        if cli_value is not None:
            value = cli_value
        elif name in mc_cfg:
            value = mc_cfg[name]
        else:
            continue
        chosen[name] = convert(value) if convert is not None else value
    settings = MonteCarloSettings(weather_file=str(weather_file), **chosen)

    result = run_montecarlo(config, settings)

    out_path = args.output or Path("monte_carlo_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.runs.to_csv(out_path, index=False)
    yearly_path = None
    if result.yearly is not None:
        yearly_path = args.yearly_output or out_path.with_name(f"{out_path.stem}_yearly.csv")
        yearly_path.parent.mkdir(parents=True, exist_ok=True)
        result.yearly.to_csv(yearly_path, index=False)

    provenance_path = args.provenance_output or out_path.with_name(f"{out_path.stem}.provenance.json")
    provenance = {
        **result.provenance,
        "command": shlex.join([sys.executable, *sys.argv]),
        "config_file": str(args.config.resolve()),
        "config_file_sha256": _sha256(args.config),
        "weather_file": str(Path(settings.weather_file).resolve()),
        "weather_file_sha256": _sha256(Path(settings.weather_file)),
        "runs_csv": str(out_path),
        "runs_csv_sha256": _sha256(out_path),
        "yearly_csv": str(yearly_path) if yearly_path is not None else None,
        "yearly_csv_sha256": _sha256(yearly_path) if yearly_path is not None else None,
        # A statistic with no defined value is written as null.
        "summary": nonfinite_to_none(result.summary),
    }
    # The load profile's file and hash come from the run's provenance
    # (load_profile.file, load_profile.sha256); these keys repeat them for an
    # external file.
    profile = result.provenance.get("load_profile", {})
    external = profile.get("file") if profile.get("packaged") is False else None
    provenance["external_rlp_file"] = external
    provenance["external_rlp_file_sha256"] = profile.get("sha256") if external is not None else None
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(_json_text(provenance, "the Monte Carlo provenance", indent=2, default=str) + "\n")
    plots_dir = None
    if args.plots:
        from breos.plotting import plot_montecarlo_simulation

        plot_montecarlo_simulation(result.runs, str(out_path.parent), verbose=not args.json)
        plots_dir = out_path.parent / "plots"

    if args.json:
        payload = {
            "result_schema_version": result.provenance["result_schema_version"],
            "currency": result.provenance["currency"],
            "settings": settings.__dict__,
            "summary": nonfinite_to_none(result.summary),
            "available_years": result.available_years,
            "results_csv": str(out_path),
            "yearly_csv": str(yearly_path) if yearly_path is not None else None,
            "provenance_json": str(provenance_path),
        }
        if plots_dir is not None:
            payload["plots_directory"] = str(plots_dir)
        print(_json_text(payload, "the Monte Carlo summary", indent=2))
        return 0

    print(
        f"Monte Carlo: {settings.n_runs} runs x "
        f"{settings.years_per_run or 'config'} years, "
        f"weather years {min(result.available_years)}-{max(result.available_years)} "
        f"({len(result.available_years)} available)"
    )
    print(f"Per-run results written to: {out_path}")
    if yearly_path is not None:
        print(f"Per-year trajectory results written to: {yearly_path}")
    print(f"Provenance written to: {provenance_path}")
    print(f"{'metric':<28}{'runs':>12}{'mean':>12}{'p5':>12}{'p50':>12}{'p95':>12}")
    for metric, stats in result.summary.items():
        runs = f"{stats['count']}/{stats['n_runs']}"
        if "mean" not in stats:
            print(f"{metric:<28}{runs:>12}{'-':>12}{'-':>12}{'-':>12}{'-':>12}")
            continue
        print(
            f"{metric:<28}{runs:>12}{stats['mean']:>12.2f}{stats['p5']:>12.2f}{stats['p50']:>12.2f}{stats['p95']:>12.2f}"
        )
    payback = result.summary.get("payback_year")
    if payback is not None:
        print(
            f"Paid back within the horizon: {payback['count']} of {payback['n_runs']} runs "
            f"({100.0 * payback['payback_probability']:.1f}%). "
            "Payback statistics cover those runs only."
        )
    return 0


def _add_run_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Generate App config override flags from the field registry."""
    for key, field in APP_CONFIG_FIELDS.items():
        if not field.cli_flags:
            continue

        kwargs: dict[str, Any] = {"dest": key, "help": field.cli_help}
        if field.cli_type is not None:
            kwargs["type"] = field.cli_type
        if field.cli_choices is not None:
            kwargs["choices"] = field.cli_choices
        parser.add_argument(*field.cli_flags, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="breos", description="Run BREOS simulations from the command line.")
    parser.add_argument("--version", action="version", version=f"breos {package_version()}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a PV + battery simulation.")
    run.add_argument("--config", type=Path, help="TOML or JSON file with App configuration.")
    _add_run_config_arguments(run)
    run.add_argument("--output", type=Path, help="Write JSON results to this file instead of stdout.")
    run.add_argument("--indent", type=int, default=2, help="JSON indentation. Use 0 for compact output.")
    run.add_argument("--dry-run", action="store_true", help="Validate and print resolved config without simulation.")
    run.set_defaults(func=_run)

    list_parser = subparsers.add_parser("list", help="List packaged option keys.")
    list_parser.add_argument(
        "category",
        choices=("locations", "modules", "cost-presets", "emissions", "load-profiles", "battery-models"),
        help="Packaged option category to list.",
    )
    list_parser.add_argument("--json", action="store_true", help="Write machine-readable JSON.")
    list_parser.set_defaults(func=_list_options_command)

    validate = subparsers.add_parser("validate-config", help="Validate and summarize an App config file.")
    validate.add_argument("config", type=Path, help="TOML or JSON config file.")
    validate.add_argument("--json", action="store_true", help="Write machine-readable JSON.")
    validate.set_defaults(func=_validate_config)

    sweep = subparsers.add_parser("sweep", help="Run a parameter-grid sweep from a config [sweep] section.")
    sweep.add_argument("--config", type=Path, required=True, help="TOML or JSON config file with a [sweep] section.")
    sweep.add_argument("--output", type=Path, default=Path("sweep_results.csv"), help="Combined results CSV path.")
    sweep.add_argument("--json", action="store_true", help="Print machine-readable summary to stdout.")
    sweep.set_defaults(func=_sweep)

    mc = subparsers.add_parser(
        "montecarlo",
        help="Run a Monte Carlo study over weather years and demand uncertainty.",
    )
    mc.add_argument("--config", type=Path, required=True, help="TOML or JSON config file with a [montecarlo] section.")
    mc.add_argument("--weather-file", help="Multi-year historical weather CSV (overrides [montecarlo].weather_file).")
    mc.add_argument("--rlp-directory", type=Path, help="Directory containing a licensed external RLP CSV.")
    mc.add_argument("--runs", type=int, help="Number of Monte Carlo runs (trajectories).")
    mc.add_argument(
        "--years", type=int, dest="years", help="Projection years per run. Defaults to config projection_years."
    )
    mc.add_argument(
        "--load-uncertainty",
        type=float,
        help="Demand uncertainty: normal standard deviation or uniform half-width around 1.",
    )
    mc.add_argument(
        "--load-distribution",
        choices=("normal", "uniform"),
        help="Demand multiplier distribution; uncertainty is sigma for normal or half-width for uniform.",
    )
    mc.add_argument(
        "--target-year", type=int, help="Study calendar year: the weather, load and tariff are all placed on it."
    )
    mc.add_argument("--weather-start-year", type=int, help="First historical weather year eligible for sampling.")
    mc.add_argument("--weather-end-year", type=int, help="Last historical weather year eligible for sampling.")
    mc.add_argument("--seed", type=int, help="Base random seed for reproducible runs.")
    mc.add_argument("--n-procs", type=int, help="Worker processes for independent trajectories (default: 1).")
    mc.add_argument(
        "--execution-backend",
        choices=EXECUTION_BACKENDS,
        help=(
            "Within-day dispatch implementation. 'python' (default) is the numerical reference; "
            "'numba' is the optional compiled backend and needs: pip install \"breos[fast]\"."
        ),
    )
    mc.add_argument("--output", type=Path, help="Per-run results CSV path (default: monte_carlo_results.csv).")
    mc.add_argument("--yearly-output", type=Path, help="Optional per-year trajectory CSV path.")
    mc.add_argument("--provenance-output", type=Path, help="Optional provenance JSON path.")
    mc.add_argument(
        "--collect-yearly",
        action="store_true",
        default=None,
        help="Write one row per run and project year for cost-envelope analysis.",
    )
    mc.add_argument(
        "--preserve-irradiance-energy",
        action="store_true",
        default=None,
        help="Preserve each source hour's irradiance energy during 15-minute resampling.",
    )
    mc.add_argument("--plots", action="store_true", help="Generate Monte Carlo distribution plots next to the CSV.")
    mc.add_argument("--json", action="store_true", help="Write machine-readable JSON summary to stdout.")
    mc.set_defaults(func=_montecarlo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    # ImportError covers a missing optional extra, such as numba for
    # execution_backend="numba".
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"breos: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
