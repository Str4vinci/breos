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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from breos.app import App
from breos.app_config import ALLOWED_CONFIG_KEYS, APP_CONFIG_FIELDS, COST_OVERRIDE_KEYS, resolve_app_config
from breos.degradation import get_battery_model_profile, list_battery_models
from breos.load_profiles import PROFILE_ALIASES, PROFILE_FILES, PROFILE_FILES_15MIN, PROFILE_NAMES
from breos.pv_modules import MODULES
from breos.resources import load_config_json
from breos.solar import resolve_pvwatts_losses


def _package_version() -> str:
    try:
        return version("breos")
    except PackageNotFoundError:
        return "0.1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_rlp_path(config: dict[str, Any]) -> Path | None:
    """Resolve the external load-profile file selected by App semantics."""
    directory = config.get("rlp_directory")
    if directory is None:
        return None
    profile = PROFILE_ALIASES.get(str(config.get("load_profile", "1")).lower(), str(config.get("load_profile", "1")))
    root = Path(directory)
    candidates: list[Path] = []
    if str(config.get("resolution", "h")) in {"15min", "15T"} and profile in PROFILE_FILES_15MIN:
        candidates.append(root / PROFILE_FILES_15MIN[profile])
    if profile in PROFILE_FILES:
        candidates.append(root / PROFILE_FILES[profile])
    if profile in PROFILE_FILES_15MIN:
        candidates.append(root / PROFILE_FILES_15MIN[profile])
    return next((path for path in candidates if path.is_file()), None)


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
    return {key.replace("-", "_"): value for key, value in data.items()}


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config) if args.config else {}

    overrides: dict[str, Any] = {}
    for key, field in APP_CONFIG_FIELDS.items():
        if not field.cli_flags:
            continue
        value = getattr(args, key)
        if value is None:
            continue
        if field.cli_normalizer is not None:
            value = field.cli_normalizer(value)
        if value is None:
            continue
        overrides[key] = value

    return {**config, **overrides}


def _run(args: argparse.Namespace) -> int:
    config = _build_config(args)
    if args.dry_run:
        return _write_payload(_resolved_config_summary(config), args)

    app = App(config)
    app.simulate()
    return _write_payload(app.result(), args)


def _write_payload(data: dict[str, Any], args: argparse.Namespace) -> int:
    indent = args.indent if args.indent > 0 else None
    payload = json.dumps(data, indent=indent)

    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def _resolved_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_app_config(config)
    cfg = resolved.cfg
    inverter_ac_kw = resolved.system_kwp / cfg["inverter_loading_ratio"]
    return {
        "valid": True,
        "location": {
            "key": resolved.loc_key,
            "latitude": resolved.lat,
            "longitude": resolved.lon,
            "timezone": resolved.timezone,
        },
        "pv": {
            "n_modules": cfg["n_modules"],
            "system_kwp": resolved.system_kwp,
            "module": resolved.pv_params.Name,
            "arrays": resolved.pv_arrays or None,
            "tilt": resolved.tilt,
            "azimuth": resolved.azimuth,
            "transposition_model": cfg["transposition_model"],
            "albedo": cfg["albedo"],
            "surface_type": cfg["surface_type"],
            "model_perez": cfg["model_perez"],
            "solar_position": cfg["solar_position"],
            "iam_model": cfg["iam_model"],
            "diffuse_iam": cfg["diffuse_iam"],
            "temperature_model": cfg["temperature_model"],
            "bifacial_model": cfg["bifacial_model"],
            "gcr": cfg["gcr"],
            "pvrow_height": cfg["pvrow_height"],
            "pvrow_pitch": cfg["pvrow_pitch"],
            "pv_loss_overrides": cfg["pv_loss_overrides"],
            "losses": resolve_pvwatts_losses(cfg["pv_loss_overrides"]),
        },
        "inverter": {
            "efficiency": cfg["inverter_efficiency"],
            "loading_ratio": cfg["inverter_loading_ratio"],
            "ac_rating_kw": inverter_ac_kw,
            "dc_coupled": cfg["dc_coupled"],
        },
        "load": {
            "annual_consumption_kwh": cfg["annual_consumption_kwh"],
            "load_profile": cfg["load_profile"],
            "rlp_directory": cfg["rlp_directory"],
            "resolution": cfg["resolution"],
            "start_date": cfg["start_date"],
        },
        "battery": {
            "capacity_kwh": cfg["battery_kwh"],
            "max_charge_power_w": cfg["battery_max_charge_power_w"],
            "max_discharge_power_w": cfg["battery_max_discharge_power_w"],
            "power_limit_c_rate": cfg["battery_power_limit_c_rate"],
            "min_soc": cfg["battery_min_soc"],
            "max_soc": cfg["battery_max_soc"],
            "eol_percentage": cfg["battery_eol_percentage"],
            "round_trip_efficiency": cfg["battery_rte"] if cfg["battery_rte"] is not None else 0.95,
            "degradation_engine": cfg["degradation_engine"],
            "blast_model": cfg["blast_model"],
            "model_profile": (
                get_battery_model_profile(cfg["blast_model"]).as_dict() if cfg["blast_model"] is not None else None
            ),
        },
        "economics": {
            "cost_preset": cfg["cost_preset"],
            "projection_years": cfg["projection_years"],
            "inflation_rate": cfg["inflation_rate"],
            "sell_price_inflation": cfg["sell_price_inflation"],
            "discount_rate": cfg["discount_rate"],
        },
        "emissions": {
            "country": cfg["emissions_country"],
            "enabled": resolved.emissions_params is not None,
            "export_factor_gco2_kwh": cfg["export_emissions_factor_gco2_kwh"],
        },
        "notes": [
            "This is a resolved configuration check only; no weather fetch or simulation was run.",
            "Packaged defaults are examples. Replace weather, load, PV, inverter, cost, and emissions inputs for real studies.",
        ],
    }


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
                "electricity_cost_eur_kwh": value.get("electricity_cost"),
                "export_price_eur_kwh": value.get("electricity_sold_cost"),
                "storage_cost_eur_kwh": value.get("storage_cost_per_kwh"),
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
        bundled = {"1"}
        aliases_by_key: dict[str, list[str]] = {}
        for alias, key in PROFILE_ALIASES.items():
            aliases_by_key.setdefault(key, []).append(alias)
        return [
            {
                "key": key,
                "name": name,
                "aliases": ", ".join(sorted(aliases_by_key.get(key, []))) or None,
                "bundled": key in bundled,
                "requires_rlp_directory": key not in bundled,
            }
            for key, name in sorted(PROFILE_NAMES.items())
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
            f"{row['key']}: buy {row['electricity_cost_eur_kwh']} EUR/kWh, "
            f"sell {row['export_price_eur_kwh']} EUR/kWh, battery {row['storage_cost_eur_kwh']} EUR/kWh"
            for row in rows
        )
    if category == "emissions":
        return "\n".join(
            f"{row['key']}: {row['country']}, {row['grid_intensity_gco2_kwh']} gCO2/kWh ({row['year']})" for row in rows
        )
    if category == "load-profiles":
        lines = []
        for row in rows:
            name = row["name"].replace(" (external file required)", "")
            status = "bundled" if row["bundled"] else "external CSV required via rlp_directory"
            alias_text = f"; aliases: {row['aliases']}" if row.get("aliases") else ""
            lines.append(f"{row['key']}: {name} ({status}{alias_text})")
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
        print(json.dumps(rows, indent=2))
    else:
        print(_format_options(args.category, rows))
    return 0


def _validate_config(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if "sweep" in config:
        _normalise_sweep_grid(config["sweep"])
    payload = _resolved_config_summary(config)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Config OK: {args.config}")
        print(f"Location: {payload['location']['key'] or 'custom'} ({payload['location']['timezone']})")
        print(f"PV: {payload['pv']['n_modules']} modules, {payload['pv']['system_kwp']:.3f} kWp")
        print(f"Inverter AC rating: {payload['inverter']['ac_rating_kw']:.3f} kW")
        print(f"Load profile: {payload['load']['load_profile']} at {payload['load']['resolution']}")
        print(f"Battery: {payload['battery']['capacity_kwh']} kWh")
        print(f"Cost preset: {payload['economics']['cost_preset'] or 'none'}")
        print(f"Emissions: {payload['emissions']['country'] or 'disabled'}")
    return 0


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
        top_level, separator, nested = key.partition(".")
        if top_level not in ALLOWED_CONFIG_KEYS:
            available = ", ".join(sorted(ALLOWED_CONFIG_KEYS))
            raise ValueError(f"Unknown sweep key '{key}'. Available: {available}")
        if separator and top_level != "costs":
            available = ", ".join(f"costs.{name}" for name in sorted(COST_OVERRIDE_KEYS))
            raise ValueError(
                f"Unknown sweep key '{key}'. Dotted keys are supported only under 'costs'. Available: {available}"
            )
        if top_level == "costs" and (not separator or nested not in COST_OVERRIDE_KEYS):
            available = ", ".join(f"costs.{name}" for name in sorted(COST_OVERRIDE_KEYS))
            raise ValueError(f"Unknown sweep key '{key}'. Available: {available}")

    keys = set(grid)
    for key in keys:
        parts = key.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            if parent in keys:
                raise ValueError(f"Sweep keys '{parent}' and '{key}' conflict")
    return grid


def _apply_sweep_values(config: dict[str, Any], varied: dict[str, Any]) -> dict[str, Any]:
    """Return a run config with top-level or dotted sweep values applied."""
    result = copy.deepcopy(config)
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


def _sweep(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    raw_grid = config.pop("sweep", None)
    if raw_grid is None:
        raise ValueError("Sweep config must include a [sweep] section.")

    grid = _normalise_sweep_grid(raw_grid)
    param_keys = list(grid)
    rows: list[dict[str, Any]] = []

    for run_idx, values in enumerate(itertools.product(*(grid[key] for key in param_keys)), start=1):
        varied = dict(zip(param_keys, values))
        run_config = _apply_sweep_values(config, varied)
        resolved = _resolved_config_summary(run_config)

        app = App(run_config)
        app.simulate()
        result = app.result()

        row: dict[str, Any] = {
            "run": run_idx,
            "breos_version": _package_version(),
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
        rows.append(row)

    _write_sweep_csv(rows, args.output)

    if args.json:
        print(json.dumps({"runs": len(rows), "results_csv": str(args.output), "rows": rows}, indent=2))
    else:
        print(f"Sweep: {len(rows)} runs written to {args.output}")
    return 0


def _montecarlo(args: argparse.Namespace) -> int:
    from breos.montecarlo import MonteCarloSettings, run_montecarlo

    config = _load_config(args.config)
    mc_cfg = config.get("montecarlo", {}) if isinstance(config.get("montecarlo"), dict) else {}
    if args.rlp_directory is not None:
        config["rlp_directory"] = str(args.rlp_directory)

    weather_file = args.weather_file or mc_cfg.get("weather_file")
    if not weather_file:
        raise ValueError("Monte Carlo needs a weather file: set [montecarlo].weather_file or pass --weather-file.")

    def _pick(cli_value: Any, key: str, default: Any) -> Any:
        return cli_value if cli_value is not None else mc_cfg.get(key, default)

    settings = MonteCarloSettings(
        weather_file=str(weather_file),
        n_runs=int(_pick(args.runs, "n_runs", 100)),
        years_per_run=_pick(args.years, "years_per_run", None),
        load_uncertainty=float(_pick(args.load_uncertainty, "load_uncertainty", 0.10)),
        load_distribution=str(_pick(args.load_distribution, "load_distribution", "normal")),
        target_year=int(_pick(args.target_year, "target_year", 2025)),
        weather_start_year=_pick(args.weather_start_year, "weather_start_year", None),
        weather_end_year=_pick(args.weather_end_year, "weather_end_year", None),
        seed=_pick(args.seed, "seed", None),
        min_load_scale=float(mc_cfg.get("min_load_scale", 0.0)),
        max_load_scale=mc_cfg.get("max_load_scale"),
        preserve_irradiance_energy=bool(_pick(args.preserve_irradiance_energy, "preserve_irradiance_energy", False)),
        collect_yearly=bool(_pick(args.collect_yearly, "collect_yearly", False)),
        n_procs=int(_pick(args.n_procs, "n_procs", 1)),
        execution_backend=str(_pick(args.execution_backend, "execution_backend", "python")),
    )

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
        "summary": result.summary,
    }
    rlp_path = _external_rlp_path(config)
    provenance["external_rlp_file"] = str(rlp_path.resolve()) if rlp_path is not None else None
    provenance["external_rlp_file_sha256"] = _sha256(rlp_path) if rlp_path is not None else None
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    plots_dir = None
    if args.plots:
        from breos.plotting import plot_montecarlo_simulation

        plot_montecarlo_simulation([], str(out_path.parent), full_df=result.runs, verbose=not args.json)
        plots_dir = out_path.parent / "plots"

    if args.json:
        payload = {
            "settings": settings.__dict__,
            "summary": result.summary,
            "available_years": result.available_years,
            "results_csv": str(out_path),
            "yearly_csv": str(yearly_path) if yearly_path is not None else None,
            "provenance_json": str(provenance_path),
        }
        if plots_dir is not None:
            payload["plots_directory"] = str(plots_dir)
        print(json.dumps(payload, indent=2))
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
    print(f"{'metric':<28}{'mean':>12}{'p5':>12}{'p50':>12}{'p95':>12}")
    for metric, stats in result.summary.items():
        print(f"{metric:<28}{stats['mean']:>12.2f}{stats['p5']:>12.2f}{stats['p50']:>12.2f}{stats['p95']:>12.2f}")
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
        if field.cli_action is not None:
            kwargs["action"] = field.cli_action
            # A missing boolean flag must not overwrite a config-file value.
            kwargs["default"] = None
        parser.add_argument(*field.cli_flags, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="breos", description="Run BREOS simulations from the command line.")
    parser.add_argument("--version", action="version", version=f"breos {_package_version()}")

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
    mc.add_argument("--target-year", type=int, help="Calendar year the weather index is mapped to.")
    mc.add_argument("--weather-start-year", type=int, help="First historical weather year eligible for sampling.")
    mc.add_argument("--weather-end-year", type=int, help="Last historical weather year eligible for sampling.")
    mc.add_argument("--seed", type=int, help="Base random seed for reproducible runs.")
    mc.add_argument("--n-procs", type=int, help="Worker processes for independent trajectories (default: 1).")
    mc.add_argument(
        "--execution-backend",
        choices=("python", "numba"),
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
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"breos: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
