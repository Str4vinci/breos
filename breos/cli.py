"""Command line interface for BREOS."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from breos.app import App
from breos.app_config import resolve_app_config
from breos.load_profiles import PROFILE_ALIASES, PROFILE_NAMES
from breos.pv_modules import MODULES
from breos.resources import load_config_json
from breos.solar import (
    DIFFUSE_IAM_METHODS,
    PEREZ_MODELS,
    SOLAR_POSITION_METHODS,
    SURFACE_TYPES,
    TRANSPOSITION_MODELS,
    resolve_pvwatts_losses,
)


def _package_version() -> str:
    try:
        return version("breos")
    except PackageNotFoundError:
        return "0.1.0"


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


def _add_override(overrides: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        overrides[key] = value


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config) if args.config else {}

    overrides: dict[str, Any] = {}
    _add_override(overrides, "location", args.location.lower() if args.location else None)
    _add_override(overrides, "n_modules", args.n_modules)
    _add_override(overrides, "annual_consumption_kwh", args.annual_consumption_kwh)
    _add_override(overrides, "battery_kwh", args.battery_kwh)
    _add_override(overrides, "pv_module", args.pv_module)
    _add_override(overrides, "load_profile", args.load_profile)
    _add_override(overrides, "rlp_directory", str(args.rlp_directory) if args.rlp_directory else None)
    _add_override(overrides, "tilt", args.tilt)
    _add_override(overrides, "azimuth", args.azimuth)
    _add_override(overrides, "transposition_model", args.transposition_model)
    _add_override(overrides, "albedo", args.albedo)
    _add_override(overrides, "surface_type", args.surface_type)
    _add_override(overrides, "model_perez", args.model_perez)
    _add_override(overrides, "solar_position", args.solar_position)
    _add_override(overrides, "diffuse_iam", args.diffuse_iam)
    _add_override(overrides, "resolution", args.resolution)
    _add_override(overrides, "projection_years", args.projection_years)
    _add_override(overrides, "inflation_rate", args.inflation_rate)
    _add_override(overrides, "sell_price_inflation", args.sell_price_inflation)
    _add_override(overrides, "discount_rate", args.discount_rate)
    _add_override(overrides, "pv_degradation_rate", args.pv_degradation_rate)
    _add_override(overrides, "calendar_model", args.calendar_model)
    _add_override(overrides, "dc_coupled", args.dc_coupled)
    _add_override(overrides, "inverter_efficiency", args.inverter_efficiency)
    _add_override(overrides, "inverter_loading_ratio", args.inverter_loading_ratio)
    _add_override(overrides, "start_date", args.start_date)

    if args.cost_preset:
        overrides["cost_preset"] = args.cost_preset.replace("-", "_")
    if args.emissions_country:
        overrides["emissions_country"] = args.emissions_country.upper()

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
            "diffuse_iam": cfg["diffuse_iam"],
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
            "min_soc": cfg["battery_min_soc"],
            "max_soc": cfg["battery_max_soc"],
            "eol_percentage": cfg["battery_eol_percentage"],
            "round_trip_efficiency": cfg["battery_rte"] if cfg["battery_rte"] is not None else 0.95,
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

    raise ValueError(f"Unknown list category: {category}")


def _format_options(category: str, rows: list[dict[str, Any]]) -> str:
    if category == "locations":
        return "\n".join(
            f"{row['key']}: {row['name']} ({row['latitude']}, {row['longitude']}, {row['timezone']})" for row in rows
        )
    if category == "modules":
        return "\n".join(f"{row['key']}: {row['power_w']} W, {row['name']}" for row in rows)
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
    for key, values in raw_grid.items():
        normalised_key = key.replace("-", "_")
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{key} must be a non-empty array of values")
        grid[normalised_key] = values

    if not grid:
        raise ValueError("Sweep config must define at least one parameter under [sweep].")
    return grid


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
        run_config = {**config, **varied}
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
        target_year=int(_pick(args.target_year, "target_year", 2025)),
        seed=_pick(args.seed, "seed", None),
        min_load_scale=float(mc_cfg.get("min_load_scale", 0.0)),
        max_load_scale=mc_cfg.get("max_load_scale"),
    )

    result = run_montecarlo(config, settings)

    out_path = args.output or Path("monte_carlo_results.csv")
    result.runs.to_csv(out_path, index=False)
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
    print(f"{'metric':<28}{'mean':>12}{'p5':>12}{'p50':>12}{'p95':>12}")
    for metric, stats in result.summary.items():
        print(f"{metric:<28}{stats['mean']:>12.2f}{stats['p5']:>12.2f}{stats['p50']:>12.2f}{stats['p95']:>12.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="breos", description="Run BREOS simulations from the command line.")
    parser.add_argument("--version", action="version", version=f"breos {_package_version()}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a PV + battery simulation.")
    run.add_argument("--config", type=Path, help="TOML or JSON file with App configuration.")
    run.add_argument("--location", help="Location preset key, for example 'porto'.")
    run.add_argument("--n-modules", type=int, help="Number of PV modules.")
    run.add_argument("--annual-consumption-kwh", type=float, help="Annual electricity demand in kWh.")
    run.add_argument("--battery-kwh", type=float, help="Battery capacity in kWh.")
    run.add_argument("--cost-preset", help="Cost preset key, for example 'residential-pt'.")
    run.add_argument("--emissions-country", help="Country code for emissions, for example 'pt'.")
    run.add_argument("--pv-module", help="PV module catalogue key.")
    run.add_argument("--load-profile", help="Load profile type.")
    run.add_argument("--rlp-directory", type=Path, help="Directory containing licensed external RLP CSV files.")
    run.add_argument("--tilt", type=float, help="PV tilt angle in degrees.")
    run.add_argument("--azimuth", type=float, help="PV surface azimuth in degrees.")
    run.add_argument(
        "--transposition-model",
        "--sky-model",
        dest="transposition_model",
        choices=TRANSPOSITION_MODELS,
        help="Sky-diffusion model for POA transposition (default: isotropic).",
    )
    run.add_argument(
        "--albedo", type=float, help="Ground reflectance 0-1 (default: pvlib 0.25). Excludes --surface-type."
    )
    run.add_argument(
        "--surface-type",
        choices=SURFACE_TYPES,
        help="Named ground cover mapped to an albedo (alternative to --albedo).",
    )
    run.add_argument(
        "--perez-model",
        dest="model_perez",
        choices=PEREZ_MODELS,
        help="Perez coefficient set (only used with --transposition-model perez).",
    )
    run.add_argument(
        "--solar-position",
        dest="solar_position",
        choices=SOLAR_POSITION_METHODS,
        help=(
            "Where within each timestep the sun position is evaluated. 'mid-interval' matches "
            "PVWatts/SAM for interval-averaged weather (default: interval-start)."
        ),
    )
    run.add_argument(
        "--diffuse-iam",
        dest="diffuse_iam",
        choices=DIFFUSE_IAM_METHODS,
        help=(
            "Whether IAM is also applied to the diffuse POA components. 'marion' weighs sky- and "
            "ground-diffuse with the view-factor-integrated ashrae IAM (default: none, beam-only)."
        ),
    )
    run.add_argument("--resolution", choices=("h", "15min"), help="Simulation time resolution.")
    run.add_argument("--projection-years", type=int, help="Economic projection horizon.")
    run.add_argument("--inflation-rate", type=float, help="Annual electricity price inflation.")
    run.add_argument(
        "--sell-price-inflation", type=float, help="Annual inflation of the grid export (sell) price. Default 0."
    )
    run.add_argument("--discount-rate", type=float, help="Discount rate for NPV calculations.")
    run.add_argument("--pv-degradation-rate", type=float, help="Annual PV degradation rate.")
    run.add_argument("--calendar-model", help="Battery calendar aging model.")
    run.add_argument("--dc-coupled", dest="dc_coupled", action="store_true", default=None, help="Use DC coupling.")
    run.add_argument("--ac-coupled", dest="dc_coupled", action="store_false", help="Use AC coupling.")
    run.add_argument("--inverter-efficiency", type=float, help="Inverter efficiency.")
    run.add_argument("--inverter-loading-ratio", type=float, help="DC/AC oversizing ratio.")
    run.add_argument("--start-date", help="Simulation start date, YYYY-MM-DD.")
    run.add_argument("--output", type=Path, help="Write JSON results to this file instead of stdout.")
    run.add_argument("--indent", type=int, default=2, help="JSON indentation. Use 0 for compact output.")
    run.add_argument("--dry-run", action="store_true", help="Validate and print resolved config without simulation.")
    run.set_defaults(func=_run)

    list_parser = subparsers.add_parser("list", help="List packaged option keys.")
    list_parser.add_argument(
        "category",
        choices=("locations", "modules", "cost-presets", "emissions", "load-profiles"),
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
    mc.add_argument("--runs", type=int, help="Number of Monte Carlo runs (trajectories).")
    mc.add_argument(
        "--years", type=int, dest="years", help="Projection years per run. Defaults to config projection_years."
    )
    mc.add_argument(
        "--load-uncertainty", type=float, help="Std-dev of the annual demand multiplier (Normal, mean 1.0)."
    )
    mc.add_argument("--target-year", type=int, help="Calendar year the weather index is mapped to.")
    mc.add_argument("--seed", type=int, help="Base random seed for reproducible runs.")
    mc.add_argument("--output", type=Path, help="Per-run results CSV path (default: monte_carlo_results.csv).")
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
