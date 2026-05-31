"""Command line interface for BREOS."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from breos.app import App


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
    _add_override(overrides, "tilt", args.tilt)
    _add_override(overrides, "azimuth", args.azimuth)
    _add_override(overrides, "resolution", args.resolution)
    _add_override(overrides, "projection_years", args.projection_years)
    _add_override(overrides, "inflation_rate", args.inflation_rate)
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
    app = App(config)
    app.simulate()
    indent = args.indent if args.indent > 0 else None
    payload = json.dumps(app.result(), indent=indent)

    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
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
    run.add_argument("--tilt", type=float, help="PV tilt angle in degrees.")
    run.add_argument("--azimuth", type=float, help="PV surface azimuth in degrees.")
    run.add_argument("--resolution", choices=("h", "15min"), help="Simulation time resolution.")
    run.add_argument("--projection-years", type=int, help="Economic projection horizon.")
    run.add_argument("--inflation-rate", type=float, help="Annual electricity price inflation.")
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
    run.set_defaults(func=_run)

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
