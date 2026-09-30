"""Benchmark the optimizer's Python and Numba execution backends end to end.

The timed call is the public ``breos.optimization.optimize_system_multi_objective``
on a projected, fixed-target time-of-use study. Every timed run therefore
includes candidate PV production, year-by-year dispatch, degradation, battery
state propagation and replacement, the tariff, economics, the NSGA-II search
and result assembly. It is not a kernel or dispatch timing.

Each resolution case goes through these steps, each in a fresh child process:

1. Parity. Both backends run the optimizer with the same seed and settings,
   one worker each, and must give the same Pareto frame (rows put in design
   order), evaluation count and generation count. A fixed witness design is
   evaluated with ``evaluate_projected_design`` and run through ``breos.App``
   on both backends: the metrics, yearly and financial tables, every
   year's per-step ledger and the degradation state must be equal. The
   witness must charge from the grid and replace its battery, or the check
   proves nothing. A case whose parity fails is not timed.
2. Cold. One run per backend. The Numba child gets an empty
   ``NUMBA_CACHE_DIR``, so its time includes the first compilation. The
   Python child's first run is the cold baseline.
3. Warm. ``--warm-repeats`` children per backend. Each runs the real
   optimizer once, untimed, at the study size, and then times one run.

Setup is timed at the ``SolarDesignProblem`` construction boundary: config
resolution, tariff and smart-charging resolution, and site and model
preparation. ``total - problem_construction`` is reported as the residual
search and wrapper time. It includes NSGA-II and result assembly as well as
the candidate simulations, so it is not a pure dispatch time.

Peak RSS is the child's ``ru_maxrss`` high-water mark, which includes native
NumPy and Numba allocations. A warm child's peak also covers its warm-up run,
which has the same size as the timed run.

Parallel workers (``--n-procs`` above 1) can change the timing and the order
in which pymoo meets its candidates. Parity always runs with one worker.

Usage:
    python tools/benchmark_optimization.py --output benchmark.json
    python tools/benchmark_optimization.py --resolution h --pop-size 4 --n-gen 1 \\
        --warm-repeats 1 --projection-years 3
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Run as a script, sys.path[0] is tools/. The children import this module
    # as tools.benchmark_optimization, so the parent resolves it the same way.
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.benchmark_montecarlo import machine_info as _montecarlo_machine_info

REPORT_SCHEMA = "breos_optimization_benchmark_v1"
DEFAULT_WEATHER_FILE = PROJECT_ROOT / "validation" / "data" / "weather" / "porto_tmy_2005_2023_pvgis-sarah3.csv.gz"
# The coordinates of the PVGIS request that produced the default weather file.
DEFAULT_LATITUDE = 41.1579
DEFAULT_LONGITUDE = -8.6291
DEFAULT_ANNUAL_CONSUMPTION_KWH = 3500.0
RESOLUTIONS = ("h", "15min")
BACKENDS = ("python", "numba")
# The PT tariff schedule is in civil time, and 2026 is its calendar year.
TIMEZONE = "Europe/Lisbon"
STUDY_YEAR = 2026
TARIFF: dict[str, Any] = {
    "schedule": "pt_mainland_2026_daily_bi",
    "currency": "EUR",
    "import_prices": {"peak": 0.50, "off_peak": 0.10},
    "export_prices": {"all": 0.03},
    "fixed_charge_per_day": 0.40,
}
FIXED_TARGET: dict[str, Any] = {
    "mode": "fixed_target",
    "target_usable_fraction": 0.5,
    "charge_periods": ["off_peak"],
    "discharge_periods": ["peak"],
    "grid_charge_efficiency": 0.95,
}
# A 93 % end of life makes the witness pack reach its replacement within a
# three-year horizon.
BATTERY_EOL_PERCENTAGE = 0.93
CONSTRAINTS = {"budget": 100000.0, "max_area_m2": 100.0}
AZIMUTH = 180.0
# (modules, battery kWh, tilt): the replacement-sensitive design of the
# backend parity tests.
WITNESS_DESIGN = (8, 5.0, 35.0)
DESIGN_COLUMNS = ("Modules", "Battery_kWh", "Tilt", "Azimuth")
# Native thread pools stay at one thread, so the backends are compared on the
# same single core and a child does not compete with itself.
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}
MAX_LISTED_DIFFERENCES = 20

_CHILD_BOOTSTRAP = (
    "import sys; sys.path.insert(0, {root!r}); "
    "from tools.benchmark_optimization import child_main; sys.exit(child_main(sys.argv[1:]))"
)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} must be 1 or more")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} must be 0 or more")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a finite number above 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark optimize_system_multi_objective on the Python and Numba backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weather-file",
        type=Path,
        default=DEFAULT_WEATHER_FILE,
        help="One complete year of hourly or 15-minute weather, .csv or .csv.gz (default: the local Porto TMY)",
    )
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--annual-consumption-kwh", type=_positive_float, default=DEFAULT_ANNUAL_CONSUMPTION_KWH)
    parser.add_argument("--resolution", nargs="+", choices=RESOLUTIONS, default=list(RESOLUTIONS))
    parser.add_argument("--projection-years", type=_positive_int, default=3)
    parser.add_argument("--pop-size", type=_positive_int, default=8)
    parser.add_argument("--n-gen", type=_positive_int, default=2)
    parser.add_argument("--seed", type=_non_negative_int, default=42)
    parser.add_argument(
        "--n-procs",
        type=_positive_int,
        default=1,
        help="Workers for the timed runs; above 1 can change timing and candidate order. Parity uses 1.",
    )
    parser.add_argument("--warm-repeats", type=_positive_int, default=3, help="Warm children per backend")
    parser.add_argument("--output", type=Path, help="Write the JSON report here")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and check the command line; an invalid value exits with status 2."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(set(args.resolution)) != len(args.resolution):
        parser.error(f"--resolution repeats a value: {' '.join(args.resolution)}")
    if not -90.0 <= args.latitude <= 90.0:
        parser.error(f"--latitude {args.latitude} is outside -90 to 90")
    if not -180.0 <= args.longitude <= 180.0:
        parser.error(f"--longitude {args.longitude} is outside -180 to 180")
    name = args.weather_file.name
    if not (name.endswith(".csv") or name.endswith(".csv.gz")):
        parser.error(f"--weather-file must be a .csv or .csv.gz file: {args.weather_file}")
    if not args.weather_file.is_file():
        parser.error(f"--weather-file does not exist: {args.weather_file}")
    return args


def missing_dependencies() -> list[str]:
    """The optional packages the benchmark needs and cannot import."""
    return [name for name in ("pymoo", "numba") if importlib.util.find_spec(name) is None]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    """A digest of a frame's index and values, for the report."""
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def stage_weather(weather_file: Path, staging_dir: Path) -> tuple[Path, str, dict[str, Any]]:
    """Put the weather file in ``staging_dir`` as a plain CSV that ``load_weather`` accepts.

    ``load_weather`` reads only ``.csv`` names of the form
    ``{location}_{type}_{years}_{source}.csv``. A ``.csv.gz`` file is
    decompressed, and a name outside that form is replaced by a canonical
    one. A metadata sidecar that is bound to the original file (its
    ``weather_sha256`` matches) is bound again to the staged CSV, so the
    loader reads the same timing metadata. Returns the staged path, the
    location key to request, and the provenance record.
    """
    from breos.weather import parse_weather_filename

    source = weather_file.resolve()
    compressed = source.name.endswith(".gz")
    plain_name = source.name[:-3] if compressed else source.name
    parsed = parse_weather_filename(plain_name)
    staged_name = plain_name if parsed is not None else f"benchmark_tmy_{STUDY_YEAR}_{STUDY_YEAR}_input.csv"
    location = parsed["location"] if parsed is not None else "benchmark"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / staged_name
    if compressed:
        with gzip.open(source, "rb") as reader, open(staged, "wb") as writer:
            shutil.copyfileobj(reader, writer)
    else:
        shutil.copyfile(source, staged)

    original_sha256 = sha256_file(source)
    staged_sha256 = sha256_file(staged)
    record: dict[str, Any] = {
        "path": str(source),
        "sha256": original_sha256,
        "compressed": compressed,
        "staged_name": staged_name,
        "staged_sha256": staged_sha256,
        "sidecar": {"status": "absent"},
    }
    sidecar = Path(f"{source}.metadata.json")
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        bound = isinstance(payload, dict) and payload.get("weather_sha256") == original_sha256
        record["sidecar"] = {
            "status": "bound" if bound else "digest_mismatch",
            "path": str(sidecar),
            "sha256": sha256_file(sidecar),
            "payload": payload,
        }
        if bound:
            restaged = {**payload, "weather_sha256": staged_sha256}
            Path(f"{staged}.metadata.json").write_text(json.dumps(restaged, indent=2, sort_keys=True) + "\n")
            record["sidecar"]["restaged_for"] = staged_name
    return staged, location, record


def check_complete_year(frame: pd.DataFrame, resolution: str, label: str) -> None:
    """Raise unless ``frame`` holds exactly one gap-free year at ``resolution``."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError(f"{label} has no timestamped rows")
    if index.tz is None:
        raise ValueError(f"{label} has timestamps without a timezone")
    if index.has_duplicates:
        raise ValueError(f"{label} repeats {int(index.duplicated().sum())} timestamp(s)")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are not in increasing order")
    step = pd.Timedelta(resolution if resolution != "h" else "1h")
    gaps = np.flatnonzero(np.asarray(index[1:] - index[:-1]) != step.to_timedelta64())
    if gaps.size:
        first = index[gaps[0]]
        raise ValueError(f"{label} is not on a regular {resolution!r} step: the first break follows {first}")
    expected = len(pd.date_range(f"{STUDY_YEAR}-01-01", f"{STUDY_YEAR + 1}-01-01", freq=step, inclusive="left"))
    if len(index) != expected:
        raise ValueError(
            f"{label} has {len(index)} rows at {resolution!r}; one year of {STUDY_YEAR} needs {expected}. "
            "Supply one complete year."
        )


def prepare_case_inputs(
    staged: Path,
    location: str,
    resolution: str,
    *,
    latitude: float,
    longitude: float,
    annual_consumption_kwh: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load one case's weather and load through BREOS's own readers, and check them."""
    from breos.app_inputs import remap_tmy_year, require_full_year_weather, resample_hourly_weather
    from breos.app_inputs import weather_input_frequency as input_frequency
    from breos.load_profiles import load_profile
    from breos.weather import WEATHER_METADATA_KEY, load_weather, resample_to_15min

    steps: list[str] = [f"load_weather(location={location!r}, weather_dir=<staging>)"]
    weather = load_weather(location=location, weather_dir=str(staged.parent))
    if weather is None:
        raise ValueError(f"load_weather found no usable weather in {staged}")
    index = weather.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{staged.name} has no datetime index")
    if index.tz is None:
        weather.index = index.tz_localize("UTC")
        steps.append("naive timestamps read as UTC, as App reads them")
    frequency = input_frequency(weather)
    hourly = bool(frequency) and "h" in str(frequency).lower() and "15" not in str(frequency)
    quarter_hour = bool(frequency) and "15" in str(frequency)
    if not (hourly or quarter_hour):
        raise ValueError(f"{staged.name} has step {frequency!r}; the benchmark needs hourly or 15-minute weather")
    if resolution == "h" and not hourly:
        raise ValueError(f"the hourly case needs hourly weather; {staged.name} has step {frequency!r}")

    weather = remap_tmy_year(weather, STUDY_YEAR)
    steps.append(f"remap_tmy_year(target_year={STUDY_YEAR})")
    if resolution == "15min" and hourly:
        weather = resample_hourly_weather(
            weather, resolution, latitude=latitude, longitude=longitude, resample=resample_to_15min
        )
        steps.append("resample_to_15min(latitude, longitude), the App default")
    require_full_year_weather(weather, STUDY_YEAR, resolution, TIMEZONE)
    check_complete_year(weather, resolution, f"weather from {staged.name}")

    load = load_profile(
        "demandlib_h0", annual_consumption_kwh, start_date=f"{STUDY_YEAR}-01-01", freq=resolution, timezone=TIMEZONE
    )
    check_complete_year(load, resolution, "the demandlib H0 load")
    hours_per_step = 1.0 if resolution == "h" else 0.25
    record = {
        "weather": {
            "input_frequency": str(frequency),
            "transformations": steps,
            "rows": len(weather),
            "first": str(weather.index[0]),
            "last": str(weather.index[-1]),
            "frame_sha256": frame_sha256(weather),
            "metadata": deepcopy(weather.attrs.get(WEATHER_METADATA_KEY, {})),
        },
        "load": {
            "profile": "demandlib_h0",
            "path": "load_profile(freq=resolution, timezone=Europe/Lisbon)",
            "annual_consumption_kwh": annual_consumption_kwh,
            "annual_kwh_measured": float(load.iloc[:, 0].sum() * hours_per_step / 1000.0),
            "rows": len(load),
            "first": str(load.index[0]),
            "last": str(load.index[-1]),
            "frame_sha256": frame_sha256(load),
        },
    }
    return weather, load, record


def study_configs(
    resolution: str,
    projection_years: int,
    *,
    latitude: float,
    longitude: float,
    annual_consumption_kwh: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The App config of the witness design, and the optimizer config derived from it.

    The optimizer config takes its costs, rates, module and inverter from the
    App's resolved config, as the three-year tariff App test does, so the
    witness run through App is the same design the optimizer scores.
    """
    from breos import App
    from breos.economics import COST_CONFIG_KEY_TO_PARAM

    n_modules, battery_kwh, tilt = WITNESS_DESIGN
    app_config: dict[str, Any] = {
        "location": {"latitude": latitude, "longitude": longitude, "timezone": TIMEZONE},
        "n_modules": n_modules,
        "annual_consumption_kwh": annual_consumption_kwh,
        "battery_kwh": battery_kwh,
        "tilt": tilt,
        "azimuth": AZIMUTH,
        "start_date": f"{STUDY_YEAR}-01-01",
        "resolution": resolution,
        "projection_years": projection_years,
        "battery_eol_percentage": BATTERY_EOL_PERCENTAGE,
        "battery_temperature": "weather",
        "execution_backend": "python",
        "tariff": deepcopy(TARIFF),
        "smart_charging": deepcopy(FIXED_TARGET),
    }
    resolved = App(app_config)._resolved
    cfg, params = resolved.cfg, resolved.cost_params
    costs = {
        key: getattr(params, field)
        for key, field in COST_CONFIG_KEY_TO_PARAM.items()
        if key not in ("electricity_cost", "electricity_sold_cost", "daily_power_cost")
    }
    costs["dc_ac_ratio"] = cfg["inverter_loading_ratio"]
    optimizer_config: dict[str, Any] = {
        "location": deepcopy(cfg["location"]),
        "optimization": {"objective_basis": "projected"},
        "constraints": dict(CONSTRAINTS),
        "mode": {"fixed_azimuth": AZIMUTH},
        "simulation": {"resolution": resolution, "years_projection": projection_years},
        "pv": {"module": cfg["pv_module"], "degradation_rate": cfg["pv_degradation_rate"]},
        "battery": {
            "temperature": "weather",
            "eol_percentage": BATTERY_EOL_PERCENTAGE,
            "enable_resistance_fade": cfg["enable_resistance_fade"],
        },
        "costs": costs,
        "financials": {
            "discount_rate": cfg["discount_rate"],
            "inflation_rate": cfg["inflation_rate"],
            "sell_price_inflation": cfg["sell_price_inflation"],
        },
        "inverter_efficiency": cfg["inverter_efficiency"],
        "tariff": deepcopy(TARIFF),
        "smart_charging": deepcopy(FIXED_TARGET),
    }
    return app_config, optimizer_config


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------


def canonical_pareto(pareto: pd.DataFrame) -> pd.DataFrame:
    """Rows in design order, so a comparison does not depend on NSGA-II row order.

    Rows sort by the design variables first, then by every other column in
    name order, so rows that share a design still have a fixed order. The
    index is reset and the frame's attrs are dropped.
    """
    design = [column for column in DESIGN_COLUMNS if column in pareto.columns]
    rest = sorted(column for column in pareto.columns if column not in design)
    ordered = pareto.sort_values(by=design + rest, kind="mergesort", na_position="last").reset_index(drop=True)
    ordered.attrs = {}
    return ordered


def _arrays_equal(left: Any, right: Any) -> bool:
    try:
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    except AssertionError:
        return False
    return True


def value_differences(left: Any, right: Any, label: str) -> list[str]:
    """Every place where two results differ, exactly; NaN equals NaN.

    Frames must have the same columns, dtypes, index and values; mappings
    the same keys; sequences the same length. An empty list means equal.
    """
    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        if not (isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame)):
            return [f"{label}: one side is not a DataFrame"]
        differences = []
        if list(left.columns) != list(right.columns):
            only_left = sorted(map(str, set(left.columns) - set(right.columns)))
            only_right = sorted(map(str, set(right.columns) - set(left.columns)))
            differences.append(f"{label}: columns differ (only left {only_left}, only right {only_right})")
        if left.shape[0] != right.shape[0]:
            return [*differences, f"{label}: {left.shape[0]} rows against {right.shape[0]}"]
        if not left.index.equals(right.index):
            differences.append(f"{label}: index differs")
        for column in (column for column in left.columns if column in right.columns):
            if left[column].dtype != right[column].dtype:
                differences.append(f"{label}[{column}]: dtype {left[column].dtype} against {right[column].dtype}")
            elif not _arrays_equal(left[column].to_numpy(), right[column].to_numpy()):
                differences.append(f"{label}[{column}]: values differ")
        return differences
    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        return value_differences(pd.DataFrame({"value": left}), pd.DataFrame({"value": right}), label)
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return [f"{label}: one side is not a mapping"]
        differences = []
        if left.keys() != right.keys():
            only_left = sorted(map(str, left.keys() - right.keys()))
            only_right = sorted(map(str, right.keys() - left.keys()))
            differences.append(f"{label}: keys differ (only left {only_left}, only right {only_right})")
        for key in (key for key in left if key in right):
            differences.extend(value_differences(left[key], right[key], f"{label}.{key}"))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{label}: length {len(left)} against {len(right)}"]
        differences = []
        for position, (item_left, item_right) in enumerate(zip(left, right)):
            differences.extend(value_differences(item_left, item_right, f"{label}[{position}]"))
        return differences
    return [] if _arrays_equal(left, right) else [f"{label}: {left!r} against {right!r}"]


def _check(name: str, differences: list[str], **detail: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": not differences,
        "differences": differences[:MAX_LISTED_DIFFERENCES],
        "n_differences": len(differences),
        **detail,
    }


def _requirement(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "differences": [], "n_differences": 0, **detail}


# ---------------------------------------------------------------------------
# Children
# ---------------------------------------------------------------------------


def peak_rss_mib() -> float | None:
    """This process's peak resident set size in MiB, or None where it cannot be read."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in KiB on Linux and the BSDs.
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return round(peak_bytes / 2**20, 1)


class _ConstructionTimer:
    """Time ``SolarDesignProblem.__init__`` inside one child process.

    The class's own ``__init__`` is wrapped, rather than the module name
    replaced, so the problem keeps its class and still pickles for workers.
    """

    def __init__(self) -> None:
        from breos import optimization

        self.calls: list[float] = []
        problem_class = optimization.SolarDesignProblem
        original = problem_class.__init__
        timings = self.calls

        def timed_init(problem: Any, *args: Any, **kwargs: Any) -> None:
            start = time.perf_counter()
            try:
                original(problem, *args, **kwargs)
            finally:
                timings.append(time.perf_counter() - start)

        problem_class.__init__ = timed_init  # type: ignore[method-assign]

    def reset(self) -> None:
        self.calls.clear()


def _numba_cache_files() -> int | None:
    cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if not cache_dir or not Path(cache_dir).is_dir():
        return None
    return sum(1 for path in Path(cache_dir).rglob("*") if path.is_file())


def _search_counts(result: Any) -> dict[str, int]:
    pymoo_result = result.details["pymoo_result"]
    constraint_values = np.atleast_2d(pymoo_result.opt.get("G"))
    return {
        "evaluations": int(pymoo_result.algorithm.evaluator.n_eval),
        "generations": int(result.iterations),
        "pareto_designs": int(len(result.details["pareto"])),
        "feasible_pareto_designs": int((constraint_values <= 0.0).all(axis=1).sum()),
    }


def _optimize(inputs: dict[str, Any], spec: dict[str, Any], backend: str, n_procs: int) -> Any:
    from breos.optimization import optimize_system_multi_objective

    return optimize_system_multi_objective(
        inputs["weather"],
        inputs["load"],
        inputs["optimizer_config"],
        pop_size=spec["pop_size"],
        n_gen=spec["n_gen"],
        seed=spec["seed"],
        n_procs=n_procs,
        execution_backend=backend,
    )


def _child_measure(spec: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    timer = _ConstructionTimer()
    backend = spec["backend"]
    cache_files_before = _numba_cache_files()
    warmup_s = None
    if spec["warmup"]:
        start = time.perf_counter()
        _optimize(inputs, spec, backend, spec["n_procs"])
        warmup_s = time.perf_counter() - start
    timer.reset()
    start = time.perf_counter()
    result = _optimize(inputs, spec, backend, spec["n_procs"])
    total_s = time.perf_counter() - start
    if len(timer.calls) != 1:
        raise RuntimeError(f"expected one SolarDesignProblem construction, timed {len(timer.calls)}")
    construction_s = timer.calls[0]
    return {
        "phase": spec["phase"],
        "backend": backend,
        "repeat": spec["repeat"],
        "workers": spec["n_procs"],
        "total_s": total_s,
        "problem_construction_s": construction_s,
        "residual_search_s": total_s - construction_s,
        "warmup_total_s": warmup_s,
        "peak_rss_mib": peak_rss_mib(),
        "numba_cache_files_before": cache_files_before,
        "numba_cache_files_after": _numba_cache_files(),
        **_search_counts(result),
    }


def _run_app(inputs: dict[str, Any], backend: str) -> dict[str, Any]:
    """Run the witness design through App on ``backend``, keeping every year's ledger."""
    from breos import App, projection
    from breos.app_inputs import AppRuntimeDependencies
    from breos.weather import build_battery_temperature_series

    weather, load = inputs["weather"], inputs["load"]
    dependencies = AppRuntimeDependencies(
        load_profile=lambda **kwargs: load.copy(),
        load_weather=lambda **kwargs: None,
        fetch_tmy_weather_data=lambda **kwargs: (weather.copy(), {}),
        resample_to_15min=lambda frame, **kwargs: frame,
        build_battery_temperature_series=build_battery_temperature_series,
    )
    ledgers: list[dict[str, pd.DataFrame]] = []
    simulate = projection.simulate_energy_balance

    def recording_simulate(**kwargs: Any) -> Any:
        output = simulate(**kwargs)
        ledgers.append({"results": output[0], "degradation": output[4]})
        return output

    projection.simulate_energy_balance = recording_simulate  # type: ignore[assignment]
    try:
        app = App({**inputs["app_config"], "execution_backend": backend})
        app._runtime_dependencies = lambda: dependencies  # type: ignore[method-assign]
        app.simulate()
    finally:
        projection.simulate_energy_balance = simulate  # type: ignore[assignment]
    artifacts = app._artifacts
    assert artifacts is not None
    return {
        "ledgers": ledgers,
        "first_year_results": artifacts.first_year_results_df,
        "yearly": artifacts.yearly_df,
        "cost_projection": artifacts.cost_projection,
        "degradation_summary": artifacts.degradation_summary,
        "current_soh": artifacts.current_soh,
        "total_replacements": artifacts.total_replacements,
        "npv_savings": app.result()["npv_savings"],
    }


def _child_parity(spec: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    from breos.optimization import evaluate_projected_design

    checks: list[dict[str, Any]] = []
    runs = {backend: _optimize(inputs, spec, backend, 1) for backend in BACKENDS}
    counts = {backend: _search_counts(result) for backend, result in runs.items()}
    pareto = {backend: canonical_pareto(result.details["pareto"]) for backend, result in runs.items()}
    design = [column for column in DESIGN_COLUMNS if column in pareto["python"].columns]
    checks.append(
        _check(
            "optimizer_pareto_identical",
            value_differences(pareto["python"], pareto["numba"], "pareto"),
            rows=len(pareto["python"]),
            duplicated_designs=int(pareto["python"].duplicated(subset=design).sum()),
        )
    )
    checks.append(_check("optimizer_counts_identical", value_differences(counts["python"], counts["numba"], "counts")))
    checks.append(_requirement("optimizer_has_feasible_design", counts["python"]["feasible_pareto_designs"] >= 1))

    n_modules, battery_kwh, tilt = WITNESS_DESIGN
    fixed = {
        backend: evaluate_projected_design(
            inputs["weather"],
            inputs["load"],
            inputs["optimizer_config"],
            n_modules=n_modules,
            battery_kwh=battery_kwh,
            tilt=tilt,
            azimuth=AZIMUTH,
            execution_backend=backend,
        )
        for backend in BACKENDS
    }
    for part in ("metrics", "yearly", "financial"):
        checks.append(
            _check(
                f"witness_{part}_identical",
                value_differences(getattr(fixed["python"], part), getattr(fixed["numba"], part), part),
            )
        )
    grid_charge_kwh = float(fixed["python"].yearly["Grid_AC_To_Battery_kWh"].sum())
    replacements = float(fixed["python"].metrics["Projected_Total_Replacements"])
    checks.append(_requirement("witness_grid_charges", grid_charge_kwh > 0.0, grid_ac_to_battery_kwh=grid_charge_kwh))
    checks.append(_requirement("witness_replaces_battery", replacements > 0, replacements=replacements))

    app = {backend: _run_app(inputs, backend) for backend in BACKENDS}
    for part in ("ledgers", "first_year_results", "yearly", "cost_projection", "degradation_summary"):
        checks.append(_check(f"app_{part}_identical", value_differences(app["python"][part], app["numba"][part], part)))
    checks.append(
        _check(
            "app_soh_and_replacements_identical",
            value_differences(
                {key: app["python"][key] for key in ("current_soh", "total_replacements", "npv_savings")},
                {key: app["numba"][key] for key in ("current_soh", "total_replacements", "npv_savings")},
                "app",
            ),
        )
    )
    # The ledger holds mean power per step (W); energy is power times the step.
    hours_per_step = 1.0 if spec["resolution"] == "h" else 0.25
    app_grid_charge_kwh = (
        hours_per_step
        / 1000.0
        * sum(float(ledger["results"]["Grid_AC_To_Battery"].sum()) for ledger in app["python"]["ledgers"])
    )
    checks.append(
        _requirement("app_grid_charges", app_grid_charge_kwh > 0.0, grid_ac_to_battery_kwh=app_grid_charge_kwh)
    )
    checks.append(
        _requirement(
            "app_replaces_battery",
            app["python"]["total_replacements"] > 0,
            replacements=int(app["python"]["total_replacements"]),
        )
    )
    # The same design through App and the fixed-design evaluator, at the
    # tolerance of the three-year tariff App test.
    try:
        pd.testing.assert_frame_equal(
            app["python"]["yearly"][fixed["python"].yearly.columns],
            fixed["python"].yearly,
            check_exact=False,
            rtol=1e-12,
            atol=1e-10,
        )
        app_matches: list[str] = []
    except (AssertionError, KeyError) as exc:
        app_matches = [f"yearly: {str(exc).splitlines()[0]}"]
    npv_gap = abs(float(app["python"]["npv_savings"]) - float(fixed["python"].metrics["Projected_NPV"]))
    if not npv_gap <= 0.0051:
        app_matches.append(
            f"npv: App {app['python']['npv_savings']} against {fixed['python'].metrics['Projected_NPV']}"
        )
    checks.append(_check("app_reproduces_fixed_design", app_matches, npv_gap=npv_gap))

    provenance = runs["python"].details["provenance"]
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "workers": 1,
        "checks": checks,
        "counts": counts,
        "witness": {
            "design": dict(zip(("modules", "battery_kwh", "tilt"), WITNESS_DESIGN), azimuth=AZIMUTH),
            "grid_ac_to_battery_kwh": grid_charge_kwh,
            "projected_total_replacements": replacements,
            "app_total_replacements": int(app["python"]["total_replacements"]),
            "app_ledger_years": len(app["python"]["ledgers"]),
        },
        "tariff": provenance.get("tariff"),
        "smart_charging": provenance.get("smart_charging"),
        "run_settings": provenance.get("run_settings"),
    }


def child_main(argv: list[str]) -> int:
    """Entry point of a child process: ``<kind> <spec.json> <result.json>``."""
    kind, spec_path, result_path = argv
    started = time.perf_counter()
    import breos

    spec = json.loads(Path(spec_path).read_text())
    with open(spec["inputs"], "rb") as handle:
        inputs = pickle.load(handle)
    startup_s = time.perf_counter() - started
    if kind == "measure":
        payload = _child_measure(spec, inputs)
    elif kind == "parity":
        payload = _child_parity(spec, inputs)
    else:
        raise ValueError(f"unknown child kind {kind!r}")
    payload["child"] = {"breos_file": breos.__file__, "startup_s": startup_s, "pid": os.getpid()}
    Path(result_path).write_text(json.dumps(payload, default=_json_default))
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_child(kind: str, spec: dict[str, Any], workdir: Path, numba_cache_dir: Path) -> dict[str, Any]:
    """Run one child in a fresh interpreter and return its JSON payload."""
    label = f"{kind}-{spec.get('resolution')}-{spec.get('phase', 'parity')}-{spec.get('backend', 'both')}"
    label = f"{label}-{spec.get('repeat', 0)}"
    spec_path = workdir / f"{label}.spec.json"
    result_path = workdir / f"{label}.result.json"
    spec_path.write_text(json.dumps(spec))
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **THREAD_ENV, "NUMBA_CACHE_DIR": str(numba_cache_dir), "PYTHONHASHSEED": "0"}
    command = [sys.executable, "-c", _CHILD_BOOTSTRAP.format(root=str(PROJECT_ROOT)), kind, str(spec_path)]
    completed = subprocess.run([*command, str(result_path)], env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-30:])
        raise RuntimeError(f"child {label} failed with exit code {completed.returncode}:\n{tail}")
    return json.loads(result_path.read_text())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "min": None}
    return {"median": statistics.median(values), "min": min(values)}


def summarize_case(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Cold and warm figures per backend, with speedups against the Python warm median.

    A speedup is the Python warm median total (or residual) divided by the
    measurement's; the Python warm median is therefore 1.
    """
    summary: dict[str, Any] = {"backends": {}, "speedup_vs_python_warm_median": {}}
    for backend in BACKENDS:
        cold = [m for m in measurements if m["backend"] == backend and m["phase"] == "cold"]
        warm = [m for m in measurements if m["backend"] == backend and m["phase"] == "warm"]
        if not cold and not warm:
            continue
        rss = [m["peak_rss_mib"] for m in cold + warm if m["peak_rss_mib"] is not None]
        summary["backends"][backend] = {
            "cold": {
                key: (cold[0][key] if cold else None)
                for key in ("total_s", "problem_construction_s", "residual_search_s", "peak_rss_mib")
            },
            "warm": {
                "repeats": len(warm),
                **{
                    key: _stats([m[key] for m in warm])
                    for key in ("total_s", "problem_construction_s", "residual_search_s")
                },
                "peak_rss_mib_max": max(
                    (m["peak_rss_mib"] for m in warm if m["peak_rss_mib"] is not None), default=None
                ),
            },
            "peak_rss_mib_max": max(rss, default=None),
        }
    python = summary["backends"].get("python")
    if python is None:
        return summary
    baseline_total = python["warm"]["total_s"]["median"]
    baseline_residual = python["warm"]["residual_search_s"]["median"]
    for backend, figures in summary["backends"].items():
        entry: dict[str, float | None] = {}
        for phase, total, residual in (
            ("cold", figures["cold"]["total_s"], figures["cold"]["residual_search_s"]),
            ("warm", figures["warm"]["total_s"]["median"], figures["warm"]["residual_search_s"]["median"]),
        ):
            entry[f"{phase}_total"] = baseline_total / total if baseline_total and total else None
            entry[f"{phase}_residual_search"] = baseline_residual / residual if baseline_residual and residual else None
        summary["speedup_vs_python_warm_median"][backend] = entry
    counts = {(m["evaluations"], m["generations"]) for m in measurements}
    summary["evaluation_counts_consistent"] = len(counts) <= 1
    return summary


def build_report(
    args: argparse.Namespace, machine: dict[str, Any], weather: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    parity_passed = all(case["parity"]["status"] == "passed" for case in cases)
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed" if parity_passed and len(cases) == len(args.resolution) else "failed",
        "entrypoint": "breos.optimization.optimize_system_multi_objective",
        "machine": machine,
        "settings": {
            "resolutions": list(args.resolution),
            "projection_years": args.projection_years,
            "pop_size": args.pop_size,
            "n_gen": args.n_gen,
            "seed": args.seed,
            "n_procs": args.n_procs,
            "parity_n_procs": 1,
            "warm_repeats": args.warm_repeats,
            "latitude": args.latitude,
            "longitude": args.longitude,
            "timezone": TIMEZONE,
            "study_year": STUDY_YEAR,
            "annual_consumption_kwh": args.annual_consumption_kwh,
            "child_environment": dict(THREAD_ENV, PYTHONHASHSEED="0"),
        },
        "timing_boundaries": {
            "total_s": "wall time of one optimize_system_multi_objective call",
            "problem_construction_s": "SolarDesignProblem.__init__: config, tariff, smart-charging, site and model "
            "preparation",
            "residual_search_s": "total_s - problem_construction_s: residual search and wrapper time, including "
            "candidate simulation, NSGA-II and result assembly; not a pure dispatch time",
            "cold": "first call in a fresh child; the Numba child starts with an empty NUMBA_CACHE_DIR",
            "warm": "a fresh child runs the optimizer once untimed at the study size, then times one call",
            "peak_rss_mib": "ru_maxrss of the child, warm-up included",
        },
        "study": {
            "tariff": TARIFF,
            "smart_charging": FIXED_TARGET,
            "battery_eol_percentage": BATTERY_EOL_PERCENTAGE,
            "constraints": CONSTRAINTS,
            "fixed_azimuth": AZIMUTH,
            "witness_design": dict(zip(("modules", "battery_kwh", "tilt"), WITNESS_DESIGN)),
        },
        "weather_file": weather,
        "cases": cases,
    }


def _format_seconds(value: float | None) -> str:
    return "     n/a" if value is None else f"{value:8.3f}"


def format_case(case: dict[str, Any]) -> list[str]:
    """Console lines for one case."""
    parity = case["parity"]
    counts = parity["counts"]["python"]
    witness = parity["witness"]
    lines = [
        f"[{case['resolution']}] parity {parity['status']}: {counts['evaluations']} evaluations, "
        f"{counts['generations']} generation(s), {counts['feasible_pareto_designs']}/{counts['pareto_designs']} "
        f"feasible Pareto designs; witness grid charge {witness['grid_ac_to_battery_kwh']:.1f} kWh, "
        f"{witness['projected_total_replacements']:g} replacement(s)",
    ]
    for check in parity["checks"]:
        if not check["passed"]:
            lines.append(f"  FAILED {check['name']}: {'; '.join(check['differences']) or 'requirement not met'}")
    summary = case.get("summary")
    if not summary:
        return lines
    for backend, figures in summary["backends"].items():
        cold, warm = figures["cold"], figures["warm"]
        speed = summary["speedup_vs_python_warm_median"].get(backend, {})
        warm_speed = speed.get("warm_total")
        lines.append(
            f"  {backend:>6} cold  total {_format_seconds(cold['total_s'])} s "
            f"(construction {_format_seconds(cold['problem_construction_s'])} s, "
            f"residual {_format_seconds(cold['residual_search_s'])} s) | peak RSS {cold['peak_rss_mib']} MiB"
        )
        lines.append(
            f"  {backend:>6} warm  total {_format_seconds(warm['total_s']['median'])} s median, "
            f"{_format_seconds(warm['total_s']['min'])} s min over {warm['repeats']} "
            f"(construction {_format_seconds(warm['problem_construction_s']['median'])} s, "
            f"residual {_format_seconds(warm['residual_search_s']['median'])} s) | "
            f"peak RSS {warm['peak_rss_mib_max']} MiB" + (f" | {warm_speed:.2f}x vs Python warm" if warm_speed else "")
        )
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def machine_info() -> dict[str, Any]:
    import breos

    info: dict[str, Any] = dict(_montecarlo_machine_info())
    info["breos"] = breos.__version__
    info["breos_file"] = breos.__file__
    try:
        import pymoo

        info["pymoo"] = pymoo.__version__
    except ImportError:
        info["pymoo"] = "not installed"
    return info


def run_case(args: argparse.Namespace, resolution: str, staged: Path, location: str, session: Path) -> dict[str, Any]:
    weather, load, input_record = prepare_case_inputs(
        staged,
        location,
        resolution,
        latitude=args.latitude,
        longitude=args.longitude,
        annual_consumption_kwh=args.annual_consumption_kwh,
    )
    app_config, optimizer_config = study_configs(
        resolution,
        args.projection_years,
        latitude=args.latitude,
        longitude=args.longitude,
        annual_consumption_kwh=args.annual_consumption_kwh,
    )
    inputs_path = session / f"inputs-{resolution}.pkl"
    with open(inputs_path, "wb") as handle:
        pickle.dump(
            {"weather": weather, "load": load, "optimizer_config": optimizer_config, "app_config": app_config},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    base = {
        "inputs": str(inputs_path),
        "resolution": resolution,
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "seed": args.seed,
        "n_procs": args.n_procs,
    }
    # The parity child fills the warm cache, which the warm children reuse.
    warm_cache = session / f"numba-cache-warm-{resolution}"
    case: dict[str, Any] = {
        "resolution": resolution,
        "inputs": input_record,
        "optimizer_config_sha256": config_sha256(optimizer_config),
        "optimizer_config": optimizer_config,
    }
    print(f"[{resolution}] parity...", flush=True)
    case["parity"] = run_child("parity", base, session, warm_cache)
    if case["parity"]["status"] != "passed":
        print("\n".join(format_case(case)), flush=True)
        print(f"[{resolution}] parity failed; the case is not timed", flush=True)
        return case

    measurements = []
    for backend in BACKENDS:
        print(f"[{resolution}] cold {backend}...", flush=True)
        cold_cache = session / f"numba-cache-cold-{resolution}-{backend}"
        spec = {**base, "backend": backend, "phase": "cold", "repeat": 0, "warmup": False}
        measurements.append(run_child("measure", spec, session, cold_cache))
    for repeat in range(args.warm_repeats):
        for backend in BACKENDS:
            print(f"[{resolution}] warm {backend} {repeat + 1}/{args.warm_repeats}...", flush=True)
            spec = {**base, "backend": backend, "phase": "warm", "repeat": repeat, "warmup": True}
            measurements.append(run_child("measure", spec, session, warm_cache))
    case["measurements"] = measurements
    case["summary"] = summarize_case(measurements)
    print("\n".join(format_case(case)), flush=True)
    return case


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    missing = missing_dependencies()
    if missing:
        print(
            f"benchmark_optimization needs {' and '.join(missing)}: install breos[optimization] for pymoo "
            "and breos[fast] for numba.",
            file=sys.stderr,
        )
        return 2

    machine = machine_info()
    cases = []
    with tempfile.TemporaryDirectory(prefix="breos-optimization-benchmark-") as tmp:
        session = Path(tmp)
        staged, location, weather_record = stage_weather(args.weather_file, session / "weather")
        for resolution in args.resolution:
            cases.append(run_case(args, resolution, staged, location, session))

    report = build_report(args, machine, weather_record, cases)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"\nwrote {args.output}")
    print(f"status: {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
