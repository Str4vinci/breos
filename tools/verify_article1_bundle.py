#!/usr/bin/env python3
"""Verify a completed result bundle for the forthcoming publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_ARTIFACT_HASH_KEYS = {
    "cost_projection_sha256",
    "metrics_sha256",
    "orientation_grid_sha256",
    "orientation_optimum_sha256",
    "pareto_sha256",
    "representatives_sha256",
    "runs_csv_sha256",
    "summary_json_sha256",
    "weather_monthly_by_year_sha256",
    "weather_monthly_comparison_sha256",
    "yearly_csv_sha256",
    "yearly_summary_sha256",
}
EXPECTED_MONTE_CARLO_YEARLY_COLUMNS = (
    "run",
    "Year",
    "PV_Production_kWh",
    "Legacy_PV_Production_kWh",
    "PV_DC_Generation_kWh",
    "Direct_PV_AC_Load_kWh",
    "PV_Origin_Battery_AC_Load_kWh",
    "Self_Consumption_kWh",
    "Curtailment_DC_kWh",
    "Load_kWh",
    "Import_kWh",
    "Export_kWh",
    "Grid_Independence_%",
    "Battery_SOH_%",
    "Battery_Cumulative_FEC",
    "Battery_Cumulative_Calendar_Seconds",
    "Battery_Cumulative_Cycle_Degradation",
    "Battery_Cumulative_Calendar_Degradation",
    "Battery_Resistance_Growth",
    "Replacements",
    "Replacement_Cost",
    "PV_Degradation_Factor",
    "Weather_Year",
    "Load_Scale",
    "PV_Direct_Inverter_Loss_kWh",
    "Battery_Inverter_Loss_kWh",
    "Battery_Charge_Input_kWh",
    "Battery_Discharge_DC_kWh",
    "Battery_AC_To_Load_kWh",
    "Battery_Charge_Loss_kWh",
    "Battery_Discharge_Loss_kWh",
    "Battery_Standby_Loss_kWh",
    "Capacity_Window_Loss_kWh",
    "Replacement_Energy_Removed_kWh",
    "Replacement_Energy_Added_kWh",
    "Battery_Carried_Energy_Wh",
    "Battery_Carried_PV_Origin_Energy_Wh",
    "Replacement_Steps",
    "Load_kWh_Financial",
    "Cost_No_Sys_Annual",
    "Cost_No_Sys_Cumulative",
    "PV_Production_kWh_Financial",
    "Export_kWh_Financial",
    "Degradation_Factor",
    "Cost_Import",
    "Revenue_Export",
    "Cost_Operation",
    "Cost_Daily",
    "Cost_Replacement",
    "Cost_System_Annual",
    "Cost_System_Cumulative",
    "Cost_No_Sys_Annual_NPV",
    "Cost_System_Annual_NPV",
    "Cost_No_Sys_Cumulative_NPV",
    "Cost_System_Cumulative_NPV",
    "Savings_Cumulative",
    "Savings_Cumulative_NPV",
    "CO2_Avoided_Total_kg",
    "CO2_Avoided_SelfConsumed_kg",
    "CO2_Avoided_Total_Cumulative_kg",
    "CO2_Avoided_SelfConsumed_Cumulative_kg",
    "CO2_Avoided_CI_gCO2_kWh",
    "CO2_Avoided_CI_Type",
    "Average_Grid_CI_gCO2_kWh",
    "Marginal_Grid_CI_gCO2_kWh",
)

# Publication-study generators export plot-independent source tables.
# Presentation-only changes therefore do not invalidate previously generated
# numerical outputs.
NON_NUMERICAL_SOURCE_PATHS = ("breos/plotting.py",)


def _report_source_paths(relative: str) -> tuple[str, ...]:
    """Return source paths that can affect one generated report."""
    if relative.startswith("monte-carlo-v1/"):
        return (
            "tools/reproduce_article1_montecarlo.py",
            "validation/article1/article1-montecarlo.toml",
        )
    if relative.startswith(("orientation/", "weather-comparison/")):
        return (
            "tools/reproduce_article1_context.py",
            "validation/article1/article1-projected-optimization.toml",
        )
    return (
        "tools/reproduce_article1.py",
        "validation/article1/article1-projected-optimization.toml",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BundleAudit:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.errors: list[str] = []
        self.reports: list[tuple[Path, dict[str, Any]]] = []
        self.source_commits: set[str] = set()

    def require_file(self, relative: str) -> Path:
        path = self.root / relative
        if not path.is_file():
            self.errors.append(f"missing file: {relative}")
        return path

    def load_report(self, relative: str) -> dict[str, Any]:
        path = self.require_file(relative)
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            self.errors.append(f"invalid JSON: {relative}: {error}")
            return {}
        self.reports.append((path, payload))
        self._verify_artifact_hashes(path.parent, payload, relative)
        self._verify_common_provenance(payload, relative)
        return payload

    def expect(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def verify_monte_carlo_yearly_schema(self, case: str) -> None:
        relative = f"monte-carlo-v1/{case}/yearly.csv"
        path = self.require_file(relative)
        if not path.is_file():
            return
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                columns = tuple(next(csv.reader(handle), ()))
        except OSError as error:
            self.errors.append(f"cannot read Monte Carlo yearly output: {relative}: {error}")
            return

        if columns == EXPECTED_MONTE_CARLO_YEARLY_COLUMNS:
            return

        missing = [column for column in EXPECTED_MONTE_CARLO_YEARLY_COLUMNS if column not in columns]
        unexpected = [column for column in columns if column not in EXPECTED_MONTE_CARLO_YEARLY_COLUMNS]
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        if not details:
            details.append("column order differs")
        self.errors.append(f"wrong Monte Carlo yearly schema: {case}: {'; '.join(details)}")

    def _verify_common_provenance(self, payload: dict[str, Any], relative: str) -> None:
        source = payload.get("breos_source", {})
        self.expect(bool(source.get("commit")), f"missing BREOS commit: {relative}")
        self.expect(source.get("tracked_worktree_dirty") is False, f"dirty source recorded: {relative}")
        module = payload.get("resolved_pv_module", {})
        parameters = module.get("parameters", {})
        pmax_coefficient = parameters.get("T_Pmax_pct")
        voc_coefficient = parameters.get("T_Voc_pct")
        self.expect(
            pmax_coefficient == -0.34,
            f"unexpected PV T_Pmax_pct in {relative}: {pmax_coefficient!r}",
        )
        self.expect(
            voc_coefficient == -0.26,
            f"unexpected PV T_Voc_pct in {relative}: {voc_coefficient!r}",
        )
        self.expect(module.get("width_m") == 1.134, f"unexpected PV width in {relative}")
        self.expect(module.get("length_m") == 2.278, f"unexpected PV length in {relative}")

    def _verify_artifact_hashes(self, directory: Path, value: Any, report_name: str) -> None:
        if isinstance(value, dict):
            for key, expected in value.items():
                if key in EXPECTED_ARTIFACT_HASH_KEYS and expected:
                    stem = key.removesuffix("_sha256")
                    path_value = value.get(stem) or value.get(f"{stem}_csv") or value.get(f"{stem}_json")
                    if not path_value:
                        self.errors.append(f"no path paired with {key} in {report_name}")
                    else:
                        path = directory / str(path_value)
                        if not path.is_file():
                            self.errors.append(f"missing hashed artifact: {path}")
                        elif _sha256(path) != expected:
                            self.errors.append(f"artifact hash mismatch: {path}")
                self._verify_artifact_hashes(directory, expected, report_name)
        elif isinstance(value, list):
            for item in value:
                self._verify_artifact_hashes(directory, item, report_name)

    @staticmethod
    def _source_object(commit: str, path: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", f"{commit}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    @staticmethod
    def _is_ancestor(commit: str, target: str) -> bool:
        return (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, target],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )

    @staticmethod
    def _numerical_source_changed(commit: str, target: str) -> bool:
        command = ["git", "diff", "--quiet", commit, target, "--", "breos"]
        command.extend(f":(exclude){path}" for path in NON_NUMERICAL_SOURCE_PATHS)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        return result.returncode == 1

    def _verify_source_compatibility(self, manifest: dict[str, Any]) -> None:
        commits = {
            str(payload.get("breos_source", {}).get("commit"))
            for _path, payload in self.reports
            if payload.get("breos_source", {}).get("commit")
        }
        manifest_commit = manifest.get("breos_source", {}).get("commit")
        if manifest_commit:
            commits.add(str(manifest_commit))
        self.source_commits = commits
        if len(commits) <= 1:
            return
        if not manifest_commit:
            self.errors.append(
                f"result bundle contains multiple BREOS commits without a manifest commit: {sorted(commits)}"
            )
            return

        target = str(manifest_commit)
        for commit in commits:
            if not self._is_ancestor(commit, target):
                self.errors.append(f"BREOS commit {commit} is not an ancestor of manifest commit {target}")

        for commit in commits - {target}:
            try:
                numerical_source_changed = self._numerical_source_changed(commit, target)
            except subprocess.CalledProcessError:
                self.errors.append(f"cannot compare BREOS numerical source between commits {commit} and {target}")
                continue
            if numerical_source_changed:
                self.errors.append(f"BREOS numerical source differs between commits {commit} and {target}")

        for report_path, payload in self.reports:
            commit = payload.get("breos_source", {}).get("commit")
            if not commit or commit == target:
                continue
            relative = str(report_path.relative_to(self.root))
            for source_path in _report_source_paths(relative):
                try:
                    recorded_object = self._source_object(str(commit), source_path)
                    target_object = self._source_object(target, source_path)
                except subprocess.CalledProcessError:
                    self.errors.append(
                        f"cannot compare {source_path} between BREOS commits {commit} and {target} for {relative}"
                    )
                    continue
                if recorded_object != target_object:
                    self.errors.append(
                        f"source changed for {relative}: {source_path} differs between BREOS commits {commit} and {target}"
                    )

        dependency_versions: dict[str, set[str]] = {}
        for _path, payload in self.reports:
            for package, version in payload.get("dependency_versions", {}).items():
                dependency_versions.setdefault(str(package), set()).add(str(version))
        conflicts = {
            package: sorted(versions) for package, versions in dependency_versions.items() if len(versions) > 1
        }
        self.expect(not conflicts, f"result bundle contains conflicting dependency versions: {conflicts}")

    def deterministic_report(
        self,
        relative: str,
        *,
        battery_cost: float = 500.0,
        resolution: str = "15min",
        profile: str = "6",
        calendar_model: str = "naumann_lam_field_calibrated",
        optimization: bool,
        fixed_labels: set[str] | None = None,
    ) -> None:
        payload = self.load_report(relative)
        if not payload:
            return
        config = payload.get("resolved_config", {})
        self.expect(payload.get("battery_cost_scenario_eur_per_kwh") == battery_cost, f"wrong battery cost: {relative}")
        self.expect(config.get("simulation", {}).get("resolution") == resolution, f"wrong resolution: {relative}")
        self.expect(config.get("load", {}).get("profile_type") == profile, f"wrong load profile: {relative}")
        self.expect(
            config.get("battery", {}).get("calendar_model") == calendar_model, f"wrong calendar model: {relative}"
        )
        self.expect(
            config.get("battery", {}).get("temperature") == "weather",
            f"battery temperature is not weather-driven: {relative}",
        )
        self.expect(
            config.get("battery", {}).get("indoor_model") == {"enabled": True},
            f"battery indoor buffering is not enabled: {relative}",
        )
        self.expect(
            config.get("optimization", {}).get("objective_basis") == "projected", f"wrong objectives: {relative}"
        )
        expected_rlp = None
        if profile != "1":
            expected_rlp = (
                "EREDES_2025_BTN_1000kwh_hourly.csv" if resolution == "h" else "EREDES_2025_BTN_1000kwh_15min.csv"
            )
        self.expect(payload.get("external_rlp_filename") == expected_rlp, f"wrong RLP provenance: {relative}")
        if fixed_labels is not None:
            actual = {str(row.get("Label")) for row in payload.get("fixed_candidates", [])}
            self.expect(actual == fixed_labels, f"wrong fixed candidates in {relative}: {sorted(actual)}")
        if optimization:
            run = payload.get("optimization", {})
            self.expect(run.get("run_type") == "full", f"full optimization missing: {relative}")
            self.expect(run.get("objective_basis") == "projected", f"projected optimization missing: {relative}")
            self.expect(len(run.get("objective_names", [])) == 2, f"optimization is not bi-objective: {relative}")

    def verify(self) -> None:
        manifest_path = self.require_file("input-manifest.json")
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                self.errors.append(f"invalid JSON: input-manifest.json: {error}")
            self.expect(
                manifest.get("schema") == "breos-article1-input-manifest-v1",
                "unexpected input manifest schema",
            )
            self.expect(
                manifest.get("breos_source", {}).get("tracked_worktree_dirty") is False,
                "input manifest records a dirty source tree",
            )
            manifest_inputs = manifest.get("inputs", {})
            self.expect(
                manifest_inputs.get("external_rlp_15min", {}).get("sha256")
                == "23becc5a7bfc927b1f7604156e0e4953dcc6bb65268ca947b38db3dc4f2b28bc",
                "unexpected 15-minute E-REDES input hash",
            )
            self.expect(
                manifest_inputs.get("external_rlp_hourly", {}).get("sha256")
                == "6ae15efed9b179537349ee1c1c5747065f18db876920e073e8670bf412d20d6b",
                "unexpected hourly E-REDES input hash",
            )
            self.expect(
                manifest_inputs.get("historical_weather", {}).get("sha256")
                == "71c26d072c09faf16dab37230cfe8b2d430bd39344333227d00c7be4e76a188a",
                "unexpected historical-weather input hash",
            )
        self.deterministic_report(
            "base-v1/reproduction.json", optimization=False, fixed_labels={"C1", "C2", "C3", "C4", "C5"}
        )
        for cost in (350, 500, 711):
            self.deterministic_report(
                f"battery-cost-sensitivity/battery-cost-{cost}/reproduction.json",
                battery_cost=float(cost),
                optimization=True,
                fixed_labels=set(),
            )
        self.deterministic_report("hourly-v1/reproduction.json", resolution="h", optimization=True, fixed_labels=set())
        self.deterministic_report(
            "load-profile-h0/reproduction.json", profile="1", optimization=False, fixed_labels={"C2"}
        )
        self.deterministic_report(
            "field-v2/reproduction.json",
            calendar_model="naumann_lam_field_calibrated_v2",
            optimization=True,
            fixed_labels={"C2"},
        )
        self.deterministic_report(
            "laboratory/reproduction.json",
            calendar_model="naumann_lam",
            optimization=True,
            fixed_labels={"C2"},
        )
        self.load_report("orientation/provenance.json")
        self.load_report("weather-comparison/provenance.json")
        for case in ("c1", "c2", "c3", "c4", "c5"):
            self.verify_monte_carlo_yearly_schema(case)
            payload = self.load_report(f"monte-carlo-v1/{case}/provenance.json")
            if payload:
                settings = payload.get("settings", {})
                self.expect(settings.get("n_runs") == 10000, f"wrong Monte Carlo run count: {case}")
                self.expect(settings.get("years_per_run") == 20, f"wrong Monte Carlo horizon: {case}")
                self.expect(settings.get("seed") == 42, f"wrong Monte Carlo seed: {case}")
                self.expect(settings.get("load_distribution") == "uniform", f"wrong load distribution: {case}")
                config = payload.get("resolved_config", {})
                self.expect(
                    config.get("battery_temperature") == "weather",
                    f"battery temperature is not weather-driven: {case}",
                )
                self.expect(
                    config.get("battery_indoor_model") == {"enabled": True},
                    f"battery indoor buffering is not enabled: {case}",
                )

        self._verify_source_compatibility(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("results/article1"))
    args = parser.parse_args()

    audit = BundleAudit(args.root)
    audit.verify()
    if audit.errors:
        print("Forthcoming publication bundle verification failed:")
        for error in audit.errors:
            print(f"- {error}")
        return 1
    print(f"Forthcoming publication bundle verification passed: {audit.root}")
    if len(audit.source_commits) == 1:
        print(f"Verified {len(audit.reports)} provenance reports from one BREOS commit.")
    else:
        print(
            f"Verified {len(audit.reports)} provenance reports across {len(audit.source_commits)} "
            "source-compatible BREOS commits."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
