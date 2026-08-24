#!/usr/bin/env python3
"""Verify a completed Article 1 result bundle without recalculating results."""

from __future__ import annotations

import argparse
import hashlib
import json
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
EXPECTED_EXTERNAL_VALIDATION_HASHES = {
    "monthly_results.csv": "d2b777e2b58abdad055abe25ec45c7fd879947f498622b6f17966b8d4803d1cb",
    "weekly_results.csv": "b5ff0311df777b62f22de1111ba277321203e57e2ab5c6eac67e6673880d397b",
    "daily_results.csv": "e376382026bc266b5895ce9ba2cf3504c632351a4fe34ab0ac8a6fe86ca857b8",
}


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

    def _verify_common_provenance(self, payload: dict[str, Any], relative: str) -> None:
        source = payload.get("breos_source", {})
        self.expect(bool(source.get("commit")), f"missing BREOS commit: {relative}")
        self.expect(source.get("tracked_worktree_dirty") is False, f"dirty source recorded: {relative}")
        module = payload.get("resolved_pv_module", {})
        coefficient = module.get("parameters", {}).get("T_Pmax_pct")
        self.expect(coefficient == -0.34, f"unexpected PV T_Pmax_pct in {relative}: {coefficient!r}")
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
        for filename, expected in EXPECTED_EXTERNAL_VALIDATION_HASHES.items():
            path = self.require_file(f"external-validation/{filename}")
            if path.is_file():
                self.expect(_sha256(path) == expected, f"external validation hash mismatch: {filename}")
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
            payload = self.load_report(f"monte-carlo-v1/{case}/provenance.json")
            if payload:
                settings = payload.get("settings", {})
                self.expect(settings.get("n_runs") == 10000, f"wrong Monte Carlo run count: {case}")
                self.expect(settings.get("years_per_run") == 20, f"wrong Monte Carlo horizon: {case}")
                self.expect(settings.get("seed") == 1, f"wrong Monte Carlo seed: {case}")
                self.expect(settings.get("load_distribution") == "uniform", f"wrong load distribution: {case}")

        commits = {
            payload.get("breos_source", {}).get("commit")
            for _path, payload in self.reports
            if payload.get("breos_source", {}).get("commit")
        }
        manifest_commit = manifest.get("breos_source", {}).get("commit")
        if manifest_commit:
            commits.add(manifest_commit)
        self.expect(len(commits) == 1, f"result bundle contains multiple BREOS commits: {sorted(commits)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("results/article1"))
    args = parser.parse_args()

    audit = BundleAudit(args.root)
    audit.verify()
    if audit.errors:
        print("Article 1 bundle verification failed:")
        for error in audit.errors:
            print(f"- {error}")
        return 1
    print(f"Article 1 bundle verification passed: {audit.root}")
    print(f"Verified {len(audit.reports)} provenance reports from one BREOS commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
