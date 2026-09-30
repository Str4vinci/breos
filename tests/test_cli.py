"""Tests for the BREOS command line interface."""

import csv
import json
import re
from pathlib import Path

import pytest

from breos import cli

EXAMPLE_CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs" / "examples").glob("*.toml"))


class FakeApp:
    seen_config = None
    simulated = False

    def __init__(self, config):
        self.config = config
        FakeApp.seen_config = config

    def simulate(self):
        FakeApp.simulated = True

    def result(self):
        return {
            "grid_independence_pct": 42.0,
            "config": self.config,
        }


def test_run_from_flags_outputs_json(monkeypatch, capsys):
    monkeypatch.setattr(cli, "App", FakeApp)

    exit_code = cli.main(
        [
            "run",
            "--location",
            "porto",
            "--n-modules",
            "10",
            "--annual-consumption-kwh",
            "4000",
            "--battery-kwh",
            "5.0",
            "--cost-preset",
            "residential-pt",
            "--emissions-country",
            "pt",
            "--load-profile",
            "eredes_btn_c",
            "--rlp-directory",
            "/tmp/external-rlp",
        ]
    )

    assert exit_code == 0
    assert FakeApp.simulated is True
    assert FakeApp.seen_config["location"] == "porto"
    assert FakeApp.seen_config["n_modules"] == 10
    assert FakeApp.seen_config["annual_consumption_kwh"] == 4000
    assert FakeApp.seen_config["battery_kwh"] == 5.0
    assert FakeApp.seen_config["cost_preset"] == "residential_pt"
    assert FakeApp.seen_config["emissions_country"] == "PT"
    assert FakeApp.seen_config["load_profile"] == "eredes_btn_c"
    assert FakeApp.seen_config["rlp_directory"] == "/tmp/external-rlp"

    output = json.loads(capsys.readouterr().out)
    assert output["grid_independence_pct"] == 42.0


def test_run_warns_and_ignores_unused_runner_sections(tmp_path, capsys):
    config_path = tmp_path / "runner-sections.toml"
    config_path.write_text(
        'location = "porto"\n'
        "n_modules = 10\n"
        "annual_consumption_kwh = 4000\n"
        "battery_kwh = 5.0\n"
        'degradation_engine = "blast"\n'
        'blast_model = "lfp_gr_250ah_prismatic"\n'
        "\n[montecarlo]\n"
        'weather_file = "unused.csv"\n'
        "n_runs = 1\n"
        "\n[sweep]\n"
        "battery_kwh = [5.0]\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match=r"breos run does not use \[montecarlo\], \[sweep\]"):
        assert cli.main(["run", "--config", str(config_path), "--dry-run"]) == 0
    assert '"degradation_engine": "blast"' in capsys.readouterr().out


class _BackendChosen(Exception):
    pass


def _record_montecarlo_backend(monkeypatch):
    """Stop run_montecarlo where it resolves the backend, and record it."""
    import breos.montecarlo as montecarlo

    observed = {}

    def fake_backend_provenance(execution_backend, *, pv_only=False):
        observed["execution_backend"] = execution_backend
        raise _BackendChosen

    monkeypatch.setattr(montecarlo, "backend_provenance", fake_backend_provenance)
    return observed


def _montecarlo_config(tmp_path, *, top_level=None, section=None):
    weather_file = tmp_path / "weather.csv"
    weather_file.write_text("weather", encoding="utf-8")
    lines = ['location = "porto"', "n_modules = 10", "annual_consumption_kwh = 4000"]
    if top_level is not None:
        lines.append(f'execution_backend = "{top_level}"')
    lines += ["", "[montecarlo]", f'weather_file = "{weather_file}"']
    if section is not None:
        lines.append(f'execution_backend = "{section}"')
    config_path = tmp_path / "montecarlo.toml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    ("top_level", "section", "flag", "expected"),
    [
        (None, None, None, "python"),
        ("numba", None, None, "numba"),
        ("numba", "python", None, "python"),
        ("python", None, "numba", "numba"),
        ("numba", "numba", "python", "python"),
    ],
)
def test_montecarlo_backend_precedence_matches_python_api(tmp_path, monkeypatch, top_level, section, flag, expected):
    # Flag, then [montecarlo], then the top-level key, then "python"; the
    # Python API must pick the same backend for the same config.
    import tomllib

    from breos.montecarlo import MonteCarloSettings, run_montecarlo

    config_path = _montecarlo_config(tmp_path, top_level=top_level, section=section)
    observed = _record_montecarlo_backend(monkeypatch)
    argv = ["montecarlo", "--config", str(config_path), "--output", str(tmp_path / "runs.csv")]
    if flag is not None:
        argv += ["--execution-backend", flag]
    with pytest.raises(_BackendChosen):
        cli._montecarlo(cli.build_parser().parse_args(argv))
    assert observed.pop("execution_backend") == expected

    if flag is None:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        settings = MonteCarloSettings(**config.pop("montecarlo"))
        with pytest.raises(_BackendChosen):
            run_montecarlo(config, settings)
        assert observed["execution_backend"] == expected


def test_montecarlo_reports_unknown_setting_before_weather_lookup(tmp_path, capsys):
    config_path = tmp_path / "montecarlo.toml"
    config_path.write_text('location = "porto"\n\n[montecarlo]\nweather_fille = "missing.csv"\n', encoding="utf-8")

    assert cli.main(["montecarlo", "--config", str(config_path)]) == 1
    assert "Unknown Monte Carlo config key(s): montecarlo.weather_fille" in capsys.readouterr().err


def test_run_flag_sell_price_inflation_reaches_config(monkeypatch, capsys):
    monkeypatch.setattr(cli, "App", FakeApp)

    exit_code = cli.main(
        [
            "run",
            "--location",
            "porto",
            "--n-modules",
            "10",
            "--annual-consumption-kwh",
            "4000",
            "--sell-price-inflation",
            "0.03",
        ]
    )

    assert exit_code == 0
    assert FakeApp.seen_config["sell_price_inflation"] == 0.03


def test_run_bifacial_flags_reach_config(monkeypatch, capsys):
    monkeypatch.setattr(cli, "App", FakeApp)

    exit_code = cli.main(
        [
            "run",
            "--location",
            "porto",
            "--n-modules",
            "10",
            "--annual-consumption-kwh",
            "4000",
            "--pv-module",
            "Generic_600W_Bifacial",
            "--bifacial-model",
            "infinite_sheds",
            "--gcr",
            "0.35",
            "--pvrow-height",
            "1.5",
            "--pvrow-pitch",
            "6.0",
        ]
    )

    assert exit_code == 0
    assert FakeApp.seen_config["bifacial_model"] == "infinite_sheds"
    assert FakeApp.seen_config["gcr"] == 0.35
    assert FakeApp.seen_config["pvrow_height"] == 1.5
    assert FakeApp.seen_config["pvrow_pitch"] == 6.0


def test_cli_threads_battery_power_limits(monkeypatch, capsys):
    monkeypatch.setattr(cli, "App", FakeApp)

    assert (
        cli.main(
            [
                "run",
                "--location",
                "porto",
                "--n-modules",
                "10",
                "--annual-consumption-kwh",
                "4000",
                "--battery-max-charge-power-w",
                "2500",
                "--battery-max-discharge-power-w",
                "1800",
                "--export-emissions-factor-gco2-kwh",
                "120",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert FakeApp.seen_config["battery_max_charge_power_w"] == 2500
    assert FakeApp.seen_config["battery_max_discharge_power_w"] == 1800
    assert FakeApp.seen_config["export_emissions_factor_gco2_kwh"] == 120


def test_cli_does_not_advertise_unsupported_ac_coupling(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--ac-coupled"])
    assert "unrecognized arguments: --ac-coupled" in capsys.readouterr().err


def test_run_from_toml_config_with_cli_override(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "App", FakeApp)
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500
battery_kwh = 0
cost_preset = "residential_pt"
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--battery-kwh",
            "5.0",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert FakeApp.seen_config["n_modules"] == 8
    assert FakeApp.seen_config["battery_kwh"] == 5.0
    assert json.loads(output_path.read_text(encoding="utf-8"))["grid_independence_pct"] == 42.0


def test_invalid_json_config_reports_filename(tmp_path, capsys):
    config_path = tmp_path / "broken_config.json"
    config_path.write_text('{"location": "porto",', encoding="utf-8")

    exit_code = cli.main(["run", "--config", str(config_path)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Invalid JSON in" in stderr
    assert "broken_config.json" in stderr


def test_list_locations_outputs_packaged_keys(capsys):
    exit_code = cli.main(["list", "locations"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "porto: Porto, Portugal" in output
    assert "Europe/Lisbon" in output


def test_list_modules_json_outputs_catalog(capsys):
    exit_code = cli.main(["list", "modules", "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert any(row["key"] == "Generic_400W" and row["power_w"] == 400 for row in output)
    bifacial = next(row for row in output if row["key"] == "Generic_600W_Bifacial")
    assert bifacial["bifaciality"] == 0.7


def test_list_modules_text_identifies_bifaciality(capsys):
    exit_code = cli.main(["list", "modules"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Generic_600W_Bifacial: 600 W" in output
    assert "bifaciality 70.0%" in output


def test_temperature_model_flags_expose_sapm_and_reject_unsourced_noct_metadata(capsys):
    sapm_exit = cli.main(
        [
            "run",
            "--location",
            "porto",
            "--n-modules",
            "10",
            "--annual-consumption-kwh",
            "4000",
            "--temperature-model",
            "sapm-close-mount-glass-glass",
            "--dry-run",
        ]
    )
    assert sapm_exit == 0

    noct_exit = cli.main(
        [
            "run",
            "--location",
            "porto",
            "--n-modules",
            "10",
            "--annual-consumption-kwh",
            "4000",
            "--temperature-model",
            "noct-sam",
            "--dry-run",
        ]
    )
    assert noct_exit == 1
    assert "NOCT metadata" in capsys.readouterr().err


def test_list_battery_models_exposes_scientific_metadata(capsys):
    exit_code = cli.main(["list", "battery-models", "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output) == 14
    lfp = next(row for row in output if row["key"] == "lfp_gr_250ah_prismatic")
    assert lfp["chemistry"] == "LFP/graphite"
    assert lfp["cell_format"] == "prismatic"
    assert lfp["experimental_range"]["cycling_temperature_c"] == [10, 45]
    assert lfp["release_phase"] == "core"


def test_run_cli_threads_blast_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "App", FakeApp)
    FakeApp.seen_config = None
    config_path = tmp_path / "quickstart.toml"
    config_path.write_text(
        'location = "porto"\nn_modules = 10\nannual_consumption_kwh = 4000\nbattery_kwh = 5.0\n',
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--degradation-engine",
            "blast",
            "--blast-model",
            "lfp_gr_250ah_prismatic",
        ]
    )

    assert exit_code == 0
    assert FakeApp.seen_config["degradation_engine"] == "blast"
    assert FakeApp.seen_config["blast_model"] == "lfp_gr_250ah_prismatic"


def test_validate_config_summarizes_without_simulation(tmp_path, capsys):
    config_path = tmp_path / "quickstart.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
cost_preset = "residential_pt"
emissions_country = "PT"
""".strip(),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-config", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Config OK" in output
    assert "Location: porto (Europe/Lisbon)" in output
    assert "PV: 10 modules" in output
    assert "Inverter AC rating" in output


def test_recommended_pv_example_validates(capsys):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "examples" / "recommended-pv.toml"

    exit_code = cli.main(["validate-config", str(config_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Config OK" in output
    assert "PV: 8 modules, 4.400 kWp" in output


def test_example_configs_are_discovered():
    # A moved or renamed directory would otherwise turn the parametrised test
    # below into zero silent cases.
    assert len(EXAMPLE_CONFIGS) >= 10


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda path: path.name)
def test_shipped_example_configs_validate(config_path, capsys):
    exit_code = cli.main(["validate-config", str(config_path)])

    assert exit_code == 0
    assert "Config OK" in capsys.readouterr().out


def test_validate_config_rejects_malformed_sweep(tmp_path, capsys):
    config_path = tmp_path / "invalid-sweep.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000

[sweep]
battery_kwh = []
""".strip()
    )

    exit_code = cli.main(["validate-config", str(config_path)])

    assert exit_code == 1
    assert "sweep.battery_kwh must be a non-empty array" in capsys.readouterr().err


def test_run_dry_run_outputs_resolved_config_without_simulating(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "App", FakeApp)
    FakeApp.simulated = False
    config_path = tmp_path / "quickstart.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "resolved.json"

    exit_code = cli.main(["run", "--config", str(config_path), "--dry-run", "--output", str(output_path)])

    assert exit_code == 0
    assert FakeApp.simulated is False
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["valid"] is True
    assert output["location"]["key"] == "porto"
    assert output["pv"]["n_modules"] == 10
    assert output["pv"]["losses"]["components_pct"]["shading"] == 3.0
    assert 14.0 < output["pv"]["losses"]["combined_pct"] < 15.0
    assert output["battery"]["degradation_engine"] == "native"
    assert output["battery"]["blast_model"] is None


def test_run_iam_model_override_is_reflected_in_dry_run(tmp_path):
    output_path = tmp_path / "resolved.json"

    exit_code = cli.main(
        [
            "run",
            "--location",
            "porto",
            "--n-modules",
            "8",
            "--annual-consumption-kwh",
            "3000",
            "--iam-model",
            "physical",
            "--dry-run",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["pv"]["iam_model"] == "physical"


def test_run_dry_run_reports_bifacial_geometry(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "App", FakeApp)
    output_path = tmp_path / "resolved.json"

    exit_code = cli.main(
        [
            "run",
            "--location",
            "porto",
            "--n-modules",
            "10",
            "--annual-consumption-kwh",
            "4000",
            "--pv-module",
            "Generic_600W_Bifacial",
            "--bifacial-model",
            "infinite_sheds",
            "--pvrow-height",
            "1.5",
            "--pvrow-pitch",
            "6.0",
            "--dry-run",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    pv = json.loads(output_path.read_text(encoding="utf-8"))["pv"]
    assert pv["bifacial_model"] == "infinite_sheds"
    assert pv["gcr"] == 0.35
    assert pv["pvrow_height"] == 1.5
    assert pv["pvrow_pitch"] == 6.0


def test_sweep_expands_grid_and_writes_combined_csv(monkeypatch, tmp_path, capsys):
    seen_configs = []

    class SweepFakeApp:
        def __init__(self, config):
            self.config = config
            seen_configs.append(config)

        def simulate(self):
            return None

        def result(self):
            return {
                "grid_independence_pct": 40.0 + self.config["n_modules"],
                "npv_savings": 1000.0 + self.config["battery_kwh"],
                "yearly": [{"year": 1}],
            }

    monkeypatch.setattr(cli, "App", SweepFakeApp)
    config_path = tmp_path / "sweep.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500
battery_kwh = 0
cost_preset = "residential_pt"

[sweep]
n_modules = [8, 10]
battery_kwh = [0.0, 5.0]
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "sweep.csv"

    exit_code = cli.main(["sweep", "--config", str(config_path), "--output", str(output_path), "--json"])

    assert exit_code == 0
    assert len(seen_configs) == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"] == 4
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert len(rows) == 4
    assert rows[0]["param_n_modules"] == "8"
    assert rows[-1]["param_battery_kwh"] == "5.0"
    assert "yearly" not in rows[0]
    assert rows[0]["grid_independence_pct"] == "48.0"


@pytest.mark.usefixtures("_patch_weather")
def test_sweep_csv_carries_the_year1_money_components(tmp_path, capsys):
    config_path = tmp_path / "money-sweep.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500
projection_years = 1

[sweep]
battery_kwh = [0.0, 5.0]
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "money-sweep.csv"

    assert cli.main(["sweep", "--config", str(config_path), "--output", str(output_path)]) == 0

    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert len(rows) == 2
    for row in rows:
        assert row["result_schema_version"] == "2.1"
        for key in (
            "grid_import_cost_year1_prices",
            "grid_export_revenue_year1_prices",
            "fixed_charge_year1_prices",
            "no_system_import_cost_year1_prices",
        ):
            assert float(row[key]) >= 0.0
        assert float(row["no_system_import_cost_year1_prices"]) > float(row["grid_import_cost_year1_prices"])


def test_sweep_applies_dotted_cost_keys_without_mutating_base_config(monkeypatch, tmp_path, capsys):
    seen_configs = []

    class SweepFakeApp:
        def __init__(self, config):
            self.config = config
            seen_configs.append(config)

        def simulate(self):
            return None

        def result(self):
            return {"npv_savings": self.config["costs"]["electricity_cost"] * 1000}

    monkeypatch.setattr(cli, "App", SweepFakeApp)
    config_path = tmp_path / "cost-sweep.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500
cost_preset = "residential_pt"

[costs]
storage_cost_per_kwh = 420.0

[sweep]
"costs.electricity_cost" = [0.20, 0.30]
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "cost-sweep.csv"

    exit_code = cli.main(["sweep", "--config", str(config_path), "--output", str(output_path), "--json"])

    assert exit_code == 0
    assert [config["costs"] for config in seen_configs] == [
        {"storage_cost_per_kwh": 420.0, "electricity_cost": 0.20},
        {"storage_cost_per_kwh": 420.0, "electricity_cost": 0.30},
    ]
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert [row["param_costs.electricity_cost"] for row in rows] == ["0.2", "0.3"]
    assert [row["npv_savings"] for row in rows] == ["200.0", "300.0"]


def test_sweep_accepts_unquoted_toml_dotted_cost_key(monkeypatch, tmp_path, capsys):
    seen_configs = []

    class SweepFakeApp:
        def __init__(self, config):
            seen_configs.append(config)

        def simulate(self):
            return None

        def result(self):
            return {}

    monkeypatch.setattr(cli, "App", SweepFakeApp)
    config_path = tmp_path / "nested-cost-sweep.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500

[sweep]
costs.electricity_cost = [0.20, 0.30]
""".strip(),
        encoding="utf-8",
    )

    exit_code = cli.main(["sweep", "--config", str(config_path), "--output", str(tmp_path / "out.csv")])

    assert exit_code == 0
    assert [config["costs"]["electricity_cost"] for config in seen_configs] == [0.20, 0.30]
    assert "Sweep: 2 runs" in capsys.readouterr().out


def test_validate_config_rejects_unknown_dotted_sweep_cost_key(tmp_path, capsys):
    config_path = tmp_path / "invalid-cost-sweep.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500

[sweep]
"costs.electricty_cost" = [0.20, 0.30]
""".strip(),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-config", str(config_path)])

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "Unknown sweep key 'costs.electricty_cost'" in error
    assert "costs.electricity_cost" in error


def test_validate_config_rejects_dotted_sweep_key_outside_costs(tmp_path, capsys):
    config_path = tmp_path / "invalid-nested-sweep.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500

[sweep]
"location.foo" = ["ignored-before-validation"]
""".strip(),
        encoding="utf-8",
    )

    exit_code = cli.main(["validate-config", str(config_path)])

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "Unknown sweep key 'location.foo'" in error
    assert (
        "Dotted keys are supported only under 'battery_indoor_model', 'costs', 'period', 'smart_charging', 'tariff'"
        in error
    )


SMART_CHARGING_SWEEP_BASE = """
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0

[tariff]
schedule = "pt_mainland_2026_daily_bi"
currency = "EUR"
import_prices = { peak = 0.2310, off_peak = 0.1210 }
export_prices = { all = 0.0500 }

[smart_charging]
mode = "fixed_target"
target_usable_fraction = 0.50
charge_periods = ["off_peak"]
discharge_periods = ["peak"]
grid_charge_efficiency = 0.95
"""


def test_sweep_applies_dotted_tariff_and_smart_charging_keys(monkeypatch, tmp_path):
    seen_configs = []

    class SweepFakeApp:
        def __init__(self, config):
            seen_configs.append(config)

        def simulate(self):
            return None

        def result(self):
            return {}

    monkeypatch.setattr(cli, "App", SweepFakeApp)
    config_path = tmp_path / "tariff-sweep.toml"
    config_path.write_text(
        SMART_CHARGING_SWEEP_BASE
        + """
[sweep]
"tariff.import_prices.off_peak" = [0.10, 0.12]
"smart_charging.target_usable_fraction" = [0.5, 0.8]
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "tariff-sweep.csv"

    assert cli.main(["sweep", "--config", str(config_path), "--output", str(output_path)]) == 0

    assert [(c["tariff"]["import_prices"], c["smart_charging"]["target_usable_fraction"]) for c in seen_configs] == [
        ({"peak": 0.2310, "off_peak": 0.10}, 0.5),
        ({"peak": 0.2310, "off_peak": 0.10}, 0.8),
        ({"peak": 0.2310, "off_peak": 0.12}, 0.5),
        ({"peak": 0.2310, "off_peak": 0.12}, 0.8),
    ]
    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert [row["param_tariff.import_prices.off_peak"] for row in rows] == ["0.1", "0.1", "0.12", "0.12"]


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("tariff.shedule", r"Unknown sweep key 'tariff\.shedule'\. Available: tariff\.boundary_policy"),
        ("smart_charging.target", r"Available: smart_charging\.charge_periods"),
        ("tariff.schedule.peak", r"'tariff\.schedule' is not a table of named entries"),
        ("tariff.import_prices.peak.low", r"'tariff\.import_prices' takes one more level, the entry name"),
        ("tariff.export_prices.all.x", r"as in 'tariff\.export_prices\.all'"),
        ("costs", r"Unknown sweep key 'costs'\. Available: costs\.daily_power_cost"),
        ("battery_indoor_model.setpoint", r"Available: battery_indoor_model\.ceiling_c"),
    ],
)
def test_sweep_rejects_dotted_keys_the_registry_does_not_have(tmp_path, capsys, key, message):
    config_path = tmp_path / "bad-key-sweep.toml"
    config_path.write_text(SMART_CHARGING_SWEEP_BASE + f'\n[sweep]\n"{key}" = [1.0]\n', encoding="utf-8")

    assert cli.main(["validate-config", str(config_path)]) == 1
    assert re.search(message, capsys.readouterr().err)


def test_sweep_resolves_every_grid_point_before_the_first_run(monkeypatch, tmp_path, capsys):
    simulated = []
    monkeypatch.setattr(cli.App, "simulate", lambda self: simulated.append(self))
    config_path = tmp_path / "bad-point-sweep.toml"
    # The second value names a period the schedule does not have.
    config_path.write_text(
        SMART_CHARGING_SWEEP_BASE + '\n[sweep]\n"smart_charging.charge_periods" = [["off_peak"], ["night"]]\n',
        encoding="utf-8",
    )

    assert cli.main(["sweep", "--config", str(config_path), "--output", str(tmp_path / "out.csv")]) == 1
    assert "'smart_charging.charge_periods' has period(s) night" in capsys.readouterr().err
    assert simulated == []
    assert not (tmp_path / "out.csv").exists()

    # validate-config checks every grid point too, not only the base config.
    assert cli.main(["validate-config", str(config_path)]) == 1
    assert "'smart_charging.charge_periods' has period(s) night" in capsys.readouterr().err


def test_sweep_builds_each_app_only_when_it_runs(monkeypatch, tmp_path):
    events = []

    class SweepFakeApp:
        def __init__(self, config):
            events.append(("build", config["n_modules"]))
            self.n_modules = config["n_modules"]

        def simulate(self):
            events.append(("run", self.n_modules))

        def result(self):
            return {}

    monkeypatch.setattr(cli, "App", SweepFakeApp)
    config_path = tmp_path / "order-sweep.toml"
    config_path.write_text('location = "porto"\nannual_consumption_kwh = 3500\n\n[sweep]\nn_modules = [8, 10]\n')

    assert cli.main(["sweep", "--config", str(config_path), "--output", str(tmp_path / "out.csv")]) == 0
    # A finished run's App, and its result, is not held until the sweep ends.
    assert events == [("build", 8), ("run", 8), ("build", 10), ("run", 10)]


def test_deep_merge_keeps_the_rest_of_a_nested_table():
    base = {"n_modules": 8, "tariff": {"schedule": "pt_mainland_2026_daily_bi", "import_prices": {"peak": 0.23}}}
    overrides = {"n_modules": 10, "tariff": {"import_prices": {"off_peak": 0.12}}, "battery_kwh": 5.0}

    merged = cli._deep_merge(base, overrides)

    assert merged == {
        "n_modules": 10,
        "tariff": {"schedule": "pt_mainland_2026_daily_bi", "import_prices": {"peak": 0.23, "off_peak": 0.12}},
        "battery_kwh": 5.0,
    }
    assert base["tariff"] == {"schedule": "pt_mainland_2026_daily_bi", "import_prices": {"peak": 0.23}}
    # A scalar override replaces a table, and a table replaces a scalar.
    assert cli._deep_merge({"costs": {"a": 1}}, {"costs": None}) == {"costs": None}
    assert cli._deep_merge({"costs": None}, {"costs": {"a": 1}}) == {"costs": {"a": 1}}
