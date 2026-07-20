"""Tests for the BREOS command line interface."""

import csv
import json

from breos import cli


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
            "6",
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
    assert FakeApp.seen_config["load_profile"] == "6"
    assert FakeApp.seen_config["rlp_directory"] == "/tmp/external-rlp"

    output = json.loads(capsys.readouterr().out)
    assert output["grid_independence_pct"] == 42.0


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
                "npv_savings_eur": 1000.0 + self.config["battery_kwh"],
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
