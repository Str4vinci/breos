"""Characterization tests for declarative App configuration metadata."""

import argparse

from breos import cli
from breos.app_config import ALLOWED_CONFIG_KEYS, APP_CONFIG_FIELDS, DEFAULTS

EXPECTED_DEFAULTS = {
    "battery_kwh": 0.0,
    "pv_arrays": None,
    "pv_module": None,
    "load_profile": "1",
    "rlp_directory": None,
    "tilt": None,
    "azimuth": None,
    "tracking": "fixed",
    "axis_tilt": 0.0,
    "axis_azimuth": None,
    "max_angle": 60.0,
    "backtrack": True,
    "gcr": 0.35,
    "cross_axis_tilt": 0.0,
    "dual_axis_max_tilt": 90.0,
    "transposition_model": "isotropic",
    "albedo": None,
    "surface_type": None,
    "model_perez": "allsitescomposite1990",
    "solar_position": "interval-start",
    "iam_model": "ashrae",
    "diffuse_iam": "none",
    "temperature_model": "faiman",
    "bifacial_model": "none",
    "pvrow_height": None,
    "pvrow_pitch": None,
    "resolution": "h",
    "projection_years": 20,
    "cost_preset": None,
    "inflation_rate": 0.02,
    "sell_price_inflation": 0.0,
    "discount_rate": 0.03,
    "emissions_country": None,
    "export_emissions_factor_gco2_kwh": None,
    "pv_degradation_rate": 0.005,
    "calendar_model": "naumann_lam_field_calibrated",
    "degradation_engine": "native",
    "blast_model": None,
    "battery_min_soc": 0.10,
    "battery_max_soc": 0.90,
    "battery_eol_percentage": 0.70,
    "battery_rte": None,
    "battery_max_charge_power_w": None,
    "battery_max_discharge_power_w": None,
    "enable_resistance_fade": False,
    "dc_coupled": True,
    "inverter_efficiency": 0.96,
    "inverter_loading_ratio": 1.25,
    "pv_loss_overrides": None,
    "start_date": "2023-01-01",
    "horizon_profile": None,
    "battery_temperature": "weather",
    "battery_indoor_model": None,
}

EXPECTED_CLI_FIELDS = [
    "location",
    "n_modules",
    "annual_consumption_kwh",
    "battery_kwh",
    "battery_max_charge_power_w",
    "battery_max_discharge_power_w",
    "cost_preset",
    "emissions_country",
    "pv_module",
    "load_profile",
    "rlp_directory",
    "tilt",
    "azimuth",
    "transposition_model",
    "albedo",
    "surface_type",
    "model_perez",
    "solar_position",
    "iam_model",
    "diffuse_iam",
    "temperature_model",
    "bifacial_model",
    "pvrow_height",
    "pvrow_pitch",
    "gcr",
    "resolution",
    "projection_years",
    "inflation_rate",
    "sell_price_inflation",
    "export_emissions_factor_gco2_kwh",
    "discount_rate",
    "pv_degradation_rate",
    "calendar_model",
    "degradation_engine",
    "blast_model",
    "dc_coupled",
    "inverter_efficiency",
    "inverter_loading_ratio",
    "start_date",
]


def _run_parser() -> argparse.ArgumentParser:
    parser = cli.build_parser()
    subcommands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    return subcommands.choices["run"]


def test_registry_preserves_defaults_and_allowed_top_level_keys():
    assert DEFAULTS == EXPECTED_DEFAULTS
    assert list(DEFAULTS) == list(EXPECTED_DEFAULTS)
    default_orders = [field.default_order for field in APP_CONFIG_FIELDS.values() if field.has_default]
    assert sorted(default_orders) == list(range(len(EXPECTED_DEFAULTS)))
    assert ALLOWED_CONFIG_KEYS == frozenset(
        set(EXPECTED_DEFAULTS)
        | {
            "location",
            "annual_consumption_kwh",
            "n_modules",
            "costs",
            "montecarlo",
            "sweep",
            "battery_type",
        }
    )


def test_registry_generates_every_app_config_cli_option():
    run_actions = _run_parser()._actions
    actions = {action.dest: action for action in run_actions}
    registered_cli_fields = {key for key, field in APP_CONFIG_FIELDS.items() if field.cli_flags}
    non_config_destinations = {"help", "config", "output", "indent", "dry_run"}

    assert set(actions) - non_config_destinations == registered_cli_fields
    assert [action.dest for action in run_actions if action.dest not in non_config_destinations] == EXPECTED_CLI_FIELDS
    for key in registered_cli_fields:
        field = APP_CONFIG_FIELDS[key]
        action = actions[key]
        assert tuple(action.option_strings) == field.cli_flags
        assert action.type is field.cli_type
        assert action.help == field.cli_help
        assert action.default is None
        actual_choices = tuple(action.choices) if action.choices is not None else None
        assert actual_choices == field.cli_choices
        if field.cli_action == "store_true":
            assert action.const is True
            assert action.default is None


def test_registry_generated_cli_values_all_reach_config_overrides():
    argv = ["run"]
    expected: dict[str, object] = {}
    for key, field in APP_CONFIG_FIELDS.items():
        if not field.cli_flags:
            continue
        argv.append(field.cli_flags[0])
        if field.cli_action == "store_true":
            expected[key] = True
            continue
        if field.cli_choices is not None:
            raw: object = field.cli_choices[0]
        elif field.cli_type is int:
            raw = 2
        elif field.cli_type is float:
            raw = 0.5
        elif key == "cost_preset":
            raw = "some-preset"
        elif key == "emissions_country":
            raw = "pt"
        elif key == "location":
            raw = "PORTO"
        else:
            raw = "/tmp/value" if field.cli_type is not None else "value"
        argv.append(str(raw))
        expected[key] = field.cli_normalizer(raw) if field.cli_normalizer is not None else raw

    args = cli.build_parser().parse_args(argv)

    assert cli._build_config(args) == expected


def test_empty_normalized_cli_values_do_not_overwrite_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'location = "berlin"\ncost_preset = "residential_de"\nemissions_country = "DE"\n',
        encoding="utf-8",
    )
    args = cli.build_parser().parse_args(
        [
            "run",
            "--config",
            str(config_path),
            "--location",
            "",
            "--cost-preset",
            "",
            "--emissions-country",
            "",
        ]
    )

    config = cli._build_config(args)

    assert config["location"] == "berlin"
    assert config["cost_preset"] == "residential_de"
    assert config["emissions_country"] == "DE"
