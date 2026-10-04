"""Characterization tests for declarative App configuration metadata."""

import re
from datetime import date

import pytest

from breos import cli
from breos.app_config import ALLOWED_CONFIG_KEYS, APP_CONFIG_FIELDS, DEFAULTS, resolve_app_config

EXPECTED_DEFAULTS = {
    "battery_kwh": 0.0,
    "pv_arrays": None,
    "pv_module": None,
    "load_profile": "demandlib_h0",
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
    "irradiance_resampling": "auto",
    "iam_model": "ashrae",
    "diffuse_iam": "none",
    "temperature_model": "faiman",
    "bifacial_model": "none",
    "pvrow_height": None,
    "pvrow_pitch": None,
    "resolution": "h",
    "projection_years": 20,
    "cost_preset": None,
    "currency": None,
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
    "battery_allow_terminal_replacement": True,
    "battery_replacement_min_remaining_years": 0.0,
    "battery_skipped_replacement_action": "keep",
    "battery_rte": None,
    "battery_max_charge_power_w": None,
    "battery_max_discharge_power_w": None,
    "battery_power_limit_c_rate": None,
    "enable_resistance_fade": False,
    "inverter_efficiency": 0.96,
    "inverter_loading_ratio": 1.25,
    "pv_loss_overrides": None,
    "start_date": "2023-01-01",
    "horizon_profile": None,
    "battery_temperature": "weather",
    "battery_indoor_model": None,
    "execution_backend": "python",
    "weather_file": None,
    "weather_source": None,
    "load_profile_file": None,
    "load_profile_column": None,
    "load_profile_unit": None,
    "tariff": None,
    "reference_tariff": None,
    "terminal_value": None,
    "smart_charging": None,
    "import_price_escalation": None,
    "om_escalation": None,
    "replacement_cost_learning": 0.0,
    "inverter_ac_rating_kw": None,
}

# Every App flag of ``breos run``, in the order ``--help`` lists them: the
# flag, a sample argument (none for a switch), and the config entry the flag
# must produce, after the key's normalisation. The spellings are the
# user-facing contract, so they are written out rather than read back from
# the registry.
RUN_FLAGS: list[tuple[str, list[str], str, object]] = [
    ("--location", ["PORTO"], "location", "porto"),
    ("--n-modules", ["12"], "n_modules", 12),
    ("--annual-consumption-kwh", ["3500"], "annual_consumption_kwh", 3500.0),
    ("--battery-kwh", ["5"], "battery_kwh", 5.0),
    ("--battery-max-charge-power-w", ["2500"], "battery_max_charge_power_w", 2500.0),
    ("--battery-max-discharge-power-w", ["2600"], "battery_max_discharge_power_w", 2600.0),
    ("--battery-power-limit-c-rate", ["0.5"], "battery_power_limit_c_rate", 0.5),
    ("--cost-preset", ["residential-pt"], "cost_preset", "residential_pt"),
    ("--currency", ["usd"], "currency", "USD"),
    ("--emissions-country", ["pt"], "emissions_country", "PT"),
    ("--pv-module", ["Suntech_STP550S_STC"], "pv_module", "Suntech_STP550S_STC"),
    ("--load-profile", ["bdew_h0"], "load_profile", "bdew_h0"),
    ("--rlp-directory", ["rlp"], "rlp_directory", "rlp"),
    ("--load-profile-file", ["load.csv"], "load_profile_file", "load.csv"),
    ("--load-profile-column", ["Load"], "load_profile_column", "Load"),
    ("--load-profile-unit", ["kWh"], "load_profile_unit", "kWh"),
    ("--tilt", ["30"], "tilt", 30.0),
    ("--azimuth", ["170"], "azimuth", 170.0),
    ("--transposition-model", ["perez"], "transposition_model", "perez"),
    ("--sky-model", ["haydavies"], "transposition_model", "haydavies"),
    ("--albedo", ["0.2"], "albedo", 0.2),
    ("--surface-type", ["grass"], "surface_type", "grass"),
    ("--perez-model", ["allsitescomposite1988"], "model_perez", "allsitescomposite1988"),
    ("--irradiance-resampling", ["clear_sky"], "irradiance_resampling", "clear_sky"),
    ("--solar-position", ["mid-interval"], "solar_position", "mid-interval"),
    ("--iam-model", ["physical"], "iam_model", "physical"),
    ("--diffuse-iam", ["marion"], "diffuse_iam", "marion"),
    ("--temperature-model", ["pvsyst-freestanding"], "temperature_model", "pvsyst-freestanding"),
    ("--bifacial-model", ["infinite_sheds"], "bifacial_model", "infinite_sheds"),
    ("--pvrow-height", ["1.5"], "pvrow_height", 1.5),
    ("--pvrow-pitch", ["5"], "pvrow_pitch", 5.0),
    ("--gcr", ["0.4"], "gcr", 0.4),
    ("--resolution", ["15min"], "resolution", "15min"),
    ("--projection-years", ["25"], "projection_years", 25),
    ("--inflation-rate", ["0.02"], "inflation_rate", 0.02),
    ("--import-price-escalation", ["0.03"], "import_price_escalation", 0.03),
    ("--om-escalation", ["0.01"], "om_escalation", 0.01),
    ("--replacement-cost-learning", ["0.02"], "replacement_cost_learning", 0.02),
    ("--sell-price-inflation", ["0.015"], "sell_price_inflation", 0.015),
    ("--export-emissions-factor-gco2-kwh", ["200"], "export_emissions_factor_gco2_kwh", 200.0),
    ("--discount-rate", ["0.04"], "discount_rate", 0.04),
    ("--pv-degradation-rate", ["0.005"], "pv_degradation_rate", 0.005),
    ("--calendar-model", ["Naumann-Lam"], "calendar_model", "naumann_lam"),
    ("--degradation-engine", ["blast"], "degradation_engine", "blast"),
    ("--blast-model", ["lfp_gr_250ah_prismatic"], "blast_model", "lfp_gr_250ah_prismatic"),
    ("--inverter-efficiency", ["0.97"], "inverter_efficiency", 0.97),
    ("--inverter-loading-ratio", ["1.2"], "inverter_loading_ratio", 1.2),
    ("--inverter-ac-rating-kw", ["4.6"], "inverter_ac_rating_kw", 4.6),
    ("--start-date", ["2024-01-01"], "start_date", "2024-01-01"),
    ("--weather-file", ["tmy.csv"], "weather_file", "tmy.csv"),
    ("--weather-source", ["pvgis"], "weather_source", "pvgis"),
    ("--execution-backend", ["numba"], "execution_backend", "numba"),
]

# The ``breos run`` options that are not App config keys.
RUN_COMMAND_OPTIONS = ("-h", "--help", "--config", "--output", "--indent", "--dry-run")


def _run_config(*argv: str) -> dict[str, object]:
    """The config overrides ``breos run`` builds from ``argv``."""
    return cli._build_config(cli.build_parser().parse_args(["run", *argv]))


def test_registry_preserves_defaults_and_allowed_top_level_keys():
    assert DEFAULTS == EXPECTED_DEFAULTS
    assert ALLOWED_CONFIG_KEYS == frozenset(
        set(EXPECTED_DEFAULTS)
        | {
            "location",
            "annual_consumption_kwh",
            "n_modules",
            "costs",
            "period",
            "montecarlo",
            "sweep",
        }
    )


@pytest.mark.parametrize(("flag", "argument", "key", "expected"), RUN_FLAGS, ids=[row[0] for row in RUN_FLAGS])
def test_each_run_flag_sets_its_config_key(flag, argument, key, expected):
    assert _run_config(flag, *argument) == {key: expected}


def test_run_help_lists_exactly_the_tabled_flags_in_order(capsys, monkeypatch):
    # Python 3.14's argparse colours help when the environment asks for it,
    # and PYTHON_COLORS outranks NO_COLOR there.
    monkeypatch.setenv("PYTHON_COLORS", "0")
    monkeypatch.setenv("NO_COLOR", "1")
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--help"])
    help_text = capsys.readouterr().out
    listed = re.findall(r"(?:^  |, )(--?[a-z][a-z0-9-]*)", help_text, flags=re.MULTILINE)

    assert [flag for flag in listed if flag not in RUN_COMMAND_OPTIONS] == [row[0] for row in RUN_FLAGS]


def test_run_without_flags_sets_no_config_key():
    assert _run_config() == {}


@pytest.mark.parametrize(("flag", "value"), [("--resolution", "30min"), ("--degradation-engine", "Blast")])
def test_choice_flags_reject_a_value_outside_their_choices(flag, value, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _run_config(flag, value)

    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_dc_coupled_is_an_unknown_app_key():
    with pytest.raises(ValueError, match="Unknown config key.*dc_coupled"):
        resolve_app_config({"dc_coupled": True})


def test_dc_coupled_has_no_cli_flag():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--dc-coupled"])


def test_registry_generated_cli_values_all_reach_config_overrides():
    argv = ["run"]
    for key, field in APP_CONFIG_FIELDS.items():
        if not field.cli_flags:
            continue
        argv.append(field.cli_flags[0])
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

    args = cli.build_parser().parse_args(argv)
    expected = {
        key: field.normalizer(getattr(args, key)) if field.normalizer is not None else getattr(args, key)
        for key, field in APP_CONFIG_FIELDS.items()
        if field.cli_flags
    }

    assert cli._build_config(args) == expected


def test_value_and_nested_key_normalization_is_shared_by_api_and_toml(tmp_path):
    api_config = {
        "location": "PORTO",
        "n_modules": 10,
        "annual_consumption_kwh": 4000,
        "cost_preset": "residential-pt",
        "emissions_country": "pt",
        "start_date": date(2023, 1, 1),
        "costs": {"electricity-cost": 0.31},
    }

    api_cfg = resolve_app_config(api_config).cfg

    assert api_cfg["location"] == "porto"
    assert api_cfg["cost_preset"] == "residential_pt"
    assert api_cfg["emissions_country"] == "PT"
    assert api_cfg["start_date"] == "2023-01-01"
    assert api_cfg["costs"] == {"electricity_cost": 0.31}

    config_path = tmp_path / "normalization.toml"
    config_path.write_text(
        'location = "PORTO"\n'
        "n_modules = 10\n"
        "annual_consumption_kwh = 4000\n"
        'cost_preset = "residential-pt"\n'
        'emissions_country = "pt"\n'
        "start_date = 2023-01-01\n"
        "\n[costs]\n"
        "electricity-cost = 0.31\n",
        encoding="utf-8",
    )

    toml_cfg = resolve_app_config(cli._load_config(config_path)).cfg

    assert toml_cfg["location"] == api_cfg["location"]
    assert toml_cfg["cost_preset"] == api_cfg["cost_preset"]
    assert toml_cfg["emissions_country"] == api_cfg["emissions_country"]
    assert toml_cfg["start_date"] == api_cfg["start_date"]
    assert toml_cfg["costs"] == api_cfg["costs"]


def test_key_normalization_rejects_ambiguous_spellings():
    with pytest.raises(ValueError, match="Duplicate config key 'electricity_cost'"):
        resolve_app_config(
            {
                "location": "porto",
                "n_modules": 10,
                "annual_consumption_kwh": 4000,
                "costs": {"electricity-cost": 0.31, "electricity_cost": 0.32},
            }
        )


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
