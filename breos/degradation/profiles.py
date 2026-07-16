"""Declarative metadata and discovery for vendored BLAST cell models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

BLAST_UPSTREAM_VERSION = "1.1.0"
BLAST_UPSTREAM_COMMIT = "d789e00bca60f628de640745c18eb724b07358bd"
BLAST_STATE_SCHEMA_VERSION = "1.0"


def _freeze_metadata(value: Any) -> Any:
    """Recursively freeze registry metadata without changing its values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _json_metadata(value: Any) -> Any:
    """Return fresh JSON-safe containers for frozen registry metadata."""
    if isinstance(value, Mapping):
        return {key: _json_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_metadata(item) for item in value]
    return value


@dataclass(frozen=True)
class BatteryModelProfile:
    """Stable public identity and scientific scope of one BLAST cell model."""

    key: str
    name: str
    source_module: str
    class_name: str
    chemistry: str
    cell_format: str
    nominal_capacity_ah: float
    experimental_range: Mapping[str, Any]
    citations: tuple[str, ...]
    output_keys: tuple[str, ...]
    operating_defaults: Mapping[str, Any]
    release_phase: str
    notes: str = ""

    @property
    def supports_capacity(self) -> bool:
        return "q" in self.output_keys

    @property
    def supports_resistance(self) -> bool:
        return "r" in self.output_keys

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "source_module": self.source_module,
            "class_name": self.class_name,
            "chemistry": self.chemistry,
            "cell_format": self.cell_format,
            "nominal_capacity_ah": self.nominal_capacity_ah,
            "experimental_range": _json_metadata(self.experimental_range),
            "citations": list(self.citations),
            "output_keys": list(self.output_keys),
            "operating_defaults": _json_metadata(self.operating_defaults),
            "release_phase": self.release_phase,
            "notes": self.notes,
            "supports_capacity": self.supports_capacity,
            "supports_resistance": self.supports_resistance,
            "calibration_basis": "cell-model",
            "pack_calibrated": False,
            "upstream": {
                "project": "BLAST-Lite",
                "version": BLAST_UPSTREAM_VERSION,
                "commit": BLAST_UPSTREAM_COMMIT,
            },
        }


def _profile(
    key: str,
    name: str,
    source_module: str,
    class_name: str,
    chemistry: str,
    cell_format: str,
    nominal_capacity_ah: float,
    experimental_range: dict[str, Any],
    citations: tuple[str, ...],
    output_keys: tuple[str, ...],
    release_phase: str = "phase3",
    notes: str = "",
) -> BatteryModelProfile:
    # BLAST model papers do not define pack-level operating defaults. Keep the
    # mapping empty rather than inventing generic chemistry assumptions.
    return BatteryModelProfile(
        key=key,
        name=name,
        source_module=source_module,
        class_name=class_name,
        chemistry=chemistry,
        cell_format=cell_format,
        nominal_capacity_ah=nominal_capacity_ah,
        experimental_range=_freeze_metadata(experimental_range),
        citations=citations,
        output_keys=output_keys,
        operating_defaults=_freeze_metadata({}),
        release_phase=release_phase,
        notes=notes,
    )


_COMMON_EST_2023 = ("https://doi.org/10.1016/j.est.2023.109042",)

BATTERY_MODEL_REGISTRY: Mapping[str, BatteryModelProfile] = MappingProxyType(
    {
        "lfp_gr_250ah_prismatic": _profile(
            "lfp_gr_250ah_prismatic",
            "LFP-Gr 250 Ah prismatic",
            "lfp_gr_250AhPrismatic_2019",
            "Lfp_Gr_250AhPrismatic",
            "LFP/graphite",
            "prismatic",
            250.0,
            {
                "cycling_temperature_c": [10, 45],
                "dod": [0.8, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 0.65,
                "max_c_rate_discharge": 1.0,
            },
            _COMMON_EST_2023,
            ("q", "q_t", "q_EFC"),
            release_phase="core",
        ),
        "nca_gr_panasonic_3ah": _profile(
            "nca_gr_panasonic_3ah",
            "NCA-Gr Panasonic 3.2 Ah",
            "nca_gr_Panasonic3Ah_2018",
            "Nca_Gr_Panasonic3Ah_Battery",
            "NCA/graphite",
            "cylindrical",
            3.2,
            {
                "cycling_temperature_c": [15, 35],
                "dod": [0.8, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 0.5,
                "max_c_rate_discharge": 2.0,
            },
            ("https://doi.org/10.1149/2.0411609jes", "https://doi.org/10.1149/1945-7111/abae37"),
            ("q", "q_t", "q_EFC"),
            release_phase="core",
        ),
        "lmo_gr_nissanleaf_66ah_2nd": _profile(
            "lmo_gr_nissanleaf_66ah_2nd",
            "LMO-Gr Nissan Leaf 66 Ah second-life",
            "lmo_gr_NissanLeaf66Ah_2ndLife_2020",
            "Lmo_Gr_NissanLeaf66Ah_2ndLife_Battery",
            "LMO/graphite",
            "pouch",
            66.0,
            {
                "cycling_temperature_c": [20, 30],
                "dod": [0.8, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 1.0,
                "max_c_rate_discharge": 1.0,
            },
            ("https://doi.org/10.1109/EEEIC/ICPSEUROPE54979.2022.9854784", "https://doi.org/10.1016/j.est.2020.101695"),
            ("q", "q_t", "q_EFC", "qNew"),
        ),
        "nmc811_grsi_lgm50_5ah": _profile(
            "nmc811_grsi_lgm50_5ah",
            "NMC811-GrSi LG M50 5 Ah",
            "nmc811_grSi_LGM50_5Ah_2021",
            "Nmc811_GrSi_LGM50_5Ah_Battery",
            "NMC811/graphite-silicon",
            "cylindrical 21700",
            5.0,
            {
                "cycling_temperature_c": [0, 25],
                "dod": [1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 0.3,
                "max_c_rate_discharge": 2.0,
            },
            (
                "https://ieeexplore.ieee.org/document/9617644",
                "https://www.sciencedirect.com/science/article/pii/S0013468622008593",
            ),
            ("q", "q_t", "q_EFC"),
        ),
        "nmc811_grsi_lgmj1_4ah": _profile(
            "nmc811_grsi_lgmj1_4ah",
            "NMC811-GrSi LG MJ1 3.5 Ah",
            "nmc811_grSi_LGMJ1_4Ah_2020",
            "Nmc811_GrSi_LGMJ1_4Ah_Battery",
            "NMC811/graphite-silicon",
            "cylindrical 18650",
            3.5,
            {
                "cycling_temperature_c": [0, 50],
                "dod": [0.2, 0.8],
                "soc": [0.1, 0.9],
                "max_c_rate_charge": 1.0,
                "max_c_rate_discharge": 3.0,
            },
            ("https://everlasting-project.eu/wp-content/uploads/2020/03/EVERLASTING_D2.3_final_20200228.pdf",),
            ("q", "q_t", "q_EFC"),
        ),
        "nmc_gr_50ah_b1": _profile(
            "nmc_gr_50ah_b1",
            "NMC-Gr B1 50 Ah",
            "nmc_gr_50Ah_B1_2020",
            "NMC_Gr_50Ah_B1",
            "NMC/graphite",
            "not stated",
            50.0,
            {
                "cycling_temperature_c": [10, 45],
                "dod": [0.8, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 1.75,
                "max_c_rate_discharge": 1.75,
            },
            _COMMON_EST_2023,
            ("q", "q_t", "q_EFC"),
        ),
        "nmc_gr_50ah_b2": _profile(
            "nmc_gr_50ah_b2",
            "NMC-Gr B2 50 Ah",
            "nmc_gr_50Ah_B2_2020",
            "NMC_Gr_50Ah_B2",
            "NMC/graphite",
            "not stated",
            50.0,
            {
                "cycling_temperature_c": [10, 45],
                "dod": [0.8, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 1.75,
                "max_c_rate_discharge": 1.75,
            },
            _COMMON_EST_2023,
            ("q", "q_t", "q_EFC"),
        ),
        "nmc_gr_75ah_a": _profile(
            "nmc_gr_75ah_a",
            "NMC-Gr A 75 Ah",
            "nmc_gr_75Ah_A_2019",
            "NMC_Gr_75Ah_A",
            "NMC/graphite",
            "not stated",
            75.0,
            {
                "cycling_temperature_c": [10, 45],
                "dod": [0.8, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 2.0,
                "max_c_rate_discharge": 2.0,
            },
            _COMMON_EST_2023,
            ("q", "q_t", "q_EFC"),
        ),
        "nmc111_gr_sanyo_2ah": _profile(
            "nmc111_gr_sanyo_2ah",
            "NMC111-Gr Sanyo 2.15 Ah",
            "nmc111_gr_Sanyo2Ah_2014",
            "Nmc111_Gr_Sanyo2Ah_Battery",
            "NMC111/graphite",
            "cylindrical 18650",
            2.15,
            {
                "cycling_temperature_c": [20, 40],
                "dod": [0, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 1.0,
                "max_c_rate_discharge": 1.0,
            },
            ("https://doi.org/10.1016/j.jpowsour.2014.02.012", "https://doi.org/10.1016/j.jpowsour.2013.09.143"),
            ("q", "q_t", "q_EFC", "r", "r_t", "r_EFC"),
        ),
        "nmc_lto_10ah": _profile(
            "nmc_lto_10ah",
            "NMC-LTO 10.2 Ah",
            "nmc_lto_10Ah_2020",
            "Nmc_Lto_10Ah_Battery",
            "NMC/LTO",
            "not stated",
            10.2,
            {
                "cycling_temperature_c": [30, 60],
                "dod": [0, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 10.0,
                "max_c_rate_discharge": 10.0,
            },
            ("https://doi.org/10.1016/j.jpowsour.2020.228566",),
            ("q", "q_t_loss", "q_t_gain", "q_EFC"),
        ),
        "lfp_gr_sonymurata_3ah": _profile(
            "lfp_gr_sonymurata_3ah",
            "LFP-Gr Sony-Murata 3 Ah",
            "lfp_gr_SonyMurata3Ah_2018",
            "Lfp_Gr_SonyMurata3Ah_Battery",
            "LFP/graphite",
            "cylindrical",
            3.0,
            {
                "cycling_temperature_c": [20, 40],
                "dod": [0.8, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 1.0,
                "max_c_rate_discharge": 2.0,
            },
            (
                "https://doi.org/10.1016/j.est.2018.01.019",
                "https://doi.org/10.1016/j.jpowsour.2019.227666",
                "https://doi.org/10.1149/1945-7111/ac86a8",
            ),
            ("q", "q_LLI_t", "q_LLI_EFC", "q_BreakIn_EFC", "r", "r_LLI_t", "r_LLI_EFC"),
        ),
        "nca_grsi_sonymurata_2p5ah": _profile(
            "nca_grsi_sonymurata_2p5ah",
            "NCA-GrSi Sony-Murata 2.5 Ah",
            "nca_grsi_SonyMurata2p5Ah_2023",
            "NCA_GrSi_SonyMurata2p5Ah_Battery",
            "NCA/graphite-silicon",
            "cylindrical",
            2.5,
            {
                "cycling_temperature_c": [5, 35],
                "dod": [0.2, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 2.0,
                "max_c_rate_discharge": 10.0,
            },
            (
                "https://doi.org/10.1016/j.jpowsour.2022.232498",
                "https://doi.org/10.1016/j.jpowsour.2023.233947",
                "https://doi.org/10.1016/j.jpowsour.2023.233208",
            ),
            ("q", "q_t", "q_EFC"),
        ),
        "nmc111_gr_kokam_75ah": _profile(
            "nmc111_gr_kokam_75ah",
            "NMC111-Gr Kokam 75 Ah",
            "nmc111_gr_Kokam75Ah_2017",
            "Nmc111_Gr_Kokam75Ah_Battery",
            "NMC111/graphite",
            "pouch",
            75.0,
            {
                "cycling_temperature_c": [0, 45],
                "dod": [0.8, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 1.0,
                "max_c_rate_discharge": 1.0,
            },
            ("https://ieeexplore.ieee.org/iel7/7951530/7962914/07963578.pdf",),
            ("q", "q_LLI", "q_LLI_t", "q_LLI_EFC", "q_LAM", "r", "r_LLI", "r_LLI_t", "r_LLI_EFC", "r_LAM"),
        ),
        "nmc622_gr_denso_50ah": _profile(
            "nmc622_gr_denso_50ah",
            "NMC622-Gr DENSO 50 Ah",
            "nmc622_gr_DENSO50Ah_2021",
            "Nmc622_Gr_DENSO50Ah_Battery",
            "NMC622/graphite",
            "not stated",
            50.0,
            {
                "cycling_temperature_c": [10, 60],
                "dod": [0.1, 1.0],
                "soc": [0, 1],
                "max_c_rate_charge": 1.0,
                "max_c_rate_discharge": 1.0,
            },
            ("https://doi.org/10.1149/1945-7111/ac2ebd",),
            ("q", "q_t", "q_EFC", "q_BreakIn"),
        ),
    }
)

CORE_BLAST_MODEL_KEYS = tuple(key for key, profile in BATTERY_MODEL_REGISTRY.items() if profile.release_phase == "core")


def get_battery_model_profile(key: str) -> BatteryModelProfile:
    """Return one stable battery-model profile by key."""
    try:
        return BATTERY_MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(BATTERY_MODEL_REGISTRY)
        raise KeyError(f"Unknown battery model {key!r}. Available: {available}") from exc


def list_battery_models(*, enabled_only: bool = False) -> list[dict[str, Any]]:
    """Return JSON-serializable discovery metadata for BLAST battery models."""
    return [
        profile.as_dict()
        for profile in BATTERY_MODEL_REGISTRY.values()
        if not enabled_only or profile.release_phase == "core"
    ]


def apply_battery_profile_defaults(
    global_defaults: Mapping[str, Any], user_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve configuration as user values > profile defaults > globals."""
    model_key = user_config.get("blast_model")
    profile = BATTERY_MODEL_REGISTRY.get(str(model_key)) if model_key is not None else None
    profile_defaults = profile.operating_defaults if profile is not None else {}
    return merge_battery_config_layers(global_defaults, profile_defaults, user_config)


def merge_battery_config_layers(
    global_defaults: Mapping[str, Any], profile_defaults: Mapping[str, Any], user_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge the documented config precedence without mutating any input."""
    return {**global_defaults, **profile_defaults, **user_config}
