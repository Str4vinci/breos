"""Battery degradation engines, vendored models, and public discovery."""

from breos.degradation.profiles import (
    BATTERY_MODEL_REGISTRY,
    BatteryModelProfile,
    get_battery_model_profile,
    list_battery_models,
)

__all__ = [
    "BATTERY_MODEL_REGISTRY",
    "BatteryModelProfile",
    "get_battery_model_profile",
    "list_battery_models",
]
