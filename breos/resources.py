"""Access helpers for packaged BREOS runtime data."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

CONFIG_PACKAGE = "breos.data.configs"
RLP_PACKAGE = "breos.data.rlp"


def load_config_json(name: str) -> dict[str, Any]:
    """Load a packaged JSON config file."""
    resource = files(CONFIG_PACKAGE).joinpath(name)
    return json.loads(resource.read_text(encoding="utf-8"))


def rlp_resource(name: str):
    """Return a traversable packaged load-profile resource."""
    return files(RLP_PACKAGE).joinpath(name)
