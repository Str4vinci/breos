"""The defaults the shipped example configs state in their comments are the real ones.

``configs/examples/pv-plus-battery.toml`` and ``montecarlo.toml`` annotate
keys with their default in brackets, ``key = value  # [default] ...``. A
bracket holding a TOML value, or ``none`` for an unset key, must equal the
registry default; descriptive brackets such as ``[equator-facing]`` are not
checked. Every example validating cleanly is covered in ``test_cli.py``.
"""

import dataclasses
import re
import tomllib
from pathlib import Path

import pytest

from breos.app_config import DEFAULTS
from breos.montecarlo import MonteCarloSettings

EXAMPLES = Path(__file__).resolve().parents[1] / "configs" / "examples"
_ANNOTATED = re.compile(r"^#?\s*([a-z_]+)\s*=\s*[^#]+#\s*\[([^\]]+)\]")
_MONTECARLO_DEFAULTS = {
    field.name: field.default
    for field in dataclasses.fields(MonteCarloSettings)
    if field.default is not dataclasses.MISSING
}


def _annotated_defaults(path):
    """Return ``{key: stated default}`` for every bracket that holds a value."""
    stated = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ANNOTATED.match(line)
        if match is None:
            continue
        key, text = match.groups()
        if text == "none":
            stated[key] = None
            continue
        try:
            stated[key] = tomllib.loads(f"value = {text}")["value"]
        except tomllib.TOMLDecodeError:
            continue  # a description, not a value
    return stated


@pytest.mark.parametrize(
    ("name", "defaults", "minimum"),
    [("pv-plus-battery.toml", DEFAULTS, 20), ("montecarlo.toml", _MONTECARLO_DEFAULTS, 7)],
)
def test_example_comments_state_the_real_defaults(name, defaults, minimum):
    stated = _annotated_defaults(EXAMPLES / name)

    # A reworded comment style would otherwise check nothing.
    assert len(stated) >= minimum
    wrong = {
        key: (value, defaults.get(key, "<no default>"))
        for key, value in stated.items()
        if defaults.get(key, object()) != value
    }
    assert wrong == {}
