"""Schemas for the nested tables of an App configuration (#181).

A top-level App key is described by ``breos.app_config.AppConfigField``. A
key whose value is a table (``costs``, ``battery_indoor_model``, each entry
of ``pv_arrays``, and the 0.7 ``[tariff]`` and ``[smart_charging]`` tables)
is described here instead: which keys it allows, how each value is checked,
which keys it requires, and a hook for rules that span several keys.

A checker is a function ``check(value, where) -> value``. It raises
``TypeError`` or ``ValueError`` naming ``where`` (the dotted key, such as
``costs.storage_cost_per_kwh``) and returns the value, normalised where that
makes sense (an enum lowercased, a number as ``float``). :meth:`TableSpec.validate`
returns the checked table and leaves the input unchanged; a caller decides
whether to keep the normalised values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Callable, Iterable, Mapping

Checker = Callable[[Any, str], Any]


def _format_keys(where: str, keys: Iterable[str]) -> str:
    return ", ".join(f"{where}.{key}" for key in keys)


@dataclass(frozen=True)
class TableSpec:
    """The keys one nested config table allows, and how each is checked.

    Args:
        name: The table's name in messages, such as ``"costs"``.
        keys: A checker per allowed key.
        required: Keys the table must set.
        check: Called with the checked table and its name once every key has
            passed, for rules that span keys (``floor_c <= ceiling_c``).
        kind: How the expected type is named when the value is not a mapping.
        docs: A description per allowed key, for the generated configuration
            key reference.
    """

    name: str
    keys: Mapping[str, Checker]
    required: frozenset[str] = field(default_factory=frozenset)
    check: Callable[[dict[str, Any], str], None] | None = None
    kind: str = "table/dict"
    docs: Mapping[str, str] = field(default_factory=dict)

    def validate(self, value: Any, where: str | None = None) -> dict[str, Any]:
        """Check a table and return it with each value as its checker returned it.

        ``where`` replaces the table's name in messages, for a table that sits
        inside a list (``pv_arrays[2]``).

        Raises:
            TypeError: If the value is not a mapping, or a key's value has the
                wrong type.
            ValueError: If a key is unknown or missing, or a value is out of
                range.
        """
        where = where or self.name
        if not isinstance(value, Mapping):
            raise TypeError(f"'{where}' must be a {self.kind}")
        unknown = sorted(str(key) for key in value if key not in self.keys)
        if unknown:
            if len(unknown) == 1:
                unknown_text = f"Unknown key '{where}.{unknown[0]}'"
            else:
                unknown_text = "Unknown keys " + ", ".join(f"'{where}.{key}'" for key in unknown)
            raise ValueError(f"{unknown_text}. Available: {_format_keys(where, sorted(self.keys))}")
        missing = sorted(key for key in self.required if key not in value)
        if missing:
            raise ValueError(f"'{where}' needs {_format_keys(where, missing)}")
        checked = {key: self.keys[key](item, f"{where}.{key}") for key, item in value.items()}
        if self.check is not None:
            self.check(checked, where)
        return checked


def anything(value: Any, where: str) -> Any:
    """Accept any value; for keys another validator checks."""
    return value


def number(
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    min_exclusive: bool = False,
    max_exclusive: bool = False,
    allow_none: bool = False,
) -> Checker:
    """A finite real number, optionally within a range; returned as ``float``."""

    def check(value: Any, where: str) -> float | None:
        if value is None and allow_none:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"'{where}' must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"'{where}' must be a finite number")
        low_ok = minimum is None or (result > minimum if min_exclusive else result >= minimum)
        high_ok = maximum is None or (result < maximum if max_exclusive else result <= maximum)
        if not (low_ok and high_ok):
            raise ValueError(f"'{where}' must be {_range_text(minimum, maximum, min_exclusive, max_exclusive)}")
        return result

    return check


def integer(*, minimum: int | None = None) -> Checker:
    """An integer, optionally at least ``minimum``; returned as ``int``.

    A bool is refused, and so is a float even when it is integral (``2.0``).
    """

    def check(value: Any, where: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"'{where}' must be an integer")
        result = int(value)
        if minimum is not None and result < minimum:
            raise ValueError(f"'{where}' must be >= {minimum}")
        return result

    return check


def _range_text(minimum: float | None, maximum: float | None, min_exclusive: bool, max_exclusive: bool) -> str:
    if minimum is not None and maximum is not None:
        bounds = []
        if min_exclusive:
            bounds.append("exclusive of the lower bound")
        if max_exclusive:
            bounds.append("exclusive of the upper bound")
        suffix = f" ({', '.join(bounds)})" if bounds else ""
        return f"between {minimum:g} and {maximum:g}{suffix}"
    if minimum is not None:
        return f"{'>' if min_exclusive else '>='} {minimum:g}"
    return f"{'<' if max_exclusive else '<='} {maximum:g}"


def boolean(value: Any, where: str) -> bool:
    """A real ``bool``: a string such as ``"no"`` is truthy and would read as on."""
    if not isinstance(value, bool):
        raise TypeError(f"'{where}' must be true or false")
    return value


def choice(options: Iterable[str], *, case_insensitive: bool = True) -> Checker:
    """One of ``options``; returned lowercased when ``case_insensitive``."""
    allowed = tuple(options)
    lookup = {option.lower(): option for option in allowed} if case_insensitive else {o: o for o in allowed}

    def check(value: Any, where: str) -> str:
        key = value.strip().lower() if case_insensitive and isinstance(value, str) else value
        if not isinstance(value, str) or key not in lookup:
            raise ValueError(f"'{where}' must be one of: {', '.join(allowed)}; got {value!r}")
        return lookup[key]

    return check


def text(value: Any, where: str) -> str:
    """A non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"'{where}' must be a non-empty string")
    return value


def list_of(item: Checker, *, min_length: int = 0) -> Checker:
    """A list whose every entry passes ``item``; entries are named ``where[i]``."""

    def check(value: Any, where: str) -> list[Any]:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"'{where}' must be a list")
        if len(value) < min_length:
            raise ValueError(f"'{where}' needs at least {min_length} entr{'y' if min_length == 1 else 'ies'}")
        return [item(entry, f"{where}[{index}]") for index, entry in enumerate(value)]

    return check


@dataclass(frozen=True)
class MappingOf:
    """The checker :func:`mapping_of` returns.

    It is a class rather than a closure so that a caller can tell a free-form
    mapping from a scalar key: ``breos sweep`` allows one more dotted level
    below it (``tariff.import_prices.P1``).
    """

    key: Checker
    value: Checker

    def __call__(self, table: Any, where: str) -> dict[Any, Any]:
        if not isinstance(table, Mapping):
            raise TypeError(f"'{where}' must be a table/dict")
        return {
            self.key(name, f"{where} key {name!r}"): self.value(item, f"{where}.{name}") for name, item in table.items()
        }


def mapping_of(key: Checker, value: Checker) -> MappingOf:
    """A mapping with free-form keys, such as a tariff's period-to-price map."""
    return MappingOf(key, value)


def table(spec: TableSpec) -> Checker:
    """A nested table checked against ``spec``."""
    return lambda value, where: spec.validate(value, where)
