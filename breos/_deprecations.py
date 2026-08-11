"""Internal helpers for BREOS's staged public API removals."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, cast

_F = TypeVar("_F", bound=Callable[..., Any])


def deprecation_message(name: str, *, replacement: str | None = None) -> str:
    """Build the standard message for APIs scheduled for BREOS 0.6.0."""

    message = f"{name} is deprecated and will be removed in BREOS 0.6.0."
    if replacement:
        message += f" Use {replacement} instead."
    return message


def deprecated(*, name: str, replacement: str | None = None):
    """Warn when a deprecated function is called or class is instantiated."""

    message = deprecation_message(name, replacement=replacement)

    def decorate(obj):
        if isinstance(obj, type):
            original_init = obj.__init__

            @wraps(original_init)
            def warned_init(self, *args, **kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            obj.__init__ = warned_init
            obj.__breos_deprecated_removal__ = "0.6.0"
            return obj

        @wraps(obj)
        def warned_call(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        warned_call.__breos_deprecated_removal__ = "0.6.0"
        return cast(_F, warned_call)

    return decorate


def warn_deprecated(name: str, *, replacement: str | None = None, stacklevel: int = 2) -> None:
    """Warn for deprecated surfaces that are not represented by a callable."""

    warnings.warn(
        deprecation_message(name, replacement=replacement),
        DeprecationWarning,
        stacklevel=stacklevel,
    )
