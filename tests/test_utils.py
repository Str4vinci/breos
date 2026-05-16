"""Tests for breos.utils."""

import pytest

from breos.utils import safe_path_slug


@pytest.mark.parametrize(
    "name,expected",
    [
        ("porto", "porto"),
        ("porto_2024", "porto_2024"),
        ("porto-de", "porto-de"),
        ("PORTO", "porto"),
        ("Berlin", "berlin"),
        ("a", "a"),
        ("a" * 64, "a" * 64),
    ],
)
def test_safe_path_slug_accepts_valid(name, expected):
    assert safe_path_slug(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "",
        "../etc/passwd",
        "/abs/path",
        "porto/sub",
        r"porto\sub",
        "porto..bad",
        ".hidden",
        "_leading",
        "-leading",
        "porto bad",
        "porto.bad",
        "porto\x00null",
        "a" * 65,
    ],
)
def test_safe_path_slug_rejects_unsafe(name):
    with pytest.raises(ValueError):
        safe_path_slug(name)


def test_safe_path_slug_rejects_non_string():
    with pytest.raises(TypeError):
        safe_path_slug(123)
