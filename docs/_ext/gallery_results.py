"""Load the stored results behind an example-gallery page.

``tools/regenerate_gallery_results.py`` writes each case to
``docs/examples/_results/<case>/``. A page loads them with :func:`load_case`
and states its numbers through :func:`say`, so the prose follows the stored
results when they are regenerated.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "examples" / "_results"
SOURCE = "https://github.com/Str4vinci/breos/blob/develop/"


class Html:
    """A block of HTML that sphinx-gallery shows as the output of a code cell."""

    def __init__(self, markup: str) -> None:
        self.markup = markup

    def _repr_html_(self) -> str:
        return self.markup


def _inline(text: str) -> str:
    """Escape ``text`` and turn ``code`` and **bold** spans into HTML."""
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


def say(*paragraphs: str) -> Html:
    """Paragraphs of prose built from stored numbers."""
    return Html("".join(f"<p>{_inline(paragraph)}</p>" for paragraph in paragraphs))


def bullets(*items: str) -> Html:
    """A bulleted list built from stored numbers."""
    return Html("<ul>" + "".join(f"<li>{_inline(item)}</li>" for item in items) + "</ul>")


GROUPED = re.compile(r"(?<![\d.])\d{5,}")


def group_digits(text: str, sep: str = "\u00a0") -> str:
    """``text`` with every run of five or more integer digits grouped by thousands: 5000, 12 345."""
    return GROUPED.sub(lambda match: f"{int(match.group()):,}".replace(",", sep), text)


def number(value: float, decimals: int = 0) -> str:
    """``value`` with ``decimals`` places and the documentation's digit grouping."""
    return number_text(f"{value:.{decimals}f}")


def table(frame: pd.DataFrame, **formats: str) -> Html:
    """``frame`` as a plain HTML table; ``formats`` maps a column to a format spec such as ``".0f"``.

    Numbers get the documentation's digit grouping (5000, 12 345) whatever the spec says.
    """
    shown = frame.copy()
    for column, spec in formats.items():
        spec = spec.replace(",", "")
        shown[column] = [("" if pd.isna(value) else number_text(format(value, spec))) for value in shown[column]]
    markup = shown.to_html(index=False, border=0, classes="table", na_rep="", escape=True)
    return Html(markup)


@dataclass
class StoredCase:
    """One case's stored files and manifest."""

    name: str

    @property
    def directory(self) -> Path:
        return RESULTS / self.name

    @cached_property
    def manifest(self) -> dict[str, Any]:
        return self.json("manifest.json")

    def json(self, filename: str) -> Any:
        return json.loads((self.directory / filename).read_text(encoding="utf-8"))

    def csv(self, filename: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(self.directory / filename, **kwargs)

    def week(self, filename: str, timezone: str) -> pd.DataFrame:
        """A stored week with its ``Datetime`` column on the local clock."""
        frame = self.csv(filename)
        frame["Datetime"] = pd.to_datetime(frame["Datetime"], utc=True).dt.tz_convert(timezone)
        return frame

    def stamp(self) -> Html:
        """Which BREOS made the stored results, from which inputs, and their sources."""
        manifest = self.manifest
        configs = ", ".join(f'<a href="{SOURCE}{path}"><code>{path}</code></a>' for path in manifest["configs"])
        weather = "".join(
            f"<li><code>{html.escape(path)}</code>: {html.escape(entry['attribution'])}</li>"
            for path, entry in manifest["weather"].items()
        )
        return Html(
            '<div class="admonition note"><p class="admonition-title">Stored results</p>'
            f"<p>Computed by BREOS {html.escape(manifest['breos_version'])} "
            f"(<code>{html.escape(manifest['git_describe'])}</code>) on {manifest['generated_utc'][:10]} "
            f"in {manifest['runtime_s']:g} s, from {configs or 'the settings shown on this page'}. "
            "The docs build only loads and plots them. "
            f"Regenerate with <code>python tools/regenerate_gallery_results.py {self.name}</code>.</p>"
            f"<p>Weather:</p><ul>{weather}</ul></div>"
        )


def load_case(name: str) -> StoredCase:
    """The stored results of gallery case ``name``."""
    case = StoredCase(name)
    if not (case.directory / "manifest.json").exists():
        raise FileNotFoundError(
            f"No stored results for {name!r}; run python tools/regenerate_gallery_results.py {name}"
        )
    return case


def number_text(text: str) -> str:
    """Formatted number ``text`` with digit grouping and a true minus sign."""
    return group_digits(text).replace("-", "−")


def money(value: float, currency: str = "EUR") -> str:
    """``value`` as whole currency units with the documentation's digit grouping, sign first."""
    sign = "−" if round(value) < 0 else ""
    return f"{sign}{group_digits(f'{abs(value):.0f}')} {currency}"
