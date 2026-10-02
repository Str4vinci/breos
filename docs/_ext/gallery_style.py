"""How the example gallery draws its figures.

``reset_style`` runs before every gallery page and applies
``gallery.mplstyle``. ``scrape`` is sphinx-gallery's matplotlib scraper with
one change: before a figure is saved, tick labels group digits by thousands
with a space from five digits up (``5000``, ``12 345``), the way the
documentation writes numbers everywhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gallery_results import group_digits

STYLE = Path(__file__).with_name("gallery.mplstyle")


def reset_style(gallery_conf: dict[str, Any], fname: str) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcdefaults()
    plt.style.use(STYLE)


def _grouped_formatter(base: type) -> type:
    class Grouped(base):  # type: ignore[misc, valid-type]
        def __call__(self, x: float, pos: int | None = None) -> str:
            return group_digits(super().__call__(x, pos), sep=" ")

    Grouped.__name__ = f"Grouped{base.__name__}"
    return Grouped


def _group_ticks() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    grouped = _grouped_formatter(ScalarFormatter)
    for number in plt.get_fignums():
        for ax in plt.figure(number).axes:
            for axis in (ax.xaxis, ax.yaxis):
                formatter = axis.get_major_formatter()
                if type(formatter) is ScalarFormatter:
                    formatter.__class__ = grouped


def scrape(block: Any, block_vars: dict[str, Any], gallery_conf: dict[str, Any], **kwargs: Any) -> str:
    from sphinx_gallery.scrapers import matplotlib_scraper

    _group_ticks()
    return matplotlib_scraper(block, block_vars, gallery_conf, **kwargs)
