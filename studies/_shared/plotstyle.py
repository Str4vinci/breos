"""Shared matplotlib style for the long-run studies (static PNG figures in REPORT.md).

Colours are the validated reference palette (light mode): categorical slots in fixed order,
one-hue blue ramp for magnitude, blue<->red with a grey midpoint for diverging data.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("breos_blue", SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "breos_div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a32a2a"]
)


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "font.family": "sans-serif",
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )
