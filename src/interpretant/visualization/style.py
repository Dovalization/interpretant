"""Shared matplotlib style for interpretant's figures.

Light surface, Open Sans, hairline grid, and a two-slot categorical palette validated for
colour-vision deficiency (blue for philosophy, orange for the field it is compared with).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")

FONTS_DIR = Path(__file__).parent / "fonts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ("#2a78d6", "#eb6834")
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def load_fonts() -> None:
    """Register the bundled Open Sans files and make them the default family."""
    for path in FONTS_DIR.glob("OpenSans-*.ttf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = "Open Sans"


WEIGHTS = {"Light": 300, "Regular": 400, "SemiBold": 600, "Bold": 700}


def font(variant: str = "Regular", size: float = 10) -> font_manager.FontProperties:
    """Return Open Sans at ``size`` in the given weight, falling back to DejaVu Sans for
    glyphs Open Sans lacks (such as arrows)."""
    return font_manager.FontProperties(
        family=["Open Sans", "DejaVu Sans"], weight=WEIGHTS[variant], size=size
    )


def style_axes(axes: Axes, grid_axis: Literal["x", "y", "both"] | None = "y") -> None:
    """Recessive chrome: surface background, left and bottom hairlines, one grid axis."""
    axes.set_facecolor(SURFACE)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["left", "bottom"]].set_color(AXIS)
    axes.spines[["left", "bottom"]].set_linewidth(0.8)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=3)
    if grid_axis is not None:
        axes.grid(axis=grid_axis, color=GRIDLINE, linewidth=0.6)
    axes.set_axisbelow(True)


def new_figure(figsize: tuple[float, float], title: str, subtitle: str = "") -> Figure:
    """Create a figure on the light surface with a left-aligned title and subtitle."""
    load_fonts()
    figure = plt.figure(figsize=figsize, facecolor=SURFACE)
    height = figsize[1]
    figure.text(
        0.04, 1 - 0.25 / height, title, color=INK, fontproperties=font("Bold", 14), va="top"
    )
    if subtitle:
        figure.text(
            0.04,
            1 - 0.58 / height,
            subtitle,
            color=INK_SECONDARY,
            fontproperties=font("Regular", 10),
            va="top",
        )
    return figure


def plot_top(height: float, header: float = 1.05) -> float:
    """Figure fraction where the plot area ends, leaving ``header`` inches for the titles."""
    return 1 - header / height


def add_footnote(figure: Figure, text: str) -> None:
    """Add a small source or method note at the bottom-left, wrapped to the figure's width."""
    wrapped = textwrap.fill(text, width=int(figure.get_figwidth() * 16))
    figure.text(
        0.04, 0.02, wrapped, color=INK_MUTED, fontproperties=font("Regular", 8), va="bottom"
    )
