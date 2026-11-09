"""Shared chart style for interpretant visualisations.

Adapted from the ai-labor-exposure-replication project style:
light background, Open Sans, minimal spines, editorial palette.
"""

from __future__ import annotations

import pathlib

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FONTS_DIR = pathlib.Path(__file__).parent / "fonts"

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

C_BG     = "#F7FAFC"   # figure / axes background
C_DARK   = "#171923"   # titles, primary text
C_MID    = "#4A5568"   # axis labels, secondary text
C_DIM    = "#718096"   # footnotes, tertiary text
C_GRID   = "#CBD5E0"   # grid lines
C_SPINE  = "#A0AEC0"   # visible spines

# Word accent colours — work on light background
WORD_COLORS: dict[str, str] = {
    "network":        "#2B6CB0",   # blue
    "learning":       "#C05621",   # orange
    "agent":          "#C53030",   # red
    "representation": "#276749",   # green
    "consciousness":  "#6B46C1",   # purple
    "inference":      "#2C7A7B",   # teal
    "information":    "#744210",   # amber-dark
    "perception":     "#97266D",   # pink
    "affordance":     "#2F855A",   # green-light
    "intentionality": "#2A4365",   # blue-dark
}

DEFAULT_WORD_COLOR = "#4A5568"


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def load_fonts() -> None:
    for name in ("OpenSans-Regular.ttf", "OpenSans-SemiBold.ttf",
                 "OpenSans-Bold.ttf", "OpenSans-Light.ttf"):
        path = FONTS_DIR / name
        if path.exists():
            fm.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = "Open Sans"


def fp(variant: str = "Regular", size: float = 10) -> fm.FontProperties:
    """Return a FontProperties for the given Open Sans variant and size."""
    fname = FONTS_DIR / f"OpenSans-{variant}.ttf"
    if fname.exists():
        return fm.FontProperties(fname=str(fname), size=size)
    return fm.FontProperties(size=size)


# ---------------------------------------------------------------------------
# Axes styling helpers
# ---------------------------------------------------------------------------

def style_axes(ax: Axes, grid_axis: str = "both") -> None:
    """Apply standard interpretant style to an axes."""
    ax.set_facecolor(C_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C_SPINE)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.tick_params(colors=C_MID, labelsize=8.5)
    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color=C_GRID, linestyle="--", lw=0.5, alpha=0.8)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color=C_GRID, linestyle="--", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def style_axes_off(ax: Axes) -> None:
    """Background only, no spines or ticks — for text/table panels."""
    ax.set_facecolor(C_BG)
    ax.axis("off")


def add_title(ax: Axes, title: str, subtitle: str = "") -> None:
    """Add bold title + optional regular subtitle above axes."""
    ax.set_title(title, color=C_DARK, fontproperties=fp("Bold", 12),
                 pad=10, loc="left")
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.01), xycoords="axes fraction",
                    color=C_MID, fontproperties=fp("Regular", 9),
                    va="bottom")


def add_footnote(fig: Figure, text: str) -> None:
    """Add a small footnote at the bottom-left of the figure."""
    fig.text(0.06, 0.015, text, color=C_DIM,
             fontproperties=fp("Regular", 8), va="bottom")


def new_figure(
    figsize: tuple[float, float] = (16, 10),
    title: str = "",
    subtitle: str = "",
) -> Figure:
    """Create a new figure with the interpretant light style applied."""
    load_fonts()
    fig = plt.figure(figsize=figsize, facecolor=C_BG)
    if title:
        fig.text(0.06, 0.97, title, color=C_DARK,
                 fontproperties=fp("Bold", 15), va="top")
    if subtitle:
        fig.text(0.06, 0.935, subtitle, color=C_MID,
                 fontproperties=fp("Regular", 10), va="top")
    return fig
