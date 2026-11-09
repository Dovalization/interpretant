"""Capture & stabilisation — three patterns of semantic capture.

Run:  uv run python scripts/plot_capture.py
Outputs: plots/capture.png
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from matplotlib.gridspec import GridSpec

from chart_style import (
    C_BG, C_DARK, C_DIM, C_GRID, C_MID, C_SPINE,
    DEFAULT_WORD_COLOR,
    add_footnote, fp, load_fonts, style_axes,
)

# ---------------------------------------------------------------------------
# Data & constants
# ---------------------------------------------------------------------------

df = pd.read_parquet("data/processed/drift.parquet")

PATTERNS = {
    "capture":    {"words": ["network", "learning"],                 "label": "capture + stabilisation"},
    "suspension": {"words": ["agent"],                               "label": "permanent suspension"},
    "continuous": {"words": ["representation", "consciousness"],     "label": "continuous contestation"},
}
PATTERN_COLORS = {
    "capture":    "#2B6CB0",
    "suspension": "#C53030",
    "continuous": "#6B46C1",
}
# Within a pattern, first word = solid, second = dashed
LINESTYLES = ["solid", "dashed"]

NEIGHBOR_DECADES = [1880, 1960, 1990, 2020]
TOPN = 5
kv_dir = pathlib.Path("models/twec")

all_decades = sorted(df["decade"].unique())
x_map = {d: i for i, d in enumerate(all_decades)}
stab_start = x_map[2000]
stab_end   = x_map[2020]
inf_x      = x_map[1960]
PATTERN_ORDER = ["capture", "suspension", "continuous"]

Y_MAX_DRIFT = 0.52
Y_MAX_NS    = 1.15


def load_neighbours(word: str, decade: int) -> list[str]:
    kv_path = kv_dir / f"{decade}.kv"
    if not kv_path.exists():
        return []
    kv = KeyedVectors.load(str(kv_path))
    return [w for w, _ in kv.most_similar(word, topn=TOPN)] if word in kv else []


def annotate_zone(ax: plt.Axes, pkey: str) -> None:
    ax.axvspan(stab_start - 0.5, stab_end + 0.5, color=C_GRID, alpha=0.12)
    ax.axvline(inf_x, color=C_SPINE, lw=1.0, linestyle="--", alpha=0.6)
    if pkey == "capture":
        ax.text(stab_start + 0.4, Y_MAX_DRIFT * 0.92, "stabilises",
                color=C_MID, fontproperties=fp("Regular", 7), ha="center")
        ax.text(inf_x + 0.2, Y_MAX_DRIFT * 0.92, "1960",
                color=C_MID, fontproperties=fp("Regular", 7))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

load_fonts()
fig = plt.figure(figsize=(20, 16), facecolor=C_BG)
fig.text(0.06, 0.97, "interpretant", color=C_DARK,
         fontproperties=fp("Bold", 16), va="top")
fig.text(0.06, 0.942,
         "Three patterns of semantic capture  ·  how contested vocabulary resolves (or doesn't)",
         color=C_MID, fontproperties=fp("Regular", 10), va="top")

gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.12,
              left=0.07, right=0.95, top=0.86, bottom=0.06,
              height_ratios=[1.4, 1.4, 1.1])

# ---------------------------------------------------------------------------
# Row 0: cosine drift — small multiples
# ---------------------------------------------------------------------------

for ci, pkey in enumerate(PATTERN_ORDER):
    ax = fig.add_subplot(gs[0, ci])
    style_axes(ax, grid_axis="y")
    pdata  = PATTERNS[pkey]
    color  = PATTERN_COLORS[pkey]

    for wi, word in enumerate(pdata["words"]):
        sub = df[df["word"] == word].sort_values("decade")
        xs  = [x_map[d] for d in sub["decade"]]
        ys  = sub["cosine_drift_corrected"].values
        ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.9,
                linestyle=LINESTYLES[wi % 2])
        ax.text(xs[-1] + 0.25, float(ys[-1]), f" {word}",
                color=color, fontproperties=fp("SemiBold", 8.5), va="center",
                clip_on=False)

    annotate_zone(ax, pkey)

    ax.set_xticks(list(x_map.values())[::2])
    ax.set_xticklabels([f"{int(all_decades[i])}s" for i in range(0, len(all_decades), 2)],
                       rotation=45, ha="right")
    ax.set_ylim(0, Y_MAX_DRIFT)
    ax.set_title(pdata["label"], color=color,
                 fontproperties=fp("Bold", 10), pad=6, loc="left")

    if ci == 0:
        ax.set_ylabel("cosine drift (corrected)", color=C_MID,
                      fontproperties=fp("Regular", 8.5))
    else:
        ax.set_yticklabels([])

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp("Regular", 7.5))
        label.set_color(C_MID)

# ---------------------------------------------------------------------------
# Row 1: neighbourhood shift — small multiples
# ---------------------------------------------------------------------------

for ci, pkey in enumerate(PATTERN_ORDER):
    ax = fig.add_subplot(gs[1, ci])
    style_axes(ax, grid_axis="y")
    pdata = PATTERNS[pkey]
    color = PATTERN_COLORS[pkey]

    for wi, word in enumerate(pdata["words"]):
        sub = df[df["word"] == word].sort_values("decade")
        xs  = [x_map[d] for d in sub["decade"]]
        ys  = sub["neighborhood_shift"].values
        ax.plot(xs, ys, color=color, linewidth=2.0,
                linestyle=LINESTYLES[wi % 2],
                marker="o", markersize=3.5, alpha=0.9)
        ax.text(xs[-1] + 0.25, float(ys[-1]), f" {word}",
                color=color, fontproperties=fp("SemiBold", 8.5), va="center",
                clip_on=False)

    ax.axvspan(stab_start - 0.5, stab_end + 0.5, color=C_GRID, alpha=0.12)
    ax.axvline(inf_x, color=C_SPINE, lw=1.0, linestyle="--", alpha=0.6)

    ax.set_xticks(list(x_map.values())[::2])
    ax.set_xticklabels([f"{int(all_decades[i])}s" for i in range(0, len(all_decades), 2)],
                       rotation=45, ha="right")
    ax.set_ylim(0, Y_MAX_NS)

    if ci == 0:
        ax.set_ylabel("neighbourhood shift (Jaccard)", color=C_MID,
                      fontproperties=fp("Regular", 8.5))
    else:
        ax.set_yticklabels([])

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp("Regular", 7.5))
        label.set_color(C_MID)

# ---------------------------------------------------------------------------
# Row 2: neighbour snapshots — one col per pattern
# ---------------------------------------------------------------------------

PANEL_BG = {
    "capture":    "#EBF8FF",
    "suspension": "#FFF5F5",
    "continuous": "#FAF5FF",
}

for ci, pkey in enumerate(PATTERN_ORDER):
    ax = fig.add_subplot(gs[2, ci])
    ax.set_facecolor(PANEL_BG[pkey])
    ax.axis("off")
    color = PATTERN_COLORS[pkey]
    pdata = PATTERNS[pkey]
    words = pdata["words"]

    n_cols    = len(NEIGHBOR_DECADES)
    col_w     = 1.0 / n_cols
    n_words   = len(words)
    N_SHOWN   = 4
    HDR       = 0.10
    WLH       = 0.09
    MAX_WORDS = 2                             # normalise all panels to 2-word height
    DATA_H    = (1.0 - HDR - MAX_WORDS * WLH) / MAX_WORDS
    BLOCK_H   = WLH + DATA_H
    top_offset = 0.0  # always align to top

    # Decade column headers
    for ci2, decade in enumerate(NEIGHBOR_DECADES):
        cx = ci2 * col_w + col_w * 0.5
        ax.text(cx, 0.99, f"{decade}s", ha="center", va="top",
                color=C_DARK, fontproperties=fp("SemiBold", 8),
                transform=ax.transAxes)
    ax.plot([0, 1], [1 - HDR, 1 - HDR], color=C_GRID, lw=0.7,
            transform=ax.transAxes)

    for ri, word in enumerate(words):
        block_top = (1.0 - HDR - top_offset) - ri * BLOCK_H
        label_bot = block_top - WLH
        data_bot  = label_bot - DATA_H

        # Shaded word-label strip
        rect = plt.Rectangle((0, label_bot), 1, WLH,
                              transform=ax.transAxes, clip_on=True,
                              facecolor=color, alpha=0.08, zorder=1)
        ax.add_patch(rect)
        ax.text(0.5, label_bot + WLH * 0.5, f'"{word}"',
                ha="center", va="center", color=color,
                fontproperties=fp("Bold", 8.5),
                transform=ax.transAxes, zorder=2)

        # Horizontal dividers
        ax.plot([0, 1], [label_bot, label_bot], color=C_GRID, lw=0.5,
                transform=ax.transAxes, zorder=2)
        if ri > 0:
            ax.plot([0, 1], [block_top, block_top], color=C_GRID, lw=0.5,
                    transform=ax.transAxes, zorder=2)

        # Vertical column dividers — only within data zone, not through label strip
        for ci2 in range(1, n_cols):
            ax.plot([ci2 * col_w, ci2 * col_w], [data_bot, label_bot],
                    color=C_GRID, lw=0.6, transform=ax.transAxes, zorder=0)

        # Neighbour words
        row_step = DATA_H / (N_SHOWN + 0.5)
        for ci2, decade in enumerate(NEIGHBOR_DECADES):
            cx = ci2 * col_w + col_w * 0.5
            neighbours = load_neighbours(word, decade)
            for ni, w in enumerate(neighbours[:N_SHOWN]):
                y = label_bot - (ni + 0.8) * row_step
                ax.text(cx, y, w, ha="center", va="center",
                        color=C_MID, fontproperties=fp("Regular", 7),
                        transform=ax.transAxes, zorder=2)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

add_footnote(fig, "interpretant  ·  TWEC alignment  ·  274 books + PubMed + arXiv + ACL  ·  1820s–2020s")

out = pathlib.Path("plots/capture.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"saved → {out}")
plt.show()
