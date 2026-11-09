"""Drift visualisations from real TWEC-computed data.

Run:  uv run python scripts/plot_drift.py
Outputs: plots/drift.png
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from chart_style import (
    C_BG, C_DARK, C_DIM, C_GRID, C_MID, C_SPINE,
    WORD_COLORS, DEFAULT_WORD_COLOR,
    add_footnote, fp, load_fonts, new_figure, style_axes, style_axes_off,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_parquet("data/processed/drift.parquet")
DECADES = sorted(df["decade"].unique().tolist())

# Ordered by pattern: capture → suspension → contestation → other
# so the three groups have visually distinct textures in the heatmap
HEATMAP_WORDS = [
    # capture + stabilisation
    "network", "learning", "classification", "recognition", "inference",
    "adaptation", "integration",
    # permanent suspension
    "agent", "agency", "model",
    # continuous contestation
    "consciousness", "representation", "intentionality", "perception",
    "experience", "embodiment", "memory", "mind",
    # broadly contested / cross-field
    "information", "signal", "feedback", "complexity", "symbol",
    "reasoning", "affordance", "interface", "form", "image", "intelligence", "evolution",
]
available = set(df["word"].unique())
HEATMAP_WORDS = [w for w in HEATMAP_WORDS if w in available]

pivot = df[df["word"].isin(HEATMAP_WORDS)].pivot_table(
    index="word", columns="decade", values="cosine_drift_corrected", aggfunc="first"
)
pivot = pivot.reindex(index=HEATMAP_WORDS)

LINE_WORDS = {
    "network":        ("#2B6CB0", 2.0),
    "representation": ("#276749", 1.8),
    "information":    ("#2C7A7B", 1.6),
    "consciousness":  ("#6B46C1", 1.6),
    "inference":      ("#C05621", 1.6),
    "recognition":    ("#C53030", 1.8),
    "intentionality": ("#A0AEC0", 1.2),
}
LINE_WORDS = {w: v for w, v in LINE_WORDS.items() if w in available}

# ---------------------------------------------------------------------------
# PCA trajectories
# ---------------------------------------------------------------------------

kv_dir = pathlib.Path("models/twec")
traj_words = ["network", "representation", "consciousness", "inference"]
traj_words = [w for w in traj_words if w in available]

decade_vecs: dict[int, dict[str, np.ndarray]] = {}
for kv_path in sorted(kv_dir.glob("*.kv")):
    try:
        decade = int(kv_path.stem)
    except ValueError:
        continue
    kv = KeyedVectors.load(str(kv_path))
    decade_vecs[decade] = {w: kv[w] for w in traj_words if w in kv}

traj_decades = sorted(decade_vecs.keys())
valid_decades = [d for d in traj_decades if all(w in decade_vecs[d] for w in traj_words)]

pca = None
trajectories: dict[str, np.ndarray] = {}
if valid_decades and traj_words:
    all_vecs = np.vstack([[decade_vecs[d][w] for w in traj_words] for d in valid_decades])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_vecs)
    for w in traj_words:
        word_vecs = np.array([decade_vecs[d][w] for d in valid_decades])
        trajectories[w] = pca.transform(word_vecs)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

load_fonts()
fig = plt.figure(figsize=(20, 15), facecolor=C_BG)
fig.text(0.06, 0.97, "interpretant", color=C_DARK,
         fontproperties=fp("Bold", 16), va="top")
fig.text(0.06, 0.942, "Semantic drift across 200 years  ·  TWEC compass alignment  ·  frequency-corrected cosine distance",
         color=C_MID, fontproperties=fp("Regular", 10), va="top")

gs = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.30,
              left=0.07, right=0.97, top=0.86, bottom=0.10)

# ---------------------------------------------------------------------------
# Panel 1: Heatmap
# ---------------------------------------------------------------------------

ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(C_BG)

matrix = pivot.values.astype(float)
n_words, n_decades = matrix.shape

cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=0.45)
im = ax1.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

ax1.set_xticks(range(n_decades))
ax1.set_xticklabels([f"{int(d)}s" for d in pivot.columns],
                    rotation=45, ha="right", fontsize=8, color=C_MID)
ax1.set_yticks(range(n_words))
ax1.set_yticklabels(pivot.index.tolist(), fontsize=8.5, color=C_DARK)
ax1.set_title("Cosine drift (frequency-corrected) — consecutive-decade transitions",
              color=C_DARK, fontproperties=fp("SemiBold", 11), pad=10, loc="left")

cbar = fig.colorbar(im, ax=ax1, fraction=0.015, pad=0.01)
cbar.ax.tick_params(labelsize=8, colors=C_MID)
cbar.outline.set_edgecolor(C_GRID)
cbar.set_label("cosine distance (corrected)", color=C_DIM, fontsize=8)

ax1.tick_params(axis="both", colors=C_MID)
for spine in ax1.spines.values():
    spine.set_edgecolor(C_GRID)
    spine.set_linewidth(0.6)

col_idx = {int(d): i for i, d in enumerate(pivot.columns)}
row_idx = {w: i for i, w in enumerate(pivot.index)}

n_heatmap_rows = len(pivot.index.tolist())
_below_y = n_heatmap_rows + 1.8
# (word, decade, label, text_y, va)
# Stagger vertically when labels share the same above/below zone and are within 2 cols
annotations = [
    ("network",        1960, "1960s transition",   -2.8,      "bottom"),
    ("representation", 1970, "cog-sci crystallise", -5.2,     "bottom"),  # higher row avoids overlap
    ("information",    1970, "info-theory peak",    _below_y + 1.5, "top"),
    ("adaptation",     2000, "ML/bio peak",         -2.8,     "bottom"),
]
for word, decade, label, text_y, va in annotations:
    if word in row_idx and decade in col_idx:
        x, y = col_idx[decade], row_idx[word]
        ax1.annotate(
            label, xy=(x, y), xytext=(x, text_y),
            fontproperties=fp("SemiBold", 7.5), color=C_DARK,
            ha="center", va=va, annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color=C_MID, lw=0.9),
            bbox=dict(facecolor="white", edgecolor=C_SPINE, alpha=0.95,
                      boxstyle="round,pad=0.35", lw=0.7),
        )

for label in ax1.get_yticklabels():
    label.set_fontproperties(fp("Regular", 8.5))
for label in ax1.get_xticklabels():
    label.set_fontproperties(fp("Regular", 8))

# ---------------------------------------------------------------------------
# Panel 2: Line chart
# ---------------------------------------------------------------------------

ax2 = fig.add_subplot(gs[1, 0])
style_axes(ax2, grid_axis="y")

label_offsets: dict[str, float] = {
    "network":        0.005, "representation":  0.010, "information":  -0.010,
    "consciousness":  0.005, "inference":       -0.010, "recognition":   0.000,
    "intentionality": 0.000,
}

for word, (color, lw) in LINE_WORDS.items():
    series = df[df["word"] == word].sort_values("decade")
    x = range(len(series))
    y = series["cosine_drift_corrected"].values
    ax2.plot(x, y, color=color, linewidth=lw, alpha=0.9)
    if len(y):
        ax2.annotate(f" {word}", xy=(len(series) - 1, float(y[-1]) + label_offsets.get(word, 0)),
                     xycoords="data", color=color, fontproperties=fp("SemiBold", 8),
                     va="center", annotation_clip=False)

tick_positions = list(range(len(DECADES)))
ax2.set_xticks(tick_positions[::2])
ax2.set_xticklabels([f"{int(DECADES[i])}s" for i in tick_positions[::2]],
                    rotation=45, ha="right", fontsize=8, color=C_MID)
ax2.set_ylabel("cosine distance (corrected)", color=C_MID, fontsize=9,
               fontproperties=fp("Regular", 9))
ax2.set_title("Drift over time — selected words", color=C_DARK,
              fontproperties=fp("SemiBold", 10), pad=8, loc="left")
ax2.set_ylim(0, 0.55)

for label in ax2.get_xticklabels():
    label.set_fontproperties(fp("Regular", 8))
for label in ax2.get_yticklabels():
    label.set_fontproperties(fp("Regular", 8))

# ---------------------------------------------------------------------------
# Panel 3: PCA trajectory
# ---------------------------------------------------------------------------

ax3 = fig.add_subplot(gs[1, 1])
style_axes(ax3)

traj_colors = {w: WORD_COLORS.get(w, DEFAULT_WORD_COLOR) for w in traj_words}
cmap_decade = plt.cm.plasma

if trajectories and pca is not None:
    n_traj = len(valid_decades)
    norm_t = mcolors.Normalize(vmin=0, vmax=max(n_traj - 1, 1))
    for word, pts in trajectories.items():
        for i in range(len(pts) - 1):
            ax3.plot(pts[i:i+2, 0], pts[i:i+2, 1],
                     color=cmap_decade(norm_t(i)), linewidth=1.8, alpha=0.85)
        for i, decade in enumerate(valid_decades):
            marker = "D" if i in (0, n_traj - 1) else "o"
            size   = 60  if i in (0, n_traj - 1) else 22
            ax3.scatter(pts[i, 0], pts[i, 1],
                        color=cmap_decade(norm_t(i)), s=size, zorder=5, marker=marker)
            if decade in (1820, 1880, 1940, 1980, 2020):
                ax3.annotate(f"{decade}s", (pts[i, 0], pts[i, 1]),
                             textcoords="offset points", xytext=(5, 4),
                             fontsize=7, color=C_MID)
        # Per-word label offsets to avoid collision at start cluster
        _label_offsets: dict[str, tuple[float, float]] = {
            "network":        ( 0.00,  0.08),
            "representation": ( 0.00, -0.10),
            "consciousness":  (-0.08,  0.06),
            "inference":      ( 0.08,  0.06),
        }
        lx, ly = _label_offsets.get(word, (0.0, 0.08))
        ax3.text(pts[0, 0] + lx, pts[0, 1] + ly, word,
                 color=traj_colors[word], fontproperties=fp("SemiBold", 9),
                 ha="center")

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.plasma,
        norm=mcolors.Normalize(vmin=valid_decades[0], vmax=valid_decades[-1]),
    )
    sm.set_array([])
    cbar2 = fig.colorbar(sm, ax=ax3, fraction=0.04, pad=0.02)
    cbar2.ax.tick_params(labelsize=8, colors=C_MID)
    cbar2.outline.set_edgecolor(C_GRID)
    cbar2.set_label("decade", color=C_DIM, fontsize=8)

ax3.set_title(
    f"PCA trajectory  (PC1 {pca.explained_variance_ratio_[0]:.1%} / "
    f"PC2 {pca.explained_variance_ratio_[1]:.1%})" if pca else "PCA trajectory",
    color=C_DARK, fontproperties=fp("SemiBold", 10), pad=8, loc="left",
)
ax3.set_xlabel("PC1", color=C_MID, fontproperties=fp("Regular", 9))
ax3.set_ylabel("PC2", color=C_MID, fontproperties=fp("Regular", 9))

for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontproperties(fp("Regular", 8))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

add_footnote(fig, "interpretant  ·  TWEC (Di Carlo et al. 2019)  ·  274 books + PubMed + arXiv + ACL  ·  1820s–2020s")

out = pathlib.Path("plots/drift.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"saved → {out}")
plt.show()
