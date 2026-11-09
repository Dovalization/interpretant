"""Deep-dive visualisation for a single word.

Run:  uv run python scripts/plot_word.py --word network
Outputs: plots/word_<word>.png
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--word", default="network")
parser.add_argument("--topn", type=int, default=8, help="Neighbours to show per decade")
parser.add_argument(
    "--neighbor-decades",
    nargs="+",
    type=int,
    default=[1850, 1880, 1960, 1980, 2000, 2020],
)
args = parser.parse_args()
WORD = args.word
TOPN = args.topn
NEIGHBOR_DECADES = args.neighbor_decades

# ---------------------------------------------------------------------------
# Load drift data
# ---------------------------------------------------------------------------

df = pd.read_parquet("data/processed/drift.parquet")
word_df = df[df["word"] == WORD].sort_values("decade").reset_index(drop=True)

if word_df.empty:
    raise SystemExit(f"Word '{WORD}' not found in drift.parquet")

DECADES = word_df["decade"].tolist()

# ---------------------------------------------------------------------------
# Load KeyedVectors for neighbours + PCA
# ---------------------------------------------------------------------------

kv_dir = pathlib.Path("models/twec")
decade_kv: dict[int, KeyedVectors] = {}
for path in sorted(kv_dir.glob("*.kv")):
    try:
        d = int(path.stem)
    except ValueError:
        continue
    decade_kv[d] = KeyedVectors.load(str(path))

# Neighbours at selected decades
neighbours: dict[int, list[tuple[str, float]]] = {}
for d in NEIGHBOR_DECADES:
    kv = decade_kv.get(d)
    if kv and WORD in kv:
        neighbours[d] = kv.most_similar(WORD, topn=TOPN)

# PCA trajectory — collect vectors across all decades where word is present
traj_decades = [d for d in sorted(decade_kv) if WORD in decade_kv[d]]
traj_vecs = np.array([decade_kv[d][WORD] for d in traj_decades])

# Fit PCA on these vectors + a handful of contextually related words for grounding
context_words = ["neural", "graph", "connection", "system", "circuit",
                 "social", "biology", "computation", "architecture", "node"]
context_words = [w for w in context_words
                 if any(w in decade_kv[d] for d in traj_decades)]

all_fit_vecs = list(traj_vecs)
word_labels_fit: list[str] = []
for w in context_words:
    vecs = [decade_kv[d][w] for d in traj_decades if w in decade_kv[d]]
    if vecs:
        all_fit_vecs.append(np.mean(vecs, axis=0))
        word_labels_fit.append(w)

pca = PCA(n_components=2, random_state=42)
pca.fit(np.array(all_fit_vecs))
traj_2d = pca.transform(traj_vecs)
context_2d = pca.transform(np.array(all_fit_vecs[len(traj_vecs):]))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

ACCENT  = "#e8e8e8"
DIM     = "#888888"
GRID_C  = "#2a2a2a"
BG      = "#151515"
GOLD    = "#ffcc44"
TEAL    = "#4ecdc4"
PURPLE  = "#c77dff"

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0f0f0f")
gs = GridSpec(
    2, 3, figure=fig,
    hspace=0.45, wspace=0.35,
    left=0.06, right=0.97, top=0.91, bottom=0.09,
    width_ratios=[2, 1.4, 1.6],
)

# ---------------------------------------------------------------------------
# Panel 1 (top-left, spans 2 cols): dual-axis drift + neighbourhood shift
# ---------------------------------------------------------------------------

ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(BG)

x = np.arange(len(DECADES))
drift_vals = word_df["cosine_drift_corrected"].values
ns_vals    = word_df["neighborhood_shift"].values

color_drift = TEAL
color_ns    = GOLD

ax1.bar(x, drift_vals, color=color_drift, alpha=0.55, width=0.6, label="cosine drift (corrected)")
ax1.plot(x, drift_vals, color=color_drift, linewidth=1.8, alpha=0.9)

ax2_twin = ax1.twinx()
ax2_twin.plot(x, ns_vals, color=color_ns, linewidth=2.0, linestyle="--",
              marker="o", markersize=4, alpha=0.9, label="neighbourhood shift (Jaccard)")
ax2_twin.set_ylim(0, 1.15)
ax2_twin.set_ylabel("neighbourhood shift", color=color_ns, fontsize=9)
ax2_twin.tick_params(axis="y", colors=color_ns)
ax2_twin.yaxis.label.set_color(color_ns)
for spine in ax2_twin.spines.values():
    spine.set_edgecolor(GRID_C)

# Annotate the 1960 spike
spike_idx = DECADES.index(1960) if 1960 in DECADES else None
if spike_idx is not None:
    ax1.annotate(
        "1960s\ncritical shift",
        xy=(spike_idx, drift_vals[spike_idx]),
        xytext=(spike_idx - 2.5, drift_vals[spike_idx] + 0.06),
        fontsize=8, color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.0),
        bbox=dict(facecolor="#1a1a1a", edgecolor=GOLD, alpha=0.85,
                  boxstyle="round,pad=0.3", lw=0.6),
    )

ax1.set_xticks(x)
ax1.set_xticklabels([f"{int(d)}s" for d in DECADES], rotation=45, ha="right",
                    fontsize=8, color=ACCENT)
ax1.set_ylabel("cosine drift (corrected)", color=color_drift, fontsize=9)
ax1.yaxis.label.set_color(color_drift)
ax1.tick_params(axis="y", colors=color_drift)
ax1.set_title(f'"{WORD}" — drift & neighbourhood shift across decades',
              color=ACCENT, fontsize=11, pad=8)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
           facecolor="#1a1a1a", edgecolor=GRID_C, labelcolor=ACCENT, loc="upper left")

for spine in ax1.spines.values():
    spine.set_edgecolor(GRID_C)
ax1.set_ylim(bottom=0)

# ---------------------------------------------------------------------------
# Panel 2 (top-right): PCA trajectory
# ---------------------------------------------------------------------------

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(BG)

cmap_t = plt.cm.plasma
n = len(traj_decades)
for i in range(n - 1):
    t = i / max(n - 2, 1)
    ax3.plot(traj_2d[i:i+2, 0], traj_2d[i:i+2, 1],
             color=cmap_t(t), linewidth=2.0, alpha=0.9)

for i, d in enumerate(traj_decades):
    t = i / max(n - 1, 1)
    marker = "D" if d in (traj_decades[0], traj_decades[-1]) else "o"
    size   = 70  if d in (traj_decades[0], traj_decades[-1]) else 28
    ax3.scatter(traj_2d[i, 0], traj_2d[i, 1], color=cmap_t(t),
                s=size, zorder=5, marker=marker)
    if d in (1820, 1860, 1920, 1960, 1990, 2020):
        ax3.annotate(f"{d}s", (traj_2d[i, 0], traj_2d[i, 1]),
                     textcoords="offset points", xytext=(5, 4),
                     fontsize=7.5, color=cmap_t(t), alpha=0.95)

# Plot context word centroids as ghost markers
for j, (w, pt) in enumerate(zip(word_labels_fit, context_2d)):
    ax3.scatter(pt[0], pt[1], color="#444", s=20, zorder=3, marker="x")
    ax3.text(pt[0] + 0.01, pt[1] + 0.01, w, color="#666", fontsize=7, ha="left")

sm = plt.cm.ScalarMappable(cmap=cmap_t,
                            norm=mcolors.Normalize(vmin=traj_decades[0], vmax=traj_decades[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax3, fraction=0.05, pad=0.02)
cbar.ax.yaxis.set_tick_params(color=ACCENT)
cbar.outline.set_edgecolor(DIM)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=ACCENT, fontsize=7)
cbar.set_label("decade", color=DIM, fontsize=8)

ax3.set_title(
    f'PCA trajectory  (PC1 {pca.explained_variance_ratio_[0]:.1%} / PC2 {pca.explained_variance_ratio_[1]:.1%})',
    color=ACCENT, fontsize=9, pad=8,
)
ax3.set_xlabel("PC1", color=DIM, fontsize=8)
ax3.set_ylabel("PC2", color=DIM, fontsize=8)
ax3.tick_params(colors=DIM)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRID_C)

# ---------------------------------------------------------------------------
# Panel 3 (bottom, spans all cols): neighbour heatmap / table
# ---------------------------------------------------------------------------

ax4 = fig.add_subplot(gs[1, :])
ax4.set_facecolor(BG)
ax4.axis("off")

n_cols = len(neighbours)
if n_cols == 0:
    ax4.text(0.5, 0.5, "No neighbour data", color=DIM, ha="center", va="center",
             transform=ax4.transAxes)
else:
    col_w = 1.0 / n_cols
    decade_list = sorted(neighbours.keys())

    for ci, d in enumerate(decade_list):
        cx = ci * col_w + col_w * 0.5
        # Decade header
        ax4.text(cx, 0.95, f"{d}s", ha="center", va="top",
                 color=GOLD, fontsize=11, fontweight="bold",
                 transform=ax4.transAxes)
        # Divider line
        ax4.plot([ci * col_w, ci * col_w], [0, 1], color=GRID_C, lw=0.8,
                 transform=ax4.transAxes, clip_on=False)

        for ri, (w, score) in enumerate(neighbours[d]):
            y = 0.82 - ri * 0.085
            # Score bar (thin rectangle behind word)
            bar_w = score * col_w * 0.85
            bar_x = cx - col_w * 0.42
            rect = plt.Rectangle(
                (bar_x, y - 0.032), bar_w, 0.055,
                transform=ax4.transAxes,
                color=TEAL, alpha=0.18, zorder=1,
            )
            ax4.add_patch(rect)
            ax4.text(cx, y, w, ha="center", va="center",
                     color=ACCENT, fontsize=9, zorder=2,
                     transform=ax4.transAxes)
            ax4.text(cx + col_w * 0.38, y, f"{score:.2f}",
                     ha="right", va="center",
                     color=DIM, fontsize=7.5, zorder=2,
                     transform=ax4.transAxes)

    ax4.set_title(f'Top-{TOPN} nearest neighbours at selected decades',
                  color=ACCENT, fontsize=10, pad=6, loc="left")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

fig.suptitle(f'interpretant — "{WORD}" semantic trajectory  1820s–2020s',
             color=ACCENT, fontsize=13, y=0.97)

out = pathlib.Path(f"plots/word_{WORD}.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"saved → {out}")
plt.show()
