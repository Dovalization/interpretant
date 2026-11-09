"""Mock drift visualisations — synthetic data matching expected empirical patterns.

Run:  uv run python scripts/plot_mock_drift.py
Outputs: plots/mock_drift.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

RNG = np.random.default_rng(42)
DECADES = list(range(1820, 2020, 10))          # 1820 … 2010
TRANSITIONS = [f"{d}s→{d+10}s" for d in DECADES[:-1]]  # 19 transitions

# ---------------------------------------------------------------------------
# Synthetic drift profiles  (cosine distance per consecutive-decade transition)
# Each value is a (mean, noise_std) pair; noise is added per-run.
# ---------------------------------------------------------------------------

def _profile(*segments: tuple[int, int, float, float]) -> np.ndarray:
    """Build a 19-element drift array from (start_idx, end_idx, mean, std) segments."""
    arr = np.zeros(19)
    for start, end, mean, std in segments:
        arr[start:end] = RNG.normal(mean, std, end - start).clip(0.02, 0.9)
    return arr


PROFILES: dict[str, np.ndarray] = {
    # AI/CS vocabulary — all show sharp 1980s–1990s shift
    "attention":      _profile((0, 18, 0.07, 0.02), (18, 19, 0.58, 0.01)),
    "network":        _profile((0, 15, 0.09, 0.03), (15, 17, 0.44, 0.03), (17, 19, 0.20, 0.03)),
    "agent":          _profile((0, 16, 0.10, 0.03), (16, 18, 0.36, 0.03), (18, 19, 0.28, 0.03)),
    "learning":       _profile((0, 14, 0.09, 0.02), (14, 17, 0.38, 0.03), (17, 19, 0.18, 0.03)),
    "model":          _profile((0, 12, 0.10, 0.03), (12, 17, 0.28, 0.03), (17, 19, 0.22, 0.03)),
    "reasoning":      _profile((0, 13, 0.08, 0.02), (13, 16, 0.30, 0.03), (16, 19, 0.15, 0.02)),
    # Philosophy of mind — earlier shifts, more stable post-1980
    "representation": _profile((0,  5, 0.10, 0.02), (5, 7, 0.27, 0.03),
                                (7, 14, 0.13, 0.02), (14, 17, 0.33, 0.03), (17, 19, 0.18, 0.02)),
    "intentionality": _profile((0, 19, 0.08, 0.02)),
    "consciousness":  _profile((0, 16, 0.09, 0.02), (16, 18, 0.33, 0.03), (18, 19, 0.22, 0.02)),
    "embodiment":     _profile((0, 15, 0.08, 0.02), (15, 18, 0.32, 0.03), (18, 19, 0.19, 0.02)),
    "perception":     _profile((0, 19, 0.11, 0.03)),
    "experience":     _profile((0, 19, 0.08, 0.02)),
    # Cybernetics / information theory — 1940s–1950s spike
    "information":    _profile((0, 12, 0.10, 0.02), (12, 14, 0.42, 0.03), (14, 19, 0.14, 0.02)),
    "feedback":       _profile((0, 12, 0.09, 0.02), (12, 14, 0.37, 0.03), (14, 19, 0.12, 0.02)),
    "signal":         _profile((0, 12, 0.10, 0.02), (12, 14, 0.36, 0.03), (14, 19, 0.13, 0.02)),
    "complexity":     _profile((0, 14, 0.09, 0.02), (14, 17, 0.27, 0.03), (17, 19, 0.15, 0.02)),
    "symbol":         _profile((0, 13, 0.09, 0.02), (13, 15, 0.26, 0.03), (15, 19, 0.14, 0.02)),
    # Design / HCI — 1970s–1980s
    "affordance":     _profile((0, 14, 0.06, 0.01), (14, 16, 0.32, 0.03), (16, 19, 0.16, 0.02)),
    "interface":      _profile((0, 15, 0.07, 0.02), (15, 17, 0.30, 0.03), (17, 19, 0.16, 0.02)),
    "user":           _profile((0, 15, 0.07, 0.02), (15, 17, 0.28, 0.03), (17, 19, 0.15, 0.02)),
    # Stable anchors
    "form":           _profile((0, 19, 0.09, 0.03)),
    "pattern":        _profile((0, 13, 0.09, 0.02), (13, 17, 0.18, 0.02), (17, 19, 0.14, 0.02)),
    "medium":         _profile((0, 14, 0.09, 0.02), (14, 16, 0.28, 0.03), (16, 19, 0.14, 0.02)),
    "behavior":       _profile((0, 11, 0.09, 0.02), (11, 13, 0.26, 0.03), (13, 19, 0.12, 0.02)),
    "image":          _profile((0, 14, 0.10, 0.02), (14, 16, 0.24, 0.03), (16, 19, 0.13, 0.02)),
    "mind":           _profile((0, 19, 0.09, 0.02)),
    "memory":         _profile((0, 14, 0.09, 0.02), (14, 16, 0.25, 0.03), (16, 19, 0.16, 0.02)),
    "simulation":     _profile((0, 14, 0.08, 0.02), (14, 17, 0.24, 0.03), (17, 19, 0.17, 0.02)),
    "intelligence":   _profile((0, 13, 0.09, 0.02), (13, 15, 0.33, 0.03), (15, 19, 0.16, 0.02)),
}

WORDS = list(PROFILES.keys())

# ---------------------------------------------------------------------------
# Simulated 2D trajectories for PCA plot
# ---------------------------------------------------------------------------

def make_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    inflection_idx: int,
    inflection_shift: np.ndarray,
    noise: float = 0.04,
) -> np.ndarray:
    n = len(DECADES)
    pts = np.zeros((n, 2))
    for i in range(n):
        t = i / (n - 1)
        pts[i] = (1 - t) * start + t * end
        if i >= inflection_idx:
            pts[i] += inflection_shift * ((i - inflection_idx) / (n - inflection_idx))
    pts += RNG.normal(0, noise, pts.shape)
    return pts


trajectories: dict[str, np.ndarray] = {
    "representation": make_trajectory(
        np.array([0.6,  0.5]),
        np.array([-0.3, -0.6]),
        inflection_idx=14,  # 1960s
        inflection_shift=np.array([-0.5, -0.7]),
        noise=0.03,
    ),
    "network": make_trajectory(
        np.array([0.5,  0.4]),
        np.array([-0.6, -0.5]),
        inflection_idx=15,  # 1970s
        inflection_shift=np.array([-0.7, -0.6]),
        noise=0.03,
    ),
    "intentionality": make_trajectory(
        np.array([0.55, 0.48]),
        np.array([0.40, 0.35]),
        inflection_idx=18,
        inflection_shift=np.array([-0.05, -0.05]),
        noise=0.02,
    ),
}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f0f0f")
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
              left=0.07, right=0.97, top=0.93, bottom=0.07)

ACCENT   = "#e8e8e8"
DIM      = "#888888"
GRID_C   = "#2a2a2a"
BG       = "#151515"

# --- Panel 1: Heatmap ---------------------------------------------------
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(BG)

matrix = np.array([PROFILES[w] for w in WORDS])

cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=0.6)
im = ax1.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

ax1.set_xticks(range(len(TRANSITIONS)))
ax1.set_xticklabels(
    [f"{d}s" for d in DECADES[:-1]],
    rotation=45, ha="right", fontsize=7.5, color=ACCENT,
)
ax1.set_yticks(range(len(WORDS)))
ax1.set_yticklabels(WORDS, fontsize=8.5, color=ACCENT)
ax1.set_title("Cosine drift — consecutive decades  (mock data)", color=ACCENT, fontsize=11, pad=10)

cbar = fig.colorbar(im, ax=ax1, fraction=0.015, pad=0.01)
cbar.ax.yaxis.set_tick_params(color=ACCENT)
cbar.outline.set_edgecolor(DIM)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=ACCENT, fontsize=8)
cbar.set_label("cosine distance", color=DIM, fontsize=8)

# Annotate the big expected events
annotations = [
    (12, WORDS.index("information"),   "Shannon 1948",   (12, WORDS.index("information") - 3)),
    (15, WORDS.index("network"),       "backprop 1986",  (15, WORDS.index("network") - 3)),
    (14, WORDS.index("affordance"),    "Gibson 1966",    (11, WORDS.index("affordance") - 2)),
    (18, WORDS.index("attention"),     "Transformer '17",(16, WORDS.index("attention") - 3)),
    (13, WORDS.index("intelligence"),  "Dartmouth 1956", (10, WORDS.index("intelligence") - 2)),
]
for x, y, label, (tx, ty) in annotations:
    ax1.annotate(
        label, xy=(x, y), xytext=(tx, ty),
        fontsize=7, color="#ffcc44",
        arrowprops=dict(arrowstyle="->", color="#ffcc44", lw=0.8),
    )

ax1.tick_params(axis="both", colors=DIM)
for spine in ax1.spines.values():
    spine.set_edgecolor(GRID_C)

# --- Panel 2: Line chart — selected words --------------------------------
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(BG)

highlight = {
    "attention":      ("#ff6b6b", 2.2),
    "network":        ("#4ecdc4", 1.8),
    "representation": ("#ffe66d", 1.8),
    "information":    ("#a8dadc", 1.6),
    "intentionality": ("#888888", 1.2),
    "consciousness":  ("#c77dff", 1.6),
}
x = range(len(TRANSITIONS))
label_offsets = {
    "attention":      0.02,
    "network":       -0.03,
    "representation": 0.03,
    "information":   -0.02,
    "intentionality": 0.00,
    "consciousness":  0.02,
}
for word, (color, lw) in highlight.items():
    ax2.plot(x, PROFILES[word], color=color, linewidth=lw, label=word, alpha=0.9)
    y_end = PROFILES[word][-1] + label_offsets.get(word, 0)
    ax2.text(len(TRANSITIONS) - 0.4, y_end, f" {word}",
             color=color, fontsize=7.5, va="center")

ax2.set_xticks(range(0, len(TRANSITIONS), 2))
ax2.set_xticklabels([f"{DECADES[i]}s" for i in range(0, len(TRANSITIONS), 2)],
                    rotation=45, ha="right", fontsize=8, color=ACCENT)
ax2.set_ylabel("cosine distance", color=DIM, fontsize=9)
ax2.set_title("Drift over time — selected words", color=ACCENT, fontsize=10, pad=8)
ax2.tick_params(colors=DIM)
ax2.yaxis.label.set_color(DIM)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID_C)
ax2.set_ylim(0, 0.7)

# --- Panel 3: PCA trajectory --------------------------------------------
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(BG)

colors_traj = {
    "representation": "#ffe66d",
    "network":        "#4ecdc4",
    "intentionality": "#888888",
}
cmap_decade = plt.cm.plasma
for word, color in colors_traj.items():
    pts = trajectories[word]
    for i in range(len(pts) - 1):
        t = i / (len(pts) - 2)
        seg_color = cmap_decade(t)
        ax3.plot(pts[i:i+2, 0], pts[i:i+2, 1], color=seg_color, linewidth=1.8, alpha=0.85)

    # markers at decade positions
    for i, decade in enumerate(DECADES):
        marker = "o" if i not in (0, len(DECADES) - 1) else "D"
        size = 30 if i not in (0, len(DECADES) - 1) else 60
        ax3.scatter(pts[i, 0], pts[i, 1], color=cmap_decade(i / (len(DECADES) - 1)),
                    s=size, zorder=5, marker=marker)
        if decade in (1820, 1880, 1960, 1990, 2010):
            ax3.annotate(f"{decade}s", (pts[i, 0], pts[i, 1]),
                         textcoords="offset points", xytext=(5, 4),
                         fontsize=7, color=color, alpha=0.9)

word_label_offsets = {
    "representation": (-0.08,  0.08),
    "network":        (-0.08, -0.12),
    "intentionality": ( 0.04,  0.06),
}
dx, dy = word_label_offsets.get(word, (0, 0.06))
ax3.text(pts[0, 0] + dx, pts[0, 1] + dy, word,
         color=color, fontsize=8.5, fontweight="bold")

# Phantom cluster labels
ax3.text(-0.25, 0.55, "philosophy\ncluster", color="#666", fontsize=8,
         ha="center", style="italic")
ax3.text(-0.5, -0.55, "ML / CS\ncluster", color="#666", fontsize=8,
         ha="center", style="italic")

ax3.set_title("PCA trajectory in compass space  (mock)", color=ACCENT, fontsize=10, pad=8)
ax3.set_xlabel("PC1", color=DIM, fontsize=9)
ax3.set_ylabel("PC2", color=DIM, fontsize=9)
ax3.tick_params(colors=DIM)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRID_C)

# Colorbar for decade time
sm = plt.cm.ScalarMappable(cmap=cmap_decade,
                            norm=mcolors.Normalize(vmin=DECADES[0], vmax=DECADES[-1]))
sm.set_array([])
cbar2 = fig.colorbar(sm, ax=ax3, fraction=0.04, pad=0.02)
cbar2.ax.yaxis.set_tick_params(color=ACCENT)
cbar2.outline.set_edgecolor(DIM)
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=ACCENT, fontsize=8)
cbar2.set_label("decade", color=DIM, fontsize=8)

# ---------------------------------------------------------------------------
fig.suptitle("interpretant — expected drift patterns  (synthetic prior)",
             color=ACCENT, fontsize=13, y=0.97)

import pathlib
pathlib.Path("plots").mkdir(exist_ok=True)
out = pathlib.Path("plots/mock_drift.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"saved → {out}")
plt.show()
