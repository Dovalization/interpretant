#!/usr/bin/env python3
"""ASCII trajectory plot for a word across decades.

Usage:
    uv run python scripts/ascii_trajectory.py network
    uv run python scripts/ascii_trajectory.py representation intentionality
    uv run python scripts/ascii_trajectory.py network --model-dir models/twec
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path("models/twec")
WIDTH = 60
HEIGHT = 24


def load_vectors(word: str, model_dir: Path) -> dict[int, np.ndarray]:
    """Load the vector for a word from each decade .kv file."""
    from gensim.models import KeyedVectors

    vectors: dict[int, np.ndarray] = {}
    for kv_path in sorted(model_dir.glob("*.kv")):
        try:
            decade = int(kv_path.stem)
        except ValueError:
            continue
        kv = KeyedVectors.load(str(kv_path))
        if word in kv:
            vectors[decade] = kv[word]
    return vectors


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project N×D matrix to N×2 via PCA."""
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def render(words: list[str], model_dir: Path) -> None:
    # Collect all vectors across all words and decades
    word_vecs: dict[str, dict[int, np.ndarray]] = {}
    all_vecs: list[np.ndarray] = []
    all_keys: list[tuple[str, int]] = []

    for word in words:
        vecs = load_vectors(word, model_dir)
        if not vecs:
            print(f"'{word}' not found in any decade model.", file=sys.stderr)
            continue
        word_vecs[word] = vecs
        for decade, vec in vecs.items():
            all_vecs.append(vec)
            all_keys.append((word, decade))

    if not all_vecs:
        print("No vectors found.", file=sys.stderr)
        sys.exit(1)

    # Project to 2D
    matrix = np.array(all_vecs)
    coords_2d = pca_2d(matrix)

    # Map to word → decade → (x, y)
    results: dict[str, dict[int, tuple[float, float]]] = {}
    for i, (word, decade) in enumerate(all_keys):
        if word not in results:
            results[word] = {}
        results[word][decade] = (coords_2d[i, 0], coords_2d[i, 1])

    # Determine plot bounds
    xs = coords_2d[:, 0]
    ys = coords_2d[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    def to_grid(x: float, y: float) -> tuple[int, int]:
        col = int((x - x_min) / x_range * (WIDTH - 1))
        row = int((1 - (y - y_min) / y_range) * (HEIGHT - 1))
        return max(0, min(col, WIDTH - 1)), max(0, min(row, HEIGHT - 1))

    # Build grid
    MARKERS = "●○◆◇▲△"
    grid: list[list[str]] = [[" "] * WIDTH for _ in range(HEIGHT)]
    labels: list[tuple[int, int, str]] = []

    for word_idx, word in enumerate(results):
        marker = MARKERS[word_idx % len(MARKERS)]
        decades = sorted(results[word])
        for i, decade in enumerate(decades):
            x, y = results[word][decade]
            col, row = to_grid(x, y)
            grid[row][col] = marker
            # Label every other decade to avoid clutter
            short = str(decade)[2:]  # "1960" → "60"
            labels.append((col, row, f"{short}s"))

    # Print
    border = "+" + "─" * WIDTH + "+"
    print(border)
    for row in grid:
        print("|" + "".join(row) + "|")
    print(border)

    # Legend
    print()
    for word_idx, word in enumerate(results):
        marker = MARKERS[word_idx % len(MARKERS)]
        decades = sorted(results[word])
        decade_range = f"{decades[0]}s–{decades[-1]}s" if decades else "–"
        print(f"  {marker}  {word}  ({decade_range}, {len(decades)} decades)")

    # Trajectory summary per word
    print()
    for word in results:
        decades = sorted(results[word])
        if len(decades) < 2:
            continue
        print(f"  {word}:")
        for i, decade in enumerate(decades):
            x, y = results[word][decade]
            col, _ = to_grid(x, y)
            bar = "█" * int(col / WIDTH * 20)
            arrow = " →" if i < len(decades) - 1 else "  "
            print(f"    {decade}s  {bar:<20}  ({x:+.3f}, {y:+.3f}){arrow}")


def demo() -> None:
    """Render a synthetic demo when no model files exist yet."""
    import math

    print("  [demo mode — synthetic data]\n")

    # Simulate 'network' drifting from sociology → ML
    np.random.seed(42)
    words = {
        "network": [(d, math.sin(i * 0.4) * 0.6, -0.8 + i * 0.12)
                    for i, d in enumerate(range(1960, 2030, 10))],
        "representation": [(d, -0.5 + math.cos(i * 0.3) * 0.4, 0.3 - i * 0.06)
                           for i, d in enumerate(range(1920, 2030, 10))],
    }

    MARKERS = "●○"
    grid: list[list[str]] = [[" "] * WIDTH for _ in range(HEIGHT)]

    all_xs = [x for pts in words.values() for _, x, _ in pts]
    all_ys = [y for pts in words.values() for _, _, y in pts]
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    def to_grid(x: float, y: float) -> tuple[int, int]:
        col = int((x - x_min) / x_range * (WIDTH - 1))
        row = int((1 - (y - y_min) / y_range) * (HEIGHT - 1))
        return max(0, min(col, WIDTH - 1)), max(0, min(row, HEIGHT - 1))

    label_grid: dict[tuple[int, int], str] = {}

    for word_idx, (word, pts) in enumerate(words.items()):
        marker = MARKERS[word_idx]
        for decade, x, y in pts:
            col, row = to_grid(x, y)
            grid[row][col] = marker
            short = str(decade)[2:]
            # place label to the right if space
            lc = min(col + 1, WIDTH - 3)
            label_grid[(lc, row)] = f"{short}s"

    # Overlay labels
    for (col, row), label in label_grid.items():
        for i, ch in enumerate(label):
            if col + i < WIDTH and grid[row][col + i] == " ":
                grid[row][col + i] = ch

    border = "+" + "─" * WIDTH + "+"
    print(border)
    for row in grid:
        print("|" + "".join(row) + "|")
    print(border)
    print()
    print("  ●  network         (1960s–2020s)")
    print("  ○  representation  (1920s–2020s)")
    print()
    print("  run again once models/twec/*.kv files exist for real data")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASCII trajectory plot")
    parser.add_argument("words", nargs="*", default=["network"], help="Words to plot")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    kv_files = list(args.model_dir.glob("*.kv"))
    if not kv_files:
        demo()
    else:
        render(args.words, args.model_dir)
