"""The figures in the README: neighbour panels, cross-perplexity by decade, the corpus grid.

Each function takes already-computed inputs (neighbour lists, a cross-perplexity table, cell
counts) and returns a matplotlib figure, so the figures can be tested on synthetic data and
regenerated from the real models and tables by ``interpretant figures``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from interpretant.visualization.style import (
    AXIS,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL,
    SERIES,
    SURFACE,
    add_footnote,
    font,
    new_figure,
    plot_top,
    style_axes,
)

FIELD_LABELS: dict[str, str] = {
    "ai": "AI",
    "books": "Books",
    "cogpsych": "Cognitive psychology",
    "communication": "Communication",
    "compsci": "Computer science",
    "hci": "HCI",
    "histphilsci": "History & philosophy of science",
    "linguistics": "Linguistics",
    "neuroscience": "Neuroscience",
    "philosophy": "Philosophy",
    "psychology": "Psychology",
    "sociology": "Sociology",
}


def field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field)


def decade_label(decade: int) -> str:
    return f"{decade}s"


def same_family(word: str, other: str, min_prefix: int = 5, share: float = 0.6) -> bool:
    """Whether two words are inflections or spelling variants of each other.

    The models are not lemmatised, so *structure* and *structures*, *embedding* and
    *embeddings*, or *conceptualisation* and *conceptualization* appear as separate
    neighbours. Two words count as one family when their common prefix is at least
    ``min_prefix`` characters and ``share`` of the shorter word.
    """
    prefix = 0
    for first, second in zip(word, other, strict=False):
        if first != second:
            break
        prefix += 1
    return prefix >= min_prefix and prefix >= share * min(len(word), len(other))


def distinct_neighbours(word: str, neighbours: list[str], k: int) -> list[str]:
    """The first ``k`` neighbours that are not variants of ``word`` or of an earlier one."""
    kept: list[str] = []
    for neighbour in neighbours:
        if same_family(word, neighbour) or any(same_family(neighbour, seen) for seen in kept):
            continue
        kept.append(neighbour)
        if len(kept) == k:
            break
    return kept


@dataclass(frozen=True)
class NeighbourColumn:
    """One cell's nearest neighbours of the word under study."""

    field: str
    decade: int
    count: int
    words: list[str]


def neighbour_panel(
    word: str,
    columns: list[NeighbourColumn],
    title: str,
    subtitle: str = "",
    footnote: str = "",
    k: int = 12,
) -> Figure:
    """Ranked neighbour lists, one column per cell, grouped and coloured by field.

    Inflections of the word and later variants of an earlier neighbour are left out
    (:func:`distinct_neighbours`). A word that also appears among another field's neighbours
    is set in muted ink, so the columns show at a glance how little two fields share.
    """
    fields = list(dict.fromkeys(column.field for column in columns))
    colours = dict(zip(fields, SERIES, strict=False))
    shown = {
        (column.field, column.decade): distinct_neighbours(word, column.words, k)
        for column in columns
    }
    shared = {
        neighbour
        for column in columns
        for neighbour in shown[(column.field, column.decade)]
        if any(
            neighbour in shown[(other.field, other.decade)]
            for other in columns
            if other.field != column.field
        )
    }
    width = 1.35 * len(columns) + 0.8
    rows = 2.4 + k * 0.78 + 1.6
    height = 0.19 * rows + 1.6
    figure = new_figure((width, height), title, subtitle)
    axes = figure.add_axes((0.04, 0.12, 0.94, plot_top(height) - 0.12))
    axes.set_xlim(0, len(columns))
    axes.set_ylim(rows, -0.2)
    axes.axis("off")
    for position, column in enumerate(columns):
        colour = colours.get(column.field, INK_SECONDARY)
        x = position + 0.08
        if position == 0 or columns[position - 1].field != column.field:
            axes.text(
                x, -0.1, field_label(column.field), color=colour, fontproperties=font("Bold", 11)
            )
        axes.text(
            x, 0.8, decade_label(column.decade), color=INK, fontproperties=font("SemiBold", 10)
        )
        axes.text(
            x,
            1.45,
            f"{column.count:,} occurrences",
            color=INK_MUTED,
            fontproperties=font("Regular", 7.5),
        )
        axes.plot([x, position + 0.92], [1.75, 1.75], color=colour, linewidth=1.2)
        for rank, neighbour in enumerate(shown[(column.field, column.decade)]):
            ink = INK_MUTED if neighbour in shared else INK
            axes.text(
                x,
                2.4 + rank * 0.78,
                neighbour.replace("_", " "),
                color=ink,
                fontproperties=font("Regular", 9.5),
            )
    if shared:
        axes.text(
            0.08,
            2.4 + k * 0.78 + 0.6,
            "grey: also among the other field's neighbours",
            color=INK_MUTED,
            fontproperties=font("Regular", 8),
        )
    if footnote:
        add_footnote(figure, footnote)
    return figure


def excess_by_decade(
    table: pd.DataFrame,
    pairs: list[tuple[str, str]],
    title: str,
    subtitle: str = "",
    footnote: str = "",
) -> Figure:
    """Excess cross-entropy per decade for ordered field pairs: seeds as points, mean as line.

    ``table`` has the columns of ``cross_perplexity_corrected.parquet``: ``seed``, ``decade``,
    ``model_field``, ``text_field``, ``excess``. Each pair is ``(model_field, text_field)``.
    """
    figure = new_figure((8, 4.8), title, subtitle)
    axes = figure.add_axes((0.1, 0.16, 0.66, plot_top(4.8) - 0.16))
    style_axes(axes)
    decades = sorted(table["decade"].unique())
    for (model_field, text_field), colour in zip(pairs, SERIES, strict=False):
        rows = table[(table["model_field"] == model_field) & (table["text_field"] == text_field)]
        means = rows.groupby("decade")["excess"].mean().reindex(decades)
        axes.scatter(
            rows["decade"],
            rows["excess"],
            s=28,
            color=colour,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )
        axes.plot(means.index, means.to_numpy(), color=colour, linewidth=2, zorder=3)
        label = f"{field_label(model_field)} → {field_label(text_field)}"
        axes.annotate(
            label,
            xy=(means.index[-1], means.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=colour,
            fontproperties=font("SemiBold", 9.5),
            va="center",
        )
    axes.set_xticks(decades)
    axes.set_xticklabels([decade_label(int(decade)) for decade in decades])
    axes.set_ylabel(
        "excess cross-entropy (bits per token)",
        color=INK_SECONDARY,
        fontproperties=font("Regular", 9),
    )
    axes.legend(
        handles=[
            Line2D([], [], color=colour, linewidth=2, label=f"{field_label(a)} → {field_label(b)}")
            for (a, b), colour in zip(pairs, SERIES, strict=False)
        ],
        loc="lower left",
        frameon=False,
        prop=font("Regular", 8.5),
        labelcolor=INK_SECONDARY,
    )
    if footnote:
        add_footnote(figure, footnote)
    return figure


def corpus_grid(
    cells: pd.DataFrame,
    title: str,
    subtitle: str = "",
    footnote: str = "",
    fields: list[str] | None = None,
) -> Figure:
    """Heatmap of documents per field × decade cell, one blue ramp, counts in the cells.

    ``cells`` has the columns ``source``, ``decade`` and ``documents``.
    """
    pivot = cells.pivot_table(index="source", columns="decade", values="documents", fill_value=0)
    if fields is not None:
        pivot = pivot.reindex(fields).fillna(0)
    matrix = pivot.to_numpy(dtype=float)
    rows, columns = matrix.shape
    height = 0.46 * rows + 2.0
    figure = new_figure((1.0 * columns + 3.4, height), title, subtitle)
    axes = figure.add_axes((0.3, 0.12, 0.68, plot_top(height) - 0.12))
    colormap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    shown = np.ma.masked_where(matrix == 0, matrix)
    colormap.set_bad(SURFACE)
    axes.imshow(shown, cmap=colormap, aspect="auto", vmin=0, vmax=matrix.max())
    axes.set_xticks(range(columns))
    axes.set_xticklabels([decade_label(int(decade)) for decade in pivot.columns])
    axes.set_yticks(range(rows))
    axes.set_yticklabels([field_label(str(source)) for source in pivot.index])
    axes.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_xticks(np.arange(-0.5, columns), minor=True)
    axes.set_yticks(np.arange(-0.5, rows), minor=True)
    axes.grid(which="minor", color=SURFACE, linewidth=2)
    axes.tick_params(which="minor", length=0)
    threshold = 0.55 * matrix.max()
    for row in range(rows):
        for column in range(columns):
            value = matrix[row, column]
            if value == 0:
                axes.text(column, row, "–", ha="center", va="center", color=AXIS, fontsize=8)
                continue
            ink = SURFACE if value > threshold else INK
            axes.text(
                column,
                row,
                f"{int(value):,}",
                ha="center",
                va="center",
                color=ink,
                fontproperties=font("Regular", 8),
            )
    axes.set_facecolor(SURFACE)
    if footnote:
        add_footnote(figure, footnote)
    return figure
