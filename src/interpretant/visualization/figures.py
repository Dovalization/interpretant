"""The figures in the README: neighbour panels, cross-perplexity by decade, the corpus grid.

Each function takes already-computed inputs (neighbour lists, a cross-perplexity table, cell
counts) and returns a matplotlib figure, so the figures can be tested on synthetic data and
regenerated from the real models and tables by ``interpretant figures``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from interpretant.drift.grid import cell_neighbours
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


def fall_by_model(
    table: pd.DataFrame,
    text_field: str,
    early: int,
    late: int,
    title: str,
    subtitle: str = "",
    footnote: str = "",
) -> Figure:
    """How much better each field's model predicts one field's text in ``late`` than ``early``.

    One bar per model (the fall of excess cross-entropy, bits per token, mean over seeds), the
    seeds as points, sorted; one hue, since the bars are one measure.
    """
    rows = table[table["text_field"] == text_field]
    rows = rows[rows["model_field"] != text_field]
    falls = (
        rows.pivot_table(index=["model_field", "seed"], columns="decade", values="excess")
        .pipe(lambda frame: frame[early] - frame[late])
        .rename("fall")
        .reset_index()
    )
    means = falls.groupby("model_field")["fall"].mean().sort_values()
    models = list(means.index)
    height = 0.42 * len(models) + 2.2
    figure = new_figure((8, height), title, subtitle)
    axes = figure.add_axes((0.3, 0.12, 0.66, plot_top(height) - 0.12))
    style_axes(axes, grid_axis="x")
    positions = range(len(models))
    axes.barh(positions, means.to_numpy(), color=SERIES[0], height=0.55, zorder=2)
    for position, model in zip(positions, models, strict=True):
        seeds = falls.loc[falls["model_field"] == model, "fall"]
        axes.scatter(
            seeds, [position] * len(seeds), s=18, color=INK, alpha=0.6, zorder=3, linewidths=0
        )
    axes.axvline(0, color=AXIS, linewidth=0.8)
    axes.set_yticks(list(positions))
    axes.set_yticklabels([field_label(model) for model in models])
    axes.tick_params(axis="y", length=0)
    axes.set_xlabel(
        f"fall in excess cross-entropy on {field_label(text_field)}'s abstracts, "
        f"{decade_label(early)} to {decade_label(late)} (bits per token)",
        color=INK_SECONDARY,
        fontproperties=font("Regular", 9),
    )
    if footnote:
        add_footnote(figure, footnote)
    return figure


def readme_figures(model_dir: Path, drift_dir: Path, audit_path: Path) -> dict[str, Figure]:
    """Draw the README's figures from the grid models, the corrected cross-perplexity table
    and the corpus audit, keyed by file stem."""

    def column(layer: str, field: str, decade: int, word: str) -> NeighbourColumn:
        vectors = KeyedVectors.load(str(model_dir / layer / f"{field}_{decade}.kv"))
        count = int(vectors.get_vecattr(word, "count")) if word in vectors.key_to_index else 0
        return NeighbourColumn(field, decade, count, cell_neighbours(vectors, word))

    drawn: dict[str, Figure] = {}

    drawn["embodiment"] = neighbour_panel(
        "embodiment",
        [
            column("openalex_full", "philosophy", decade, "embodiment")
            for decade in (1990, 2000, 2010, 2020)
        ]
        + [column("openalex_full", "hci", decade, "embodiment") for decade in (2000, 2010, 2020)],
        title="Embodiment means two different things",
        subtitle="Nearest neighbours of embodiment in every philosophy and HCI abstract, by decade",
        footnote="TWEC models on the full OpenAlex cells; 25 neighbours from words with at least 50 occurrences, the first 12 distinct words shown (inflections and spelling variants collapsed). Same result with a second seed and with PPMI-SVD.",  # noqa: E501
    )
    drawn["representation_ai"] = neighbour_panel(
        "representation",
        [column("openalex", "ai", decade, "representation") for decade in (1980, 2000, 2020)],
        title="Representation in AI, from symbolic to learned",
        subtitle="Nearest neighbours of representation in 10,000-abstract AI cells",
        footnote="TWEC models on the main OpenAlex grid, first 12 distinct neighbours (inflections and spelling variants collapsed); the symbolic 1980s holds in TWEC only, the learned 2020s in every run.",  # noqa: E501
    )
    table = pd.read_parquet(drift_dir / "cross_perplexity_corrected.parquet")
    drawn["philosophy_hci"] = excess_by_decade(
        table[table["decade"] >= 1990],
        [("philosophy", "hci"), ("hci", "philosophy")],
        title="Philosophy's model learned to predict HCI; HCI's did not learn philosophy",
        subtitle="Excess cross-entropy of one field's held-out abstracts under the other field's bigram model",  # noqa: E501
        footnote="Kneser-Ney bigrams, 200,000 training tokens per cell, shared abstracts removed, 1,000 held-out abstracts; points are 5 resampled runs, lines their mean.",  # noqa: E501
    )
    drawn["hci_gains"] = fall_by_model(
        table,
        "hci",
        1990,
        2020,
        title="HCI's language moved towards the behavioural and social sciences'",
        subtitle="How much better each field's model predicts HCI's abstracts in the 2020s than in the 1990s",  # noqa: E501
        footnote="Bars: mean over five resampled runs; points: the runs. Corrected design (200,000 training tokens, shared abstracts removed, 1,000 held-out abstracts). AI's and computer science's models are the two that did not gain.",  # noqa: E501
    )
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(report["cells"])[["source", "decade", "documents"]]
    fields = sorted(
        {str(source) for source in cells["source"] if source != "books"},
        key=lambda field: field_label(field).lower(),
    )
    drawn["corpus_grid"] = corpus_grid(
        cells[cells["source"] != "books"],
        title="The corpus is a grid",
        subtitle="Research abstracts per field × decade cell after cleaning (sampled to 10,000 per cell)",  # noqa: E501
        footnote="OpenAlex, seeded sample of English research prose; duplicates, catalogue records, scraped pages and non-English text removed; no author above 1% of a cell.",  # noqa: E501
        fields=fields,
    )
    return drawn
