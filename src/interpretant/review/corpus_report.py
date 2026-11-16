"""The corpus report (decision agenda 4b, layer 2): what each rule did, by decade and field.

The author reads it before the corpus review, and the datasheet takes its counts from it. Each
section is a small table computed from the abstracts table and, for repository-only works, the
works' primary sources, so that archives can be told apart from repositories of papers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from interpretant.corpus.work_metadata import REPOSITORY_ONLY

TOP_ROWS = 30
# The agenda's measure of a short abstract (4b: 34% of 1950s abstracts against 3.6% of the
# 2020s' were under 250 characters in the raw cells).
SHORT_CHARACTERS = 250
SAMPLE_CHARACTERS = 100
_KEPT = "kept"


@dataclass(frozen=True)
class ReportSection:
    """One section of the report: a heading, a sentence on how to read it, and its table."""

    title: str
    note: str
    table: pl.DataFrame


def build_corpus_report(
    abstracts: pl.DataFrame, sources: pl.DataFrame, *, top: int = TOP_ROWS
) -> list[ReportSection]:
    """Every section of the report, from the abstracts table and each work's primary source.

    ``sources`` has a row per work with its ``work_id`` and ``source_name``, as OpenAlex's
    staging table has them.
    """
    return [
        _totals(abstracts),
        _dropped_by(abstracts, "decade"),
        _dropped_by(abstracts.explode("fields"), "fields"),
        _removals(abstracts),
        _template_sentences(abstracts, top),
        _flags_by_decade(abstracts),
        _repository_outcomes(abstracts),
        _repositories(abstracts, sources, top),
        _near_duplicates(abstracts),
        _short_abstracts(abstracts),
    ]


def corpus_metrics(abstracts: pl.DataFrame) -> dict[str, object]:
    """The corpus's headline counts, for DVC to compare between runs (``dvc metrics diff``)."""
    dropped = abstracts["dropped_reason"].drop_nulls()
    removals = abstracts["removals"].explode().drop_nulls().struct.field("reason")
    flags = abstracts["flags"].explode().drop_nulls()
    return {
        "abstracts": abstracts.height,
        _KEPT: abstracts.height - dropped.len(),
        "dropped": dropped.len(),
        "dropped_by_reason": _value_counts(dropped),
        "removed_ranges_by_reason": _value_counts(removals),
        "flagged_by_flag": _value_counts(flags),
    }


def compare_metrics(previous: Mapping[str, object], current: Mapping[str, object]) -> ReportSection:
    """Every count that differs between the previous run's metrics and this run's."""
    before = _flatten(previous)
    after = _flatten(current)
    changed = sorted(
        name for name in before.keys() | after.keys() if before.get(name, 0) != after.get(name, 0)
    )
    table = pl.DataFrame(
        {
            "count": changed,
            "previous": [before.get(name, 0) for name in changed],
            "now": [after.get(name, 0) for name in changed],
            "change": [after.get(name, 0) - before.get(name, 0) for name in changed],
        },
        schema={"count": pl.String, "previous": pl.Int64, "now": pl.Int64, "change": pl.Int64},
    )
    note = (
        "Each count that changed since the metrics were last written (metrics/corpus.json, kept "
        "in git); `dvc metrics diff` compares any two commits the same way."
    )
    return ReportSection("Compared with the previous run", note, table)


def render_markdown(sections: Sequence[ReportSection], *, title: str) -> str:
    """The report as a Markdown document, each table written by Polars."""
    parts = [f"# {title}\n"]
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_rows=-1,
        tbl_cols=-1,
        # Wide enough that no cell wraps: a wrapped cell breaks the Markdown table.
        tbl_width_chars=60_000,
        fmt_str_lengths=SAMPLE_CHARACTERS + 20,
        thousands_separator=",",
    ):
        parts.extend(
            f"## {section.title}\n\n{section.note}\n\n{section.table}\n" for section in sections
        )
    return "\n".join(parts)


def _flatten(metrics: Mapping[str, object], prefix: str = "") -> dict[str, int]:
    flat: dict[str, int] = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            flat |= _flatten(value, f"{prefix}{key}.")
        elif isinstance(value, int):
            flat[f"{prefix}{key}"] = value
    return flat


def _value_counts(values: pl.Series) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values.to_list()).items())}


def _totals(abstracts: pl.DataFrame) -> ReportSection:
    dropped = abstracts["dropped_reason"].is_not_null().sum()
    counts = [abstracts.height, abstracts.height - dropped, dropped]
    table = pl.DataFrame(
        {
            "abstracts": ["all", _KEPT, "dropped"],
            "count": counts,
            "share": [f"{count / abstracts.height:.1%}" for count in counts],
        }
    )
    return ReportSection("Totals", "Every abstract of the corpus, kept or dropped.", table)


def _dropped_by(abstracts: pl.DataFrame, column: str) -> ReportSection:
    label = "decade" if column == "decade" else "field"
    counts = (
        abstracts.with_columns(
            pl.col("dropped_reason").fill_null(_KEPT), pl.col(column).cast(pl.String())
        )
        .group_by("dropped_reason", column)
        .len()
        .pivot(on=column, index="dropped_reason", values="len")
        .fill_null(0)
    )
    groups = sorted(name for name in counts.columns if name != "dropped_reason")
    table = (
        counts.select("dropped_reason", *groups)
        .with_columns(pl.sum_horizontal(groups).alias("all"))
        .sort(pl.col("dropped_reason") == _KEPT, "all", descending=[False, True])
        .rename({"dropped_reason": "reason"})
    )
    note = f"Abstracts by why they were dropped, per {label}; the last row counts those kept." + (
        " A work in two fields counts in both." if label == "field" else ""
    )
    return ReportSection(f"Dropped abstracts by {label}", note, table)


def _removals(abstracts: pl.DataFrame) -> ReportSection:
    table = (
        abstracts.select("work_id", "removals")
        .explode("removals")
        .drop_nulls("removals")
        .unnest("removals")
        .group_by("reason")
        .agg(pl.len().alias("ranges"), pl.col("work_id").n_unique().alias("abstracts"))
        .sort("ranges", descending=True)
    )
    note = "Ranges of text the views leave out, by reason, and how many abstracts hold one."
    return ReportSection("Removed ranges", note, table)


def _template_sentences(abstracts: pl.DataFrame, top: int) -> ReportSection:
    removed = (
        abstracts.select("text", "removals")
        .explode("removals")
        .drop_nulls("removals")
        .unnest("removals")
        .filter(pl.col("reason") == "template_sentence")
    )
    table = (
        removed.select(
            pl.col("text")
            .str.slice(pl.col("start"), pl.col("end") - pl.col("start"))
            .str.replace_all(r"\d+", "0")
            .str.slice(0, SAMPLE_CHARACTERS)
            .alias("sentence (digits masked)")
        )
        .group_by("sentence (digits masked)")
        .len()
        .rename({"len": "removed"})
        .sort("removed", "sentence (digits masked)", descending=[True, False])
        .head(top)
    )
    note = (
        f"The {top} template sentences removed most often (shared by 20 or more abstracts once "
        "digits are masked), out of the distinct template sentences found."
    )
    return ReportSection("Template sentences", note, table)


def _flags_by_decade(abstracts: pl.DataFrame) -> ReportSection:
    flagged = abstracts.select("decade", "flags").explode("flags").drop_nulls("flags")
    if flagged.is_empty():
        return ReportSection("Flags by decade", "No abstract is flagged.", flagged)
    counts = (
        flagged.with_columns(pl.col("decade").cast(pl.String()))
        .group_by("flags", "decade")
        .len()
        .pivot(on="decade", index="flags", values="len")
        .fill_null(0)
    )
    decades = sorted(name for name in counts.columns if name != "flags")
    table = (
        counts.select("flags", *decades)
        .with_columns(pl.sum_horizontal(decades).alias("all"))
        .sort("all", descending=True)
        .rename({"flags": "flag"})
    )
    note = "Abstracts flagged for the review, kept or dropped, by decade."
    return ReportSection("Flags by decade", note, table)


def _repository_outcomes(abstracts: pl.DataFrame) -> ReportSection:
    repository = abstracts.filter(pl.col("flags").list.contains(REPOSITORY_ONLY))
    table = (
        repository.group_by(pl.col("dropped_reason").fill_null(_KEPT).alias("outcome"))
        .len()
        .rename({"len": "works"})
        .with_columns((pl.col("works") / max(repository.height, 1)).round(3).alias("share"))
        .sort("works", descending=True)
    )
    note = (
        "What the text rules did with the repository-only works (a repository source, listed by "
        "none of Crossref, PubMed, arXiv or DOAJ)."
    )
    return ReportSection("Repository-only works by outcome", note, table)


def _repositories(abstracts: pl.DataFrame, sources: pl.DataFrame, top: int) -> ReportSection:
    repository = (
        abstracts.filter(pl.col("flags").list.contains(REPOSITORY_ONLY))
        .join(
            sources.select("work_id", pl.col("source_name").alias("repository")),
            on="work_id",
            how="left",
        )
        .with_columns(
            pl.col("repository").fill_null("(no source)"),
            pl.col("dropped_reason").is_null().alias("is_kept"),
        )
    )
    table = (
        repository.group_by("repository")
        .agg(
            pl.col("is_kept").sum().alias(_KEPT),
            (~pl.col("is_kept")).sum().alias("dropped"),
            pl.col("text")
            .filter("is_kept")
            .sort_by(pl.col("work_id").filter("is_kept"))
            .first()
            .str.slice(0, SAMPLE_CHARACTERS)
            .alias("a kept text"),
        )
        .sort(_KEPT, "repository", descending=[True, False])
        .head(top)
    )
    note = (
        f"The {top} repositories holding the most kept repository-only works, with how many of "
        "theirs the text rules already dropped and the start of one kept text: archives "
        "(newspapers, photographs, letters) can be listed for dropping, repositories of papers "
        "kept."
    )
    return ReportSection("Repository-only works by repository", note, table)


def _near_duplicates(abstracts: pl.DataFrame) -> ReportSection:
    kept = abstracts.select(
        pl.col("work_id").alias("duplicate_of"),
        pl.col("fields").alias("kept_fields"),
        pl.col("decade").alias("kept_decade"),
    )
    pairs = abstracts.filter(pl.col("dropped_reason") == "near_duplicate").join(
        kept, on="duplicate_of", how="left"
    )
    shared = pl.col("fields").list.set_intersection("kept_fields").list.len()
    table = (
        pairs.with_columns(
            pl.when(pl.col("fields").list.sort() == pl.col("kept_fields").list.sort())
            .then(pl.lit("same fields"))
            .when(shared > 0)
            .then(pl.lit("a field in common"))
            .otherwise(pl.lit("no field in common"))
            .alias("fields"),
            pl.when(pl.col("decade") == pl.col("kept_decade"))
            .then(pl.lit("same decade"))
            .otherwise(pl.lit("different decades"))
            .alias("decades"),
        )
        .group_by("fields", "decades")
        .len()
        .rename({"len": "pairs"})
        .sort("pairs", descending=True)
    )
    note = (
        "Near-duplicate pairs (the dropped work and the work it duplicates) by whether they share "
        "their fields and their decade (decision agenda 2.5 waits on these counts)."
    )
    return ReportSection("Near-duplicates across fields and decades", note, table)


def _short_abstracts(abstracts: pl.DataFrame) -> ReportSection:
    table = (
        abstracts.filter(pl.col("dropped_reason").is_null())
        .with_columns(pl.col("text").str.len_chars().alias("characters"))
        .group_by("decade")
        .agg(
            pl.len().alias(_KEPT),
            (pl.col("characters") < SHORT_CHARACTERS).mean().round(3).alias("short share"),
            pl.col("characters").median().round().cast(pl.Int64()).alias("median characters"),
        )
        .sort("decade")
        .with_columns(pl.col("decade").cast(pl.String()))
    )
    note = (
        f"Kept abstracts per decade, the share under {SHORT_CHARACTERS} characters, and the median."
    )
    return ReportSection("Short abstracts by decade", note, table)
