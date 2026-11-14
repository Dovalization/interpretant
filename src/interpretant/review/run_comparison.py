"""What changed between two builds of the abstracts table, for the review of each run (4b).

The corpus report counts what every rule did in one build; this comparison shows which works
changed outcome between two builds, and how the removals and sentence languages of the works kept
in both changed, with samples to read. Works are matched by id; sentences and removed ranges by
their text, since a repair change can shift character offsets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

KEPT = "kept"
ABSENT = "absent"
SAMPLE_SIZE = 15
EXCERPT_CHARACTERS = 320
SEED = 0
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class SampleGroup:
    """A heading and the sampled lines under it."""

    heading: str
    lines: list[str]


@dataclass(frozen=True)
class RunComparison:
    """The comparison's tables, each followed by its samples in the report."""

    outcomes: pl.DataFrame
    outcome_samples: list[SampleGroup]
    removals: pl.DataFrame
    removal_samples: list[SampleGroup]
    sentences: pl.DataFrame
    sentence_samples: list[SampleGroup]
    languages: pl.DataFrame
    language_samples: list[SampleGroup]


def compare_runs(
    previous: pl.DataFrame, current: pl.DataFrame, *, sample_size: int = SAMPLE_SIZE
) -> RunComparison:
    """Compare two builds of the abstracts table: outcomes, then removals and languages."""
    joined = (
        _outcomes(previous)
        .join(_outcomes(current), on="work_id", how="full", suffix="_now", coalesce=True)
        .with_columns(pl.col("outcome", "outcome_now").fill_null(ABSENT))
    )
    kept_in_both = joined.filter(
        (pl.col("outcome") == KEPT) & (pl.col("outcome_now") == KEPT)
    ).select("work_id")
    removals, removal_samples = _removal_changes(previous, current, kept_in_both, sample_size)
    sentences, sentence_samples, languages, language_samples = _sentence_changes(
        previous, current, kept_in_both, sample_size
    )
    return RunComparison(
        outcomes=joined.group_by("outcome", "outcome_now")
        .len("works")
        .sort("works", "outcome", "outcome_now", descending=[True, False, False]),
        outcome_samples=_outcome_samples(joined, sample_size),
        removals=removals,
        removal_samples=removal_samples,
        sentences=sentences,
        sentence_samples=sentence_samples,
        languages=languages,
        language_samples=language_samples,
    )


def render_comparison(comparison: RunComparison, *, title: str) -> str:
    """The comparison as a Markdown document: each table, then its samples as lists."""
    parts = [f"# {title}\n"]
    sections = [
        (
            "Outcomes",
            "Every work's outcome in the previous build and in this one (kept, a drop reason, or"
            " absent from a build).",
            comparison.outcomes,
            comparison.outcome_samples,
        ),
        (
            "Removals in works kept in both",
            "Removed ranges by reason, matched by their text: how many there were, how many are"
            " new and how many are no longer removed.",
            comparison.removals,
            comparison.removal_samples,
        ),
        (
            "Sentences in works kept in both",
            "Sentences matched by their text: how many are only in one build (their boundaries"
            " moved) and in how many works.",
            comparison.sentences,
            comparison.sentence_samples,
        ),
        (
            "Sentence languages in works kept in both",
            "Sentences found in both builds whose language label changed.",
            comparison.languages,
            comparison.language_samples,
        ),
    ]
    for heading, note, table, samples in sections:
        parts.append(f"## {heading}\n\n{note}\n\n{_markdown_table(table)}\n")
        for group in samples:
            parts.append(f"### {group.heading}\n")
            parts.extend(f"- {line}" for line in group.lines)
            parts.append("")
    return "\n".join(parts)


def _outcomes(table: pl.DataFrame) -> pl.DataFrame:
    return table.select(
        "work_id",
        "decade",
        pl.col("fields").list.first().alias("field"),
        "text",
        pl.col("dropped_reason").fill_null(KEPT).alias("outcome"),
    )


def _outcome_samples(joined: pl.DataFrame, sample_size: int) -> list[SampleGroup]:
    changed = (
        joined.filter(pl.col("outcome") != pl.col("outcome_now"))
        .group_by("outcome", "outcome_now")
        .len()
        .sort("len", "outcome", "outcome_now", descending=[True, False, False])
    )
    groups = []
    for before, now, count in changed.iter_rows():
        rows = joined.filter((pl.col("outcome") == before) & (pl.col("outcome_now") == now))
        sample = rows.sort("work_id").sample(min(sample_size, count), seed=SEED)
        lines = [
            f"`{work_id}` {decade_now or decade} {field_now or field}: "
            f"{_excerpt(text_now if text_now is not None else text)}"
            for work_id, decade, field, text, decade_now, field_now, text_now in sample.select(
                "work_id", "decade", "field", "text", "decade_now", "field_now", "text_now"
            ).iter_rows()
        ]
        groups.append(SampleGroup(f"{before} → {now} ({count:,} works)", lines))
    return groups


def _removal_changes(
    previous: pl.DataFrame, current: pl.DataFrame, works: pl.DataFrame, sample_size: int
) -> tuple[pl.DataFrame, list[SampleGroup]]:
    before = _removed_pieces(previous, works)
    now = _removed_pieces(current, works)
    keys = ["work_id", "reason", "piece"]
    added = now.join(before, on=keys, how="anti")
    gone = before.join(now, on=keys, how="anti")
    table = (
        before.group_by("reason")
        .len("before")
        .join(now.group_by("reason").len("now"), on="reason", how="full", coalesce=True)
        .join(added.group_by("reason").len("new"), on="reason", how="full", coalesce=True)
        .join(gone.group_by("reason").len("no_longer"), on="reason", how="full", coalesce=True)
        .fill_null(0)
        .sort("reason")
    )
    groups = []
    for label, pieces in (("newly removed", added), ("no longer removed", gone)):
        for reason, count in pieces.group_by("reason").len().sort("len", descending=True).rows():
            sample = pieces.filter(pl.col("reason") == reason).sort(keys)
            sample = sample.sample(min(sample_size, count), seed=SEED)
            lines = [
                f"`{work_id}`: {_excerpt(piece)}"
                for work_id, piece in sample.select("work_id", "piece").iter_rows()
            ]
            groups.append(SampleGroup(f"{reason}, {label} ({count:,})", lines))
    return table, groups


def _removed_pieces(table: pl.DataFrame, works: pl.DataFrame) -> pl.DataFrame:
    return (
        table.join(works, on="work_id")
        .select("work_id", "text", "removals")
        .explode("removals")
        .drop_nulls("removals")
        .select(
            "work_id",
            pl.col("removals").struct.field("reason").alias("reason"),
            pl.col("text")
            .str.slice(
                pl.col("removals").struct.field("start"),
                pl.col("removals").struct.field("end") - pl.col("removals").struct.field("start"),
            )
            .alias("piece"),
        )
        .unique()
    )


def _sentence_changes(
    previous: pl.DataFrame, current: pl.DataFrame, works: pl.DataFrame, sample_size: int
) -> tuple[pl.DataFrame, list[SampleGroup], pl.DataFrame, list[SampleGroup]]:
    before = _sentence_rows(previous, works)
    now = _sentence_rows(current, works)
    keys = ["work_id", "sentence"]
    only_before = before.join(now, on=keys, how="anti")
    only_now = now.join(before, on=keys, how="anti")
    sentences = pl.DataFrame(
        {
            "sentences": ["only in the previous build", "only in this build"],
            "count": [only_before.height, only_now.height],
            "works": [only_before["work_id"].n_unique(), only_now["work_id"].n_unique()],
        }
    )
    sentence_samples = [
        SampleGroup(
            f"Sentences {label} ({rows.height:,})",
            [
                f"`{work_id}`: {_excerpt(sentence)}"
                for work_id, sentence in rows.sort(keys)
                .sample(min(sample_size, rows.height), seed=SEED)
                .select(keys)
                .iter_rows()
            ],
        )
        for label, rows in (("only in this build", only_now), ("only in the previous", only_before))
        if rows.height
    ]
    relabelled = before.join(now, on=keys, suffix="_now").filter(
        pl.col("language") != pl.col("language_now")
    )
    languages = (
        relabelled.group_by("language", "language_now")
        .len("sentences")
        .sort("sentences", "language", "language_now", descending=[True, False, False])
    )
    language_samples = []
    for before_language, now_language, count in languages.head(10).iter_rows():
        rows = relabelled.filter(
            (pl.col("language") == before_language) & (pl.col("language_now") == now_language)
        ).sort(keys)
        language_samples.append(
            SampleGroup(
                f"{before_language} → {now_language} ({count:,} sentences)",
                [
                    f"`{work_id}`: {_excerpt(sentence)}"
                    for work_id, sentence in rows.sample(min(sample_size, count), seed=SEED)
                    .select(keys)
                    .iter_rows()
                ],
            )
        )
    return sentences, sentence_samples, languages, language_samples


def _sentence_rows(table: pl.DataFrame, works: pl.DataFrame) -> pl.DataFrame:
    return (
        table.join(works, on="work_id")
        .select("work_id", "text", "sentences")
        .explode("sentences")
        .drop_nulls("sentences")
        .select(
            "work_id",
            pl.col("text")
            .str.slice(
                pl.col("sentences").struct.field("start"),
                pl.col("sentences").struct.field("end") - pl.col("sentences").struct.field("start"),
            )
            .alias("sentence"),
            pl.col("sentences").struct.field("language").alias("language"),
        )
        .unique(["work_id", "sentence"], keep="first")
    )


def _excerpt(text: str) -> str:
    flat = _SPACES.sub(" ", text).strip()
    return flat if len(flat) <= EXCERPT_CHARACTERS else flat[: EXCERPT_CHARACTERS - 1] + "…"


def _markdown_table(table: pl.DataFrame) -> str:
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=60_000,
        thousands_separator=",",
    ):
        return str(table)
