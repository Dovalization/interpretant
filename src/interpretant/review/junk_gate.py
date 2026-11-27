"""The junk gate's draws (`docs/junk_gate.md`): its sets, and records as the judges' items.

Every draw takes the smallest draw keys (``usages.identifiers.build_draw_key``) under its own seed,
so whether a usage is drawn depends on its own key, never on row order. The labelled sample L takes
each decade's drawn usages with the smallest keys; the enrichment set E takes the records of each
stratum likely to be junk with the smallest keys, skipping L's records; the confirmation samples
take each flagged cell's drawn usages with the smallest keys. An item shows a record's kept text,
cut to length, with the drawn usage's sentence between ⟦ and ⟧, and nothing else: the judges see
no word, field, decade or year. The key maps each item to its usage, words by their reported names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal

import polars as pl

from interpretant.catalog.junk_gate import JunkLabels, Run3Verdict
from interpretant.catalog.schemas import REMOVAL
from interpretant.catalog.words import Word
from interpretant.corpus.removals import Removal, kept_text
from interpretant.corpus.sentence_view import merge_ranges
from interpretant.review.usages_audit import build_with_keys
from interpretant.usages.identifiers import build_draw_key

DrawSet = Literal["sample", "run3", "confirmation"]
DRAW_SETS: tuple[DrawSet, ...] = ("sample", "run3", "confirmation")
# The seeds and sizes are pre-registered (docs/junk_gate.md), so they live here, not in params.yaml
LABELLED_SEED = 20261401
ENRICHMENT_SEED = 20261402
CONFIRMATION_SEED = 20261404
ORDER_SEED = 20261405
LABELLED_PER_DECADE = 500
ENRICHMENT_PER_STRATUM = 200
CONFIRMATION_PER_CELL = 40
# A repository-only record shorter than this, in kept characters, belongs to E's third stratum
SHORT_RECORD_CHARACTERS = 500
# The screen gates the cells with at least this many drawn usages
GATED_CELL_USAGES = 100
# A shown text up to this length is whole; a longer one keeps its head and tail, and the marked
# sentence with a window on each side when any of it falls between them
WHOLE_CHARACTERS = 6000
HEAD_CHARACTERS = 3500
TAIL_CHARACTERS = 1500
WINDOW_CHARACTERS = 300
OPEN_MARK = "⟦"
CLOSE_MARK = "⟧"
CUT = " [… cut …] "
NO_FINITE_VERB = "no_finite_verb"
LIST_LIKE = "list_like"
SHORT_REPOSITORY_ONLY = "short_repository_only"
# E's strata in the order they are drawn: a record drawn in one is skipped by the next
STRATA = (NO_FINITE_VERB, LIST_LIKE, SHORT_REPOSITORY_ONLY)
# The usages table's columns the draws read
USAGE_COLUMNS = [
    "work_id",
    "word",
    "content_key",
    "sentence_start",
    "sentence_end",
    "in_scope",
    "drawn",
    "list_like",
]
# What the enrichment draw needs to know of each record behind a drawn usage
RECORDS_SCHEMA = pl.Schema(
    {
        "work_id": pl.String(),
        "removals": pl.List(REMOVAL),
        "has_finite_verb": pl.Boolean(),
        "is_repository_only": pl.Boolean(),
        "kept_characters": pl.Int64(),
    }
)
# A character range of a text: its start and end
Span = tuple[int, int]


def render_record(text: str, removals: Sequence[Removal], sentence: Span | None = None) -> str:
    """A record as its item shows it: the kept text, the marked sentence, cut to length.

    ``sentence`` is the marked sentence's span in ``text``, before removals; it is shown between
    ⟦ and ⟧. A shown text of up to 6,000 characters is whole. A longer one keeps its first 3,500
    and last 1,500 characters, and the marked sentence with 300 characters on each side when any
    of it falls between them; ``[… cut …]`` stands for each gap.
    """
    kept = kept_text(text, removals)
    if sentence is None:
        return _cut(kept, None)
    start = _kept_offset(text, removals, sentence[0])
    end = _kept_offset(text, removals, _start_of_removal_around(removals, sentence[1]))
    shown = f"{kept[:start]}{OPEN_MARK}{kept[start:end]}{CLOSE_MARK}{kept[end:]}"
    return _cut(shown, (start, end + len(OPEN_MARK) + len(CLOSE_MARK)))


def parse_removals(rows: Sequence[Mapping[str, Any]]) -> list[Removal]:
    """The removed ranges of a table's ``removals`` column, as removals."""
    return [Removal(row["start"], row["end"], row["reason"]) for row in rows]


def draw_labelled(drawn: pl.DataFrame, *, seed: int, per_decade: int) -> pl.DataFrame:
    """In each decade, the ``per_decade`` drawn usages with the smallest keys, over all words."""
    keyed = build_with_keys(drawn, seed)
    return keyed.sort("decade", "key").group_by("decade", maintain_order=True).head(per_decade)


def build_strata(drawn: pl.DataFrame, records: pl.DataFrame) -> pl.DataFrame:
    """Each record behind a drawn usage, with whether it belongs to each of E's strata.

    ``records`` describes the records (``RECORDS_SCHEMA``). The strata: no finite verb outside
    removed ranges; a drawn usage in a list-like sentence; flagged repository-only with a kept text
    under 500 characters.

    Raises:
        ValueError: if a record behind a drawn usage has no description.
    """
    behind = drawn.group_by("work_id").agg(pl.col("list_like").any().alias(LIST_LIKE))
    undescribed = behind.join(records, on="work_id", how="anti")
    if undescribed.height:
        raise ValueError(f"{undescribed.height} records behind drawn usages have no description")
    return behind.join(records, on="work_id").select(
        "work_id",
        (~pl.col("has_finite_verb")).alias(NO_FINITE_VERB),
        LIST_LIKE,
        (
            pl.col("is_repository_only") & (pl.col("kept_characters") < SHORT_RECORD_CHARACTERS)
        ).alias(SHORT_REPOSITORY_ONLY),
    )


def draw_enrichment(
    drawn: pl.DataFrame,
    strata: pl.DataFrame,
    skipped: Collection[str],
    *,
    seed: int,
    per_stratum: int,
) -> pl.DataFrame:
    """Each stratum's ``per_stratum`` records with the smallest keys, each with its marked usage.

    A record's key is the key of the usage it marks: its drawn usage with the smallest key, or in
    the list-like stratum its list-like one with the smallest key. The records in ``skipped`` (L's)
    are left out, and the strata are drawn in order (``STRATA``), each skipping the records drawn
    before, so E's records are distinct. Each row is the marked usage, with its ``stratum``.
    """
    keyed = build_with_keys(_without(drawn, skipped), seed)
    first = _smallest_per_record(keyed)
    candidates = {
        NO_FINITE_VERB: first.join(strata.filter(NO_FINITE_VERB), on="work_id", how="semi"),
        LIST_LIKE: _smallest_per_record(keyed.filter(LIST_LIKE)),
        SHORT_REPOSITORY_ONLY: first.join(
            strata.filter(SHORT_REPOSITORY_ONLY), on="work_id", how="semi"
        ),
    }
    taken: set[str] = set()
    chosen = []
    for stratum in STRATA:
        picked = _without(candidates[stratum], taken).sort("key").head(per_stratum)
        taken.update(picked["work_id"])
        chosen.append(picked.with_columns(pl.lit(stratum).alias("stratum")))
    return pl.concat(chosen)


def count_strata(strata: pl.DataFrame) -> dict[str, int]:
    """The records in each of E's strata before the draw."""
    return {stratum: int(strata[stratum].sum()) for stratum in STRATA}


def draw_confirmation(
    drawn: pl.DataFrame, cells: pl.DataFrame, *, seed: int, per_cell: int
) -> pl.DataFrame:
    """For each flagged cell, its ``per_cell`` drawn usages with the smallest keys, in key order.

    A cell is a word and a decade over all fields; ``cells`` says which are flagged.
    """
    flagged = cells.filter("flagged").select("word", pl.col("decade").cast(drawn.schema["decade"]))
    keyed = build_with_keys(drawn.join(flagged, on=["word", "decade"], how="semi"), seed)
    return (
        keyed.sort("word", "decade", "key")
        .group_by("word", "decade", maintain_order=True)
        .head(per_cell)
    )


def build_items(
    samples: Sequence[Mapping[str, Any]],
    texts: Mapping[str, str],
    removals: Mapping[str, Sequence[Removal]],
    words: Mapping[str, Word],
    *,
    prefix: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """The drawn usages as items, in seeded order and numbered from 1 after ``prefix``, and the key.

    The order is the usages' draw keys under the order seed: a seeded shuffle in which no usage's
    place relative to another depends on the rest. Each sample row gives its usage, its sentence's
    span, its part, decade and ``set``, and an enrichment row its ``stratum``. An item holds its id
    and its text only; its key entry holds the set (and stratum), work, part, decade, word by its
    reported name, and content key.
    """
    ordered = sorted(
        samples,
        key=lambda row: build_draw_key(ORDER_SEED, row["work_id"], row["word"], row["content_key"]),
    )
    items, key = [], {}
    for number, row in enumerate(ordered, start=1):
        item_id = f"{prefix}{number:04d}"
        work_id = row["work_id"]
        sentence = (row["sentence_start"], row["sentence_end"])
        items.append(
            {"id": item_id, "text": render_record(texts[work_id], removals[work_id], sentence)}
        )
        entry: dict[str, Any] = {"set": row["set"]}
        if row.get("stratum") is not None:
            entry["stratum"] = row["stratum"]
        key[item_id] = entry | {
            "work_id": work_id,
            "part": row["part"],
            "decade": row["decade"],
            "word": words[row["word"]].reported_name,
            "content_key": row["content_key"],
        }
    return items, key


def build_confirmation_items(
    chosen: pl.DataFrame,
    labels: JunkLabels,
    texts: Mapping[str, str],
    removals: Mapping[str, Sequence[Removal]],
    words: Mapping[str, Word],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Items for the chosen usages not yet labelled, and a key listing every cell's usages.

    A usage that already has a label keeps it and gets no item: its cell lists the labelled item's
    id. The key holds the new items' entries (``items``, numbered ``C0001`` on) and each cell's
    usages as item ids in key order (``cells``: word by its reported name, decade, ids), the
    cells sorted by word and decade.
    """
    labelled = {
        (label.work_id, label.word, label.content_key): item_id
        for item_id, label in labels.items.items()
    }
    rows = chosen.to_dicts()
    new = [row | {"set": "confirmation"} for row in rows if _identity(row, words) not in labelled]
    items, entries = build_items(new, texts, removals, words, prefix="C")
    item_ids = labelled | {
        (entry["work_id"], entry["word"], entry["content_key"]): item_id
        for item_id, entry in entries.items()
    }
    cells: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        cell = (words[row["word"]].reported_name, row["decade"])
        cells.setdefault(cell, []).append(item_ids[_identity(row, words)])
    listed = [
        {"word": word, "decade": decade, "items": cells[word, decade]}
        for word, decade in sorted(cells)
    ]
    return items, {"items": entries, "cells": listed}


def build_run3_items(
    verdicts: Sequence[Run3Verdict],
    texts: Mapping[str, str],
    removals: Mapping[str, Sequence[Removal]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """Run 3's judged records as items under run 3's own ids, no sentence marked, and the key.

    Raises:
        ValueError: if a judged record has no text.
    """
    missing = [verdict.work_id for verdict in verdicts if verdict.work_id not in texts]
    if missing:
        raise ValueError(
            f"{len(missing)} of run 3's judged records have no text, such as {missing[:5]}"
        )
    items = [
        {"id": verdict.id, "text": render_record(texts[verdict.work_id], removals[verdict.work_id])}
        for verdict in verdicts
    ]
    key = {
        verdict.id: {
            "set": "run3",
            "work_id": verdict.work_id,
            "decade": verdict.decade,
            "field": verdict.field,
        }
        for verdict in verdicts
    }
    return items, key


def count_sample(
    usages: pl.DataFrame,
    words: Mapping[str, Word],
    strata_sizes: Mapping[str, int],
    key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """The sample's counts: per decade, per word and decade, the gated cells and E's strata.

    Per decade, the drawn usages and the records behind them; per word (by its reported name) and
    decade, the usages matched, in scope and drawn; the cells with at least 100 drawn usages; each
    stratum's records before the draw; and the items by set.
    """
    drawn = usages.filter("drawn")
    decades = (
        drawn.group_by("decade")
        .agg(pl.len().alias("usages"), pl.col("work_id").n_unique().alias("records"))
        .sort("decade")
    )
    cells = usages.group_by("word", "decade").agg(
        pl.len().alias("matched"),
        pl.col("in_scope").sum().alias("in_scope"),
        pl.col("drawn").sum().alias("drawn"),
    )
    by_word: dict[str, dict[int, dict[str, int]]] = {}
    for word, decade, matched, in_scope, drawn_count in cells.iter_rows():
        by_word.setdefault(words[word].reported_name, {})[decade] = {
            "matched": matched,
            "in_scope": in_scope,
            "drawn": drawn_count,
        }
    return {
        "items": dict(Counter(entry["set"] for entry in key.values())),
        "drawn_usages_by_decade": {str(decade): n for decade, n, _ in decades.iter_rows()},
        "records_by_decade": {str(decade): n for decade, _, n in decades.iter_rows()},
        "usages_by_word_and_decade": {
            name: {str(decade): by_word[name][decade] for decade in sorted(by_word[name])}
            for name in sorted(by_word)
        },
        "cells_with_at_least_100_drawn": int((cells["drawn"] >= GATED_CELL_USAGES).sum()),
        "enrichment_strata": dict(strata_sizes),
        "held_out_words": sum(word.held_out for word in words.values()),
    }


def _kept_offset(text: str, removals: Sequence[Removal], position: int) -> int:
    """Where ``position`` of the text falls in its kept text.

    The kept text of the text's first ``position`` characters begins the whole kept text, so its
    length is the offset. A position inside a removed range falls where the kept text resumes.
    """
    before = [
        Removal(removal.start, min(removal.end, position), removal.reason)
        for removal in removals
        if removal.start < position
    ]
    return len(kept_text(text[:position], before))


def _start_of_removal_around(removals: Sequence[Removal], position: int) -> int:
    """The start of the removed range ``position`` falls inside, else ``position`` itself.

    A closing mark placed there follows the kept text before the range, not the space after it.
    """
    for start, end in merge_ranges((removal.start, removal.end) for removal in removals):
        if start < position < end:
            return start
    return position


def _cut(shown: str, marked: Span | None) -> str:
    """The shown text, cut to its head and tail (and the marked sentence's window) when long."""
    if len(shown) <= WHOLE_CHARACTERS:
        return shown.strip()
    tail_start = len(shown) - TAIL_CHARACTERS
    spans = [(0, HEAD_CHARACTERS), (tail_start, len(shown))]
    if marked is not None and marked[1] > HEAD_CHARACTERS and marked[0] < tail_start:
        spans.append(
            (max(0, marked[0] - WINDOW_CHARACTERS), min(len(shown), marked[1] + WINDOW_CHARACTERS))
        )
    return CUT.join(shown[start:end] for start, end in merge_ranges(spans)).strip()


def _smallest_per_record(keyed: pl.DataFrame) -> pl.DataFrame:
    """Each record's usage with the smallest key."""
    return keyed.sort("key").group_by("work_id", maintain_order=True).first()


def _without(usages: pl.DataFrame, work_ids: Collection[str]) -> pl.DataFrame:
    """The usages of records other than ``work_ids``."""
    left_out = pl.DataFrame({"work_id": sorted(work_ids)}, schema={"work_id": pl.String()})
    return usages.join(left_out, on="work_id", how="anti")


def _identity(row: Mapping[str, Any], words: Mapping[str, Word]) -> tuple[str, str, str]:
    """A usage's work, word by its reported name, and content key, as a label names it."""
    return (row["work_id"], words[row["word"]].reported_name, row["content_key"])
