"""The word repairs of the cleaning runs (decision agenda 2.3), counted over the whole corpus.

Every selected cell's works are merged and repaired as the spaCy pass repairs them, part by part
in worker processes. Each repair is then decided in the order the pass makes them, each from
counts over the whole corpus of the text the earlier ones produce: the glued words of the
repaired texts (run 7), split or kept joined; then the words broken by a space or a line-end
hyphen in the texts with those splits made (run 7c), joined or kept apart; then the lower-case
glued words of the texts with those joins made (run 8), split or kept joined. Each table is written
whole, so the decisions can be reviewed before the pass reads them, and so are the texts with
every repair made, one part per field and decade, which the spaCy pass reads instead of
repairing in its main process (run 7c's profile, decision agenda step 9).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.curated import load_kept_apart_words, load_kept_joined_words
from interpretant.catalog.glued_words import save_glued_words
from interpretant.catalog.lower_glued_words import save_lower_glued_words
from interpretant.catalog.repaired_texts import REPAIRED_TEXTS_SCHEMA, save_repaired_texts
from interpretant.catalog.split_words import save_split_words
from interpretant.corpus.glued_words import (
    count_glued_words,
    decide_glued_word_splits,
    glued_word_splits,
    split_glued_words,
)
from interpretant.corpus.lower_glued_words import (
    HYPHENATED_CHUNK,
    LowerGluedSplits,
    count_lower_glued_words,
    decide_hyphenated_chunks,
    decide_lower_glued_word_splits,
    lower_glued_word_splits,
    split_lower_glued_words,
)
from interpretant.corpus.repair import repair_text
from interpretant.corpus.split_words import (
    HYPHEN,
    HYPHEN_SPACE,
    SPACE,
    SplitWordJoins,
    count_split_words,
    decide_split_word_joins,
    join_split_words,
    split_word_joins,
)
from interpretant.corpus.works import part_of
from interpretant.pipelines.corpus import corpus_cells, corpus_parts, load_corpus_works, part_name
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts


class GluedWordSettings(BaseModel):
    """When a glued word is split (run 7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    apart_margin: float = Field(gt=0)
    opener_count: int = Field(ge=0)
    opener_min_left_letters: int = Field(ge=2)
    opener_min_right_letters: int = Field(ge=1)


class SplitWordSettings(BaseModel):
    """When a word broken by a space or a line-end hyphen is joined (run 7c)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_word_count: int = Field(ge=1)
    join_margin: float = Field(gt=0)
    hyphen_margin: float = Field(gt=0)


class LowerGluedWordSettings(BaseModel):
    """When a lower-case glued word is split (run 8)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    apart_margin: float = Field(gt=0)
    min_part_count: int = Field(ge=1)
    min_first_part_letters: int = Field(ge=2)
    min_dense_splits: int = Field(ge=1)


class CountWordRepairsSettings(ParamsSettings):
    """The ``count_word_repairs`` section of ``params.yaml``: which words repair changes."""

    model_config = SettingsConfigDict(yaml_config_section="count_word_repairs")

    glued_words: GluedWordSettings
    split_words: SplitWordSettings
    lower_glued_words: LowerGluedWordSettings


@dataclass(frozen=True)
class WordRepairsResult:
    """What the stage wrote: the abstracts read and each repair's words found and changed.

    A broken word can be joined in both ways, so ``split_joined_count`` counts each once and
    ``split_join_counts`` counts those joined after a space or a hyphen and a space (``spaced``)
    and after a hyphen alone (``hyphenated``). ``lower_glued_split_counts`` counts the
    lower-case glued words decided split by pattern, ``lower_glued_split_occurrences`` the splits
    made in the texts by pattern (the guarded patterns only in damaged texts).
    """

    abstract_count: int
    glued_word_count: int
    glued_split_counts: dict[str, int]
    glued_split_occurrences: int
    split_word_count: int
    split_joined_count: int
    split_join_counts: dict[str, int]
    split_join_occurrences: int
    lower_glued_word_count: int
    lower_glued_split_counts: dict[str, int]
    lower_glued_split_occurrences: dict[str, int]


def count_word_repairs(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: CountWordRepairsSettings,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> WordRepairsResult:
    """Count the word repairs of the selected cells' repaired texts and write their decisions.

    Raises:
        ValueError: if no raw cell matches the selection, or a cell's metadata has not been
            fetched or is incomplete.
        FileNotFoundError: if the kept-joined or kept-apart words file does not exist.
    """
    kept_joined = [entry.word for entry in load_kept_joined_words(catalog)]
    kept_apart = [(entry.first, entry.second) for entry in load_kept_apart_words(catalog)]
    cells = corpus_cells(catalog, selection)
    abstracts = load_corpus_works(catalog, cells, progress)
    by_part: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for abstract in abstracts:
        by_part[part_name(*part_of(abstract))].append((abstract.work_id, abstract.text))
    del abstracts
    progress.stage("repairing the texts", total=len(by_part))
    repaired = map_parts(
        _repair_part, dict(by_part), process_count=process_count, progress=progress
    )
    texts = pl.DataFrame(
        [row for rows in repaired.values() for row in rows],
        schema={"work_id": pl.String(), "text": pl.String()},
        orient="row",
    )
    progress.stage("counting glued words")
    counts = count_glued_words(texts)
    progress.stage("deciding the splits")
    glued = settings.glued_words
    decisions = decide_glued_word_splits(
        counts,
        apart_margin=glued.apart_margin,
        opener_count=glued.opener_count,
        opener_min_left_letters=glued.opener_min_left_letters,
        opener_min_right_letters=glued.opener_min_right_letters,
        kept_joined=kept_joined,
    )
    save_glued_words(catalog, decisions)
    split = decisions.filter(pl.col("split"))
    splits = glued_word_splits(decisions)
    progress.stage("splitting the glued words", total=len(repaired))
    split_texts = map_parts(
        _split_part,
        {part: (rows, splits) for part, rows in repaired.items()},
        process_count=process_count,
        progress=progress,
    )
    del repaired, texts
    texts = pl.DataFrame(
        [row for rows in split_texts.values() for row in rows],
        schema={"work_id": pl.String(), "text": pl.String()},
        orient="row",
    )
    progress.stage("counting split words")
    broken = settings.split_words
    split_counts = count_split_words(texts, min_word_count=broken.min_word_count)
    progress.stage("deciding the joins")
    joins = decide_split_word_joins(
        split_counts,
        join_margin=broken.join_margin,
        hyphen_margin=broken.hyphen_margin,
        kept_apart=kept_apart,
    )
    save_split_words(catalog, joins)
    word_joins = split_word_joins(joins)
    progress.stage("joining the split words", total=len(split_texts))
    joined_texts = map_parts(
        _join_part,
        {part: (rows, word_joins) for part, rows in split_texts.items()},
        process_count=process_count,
        progress=progress,
    )
    del split_texts
    texts = pl.DataFrame(
        [row for rows in joined_texts.values() for row in rows],
        schema={"work_id": pl.String(), "text": pl.String()},
        orient="row",
    )
    progress.stage("counting lower-case glued words")
    lower = settings.lower_glued_words
    lower_counts = count_lower_glued_words(
        texts,
        min_part_count=lower.min_part_count,
        min_first_part_letters=lower.min_first_part_letters,
    )
    progress.stage("deciding the lower-case splits")
    lower_decisions = decide_lower_glued_word_splits(
        lower_counts, apart_margin=lower.apart_margin, kept_joined=kept_joined
    )
    lower_decisions = pl.concat([lower_decisions, decide_hyphenated_chunks(texts, lower_decisions)])
    save_lower_glued_words(catalog, lower_decisions)
    lower_splits = lower_glued_word_splits(lower_decisions)
    progress.stage("splitting the lower-case glued words", total=len(joined_texts))
    lower_split = map_parts(
        _lower_split_part,
        {part: (rows, lower_splits, lower.min_dense_splits) for part, rows in joined_texts.items()},
        process_count=process_count,
        progress=progress,
    )
    del joined_texts
    occurrences: Counter[str] = Counter()
    for _, part_counts in lower_split.values():
        occurrences.update(part_counts)
    progress.stage("writing the repaired texts")
    for part in corpus_parts(cells):
        rows = lower_split[part][0] if part in lower_split else []
        save_repaired_texts(
            catalog, pl.DataFrame(rows, schema=REPAIRED_TEXTS_SCHEMA, orient="row"), part
        )
    spaced = joins.filter(pl.col("join"))
    hyphenated = joins.filter(pl.col("join_hyphenated"))
    return WordRepairsResult(
        abstract_count=texts.height,
        glued_word_count=decisions.height,
        glued_split_counts=dict(split["reason"].value_counts(sort=True).iter_rows()),
        glued_split_occurrences=int(split["occurrences"].sum()),
        split_word_count=joins.height,
        split_joined_count=joins.filter(pl.col("join") | pl.col("join_hyphenated")).height,
        split_join_counts={"spaced": spaced.height, "hyphenated": hyphenated.height},
        split_join_occurrences=int((spaced[SPACE] + spaced[HYPHEN_SPACE]).sum())
        + int(hyphenated[HYPHEN].sum()),
        lower_glued_word_count=lower_counts.height,
        lower_glued_split_counts=dict(
            lower_decisions.filter(pl.col("split") & (pl.col("pattern") != HYPHENATED_CHUNK))[
                "pattern"
            ]
            .value_counts(sort=True)
            .iter_rows()
        ),
        lower_glued_split_occurrences=dict(occurrences.most_common()),
    )


def _repair_part(_part: str, works: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One part's works with their texts repaired."""
    return [(work_id, repair_text(text)) for work_id, text in works]


def _split_part(
    _part: str, works_and_splits: tuple[list[tuple[str, str]], Mapping[str, str]]
) -> list[tuple[str, str]]:
    """One part's repaired works with their glued words split."""
    works, splits = works_and_splits
    return [(work_id, split_glued_words(text, splits)) for work_id, text in works]


def _join_part(
    _part: str, works_and_joins: tuple[list[tuple[str, str]], SplitWordJoins]
) -> list[tuple[str, str]]:
    """One part's works with their split words joined."""
    works, joins = works_and_joins
    return [(work_id, join_split_words(text, joins)) for work_id, text in works]


def _lower_split_part(
    _part: str, works_and_splits: tuple[list[tuple[str, str]], LowerGluedSplits, int]
) -> tuple[list[tuple[str, str]], Counter[str]]:
    """One part's works with their lower-case glued words split, and the splits made by pattern.

    The last of the word repairs.
    """
    works, splits, min_dense_splits = works_and_splits
    rows: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    for work_id, text in works:
        repaired, made = split_lower_glued_words(text, splits, min_dense_splits=min_dense_splits)
        rows.append((work_id, repaired))
        counts.update(made)
    return rows, counts
