"""The word repairs of the cleaning runs (decision agenda 2.3), counted over the whole corpus.

The repair-texts stage has merged the works and repaired their texts; this stage reads those base
texts and their sentence languages, part by part. Each word repair is decided in turn, each from
counts over the whole corpus of the text the earlier ones produce, and changes only English
sentences and undetermined ones in mostly English abstracts (``sentence_pieces``); the counts read
every sentence (``_whole_texts``). The repairs are: the glued words of the
repaired texts (run 7), split or kept joined; then the words broken by a space or a line-end
hyphen in the texts with those splits made (run 7c), joined or kept apart; then the lower-case
glued words of the texts with those joins made (run 8), split or kept joined. Each table is written
whole, so the decisions can be reviewed before the pass reads them, and so are the texts with
every repair made, one part per field and decade, which the spaCy pass reads instead of
repairing in its main process (run 7c's profile, decision agenda step 9).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog import (
    REPAIRED_TEXTS_SCHEMA,
    SENTENCE_LANGUAGES_SCHEMA,
    CatalogConfig,
    load_base_sentence_languages,
    load_base_texts,
    load_kept_apart_words,
    load_kept_joined_words,
    save_glued_words,
    save_lower_glued_words,
    save_repaired_sentence_languages,
    save_repaired_texts,
    save_split_words,
)
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
    count_two_word_splits,
    decide_hyphenated_chunks,
    decide_lower_glued_word_splits,
    lower_glued_word_splits,
    split_lower_glued_words,
)
from interpretant.corpus.sentence_pieces import (
    PiecedText,
    pieced_texts,
    repair_pieces,
    sentence_ranges,
)
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
from interpretant.pipelines.corpus import corpus_cells, corpus_parts
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
    """Count the word repairs of the selected cells' base texts and write their decisions.

    Raises:
        ValueError: if no raw cell matches the selection.
        FileNotFoundError: if the kept-joined or kept-apart words file does not exist, or a
            part's texts have not been repaired.
    """
    kept_joined = [entry.word for entry in load_kept_joined_words(catalog)]
    kept_apart = [(entry.first, entry.second) for entry in load_kept_apart_words(catalog)]
    cells = corpus_cells(catalog, selection)
    parts = corpus_parts(cells)
    progress.stage("reading the base texts and their languages", total=len(parts))
    works: dict[str, list[PiecedText]] = {}
    for part in parts:
        base_texts = load_base_texts(catalog, part)
        if base_texts.height:
            languages = load_base_sentence_languages(catalog, part)
            works[part] = pieced_texts(base_texts.iter_rows(), list(languages.iter_rows()))
        progress.advance()
    texts = _whole_texts(works)
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
    progress.stage("splitting the glued words", total=len(works))
    works = map_parts(
        _split_part,
        {part: (part_works, splits) for part, part_works in works.items()},
        process_count=process_count,
        progress=progress,
    )
    texts = _whole_texts(works)
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
    progress.stage("joining the split words", total=len(works))
    works = map_parts(
        _join_part,
        {part: (part_works, word_joins) for part, part_works in works.items()},
        process_count=process_count,
        progress=progress,
    )
    texts = _whole_texts(works)
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
    progress.stage("splitting the lower-case glued words", total=len(works))
    lower_split = map_parts(
        _lower_split_part,
        {
            part: (part_works, lower_splits, lower.min_dense_splits)
            for part, part_works in works.items()
        },
        process_count=process_count,
        progress=progress,
    )
    occurrences: Counter[str] = Counter()
    for _, part_counts in lower_split.values():
        occurrences.update(part_counts)
    progress.stage("writing the repaired texts and their languages", total=len(parts))
    for part in parts:
        part_works = lower_split[part][0] if part in lower_split else []
        repaired_texts = [(work.work_id, work.text) for work in part_works]
        save_repaired_texts(
            catalog,
            pl.DataFrame(repaired_texts, schema=REPAIRED_TEXTS_SCHEMA, orient="row"),
            part,
        )
        moved = [
            (work.work_id, *sentence) for work in part_works for sentence in sentence_ranges(work)
        ]
        save_repaired_sentence_languages(
            catalog, pl.DataFrame(moved, schema=SENTENCE_LANGUAGES_SCHEMA, orient="row"), part
        )
        progress.advance()
    spaced = joins.filter(pl.col("join"))
    hyphenated = joins.filter(pl.col("join_hyphenated"))
    return WordRepairsResult(
        abstract_count=sum(len(part_works) for part_works in works.values()),
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


def _whole_texts(works: Mapping[str, list[PiecedText]]) -> pl.DataFrame:
    """Each work's whole text, what the counts read.

    The counts read every sentence, foreign ones too, although the repairs change only English
    ones: counting English sentences alone flipped decisions at the margins (one use fewer of
    "robotic and") and took away the joined uses that keep foreign words whole (*forma* in Latin
    terms), which made more kept abstracts worse than better (the check run of 2026-11-16).
    """
    return pl.DataFrame(
        [(work.work_id, work.text) for part_works in works.values() for work in part_works],
        schema={"work_id": pl.String(), "text": pl.String()},
        orient="row",
    )


def _split_part(
    _part: str, works_and_splits: tuple[list[PiecedText], Mapping[str, str]]
) -> list[PiecedText]:
    """One part's works with the glued words of their English sentences split."""
    works, splits = works_and_splits
    return [repair_pieces(work, lambda text: split_glued_words(text, splits)) for work in works]


def _join_part(
    _part: str, works_and_joins: tuple[list[PiecedText], SplitWordJoins]
) -> list[PiecedText]:
    """One part's works with the split words of their English sentences joined."""
    works, joins = works_and_joins
    return [repair_pieces(work, lambda text: join_split_words(text, joins)) for work in works]


def _lower_split_part(
    _part: str, works_and_splits: tuple[list[PiecedText], LowerGluedSplits, int]
) -> tuple[list[PiecedText], Counter[str]]:
    """One part's works with their English sentences' lower-case glue split, and the splits.

    The last of the word repairs. A work is damaged when its text holds at least
    ``min_dense_splits`` two-word glues; the guarded patterns apply only there. The splits made
    are counted by pattern.
    """
    works, splits, min_dense_splits = works_and_splits
    repaired: list[PiecedText] = []
    counts: Counter[str] = Counter()
    for work in works:
        is_damaged = count_two_word_splits(work.text, splits) >= min_dense_splits

        repaired.append(
            repair_pieces(work, partial(_split_counted, splits, counts, is_damaged=is_damaged))
        )
    return repaired, counts


def _split_counted(
    splits: LowerGluedSplits, counts: Counter[str], text: str, *, is_damaged: bool
) -> str:
    """One piece with its lower-case glue split, the splits added to ``counts``."""
    split_text, made = split_lower_glued_words(text, splits, is_damaged=is_damaged)
    counts.update(made)
    return split_text
