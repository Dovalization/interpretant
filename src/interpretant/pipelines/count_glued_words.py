"""Run 7 of the cleaning runs (decision agenda 2.3): the corpus's glued words and their splits.

Every selected cell's works are merged and repaired as the spaCy pass repairs them, part by part
in worker processes; the glued words of the repaired texts are counted over the whole corpus
and each is decided: split before the spaCy pass, or kept joined. The table is written whole,
so the decisions can be reviewed before the pass reads them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import polars as pl
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog import CatalogConfig, load_kept_joined_words, save_glued_words
from interpretant.corpus.glued_words import count_glued_words as count_words
from interpretant.corpus.glued_words import decide_glued_word_splits
from interpretant.corpus.repair import repair_text
from interpretant.corpus.works import part_of
from interpretant.pipelines.corpus import corpus_cells, load_corpus_works, part_name
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts


class CountGluedWordsSettings(ParamsSettings):
    """The ``count_glued_words`` section of ``params.yaml``: when a glued word is split."""

    model_config = SettingsConfigDict(yaml_config_section="count_glued_words")

    apart_margin: float = Field(gt=0)
    opener_count: int = Field(ge=0)
    opener_min_left_letters: int = Field(ge=2)


@dataclass(frozen=True)
class GluedWordsResult:
    """What the stage wrote: the abstracts read, the glued words found and those split."""

    abstract_count: int
    word_count: int
    split_counts: dict[str, int]
    split_occurrences: int


def count_glued_words(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: CountGluedWordsSettings,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> GluedWordsResult:
    """Count the glued words of the selected cells' repaired texts and write their decisions.

    Raises:
        ValueError: if no raw cell matches the selection, or a cell's metadata has not been
            fetched or is incomplete.
        FileNotFoundError: if the kept-joined words file does not exist.
    """
    kept_joined = [entry.word for entry in load_kept_joined_words(catalog)]
    abstracts = load_corpus_works(catalog, corpus_cells(catalog, selection), progress)
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
    counts = count_words(texts)
    progress.stage("deciding the splits")
    decisions = decide_glued_word_splits(
        counts,
        apart_margin=settings.apart_margin,
        opener_count=settings.opener_count,
        opener_min_left_letters=settings.opener_min_left_letters,
        kept_joined=kept_joined,
    )
    save_glued_words(catalog, decisions)
    split = decisions.filter(pl.col("split"))
    return GluedWordsResult(
        abstract_count=texts.height,
        word_count=decisions.height,
        split_counts=dict(split["reason"].value_counts(sort=True).iter_rows()),
        split_occurrences=int(split["occurrences"].sum()),
    )


def _repair_part(_part: str, works: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """One part's works with their texts repaired as the spaCy pass repairs them."""
    return [(work_id, repair_text(text)) for work_id, text in works]
