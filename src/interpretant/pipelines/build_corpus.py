"""Stage 3 of the run order (decision agenda 9.1): the corpus-level rules over the annotated table.

Reads every part of the selected cells as the spaCy pass left it (``annotate-corpus``), runs the
corpus-level rules over the whole table at once (template sentences, repeated sentences,
near-duplicates, the author cap: 2.3, 2.5), flags abstracts for the corpus review from their
works' OpenAlex metadata and from their text (4b), and saves the abstracts table's parts. The
rules compare abstracts across the corpus, so this stage always reads every part; it takes
minutes, so changing a rule's threshold never redoes the pass.
"""

from __future__ import annotations

import polars as pl
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog import CatalogConfig, load_annotated_abstracts, save_abstracts
from interpretant.corpus.cleaning import (
    mark_author_cap,
    mark_repeated_sentences,
    mark_template_sentences,
    mark_truncated,
)
from interpretant.corpus.deduplication import mark_near_duplicates
from interpretant.corpus.work_metadata import index_work_metadata, mark_metadata_flags
from interpretant.pipelines.corpus import (
    CorpusResult,
    corpus_cells,
    load_corpus_metadata,
    part_name,
    summarise,
)
from interpretant.pipelines.params import CorpusSelection, ParamsSettings

_PART = "part"


class BuildCorpusSettings(ParamsSettings):
    """The ``build_corpus`` section of ``params.yaml``: the corpus-level rules' thresholds."""

    model_config = SettingsConfigDict(yaml_config_section="build_corpus")

    template_min_abstracts: int = Field(ge=2)
    near_duplicate_threshold: float = Field(gt=0, le=1)
    author_cap_share: float = Field(gt=0, le=1)
    author_cap_floor: int = Field(ge=1)


def build_corpus(
    catalog: CatalogConfig, selection: CorpusSelection, settings: BuildCorpusSettings
) -> CorpusResult:
    """Apply the corpus-level rules and the review flags to every annotated part, and save them.

    Raises:
        ValueError: if no raw cell matches the selection, or a cell's metadata has not been
            fetched or is incomplete.
        FileNotFoundError: naming the parts the spaCy pass has not annotated yet.
    """
    cells = corpus_cells(catalog, selection)
    parts = [part_name(cell.field, cell.decade) for cell in cells]
    annotated = load_annotated_abstracts(catalog, parts)
    table = pl.concat(
        part.with_columns(pl.lit(name).alias(_PART)) for name, part in annotated.items()
    )
    metadata = index_work_metadata(load_corpus_metadata(catalog, cells), table["work_id"])
    table = mark_template_sentences(table, min_abstracts=settings.template_min_abstracts)
    table = mark_repeated_sentences(table)
    table = mark_near_duplicates(table, threshold=settings.near_duplicate_threshold)
    table = mark_author_cap(
        table, max_share=settings.author_cap_share, floor=settings.author_cap_floor
    )
    table = mark_truncated(mark_metadata_flags(table, metadata))
    tables = {part: table.filter(pl.col(_PART) == part).drop(_PART) for part in parts}
    for part, rows in tables.items():
        save_abstracts(catalog, rows, part=part)
    return summarise(tables)
