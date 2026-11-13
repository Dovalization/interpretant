"""Build the corpus: the spaCy pass and the corpus-level rules over the raw OpenAlex cells.

Stages 2 and 3 of the decision agenda's run order (9.1): every selected cell's works, merged
across cells, are repaired and annotated in one stream; each part's documents go to the
annotation cache as the stream reaches its end; then the corpus-level rules run over the whole
table and its parts are saved. A part is a field in a decade, the first of a work's fields.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from interpretant.catalog import (
    ABSTRACTS_COLUMNS,
    CatalogConfig,
    list_raw_cells,
    load_cleaning_patterns,
    load_raw_cell,
    save_abstracts,
    save_annotations,
)
from interpretant.corpus.annotation import (
    AnnotatedAbstract,
    RawAbstract,
    annotate_abstracts,
    build_pipeline,
)
from interpretant.corpus.cleaning import (
    mark_author_cap,
    mark_repeated_sentences,
    mark_template_sentences,
)
from interpretant.corpus.deduplication import mark_near_duplicates, merge_cells

PARAMS_FILE = Path(__file__).resolve().parents[3] / "params.yaml"
REGULAR = "regular"
_PART = "part"


class BuildCorpusSettings(BaseSettings):
    """The ``build_corpus`` section of ``params.yaml``; keyword arguments override it."""

    model_config = SettingsConfigDict(
        extra="forbid", frozen=True, yaml_file=PARAMS_FILE, yaml_config_section="build_corpus"
    )

    fields: tuple[str, ...] = Field(min_length=1)
    decades: tuple[int, ...] = Field(min_length=1)
    languages: tuple[str, ...] = Field(min_length=1)
    english_threshold: float = Field(gt=0, le=1)
    max_not_english_share: float = Field(ge=0, le=1)
    template_min_abstracts: int = Field(ge=2)
    near_duplicate_threshold: float = Field(gt=0, le=1)
    author_cap_share: float = Field(gt=0, le=1)
    author_cap_floor: int = Field(ge=1)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keyword arguments first, then the parameters file; no environment variables."""
        return init_settings, YamlConfigSettingsSource(settings_cls)


@dataclass(frozen=True)
class BuildCorpusResult:
    """What a run wrote, and how many abstracts and ranges each rule took."""

    cell_count: int
    abstract_count: int
    dropped_counts: dict[str, int]
    removal_counts: dict[str, int]
    parts: list[str]


def build_corpus(
    catalog: CatalogConfig,
    settings: BuildCorpusSettings,
    *,
    batch_size: int = 256,
    process_count: int = 1,
) -> BuildCorpusResult:
    """Annotate the selected raw cells, apply the corpus-level rules and save both outputs.

    Saving a part replaces it; parts of cells outside the selection are left as they were.

    Raises:
        ValueError: if no raw cell matches the selected fields and decades, or they hold no
            works.
    """
    cells = [
        cell
        for cell in list_raw_cells(catalog)
        if cell.field in settings.fields and cell.decade in settings.decades
    ]
    if not cells:
        raise ValueError(f"no raw cell for fields {settings.fields} in decades {settings.decades}")
    patterns = load_cleaning_patterns(catalog)
    nlp = build_pipeline(
        patterns, languages=settings.languages, english_threshold=settings.english_threshold
    )
    abstracts = merge_cells(
        ((cell.field, load_raw_cell(catalog, cell)) for cell in cells), source=REGULAR
    )
    abstracts.sort(key=_part_name)
    annotated = annotate_abstracts(
        abstracts,
        nlp,
        patterns,
        max_not_english_share=settings.max_not_english_share,
        batch_size=batch_size,
        process_count=process_count,
    )
    part_tables = []
    for part, pairs in groupby(annotated, key=lambda pair: _part_name(pair[1])):
        docs, rows = zip(*pairs, strict=True)
        save_annotations(catalog, docs, part=part)
        part_tables.append(_table(rows).with_columns(pl.lit(part).alias(_PART)))
    if not part_tables:
        raise ValueError(f"the raw cells {cells} hold no works")
    table = _apply_corpus_rules(pl.concat(part_tables), settings)
    parts = table[_PART].unique(maintain_order=True).to_list()
    for part in parts:
        save_abstracts(catalog, table.filter(pl.col(_PART) == part).drop(_PART), part=part)
    return BuildCorpusResult(
        cell_count=len(cells),
        abstract_count=table.height,
        dropped_counts=_counts(table["dropped_reason"].drop_nulls()),
        removal_counts=_counts(table["removals"].explode().drop_nulls().struct.field("reason")),
        parts=parts,
    )


def _part_name(abstract: RawAbstract | AnnotatedAbstract) -> str:
    return f"{abstract.fields[0]}_{abstract.year - abstract.year % 10}"


def _table(rows: tuple[AnnotatedAbstract, ...]) -> pl.DataFrame:
    records: list[dict[str, Any]] = [asdict(row) for row in rows]
    return pl.DataFrame(records, schema=ABSTRACTS_COLUMNS)


def _apply_corpus_rules(table: pl.DataFrame, settings: BuildCorpusSettings) -> pl.DataFrame:
    """The corpus-level rules in the agenda's order (9.1)."""
    table = mark_template_sentences(table, min_abstracts=settings.template_min_abstracts)
    table = mark_repeated_sentences(table)
    table = mark_near_duplicates(table, threshold=settings.near_duplicate_threshold)
    return mark_author_cap(
        table, max_share=settings.author_cap_share, floor=settings.author_cap_floor
    )


def _counts(reasons: pl.Series) -> dict[str, int]:
    return {str(reason): count for reason, count in Counter(reasons.to_list()).most_common()}
