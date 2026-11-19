"""The usages table, saved in parts (decision agenda 4.1-4.6; `docs/preprocessing_phase.md` step 6).

One row per target mention in a kept work, outside removed ranges: its usage id (work, word, content
key) and draw key; the sentence-view sentence it sits in (number, span, word count, language and
confidence, exclusion); the target's span in the repaired text, its compound and position (head or
modifier) when it is part of a hyphenated word; the head token's part of speech, tag, morphology
and lemma, the entity it sits in, and the static view's entry for it; whether it is in scope;
whether it is the one usage drawn for its work and word; the model that annotated it; and whether
the model read only a window around the target (a sentence over 512 tokens). Text is not copied:
the abstracts table holds it. A part is written whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import pandera.polars as pa
import polars as pl
from pandera.polars import PolarsData

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

MODELS = ("en_core_web_sm", "en_core_web_trf")
USAGES_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "word": pl.String(),
        "content_key": pl.String(),
        "draw_key": pl.String(),
        "sentence_number": pl.Int64(),
        "sentence_start": pl.Int64(),
        "sentence_end": pl.Int64(),
        "target_start": pl.Int64(),
        "target_end": pl.Int64(),
        "compound": pl.String(),
        "position": pl.String(),
        "pos": pl.String(),
        "tag": pl.String(),
        "morph": pl.String(),
        "lemma": pl.String(),
        "entity": pl.String(),
        "static_token": pl.String(),
        "sentence_word_count": pl.Int64(),
        "sentence_language": pl.String(),
        "sentence_confidence": pl.Float64(),
        "sentence_exclusion": pl.String(),
        "in_scope": pl.Boolean(),
        "drawn": pl.Boolean(),
        "model": pl.String(),
        "windowed": pl.Boolean(),
    }
)
_SUFFIX = ".parquet"


def _offsets_are_ordered(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        (pl.col("sentence_start") <= pl.col("target_start"))
        & (pl.col("target_start") < pl.col("target_end"))
        & (pl.col("target_end") <= pl.col("sentence_end"))
    )
    return checked


def _drawn_usages_are_eligible(data: PolarsData) -> pl.LazyFrame:
    checked: pl.LazyFrame = data.lazyframe.select(
        ~pl.col("drawn") | (pl.col("in_scope") & pl.col("sentence_exclusion").is_null())
    )
    return checked


USAGES_SCHEMA = pa.DataFrameSchema(
    {
        "work_id": pa.Column(
            USAGES_COLUMNS["work_id"], checks=pa.Check.str_startswith("https://openalex.org/W")
        ),
        "word": pa.Column(USAGES_COLUMNS["word"]),
        "content_key": pa.Column(
            USAGES_COLUMNS["content_key"], checks=pa.Check.str_matches(r"^[0-9a-f]{20}$")
        ),
        "draw_key": pa.Column(
            USAGES_COLUMNS["draw_key"], checks=pa.Check.str_matches(r"^[0-9a-f]{64}$")
        ),
        "sentence_number": pa.Column(USAGES_COLUMNS["sentence_number"], checks=pa.Check.ge(0)),
        "sentence_start": pa.Column(USAGES_COLUMNS["sentence_start"], checks=pa.Check.ge(0)),
        "sentence_end": pa.Column(USAGES_COLUMNS["sentence_end"]),
        "target_start": pa.Column(USAGES_COLUMNS["target_start"]),
        "target_end": pa.Column(USAGES_COLUMNS["target_end"]),
        "compound": pa.Column(USAGES_COLUMNS["compound"], nullable=True),
        "position": pa.Column(
            USAGES_COLUMNS["position"], nullable=True, checks=pa.Check.isin(["head", "modifier"])
        ),
        "pos": pa.Column(USAGES_COLUMNS["pos"]),
        "tag": pa.Column(USAGES_COLUMNS["tag"]),
        "morph": pa.Column(USAGES_COLUMNS["morph"]),
        "lemma": pa.Column(USAGES_COLUMNS["lemma"]),
        "entity": pa.Column(USAGES_COLUMNS["entity"], nullable=True),
        "static_token": pa.Column(USAGES_COLUMNS["static_token"], nullable=True),
        "sentence_word_count": pa.Column(
            USAGES_COLUMNS["sentence_word_count"], checks=pa.Check.ge(0)
        ),
        "sentence_language": pa.Column(USAGES_COLUMNS["sentence_language"]),
        "sentence_confidence": pa.Column(USAGES_COLUMNS["sentence_confidence"]),
        "sentence_exclusion": pa.Column(
            USAGES_COLUMNS["sentence_exclusion"], nullable=True, checks=pa.Check.isin(["cut_off"])
        ),
        "in_scope": pa.Column(USAGES_COLUMNS["in_scope"]),
        "drawn": pa.Column(USAGES_COLUMNS["drawn"]),
        "model": pa.Column(USAGES_COLUMNS["model"], checks=pa.Check.isin(MODELS)),
        "windowed": pa.Column(USAGES_COLUMNS["windowed"]),
    },
    checks=[
        pa.Check(_offsets_are_ordered, name="the target lies inside its sentence"),
        pa.Check(_drawn_usages_are_eligible, name="a drawn usage is in scope and not excluded"),
    ],
    unique=["work_id", "word", "content_key"],
    strict=True,
    name="usages",
    description=(
        "One row per target mention in a kept work, pointing into the abstracts table "
        "(decision agenda 4.6)."
    ),
)


def save_usages(catalog: CatalogConfig, usages: pl.DataFrame, part: str) -> Path:
    """Check one part's usages against the schema and write them, replacing the part.

    Raises:
        ValueError: if the part name is not safe as a file name.
        pandera.errors.SchemaError: if the rows break the schema; nothing is written.
    """
    validated: pl.DataFrame = USAGES_SCHEMA.validate(usages)
    target = part_path(catalog.usages.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    validated.write_parquet(staging)
    staging.replace(target)
    return target


def load_usages(catalog: CatalogConfig) -> pl.LazyFrame:
    """Every saved part as one lazy table.

    Raises:
        FileNotFoundError: if no part has been saved yet.
    """
    parts = sorted(catalog.usages.path.glob(f"*{_SUFFIX}"))
    if not parts:
        raise FileNotFoundError(f"no part of the usages table under {catalog.usages.path}")
    return pl.scan_parquet(parts, schema=USAGES_COLUMNS)
