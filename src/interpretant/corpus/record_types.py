"""What kind of record each work is, from the staging tables' signals (decision agenda 2.3).

The record-type batch finds the kept works that are not the work's own abstract (errata,
obituaries, book reviews, news, letters) from several sources that disagree, so each source's
value is kept beside the others: OpenAlex's type and title, Crossref's type and whether the
record is itself an update (a correction or retraction notice), PubMed's publication types, the
page count, Crossref's where it has one, PubMed's otherwise, and how the text opens ("To the
Editor"). The kinds, what shows each and what is done with it are in the curated record kinds
file; a work showing several kinds takes the first in the file's order, and its kind's action
keeps, flags or drops it.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from interpretant.catalog import RecordKind
from interpretant.corpus.cleaning import add_flag, mark_dropped

RECORD_SIGNAL_COLUMNS = pl.Schema(
    {
        "work_id": pl.String(),
        "title": pl.String(),
        "publication_year": pl.Int64(),
        "openalex_type": pl.String(),
        "is_paratext": pl.Boolean(),
        "crossref_type": pl.String(),
        "update_types": pl.List(pl.String()),
        "pubmed_types": pl.List(pl.String()),
        "page_count": pl.Int64(),
        "page_count_source": pl.String(),
        "opening": pl.String(),
    }
)
# Enough of the text for a letter's salutation after an "Abstract" label.
OPENING_LENGTH = 100
_OPENING_LABEL = r"^\s*(?:abstract\W*)?"
# An obituary's title is a name and a life span: "GEORGE ROSEN (1910–1977)", "Charles Donald
# O'Malley. 1 April 1907–7 April 1970". A history title with a period has a phrase before it
# ("Canada's Immigration Policy, 1957-1962"), told apart by its lower-case or function words.
_YEAR = r"(1[6-9]\d\d|20\d\d)"
_DAY = r"(?:\d{1,2}\s+[A-Za-z]+\.?\s+)?"
_NAME_AND_LIFE_SPAN = (
    rf"^\s*([^\d()\[\]]+?)[\s,.:;(\[—–-]*"
    rf"{_DAY}{_YEAR}\s*[-–—]+\s*{_DAY}{_YEAR}\s*[)\]]?\s*\.?\s*$"
)
_NAME_PARTICLES = ["de", "van", "von", "der", "den", "da", "du", "le", "la", "di", "del", "ten"]
_FUNCTION_WORDS = [
    "and", "of", "in", "the", "for", "on", "to", "a", "an", "with", "at", "from", "by", "&",
    "its", "his", "her", "their",
]  # fmt: skip
_NAME_WORDS = (1, 8)
_LIFE_SPAN_YEARS = (20, 110)


def join_record_signals(
    works: pl.DataFrame, openalex: pl.DataFrame, crossref: pl.DataFrame, pubmed: pl.DataFrame
) -> pl.DataFrame:
    """Each work's record-type signals, one row per work in ``works`` and in its order.

    ``works`` needs ``work_id`` and ``text`` columns; the other three are the staging tables. A
    source that holds no record for a work, or did not find it, leaves its columns null.
    """
    found_crossref = crossref.filter("is_found").select(
        "work_id", "crossref_type", "update_types", pl.col("page_count").alias("crossref_pages")
    )
    found_pubmed = pubmed.filter("is_found").select(
        "work_id",
        pl.col("publication_types").alias("pubmed_types"),
        pl.col("page_count").alias("pubmed_pages"),
    )
    joined = (
        works.select("work_id", pl.col("text").str.slice(0, OPENING_LENGTH).alias("opening"))
        .join(
            openalex.select("work_id", "title", "publication_year", "openalex_type", "is_paratext"),
            on="work_id",
            how="left",
        )
        .join(found_crossref, on="work_id", how="left")
        .join(found_pubmed, on="work_id", how="left")
    )
    return joined.select(
        "work_id",
        "title",
        "publication_year",
        "openalex_type",
        "is_paratext",
        "crossref_type",
        "update_types",
        "pubmed_types",
        pl.coalesce("crossref_pages", "pubmed_pages").alias("page_count"),
        pl.when(pl.col("crossref_pages").is_not_null())
        .then(pl.lit("crossref"))
        .when(pl.col("pubmed_pages").is_not_null())
        .then(pl.lit("pubmed"))
        .alias("page_count_source"),
        "opening",
    )


def decide_record_types(signals: pl.DataFrame, kinds: Mapping[str, RecordKind]) -> pl.DataFrame:
    """Each work's record type: the first kind, in order, whose signals it shows; else null."""
    shown = [pl.when(_shows(kind)).then(pl.lit(name)) for name, kind in kinds.items()]
    record_type = pl.coalesce(shown) if shown else pl.lit(None, dtype=pl.String())
    return signals.select("work_id", record_type.alias("record_type"))


def mark_record_types(
    abstracts: pl.DataFrame, record_types: pl.DataFrame, kinds: Mapping[str, RecordKind]
) -> pl.DataFrame:
    """Apply each kind's action to the works of that type: drop (the kind is the reason) or flag.

    A kept kind changes nothing; a work already dropped by an earlier rule keeps its reason.
    """
    for name, kind in kinds.items():
        if kind.action == "keep":
            continue
        work_ids = record_types.filter(pl.col("record_type") == name)["work_id"]
        if kind.action == "drop":
            abstracts = mark_dropped(abstracts, work_ids, reason=name)
        else:
            abstracts = add_flag(abstracts, work_ids.to_list(), flag=name)
    return abstracts


def _has_obituary_title(within_years: int) -> pl.Expr:
    """Whether the title is a name, then a life span ending ``within_years`` or less before."""
    title = pl.col("title")
    name = title.str.extract(_NAME_AND_LIFE_SPAN, 1).str.strip_chars(" ,.:;-–—")
    born = title.str.extract(_NAME_AND_LIFE_SPAN, 2).cast(pl.Int64())
    died = title.str.extract(_NAME_AND_LIFE_SPAN, 3).cast(pl.Int64())
    words = name.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    is_name_word = pl.element().str.contains(r"^[\p{Lu}\"“'(]") | pl.element().is_in(
        _NAME_PARTICLES
    )
    is_function_word = pl.element().str.to_lowercase().is_in(_FUNCTION_WORDS)
    return (
        (pl.col("publication_year") - died).is_between(0, within_years)
        & (died - born).is_between(*_LIFE_SPAN_YEARS)
        & words.list.len().is_between(*_NAME_WORDS)
        & words.list.eval(is_name_word).list.all()
        & ~words.list.eval(is_function_word).list.any()
    )


def _shows(kind: RecordKind) -> pl.Expr:
    """Whether a work shows any of the kind's signals; a missing value shows nothing."""
    signals: list[pl.Expr] = []
    if kind.openalex_types:
        signals.append(pl.col("openalex_type").is_in(sorted(kind.openalex_types)))
    if kind.crossref_update_types:
        signals.append(_holds_any("update_types", kind.crossref_update_types))
    if kind.pubmed_types:
        signals.append(_holds_any("pubmed_types", kind.pubmed_types))
    if kind.pubmed_types_together:
        signals.append(_holds_all("pubmed_types", kind.pubmed_types_together))
    if kind.title_patterns:
        signals.append(pl.col("title").str.contains(f"(?i){_either(kind.title_patterns)}"))
    if kind.opening_patterns:
        opens = f"(?i){_OPENING_LABEL}(?:{_either(kind.opening_patterns)})"
        signals.append(pl.col("opening").str.contains(opens))
    if kind.life_span_within_years is not None:
        signals.append(_has_obituary_title(kind.life_span_within_years))
    if kind.max_pages is not None:
        signals.append(pl.col("page_count") <= kind.max_pages)
    return pl.any_horizontal(signal.fill_null(value=False) for signal in signals)


def _either(patterns: tuple[str, ...]) -> str:
    return "|".join(f"(?:{pattern})" for pattern in patterns)


def _holds_any(column: str, values: frozenset[str]) -> pl.Expr:
    return pl.col(column).list.eval(pl.element().is_in(sorted(values))).list.any()


def _holds_all(column: str, values: frozenset[str]) -> pl.Expr:
    return pl.all_horizontal(pl.col(column).list.contains(value) for value in sorted(values))
