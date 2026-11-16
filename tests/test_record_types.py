"""The record-type signals joined from the staging tables, the kinds decided from them, and
what each kind's action does."""

from __future__ import annotations

import polars as pl
import pytest

from interpretant.catalog.config import read_catalog
from interpretant.catalog.curated import RecordKind, load_record_kinds
from interpretant.catalog.schemas import (
    CROSSREF_WORKS_COLUMNS,
    OPENALEX_WORKS_COLUMNS,
    PUBMED_WORKS_COLUMNS,
)
from interpretant.corpus.record_types import (
    RECORD_SIGNAL_COLUMNS,
    decide_record_types,
    join_record_signals,
    mark_record_types,
)


def staging(columns: pl.Schema, rows: list[dict[str, object]]) -> pl.DataFrame:
    """A staging table with only the given values set, the rest null."""
    return pl.DataFrame([{name: row.get(name) for name in columns} for row in rows], schema=columns)


def signals(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Record-type signals with only the given values set, the rest null."""
    return staging(RECORD_SIGNAL_COLUMNS, rows)


def work(number: int) -> str:
    return f"https://openalex.org/W{number}"


def test_join_record_signals_keeps_each_source_and_prefers_crossrefs_page_count() -> None:
    works = pl.DataFrame({"work_id": [work(1), work(2), work(3)]})
    openalex = staging(
        OPENALEX_WORKS_COLUMNS,
        [
            {"work_id": work(1), "title": "Errata", "openalex_type": "article",
             "publication_year": 1974},
            {"work_id": work(2), "title": "Letter", "openalex_type": "article"},
            {"work_id": work(3), "title": "Lost", "openalex_type": "article"},
        ],
    )  # fmt: skip
    crossref = staging(
        CROSSREF_WORKS_COLUMNS,
        [
            {"work_id": work(1), "is_found": True,
             "crossref_type": "journal-article", "update_types": ["correction"], "page_count": 1},
            {"work_id": work(2), "is_found": False, "page_count": None},
        ],
    )  # fmt: skip
    pubmed = staging(
        PUBMED_WORKS_COLUMNS,
        [
            {"work_id": work(1), "is_found": True,
             "publication_types": ["Published Erratum"], "page_count": 3},
            {"work_id": work(2), "is_found": True,
             "publication_types": ["Letter"], "page_count": 2},
        ],
    )  # fmt: skip
    joined = join_record_signals(works, openalex, crossref, pubmed)
    assert joined.schema == RECORD_SIGNAL_COLUMNS
    first, second, third = joined.to_dicts()
    assert (first["update_types"], first["pubmed_types"]) == (["correction"], ["Published Erratum"])
    assert (first["publication_year"], first["page_count"], first["page_count_source"]) == (
        1974,
        1,
        "crossref",
    )
    assert (second["crossref_type"], second["pubmed_types"]) == (None, ["Letter"])
    assert (second["page_count"], second["page_count_source"]) == (2, "pubmed")
    assert (third["title"], third["page_count"], third["page_count_source"]) == ("Lost", None, None)


KINDS = {
    "erratum": RecordKind(
        action="drop",
        openalex_types=frozenset({"erratum"}),
        crossref_update_types=frozenset({"correction"}),
        pubmed_types=frozenset({"Published Erratum"}),
    ),
    "letter": RecordKind(
        action="flag", pubmed_types=frozenset({"Letter"}), title_patterns=(r"^letters$",)
    ),
    "short_record": RecordKind(action="keep", max_pages=2),
}


def test_decide_record_types_takes_the_first_kind_a_work_shows() -> None:
    table = signals(
        [
            {"work_id": work(1), "pubmed_types": ["Letter", "Published Erratum"], "page_count": 1},
            {"work_id": work(2), "title": "LETTERS", "page_count": 1},
            {"work_id": work(3), "update_types": ["new_version", "correction"]},
            {"work_id": work(4), "openalex_type": "erratum"},
            {"work_id": work(5), "page_count": 2},
            {"work_id": work(6), "title": "Letters on the Kantian philosophy", "page_count": 9},
            {"work_id": work(7)},
        ]
    )
    decided = decide_record_types(table, KINDS)
    assert decided["work_id"].to_list() == [work(number) for number in range(1, 8)]
    assert decided["record_type"].to_list() == [
        "erratum",
        "letter",
        "erratum",
        "erratum",
        "short_record",
        None,
        None,
    ]


def test_mark_record_types_drops_flags_or_keeps_each_kind() -> None:
    abstracts = pl.DataFrame(
        {
            "work_id": [work(1), work(2), work(3), work(4)],
            "flags": [[], [], [], []],
            "dropped_reason": [None, None, None, "too_short"],
        },
        schema={
            "work_id": pl.String(),
            "flags": pl.List(pl.String()),
            "dropped_reason": pl.String(),
        },
    )
    record_types = pl.DataFrame(
        {"work_id": [work(1), work(2), work(3), work(4)],
         "record_type": ["erratum", "letter", "short_record", "erratum"]}
    )  # fmt: skip
    marked = mark_record_types(abstracts, record_types, KINDS)
    assert marked["dropped_reason"].to_list() == ["erratum", None, None, "too_short"]
    assert marked["flags"].to_list() == [[], ["letter"], [], []]


@pytest.mark.parametrize(
    ("title", "year", "kind"),
    [
        ("Errata", 1974, "erratum"),
        ("Errata to: How Mechanics Shaped the Modern World", 2014, "erratum"),
        ("Correction: Balancing Act: A Comprehensive Review of Vestibular Evaluation", 2025,
         "erratum"),
        ("NECROLOGY", 1953, "obituary"),
        ("GEORGE ROSEN (1910–1977)", 1977, "obituary"),
        ("Charles Donald O'Malley. 1 April 1907–7 April 1970", 1970, "obituary"),
        ("Professor Eric William Hawkins (8 January 1915 – 31 October 2010)", 2011, "obituary"),
        ("Werner Schmid, 1930–2002", 2010, None),
        ("Canada's Immigration Policy, 1957-1962", 1963, None),
        ("History of Langston, Oklahoma, 1890-1950", 1952, None),
        ("Review of Dahl (1985): Tense and Aspect Systems", 1987, "book_review"),
        ("THE JUVENILE IN DELINQUENT SOCIETY. By Milton L. Barron. New York: Alfred A. Knopf, "
         "1954. 349 pp. $5.00", 1955, "book_review"),
        ("Review of the literature on the dangers of zolpidem use", 2022, None),
        ("America's children and public policy in the 1980s.", 1984, None),
        ("'Kinship and Credit among the Nuer' (Africa, October 1971, pp. 306–19). A Comment by "
         "J. Y. Okamura", 1972, None),
        ("News and Comment", 1957, "news"),
        ("News Coverage of the Gulf Crisis and Public Opinion", 1993, None),
        ("News : a reader", 1999, None),
        ("LETTERS", 1964, "letter"),
        ("Letter from Dr. Koch to FA Davis Company", 1956, "letter"),
        ("LETTER XXIV. Newton to Cotes, June 7, 1711", 2014, "letter"),
        ("Letter Cueing in Oral Translation Exercises", 1975, None),
        ("Guest Editorial", 2000, "editorial"),
        ("Reply", 1980, "comment"),
    ],
)  # fmt: skip
def test_the_repository_kinds_read_titles_as_their_census_showed(
    title: str, year: int, kind: str | None
) -> None:
    table = signals([{"work_id": work(1), "title": title, "publication_year": year}])
    decided = decide_record_types(table, load_record_kinds(read_catalog()))
    assert decided["record_type"].to_list() == [kind]


@pytest.mark.parametrize(
    ("pubmed_types", "kind"),
    [
        (["Letter", "Comment"], "letter"),
        (["Comment", "Journal Article", "Letter"], "letter"),
        (["Letter"], None),
        (["Letter", "Research Support, Non-U.S. Gov't"], None),
        (["Comment"], "comment"),
        (None, None),
    ],
    ids=[
        "a reply to an article",
        "a reply among other types",
        "a research letter",
        "a funded research letter",
        "a commentary",
        "not in PubMed",
    ],
)
def test_the_repository_kinds_take_a_pubmed_letter_only_as_a_reply(
    pubmed_types: list[str] | None, kind: str | None
) -> None:
    table = signals([{"work_id": work(1), "pubmed_types": pubmed_types}])
    decided = decide_record_types(table, load_record_kinds(read_catalog()))
    assert decided["record_type"].to_list() == [kind]
