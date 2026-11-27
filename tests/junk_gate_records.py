"""Synthetic records behind drawn usages for the junk gate's tests, written into a catalog.

Each record is an abstract of three sentences, prose or junk, whose middle sentence holds the
drawn usages: *attention* (a development word) and, for some records, *network* (a held-out word
under a sealed name). A junk record's middle sentence is a run-on list with OCR damage, it has no
finite verb, and its vector leans along one dimension where a prose record's leans along another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from interpretant.catalog.abstracts import save_abstracts
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.junk_gate import JunkLabel
from interpretant.catalog.record_vectors import RECORD_VECTORS_COLUMNS, VECTOR_SIZE
from interpretant.catalog.record_vectors import save_record_vectors as save_vectors
from interpretant.catalog.schemas import OPENALEX_WORKS_COLUMNS
from interpretant.catalog.staging import save_staging_table
from interpretant.catalog.static_view import STATIC_VIEW_SCHEMA, save_static_view
from interpretant.catalog.usages import USAGES_COLUMNS, save_usages
from interpretant.usages.identifiers import build_content_key, build_draw_key
from tests.abstracts_tables import abstract_row, abstracts_table

SEALED = "held-out:0123456789ab"
PROSE = [
    "We study attention in network models.",
    "Attention improves recall in every network we tried.",
    "The results hold across several datasets and tasks.",
]
JUNK = [
    "CONTENTS OF VOLUME 3.",
    "Attention tbe qnick brovvn wvrds.-Index of every network.-Errata",
    "Notes 1-12.",
]


@dataclass(frozen=True)
class SyntheticRecord:
    """One record: its work number, year, kind, whether it holds *network*, and its fields."""

    number: int
    year: int
    is_junk: bool
    has_network: bool
    fields: tuple[str, ...] = ("ai",)


def work_id(number: int) -> str:
    return f"https://openalex.org/W{number}"


def write_words(catalog: CatalogConfig) -> None:
    """*attention* among the development words, *network* held out under its sealed name."""
    catalog.words.path.parent.mkdir(parents=True, exist_ok=True)
    catalog.words.path.write_text("lemma\tpos\tmatch\nattention\tNOUN|PROPN\tlemma\n")
    catalog.held_out_words.path.write_text(
        f"lemma\tpos\tmatch\tsealed_name\nnetwork\tNOUN\tlemma\t{SEALED}\n"
    )


def build_abstract(record: SyntheticRecord) -> dict[str, Any]:
    return abstract_row(
        record.number,
        JUNK if record.is_junk else PROSE,
        year=record.year,
        fields=list(record.fields),
    )


def build_usage(abstract: dict[str, Any], word: str) -> dict[str, Any]:
    """The drawn usage of ``word`` in the abstract's middle sentence."""
    text, sentence = abstract["text"], abstract["sentences"][1]
    form = "Attention" if word == "attention" else word
    start = text.index(form, sentence["start"])
    end = start + len(word)
    content_key = build_content_key(
        text[sentence["start"] : sentence["end"]],
        start - sentence["start"],
        end - sentence["start"],
    )
    return {
        "work_id": abstract["work_id"],
        "word": word,
        "content_key": content_key,
        "draw_key": build_draw_key(1, abstract["work_id"], word, content_key),
        "sentence_number": 1,
        "sentence_start": sentence["start"],
        "sentence_end": sentence["end"],
        "target_start": start,
        "target_end": end,
        "compound": None,
        "position": None,
        "pos": "NOUN",
        "tag": "NN",
        "morph": "Number=Sing",
        "lemma": word,
        "entity": None,
        "static_token": word,
        "sentence_word_count": 8,
        "sentence_language": "en",
        "sentence_confidence": 0.99,
        "sentence_exclusion": None,
        "inside_name": False,
        "in_scope": True,
        "drawn": True,
        "model": "en_core_web_sm",
        "windowed": False,
        "role": "head",
        "capitalised": False,
        "mention": False,
        "list_like": False,
        "copy_of": None,
    }


def build_static_entry(abstract: dict[str, Any], *, has_finite_verb: bool) -> dict[str, Any]:
    """The abstract's space-separated tokens, with a finite verb at *improves* when it has one."""
    text = abstract["text"]
    tokens, starts, position = [], [], 0
    for token in text.split(" "):
        tokens.append(token.lower())
        starts.append(text.index(token, position))
        position = starts[-1] + len(token)
    return {
        "work_id": abstract["work_id"],
        "tokens": tokens,
        "starts": starts,
        "finite_verbs": [has_finite_verb and token == "improves" for token in tokens],
    }


def build_vector(record: SyntheticRecord) -> np.ndarray:
    """A unit vector leaning along dimension 0 for junk and dimension 1 for prose."""
    generator = np.random.default_rng(record.number)
    vector = generator.standard_normal(VECTOR_SIZE) * 0.05
    vector[0 if record.is_junk else 1] += 1.0
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def write_records(
    catalog: CatalogConfig, records: Sequence[SyntheticRecord]
) -> dict[tuple[int, str], dict[str, Any]]:
    """Write the records' parts (abstracts, static view, usages, vectors) and their sources.

    A record's source is an ISSN-L for most, a source name for every seventh and nothing for
    every eleventh. Returns each drawn usage by work number and word.
    """
    usages_by_record: dict[tuple[int, str], dict[str, Any]] = {}
    decades = sorted({record.year - record.year % 10 for record in records})
    for decade in decades:
        part = f"ai_{decade}"
        in_part = [record for record in records if record.year - record.year % 10 == decade]
        abstracts, entries, usages = [], [], []
        for record in in_part:
            abstract = build_abstract(record)
            abstracts.append(abstract)
            entries.append(build_static_entry(abstract, has_finite_verb=not record.is_junk))
            for word in ("attention", "network") if record.has_network else ("attention",):
                usage = build_usage(abstract, word)
                usages.append(usage)
                usages_by_record[record.number, word] = usage
        save_abstracts(catalog, abstracts_table(abstracts), part)
        save_static_view(catalog, pl.DataFrame(entries, schema=STATIC_VIEW_SCHEMA), part)
        save_usages(catalog, pl.DataFrame(usages, schema=USAGES_COLUMNS), part)
        vectors = pl.DataFrame(
            {
                "work_id": [work_id(record.number) for record in in_part],
                "vector": np.stack([build_vector(record) for record in in_part]),
            },
            schema=RECORD_VECTORS_COLUMNS,
        )
        save_vectors(catalog, vectors, part)
    save_staging_table(catalog, "openalex", build_sources(records))
    return usages_by_record


def build_sources(records: Sequence[SyntheticRecord]) -> pl.DataFrame:
    rows = [
        {
            "work_id": work_id(record.number),
            "issn_l": None
            if record.number % 7 == 0 or record.number % 11 == 0
            else f"0000-{record.number % 25:04d}",
            "source_name": f"Journal {record.number % 5}"
            if record.number % 7 == 0 and record.number % 11 != 0
            else None,
        }
        for record in records
    ]
    return pl.DataFrame(rows, schema=OPENALEX_WORKS_COLUMNS)


def build_label(
    usage: dict[str, Any],
    item_set: str,
    *,
    decade: int,
    verdict: str,
    sentence: str = "own",
    damaged: str = "no",
) -> JunkLabel:
    """A label on a drawn usage of the decade's part, its word by its reported name."""
    return JunkLabel(
        set=item_set,
        stratum="list_like" if item_set == "enrichment" else None,
        work_id=usage["work_id"],
        part=f"ai_{decade}",
        decade=decade,
        word=SEALED if usage["word"] == "network" else usage["word"],
        content_key=usage["content_key"],
        verdict=verdict,
        kind="contents_or_list" if verdict == "junk" else "none",
        sentence=sentence,
        damaged=damaged,
    )
