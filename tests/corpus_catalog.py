"""A small corpus in a temporary catalog, for the corpus stages' tests.

Three raw cells with their work metadata; the cleaning rules and patterns are the repository's.
Work 1 sits in AI and computer science, work 2 is too short, work 3 is a repository-only record,
work 4 revises work 3's text (a near-duplicate) and work 5 is in a later decade.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path

from interpretant.catalog import RawCell, append_work_metadata, read_catalog
from interpretant.pipelines.annotate_corpus import AnnotateCorpusSettings
from interpretant.pipelines.params import CorpusSelection
from tests.metadata_records import metadata_record

PARAGRAPH = (
    "Attention mechanisms let a neural network weigh the parts of its input that matter most for "
    "the current prediction. They were first used in machine translation, where the decoder looks "
    "back at every encoded word instead of a single summary vector. Since then the idea has spread "
    "to speech recognition, image captioning and protein structure prediction, and it now sits at "
    "the centre of the transformer architecture. We review how the term attention moved from "
    "psychology into computer science, compare the main families of attention mechanisms, and ask "
    "which properties of human attention the computational versions keep and which they discard. "
    "We close with open questions about interpretability and the cost of long inputs."
)
REVISED = PARAGRAPH.replace("We close with", "We end with")
SHARED = (
    "Language models learn word senses from the contexts in which the words appear, and those "
    "senses drift as fields borrow terms from one another over the decades."
)
CELLS = {
    ("ai", 2010): [(1, SHARED, 2012), (2, "Short note here.", 2012), (3, PARAGRAPH, 2012)],
    ("compsci", 2010): [(1, SHARED, 2012), (4, REVISED, 2015)],
    ("ai", 2020): [(5, SHARED, 2021)],
}
# The two 2010s cells, read by English and German candidates.
SELECTION = CorpusSelection(fields=("ai", "compsci"), decades=(2010,))
ANNOTATE_SETTINGS = AnnotateCorpusSettings(languages=("en", "de"))


def write_corpus_catalog(folder: Path) -> Path:
    """Write the catalog file, the raw cells and their metadata under ``folder``."""
    repository = read_catalog()
    catalog_file = folder / "catalog.yaml"
    catalog_file.write_text(
        "raw_abstracts:\n  path: raw\n  format: jsonl.gz\n"
        "work_metadata:\n  path: metadata\n  format: jsonl.gz\n"
        "annotated_abstracts:\n  path: annotated\n  format: parquet\n"
        "abstracts:\n  path: abstracts\n  format: parquet\n"
        f"cleaning_rules:\n  path: '{repository.cleaning_rules.path}'\n  format: yaml\n"
        f"cleaning_patterns:\n  path: '{repository.cleaning_patterns.path}'\n  format: jsonl\n"
        "annotations:\n  path: annotations\n  format: docbin\n"
    )
    for (field, decade), works in CELLS.items():
        write_raw_cell(catalog_file, field, decade, works)
    return catalog_file


def write_raw_cell(
    catalog_file: Path, field: str, decade: int, works: Sequence[tuple[int, str, int]]
) -> None:
    """One raw cell of (work, text, year) entries, and its metadata (work 3: repository-only)."""
    path = catalog_file.parent / "raw" / field / f"{decade}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as cell:
        for work, text, year in works:
            record = {
                "id": f"https://openalex.org/W{work}",
                "year": year,
                "type": "article",
                "authors": [f"https://openalex.org/A{work}"],
                "abstract": text,
            }
            cell.write(json.dumps(record) + "\n")
    records = [
        metadata_record(work, source_type="repository", indexed_in=())
        if work == 3
        else metadata_record(work)
        for work, _, _ in works
    ]
    append_work_metadata(read_catalog(catalog_file), RawCell(field, decade), records)
