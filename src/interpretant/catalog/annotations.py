"""The annotation cache: the spaCy pass's documents, saved per part with DocBin.

A rebuildable cache, tied to spaCy's version (decision agenda 4.6): later stages apply lemma
exceptions to it instead of parsing again (3.6). DocBin keeps spaCy's default attributes, which in
the installed spaCy include sentence starts and norms, and always keeps span groups, so the
removal spans survive.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

_SUFFIX = ".spacy"


def save_annotations(catalog: CatalogConfig, docs: Iterable[Doc], part: str) -> Path:
    """Write one part's documents to the cache, whole or not at all.

    Raises:
        ValueError: if the part name is not safe as a file name.
    """
    target = part_path(catalog.annotations.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    DocBin(docs=docs).to_disk(staging)
    staging.replace(target)
    return target


def load_annotations(catalog: CatalogConfig, part: str, vocab: Vocab) -> Iterator[Doc]:
    """One part's documents from the cache, on the given vocabulary.

    Raises:
        ValueError: if the part name is not safe as a file name.
        FileNotFoundError: if the part has not been cached.
    """
    path = part_path(catalog.annotations.path, part, _SUFFIX)
    if not path.exists():
        raise FileNotFoundError(f"no cached annotations for part {part!r} at {path}")
    yield from DocBin().from_disk(path).get_docs(vocab)
