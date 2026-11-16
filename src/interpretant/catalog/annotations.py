"""The annotation cache: the spaCy pass's documents, saved per part with DocBin.

A rebuildable cache, tied to spaCy's version (decision agenda 4.6): the language and cleaning
stages read it, and later stages apply lemma exceptions to it instead of parsing again (3.6).
DocBin keeps spaCy's default attributes, which in the installed spaCy include sentence starts and
norms; ``store_user_data`` keeps each document's ``user_data``, where the work id is.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from srsly.msgpack import msgpack_decoders, msgpack_encoders

from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.parts import part_path

_SUFFIX = ".spacy"
# The key of ``Doc.user_data`` holding each cached document's work id.
WORK_ID = "work_id"
# The key of ``Doc.user_data`` holding the hash of the pass and the text a document was made from,
# by which a later pass reuses it (pipelines.annotate_corpus).
TEXT_KEY = "text_key"


def save_annotations(catalog: CatalogConfig, docs: Iterable[Doc], part: str) -> Path:
    """Write one part's documents to the cache, whole or not at all.

    Raises:
        ValueError: if the part name is not safe as a file name.
    """
    target = part_path(catalog.annotations.path, part, _SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{_SUFFIX}.partial")
    DocBin(docs=docs, store_user_data=True).to_disk(staging)
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
    cache_serialiser_extensions()
    yield from DocBin().from_disk(path).get_docs(vocab)


def cache_serialiser_extensions() -> None:
    """Look srsly's msgpack extensions up once in this process, instead of on every unpack.

    srsly asks its two catalogue registries for extension encoders and decoders each time it
    packs or unpacks, and each ask rescans the installed packages' entry points and filters
    catalogue's global registry, where spaCy keeps thousands of functions: about a third of the
    spaCy pass's main process, which unpacks every finished document (profiled 2026-11-13), and
    in run 7c's profile 28% of the language stage and 10% of the cleaning workers, which read
    the cache. The two registries' contents (srsly's numpy functions; no installed package adds
    any) are read once and served from then on; a function registered later would be missed.
    Worker processes started by fork inherit the cache; those started by spawn set it again when
    they read the cache. Calling it again changes nothing.
    """
    for registry in (msgpack_encoders, msgpack_decoders):
        functions = registry.get_all()
        registry.get_all = lambda functions=functions: functions
