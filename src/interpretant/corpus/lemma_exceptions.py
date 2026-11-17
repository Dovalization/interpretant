"""The lemma exceptions as spaCy's attribute ruler, for the cached documents (agenda 3.2, 3.6).

Each exception sets one form's lemma under one reading: a single-token pattern on ``LOWER`` and
``TAG`` with the ``LEMMA`` attribute, added through the ruler's ``add`` (spaCy's
``attribute_ruler.add``, 8.2). The project's ``LookupAttributeRuler`` finds such rules by lookup,
so 3,297 of them cost no more than a few. The documents are not parsed again: the ruler runs on
documents loaded from the annotation cache, whose tags it reads.

Build the ruler on the vocab the documents load into. spaCy interns a rule's lemma in the ruler's
string store, and a token reads its lemma from its document's.
"""

from __future__ import annotations

from collections.abc import Iterable

from spacy.vocab import Vocab

from interpretant.catalog.lemma_exceptions import LemmaException
from interpretant.corpus.attribute_rules import LookupAttributeRuler

RULER_NAME = "lemma_exceptions"


def lemma_exception_ruler(
    vocab: Vocab, exceptions: Iterable[LemmaException]
) -> LookupAttributeRuler:
    """An attribute ruler setting each exception's lemma on the tokens of its form and tag."""
    ruler = LookupAttributeRuler(vocab, RULER_NAME)
    for exception in exceptions:
        ruler.add([[{"LOWER": exception.form, "TAG": exception.tag}]], {"LEMMA": exception.lemma})
    return ruler
