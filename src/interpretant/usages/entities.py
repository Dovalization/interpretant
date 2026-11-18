"""Named entities on target sentences, and the transformer's annotation of them (agenda 4.1, 4.9).

The corpus pass leaves named entity recognition out, so entities are recognised here on each target
sentence alone, as agenda 4.1 decided from its own measurement (99.87% agreement with the whole
abstract on 15,018 mentions); the model comparison re-checks sentence against abstract on its
sample. The small model's ``ner`` has its own embedding layer, so it runs on the cached tokens
without re-tagging. A document must hold entity annotation before matching reads it: the cached
documents' empty labels would read as "no entity", so the sentence's labels are first set to
missing, and a document without annotation is refused rather than read as having no names.

The transformer (``en_core_web_trf``) re-annotates a target sentence's tags, lemmas and entities
within the small model's boundaries, one sentence per ``Doc`` (it has no sentence splitter), with
the lemma exceptions applied after it, for the comparison of agenda 4.9 and, if it wins, the stage.
"""

from __future__ import annotations

from collections.abc import Callable

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

SMALL_MODEL = "en_core_web_sm"
TRANSFORMER_MODEL = "en_core_web_trf"


class MissingEntityAnnotationError(ValueError):
    """A document read for names holds no entity annotation."""


def load_entity_recognizer() -> Language:
    """The small model with only its entity recognizer enabled."""
    nlp = spacy.load(SMALL_MODEL)
    for name in nlp.pipe_names:
        if name != "ner":
            nlp.disable_pipe(name)
    return nlp


def load_transformer() -> Language:
    """The transformer model without its parser (dependencies are not read)."""
    return spacy.load(TRANSFORMER_MODEL, exclude=["parser"])


def require_entity_annotation(doc: Doc) -> None:
    """Refuse a document without entity annotation.

    Raises:
        MissingEntityAnnotationError: if no token has an entity label or an explicit outside one.
    """
    if not doc.has_annotation("ENT_IOB"):
        raise MissingEntityAnnotationError("the document holds no entity annotation")


def recognise_sentence_entities(sentence: Span, recognizer: Language) -> Doc:
    """The sentence as its own document, with entities recognised on it alone.

    Raises:
        MissingEntityAnnotationError: if recognition left no annotation.
    """
    doc = sentence.as_doc()
    doc.set_ents([], default="missing")
    doc = recognizer.get_pipe("ner")(doc)
    require_entity_annotation(doc)
    return doc


def annotate_with_transformer(
    sentence: Span, transformer: Language, exceptions: Callable[[Doc], Doc]
) -> Doc:
    """The sentence's tokens re-annotated by the transformer, then the lemma exceptions.

    Raises:
        MissingEntityAnnotationError: if the transformer left no entity annotation.
    """
    doc = Doc(
        transformer.vocab,
        words=[token.text for token in sentence],
        spaces=[bool(token.whitespace_) for token in sentence],
    )
    doc = exceptions(transformer(doc))
    require_entity_annotation(doc)
    return doc


def entity_label(doc: Doc, start_char: int, end_char: int) -> str | None:
    """The label of the entity overlapping the span ``[start_char, end_char)``, if any."""
    for entity in doc.ents:
        if entity.start_char < end_char and start_char < entity.end_char:
            return str(entity.label_)
    return None
