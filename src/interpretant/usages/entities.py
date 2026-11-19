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

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import spacy
from spacy.language import Language
from spacy.pipeline import AttributeRuler, EntityRecognizer
from spacy.tokens import Doc, Span

from interpretant.corpus.attribute_rules import LOOKUP_ATTRIBUTE_RULER, LookupAttributeRuler

SMALL_MODEL = "en_core_web_sm"
TRANSFORMER_MODEL = "en_core_web_trf"
# Spans per transformer forward pass: 128 keeps four processes within the GPU's 12 GB.
SPANS_PER_FORWARD = 128


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
    """The transformer without its parser, in float16, with the lookup attribute ruler.

    Measured on target sentences (`docs/performance.md`, 2026-11-18): float16 autocast on the GPU's
    tensor cores changed 5 tags and 4 lemmas in 192,828 tokens and no entity; the lookup attribute
    ruler gives spaCy's annotation a fifth faster. Run it on sentences sorted by length, in a few
    processes sharing the GPU, each with its own model.
    """
    nlp = spacy.load(
        TRANSFORMER_MODEL,
        exclude=["parser"],
        config={
            "components": {
                "transformer": {
                    "model": {
                        "mixed_precision": True,
                        "with_spans": {"batch_size": SPANS_PER_FORWARD},
                    }
                }
            }
        },
    )
    _use_lookup_attribute_ruler(nlp)
    # The transformer's output stays on each document, about 0.35 MB of GPU memory a token; it is
    # dropped once the tagger and the recognizer have read it (spaCy's "doc_cleaner").
    nlp.add_pipe("doc_cleaner", config={"attrs": {"tensor": None, "_.trf_data": None}})
    return nlp


def _use_lookup_attribute_ruler(nlp: Language) -> None:
    """Replace the model's attribute ruler with the lookup one, without the parser's rules."""
    ruler = nlp.get_pipe("attribute_ruler")
    if not isinstance(ruler, AttributeRuler):
        raise TypeError(f"expected spaCy's attribute ruler, got {type(ruler).__name__}")
    rules = [rule for rule in ruler.patterns if not _tests_dependency(dict(rule))]
    lookup = nlp.replace_pipe("attribute_ruler", LOOKUP_ATTRIBUTE_RULER)
    if not isinstance(lookup, LookupAttributeRuler):
        raise TypeError(f"expected the lookup attribute ruler, got {type(lookup).__name__}")
    lookup.add_patterns(rules)


def _tests_dependency(rule: dict[str, Any]) -> bool:
    """Whether one of an attribute rule's token patterns tests the dependency label."""
    patterns: list[list[dict[str, Any]]] = rule["patterns"]
    return any("DEP" in token for pattern in patterns for token in pattern)


def require_entity_annotation(doc: Doc) -> None:
    """Refuse a document without entity annotation.

    Raises:
        MissingEntityAnnotationError: if no token has an entity label or an explicit outside one.
    """
    if not doc.has_annotation("ENT_IOB"):
        raise MissingEntityAnnotationError("the document holds no entity annotation")


def recognise_sentence_entities(
    sentences: Iterable[Span], recognizer: Language, *, batch_size: int
) -> Iterator[Doc]:
    """Each sentence as its own document, with entities recognised on it alone, in batches.

    Raises:
        TypeError: if the recognizer's ``ner`` is not spaCy's entity recognizer.
        MissingEntityAnnotationError: if recognition left a sentence without annotation.
    """
    ner = recognizer.get_pipe("ner")
    if not isinstance(ner, EntityRecognizer):
        raise TypeError(f"expected spaCy's entity recognizer, got {type(ner).__name__}")
    docs = (_without_entities(sentence) for sentence in sentences)
    for doc in ner.pipe(docs, batch_size=batch_size):
        require_entity_annotation(doc)
        yield doc


def _without_entities(sentence: Span) -> Doc:
    """The sentence as its own document, its entity labels set to missing for recognition."""
    doc = sentence.as_doc()
    doc.set_ents([], default="missing")
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
