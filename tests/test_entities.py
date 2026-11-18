"""Entities on target sentences: recognised on the sentence alone, refused when missing."""

from __future__ import annotations

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from interpretant.usages.entities import (
    MissingEntityAnnotationError,
    annotate_with_transformer,
    entity_label,
    load_entity_recognizer,
    recognise_sentence_entities,
    require_entity_annotation,
)


@pytest.fixture(scope="module")
def recognizer() -> Language:
    """The small model's entity recognizer."""
    return load_entity_recognizer()


def test_require_entity_annotation_refuses_a_document_without_it() -> None:
    doc = spacy.blank("en")("Attention is all you need.")
    with pytest.raises(MissingEntityAnnotationError):
        require_entity_annotation(doc)


def test_recognise_sentence_entities_finds_names_on_the_sentence_alone(
    recognizer: Language,
) -> None:
    doc = recognizer.make_doc(
        "Our model was trained at Google. John Smith built the network there."
    )
    doc.set_ents([], default="outside")
    second = doc[7:]
    sentence = recognise_sentence_entities(second, recognizer)
    labels = {entity.text: entity.label_ for entity in sentence.ents}
    assert labels.get("John Smith") == "PERSON"
    network = sentence.text.index("network")
    assert entity_label(sentence, network, network + len("network")) is None


def test_entity_label_reads_the_entity_overlapping_the_target(recognizer: Language) -> None:
    doc = recognizer.make_doc("They visited the Cable News Network in Atlanta.")
    sentence = recognise_sentence_entities(doc[:], recognizer)
    network = sentence.text.index("Network")
    assert entity_label(sentence, network, network + len("Network")) == "ORG"


def test_annotate_with_transformer_rebuilds_the_tokens_and_applies_the_exceptions() -> None:
    stand_in = spacy.blank("en")
    ruler = stand_in.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "ORG", "pattern": "Google"}])  # type: ignore[attr-defined]
    source = stand_in.make_doc("Models at Google learn.")
    seen: list[str] = []

    def exceptions(doc: Doc) -> Doc:
        seen.append(doc.text)
        return doc

    annotated = annotate_with_transformer(source[:], stand_in, exceptions)
    assert [token.text for token in annotated] == [token.text for token in source]
    assert seen == ["Models at Google learn."]
    assert [(entity.text, entity.label_) for entity in annotated.ents] == [("Google", "ORG")]
