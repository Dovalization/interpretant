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
    entity_over_target,
    in_named_instrument,
    load_entity_recognizer,
    recognise_sentence_entities,
    require_entity_annotation,
)
from interpretant.usages.usage_selection import NAME_LABELS

VOCAB = spacy.blank("en").vocab


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
    [sentence] = recognise_sentence_entities([second], recognizer, batch_size=8)
    labels = {entity.text: entity.label_ for entity in sentence.ents}
    assert labels.get("John Smith") == "PERSON"
    network = sentence.text.index("network")
    assert entity_label(sentence, network, network + len("network")) is None


def test_recognise_sentence_entities_gives_each_sentence_its_own_result_in_order(
    recognizer: Language,
) -> None:
    texts = ["John Smith built it.", "The network grew.", "They moved to Atlanta."]
    spans = [recognizer.make_doc(text)[:] for text in texts]
    alone = [next(recognise_sentence_entities([span], recognizer, batch_size=1)) for span in spans]
    batched = list(recognise_sentence_entities(spans, recognizer, batch_size=2))
    assert [doc.text for doc in batched] == texts
    assert [[(e.text, e.label_) for e in doc.ents] for doc in batched] == [
        [(e.text, e.label_) for e in doc.ents] for doc in alone
    ]


def test_entity_label_reads_the_entity_overlapping_the_target(recognizer: Language) -> None:
    doc = recognizer.make_doc("They visited the Cable News Network in Atlanta.")
    [sentence] = recognise_sentence_entities([doc[:]], recognizer, batch_size=8)
    network = sentence.text.index("Network")
    assert entity_label(sentence, network, network + len("Network")) == "ORG"


def over(words: list[str], ents: list[str], target: str) -> tuple[str | None, bool]:
    """The entity over the target in a sentence with these entity tags (IOB)."""
    doc = Doc(VOCAB, words=words, ents=ents)
    start = doc.text.index(target)
    return entity_over_target(doc, start, start + len(target), NAME_LABELS)


def test_an_entity_with_other_words_puts_the_target_inside_a_name() -> None:
    words = ["the", "Cable", "News", "Network", "grew"]
    ents = ["O", "B-ORG", "I-ORG", "I-ORG", "O"]
    assert over(words, ents, "Network") == ("ORG", True)


def test_an_entity_that_is_the_target_alone_is_a_generic_term_not_a_name() -> None:
    words = ["work", "in", "the", "Artificial", "Intelligence", "field"]
    ents = ["O", "O", "B-ORG", "I-ORG", "I-ORG", "O"]
    assert over(words, ents, "Artificial Intelligence") == ("ORG", False)
    assert over(["Cortex", "was", "thin"], ["B-PRODUCT", "O", "O"], "Cortex") == ("PRODUCT", False)


def test_a_product_ending_in_the_target_is_generic_but_one_holding_it_inside_is_a_name() -> None:
    words = ["a", "User", "Interface", "Management", "System"]
    ents = ["O", "B-PRODUCT", "I-PRODUCT", "I-PRODUCT", "I-PRODUCT"]
    assert over(words, ents, "Interface") == ("PRODUCT", True)
    assert over(words, ents, "System") == ("PRODUCT", False)
    organisation = ["O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]
    assert over(words, organisation, "System") == ("ORG", True)


def test_no_entity_or_one_without_a_name_label_is_no_name() -> None:
    assert over(["the", "network", "grew"], ["O", "O", "O"], "network") == (None, False)
    words = ["the", "1990s", "network"]
    assert over(words, ["B-DATE", "I-DATE", "O"], "network") == (None, False)
    assert over(["network", "1990s"], ["B-DATE", "I-DATE"], "network") == ("DATE", False)


def instrument(words: list[str], target: str) -> bool:
    """Whether the target sits in a named instrument in a sentence of these tokens."""
    doc = Doc(VOCAB, words=words)
    start = doc.text.index(target)
    return in_named_instrument(doc, start, start + len(target))


def test_a_capitalised_run_ending_in_an_instrument_word_is_a_name() -> None:
    assert instrument(["the", "Wechsler", "Memory", "Scale", "was", "used"], "Memory")
    words = ["the", "Strengths", "and", "Difficulties", "Questionnaire", "(", "SDQ", ")"]
    assert instrument(words, "Strengths")
    assert instrument(["the", "State", "-", "Trait", "Anxiety", "Inventory"], "State")


def test_a_target_outside_such_a_run_is_not_in_a_named_instrument() -> None:
    assert not instrument(["a", "memory", "test", "was", "used"], "memory")
    assert not instrument(["Memory", "improved", "on", "the", "Test"], "Memory")
    assert not instrument(["the", "Memory", "Test"], "Test")
    assert not instrument(["the", "Memory", "Model"], "Memory")


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
