"""Sentence languages by lingua, on documents with fixed sentence boundaries."""

from __future__ import annotations

import pickle

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from interpretant.corpus.language import (
    SENTENCE_LANGUAGES,
    UNDETERMINED,
    SentenceLanguage,
    SentenceLanguageDetector,
    build_language_detector,
    read_sentence_languages,
)


def sentences_doc(nlp: Language, sentences: list[str]) -> Doc:
    """A document whose sentences are exactly the given ones, split into words on spaces."""
    words: list[str] = []
    sentence_starts: list[bool | int | None] = []
    for sentence in sentences:
        sentence_words = sentence.split()
        words.extend(sentence_words)
        sentence_starts.extend([True] + [False] * (len(sentence_words) - 1))
    return Doc(nlp.vocab, words=words, sent_starts=sentence_starts)


def test_build_language_detector_requires_english() -> None:
    with pytest.raises(ValueError, match="English"):
        build_language_detector(["de", "fr"], minimum_relative_distance=0.1)


def test_build_language_detector_rejects_an_unknown_code() -> None:
    with pytest.raises(ValueError, match="xx"):
        build_language_detector(["en", "xx"], minimum_relative_distance=0.1)


def test_the_component_checks_its_language_codes_when_added() -> None:
    with pytest.raises(ValueError, match="English"):
        spacy.blank("en").add_pipe(
            SENTENCE_LANGUAGES,
            config={
                "languages": ["de"],
                "english_threshold": 0.9,
                "minimum_relative_distance": 0.1,
            },
        )


def test_the_component_leaves_sentences_without_letters_undetermined(nlp: Language) -> None:
    doc = sentences_doc(
        nlp, ["Attention mechanisms improve machine translation in many settings .", "12 34 ."]
    )
    english, numbers = read_sentence_languages(nlp.get_pipe(SENTENCE_LANGUAGES)(doc))
    assert english.language == "en"
    assert numbers == SentenceLanguage(UNDETERMINED, 0.0)


def test_the_component_pickles_without_its_detector(nlp: Language) -> None:
    component = nlp.get_pipe(SENTENCE_LANGUAGES)
    component(sentences_doc(nlp, ["Attention mechanisms improve machine translation ."]))
    copy = pickle.loads(pickle.dumps(component))
    assert isinstance(copy, SentenceLanguageDetector)
    assert (copy.languages, copy.english_threshold, copy.minimum_relative_distance) == (
        ("en", "de", "fr", "es"),
        0.9,
        0.1,
    )


def test_read_sentence_languages_refuses_a_document_the_component_has_not_seen(
    nlp: Language,
) -> None:
    with pytest.raises(ValueError, match=SENTENCE_LANGUAGES):
        read_sentence_languages(sentences_doc(nlp, ["one two three"]))


def test_the_component_labels_each_sentence_of_a_mixed_abstract(nlp: Language) -> None:
    doc = sentences_doc(
        nlp,
        [
            "Attention mechanisms improve machine translation in many settings and tasks .",
            "Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung .",
            "Les mécanismes d'attention améliorent la traduction automatique .",
        ],
    )
    languages = read_sentence_languages(nlp.get_pipe(SENTENCE_LANGUAGES)(doc))
    assert [language.language for language in languages] == ["en", "de", "fr"]


def test_the_component_leaves_a_too_close_call_undetermined(nlp: Language) -> None:
    component = SentenceLanguageDetector(["en", "de", "fr"], 0.9, minimum_relative_distance=0.9)
    doc = sentences_doc(
        nlp, ["Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung .", "prologue ."]
    )
    languages = read_sentence_languages(component(doc))
    assert languages[1] == SentenceLanguage(UNDETERMINED, 0.0)
