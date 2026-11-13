"""Sentence languages by lingua, on documents with fixed sentence boundaries."""

from __future__ import annotations

import pytest
from lingua import LanguageDetector
from spacy.language import Language
from spacy.tokens import Doc

from interpretant.corpus.language import (
    CONFIDENCE_ATTRIBUTE,
    LANGUAGE_ATTRIBUTE,
    UNDETERMINED,
    SentenceLanguage,
    build_language_detector,
    compute_not_english_share,
    detect_sentence_languages,
    set_sentence_languages,
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
        build_language_detector(["de", "fr"])


def test_build_language_detector_rejects_an_unknown_code() -> None:
    with pytest.raises(ValueError, match="xx"):
        build_language_detector(["en", "xx"])


def test_detect_sentence_languages_leaves_sentences_without_letters_undetermined(
    nlp: Language, detector: LanguageDetector
) -> None:
    doc = sentences_doc(
        nlp, ["Attention mechanisms improve machine translation in many settings .", "12 34 ."]
    )
    ((english, numbers),) = detect_sentence_languages([doc], detector, threshold=0.5)
    assert english.language == "en"
    assert numbers == SentenceLanguage(UNDETERMINED, 0.0)


def test_compute_not_english_share_counts_judged_characters_only(nlp: Language) -> None:
    doc = sentences_doc(nlp, ["one two three", "vier fünf", "1 2"])
    languages = [
        SentenceLanguage("en", 0.9),
        SentenceLanguage("de", 0.8),
        SentenceLanguage(UNDETERMINED, 0.0),
    ]
    assert compute_not_english_share(doc, languages) == pytest.approx(9 / (13 + 9))


def test_set_sentence_languages_writes_each_sentence_attribute(nlp: Language) -> None:
    doc = sentences_doc(nlp, ["one two three", "vier fünf"])
    set_sentence_languages(doc, [SentenceLanguage("en", 0.9), SentenceLanguage("de", 0.8)])
    assert [(s._.get(LANGUAGE_ATTRIBUTE), s._.get(CONFIDENCE_ATTRIBUTE)) for s in doc.sents] == [
        ("en", 0.9),
        ("de", 0.8),
    ]
