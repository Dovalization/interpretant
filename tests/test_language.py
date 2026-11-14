"""Sentence languages by lingua, on documents with fixed sentence boundaries."""

from __future__ import annotations

import pytest
from lingua import LanguageDetector
from spacy.language import Language
from spacy.tokens import Doc

from interpretant.corpus.language import (
    UNDETERMINED,
    SentenceLanguage,
    build_language_detector,
    detect_sentence_languages,
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


def test_detect_sentence_languages_leaves_sentences_without_letters_undetermined(
    nlp: Language, detector: LanguageDetector
) -> None:
    doc = sentences_doc(
        nlp, ["Attention mechanisms improve machine translation in many settings .", "12 34 ."]
    )
    english, numbers = detect_sentence_languages(doc, detector, english_threshold=0.9)
    assert english.language == "en"
    assert numbers == SentenceLanguage(UNDETERMINED, 0.0)


def test_detect_sentence_languages_labels_each_sentence_of_a_mixed_abstract(
    nlp: Language, detector: LanguageDetector
) -> None:
    doc = sentences_doc(
        nlp,
        [
            "Attention mechanisms improve machine translation in many settings and tasks .",
            "Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung .",
            "Les mécanismes d'attention améliorent la traduction automatique .",
        ],
    )
    languages = detect_sentence_languages(doc, detector, english_threshold=0.9)
    assert [language.language for language in languages] == ["en", "de", "fr"]


def test_detect_sentence_languages_leaves_a_too_close_call_undetermined(nlp: Language) -> None:
    strict = build_language_detector(["en", "de", "fr"], minimum_relative_distance=0.9)
    doc = sentences_doc(
        nlp, ["Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung .", "prologue ."]
    )
    languages = detect_sentence_languages(doc, strict, english_threshold=0.9)
    assert languages[1] == SentenceLanguage(UNDETERMINED, 0.0)


ENGLISH = "Attention mechanisms improve machine translation in many settings and tasks ."


def test_detect_sentence_languages_takes_the_likelier_of_two_close_foreign_languages(
    nlp: Language,
) -> None:
    detector = build_language_detector(["en", "hr", "bs"], minimum_relative_distance=0.1)
    doc = sentences_doc(
        nlp,
        [
            "Te zatim dodaje : Filozofija je plodotvorno djelovala na razvoj znanosti .",
            "Rad je zamisljen kao prilog suvremenim istrazivanjima strucnih terminologija .",
        ],
    )
    languages = detect_sentence_languages(doc, detector, english_threshold=0.9)
    assert {language.language for language in languages} <= {"hr", "bs"}


def test_detect_sentence_languages_finds_another_script_in_an_english_abstract(
    nlp: Language,
) -> None:
    detector = build_language_detector(["en", "ko", "de"], minimum_relative_distance=0.1)
    korean = "또한 다중 CPU를 사용한 MPI 병렬프로그램과 비교하였다 ."
    doc = sentences_doc(nlp, [ENGLISH, ENGLISH, ENGLISH, ENGLISH, korean])
    languages = detect_sentence_languages(doc, detector, english_threshold=0.9)
    assert [language.language for language in languages] == ["en", "en", "en", "en", "ko"]


def test_detect_sentence_languages_finds_a_long_foreign_sentence_in_an_english_abstract(
    nlp: Language, detector: LanguageDetector
) -> None:
    german = (
        "Man fand , daß weder die visuell evozierten Reaktionen noch die Latenzen sich änderten ."
    )
    doc = sentences_doc(nlp, [ENGLISH] * 5 + [german])
    languages = detect_sentence_languages(doc, detector, english_threshold=0.9)
    assert [language.language for language in languages] == ["en"] * 5 + ["de"]


def test_detect_sentence_languages_leaves_english_headings_and_references_english(
    nlp: Language, detector: LanguageDetector
) -> None:
    doc = sentences_doc(
        nlp, [ENGLISH, "Four pages .", "Acta Oto-Laryngologica : Vol. 45 , No. 2 , pp. 1-10 ."]
    )
    languages = detect_sentence_languages(doc, detector, english_threshold=0.9)
    assert [language.language for language in languages] == ["en", "en", "en"]
