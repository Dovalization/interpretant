"""Sentence languages by lingua, read from the spaCy pass's sentences.

The decision agenda, step 2.4: lingua alone, restricted to English and the languages found in the
corpus (scripts/evidence/language_scan.py); per abstract first, per sentence only for abstracts
not confidently English. It runs as its own stage on the documents in the annotation cache, so a
change to its settings reruns neither the spaCy models nor anything but the later stages.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from lingua import ConfidenceValue, IsoCode639_1, LanguageDetector, LanguageDetectorBuilder
from lingua import Language as LinguaLanguage
from spacy.tokens import Doc, Span

ENGLISH = "en"
# ISO 639-2's code for an undetermined language: a sentence with no letters, or no verdict.
UNDETERMINED = "und"
# lingua's confidence for the same text varies in its last digits from call to call, so it is
# rounded where it is read: the threshold decision and the stored table then reproduce.
CONFIDENCE_DECIMALS = 4
# In an abstract read as English, a sentence is detected on its own when it is mostly in another
# script, or has at least this many words and none of these English function words; it is taken
# as another language when lingua is this confident (2.4, from the audit of run 2: such
# sentences were Korean, Arabic, French "Résumé"s, German and Spanish).
MIN_CHECKED_WORDS = 8
FOREIGN_CONFIDENCE = 0.99
ENGLISH_FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "between",
        "by",
        "can",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "not",
        "of",
        "on",
        "or",
        "our",
        "than",
        "that",
        "the",
        "their",
        "these",
        "this",
        "to",
        "was",
        "we",
        "were",
        "which",
        "with",
    ]
)
_LETTER = re.compile(r"[^\W\d_]")
# A letter outside the Latin script (Basic Latin, Latin-1, Latin Extended A and B and Additional).
_NON_LATIN_LETTER = re.compile(r"[^\W\d_A-Za-z\u00c0-\u024f\u1e00-\u1eff]")
_WORD = re.compile(r"[^\W\d_]+")


@dataclass(frozen=True)
class SentenceLanguage:
    """A sentence's language (an ISO 639-1 code, or ``und``) and lingua's confidence in it."""

    language: str
    confidence: float


_NO_LANGUAGE = SentenceLanguage(UNDETERMINED, 0.0)


def build_language_detector(
    languages: Sequence[str], *, minimum_relative_distance: float
) -> LanguageDetector:
    """A lingua detector restricted to the given ISO 639-1 codes, which must include English.

    Restricting the candidates improves accuracy when the text's languages are among them
    (lingua's documentation). Below ``minimum_relative_distance`` between the two likeliest
    languages, detecting a text's language gives no verdict; confidence values are unaffected.

    Raises:
        ValueError: if English is missing or a code is not an ISO 639-1 code lingua knows.
    """
    return (
        LanguageDetectorBuilder.from_languages(*_lingua_languages(languages))
        .with_minimum_relative_distance(minimum_relative_distance)
        .with_preloaded_language_models()
        .build()
    )


def detect_sentence_languages(
    doc: Doc, detector: LanguageDetector, *, english_threshold: float
) -> list[SentenceLanguage]:
    """Each sentence's language and lingua's confidence in it, in order.

    When lingua gives English at least ``english_threshold`` confidence for the whole abstract,
    every sentence with letters takes that verdict, except a sentence worth checking (mostly in
    another script, or long with no English function word) that lingua reads as another
    language: in another script at any confidence, in the Latin script at
    ``FOREIGN_CONFIDENCE``. Otherwise each sentence is detected on its own, with lingua's
    confidence for the language found. When its two likeliest languages are closer than the
    detector's minimum relative distance, it is undetermined if one of them is English (short
    English fragments, headings, references) and takes the likelier one if neither is (Croatian
    and Bosnian, Indonesian and Malay). A sentence without letters is undetermined. (Until
    2026-11-13 lingua's experimental detection of several languages in one text split such
    abstracts; the corpus audit found it read real English as German or French, decision agenda
    2.4.)
    """
    whole = _top_language(detector.compute_language_confidence_values(doc.text))
    is_english = whole.language == ENGLISH and whole.confidence >= english_threshold
    return [
        _sentence_language(sentence, whole if is_english else None, detector)
        for sentence in doc.sents
    ]


def _lingua_languages(codes: Sequence[str]) -> list[LinguaLanguage]:
    """The lingua languages for ISO 639-1 codes, which must include English."""
    unique_codes = sorted({code.lower() for code in codes})
    if ENGLISH not in unique_codes:
        raise ValueError("the language detector must include English")
    unknown = [code for code in unique_codes if not hasattr(IsoCode639_1, code.upper())]
    if unknown:
        raise ValueError(f"not ISO 639-1 codes lingua knows: {unknown}")
    return [
        LinguaLanguage.from_iso_code_639_1(getattr(IsoCode639_1, code.upper()))
        for code in unique_codes
    ]


def _top_language(values: Sequence[ConfidenceValue]) -> SentenceLanguage:
    if not values:
        return _NO_LANGUAGE
    best = values[0]
    return _sentence_language_of(best.language, best.value)


def _sentence_language_of(language: LinguaLanguage, confidence: float) -> SentenceLanguage:
    return SentenceLanguage(_code(language), round(confidence, CONFIDENCE_DECIMALS))


def _sentence_language(
    sentence: Span, whole_english: SentenceLanguage | None, detector: LanguageDetector
) -> SentenceLanguage:
    text = sentence.text
    if not _LETTER.search(text):
        return _NO_LANGUAGE
    if whole_english is None:
        return _detect(text, detector)
    if not _is_worth_checking(text):
        return whole_english
    found = _detect(text, detector)
    is_foreign = found.language not in (ENGLISH, UNDETERMINED) and (
        _is_mostly_non_latin(text) or found.confidence >= FOREIGN_CONFIDENCE
    )
    return found if is_foreign else whole_english


def _detect(text: str, detector: LanguageDetector) -> SentenceLanguage:
    found = detector.detect_language_of(text)
    if found is not None:
        return _sentence_language_of(found, detector.compute_language_confidence(text, found))
    values = detector.compute_language_confidence_values(text)
    if len(values) < 2 or ENGLISH in (_code(values[0].language), _code(values[1].language)):
        return _NO_LANGUAGE
    return _sentence_language_of(values[0].language, values[0].value)


def _is_worth_checking(text: str) -> bool:
    if _is_mostly_non_latin(text):
        return True
    words = [word.lower() for word in _WORD.findall(text)]
    return len(words) >= MIN_CHECKED_WORDS and ENGLISH_FUNCTION_WORDS.isdisjoint(words)


def _is_mostly_non_latin(text: str) -> bool:
    if not _NON_LATIN_LETTER.search(text):
        return False
    return 2 * len(_NON_LATIN_LETTER.findall(text)) > len(_LETTER.findall(text))


def _code(language: LinguaLanguage) -> str:
    return language.iso_code_639_1.name.lower()
