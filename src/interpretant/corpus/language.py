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
_LETTER = re.compile(r"[^\W\d_]")


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
    every sentence with letters takes that verdict. Otherwise each sentence is detected on its
    own, with lingua's confidence for the language found; a sentence whose two likeliest
    languages are closer than the detector's minimum relative distance is undetermined, as is a
    sentence without letters. (Until 2026-11-13 lingua's experimental detection of several
    languages in one text split such abstracts; the corpus audit found it read real English as
    German or French, decision agenda 2.4.)
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
    return SentenceLanguage(
        language.iso_code_639_1.name.lower(), round(confidence, CONFIDENCE_DECIMALS)
    )


def _sentence_language(
    sentence: Span, whole_english: SentenceLanguage | None, detector: LanguageDetector
) -> SentenceLanguage:
    if not _LETTER.search(sentence.text):
        return _NO_LANGUAGE
    if whole_english is not None:
        return whole_english
    found = detector.detect_language_of(sentence.text)
    if found is None:
        return _NO_LANGUAGE
    return _sentence_language_of(found, detector.compute_language_confidence(sentence.text, found))
