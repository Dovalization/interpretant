"""Sentence languages by lingua, read from the spaCy pass's sentences.

The decision agenda, step 2.4: lingua alone, restricted to English and the languages found in the
corpus (scripts/evidence/language_scan.py); per abstract first, per sentence only for abstracts
not confidently English. It runs as its own stage on the documents in the annotation cache, so a
change to its settings reruns neither the spaCy models nor anything but the later stages, and
reads them in batches with lingua's multi-threaded methods.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from lingua import ConfidenceValue, IsoCode639_1, LanguageDetector, LanguageDetectorBuilder
from lingua import Language as LinguaLanguage
from spacy.tokens import Doc

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


def build_language_detector(languages: Sequence[str]) -> LanguageDetector:
    """A lingua detector restricted to the given ISO 639-1 codes, which must include English.

    Restricting the candidates improves accuracy when the text's languages are among them
    (lingua's documentation). The minimum relative distance between the two likeliest languages
    is applied by ``detect_sentence_languages``, which reads each text's confidence values once.

    Raises:
        ValueError: if English is missing or a code is not an ISO 639-1 code lingua knows.
    """
    return (
        LanguageDetectorBuilder.from_iso_codes_639_1(*_iso_codes(languages))
        .with_preloaded_language_models()
        .build()
    )


def detect_sentence_languages(
    docs: Sequence[Doc],
    detector: LanguageDetector,
    *,
    english_threshold: float,
    minimum_relative_distance: float,
) -> list[list[SentenceLanguage]]:
    """Each document's sentence languages, with lingua's confidence in each, in order.

    When lingua gives English at least ``english_threshold`` confidence for the whole abstract,
    every sentence with letters takes that verdict, except a sentence worth checking (mostly in
    another script, or long with no English function word) that lingua reads as another
    language: in another script at any confidence, in the Latin script at
    ``FOREIGN_CONFIDENCE``. Otherwise each sentence is detected on its own, with lingua's
    confidence for the language found. When its two likeliest languages are closer than
    ``minimum_relative_distance``, it is undetermined if one of them is English (short
    English fragments, headings, references) and takes the likelier one if neither is (Croatian
    and Bosnian, Indonesian and Malay). A sentence without letters is undetermined. (Until
    2026-11-13 lingua's experimental detection of several languages in one text split such
    abstracts; the corpus audit found it read real English as German or French, decision agenda
    2.4.)

    The documents are read together with lingua's multi-threaded methods, which use every core
    with one copy of the language models (lingua's documentation).
    """
    wholes = [
        _top_language(values)
        for values in detector.compute_language_confidence_values_in_parallel(
            [doc.text for doc in docs]
        )
    ]
    labels: list[list[SentenceLanguage]] = []
    checks: list[tuple[int, int, str, bool]] = []
    for doc_index, (doc, whole) in enumerate(zip(docs, wholes, strict=True)):
        is_english = whole.language == ENGLISH and whole.confidence >= english_threshold
        row: list[SentenceLanguage] = []
        for sentence_index, sentence in enumerate(doc.sents):
            text = sentence.text
            if not _LETTER.search(text):
                row.append(_NO_LANGUAGE)
                continue
            row.append(whole if is_english else _NO_LANGUAGE)
            if not is_english or _is_worth_checking(text):
                checks.append((doc_index, sentence_index, text, is_english))
        labels.append(row)
    detected = _detect_texts(
        [text for _, _, text, _ in checks], detector, minimum_relative_distance
    )
    for (doc_index, sentence_index, text, in_english), language in zip(
        checks, detected, strict=True
    ):
        if in_english and not _is_foreign_in_english(text, language):
            continue
        labels[doc_index][sentence_index] = language
    return labels


def _detect_texts(
    texts: list[str], detector: LanguageDetector, minimum_relative_distance: float
) -> list[SentenceLanguage]:
    """Each text's language and lingua's confidence in it; close calls as the docstring above.

    One call reads each text's confidence values; the likeliest language is the verdict unless
    the second is as likely or within ``minimum_relative_distance`` of it, the rule lingua's own
    detection applies, which would read every text a second time.
    """
    results = []
    for values in detector.compute_language_confidence_values_in_parallel(texts):
        top = values[0]
        second = values[1].value if len(values) > 1 else 0.0
        if top.value == second or top.value - second < minimum_relative_distance:
            results.append(_close_call(values))
        else:
            results.append(_sentence_language_of(top.language, top.value))
    return results


def _close_call(values: Sequence[ConfidenceValue]) -> SentenceLanguage:
    if len(values) < 2 or ENGLISH in (_code(values[0].language), _code(values[1].language)):
        return _NO_LANGUAGE
    return _sentence_language_of(values[0].language, values[0].value)


def _is_foreign_in_english(text: str, language: SentenceLanguage) -> bool:
    return language.language not in (ENGLISH, UNDETERMINED) and (
        _is_mostly_non_latin(text) or language.confidence >= FOREIGN_CONFIDENCE
    )


def _iso_codes(codes: Sequence[str]) -> list[IsoCode639_1]:
    """Lingua's ISO 639-1 codes for the given ones, which must include English."""
    unique_codes = sorted({code.lower() for code in codes})
    if ENGLISH not in unique_codes:
        raise ValueError("the language detector must include English")
    known, unknown = [], []
    for code in unique_codes:
        try:
            known.append(IsoCode639_1.from_str(code))
        except ValueError:
            unknown.append(code)
    if unknown:
        raise ValueError(f"not ISO 639-1 codes lingua knows: {unknown}")
    return known


def _top_language(values: Sequence[ConfidenceValue]) -> SentenceLanguage:
    if not values:
        return _NO_LANGUAGE
    best = values[0]
    return _sentence_language_of(best.language, best.value)


def _sentence_language_of(language: LinguaLanguage, confidence: float) -> SentenceLanguage:
    return SentenceLanguage(_code(language), round(confidence, CONFIDENCE_DECIMALS))


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
