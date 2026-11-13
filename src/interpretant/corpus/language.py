"""Sentence languages by lingua, recorded on the spaCy pipeline's sentences.

The decision agenda, step 2.4: lingua alone, restricted to English and the languages found in the
corpus; per abstract first, per sentence only for abstracts not confidently English. Each
sentence's language and confidence are set as span extension attributes, so the annotation cache
keeps them (4.6), and copied into the abstracts table.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from lingua import (
    ConfidenceValue,
    DetectionResult,
    IsoCode639_1,
    LanguageDetector,
    LanguageDetectorBuilder,
)
from lingua import Language as LinguaLanguage
from spacy.tokens import Doc, Span

ENGLISH = "en"
# ISO 639-2's code for an undetermined language: a sentence with no letters, or no verdict.
UNDETERMINED = "und"
# lingua's confidence for the same text varies in its last digits from call to call, so it is
# rounded where it is read: the threshold decision and the stored table then reproduce.
CONFIDENCE_DECIMALS = 4
# The span extension attributes holding each sentence's language and lingua's confidence in it.
LANGUAGE_ATTRIBUTE = "language"
CONFIDENCE_ATTRIBUTE = "language_confidence"
_LETTER = re.compile(r"[^\W\d_]")


@dataclass(frozen=True)
class SentenceLanguage:
    """A sentence's language (an ISO 639-1 code, or ``und``) and lingua's confidence in it."""

    language: str
    confidence: float


_NO_LANGUAGE = SentenceLanguage(UNDETERMINED, 0.0)


def build_language_detector(languages: Sequence[str]) -> LanguageDetector:
    """A lingua detector restricted to the given ISO 639-1 codes, which must include English.

    Restricting the candidates improves accuracy when the text's languages are among them
    (lingua's documentation).

    Raises:
        ValueError: if English is missing or a code is not an ISO 639-1 code lingua knows.
    """
    codes = sorted({code.lower() for code in languages})
    if ENGLISH not in codes:
        raise ValueError("the language detector must include English")
    unknown = [code for code in codes if not hasattr(IsoCode639_1, code.upper())]
    if unknown:
        raise ValueError(f"not ISO 639-1 codes lingua knows: {unknown}")
    candidates = [
        LinguaLanguage.from_iso_code_639_1(getattr(IsoCode639_1, code.upper())) for code in codes
    ]
    return (
        LanguageDetectorBuilder.from_languages(*candidates).with_preloaded_language_models().build()
    )


def declare_sentence_attributes() -> None:
    """Declare the sentence language attributes on spaCy's ``Span`` class, once.

    spaCy keeps extension attributes globally, and a document read back from the annotation cache
    only exposes the stored values once the attributes are declared again (spaCy's documentation,
    "Serializing Doc objects efficiently").
    """
    for name in (LANGUAGE_ATTRIBUTE, CONFIDENCE_ATTRIBUTE):
        if not Span.has_extension(name):
            Span.set_extension(name, default=None)


def detect_sentence_languages(
    docs: Sequence[Doc], detector: LanguageDetector, *, threshold: float
) -> list[list[SentenceLanguage]]:
    """Each document's sentence languages, in sentence order, judged by lingua in parallel.

    When lingua gives English at least ``threshold`` confidence for the whole abstract, every
    sentence with letters takes that verdict. Otherwise lingua's detection of several languages
    in one text (documented as experimental, so validated on the sample: decision agenda 2.4)
    splits the abstract into single-language ranges; each sentence takes the language covering
    most of its characters, with lingua's confidence for that language in that sentence.
    Sentences without letters are undetermined.
    """
    whole = [
        _top_language(values)
        for values in detector.compute_language_confidence_values_in_parallel(
            [doc.text for doc in docs]
        )
    ]
    uncertain = [
        index
        for index, verdict in enumerate(whole)
        if not (verdict.language == ENGLISH and verdict.confidence >= threshold)
    ]
    ranges = detector.detect_multiple_languages_in_parallel_of(
        [docs[index].text for index in uncertain]
    )
    ranges_by_doc = dict(zip(uncertain, ranges, strict=True))
    languages = []
    for index, doc in enumerate(docs):
        doc_ranges = ranges_by_doc.get(index)
        languages.append(
            [
                _sentence_language(sentence, whole[index], doc_ranges, detector)
                for sentence in doc.sents
            ]
        )
    return languages


def set_sentence_languages(doc: Doc, languages: Sequence[SentenceLanguage]) -> None:
    """Write each sentence's language and confidence to its span extension attributes."""
    for sentence, language in zip(doc.sents, languages, strict=True):
        sentence._.set(LANGUAGE_ATTRIBUTE, language.language)
        sentence._.set(CONFIDENCE_ATTRIBUTE, language.confidence)


def compute_not_english_share(doc: Doc, languages: Sequence[SentenceLanguage]) -> float:
    """The share of the judged sentences' characters that lingua reads as another language.

    Undetermined sentences are left out of both sides; an abstract with none judged scores 0.
    """
    judged = 0
    other = 0
    for sentence, language in zip(doc.sents, languages, strict=True):
        if language.language == UNDETERMINED:
            continue
        length = sentence.end_char - sentence.start_char
        judged += length
        if language.language != ENGLISH:
            other += length
    return other / judged if judged else 0.0


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
    sentence: Span,
    whole: SentenceLanguage,
    ranges: Sequence[DetectionResult] | None,
    detector: LanguageDetector,
) -> SentenceLanguage:
    if not _LETTER.search(sentence.text):
        return _NO_LANGUAGE
    if ranges is None:
        return whole
    overlaps = [
        (
            min(found.end_index, sentence.end_char) - max(found.start_index, sentence.start_char),
            found,
        )
        for found in ranges
    ]
    overlap, best = max(overlaps, key=lambda pair: pair[0], default=(0, None))
    if best is None or overlap <= 0:
        return _NO_LANGUAGE
    confidence = detector.compute_language_confidence(sentence.text, best.language)
    return _sentence_language_of(best.language, confidence)
