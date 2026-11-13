"""Sentence languages by lingua, as a spaCy pipeline component.

The decision agenda, step 2.4: lingua alone, restricted to English and the languages found in the
corpus; per abstract first, per sentence only for abstracts not confidently English. The
``sentence_languages`` component sets each sentence's language and confidence as span extension
attributes, so the annotation cache keeps them (4.6) and the abstracts table copies them. As a
component it runs inside spaCy's worker processes, and its settings are part of the pipeline's
config.
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
from spacy.language import Language
from spacy.tokens import Doc, Span

# The component's registered factory name, which is also its name in the pipeline.
SENTENCE_LANGUAGES = "sentence_languages"
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
    return (
        LanguageDetectorBuilder.from_languages(*_lingua_languages(languages))
        .with_preloaded_language_models()
        .build()
    )


class SentenceLanguageDetector:
    """The pipeline component setting each sentence's language and lingua's confidence in it.

    When lingua gives English at least ``english_threshold`` confidence for the whole abstract,
    every sentence with letters takes that verdict. Otherwise lingua's detection of several
    languages in one text (documented as experimental, so validated on the sample: decision
    agenda 2.4) splits the abstract into single-language ranges; each sentence takes the language
    covering most of its characters, with lingua's confidence for that language in that
    sentence. Sentences without letters are undetermined.

    lingua's detector cannot be pickled, and spaCy pickles each component to start its worker
    processes, so the detector is built on first use and left out of the pickled state: each
    worker builds its own. Its single-threaded calls are used, since spaCy's worker processes
    already run the abstracts in parallel.
    """

    def __init__(self, languages: Sequence[str], english_threshold: float) -> None:
        """Check the language codes now; build the detector when the first document arrives.

        Raises:
            ValueError: if English is missing or a code is not an ISO 639-1 code lingua knows.
        """
        _lingua_languages(languages)
        self.languages = tuple(languages)
        self.english_threshold = english_threshold
        self._detector: LanguageDetector | None = None

    def __call__(self, doc: Doc) -> Doc:
        """Set the language attributes of every sentence of the document."""
        detector = self._loaded_detector()
        whole = _top_language(detector.compute_language_confidence_values(doc.text))
        is_english = whole.language == ENGLISH and whole.confidence >= self.english_threshold
        ranges = None if is_english else detector.detect_multiple_languages_of(doc.text)
        for sentence in doc.sents:
            language = _sentence_language(sentence, whole, ranges, detector)
            sentence._.set(LANGUAGE_ATTRIBUTE, language.language)
            sentence._.set(CONFIDENCE_ATTRIBUTE, language.confidence)
        return doc

    def __getstate__(self) -> dict[str, object]:
        """The settings without the detector, which each process builds for itself."""
        return {**self.__dict__, "_detector": None}

    def _loaded_detector(self) -> LanguageDetector:
        if self._detector is None:
            self._detector = build_language_detector(self.languages)
        return self._detector


@Language.factory(
    SENTENCE_LANGUAGES,
    assigns=[f"span._.{LANGUAGE_ATTRIBUTE}", f"span._.{CONFIDENCE_ATTRIBUTE}"],
    requires=["token.is_sent_start"],
)
def create_sentence_language_detector(
    nlp: Language, name: str, languages: list[str], english_threshold: float
) -> SentenceLanguageDetector:
    """The factory spaCy calls to add the component; declares the span attributes it sets."""
    declare_sentence_attributes()
    return SentenceLanguageDetector(languages, english_threshold)


def read_sentence_languages(doc: Doc) -> list[SentenceLanguage]:
    """Each sentence's language and confidence, as the component set them.

    Raises:
        ValueError: if the component has not run on the document.
    """
    languages = []
    for sentence in doc.sents:
        language = sentence._.get(LANGUAGE_ATTRIBUTE)
        if language is None:
            raise ValueError(f"no sentence languages: run the {SENTENCE_LANGUAGES} component")
        languages.append(SentenceLanguage(language, sentence._.get(CONFIDENCE_ATTRIBUTE)))
    return languages


def declare_sentence_attributes() -> None:
    """Declare the sentence language attributes on spaCy's ``Span`` class, once.

    spaCy keeps extension attributes globally, and a document read back from the annotation cache
    only exposes the stored values once the attributes are declared again (spaCy's documentation,
    "Serializing Doc objects efficiently").
    """
    for name in (LANGUAGE_ATTRIBUTE, CONFIDENCE_ATTRIBUTE):
        if not Span.has_extension(name):
            Span.set_extension(name, default=None)


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
