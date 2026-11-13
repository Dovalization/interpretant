"""The spaCy pass: tokens, sentences, tags, lemmas, entities and labelled removal spans.

One pass annotates each repaired abstract (decision agenda, steps 2 and 4.9): sentences by
``senter`` with abbreviation special cases, hyphenated words kept whole, DOIs kept as single
tokens, and every cleaning pattern's matches written as labelled spans in
``doc.spans["removals"]``. Sentence languages come from lingua, per abstract first and, for an
abstract not confidently English, from lingua's detection of several languages in one text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice

import spacy
from lingua import (
    ConfidenceValue,
    DetectionResult,
    IsoCode639_1,
    LanguageDetector,
    LanguageDetectorBuilder,
)
from lingua import Language as LinguaLanguage
from spacy.lang.char_classes import ALPHA, HYPHENS
from spacy.lang.tokenizer_exceptions import URL_PATTERN
from spacy.language import Language
from spacy.pipeline import SpanRuler
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Span
from spacy.util import compile_infix_regex

from interpretant.catalog import START_ID, CleaningPatterns, Scope
from interpretant.corpus.repair import compile_signatures, has_mojibake, repair_text

SPANS_KEY = "removals"
ENGLISH = "en"
# ISO 639-2's code for an undetermined language: a sentence with no letters, or no verdict.
UNDETERMINED = "und"
# Abbreviations whose full stop would otherwise end a sentence (docs/spacy.md).
ABBREVIATIONS = ("cf.", "Cf.", "Fig.", "Figs.", "Sect.", "Eq.", "Eqs.", "al.", "approx.")
# English's infix rule for a hyphen after a letter or digit and before a letter
# (spacy/lang/en/punctuation.py); removing it keeps *self-attention* one token (decision agenda
# 2.3), as spaCy's documentation shows under "Modifying existing rule sets".
_HYPHEN_INFIX = rf"(?<=[{ALPHA}0-9])(?:{HYPHENS})(?=[{ALPHA}])"
_DOI = r"^(?:doi:)?10\.\d{4,9}/\S+$"
# spaCy's own URL rule extended with DOIs, as one compiled pattern so the tokenizer can still be
# serialised to worker processes.
URL_OR_DOI = re.compile(f"(?u)(?:{_DOI})|(?:{URL_PATTERN})")
_LETTER = re.compile(r"[^\W\d_]")
# lingua's confidence for the same text varies in its last digits from call to call, so it is
# rounded where it is read: the threshold decision and the stored table then reproduce.
CONFIDENCE_DECIMALS = 4


@dataclass(frozen=True)
class RawAbstract:
    """One abstract as collected, before repair, with the metadata the table keeps."""

    work_id: str
    year: int
    fields: tuple[str, ...]
    source: str
    document_type: str
    author_ids: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class SentenceLanguage:
    """A sentence's language (an ISO 639-1 code, or ``und``) and lingua's confidence in it."""

    language: str
    confidence: float


@dataclass(frozen=True, order=True)
class Removal:
    """A labelled character range of the repaired text that the views leave out."""

    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class Sentence:
    """A sentence's character range and its language."""

    start: int
    end: int
    language: str
    confidence: float


@dataclass(frozen=True)
class AnnotatedAbstract:
    """One row of the abstracts table, as the spaCy pass produces it."""

    work_id: str
    year: int
    decade: int
    fields: list[str]
    source: str
    document_type: str
    author_ids: list[str]
    text: str
    sentences: list[Sentence] = field(default_factory=list)
    removals: list[Removal] = field(default_factory=list)
    dropped_reason: str | None = None


_NO_LANGUAGE = SentenceLanguage(UNDETERMINED, 0.0)


def build_pipeline(patterns: CleaningPatterns, *, model: str = "en_core_web_sm") -> Language:
    """The spaCy pipeline of the pass, with the cleaning patterns as a span ruler.

    The parser is left out and ``senter`` sets sentences (about ten times faster, and it matched
    the parser on the sample once the abbreviation special cases were in: docs/spacy.md).

    Raises:
        ValueError: if spaCy's English hyphen rule has changed and cannot be removed.
    """
    nlp = spacy.load(model, exclude=["parser"])
    nlp.enable_pipe("senter")
    tokenizer = nlp.tokenizer
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(f"expected spaCy's rule-based tokenizer, got {type(tokenizer).__name__}")
    for abbreviation in ABBREVIATIONS:
        tokenizer.add_special_case(abbreviation, [{ORTH: abbreviation}])
    if nlp.Defaults.infixes is None:
        raise ValueError("the pipeline's language defines no infix rules")
    infixes = list(nlp.Defaults.infixes)
    infixes.remove(_HYPHEN_INFIX)
    tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    tokenizer.url_match = URL_OR_DOI.match
    # Overlapping matches keep the first longest, with spaCy's registered filter (the entity
    # ruler's default), so "Abstract:" wins over "Abstract".
    ruler = nlp.add_pipe(
        "span_ruler",
        config={
            "spans_key": SPANS_KEY,
            "validate": True,
            "spans_filter": {"@misc": "spacy.first_longest_spans_filter.v1"},
        },
    )
    if not isinstance(ruler, SpanRuler):
        raise TypeError(f"expected a SpanRuler, got {type(ruler).__name__}")
    ruler.add_patterns(patterns.span_ruler_patterns())
    return nlp


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


def find_removals(
    doc: Doc, scopes: Mapping[str, Scope], languages: Sequence[SentenceLanguage]
) -> tuple[list[Removal], list[str]]:
    """The removal ranges of one abstract, sorted, and the labels that drop it whole.

    A tokens-scope match removes its own characters; a sentence-scope match removes its whole
    sentence; an abstract-scope match removes its characters and drops the abstract. Matches of
    patterns with the id ``start`` count only at the first token. A sentence lingua reads as
    another language is removed with the reason ``not_english``.
    """
    removals: set[Removal] = set()
    drop_reasons: list[str] = []
    for span in doc.spans[SPANS_KEY]:
        if span.id_ == START_ID and span.start != 0:
            continue
        label = span.label_
        if scopes[label] == "sentence":
            sentence = span.sent
            removals.add(Removal(sentence.start_char, sentence.end_char, label))
            continue
        removals.add(Removal(span.start_char, span.end_char, label))
        if scopes[label] == "abstract" and label not in drop_reasons:
            drop_reasons.append(label)
    for sentence, language in zip(doc.sents, languages, strict=True):
        if language.language not in (ENGLISH, UNDETERMINED):
            removals.add(Removal(sentence.start_char, sentence.end_char, "not_english"))
    return sorted(removals), drop_reasons


def annotate_abstracts(
    abstracts: Iterable[RawAbstract],
    nlp: Language,
    detector: LanguageDetector,
    patterns: CleaningPatterns,
    *,
    language_threshold: float,
    batch_size: int = 256,
    process_count: int = 1,
) -> Iterator[tuple[Doc, AnnotatedAbstract]]:
    """Repair and annotate each abstract: its spaCy document and its abstracts-table row.

    Abstracts showing mojibake (read on the raw text) are annotated too and marked dropped,
    so the corpus review can sample them; otherwise an abstract-scope pattern's label, if any,
    is the drop reason.
    """
    signatures = compile_signatures(patterns.mojibake_signatures)
    pairs = ((repair_text(abstract.text), abstract) for abstract in abstracts)
    annotated = nlp.pipe(pairs, as_tuples=True, batch_size=batch_size, n_process=process_count)
    while chunk := list(islice(annotated, batch_size)):
        docs = [doc for doc, _ in chunk]
        chunk_languages = detect_sentence_languages(docs, detector, threshold=language_threshold)
        for (doc, abstract), languages in zip(chunk, chunk_languages, strict=True):
            yield doc, _annotated_abstract(doc, abstract, languages, patterns, signatures)


def _annotated_abstract(
    doc: Doc,
    abstract: RawAbstract,
    languages: Sequence[SentenceLanguage],
    patterns: CleaningPatterns,
    signatures: re.Pattern[str],
) -> AnnotatedAbstract:
    removals, drop_reasons = find_removals(doc, patterns.labels, languages)
    if has_mojibake(abstract.text, signatures):
        dropped_reason: str | None = "mojibake"
    else:
        dropped_reason = drop_reasons[0] if drop_reasons else None
    sentences = [
        Sentence(sentence.start_char, sentence.end_char, language.language, language.confidence)
        for sentence, language in zip(doc.sents, languages, strict=True)
    ]
    return AnnotatedAbstract(
        work_id=abstract.work_id,
        year=abstract.year,
        decade=abstract.year - abstract.year % 10,
        fields=list(abstract.fields),
        source=abstract.source,
        document_type=abstract.document_type,
        author_ids=list(abstract.author_ids),
        text=doc.text,
        sentences=sentences,
        removals=removals,
        dropped_reason=dropped_reason,
    )


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
