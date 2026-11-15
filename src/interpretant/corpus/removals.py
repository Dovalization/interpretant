"""The cleaning patterns applied to annotated documents: removal ranges, drop reasons and flags.

The spaCy pass's documents carry tokens and sentences; this module matches the cleaning patterns
on them with spaCy's ``SpanRuler`` and applies the keeping rules of decision agenda 2.3. The
patterns read only word-level attributes (text, lower case, shape, ``like_url``, ``like_num``,
punctuation, the space after a token), so a blank English pipeline matches them: the statistical
models are not loaded. Every removal is a labelled character range of the repaired text, which
is never edited (2.2).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import spacy
from spacy.language import Language
from spacy.pipeline import SpanRuler
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans, registry

from interpretant.catalog import START_ID, CleaningPatterns, Scope
from interpretant.corpus.language import ENGLISH, UNDETERMINED, SentenceLanguage
from interpretant.corpus.repair import has_mojibake
from interpretant.corpus.works import RawAbstract

SpansFilter = Callable[[Iterable[Span], Iterable[Span]], list[Span]]

SPANS_KEY = "removals"
# The label of the leading "Abstract" or "Summary" label; a start-only match may follow it.
LEADING_LABEL = "abstract_label"
START_AWARE_SPANS_FILTER = "interpretant.start_aware_spans_filter.v1"
# The keeping rules of decision agenda 2.3: an abstract needs five words outside the removed
# ranges, and a "sentence" longer than about 150 words means missing sentence punctuation.
MIN_WORDS = 5
MALFORMED_SENTENCE_WORDS = 150
MALFORMED = "malformed"
NOT_ENGLISH = "not_english"


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
    """One row of the annotated abstracts table, before the corpus-level rules."""

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
    flags: list[str] = field(default_factory=list)
    dropped_reason: str | None = None
    duplicate_of: str | None = None


def make_start_aware_spans_filter(start_id: str, leading_label: str) -> SpansFilter:
    """The first-longest spans filter, after dropping start-only matches away from the start.

    A match of a pattern with the id ``start_id`` counts when it begins at the first token or
    right after the longest ``leading_label`` match there ("Abstract: To the Editor…"). The check
    comes before the overlap filter, so a start-only match cannot lose to a longer match that is
    not allowed where it begins: in run 6c, the letter match of "to the editor" from the second
    word beat the catalogue match of "Letter to" and was then discarded, keeping the record.
    """

    def start_aware_spans_filter(existing: Iterable[Span], matches: Iterable[Span]) -> list[Span]:
        candidates = [*existing, *matches]
        label_ends = [
            span.end for span in candidates if span.label_ == leading_label and span.start == 0
        ]
        openings = {0, max(label_ends, default=0)}
        return filter_spans(
            span for span in candidates if span.id_ != start_id or span.start in openings
        )

    return start_aware_spans_filter


# Registered by a call rather than as a decorator, since catalogue's decorator is untyped.
registry.misc.register(START_AWARE_SPANS_FILTER, func=make_start_aware_spans_filter)


def build_cleaning_pipeline(patterns: CleaningPatterns) -> Language:
    """A blank English pipeline whose only component matches the cleaning patterns.

    Start-only matches count only at the start of the text or after its leading label, and
    overlapping matches then keep the first longest, as in spaCy's registered filter (the entity
    ruler's default), so "Abstract:" wins over "Abstract"; matches already on a document are
    replaced. Documents read from the annotation cache with this pipeline's vocabulary can be
    passed to it.
    """
    if LEADING_LABEL not in patterns.labels:
        raise ValueError(f"the cleaning rules declare no {LEADING_LABEL!r} label")
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe(
        "span_ruler",
        config={
            "spans_key": SPANS_KEY,
            "overwrite": True,
            "validate": True,
            "spans_filter": {
                "@misc": START_AWARE_SPANS_FILTER,
                "start_id": START_ID,
                "leading_label": LEADING_LABEL,
            },
        },
    )
    if not isinstance(ruler, SpanRuler):
        raise TypeError(f"expected a SpanRuler, got {type(ruler).__name__}")
    ruler.add_patterns(patterns.span_ruler_patterns())
    return nlp


def clean_abstract(
    doc: Doc,
    abstract: RawAbstract,
    languages: Sequence[SentenceLanguage],
    *,
    cleaning: Language,
    patterns: CleaningPatterns,
    signatures: re.Pattern[str],
    min_english_words: int,
) -> AnnotatedAbstract:
    """The annotated row of one abstract: its sentences, removals, flags and drop reason.

    Dropped abstracts get a row too, with their reason, so the corpus review can sample them. The
    reasons, first match wins: ``mojibake`` (read on the raw text), an abstract-scope pattern's
    label, ``not_english`` (fewer than ``min_english_words`` words left once text in another
    language is removed, and more words removed as other-language than left: a provisional
    threshold, decision agenda 2.4), ``too_short``
    (fewer than five words kept). An abstract printed in several languages keeps its English
    part: other-language sentences are removed, not counted against it.
    """
    doc = cleaning(doc)
    removals, drop_reasons = find_removals(doc, patterns.labels, languages)
    if has_mojibake(abstract.text, signatures):
        drop_reasons.insert(0, "mojibake")
    kept_words = count_kept_words(doc, removals)
    other_language_words = _count_words_inside(
        doc, [removal for removal in removals if removal.reason == NOT_ENGLISH]
    )
    if kept_words < min_english_words and other_language_words > kept_words:
        drop_reasons.append(NOT_ENGLISH)
    if kept_words < MIN_WORDS:
        drop_reasons.append("too_short")
    is_malformed = any(
        sum(1 for token in sentence if _is_word(token)) > MALFORMED_SENTENCE_WORDS
        for sentence in doc.sents
    )
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
        flags=[MALFORMED] if is_malformed else [],
        dropped_reason=drop_reasons[0] if drop_reasons else None,
    )


def find_removals(
    doc: Doc, scopes: Mapping[str, Scope], languages: Sequence[SentenceLanguage]
) -> tuple[list[Removal], list[str]]:
    """The removal ranges of one abstract, sorted, and the labels that drop it whole.

    A tokens-scope match removes its own characters; a rest-of-sentence match removes from its
    start to the end of its sentence, so text before a notice survives; an abstract-scope match
    removes its characters and drops the abstract (start-only matches away from the start are
    already filtered out by the cleaning pipeline). A sentence lingua reads as another language
    is removed with the reason ``not_english``. A range inside another range with the same reason
    is left out, as when one notice holds several matches ("(PsycINFO Database Record (c) 2006").
    """
    removals: set[Removal] = set()
    drop_reasons: list[str] = []
    for span in doc.spans[SPANS_KEY]:
        label = span.label_
        if scopes[label] == "rest_of_sentence":
            removals.add(Removal(span.start_char, span.sent.end_char, label))
            continue
        removals.add(Removal(span.start_char, span.end_char, label))
        if scopes[label] == "abstract" and label not in drop_reasons:
            drop_reasons.append(label)
    for sentence, language in zip(doc.sents, languages, strict=True):
        if language.language not in (ENGLISH, UNDETERMINED):
            removals.add(Removal(sentence.start_char, sentence.end_char, NOT_ENGLISH))
    return _outermost(removals), drop_reasons


def kept_text(text: str, removals: Sequence[Removal]) -> str:
    """The text outside the removed ranges, the pieces joined by single spaces."""
    pieces = []
    position = 0
    for removal in sorted(removals):
        if removal.start > position:
            pieces.append(text[position : removal.start])
        position = max(position, removal.end)
    pieces.append(text[position:])
    return " ".join(pieces)


def count_kept_words(doc: Doc, removals: Sequence[Removal]) -> int:
    """The words (tokens that are neither punctuation nor space) outside every removed range."""
    return sum(
        1
        for token in doc
        if _is_word(token)
        and not any(removal.start <= token.idx < removal.end for removal in removals)
    )


def _count_words_inside(doc: Doc, removals: Sequence[Removal]) -> int:
    return sum(
        1
        for token in doc
        if _is_word(token) and any(removal.start <= token.idx < removal.end for removal in removals)
    )


def _outermost(removals: set[Removal]) -> list[Removal]:
    """The removals, sorted, without those inside another removal with the same reason."""
    return sorted(
        removal
        for removal in removals
        if not any(
            other != removal
            and other.reason == removal.reason
            and other.start <= removal.start
            and removal.end <= other.end
            for other in removals
        )
    )


def _is_word(token: Token) -> bool:
    return not (token.is_punct or token.is_space)
