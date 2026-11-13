"""The spaCy pass: tokens, sentences, tags, lemmas, entities and labelled removal spans.

One pass annotates each repaired abstract (decision agenda, steps 2 and 4.9): sentences by
``senter`` with abbreviation special cases, hyphenated words kept whole, DOIs kept as single
tokens, and every cleaning pattern's matches written as labelled spans in
``doc.spans["removals"]``. Sentence languages come from lingua, as the pipeline's last component
(``corpus.language``). The keeping rules of the pass (2.3) drop an abstract with mojibake, an
abstract-scope pattern, mostly another language or fewer than five words, and flag one whose
"sentence" runs past about 150 words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import spacy
from spacy.lang.char_classes import ALPHA, HYPHENS
from spacy.lang.tokenizer_exceptions import URL_PATTERN
from spacy.language import Language
from spacy.pipeline import SpanRuler
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Token
from spacy.util import compile_infix_regex

from interpretant.catalog import START_ID, CleaningPatterns, Scope
from interpretant.corpus.language import (
    ENGLISH,
    SENTENCE_LANGUAGES,
    UNDETERMINED,
    SentenceLanguage,
    compute_not_english_share,
    read_sentence_languages,
)
from interpretant.corpus.repair import compile_signatures, has_mojibake, repair_text

SPANS_KEY = "removals"
# Abbreviations whose full stop would otherwise end a sentence (docs/spacy.md).
ABBREVIATIONS = ("cf.", "Cf.", "Fig.", "Figs.", "Sect.", "Eq.", "Eqs.", "al.", "approx.")
# The keeping rules of decision agenda 2.3: an abstract needs five words outside the removed
# ranges, and a "sentence" longer than about 150 words means missing sentence punctuation.
MIN_WORDS = 5
MALFORMED_SENTENCE_WORDS = 150
MALFORMED = "malformed"
# English's infix rule for a hyphen after a letter or digit and before a letter
# (spacy/lang/en/punctuation.py); removing it keeps *self-attention* one token (decision agenda
# 2.3), as spaCy's documentation shows under "Modifying existing rule sets".
_HYPHEN_INFIX = rf"(?<=[{ALPHA}0-9])(?:{HYPHENS})(?=[{ALPHA}])"
_DOI = r"^(?:doi:)?10\.\d{4,9}/\S+$"
# spaCy's own URL rule extended with DOIs, as one compiled pattern so the tokenizer can still be
# serialised to worker processes.
URL_OR_DOI = re.compile(f"(?u)(?:{_DOI})|(?:{URL_PATTERN})")


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
    flags: list[str] = field(default_factory=list)
    dropped_reason: str | None = None
    duplicate_of: str | None = None


def build_pipeline(
    patterns: CleaningPatterns,
    *,
    languages: Sequence[str],
    english_threshold: float,
    model: str = "en_core_web_sm",
) -> Language:
    """The spaCy pipeline of the pass: the cleaning patterns as a span ruler, then lingua.

    The parser is left out and ``senter`` sets sentences (about ten times faster, and it matched
    the parser on the sample once the abbreviation special cases were in: docs/spacy.md). The
    language component reads the given ISO 639-1 codes and takes an abstract with at least
    ``english_threshold`` confidence in English as English throughout (decision agenda 2.4); it
    declares the sentence language attributes on spaCy's ``Span`` class as a side effect.

    Raises:
        ValueError: if spaCy's English hyphen rule has changed and cannot be removed, or the
            language codes lack English or include one lingua does not know.
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
    nlp.add_pipe(
        SENTENCE_LANGUAGES,
        config={"languages": list(languages), "english_threshold": english_threshold},
    )
    # The tok2vec vectors left on each document are neither cached nor read later; dropping them
    # makes a document about a third the size to hold and to pass between worker processes
    # (spaCy's documentation, "Memory management").
    nlp.add_pipe("doc_cleaner", config={"attrs": {"tensor": None}})
    return nlp


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


def annotate_abstracts(
    abstracts: Iterable[RawAbstract],
    nlp: Language,
    patterns: CleaningPatterns,
    *,
    max_not_english_share: float,
    batch_size: int = 256,
    process_count: int = 1,
) -> Iterator[tuple[Doc, AnnotatedAbstract]]:
    """Repair and annotate each abstract: its spaCy document and its abstracts-table row.

    Dropped abstracts are annotated too, with their reason, so the corpus review can sample
    them. The reasons, first match wins: ``mojibake`` (read on the raw text), an abstract-scope
    pattern's label, ``not_english`` (more than ``max_not_english_share`` of the judged text, a
    provisional threshold: decision agenda 2.4), ``too_short`` (fewer than five words kept).
    """
    signatures = compile_signatures(patterns.mojibake_signatures)
    pairs = ((repair_text(abstract.text), abstract) for abstract in abstracts)
    annotated = nlp.pipe(pairs, as_tuples=True, batch_size=batch_size, n_process=process_count)
    for doc, abstract in annotated:
        yield (
            doc,
            _annotated_abstract(doc, abstract, patterns, signatures, max_not_english_share),
        )


def _annotated_abstract(
    doc: Doc,
    abstract: RawAbstract,
    patterns: CleaningPatterns,
    signatures: re.Pattern[str],
    max_not_english_share: float,
) -> AnnotatedAbstract:
    languages = read_sentence_languages(doc)
    removals, drop_reasons = find_removals(doc, patterns.labels, languages)
    if has_mojibake(abstract.text, signatures):
        drop_reasons.insert(0, "mojibake")
    if compute_not_english_share(doc, languages) > max_not_english_share:
        drop_reasons.append("not_english")
    if count_kept_words(doc, removals) < MIN_WORDS:
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


def _is_word(token: Token) -> bool:
    return not (token.is_punct or token.is_space)
