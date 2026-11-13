"""The spaCy pass: tokens, sentences, tags, lemmas, entities and labelled removal spans.

One pass annotates each repaired abstract (decision agenda, steps 2 and 4.9): sentences by
``senter`` with abbreviation special cases and two sentence rules (a start after punctuation
glued to the next sentence, none inside square brackets), hyphenated words kept whole, DOIs kept
as single tokens, and every cleaning pattern's matches written as labelled spans in
``doc.spans["removals"]``. Sentence languages come from lingua, as the pipeline's last component
(``corpus.language``). The keeping rules of the pass (2.3) drop an abstract with mojibake, an
abstract-scope pattern, too little English beside another language or fewer than five words,
and flag one whose
"sentence" runs past about 150 words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.tokenizer_exceptions import URL_PATTERN
from spacy.language import Language
from spacy.pipeline import SpanRuler
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Token
from spacy.util import compile_infix_regex
from srsly.msgpack import msgpack_decoders, msgpack_encoders

from interpretant.catalog import START_ID, CleaningPatterns, Scope
from interpretant.corpus.language import (
    ENGLISH,
    SENTENCE_LANGUAGES,
    UNDETERMINED,
    SentenceLanguage,
    read_sentence_languages,
)
from interpretant.corpus.repair import compile_signatures, has_mojibake, repair_text

SPANS_KEY = "removals"
# Abbreviations whose full stop would otherwise end a sentence (docs/spacy.md; "viz." to "ca."
# added after the corpus audit, decision agenda 2.3).
ABBREVIATIONS = (
    "cf.",
    "Cf.",
    "Fig.",
    "Figs.",
    "Sect.",
    "Eq.",
    "Eqs.",
    "al.",
    "approx.",
    "viz.",
    "pp.",
    "resp.",
    "Vol.",
    "vol.",
    "ca.",
)
# The keeping rules of decision agenda 2.3: an abstract needs five words outside the removed
# ranges, and a "sentence" longer than about 150 words means missing sentence punctuation.
MIN_WORDS = 5
MALFORMED_SENTENCE_WORDS = 150
MALFORMED = "malformed"
# English's infix rule for a hyphen after a letter or digit and before a letter
# (spacy/lang/en/punctuation.py); removing it keeps *self-attention* one token (decision agenda
# 2.3), as spaCy's documentation shows under "Modifying existing rule sets".
_HYPHEN_INFIX = rf"(?<=[{ALPHA}0-9])(?:{HYPHENS})(?=[{ALPHA}])"
# Sentence punctuation with no space before the next sentence, as in "(r = 0.922).In addition"
# or "in 1950.French delegates" (decision agenda 2.3). spaCy's English rules split a full stop
# between two letters only; these two infix rules also split it after a digit, a closing bracket
# or a quote, and split that bracket or quote off the word before it.
_CLOSERS = r"\)\]\"”’"
_NEXT_SENTENCE = rf"[{ALPHA_UPPER}][{ALPHA_LOWER}]"
GLUED_STOP_INFIX = rf"(?<=[{ALPHA_LOWER}0-9{_CLOSERS}])[.?!](?={_NEXT_SENTENCE})"
GLUED_CLOSER_INFIX = rf"(?<=[{ALPHA}0-9])[{_CLOSERS}](?=[.?!]{_NEXT_SENTENCE})"
_GLUED_STOP = re.compile(GLUED_STOP_INFIX)
# A short square-bracket group, not nested: citations ("[2]", "[4, pp. 258-265]", "[Labov,
# 1966]"), abbreviations ("[PSQI]"), phonetic symbols ("[ta]"), "[Formula: see text]". spaCy's
# sentence model often starts a sentence at its bracket (decision agenda 2.3).
_SQUARE_BRACKETS = re.compile(r"\[[^\[\]]{1,40}\]")
_SENTENCE_FINAL = frozenset({".", "?", "!"})
# The component setting the sentence boundaries the sentence model misses, before ``senter``.
SENTENCE_RULES = "sentence_rules"
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
    named_entity_recognition: bool = False,
    model: str = "en_core_web_sm",
) -> Language:
    """The spaCy pipeline of the pass: the cleaning patterns as a span ruler, then lingua.

    The parser is left out and ``senter`` sets sentences (about ten times faster, and it matched
    the parser on the sample once the abbreviation special cases were in: docs/spacy.md). The
    language component reads the given ISO 639-1 codes and takes an abstract with at least
    ``english_threshold`` confidence in English as English throughout (decision agenda 2.4); it
    declares the sentence language attributes on spaCy's ``Span`` class as a side effect.

    The model's named entity recognition (``ner``) is left out unless
    ``named_entity_recognition`` is set: nothing in cleaning reads named entities, and it takes
    about 44% of a worker's time (timed on bazzite: 30 s instead of
    44 s per 10,000 abstracts). Matching recognises them on target sentences instead (decision
    agenda 4.1); there they agree with entities found on whole abstracts for 99.87% of 15,018
    target mentions.

    Raises:
        ValueError: if spaCy's English hyphen rule has changed and cannot be removed, or the
            language codes lack English or include one lingua does not know.
    """
    cache_serialiser_extensions()
    nlp = spacy.load(model, exclude=["parser"] if named_entity_recognition else ["parser", "ner"])
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
    infixes += [GLUED_STOP_INFIX, GLUED_CLOSER_INFIX]
    tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    tokenizer.url_match = URL_OR_DOI.match
    nlp.add_pipe(SENTENCE_RULES, before="senter")
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


@Language.component(SENTENCE_RULES)
def apply_sentence_rules(doc: Doc) -> Doc:
    """Set the sentence boundaries spaCy's sentence model gets wrong in abstracts.

    spaCy's documentation offers such rules on top of the statistical segmenter, set on
    ``Token.is_sent_start`` before it; ``senter`` keeps what is set (its ``overwrite`` setting is
    off). Two rules, from run 1's audit:

    - A sentence starts after sentence punctuation glued to the next sentence. The tokenizer
      splits the punctuation off (``GLUED_STOP_INFIX``), but the model, trained on text with a
      space after each sentence, put a boundary at only 48% of these places, and the sampled
      misses were nearly all real sentence ends.
    - No sentence starts inside a short square-bracket group, nor at its opening bracket unless
      sentence-final punctuation comes before it. The model started one at the bracket even with
      normal spacing ("Goodman | [ | 2] and Haberman"): 56,208 broken pieces in 21,440 of run
      1's abstracts counting numbered citations alone; on 6,000 abstracts with brackets, broken
      pieces fell from 13,315 to 1,400.
    """
    _start_after_glued_punctuation(doc)
    _join_square_brackets(doc)
    return doc


def _start_after_glued_punctuation(doc: Doc) -> None:
    for match in _GLUED_STOP.finditer(doc.text):
        punctuation = doc.char_span(match.start(), match.end())
        if punctuation is None or punctuation.end == len(doc):
            continue
        doc[punctuation.end].is_sent_start = True


def _join_square_brackets(doc: Doc) -> None:
    for match in _SQUARE_BRACKETS.finditer(doc.text):
        group = doc.char_span(match.start(), match.end(), alignment_mode="expand")
        if group is None:
            continue
        for token in group[1:]:
            token.is_sent_start = False
        opening = group[0]
        if opening.i == 0 or doc[opening.i - 1].text in _SENTENCE_FINAL:
            continue
        opening.is_sent_start = False


def cache_serialiser_extensions() -> None:
    """Look srsly's msgpack extensions up once, instead of on every pack and unpack.

    srsly asks its two catalogue registries for extension encoders and decoders each time it
    packs or unpacks, and each ask rescans the installed packages' entry points and filters
    catalogue's global registry, where spaCy keeps thousands of functions: about a third of the
    main process's time in a multi-process pass, which unpacks every finished document (profiled
    2026-11-13). The two registries' contents (srsly's numpy functions; no installed package adds
    any) are read once and served from then on; a function registered later would be missed.
    Worker processes started by fork inherit the cache.
    """
    for registry in (msgpack_encoders, msgpack_decoders):
        functions = registry.get_all()
        registry.get_all = lambda functions=functions: functions


def find_removals(
    doc: Doc, scopes: Mapping[str, Scope], languages: Sequence[SentenceLanguage]
) -> tuple[list[Removal], list[str]]:
    """The removal ranges of one abstract, sorted, and the labels that drop it whole.

    A tokens-scope match removes its own characters; a rest-of-sentence match removes from its
    start to the end of its sentence, so text before a notice survives; an abstract-scope match
    removes its characters and drops the abstract. Matches of patterns with the id ``start``
    count only at the first token. A sentence lingua reads as another language is removed with
    the reason ``not_english``. A range inside another range with the same reason is left out,
    as when one notice holds several matches ("(PsycINFO Database Record (c) 2006").
    """
    removals: set[Removal] = set()
    drop_reasons: list[str] = []
    for span in doc.spans[SPANS_KEY]:
        if span.id_ == START_ID and span.start != 0:
            continue
        label = span.label_
        if scopes[label] == "rest_of_sentence":
            removals.add(Removal(span.start_char, span.sent.end_char, label))
            continue
        removals.add(Removal(span.start_char, span.end_char, label))
        if scopes[label] == "abstract" and label not in drop_reasons:
            drop_reasons.append(label)
    for sentence, language in zip(doc.sents, languages, strict=True):
        if language.language not in (ENGLISH, UNDETERMINED):
            removals.add(Removal(sentence.start_char, sentence.end_char, "not_english"))
    return _outermost(removals), drop_reasons


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
    min_english_words: int,
    batch_size: int = 256,
    process_count: int = 1,
) -> Iterator[tuple[Doc, AnnotatedAbstract]]:
    """Repair and annotate each abstract: its spaCy document and its abstracts-table row.

    Dropped abstracts are annotated too, with their reason, so the corpus review can sample
    them. The reasons, first match wins: ``mojibake`` (read on the raw text), an abstract-scope
    pattern's label, ``not_english`` (some text in another language and fewer than
    ``min_english_words`` words left once it is removed, a provisional threshold: decision agenda
    2.4), ``too_short`` (fewer than five words kept). An abstract printed in several languages
    keeps its English part: other-language sentences are removed, not counted against it.
    """
    signatures = compile_signatures(patterns.mojibake_signatures)
    pairs = ((repair_text(abstract.text), abstract) for abstract in abstracts)
    annotated = nlp.pipe(pairs, as_tuples=True, batch_size=batch_size, n_process=process_count)
    for doc, abstract in annotated:
        yield (
            doc,
            _annotated_abstract(doc, abstract, patterns, signatures, min_english_words),
        )


def _annotated_abstract(
    doc: Doc,
    abstract: RawAbstract,
    patterns: CleaningPatterns,
    signatures: re.Pattern[str],
    min_english_words: int,
) -> AnnotatedAbstract:
    languages = read_sentence_languages(doc)
    removals, drop_reasons = find_removals(doc, patterns.labels, languages)
    if has_mojibake(abstract.text, signatures):
        drop_reasons.insert(0, "mojibake")
    kept_words = count_kept_words(doc, removals)
    has_other_language = any(
        language.language not in (ENGLISH, UNDETERMINED) for language in languages
    )
    if has_other_language and kept_words < min_english_words:
        drop_reasons.append("not_english")
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


def _is_word(token: Token) -> bool:
    return not (token.is_punct or token.is_space)
