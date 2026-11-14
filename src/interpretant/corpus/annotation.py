"""The spaCy pass: tokens, sentences, part-of-speech tags and lemmas of each repaired abstract.

One pass annotates each repaired abstract (decision agenda, steps 2 and 4.9): sentences by
``senter`` with abbreviation special cases and sentence rules (starts after glued punctuation,
none inside brackets or at a lowercase word), hyphenated words kept whole, DOIs kept
as single tokens. Each document carries its work id. Sentence languages (``corpus.language``) and
the cleaning patterns and keeping rules (``corpus.removals``) are later stages that read these
documents from the annotation cache, so changing them does not rerun the models.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.tokenizer_exceptions import URL_PATTERN
from spacy.language import Language
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import compile_infix_regex
from srsly.msgpack import msgpack_decoders, msgpack_encoders

from interpretant.catalog import WORK_ID
from interpretant.corpus.repair import repair_text
from interpretant.corpus.works import RawAbstract

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
# English's infix rule for a hyphen after a letter or digit and before a letter
# (spacy/lang/en/punctuation.py); removing it keeps *self-attention* one token (decision agenda
# 2.3), as spaCy's documentation shows under "Modifying existing rule sets".
_HYPHEN_INFIX = rf"(?<=[{ALPHA}0-9])(?:{HYPHENS})(?=[{ALPHA}])"
# Sentence punctuation with no space before the next sentence, as in "(r = 0.922).In addition"
# or "in 1950.French delegates" (decision agenda 2.3). spaCy's English rules split a full stop
# between two letters only; these two infix rules also split it after a digit, a closing bracket
# or a quote, and split that bracket or quote off the word before it.
_CLOSERS = r"\)\]\"”’"
_QUOTES = r"\"”’"
_NEXT_SENTENCE = rf"[{ALPHA_UPPER}][{ALPHA_LOWER}]"
GLUED_STOP_INFIX = rf"(?<=[{ALPHA_LOWER}0-9{_CLOSERS}])[.?!](?=[{_QUOTES}]?{_NEXT_SENTENCE})"
GLUED_CLOSER_INFIX = rf"(?<=[{ALPHA}0-9])[{_CLOSERS}](?=[.?!]{_NEXT_SENTENCE})"
# A closing quote after the full stop, glued to the next sentence: 'learns."Acquaintance'.
GLUED_QUOTE_INFIX = rf"(?<=[.?!])[{_QUOTES}](?={_NEXT_SENTENCE})"
# A footnote number glued to the full stop of a sentence: "three weeks.4 Just prior" (12,257 in
# run 2). The three lowercase letters before the stop keep "Vol.3", "Fig.4" and "pp.12" whole.
GLUED_FOOTNOTE_INFIX = r"(?<=[a-z]{3})[.?!](?=\d{1,3}$)"
# Dashes between two words, kept whole when the hyphen rule was removed: "theory—namely",
# "first--applied" (about 46,000 in run 2). En dashes stay whole, like hyphens: they mostly join
# compounds of two equal parts ("sleep–wake", "Mann–Whitney").
DASH_INFIX = rf"(?<=[{ALPHA}])(?:—+|-{{2,3}}|~)(?=[{ALPHA}])"
_GLUED_STOP = re.compile(rf"[.?!][{_QUOTES}]?(?={_NEXT_SENTENCE})")
_GLUED_FOOTNOTE = re.compile(r"(?<=[a-z]{3})[.?!]\d{1,3}\.?\s+(?=[A-Z][a-z])")
# A short bracket group, square or round, not nested: citations ("[2]", "[4, pp. 258-265]",
# "(Smith, 2001)"), abbreviations ("[PSQI]"), phonetic symbols ("[ta]"), "[Formula: see text]".
# spaCy's sentence model often starts a sentence inside one (decision agenda 2.3).
_BRACKET_GROUPS = re.compile(r"\[[^\[\]]{1,40}\]|\([^()]{1,40}\)")
_OPENING_BRACKETS = frozenset({"(", "["})
_SENTENCE_FINAL = frozenset({".", "?", "!"})
# The component setting the sentence boundaries the sentence model misses, before ``senter``.
SENTENCE_RULES = "sentence_rules"
_DOI = r"^(?:doi:)?10\.\d{4,9}/\S+$"
# spaCy's own URL rule extended with DOIs, as one compiled pattern so the tokenizer can still be
# serialised to worker processes.
URL_OR_DOI = re.compile(f"(?u)(?:{_DOI})|(?:{URL_PATTERN})")


def build_pipeline(
    *, named_entity_recognition: bool = False, model: str = "en_core_web_sm"
) -> Language:
    """The spaCy pipeline of the pass: the tokenizer rules, the sentence rules and the models.

    The parser is left out and ``senter`` sets sentences (about ten times faster, and it matched
    the parser on the sample once the abbreviation special cases were in: docs/spacy.md).

    The model's named entity recognition (``ner``) is left out unless
    ``named_entity_recognition`` is set: nothing in cleaning reads named entities, and it takes
    about 44% of a worker's time (timed on bazzite: 30 s instead of 44 s per 10,000 abstracts).
    Matching recognises them on target sentences instead (decision agenda 4.1); there they agree
    with entities found on whole abstracts for 99.87% of 15,018 target mentions.

    Raises:
        ValueError: if spaCy's English hyphen rule has changed and cannot be removed.
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
    infixes += [
        GLUED_STOP_INFIX,
        GLUED_CLOSER_INFIX,
        GLUED_QUOTE_INFIX,
        GLUED_FOOTNOTE_INFIX,
        DASH_INFIX,
    ]
    tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    tokenizer.url_match = URL_OR_DOI.match
    nlp.add_pipe(SENTENCE_RULES, before="senter")
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
    off). The rules, from the audits of runs 1 and 2:

    - A sentence starts after sentence punctuation glued to the next sentence, with or without a
      closing quote ("1950.French", 'learns."Acquaintance'), and after a footnote number glued
      to the full stop ("weeks.4 Just"). The tokenizer splits the punctuation off, but the model,
      trained on text with a space after each sentence, put a boundary at only 48% of the first
      kind and 7% of the footnotes.
    - No sentence starts inside a short bracket group (square or round, up to 40 characters),
      right after an opening bracket, or at the bracket unless sentence-final punctuation comes
      before it. The model started one at a citation's bracket even with normal spacing
      ("Goodman | [ | 2] and Haberman"): 56,208 broken pieces in 21,440 of run 1's abstracts.
    - No sentence starts at a word beginning with a lowercase letter, unless it has a capital or
      a digit inside ("mRNA", "fMRI"). Of run 2's 35,320 such starts, about 64% were wrong splits
      (after units, "etc.)", OCR breaks, abbreviations the tokenizer does not know) and about 20%
      the author's own lowercase, now joined to the sentence before.
    """
    _start_after_glued_punctuation(doc)
    _start_after_glued_footnotes(doc)
    _join_bracket_groups(doc)
    _join_inside_sentences(doc)
    return doc


def _start_after_glued_punctuation(doc: Doc) -> None:
    for match in _GLUED_STOP.finditer(doc.text):
        punctuation = doc.char_span(match.start(), match.end())
        if punctuation is None or punctuation.end == len(doc):
            continue
        doc[punctuation.end].is_sent_start = True


def _start_after_glued_footnotes(doc: Doc) -> None:
    for match in _GLUED_FOOTNOTE.finditer(doc.text):
        following = doc.char_span(match.end(), match.end() + 1, alignment_mode="expand")
        if following is None or following.start_char != match.end():
            continue
        following[0].is_sent_start = True


def _join_bracket_groups(doc: Doc) -> None:
    for match in _BRACKET_GROUPS.finditer(doc.text):
        group = doc.char_span(match.start(), match.end(), alignment_mode="expand")
        if group is None:
            continue
        for token in group[1:]:
            token.is_sent_start = False


def _join_inside_sentences(doc: Doc) -> None:
    for token in doc[1:]:
        if token.text in _OPENING_BRACKETS:
            if token.i + 1 < len(doc):
                doc[token.i + 1].is_sent_start = False
            if doc[token.i - 1].text not in _SENTENCE_FINAL:
                token.is_sent_start = False
        elif _is_lowercase_word(token.text):
            token.is_sent_start = False


def _is_lowercase_word(text: str) -> bool:
    """Whether a token starts with a lowercase letter and has no capital or digit after it."""
    return text[:1].islower() and not any(
        character.isupper() or character.isdigit() for character in text[1:]
    )


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


def annotate_texts(
    abstracts: Iterable[RawAbstract],
    nlp: Language,
    *,
    batch_size: int = 256,
    process_count: int = 1,
) -> Iterator[Doc]:
    """Repair and annotate each abstract, in order; each document carries its work id."""
    pairs = ((repair_text(abstract.text), abstract.work_id) for abstract in abstracts)
    for doc, work_id in nlp.pipe(
        pairs, as_tuples=True, batch_size=batch_size, n_process=process_count
    ):
        doc.user_data[WORK_ID] = work_id
        yield doc
