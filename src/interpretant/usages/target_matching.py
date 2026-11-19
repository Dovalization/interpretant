"""Finding the word table's targets in a cached spaCy document (decision agenda 4.1, 4.2).

A token matches a single-word target when its spelled lemma or its spelled form is the word, the
form alone for words the table matches by form (*having*), so a verb reading such as *learning*
(lemma *learn*) is still a row whose part of speech decides its scope. Spelled means folded and
written by the spelling map, per hyphen segment, as in the static view, so the two pipelines' counts
reconcile; the lemma exceptions are applied to the document first (*media* keeps its form, apart
from *medium*). A hyphenated token that is not itself a target is searched part by part (4.2): each
part that is a target, or the last part as a target's plural, is a mention recording the compound
and the part's position (head or modifier), its offsets the part's alone, so an encoder can mark
only it (*self-⟨t⟩attention⟨/t⟩*). A phrase target is its modifier's form followed, after a space,
by its head's form or lemma (*neural networks*), or the two as consecutive parts of one hyphenated
token. The matching is a plain pass over the tokens rather than spaCy's ``Matcher``, since it needs
exactly the static view's normalisation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from spacy.tokens import Doc, Token

from interpretant.catalog.words import Word
from interpretant.corpus.spelling import spell
from interpretant.corpus.static_view import fold

Position = Literal["head", "modifier"]
# Plural tags whose lemma can stay the form (a capitalised plural read as a proper noun).
_PLURAL_TAGS = frozenset({"NNS", "NNPS"})


@dataclass(frozen=True)
class TargetMention:
    """One match: the word, its head token, its character span, its compound and position."""

    word: str
    token_index: int
    start_char: int
    end_char: int
    compound: str | None
    position: Position | None


@dataclass(frozen=True)
class TargetMatcher:
    """The words to find, by how they match, and the spelling map."""

    by_form: frozenset[str]
    by_lemma: frozenset[str]
    phrases: Mapping[tuple[str, str], str]
    spellings: Mapping[str, str]
    # Each text's folded and spelled forms, filled as tokens are read: a profile of one part spent
    # a third of the stage's time folding the same texts again (`docs/performance.md`).
    folded: dict[str, str] = field(default_factory=dict, compare=False)
    spelled: dict[str, str] = field(default_factory=dict, compare=False)


def build_target_matcher(words: Iterable[Word], spellings: Mapping[str, str]) -> TargetMatcher:
    """The matcher for the word table's words: single words by form and lemma, and phrases."""
    words = list(words)
    singles = [word for word in words if not word.is_phrase]
    phrases = {}
    for word in words:
        if word.is_phrase:
            modifier, head = word.lemma.split(" ")
            phrases[(modifier, head)] = word.lemma
    return TargetMatcher(
        by_form=frozenset(word.lemma for word in singles),
        by_lemma=frozenset(word.lemma for word in singles if word.match == "lemma"),
        phrases=phrases,
        spellings=dict(spellings),
    )


def match_targets(doc: Doc, matcher: TargetMatcher) -> list[TargetMention]:
    """Every target mention in the document, single words first, in text order within each kind."""
    mentions: list[TargetMention] = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        word = _single_word(token, matcher)
        if word is not None:
            mentions.append(
                TargetMention(word, token.i, token.idx, token.idx + len(token.text), None, None)
            )
        elif "-" in _folded(token.text, matcher):
            mentions.extend(_hyphen_parts(token, matcher))
    mentions.extend(_phrases(doc, matcher))
    return mentions


def _folded(text: str, matcher: TargetMatcher) -> str:
    folded = matcher.folded.get(text)
    if folded is None:
        folded = matcher.folded[text] = fold(text)
    return folded


def _spelled(text: str, matcher: TargetMatcher) -> str:
    spelled = matcher.spelled.get(text)
    if spelled is None:
        spelled = matcher.spelled[text] = spell(_folded(text, matcher), matcher.spellings)
    return spelled


def _single_word(token: Token, matcher: TargetMatcher) -> str | None:
    """The single-word target a token is: by form, by lemma, or as a plural (*Networks*, NNPS)."""
    form = _spelled(token.text, matcher)
    if form in matcher.by_form:
        return form
    lemma = _spelled(token.lemma_, matcher)
    if lemma in matcher.by_lemma:
        return lemma
    return _singular(form, matcher) if token.tag_ in _PLURAL_TAGS else None


def _singular(part: str, matcher: TargetMatcher) -> str | None:
    """The target a compound's last part is the plural of (*networks*, *complexities*), if any."""
    candidates = []
    if part.endswith("ies"):
        candidates.append(part[:-3] + "y")
    if part.endswith("es"):
        candidates.append(part[:-2])
    if part.endswith("s"):
        candidates.append(part[:-1])
    return next((word for word in candidates if word in matcher.by_lemma), None)


def _hyphen_parts(token: Token, matcher: TargetMatcher) -> list[TargetMention]:
    """The targets among a hyphenated token's parts, each with its own offsets."""
    parts = _folded(token.text, matcher).split("-")
    mentions = []
    offset = 0
    for index, part in enumerate(parts):
        spelled = spell(part, matcher.spellings)
        last = index == len(parts) - 1
        word = spelled if spelled in matcher.by_form else None
        if word is None and last:
            word = _singular(spelled, matcher)
        if word is not None and part:
            start = token.idx + offset
            position: Position = "head" if last else "modifier"
            mentions.append(
                TargetMention(word, token.i, start, start + len(part), token.text, position)
            )
        offset += len(part) + 1
    return mentions


def _phrases(doc: Doc, matcher: TargetMatcher) -> list[TargetMention]:
    """The phrase targets: two tokens after a space, or two consecutive hyphen parts."""
    if not matcher.phrases:
        return []
    mentions = []
    for first, second in pairwise(doc):
        if not first.whitespace_:
            continue
        modifier = _spelled(first.text, matcher)
        for head in (_spelled(second.text, matcher), _spelled(second.lemma_, matcher)):
            phrase = matcher.phrases.get((modifier, head))
            if phrase is not None:
                end = second.idx + len(second.text)
                mentions.append(TargetMention(phrase, second.i, first.idx, end, None, None))
                break
    for token in doc:
        if "-" in token.text:
            mentions.extend(_hyphenated_phrases(token, matcher))
    return mentions


def _hyphenated_phrases(token: Token, matcher: TargetMatcher) -> list[TargetMention]:
    """Phrase targets written as consecutive parts of one hyphenated token (*neural-network*)."""
    parts = _folded(token.text, matcher).split("-")
    starts = [sum(len(part) + 1 for part in parts[:index]) for index in range(len(parts))]
    mentions = []
    for index in range(len(parts) - 1):
        modifier = spell(parts[index], matcher.spellings)
        head = spell(parts[index + 1], matcher.spellings)
        last = index + 1 == len(parts) - 1
        phrase = matcher.phrases.get((modifier, head))
        if phrase is None and last:
            singular = _singular_head(head, modifier, matcher)
            phrase = matcher.phrases.get((modifier, singular)) if singular else None
        if phrase is None:
            continue
        start = token.idx + starts[index]
        end = token.idx + starts[index + 1] + len(parts[index + 1])
        whole = len(parts) == 2
        position: Position | None = None if whole else ("head" if last else "modifier")
        mentions.append(TargetMention(phrase, token.i, start, end, token.text, position))
    return mentions


def _singular_head(head: str, modifier: str, matcher: TargetMatcher) -> str | None:
    """The singular of a phrase's plural head (*neural-networks*), if the phrase knows it."""
    candidates = [head[:-2]] if head.endswith("es") else []
    if head.endswith("s"):
        candidates.append(head[:-1])
    return next((word for word in candidates if (modifier, word) in matcher.phrases), None)
