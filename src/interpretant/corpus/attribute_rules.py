"""spaCy's attribute ruler with its single-token rules found by dictionary lookup.

The model's attribute ruler maps each token's fine-grained tag (and, for some words, its
lower-case form) to its part of speech, morphology and lemma. spaCy's matcher tries every rule at
every token, about 80 ns a rule: 12.8 µs per token for the 166 rules the pass keeps, a quarter of
the spaCy workers' time (timed on the remote runner). Most rules are one token tested on its tag,
its lower-case form or both, so they are looked up by those values instead; the others (two tokens,
regular expressions, spaces) stay with a matcher of their own. The matches are the tuples spaCy's
``AttributeRuler.match`` gives, sorted the same way, and spaCy's ``set_annotations`` applies
them: 2.6 µs per token, every tag, part of speech, morphology and lemma the same on 3,000
abstracts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
from typing import Any

from spacy.language import Language
from spacy.matcher import Matcher
from spacy.pipeline import AttributeRuler
from spacy.strings import StringStore
from spacy.tokens import Doc
from spacy.vocab import Vocab

LOOKUP_ATTRIBUTE_RULER = "lookup_attribute_ruler"
_LOOKUP_KEYS = frozenset({"TAG", "LOWER"})

# One match as spaCy's AttributeRuler.match gives it: rule number, match id, start, end.
Match = tuple[int, int, int, int]


class LookupAttributeRuler(AttributeRuler):
    """An ``AttributeRuler`` whose single-token TAG and LOWER rules are found by lookup."""

    def __init__(self, vocab: Vocab, name: str = "attribute_ruler") -> None:
        super().__init__(vocab, name)
        self._reset_index()

    def clear(self) -> None:
        """Remove every rule, from spaCy's matcher and from the lookups."""
        super().clear()
        self._reset_index()

    def add(self, patterns: Iterable[Any], attrs: dict[str, Any], index: int = 0) -> None:
        """Add a rule as spaCy does, and index each of its patterns for ``match``."""
        patterns = list(patterns)
        rule = len(self.attrs)
        super().add(patterns, attrs, index)
        match_id = self.vocab.strings[str(rule)]
        self._match_ids.append(match_id)
        for pattern in patterns:
            keys = _lookup_keys(pattern, self.vocab.strings)
            if keys is None:
                self._other_rules.add(match_id, [pattern])
                continue
            for tag, lower in keys:
                self._rules_by_key[(tag, lower)].append(rule)

    def match(self, doc: Doc) -> list[Match]:
        """Every rule matching in the document, as spaCy's ``AttributeRuler.match`` gives them."""
        lookups = self._rules_by_key
        matches: list[Match] = [
            (rule, self._match_ids[rule], position, position + 1)
            for position, (tag, lower) in enumerate(doc.to_array(["TAG", "LOWER"]).tolist())
            for rule in chain(
                lookups.get((tag, None), ()),
                lookups.get((tag, lower), ()),
                lookups.get((None, lower), ()),
            )
        ]
        if len(self._other_rules):
            matches.extend(
                (int(self.vocab.strings[match_id]), match_id, start, end)
                for match_id, start, end in self._other_rules(doc, allow_missing=True)
            )
        matches.sort()
        return matches

    def _reset_index(self) -> None:
        self._rules_by_key: defaultdict[tuple[int | None, int | None], list[int]] = defaultdict(
            list
        )
        self._match_ids: list[int] = []
        self._other_rules = Matcher(self.vocab)


def _lookup_keys(
    pattern: list[dict[str, Any]], strings: StringStore
) -> list[tuple[int | None, int | None]] | None:
    """The (tag, lower-case form) hashes a one-token pattern matches, or None if not a lookup."""
    if len(pattern) != 1 or not set(pattern[0]) <= _LOOKUP_KEYS:
        return None
    token = pattern[0]
    tags = _values(token["TAG"], strings) if "TAG" in token else [None]
    lowers = _values(token["LOWER"], strings) if "LOWER" in token else [None]
    if tags is None or lowers is None:
        return None
    return [(tag, lower) for tag in tags for lower in lowers]


def _values(spec: object, strings: StringStore) -> list[int | None] | None:
    """The hashes of the exact values a token test accepts: one value or an IN list."""
    if isinstance(spec, str):
        return [strings.add(spec)]
    if isinstance(spec, dict) and set(spec) == {"IN"}:
        return [strings.add(value) for value in spec["IN"]]
    return None


@Language.factory(LOOKUP_ATTRIBUTE_RULER)
def create_lookup_attribute_ruler(nlp: Language, name: str) -> LookupAttributeRuler:
    """The factory spaCy calls to add the lookup attribute ruler."""
    return LookupAttributeRuler(nlp.vocab, name)
