"""Spelling variants written as one standard spelling (decision agenda 3.4; phase step 2).

The map (`curated/spellings.tsv`) sends each variant form to the spelling the corpus uses more
often (the author, 2026-11-17), a word-table lemma keeping its own (*behaviour* → *behavior*). It
applies to each hyphen segment of a folded token, one token for one, so a static view's offsets do
not change, and the same function serves target matching on the folded lemma (4.1).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def spell(word: str, spellings: Mapping[str, str]) -> str:
    """The word with each hyphen segment written in its standard spelling."""
    if "-" not in word:
        return spellings.get(word, word)
    return "-".join(spellings.get(segment, segment) for segment in word.split("-"))


def mapped_away(spellings: Mapping[str, str], words: Iterable[str]) -> list[str]:
    """The words the map would write as another spelling (none of the word table may be)."""
    return sorted(word for word in words if spell(word, spellings) != word)
