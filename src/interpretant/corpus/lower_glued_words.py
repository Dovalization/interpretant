r"""Lower-case glued words: words run together without a space, found from corpus counts.

Decision agenda 2.3, run 8. OCR and conversion lose the space between words (*ofthe*,
*theresults*, *socalled*). A lower-case word the lemma index does not know is a candidate when it
splits into words spaCy knows (``corpus.dictionary``) that the corpus uses at least
``min_part_count`` times; it is split when the corpus writes the parts apart, with a space or a
hyphen, at least ``apart_margin`` times as often as joined, with the separator the corpus uses
more (*of the*, *so-called*, *real-time*). Solid compounds (*login*) fall below the margin.

Two words, the first of two letters or more, are split wherever they appear. Three patterns hit
mostly real words, foreign words and names outside damaged text, so they apply only in a text
where the two-word rule splits at least ``min_dense_splits`` words (mostly OCR text): *a* and a
word (*asubstantial*); three or more words (*oftheproposed*), segmented by symspellpy's
``word_segmentation`` on the corpus's word counts with spelling correction off, every
neighbouring pair written apart that often and every part of one or two letters a stop word; and
a capitalised two-word glue, the capital kept on the first part (*Thestudy*). Where a glued word
touches a hyphen, the separator follows how the corpus writes the whole hyphenated chunk
(*stateof-the-art* into *state-of-the-art*, *ofman-machine* into *of man-machine*), since the
tokenizer keeps hyphenated words whole. Nothing right after a backslash is split (LaTeX's
``\forall``), nor a word touching a digit or inside a web address; the curated kept-joined list
keeps its words whole.

Counting reads Polars expressions over the whole corpus; the split is applied text by text before
the spaCy pass, after the broken words are joined (run 7c), so a broken word is never cut.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from itertools import pairwise

import polars as pl
from spacy.lang.en.stop_words import STOP_WORDS
from symspellpy import SymSpell

from interpretant.catalog import LOWER_GLUED_WORDS_SCHEMA
from interpretant.corpus.dictionary import is_dictionary_word
from interpretant.corpus.phrase_counts import count_forms
from interpretant.corpus.repair import inside_address

# The patterns, as the table names them.
TWO_WORDS = "two_words"
A_AND_WORD = "a_and_word"
THREE_OR_MORE_WORDS = "three_or_more_words"
HYPHENATED_CHUNK = "hyphenated_chunk"
# Applied, a capitalised two-word glue is counted apart.
CAPITALISED = "capitalised"
SPACE = " "
HYPHEN = "-"
# A shorter word is never a candidate.
MIN_WORD_LETTERS = 3
# Only a word at least this long is segmented into three or more words.
MIN_SEGMENTED_LETTERS = 6
# The longest part the segmentation considers.
MAX_PART_LETTERS = 20
# The only part of one letter.
_ONE_LETTER_PART = "a"

# In Polars' syntax (the regex crate): a word for the word counts; a lower-case or capitalised
# word, the shapes a glued word is looked for in; a run of words joined by hyphens.
_WORD_PATTERN = r"\p{L}+"
_GLUED_FORM_PATTERN = r"\b\p{Lu}?\p{Ll}{2,}\b"
_CHUNK_PATTERN = r"\p{L}+(?:-\p{L}+)+"
# A word in Python's syntax: letters only.
_WORD = re.compile(r"[^\W\d_]+")

# The counts are the decision table's columns before ``kept_joined`` and ``split``.
LOWER_GLUED_WORD_COUNTS_SCHEMA = pl.Schema(
    {
        name: dtype
        for name, dtype in LOWER_GLUED_WORDS_SCHEMA.items()
        if name not in ("kept_joined", "split")
    }
)


@dataclass(frozen=True)
class GluedSplit:
    """How a glued word is split: its pattern, its parts and the separators between them."""

    pattern: str
    parts: tuple[str, ...]
    separators: tuple[str, ...]


@dataclass(frozen=True)
class LowerGluedSplits:
    """The words to split, folded, and each hyphenated chunk's separator for its glued word.

    Two-word splits apply everywhere, the guarded patterns only in damaged texts.
    """

    two_words: Mapping[str, GluedSplit] = field(default_factory=dict)
    guarded: Mapping[str, GluedSplit] = field(default_factory=dict)
    chunks: Mapping[str, str] = field(default_factory=dict)


def count_lower_glued_words(
    texts: pl.DataFrame, *, min_part_count: int, min_first_part_letters: int
) -> pl.DataFrame:
    """Each candidate glued word of the texts, folded, with the counts that decide its split.

    ``texts`` has a ``text`` column. A word of at least three letters, written in lower case or
    capitalised and unknown to spaCy, is cut into two parts at every place where both are known
    words (or the first is *a*) used at least ``min_part_count`` times; the cut written apart
    most often is kept, as ``two_words`` when its first part has ``min_first_part_letters`` or
    more, as ``a_and_word`` when it is *a*. A word with no such cut is segmented by symspellpy
    into three or more such parts (``three_or_more_words``), every part of one or two letters a
    stop word. ``joined`` counts the word whole; ``apart`` its parts written apart, with a space
    or a hyphen (the weakest pair of three or more); ``uses`` and ``capitalised_uses`` its uses
    in lower case and capitalised. Sorted by uses.
    """
    words: dict[str, int] = dict(
        texts.lazy()
        .select(word=pl.col("text").str.to_lowercase().str.extract_all(_WORD_PATTERN))
        .explode("word")
        .drop_nulls()
        .group_by("word")
        .len("count")
        .collect(engine="streaming")
        .iter_rows()
    )
    forms = (
        texts.lazy()
        .select(token=pl.col("text").str.extract_all(_GLUED_FORM_PATTERN))
        .explode("token")
        .drop_nulls()
        .group_by(pl.col("token").str.to_lowercase().alias("word"))
        .agg(
            uses=pl.col("token").str.contains(r"^\p{Ll}").sum().cast(pl.Int64()),
            capitalised_uses=pl.col("token").str.contains(r"^\p{Lu}").sum().cast(pl.Int64()),
        )
        .collect(engine="streaming")
    )
    parts = _PartChecker(words, min_part_count)
    segmenter = _segmenter(words, parts)
    candidates: list[tuple[str, str, tuple[str, ...], int, int]] = []
    for word, uses, capitalised_uses in forms.iter_rows():
        if len(word) < MIN_WORD_LETTERS:
            continue
        cuts = [
            (word[:cut], word[cut:])
            for cut in range(1, len(word))
            if parts.is_part(word[:cut]) and parts.is_part(word[cut:])
        ]
        if cuts:
            candidates.extend(
                (word, _two_part_pattern(left), (left, right), uses, capitalised_uses)
                for left, right in cuts
                if left == _ONE_LETTER_PART or len(left) >= min_first_part_letters
            )
            continue
        if len(word) < MIN_SEGMENTED_LETTERS:
            continue
        found = tuple(
            segmenter.word_segmentation(
                word, max_edit_distance=0, max_segmentation_word_length=MAX_PART_LETTERS
            ).segmented_string.split()
        )
        if len(found) >= 3 and all(_is_segment(part, parts) for part in found):
            candidates.append((word, THREE_OR_MORE_WORDS, found, uses, capitalised_uses))
    candidates = [row for row in candidates if not is_dictionary_word(row[0])]
    pairs = {
        (row[2][index], row[2][index + 1]) for row in candidates for index in range(len(row[2]) - 1)
    }
    spaced, hyphenated = _pair_counts(texts, pairs)
    best: dict[str, tuple[int, str, tuple[str, ...], tuple[str, ...], int, int]] = {}
    for word, pattern, cut, uses, capitalised_uses in candidates:
        neighbours = list(pairwise(cut))
        apart = min(spaced.get(pair, 0) + hyphenated.get(pair, 0) for pair in neighbours)
        separators = tuple(
            HYPHEN if hyphenated.get(pair, 0) > spaced.get(pair, 0) else SPACE
            for pair in neighbours
        )
        if word not in best or apart > best[word][0]:
            best[word] = (apart, pattern, cut, separators, uses, capitalised_uses)
    rows = [
        {
            "word": word,
            "pattern": pattern,
            "parts": list(cut),
            "separators": list(separators),
            "joined": words.get(word, 0),
            "apart": apart,
            "uses": uses,
            "capitalised_uses": capitalised_uses,
        }
        for word, (apart, pattern, cut, separators, uses, capitalised_uses) in best.items()
    ]
    return (
        pl.DataFrame(rows, schema=LOWER_GLUED_WORD_COUNTS_SCHEMA)
        .with_columns(total=pl.col("uses") + pl.col("capitalised_uses"))
        .sort(["total", "word"], descending=[True, False])
        .drop("total")
    )


def decide_lower_glued_word_splits(
    counts: pl.DataFrame, *, apart_margin: float, kept_joined: Collection[str]
) -> pl.DataFrame:
    """The counts with each word's decisions: ``kept_joined`` and ``split``.

    A word is split when the corpus writes its parts apart at least ``apart_margin`` times as
    often as joined, unless it is in ``kept_joined`` (any case).
    """
    folded = sorted({word.lower() for word in kept_joined})
    return counts.with_columns(kept_joined=pl.col("word").is_in(folded)).with_columns(
        split=(pl.col("apart") >= apart_margin * pl.col("joined")) & ~pl.col("kept_joined")
    )


def decide_hyphenated_chunks(texts: pl.DataFrame, decisions: pl.DataFrame) -> pl.DataFrame:
    """The separator of each hyphenated chunk holding one two-part word to split, as decision rows.

    A chunk is words joined by hyphens (*stateof-the-art*). The two ways of writing it with the
    word split, by a hyphen (*state-of-the-art*) or a space (*state of-the-art*), are counted,
    and the more frequent gives the chunk's separator; a chunk written neither way is left to the
    word's own separator. Each row has the chunk as ``word``, the pattern ``hyphenated_chunk``,
    the glued word's parts and the chosen separator; ``joined`` counts the chunk as written and
    ``apart`` the chosen form.
    """
    two_part = decisions.filter(pl.col("split") & (pl.col("parts").list.len() == 2))
    to_split: dict[str, list[str]] = dict(two_part.select("word", "parts").iter_rows())
    chunks = (
        texts.lazy()
        .select(chunk=pl.col("text").str.to_lowercase().str.extract_all(_CHUNK_PATTERN))
        .explode("chunk")
        .drop_nulls()
        .group_by("chunk")
        .len("joined")
        .collect(engine="streaming")
    )
    forms: dict[str, tuple[list[str], str, str, int]] = {}
    for chunk, joined in chunks.iter_rows():
        pieces = chunk.split(HYPHEN)
        glued = [index for index, piece in enumerate(pieces) if piece in to_split]
        if len(glued) != 1:
            continue
        parts = to_split[pieces[glued[0]]]

        def written(
            separator: str,
            pieces: list[str] = pieces,
            index: int = glued[0],
            parts: list[str] = parts,
        ) -> str:
            return HYPHEN.join([*pieces[:index], separator.join(parts), *pieces[index + 1 :]])

        forms[chunk] = (parts, written(HYPHEN), written(SPACE), joined)
    found = dict(
        count_forms(
            texts, pl.Series([form for _, *written, _ in forms.values() for form in written])
        ).iter_rows()
    )
    rows = []
    for chunk, (parts, with_hyphen, with_space, joined) in forms.items():
        hyphen_count, space_count = found.get(with_hyphen, 0), found.get(with_space, 0)
        if hyphen_count == space_count == 0:
            continue
        rows.append(
            {
                "word": chunk,
                "pattern": HYPHENATED_CHUNK,
                "parts": parts,
                "separators": [HYPHEN if hyphen_count > space_count else SPACE],
                "joined": joined,
                "apart": max(hyphen_count, space_count),
                "uses": joined,
                "capitalised_uses": 0,
                "kept_joined": False,
                "split": True,
            }
        )
    return pl.DataFrame(rows, schema=LOWER_GLUED_WORDS_SCHEMA).sort("word")


def lower_glued_word_splits(decisions: pl.DataFrame) -> LowerGluedSplits:
    """The splits the decisions make, by pattern, and the hyphenated chunks' separators."""
    two_words: dict[str, GluedSplit] = {}
    guarded: dict[str, GluedSplit] = {}
    chunks: dict[str, str] = {}
    chosen = decisions.filter(pl.col("split")).select("word", "pattern", "parts", "separators")
    for word, pattern, parts, separators in chosen.iter_rows():
        if pattern == HYPHENATED_CHUNK:
            chunks[word] = separators[0]
            continue
        split = GluedSplit(pattern, tuple(parts), tuple(separators))
        (two_words if pattern == TWO_WORDS else guarded)[word] = split
    return LowerGluedSplits(two_words=two_words, guarded=guarded, chunks=chunks)


def split_lower_glued_words(
    text: str, splits: LowerGluedSplits, *, min_dense_splits: int
) -> tuple[str, Counter[str]]:
    """The text with its glued words split, and how many were split by pattern.

    A lower-case two-word glue is split everywhere; when at least ``min_dense_splits`` of them
    are split in the text, so are the guarded patterns in lower case and the capitalised
    two-word glue (counted as ``capitalised``), the capital kept on the first part. A word right
    after a backslash, touching a digit or inside a web address is left whole; a word touching a
    hyphen takes its hyphenated chunk's separator when the chunk has one.
    """
    if not (splits.two_words or splits.guarded):
        return text, Counter()
    words = [
        word for word in _WORD.finditer(text) if len(word.group().lower()) == len(word.group())
    ]
    lower_two_word = sum(
        1
        for word in words
        if word.group() in splits.two_words and word.group().islower() and _may_split(text, word)
    )
    is_damaged = lower_two_word >= min_dense_splits
    pieces: list[str] = []
    counts: Counter[str] = Counter()
    position = 0
    for word in words:
        token = word.group()
        split, label = _split_for(token, splits, is_damaged=is_damaged)
        if split is None or not _may_split(text, word):
            continue
        separators = split.separators
        chunk_separator = splits.chunks.get(_chunk_around(text, word.start(), word.end()))
        if chunk_separator is not None and len(split.parts) == 2:
            separators = (chunk_separator,)
        pieces.append(text[position : word.start()])
        pieces.append(_written_apart(token, split.parts, separators))
        position = word.end()
        counts[label] += 1
    pieces.append(text[position:])
    return "".join(pieces), counts


def _two_part_pattern(left: str) -> str:
    return A_AND_WORD if left == _ONE_LETTER_PART else TWO_WORDS


def _is_segment(part: str, parts: _PartChecker) -> bool:
    """Whether a segment is a known part, and a stop word if it has one or two letters."""
    return parts.is_part(part) and (len(part) > 2 or part in STOP_WORDS)


class _PartChecker:
    """Whether a piece of a glued word can be a part.

    A part is *a*, or a word spaCy knows that the corpus uses at least ``min_part_count`` times.
    """

    def __init__(self, words: Mapping[str, int], min_part_count: int) -> None:
        self._words = words
        self._min_part_count = min_part_count
        self._known: dict[str, bool] = {}

    def is_part(self, piece: str) -> bool:
        if len(piece) == 1:
            return piece == _ONE_LETTER_PART
        if piece not in self._known:
            self._known[piece] = self._words.get(
                piece, 0
            ) >= self._min_part_count and is_dictionary_word(piece)
        return self._known[piece]


def _segmenter(words: Mapping[str, int], parts: _PartChecker) -> SymSpell:
    """The segmenter of symspellpy on the corpus's counts of the words that can be parts.

    Spelling correction is off (an edit distance of 0), so the segments are the word's letters.
    """
    segmenter = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    for word, count in words.items():
        if len(word) <= MAX_PART_LETTERS and parts.is_part(word):
            segmenter.create_dictionary_entry(word, count)
    return segmenter


def _pair_counts(
    texts: pl.DataFrame, pairs: Collection[tuple[str, str]]
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """How often each pair of words is written with a space and with a hyphen between them."""
    if not pairs:
        return {}, {}
    forms = {
        f"{left}{separator}{right}": (left, right)
        for left, right in pairs
        for separator in (SPACE, HYPHEN)
    }
    found = dict(count_forms(texts, pl.Series(list(forms))).iter_rows())
    spaced = {(left, right): found.get(f"{left}{SPACE}{right}", 0) for left, right in pairs}
    hyphenated = {(left, right): found.get(f"{left}{HYPHEN}{right}", 0) for left, right in pairs}
    return spaced, hyphenated


def _split_for(
    token: str, splits: LowerGluedSplits, *, is_damaged: bool
) -> tuple[GluedSplit | None, str]:
    """The split a word as written takes here, and the label it is counted under."""
    if token.islower():
        if token in splits.two_words:
            return splits.two_words[token], TWO_WORDS
        if is_damaged and token in splits.guarded:
            split = splits.guarded[token]
            return split, split.pattern
        return None, ""
    if is_damaged and token[0].isupper() and token[1:].islower():
        return splits.two_words.get(token.lower()), CAPITALISED
    return None, ""


def _may_split(text: str, word: re.Match[str]) -> bool:
    """Whether a word may be split here: not after a backslash, a digit or inside an address."""
    start, end = word.start(), word.end()
    if start > 0 and (text[start - 1] == "\\" or text[start - 1].isdigit()):
        return False
    if end < len(text) and text[end].isdigit():
        return False
    return not inside_address(text, start, end)


def _chunk_around(text: str, start: int, end: int) -> str:
    """The hyphenated chunk the word at ``start``-``end`` belongs to, folded ("" if none)."""
    first, last = start, end
    while first > 1 and text[first - 1] == HYPHEN and text[first - 2].isalpha():
        first -= 1
        while first > 0 and text[first - 1].isalpha():
            first -= 1
    while last < len(text) - 1 and text[last] == HYPHEN and text[last + 1].isalpha():
        last += 1
        while last < len(text) and text[last].isalpha():
            last += 1
    if (first, last) == (start, end):
        return ""
    return text[first:last].lower()


def _written_apart(token: str, parts: tuple[str, ...], separators: tuple[str, ...]) -> str:
    """The word as written, cut after each part's letters with its separator."""
    pieces = []
    position = 0
    for index, part in enumerate(parts):
        pieces.append(token[position : position + len(part)])
        position += len(part)
        if index < len(separators):
            pieces.append(separators[index])
    return "".join(pieces)
