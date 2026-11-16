"""Split words: words broken by a space or a line-end hyphen, found from corpus counts.

Decision agenda 2.3, run 7c. A word broken at the end of a line keeps the typesetter's hyphen and
a space (*fol- lowing*) or only the hyphen (*infor-mation*), and OCR text breaks words with a
space (*in vestigation*, *sys tem*). The corpus tells a broken word from two words: it writes a
word broken only here and there whole hundreds of times as often, while a real phrase or a real
hyphenated word (*multi-modal*, *self-esteem*) is written apart or hyphenated far more often. A
break is a candidate only when one of its pieces is not a word spaCy knows, so *a void* is never
read as *avoid*. Real hyphenated words stay as the authors wrote them; the spellings step gives
them the standard spelling (3.4).

Counting reads Polars expressions over the whole corpus (Aho-Corasick for the many literal broken
forms); the join is applied text by text before the spaCy pass, as a lookup on each pair of
neighbouring words, outside web addresses.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field

import polars as pl

from interpretant.corpus.dictionary import is_dictionary_word
from interpretant.corpus.repair import inside_address

# The three ways a word is broken, by what separates its pieces.
SPACE = "space"
HYPHEN_SPACE = "hyphen_space"
HYPHEN = "hyphen"
_SEPARATORS = {SPACE: " ", HYPHEN_SPACE: "- ", HYPHEN: "-"}
# A word shorter than this is never a candidate (a two-letter word broken is mostly two words).
MIN_WORD_LETTERS = 4
# A hyphen alone joins only pieces of at least this many letters (*x-ray*, *t-test*, *k-means*).
MIN_HYPHENATED_PIECE_LETTERS = 2

# In Polars' syntax (the regex crate): a word, the characters the counts read as a space, and
# runs of white space.
_WORD_PATTERN = r"\p{L}+"
_NOT_TEXT_PATTERN = r"[^\p{L}\p{N}\s-]+"
_SPACES_PATTERN = r"\s+"
# A word in Python's syntax: letters only. A word is a run of letters, so what touches it is not
# a letter, but it can be a digit (*1st*, *3rd*), and such a piece is never joined.
_WORD = re.compile(r"[^\W\d_]+")
# The citation formula "et al.": a name before it is never joined to its "et" (*Bull et al.*).
_CITATION_ET = "et"
_CITATION_AL = re.compile(r"\s*al\b")

SPLIT_WORD_COUNTS_SCHEMA = pl.Schema(
    {
        "word": pl.String(),
        "left": pl.String(),
        "right": pl.String(),
        "left_is_word": pl.Boolean(),
        "right_is_word": pl.Boolean(),
        "joined": pl.Int64(),
        SPACE: pl.Int64(),
        HYPHEN_SPACE: pl.Int64(),
        HYPHEN: pl.Int64(),
    }
)


@dataclass(frozen=True)
class SplitWordJoins:
    """The pieces to join, folded: each second piece with the first pieces it joins.

    ``spaced`` joins after a space or a hyphen and a space, ``hyphenated`` after a hyphen alone.
    """

    spaced: Mapping[str, frozenset[str]] = field(default_factory=dict)
    hyphenated: Mapping[str, frozenset[str]] = field(default_factory=dict)


def count_split_words(texts: pl.DataFrame, *, min_word_count: int) -> pl.DataFrame:
    """Each break of a word in the texts, with the counts that decide whether it is joined.

    ``texts`` has a ``text`` column. A word the corpus writes at least ``min_word_count`` times
    (case folded) and at least four letters long is cut in two at every place where both pieces
    also occur as words and at least one is not a word spaCy knows. One row per cut found broken
    at least once, sorted by how often: the word, its pieces and whether spaCy knows each, how
    often the corpus writes it whole, and how often broken by a space, by a hyphen and a space,
    and by a hyphen (case folded; any character other than a letter, digit, space or hyphen reads
    as a space).
    """
    words = (
        texts.lazy()
        .select(word=pl.col("text").str.to_lowercase().str.extract_all(_WORD_PATTERN))
        .explode("word")
        .drop_nulls("word")
        .group_by("word")
        .len("joined")
        .collect(engine="streaming")
    )
    cuts = pl.DataFrame(
        list(_cuts(dict(words.iter_rows()), min_word_count)),
        schema=["word", "left", "right", "left_is_word", "right_is_word"],
        orient="row",
    )
    if cuts.is_empty():
        return pl.DataFrame(schema=SPLIT_WORD_COUNTS_SCHEMA)
    forms = cuts.select(
        "word",
        "left",
        "right",
        **{
            kind: pl.concat_str(pl.col("left"), pl.lit(separator), pl.col("right"))
            for kind, separator in _SEPARATORS.items()
        },
    )
    found = _count_forms(texts, pl.concat([forms[kind] for kind in _SEPARATORS]))
    table = cuts.join(words, on="word")
    for kind in _SEPARATORS:
        table = table.join(
            forms.select("word", "left", form=kind)
            .join(found, on="form", how="left")
            .select("word", "left", pl.col("count").fill_null(0).alias(kind)),
            on=["word", "left"],
        )
    broken = sum(pl.col(kind) for kind in _SEPARATORS)
    return (
        table.filter(broken > 0)
        .select(SPLIT_WORD_COUNTS_SCHEMA.names())
        .cast(SPLIT_WORD_COUNTS_SCHEMA)
        .sort([broken, "word", "left"], descending=[True, False, False])
    )


def decide_split_word_joins(
    counts: pl.DataFrame,
    *,
    join_margin: float,
    hyphen_margin: float,
    kept_apart: Collection[tuple[str, str]],
) -> pl.DataFrame:
    """The counts with each break's decisions: ``kept_apart``, ``join`` and ``join_hyphenated``.

    A break by a space, or by a hyphen and a space, is joined when the corpus writes the word
    whole at least ``join_margin`` times as often as broken in any of the three ways (``join``),
    so a word often hyphenated (*multi-modal*) is a real hyphenated word and stays apart. A break
    by a hyphen alone is joined only when its second piece is not a word spaCy knows, both pieces
    have at least two letters (*x-ray*, *t-test* stay) and the corpus writes the word whole at
    least ``hyphen_margin`` times as often (``join_hyphenated``). A pair of pieces in
    ``kept_apart`` (any case) is a real phrase and stays apart whatever the counts say.
    """
    folded = sorted(f"{first.lower()} {second.lower()}" for first, second in kept_apart)
    is_kept_apart = pl.concat_str(pl.col("left"), pl.col("right"), separator=" ").is_in(folded)
    whole = pl.col("joined")
    broken = pl.col(SPACE) + pl.col(HYPHEN_SPACE) + pl.col(HYPHEN)
    pieces_long_enough = (pl.col("left").str.len_chars() >= MIN_HYPHENATED_PIECE_LETTERS) & (
        pl.col("right").str.len_chars() >= MIN_HYPHENATED_PIECE_LETTERS
    )
    return counts.with_columns(kept_apart=is_kept_apart).with_columns(
        join=(whole >= join_margin * broken) & ~pl.col("kept_apart"),
        join_hyphenated=(whole >= hyphen_margin * broken)
        & ~pl.col("right_is_word")
        & pieces_long_enough
        & ~pl.col("kept_apart"),
    )


def split_word_joins(decisions: pl.DataFrame) -> SplitWordJoins:
    """The pieces each decision joins, folded."""
    return SplitWordJoins(
        spaced=_pairs(decisions.filter(pl.col("join"))),
        hyphenated=_pairs(decisions.filter(pl.col("join_hyphenated"))),
    )


def join_split_words(text: str, joins: SplitWordJoins) -> str:
    """The text with each broken word in ``joins`` written whole, outside web addresses.

    Two neighbouring words are joined when they are a pair to join after what separates them (a
    run of white space, or a hyphen followed by one; or a hyphen alone) and read as one word in
    their case: the first piece in lower case or capitalised, the second in lower case, so
    initials and acronyms (*William S.*, *MOR as*) stay apart. A joined word is not joined again
    with the next.
    """
    if not (joins.spaced or joins.hyphenated):
        return text
    words = list(_WORD.finditer(text))
    pieces: list[str] = []
    position = 0
    index = 0
    while index < len(words) - 1:
        left, right = words[index], words[index + 1]
        if _joins(text, left, right, joins):
            pieces.append(text[position : left.start()])
            pieces.append(left.group() + right.group())
            position = right.end()
            index += 2
        else:
            index += 1
    pieces.append(text[position:])
    return "".join(pieces)


def _cuts(
    counts: dict[str, int], min_word_count: int
) -> Iterator[tuple[str, str, str, bool, bool]]:
    """Each word's cuts into two pieces that occur as words, not both words spaCy knows."""
    for word, count in counts.items():
        if count < min_word_count or len(word) < MIN_WORD_LETTERS:
            continue
        for cut in range(1, len(word)):
            left, right = word[:cut], word[cut:]
            if left not in counts or right not in counts:
                continue
            left_is_word, right_is_word = is_dictionary_word(left), is_dictionary_word(right)
            if not (left_is_word and right_is_word):
                yield word, left, right, left_is_word, right_is_word


def _count_forms(texts: pl.DataFrame, forms: pl.Series) -> pl.DataFrame:
    """How often each form appears between spaces, case folded, other characters as spaces.

    One lazy query, which Polars runs in parallel (as ``Series`` methods it ran on one core).
    """
    normalised = (
        " "
        + pl.col("text")
        .str.to_lowercase()
        .str.replace_all(_NOT_TEXT_PATTERN, " ")
        .str.replace_all(_SPACES_PATTERN, " ")
        + " "
    )
    patterns = (" " + forms.unique() + " ").to_list()
    return (
        texts.lazy()
        .select(form=normalised.str.extract_many(patterns, overlapping=True))
        .explode("form")
        .drop_nulls()
        .group_by(pl.col("form").str.strip_chars(" "))
        .len("count")
        .collect(engine="streaming")
    )


def _pairs(decisions: pl.DataFrame) -> dict[str, frozenset[str]]:
    """Each second piece with the first pieces it joins."""
    lefts: defaultdict[str, set[str]] = defaultdict(set)
    for left, right in decisions.select("left", "right").iter_rows():
        lefts[right].add(left)
    return {right: frozenset(firsts) for right, firsts in lefts.items()}


def _joins(text: str, left: re.Match[str], right: re.Match[str], joins: SplitWordJoins) -> bool:
    """Whether two neighbouring words are a broken word to join here.

    The pairs are looked up by the second piece as written, which matches only in lower case.
    """
    gap = text[left.end() : right.start()]
    hyphen_alone = gap == "-"
    firsts = (joins.hyphenated if hyphen_alone else joins.spaced).get(right.group())
    if firsts is None:
        return False
    if not hyphen_alone and not (gap.isspace() or (gap[0] == "-" and gap[1:].isspace())):
        return False
    first = left.group()
    if not (first.islower() or first.istitle()) or first.lower() not in firsts:
        return False
    if _touches_a_digit(text, left.start(), right.end()):
        return False
    if right.group() == _CITATION_ET and _CITATION_AL.match(text, right.end()):
        return False
    return not inside_address(text, left.start(), right.end())


def _touches_a_digit(text: str, start: int, end: int) -> bool:
    """Whether a digit comes right before ``start`` or right after ``end``."""
    return (start > 0 and text[start - 1].isdigit()) or (end < len(text) and text[end].isdigit())
