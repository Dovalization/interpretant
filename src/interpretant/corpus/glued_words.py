"""Glued words: two words run together where the source lost the space, found from corpus counts.

Decision agenda 2.3, run 7. A word with a lower-case letter run straight into a capital is either
two words that lost their space (*AbstractThe*, *ofThe*, *To the EditorWe*) or a name or term
written that way (*PubMed*, *YouTube*, *McCarthy*). The corpus tells them apart: it mostly writes
a lost space's two words apart, and a name joined. Only words with one such boundary and at least
two letters before it are counted, so *mRNA*, *pH* and menus run together
(*BinderNameCancelCreateExport*) are never split.

Counting reads Polars expressions over the whole corpus (Aho-Corasick for the many literal
phrases); the split itself is applied text by text before the spaCy pass, as a dictionary lookup
on each candidate word, so a word is replaced only whole and never inside a web address.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

import polars as pl
from spacy.lang.char_classes import ALPHA_LOWER, ALPHA_UPPER

from interpretant.corpus.repair import inside_address

# The reasons a glued word is split, or kept although the counts would split it.
APART = "apart"
SENTENCE_OPENER = "sentence_opener"
KEPT_JOINED = "kept_joined"
# A first part of one letter is a prefix or a symbol (*mRNA*, *pH*, *eLearning*), never a word.
MIN_FIRST_PART_LETTERS = 2

# In Polars' syntax (the regex crate): a whole word holding a lower-case letter before a capital,
# that boundary, the word's two parts, and a capitalised word opening a sentence.
_GLUED_WORD_PATTERN = r"\b\p{L}*\p{Ll}\p{Lu}\p{L}*\b"
_BOUNDARY_PATTERN = r"\p{Ll}\p{Lu}"
_PARTS_PATTERN = r"^(?<left>\p{L}*\p{Ll})(?<right>\p{Lu}\p{L}*)$"
_SENTENCE_OPENER_PATTERN = r"(?:^|[.!?] )\p{Lu}\p{Ll}*\b"
# What separates two words in the counts: any run of characters that are not letters or digits.
_SEPARATORS_PATTERN = r"[^\p{L}\p{N}]+"
# The same candidate words in Python's syntax, with spaCy's letter classes as in repair.
_GLUED_WORD = re.compile(rf"\b[^\W\d_]*[{ALPHA_LOWER}][{ALPHA_UPPER}][^\W\d_]*\b")

GLUED_WORD_COUNTS_SCHEMA = pl.Schema(
    {
        "word": pl.String(),
        "left": pl.String(),
        "right": pl.String(),
        "works": pl.Int64(),
        "occurrences": pl.Int64(),
        "joined": pl.Int64(),
        "apart": pl.Int64(),
        "left_alone": pl.Int64(),
        "opener_rank": pl.Int64(),
    }
)


def count_glued_words(texts: pl.DataFrame) -> pl.DataFrame:
    """Each glued word of the texts, with the counts that decide whether it is split.

    ``texts`` has a ``work_id`` and a ``text`` column. One row per glued word with a single
    boundary and at least two letters before it, sorted by occurrences: its two parts, how many
    works hold it and how often, as written; then, with case folded, how often the corpus writes
    it joined, how often its parts apart (only spaces or punctuation between), how often its
    first part stands alone, and the rank of its second part among the capitalised words that
    open sentences (1 for the most common; null when it never opens one).
    """
    forms = (
        texts.select("work_id", word=pl.col("text").str.extract_all(_GLUED_WORD_PATTERN))
        .explode("word")
        .drop_nulls("word")
        .filter(pl.col("word").str.count_matches(_BOUNDARY_PATTERN) == 1)
        .group_by("word")
        .agg(works=pl.col("work_id").n_unique(), occurrences=pl.len())
        .with_columns(parts=pl.col("word").str.extract_groups(_PARTS_PATTERN))
        .unnest("parts")
        .filter(pl.col("left").str.len_chars() >= MIN_FIRST_PART_LETTERS)
    )
    if forms.is_empty():
        return pl.DataFrame(schema=GLUED_WORD_COUNTS_SCHEMA)
    phrases = forms.select(
        "word",
        joined_phrase=pl.col("word").str.to_lowercase(),
        apart_phrase=pl.concat_str(
            pl.col("left").str.to_lowercase(), pl.col("right").str.to_lowercase(), separator=" "
        ),
        left_phrase=pl.col("left").str.to_lowercase(),
    )
    counts = _count_phrases(
        texts["text"],
        pl.concat(
            [phrases["joined_phrase"], phrases["apart_phrase"], phrases["left_phrase"]]
        ).unique(),
    )
    table = phrases
    for phrase, column in (
        ("joined_phrase", "joined"),
        ("apart_phrase", "apart"),
        ("left_phrase", "left_alone"),
    ):
        table = table.join(
            counts.rename({"count": column}), left_on=phrase, right_on="phrase", how="left"
        )
    return (
        forms.join(table.select("word", "joined", "apart", "left_alone"), on="word")
        .join(_rank_sentence_openers(texts["text"]), left_on="right", right_on="opener", how="left")
        .with_columns(pl.col("joined", "apart", "left_alone").fill_null(0))
        .select(GLUED_WORD_COUNTS_SCHEMA.names())
        .cast(GLUED_WORD_COUNTS_SCHEMA)
        .sort(["occurrences", "word"], descending=[True, False])
    )


def decide_glued_word_splits(
    counts: pl.DataFrame,
    *,
    apart_margin: float,
    opener_count: int,
    opener_min_left_letters: int,
    opener_min_right_letters: int,
    kept_joined: Collection[str],
) -> pl.DataFrame:
    """The counts with each glued word's decision: ``split``, and the ``reason`` for it.

    A word is split when the corpus writes its parts apart at least ``apart_margin`` times as
    often as joined (``apart``), or when its second part is among the ``opener_count`` most common
    sentence openers, its first part has at least ``opener_min_left_letters`` letters and its
    second part at least ``opener_min_right_letters`` (``sentence_opener``: a heading run into the
    first word of the text, *EditorsThis*, joined in every issue of a journal; a second part of
    one letter is mostly a final "l" read as "I", *proposaI*). A word in ``kept_joined`` (any
    case) stays joined whatever the counts say (``kept_joined``).
    """
    folded = sorted({word.lower() for word in kept_joined})
    is_kept = pl.col("word").str.to_lowercase().is_in(folded)
    is_apart = pl.col("apart") >= apart_margin * pl.max_horizontal(pl.col("joined"), 1)
    opens_sentences = (
        (pl.col("opener_rank") <= opener_count)
        & (pl.col("left").str.len_chars() >= opener_min_left_letters)
        & (pl.col("right").str.len_chars() >= opener_min_right_letters)
    )
    reason = (
        pl.when(is_kept)
        .then(pl.lit(KEPT_JOINED))
        .when(is_apart)
        .then(pl.lit(APART))
        .when(opens_sentences.fill_null(value=False))
        .then(pl.lit(SENTENCE_OPENER))
    )
    return counts.with_columns(reason=reason).with_columns(
        split=pl.col("reason").is_in([APART, SENTENCE_OPENER]).fill_null(value=False)
    )


def glued_word_splits(decisions: pl.DataFrame) -> dict[str, str]:
    """Each glued word the decisions split, mapped to its two parts with a space between."""
    chosen = decisions.filter(pl.col("split"))
    spaced = chosen["left"] + " " + chosen["right"]
    return dict(zip(chosen["word"].to_list(), spaced.to_list(), strict=True))


def split_glued_words(text: str, splits: Mapping[str, str]) -> str:
    """The text with each whole word found in ``splits`` replaced, outside web addresses."""
    if not splits:
        return text

    def split(match: re.Match[str]) -> str:
        parts = splits.get(match.group())
        if parts is None or inside_address(text, match.start(), match.end()):
            return match.group()
        return parts

    return _GLUED_WORD.sub(split, text)


def _count_phrases(texts: pl.Series, phrases: pl.Series) -> pl.DataFrame:
    """How often each phrase appears as whole words, case folded, separators read as spaces."""
    normalised = " " + texts.str.to_lowercase().str.replace_all(_SEPARATORS_PATTERN, " ") + " "
    patterns = (" " + phrases + " ").to_list()
    hits = normalised.str.extract_many(patterns, overlapping=True).explode().drop_nulls()
    return hits.str.strip_chars(" ").rename("phrase").value_counts()


def _rank_sentence_openers(texts: pl.Series) -> pl.DataFrame:
    """Each capitalised word that opens a sentence, ranked by how often (1 for the most)."""
    openers = texts.str.extract_all(_SENTENCE_OPENER_PATTERN).explode().drop_nulls()
    return (
        openers.str.strip_chars_start(".!? ")
        .rename("opener")
        .value_counts()
        .select("opener", opener_rank=pl.col("count").rank("min", descending=True))
    )
