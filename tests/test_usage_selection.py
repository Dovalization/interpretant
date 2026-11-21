"""Scope and the draw: one in-scope usage per abstract and word, independent of everything else."""

from __future__ import annotations

import polars as pl

from interpretant.catalog.words import Word
from interpretant.usages.identifiers import build_draw_key
from interpretant.usages.usage_selection import find_copies, is_in_scope, mark_copies, mark_drawn

MEDIUM = Word(lemma="medium", pos=("NOUN", "PROPN"), match="lemma", held_out=False)
SEED = 20261301


def usage(
    work: int, word: str, key: str, *, in_scope: bool = True, excluded: str | None = None
) -> dict[str, object]:
    """One usages row with the columns the draw reads."""
    work_id = f"https://openalex.org/W{work}"
    return {
        "work_id": work_id,
        "word": word,
        "draw_key": build_draw_key(SEED, work_id, word, key),
        "in_scope": in_scope,
        "sentence_exclusion": excluded,
    }


def drawn(rows: list[dict[str, object]]) -> set[tuple[str, str, str]]:
    """The drawn usages as (work, word, draw key)."""
    table = mark_drawn(pl.DataFrame(rows))
    return {
        (w, d, k) for w, d, k in table.filter("drawn").select("work_id", "word", "draw_key").rows()
    }


def test_is_in_scope_needs_the_part_of_speech_and_no_name() -> None:
    assert is_in_scope("NOUN", MEDIUM, inside_name=False, hyphen_part=False)
    assert not is_in_scope("ADJ", MEDIUM, inside_name=False, hyphen_part=False)
    assert not is_in_scope("PROPN", MEDIUM, inside_name=True, hyphen_part=False)


def test_is_in_scope_keeps_a_hyphen_part_whatever_its_words_tag_but_not_in_a_name() -> None:
    assert is_in_scope("ADJ", MEDIUM, inside_name=False, hyphen_part=True)
    assert not is_in_scope("ADJ", MEDIUM, inside_name=True, hyphen_part=True)


def test_is_in_scope_leaves_out_a_hyphen_part_read_as_a_foreign_word() -> None:
    assert not is_in_scope("X", MEDIUM, inside_name=False, hyphen_part=True)


def test_mark_drawn_draws_one_eligible_usage_per_work_and_word() -> None:
    rows = [
        usage(1, "network", "a" * 20),
        usage(1, "network", "b" * 20),
        usage(1, "network", "c" * 20, in_scope=False),
        usage(1, "network", "d" * 20, excluded="cut_off"),
        usage(1, "memory", "a" * 20),
        usage(2, "network", "a" * 20, in_scope=False),
    ]
    table = mark_drawn(pl.DataFrame(rows))
    counts = dict(
        table.group_by("work_id", "word")
        .agg(pl.col("drawn").sum())
        .select(pl.concat_str("work_id", "word", separator="|"), "drawn")
        .iter_rows()
    )
    assert counts == {
        "https://openalex.org/W1|network": 1,
        "https://openalex.org/W1|memory": 1,
        "https://openalex.org/W2|network": 0,
    }
    picked = table.filter("drawn", pl.col("word") == "network")["draw_key"].item()
    assert picked == min(str(rows[0]["draw_key"]), str(rows[1]["draw_key"]))


def test_the_draw_does_not_depend_on_other_abstracts_words_or_row_order() -> None:
    base = [
        usage(1, "network", "a" * 20),
        usage(1, "network", "b" * 20),
        usage(2, "network", "e" * 20),
    ]
    before = drawn(base)
    more = [*base, usage(3, "network", "f" * 20), usage(1, "memory", "g" * 20)]
    assert before <= drawn(more)
    assert drawn(list(reversed(base))) == before
    fewer = [row for row in base if row["work_id"] != "https://openalex.org/W2"]
    assert drawn(fewer) == {row for row in before if row[0] != "https://openalex.org/W2"}


def shared(*rows: tuple[str, str, int]) -> pl.DataFrame:
    """Usages as (work, content key, sentence number)."""
    return pl.DataFrame(rows, schema=["work_id", "content_key", "sentence_number"], orient="row")


def test_works_sharing_three_target_sentences_are_one_work_twice() -> None:
    usages = shared(
        *[("W1", f"k{n}", n) for n in range(3)],
        *[("W2", f"k{n}", n + 4) for n in range(3)],
        ("W3", "k0", 0),
        ("W3", "k1", 1),
    )
    years = {"W1": 1995, "W2": 1991, "W3": 1980}
    assert find_copies(usages, years) == {"W1": "W2"}


def test_targets_in_one_sentence_count_once_and_a_group_keeps_its_earliest_work() -> None:
    one_sentence = shared(*[(w, f"k{n}", 0) for w in ("W1", "W2") for n in range(3)])
    assert find_copies(one_sentence, {"W1": 1990, "W2": 1991}) == {}
    chain = shared(
        *[(w, f"a{n}", n) for w in ("W1", "W2") for n in range(3)],
        *[(w, f"b{n}", n) for w in ("W2", "W3") for n in range(3)],
    )
    years = {"W1": 2001, "W2": 1999, "W3": 2001}
    assert find_copies(chain, years) == {"W1": "W2", "W3": "W2"}


def test_a_copys_usages_are_marked_and_not_drawn() -> None:
    table = pl.DataFrame({"work_id": ["W1", "W2"], "drawn": [True, True]})
    marked = mark_copies(table, {"W1": "W2"})
    assert marked["copy_of"].to_list() == ["W2", None]
    assert marked["drawn"].to_list() == [False, True]
