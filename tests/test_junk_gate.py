"""The junk gate's draws: the items' text, the sets, the order, the key and the counts."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from interpretant.catalog.junk_gate import JunkLabel, JunkLabels, Run3Verdict
from interpretant.catalog.words import Word
from interpretant.corpus.removals import Removal
from interpretant.review.junk_gate import (
    CONFIRMATION_SEED,
    CUT,
    ENRICHMENT_SEED,
    LABELLED_SEED,
    LIST_LIKE,
    NO_FINITE_VERB,
    RECORDS_SCHEMA,
    SHORT_REPOSITORY_ONLY,
    build_confirmation_items,
    build_items,
    build_run3_items,
    build_strata,
    count_sample,
    count_strata,
    draw_confirmation,
    draw_enrichment,
    draw_labelled,
    render_record,
)
from interpretant.usages.identifiers import build_draw_key

ATTENTION = Word(lemma="attention", pos=("NOUN",), match="lemma", held_out=False)
MEMORY = Word(
    lemma="memory", pos=("NOUN",), match="lemma", held_out=True, sealed_name="held-out:0123456789ab"
)
WORDS = {"attention": ATTENTION, "memory": MEMORY}
TEXT = "Abstract: We study attention. It matters a lot. Copyright 2001 Elsevier."
COPYRIGHT = TEXT.index("Copyright")
REMOVALS = [Removal(0, 9, "abstract_label"), Removal(COPYRIGHT, len(TEXT), "copyright")]


def span_of(text: str, sentence: str) -> tuple[int, int]:
    start = text.index(sentence)
    return start, start + len(sentence)


def test_render_record_marks_the_sentence_in_the_kept_text() -> None:
    shown = render_record(TEXT, REMOVALS, span_of(TEXT, "It matters a lot."))

    assert shown == "We study attention. ⟦It matters a lot.⟧"


def test_render_record_without_a_sentence_shows_the_kept_text_only() -> None:
    assert render_record(TEXT, REMOVALS) == "We study attention. It matters a lot."


def test_render_record_keeps_a_mark_that_falls_in_a_removed_range_beside_the_kept_text() -> None:
    start = TEXT.index("It matters")
    shown = render_record(TEXT, REMOVALS, (start, COPYRIGHT + 5))

    assert "Copyright" not in shown
    assert shown.startswith("We study attention. ⟦It matters a lot.")
    assert shown.endswith("⟧")


def marked(before: int, sentence: int, after: int) -> tuple[str, str, tuple[int, int]]:
    """A text of letters, its shown form with the middle run marked, and that run's span."""
    text = "a" * before + "S" * sentence + "b" * after
    shown = "a" * before + "⟦" + "S" * sentence + "⟧" + "b" * after
    return text, shown, (before, before + sentence)


def test_render_record_shows_a_text_of_up_to_six_thousand_characters_whole() -> None:
    text, shown, sentence = marked(3000, 20, 2978)

    assert len(shown) == 6000
    assert render_record(text, [], sentence) == shown


def test_render_record_keeps_the_start_and_end_of_a_long_text() -> None:
    text, shown, sentence = marked(3000, 20, 7000)

    assert render_record(text, [], sentence) == shown[:3500] + CUT + shown[-1500:]
    assert render_record("a" * 7000, []) == "a" * 3500 + CUT + "a" * 1500


def test_render_record_shows_a_sentence_in_the_gap_with_three_hundred_characters_around() -> None:
    text, shown, sentence = marked(5000, 20, 5000)

    assert render_record(text, [], sentence) == (
        shown[:3500] + CUT + shown[4700:5322] + CUT + shown[-1500:]
    )


def test_render_record_joins_a_sentence_window_that_meets_the_start() -> None:
    text, shown, sentence = marked(3400, 200, 7000)

    assert render_record(text, [], sentence) == shown[:3902] + CUT + shown[-1500:]


def usage(
    work: int, word: str, decade: int, *, list_like: bool = False, drawn: bool = True
) -> dict[str, Any]:
    return {
        "work_id": f"W{work}",
        "word": word,
        "content_key": f"{work}:{word}",
        "sentence_start": 0,
        "sentence_end": 10,
        "in_scope": True,
        "drawn": drawn,
        "list_like": list_like,
        "part": f"ai_{decade}",
        "decade": decade,
    }


def record(
    work: int,
    *,
    has_finite_verb: bool = True,
    is_repository_only: bool = False,
    kept_characters: int = 1000,
) -> dict[str, Any]:
    return {
        "work_id": f"W{work}",
        "removals": [],
        "has_finite_verb": has_finite_verb,
        "is_repository_only": is_repository_only,
        "kept_characters": kept_characters,
    }


def records_table(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=RECORDS_SCHEMA)


def key_of(seed: int, row: dict[str, Any]) -> str:
    return build_draw_key(seed, row["work_id"], row["word"], row["content_key"])


def identities(table: pl.DataFrame) -> set[tuple[str, str, str]]:
    return set(table.select("work_id", "word", "content_key").iter_rows())


def two_decades() -> list[dict[str, Any]]:
    rows = [usage(work, "attention", 1990) for work in range(20)]
    rows += [usage(100 + work, "memory", 1990) for work in range(10)]
    return rows + [usage(200 + work, "attention", 2000) for work in range(4)]


def test_draw_labelled_takes_each_decades_smallest_keys_over_all_words() -> None:
    rows = two_decades()

    labelled = draw_labelled(pl.DataFrame(rows), seed=LABELLED_SEED, per_decade=5)

    nineties = sorted(key_of(LABELLED_SEED, row) for row in rows if row["decade"] == 1990)
    assert dict(labelled.group_by("decade").len().iter_rows()) == {1990: 5, 2000: 4}
    assert labelled.filter(pl.col("decade") == 1990)["key"].to_list() == nineties[:5]


def test_draw_labelled_gives_the_same_rows_whatever_the_tables_order() -> None:
    usages = pl.DataFrame(two_decades())
    expected = draw_labelled(usages, seed=LABELLED_SEED, per_decade=5).to_dicts()

    for seed in range(5):
        shuffled = usages.sample(fraction=1.0, shuffle=True, seed=seed)
        assert draw_labelled(shuffled, seed=LABELLED_SEED, per_decade=5).to_dicts() == expected


def test_an_added_record_takes_at_most_the_place_of_the_last_one_drawn() -> None:
    rows = two_decades()
    before = draw_labelled(pl.DataFrame(rows), seed=LABELLED_SEED, per_decade=5)
    cutoff = before.filter(pl.col("decade") == 1990)["key"].max()
    added = next(
        row
        for row in (usage(1000 + work, "attention", 1990) for work in range(1000))
        if key_of(LABELLED_SEED, row) < cutoff
    )

    after = draw_labelled(pl.DataFrame([*rows, added]), seed=LABELLED_SEED, per_decade=5)

    displaced = before.filter(pl.col("key") == cutoff)
    assert identities(after) == identities(before) - identities(displaced) | identities(
        pl.DataFrame([added])
    )


def enrichment_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Usages and records whose strata do not overlap: 10 records in each."""
    rows = [usage(work, "attention", 1990) for work in range(10)]
    rows += [usage(100 + work, "attention", 1990, list_like=True) for work in range(10)]
    rows += [usage(200 + work, "memory", 2000) for work in range(10)]
    rows += [usage(300 + work, "memory", 2000) for work in range(10)]
    records = [record(work, has_finite_verb=False) for work in range(10)]
    records += [record(100 + work) for work in range(10)]
    records += [
        record(200 + work, is_repository_only=True, kept_characters=300) for work in range(10)
    ]
    records += [record(300 + work) for work in range(10)]
    return rows, records


def draw_sample_sets(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """L and E as the sample draws them, with small sizes."""
    drawn = pl.DataFrame(rows)
    labelled = draw_labelled(drawn, seed=LABELLED_SEED, per_decade=5)
    strata = build_strata(drawn, records_table(records))
    enrichment = draw_enrichment(
        drawn, strata, set(labelled["work_id"]), seed=ENRICHMENT_SEED, per_stratum=3
    )
    return labelled, enrichment


def select(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[Any, str]:
    """Each drawn usage of L and E, with its set or stratum."""
    labelled, enrichment = draw_sample_sets(rows, records)
    chosen = dict.fromkeys(identities(labelled), "labelled")
    for row in enrichment.to_dicts():
        chosen[(row["work_id"], row["word"], row["content_key"])] = row["stratum"]
    return chosen


def test_an_added_record_outside_the_draws_changes_no_other_records_selection() -> None:
    rows, records = enrichment_corpus()
    labelled, enrichment = draw_sample_sets(rows, records)
    labelled_cutoff = labelled.filter(pl.col("decade") == 1990)["key"].max()
    enrichment_cutoff = enrichment.filter(pl.col("stratum") == NO_FINITE_VERB)["key"].max()
    added = next(
        row
        for row in (usage(1000 + work, "attention", 1990) for work in range(1000))
        if key_of(LABELLED_SEED, row) > labelled_cutoff
        and key_of(ENRICHMENT_SEED, row) > enrichment_cutoff
    )
    added_record = record(int(added["work_id"][1:]), has_finite_verb=False)
    before = select(rows, records)

    assert select([*rows, added], [*records, added_record]) == before


def test_an_added_record_in_a_stratum_takes_only_the_place_of_its_last_record() -> None:
    rows, records = enrichment_corpus()
    drawn = pl.DataFrame(rows)
    strata = build_strata(drawn, records_table(records))
    before = draw_enrichment(drawn, strata, set(), seed=ENRICHMENT_SEED, per_stratum=3)
    first_stratum = before.filter(pl.col("stratum") == NO_FINITE_VERB)
    cutoff = first_stratum["key"].max()
    added = next(
        row
        for row in (usage(1000 + work, "attention", 1990) for work in range(1000))
        if key_of(ENRICHMENT_SEED, row) < cutoff
    )
    more = pl.DataFrame([*rows, added])
    after = draw_enrichment(
        more,
        build_strata(
            more,
            records_table([*records, record(int(added["work_id"][1:]), has_finite_verb=False)]),
        ),
        set(),
        seed=ENRICHMENT_SEED,
        per_stratum=3,
    )

    displaced = first_stratum.filter(pl.col("key") == cutoff)
    assert identities(after) == identities(before) - identities(displaced) | identities(
        pl.DataFrame([added])
    )


def test_draw_enrichment_skips_the_labelled_records_and_draws_each_record_once() -> None:
    rows = [usage(work, "attention", 1990, list_like=True) for work in range(12)]
    records = [
        record(work, has_finite_verb=False, is_repository_only=True, kept_characters=100)
        for work in range(12)
    ]
    drawn = pl.DataFrame(rows)
    strata = build_strata(drawn, records_table(records))

    enrichment = draw_enrichment(drawn, strata, {"W0", "W1"}, seed=ENRICHMENT_SEED, per_stratum=4)

    assert not {"W0", "W1"} & set(enrichment["work_id"])
    assert enrichment["work_id"].n_unique() == enrichment.height == 10
    assert dict(enrichment.group_by("stratum").len().iter_rows()) == {
        NO_FINITE_VERB: 4,
        LIST_LIKE: 4,
        SHORT_REPOSITORY_ONLY: 2,
    }


def test_draw_enrichment_marks_a_records_list_like_usage_or_its_smallest_key() -> None:
    rows = [
        usage(1, "attention", 1990, list_like=True),
        usage(1, "memory", 1990),
        usage(2, "attention", 1990),
        usage(2, "memory", 1990),
    ]
    records = [record(1), record(2, has_finite_verb=False)]
    drawn = pl.DataFrame(rows)

    enrichment = draw_enrichment(
        drawn, build_strata(drawn, records_table(records)), set(), seed=3, per_stratum=5
    )

    marked = {row["stratum"]: row for row in enrichment.to_dicts()}
    assert marked[LIST_LIKE]["word"] == "attention"
    smallest = min(rows[2:], key=lambda row: key_of(3, row))
    assert marked[NO_FINITE_VERB]["word"] == smallest["word"]


def test_build_strata_reads_each_stratum_from_the_records() -> None:
    rows = [usage(1, "attention", 1990, list_like=True), usage(2, "attention", 1990)]
    rows += [usage(3, "attention", 1990), usage(4, "attention", 1990)]
    records = [
        record(1),
        record(2, has_finite_verb=False),
        record(3, is_repository_only=True, kept_characters=499),
        record(4, is_repository_only=True, kept_characters=500),
    ]

    strata = build_strata(pl.DataFrame(rows), records_table(records))

    assert count_strata(strata) == {NO_FINITE_VERB: 1, LIST_LIKE: 1, SHORT_REPOSITORY_ONLY: 1}


def test_build_strata_refuses_a_record_without_a_description() -> None:
    rows = [usage(1, "attention", 1990), usage(2, "attention", 1990)]

    with pytest.raises(ValueError, match="1 records"):
        build_strata(pl.DataFrame(rows), records_table([record(1)]))


def test_draw_confirmation_takes_each_flagged_cells_smallest_keys() -> None:
    rows = [usage(work, "attention", 1990) for work in range(10)]
    rows += [usage(100 + work, "memory", 1990) for work in range(10)]
    rows += [usage(200 + work, "attention", 2000) for work in range(10)]
    drawn = pl.DataFrame(rows).with_columns(pl.col("decade").cast(pl.Int32))
    cells = pl.DataFrame(
        {"word": ["attention", "memory", "attention"], "decade": [1990, 1990, 2000]},
        schema={"word": pl.String(), "decade": pl.Int64()},
    ).with_columns(flagged=pl.Series([True, False, True]))

    chosen = draw_confirmation(drawn, cells, seed=CONFIRMATION_SEED, per_cell=4)

    assert sorted(chosen.group_by("word", "decade").len().iter_rows()) == [
        ("attention", 1990, 4),
        ("attention", 2000, 4),
    ]
    nineties = sorted(key_of(CONFIRMATION_SEED, row) for row in rows[:10])
    assert chosen.filter(pl.col("decade") == 1990)["key"].to_list() == nineties[:4]


def sample_row(work: int, word: str, item_set: str, stratum: str | None = None) -> dict[str, Any]:
    return {
        "work_id": f"W{work}",
        "word": word,
        "content_key": f"{work}:{word}",
        "sentence_start": TEXT.index("It"),
        "sentence_end": COPYRIGHT - 1,
        "part": "ai_1990",
        "decade": 1990,
        "set": item_set,
        "stratum": stratum,
    }


def test_build_items_numbers_items_in_seeded_order_and_shows_only_their_text() -> None:
    samples = [sample_row(work, "attention", "labelled") for work in range(5)]
    samples.append(sample_row(9, "memory", "enrichment", LIST_LIKE))
    texts = {row["work_id"]: TEXT for row in samples}
    removals = {row["work_id"]: REMOVALS for row in samples}

    items, key = build_items(samples, texts, removals, WORDS, prefix="S")

    assert [item["id"] for item in items] == [f"S{number:04d}" for number in range(1, 7)]
    assert all(set(item) == {"id", "text"} for item in items)
    assert items[0]["text"] == "We study attention. ⟦It matters a lot.⟧"
    assert build_items(list(reversed(samples)), texts, removals, WORDS, prefix="S") == (
        items,
        key,
    )
    order = sorted(samples, key=lambda row: key_of(20261405, row))
    assert [entry["work_id"] for entry in key.values()] == [row["work_id"] for row in order]
    [enrichment] = [entry for entry in key.values() if entry["set"] == "enrichment"]
    assert enrichment["stratum"] == LIST_LIKE
    assert enrichment["word"] == MEMORY.reported_name
    assert all("stratum" not in entry for entry in key.values() if entry["set"] == "labelled")


def label(work: int, lemma: str) -> JunkLabel:
    return JunkLabel(
        set="labelled",
        stratum=None,
        work_id=f"W{work}",
        part="ai_1990",
        decade=1990,
        word=WORDS[lemma].reported_name,
        content_key=f"{work}:{lemma}",
        verdict="abstract",
        kind="none",
        sentence="own",
        damaged="no",
    )


def test_build_confirmation_items_reuses_a_label_and_lists_every_usage_of_a_cell() -> None:
    chosen = pl.DataFrame([sample_row(work, "memory", "") for work in range(3)]).drop(
        "set", "stratum"
    )
    labels = JunkLabels(codebook_sha256="0" * 64, items={"S0042": label(1, "memory")})
    texts = {f"W{work}": TEXT for work in range(3)}
    removals = {f"W{work}": REMOVALS for work in range(3)}

    items, key = build_confirmation_items(chosen, labels, texts, removals, WORDS)

    assert [item["id"] for item in items] == ["C0001", "C0002"]
    assert {entry["work_id"] for entry in key["items"].values()} == {"W0", "W2"}
    assert all(entry["set"] == "confirmation" for entry in key["items"].values())
    [cell] = key["cells"]
    assert (cell["word"], cell["decade"]) == (MEMORY.reported_name, 1990)
    by_work = dict(zip(["W0", "W1", "W2"], cell["items"], strict=True))
    assert by_work["W1"] == "S0042"
    assert {by_work["W0"], by_work["W2"]} == {"C0001", "C0002"}


def test_build_run3_items_keeps_run3s_ids_and_marks_no_sentence() -> None:
    verdicts = [
        Run3Verdict(id="r000", work_id="W1", decade=1950, field="psychology", verdict="abstract"),
        Run3Verdict(id="r001", work_id="W2", decade=1960, field="ai", verdict="unsure"),
    ]
    texts = {"W1": TEXT, "W2": "A plain abstract."}
    removals: dict[str, list[Removal]] = {"W1": REMOVALS, "W2": []}

    items, key = build_run3_items(verdicts, texts, removals)

    assert items == [
        {"id": "r000", "text": "We study attention. It matters a lot."},
        {"id": "r001", "text": "A plain abstract."},
    ]
    assert key["r001"] == {"set": "run3", "work_id": "W2", "decade": 1960, "field": "ai"}


def test_build_run3_items_refuses_a_judged_record_without_its_text() -> None:
    verdicts = [Run3Verdict(id="r000", work_id="W1", decade=1950, field="ai", verdict="abstract")]

    with pytest.raises(ValueError, match="1 of run 3's judged records"):
        build_run3_items(verdicts, {}, {})


def test_count_sample_counts_decades_words_and_the_gated_cells() -> None:
    rows = [usage(work, "attention", 1990) for work in range(100)]
    rows += [usage(work, "memory", 1990) for work in range(3)]
    rows += [usage(200, "memory", 2000), usage(201, "memory", 2000, drawn=False)]
    usages = pl.DataFrame(rows)
    key = {"S0001": {"set": "labelled"}, "S0002": {"set": "enrichment"}}
    strata = {NO_FINITE_VERB: 3, LIST_LIKE: 0, SHORT_REPOSITORY_ONLY: 1}

    counts = count_sample(usages, WORDS, strata, key)

    assert counts["drawn_usages_by_decade"] == {"1990": 103, "2000": 1}
    assert counts["records_by_decade"] == {"1990": 100, "2000": 1}
    assert counts["usages_by_word_and_decade"] == {
        "attention": {"1990": {"matched": 100, "in_scope": 100, "drawn": 100}},
        MEMORY.reported_name: {
            "1990": {"matched": 3, "in_scope": 3, "drawn": 3},
            "2000": {"matched": 2, "in_scope": 2, "drawn": 1},
        },
    }
    assert counts["cells_with_at_least_100_drawn"] == 1
    assert counts["enrichment_strata"] == strata
    assert counts["items"] == {"labelled": 1, "enrichment": 1}
    assert counts["held_out_words"] == 1
