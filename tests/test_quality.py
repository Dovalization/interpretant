from __future__ import annotations

from interpretant.corpus.quality import (
    clean_abstract,
    clean_cells,
    english_share,
    shared_sentences,
)

ENGLISH = (
    "We study how the meaning of a word changes over time in the abstracts of a field, and we "
    "show that the shares of its senses move when the field takes up a new method."
)
FRENCH = (
    "L'argument de Davidson de l'interprete omniscient repose sur le presuppose qu'un interprete "
    "omniscient est possible, et que certaines conditions sont remplies dans chaque cas etudie."
)
NOTICE = "These keywords were added by machine and not by the authors."
NEWSPAPER = "East Carolina's student-run campus newspaper was first published in 1923 as a weekly."


def test_english_function_words_separate_english_from_french() -> None:
    assert english_share(ENGLISH) > 0.3
    assert english_share(FRENCH) < 0.15


def test_a_non_english_abstract_is_dropped_and_an_english_one_kept() -> None:
    assert clean_abstract(FRENCH, set()) is None
    assert clean_abstract(ENGLISH, set()) == ENGLISH


def test_a_short_text_is_not_judged_by_language() -> None:
    assert clean_abstract("Analyse de texte", set()) == "Analyse de texte"


def test_a_notice_sentence_is_stripped_from_a_real_abstract() -> None:
    shared = {NOTICE.lower()}
    assert clean_abstract(f"{ENGLISH} {NOTICE}", shared) == ENGLISH


def test_a_record_made_mostly_of_shared_sentences_is_dropped() -> None:
    shared = {NEWSPAPER.lower()}
    assert clean_abstract(f"{NEWSPAPER} Issue 4.", shared) is None


def test_exam_dump_advertising_is_dropped() -> None:
    spam = "Choose Dumps4it AI-900 actual questions to pass the Microsoft certification exam."
    assert clean_abstract(spam, set()) is None


def test_shared_sentences_need_enough_abstracts() -> None:
    texts = [f"{ENGLISH} {NOTICE}"] * 3 + [ENGLISH]
    assert NOTICE.lower() in shared_sentences(texts, min_abstracts=3)
    assert NOTICE.lower() not in shared_sentences(texts, min_abstracts=4)


def test_clean_cells_keeps_ids_and_leaves_dropped_records_out() -> None:
    cells = {("ai", 2020): [("w1", ENGLISH), ("w2", FRENCH)]}
    assert clean_cells(cells, set()) == {("ai", 2020): [("w1", ENGLISH)]}
