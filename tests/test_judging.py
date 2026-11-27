"""Review items as the judges see them (`review/judging.py`): the answers and the rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, get_args

from interpretant.review.judging import (
    ADJUDICATION,
    RECORD_DAMAGE,
    RECORD_KINDS,
    RECORD_SENTENCES,
    RECORD_VERDICTS,
    SCHEMAS,
    Answers,
    build_batch_schema,
    render,
    render_message,
)

REPOSITORY = Path(__file__).resolve().parents[1]
JUNK_GATE_CODEBOOK = REPOSITORY / "curated" / "judging" / "junk_gate" / "prompt.md"
ITEMS: dict[str, dict[str, Any]] = {
    "usages": {
        "id": "U0001",
        "kind": "left_out",
        "word": "network",
        "parts_of_speech": ["NOUN", "PROPN"],
        "reason": "name",
        "sentence": "The ⟨t⟩Network⟨/t⟩ of Sensors met.",
    },
    "comparison": {
        "id": "C0001",
        "word": "agent",
        "parts_of_speech": ["NOUN"],
        "sentence": "An ⟨t⟩Agent⟨/t⟩ acts.",
        "A": {"pos": "NOUN", "lemma": "agent", "name": None},
        "B": {"pos": "PROPN", "lemma": "Agent", "name": "ORG"},
    },
    "records": {
        "id": "R0001",
        "text": "We study attention. ⟦It drifts.⟧\n[… cut …]\nWe conclude.",
    },
}
JUNK = {
    "note": "A table of contents.",
    "verdict": "junk",
    "kind": "contents_or_list",
    "sentence": "other",
    "damaged": "no",
}
ABSTRACT = {
    "note": "A short summary.",
    "verdict": "abstract",
    "kind": "none",
    "sentence": "own",
    "damaged": "no",
}


def test_every_kind_of_answers_has_a_schema_and_a_rendering() -> None:
    assert sorted(SCHEMAS) == sorted(get_args(Answers))
    for answers in get_args(Answers):
        assert render(ITEMS[answers], answers).startswith(f"Item id: {ITEMS[answers]['id']}\n")


def test_a_usage_is_rendered_with_its_word_and_why_it_was_left_out() -> None:
    assert render(ITEMS["usages"], "usages") == (
        "Item id: U0001\n"
        "Kind: left_out\n"
        "Target word: network\n"
        "Parts of speech it is studied in: NOUN, PROPN\n"
        "The program left it out for: name\n"
        "Sentence: The ⟨t⟩Network⟨/t⟩ of Sensors met."
    )


def test_a_comparison_is_rendered_with_both_readings() -> None:
    assert render(ITEMS["comparison"], "comparison") == (
        "Item id: C0001\n"
        "Target word: agent\n"
        "Parts of speech it is studied in: NOUN\n"
        "Sentence: An ⟨t⟩Agent⟨/t⟩ acts.\n"
        "Reading A: part of speech NOUN, lemma agent, name none\n"
        "Reading B: part of speech PROPN, lemma Agent, name ORG"
    )


def test_a_record_is_rendered_as_its_id_and_its_text() -> None:
    assert render(ITEMS["records"], "records") == (
        "Item id: R0001\nWe study attention. ⟦It drifts.⟧\n[… cut …]\nWe conclude."
    )


def test_a_disputed_record_is_sent_with_both_answers_and_the_instruction() -> None:
    disputed = ITEMS["records"] | {"judge_1": JUNK, "judge_2": ABSTRACT}

    text = render_message([disputed], "records")

    assert text.startswith(ADJUDICATION)
    assert text.endswith(
        "We conclude.\n"
        f"Judge 1 answered: {json.dumps(JUNK)}\n"
        f"Judge 2 answered: {json.dumps(ABSTRACT)}"
    )


def test_a_records_answer_gives_its_note_before_its_verdict() -> None:
    schema = SCHEMAS["records"]

    assert list(schema["properties"]) == ["note", "verdict", "kind", "sentence", "damaged"]
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["verdict"]["enum"] == list(RECORD_VERDICTS)
    assert schema["properties"]["kind"]["enum"] == list(RECORD_KINDS)
    assert schema["properties"]["sentence"]["enum"] == list(RECORD_SENTENCES)
    assert schema["properties"]["damaged"]["enum"] == list(RECORD_DAMAGE)


def test_several_records_are_answered_with_each_ones_item_id() -> None:
    answer = build_batch_schema("records")["properties"]["answers"]["items"]

    assert answer["required"] == ["id", "note", "verdict", "kind", "sentence", "damaged"]
    assert list(answer["properties"]) == answer["required"]


def test_the_junk_gates_codebook_and_the_records_schema_name_the_same_answers() -> None:
    """The codebook writes every field and value in backticks, and nothing else in lower case."""
    codebook = JUNK_GATE_CODEBOOK.read_text(encoding="utf-8")
    named_in_codebook = set(re.findall(r"`([a-z_]+)`", codebook))
    properties = SCHEMAS["records"]["properties"]
    values = {value for answer in properties.values() for value in answer.get("enum", [])}

    assert values - named_in_codebook == set(), "values the codebook never names"
    assert set(properties) - named_in_codebook == set(), "fields the codebook never names"
    assert named_in_codebook - values - set(properties) == set(), "names the schema lacks"


def test_the_junk_gates_codebook_asks_for_the_answer_in_the_schemas_order() -> None:
    codebook = JUNK_GATE_CODEBOOK.read_text(encoding="utf-8")
    _, answer_section = codebook.split("## Your answer\n")

    asked = re.findall(r"^- `([a-z_]+)`:", answer_section, flags=re.MULTILINE)

    assert asked == SCHEMAS["records"]["required"]
