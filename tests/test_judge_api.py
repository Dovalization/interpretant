"""The API judge (`pipelines/judge_api.py`): requests, the cap, retries, the usage."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from interpretant.pipelines import judge_api as judge
from interpretant.pipelines.judge_api import ApiJudging
from interpretant.review.judging import SCHEMAS, render

MODEL = "claude-sonnet-5-5"


def usage(**tokens: int) -> SimpleNamespace:
    """A usage record with the given token counts and zero for the rest."""
    fields = dict.fromkeys(judge.USAGE_FIELDS, 0) | tokens
    return SimpleNamespace(**fields)


def item(item_id: str) -> dict[str, Any]:
    """One kept item of the matching audit."""
    return {
        "id": item_id,
        "kind": "kept",
        "word": "network",
        "parts_of_speech": ["NOUN", "PROPN"],
        "sentence": "A ⟨t⟩network⟨/t⟩ of sensors.",
    }


def succeeded(item_id: str, text: str, stop: str = "end_turn") -> SimpleNamespace:
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=usage(input_tokens=100, cache_read_input_tokens=1500, output_tokens=200),
        stop_reason=stop,
    )
    return SimpleNamespace(
        custom_id=item_id, result=SimpleNamespace(type="succeeded", message=message)
    )


def errored(item_id: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=item_id, result=SimpleNamespace(type="errored"))


class FakeBatches:
    """Batches that end at once and return scripted results, one list per submitted batch."""

    def __init__(self, scripted: list[list[SimpleNamespace]]) -> None:
        self.scripted = scripted
        self.submitted: list[list[dict[str, Any]]] = []

    def create(self, requests: list[dict[str, Any]]) -> SimpleNamespace:
        self.submitted.append(requests)
        return SimpleNamespace(id=f"batch-{len(self.submitted)}")

    def retrieve(self, _batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(processing_status="ended")

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        return self.scripted[int(batch_id.rsplit("-", 1)[1]) - 1]


class FakeClient:
    """A client whose token count and batches are scripted."""

    def __init__(self, scripted: list[list[SimpleNamespace]]) -> None:
        self.messages = SimpleNamespace(
            count_tokens=lambda **_: SimpleNamespace(input_tokens=1500),
            batches=FakeBatches(scripted),
        )


def options(tmp_path: Path, max_dollars: float = 5.0) -> ApiJudging:
    codebook = tmp_path / "codebook.md"
    codebook.write_text("Judge whether the marked place is a use of the word.")
    return ApiJudging(
        codebook=codebook,
        answers="usages",
        model=MODEL,
        effort="medium",
        max_dollars=max_dollars,
    )


def test_cost_prices_each_kind_of_token_at_batch_prices() -> None:
    spent = judge.compute_cost(
        usage(
            input_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=1_000_000,
            output_tokens=1_000_000,
        ),
        MODEL,
    )
    assert spent == pytest.approx((2.00 + 4.00 + 0.10 + 10.00) * 0.5)


def test_worst_case_covers_a_typical_request() -> None:
    typical = judge.compute_cost(
        usage(input_tokens=200, cache_read_input_tokens=1500, output_tokens=600), MODEL
    )
    assert judge.compute_worst_case(render(item("U1"), "usages"), 1500, 4096, MODEL) > typical


def test_judge_retries_an_errored_request_once_and_records_a_refusal(tmp_path: Path) -> None:
    right = json.dumps({"verdict": "right", "reason": "none", "note": "a use"})
    client = FakeClient(
        [
            [succeeded("U1", right), errored("U2"), succeeded("U3", right, stop="refusal")],
            [succeeded("U2", right)],
        ]
    )
    verdicts: dict[str, Any] = {}
    manifest = judge.generate_verdicts(
        [item("U1"), item("U2"), item("U3")],
        options(tmp_path),
        client,
        verdicts,
        tmp_path / "verdicts.json",
    )
    assert verdicts["U1"] == {"verdict": "right", "reason": None, "note": "a use"}
    assert "U2" in verdicts
    assert manifest["failed"] == {"U3": "no valid answer (refusal)"}
    assert manifest["batches"] == ["batch-1", "batch-2"]
    assert manifest["tokens"]["cache_read_input_tokens"] == 4500
    assert manifest["spent_dollars"] > 0
    assert json.loads((tmp_path / "verdicts.json").read_text()) == verdicts
    assert "cost at batch prices" in judge.render_report(manifest)


def test_judge_stops_before_a_chunk_that_could_pass_the_cap(tmp_path: Path) -> None:
    client = FakeClient([])
    manifest = judge.generate_verdicts(
        [item("U1"), item("U2")], options(tmp_path, 0.001), client, {}, tmp_path / "out.json"
    )
    assert client.messages.batches.submitted == []
    assert manifest["left_by_cap"] == ["U1", "U2"]
    assert manifest["spent_dollars"] == 0


def test_judge_skips_items_already_judged_and_sends_the_cached_codebook(tmp_path: Path) -> None:
    right = json.dumps({"verdict": "wrong", "reason": "name", "note": "a title"})
    client = FakeClient([[succeeded("U2", right)]])
    verdicts = {"U1": {"verdict": "right", "reason": None, "note": "earlier"}}
    judge.generate_verdicts(
        [item("U1"), item("U2")], options(tmp_path), client, verdicts, tmp_path / "out.json"
    )
    [requests] = client.messages.batches.submitted
    assert [request["custom_id"] for request in requests] == ["U2"]
    params = requests[0]["params"]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert "Target word: network" in params["messages"][0]["content"]


def test_thinking_off_asks_for_an_answer_without_reasoning(tmp_path: Path) -> None:
    settings = options(tmp_path)
    system = [{"type": "text", "text": "codebook"}]
    assert "thinking" not in judge.build_request(item("U1"), settings, system)["params"]
    params = judge.build_request(item("U1"), dataclasses.replace(settings, thinking="off"), system)[
        "params"
    ]
    assert params["thinking"] == {"type": "between_tools"}


def test_a_record_is_judged_on_its_text_with_the_records_answers(tmp_path: Path) -> None:
    abstract = {
        "note": "A short summary.",
        "verdict": "abstract",
        "kind": "none",
        "sentence": "own",
        "damaged": "no",
    }
    client = FakeClient([[succeeded("R1", json.dumps(abstract))]])
    verdicts: dict[str, Any] = {}

    judge.generate_verdicts(
        [{"id": "R1", "text": "We study attention. ⟦It drifts.⟧"}],
        dataclasses.replace(options(tmp_path), answers="records"),
        client,
        verdicts,
        tmp_path / "out.json",
    )

    [[request]] = client.messages.batches.submitted
    params = request["params"]
    assert params["messages"][0]["content"] == "Item id: R1\nWe study attention. ⟦It drifts.⟧"
    assert params["output_config"]["format"]["schema"] == SCHEMAS["records"]
    assert verdicts == {"R1": abstract}
