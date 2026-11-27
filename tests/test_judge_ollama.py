"""The local judge (`pipelines/judge_ollama.py`): requests, answers, parts, resumption, manifest."""

from __future__ import annotations

import dataclasses
import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from interpretant.catalog.config import CatalogConfig
from interpretant.cli.judge_ollama import run_judge_ollama
from interpretant.pipelines.judge_ollama import (
    InvalidAnswerError,
    JudgeKind,
    OllamaJudging,
    OllamaRequestError,
    build_chat_request,
    build_decision_questions,
    build_decision_request,
    build_http_transport,
    build_merged_manifest,
    generate_verdicts,
    parse_chat_answer,
    parse_decision_answer,
)
from interpretant.pipelines.progress import SILENT
from interpretant.review.judging import RECORD_KINDS, SCHEMAS

REPOSITORY = Path(__file__).resolve().parents[1]
CODEBOOK = REPOSITORY / "curated" / "judging" / "junk_gate" / "prompt.md"
ABSTRACT = {
    "note": "An abstract.",
    "verdict": "abstract",
    "kind": "none",
    "sentence": "own",
    "damaged": "no",
}
SECONDS_PER_REQUEST = 2.0
GIGABYTE = 2**30


def record_item(item_id: str, text: str = "We study ⟦attention⟧ in readers.") -> dict[str, Any]:
    """A records item as the junk gate's draw writes it."""
    return {"id": item_id, "text": text}


def judging(kind: JudgeKind = "chat") -> OllamaJudging:
    return OllamaJudging(
        name="local",
        codebook=CODEBOOK,
        answers="records",
        model="qwen3.5:9b",
        kind=kind,
        host_label="runner",
    )


def chat_response(answer: object) -> dict[str, Any]:
    content = answer if isinstance(answer, str) else json.dumps(answer)
    return {"model": "qwen3.5:9b", "message": {"role": "assistant", "content": content}}


def decision_response(own_prose: float, sentence_own: float, top_kind: str) -> dict[str, Any]:
    rest = (1 - 0.6) / (len(RECORD_KINDS) - 1)
    kinds = {kind: 0.6 if kind == top_kind else rest for kind in RECORD_KINDS}
    return {
        "model": "pplx-decider-flash:4b",
        "answers": {
            "own_prose": {"type": "noul", "noul": own_prose},
            "sentence_own": {"type": "noul", "noul": sentence_own},
            "kind": {"type": "choice", "choice": top_kind, "probabilities": kinds},
        },
    }


class FakeOllama:
    """A stand-in server: answers chat requests in turn from ``answers``, then always ``ABSTRACT``.

    Each chat request takes ``SECONDS_PER_REQUEST`` on its clock; the running models list the
    model at 8 GB, 6 of them on the graphics card.
    """

    def __init__(self, answers: list[object] | None = None) -> None:
        self.answers = list(answers or [])
        self.requests: list[tuple[str, dict[str, Any] | None]] = []
        self.now = 0.0
        self.lock = threading.Lock()

    def __call__(self, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        with self.lock:
            self.requests.append((path, body))
            if path == "/api/ps":
                model = {"name": "qwen3.5:9b", "size": 8 * GIGABYTE, "size_vram": 6 * GIGABYTE}
                return {"models": [model]}
            self.now += SECONDS_PER_REQUEST
            answer = self.answers.pop(0) if self.answers else ABSTRACT
        if isinstance(answer, Exception):
            raise answer
        return chat_response(answer)

    def clock(self) -> float:
        with self.lock:
            return self.now

    def chat_requests(self) -> list[dict[str, Any]]:
        return [body for path, body in self.requests if path == "/api/chat" and body]


def test_a_chat_request_sends_the_codebook_and_the_schema_with_thinking_off(
    tmp_path: Path,
) -> None:
    item = record_item("R0001")

    request = build_chat_request(item, judging(), "The codebook.")

    assert request["model"] == "qwen3.5:9b"
    assert request["messages"] == [
        {"role": "system", "content": "The codebook."},
        {"role": "user", "content": "Item id: R0001\nWe study ⟦attention⟧ in readers."},
    ]
    assert request["format"] == SCHEMAS["records"]
    assert request["think"] is False
    assert request["stream"] is False
    assert request["keep_alive"] == "2m"
    assert request["options"] == {"temperature": 0, "seed": 20261408}


def test_a_chat_answer_is_parsed_and_checked_against_the_schema() -> None:
    assert parse_chat_answer(chat_response(ABSTRACT), SCHEMAS["records"]) == ABSTRACT
    for broken in (
        "not json",
        ["a list"],
        {key: value for key, value in ABSTRACT.items() if key != "damaged"},
        ABSTRACT | {"confidence": "high"},
        ABSTRACT | {"verdict": "maybe"},
        ABSTRACT | {"note": 3},
    ):
        with pytest.raises(InvalidAnswerError):
            parse_chat_answer(chat_response(broken), SCHEMAS["records"])


def test_an_invalid_chat_answer_is_retried_once_then_the_item_fails(tmp_path: Path) -> None:
    server = FakeOllama([ABSTRACT | {"verdict": "maybe"}])
    run = generate_verdicts({"00": [record_item("R0001")]}, judging(), tmp_path / "once", server)
    assert run.items_judged == 1
    assert len(server.chat_requests()) == 2

    server = FakeOllama([ABSTRACT | {"verdict": "maybe"}, OllamaRequestError("/api/chat: down")])
    run = generate_verdicts({"00": [record_item("R0001")]}, judging(), tmp_path / "twice", server)
    assert run.items_failed == 1
    assert run.failed == {"00": "R0001: /api/chat: down"}
    assert len(server.chat_requests()) == 2
    assert not (tmp_path / "twice" / "local_part_00.json").exists()


def test_the_decision_questions_take_their_criteria_from_the_codebook() -> None:
    questions = build_decision_questions(CODEBOOK.read_text(encoding="utf-8"))

    assert list(questions) == ["own_prose", "kind", "sentence_own"]
    own_prose = questions["own_prose"]
    assert own_prose["type"] == "noul"
    assert "- a book review, an editorial, a comment or a reply" in own_prose["criteria"]["true"]
    assert "a citation line, author and journal details" in own_prose["criteria"]["false"]
    kinds = questions["kind"]["criteria"]
    assert questions["kind"]["type"] == "choice"
    assert list(kinds) == list(RECORD_KINDS)
    assert kinds["citation_or_metadata"].endswith("a list of keywords with no sentence")
    assert questions["sentence_own"]["type"] == "noul"
    request = build_decision_request(record_item("R0001"), judging("decision"), questions)
    assert request == {
        "model": "qwen3.5:9b",
        "state": "We study ⟦attention⟧ in readers.",
        "questions": questions,
        "keep_alive": "2m",
    }


def test_the_decision_questions_refuse_a_codebook_without_every_kind() -> None:
    codebook = CODEBOOK.read_text(encoding="utf-8").replace("  - `fragment`:", "  - `piece`:")

    with pytest.raises(ValueError, match="kinds of junk"):
        build_decision_questions(codebook)


def test_a_decision_response_is_mapped_to_junk_or_abstract() -> None:
    marked = "Contents. ⟦Chapter 1.⟧"

    junk = parse_decision_answer(decision_response(0.2, 0.1, "contents_or_list"), marked)
    abstract = parse_decision_answer(decision_response(0.9, 0.7, "none"), marked)
    unmarked = parse_decision_answer(decision_response(0.5, 0.9, "none"), "We study attention.")

    assert {key: junk[key] for key in ("verdict", "kind", "sentence", "damaged")} == {
        "verdict": "junk",
        "kind": "contents_or_list",
        "sentence": "other",
        "damaged": None,
    }
    assert junk["probabilities"]["own_prose"] == 0.2
    assert junk["probabilities"]["kind"]["contents_or_list"] == 0.6
    assert (abstract["verdict"], abstract["kind"], abstract["sentence"]) == (
        "abstract",
        "none",
        "own",
    )
    assert (unmarked["verdict"], unmarked["sentence"]) == ("abstract", "none")


def test_a_junk_verdict_never_takes_the_kind_none() -> None:
    answer = parse_decision_answer(decision_response(0.1, 0.1, "none"), "Text.")

    assert answer["verdict"] == "junk"
    assert answer["kind"] == "citation_or_metadata"


def test_a_decision_response_without_every_answer_is_invalid() -> None:
    response = decision_response(0.2, 0.1, "fragment")
    del response["answers"]["kind"]["probabilities"]["other"]

    with pytest.raises(InvalidAnswerError):
        parse_decision_answer(response, "Text.")
    with pytest.raises(InvalidAnswerError):
        parse_decision_answer(decision_response(1.2, 0.1, "fragment"), "Text.")


def test_a_part_is_written_only_when_all_its_items_succeed(tmp_path: Path) -> None:
    broken = ABSTRACT | {"kind": "spam"}
    server = FakeOllama([ABSTRACT, broken, broken])
    parts = {"00": [record_item("R0001"), record_item("R0002")], "01": [record_item("R0003")]}

    run = generate_verdicts(parts, judging(), tmp_path / "verdicts", server)

    assert not (tmp_path / "verdicts" / "local_part_00.json").exists()
    written = json.loads((tmp_path / "verdicts" / "local_part_01.json").read_text())
    assert written == {"R0003": ABSTRACT}
    assert run.judged == 1
    assert (run.items_judged, run.items_failed) == (2, 1)
    assert list(run.failed) == ["00"]


def test_a_resumed_run_skips_the_parts_already_written(tmp_path: Path) -> None:
    parts = {"00": [record_item("R0001")], "01": [record_item("R0002")]}
    verdicts = tmp_path / "verdicts"
    first = FakeOllama([ABSTRACT, OllamaRequestError("down"), OllamaRequestError("down")])
    generate_verdicts(parts, judging(), verdicts, first)

    again = FakeOllama()
    run = generate_verdicts(parts, judging(), verdicts, again)

    assert [body["messages"][1]["content"] for body in again.chat_requests()] == [
        "Item id: R0002\nWe study ⟦attention⟧ in readers."
    ]
    assert run.judged == 1
    assert sorted(path.name for path in verdicts.iterdir()) == [
        "local_part_00.json",
        "local_part_01.json",
    ]


def test_the_manifest_records_each_runs_speed_and_placement(tmp_path: Path) -> None:
    parts = {"00": [record_item(f"R{number:04d}") for number in range(3)]}
    server = FakeOllama([OllamaRequestError("busy")])
    setup = judging()

    run = generate_verdicts(parts, setup, tmp_path / "verdicts", server, clock=server.clock)
    manifest = build_merged_manifest(
        {"judged": 50, "runs": [{"model": "earlier"}]}, run, setup, "abc"
    )

    assert manifest["judged"] == 53
    assert manifest["codebook_sha256"] == "abc"
    assert (manifest["model"], manifest["kind"], manifest["answers"]) == (
        "qwen3.5:9b",
        "chat",
        "records",
    )
    earlier, record = manifest["runs"]
    assert earlier == {"model": "earlier"}
    assert record["host_label"] == "runner"
    assert (record["items_judged"], record["items_failed"]) == (3, 0)
    assert record["seconds_by_item"] == {"R0000": 4.0, "R0001": 2.0, "R0002": 2.0}
    assert record["seconds_total"] == 8.0
    assert record["seconds_median"] == 2.0
    assert record["items_per_minute"] == 22.5
    assert (record["size"], record["size_vram"]) == (8 * GIGABYTE, 6 * GIGABYTE)
    assert [path for path, _ in server.requests].count("/api/ps") == 1


def test_only_records_are_judged() -> None:
    with pytest.raises(ValueError, match="records only"):
        OllamaJudging(
            name="local",
            codebook=CODEBOOK,
            answers="usages",
            model="m",
            kind="chat",
            host_label="runner",
        )


@pytest.fixture
def loopback_server() -> Iterator[str]:
    """An Ollama stand-in on the loopback interface, answering every chat with an abstract."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            json.loads(self.rfile.read(length))
            self.reply({"message": {"role": "assistant", "content": json.dumps(ABSTRACT)}})

        def do_GET(self) -> None:
            if self.path != "/api/ps":
                self.send_error(404)
                return
            self.reply({"models": [{"name": "qwen3.5:9b", "size": 10, "size_vram": 10}]})

        def reply(self, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_the_http_transport_posts_json_and_names_only_the_path_in_errors(
    loopback_server: str,
) -> None:
    send = build_http_transport(loopback_server, timeout_seconds=5)

    assert send("/api/chat", {"model": "m"})["message"]["role"] == "assistant"
    assert send("/api/ps", None)["models"][0]["size"] == 10
    with pytest.raises(OllamaRequestError) as raised:
        send("/api/tags", None)
    assert "127.0.0.1" not in str(raised.value)
    with pytest.raises(ValueError, match="http or https URL"):
        build_http_transport("127.0.0.1:11434")


def test_no_written_file_names_the_host_only_its_label(
    tmp_path: Path, catalog: CatalogConfig, loopback_server: str
) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps(record_item(f"R{n:04d}")) + "\n" for n in range(60)))
    verdicts = tmp_path / "verdicts"
    output = io.StringIO()

    counts = run_judge_ollama(
        catalog,
        items,
        verdicts,
        judging(),
        loopback_server,
        progress=SILENT,
        console=Console(file=output),
    )

    assert counts == {"judged": 60, "items_judged": 60, "items_failed": 0, "failed_parts": 0}
    written = sorted(path.name for path in verdicts.iterdir())
    assert written == ["local.manifest.json", "local_part_00.json", "local_part_01.json"]
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "127.0.0.1" not in path.read_text(encoding="utf-8"), path
    manifest = json.loads((verdicts / "local.manifest.json").read_text())
    assert manifest["runs"][0]["host_label"] == "runner"
    assert "127.0.0.1" not in output.getvalue()


def test_requests_run_in_parallel_and_parts_keep_their_items_order(tmp_path: Path) -> None:
    server = FakeOllama()
    parts = {"00": [record_item(f"R{number:04d}") for number in range(8)]}
    setup = dataclasses.replace(judging(), parallel=4)

    generate_verdicts(parts, setup, tmp_path / "verdicts", server)

    written = json.loads((tmp_path / "verdicts" / "local_part_00.json").read_text())
    assert list(written) == [f"R{number:04d}" for number in range(8)]
