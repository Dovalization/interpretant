"""Judging review items with local models on an Ollama server (`docs/junk_gate.md`, local trial).

The items, their parts and the verdict files are the Claude Code judge's
(``pipelines.judge_claude_code``): each part's answers go to ``<verdicts>/<name>_part_<NN>.json``,
item id to answer, once every item of the part has answered; a part whose file exists is skipped,
so a stopped run resumes. One item goes in each request; a request that fails or answers outside
the schema is sent once more, then the item fails and its part is left for the next run.

Chat models answer through ``/api/chat``: the codebook as the whole system prompt, the answer
constrained to its schema, thinking off, temperature 0 and the trial's seed. Decision models answer
through ``/v1/systemone`` with probabilities and no text: three questions about the item, whose
criteria are the codebook's own definitions (``build_decision_questions``), mapped to an answer
(``parse_decision_answer``).

For the trial's accuracy-against-speed table, each item's wall time is measured, and after the
first item the server's running models say how much of the model sits on the graphics card. The
server's address reaches only the transport; what the run writes names the server by the label
the caller gives it.
"""

from __future__ import annotations

import http.client
import json
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.judging import RECORD_KINDS, SCHEMAS, Answers, render

JudgeKind = Literal["chat", "decision"]
JUDGE_KINDS: tuple[JudgeKind, ...] = ("chat", "decision")
ATTEMPTS = 2
# A model partly in system memory, or loading for the first request, can take minutes per item
TIMEOUT_SECONDS = 600
SEED = 20261408
CHAT_PATH = "/api/chat"
DECISION_PATH = "/v1/systemone"
RUNNING_MODELS_PATH = "/api/ps"
# An empty generate request with keep_alive 0 unloads the model (Ollama's API)
UNLOAD_PATH = "/api/generate"
# A decision model calls a record junk when the probability that it is the work's own prose is
# under this, and a marked sentence the work's own at this or more (docs/junk_gate.md)
PROBABILITY_CUT = 0.5
MARK = "⟦"
NO_KIND = "none"
NO_KIND_DESCRIPTION = "Not junk: the text is the work's own prose."
OWN_PROSE_QUESTION = (
    "Is the text the work's own prose: the authors writing about their own work, in English?"
)
KIND_QUESTION = "If the text is not the work's own prose, what kind of text is it?"
SENTENCE_QUESTION = "Is the sentence marked between ⟦ and ⟧ part of the work's own prose?"
# Sends a JSON body to a path of the server (a GET when the body is None); returns its JSON object.
Transport = Callable[[str, dict[str, Any] | None], dict[str, Any]]
# One item to its answer, in one request (raises OllamaRequestError or InvalidAnswerError).
Judge = Callable[[dict[str, Any]], dict[str, Any]]
Clock = Callable[[], float]
_DEFINITION_BULLET = re.compile(r"^- \*\*`([a-z]+)`\*\*: (.*)$")
_KIND_BULLET = re.compile(r"^- `([a-z_]+)`: (.*?)[;.]?$")


class OllamaRequestError(RuntimeError):
    """A request to the server failed: refused, timed out, an HTTP error, or not a JSON object."""


class InvalidAnswerError(ValueError):
    """A response whose answer breaks the schema or the questions asked."""


@dataclass(frozen=True)
class OllamaJudging:
    """How a local judge judges: its name, codebook and answers, the model, its kind, the server.

    ``host_label`` names the server in the manifest; its address is never part of the setup.

    Raises:
        ValueError: for answers other than records, the only ones the local trial judges.
    """

    name: str
    codebook: Path
    answers: Answers
    model: str
    kind: JudgeKind
    host_label: str
    parallel: int = 1
    keep_alive: str = "2m"
    context_length: int = 8192

    def __post_init__(self) -> None:
        if self.answers != "records":
            raise ValueError(
                f"judge-ollama judges records only (the junk gate's answers), not {self.answers}"
            )


@dataclass(frozen=True)
class OllamaRunResult:
    """What one run did.

    ``judged`` counts the items of the parts it wrote; ``failed`` maps each part it could not
    write to the problem of a failed item; ``seconds_by_item`` holds the wall time of every item
    it finished, answered or failed; ``placement`` is the model's ``size`` and ``size_vram`` in
    bytes after the first item, both None when the server did not list the model or no item ran.
    """

    judged: int
    failed: dict[str, str]
    items_judged: int
    items_failed: int
    seconds_by_item: dict[str, float]
    seconds_total: float
    placement: dict[str, int | None]


def build_http_transport(host: str, *, timeout_seconds: float = TIMEOUT_SECONDS) -> Transport:
    """Requests to the Ollama server at ``host``, each within ``timeout_seconds``.

    Errors name the path, never the address, so no file a run writes can carry it.

    Raises:
        ValueError: if ``host`` is not an http or https URL with a host and a valid port.
    """
    address = urllib.parse.urlsplit(host)
    try:
        has_valid_port = address.port is None or address.port > 0
    except ValueError:
        has_valid_port = False
    if address.scheme not in ("http", "https") or not address.hostname or not has_valid_port:
        raise ValueError("the Ollama host must be an http or https URL, such as http://name:11434")
    base = host.rstrip("/")

    def send(path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            base + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise OllamaRequestError(f"{path}: HTTP {exc.code} {_error_text(exc)}") from exc
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise OllamaRequestError(f"{path}: {exc}") from exc
        if not isinstance(result, dict):
            raise OllamaRequestError(f"{path}: the response is not a JSON object")
        return result

    return send


def build_chat_request(
    item: dict[str, Any], judging: OllamaJudging, codebook: str
) -> dict[str, Any]:
    """One item for a chat model: the codebook as the whole system prompt, the schema's answer."""
    return {
        "model": judging.model,
        "messages": [
            {"role": "system", "content": codebook},
            {"role": "user", "content": render(item, judging.answers)},
        ],
        "format": SCHEMAS[judging.answers],
        "think": False,
        "stream": False,
        "keep_alive": judging.keep_alive,
        # Per request, so the server's own default (which can be very large) is left alone and a
        # model needs only the memory one record takes
        "options": {"temperature": 0, "seed": SEED, "num_ctx": judging.context_length},
    }


def parse_chat_answer(response: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """The answer in a chat model's message, checked against the schema.

    Raises:
        InvalidAnswerError: if the message is not a JSON object, lacks a required field, has a
            field the schema does not, or gives a value outside a field's allowed values.
    """
    content = (response.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise InvalidAnswerError("the response has no message")
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidAnswerError("the message is not JSON") from exc
    if not isinstance(answer, dict):
        raise InvalidAnswerError("the message is not a JSON object")
    missing = [name for name in schema["required"] if name not in answer]
    extra = [name for name in answer if name not in schema["properties"]]
    if missing or extra:
        raise InvalidAnswerError(f"fields missing {missing}, fields not asked for {extra}")
    for name, rule in schema["properties"].items():
        value, allowed = answer[name], rule.get("enum")
        if not isinstance(value, str) or (allowed is not None and value not in allowed):
            raise InvalidAnswerError(f"{name} is not one of the allowed values")
    return answer


def build_decision_questions(codebook: str) -> dict[str, Any]:
    """The decision model's three questions, their criteria taken from the codebook.

    ``own_prose`` asks whether the text is the work's own prose, with the codebook's ``abstract``
    definition as what counts as yes and its ``junk`` definition as what counts as no; ``kind``
    chooses among the records' kinds, each described by the codebook's line for it; and
    ``sentence_own`` asks whether the marked sentence is the work's own prose.

    Raises:
        ValueError: if the codebook does not define the abstract, the junk and each kind of junk.
    """
    definitions = _definitions(codebook)
    if not {"abstract", "junk"} <= set(definitions):
        raise ValueError("the codebook must define the abstract and junk verdicts")
    kinds = {
        bullet[1]: bullet[2]
        for line in definitions["junk"][1:]
        if (bullet := _KIND_BULLET.match(line))
    }
    expected = set(RECORD_KINDS) - {NO_KIND}
    if set(kinds) != expected:
        raise ValueError(f"the codebook's kinds of junk {sorted(kinds)} are not {sorted(expected)}")
    return {
        "own_prose": {
            "type": "noul",
            "instructions": OWN_PROSE_QUESTION,
            "criteria": {
                "true": "\n".join(definitions["abstract"]),
                "false": "\n".join(definitions["junk"]),
            },
        },
        "kind": {
            "type": "choice",
            "instructions": KIND_QUESTION,
            "criteria": {
                kind: NO_KIND_DESCRIPTION if kind == NO_KIND else kinds[kind]
                for kind in RECORD_KINDS
            },
        },
        "sentence_own": {"type": "noul", "instructions": SENTENCE_QUESTION},
    }


def build_decision_request(
    item: dict[str, Any], judging: OllamaJudging, questions: dict[str, Any]
) -> dict[str, Any]:
    """One item for a decision model: its text as the state the questions are about."""
    return {
        "model": judging.model,
        "state": item["text"],
        "questions": questions,
        "keep_alive": judging.keep_alive,
    }


def parse_decision_answer(response: dict[str, Any], text: str) -> dict[str, Any]:
    """A decision model's probabilities as a records answer, the probabilities kept in it.

    The verdict is junk when the probability of the work's own prose is under 0.5, and abstract
    otherwise (a decision model never says mixed); the kind is then the likeliest kind other than
    none, and none for an abstract. The sentence is none when the text marks no sentence, else
    own at a probability of 0.5 or more and other under it. Damage is not asked: it is null.

    Raises:
        InvalidAnswerError: if an answer is missing or not a probability, or the kinds' answer
            does not cover exactly the records' kinds.
    """
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise InvalidAnswerError("the response has no answers")
    own_prose = _probability((answers.get("own_prose") or {}).get("noul"), "own_prose")
    sentence_own = _probability((answers.get("sentence_own") or {}).get("noul"), "sentence_own")
    kinds = (answers.get("kind") or {}).get("probabilities")
    if not isinstance(kinds, dict) or set(kinds) != set(RECORD_KINDS):
        raise InvalidAnswerError("the kind's probabilities do not cover the records' kinds")
    probabilities = {kind: _probability(kinds[kind], kind) for kind in RECORD_KINDS}
    is_junk = own_prose < PROBABILITY_CUT
    junk_kinds = [kind for kind in RECORD_KINDS if kind != NO_KIND]
    if MARK not in text:
        sentence = "none"
    else:
        sentence = "own" if sentence_own >= PROBABILITY_CUT else "other"
    return {
        "verdict": "junk" if is_junk else "abstract",
        "kind": max(junk_kinds, key=probabilities.__getitem__) if is_junk else NO_KIND,
        "sentence": sentence,
        "damaged": None,
        "probabilities": {
            "own_prose": own_prose,
            "sentence_own": sentence_own,
            "kind": probabilities,
        },
    }


def build_own_prose_question(codebook: str) -> dict[str, Any]:
    """The own-prose question alone, its criteria the codebook's (``build_decision_questions``)."""
    return {"own_prose": build_decision_questions(codebook)["own_prose"]}


def ask_not_own_prose(
    transport: Transport, model: str, text: str, questions: dict[str, Any], keep_alive: str
) -> float:
    """The probability, by one request, that the text is not the work's own prose.

    Raises:
        OllamaRequestError: if the request fails.
        InvalidAnswerError: if the answer is missing or not a probability.
    """
    response = transport(
        DECISION_PATH,
        {"model": model, "state": text, "questions": questions, "keep_alive": keep_alive},
    )
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise InvalidAnswerError("the response has no answers")
    return 1.0 - _probability((answers.get("own_prose") or {}).get("noul"), "own_prose")


def unload_model(transport: Transport, model: str) -> None:
    """Unload the model now, so the next model has the graphics card to itself.

    Raises:
        OllamaRequestError: if the request fails.
    """
    transport(UNLOAD_PATH, {"model": model, "keep_alive": 0})


def build_judge(judging: OllamaJudging, transport: Transport) -> Judge:
    """One item to its answer in one request, as the judge's kind asks it."""
    codebook = judging.codebook.read_text(encoding="utf-8")
    if judging.kind == "chat":
        schema = SCHEMAS[judging.answers]

        def judge_by_chat(item: dict[str, Any]) -> dict[str, Any]:
            response = transport(CHAT_PATH, build_chat_request(item, judging, codebook))
            return parse_chat_answer(response, schema)

        return judge_by_chat
    questions = build_decision_questions(codebook)

    def judge_by_decision(item: dict[str, Any]) -> dict[str, Any]:
        response = transport(DECISION_PATH, build_decision_request(item, judging, questions))
        return parse_decision_answer(response, item["text"])

    return judge_by_decision


def generate_item_answer(
    item: dict[str, Any], judge: Judge, clock: Clock
) -> tuple[dict[str, Any] | None, float, str]:
    """One item's answer (None if both attempts failed), its wall time in seconds, its problem."""
    started = clock()
    problem = ""
    for _ in range(ATTEMPTS):
        try:
            return judge(item), clock() - started, ""
        except (OllamaRequestError, InvalidAnswerError) as exc:
            problem = str(exc)[:200]
    return None, clock() - started, problem


def fetch_placement(transport: Transport, model: str) -> dict[str, int | None]:
    """The model's size and the bytes of it on the graphics card, from the server's running models.

    Both are None when the server does not list the model (it failed to load, or was unloaded).
    """
    names = {model, f"{model}:latest"}
    for running in transport(RUNNING_MODELS_PATH, None).get("models", []):
        if running.get("name") in names or running.get("model") in names:
            return {"size": running.get("size"), "size_vram": running.get("size_vram")}
    return {"size": None, "size_vram": None}


def generate_verdicts(
    parts: dict[str, list[dict[str, Any]]],
    judging: OllamaJudging,
    verdicts: Path,
    transport: Transport,
    *,
    progress: Progress = SILENT,
    clock: Clock = time.monotonic,
) -> OllamaRunResult:
    """Judge every item of the parts without a verdict file under ``verdicts``, one per request.

    ``judging.parallel`` requests run at once. A part's file is written, its answers in the part's
    order, once all of its items have answered; a part with a failed item is reported and left for
    the next run. An interrupted run cancels the requests not yet sent; those running finish
    within their timeout.
    """
    verdicts.mkdir(parents=True, exist_ok=True)
    pending = {
        part: items
        for part, items in parts.items()
        if not _part_path(verdicts, judging.name, part).exists()
    }
    progress.stage("judging", total=sum(len(items) for items in pending.values()))
    judge = build_judge(judging, transport)
    answered: dict[str, dict[str, Any]] = {part: {} for part in pending}
    items_left = {part: len(items) for part, items in pending.items()}
    failed: dict[str, str] = {}
    seconds_by_item: dict[str, float] = {}
    placement: dict[str, int | None] = {}
    judged = items_judged = items_failed = 0
    started = clock()
    with ThreadPoolExecutor(max_workers=judging.parallel) as pool:
        futures = {
            pool.submit(generate_item_answer, item, judge, clock): (part, item["id"])
            for part, items in pending.items()
            for item in items
        }
        try:
            for future in as_completed(futures):
                part, item_id = futures[future]
                answer, seconds, problem = future.result()
                seconds_by_item[item_id] = round(seconds, 3)
                if not placement:
                    placement = fetch_placement(transport, judging.model)
                progress.advance()
                items_left[part] -= 1
                if answer is None:
                    items_failed += 1
                    failed.setdefault(part, f"{item_id}: {problem}")
                else:
                    items_judged += 1
                    answered[part][item_id] = answer
                if items_left[part] == 0 and part not in failed:
                    ordered = {item["id"]: answered[part][item["id"]] for item in pending[part]}
                    _part_path(verdicts, judging.name, part).write_text(
                        json.dumps(ordered, indent=1, ensure_ascii=False) + "\n"
                    )
                    judged += len(ordered)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return OllamaRunResult(
        judged=judged,
        failed=failed,
        items_judged=items_judged,
        items_failed=items_failed,
        seconds_by_item=seconds_by_item,
        seconds_total=clock() - started,
        placement=placement or {"size": None, "size_vram": None},
    )


def build_run_record(run: OllamaRunResult, judging: OllamaJudging) -> dict[str, Any]:
    """The manifest's entry for one run: the setup, its items, speed and placement.

    The median is over every item the run finished; the rate counts the items answered per minute
    of the run's wall time, so with parallel requests it is not the median's inverse.
    """
    seconds = list(run.seconds_by_item.values())
    return {
        "model": judging.model,
        "kind": judging.kind,
        "host_label": judging.host_label,
        "parallel": judging.parallel,
        "keep_alive": judging.keep_alive,
        "context_length": judging.context_length,
        "items_judged": run.items_judged,
        "items_failed": run.items_failed,
        "seconds_total": round(run.seconds_total, 3),
        "seconds_median": round(statistics.median(seconds), 3) if seconds else None,
        "items_per_minute": (
            round(run.items_judged * 60 / run.seconds_total, 2) if run.seconds_total > 0 else None
        ),
        "size": run.placement["size"],
        "size_vram": run.placement["size_vram"],
        "seconds_by_item": run.seconds_by_item,
    }


def build_merged_manifest(
    earlier: dict[str, Any], run: OllamaRunResult, judging: OllamaJudging, codebook_sha256: str
) -> dict[str, Any]:
    """The manifest after this run: the setup, all runs' items judged, its failures, every run."""
    return {
        "model": judging.model,
        "kind": judging.kind,
        "answers": judging.answers,
        "codebook_sha256": codebook_sha256,
        "judged": int(earlier.get("judged", 0)) + run.judged,
        "failed": run.failed,
        "runs": [*earlier.get("runs", []), build_run_record(run, judging)],
    }


def _part_path(verdicts: Path, name: str, part: str) -> Path:
    return verdicts / f"{name}_part_{part}.json"


def _error_text(error: urllib.error.HTTPError) -> str:
    """The server's own message in an error response (Ollama sends ``{"error": ...}``)."""
    try:
        return str(json.loads(error.read(500)).get("error", ""))[:200]
    except (OSError, ValueError, AttributeError):
        return ""


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise InvalidAnswerError(f"{name} is not a probability")
    return float(value)


def _definitions(codebook: str) -> dict[str, list[str]]:
    """Each term the codebook defines (the verdicts among them): its text, then its sub-bullets.

    A definition is a top-level bullet naming its term in bold backticks and the indented lines
    under it; lines wrapped for width are joined, so each entry is one line.
    """
    definitions: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in codebook.splitlines():
        bullet = _DEFINITION_BULLET.match(line)
        if bullet:
            current = definitions.setdefault(bullet[1], [bullet[2]])
        elif current is not None and line.startswith(" ") and line.strip():
            text = line.strip()
            if text.startswith("- "):
                current.append(text)
            else:
                current[-1] += f" {text}"
        else:
            current = None
    return definitions
