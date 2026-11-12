"""Labeller pilot: Claude Sonnet 5.5 and JEV on check 2b's usages (docs/plan_sense_labels.md).

Committed before either labeller was run. Three steps:

    uv run python scripts/pilot_labellers.py sonnet    # labels through the Claude Code CLI
    uv run --extra labels python scripts/pilot_labellers.py jev   # TYPESAFE_API_KEY in .env
    uv run python scripts/pilot_labellers.py compare
    uv run --extra labels python scripts/pilot_labellers.py sonnet_api   # ANTHROPIC_API_KEY in .env
    uv run python scripts/pilot_labellers.py recheck

Labels and each labeller's usage figures are written to data/labels/pilot/ (identifiers and labels
only); raw responses stay under data/labels/pilot/raw/, which is not committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from interpretant.drift.labels import (
    OTHER,
    REPEAT_SUFFIX,
    UNCLEAR,
    BatchLabeller,
    Item,
    Labeller,
    SenseList,
    TypeSafeSettings,
    blocks,
    label_blocks,
    load_sense_lists,
    shown,
)
from interpretant.drift.pairs import Usage

KEY = Path("data/diagnostics/gold_pairs_key.tsv")
JUDGED = Path("data/diagnostics/gold_pairs.md")
CORRECTIONS = Path("data/diagnostics/gold_pairs_author_corrections.tsv")
LISTS = Path("data/sense_lists/pilot_check_2b.toml")
OUT = Path("data/labels/pilot")
SONNET = Labeller(model="claude-sonnet-5-5", effort="low")
SONNET_API = BatchLabeller(model="claude-sonnet-5-5", effort="low")
JEV_MODEL = "jev-1.14.0"
BLOCK = 20
REPEATS = 0.1
SEED = 2026
YES, NO = 0.7, 0.3


def key_rows() -> list[dict[str, str]]:
    with KEY.open(encoding="utf-8") as table:
        return list(csv.DictReader(table, delimiter="\t"))


def usage_id(word: str, usage: Usage) -> str:
    digest = hashlib.sha256(usage.text.encode("utf-8")).hexdigest()[:8]
    return f"{word}:{usage.start}:{usage.end}:{digest}"


def usages() -> dict[str, tuple[str, Usage]]:
    """Every distinct usage in check 2b's pairs, by a stable identifier."""
    found: dict[str, tuple[str, Usage]] = {}
    for row in key_rows():
        for side in ("a", "b"):
            usage = Usage(row[f"text_{side}"], int(row[f"start_{side}"]), int(row[f"end_{side}"]))
            found[usage_id(row["word"], usage)] = (row["word"], usage)
    return found


def write_labels(name: str, labels: dict[str, str], extra: dict[str, str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label", "detail"])
        for key in sorted(labels):
            writer.writerow([key, labels[key], (extra or {}).get(key, "")])


def read_labels(name: str) -> dict[str, str]:
    with (OUT / f"{name}_labels.tsv").open(encoding="utf-8") as table:
        return {row["usage"]: row["label"] for row in csv.DictReader(table, delimiter="\t")}


def sonnet() -> None:
    lists = load_sense_lists(LISTS)
    by_word: dict[str, list[Item]] = {}
    for key, (word, usage) in usages().items():
        by_word.setdefault(word, []).append(Item(key, usage))
    work = [
        (lists[word], block, OUT / "raw" / "sonnet" / f"{word}_{index:02d}.json")
        for word, items in sorted(by_word.items())
        for index, block in enumerate(blocks(items, BLOCK, REPEATS, f"{SEED}:{word}"))
    ]
    started = time.monotonic()
    labels = label_blocks(SONNET, work, workers=4)
    elapsed = time.monotonic() - started
    write_labels("sonnet", labels)
    calls = [json.loads(path.read_text(encoding="utf-8")) for _, _, path in work]
    responses = [response for stored in calls for response in stored]
    tokens = [response.get("usage") or {} for response in responses]
    fields = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    input_tokens = sum(int(t.get(field, 0)) for t in tokens for field in fields)
    output_tokens = sum(int(t.get("output_tokens", 0)) for t in tokens)
    cost = sum(float(r.get("total_cost_usd") or 0) for r in responses)
    print(
        f"{len(work)} blocks, {len(labels)} labels in {elapsed:.0f}s; input {input_tokens}, "
        f"output {output_tokens} tokens; reported cost ${cost:.3f}"
    )


def jev_questions(senses: SenseList) -> dict[str, object]:
    from typesafe_sdk import Choice, Noul

    listed = "; ".join(f"{s.key}: {s.definition}" for s in senses.senses)
    return {
        "sense": Choice(
            instructions="Which sense does the word marked with ⟦ and ⟧ in `usage` have?",
            criteria={s.key: f"{s.definition}. Example: {s.example}" for s in senses.senses},
        ),
        "listed": Noul(
            instructions=(
                f"Does the word marked with ⟦ and ⟧ in `usage` have one of these senses: {listed}?"
            ),
            criteria={
                "true": "The marked word is used in one of the listed senses.",
                "false": (
                    "The marked word is used in a sense that none of the listed senses describes."
                ),
            },
        ),
    }


def jev(limit: int | None = None) -> None:
    from typesafe_sdk import TypeSafeClient

    lists = load_sense_lists(LISTS)
    client = TypeSafeClient(api_key=TypeSafeSettings().typesafe_api_key, model=JEV_MODEL)  # type: ignore[call-arg]
    labels: dict[str, str] = {}
    detail: dict[str, str] = {}
    raw = OUT / "raw" / "jev"
    raw.mkdir(parents=True, exist_ok=True)
    tokens = 0
    started = time.monotonic()
    for key, (word, usage) in sorted(usages().items())[:limit]:
        state = {"word": word, "usage": shown(usage)}
        response = client.system_one(state, jev_questions(lists[word]), model=JEV_MODEL)
        choice = response.choices["sense"]
        listed = response.nouls["listed"].noul
        labels[key] = choice.choice if listed > YES else OTHER if listed < NO else UNCLEAR
        detail[key] = json.dumps({"listed": listed, "probabilities": choice.probabilities})
        tokens += response.usage.input_tokens or 0
        (raw / f"{key.replace(':', '_')}.json").write_text(
            json.dumps(
                {"model": response.model, "state": state, "answer": json.loads(detail[key])}
            ),
            encoding="utf-8",
        )
    write_labels("jev", labels, detail)
    print(
        f"{len(labels)} labels in {time.monotonic() - started:.0f}s; {tokens} input tokens; "
        f"model {response.model}"
    )


def sonnet_api() -> None:
    """The plan's API recheck: Sonnet 5.5 through the Batch API, one usage per request."""
    lists = load_sense_lists(LISTS)
    found = sorted(usages().items())
    items = [Item(key, usage) for key, (_, usage) in found]
    word_of = {key: word for key, (word, _) in found}
    rng = random.Random(f"{SEED}:api")
    repeats = [
        Item(i.key + REPEAT_SUFFIX, i.usage) for i in rng.sample(items, round(REPEATS * len(items)))
    ]
    work = [(lists[word_of[i.key.removesuffix(REPEAT_SUFFIX)]], i) for i in items + repeats]
    started = time.monotonic()
    labels = SONNET_API.label(work, OUT / "raw" / "sonnet_api")
    write_labels("sonnet_api", labels)
    results = [
        json.loads(line)
        for line in (OUT / "raw" / "sonnet_api" / "results.jsonl").read_text().splitlines()
    ]
    input_tokens = sum(int(r.get("input_tokens", 0)) for r in results)
    output_tokens = sum(int(r.get("output_tokens", 0)) for r in results)
    cost = (input_tokens * 1.0 + output_tokens * 5.0) / 1_000_000
    print(
        f"{len(labels)} labels in {time.monotonic() - started:.0f}s; input {input_tokens}, "
        f"output {output_tokens} tokens (about ${cost:.3f} at batch prices); "
        f"types {sorted({str(r['type']) for r in results})}"
    )


def recheck() -> None:
    """Sonnet through the API against the pilot's CLI blocks, JEV and the author."""
    lists = load_sense_lists(LISTS)
    api, cli, jev_labels = read_labels("sonnet_api"), read_labels("sonnet"), read_labels("jev")
    originals = [k for k in api if not k.endswith(REPEAT_SUFFIX)]
    for name, other in (("the pilot's CLI blocks", cli), ("JEV", jev_labels)):
        first, second = [api[k] for k in originals], [other[k] for k in originals]
        exact = statistics.mean(a == b for a, b in zip(first, second, strict=True))
        print(
            f"API labels and {name}: agree on {exact:.0%}, "
            f"kappa {cohen_kappa_score(first, second):.3f}"
        )
    repeats = [k for k in api if k.endswith(REPEAT_SUFFIX)]
    agreement = statistics.mean(api[k] == api[k.removesuffix(REPEAT_SUFFIX)] for k in repeats)
    shares = {
        label: statistics.mean(api[k] == label for k in originals) for label in (OTHER, UNCLEAR)
    }
    print(
        f"consistency on {len(repeats)} repeats {agreement:.0%}; "
        f"other {shares[OTHER]:.0%}, unclear {shares[UNCLEAR]:.0%}"
    )
    rows = [r for r in key_rows() if not r["repeat_of"]]
    for corrected in (False, True):
        judged = judgements(corrected)
        pairs = [r for r in rows if judged[r["item"]] > 0]
        scores = []
        for r in pairs:
            a, b = (
                usage_id(r["word"], Usage(r[f"text_{s}"], int(r[f"start_{s}"]), int(r[f"end_{s}"])))
                for s in ("a", "b")
            )
            listed = {s.key for s in lists[r["word"]].senses}
            scores.append(float(api[a] == api[b] and api[a] in listed))
        rho = float(spearmanr(scores, [judged[r["item"]] for r in pairs]).statistic)
        label = "with the correction pass" if corrected else "original judgements"
        print(f"API labels against the author ({label}, {len(pairs)} pairs): Spearman {rho:.3f}")


def judgements(corrected: bool) -> dict[str, int]:
    import re

    text = JUDGED.read_text(encoding="utf-8")
    found = {
        item: int(value)
        for item, value in re.findall(r"## Item (\d+):.*?Judgement:[ \t]*([0-4])", text, re.DOTALL)
    }
    if corrected:
        with CORRECTIONS.open(encoding="utf-8") as table:
            for row in csv.DictReader(table, delimiter="\t"):
                found[row["item"]] = int(row["judgement"])
    return found


def compare() -> None:
    lists = load_sense_lists(LISTS)
    sonnet_labels, jev_labels = read_labels("sonnet"), read_labels("jev")
    originals = [key for key in sonnet_labels if not key.endswith(REPEAT_SUFFIX)]
    both = [key for key in originals if key in jev_labels]
    first, second = [sonnet_labels[k] for k in both], [jev_labels[k] for k in both]
    exact = statistics.mean(a == b for a, b in zip(first, second, strict=True))
    print(
        f"labellers agree on {exact:.0%} of {len(both)} usages, "
        f"kappa {cohen_kappa_score(first, second):.3f}"
    )
    repeats = [k for k in sonnet_labels if k.endswith(REPEAT_SUFFIX)]
    agreement = statistics.mean(
        sonnet_labels[k] == sonnet_labels[k.removesuffix(REPEAT_SUFFIX)] for k in repeats
    )
    print(f"Sonnet's consistency on {len(repeats)} repeats: {agreement:.0%}")
    for name, labels in (("sonnet", sonnet_labels), ("jev", jev_labels)):
        shares = {
            label: statistics.mean(labels[k] == label for k in originals)
            for label in (OTHER, UNCLEAR)
        }
        print(f"{name}: other {shares[OTHER]:.0%}, unclear {shares[UNCLEAR]:.0%}")

    with (OUT / "jev_labels.tsv").open(encoding="utf-8") as table:
        jev_detail = {
            row["usage"]: json.loads(row["detail"]) for row in csv.DictReader(table, delimiter="\t")
        }
    rows = [r for r in key_rows() if not r["repeat_of"]]
    for corrected in (False, True):
        judged = judgements(corrected)
        pairs = [r for r in rows if judged[r["item"]] > 0]
        ids = [
            tuple(
                usage_id(r["word"], Usage(r[f"text_{s}"], int(r[f"start_{s}"]), int(r[f"end_{s}"])))
                for s in ("a", "b")
            )
            for r in pairs
        ]
        values = [judged[r["item"]] for r in pairs]

        def same(labels: dict[str, str], word: str, a: str, b: str) -> float:
            listed = {s.key for s in lists[word].senses}
            return float(labels[a] == labels[b] and labels[a] in listed)

        def overlap(a: str, b: str) -> float:
            pa, pb = jev_detail[a]["probabilities"], jev_detail[b]["probabilities"]
            return sum(pa[s] * pb[s] for s in pa)

        measures = {
            "Sonnet, same listed sense": [
                same(sonnet_labels, r["word"], a, b) for r, (a, b) in zip(pairs, ids, strict=True)
            ],
            "JEV, same listed sense": [
                same(jev_labels, r["word"], a, b) for r, (a, b) in zip(pairs, ids, strict=True)
            ],
            "JEV, probability overlap": [overlap(a, b) for a, b in ids],
        }
        label = "with the correction pass" if corrected else "original judgements"
        print(f"against the author ({label}, {len(pairs)} pairs):")
        for name, scores in measures.items():
            rho = float(spearmanr(scores, values).statistic)
            print(f"  {name}: Spearman {rho:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "jev":
        jev(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        {"sonnet": sonnet, "compare": compare, "sonnet_api": sonnet_api, "recheck": recheck}[
            sys.argv[1]
        ]()
