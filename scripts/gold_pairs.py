"""Check 2b: does XL-LEXEME judge this corpus's usage pairs as the author does?

See docs/diagnostics.md.

Committed before the pairs were drawn. Two steps:

    uv run python scripts/gold_pairs.py sample   # writes the pairs to judge and their key
    uv run python scripts/gold_pairs.py score    # after judging: XL-LEXEME against the judgements

`sample` writes data/diagnostics/gold_pairs.md, the items the author judges blind (word and two
sentences, nothing about field, decade or cell), and data/diagnostics/gold_pairs_key.tsv, which
the author does not open. `score` reads the judgements written after "Judgement:", encodes each
pair with XL-LEXEME (cached under data/drift/), and correlates.
"""

from __future__ import annotations

import csv
import random
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from interpretant.drift.pairs import Usage, XLLexeme
from interpretant.drift.senses import Sampling, cased_abstracts, drop_shared, sample_occurrences
from interpretant.predictions import report

GRID = Path("data/raw/openalex")
WORDS = Path("data/raw/openalex_words")
OUT = Path("data/diagnostics")
CACHE = Path("data/drift/gold_pairs.npz")
SAMPLING = Sampling(cap=150, plural=True, quality=True)
SEED = 2026
PLAN = {
    "attention": (GRID, "ai", 2010, 2020, True),
    "prompt": (WORDS / "prompt", "ai", 2000, 2020, True),
    "platform": (GRID, "communication", 2000, 2010, True),
    "network": (GRID, "ai", 1980, 1990, True),
    "component": (GRID, "ai", 2010, 2020, False),
    "having": (GRID, "ai", 2010, 2020, False),
}
COUNTS = {"within early": 5, "within late": 5, "across": 6}
FIELDS = [
    "item",
    "word",
    "kind",
    "repeat_of",
    "text_a",
    "start_a",
    "end_a",
    "text_b",
    "start_b",
    "end_b",
]


def usages(word: str) -> dict[int, list[Usage]]:
    root, field, early, late, _ = PLAN[word]
    cells = drop_shared({(field, d): cased_abstracts(root, field, d) for d in (early, late)})
    found = sample_occurrences(cells, word, seed=1, sampling=SAMPLING)
    return {
        d: [Usage(o.context, o.start, o.end) for o in found if o.decade == d] for d in (early, late)
    }


def shown(usage: Usage) -> str:
    before, word, after = (
        usage.text[: usage.start],
        usage.text[usage.start : usage.end],
        usage.text[usage.end :],
    )
    return f"{before}⟦{word}⟧{after}"


@dataclass
class Pair:
    word: str
    kind: str
    a: Usage
    b: Usage
    repeat_of: Pair | None = None


def sample() -> None:
    rng = random.Random(SEED)
    pairs: list[Pair] = []
    for word, (_, _, early, late, known) in PLAN.items():
        found = usages(word)
        word_pairs = []
        for kind, count in COUNTS.items():
            for _ in range(count):
                if kind == "across":
                    a, b = rng.choice(found[early]), rng.choice(found[late])
                else:
                    a, b = rng.sample(found[early if kind == "within early" else late], 2)
                word_pairs.append(Pair(word, kind, a, b))
        pairs.extend(word_pairs)
        if known:
            original = rng.choice(word_pairs)
            pairs.append(Pair(word, original.kind, original.b, original.a, original))
    rng.shuffle(pairs)
    number = {id(pair): index for index, pair in enumerate(pairs, start=1)}
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Usage pairs to judge (check 2b)",
        "",
        "See `docs/gold_guidelines.md`. Write one number after each `Judgement:`: 4 identical,",
        "3 closely related, 2 distantly related, 1 unrelated, 0 cannot decide.",
        "",
    ]
    with (OUT / "gold_pairs_key.tsv").open("w", encoding="utf-8", newline="") as key:
        writer = csv.writer(key, delimiter="\t")
        writer.writerow(FIELDS)
        for index, pair in enumerate(pairs, start=1):
            repeat = number[id(pair.repeat_of)] if pair.repeat_of else ""
            a, b = pair.a, pair.b
            writer.writerow(
                [
                    index,
                    pair.word,
                    pair.kind,
                    repeat,
                    a.text,
                    a.start,
                    a.end,
                    b.text,
                    b.start,
                    b.end,
                ]
            )
            lines += [f"## Item {index}: {pair.word}", "", f"A: {shown(a)}", "", f"B: {shown(b)}"]
            lines += ["", "Judgement:", ""]
    (OUT / "gold_pairs.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(pairs)} items → {OUT / 'gold_pairs.md'} (key kept separately)")


def judgements() -> dict[str, int]:
    text = (OUT / "gold_pairs.md").read_text(encoding="utf-8")
    found = dict(re.findall(r"## Item (\d+):.*?Judgement:[ \t]*([0-4])?", text, flags=re.DOTALL))
    missing = [item for item, value in found.items() if not value]
    if missing:
        raise SystemExit(f"{len(missing)} items not judged yet, for example item {missing[0]}")
    return {item: int(value) for item, value in found.items()}


def score() -> None:
    judged = judgements()
    with (OUT / "gold_pairs_key.tsv").open(encoding="utf-8") as table:
        key = list(csv.DictReader(table, delimiter="\t"))
    stored = dict(np.load(CACHE)) if CACHE.exists() else {}
    if not stored:
        encoder = XLLexeme()
        for side in ("a", "b"):
            vectors = encoder.encode(
                [
                    Usage(r[f"text_{side}"], int(r[f"start_{side}"]), int(r[f"end_{side}"]))
                    for r in key
                ]
            )
            stored[side] = vectors
        np.savez(CACHE, **stored)
    a = stored["a"] / np.linalg.norm(stored["a"], axis=1, keepdims=True)
    b = stored["b"] / np.linalg.norm(stored["b"], axis=1, keepdims=True)
    similarity = {row["item"]: float(a[i] @ b[i]) for i, row in enumerate(key)}

    distinct = [r for r in key if not r["repeat_of"] and judged[r["item"]] > 0]
    rho = spearmanr(
        [similarity[r["item"]] for r in distinct], [judged[r["item"]] for r in distinct]
    ).statistic
    for kind in COUNTS:
        rows = [r for r in distinct if r["kind"] == kind]
        mean_judged = statistics.mean(judged[r["item"]] for r in rows)
        mean_score = statistics.mean(similarity[r["item"]] for r in rows)
        print(
            f"{kind}: {len(rows)} pairs, mean judgement {mean_judged:.2f}, "
            f"mean similarity {mean_score:.3f}"
        )
    repeats = [(r["item"], r["repeat_of"]) for r in key if r["repeat_of"]]
    differences = [abs(judged[item] - judged[original]) for item, original in repeats]
    print(f"author's consistency on {len(repeats)} repeated pairs: differences {differences}")
    report(
        "G1 XL-LEXEME agrees with the author on this corpus",
        float(rho) >= 0.5,
        f"Spearman {rho:.3f} over {len(distinct)} pairs "
        "(needed 0.5; published about 0.63 on DWUG EN)",
    )


def second() -> None:
    """Check 2b′: agreement between the author and a second judge, and XL-LEXEME with each."""
    author = judgements()
    with (OUT / "gold_pairs_second_judge.tsv").open(encoding="utf-8") as table:
        other = {
            row["item"]: int(row["judgement"]) for row in csv.DictReader(table, delimiter="\t")
        }
    with (OUT / "gold_pairs_key.tsv").open(encoding="utf-8") as table:
        key = list(csv.DictReader(table, delimiter="\t"))
    stored = np.load(CACHE)
    a = stored["a"] / np.linalg.norm(stored["a"], axis=1, keepdims=True)
    b = stored["b"] / np.linalg.norm(stored["b"], axis=1, keepdims=True)
    similarity = {row["item"]: float(a[i] @ b[i]) for i, row in enumerate(key)}
    rows = [r for r in key if not r["repeat_of"] and author[r["item"]] > 0 and other[r["item"]] > 0]
    items = [r["item"] for r in rows]

    def rho(first: list[float], second_values: list[float]) -> float:
        return float(spearmanr(first, second_values).statistic)

    between = rho([author[i] for i in items], [other[i] for i in items])
    exact = statistics.mean(author[i] == other[i] for i in items)
    within_one = statistics.mean(abs(author[i] - other[i]) <= 1 for i in items)
    mean_judge = [(author[i] + other[i]) / 2 for i in items]
    scores = [similarity[i] for i in items]
    repeats = [(r["item"], r["repeat_of"]) for r in key if r["repeat_of"]]
    print(f"{len(items)} distinct pairs judged by both")
    print(
        f"between judges: Spearman {between:.3f}, exact {exact:.0%}, "
        f"within one point {within_one:.0%}"
    )
    print(
        f"XL-LEXEME with the author {rho(scores, [author[i] for i in items]):.3f}, "
        f"with the second judge {rho(scores, [other[i] for i in items]):.3f}, "
        f"with their mean {rho(scores, mean_judge):.3f}"
    )
    print(
        f"second judge's consistency on repeated pairs: "
        f"{[abs(other[i] - other[o]) for i, o in repeats]}"
    )


if __name__ == "__main__":
    {"sample": sample, "score": score, "second": second}[sys.argv[1]]()
