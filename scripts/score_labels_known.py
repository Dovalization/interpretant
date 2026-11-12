"""Rung 3: round A (A1-A8), the classifier on known changes over time, with a planted change and
a placebo (docs/predictions_labels.md, with its amendments up to the labeller after round U).

Committed before it was run. Steps:

    uv run python scripts/score_labels_known.py collect   # training samples and measured usages
    uv run python scripts/score_labels_known.py encode    # XL-LEXEME vectors, cached
    uv run python scripts/score_labels_known.py classify  # cue-first labels and confusion matrices
    uv run --extra labels python scripts/score_labels_known.py sonnet
    uv run --extra labels python scripts/score_labels_known.py jev
    uv run python scripts/score_labels_known.py score

Lists: data/sense_lists/corpus_v2.toml (target words, check 2b's controls) and
data/sense_lists/corpus_controls.toml (rung 3's control words). Abstract text and vectors stay
under data/labels/known/raw/, which is not committed. Vectors already encoded for the same usages
(rounds T and U, and the list reviews, saved as raw/vectors_review.npz) are reused by key.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import tomllib
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).parent))
import check_known_presence as check_1a

from interpretant.drift.classifier import (
    adjusted_shares,
    boilerplate,
    confusion,
    cue_first,
    cue_sense,
    example_usage,
    normalised,
    out_of_scope,
    sqrt_balanced,
    tidy_usage,
)
from interpretant.drift.labels import (
    BatchLabeller,
    Item,
    JevLabeller,
    SenseList,
    load_sense_lists,
)
from interpretant.drift.pairs import Usage, XLLexeme
from interpretant.drift.senses import (
    Sampling,
    cased_abstracts,
    drop_shared,
    sample_occurrences,
)
from interpretant.predictions import report

GRID = Path("data/raw/openalex")
WORDS_ROOT = Path("data/raw/openalex_words")
TARGET_LISTS = Path("data/sense_lists/corpus_v2.toml")
CONTROL_LISTS = Path("data/sense_lists/corpus_controls.toml")
CONTROLS = [
    line.strip()
    for line in Path("data/control_words_rung3.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]
T_SAMPLE = Path("data/labels/corpus/raw/usages.jsonl")
OUT = Path("data/labels/known")
RAW = OUT / "raw"
TRAINING = RAW / "training.jsonl"
MEASURED = RAW / "measured.jsonl"
VECTORS = RAW / "vectors.npz"
CACHES = (
    VECTORS,
    RAW / "vectors_review.npz",
    Path("data/labels/corpus/raw/vectors.npz"),
    Path("data/labels/round_u/raw/vectors.npz"),
)
CELLS = Sampling(cap=150, plural=True, quality=True)
MEASURE_SEED = 2
MIN_USAGES = 100
SHUFFLES = 200
SHARES = (0.1, 0.2, 0.3, 0.4, 0.6)
PLACEBO_SHARE = 0.4
DRAWS = 5
SONNET = BatchLabeller(model="claude-sonnet-5-5", effort="low")
JEV = JevLabeller(model="jev-1.14.0")


@dataclass(frozen=True)
class Case:
    name: str
    word: str
    root: Path
    field: str
    early: int
    late: int
    kind: str  # "documented", "weak" or "quiet"
    sense: str = ""  # the documented sense A3 names


CASES = (
    Case(
        "attention, AI 2010s→2020s",
        "attention",
        GRID,
        "ai",
        2010,
        2020,
        "documented",
        "attention_mechanism",
    ),
    Case(
        "prompt, AI 2000s→2020s",
        "prompt",
        WORDS_ROOT / "prompt",
        "ai",
        2000,
        2020,
        "documented",
        "language_model_input",
    ),
    Case(
        "platform, communication 2000s→2010s",
        "platform",
        GRID,
        "communication",
        2000,
        2010,
        "documented",
        "digital_or_social_media_platform",
    ),
    Case(
        "network, AI 1980s→1990s",
        "network",
        GRID,
        "ai",
        1980,
        1990,
        "documented",
        "artificial_neural_network",
    ),
    Case("alignment, AI 2010s→2020s", "alignment", GRID, "ai", 2010, 2020, "weak"),
    Case("attention, AI 1990s→2000s", "attention", GRID, "ai", 1990, 2000, "quiet"),
    Case("attention, psychology 2010s→2020s", "attention", GRID, "psychology", 2010, 2020, "quiet"),
)
TRANSITIONS = sorted({(c.field, c.early, c.late) for c in CASES})


def key_of(text: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{start}:{end}:{text}".encode()).hexdigest()[:20]


def seed_of(name: str) -> int:
    return zlib.crc32(name.encode())


@cache
def lists() -> dict[str, SenseList]:
    return load_sense_lists(TARGET_LISTS) | load_sense_lists(CONTROL_LISTS)


def sense_keys(word: str) -> list[str]:
    """The word's listed senses, without the ``other`` and ``unclear`` every list carries."""
    return [s.key for s in lists()[word].senses]


@cache
def tables() -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for path in (TARGET_LISTS, CONTROL_LISTS):
        found |= tomllib.loads(path.read_text(encoding="utf-8"))
    return found


def cues(word: str) -> dict[str, list[str]]:
    senses = tables()[word]["senses"]
    return {s["key"]: s["cues"] for s in senses}  # type: ignore[attr-defined]


def scope(word: str) -> list[str]:
    words = tables()[word].get("out_of_scope", [])
    return [str(w) for w in words] if isinstance(words, list) else []


def as_usage(row: dict[str, object]) -> Usage:
    return Usage(str(row["text"]), int(row["start"]), int(row["end"]))  # type: ignore[call-overload]


def written(occurrences: list[object], **fields: object) -> list[dict[str, object]]:
    return [
        {**fields, "field": o.field, "decade": o.decade, "text": o.context}  # type: ignore[attr-defined]
        | {"start": o.start, "end": o.end}  # type: ignore[attr-defined]
        for o in occurrences
    ]


def collect() -> None:
    """The whole-corpus training samples not drawn by round T (alignment, the control words,
    prompt uncapped in AI's and computer science's 2020s cells), and every usage round A measures:
    the cases' cells, the control words' cells, the planting pool and the placebo pool."""
    RAW.mkdir(parents=True, exist_ok=True)
    fields = sorted(p.name for p in GRID.iterdir() if p.is_dir())
    training: list[dict[str, object]] = []
    for field in fields:
        for decade in range(1950, 2030, 10):
            try:
                cell = {(field, decade): cased_abstracts(GRID, field, decade)}
            except FileNotFoundError:
                continue
            for word in ["alignment", *CONTROLS]:
                found = sample_occurrences(cell, word, seed=1, sampling=CELLS)
                training += written(found, word=word)  # type: ignore[arg-type]
        print(field, "sampled", flush=True)
    for field in ("ai", "compsci"):
        cell = {(field, 2020): cased_abstracts(GRID, field, 2020)}
        uncapped = Sampling(cap=100_000, plural=True, quality=True)
        found = sample_occurrences(cell, "prompt", seed=1, sampling=uncapped)
        training += written(found, word="prompt")  # type: ignore[arg-type]
    write_jsonl(TRAINING, training)

    measured: list[dict[str, object]] = []
    for case in CASES:
        cells = drop_shared(
            {
                (case.field, d): cased_abstracts(case.root, case.field, d)
                for d in (case.early, case.late)
            }
        )
        found = sample_occurrences(cells, case.word, seed=MEASURE_SEED, sampling=CELLS)
        measured += written(found, set="case", case=case.name, word=case.word)  # type: ignore[arg-type]
    for field, early, late in TRANSITIONS:
        cells = drop_shared({(field, d): cased_abstracts(GRID, field, d) for d in (early, late)})
        name = f"{field} {early}s→{late}s"
        for word in CONTROLS:
            found = sample_occurrences(cells, word, seed=MEASURE_SEED, sampling=CELLS)
            measured += written(found, set="control", case=name, word=word)  # type: ignore[arg-type]
    ai_2020 = {("ai", 2020): cased_abstracts(GRID, "ai", 2020)}
    uncapped = Sampling(cap=100_000, plural=True, quality=True)
    pool = [
        o
        for o in sample_occurrences(ai_2020, "attention", seed=1, sampling=uncapped)
        if check_1a.marked("attention", o.context)
    ]
    measured += written(pool, set="planting", case="planting", word="attention")  # type: ignore[arg-type]
    sociology = {("sociology", 2020): cased_abstracts(GRID, "sociology", 2020)}
    found = sample_occurrences(sociology, "years", seed=MEASURE_SEED, sampling=CELLS)
    measured += written(found, set="placebo", case="placebo", word="years")  # type: ignore[arg-type]
    for n, row in enumerate(measured):
        row["id"] = f"m{n:05d}"
    write_jsonl(MEASURED, measured)
    print(f"{len(training)} training rows, {len(measured)} measured usages")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tidied(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        usage = tidy_usage(as_usage(row))
        out.append({**row, "text": usage.text, "start": usage.start, "end": usage.end})
    return out


@cache
def repeated() -> frozenset[tuple[str, str]]:
    """Boilerplate: sentences found in five or more usages of the same word across its training
    sample and its measured usages."""
    rows = training_rows_raw() + tidied(read_jsonl(MEASURED))
    by_word: dict[str, list[str]] = {}
    for row in rows:
        by_word.setdefault(str(row["word"]), []).append(str(row["text"]))
    return frozenset((w, t) for w, texts in by_word.items() for t in boilerplate(texts))


@cache
def training_rows_raw() -> list[dict[str, object]]:
    rows = [r for r in read_jsonl(T_SAMPLE) if r["word"] in tables()]
    return tidied(rows + read_jsonl(TRAINING))


def kept(row: dict[str, object]) -> bool:
    word = str(row["word"])
    return (word, str(row["text"])) not in repeated() and not out_of_scope(
        as_usage(row), scope(word)
    )


@cache
def seeds(word: str) -> tuple[tuple[Usage, str], ...]:
    """Cue-seeded training usages of one word, each distinct sentence once."""
    found, seen = [], set()
    for row in training_rows_raw():
        text = str(row["text"])
        if row["word"] != word or text in seen or not kept(row):
            continue
        seen.add(text)
        usage = as_usage(row)
        sense = cue_sense(usage, cues(word))
        if sense is not None:
            found.append((usage, sense))
    return tuple(found)


@cache
def measured() -> tuple[dict[str, object], ...]:
    return tuple(r for r in tidied(read_jsonl(MEASURED)) if kept(r))


def examples(word: str) -> list[tuple[Usage, str]]:
    senses = lists()[word]
    found = [(example_usage(senses.word, s.example), s.key) for s in senses.senses]
    return [(u, k) for u, k in found if u is not None]


def all_words() -> list[str]:
    return sorted({str(r["word"]) for r in measured()})


def encode() -> None:
    cache_: dict[str, np.ndarray] = {}
    for path in reversed(CACHES):
        if path.exists():
            stored = np.load(path)
            cache_ |= {str(k): v for k, v in zip(stored["keys"], stored["vectors"], strict=True)}
    pending: dict[str, Usage] = {}
    for word in all_words():
        usages = [u for u, _ in seeds(word)] + [u for u, _ in examples(word)]
        usages += [as_usage(r) for r in measured() if r["word"] == word]
        for usage in usages:
            pending.setdefault(key_of(usage.text, usage.start, usage.end), usage)
    todo = {k: u for k, u in pending.items() if k not in cache_}
    print(f"{len(todo)} of {len(pending)} vectors to encode", flush=True)
    if todo:
        cache_ |= dict(zip(todo, XLLexeme().encode(list(todo.values())), strict=True))
    needed = {k: cache_[k] for k in pending}
    np.savez(VECTORS, keys=np.array(list(needed)), vectors=np.array(list(needed.values())))


def classify() -> None:
    """Per word: a square-root balanced logistic regression on the seeds and examples, cue-first
    labels for every measured usage, and the confusion matrix from five-fold out-of-fold
    predictions on the seeds."""
    from sklearn.linear_model import LogisticRegression

    stored = np.load(VECTORS)
    vectors: dict[str, np.ndarray] = {
        str(k): np.asarray(v) for k, v in zip(stored["keys"], stored["vectors"], strict=True)
    }

    def vector(usage: Usage) -> np.ndarray:
        return vectors[key_of(usage.text, usage.start, usage.end)]

    def fit(x: list[np.ndarray], y: list[str]) -> LogisticRegression:
        model = LogisticRegression(C=1.0, class_weight=sqrt_balanced(y), max_iter=1000)
        return model.fit(normalised(np.array(x)), y)

    rows_out: list[list[str]] = []
    matrices: dict[str, list[list[float]]] = {}
    for word in all_words():
        trained = list(seeds(word)) + examples(word)
        x, y = [vector(u) for u, _ in trained], [s for _, s in trained]
        model = fit(x, y)
        items = [r for r in measured() if r["word"] == word]
        usages = [as_usage(r) for r in items]
        predicted = [
            str(p) for p in model.predict(normalised(np.array([vector(u) for u in usages])))
        ]
        labels = cue_first(usages, cues(word), predicted)
        for row, usage, label in zip(items, usages, labels, strict=True):
            decided = "cue" if cue_sense(usage, cues(word)) else "regression"
            rows_out.append([str(row["id"]), label, decided])
        n_seeds = len(seeds(word))
        order = list(range(n_seeds))
        random.Random(f"round-a:{word}").shuffle(order)
        fold = {i: p % 5 for p, i in enumerate(order)}
        truth, guess = [], []
        for k in range(5):
            train = [i for i in range(n_seeds) if fold[i] != k]
            test = [i for i in range(n_seeds) if fold[i] == k]
            if not test:
                continue
            xs = [vector(seeds(word)[i][0]) for i in train] + [vector(u) for u, _ in examples(word)]
            ys = [seeds(word)[i][1] for i in train] + [s for _, s in examples(word)]
            if len(set(ys)) < 2:
                continue
            held = normalised(np.array([vector(seeds(word)[i][0]) for i in test]))
            guess += [str(p) for p in fit(xs, ys).predict(held)]
            truth += [seeds(word)[i][1] for i in test]
        matrices[word] = confusion(truth, guess, sense_keys(word)).tolist()
        print(f"{word}: {len(items)} labelled, {n_seeds} seeds", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "classifier_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label", "decided_by"])
        writer.writerows(rows_out)
    (OUT / "confusion.json").write_text(json.dumps(matrices, indent=1), encoding="utf-8")


def language_model(name: str) -> None:
    work = [
        (lists()[str(r["word"])], Item(str(r["id"]), as_usage(r)))
        for r in measured()
        if r["set"] == "case"
    ]
    labeller = {"sonnet": SONNET, "jev": JEV}[name]
    labels = labeller.label(work, RAW / name)  # type: ignore[attr-defined]
    with (OUT / f"{name}_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label"])
        writer.writerows(sorted(labels.items()))
    print(f"{name}: {len(labels)} labels")


@dataclass(frozen=True)
class Labelled:
    label: str
    by_cue: bool


def shares(items: list[Labelled], word: str, matrices: dict[str, np.ndarray] | None) -> np.ndarray:
    """A cell's shares over the word's senses (plus "other" for anything off the list): raw when
    ``matrices`` is None; otherwise cue-decided usages counted as labelled and the rest corrected
    by adjusted classify and count with the word's confusion matrix."""
    senses = sense_keys(word)
    names = [*senses, "other"]
    if not items:
        return np.zeros(len(names))
    if matrices is None:
        counts = Counter(i.label if i.label in senses else "other" for i in items)
        return np.array([counts[n] for n in names], dtype=float) / len(items)
    cue_counts = Counter(i.label for i in items if i.by_cue)
    by_regression = [i.label for i in items if not i.by_cue]
    total = np.array([cue_counts[s] for s in senses], dtype=float)
    if by_regression:
        total += len(by_regression) * adjusted_shares(by_regression, matrices[word], senses)
    return np.append(total / len(items), 0.0)


def divergence(first: np.ndarray, second: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits."""
    if first.sum() == 0 or second.sum() == 0:
        return 0.0
    return float(jensenshannon(first, second, base=2) ** 2)


def shuffled(
    early: list[Labelled],
    late: list[Labelled],
    word: str,
    matrices: dict[str, np.ndarray] | None,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    pooled = early + late
    values = []
    for _ in range(SHUFFLES):
        order = rng.permutation(len(pooled))
        a = [pooled[i] for i in order[: len(early)]]
        b = [pooled[i] for i in order[len(early) :]]
        values.append(divergence(shares(a, word, matrices), shares(b, word, matrices)))
    return values


def planted_change(
    before: list[Labelled],
    after: list[Labelled],
    pool: list[Labelled],
    word: str,
    share: float,
    control_bar: float,
    matrices: dict[str, np.ndarray],
    seed: str,
) -> bool:
    """Replace ``share`` of the later cell's usages with usages drawn from ``pool`` and judge the
    change with both bars: its own shuffles' 95th percentile and the controls' 90th."""
    rng = random.Random(seed)
    count = round(share * len(after))
    replaced = list(after)
    positions = rng.sample(range(len(after)), count)
    for position, donor in zip(positions, rng.sample(range(len(pool)), count), strict=True):
        replaced[position] = pool[donor]
    value = divergence(shares(before, word, matrices), shares(replaced, word, matrices))
    null = shuffled(before, replaced, word, matrices, rng.randrange(10**6))
    return value > float(np.percentile(np.array(null), 95)) and value > control_bar


def score() -> None:
    labels = {
        row["usage"]: Labelled(row["label"], row["decided_by"] == "cue")
        for row in csv.DictReader(
            (OUT / "classifier_labels.tsv").open(encoding="utf-8"), delimiter="\t"
        )
    }
    raw_matrices = json.loads((OUT / "confusion.json").read_text(encoding="utf-8"))
    matrices = {w: np.array(m) for w, m in raw_matrices.items()}
    rows = measured()

    def cell(
        set_: str, case: str, word: str, decade: int, source: dict[str, str] | None = None
    ) -> list[Labelled]:
        chosen = [
            r
            for r in rows
            if r["set"] == set_
            and r["case"] == case
            and r["word"] == word
            and int(r["decade"]) == decade  # type: ignore[call-overload]
        ]
        if source is None:
            return [labels[str(r["id"])] for r in chosen]
        return [Labelled(source[str(r["id"])], False) for r in chosen if str(r["id"]) in source]

    # Control words on each transition: divergence, and A6's bar (a) alone
    control_values: dict[tuple[str, int, int], dict[str, float]] = {}
    a6 = []
    for field, early, late in TRANSITIONS:
        name = f"{field} {early}s→{late}s"
        control_values[(field, early, late)] = {}
        for word in CONTROLS:
            e, l_ = cell("control", name, word, early), cell("control", name, word, late)
            if min(len(e), len(l_)) < MIN_USAGES:
                continue
            value = divergence(shares(e, word, matrices), shares(l_, word, matrices))
            control_values[(field, early, late)][word] = value
            null = shuffled(e, l_, word, matrices, seed_of(f"control:{name}:{word}"))
            a6.append(value > np.percentile(np.array(null), 95))
        print(f"{name}: {len(control_values[(field, early, late)])} controls measured")

    found: dict[str, bool] = {}
    rises: dict[str, tuple[float, float]] = {}
    for case in CASES:
        e, l_ = (
            cell("case", case.name, case.word, case.early),
            cell("case", case.name, case.word, case.late),
        )
        first, second = shares(e, case.word, matrices), shares(l_, case.word, matrices)
        value = divergence(first, second)
        bar_a = float(np.percentile(shuffled(e, l_, case.word, matrices, seed_of(case.name)), 95))
        bar_b = float(
            np.percentile(list(control_values[(case.field, case.early, case.late)].values()), 90)
        )
        found[case.name] = value > bar_a and value > bar_b
        senses = sense_keys(case.word)
        raw_first, raw_second = shares(e, case.word, None), shares(l_, case.word, None)
        print(
            f"\n{case.name} ({case.kind}): {len(e)} and {len(l_)} usages; JSD {value:.4f} bits, "
            f"shuffles {bar_a:.4f}, controls {bar_b:.4f}: "
            + ("found" if found[case.name] else "not found")
        )
        for i, s in enumerate(senses):
            print(
                f"   {s:36s} adjusted {first[i]:.3f} → {second[i]:.3f}   "
                f"raw {raw_first[i]:.3f} → {raw_second[i]:.3f}"
            )
        if case.sense:
            i = senses.index(case.sense)
            rises[case.name] = (float(first[i]), float(second[i]))

    documented = [c.name for c in CASES if c.kind == "documented"]
    quiet = [c.name for c in CASES if c.kind == "quiet"]
    hits = [n for n in documented if found[n]]
    report(
        "A1 the classifier finds the documented changes",
        len(hits) >= 3,
        f"found {len(hits)} of 4: {hits}",
    )
    report(
        "A2 it stays quiet where nothing documented changed",
        not any(found[n] for n in quiet),
        ", ".join(f"{n}: {'found' if found[n] else 'not found'}" for n in quiet),
    )
    report(
        "A3 in the right direction",
        all(b - a > 0.05 for a, b in rises.values()),
        "; ".join(f"{n}: {a:.3f} → {b:.3f}" for n, (a, b) in rises.items()),
    )

    # Planted change: psychology's attention, 2010s → 2020s
    case = next(c for c in CASES if c.name == "attention, psychology 2010s→2020s")
    before = cell("case", case.name, "attention", 2010)
    after = cell("case", case.name, "attention", 2020)
    pool = cell("planting", "planting", "attention", 2020)
    psych_bar = float(np.percentile(list(control_values[("psychology", 2010, 2020)].values()), 90))
    planted_found = {
        share: sum(
            planted_change(
                before,
                after,
                pool,
                "attention",
                share,
                psych_bar,
                matrices,
                f"round-a:planted:{share}:{draw}",
            )
            for draw in range(DRAWS)
        )
        for share in SHARES
    }
    detected = [s for s in SHARES if planted_found[s] >= 4]
    smallest = min(detected) if detected else None
    report(
        "A4 it sees a planted change of a third or less",
        smallest is not None
        and smallest <= 0.3
        and all(s in detected for s in SHARES if s >= smallest),
        f"pool {len(pool)} marked usages; "
        + ", ".join(f"{s:.0%}: {planted_found[s]}/{DRAWS}" for s in SHARES),
    )

    # Placebo: psychology's years, 2010s → 2020s, with sociology's 2020s years planted
    transition = "psychology 2010s→2020s"
    others = [v for w, v in control_values[("psychology", 2010, 2020)].items() if w != "years"]
    placebo = sum(
        planted_change(
            cell("control", transition, "years", 2010),
            cell("control", transition, "years", 2020),
            cell("placebo", "placebo", "years", 2020),
            "years",
            PLACEBO_SHARE,
            float(np.percentile(others, 90)),
            matrices,
            f"round-a:placebo:{draw}",
        )
        for draw in range(DRAWS)
    )
    report(
        "A5 it does not see a change of source alone",
        placebo <= 1,
        f"placebo found in {placebo} of {DRAWS} draws",
    )
    report(
        "A6 every word drifts a little",
        "descriptive",
        f"control-word transitions found under bar (a) alone: {sum(a6)} of {len(a6)} "
        f"({np.mean(a6):.0%})",
    )

    case = next(c for c in CASES if c.kind == "weak")
    senses = sense_keys("alignment")
    i = senses.index("goal_or_value_alignment")
    e, l_ = cell("case", case.name, "alignment", 2010), cell("case", case.name, "alignment", 2020)
    report(
        "A7 the weak case",
        "descriptive",
        f"goal or value alignment (merged with AI alignment), adjusted share "
        f"{shares(e, 'alignment', matrices)[i]:.3f} → {shares(l_, 'alignment', matrices)[i]:.3f}; "
        f"{'found' if found[case.name] else 'not found'} under both bars",
    )

    for name in ("sonnet", "jev"):
        path = OUT / f"{name}_labels.tsv"
        if not path.exists():
            continue
        source = {
            r["usage"]: r["label"]
            for r in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t")
        }
        verdicts = []
        for case in CASES:
            if case.kind == "weak":
                continue
            e = cell("case", case.name, case.word, case.early, source)
            l_ = cell("case", case.name, case.word, case.late, source)
            value = divergence(shares(e, case.word, None), shares(l_, case.word, None))
            bar_a = float(
                np.percentile(shuffled(e, l_, case.word, None, seed_of(f"{name}:{case.name}")), 95)
            )
            verdicts.append(
                f"{case.name} ({case.kind}): {'found' if value > bar_a else 'not found'}"
            )
        report(
            f"A8 the comparison labellers: {name}, raw shares, bar (a)",
            "descriptive",
            "; ".join(verdicts),
        )


if __name__ == "__main__":
    steps = {"collect": collect, "encode": encode, "classify": classify, "score": score}
    if sys.argv[1] in ("sonnet", "jev"):
        language_model(sys.argv[1])
    else:
        steps[sys.argv[1]]()
