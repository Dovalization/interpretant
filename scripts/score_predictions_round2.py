"""Score round 2 of docs/predictions.md (R1-R9), exactly as the predictions define.

Seed-42 evidence comes from data/drift/*.parquet and models/twec_grid; seed-43 evidence from
models/twec_grid_seed43, with its drift tables computed here by the same functions as
`drift grid`. Neighbours are a word's 25 nearest words in one cell, drawn from words with at
least 50 occurrences in that cell.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.stats import binom

from interpretant.corpus.audit import load_target_words
from interpretant.drift.grid import add_noise_floor, compare_cells
from interpretant.drift.metrics import batch_neighborhood_shift

SEED42, SEED43 = Path("models/twec_grid"), Path("models/twec_grid_seed43")
WITHIN42 = pd.read_parquet("data/drift/within_field.parquet")
CROSS42 = pd.read_parquet("data/drift/cross_field.parquet")


@cache
def model(root: Path, layer: str, field: str, decade: int) -> KeyedVectors:
    return KeyedVectors.load(str(root / layer / f"{field}_{decade}.kv"))


def count(vectors: KeyedVectors, word: str) -> int:
    return int(vectors.get_vecattr(word, "count")) if word in vectors.key_to_index else 0


def pool(vectors: KeyedVectors) -> list[str]:
    return [w for w in vectors.index_to_key if count(vectors, w) >= 50]


def neighbours(root: Path, layer: str, field: str, decade: int, word: str) -> list[str]:
    vectors = model(root, layer, field, decade)
    if count(vectors, word) < 150:
        return []
    candidates = [w for w in pool(vectors) if w != word]
    matrix = np.asarray([vectors[w] for w in candidates])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    order = np.argsort(-(matrix @ (vectors[word] / np.linalg.norm(vectors[word]))))[:25]
    return [candidates[i] for i in order]


def near(words: list[str], wordset: set[str]) -> int:
    return len(set(words) & wordset)


def pair43(layer: str, field_a: str, decade_a: int, field_b: str, decade_b: int) -> pd.DataFrame:
    return add_noise_floor(
        compare_cells(
            model(SEED43, layer, field_a, decade_a), model(SEED43, layer, field_b, decade_b)
        )
    )


def row43(layer: str, field_a: str, decade_a: int, field_b: str, decade_b: int, word: str):  # noqa: ANN201
    rows = pair43(layer, field_a, decade_a, field_b, decade_b)
    rows = rows[rows.word == word]
    return rows.iloc[0] if len(rows) else None


def report(name: str, verdict: str, detail: str) -> None:
    print(f"\n{name}: {verdict.upper()}\n  {detail}")


def r1() -> None:
    symbolic = {"knowledge", "logic", "rule", "symbolic", "reasoning", "frame"}
    learned = {"feature", "learned", "embedding", "deep", "latent", "encoder"}
    early = neighbours(SEED43, "openalex", "ai", 1980, "representation")
    late = {d: neighbours(SEED43, "openalex", "ai", d, "representation") for d in (2010, 2020)}
    ok = near(early, symbolic) >= 2 and any(near(n, learned) >= 2 for n in late.values())
    report(
        "R1 P3 replicates",
        "passed" if ok else "falsified",
        f"1980s symbolic {near(early, symbolic)}: {early[:10]}\n  "
        + "\n  ".join(f"{d}s learned {near(n, learned)}: {n[:10]}" for d, n in late.items()),
    )


def r2() -> None:
    parts, flagged = [], False
    for decade in (1980, 1990, 2000, 2010):
        row = row43(
            "openalex_full", "philosophy", decade, "philosophy", decade + 10, "intentionality"
        )
        flagged |= bool(row is not None and row.candidate)
        parts.append(
            f"{decade}s: " + ("untestable" if row is None else f"candidate={row.candidate}")
        )
    report("R2 P7 replicates", "falsified" if flagged else "passed", "; ".join(parts))


def sense(
    name: str, field: str, word: str, wordset: set[str], evidence: list[tuple[Path, int]]
) -> None:
    lines, ok = [], True
    for root, decade in evidence:
        found = neighbours(root, "openalex_full", field, decade, word)
        hits = near(found, wordset)
        ok &= bool(found) and hits >= 3
        lines.append(f"{root.name} {decade}s: {hits} hits {found[:12]}")
    report(name, "passed" if ok else "falsified", "\n  ".join(lines))


def r5() -> None:
    computational = {
        "computation",
        "computational",
        "neural",
        "substrate",
        "information_processing",
        "informational",
        "intelligence",
    }
    lines, ok = [], True
    for root, layer in ((SEED43, "openalex_full"), (SEED42, "openalex")):
        if root == SEED43:
            row = row43(layer, "philosophy", 2010, "philosophy", 2020, "consciousness")
        else:
            rows = WITHIN42[
                (WITHIN42.layer == layer)
                & (WITHIN42.field_a == "philosophy")
                & (WITHIN42.decade_a == 2010)
                & (WITHIN42.word == "consciousness")
            ]
            row = rows.iloc[0] if len(rows) else None
        found = neighbours(root, layer, "philosophy", 2020, "consciousness")
        hits = near(found, computational)
        ok &= row is not None and bool(row.candidate) and hits >= 3
        candidate = "untestable" if row is None else f"candidate={row.candidate}"
        lines.append(f"{root.name}/{layer}: {candidate}, {hits} hits {found[:12]}")
    report("R5 consciousness computational", "passed" if ok else "falsified", "\n  ".join(lines))


def r6() -> None:
    lines, ok = [], True
    for label, root, field in (
        ("seed43 psychology", SEED43, "psychology"),
        ("seed42 cogpsych", SEED42, "cogpsych"),
    ):
        ranks = {}
        for decade in (1960, 1970, 2000, 2010):
            a, b = sorted(["compsci", field])
            if root == SEED43:
                row = row43("openalex", a, decade, b, decade, "information")
                ranks[decade] = None if row is None else float(row.cosine_rank)
            else:
                rows = CROSS42[
                    (CROSS42.field_a == a)
                    & (CROSS42.field_b == b)
                    & (CROSS42.decade_a == decade)
                    & (CROSS42.word == "information")
                ]
                ranks[decade] = float(rows.iloc[0].cosine_rank) if len(rows) else None
        early = [r for d, r in ranks.items() if d < 2000 and r is not None]
        late = [r for d, r in ranks.items() if d >= 2000 and r is not None]
        passed = bool(early and late) and np.mean(early) - np.mean(late) >= 15
        ok &= passed
        lines.append(f"{label}: {ranks}")
    report("R6 information converges", "passed" if ok else "falsified", "\n  ".join(lines))


def r8() -> None:
    controls = set(load_target_words(Path("data/control_words_cross_field.txt")))
    rate = CROSS42.candidate.mean()
    sub = CROSS42[CROSS42.word.isin(controls)]
    broken = [
        word
        for word, rows in sub.groupby("word")
        if rows.candidate.sum() > binom.ppf(0.99, len(rows), rate)
    ]
    ok = sub.candidate.mean() < 0.05 and not broken
    report(
        "R8 cross-field sanity",
        "passed" if ok else "falsified",
        f"{sub.candidate.mean():.2%} candidates; over chance bound: {broken}",
    )


def seed_noise(layer: str, field: str, decade: int, word: str) -> float | None:
    a, b = model(SEED42, layer, field, decade), model(SEED43, layer, field, decade)
    shared = [w for w in pool(a) if count(b, w) >= 50]
    if word not in shared:
        return None

    def unit(vectors: KeyedVectors) -> np.ndarray:
        matrix = np.asarray([vectors[w] for w in shared])
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    return batch_neighborhood_shift([word], shared, unit(a), shared, unit(b), k=25).get(word)


def r9() -> None:
    lines, ok = [], True
    for label, table in (("within", WITHIN42), ("cross", CROSS42)):
        candidates = table[table.is_target & table.candidate]
        passed = 0
        for row in candidates.itertuples():
            noise = [
                seed_noise(row.layer, row.field_a, row.decade_a, row.word),
                seed_noise(row.layer, row.field_b, row.decade_b, row.word),
            ]
            passed += all(n is not None and row.neighbour_shift > n for n in noise)
        share = passed / len(candidates) if len(candidates) else 0.0
        ok &= share >= 0.8
        total = len(candidates)
        lines.append(
            f"{label}: {passed} of {total} target candidates beat seed noise ({share:.0%})"
        )
    report("R9 drift exceeds training noise", "passed" if ok else "falsified", "; ".join(lines))


if __name__ == "__main__":
    r1()
    r2()
    sense(
        "R3 HCI embodiment is the VR sense",
        "hci",
        "embodiment",
        {
            "presence",
            "ownership",
            "illusion",
            "immersion",
            "self-location",
            "avatar",
            "virtual",
            "agency",
        },
        [(SEED43, 2020), (SEED42, 2010)],
    )
    sense(
        "R4 philosophy embodiment is phenomenological",
        "philosophy",
        "embodiment",
        {
            "corporeality",
            "spatiality",
            "intersubjectivity",
            "affectivity",
            "temporality",
            "lived",
            "flesh",
            "embodied",
        },
        [(SEED43, 2020), (SEED42, 2010)],
    )
    r5()
    r6()
    sense(
        "R7 HCI affordance is experiential",
        "hci",
        "affordance",
        {
            "experiential",
            "qualities",
            "physicality",
            "interactivity",
            "metaphors",
            "sensorial",
            "materiality",
        },
        [(SEED43, 2010), (SEED43, 2020)],
    )
    r8()
    r9()
