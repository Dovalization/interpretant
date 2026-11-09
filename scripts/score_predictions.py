"""Score docs/predictions.md against the drift grid results, exactly as the predictions define.

Reads data/drift/within_field.parquet, data/drift/cross_field.parquet and the cell models in
models/twec_grid. Neighbours are a word's 25 nearest words in one cell, drawn from words with
at least 50 occurrences in that cell. Prints each prediction's verdict with the numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

# Paths moved on 2026-11-10 when the lemmatised grid took the default paths; the evidence is unchanged.
MODELS = Path("models/twec_grid_unlemmatised")
WITHIN = pd.read_parquet("data/drift/unlemmatised/within_field.parquet")
CROSS = pd.read_parquet("data/drift/unlemmatised/cross_field.parquet")
MEASURE_MIN = 150


def model(layer: str, field: str, decade: int) -> KeyedVectors:
    return KeyedVectors.load(str(MODELS / layer / f"{field}_{decade}.kv"))


def count(vectors: KeyedVectors, word: str) -> int:
    return int(vectors.get_vecattr(word, "count")) if word in vectors.key_to_index else 0


def neighbours(layer: str, field: str, decade: int, word: str, k: int = 25) -> list[str]:
    vectors = model(layer, field, decade)
    if count(vectors, word) < MEASURE_MIN:
        return []
    pool = [w for w in vectors.index_to_key if count(vectors, w) >= 50 and w != word]
    matrix = np.asarray([vectors[w] for w in pool])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    target = vectors[word] / np.linalg.norm(vectors[word])
    order = np.argsort(-(matrix @ target))[:k]
    return [pool[i] for i in order]


def within(layer: str, field: str, decade: int, word: str) -> pd.Series | None:
    rows = WITHIN[
        (WITHIN.layer == layer)
        & (WITHIN.field_a == field)
        & (WITHIN.decade_a == decade)
        & (WITHIN.word == word)
    ]
    return rows.iloc[0] if len(rows) else None


def cross_rank(field_a: str, field_b: str, decade: int, word: str) -> float | None:
    first, second = sorted([field_a, field_b])
    rows = CROSS[
        (CROSS.field_a == first)
        & (CROSS.field_b == second)
        & (CROSS.decade_a == decade)
        & (CROSS.word == word)
    ]
    return float(rows.iloc[0]["cosine_rank"]) if len(rows) else None


def report(name: str, verdict: str, detail: str) -> None:
    print(f"\n{name}: {verdict.upper()}\n  {detail}")


def p1() -> None:
    lines, ok = [], True
    for label, table in (("within", WITHIN), ("cross", CROSS)):
        controls = table[table.is_control]
        share = controls.candidate.mean()
        worst = controls[controls.candidate].groupby("word").size()
        top = worst.max() if len(worst) else 0
        ok &= share < 0.05 and top <= 3
        worst_word = worst.idxmax() if len(worst) else "-"
        lines.append(f"{label}: {share:.1%} control candidates, max {top} pairs ({worst_word})")
    report("P1 method sanity", "passed" if ok else "falsified", "; ".join(lines))


def p2() -> None:
    rows = {d: within("openalex", "ai", d, "network") for d in (1980, 2000)}
    parts = [
        f"{d}s→{d + 10}s: "
        + (
            "untestable"
            if r is None
            else f"candidate={r.candidate} (cos {r.cosine_rank:.0f}, nb {r.shift_rank:.0f})"
        )
        for d, r in rows.items()
    ]
    tested = [r for r in rows.values() if r is not None]
    verdict = (
        "untestable"
        if not tested
        else "passed"
        if any(r.candidate for r in tested)
        else "falsified"
    )
    report("P2 network in AI", verdict, "; ".join(parts))


def near(words: list[str], wordset: set[str]) -> int:
    return len(set(words) & wordset)


def p3() -> None:
    symbolic = {"knowledge", "logic", "rule", "symbolic", "reasoning", "frame"}
    learned = {"feature", "learned", "embedding", "deep", "latent", "encoder"}
    early = neighbours("openalex", "ai", 1980, "representation")
    late = {d: neighbours("openalex", "ai", d, "representation") for d in (2010, 2020)}
    early_ok = near(early, symbolic) >= 2
    late_ok = any(near(n, learned) >= 2 for n in late.values())
    verdict = (
        "untestable"
        if not early or not any(late.values())
        else "passed"
        if early_ok and late_ok
        else "falsified"
    )
    report(
        "P3 representation in AI",
        verdict,
        f"1980s symbolic {near(early, symbolic)}: {early[:12]}\n  "
        + "\n  ".join(f"{d}s learned {near(n, learned)}: {n[:12]}" for d, n in late.items()),
    )


def p4() -> None:
    design = {"design", "interface", "user", "usability", "interaction", "device", "button"}
    ecological = {
        "perception",
        "ecological",
        "environment",
        "action",
        "perceive",
        "organism",
        "gibson",
    }
    ok, lines, untestable = True, [], False
    for field, leaning in (("hci", "design"), ("philosophy", "ecological")):
        for decade in (2010, 2020):
            n = neighbours("openalex_full", field, decade, "affordance")
            if not n:
                untestable = True
                lines.append(f"{field} {decade}s untestable")
                continue
            d, e = near(n, design), near(n, ecological)
            ok &= d > e if leaning == "design" else e > d
            lines.append(f"{field} {decade}s design {d} / ecological {e}: {n[:10]}")
    verdict = "untestable" if untestable else "passed" if ok else "falsified"
    report("P4 affordance HCI vs philosophy", verdict, "\n  ".join(lines))


def p5() -> None:
    hci_set = {"tangible", "interaction", "physical", "wearable", "design", "gesture"}
    phil_set = {"phenomenology", "mind", "cognition", "merleau-ponty", "perception", "enactive"}
    hci = neighbours("openalex_full", "hci", 2020, "embodiment")
    phil = neighbours("openalex_full", "philosophy", 2020, "embodiment")
    ok = near(hci, hci_set) >= 2 and near(phil, phil_set) >= 2
    verdict = "untestable" if not hci or not phil else "passed" if ok else "falsified"
    report(
        "P5 embodiment HCI vs philosophy",
        verdict,
        f"HCI {near(hci, hci_set)}: {hci[:12]}\n  philosophy {near(phil, phil_set)}: {phil[:12]}",
    )


def p6() -> None:
    ai_words = {
        "artificial",
        "machine",
        "ai",
        "robot",
        "computational",
        "llm",
        "chatbot",
        "algorithm",
    }
    row = within("openalex_full", "philosophy", 2010, "consciousness")
    before = neighbours("openalex_full", "philosophy", 2010, "consciousness")
    after = neighbours("openalex_full", "philosophy", 2020, "consciousness")
    new_ai = (set(after) - set(before)) & ai_words
    if row is None or not after:
        report("P6 consciousness towards AI", "untestable", "")
        return
    ok = bool(row.candidate) and len(new_ai) >= 2
    report(
        "P6 consciousness towards AI",
        "passed" if ok else "falsified",
        f"candidate={row.candidate} (cos {row.cosine_rank:.0f}, nb {row.shift_rank:.0f}); "
        f"new AI neighbours {sorted(new_ai)}\n"
        f"  2010s: {before[:12]}\n  2020s: {after[:12]}",
    )


def p7() -> None:
    rows = [
        within("openalex_full", "philosophy", d, "intentionality") for d in (1980, 1990, 2000, 2010)
    ]
    parts = [
        f"{d}s: "
        + (
            "untestable"
            if r is None
            else f"candidate={r.candidate} (cos {r.cosine_rank:.0f}, nb {r.shift_rank:.0f})"
        )
        for d, r in zip((1980, 1990, 2000, 2010), rows, strict=True)
    ]
    tested = [r for r in rows if r is not None]
    verdict = (
        "untestable"
        if not tested
        else "falsified"
        if any(r.candidate for r in tested)
        else "passed"
    )
    report("P7 intentionality stable", verdict, "; ".join(parts))


def p8() -> None:
    ranks = {
        d: cross_rank("psychology", "compsci", d, "information") for d in (1960, 1970, 2000, 2010)
    }
    early = [r for d, r in ranks.items() if d < 2000 and r is not None]
    late = [r for d, r in ranks.items() if d >= 2000 and r is not None]
    if not early or not late:
        report("P8 information diverges", "untestable", str(ranks))
        return
    ok = np.mean(early) < np.mean(late)
    report(
        "P8 information diverges",
        "passed" if ok else "falsified",
        f"ranks {ranks}; early {np.mean(early):.0f} vs late {np.mean(late):.0f}",
    )


def p9() -> None:
    ranks = {d: cross_rank("sociology", "compsci", d, "network") for d in (1990, 2010)}
    if None in ranks.values():
        report("P9 network converges", "untestable", str(ranks))
        return
    ok = ranks[2010] < ranks[1990]
    report("P9 network converges", "passed" if ok else "falsified", f"ranks {ranks}")


if __name__ == "__main__":
    for prediction in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
        prediction()
