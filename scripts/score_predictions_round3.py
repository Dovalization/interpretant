"""Score round 3 of docs/predictions.md (T0-T7) on the PPMI-SVD grid, as the predictions define.

PPMI cells are not aligned, so only neighbour lists and neighbourhood shift are used. Shift
ranks come from compare_cells and add_noise_floor, the functions behind drift grid.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from interpretant.corpus.audit import load_target_words
from interpretant.drift.grid import add_noise_floor, compare_cells

TWEC, PPMI = Path("models/twec_grid"), Path("models/ppmi_grid")


@cache
def model(root: Path, layer: str, field: str, decade: int) -> KeyedVectors:
    return KeyedVectors.load(str(root / layer / f"{field}_{decade}.kv"))


def count(vectors: KeyedVectors, word: str) -> int:
    return int(vectors.get_vecattr(word, "count")) if word in vectors.key_to_index else 0


def neighbours(root: Path, layer: str, field: str, decade: int, word: str) -> list[str]:
    vectors = model(root, layer, field, decade)
    if count(vectors, word) < 150:
        return []
    pool = [w for w in vectors.index_to_key if count(vectors, w) >= 50 and w != word]
    matrix = np.asarray([vectors[w] for w in pool])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    order = np.argsort(-(matrix @ (vectors[word] / np.linalg.norm(vectors[word]))))[:25]
    return [pool[i] for i in order]


def near(words: list[str], wordset: set[str]) -> int:
    return len(set(words) & wordset)


def shift_rank(layer: str, a: tuple[str, int], b: tuple[str, int], word: str) -> float | None:
    pair = add_noise_floor(compare_cells(model(PPMI, layer, *a), model(PPMI, layer, *b)))
    rows = pair[pair.word == word]
    return float(rows.iloc[0].shift_rank) if len(rows) else None


def report(name: str, passed: bool | None, detail: str) -> None:
    verdict = "untestable" if passed is None else "passed" if passed else "falsified"
    print(f"\n{name}: {verdict.upper()}\n  {detail}")


def t0() -> None:
    targets = load_target_words(Path("data/words_of_interest.txt"))
    similarities = []
    for field in ("philosophy", "hci"):
        for decade in (2010, 2020):
            for word in targets:
                a = neighbours(TWEC, "openalex_full", field, decade, word)
                b = neighbours(PPMI, "openalex_full", field, decade, word)
                if a and b:
                    similarities.append(len(set(a) & set(b)) / len(set(a) | set(b)))
    median = float(np.median(similarities)) if similarities else 0.0
    report(
        "T0 methods agree",
        median >= 0.2,
        f"median Jaccard {median:.2f} over {len(similarities)} word-cells",
    )


def t1() -> None:
    symbolic = {"knowledge", "logic", "rule", "symbolic", "reasoning", "frame"}
    learned = {"feature", "learned", "embedding", "deep", "latent", "encoder"}
    early = neighbours(PPMI, "openalex", "ai", 1980, "representation")
    late = {d: neighbours(PPMI, "openalex", "ai", d, "representation") for d in (2010, 2020)}
    passed = near(early, symbolic) >= 2 and any(near(n, learned) >= 2 for n in late.values())
    detail = f"1980s symbolic {near(early, symbolic)}: {early[:10]}\n  " + "\n  ".join(
        f"{d}s learned {near(n, learned)}: {n[:10]}" for d, n in late.items()
    )
    report("T1 representation in AI", passed if early else None, detail)


def sense(name: str, field: str, word: str, wordset: set[str]) -> None:
    lines, passed = [], True
    for decade in (2010, 2020):
        found = neighbours(PPMI, "openalex_full", field, decade, word)
        hits = near(found, wordset)
        passed &= hits >= 3
        lines.append(f"{decade}s: {hits} hits {found[:12]}")
    report(name, passed, "\n  ".join(lines))


def t5() -> None:
    computational = {
        "computation",
        "computational",
        "neural",
        "substrate",
        "information_processing",
        "informational",
        "intelligence",
    }
    rank = shift_rank("openalex_full", ("philosophy", 2010), ("philosophy", 2020), "consciousness")
    found = neighbours(PPMI, "openalex_full", "philosophy", 2020, "consciousness")
    hits = near(found, computational)
    passed = None if rank is None else rank > 95 and hits >= 3
    report("T5 consciousness computational", passed, f"shift rank {rank}; {hits} hits {found[:12]}")


def t6() -> None:
    ranks = {
        d: shift_rank("openalex", ("compsci", d), ("psychology", d), "information")
        for d in (1960, 1970, 2000, 2010)
    }
    early = [r for d, r in ranks.items() if d < 2000 and r is not None]
    late = [r for d, r in ranks.items() if d >= 2000 and r is not None]
    passed = None if not early or not late else np.mean(early) - np.mean(late) >= 15
    report(
        "T6 information converges",
        passed,
        str({d: None if r is None else round(r) for d, r in ranks.items()}),
    )


def t7() -> None:
    ranks = {
        d: shift_rank("openalex_full", ("philosophy", d), ("philosophy", d + 10), "intentionality")
        for d in (1980, 1990, 2000, 2010)
    }
    tested = [r for r in ranks.values() if r is not None]
    passed = None if not tested else all(r <= 95 for r in tested)
    report(
        "T7 intentionality stable",
        passed,
        str({d: None if r is None else round(r) for d, r in ranks.items()}),
    )


if __name__ == "__main__":
    t0()
    t1()
    sense(
        "T2 HCI embodiment",
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
    )
    sense(
        "T3 philosophy embodiment",
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
    )
    sense(
        "T4 HCI affordance",
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
    )
    t5()
    t6()
    t7()
