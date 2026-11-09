"""Score round 4 (L0-L10): the findings on the lemmatised corpus, as the predictions define.

Committed before the lemmatised corpus, its models or its tables existed. Lemmatised evidence
is at the default paths (models/twec_grid, models/twec_grid_seed43, data/drift/*.parquet);
the unlemmatised cells L1 compares against are under models/twec_grid_unlemmatised.
Neighbours are a word's 25 nearest words in one cell, drawn from words with at least 50
occurrences in that cell (interpretant.drift.grid.cell_neighbours).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from interpretant.drift.grid import add_noise_floor, cell_neighbours, compare_cells

SEED42, SEED43 = Path("models/twec_grid"), Path("models/twec_grid_seed43")
UNLEMMATISED = Path("models/twec_grid_unlemmatised")
DRIFT = Path("data/drift")
PERPLEXITY = DRIFT / "cross_perplexity_corrected.parquet"
SEEDS = (1, 2, 3, 4, 5)

VR = {
    "presence",
    "ownership",
    "illusion",
    "immersion",
    "self-location",
    "avatar",
    "virtual",
    "agency",
}
PHENOMENOLOGICAL = {
    "corporeality",
    "spatiality",
    "intersubjectivity",
    "affectivity",
    "temporality",
    "lived",
    "flesh",
    "embodied",
}
SYMBOLIC = {"knowledge", "logic", "rule", "symbolic", "reasoning", "frame"}
LEARNED = {"feature", "learned", "embedding", "deep", "latent", "encoder"}
EXPERIENTIAL = {
    "experiential",
    "qualities",
    "physicality",
    "interactivity",
    "metaphors",
    "sensorial",
    "materiality",
}


@cache
def model(root: Path, layer: str, field: str, decade: int) -> KeyedVectors:
    return KeyedVectors.load(str(root / layer / f"{field}_{decade}.kv"))


@cache
def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(DRIFT / f"{name}.parquet")


def count(vectors: KeyedVectors, word: str) -> int:
    return int(vectors.get_vecattr(word, "count")) if word in vectors.key_to_index else 0


def pool_size(vectors: KeyedVectors) -> int:
    return sum(1 for word in vectors.index_to_key if count(vectors, word) >= 50)


def neighbours(root: Path, layer: str, field: str, decade: int, word: str) -> list[str]:
    return cell_neighbours(model(root, layer, field, decade), word)


def near(words: list[str], wordset: set[str]) -> int:
    return len(set(words) & wordset)


def within42(layer: str, field: str, decade: int, word: str) -> pd.Series | None:
    rows = table("within_field")
    rows = rows[
        (rows.layer == layer)
        & (rows.field_a == field)
        & (rows.decade_a == decade)
        & (rows.word == word)
    ]
    return rows.iloc[0] if len(rows) else None


def within43(layer: str, field: str, decade: int, word: str) -> pd.Series | None:
    pair = add_noise_floor(
        compare_cells(model(SEED43, layer, field, decade), model(SEED43, layer, field, decade + 10))
    )
    rows = pair[pair.word == word]
    return rows.iloc[0] if len(rows) else None


def cross_rank(field_a: str, field_b: str, decade: int, word: str) -> float | None:
    first, second = sorted([field_a, field_b])
    rows = table("cross_field")
    rows = rows[
        (rows.field_a == first)
        & (rows.field_b == second)
        & (rows.decade_a == decade)
        & (rows.word == word)
    ]
    return float(rows.iloc[0]["cosine_rank"]) if len(rows) else None


def excess(rows: pd.DataFrame, decade: int, model_field: str, text_field: str) -> float:
    hit = rows[
        (rows.decade == decade)
        & (rows.model_field == model_field)
        & (rows.text_field == text_field)
    ]
    return float(hit.iloc[0].excess)


def report(name: str, passed: bool | None, detail: str) -> None:
    verdict = "UNTESTABLE" if passed is None else "PASSED" if passed else "FALSIFIED"
    print(f"\n{name}: {verdict}\n  {detail}")


def l0() -> None:
    lines, ok = [], True
    for name in ("within_field", "cross_field"):
        rows = table(name)
        controls = rows[rows.is_control]
        others = controls[controls.word != "food"]
        share_all, share_others = controls.candidate.mean(), others.candidate.mean()
        pairs = others[others.candidate].groupby("word").size()
        top = int(pairs.max()) if len(pairs) else 0
        food = int(controls[(controls.word == "food") & controls.candidate].shape[0])
        ok &= share_others < 0.05 and top <= 3
        lines.append(
            f"{name}: {share_all:.1%} control candidates ({share_others:.1%} without food); "
            f"max pairs {top} ({pairs.idxmax() if len(pairs) else '-'}); food in {food} pairs"
        )
    report("L0 method sanity", ok, "; ".join(lines))


def l1() -> None:
    misses, shrinks = [], []
    for path in sorted(SEED42.glob("*/*.kv")):
        layer, stem = path.parent.name, path.stem
        field, decade = stem.rsplit("_", 1)
        before = len(model(UNLEMMATISED, layer, field, int(decade)).index_to_key)
        after = len(model(SEED42, layer, field, int(decade)).index_to_key)
        shrink = 1 - after / before
        shrinks.append(shrink)
        floor = 0.08 if layer in ("books", "openalex") else 0.04
        if shrink < floor:
            misses.append(f"{layer}/{stem} {shrink:.1%}")
    pools = {}
    for field in ("philosophy", "hci"):
        for decade in (2010, 2020):
            before = pool_size(model(UNLEMMATISED, "openalex_full", field, decade))
            after = pool_size(model(SEED42, "openalex_full", field, decade))
            pools[f"{field} {decade}s"] = round(after / before - 1, 3)
    passed = not misses and all(growth >= 0.03 for growth in pools.values())
    report(
        "L1 lemmatisation did its job",
        passed,
        f"vocabulary shrink: median {np.median(shrinks):.1%}, min {min(shrinks):.1%}, "
        f"cells below their floor: {misses or 'none'}; neighbour pool growth: {pools}",
    )


def l2() -> None:
    lines, ok = [], True
    for root, seed in ((SEED42, 42), (SEED43, 43)):
        hci = neighbours(root, "openalex_full", "hci", 2020, "embodiment")
        phil = neighbours(root, "openalex_full", "philosophy", 2020, "embodiment")
        ok &= near(hci, VR) >= 3 and near(phil, PHENOMENOLOGICAL) >= 3
        lines.append(
            f"seed {seed}: HCI {near(hci, VR)} of VR {sorted(set(hci) & VR)}; philosophy "
            f"{near(phil, PHENOMENOLOGICAL)} of phenomenological "
            f"{sorted(set(phil) & PHENOMENOLOGICAL)}"
        )
    report("L2 embodiment's two senses", ok, "\n  ".join(lines))


def l3() -> None:
    lines, ok, tested = [], True, False
    for seed, lookup in ((42, within42), (43, within43)):
        rows = [
            lookup("openalex_full", "philosophy", d, "intentionality")
            for d in (1980, 1990, 2000, 2010)
        ]
        found = [r for r in rows if r is not None]
        tested |= bool(found)
        ok &= not any(r.candidate for r in found)
        lines.append(
            f"seed {seed}: "
            + "; ".join(
                f"{d}s "
                + (
                    "untestable"
                    if r is None
                    else f"cos {r.cosine_rank:.0f} nb {r.shift_rank:.0f} cand={bool(r.candidate)}"
                )
                for d, r in zip((1980, 1990, 2000, 2010), rows, strict=True)
            )
        )
    report("L3 intentionality stable", ok if tested else None, "\n  ".join(lines))


def l4() -> None:
    early = neighbours(SEED42, "openalex", "ai", 1980, "representation")
    late = {d: neighbours(SEED42, "openalex", "ai", d, "representation") for d in (2010, 2020)}
    ok = near(early, SYMBOLIC) >= 2 and any(near(n, LEARNED) >= 2 for n in late.values())
    report(
        "L4 representation in AI",
        ok,
        f"1980s {near(early, SYMBOLIC)} of symbolic {sorted(set(early) & SYMBOLIC)}; "
        + "; ".join(
            f"{d}s {near(n, LEARNED)} of learned {sorted(set(n) & LEARNED)}"
            for d, n in late.items()
        ),
    )


def l5() -> None:
    words = neighbours(SEED42, "openalex_full", "hci", 2010, "affordance")
    report(
        "L5 affordance experiential",
        near(words, EXPERIENTIAL) >= 3,
        f"{near(words, EXPERIENTIAL)} of the set {sorted(set(words) & EXPERIENTIAL)}; neighbours {words[:12]}",
    )


def l6() -> None:
    ranks = {
        d: cross_rank("psychology", "compsci", d, "information") for d in (1960, 1970, 2000, 2010)
    }
    early = [r for d, r in ranks.items() if d < 2000 and r is not None]
    late = [r for d, r in ranks.items() if d >= 2000 and r is not None]
    if not early or not late:
        report("L6 information converges", None, str(ranks))
        return
    gap = float(np.mean(early) - np.mean(late))
    report(
        "L6 information converges",
        gap >= 15,
        f"ranks {ranks}; early {np.mean(early):.0f} vs late {np.mean(late):.0f}, gap {gap:.0f}",
    )


def l7() -> None:
    row = within42("openalex_full", "philosophy", 2010, "consciousness")
    words = neighbours(SEED42, "openalex_full", "philosophy", 2020, "consciousness")
    if row is None:
        report("L7 consciousness moves", None, f"2020s neighbours {words}")
        return
    report(
        "L7 consciousness moves",
        bool(row.candidate),
        f"2010s->2020s cos {row.cosine_rank:.0f} nb {row.shift_rank:.0f} candidate={bool(row.candidate)}; "
        f"2020s neighbours {words[:15]}",
    )


def l8() -> None:
    full = pd.read_parquet(PERPLEXITY)
    changes = {}
    for seed in SEEDS:
        rows = full[full.seed == seed]
        changes[seed] = (
            round(
                excess(rows, 2020, "philosophy", "hci") - excess(rows, 1990, "philosophy", "hci"), 3
            ),
            round(
                excess(rows, 2020, "hci", "philosophy") - excess(rows, 1990, "hci", "philosophy"), 3
            ),
        )
    passed = all(phil < 0 and abs(back) < abs(phil) for phil, back in changes.values())
    report(
        "L8 philosophy predicts HCI better, one-sidedly",
        passed,
        f"per seed (dE(philosophy->hci), dE(hci->philosophy)) 1990s->2020s: {changes}",
    )


def l9() -> None:
    full = pd.read_parquet(PERPLEXITY)
    changes = {}
    for seed in SEEDS:
        rows = full[full.seed == seed]
        fields = set(rows[rows.decade == 1980].text_field) & set(
            rows[rows.decade == 2010].text_field
        )
        means = []
        for decade in (1980, 2010):
            pairs = rows[
                (rows.decade == decade)
                & rows.model_field.isin(fields)
                & rows.text_field.isin(fields)
                & (rows.model_field != rows.text_field)
            ]
            means.append(float(pairs.excess.mean()))
        changes[seed] = (len(fields), round(means[1] - means[0], 3))
    report(
        "L9 fields converge",
        all(change < 0 for _, change in changes.values()),
        f"(fields, 2010s minus 1980s) per seed: {changes}",
    )


def l10() -> None:
    full = pd.read_parquet(PERPLEXITY)
    mean = full.groupby(["decade", "model_field", "text_field"], as_index=False).excess.mean()
    lines, passed = [], True
    for parent, child in (("compsci", "ai"), ("psychology", "cogpsych")):
        decades = sorted(
            set(mean[(mean.model_field == parent) & (mean.text_field == child)].decade)
            & set(mean[(mean.model_field == child) & (mean.text_field == parent)].decade)
        )
        hits = [
            d for d in decades if excess(mean, d, child, parent) < excess(mean, d, parent, child)
        ]
        needed = 5 if len(decades) >= 6 else len(decades)
        passed &= len(hits) >= needed
        lines.append(
            f"{child}->{parent} lower in {len(hits)}/{len(decades)} decades {decades} (needed {needed})"
        )
    report("L10 subfield asymmetry", passed, "\n  ".join(lines))


if __name__ == "__main__":
    for prediction in (l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10):
        prediction()
