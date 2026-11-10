"""Score round 6 (B1-B4): the grid's method on SemEval-2020 Task 1 English.

Committed before it was run. Unpacks the lemmatised corpora, trains the grid's TWEC
configuration on them (a compass on 6,500,000 tokens from each corpus, then one model per corpus)
for seeds 42 and 43, measures drift with the grid's functions at the grid's occurrence floor (150)
and at a lowered one (85), and scores the predictions in docs/predictions.md. Models are saved
under models/semeval_twec_seed<seed>/ and reused on a rerun.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

from gensim.models import KeyedVectors
from scipy.stats import spearmanr

from interpretant.drift.grid import add_noise_floor, compare_cells
from interpretant.embedding.grid import Cell, write_compass_sample
from interpretant.embedding.twec import TWECTrainer
from interpretant.predictions import report

ROOT = Path("data/external/semeval2020_eng/semeval2020_ulscd_eng")
TEXTS = Path("data/external/semeval2020_eng/texts")
SEEDS = (42, 43)
COMPASS_TOKENS = 6_500_000
FLOORS = {"as built": 150, "extended": 85}
MAJORITY = 0.568


def texts() -> list[Cell]:
    """The two lemmatised corpora as plain text files, one sentence per line."""
    TEXTS.mkdir(parents=True, exist_ok=True)
    cells = []
    for number, period in ((1, 1810), (2, 1960)):
        path = TEXTS / f"ccoha{number}.txt"
        if not path.exists():
            with gzip.open(ROOT / f"corpus{number}/lemma/ccoha{number}.txt.gz", "rt") as source:
                path.write_text(source.read(), encoding="utf-8")
        cells.append(Cell("semeval", f"ccoha{number}", period, path))
    return cells


def models(cells: list[Cell], seed: int) -> tuple[KeyedVectors, KeyedVectors]:
    """The two corpus models of one seed, trained once and saved."""
    output = Path(f"models/semeval_twec_seed{seed}")
    paths = [output / f"{cell.name}.kv" for cell in cells]
    if not all(path.exists() for path in paths):
        trainer = TWECTrainer(
            vector_size=100, window=5, min_count=5, workers=6, epochs=10, negative=5, seed=seed
        )
        sample = output / "compass_sample.txt"
        output.mkdir(parents=True, exist_ok=True)
        write_compass_sample(cells, COMPASS_TOKENS, sample)
        trainer.train_compass([sample])
        sample.unlink()
        for cell, path in zip(cells, paths, strict=True):
            trainer.train_decade([cell.path], cell.decade).save(str(path))
        print(f"  seed {seed}: models trained", flush=True)
    first, second = (KeyedVectors.load(str(path)) for path in paths)
    return first, second


def truth(name: str) -> dict[str, float]:
    lines = (ROOT / "truth" / name).read_text(encoding="utf-8").splitlines()
    return {word: float(value) for word, value in (line.split("\t") for line in lines)}


@dataclass(frozen=True)
class Result:
    seed: int
    version: str
    measured: int
    rho: float
    flagged: list[str]
    flagged_changed: int
    accuracy: float


def main() -> None:
    graded, binary = truth("graded.txt"), truth("binary.txt")
    cells = texts()
    results: list[Result] = []
    for seed in SEEDS:
        first, second = models(cells, seed)
        for version, floor in FLOORS.items():
            pair = add_noise_floor(compare_cells(first, second, measure_min=floor))
            measured = pair[pair.word.isin(set(graded))]
            drift = dict(zip(measured.word, measured.cosine_drift, strict=True))
            candidate = dict(zip(measured.word, measured.candidate, strict=True))
            words = sorted(drift)
            rho = spearmanr([drift[w] for w in words], [graded[w] for w in words]).statistic
            flagged = [w for w in words if candidate[w]]
            correct = sum((w in flagged) == (binary[w] == 1) for w in graded)
            results.append(
                Result(
                    seed,
                    version,
                    len(words),
                    float(rho),
                    flagged,
                    sum(binary[w] == 1 for w in flagged),
                    correct / len(graded),
                )
            )
    for r in results:
        print(
            f"seed {r.seed}, {r.version}: {r.measured} measured, ρ {r.rho:.3f}, accuracy "
            f"{r.accuracy:.3f}, flagged {r.flagged} ({r.flagged_changed} changed)"
        )

    built = [r for r in results if r.version == "as built"]
    extended = [r for r in results if r.version == "extended"]
    report(
        "B1 the grid as built tracks graded change",
        all(r.rho >= 0.25 for r in built),
        ", ".join(f"seed {r.seed}: ρ {r.rho:.3f} over {r.measured}" for r in built),
    )
    report(
        "B2 so does the extended grid, at published systems' level",
        all(r.rho >= 0.25 for r in extended),
        ", ".join(f"seed {r.seed}: ρ {r.rho:.3f} over {r.measured}" for r in extended),
    )
    report(
        "B3 the grid is cautious on clean data too",
        all(
            len(r.flagged) <= 4 and (not r.flagged or r.flagged_changed / len(r.flagged) >= 0.75)
            for r in built
        ),
        ", ".join(
            f"seed {r.seed}: {len(r.flagged)} flagged, {r.flagged_changed} changed" for r in built
        ),
    )
    report(
        "B4 the candidate rule beats always predicting no change",
        all(r.accuracy > MAJORITY for r in extended),
        ", ".join(f"seed {r.seed}: accuracy {r.accuracy:.3f}" for r in extended)
        + f" (majority {MAJORITY})",
    )


if __name__ == "__main__":
    main()
