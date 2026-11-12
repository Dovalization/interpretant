"""Export the data behind the web explorer (``web/data``) as JSON.

Everything a reader needs to explore the findings fits in a few megabytes: each target word's
nearest neighbours per cell, its rows in the drift tables, the cell counts, the
cross-perplexity table and the prediction verdicts. The explorer imports these files at build
time and validates them against a mirrored Zod schema, so the shapes here are a contract.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from interpretant.corpus.audit import load_target_words
from interpretant.drift.grid import MEASURE_MIN, NEIGHBOUR_MIN, NEIGHBOURS, discover_models
from interpretant.visualization.figures import FIELD_LABELS

LAYERS = ("openalex", "openalex_full")
METHODS = {"twec": "model_dir", "ppmi": "ppmi_dir"}
ROUNDS: dict[str, tuple[str, str]] = {
    "P": ("Round 1", "the trained models"),
    "R": ("Round 2", "a second seed, unseen decades and field pairs"),
    "T": ("Round 3", "count-based PPMI-SVD"),
    "L": ("Round 4", "the lemmatised corpus"),
    "C": ("Cross-perplexity, first run", "bigram models per cell"),
    "D": (
        "Cross-perplexity, corrected run",
        "shared abstracts removed, one training size, five seeds",
    ),
    "F": ("Follow-up", "HCI and philosophy"),
    "H": ("Follow-up", "philosophy's turn towards technology"),
    "G": ("Decomposition", "where the philosophy → HCI gain sits"),
    "J": ("Mechanism", "is the philosophy → HCI effect HCI's own word mix"),
    "K": ("Venues", "the field's move or OpenAlex's label (scored as round K′)"),
    "S": ("Separability", "a classifier that guesses each abstract's field"),
    "V": ("Technology vocabulary", "round S's classifier without 29 technology terms"),
    "M": ("Held-out check", "does every cell's model generalise to text it never saw"),
    "N": ("Direction", "who moved towards whom, net of every field's drift"),
}
_TITLE = re.compile(r"^\*\*([A-Z]\d+) — ([\s\S]*?)\*\*", re.MULTILINE)
_VERDICT = re.compile(
    r"^\| ([A-Z]\d+) \| \*\*(Passed|Falsified|Untestable)\*\*\s*(\([^)]*\))? \| (.*) \|\s*$",
    re.MULTILINE,
)
_TIER = re.compile(r"^# --- (.*?)[:.]", re.MULTILINE)


def read_tiers(words_file: Path) -> dict[str, str]:
    """Map each target word to the tier named by the ``# --- Tier: ...`` header above it."""
    tiers: dict[str, str] = {}
    tier = "core"
    for line in words_file.read_text(encoding="utf-8").splitlines():
        header = _TIER.match(line)
        if header:
            tier = header.group(1).strip().lower()
            continue
        word = line.split("#", 1)[0].strip()
        if word:
            tiers[word] = tier
    return tiers


@dataclass
class NeighbourIndex:
    """One cell's neighbour pool as a unit matrix, so many words can be looked up cheaply."""

    vectors: KeyedVectors
    counts: dict[str, int]
    pool: list[str]
    matrix: np.ndarray

    @classmethod
    def build(cls, vectors: KeyedVectors, pool_min: int = NEIGHBOUR_MIN) -> NeighbourIndex:
        counts = {word: int(vectors.get_vecattr(word, "count")) for word in vectors.index_to_key}
        pool = [word for word, count in counts.items() if count >= pool_min]
        matrix = np.asarray([vectors[word] for word in pool], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.where(norms == 0, 1.0, norms)
        return cls(vectors, counts, pool, matrix)

    def neighbours(
        self, word: str, k: int = NEIGHBOURS, measure_min: int = MEASURE_MIN
    ) -> list[str]:
        if self.counts.get(word, 0) < measure_min or not self.pool:
            return []
        vector = np.asarray(self.vectors[word], dtype=np.float32)
        vector = vector / (np.linalg.norm(vector) or 1.0)
        order = np.argsort(-(self.matrix @ vector))
        return [self.pool[i] for i in order if self.pool[i] != word][:k]


def parse_predictions(paths: Iterable[Path]) -> dict[str, Any]:
    """Read the prediction files: every ``**ID — title**`` line and every scored verdict row."""
    titles: dict[str, str] = {}
    verdicts: dict[str, dict[str, str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in _TITLE.finditer(text):
            titles.setdefault(match.group(1), " ".join(match.group(2).split()).rstrip("."))
        for match in _VERDICT.finditer(text):
            identifier, verdict, note, numbers = match.groups()
            verdicts[identifier] = {
                "verdict": verdict.lower(),
                "note": (note or "").strip("()"),
                "numbers": numbers.strip(),
                "file": path.name,
            }
    predictions = [
        {"id": identifier, "round": identifier[0], "title": title, **verdicts.get(identifier, {})}
        for identifier, title in titles.items()
    ]
    rounds = [
        {"id": prefix, "title": title, "subtitle": subtitle}
        for prefix, (title, subtitle) in ROUNDS.items()
        if any(prediction["round"] == prefix for prediction in predictions)
    ]
    return {"rounds": rounds, "predictions": predictions}


def _rows(table: pd.DataFrame, rename: dict[str, str]) -> list[dict[str, Any]]:
    selected = table[list(rename)].rename(columns=rename)
    return [
        {str(key): value.item() if hasattr(value, "item") else value for key, value in row.items()}
        for row in selected.to_dict(orient="records")
    ]


WITHIN_COLUMNS = {
    "layer": "layer",
    "field_a": "field",
    "decade_a": "decadeA",
    "decade_b": "decadeB",
    "count_a": "countA",
    "count_b": "countB",
    "cosine_drift": "cosineDrift",
    "neighbour_shift": "neighbourShift",
    "cosine_rank": "cosineRank",
    "shift_rank": "shiftRank",
    "candidate": "candidate",
}
CROSS_COLUMNS = {
    "field_a": "fieldA",
    "field_b": "fieldB",
    "decade_a": "decade",
    "count_a": "countA",
    "count_b": "countB",
    "cosine_rank": "cosineRank",
    "shift_rank": "shiftRank",
    "candidate": "candidate",
}


def export_web(  # noqa: PLR0913
    model_dir: Path,
    ppmi_dir: Path,
    drift_dir: Path,
    audit_path: Path,
    predictions_paths: Iterable[Path],
    words_file: Path,
    output_dir: Path,
    log: Any = print,
) -> None:
    """Write corpus.json, words.json, words/<word>.json, perplexity.json and predictions.json."""
    tiers = read_tiers(words_file)
    targets = load_target_words(words_file)
    audit = {
        (cell["layer"], cell["source"], cell["decade"]): cell
        for cell in json.loads(audit_path.read_text(encoding="utf-8"))["cells"]
    }
    (output_dir / "words").mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    counts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    neighbours: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method, root in (("twec", model_dir), ("ppmi", ppmi_dir)):
        for cell in discover_models(root, LAYERS):
            index = NeighbourIndex.build(KeyedVectors.load(str(cell.path)))
            tokens = sum(index.counts.values())
            if method == "twec":
                audited = audit.get((cell.layer, cell.name, cell.decade), {})
                cells.append(
                    {
                        "layer": cell.layer,
                        "field": cell.name,
                        "decade": cell.decade,
                        "documents": audited.get("documents"),
                        "tokens": audited.get("tokens", tokens),
                        "vocabulary": len(index.counts),
                        "pool": len(index.pool),
                    }
                )
            for word in targets:
                count = index.counts.get(word, 0)
                if method == "twec" and count:
                    counts[word].append(
                        {
                            "layer": cell.layer,
                            "field": cell.name,
                            "decade": cell.decade,
                            "count": count,
                            "perMillion": round(count / tokens * 1_000_000, 2),
                        }
                    )
                words = index.neighbours(word)
                if words:
                    neighbours[word].append(
                        {
                            "layer": cell.layer,
                            "field": cell.name,
                            "decade": cell.decade,
                            "method": method,
                            "words": words,
                        }
                    )
            log(f"  {method} {cell.layer}/{cell.name}_{cell.decade}")

    within = pd.read_parquet(drift_dir / "within_field.parquet")
    # Drift tables measured before the books layer was retired still hold its rows.
    within = within[within.layer.isin(LAYERS)]
    cross = pd.read_parquet(drift_dir / "cross_field.parquet")
    fields = sorted({cell["field"] for cell in cells})
    _dump(
        output_dir / "corpus.json",
        {
            "fields": [
                {"slug": field, "label": FIELD_LABELS.get(field, field)} for field in fields
            ],
            "decades": sorted({cell["decade"] for cell in cells}),
            "cells": cells,
        },
    )
    _dump(
        output_dir / "words.json",
        {
            "words": [
                {
                    "word": word,
                    "tier": tiers.get(word, "core"),
                    "measurableCells": sum(
                        1 for entry in neighbours[word] if entry["method"] == "twec"
                    ),
                }
                for word in targets
            ]
        },
    )
    for word in targets:
        _dump(
            output_dir / "words" / f"{word}.json",
            {
                "word": word,
                "counts": counts[word],
                "neighbours": neighbours[word],
                "within": _rows(within[within.word == word], WITHIN_COLUMNS),
                "cross": _rows(cross[cross.word == word], CROSS_COLUMNS),
            },
        )
    perplexity = pd.read_parquet(drift_dir / "cross_perplexity_corrected.parquet")
    grouped = (
        perplexity.sort_values("seed")
        .groupby(["decade", "model_field", "text_field"])["excess"]
        .apply(list)
        .reset_index()
    )
    _dump(
        output_dir / "perplexity.json",
        {
            "fields": sorted(set(perplexity.text_field)),
            "decades": sorted(int(decade) for decade in set(perplexity.decade)),
            "rows": [
                {
                    "decade": int(row["decade"]),
                    "model": row["model_field"],
                    "text": row["text_field"],
                    "excess": [round(float(value), 4) for value in row["excess"]],
                }
                for row in grouped.to_dict(orient="records")
            ],
        },
    )
    falls_path = drift_dir / "cross_perplexity_decomposition.parquet"
    if falls_path.exists():
        falls = pd.read_parquet(falls_path)
        grouped_falls = (
            falls.sort_values("seed")
            .groupby(["model_field", "text_field", "early", "late"])
            .agg(
                fall=("fall", "mean"),
                mix=("mix", "mean"),
                rate=("rate", "mean"),
                seeds=("fall", list),
            )
            .reset_index()
        )
        _dump(
            output_dir / "falls.json",
            {
                "rows": [
                    {
                        "model": row["model_field"],
                        "text": row["text_field"],
                        "early": int(row["early"]),
                        "late": int(row["late"]),
                        "fall": round(float(row["fall"]), 4),
                        "mix": round(float(row["mix"]), 4),
                        "rate": round(float(row["rate"]), 4),
                        "seeds": [round(float(value), 4) for value in row["seeds"]],
                    }
                    for row in grouped_falls.to_dict(orient="records")
                ]
            },
        )
    _dump(output_dir / "predictions.json", parse_predictions(predictions_paths))
    log(f"wrote {output_dir}")


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8")


PREDICTIONS_PATHS = (Path("docs/predictions.md"), Path("docs/predictions_cross_perplexity.md"))


if __name__ == "__main__":
    import sys

    export_web(
        model_dir=Path("models/twec_grid"),
        ppmi_dir=Path("models/ppmi_grid"),
        drift_dir=Path("data/drift"),
        audit_path=Path("data/phrased/audit.json"),
        predictions_paths=PREDICTIONS_PATHS,
        words_file=Path("data/words_of_interest.txt"),
        output_dir=Path(sys.argv[1]) if len(sys.argv) > 1 else Path("web/data"),
    )
