"""End-to-end smoke test: the whole CLI pipeline on a synthetic two-field, two-decade grid.

Each stage reads what the previous one wrote, as in a real run, so a broken hand-off between
commands fails here even when every library function passes its own tests.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl
from click.testing import CliRunner
from gensim.models import KeyedVectors

from interpretant.__main__ import cli

SHARED = ["the", "of", "and", "in", "a", "study", "model", "user", "system", "theory"]
FIELD_WORDS = {
    "philosophy": ["mind", "consciousness", "perception", "experience", "world", "body"],
    "hci": ["interface", "interfaces", "interaction", "design", "usability", "screen"],
}
TARGETS = ["consciousness", "interface", "model"]
CONTROLS = ["study", "theory"]
LINES_PER_CELL = 1_300


def write_cell(path: Path, field: str, decade: int) -> None:
    rng = random.Random(f"{field}{decade}")
    vocabulary = SHARED + FIELD_WORDS[field]
    lines = [" ".join(rng.choices(vocabulary, k=12)) for _ in range(LINES_PER_CELL)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*args: str | Path) -> str:
    result = CliRunner().invoke(cli, [str(arg) for arg in args])
    assert result.exit_code == 0, f"{args[:2]} failed:\n{result.output}\n{result.exception!r}"
    return result.output


def test_pipeline_runs_from_processed_text_to_drift_tables(tmp_path: Path) -> None:
    processed, lemmatised, phrased = (tmp_path / name for name in ("p", "l", "ph"))
    models, ppmi, drift = (tmp_path / name for name in ("twec", "ppmi", "drift"))
    for field in FIELD_WORDS:
        for decade in (1990, 2000):
            write_cell(processed / "openalex" / "texts" / f"{field}_{decade}.txt", field, decade)
    words, controls, overrides = (tmp_path / name for name in ("w.txt", "c.txt", "o.tsv"))
    words.write_text("\n".join(TARGETS))
    controls.write_text("\n".join(CONTROLS))
    overrides.write_text("# none\n")
    texts = phrased / "openalex" / "texts"

    run("corpus", "lemmas", "learn", "--processed-dir", processed, "--source", "openalex",
        "--words-file", words, "--overrides", overrides,
        "--output", processed / "lemmas.tsv")  # fmt: skip
    run("corpus", "lemmas", "apply", "--processed-dir", processed, "--source", "openalex",
        "--map", processed / "lemmas.tsv", "--output-dir", lemmatised)  # fmt: skip
    run("corpus", "phrases", "learn", "--processed-dir", lemmatised, "--source", "openalex",
        "--words-file", words, "--tokens-per-file", "5000",
        "--output", lemmatised / "p.model")  # fmt: skip
    run("corpus", "phrases", "apply", "--processed-dir", lemmatised, "--source", "openalex",
        "--model", lemmatised / "p.model", "--output-dir", phrased)  # fmt: skip
    run("corpus", "audit", "--processed-dir", phrased, "--words-file", words,
        "--output", phrased / "audit.json")  # fmt: skip
    run("twec", "grid", "--input-dir", phrased, "--output-dir", models, "--vector-size", "8",
        "--epochs", "1", "--workers", "1", "--min-count", "1",
        "--compass-tokens-per-cell", "2000")  # fmt: skip
    run("ppmi", "grid", "--input-dir", phrased, "--layer", "openalex", "--output-dir", ppmi,
        "--dimensions", "4", "--min-count", "5")  # fmt: skip
    run("drift", "grid", "--model-dir", models, "--output-dir", drift, "--words-file", words,
        "--controls-file", controls, "--measure-min", "20", "--neighbour-min", "10",
        "--neighbours", "5")  # fmt: skip
    run("drift", "perplexity", "--texts-dir", texts, "--output", drift / "cross.parquet",
        "--fixed-budget", "2000", "--drop-shared", "--seed", "1")  # fmt: skip
    run("drift", "decompose", "--texts-dir", texts, "--early", "1990", "--late", "2000",
        "--seed", "1", "--fixed-budget", "2000", "--output", drift / "falls.parquet")  # fmt: skip

    hci = (texts / "hci_1990.txt").read_text(encoding="utf-8").split()
    assert "interfaces" not in hci
    assert "interface" in hci
    audit = json.loads((phrased / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["cells"]) == 4
    for root in (models, ppmi):
        vectors = KeyedVectors.load(str(root / "openalex" / "hci_2000.kv"))
        assert "interface" in vectors.key_to_index
    within = pl.read_parquet(drift / "within_field.parquet")
    cross = pl.read_parquet(drift / "cross_field.parquet")
    assert set(within["field_a"]) == {"hci", "philosophy"}
    assert set(cross["decade_a"]) == {1990, 2000}
    assert within["is_target"].any()
    assert within["is_control"].any()
    perplexity = pl.read_parquet(drift / "cross.parquet")
    assert len(perplexity) == 2 * 4
    own = perplexity.filter(pl.col("model_field") == pl.col("text_field"))
    assert (own["excess"] == 0).all()
    falls = pl.read_parquet(drift / "falls.parquet")
    assert set(falls["text_field"]) == {"hci", "philosophy"}
