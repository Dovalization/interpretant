"""Whether a stronger encoder lifts the junk classifier over its bar (an experiment, not the gate).

On L and E only: each candidate encoder encodes the records' kept text, and the classifier is
cross-fitted with the pre-registered rules and settings (`review.junk_classifier.cross_fit`, the
13 surface features from the gate's stage, folds by source). The stage's own bge-base vectors are
scored the same way first, to check the set-up reproduces the stage's figures. Prints and writes
rates only. Run on the runner from the repository root: `python encoder_comparison.py OUT`.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import read_catalog
from interpretant.catalog.junk_gate import load_junk_labels
from interpretant.catalog.junk_gate_tables import load_junk_scores
from interpretant.catalog.record_vectors import load_record_vectors
from interpretant.catalog.staging import load_staging_table
from interpretant.corpus.removals import Removal, kept_text
from interpretant.pipelines.embed_records import encode_token_ids
from interpretant.review.junk_classifier import (
    FEATURE_NAMES,
    ClassifierSpec,
    TrainingSet,
    build_record_labels,
    build_sources,
    compute_quality,
    cross_fit,
)

MAX_TOKENS = 512
CANDIDATES = {
    "bge-large-en-v1.5": (
        "BAAI/bge-large-en-v1.5",
        "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "first",
    ),
    "Qwen3-Embedding-0.6B": (
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "last",
    ),
}


def build_encoder(name: str, revision: str, pooling: str) -> Callable[[Sequence[str]], np.ndarray]:
    """A frozen encoder: the first token's state (BERT-style) or the last real token's (Qwen)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)  # type: ignore[no-untyped-call]
    model = (
        AutoModel.from_pretrained(name, revision=revision)
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    pad_id = int(
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        states = model(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
        ).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        pooled: torch.Tensor = (
            states[:, 0]
            if pooling == "first"
            else states[torch.arange(states.size(0)), last.to(device)]
        )
        return pooled.cpu()

    def encode(texts: Sequence[str]) -> np.ndarray:
        ids = tokenizer(list(texts), truncation=True, max_length=MAX_TOKENS - (pooling == "last"))[
            "input_ids"
        ]
        if pooling == "last":
            ids = [[*row, tokenizer.eos_token_id] for row in ids]
        return encode_token_ids(ids, forward, pad_id=pad_id, batch_size=32)

    return encode


def main() -> None:
    """Encode L and E with each candidate, cross-fit the classifier on each, write the rates."""
    catalog = read_catalog()
    labels = load_junk_labels(catalog)
    records = build_record_labels(labels).table.sort("work_id")
    part_of: dict[str, str] = {}
    for _, item in sorted(labels.items.items()):
        if item.set in ("labelled", "enrichment"):
            part_of.setdefault(item.work_id, item.part)
    records = records.with_columns(pl.col("work_id").replace_strict(part_of).alias("part"))
    features_table = load_junk_scores(catalog).select("work_id", *FEATURE_NAMES)
    records = records.join(features_table, on="work_id", how="left").sort("work_id")
    staged = (
        load_staging_table(catalog, "openalex").select("work_id", "issn_l", "source_name").collect()
    )
    sources = tuple(build_sources(records.join(staged, on="work_id", how="left").sort("work_id")))
    features = records.select(list(FEATURE_NAMES)).to_numpy().astype(np.float64)
    is_junk = records["is_junk"].to_numpy()
    in_labelled = records["in_labelled_sample"].to_numpy()
    texts: dict[str, str] = {}
    for part, group in records.group_by("part"):
        wanted = group["work_id"].to_list()
        table = load_abstracts_part(catalog, str(part[0]), ["work_id", "text", "removals"]).filter(
            pl.col("work_id").is_in(wanted)
        )
        for work_id, text, removals in table.iter_rows():
            texts[work_id] = kept_text(text, [Removal(**removal) for removal in removals])
    ordered_texts = [texts[work_id] for work_id in records["work_id"]]
    settings = yaml.safe_load(Path("params.yaml").read_text())["junk_gate"]["classifier"]
    spec = ClassifierSpec(
        regularisation=settings["regularisation"],
        class_weight=settings["class_weight"],
        folds=settings["folds"],
        specificity_target=settings["specificity_target"],
        max_iterations=settings["max_iterations"],
    )
    stage_vectors = (
        load_record_vectors(catalog).filter(pl.col("work_id").is_in(records["work_id"])).collect()
    )
    stage_vectors = records.select("work_id").join(stage_vectors, on="work_id", how="left")
    vector_sets: dict[str, np.ndarray] = {
        "bge-base-en-v1.5 (the stage's vectors)": np.asarray(
            stage_vectors["vector"].to_list(), dtype=np.float32
        )
    }
    timings: dict[str, float] = {}
    for candidate, (name, revision, pooling) in CANDIDATES.items():
        started = time.time()
        vector_sets[candidate] = build_encoder(name, revision, pooling)(ordered_texts)
        timings[candidate] = round(len(ordered_texts) / (time.time() - started), 1)
        torch.cuda.empty_cache()
    results: dict[str, dict[str, Any]] = {}
    for candidate, vectors in vector_sets.items():
        training = TrainingSet(
            vectors=vectors.astype(np.float32),
            features=features,
            is_junk=is_junk,
            in_labelled_sample=in_labelled,
            sources=sources,
        )
        try:
            fitted = cross_fit(training, spec)
            quality = compute_quality(is_junk[in_labelled], fitted.predicted_junk[in_labelled])
            results[candidate] = {
                "dimensions": int(vectors.shape[1]),
                "records_per_second": timings.get(candidate),
                "sensitivity": [
                    quality.sensitivity.count,
                    quality.sensitivity.total,
                    round(quality.sensitivity.share, 3),
                ],
                "specificity": [
                    quality.specificity.count,
                    quality.specificity.total,
                    round(quality.specificity.share, 4),
                ],
                "meets_bar": quality.sensitivity.share >= 0.70
                and quality.specificity.share >= 0.97,
            }
        except Exception as error:
            results[candidate] = {"error": f"{type(error).__name__}: {error}"[:300]}
        sys.stdout.write(f"{candidate} {json.dumps(results[candidate])}\n")
        sys.stdout.flush()
    Path(sys.argv[1]).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
