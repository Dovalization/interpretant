"""Whether other encoder or classifier settings lift the junk classifier over its bar (an experiment).

On L and E only, leaving the gate alone. Four frozen encoder variants, each vector normalised to
length 1:

- the stage's bge-base vectors (the first token's state of the first 512 tokens);
- bge-base, the tokens' average state over the first 512 tokens;
- bge-base, the first token's state of the first 512 tokens and of the last 512, averaged, so that
  junk at a record's end (reference lists, citation trails) is read;
- bge-large, the tokens' average state over the first 512 tokens.

Each is crossed with a grid of logistic-regression settings (C, class weight, cut-off target), and
the 13 surface features from the gate's stage are added as before. The choice is made by nested
cross-validation, so L's held-out figures never see it. In each outer fold of L (the
pre-registered five folds by source), every combination is scored by an inner cross-fit on the
outer training rows alone. Each inner fold's cut-off is set on the other inner folds' abstracts,
and the most sensitive combination that reaches an inner specificity of 0.97 is picked (else the
most specific). That combination is trained on the outer training rows with the pre-registered
procedure (`review.junk_classifier.train_classifier`) and calls the held-out fold.

Also reported:

- the pre-registered settings on the stage's vectors, through `cross_fit`, as a check that the
  set-up reproduces the stage's figures;
- each combination's own held-out figures on L, which are optimistic because they are read after
  seeing L;
- the combination that the same selection picks on all of L, which is the one that would be used.

Prints and writes rates only. Run on the runner from the repository root:
`python scripts/evidence/classifier_settings.py OUT`.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import polars as pl
import torch
import yaml
from joblib import Parallel, delayed
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits
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
    SENSITIVITY_BAR,
    SPECIFICITY_BAR,
    ClassifierSpec,
    TrainingSet,
    _fit_scorer,
    build_folds,
    build_record_labels,
    build_sources,
    compute_cut_off,
    compute_quality,
    cross_fit,
    train_classifier,
)

MAX_TOKENS = 512
BATCH_SIZE = 32
BASE = ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
LARGE = ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09")
# name: (model, pooling, reading); None for the stage's own vectors
VARIANTS: dict[str, tuple[tuple[str, str], str, str] | None] = {
    "bge-base first token, head (the stage's)": None,
    "bge-base average, head": (BASE, "average", "head"),
    "bge-base first token, head and tail": (BASE, "first", "head_and_tail"),
    "bge-large average, head": (LARGE, "average", "head"),
}
REGULARISATIONS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CLASS_WEIGHTS = ("balanced", None)
TARGETS = (0.97, 0.975, 0.98)
SCOPE_ALL = -1  # the selection's scope over all of L, beside the outer folds 0..4


@dataclass(frozen=True)
class Setting:
    """One combination of the grid."""

    variant: str
    regularisation: float
    class_weight: str | None
    target: float


def build_encoder(model_name: str, revision: str, pooling: str, reading: str) -> Callable[[Sequence[str]], NDArray[np.float32]]:
    """A frozen encoder: first-token or average pooling, over the head or the head and the tail."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device=device, dtype=torch.float32).eval()
    pad_id = int(tokenizer.pad_token_id)

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        states = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).last_hidden_state
        if pooling == "first":
            return states[:, 0].cpu()
        mask = attention_mask.to(device).unsqueeze(-1).to(states.dtype)
        return ((states * mask).sum(dim=1) / mask.sum(dim=1)).cpu()

    def encode(token_ids: Sequence[Sequence[int]]) -> NDArray[np.float32]:
        return encode_token_ids(token_ids, forward, pad_id=pad_id, batch_size=BATCH_SIZE)

    def encode_texts(texts: Sequence[str]) -> NDArray[np.float32]:
        if reading == "head":
            return encode(tokenizer(list(texts), truncation=True, max_length=MAX_TOKENS)["input_ids"])
        heads, tails = [], []
        for row in tokenizer(list(texts), truncation=False)["input_ids"]:
            if len(row) <= MAX_TOKENS:
                heads.append(row)
                tails.append(row)
            else:  # keep the closing and the opening special tokens
                heads.append([*row[: MAX_TOKENS - 1], row[-1]])
                tails.append([row[0], *row[-(MAX_TOKENS - 1) :]])
        averaged = encode(heads) + encode(tails)
        return (averaged / np.linalg.norm(averaged, axis=1, keepdims=True)).astype(np.float32)

    return encode_texts


def held_out_scores(training: TrainingSet, rows: NDArray[np.intp], spec: ClassifierSpec, folds: NDArray[np.int64]) -> NDArray[np.float64]:
    """Each L row of ``rows`` scored by a model fitted on the other rows, E always among them."""
    with threadpool_limits(1):
        labelled = training.in_labelled_sample[rows]
        scores = np.full(rows.size, np.nan)
        for fold in range(spec.folds):
            is_held_out = labelled & (folds == fold)
            if not is_held_out.any():
                continue
            scorer = _fit_scorer(training, rows[~is_held_out], spec)
            held_out = rows[is_held_out]
            scores[is_held_out] = scorer.compute_scores(training.vectors[held_out], training.features[held_out])
        return scores


def fit_or_none(training: TrainingSet, rows: NDArray[np.intp], spec: ClassifierSpec, folds: NDArray[np.int64]) -> NDArray[np.float64] | None:
    """``held_out_scores``, or None when a fit does not converge (not the model the spec describes)."""
    try:
        return held_out_scores(training, rows, spec, folds)
    except Exception:
        return None


def calls_with_fold_cut_offs(scores: NDArray[np.float64], is_junk: NDArray[np.bool_], labelled: NDArray[np.bool_], folds: NDArray[np.int64], target: float, fold_count: int) -> NDArray[np.bool_]:
    """Each fold's calls at the cut-off set on the other folds' abstracts."""
    calls = np.zeros(scores.size, dtype=np.bool_)
    for fold in range(fold_count):
        is_held_out = labelled & (folds == fold)
        if not is_held_out.any():
            continue
        reference = labelled & ~is_junk & (folds != fold)
        calls[is_held_out] = scores[is_held_out] >= compute_cut_off(scores[reference], target=target)
    return calls


def rates(is_junk: NDArray[np.bool_], calls: NDArray[np.bool_]) -> dict[str, list[float]]:
    """Sensitivity and specificity as [count, total, share]."""
    quality = compute_quality(is_junk, calls)
    return {
        name: [rate.count, rate.total, round(rate.share, 4)]
        for name, rate in (("sensitivity", quality.sensitivity), ("specificity", quality.specificity))
    }


def pick(results: dict[Setting, dict[str, list[float]]]) -> Setting:
    """The most sensitive setting reaching the specificity bar; else the most specific."""
    reaching = [s for s, r in results.items() if r["specificity"][2] >= SPECIFICITY_BAR]
    if reaching:
        return max(reaching, key=lambda s: (results[s]["sensitivity"][2], results[s]["specificity"][2]))
    return max(results, key=lambda s: (results[s]["specificity"][2], results[s]["sensitivity"][2]))


def load_training_data() -> tuple[pl.DataFrame, NDArray[np.float64], tuple[str, ...], list[str]]:
    """L and E's records (sorted by work id), their features, sources and kept texts."""
    catalog = read_catalog()
    labels = load_junk_labels(catalog)
    records = build_record_labels(labels).table.sort("work_id")
    part_of: dict[str, str] = {}
    for _, label in sorted(labels.items.items()):
        if label.set in ("labelled", "enrichment"):
            part_of.setdefault(label.work_id, label.part)
    records = records.with_columns(pl.col("work_id").replace_strict(part_of).alias("part"))
    features_table = load_junk_scores(catalog).select("work_id", *FEATURE_NAMES)
    records = records.join(features_table, on="work_id", how="left").sort("work_id")
    staged = load_staging_table(catalog, "openalex").select("work_id", "issn_l", "source_name").collect()
    sources = tuple(build_sources(records.join(staged, on="work_id", how="left").sort("work_id")))
    features = records.select(list(FEATURE_NAMES)).to_numpy().astype(np.float64)
    texts: dict[str, str] = {}
    for part, group in records.group_by("part"):
        table = load_abstracts_part(catalog, str(part[0]), ["work_id", "text", "removals"]).filter(
            pl.col("work_id").is_in(group["work_id"].to_list())
        )
        for work_id, text, removals in table.iter_rows():
            texts[work_id] = kept_text(text, [Removal(**removal) for removal in removals])
    stage_vectors = load_record_vectors(catalog).filter(pl.col("work_id").is_in(records["work_id"])).collect()
    stage_vectors = records.select("work_id").join(stage_vectors, on="work_id", how="left")
    records = records.with_columns(pl.Series("stage_vector", stage_vectors["vector"]))
    return records, features, sources, [texts[w] for w in records["work_id"]]


def main() -> None:
    started = time.time()
    records, features, sources, texts = load_training_data()
    is_junk = records["is_junk"].to_numpy()
    in_labelled = records["in_labelled_sample"].to_numpy()
    settings = yaml.safe_load(open("params.yaml"))["junk_gate"]["classifier"]
    base_spec = ClassifierSpec(
        regularisation=settings["regularisation"], class_weight=settings["class_weight"],
        folds=settings["folds"], specificity_target=settings["specificity_target"],
        max_iterations=settings["max_iterations"],
    )
    vectors: dict[str, NDArray[np.float32]] = {}
    for name, variant in VARIANTS.items():
        if variant is None:
            vectors[name] = np.stack(records["stage_vector"].to_numpy()).astype(np.float32)
        else:
            (model_name, revision), pooling, reading = variant
            vectors[name] = build_encoder(model_name, revision, pooling, reading)(texts)
            torch.cuda.empty_cache()
        print(f"encoded: {name} ({time.time() - started:.0f} s)", flush=True)
    training = {
        name: TrainingSet(vectors=v, features=features, is_junk=is_junk, in_labelled_sample=in_labelled, sources=sources)
        for name, v in vectors.items()
    }

    # The check: the pre-registered settings on the stage's vectors give the stage's figures
    stage = next(iter(VARIANTS))
    check = cross_fit(training[stage], base_spec)
    reproduction = rates(is_junk[in_labelled], check.predicted_junk[in_labelled])
    print("reproduction of the stage's figures:", json.dumps(reproduction), flush=True)

    # Held-out scores for each (variant, C, class weight) and each scope: the five outer folds,
    # each scored by an inner cross-fit of depth 1, and all of L, by the outer folds of depth 0
    labelled_rows = np.flatnonzero(in_labelled)
    enrichment_rows = np.flatnonzero(~in_labelled)
    outer = build_folds(sources, folds=base_spec.folds, depth=0)
    scope_rows = {SCOPE_ALL: np.arange(is_junk.size)}
    scope_folds = {SCOPE_ALL: outer}
    for fold in range(base_spec.folds):
        rows = np.concatenate([labelled_rows[outer[labelled_rows] != fold], enrichment_rows])
        scope_rows[fold] = rows
        scope_folds[fold] = build_folds([sources[row] for row in rows], folds=base_spec.folds, depth=1)
    fits = list(itertools.product(VARIANTS, REGULARISATIONS, CLASS_WEIGHTS, scope_rows))
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(fits)} cross-fits of 5 models each, on {workers} workers", flush=True)
    scored = Parallel(n_jobs=workers, verbose=5)(
        delayed(fit_or_none)(
            training[variant], scope_rows[scope],
            replace(base_spec, regularisation=regularisation, class_weight=class_weight),
            scope_folds[scope],
        )
        for variant, regularisation, class_weight, scope in fits
    )
    scores = dict(zip(fits, scored, strict=True))
    failed = sorted({f"{v} C={c} {w}" for (v, c, w, _), s in scores.items() if s is None})

    def results_in(scope: int) -> dict[Setting, dict[str, list[float]]]:
        rows, folds = scope_rows[scope], scope_folds[scope]
        labelled, junk = in_labelled[rows], is_junk[rows]
        results = {}
        for variant, regularisation, class_weight in itertools.product(VARIANTS, REGULARISATIONS, CLASS_WEIGHTS):
            fold_scores = scores[(variant, regularisation, class_weight, scope)]
            if fold_scores is None:
                continue
            for target in TARGETS:
                calls = calls_with_fold_cut_offs(fold_scores, junk, labelled, folds, target, base_spec.folds)
                results[Setting(variant, regularisation, class_weight, target)] = rates(junk[labelled], calls[labelled])
        return results

    # Nested: pick inside each outer fold's training rows, train there, call the held-out fold
    nested_calls = np.zeros(is_junk.size, dtype=np.bool_)
    chosen = {}
    for fold in range(base_spec.folds):
        setting = pick(results_in(fold))
        chosen[fold] = setting
        spec = replace(base_spec, regularisation=setting.regularisation, class_weight=setting.class_weight, specificity_target=setting.target)
        classifier = train_classifier(training[setting.variant], scope_rows[fold], spec)
        held_out = labelled_rows[outer[labelled_rows] == fold]
        data = training[setting.variant]
        nested_calls[held_out] = classifier.scorer.compute_scores(data.vectors[held_out], data.features[held_out]) >= classifier.cut_off
        print(f"outer fold {fold}: picked {setting}", flush=True)
    nested = rates(is_junk[in_labelled], nested_calls[in_labelled])
    nested["meets_bar"] = nested["sensitivity"][2] >= SENSITIVITY_BAR and nested["specificity"][2] >= SPECIFICITY_BAR

    optimistic = results_in(SCOPE_ALL)
    picked_on_all = pick(optimistic)
    result = {
        "reproduction_of_the_stage": reproduction,
        "nested": nested,
        "chosen_by_outer_fold": {str(f): s.__dict__ for f, s in chosen.items()},
        "picked_on_all_of_l": picked_on_all.__dict__ | {"optimistic_rates": optimistic[picked_on_all]},
        "optimistic_by_setting": [s.__dict__ | r for s, r in sorted(optimistic.items(), key=lambda kv: -kv[1]["sensitivity"][2])],
        "unconverged": failed,
        "bar": {"sensitivity": SENSITIVITY_BAR, "specificity": SPECIFICITY_BAR},
        "seconds": round(time.time() - started),
    }
    print("nested:", json.dumps(nested), flush=True)
    print("picked on all of L:", json.dumps(result["picked_on_all_of_l"]), flush=True)
    reaching = [row for row in result["optimistic_by_setting"] if row["specificity"][2] >= SPECIFICITY_BAR]
    print(f"optimistic: {len(reaching)} of {len(optimistic)} settings reach the bar's specificity; best five:", flush=True)
    for row in reaching[:5]:
        print("  ", json.dumps(row), flush=True)
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
