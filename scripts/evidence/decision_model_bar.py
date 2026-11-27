"""Whether the local decision model meets the junk classifier's bar (`docs/junk_gate.md`).

The pre-registration's fallback when the classifier fails its bar: the decision model's junk
probability takes the classifier's place if, with its cut-off set by the same rule, it meets the
bar (sensitivity of at least 0.70 and specificity of at least 0.97 on L's held-out calls).

The rule, for a model that is not trained: L's records are split in 5 folds by their source (as
`review.junk_classifier.build_folds`, depth 0); each fold's cut-off is the lowest probability that
still clears 97% of the abstracts in the other folds (`compute_cut_off`), and calls the held-out
fold. The final cut-off is set the same way on all of L. A record's probability and label are its
item with the smallest id, as the classifier's labels take theirs.

Run on the runner from the repository root, after `judge-ollama` has judged the sample:
`python scripts/evidence/decision_model_bar.py SAMPLE_FOLDER SETUP OUT`, where SETUP names the
decision model's answers under `SAMPLE_FOLDER/verdicts/local/`. Writes counts and rates only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from interpretant.catalog.config import read_catalog
from interpretant.catalog.junk_gate import JunkLabels, load_junk_labels
from interpretant.catalog.staging import load_staging_table
from interpretant.review.junk_classifier import (
    SENSITIVITY_BAR,
    SPECIFICITY_BAR,
    build_folds,
    build_record_labels,
    build_sources,
    compute_cut_off,
    compute_quality,
)
from interpretant.review.verdicts import load_parts

FOLDS = 5


def build_record_probabilities(
    labels: JunkLabels, answers: dict[str, dict[str, Any]]
) -> pl.DataFrame:
    """Each L or E record's junk probability, from its item with the smallest id."""
    rows = [
        (item_id, label.work_id, 1.0 - float(answers[item_id]["probabilities"]["own_prose"]))
        for item_id, label in sorted(labels.items.items())
        if label.set in ("labelled", "enrichment")
    ]
    items = pl.DataFrame(
        rows,
        schema={"item_id": pl.String(), "work_id": pl.String(), "probability": pl.Float64()},
        orient="row",
    )
    return items.group_by("work_id", maintain_order=True).agg(pl.col("probability").first())


def evaluate(records: pl.DataFrame, sources: list[str]) -> dict[str, object]:
    """Cross-fitted calls on L with the classifier's cut-off rule, and the final cut-off.

    ``records`` holds L's records in the order of ``sources``: ``is_junk`` and ``probability``.
    """
    probability = records["probability"].to_numpy()
    is_junk = records["is_junk"].to_numpy()
    folds = build_folds(sources, folds=FOLDS, depth=0)
    predicted = np.zeros(is_junk.size, dtype=np.bool_)
    cut_offs = {}
    for fold in range(FOLDS):
        held_out = folds == fold
        if not held_out.any():
            continue
        training_abstracts = probability[~held_out & ~is_junk]
        cut_off = compute_cut_off(training_abstracts, target=SPECIFICITY_BAR)
        predicted[held_out] = probability[held_out] >= cut_off
        cut_offs[fold] = cut_off
    quality = compute_quality(is_junk, predicted)
    final = compute_cut_off(probability[~is_junk], target=SPECIFICITY_BAR)
    return {
        "records": int(is_junk.size),
        "junk_records": int(is_junk.sum()),
        "sensitivity": quality.sensitivity.__dict__,
        "specificity": quality.specificity.__dict__,
        "meets_bar": quality.sensitivity.share >= SENSITIVITY_BAR
        and quality.specificity.share >= SPECIFICITY_BAR,
        "cut_offs_by_fold": cut_offs,
        "final_cut_off": final,
        "auc": float(roc_auc_score(is_junk, probability)),
    }


def main() -> None:
    """Evaluate the named decision model's answers on the sample and write the result."""
    sample, setup, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    catalog = read_catalog()
    labels = load_junk_labels(catalog)
    answers = load_parts(sample / "verdicts" / "local", setup)
    record_labels = build_record_labels(labels).table
    labelled = (
        record_labels.filter(pl.col("in_labelled_sample"))
        .join(build_record_probabilities(labels, answers), on="work_id", how="left")
        .sort("work_id")
    )
    if labelled["probability"].null_count():
        raise SystemExit(f"{labelled['probability'].null_count()} records of L have no answer")
    staged = load_staging_table(catalog, "openalex").select("work_id", "issn_l", "source_name")
    with_sources = labelled.join(staged.collect(), on="work_id", how="left").sort("work_id")
    result = {"setup": setup, **evaluate(labelled, build_sources(with_sources))}
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
