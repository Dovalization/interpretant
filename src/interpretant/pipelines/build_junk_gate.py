"""The junk gate's stage (`docs/junk_gate.md`): the records' scores, the OCR scores and the screen.

Worker processes read the corpus part by part twice. First they count, for each lower-case word,
the kept works whose kept text holds it: a word in enough of them is known. Then they describe each
record behind a drawn usage (its 13 surface features, the finite verb read from the static view as
round S1 reads it) and each drawn usage (its sentence's unknown words, the target left out). The
classifier learns from the judges' labels on L and E (the catalog's ``junk_labels``), from each
record's vector (``record_vectors``) and its source (the OpenAlex staging table's ISSN-L or source
name): it is cross-fitted on L by source, and the final model, trained on L and E, scores every
record outside L. The screen flags the cells with more drawn usages from records called junk than
their decade's rate gives by chance. The metrics name words by their reported names only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog.abstracts import abstracts_parts, load_abstracts_part
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.junk_gate import JunkLabels, load_junk_labels
from interpretant.catalog.junk_gate_tables import (
    save_junk_cells,
    save_junk_classifier,
    save_junk_gate_metrics,
    save_junk_scores,
    save_usage_ocr_scores,
)
from interpretant.catalog.record_vectors import load_record_vectors
from interpretant.catalog.staging import load_staging_table
from interpretant.catalog.static_view import load_static_view
from interpretant.catalog.usages import load_usages_part
from interpretant.catalog.words import Word, load_words
from interpretant.corpus.removals import kept_text
from interpretant.pipelines.params import ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts, worker_pool
from interpretant.review.finite_verbs import has_finite_verb_outside
from interpretant.review.junk_classifier import (
    FEATURE_NAMES,
    SENSITIVITY_BAR,
    SPECIFICITY_BAR,
    ClassifierQuality,
    ClassifierSpec,
    CrossFitResult,
    JunkClassifier,
    Rate,
    TrainingSet,
    build_record_labels,
    build_sources,
    compute_corrections,
    compute_false_alarms,
    compute_quality,
    compute_rate,
    compute_removed_share,
    compute_surface_features,
    cross_fit,
    format_classifier,
    train_final_classifier,
)
from interpretant.review.junk_gate import parse_removals
from interpretant.review.junk_rules import screen_cells
from interpretant.review.ocr_damage import (
    AUC_BAR,
    build_corpus_words,
    compute_ocr_check,
    count_document_frequencies,
    count_unknown_words,
    find_lower_case_words,
)

# The usages table's columns a drawn usage is described from
_USAGE_COLUMNS = [
    "work_id",
    "word",
    "content_key",
    "sentence_start",
    "sentence_end",
    "target_start",
    "target_end",
    "drawn",
]
# The OCR scores' table's columns, from the described usages
_USAGE_OCR_COLUMNS = ["work_id", "word", "content_key", "ocr_unknown_share", "ocr_words"]
_RECORDS_SCHEMA = pl.Schema(
    {
        "work_id": pl.String(),
        "decade": pl.Int64(),
        "fields": pl.List(pl.String()),
        **dict.fromkeys(FEATURE_NAMES, pl.Float64()),
    }
)
_USAGES_SCHEMA = pl.Schema(
    {
        "work_id": pl.String(),
        "word": pl.String(),
        "content_key": pl.String(),
        "decade": pl.Int64(),
        "ocr_unknown": pl.Int64(),
        "ocr_words": pl.Int64(),
    }
)


class ClassifierSettings(ParamsSettings):
    """The ``junk_gate.classifier`` section of ``params.yaml``: training and the cut-off."""

    model_config = SettingsConfigDict(yaml_config_section="junk_gate.classifier")

    regularisation: float = Field(gt=0)
    class_weight: Literal["balanced"]
    folds: int = Field(ge=2)
    specificity_target: float = Field(gt=0, le=1)
    max_iterations: int = Field(gt=0)


class ScreenSettings(ParamsSettings):
    """The ``junk_gate.screen`` section of ``params.yaml``: which cells are gated and flagged."""

    model_config = SettingsConfigDict(yaml_config_section="junk_gate.screen")

    min_drawn_usages: int = Field(gt=0)
    alpha: float = Field(gt=0, lt=1)


class OcrSettings(ParamsSettings):
    """The ``junk_gate.ocr`` section of ``params.yaml``: the words read and which are known."""

    model_config = SettingsConfigDict(yaml_config_section="junk_gate.ocr")

    min_letters: int = Field(gt=0)
    min_words: int = Field(gt=0)
    min_known_works: int = Field(gt=0)


@dataclass(frozen=True)
class WordCountJob:
    """What a worker counting one part's words needs (picklable)."""

    catalog: CatalogConfig
    min_letters: int


@dataclass(frozen=True)
class DescribeJob:
    """What a worker describing one part's records and drawn usages needs (picklable)."""

    catalog: CatalogConfig
    corpus_words: frozenset[str]
    min_letters: int


@dataclass(frozen=True)
class PartDescription:
    """One part's records behind drawn usages, and its drawn usages' unknown words."""

    records: pl.DataFrame
    usages: pl.DataFrame


@dataclass(frozen=True)
class JunkGateResult:
    """What the stage wrote: its records and usages, the classifier's quality and the screen."""

    record_count: int
    usage_count: int
    labelled_count: int
    quality: ClassifierQuality
    gated_count: int
    flagged_count: int


def build_junk_gate(
    catalog: CatalogConfig,
    classifier: ClassifierSettings,
    screen: ScreenSettings,
    ocr: OcrSettings,
    *,
    process_count: int,
    progress: Progress = SILENT,
) -> JunkGateResult:
    """Score every record behind a drawn usage and every drawn usage, screen the cells, write all.

    Raises:
        FileNotFoundError: if the abstracts table has no part, or an input is missing.
        ValueError: if a record behind a drawn usage lacks its text, static view or vector, or a
            labelled usage is not a drawn usage of the table.
        sklearn.exceptions.ConvergenceWarning: as an error, if a fit does not converge.
    """
    parts = abstracts_parts(catalog)
    if not parts:
        raise FileNotFoundError(f"no part of the abstracts table under {catalog.abstracts.path}")
    words = {word.lemma: word for word in load_words(catalog)}
    labels = load_junk_labels(catalog)
    records, usages = describe_corpus(
        catalog, parts, ocr, process_count=process_count, progress=progress
    )
    vectors = load_vectors(catalog, records)
    features = records.select(list(FEATURE_NAMES)).to_numpy().astype(np.float64)
    record_labels = build_record_labels(labels)
    labelled = find_labelled_rows(records, record_labels.table)
    training = build_training_set(
        vectors, features, labelled, build_sources(load_sources(catalog, records))
    )
    spec = ClassifierSpec(
        regularisation=classifier.regularisation,
        class_weight=classifier.class_weight,
        folds=classifier.folds,
        specificity_target=classifier.specificity_target,
        max_iterations=classifier.max_iterations,
    )
    progress.stage("training the classifier", total=spec.folds + 1)
    cross_fitted = cross_fit(training, spec, report=progress.advance)
    final = train_final_classifier(training, spec)
    progress.advance()
    progress.stage("scoring the records and screening the cells")
    scores = build_scores(records, vectors, features, final, labelled, cross_fitted)
    cells = screen_cells(
        usages.join(scores.select("work_id", "predicted_junk"), on="work_id"),
        min_drawn_usages=screen.min_drawn_usages,
        alpha=screen.alpha,
    )
    is_held_out = training.in_labelled_sample
    quality = compute_quality(
        training.is_junk[is_held_out], cross_fitted.predicted_junk[is_held_out]
    )
    held_out = labelled.filter("in_labelled_sample").with_columns(
        pl.Series("predicted_junk", cross_fitted.predicted_junk[is_held_out])
    )
    items = build_labelled_items(labels, words, usages, scores)
    metrics = {
        "records": _count_records(scores, record_labels.disagreeing_count, labelled),
        "usages": _count_usages(usages, cells),
        "classifier": _report_classifier(quality, cross_fitted, final, records, held_out, items),
        "ocr": _report_ocr(items),
        "cells": _report_cells(cells, words),
    }
    progress.stage("writing the tables")
    save_junk_scores(catalog, scores)
    save_usage_ocr_scores(catalog, usages.select(_USAGE_OCR_COLUMNS))
    save_junk_cells(catalog, cells)
    save_junk_classifier(catalog, format_classifier(final))
    save_junk_gate_metrics(catalog, metrics)
    return JunkGateResult(
        record_count=records.height,
        usage_count=usages.height,
        labelled_count=int(is_held_out.sum()),
        quality=quality,
        gated_count=int(cells["gated"].sum()),
        flagged_count=int(cells["flagged"].sum()),
    )


def describe_corpus(
    catalog: CatalogConfig,
    parts: Sequence[str],
    settings: OcrSettings,
    *,
    process_count: int,
    progress: Progress,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every record behind a drawn usage, described, and every drawn usage's OCR score.

    The records come one per work, sorted by work id: its decade, fields and surface features. The
    usages come sorted by work, word and content key, with their decade, unknown words, words read
    (``ocr_words``) and the share unknown (``ocr_unknown_share``, null under ``min_words``).
    """
    with worker_pool(process_count) as pool:
        progress.stage("counting the words of every kept work", total=len(parts))
        frequencies = map_parts(
            count_words_in_part,
            dict.fromkeys(parts, WordCountJob(catalog, settings.min_letters)),
            process_count=process_count,
            progress=progress,
            pool=pool,
        )
        corpus_words = build_corpus_words(frequencies.values(), min_works=settings.min_known_works)
        progress.stage("describing the records behind drawn usages", total=len(parts))
        described = map_parts(
            describe_part,
            dict.fromkeys(parts, DescribeJob(catalog, corpus_words, settings.min_letters)),
            process_count=process_count,
            progress=progress,
            pool=pool,
        )
    records = (
        pl.concat(
            [pl.DataFrame(schema=_RECORDS_SCHEMA), *(part.records for part in described.values())]
        )
        .unique("work_id", keep="first", maintain_order=True)
        .sort("work_id")
    )
    usages = (
        pl.concat(
            [pl.DataFrame(schema=_USAGES_SCHEMA), *(part.usages for part in described.values())]
        )
        .with_columns(
            pl.when(pl.col("ocr_words") >= settings.min_words)
            .then(pl.col("ocr_unknown") / pl.col("ocr_words"))
            .alias("ocr_unknown_share")
        )
        .sort("work_id", "word", "content_key")
    )
    return records, usages


def count_words_in_part(part: str, job: WordCountJob) -> pl.DataFrame:
    """For each lower-case word, the part's kept works whose kept text holds it."""
    abstracts = load_abstracts_part(
        job.catalog, part, ["text", "removals", "dropped_reason"]
    ).filter(pl.col("dropped_reason").is_null())
    texts = (
        kept_text(text, parse_removals(removals)) for text, removals, _ in abstracts.iter_rows()
    )
    return count_document_frequencies(texts, min_letters=job.min_letters)


def describe_part(part: str, job: DescribeJob) -> PartDescription:
    """One part's records behind drawn usages and its drawn usages.

    A record's features are read from its kept text, with its finite verb from the static view's
    tokens outside removed ranges and its unknown words against the corpus's known words. A
    usage's words are its sentence's, the removed ranges and the target left out.

    Raises:
        ValueError: if a record behind a drawn usage lacks its text or its static view.
    """
    usages = load_usages_part(job.catalog, part, _USAGE_COLUMNS).filter("drawn")
    behind = usages.select("work_id").unique()
    abstracts = load_abstracts_part(
        job.catalog, part, ["work_id", "decade", "fields", "text", "removals"]
    ).join(behind, on="work_id", how="semi")
    view = load_static_view(job.catalog, part, ["work_id", "starts", "finite_verbs"])
    joined = abstracts.join(view, on="work_id", how="inner")
    if joined.height != behind.height:
        raise ValueError(
            f"{behind.height - joined.height} records behind drawn usages of part {part} lack "
            "their text or static view"
        )
    records, texts, removed, decades = [], {}, {}, {}
    for work_id, decade, fields, text, removals, starts, finite_verbs in joined.iter_rows():
        spans = [(removal["start"], removal["end"]) for removal in removals]
        kept = kept_text(text, parse_removals(removals))
        kept_words = find_lower_case_words(kept, min_letters=job.min_letters)
        unknown = count_unknown_words(kept_words, job.corpus_words)
        features = compute_surface_features(
            kept,
            has_finite_verb=has_finite_verb_outside(starts, finite_verbs, spans),
            removed_share=compute_removed_share(len(text), spans),
            unknown_word_share=unknown / len(kept_words) if kept_words else 0.0,
        )
        records.append((work_id, decade, fields, *features))
        texts[work_id], removed[work_id], decades[work_id] = text, spans, decade
    described = []
    for work_id, word, content_key, start, end, target_start, target_end, _ in usages.iter_rows():
        sentence_words = find_lower_case_words(
            texts[work_id],
            min_letters=job.min_letters,
            within=(start, end),
            left_out=[*removed[work_id], (target_start, target_end)],
        )
        unknown = count_unknown_words(sentence_words, job.corpus_words)
        described.append(
            (work_id, word, content_key, decades[work_id], unknown, len(sentence_words))
        )
    return PartDescription(
        records=pl.DataFrame(records, schema=_RECORDS_SCHEMA, orient="row"),
        usages=pl.DataFrame(described, schema=_USAGES_SCHEMA, orient="row"),
    )


def load_vectors(catalog: CatalogConfig, records: pl.DataFrame) -> NDArray[np.float32]:
    """The records' vectors, in the records' order (sorted by work id).

    Raises:
        ValueError: if a record has no vector.
    """
    vectors = (
        load_record_vectors(catalog)
        .join(records.select("work_id").lazy(), on="work_id", how="semi")
        .sort("work_id")
        .collect()
    )
    if not vectors["work_id"].equals(records["work_id"]):
        raise ValueError(
            f"{records.height - vectors.height} records behind drawn usages have no vector: "
            "rerun embed-records"
        )
    matrix: NDArray[np.float32] = vectors["vector"].to_numpy()
    return matrix


def load_sources(catalog: CatalogConfig, records: pl.DataFrame) -> pl.DataFrame:
    """Each record's ISSN-L and source name from the OpenAlex staging table, null when unknown."""
    staged = (
        load_staging_table(catalog, "openalex").select("work_id", "issn_l", "source_name").collect()
    )
    return records.select("work_id").join(staged, on="work_id", how="left").sort("work_id")


def find_labelled_rows(records: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Each labelled record's row among the records, with its label and set, by work id.

    Raises:
        ValueError: if a labelled record is not behind a drawn usage.
    """
    indexed = records.select("work_id").with_row_index("row")
    joined = labels.join(indexed, on="work_id", how="left").sort("work_id")
    missing = joined.filter(pl.col("row").is_null())["work_id"].to_list()
    if missing:
        raise ValueError(
            f"{len(missing)} labelled records are not behind a drawn usage, such as {missing[:3]}"
        )
    return joined


def build_training_set(
    vectors: NDArray[np.float32],
    features: NDArray[np.float64],
    labelled: pl.DataFrame,
    sources: Sequence[str],
) -> TrainingSet:
    """The labelled records' vectors, features, labels, sets and sources, by work id.

    ``labelled`` gives each labelled record's row in the other arguments (``find_labelled_rows``).
    """
    rows = labelled["row"].to_numpy()
    return TrainingSet(
        vectors=vectors[rows],
        features=features[rows],
        is_junk=labelled["is_junk"].to_numpy(),
        in_labelled_sample=labelled["in_labelled_sample"].to_numpy(),
        sources=tuple(sources[row] for row in rows),
    )


def build_scores(
    records: pl.DataFrame,
    vectors: NDArray[np.float32],
    features: NDArray[np.float64],
    final: JunkClassifier,
    labelled: pl.DataFrame,
    cross_fitted: CrossFitResult,
) -> pl.DataFrame:
    """Every record's score and call: L's held out, with its fold; the final model's elsewhere."""
    scores = final.scorer.compute_scores(vectors, features)
    predicted = scores >= final.cut_off
    folds = np.full(records.height, -1, dtype=np.int64)
    held_out = labelled["in_labelled_sample"].to_numpy()
    rows = labelled["row"].to_numpy()[held_out]
    scores[rows] = cross_fitted.scores[held_out]
    predicted[rows] = cross_fitted.predicted_junk[held_out]
    folds[rows] = cross_fitted.folds[held_out]
    return (
        records.select("work_id", *FEATURE_NAMES)
        .with_columns(
            pl.Series("score", scores),
            pl.Series("predicted_junk", predicted),
            pl.Series("fold", folds),
        )
        .with_columns(pl.when(pl.col("fold") >= 0).then("fold").alias("fold"))
    )


def build_labelled_items(
    labels: JunkLabels, words: Mapping[str, Word], usages: pl.DataFrame, scores: pl.DataFrame
) -> pl.DataFrame:
    """L's items, each with its sentence's OCR score and its record's held-out call.

    ``is_junk_usage`` is the judges' label of the usage (verdict ``junk`` or sentence ``other``),
    ``is_damaged`` their damage mark.

    Raises:
        ValueError: if an item's usage is not a drawn usage of the table.
    """
    lemmas = {word.reported_name: lemma for lemma, word in words.items()}
    items = pl.DataFrame(
        [
            (
                label.work_id,
                lemmas.get(label.word),
                label.content_key,
                label.decade,
                label.verdict == "junk" or label.sentence == "other",
                label.damaged == "yes",
            )
            for label in labels.items.values()
            if label.set == "labelled"
        ],
        schema={
            "work_id": pl.String(),
            "word": pl.String(),
            "content_key": pl.String(),
            "decade": pl.Int64(),
            "is_junk_usage": pl.Boolean(),
            "is_damaged": pl.Boolean(),
        },
        orient="row",
    )
    matched = items.join(
        usages.select("work_id", "word", "content_key", "ocr_unknown_share"),
        on=["work_id", "word", "content_key"],
        how="inner",
    )
    if matched.height != items.height:
        raise ValueError(f"{items.height - matched.height} labelled usages are not drawn usages")
    return matched.join(scores.select("work_id", "predicted_junk"), on="work_id")


def _count_records(
    scores: pl.DataFrame, disagreeing_count: int, labelled: pl.DataFrame
) -> dict[str, int]:
    """The records behind drawn usages, those labelled in each set, and those called junk."""
    in_sample = labelled.filter("in_labelled_sample")
    enrichment = labelled.filter(~pl.col("in_labelled_sample"))
    return {
        "behind_drawn_usages": scores.height,
        "called_junk": int(scores["predicted_junk"].sum()),
        "labelled_sample": in_sample.height,
        "labelled_sample_junk": int(in_sample["is_junk"].sum()),
        "enrichment": enrichment.height,
        "enrichment_junk": int(enrichment["is_junk"].sum()),
        "with_disagreeing_items": disagreeing_count,
    }


def _count_usages(usages: pl.DataFrame, cells: pl.DataFrame) -> dict[str, int]:
    """The drawn usages, those from records called junk, and those with an OCR score."""
    return {
        "drawn": usages.height,
        "from_records_called_junk": int(cells["predicted_junk"].sum()),
        "with_ocr_score": int(usages["ocr_unknown_share"].is_not_null().sum()),
    }


def _report_classifier(
    quality: ClassifierQuality,
    cross_fitted: CrossFitResult,
    final: JunkClassifier,
    records: pl.DataFrame,
    held_out: pl.DataFrame,
    items: pl.DataFrame,
) -> dict[str, Any]:
    """The cross-fitted quality against the bar, the false alarms and each decade's correction.

    ``held_out`` holds L's records with their labels and held-out calls, ``items`` L's usages.
    """
    described = held_out.join(records.select("work_id", "decade", "fields"), on="work_id")
    return {
        "sensitivity": _format_rate(quality.sensitivity),
        "specificity": _format_rate(quality.specificity),
        "bar": {"sensitivity": SENSITIVITY_BAR, "specificity": SPECIFICITY_BAR},
        "meets_bar": quality.meets_bar,
        "cut_offs_by_fold": {str(fold): cut for fold, cut in cross_fitted.cut_offs.items()},
        "final_cut_off": final.cut_off,
        "false_alarms_by_decade": {
            decade: _format_rate(rate)
            for decade, rate in compute_false_alarms(described, "decade").items()
        },
        "false_alarms_by_field": {
            field: _format_rate(rate)
            for field, rate in compute_false_alarms(described, "fields").items()
        },
        "correction_by_decade": {
            str(decade): {
                "mean": round(difference.mean, 4),
                "interval": [round(difference.low, 4), round(difference.high, 4)],
                "items": difference.items,
            }
            for decade, difference in compute_corrections(items).items()
        },
    }


def _report_ocr(items: pl.DataFrame) -> dict[str, Any]:
    """The OCR score against the judges' damage marks on L's sentences, and its best cut."""
    scored = items.filter(pl.col("ocr_unknown_share").is_not_null())
    counts = {
        "sentences": items.height,
        "without_score": items.height - scored.height,
        "damaged": int(scored["is_damaged"].sum()),
        "bar": AUC_BAR,
    }
    damaged = scored["is_damaged"].to_numpy()
    if damaged.all() or not damaged.any():
        return counts | {"testable": False}
    check = compute_ocr_check(scored["ocr_unknown_share"].to_numpy(), damaged)
    return counts | {
        "testable": True,
        "auc": round(check.auc, 4),
        "meets_bar": check.meets_bar,
        "best_cut": check.best_cut,
        "sensitivity": round(check.sensitivity, 4),
        "specificity": round(check.specificity, 4),
    }


def _report_cells(cells: pl.DataFrame, words: Mapping[str, Word]) -> dict[str, Any]:
    """The screen's counts and decade rates, the flagged cells, and the ungated cells' shares."""
    rates = cells.select("decade", "decade_rate").unique().sort("decade")
    flagged = cells.filter("flagged")
    ungated = cells.filter(~pl.col("gated"))
    return {
        "cells": cells.height,
        "gated": int(cells["gated"].sum()),
        "flagged": flagged.height,
        "decade_rates": {str(decade): round(rate, 4) for decade, rate in rates.iter_rows()},
        "flagged_cells": [
            {
                "word": words[row["word"]].reported_name,
                "decade": row["decade"],
                "drawn": row["drawn"],
                "predicted_junk": row["predicted_junk"],
                "tail_probability": float(f"{row['tail_probability']:.3g}"),
            }
            for row in flagged.iter_rows(named=True)
        ],
        "ungated_cells": [
            {"word": words[row["word"]].reported_name, "decade": row["decade"]}
            | _format_rate(compute_rate(row["predicted_junk"], row["drawn"]))
            for row in ungated.iter_rows(named=True)
        ],
    }


def _format_rate(rate: Rate) -> dict[str, Any]:
    return {
        "count": rate.count,
        "of": rate.total,
        "share": round(rate.share, 4),
        "interval": [round(rate.low, 4), round(rate.high, 4)],
    }
