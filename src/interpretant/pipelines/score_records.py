"""Every record behind a drawn usage scored by two decision models (`docs/junk_gate.md`).

The junk gate's call since the deviation of 2026-11-28. Each record is read as the gate shows it,
its kept text cut for length with no sentence marked (``review.junk_gate.render_record``), and
asked the codebook's own-prose question alone (``judge_ollama.ask_not_own_prose``).

- **The screening model** scores every record.
- **The confirming model** scores L's and E's records and every record whose screening probability
  reaches the lowest of the screening model's cut-offs on L (``review.junk_calls.set_cut_offs``):
  no other record can be called junk.
- **One model at a time.** The screening model is unloaded before the confirming model loads, so
  neither spills out of the graphics card.
- **Resuming.** A pass scores only the records missing from its parts, in chunks, adding each chunk
  as a part, so a stopped run resumes: the stage's output persists between runs.
- **Failures.** A request that fails or answers badly is sent once more; a record that still fails
  is left out, and the pass ends with an error after its other records, so the next run retries
  only those.
- **The manifest.** Each pass keeps what its scores depend on (the model, the codebook's sha256,
  the question and the rendering). A different one stops the stage, which never deletes scores.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog.abstracts import abstracts_parts, load_abstracts_part
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.junk_gate import load_junk_labels
from interpretant.catalog.record_decisions import (
    Pass,
    load_decisions,
    read_pass_manifest,
    save_decision_part,
    write_pass_manifest,
)
from interpretant.catalog.staging import load_staging_table
from interpretant.catalog.usages import load_usages_part
from interpretant.pipelines.judge_ollama import (
    OWN_PROSE_QUESTION,
    Clock,
    Transport,
    ask_not_own_prose,
    build_own_prose_question,
    generate_item_answer,
    unload_model,
)
from interpretant.pipelines.params import ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.junk_calls import (
    build_folds,
    build_record_labels,
    build_sources,
    set_cut_offs,
)
from interpretant.review.junk_gate import parse_removals, render_record

# What a record's text is: kept, cut to its head and tail when long, no sentence marked
RENDERING = "review.junk_gate.render_record, no sentence marked"


class DecisionSettings(ParamsSettings):
    """The ``junk_gate.decision`` section of ``params.yaml``: the two models and their cut-offs."""

    model_config = SettingsConfigDict(yaml_config_section="junk_gate.decision")

    codebook: Path
    screen_model: str
    confirm_model: str
    specificity_target: float = Field(gt=0, le=1)
    folds: int = Field(ge=2)
    keep_alive: str
    chunk_size: int = Field(gt=0)


class ScoringIncompleteError(RuntimeError):
    """Some records failed twice; the next run scores only those."""


@dataclass(frozen=True)
class PassResult:
    """What one pass did: the records it needed, those it scored now, and those that failed."""

    needed: int
    scored: int
    failed: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreRecordsResult:
    """Both passes, and the screening probability under which no record is confirmed."""

    screen: PassResult
    confirm: PassResult
    lowest_screen_cut_off: float | None


def find_records(catalog: CatalogConfig, *, limit: int | None = None) -> dict[str, list[str]]:
    """Each part's records behind drawn usages, sorted, each record in its first part only.

    ``limit`` keeps only the first records, in part then work-id order, for a trial.

    Raises:
        FileNotFoundError: if the abstracts table has no part, or a usages part is missing.
    """
    parts = abstracts_parts(catalog)
    if not parts:
        raise FileNotFoundError(f"no part of the abstracts table under {catalog.abstracts.path}")
    records: dict[str, list[str]] = {}
    seen: set[str] = set()
    for part in parts:
        drawn = load_usages_part(catalog, part, ["work_id", "drawn"]).filter("drawn")
        new = [work for work in drawn["work_id"].unique().sort().to_list() if work not in seen]
        if limit is not None:
            new = new[: max(0, limit - len(seen))]
        records[part] = new
        seen.update(new)
    return records


def score_records(
    catalog: CatalogConfig,
    settings: DecisionSettings,
    transport: Transport,
    *,
    root: Path,
    limit: int | None = None,
    progress: Progress = SILENT,
    clock: Clock = time.monotonic,
) -> ScoreRecordsResult:
    """Score every record with the screening model, then the confirming model's records.

    With ``limit`` (a trial), only the first records are scored, by both models, and no cut-off
    is set.

    Raises:
        ValueError: if a pass's manifest differs from the settings, or one of L's records has no
            screening score.
        ScoringIncompleteError: if a record failed twice; the other records are saved.
    """
    codebook = settings.codebook.read_text(encoding="utf-8")
    questions = build_own_prose_question(codebook)
    records = find_records(catalog, limit=limit)
    screen = score_pass(
        catalog,
        "screen",
        settings.screen_model,
        records,
        settings,
        transport,
        questions,
        root=root,
        codebook=codebook,
        progress=progress,
        clock=clock,
    )
    if screen.failed:
        raise ScoringIncompleteError(f"{len(screen.failed)} records failed the screening pass")
    if limit is None:
        labelled = labelled_records(catalog)
        screen_scores = load_decisions(root, "screen")
        lowest = find_lowest_screen_cut_off(labelled, screen_scores, settings)
        to_confirm = select_for_confirming(records, screen_scores, lowest, labelled)
    else:
        lowest, to_confirm = None, records
    confirm = score_pass(
        catalog,
        "confirm",
        settings.confirm_model,
        to_confirm,
        settings,
        transport,
        questions,
        root=root,
        codebook=codebook,
        progress=progress,
        clock=clock,
    )
    if confirm.failed:
        raise ScoringIncompleteError(f"{len(confirm.failed)} records failed the confirming pass")
    return ScoreRecordsResult(screen=screen, confirm=confirm, lowest_screen_cut_off=lowest)


def score_pass(
    catalog: CatalogConfig,
    pass_name: Pass,
    model: str,
    records: Mapping[str, Sequence[str]],
    settings: DecisionSettings,
    transport: Transport,
    questions: dict[str, Any],
    *,
    root: Path,
    codebook: str,
    progress: Progress,
    clock: Clock,
) -> PassResult:
    """Score the records the pass lacks, part by part, a chunk at a time, then unload the model.

    Raises:
        ValueError: if the pass's manifest differs from this pass's.
    """
    check_manifest(root, pass_name, build_manifest(model, codebook))
    done = set(load_decisions(root, pass_name)["work_id"].to_list())
    missing = {part: [w for w in works if w not in done] for part, works in records.items()}
    needed = sum(len(works) for works in records.values())
    progress.stage(f"{pass_name}: {model}", total=sum(len(works) for works in missing.values()))

    def judge(item: dict[str, Any]) -> dict[str, Any]:
        probability = ask_not_own_prose(
            transport, model, item["text"], questions, settings.keep_alive
        )
        return {"probability": probability}

    scored, failed = 0, {}
    for part, works in missing.items():
        if not works:
            continue
        texts = render_texts(catalog, part, works)
        for start in range(0, len(works), settings.chunk_size):
            rows = []
            for work_id in works[start : start + settings.chunk_size]:
                answer, _, problem = generate_item_answer(
                    {"id": work_id, "text": texts[work_id]}, judge, clock
                )
                if answer is None:
                    failed[work_id] = problem
                else:
                    rows.append((work_id, answer["probability"]))
                progress.advance()
            if rows:
                chunk = pl.DataFrame(rows, schema=["work_id", "probability"], orient="row")
                save_decision_part(root, pass_name, chunk)
                scored += len(rows)
    if scored or failed:
        unload_model(transport, model)
    return PassResult(needed=needed, scored=scored, failed=failed)


def build_manifest(model: str, codebook: str) -> dict[str, str]:
    """What a pass's scores depend on."""
    return {
        "model": model,
        "codebook_sha256": hashlib.sha256(codebook.encode()).hexdigest(),
        "question": OWN_PROSE_QUESTION,
        "rendering": RENDERING,
    }


def check_manifest(root: Path, pass_name: Pass, manifest: dict[str, str]) -> None:
    """Write the pass's manifest if it has none; refuse one that differs.

    Raises:
        ValueError: if the saved manifest differs, naming what changed.
    """
    saved = read_pass_manifest(root, pass_name)
    if not saved:
        write_pass_manifest(root, pass_name, manifest)
        return
    changed = sorted(
        key for key in manifest.keys() | saved.keys() if manifest.get(key) != saved.get(key)
    )
    if changed:
        raise ValueError(
            f"the {pass_name} pass's saved scores were made with another {', '.join(changed)}; "
            f"move {root / pass_name} aside to score again"
        )


def render_texts(catalog: CatalogConfig, part: str, works: Sequence[str]) -> dict[str, str]:
    """Each work's text as the gate shows it, from one part of the abstracts table.

    Raises:
        ValueError: if a work is not in the part.
    """
    abstracts = load_abstracts_part(catalog, part, ["work_id", "text", "removals"]).filter(
        pl.col("work_id").is_in(list(works))
    )
    texts = {
        work_id: render_record(text, parse_removals(removals))
        for work_id, text, removals in abstracts.iter_rows()
    }
    if len(texts) != len(works):
        raise ValueError(f"{len(works) - len(texts)} records of part {part} have no text")
    return texts


def labelled_records(catalog: CatalogConfig) -> pl.DataFrame:
    """L's and E's records with their labels, set and source, sorted by work id."""
    labels = build_record_labels(load_junk_labels(catalog)).table
    staged = load_staging_table(catalog, "openalex").select("work_id", "issn_l", "source_name")
    with_sources = labels.join(staged.collect(), on="work_id", how="left").sort("work_id")
    return labels.with_columns(pl.Series("source", build_sources(with_sources)))


def find_lowest_screen_cut_off(
    labelled: pl.DataFrame, screen: pl.DataFrame, settings: DecisionSettings
) -> float:
    """The lowest of the screening model's cut-offs on L: each fold's and the final one.

    ``labelled`` holds L's and E's records (``labelled_records``).

    Raises:
        ValueError: if one of L's records has no screening score.
    """
    joined = labelled.filter("in_labelled_sample").join(screen, on="work_id", how="left")
    if joined["probability"].null_count():
        raise ValueError(f"{joined['probability'].null_count()} of L's records have no score")
    folds = build_folds(joined["source"].to_list(), folds=settings.folds, depth=0)
    cut_offs = set_cut_offs(
        joined["probability"].to_numpy(),
        joined["is_junk"].to_numpy(),
        folds,
        fold_count=settings.folds,
        target=settings.specificity_target,
    )
    return cut_offs.lowest


def select_for_confirming(
    records: Mapping[str, Sequence[str]],
    screen: pl.DataFrame,
    lowest_cut_off: float,
    labelled: pl.DataFrame,
) -> dict[str, list[str]]:
    """Each part's records for the confirming model: L's, E's and those at the lowest cut-off."""
    reaching = screen.filter(pl.col("probability") >= lowest_cut_off)["work_id"].to_list()
    wanted = set(reaching) | set(labelled["work_id"].to_list())
    return {part: [w for w in works if w in wanted] for part, works in records.items()}
