"""The compound audit's items: each pair of the sample and the census with corpus sentences.

`docs/compounds.md` fixes the audit. Every pair of the sample and of the census (each form once)
becomes a blind item: the form, the standard, and up to two sentences from kept works for each,
the occurrence marked by its offsets in the shown text. A spaced form is drawn only from
occurrences that pass the view's noun-compound gate. Occurrences are drawn by the smallest sha256
of (seed, work id, offset), so the draw does not depend on part order. The items are shuffled
(seed 20261123) into batches for the agent judges; key.json maps each blind id to its pair and its
sets; calibration.json holds the 30 sample pairs the author judges (seed 20261124). Run from the
repository root on the run machine: `python scripts/review/compound_audit.py REVIEW_DIR OUT
[PROCESSES]`, REVIEW_DIR holding the builder's sample.parquet and census.parquet.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import polars as pl
import spacy
from rich.console import Console

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import LemmaException, load_lemma_exceptions
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.static_view import load_static_view, static_view_parts
from interpretant.cli.progress import LiveProgress
from interpretant.corpus.compound_spelling import NO_COMPOUNDS, is_noun_compound
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.spelling import has_accent
from interpretant.corpus.static_view import static_tokens
from interpretant.pipelines.workers import map_parts

SEED = 20261123
CALIBRATION_SEED = 20261124
CALIBRATION = 30
SENTENCES = 2
BATCH = 125
WIDTH = 300
# An occurrence: its draw key, work id, and character span.
Occurrence = tuple[str, str, int, int]


@dataclass(frozen=True)
class PartJob:
    """What a worker needs for one part (picklable)."""

    catalog: CatalogConfig
    exceptions: tuple[LemmaException, ...]
    spellings: dict[str, str]
    one_token: frozenset[str]
    spaced: frozenset[str]
    removals: dict[str, list[tuple[int, int]]]


def key(work_id: str, start: int) -> str:
    """The occurrence draw's key."""
    return hashlib.sha256(f"{SEED}:{work_id}:{start}".encode()).hexdigest()


def find_occurrences(part: str, job: PartJob) -> dict[str, list[Occurrence]]:
    """The part's best occurrences of each wanted form, a few per form."""
    vocab = spacy.blank("en").vocab
    ruler = lemma_exception_ruler(vocab, job.exceptions)
    best: dict[str, list[Occurrence]] = {}

    def keep(form: str, work_id: str, start: int, end: int) -> None:
        found = best.setdefault(form, [])
        found.append((key(work_id, start), work_id, start, end))
        if len(found) > 8 * SENTENCES:
            best[form] = sorted(found)[:SENTENCES]

    for doc in load_annotations(job.catalog, part, vocab):
        work_id = doc.user_data[WORK_ID]
        if work_id not in job.removals:
            continue
        removals = job.removals[work_id]
        doc = ruler(doc)
        view = static_tokens(doc, job.spellings, NO_COMPOUNDS)
        by_start = {token.idx: token for token in doc}
        entries = [
            (by_start[start], word)
            for word, start in zip(view.tokens, view.starts, strict=True)
            if not any(low <= start < high for low, high in removals)
            and not has_accent(by_start[start].text)
        ]
        for token, word in entries:
            if word in job.one_token:
                keep(word, work_id, token.idx, token.idx + len(token.text))
        for (first, left), (second, right) in pairwise(entries):
            form = f"{left} {right}"
            if form in job.spaced and is_noun_compound(first, second):
                keep(form, work_id, first.idx, second.idx + len(second.text))
    return {form: sorted(found)[:SENTENCES] for form, found in best.items()}


def window(sentence: str, start: int, end: int) -> dict[str, object]:
    """The sentence, or a window of it around the occurrence, with the occurrence's offsets."""
    if len(sentence) <= WIDTH:
        stripped = sentence.lstrip()
        shift = len(sentence) - len(stripped)
        return {"text": stripped.rstrip(), "mark": [start - shift, end - shift]}
    begin = max(0, min((start + end) // 2 - WIDTH // 2, len(sentence) - WIDTH))
    prefix = "…" if begin else ""
    suffix = "…" if begin + WIDTH < len(sentence) else ""
    shown = prefix + sentence[begin : begin + WIDTH] + suffix
    offset = len(prefix) - begin
    return {"text": shown, "mark": [start + offset, end + offset]}


def main() -> None:
    """Build the items, the batches, the key and the calibration set."""
    review, out = Path(sys.argv[1]), Path(sys.argv[2])
    processes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    out.mkdir(parents=True, exist_ok=True)
    sample = pl.read_parquet(review / "sample.parquet")
    census = pl.read_parquet(review / "census.parquet")
    pairs: dict[str, dict[str, object]] = {}
    for row in sample.iter_rows(named=True):
        pairs[row["form"]] = {
            "form": row["form"],
            "standard": row["standard"],
            "kind": row["kind"],
            "sets": ["sample"],
            "drawn": row["drawn"],
        }
    for row in census.iter_rows(named=True):
        entry = pairs.setdefault(
            row["form"],
            {
                "form": row["form"],
                "standard": row["standard"],
                "kind": row["kind"],
                "sets": [],
                "drawn": None,
            },
        )
        entry["sets"] = [*entry["sets"], f"census:{row['reason']}"]  # type: ignore[misc]
    needed = {str(p["form"]) for p in pairs.values()} | {str(p["standard"]) for p in pairs.values()}

    catalog = read_catalog()
    kept = (
        load_abstracts(catalog)
        .select("work_id", "text", "sentences", "removals", "dropped_reason")
        .filter(pl.col("dropped_reason").is_null())
        .drop("dropped_reason")
        .collect()
    )
    removals = {
        work_id: [(r["start"], r["end"]) for r in ranges]
        for work_id, ranges in kept.select("work_id", "removals").iter_rows()
    }
    spellings = {spelling.form: spelling.standard for spelling in load_spellings(catalog)}
    exceptions = tuple(load_lemma_exceptions(catalog))
    one_token = frozenset(form for form in needed if " " not in form)
    spaced = frozenset(form for form in needed if " " in form)
    jobs = {}
    for part in static_view_parts(catalog):
        ids = load_static_view(catalog, part, columns=["work_id"])["work_id"]
        own = {work_id: removals[work_id] for work_id in ids if work_id in removals}
        jobs[part] = PartJob(catalog, exceptions, spellings, one_token, spaced, own)
    with LiveProgress(Console()) as live:
        live.stage("finding occurrences", total=len(jobs))
        results = map_parts(find_occurrences, jobs, process_count=processes, progress=live)
    chosen: dict[str, list[Occurrence]] = {}
    for found in results.values():
        for form, occurrences in found.items():
            chosen[form] = sorted(chosen.get(form, []) + occurrences)[:SENTENCES]
    wanted = {work_id for found in chosen.values() for _, work_id, _, _ in found}
    texts = {
        row[0]: (row[1], row[2])
        for row in kept.filter(pl.col("work_id").is_in(sorted(wanted)))
        .select("work_id", "text", "sentences")
        .iter_rows()
    }

    def sentences_of(form: str) -> list[dict[str, object]]:
        shown = []
        for _, work_id, start, end in chosen.get(form, []):
            text, bounds = texts[work_id]
            sentence = next((s for s in bounds if s["start"] <= start < s["end"]), None)
            if sentence is not None:
                body = text[sentence["start"] : sentence["end"]]
                low = start - sentence["start"]
                shown.append(window(body, low, min(end - sentence["start"], len(body))))
        return shown

    rng = random.Random(SEED)
    order = sorted(pairs)
    rng.shuffle(order)
    items, key_map = [], {}
    for index, form in enumerate(order):
        entry = pairs[form]
        item_id = f"C{index:03d}"
        key_map[item_id] = entry
        items.append(
            {
                "id": item_id,
                "form": form,
                "standard": entry["standard"],
                "form_sentences": sentences_of(form),
                "standard_sentences": sentences_of(str(entry["standard"])),
            }
        )
    for batch, start in enumerate(range(0, len(items), BATCH)):
        with (out / f"batch_{batch:02d}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items[start : start + BATCH]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "key.json").write_text(json.dumps(key_map, indent=1, ensure_ascii=False))
    sample_ids = sorted(
        item_id
        for item_id, entry in key_map.items()
        if "sample" in entry["sets"]  # type: ignore[operator]
    )
    calibration = random.Random(CALIBRATION_SEED).sample(sample_ids, CALIBRATION)
    by_id = {item["id"]: item for item in items}
    (out / "calibration.json").write_text(
        json.dumps(
            [by_id[item_id] for item_id in sorted(calibration)], indent=1, ensure_ascii=False
        )
    )
    missing = sum(1 for item in items if not item["form_sentences"])
    sys.stdout.write(
        f"{len(items)} items in {(len(items) + BATCH - 1) // BATCH} batches; "
        f"{len(sample_ids)} from the sample; {CALIBRATION} for calibration; "
        f"{missing} forms without a sentence\n"
    )


if __name__ == "__main__":
    main()
