"""The small model against the transformer on sampled usages (agenda 4.9; `docs/usages_audit.md`).

Draws 300 usages per word from the usages table, spread evenly over decades (seed 20261302). Worker
processes read each sampled usage's sentence tokens from the cached documents and recognise the
small model's entities on the whole abstract too, in batches (the sentence context check). The
transformer then re-annotates every sampled sentence on the GPU
(`usages/entities.load_transformer`, sentences sorted by length), followed by the lemma
exceptions, and each usage is matched and scoped again under the same rules. A usage flips if the
transformer's lemma stops matching it (whole tokens only) or its scope changes. Writes
`summary.json` (counts only, no sentence; held-out words named by a hash), the 50
disagreements to judge (seed 20261303, readings shown as A and B in an order set by the key) and
their key. Run from the repository root on the GPU machine:
`python scripts/evidence/model_comparison.py OUT_DIR [PROCESSES]`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console
from spacy.tokens import Doc
from thinc.api import require_gpu, set_gpu_allocator

from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import load_lemma_exceptions
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.words import Word, load_words
from interpretant.cli.progress import LiveProgress
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.pipelines.build_usages import UsagesSettings
from interpretant.pipelines.workers import map_parts
from interpretant.usages.entities import (
    entity_label,
    load_entity_recognizer,
    load_transformer,
    recognise_sentence_entities,
)
from interpretant.usages.identifiers import build_draw_key
from interpretant.usages.target_matching import build_target_matcher, match_targets
from interpretant.usages.usage_selection import is_in_scope

SAMPLE_SEED = 20261302
DISAGREEMENT_SEED = 20261303
PER_WORD = 300
DISAGREEMENTS = 50
COLUMNS = [
    "work_id",
    "word",
    "content_key",
    "sentence_start",
    "sentence_end",
    "target_start",
    "target_end",
    "position",
    "pos",
    "lemma",
    "entity",
    "in_scope",
]


def allocate(available: dict[int, int], total: int) -> dict[int, int]:
    """An equal share per decade; a decade short of its share gives all it has, the rest spread."""
    shares = {}
    remaining = total
    by_size = sorted(available, key=lambda decade: (available[decade], decade))
    for index, decade in enumerate(by_size):
        share = min(available[decade], remaining // (len(by_size) - index))
        shares[decade] = share
        remaining -= share
    return shares


def draw_sample(catalog: CatalogConfig) -> pl.DataFrame:
    """300 usages per word, spread evenly over decades, the smallest sample keys in each decade."""
    frames = [
        pl.read_parquet(path, columns=COLUMNS).with_columns(
            pl.lit(path.stem).alias("part"),
            pl.lit(int(path.stem.rsplit("_", 1)[1])).alias("decade"),
        )
        for path in sorted(catalog.usages.path.glob("*.parquet"))
    ]
    usages = pl.concat(frames)
    keyed = usages.with_columns(
        pl.struct("work_id", "word", "content_key")
        .map_elements(
            lambda row: build_draw_key(
                SAMPLE_SEED, row["work_id"], row["word"], row["content_key"]
            ),
            return_dtype=pl.String,
        )
        .alias("sample_key")
    )
    chosen = []
    for (word,), rows in keyed.group_by("word"):
        counts = dict(rows.group_by("decade").len().iter_rows())
        for decade, share in allocate(counts, PER_WORD).items():
            in_decade = rows.filter(pl.col("decade") == decade).sort("sample_key")
            chosen.append(in_decade.head(share))
        del word
    return pl.concat(chosen)


@dataclass(frozen=True)
class PartJob:
    """The sampled usages of one part (picklable)."""

    catalog: CatalogConfig
    rows: tuple[dict[str, Any], ...]


def read_part(part: str, job: PartJob) -> list[dict[str, Any]]:
    """Each sampled usage with its sentence's tokens and the abstract's entity at the target."""
    recognizer = load_entity_recognizer()
    wanted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in job.rows:
        wanted[row["work_id"]].append(row)
    read: list[tuple[Doc, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for doc in load_annotations(job.catalog, part, recognizer.vocab):
        rows = wanted.get(doc.user_data[WORK_ID])
        if not rows:
            continue
        sentences = []
        for row in rows:
            span = doc.char_span(
                row["sentence_start"], row["sentence_end"], alignment_mode="expand"
            )
            if span is None:
                raise ValueError(f"{row['work_id']}: a sampled sentence has no tokens")
            sentences.append(
                {
                    "words": [token.text for token in span],
                    "spaces": [bool(token.whitespace_) for token in span],
                    "offset": span.start_char,
                    "sentence": doc.text[row["sentence_start"] : row["sentence_end"]],
                }
            )
        read.append((doc, rows, sentences))
    abstracts = recognise_sentence_entities(
        (doc[:] for doc, _, _ in read), recognizer, batch_size=64
    )
    found = []
    for (_, rows, sentences), abstract in zip(read, abstracts, strict=True):
        for row, sentence in zip(rows, sentences, strict=True):
            entity = entity_label(abstract, row["target_start"], row["target_end"])
            found.append(row | sentence | {"abstract_entity": entity})
    return found


def reannotate(
    records: list[dict[str, Any]],
    words: dict[str, Word],
    catalog: CatalogConfig,
    live: LiveProgress,
) -> None:
    """Add the transformer's reading, match and scope to each record, on the GPU."""
    set_gpu_allocator("pytorch")
    require_gpu(0)
    transformer = load_transformer()
    exceptions = lemma_exception_ruler(transformer.vocab, load_lemma_exceptions(catalog))
    matcher = build_target_matcher(
        words.values(), {s.form: s.standard for s in load_spellings(catalog)}
    )
    name_labels = UsagesSettings().name_labels
    order = sorted(range(len(records)), key=lambda index: len(records[index]["words"]))
    docs = (
        Doc(transformer.vocab, words=records[i]["words"], spaces=records[i]["spaces"])
        for i in order
    )
    live.stage("annotating with the transformer", total=len(records))
    for index, doc in zip(order, transformer.pipe(docs, batch_size=64), strict=True):
        record = records[index]
        doc = exceptions(doc)
        start = record["target_start"] - record["offset"]
        end = record["target_end"] - record["offset"]
        span = doc.char_span(start, end, alignment_mode="expand")
        if span is None:
            raise ValueError(f"{record['work_id']}: the target has no token")
        head = span[-1]
        hyphen_part = record["position"] is not None
        matched = hyphen_part or any(
            mention.word == record["word"]
            and mention.start_char == start
            and mention.end_char == end
            for mention in match_targets(doc, matcher)
        )
        entity = entity_label(doc, start, end)
        record |= {
            "trf_pos": head.pos_,
            "trf_lemma": head.lemma_,
            "trf_entity": entity,
            "trf_matched": matched,
            "trf_in_scope": matched
            and is_in_scope(
                head.pos_, words[record["word"]], entity, name_labels, hyphen_part=hyphen_part
            ),
        }
        live.advance()


def summarise(records: list[dict[str, Any]], words: dict[str, Word]) -> dict[str, Any]:
    """The flip shares pooled, per word (held-out words by their hash) and per decade."""
    frame = pl.DataFrame(
        [
            {
                "word": words[r["word"]].reported_name,
                "decade": r["decade"],
                "match_flip": not r["trf_matched"],
                "scope_flip": r["trf_matched"] and r["trf_in_scope"] != r["in_scope"],
                "lemma_differs": r["trf_lemma"] != r["lemma"],
                "abstract_agrees": r["abstract_entity"] == r["entity"],
                "in_scope": r["in_scope"],
                "trf_in_scope": r["trf_in_scope"],
            }
            for r in records
        ]
    ).with_columns((pl.col("match_flip") | pl.col("scope_flip")).alias("flip"))
    shares = [pl.col(name).mean().round(4) for name in ("flip", "match_flip", "scope_flip")]
    by_word = (
        frame.group_by("word")
        .agg(pl.len().alias("usages"), *shares)
        .sort("flip", descending=True)
    )
    by_decade = frame.group_by("decade").agg(pl.len().alias("usages"), *shares).sort("decade")
    return {
        "usages": frame.height,
        "words": frame["word"].n_unique(),
        "held_out_words": sum(name.startswith("held-out:") for name in set(frame["word"])),
        "flip_share": round(frame.select(pl.col("flip").mean()).item(), 4),
        "match_flips": int(frame["match_flip"].sum()),
        "scope_flips": int(frame["scope_flip"].sum()),
        "into_scope": int((frame["scope_flip"] & frame["trf_in_scope"]).sum()),
        "out_of_scope": int((frame["scope_flip"] & frame["in_scope"]).sum()),
        "raw_lemma_differences": int(frame["lemma_differs"].sum()),
        "abstract_entity_agreement": round(
            frame.select(pl.col("abstract_agrees").mean()).item(), 5
        ),
        "by_word": by_word.to_dicts(),
        "by_decade": by_decade.to_dicts(),
    }


def marked(record: dict[str, Any]) -> str:
    """The sentence with the target between ⟨t⟩ and ⟨/t⟩."""
    sentence, base = record["sentence"], record["sentence_start"]
    start, end = record["target_start"] - base, record["target_end"] - base
    return f"{sentence[:start]}⟨t⟩{sentence[start:end]}⟨/t⟩{sentence[end:]}"


def disagreements(
    records: list[dict[str, Any]], words: dict[str, Word]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 50 judged disagreements, A and B in an order set by each one's key, and the key."""
    flipped = [
        r
        for r in records
        if not r["trf_matched"] or r["trf_in_scope"] != r["in_scope"]
    ]
    keyed = sorted(
        flipped,
        key=lambda r: build_draw_key(DISAGREEMENT_SEED, r["work_id"], r["word"], r["content_key"]),
    )[:DISAGREEMENTS]
    items, key = [], {}
    for number, record in enumerate(keyed, start=1):
        item_id = f"m{number:02d}"
        draw_key = build_draw_key(
            DISAGREEMENT_SEED, record["work_id"], record["word"], record["content_key"]
        )
        small = {"pos": record["pos"], "lemma": record["lemma"], "name": record["entity"]}
        transformer = {
            "pos": record["trf_pos"],
            "lemma": record["trf_lemma"],
            "name": record["trf_entity"],
        }
        transformer_first = int(draw_key[-1], 16) % 2 == 0
        first, second = (transformer, small) if transformer_first else (small, transformer)
        items.append(
            {
                "id": item_id,
                "word": record["word"],
                "parts_of_speech": list(words[record["word"]].pos),
                "sentence": marked(record),
                "A": first,
                "B": second,
            }
        )
        key[item_id] = {
            "A": "transformer" if transformer_first else "small",
            "work_id": record["work_id"],
            "word": words[record["word"]].reported_name,
            "content_key": record["content_key"],
            "kind": "match" if not record["trf_matched"] else "scope",
        }
    return items, key


def main() -> None:
    """Draw the sample, read it in workers, re-annotate it on the GPU, and write the results."""
    out = Path(sys.argv[1])
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    words = {word.lemma: word for word in load_words(catalog)}
    sample = draw_sample(catalog)
    jobs = {
        part: PartJob(catalog, tuple(rows.drop("part").to_dicts()))
        for (part,), rows in sample.group_by("part")
    }
    with LiveProgress(Console()) as live:
        live.stage("reading the sampled sentences", total=len(jobs))
        results = map_parts(read_part, jobs, process_count=processes, progress=live)
        records = [record for part in sorted(results) for record in results[part]]
        reannotate(records, words, catalog, live)
    summary = summarise(records, words)
    items, key = disagreements(records, words)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    with (out / "items.jsonl").open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    kinds = Counter(entry["kind"] for entry in key.values())
    sys.stdout.write(
        f"{summary['usages']} usages, {summary['words']} words: flips {summary['flip_share']:.2%} "
        f"(match {summary['match_flips']}, scope {summary['scope_flips']}); abstract entities "
        f"agree {summary['abstract_entity_agreement']:.3%}; {len(items)} disagreements "
        f"to judge ({dict(kinds)})\n"
    )


if __name__ == "__main__":
    main()
