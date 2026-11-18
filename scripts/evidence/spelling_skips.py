"""How many tokens the spelling map would skip as names or accented words (`docs/spellings.md`).

The author's calibration (2026-11-17) found the map rewriting names (Professor Armour, the
Technicolor Corporation) and folded French words (*centralisé* read as *centralise*). The proposed
rule leaves alone tokens tagged as proper nouns (`PROPN`) and tokens whose text carries an accent.
This script measures both over kept works, with the lemma exceptions applied as in the static view:
the mapped tokens in total, those each rule would skip, and per form the counts and two seeded
contexts of each skipped kind, so a rule that also skips good merges shows. Run from the
repository root on the run machine:
`python scripts/evidence/spelling_skips.py PROPOSAL.parquet OUT.json [PROCESSES]`.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
import spacy

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import LemmaException, load_lemma_exceptions
from interpretant.catalog.static_view import static_view_parts
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.static_view import static_tokens

SEED = 20261122
EXAMPLES = 2
KEPT_OUT = ("programme", "programmes")  # the author's census call, 2026-11-17


def draw_key(work_id: str, start: int) -> str:
    """The example draw's key."""
    return hashlib.sha256(f"{SEED}:{work_id}:{start}".encode()).hexdigest()


def measure_part(
    part: str,
    catalog: CatalogConfig,
    exceptions: tuple[LemmaException, ...],
    spellings: dict[str, str],
    kept_ids: frozenset[str],
) -> dict[str, Any]:
    """One part's mapped tokens, and per form those tagged PROPN or written with an accent."""
    vocab = spacy.blank("en").vocab
    ruler = lemma_exception_ruler(vocab, exceptions)
    totals: Counter[str] = Counter()
    by_form: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc in load_annotations(catalog, part, vocab):
        work_id = doc.user_data[WORK_ID]
        if work_id not in kept_ids:
            continue
        doc = ruler(doc)
        view = static_tokens(doc)
        by_start = {token.idx: token for token in doc}
        for word, start in zip(view.tokens, view.starts, strict=True):
            mapped = [segment for segment in word.split("-") if segment in spellings]
            if not mapped:
                continue
            token = by_start[start]
            form = mapped[0]
            name = token.pos_ == "PROPN"
            accented = any(
                unicodedata.combining(character)
                for character in unicodedata.normalize("NFKD", token.text)
            )
            totals["mapped"] += 1
            by_form[form]["mapped"] += 1
            for kind, applies in (("propn", name), ("accented", accented)):
                if not applies:
                    continue
                totals[kind] += 1
                by_form[form][kind] += 1
                left = max(0, token.idx - 80)
                context = doc.text[left : token.idx + len(token.text) + 80].replace("\n", " ")
                examples[f"{form}|{kind}"].append((draw_key(work_id, start), context))
            if name or accented:
                totals["either"] += 1
                by_form[form]["either"] += 1
    kept_examples = {key: sorted(found)[:EXAMPLES] for key, found in examples.items()}
    return {
        "totals": dict(totals),
        "by_form": {form: dict(counts) for form, counts in by_form.items()},
        "examples": kept_examples,
    }


def main() -> None:
    """Measure every part in parallel and write the report."""
    proposal = pl.read_parquet(sys.argv[1]).filter(~pl.col("form").is_in(KEPT_OUT))
    out = Path(sys.argv[2])
    processes = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, multiprocessing.cpu_count() - 2)
    spellings = dict(zip(proposal["form"], proposal["standard"], strict=True))
    catalog = read_catalog()
    exceptions = tuple(load_lemma_exceptions(catalog))
    kept_ids = frozenset(
        load_abstracts(catalog)
        .select("work_id", "dropped_reason")
        .filter(pl.col("dropped_reason").is_null())
        .collect()["work_id"]
    )
    parts = static_view_parts(catalog)
    totals: Counter[str] = Counter()
    by_form: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    began = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(processes, mp_context=context) as pool:
        futures = [
            pool.submit(measure_part, part, catalog, exceptions, spellings, kept_ids)
            for part in parts
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            totals.update(result["totals"])
            for form, counts in result["by_form"].items():
                by_form[form].update(counts)
            for key, found in result["examples"].items():
                examples[key] = sorted([*examples[key], *map(tuple, found)])[:EXAMPLES]
            elapsed = time.monotonic() - began
            remaining = elapsed / done * (len(parts) - done)
            sys.stdout.write(
                f"parts {done}/{len(parts)} ({100 * done / len(parts):.0f}%), "
                f"{done / elapsed * 60:.1f} parts/min, about {remaining / 60:.1f} min left\n"
            )
            sys.stdout.flush()
    report = {
        "totals": dict(totals),
        "by_form": {
            form: dict(counts)
            for form, counts in sorted(by_form.items(), key=lambda item: -item[1]["mapped"])
        },
        "examples": {key: [context for _, context in found] for key, found in examples.items()},
    }
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    sys.stdout.write(f"{json.dumps(report['totals'])}\nwritten to {out}\n")


if __name__ == "__main__":
    main()
