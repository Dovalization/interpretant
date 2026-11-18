"""How the static view writes the target words, without and with the lemma exceptions (3.6).

Over the kept text of the build of record (kept works, removed ranges skipped), each cached
document's static tokens are taken twice: as spaCy lemmatised them (view A of round L1) and after
the lemma exceptions' attribute ruler (view B). For each single-word target it counts:

- tokens of the target's own spelling that view B writes as another word (expected none: a target
  word must not be mapped away, `docs/lemma_review.md` rule 3);
- the forms each view writes as the target, so the forms entering and leaving it show.

Corpus counts only; no drift is computed. Writes OUT/target_lemmas.json and prints the summary.
Run from the repository root on the run machine:
`python scripts/evidence/target_lemmas.py OUT [PROCESSES]`.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import read_catalog
from interpretant.catalog.lemma_exceptions import LemmaException, load_lemma_exceptions
from interpretant.catalog.static_view import load_static_view, static_view_parts
from interpretant.corpus.audit import load_target_words
from interpretant.corpus.compound_spelling import NO_COMPOUNDS
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.static_view import fold, static_tokens

TARGETS = Path("data/words_of_interest.txt")
Ranges = dict[str, list[tuple[int, int]]]
# (target, written as, tag) for a target's own spelling written as another word.
Away = Counter[tuple[str, str, str]]
# (target, view, form) for every token a view writes as a target.
Written = Counter[tuple[str, str, str]]


def count_part(
    part: str, removals: Ranges, exceptions: list[LemmaException], targets: frozenset[str]
) -> tuple[Away, Written]:
    """One part's counts: the targets' own spellings written away, and the forms written as them."""
    import spacy

    vocab = spacy.blank("en").vocab
    ruler = lemma_exception_ruler(vocab, exceptions)
    away: Away = Counter()
    written: Written = Counter()
    for doc in load_annotations(read_catalog(), part, vocab):
        ranges = removals.get(doc.user_data[WORK_ID])
        if ranges is None:
            continue
        without = static_tokens(doc, {}, NO_COMPOUNDS)
        with_exceptions = static_tokens(ruler(doc), {}, NO_COMPOUNDS)
        by_start = {token.idx: token for token in doc}
        for start, a, b in zip(without.starts, without.tokens, with_exceptions.tokens, strict=True):
            if any(low <= start < high for low, high in ranges):
                continue
            token = by_start[start]
            form = fold(token.text)
            if form in targets and b != form:
                away[(form, b, token.tag_)] += 1
            if a in targets:
                written[(a, "A", form)] += 1
            if b in targets:
                written[(b, "B", form)] += 1
    return away, written


def main() -> None:
    """Count every part on a spawn pool, then write and print the summary."""
    out = Path(sys.argv[1])
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    kept = load_abstracts(catalog).select("work_id", "removals", "dropped_reason").collect()
    kept = kept.filter(pl.col("dropped_reason").is_null())
    removals = {
        work_id: [(r["start"], r["end"]) for r in rows]
        for work_id, rows in kept.select("work_id", "removals").iter_rows()
    }
    exceptions = load_lemma_exceptions(catalog)
    targets = frozenset(word for word in load_target_words(TARGETS) if "_" not in word)
    parts = static_view_parts(catalog)
    away: Away = Counter()
    written: Written = Counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=processes, mp_context=context) as pool:
        futures = []
        for part in parts:
            work_ids = load_static_view(catalog, part, columns=["work_id"])["work_id"]
            own = {work_id: removals[work_id] for work_id in work_ids if work_id in removals}
            futures.append(pool.submit(count_part, part, own, exceptions, targets))
        for done, future in enumerate(as_completed(futures), start=1):
            part_away, part_written = future.result()
            away.update(part_away)
            written.update(part_written)
            sys.stdout.write(f"counting the target words: {done}/{len(parts)} parts\n")
            sys.stdout.flush()

    summary: dict[str, dict[str, Any]] = {}
    for target in sorted(targets):
        forms = {
            view: Counter({f: n for (t, v, f), n in written.items() if t == target and v == view})
            for view in ("A", "B")
        }
        summary[target] = {
            "tokens_a": sum(forms["A"].values()),
            "tokens_b": sum(forms["B"].values()),
            "own_spelling_written_away": [
                [as_written, tag, n] for (t, as_written, tag), n in away.items() if t == target
            ],
            "forms_leaving": (forms["A"] - forms["B"]).most_common(),
            "forms_entering": (forms["B"] - forms["A"]).most_common(),
        }
    (out / "target_lemmas.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    lines = ["target: tokens in A -> B; own spelling written away; leaving; entering"]
    for target, row in summary.items():
        lines.append(
            f"  {target}: {row['tokens_a']:,} -> {row['tokens_b']:,}; "
            f"away {row['own_spelling_written_away']}; leaving {row['forms_leaving'][:5]}; "
            f"entering {row['forms_entering'][:5]}"
        )
    total_away = sum(away.values())
    lines.append(
        f"tokens of a target's own spelling written as another word in view B: {total_away}"
    )
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
