"""The lemma review's list (agenda 3.3), rebuilt from the signed-off corpus.

Reads the cached spaCy documents of the kept works of the current build, skipping each work's
removed ranges, and counts every plural noun (`NNS`) and third-person singular verb (`VBZ`) by
its lowercased form and spaCy's in-context lemma. A form goes on the list when both spaCy and the
voted lemma map (`data/processed/lemmas.tsv`, with `data/lemma_overrides.tsv`) merge it and they
disagree, the difference not being spelling only (the British and American pairs of the first
audit, `scripts/evidence/first_review/review_coverage.py`). The author reviews the 100 most
frequent forms and every form that touches a target word; 30 forms of the rest are drawn at random
(seed 20261116) for the tail check (at least 27 of 30 right accepts the voted map for the tail).

Writes OUT/review_list.parquet (form, count, spaCy's lemma and its share, the voted lemma, the
group, up to three contexts) and prints the summary. Read-only; run from the repository root on
the run machine: `python scripts/evidence/lemma_review_list.py OUT [PROCESSES]`.
"""

from __future__ import annotations

import multiprocessing
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import read_catalog
from interpretant.catalog.static_view import load_static_view, static_view_parts
from interpretant.corpus.audit import load_target_words

OUT = Path(sys.argv[1])
PROCESSES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEED = 20261116
TOP = 100
TAIL_SAMPLE = 30
CONTEXTS = 3
TAGS = frozenset({"NNS", "VBZ"})
SPELLING = [
    ("our", "or"),
    ("ise", "ize"),
    ("yse", "yze"),
    ("isation", "ization"),
    ("tre", "ter"),
    ("ence", "ense"),
    ("ogue", "og"),
    ("ae", "e"),
    ("oe", "e"),
    ("ll", "l"),
    ("dgement", "dgment"),
    ("mme", "m"),
]


def spelling_only(first: str, second: str) -> bool:
    """Whether two lemmas differ only by a British and American spelling pair."""
    return any(
        first.replace(a, b) == second or second.replace(a, b) == first for a, b in SPELLING
    ) or re.sub(r"[sz]", "z", first) == re.sub(r"[sz]", "z", second)


def voted_map() -> dict[str, str]:
    """The voted map's lemma for each form, the curated overrides taking precedence."""
    table = pl.read_csv("data/processed/lemmas.tsv", separator="\t", quote_char=None)
    mapping = dict(zip(table["token"], table["lemma"], strict=True))
    for line in Path("data/lemma_overrides.tsv").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].rstrip()
        if "\t" in line:
            token, lemma = line.split("\t", 1)
            mapping[token.strip()] = lemma.strip() or token.strip()
    return mapping


def count_part(
    part: str, removals: dict[str, list[tuple[int, int]]], voted: dict[str, str]
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], list[str]]]:
    """One part's (form, spaCy lemma) counts, and a few contexts of the disagreeing forms."""
    import spacy

    vocab = spacy.blank("en").vocab
    pairs: Counter[tuple[str, str]] = Counter()
    contexts: dict[tuple[str, str], list[str]] = defaultdict(list)
    for doc in load_annotations(read_catalog(), part, vocab):
        work_id = doc.user_data[WORK_ID]
        if work_id not in removals:
            continue
        ranges = removals[work_id]
        for token in doc:
            if token.tag_ not in TAGS or any(s <= token.idx < e for s, e in ranges):
                continue
            form, lemma = token.text.lower(), token.lemma_.lower()
            pairs[(form, lemma)] += 1
            other = voted.get(form, form)
            disagree = lemma != form and other != form and lemma != other
            if (
                disagree
                and not spelling_only(lemma, other)
                and len(contexts[(form, lemma)]) < CONTEXTS
            ):
                sentence = token.sent.text.strip()
                contexts[(form, lemma)].append(sentence[:300])
    return pairs, dict(contexts)


def main() -> None:
    """Count, compare with the voted map, group and write the review list."""
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    kept = load_abstracts(catalog).select("work_id", "removals", "dropped_reason").collect()
    kept = kept.filter(pl.col("dropped_reason").is_null())
    removals = {
        work_id: [(r["start"], r["end"]) for r in rows]
        for work_id, rows in kept.select("work_id", "removals").iter_rows()
    }
    voted = voted_map()
    pairs: Counter[tuple[str, str]] = Counter()
    contexts: dict[tuple[str, str], list[str]] = defaultdict(list)
    parts = static_view_parts(catalog)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=PROCESSES, mp_context=context) as pool:
        futures = []
        for part in parts:
            work_ids = load_static_view(catalog, part, columns=["work_id"])["work_id"]
            own = {work_id: removals[work_id] for work_id in work_ids if work_id in removals}
            futures.append(pool.submit(count_part, part, own, voted))
        for done, future in enumerate(as_completed(futures), start=1):
            part_pairs, part_contexts = future.result()
            pairs.update(part_pairs)
            for key, sentences in part_contexts.items():
                contexts[key].extend(sentences)
            sys.stdout.write(f"counting the cached documents: {done}/{len(parts)} parts\n")
            sys.stdout.flush()

    by_form: dict[str, Counter[str]] = defaultdict(Counter)
    for (form, lemma), count in pairs.items():
        by_form[form][lemma] += count
    targets = {
        word for word in load_target_words(Path("data/words_of_interest.txt")) if "_" not in word
    }
    rng = random.Random(SEED)
    rows = []
    total = sum(pairs.values())
    for form, lemmas in by_form.items():
        lemma, lemma_count = lemmas.most_common(1)[0]
        count = sum(lemmas.values())
        other = voted.get(form, form)
        if form in (lemma, other) or lemma == other or spelling_only(lemma, other):
            continue
        sentences = contexts.get((form, lemma), [])
        rows.append(
            {
                "form": form,
                "count": count,
                "spacy_lemma": lemma,
                "spacy_share": lemma_count / count,
                "voted_lemma": other,
                "touches_target": bool({form, lemma, other} & targets),
                "contexts": rng.sample(sentences, min(CONTEXTS, len(sentences))),
            }
        )
    table = pl.DataFrame(rows).sort("count", descending=True).with_row_index("rank", offset=1)
    tail = table.filter((pl.col("rank") > TOP) & ~pl.col("touches_target"))
    sample = set(rng.sample(tail["form"].to_list(), min(TAIL_SAMPLE, tail.height)))
    table = table.with_columns(
        group=pl.when((pl.col("rank") <= TOP) | pl.col("touches_target"))
        .then(pl.lit("author"))
        .when(pl.col("form").is_in(sorted(sample)))
        .then(pl.lit("tail_sample"))
        .otherwise(pl.lit("tail"))
    )
    table.write_parquet(OUT / "review_list.parquet")
    listed = int(table["count"].sum())
    lines = [
        f"\nplural-noun and third-person-verb tokens in kept text: {total:,}",
        f"forms where both merge and disagree, spelling aside: {table.height:,} "
        f"({listed:,} tokens, {listed / total:.3%})",
    ]
    for group in ("author", "tail_sample", "tail"):
        members = table.filter(pl.col("group") == group)
        lines.append(f"  {group}: {members.height:,} forms, {int(members['count'].sum()):,} tokens")
    shown = table.select("rank", "form", "count", "spacy_lemma", "voted_lemma", "group").head(25)
    lines.append(str(shown))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
