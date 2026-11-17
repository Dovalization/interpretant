"""The lemma review's list (agenda 3.3), rebuilt from the signed-off corpus.

Reads the cached spaCy documents of the kept works of the current build, skipping each work's
removed ranges, and counts every plural noun (`NNS`) and third-person singular verb (`VBZ`) by
its lowercased form and spaCy's in-context lemma, and every token's coarse tag by form. Three
kinds of form go on the list:

- `disagree`: both spaCy and the voted lemma map (`data/processed/lemmas.tsv`, with
  `data/lemma_overrides.tsv`) merge the form and they disagree, the difference not being spelling
  only (the British and American pairs of `scripts/evidence/first_review/review_coverage.py`);
- `hyphenated`: a hyphenated form whose spaCy lemma differs from the last segment's voted lemma
  rejoined (*self-reports* → *self-report*; 3.5's note);
- `suspect_merge`: both merge it to the same lemma, but the lemma is rare beside the form
  (*cosmos* → *cosmo*), is mostly another part of speech (*novels* → *novel*, *thanks* →
  *thank*), or the form is mostly a name (*hans*). Only forms with at least 20 uses.

The author reviews the 100 most frequent forms and every form that touches a target word; 30
forms of the rest are drawn at random (seed 20261116) for the tail check (at least 27 of 30 right
accepts the voted map for the tail).

Writes OUT/review_list.parquet (form, count, kind, spaCy's lemma and its share, the voted lemma,
the signals, the group, up to three contexts) and prints the summary. Read-only; run from the
repository root on the run machine: `python scripts/evidence/lemma_review_list.py OUT [PROCESSES]`.
"""

from __future__ import annotations

import multiprocessing
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TypeVar

import polars as pl
from spacy.tokens import Doc

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
CONTEXTS_PER_PART = 5
TAGS = frozenset({"NNS", "VBZ"})
SUSPECT_MIN_USES = 20
RARE_LEMMA_RATIO = 0.25
OWN_FAMILY_SHARE = 0.25
PROPER_SHARE = 0.3
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
Ranges = dict[str, list[tuple[int, int]]]
Pairs = Counter[tuple[str, str, str]]
TagCounts = Counter[tuple[str, str]]
Result = TypeVar("Result")


def spelling_only(first: str, second: str) -> bool:
    """Whether two lemmas differ only by a British and American spelling pair."""
    return any(
        first.replace(a, b) == second or second.replace(a, b) == first for a, b in SPELLING
    ) or re.sub(r"[sz]", "z", first) == re.sub(r"[sz]", "z", second)


def family(tag: str) -> str:
    """The coarse family of a Penn tag: proper, noun, verb, adjective, adverb or other."""
    if tag in ("NNP", "NNPS"):
        return "proper"
    for prefix, name in (("NN", "noun"), ("VB", "verb"), ("JJ", "adjective"), ("RB", "adverb")):
        if tag.startswith(prefix):
            return name
    return "other"


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


def kept_documents(part: str, removals: Ranges) -> list[tuple[Doc, list[tuple[int, int]]]]:
    """The part's documents of kept works, each with its removed ranges."""
    import spacy

    vocab = spacy.blank("en").vocab
    return [
        (doc, removals[doc.user_data[WORK_ID]])
        for doc in load_annotations(read_catalog(), part, vocab)
        if doc.user_data[WORK_ID] in removals
    ]


def count_part(part: str, removals: Ranges) -> tuple[Pairs, TagCounts]:
    """One part's (form, tag, spaCy lemma) counts of inflected tokens, and every form's families."""
    pairs: Pairs = Counter()
    families: TagCounts = Counter()
    for doc, ranges in kept_documents(part, removals):
        for token in doc:
            if not token.is_alpha and "-" not in token.text:
                continue
            if any(start <= token.idx < end for start, end in ranges):
                continue
            form = token.text.lower()
            families[(form, family(token.tag_))] += 1
            if token.tag_ in TAGS:
                pairs[(form, token.tag_, token.lemma_.lower())] += 1
    return pairs, families


def contexts_part(part: str, removals: Ranges, forms: frozenset[str]) -> dict[str, list[str]]:
    """Up to a few sentences per listed form, from one part's inflected uses."""
    found: dict[str, list[str]] = defaultdict(list)
    for doc, ranges in kept_documents(part, removals):
        for token in doc:
            form = token.text.lower()
            if token.tag_ not in TAGS or form not in forms:
                continue
            if len(found[form]) >= CONTEXTS_PER_PART:
                continue
            if any(start <= token.idx < end for start, end in ranges):
                continue
            found[form].append(token.sent.text.strip()[:300])
    return dict(found)


def over_parts(
    label: str, parts: list[str], submit: Callable[[ProcessPoolExecutor, str], Future[Result]]
) -> list[Result]:
    """Run one task per part on a spawn pool, with a progress line per finished part."""
    results: list[Result] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=PROCESSES, mp_context=context) as pool:
        futures = [submit(pool, part) for part in parts]
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            sys.stdout.write(f"{label}: {done}/{len(parts)} parts\n")
            sys.stdout.flush()
    return results


def classify(
    form: str, lemmas: Counter[str], tags: Counter[str], voted: dict[str, str], families: TagCounts
) -> dict[str, object] | None:
    """The form's row on the list, or None when it needs no review."""
    lemma, lemma_count = lemmas.most_common(1)[0]
    count = sum(lemmas.values())
    if lemma == form:
        return None
    row: dict[str, object] = {
        "form": form,
        "count": count,
        "tag": tags.most_common(1)[0][0],
        "spacy_lemma": lemma,
        "spacy_share": lemma_count / count,
    }
    if "-" in form and form not in voted:
        prefix, last = form.rsplit("-", 1)
        rejoined = f"{prefix}-{voted.get(last, last)}"
        if rejoined in (form, lemma) or spelling_only(lemma, rejoined):
            return None
        return row | {"kind": "hyphenated", "voted_lemma": rejoined}
    other = voted.get(form, form)
    if other == form:
        return None
    if other != lemma and not spelling_only(lemma, other):
        return row | {"kind": "disagree", "voted_lemma": other}
    if count < SUSPECT_MIN_USES:
        return None
    own = "noun" if row["tag"] == "NNS" else "verb"
    lemma_uses = sum(n for (word, _), n in families.items() if word == lemma)
    form_uses = sum(n for (word, _), n in families.items() if word == form)
    signals = {
        "lemma_ratio": lemma_uses / count,
        "own_family_share": families[(lemma, own)] / lemma_uses if lemma_uses else 0.0,
        "proper_share": families[(form, "proper")] / form_uses if form_uses else 0.0,
    }
    suspect = (
        signals["lemma_ratio"] < RARE_LEMMA_RATIO
        or signals["own_family_share"] < OWN_FAMILY_SHARE
        or signals["proper_share"] > PROPER_SHARE
    )
    return row | {"kind": "suspect_merge", "voted_lemma": other} | signals if suspect else None


def main() -> None:
    """Count, classify, group, gather contexts and write the review list."""
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    kept = load_abstracts(catalog).select("work_id", "removals", "dropped_reason").collect()
    kept = kept.filter(pl.col("dropped_reason").is_null())
    removals = {
        work_id: [(r["start"], r["end"]) for r in rows]
        for work_id, rows in kept.select("work_id", "removals").iter_rows()
    }
    parts = static_view_parts(catalog)
    own_removals: dict[str, Ranges] = {}
    for part in parts:
        work_ids = load_static_view(catalog, part, columns=["work_id"])["work_id"]
        own_removals[part] = {w: removals[w] for w in work_ids if w in removals}
    voted = voted_map()

    counted: list[tuple[Pairs, TagCounts]] = over_parts(
        "counting the cached documents",
        parts,
        lambda pool, part: pool.submit(count_part, part, own_removals[part]),
    )
    pairs: Pairs = Counter()
    families: TagCounts = Counter()
    for part_pairs, part_families in counted:
        pairs.update(part_pairs)
        families.update(part_families)
    by_form: dict[str, Counter[str]] = defaultdict(Counter)
    tags_by_form: dict[str, Counter[str]] = defaultdict(Counter)
    for (form, tag, lemma), count in pairs.items():
        by_form[form][lemma] += count
        tags_by_form[form][tag] += count
    families_of: dict[str, TagCounts] = defaultdict(Counter)
    for (word, name), count in families.items():
        families_of[word][(word, name)] = count

    rows = []
    for form, lemmas in by_form.items():
        lemma = lemmas.most_common(1)[0][0]
        local = families_of[form] + families_of[lemma]
        row = classify(form, lemmas, tags_by_form[form], voted, local)
        if row is not None:
            rows.append(row)
    words = load_target_words(Path("data/words_of_interest.txt"))
    targets = {word for word in words if "_" not in word}
    rng = random.Random(SEED)
    table = (
        pl.DataFrame(rows)
        .with_columns(
            touches_target=pl.concat_list("form", "spacy_lemma", "voted_lemma")
            .list.eval(pl.element().is_in(sorted(targets)))
            .list.any()
        )
        .sort("count", "form", descending=[True, False])
        .with_row_index("rank", offset=1)
    )
    tail = table.filter((pl.col("rank") > TOP) & ~pl.col("touches_target"))
    sample = set(rng.sample(tail["form"].to_list(), min(TAIL_SAMPLE, tail.height)))
    table = table.with_columns(
        group=pl.when((pl.col("rank") <= TOP) | pl.col("touches_target"))
        .then(pl.lit("author"))
        .when(pl.col("form").is_in(sorted(sample)))
        .then(pl.lit("tail_sample"))
        .otherwise(pl.lit("tail"))
    )

    listed = frozenset(table["form"].to_list())
    gathered: list[dict[str, list[str]]] = over_parts(
        "gathering contexts",
        parts,
        lambda pool, part: pool.submit(contexts_part, part, own_removals[part], listed),
    )
    sentences: dict[str, list[str]] = defaultdict(list)
    for part_sentences in gathered:
        for form, found in part_sentences.items():
            sentences[form].extend(found)
    contexts = [
        rng.sample(sorted(sentences[form]), min(CONTEXTS, len(sentences[form])))
        for form in table["form"]
    ]
    table = table.with_columns(contexts=pl.Series(contexts, dtype=pl.List(pl.String)))
    table.write_parquet(OUT / "review_list.parquet")

    total = sum(pairs.values())
    listed_tokens = int(table["count"].sum())
    lines = [
        f"\nplural-noun and third-person-verb tokens in kept text: {total:,}",
        f"forms listed: {table.height:,} ({listed_tokens:,} tokens, {listed_tokens / total:.3%})",
    ]
    for column in ("kind", "group"):
        summary = table.group_by(column).agg(forms=pl.len(), tokens=pl.col("count").sum())
        lines.append(str(summary.sort(column)))
    lines.append(f"touching a target: {table['touches_target'].sum()}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
