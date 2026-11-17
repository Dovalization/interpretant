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
- `suspect_merge`: spaCy merges it (whether or not the voted map does too), but the lemma is
  rare beside the form (*cosmos* → *cosmo*) or the form is mostly tagged as a name (*hans*);
- `policy_keep`: spaCy merges it, but the form has its own noun entry in spaCy's WordNet lemma
  index and its singular is mostly read as an adjective (*goods*): the 2026-11-11 rule of
  `data/lemma_overrides.tsv` keeps such plurals, so the author confirms each new one;
- `homograph`: spaCy merges it into a singular that is rarely the same part of speech (*novels*
  → *novel*, *thanks* → *thank*, but also *embeddings* → *embedding*);
- `override`: spaCy merges a form the override file keeps or maps elsewhere; the override
  carries into the exceptions file.

The last four need at least 20 uses. Auxiliaries (`AUX`) are left out: the static view keeps
them as they are (agenda 3.2).

The author reviews the 100 most frequent `disagree` forms, every form touching a target word,
every `suspect_merge` and every `policy_keep`; 30 other `disagree` forms are drawn at random for
the tail check (at least 27 of 30 right accepts the voted map for the tail), and 30 `hyphenated`
forms check the rejoined-segment rule (3.5, decided 2026-11-13) the same way. Draws use seed
20261116.

Writes OUT/review_list.parquet (form, count, kind, spaCy's lemma and its share, the voted lemma,
the signals, the group, up to three contexts) and OUT/plural_merges.parquet (every plural spaCy
merges, with its singular's part-of-speech shares), and prints the summary. Read-only; run from the
repository root on the run machine:
`python scripts/evidence/lemma_review_list.py OUT [PROCESSES] [--tail]`. With `--tail` the
untested `disagree` tail also gets contexts, for the agent judges chosen after the tail check
failed (`docs/lemma_review.md`); they are drawn after the others with seed 20261117, so every
other form's contexts stay as without it.
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
from interpretant.corpus.lemmas import read_overrides

OUT = Path(sys.argv[1])
PROCESSES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
TAIL_CONTEXTS = "--tail" in sys.argv[3:]
SEED = 20261116
TAIL_SEED = 20261117
TOP = 100
TAIL_SAMPLE = 30
CONTEXTS = 3
CONTEXTS_PER_PART = 5
TAGS = frozenset({"NNS", "VBZ"})
SUSPECT_MIN_USES = 20
RARE_LEMMA_RATIO = 0.05
PROPER_SHARE = 0.5
ADJECTIVE_SHARE = 0.5
HOMOGRAPH_SHARE = 0.1
UNREVIEWED = ("tail", "hyphen_rest", "override")
EXAMPLES = ("cosmos", "thanks", "mises", "forceps", "novels", "currents", "logos", "hans", "rogers")
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


def voted_map(overrides: dict[str, str]) -> dict[str, str]:
    """The voted map's lemma for each form, the curated overrides taking precedence."""
    table = pl.read_csv("data/processed/lemmas.tsv", separator="\t", quote_char=None)
    mapping = dict(zip(table["token"], table["lemma"], strict=True))
    return mapping | {token: lemma or token for token, lemma in overrides.items()}


def noun_index() -> frozenset[str]:
    """The nouns of spaCy's WordNet lemma index, as the small English model ships it."""
    import spacy

    pipeline = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "parser", "ner"])
    lemmatizer = pipeline.get_pipe("lemmatizer")
    return frozenset(lemmatizer.lookups.get_table("lemma_index")["noun"])  # type: ignore[attr-defined]


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
            if token.tag_ in TAGS and token.pos_ != "AUX":
                pairs[(form, token.tag_, token.lemma_.lower())] += 1
    return pairs, families


def contexts_part(part: str, removals: Ranges, forms: frozenset[str]) -> dict[str, list[str]]:
    """Up to a few sentences per listed form, from one part's inflected uses."""
    found: dict[str, list[str]] = defaultdict(list)
    for doc, ranges in kept_documents(part, removals):
        for token in doc:
            form = token.text.lower()
            if token.tag_ not in TAGS or token.pos_ == "AUX" or form not in forms:
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
    form: str,
    lemmas: Counter[str],
    tags: Counter[str],
    voted: dict[str, str],
    overrides: dict[str, str],
    families: TagCounts,
    nouns: frozenset[str],
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
    if form in overrides and other != lemma:
        return row | {"kind": "override", "voted_lemma": other}
    if other not in (form, lemma) and not spelling_only(lemma, other):
        return row | {"kind": "disagree", "voted_lemma": other}
    if count < SUSPECT_MIN_USES:
        return None
    own = "noun" if row["tag"] == "NNS" else "verb"
    lemma_uses = sum(n for (word, _), n in families.items() if word == lemma)
    form_uses = sum(n for (word, _), n in families.items() if word == form)
    signals = {
        "voted_merges": other != form,
        "lemma_ratio": lemma_uses / count,
        "own_family_share": families[(lemma, own)] / lemma_uses if lemma_uses else 0.0,
        "adjective_share": families[(lemma, "adjective")] / lemma_uses if lemma_uses else 0.0,
        "proper_share": families[(form, "proper")] / form_uses if form_uses else 0.0,
    }
    row = row | {"voted_lemma": other} | signals
    if own == "noun" and form in nouns and signals["adjective_share"] > ADJECTIVE_SHARE:
        return row | {"kind": "policy_keep"}
    if signals["lemma_ratio"] < RARE_LEMMA_RATIO or signals["proper_share"] > PROPER_SHARE:
        return row | {"kind": "suspect_merge"}
    if signals["own_family_share"] < HOMOGRAPH_SHARE:
        return row | {"kind": "homograph"}
    return None


def plural_merges(
    by_form: dict[str, Counter[str]],
    tags_by_form: dict[str, Counter[str]],
    families_of: dict[str, TagCounts],
) -> pl.DataFrame:
    """Every form read mostly as a plural noun that spaCy merges, with its singular's family shares.

    For the rule on plurals whose singular is mostly another part of speech
    (`docs/lemma_review.md`, the consistency pass); the shares are over every use of the
    singular's spelling in the kept text.
    """
    rows = []
    for form, lemmas in by_form.items():
        lemma = lemmas.most_common(1)[0][0]
        if lemma == form or tags_by_form[form].most_common(1)[0][0] != "NNS":
            continue
        uses = families_of[lemma]
        total = sum(uses.values())
        row: dict[str, object] = {
            "form": form,
            "count": sum(lemmas.values()),
            "spacy_lemma": lemma,
            "singular_uses": total,
        }
        for name in ("noun", "proper", "adjective", "verb", "adverb", "other"):
            row[f"{name}_share"] = uses[(lemma, name)] / total if total else 0.0
        rows.append(row)
    return pl.DataFrame(rows)


def grouped(table: pl.DataFrame, targets: set[str], rng: random.Random) -> pl.DataFrame:
    """The table with each form's review group and its rank within its kind."""
    table = table.with_columns(
        kind_rank=pl.col("count").rank("ordinal", descending=True).over("kind"),
        touches_target=pl.concat_list("form", "spacy_lemma", "voted_lemma")
        .list.eval(pl.element().is_in(sorted(targets)))
        .list.any(),
    )
    disagree = table.filter(pl.col("kind") == "disagree")
    tail = disagree.filter((pl.col("kind_rank") > TOP) & ~pl.col("touches_target"))
    tail_sample = rng.sample(sorted(tail["form"]), min(TAIL_SAMPLE, tail.height))
    hyphenated = sorted(table.filter(pl.col("kind") == "hyphenated")["form"])
    hyphen_sample = rng.sample(hyphenated, min(TAIL_SAMPLE, len(hyphenated)))
    kind = pl.col("kind")
    return table.with_columns(
        group=pl.when(kind.is_in(["suspect_merge", "policy_keep", "homograph", "override"]))
        .then(kind)
        .when((kind == "disagree") & ((pl.col("kind_rank") <= TOP) | pl.col("touches_target")))
        .then(pl.lit("author"))
        .when(pl.col("form").is_in(tail_sample))
        .then(pl.lit("tail_sample"))
        .when(pl.col("form").is_in(hyphen_sample))
        .then(pl.lit("hyphen_sample"))
        .when(kind == "hyphenated")
        .then(pl.lit("hyphen_rest"))
        .otherwise(pl.lit("tail"))
    ).sort("count", "form", descending=[True, False])


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
    overrides = read_overrides(Path("data/lemma_overrides.tsv"))
    voted = voted_map(overrides)
    nouns = noun_index()

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

    plural_merges(by_form, tags_by_form, families_of).write_parquet(OUT / "plural_merges.parquet")
    rows = []
    for form, lemmas in by_form.items():
        lemma = lemmas.most_common(1)[0][0]
        local = families_of[form] + families_of[lemma]
        row = classify(form, lemmas, tags_by_form[form], voted, overrides, local, nouns)
        if row is not None:
            rows.append(row)
    words = load_target_words(Path("data/words_of_interest.txt"))
    targets = {word for word in words if "_" not in word}
    rng = random.Random(SEED)
    table = grouped(pl.DataFrame(rows), targets, rng)

    tail = frozenset(table.filter(pl.col("group") == "tail")["form"].to_list())
    listed = frozenset(table.filter(~pl.col("group").is_in(UNREVIEWED))["form"].to_list())
    if TAIL_CONTEXTS:
        listed |= tail
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
        []
        if form in tail
        else rng.sample(sorted(sentences[form]), min(CONTEXTS, len(sentences[form])))
        for form in table["form"]
    ]
    if TAIL_CONTEXTS:
        tail_rng = random.Random(TAIL_SEED)
        contexts = [
            tail_rng.sample(sorted(sentences[form]), min(CONTEXTS, len(sentences[form])))
            if form in tail
            else drawn
            for form, drawn in zip(table["form"], contexts, strict=True)
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
    spacy_only = [
        sum(lemmas.values())
        for form, lemmas in by_form.items()
        if lemmas.most_common(1)[0][0] != form and voted.get(form, form) == form
    ]
    lines.append(
        f"spaCy merges, voted map keeps (auxiliaries aside): {len(spacy_only):,} forms, "
        f"{sum(spacy_only):,} tokens"
    )
    for example in EXAMPLES:
        lemmas = by_form.get(example, Counter())
        lines.append(f"  {example}: {dict(lemmas.most_common(3))}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
