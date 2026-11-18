"""The compound map's proposal, its audit sample and census, and each form's share per decade.

Phase step 3 (`docs/preprocessing_phase.md`; `docs/compounds.md`). Over the kept text of the
build of record (kept works, removed ranges skipped, accented tokens left out), with the lemma
exceptions and the spelling map applied as in the static view, each two-part compound is counted
in its three forms: hyphenated (*data-base*), closed (*database*) and spaced (*data base*). A
spaced pair counts only where it passes the view's noun-compound gate
(`corpus/compound_spelling.is_noun_compound`); its count without the gate is kept for the report.

Each family's standard is its most frequent form (ties: closed, hyphenated, spaced), or a
word-table lemma. A form becomes a pair when it and the standard both have at least 20 uses.
Families are filtered, each with its reason:
- letter codes and broken words (fewer than 4 letters, a Roman-numeral segment, a vowelless
  segment, a one-letter second segment: *gaba-a*, *group s*);
- spaced forms whose share passing the gate is under 0.5 (mostly syntax);
- spaced forms that contain a word-table lemma (merging them would remove target tokens).

Writes OUT/families.parquet, OUT/proposal.parquet, OUT/sample.parquet (150 pairs, seed 20261123:
75 drawn by tokens, 75 uniform), OUT/census.parquet (pairs touching the word table, the 50 with most
tokens, every re- family, every one-letter first segment, co-, de-, un-, pre- and pro- families
with at least 100 hyphenated uses), OUT/shares.parquet (each form's count per field and decade)
and OUT/summary.json. Run from the repository root on the run machine:
`python scripts/evidence/compound_table.py OUT [PROCESSES]`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import spacy
from rich.console import Console

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.lemma_exceptions import LemmaException, load_lemma_exceptions
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.static_view import load_static_view, static_view_parts
from interpretant.catalog.words import load_words
from interpretant.cli.progress import LiveProgress
from interpretant.corpus.compound_spelling import NO_COMPOUNDS, compound_key, is_noun_compound
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.spelling import has_accent, spell
from interpretant.corpus.static_view import static_tokens
from interpretant.pipelines.workers import map_parts

FLOOR = 20
SEED = 20261123
SAMPLE_BY_TOKENS = 75
SAMPLE_UNIFORM = 75
TOP_BY_TOKENS = 50
CENSUS_PREFIXES = frozenset({"co", "de", "un", "pre", "pro"})
CENSUS_PREFIX_MIN = 100
MIN_GATED_SHARE = 0.5
MIN_LETTERS = 4
HYPHENATED = re.compile(r"^[a-z]+-[a-z]+$")
CLOSED = re.compile(r"^[a-z]+$")
THREE_PARTS = re.compile(r"^[a-z]+(?:-[a-z]+){2,}$")
ROMAN = frozenset({"ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii", "xiv", "xv"})
VOWELS = frozenset("aeiouy")
Works = dict[str, tuple[int, list[tuple[int, int]]]]


@dataclass(frozen=True)
class PartJob:
    """What a worker needs for one part (picklable)."""

    catalog: CatalogConfig
    exceptions: tuple[LemmaException, ...]
    spellings: dict[str, str]
    closed_words: frozenset[str]
    keys: frozenset[str]
    works: Works


# One part's counts over kept text, as plain counters: a worker cannot return a class defined in
# this script, which the parent process would fail to unpickle.
COUNTERS = ("hyphenated", "closed", "spaced_gated", "spaced_any", "straddles", "three_parts")
PartCounts = dict[str, Counter[str]]


def count_part(part: str, job: PartJob) -> PartCounts:
    """One part's forms: every hyphenated and closed candidate, and each spaced pair.

    The ``totals`` counter holds the kept tokens and the hyphenated tokens of other shapes.
    """
    vocab = spacy.blank("en").vocab
    ruler = lemma_exception_ruler(vocab, job.exceptions)
    counts: PartCounts = {name: Counter() for name in (*COUNTERS, "totals")}
    for doc in load_annotations(job.catalog, part, vocab):
        work = job.works.get(doc.user_data[WORK_ID])
        if work is None:
            continue
        _, removals = work
        doc = ruler(doc)
        view = static_tokens(doc, job.spellings, NO_COMPOUNDS)
        by_start = {token.idx: token for token in doc}
        entries = [
            (by_start[start], word, any(low <= start < high for low, high in removals))
            for word, start in zip(view.tokens, view.starts, strict=True)
        ]
        for token, word, removed in entries:
            if removed:
                continue
            counts["totals"]["kept_tokens"] += 1
            if has_accent(token.text):
                continue
            if HYPHENATED.match(word):
                counts["hyphenated"][word] += 1
            elif word in job.closed_words:
                counts["closed"][word] += 1
            elif THREE_PARTS.match(word):
                counts["three_parts"][word] += 1
            elif "-" in word:
                counts["totals"]["other_hyphenated"] += 1
        for (first, left, first_removed), (second, right, second_removed) in pairwise(entries):
            if not (CLOSED.match(left) and CLOSED.match(right)):
                continue
            if left + right not in job.keys:
                continue
            if has_accent(first.text) or has_accent(second.text):
                continue
            adjacent = (
                second.i == first.i + 1 and bool(first.whitespace_) and not second.is_sent_start
            )
            if not adjacent:
                continue
            gated = is_noun_compound(first, second)
            if first_removed or second_removed:
                if gated and first_removed != second_removed:
                    counts["straddles"][f"{left} {right}"] += 1
                continue
            counts["spaced_any"][f"{left} {right}"] += 1
            if gated:
                counts["spaced_gated"][f"{left} {right}"] += 1
    return counts


def candidates(
    catalog: CatalogConfig, spellings: dict[str, str]
) -> tuple[frozenset[str], frozenset[str]]:
    """The closed words a family can have, and the keys of the hyphenated forms it can have.

    Spelled letter-only types with at least the floor, counted over the whole static view (dropped
    works and removed ranges included), a superset of the kept counts, so no family is missed. A
    spaced pair is counted when its letters are a closed word or a hyphenated form's key.
    """
    types = (
        pl.scan_parquet(catalog.static_view.path / "*.parquet")
        .select(pl.col("tokens").explode().alias("token"))
        .group_by("token")
        .len()
        .collect()
    )
    closed: Counter[str] = Counter()
    hyphenated: Counter[str] = Counter()
    for token, count in types.iter_rows():
        if token is None:
            continue
        if CLOSED.match(token) and len(token) >= MIN_LETTERS:
            closed[spell(token, spellings)] += count
        elif HYPHENATED.match(token):
            hyphenated[compound_key(spell(token, spellings))] += count
    return (
        frozenset(word for word, count in closed.items() if count >= FLOOR),
        frozenset(key for key, count in hyphenated.items() if count >= FLOOR),
    )


def segments_of(forms: list[str]) -> list[str]:
    """The two parts of a family, from its hyphenated or spaced form."""
    for form in forms:
        if "-" in form:
            return form.split("-")
        if " " in form:
            return form.split(" ")
    return []


def filter_reason(key: str, segments: list[str]) -> str | None:
    """Why a family is left out as a letter code, or None."""
    if len(key) < MIN_LETTERS:
        return "letter_code"
    if any(segment in ROMAN for segment in segments):
        return "roman_numeral"
    if any(len(segment) >= 2 and not set(segment) & VOWELS for segment in segments):
        return "no_vowel"
    if len(segments) == 2 and len(segments[1]) == 1:
        return "one_letter_second"
    return None


def choose_standard(
    forms: dict[str, tuple[str, int]], lemmas: frozenset[str]
) -> tuple[str, str] | None:
    """The family's standard and its source: a word-table lemma, else the most frequent form."""
    for kind in ("closed", "hyphenated", "spaced"):
        if kind in forms and forms[kind][0] in lemmas:
            return forms[kind][0], "word_table"
    order = {"closed": 0, "hyphenated": 1, "spaced": 2}
    ranked = sorted(forms.items(), key=lambda item: (-item[1][1], order[item[0]]))
    return (ranked[0][1][0], "frequency") if ranked else None


def kind_of(form: str, standard: str) -> str:
    """How a pair changes the view: rejoin (one token), merge (two to one) or split (one to two)."""
    if " " in form:
        return "merge"
    if " " in standard:
        return "split"
    return "rejoin"


def touches(text: str, lemmas: frozenset[str]) -> bool:
    """Whether a form is, or has a word that is, a word-table lemma (phrases included)."""
    return text in lemmas or any(word in lemmas for word in re.split(r"[- ]", text))


def main() -> None:
    """Count every part, build the families and pairs, and write the outputs."""
    out = Path(sys.argv[1])
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    words = load_words(catalog)
    lemmas = frozenset(word.lemma for word in words)
    # Held-out words are read, never printed (docs/preprocessing_phase.md, ground rules).
    shown = frozenset(word.lemma for word in words if not word.held_out)
    spellings = {spelling.form: spelling.standard for spelling in load_spellings(catalog)}
    kept = (
        load_abstracts(catalog)
        .select("work_id", "decade", "removals", "dropped_reason")
        .filter(pl.col("dropped_reason").is_null())
        .collect()
    )
    works: Works = {
        work_id: (decade, [(r["start"], r["end"]) for r in removals])
        for work_id, decade, removals in kept.select("work_id", "decade", "removals").iter_rows()
    }
    closed_words, hyphen_keys = candidates(catalog, spellings)
    exceptions = tuple(load_lemma_exceptions(catalog))
    parts = static_view_parts(catalog)
    jobs = {}
    for part in parts:
        ids = load_static_view(catalog, part, columns=["work_id"])["work_id"]
        own = {work_id: works[work_id] for work_id in ids if work_id in works}
        jobs[part] = PartJob(
            catalog, exceptions, spellings, closed_words, closed_words | hyphen_keys, own
        )
    with LiveProgress(Console()) as live:
        live.stage("counting compound forms", total=len(parts))
        results = map_parts(count_part, jobs, process_count=processes, progress=live)

    totals: PartCounts = {name: Counter() for name in (*COUNTERS, "totals")}
    for counts in results.values():
        for name, counter in counts.items():
            totals[name].update(counter)
    kept_tokens = totals["totals"]["kept_tokens"]

    by_key: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for form, count in totals["hyphenated"].items():
        by_key[compound_key(form)]["hyphenated"][form] += count
    for form, count in totals["spaced_gated"].items():
        by_key[compound_key(form)]["spaced"][form] += count
    family_keys = {
        key
        for key, kinds in by_key.items()
        if max(kinds["hyphenated"].values(), default=0) >= FLOOR
        or max(kinds["spaced"].values(), default=0) >= FLOOR
    }

    family_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for key in sorted(family_keys):
        kinds = by_key[key]
        hyphen = kinds["hyphenated"].most_common(1)[0] if kinds["hyphenated"] else None
        spaced = kinds["spaced"].most_common(1)[0] if kinds["spaced"] else None
        closed = (key, totals["closed"][key]) if totals["closed"][key] else None
        segments = segments_of([found[0] for found in (hyphen, spaced) if found is not None])
        reason = filter_reason(key, segments)
        spaced_any = totals["spaced_any"][spaced[0]] if spaced else 0
        gated_share = spaced[1] / spaced_any if spaced and spaced_any else 0.0
        forms = {
            kind: found
            for kind, found in (("closed", closed), ("hyphenated", hyphen), ("spaced", spaced))
            if found is not None
        }
        spaced_out = None
        if spaced is not None:
            if hyphen is not None and hyphen[0].split("-") != spaced[0].split(" "):
                spaced_out = "different_split"
            elif gated_share < MIN_GATED_SHARE:
                spaced_out = "syntax"
        if spaced_out is not None:
            forms.pop("spaced", None)
        chosen = choose_standard(forms, lemmas) if reason is None else None
        family_rows.append(
            {
                "key": key,
                "closed": closed[1] if closed else 0,
                "hyphenated": hyphen[1] if hyphen else 0,
                "hyphenated_form": hyphen[0] if hyphen else None,
                "spaced": spaced[1] if spaced else 0,
                "spaced_form": spaced[0] if spaced else None,
                "spaced_any": spaced_any,
                "gated_share": round(gated_share, 4),
                "standard": chosen[0] if chosen else None,
                "reason": reason,
                "spaced_reason": spaced_out,
            }
        )
        if chosen is None:
            continue
        standard, source = chosen
        standard_count = next(count for form, count in forms.values() if form == standard)
        for form, count in forms.values():
            if form == standard or count < FLOOR or standard_count < FLOOR:
                continue
            kind = kind_of(form, standard)
            if kind == "merge" and touches(form, lemmas):
                continue
            if form in lemmas:
                continue
            pair_rows.append(
                {
                    "form": form,
                    "standard": standard,
                    "kind": kind,
                    "source": source,
                    "form_count": count,
                    "standard_count": standard_count,
                    "gated_share": round(gated_share, 4) if kind == "merge" else None,
                    "touches_word_table": touches(form, lemmas) or touches(standard, lemmas),
                    "one_letter_first": len(segments[0]) == 1 if segments else False,
                    "prefix": segments[0] if segments else None,
                    "hyphenated_count": hyphen[1] if hyphen else 0,
                    "straddles": totals["straddles"][form] if kind == "merge" else 0,
                }
            )

    families = pl.DataFrame(family_rows)
    families.write_parquet(out / "families.parquet")
    proposal = pl.DataFrame(pair_rows).sort("form_count", descending=True)
    proposal.write_parquet(out / "proposal.parquet")

    rng = np.random.default_rng(SEED)
    weights = proposal["form_count"].to_numpy().astype(float)
    by_tokens = rng.choice(
        proposal.height, size=SAMPLE_BY_TOKENS, replace=False, p=weights / weights.sum()
    )
    rest = np.setdiff1d(np.arange(proposal.height), by_tokens)
    uniform = rng.choice(rest, size=SAMPLE_UNIFORM, replace=False)
    sample = pl.concat(
        [
            proposal[by_tokens.tolist()].with_columns(drawn=pl.lit("by_tokens")),
            proposal[uniform.tolist()].with_columns(drawn=pl.lit("uniform")),
        ]
    )
    sample.write_parquet(out / "sample.parquet")
    census = pl.concat(
        [
            proposal.filter(pl.col("touches_word_table")).with_columns(reason=pl.lit("word_table")),
            proposal.head(TOP_BY_TOKENS).with_columns(reason=pl.lit("top_tokens")),
            proposal.filter(pl.col("prefix") == "re").with_columns(reason=pl.lit("re_prefix")),
            proposal.filter(pl.col("one_letter_first")).with_columns(reason=pl.lit("one_letter")),
            proposal.filter(
                pl.col("prefix").is_in(sorted(CENSUS_PREFIXES))
                & (pl.col("hyphenated_count") >= CENSUS_PREFIX_MIN)
            ).with_columns(reason=pl.lit("prefix")),
        ]
    ).unique(subset=["form"], keep="first", maintain_order=True)
    census.write_parquet(out / "census.parquet")
    paired = set(proposal["form"]) | set(proposal["standard"])
    share_rows = []
    for part, counts in results.items():
        field_name, decade = part.rsplit("_", 1)
        for name in ("hyphenated", "closed", "spaced_gated"):
            for form, count in counts[name].items():
                if form in paired:
                    share_rows.append(
                        {"form": form, "field": field_name, "decade": int(decade), "count": count}
                    )
    pl.DataFrame(share_rows).write_parquet(out / "shares.parquet")

    by_kind = proposal.group_by("kind").agg(pairs=pl.len(), tokens=pl.col("form_count").sum())
    merges = proposal.filter(pl.col("kind") == "merge")["form_count"].sum()
    splits = proposal.filter(pl.col("kind") == "split")
    gained: Counter[str] = Counter()
    for standard, count in splits.select("standard", "form_count").iter_rows():
        for word in standard.split(" "):
            if word in lemmas:
                gained[word if word in shown else "(held out)"] += count
    ungated_only = sorted(
        (
            (form, count, totals["spaced_gated"][form])
            for form, count in totals["spaced_any"].items()
            if count >= FLOOR and totals["spaced_gated"][form] < FLOOR
        ),
        key=lambda row: -row[1],
    )[:20]
    summary = {
        "kept_tokens": kept_tokens,
        "families": families.height,
        "families_by_reason": dict(families.group_by("reason").len().iter_rows()),
        "spaced_forms_left_out": dict(
            families.drop_nulls("spaced_reason").group_by("spaced_reason").len().iter_rows()
        ),
        "pairs_by_kind": {kind: {"pairs": n, "tokens": t} for kind, n, t in by_kind.iter_rows()},
        "changed_share_of_view": round(int(proposal["form_count"].sum()) / kept_tokens, 6),
        "words_removed_by_merges": int(merges or 0),
        "words_added_by_splits": int(splits["form_count"].sum() or 0),
        "targets_gaining_by_splits": dict(gained.most_common()),
        "straddles": int(proposal["straddles"].sum() or 0),
        "sample": sample.height,
        "census_by_reason": dict(census.group_by("reason").len().iter_rows()),
        "ungated_only_top": ungated_only,
        "three_part_tokens": sum(totals["three_parts"].values()),
        "three_part_top": totals["three_parts"].most_common(15),
        "other_hyphenated_tokens": totals["totals"]["other_hyphenated"],
        "top_pairs": proposal.head(25).select("form", "standard", "kind", "form_count").rows(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    sys.stdout.write(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
