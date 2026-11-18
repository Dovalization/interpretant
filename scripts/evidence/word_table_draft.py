"""The evidence for drafting the word table (`curated/words.tsv`, agenda 4.7; phase step 1).

Over the kept text of the build of record (kept works, removed ranges skipped), with the lemma
exceptions applied, every token of a candidate word is counted by decade, coarse and fine part of
speech, and the route that found it:

- `lemma`: its lemma, folded, is the word;
- `form`: its folded text is the word but its lemma is not (*learning* read as a verb, lemma
  *learn*; *having*);
- `spelling`: neither, but spaCy's spelling table maps its lemma onto the word (*behaviour*), a
  preview of the spelling step (3.4).

The candidates are the static targets of `data/words_of_interest.txt` and the agenda's other
words (7.1's development words, its second wave and 4.7's six-word shortlist). For each word and
minority reading (a part of speech at 5% or more that is not the word's most frequent), ten
example sentences are drawn by the smallest sha256 of (seed, work id, offset), so the draw does
not depend on part order. The phrase targets are counted as a spaced pair (the head by lemma or
form), as one hyphenated token and as one solid token. Corpus counts only; no drift.

Writes OUT/word_table_draft.json and prints a summary. Run from the repository root on the run
machine: `python scripts/evidence/word_table_draft.py OUT [PROCESSES]`.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import sys
from collections import Counter, defaultdict
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
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.static_view import fold

TARGETS = Path("data/words_of_interest.txt")
# Agenda 7.1 (development words and the second wave) and 4.7 (the six-word shortlist).
OTHER_WORDS = (
    "platform alignment prompt hallucination component having transformer embedding cloud "
    "presence strength fusion agency interpretation"
).split()
SEED = 20261118
EXAMPLES = 10
MINORITY_SHARE = 0.05
Works = dict[str, tuple[int, list[tuple[int, int]]]]
# (word, route, decade, coarse tag, fine tag) -> tokens
Counts = Counter[tuple[str, str, int, str, str]]


def draw_key(work_id: str, offset: int) -> str:
    """The example draw's key: independent of the part a document is in."""
    return hashlib.sha256(f"{SEED}:{work_id}:{offset}".encode()).hexdigest()


def count_part(
    part: str,
    works: Works,
    exceptions: list[LemmaException],
    words: frozenset[str],
    phrases: dict[str, tuple[str, str]],
) -> tuple[
    Counts,
    Counter[tuple[str, str]],
    dict[tuple[str, str], list[tuple[str, str, str]]],
    Counter[tuple[str, str, int]],
]:
    """One part's counts, forms, example candidates and phrase counts."""
    import spacy

    vocab = spacy.blank("en").vocab
    ruler = lemma_exception_ruler(vocab, exceptions)
    norms = spacy.load("en_core_web_sm").vocab.lookups.get_table("lexeme_norm")
    counts: Counts = Counter()
    forms: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    phrase_counts: Counter[tuple[str, str, int]] = Counter()
    by_head = {head: (name, modifier) for name, (modifier, head) in phrases.items()}
    joined = {f"{modifier}-{head}": name for name, (modifier, head) in phrases.items()} | {
        f"{modifier}{head}": name for name, (modifier, head) in phrases.items()
    }
    for doc in load_annotations(read_catalog(), part, vocab):
        work = works.get(doc.user_data[WORK_ID])
        if work is None:
            continue
        decade, ranges = work
        ruler(doc)
        previous = ""
        for token in doc:
            if any(low <= token.idx < high for low, high in ranges):
                previous = ""
                continue
            form, lemma = fold(token.text), fold(token.lemma_)
            spelled = fold(norms.get(token.lemma_.lower(), token.lemma_.lower()))
            word, route = "", ""
            if lemma in words:
                word, route = lemma, "lemma"
            elif form in words:
                word, route = form, "form"
            elif spelled in words:
                word, route = spelled, "spelling"
            if word:
                counts[(word, route, decade, token.pos_, token.tag_)] += 1
                forms[(word, form)] += 1
                key = (word, token.pos_)
                examples[key].append(
                    (
                        draw_key(doc.user_data[WORK_ID], token.idx),
                        token.sent.text.strip()[:300],
                        form,
                    )
                )
                if len(examples[key]) > 4 * EXAMPLES:
                    examples[key] = sorted(examples[key])[:EXAMPLES]
            head = by_head.get(lemma) or by_head.get(form)
            if head and previous == head[1]:
                phrase_counts[(head[0], "spaced", decade)] += 1
            base = lemma if lemma in joined else form
            for candidate in {base, base.rstrip("s")}:
                if candidate in joined:
                    phrase_counts[
                        (joined[candidate], "hyphenated" if "-" in candidate else "solid", decade)
                    ] += 1
                    break
            previous = form
    trimmed = {key: sorted(found)[:EXAMPLES] for key, found in examples.items()}
    return counts, forms, trimmed, phrase_counts


def summarise(
    counts: Counts,
    forms: Counter[tuple[str, str]],
    examples: dict[tuple[str, str], list[tuple[str, str, str]]],
    words: list[str],
) -> dict[str, Any]:
    """Each word's routes, part-of-speech shares overall and per decade, forms and examples."""
    summary: dict[str, Any] = {}
    for word in words:
        rows = [
            (route, decade, pos, tag, n)
            for (w, route, decade, pos, tag), n in counts.items()
            if w == word
        ]
        total = sum(n for *_, n in rows)
        pos_total: Counter[str] = Counter()
        tag_total: Counter[str] = Counter()
        routes: Counter[str] = Counter()
        by_decade: dict[int, Counter[str]] = defaultdict(Counter)
        for route, decade, pos, tag, n in rows:
            pos_total[pos] += n
            tag_total[tag] += n
            routes[route] += n
            by_decade[decade][pos] += n
        majority = pos_total.most_common(1)[0][0] if pos_total else ""
        minority = [
            pos for pos, n in pos_total.items() if pos != majority and n >= MINORITY_SHARE * total
        ]
        summary[word] = {
            "tokens": total,
            "routes": dict(routes.most_common()),
            "pos_shares": {pos: round(n / total, 4) for pos, n in pos_total.most_common()}
            if total
            else {},
            "tag_shares": {tag: round(n / total, 4) for tag, n in tag_total.most_common(8)}
            if total
            else {},
            "by_decade": {
                str(decade): {
                    "tokens": sum(c.values()),
                    **{pos: round(n / sum(c.values()), 3) for pos, n in c.most_common(4)},
                }
                for decade, c in sorted(by_decade.items())
            },
            "forms": Counter({f: n for (w, f), n in forms.items() if w == word}).most_common(12),
            "examples": {
                pos: [[text, form] for _, text, form in examples.get((word, pos), [])]
                for pos in minority
            },
        }
    return summary


def main() -> None:
    """Count every part on a spawn pool, then write and print the summary."""
    out = Path(sys.argv[1])
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
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
    targets = load_target_words(TARGETS)
    words = sorted({w for w in targets if "_" not in w} | set(OTHER_WORDS))
    phrases = {w: tuple(w.split("_", 1)) for w in targets if "_" in w}
    exceptions = load_lemma_exceptions(catalog)
    counts: Counts = Counter()
    forms: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    phrase_counts: Counter[tuple[str, str, int]] = Counter()
    parts = static_view_parts(catalog)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=processes, mp_context=context) as pool:
        futures = []
        for part in parts:
            ids = load_static_view(catalog, part, columns=["work_id"])["work_id"]
            own = {work_id: works[work_id] for work_id in ids if work_id in works}
            futures.append(
                pool.submit(count_part, part, own, exceptions, frozenset(words), phrases)
            )
        for done, future in enumerate(as_completed(futures), start=1):
            part_counts, part_forms, part_examples, part_phrases = future.result()
            counts.update(part_counts)
            forms.update(part_forms)
            phrase_counts.update(part_phrases)
            for key, found in part_examples.items():
                examples[key] = sorted(examples[key] + found)[:EXAMPLES]
            sys.stdout.write(f"counting the candidate words: {done}/{len(parts)} parts\n")
            sys.stdout.flush()
    summary = summarise(counts, forms, examples, words)
    phrase_summary = {
        name: {
            kind: {
                str(d): n
                for (p, k, d), n in sorted(phrase_counts.items())
                if p == name and k == kind
            }
            for kind in ("spaced", "hyphenated", "solid")
        }
        for name in sorted(phrases)
    }
    (out / "word_table_draft.json").write_text(
        json.dumps({"words": summary, "phrases": phrase_summary}, indent=1, ensure_ascii=False)
    )
    lines = ["word: tokens; routes; part-of-speech shares"]
    for word, row in summary.items():
        shares = ", ".join(
            f"{pos} {share:.0%}" for pos, share in list(row["pos_shares"].items())[:4]
        )
        lines.append(f"  {word}: {row['tokens']:,}; {row['routes']}; {shares}")
    for name, kinds in phrase_summary.items():
        lines.append(
            f"  {name}: " + ", ".join(f"{kind} {sum(v.values()):,}" for kind, v in kinds.items())
        )
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
