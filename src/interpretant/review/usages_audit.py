"""The matching audit's draws (`docs/usages_audit.md`): kept, left out and missed usages as items.

Precision draws 20 drawn in-scope usages per word spread over decades, and scope 10 usages per word
the table left out, each the smallest audit keys under the audit's seed. Recall searches every kept
work's text for each word's written forms (``build_search_pattern``); a hit no mention of the word
overlaps is a candidate miss, and 20 per word are drawn. The items are shuffled with the seed and
cut into batches; the key maps each item to its usage, the calibration set (its own seed) picks the
items the author judges, and the counts hold what the estimates of real uses lost need.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl

from interpretant.catalog.words import Word
from interpretant.usages.identifiers import build_draw_key

AUDIT_SEED = 20261304
CALIBRATION_SEED = 20261305
KEPT_PER_WORD = 20
LEFT_OUT_PER_WORD = 10
MISSED_PER_WORD = 20
CALIBRATION = {"kept": 30, "left_out": 15, "missed": 15}
BATCH = 50
# The longest sentence shown whole; a longer one is cut to a window around the target
WIDTH = 600
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
    "entity",
    "inside_name",
    "in_scope",
    "drawn",
]
# One drawn usage or hit: its kind (kept, left_out or missed) and its row
Sample = tuple[str, dict[str, Any]]


def compute_shares(available: Mapping[int, int], total: int) -> dict[int, int]:
    """An equal share per decade; a decade short of its share gives all it has, the rest spread."""
    shares = {}
    remaining = total
    by_size = sorted(available, key=lambda decade: (available[decade], decade))
    for index, decade in enumerate(by_size):
        share = min(available[decade], remaining // (len(by_size) - index))
        shares[decade] = share
        remaining -= share
    return shares


def build_audit_key(seed: int, work_id: str, word: str, place: str) -> str:
    """The seeded key ordering one usage or hit in the draws."""
    return build_draw_key(seed, work_id, word, place)


def build_forms(word: Word, spellings: Mapping[str, str]) -> list[str]:
    """The written forms the search looks for: the word, its other spellings, and their plurals."""
    if word.match == "form":
        return [word.lemma]
    bases = {word.lemma} | {form for form, standard in spellings.items() if standard == word.lemma}
    forms = set(bases)
    for base in bases:
        forms |= {base + "s", base + "es"}
        if base.endswith("y"):
            forms.add(base[:-1] + "ies")
    return sorted(forms, key=len, reverse=True)


def build_search_pattern(word: Word, spellings: Mapping[str, str]) -> re.Pattern[str]:
    """The case-insensitive search for one word, alone or as a hyphen part."""
    if word.is_phrase:
        modifier, head = word.lemma.split(" ")
        heads = "|".join(re.escape(form) for form in build_forms(_as_head(head), spellings))
        body = rf"{re.escape(modifier)}[\s-]+(?:{heads})"
    else:
        body = "|".join(re.escape(form) for form in build_forms(word, spellings))
    return re.compile(rf"(?<![^\W_])(?:{body})(?![^\W_])", re.IGNORECASE)


def build_with_keys(usages: pl.DataFrame, seed: int) -> pl.DataFrame:
    """The usages with their audit keys."""
    return usages.with_columns(
        pl.struct("work_id", "word", "content_key")
        .map_elements(
            lambda row: build_audit_key(seed, row["work_id"], row["word"], row["content_key"]),
            return_dtype=pl.String,
        )
        .alias("key")
    )


def draw_kept(usages: pl.DataFrame, seed: int) -> pl.DataFrame:
    """20 drawn in-scope usages per word, spread over decades, the smallest keys in each."""
    drawn = build_with_keys(usages.filter(pl.col("drawn") & pl.col("in_scope")), seed)
    chosen = []
    for (_,), rows in drawn.group_by("word"):
        counts = dict(rows.group_by("decade").len().iter_rows())
        for decade, share in compute_shares(counts, KEPT_PER_WORD).items():
            chosen.append(rows.filter(pl.col("decade") == decade).sort("key").head(share))
    return pl.concat(chosen)


def draw_left_out(usages: pl.DataFrame, seed: int) -> pl.DataFrame:
    """10 usages per word the table left out of scope, the smallest keys."""
    left_out = build_with_keys(usages.filter(~pl.col("in_scope")), seed)
    return left_out.sort("key").group_by("word", maintain_order=True).head(LEFT_OUT_PER_WORD)


def draw_missed(
    smallest_by_part: Iterable[Mapping[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """20 candidate misses per word, the smallest keys, from each part's smallest-keyed few."""
    pool: dict[str, list[dict[str, Any]]] = {}
    for smallest in smallest_by_part:
        for word, hits in smallest.items():
            pool.setdefault(word, []).extend(hits)
    return [
        hit
        for hits in pool.values()
        for hit in sorted(hits, key=lambda hit: hit["key"])[:MISSED_PER_WORD]
    ]


def build_items(
    samples: list[Sample], texts: Mapping[str, str], words: Mapping[str, Word], seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The samples shuffled with the seed, as the judges' items and the key to their usages."""
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    items, key = [], {}
    for index, (kind, row) in enumerate(shuffled):
        item_id = f"U{index:04d}"
        word = words[row["word"]]
        item = {
            "id": item_id,
            "kind": kind,
            "word": row["word"],
            "parts_of_speech": list(word.pos),
            "sentence": render_marked_sentence(texts[row["work_id"]], row),
        }
        if kind == "left_out":
            item["reason"] = format_left_out_reason(row)
        items.append(item)
        key[item_id] = {
            "kind": kind,
            "work_id": row["work_id"],
            "word": word.reported_name,
            "target_start": row["target_start"],
            "target_end": row["target_end"],
            "part": row["part"],
        }
    return items, key


def draw_calibration(key: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The items the author judges: 30 kept, 15 left out and 15 missed, by their own seed."""
    rng = random.Random(CALIBRATION_SEED)
    calibration = []
    for kind, size in CALIBRATION.items():
        of_kind = sorted(item_id for item_id, entry in key.items() if entry["kind"] == kind)
        calibration += rng.sample(of_kind, min(size, len(of_kind)))
    return sorted(calibration)


def count_usages(
    usages: pl.DataFrame,
    candidates: Mapping[str, int],
    key: Mapping[str, Mapping[str, Any]],
    words: Mapping[str, Word],
) -> dict[str, Any]:
    """Each word's in-scope and left-out usages and candidate misses, and the usages by decade.

    Words appear by their reported names (held-out words by a hash).
    """
    in_scope = usages.filter("in_scope").group_by("word").len().iter_rows()
    left_out = usages.filter(~pl.col("in_scope"))
    return {
        "in_scope_usages": {words[word].reported_name: n for word, n in sorted(in_scope)},
        "left_out_usages": {
            words[word].reported_name: n
            for word, n in sorted(left_out.group_by("word").len().iter_rows())
        },
        "in_scope_by_decade": count_by_decade(usages.filter("in_scope")),
        "left_out_by_decade": count_by_decade(left_out),
        "candidate_misses": {
            words[word].reported_name: n for word, n in sorted(candidates.items())
        },
        "items": dict(Counter(entry["kind"] for entry in key.values())),
        "held_out_words": sum(word.held_out for word in words.values()),
    }


def count_by_decade(usages: pl.DataFrame) -> dict[str, int]:
    """The usages per decade, keyed as the parts name it."""
    return {str(decade): n for decade, n in sorted(usages.group_by("decade").len().iter_rows())}


def render_marked_sentence(text: str, row: Mapping[str, Any]) -> str:
    """The sentence, or a window of it, with the target between ⟨t⟩ and ⟨/t⟩."""
    sentence = text[row["sentence_start"] : row["sentence_end"]]
    start = row["target_start"] - row["sentence_start"]
    end = row["target_end"] - row["sentence_start"]
    shown = f"{sentence[:start]}⟨t⟩{sentence[start:end]}⟨/t⟩{sentence[end:]}".strip()
    if len(shown) <= WIDTH:
        return shown
    middle = shown.index("⟨t⟩")
    begin = max(0, min(middle - WIDTH // 2, len(shown) - WIDTH))
    prefix = "…" if begin else ""
    suffix = "…" if begin + WIDTH < len(shown) else ""
    return prefix + shown[begin : begin + WIDTH] + suffix


def format_left_out_reason(row: Mapping[str, Any]) -> str:
    """Why the table left a usage out: the name it sits in, or its part of speech."""
    if row["inside_name"]:
        return f"inside a name ({row['entity'] or 'a named instrument'})"
    return f"part of speech {row['pos']}"


def _as_head(lemma: str) -> Word:
    """A phrase's head as a word matched by lemma, for its forms."""
    return Word(lemma=lemma, pos=("NOUN",), match="lemma", held_out=False)
