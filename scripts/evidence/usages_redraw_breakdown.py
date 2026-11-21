"""The redraw's judged items broken down, for `docs/usages_audit.md` ("The redraw's results").

The figures in "Where the kept errors are" and "Where the scope cost is" come from this report
(2026-11-21). Its sections:

- **kept items by class** (`usage_rule_profile.classify`), with their errors and the first judge's
  reasons; kept items by flag (capitalised, list-like, mention); the generic-entity rule's labels;
  the words with the most errors;
- **kept errors by decade**, and the capitalised kept items by decade and verdict;
- **candidate rules** on the kept items: errors removed and real uses lost, raw and weighted by
  each word's in-scope usages over its judged items (`counts.json`), an estimate for the table;
- **left-out items by class**, pooled and for the words over the per-word scope bar;
- *prompt*'s and *information processing*'s usages per decade by tag and role, and the
  capitalised share of in-scope usages per decade;
- **copies:** the pairs of works sharing three or more usages (a content key held by just the two),
  and what the copy rule, which counts shared sentences, marks (`usage_selection.find_copies`).

Classes and flags are read from the usages table as it stands. The table has changed in one way
since the redraw was drawn: *information processing*'s verb-tagged usages came into scope (the
author, 2026-11-21), so its judged verb-tagged left-out items now show a kept class; the rest
matches the table the redraw was drawn from. Counts only; held-out words appear under their sealed
names. Run from the repository root on the machine holding the table and the audit's items:
`python scripts/evidence/usages_redraw_breakdown.py AUDIT_DIR OUT.json`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_usages import final_verdicts, load_items, load_judge
from usage_rule_profile import classify

from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import load_words

COLUMNS = ["work_id", "word", "target_start", "position", "pos", "entity", "in_scope"]
COLUMNS += ["capitalised", "list_like", "mention", "role", "content_key", "copy_of"]
JOIN = ["part", "work_id", "word", "target_start"]
OPEN, CLOSE = "⟨t⟩", "⟨/t⟩"
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
LINKS = frozenset({"of", "the", "and", "for", "in", "on", "a", "an", "to", "with", "&"})
TRACED_WORDS = ("prompt", "information processing")


def is_capitalised(word: str) -> bool:
    """Whether a word starts with a capital letter."""
    return word[:1].isupper()


def neighbours(sentence: str) -> dict[str, bool]:
    """Features of the words around the marked target."""
    before, rest = sentence.split(OPEN, 1)
    target, after = rest.split(CLOSE, 1)
    left, right = WORD.findall(before)[-3:], WORD.findall(after)[:3]
    left_content = [word for word in left if word.lower() not in LINKS][-1:]
    right_content = [word for word in right if word.lower() not in LINKS][:1]
    left_capitalised = bool(left_content) and is_capitalised(left_content[0])
    right_capitalised = bool(right_content) and is_capitalised(right_content[0])
    return {
        "previous_capitalised": bool(left) and is_capitalised(left[-1]),
        "next_capitalised": bool(right) and is_capitalised(right[0]),
        "title_run": left_capitalised or right_capitalised,
        "both_sides": left_capitalised and right_capitalised,
        "all_caps": target.isupper() and len(target) > 1,
        "digits_near": bool(re.search(r"\d", before[-25:] + after[:25])),
        "bracket_after": after.lstrip()[:1] == "(",
    }


def judged_items(audit: Path, usages: pl.DataFrame, lemma_of: dict[str, str]) -> pl.DataFrame:
    """The audit's kept and left-out items with their final verdict, class and features."""
    key = json.loads((audit / "key.json").read_text())
    final, _, _ = final_verdicts(audit, key)
    first = load_judge(audit / "verdicts", "A")
    items = load_items(audit)
    rows = [
        {
            "part": entry["part"],
            "decade": entry["part"].rsplit("_", 1)[1],
            "work_id": entry["work_id"],
            "word": lemma_of[entry["word"]],
            "reported": entry["word"],
            "target_start": entry["target_start"],
            "kind": entry["kind"],
            "wrong": final[item_id] == "wrong",
            "reason": str(first[item_id].get("reason") or "none"),
            **neighbours(items[item_id]["sentence"]),
        }
        for item_id, entry in key.items()
        if entry["kind"] in {"kept", "left_out"}
    ]
    table = usages.select(*JOIN, "class", "pos", "entity", "capitalised", "list_like", "mention")
    return pl.DataFrame(rows).join(table, on=JOIN, how="left")


def kept_breakdown(kept: pl.DataFrame) -> dict[str, Any]:
    """Kept items by class, flag, generic-entity label, word and decade."""
    by_class: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"judged": 0, "errors": 0, "reasons": Counter()}
    )
    for name, wrong, reason in kept.select("class", "wrong", "reason").rows():
        by_class[name]["judged"] += 1
        if wrong:
            by_class[name]["errors"] += 1
            by_class[name]["reasons"][reason] += 1
    report: dict[str, Any] = {
        "by_class": {
            name: tally | {"reasons": dict(tally["reasons"])}
            for name, tally in sorted(by_class.items(), key=lambda pair: -pair[1]["errors"])
        }
    }
    for flag in ("capitalised", "list_like", "mention"):
        rows = kept.select(flag, "wrong").rows()
        tally = Counter(f"{bool(value)}:{wrong}" for value, wrong in rows)
        report[f"by_{flag}"] = dict(sorted(tally.items()))
    generic = kept.filter(pl.col("class") == "kept:generic entity")
    report["generic_entity_by_label"] = dict(
        Counter(f"{label}:{wrong}" for label, wrong in generic.select("entity", "wrong").rows())
    )
    errors = kept.filter("wrong")
    report["errors_per_word_top"] = dict(Counter(errors["reported"].to_list()).most_common(12))
    report["errors_by_decade"] = {
        decade: f"{group['wrong'].sum()}/{group.height}"
        for (decade,), group in sorted(kept.group_by("decade"))
    }
    capitalised = kept.filter("capitalised")
    report["capitalised_by_decade"] = {
        decade: f"{group['wrong'].sum()}/{group.height}"
        for (decade,), group in sorted(capitalised.group_by("decade"))
    }
    report["capitalised_by_word"] = {
        word: f"{group['wrong'].sum()}/{group.height}"
        for (word,), group in sorted(capitalised.group_by("reported"))
    }
    not_capitalised = errors.filter(~pl.col("capitalised"))
    report["errors_not_capitalised_by_reason"] = dict(Counter(not_capitalised["reason"].to_list()))
    return report


def candidate_rules(kept: pl.DataFrame, in_scope: dict[str, int]) -> dict[str, Any]:
    """Each rule's errors removed and real uses lost, raw and weighted to the table."""
    judged_per_word = dict(kept.group_by("reported").len().iter_rows())
    weights = [in_scope[word] / judged_per_word[word] for word in kept["reported"]]
    kept = kept.with_columns(pl.Series("weight", weights))
    total_weight = kept["weight"].sum()
    errors_weight = kept.filter("wrong")["weight"].sum()
    capitalised, proper_noun = pl.col("capitalised"), pl.col("pos") == "PROPN"
    rules = {
        "capitalised": capitalised,
        "PROPN": proper_noun,
        "capitalised & PROPN": capitalised & proper_noun,
        "capitalised & title run": capitalised & pl.col("title_run"),
        "capitalised & both sides": capitalised & pl.col("both_sides"),
        "capitalised & next capitalised": capitalised & pl.col("next_capitalised"),
        "capitalised & previous capitalised": capitalised & pl.col("previous_capitalised"),
        "capitalised & list-like": capitalised & pl.col("list_like"),
        "PROPN & title run": proper_noun & pl.col("title_run"),
        "PROPN & both sides": proper_noun & pl.col("both_sides"),
        "list-like": pl.col("list_like"),
        "all caps": pl.col("all_caps"),
        "capitalised & digits near": capitalised & pl.col("digits_near"),
        "capitalised & bracket after": capitalised & pl.col("bracket_after"),
    }
    report: dict[str, Any] = {
        "kept_judged": kept.height,
        "errors": int(kept["wrong"].sum()),
        "weighted_error_rate": round(errors_weight / total_weight, 4),
        "rules": {},
    }
    for name, rule in rules.items():
        caught = kept.filter(rule)
        wrong, right = caught.filter("wrong"), caught.filter(~pl.col("wrong"))
        report["rules"][name] = {
            "errors_removed": wrong.height,
            "real_uses_lost": right.height,
            "weighted_error_rate_after": round(
                (errors_weight - wrong["weight"].sum()) / (total_weight - caught["weight"].sum()), 4
            ),
            "weighted_real_uses_lost_share": round(right["weight"].sum() / total_weight, 4),
        }
    return report


def left_out_breakdown(left_out: pl.DataFrame, over_bar: list[str]) -> dict[str, Any]:
    """Left-out items by class, pooled and for the words over the per-word scope bar."""

    def tally(frame: pl.DataFrame) -> dict[str, int]:
        found = Counter(
            f"{name}:{'real use' if wrong else 'not a use'}"
            for name, wrong in frame.select("class", "wrong").rows()
        )
        return dict(sorted(found.items()))

    return {
        "pooled_by_class": tally(left_out),
        "words_over_scope_bar": {
            word: tally(left_out.filter(pl.col("reported") == word)) for word in over_bar
        },
    }


def table_by_decade(usages: pl.DataFrame) -> dict[str, Any]:
    """The traced words' usages per decade by tag and role, and the capitalised in-scope share."""

    def per_decade(frame: pl.DataFrame) -> dict[str, int]:
        return dict(sorted(frame.group_by("decade").len().iter_rows()))

    report: dict[str, Any] = {}
    for word in TRACED_WORDS:
        usages_of_word = usages.filter(pl.col("word") == word)
        adjective = usages_of_word.filter(pl.col("pos") == "ADJ")
        report[word] = {
            "in_scope": per_decade(usages_of_word.filter("in_scope")),
            "ADJ modifier": per_decade(adjective.filter(pl.col("role") == "modifier")),
            "ADJ other": per_decade(adjective.filter(pl.col("role") != "modifier")),
            "VERB": per_decade(usages_of_word.filter(pl.col("pos") == "VERB")),
        }
    shares = usages.filter("in_scope").group_by("decade").agg(pl.col("capitalised").mean().round(4))
    report["capitalised_share_of_in_scope"] = dict(sorted(shares.iter_rows()))
    return report


def copies(usages: pl.DataFrame) -> dict[str, int]:
    """Work pairs sharing three or more usages, and the works and usages the copy rule marks."""
    works_per_key = usages.group_by("content_key").agg(pl.col("work_id").unique().sort())
    pairs = works_per_key.filter(pl.col("work_id").list.len() == 2)
    usages_per_pair = pairs.group_by(pl.col("work_id").list.join("|")).len()
    sharing_three = usages_per_pair.filter(pl.col("len") >= 3)
    marked = usages.filter(pl.col("copy_of").is_not_null())
    return {
        "work_pairs_sharing_three_or_more_usages": sharing_three.height,
        "works_marked_as_copies": marked["work_id"].n_unique(),
        "usages_of_copies": marked.height,
    }


def main() -> None:
    """Break the redraw's judged items down and write the report."""
    audit, out = Path(sys.argv[1]), Path(sys.argv[2])
    catalog = read_catalog()
    words = {word.lemma: word for word in load_words(catalog)}
    lemma_of = {word.reported_name: word.lemma for word in words.values()}
    usages = pl.concat(
        [
            pl.read_parquet(path, columns=COLUMNS).with_columns(
                pl.lit(path.stem).alias("part"), pl.lit(path.stem.rsplit("_", 1)[1]).alias("decade")
            )
            for path in sorted(catalog.usages.path.glob("*.parquet"))
        ]
    )
    usages = classify(usages, words)
    judged = judged_items(audit, usages, lemma_of)
    counts = json.loads((audit / "counts.json").read_text())
    results = json.loads((audit / "results.json").read_text())
    kept = judged.filter(pl.col("kind") == "kept")
    report = {
        "unmatched_items": judged["class"].null_count(),
        "kept": kept_breakdown(kept),
        "candidate_rules": candidate_rules(kept, counts["in_scope_usages"]),
        "left_out": left_out_breakdown(
            judged.filter(pl.col("kind") == "left_out"), results["scope"]["words_over_two_percent"]
        ),
        "table_by_decade": table_by_decade(usages),
        "copies": copies(usages),
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(f"{judged.height} judged items broken down; written to {out}\n")


if __name__ == "__main__":
    main()
