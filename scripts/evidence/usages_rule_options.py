"""What candidate scope rules would change, measured on the matching audit (`docs/usages_audit.md`).

The audit of record failed its scope bar: 47 words had 3 or more of their 10 left-out usages judged
wrongly left out. Before a rule is fixed, this script measures candidate rules on the whole usages
table and the judged items. Each left-out usage falls in one stratum: the part of speech that kept
it out (`pos:ADJ`, split by whether the word is studied as a noun), or the name label that did
(`name:PRODUCT`), split by how the target is written: in lower case, opening its sentence, or
capitalised inside it. Each stratum's rate of wrongly left out comes from the judged left-out items
in it (the coarser class's rate when it has fewer than 5).

A candidate keeps some strata in scope. For each, the script reports the usages it brings in, the
judged items among them and how many were real uses, the estimated real uses recovered and errors
added to the kept usages, and two forecasts for a fresh draw by the audit's rules: the expected
number of words sent back on scope (each word's 10 left-out usages drawn from the strata still
left out), and the expected errors it adds to the 20 kept usages per word (the pooled bar left room
for 9). The status quo's forecast, set against the 47 words observed, checks the model. It also
lists the reasons behind the words that failed the precision and recall screens. Words appear by
their reported names. Run from the repository root on the machine holding the table and the audit
items: `python scripts/evidence/usages_rule_options.py AUDIT_DIR OUT.json`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
from score_usages import final_verdicts, load_parts

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import Word, load_words

LEFT_OUT_PER_WORD = 10
KEPT_PER_WORD = 20
SENT_BACK_AT = 3
MIN_STRATUM_ITEMS = 5
PRECISION_ROOM = 51 - 42
COLUMNS = [
    "work_id",
    "word",
    "sentence_start",
    "target_start",
    "position",
    "pos",
    "entity",
    "in_scope",
]


def load_usages() -> pl.DataFrame:
    """Every usage, with its part."""
    catalog = read_catalog()
    return pl.concat(
        [
            pl.read_parquet(path, columns=COLUMNS).with_columns(pl.lit(path.stem).alias("part"))
            for path in sorted(catalog.usages.path.glob("*.parquet"))
        ]
    )


def writing(rows: pl.DataFrame) -> list[str]:
    """How each name-stratum target is written: `lower`, `initial` or `capital`."""
    catalog = read_catalog()
    shapes = {}
    for (part,), group in rows.group_by("part"):
        ids = group["work_id"].unique().to_list()
        texts = dict(
            load_abstracts_part(catalog, str(part), ["work_id", "text"])
            .filter(pl.col("work_id").is_in(ids))
            .iter_rows()
        )
        for row in group.iter_rows(named=True):
            text = texts[row["work_id"]]
            if text[row["target_start"]].islower():
                shape = "lower"
            elif not text[row["sentence_start"] : row["target_start"]].strip(" \"'‘“([-"):
                shape = "initial"
            else:
                shape = "capital"
            shapes[(row["part"], row["work_id"], row["word"], row["target_start"])] = shape
    return [
        shapes.get((r["part"], r["work_id"], r["word"], r["target_start"]), "")
        for r in rows.iter_rows(named=True)
    ]


def strata(left_out: pl.DataFrame, words: dict[str, Word]) -> pl.DataFrame:
    """The left-out usages with their stratum, by the audit's reason rule."""
    names, pos_reasons = [], []
    for row in left_out.iter_rows(named=True):
        word = words[row["word"]]
        if row["entity"] is not None and (row["position"] is not None or row["pos"] in word.pos):
            names.append(True)
            pos_reasons.append(f"name:{row['entity']}")
        else:
            names.append(False)
            noun = "noun word" if "NOUN" in word.pos else "other word"
            pos_reasons.append(f"pos:{row['pos']} ({noun})")
    frame = left_out.with_columns(pl.Series("is_name", names), pl.Series("reason", pos_reasons))
    named = frame.filter("is_name")
    shapes = pl.Series("shape", writing(named))
    named = named.with_columns(shapes)
    return pl.concat(
        [
            named.with_columns((pl.col("reason") + "|" + pl.col("shape")).alias("stratum")),
            frame.filter(~pl.col("is_name")).with_columns(
                pl.col("reason").alias("stratum"), pl.lit("").alias("shape")
            ),
        ],
        how="diagonal",
    )


def judged_left_out(audit: Path, words: dict[str, Word], left_out: pl.DataFrame) -> pl.DataFrame:
    """The judged left-out items matched to their usages; `real_use` if wrongly left out."""
    key = json.loads((audit / "key.json").read_text())
    final, _, _ = final_verdicts(audit, key)
    lemma_of = {word.reported_name: word.lemma for word in words.values()}
    rows = [
        {
            "item": item_id,
            "part": entry["part"],
            "work_id": entry["work_id"],
            "word": lemma_of[entry["word"]],
            "target_start": entry["target_start"],
            "real_use": final[item_id] == "wrong",
        }
        for item_id, entry in key.items()
        if entry["kind"] == "left_out"
    ]
    items = pl.DataFrame(rows)
    matched = items.join(left_out, on=["part", "work_id", "word", "target_start"], how="left")
    if matched["stratum"].null_count():
        raise SystemExit(f"{matched['stratum'].null_count()} judged items match no left-out usage")
    return matched


def stratum_rates(judged: pl.DataFrame) -> Callable[[str], float]:
    """Each stratum's share wrongly left out; the coarser class's when it has few judged items."""
    by_stratum = {s: (n, w) for s, n, w in _tally(judged, "stratum")}
    by_reason = {r: (n, w) for r, n, w in _tally(judged, "reason")}
    overall = (judged.height, int(judged["real_use"].sum()))

    def rate(stratum: str) -> float:
        for counts in (by_stratum.get(stratum), by_reason.get(split(stratum)[0]), overall):
            if counts is not None and counts[0] >= MIN_STRATUM_ITEMS:
                return counts[1] / counts[0]
        return overall[1] / overall[0]

    return rate


def split(stratum: str) -> tuple[str, str]:
    """A stratum's reason and, for a name, how the target is written."""
    reason, _, shape = stratum.partition("|")
    return reason, shape


def _tally(judged: pl.DataFrame, column: str) -> list[tuple[str, int, int]]:
    return [
        (str(value), n, int(wrong))
        for value, n, wrong in judged.group_by(column)
        .agg(pl.len(), pl.col("real_use").sum())
        .iter_rows()
    ]


def sent_back_chance(counts: dict[str, int], rate: Callable[[str], float]) -> float:
    """The chance a fresh draw of 10 from these left-out strata has 3 or more real uses."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    share = sum(n * rate(s) for s, n in counts.items()) / total
    drawn = min(LEFT_OUT_PER_WORD, total)
    return float(binom.sf(SENT_BACK_AT - 1, drawn, share))


# Each candidate keeps in scope the left-out usages whose (reason, writing) it accepts.
Keep = Callable[[str, str], bool]
CANDIDATES: dict[str, Keep] = {
    "nouns take ADJ": lambda reason, _: reason == "pos:ADJ (noun word)",
    "PRODUCT is not a name": lambda reason, _: reason == "name:PRODUCT",
    "LAW is not a name": lambda reason, _: reason == "name:LAW",
    "ORG is not a name": lambda reason, _: reason == "name:ORG",
    "WORK_OF_ART is not a name": lambda reason, _: reason == "name:WORK_OF_ART",
    "a lower-case target is not in a name": lambda _, shape: shape == "lower",
    "a sentence-opening target is not in a name": lambda _, shape: shape == "initial",
}


def either(*names: str) -> Keep:
    """A candidate keeping what any of the named candidates keeps."""
    return lambda reason, shape: any(CANDIDATES[name](reason, shape) for name in names)


CANDIDATES["nouns take ADJ + lower case"] = either(
    "nouns take ADJ", "a lower-case target is not in a name"
)
CANDIDATES["nouns take ADJ + lower case + PRODUCT"] = either(
    "nouns take ADJ + lower case", "PRODUCT is not a name"
)
CANDIDATES["nouns take ADJ + lower case + sentence opening"] = either(
    "nouns take ADJ + lower case", "a sentence-opening target is not in a name"
)


def forecast(
    keep: Keep | None,
    by_word: dict[str, Counter[str]],
    in_scope: dict[str, int],
    rate: Callable[[str], float],
    judged: pl.DataFrame,
) -> dict[str, Any]:
    """One candidate's effect: usages brought in, judged items among them, the two forecasts."""

    def moved_rows(stratum: str) -> bool:
        return keep is not None and keep(*split(stratum))

    moved = recovered = added = sent_back = precision_added = 0.0
    for word, counts in by_word.items():
        brought = {s: n for s, n in counts.items() if moved_rows(s)}
        still = {s: n for s, n in counts.items() if not moved_rows(s)}
        word_moved = sum(brought.values())
        word_added = sum(n * (1 - rate(s)) for s, n in brought.items())
        moved += word_moved
        recovered += sum(n * rate(s) for s, n in brought.items())
        added += word_added
        sent_back += sent_back_chance(still, rate)
        base = in_scope.get(word, 0) + word_moved
        precision_added += KEPT_PER_WORD * word_added / base if base else 0.0
    items = judged.filter(pl.Series([moved_rows(s) for s in judged["stratum"]]))
    return {
        "usages_brought_in": int(moved),
        "judged_items_brought_in": items.height,
        "of_them_real_uses": int(items["real_use"].sum()),
        "estimated_real_uses_recovered": round(recovered),
        "estimated_errors_added_to_kept": round(added),
        "share_of_kept_usages_added": round(moved / sum(in_scope.values()), 4),
        "forecast_words_sent_back": round(sent_back, 1),
        "forecast_errors_added_to_precision_draw": round(precision_added, 1),
        "precision_room": PRECISION_ROOM,
    }


def screen_reasons(audit: Path) -> dict[str, Any]:
    """The kept errors' reasons for words over 2 errors, and the recall-screen words' misses."""
    key = json.loads((audit / "key.json").read_text())
    final, _, _ = final_verdicts(audit, key)
    folder = audit / "verdicts"
    first = load_parts(folder, "judge_A")
    settled = load_parts(folder, "opus_adjudication")
    results = json.loads((audit / "results.json").read_text())
    out: dict[str, Any] = {"precision": {}, "recall": {}}
    for word in results["precision"]["words_over_two_errors"]:
        reasons: Counter[str] = Counter()
        for item_id, entry in key.items():
            if entry["word"] == word and entry["kind"] == "kept" and final[item_id] == "wrong":
                verdict = settled.get(item_id) or first[item_id]
                reasons[str(verdict.get("reason") or "unstated")] += 1
        out["precision"][word] = dict(reasons)
    for word in results["recall"]["words_over_two_percent"]:
        out["recall"][word] = results["recall"]["by_word"][word]
    return out


def main() -> None:
    """Measure the candidates and write them, with the strata and the screens' reasons."""
    audit, out = Path(sys.argv[1]), Path(sys.argv[2])
    words = {word.lemma: word for word in load_words(read_catalog())}
    usages = load_usages()
    in_scope = dict(usages.filter("in_scope").group_by("word").len().iter_rows())
    left_out = strata(usages.filter(~pl.col("in_scope")), words)
    judged = judged_left_out(audit, words, left_out)
    rate = stratum_rates(judged)
    by_word: dict[str, Counter[str]] = defaultdict(Counter)
    for word, stratum, n in left_out.group_by("word", "stratum").len().iter_rows():
        by_word[word][stratum] = n
    population = dict(left_out.group_by("stratum").len().iter_rows())
    judged_tally = {s: (n, w) for s, n, w in _tally(judged, "stratum")}
    strata_table = sorted(
        (
            {
                "stratum": stratum,
                "left_out_usages": population[stratum],
                "judged": judged_tally.get(stratum, (0, 0))[0],
                "real_uses": judged_tally.get(stratum, (0, 0))[1],
                "rate_used": round(rate(stratum), 3),
            }
            for stratum in population
        ),
        key=lambda row: -row["left_out_usages"],
    )
    report = {
        "status_quo": forecast(None, by_word, in_scope, rate, judged),
        "candidates": {
            name: forecast(keep, by_word, in_scope, rate, judged)
            for name, keep in CANDIDATES.items()
        },
        "strata": strata_table,
        "screens": screen_reasons(audit),
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(
        json.dumps({k: report[k] for k in ("status_quo", "candidates")}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
