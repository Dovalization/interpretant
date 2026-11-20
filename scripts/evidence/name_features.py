"""Features of the names that keep target usages out of scope, against the audit's verdicts.

The usages audit found the transformer's name labels wrongly excluding generic terms: 46% of the
judged PRODUCT exclusions were real uses, 31% of LAW, 22% of ORG and 18% of WORK_OF_ART
(`docs/usages_audit.md`). This script runs the transformer, loaded as the stage loads it, on the
sentence of every usage the current word table leaves out for a name, finds the entity over the
target and records features of it:
- its label and its length in tokens;
- whether the target is its last token;
- whether its other words are all common lower-case words in the corpus (at least 10 lower-case
  uses in the abstracts read);
- whether an acronym in brackets follows it;
- whether a digit or an all-capital word is in it;
- whether *the* comes before it;
- how the target is written (lower case, opening the sentence, capitalised inside it);
- the share of the sentence's words that are capitalised.

The features go to FEATURES.parquet, outside the repository as it quotes the entities; a rerun
reads it instead of running the transformer. For each feature value the output counts the judged
left-out items (real uses and names) and the left-out usages, and for each candidate rule
forecasts, as `usages_rule_options.py` does, the words a fresh draw would send back on scope and
the errors it would add to the precision draw. Words appear by their reported names. Run from the
repository root on the GPU machine: `python scripts/evidence/name_features.py AUDIT_DIR
FEATURES.parquet OUT.json`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
from scipy.stats import binom
from spacy.tokens import Doc, Span
from thinc.api import require_gpu, set_gpu_allocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
from score_usages import final_verdicts

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import Word, load_words
from interpretant.usages.entities import load_transformer
from interpretant.usages.usage_selection import NAME_LABELS

COMMON = 10
MIN_ITEMS = 5
LEFT_OUT_PER_WORD = 10
KEPT_PER_WORD = 20
SENT_BACK_AT = 3
COLUMNS = [
    "work_id",
    "word",
    "sentence_start",
    "sentence_end",
    "target_start",
    "target_end",
    "position",
    "pos",
    "entity",
]
JOIN = ["part", "work_id", "word", "target_start"]
ACRONYM_AFTER = re.compile(r"\s*\(\s*[A-Za-z0-9-]*[A-Z][A-Za-z0-9-]*[A-Z][A-Za-z0-9-]*\s*\)")
LOWER_WORD = re.compile(r"\b[a-z][a-z-]*\b")


def load_usages(words: dict[str, Word]) -> pl.DataFrame:
    """Every usage, with its part and its scope and reason under the current word table."""
    catalog = read_catalog()
    usages = pl.concat(
        [
            pl.read_parquet(path, columns=COLUMNS).with_columns(pl.lit(path.stem).alias("part"))
            for path in sorted(catalog.usages.path.glob("*.parquet"))
        ]
    )
    allowed, reasons = [], []
    for word, position, pos, entity in usages.select("word", "position", "pos", "entity").rows():
        tag_allowed = position is not None or pos in words[word].pos
        is_name = entity in NAME_LABELS
        allowed.append(tag_allowed and not is_name)
        if tag_allowed and is_name:
            reasons.append(f"name:{entity}")
        else:
            noun = "noun word" if "NOUN" in words[word].pos else "other word"
            reasons.append(f"pos:{pos} ({noun})")
    return usages.with_columns(pl.Series("in_scope", allowed), pl.Series("reason", reasons))


def entity_over(doc: Doc, start: int, end: int) -> Span | None:
    """The entity overlapping the character span, if any."""
    return next((e for e in doc.ents if e.start_char < end and start < e.end_char), None)


def features(rows: pl.DataFrame) -> pl.DataFrame:
    """Run the transformer on each name row's sentence and describe the entity over its target."""
    catalog = read_catalog()
    sentence_of: dict[tuple[str, str, int], str] = {}
    texts_seen: list[str] = []
    for (part,), group in rows.group_by("part"):
        ids = group["work_id"].unique().to_list()
        texts = dict(
            load_abstracts_part(catalog, str(part), ["work_id", "text"])
            .filter(pl.col("work_id").is_in(ids))
            .iter_rows()
        )
        texts_seen += texts.values()
        for work_id, start, end in group.select("work_id", "sentence_start", "sentence_end").rows():
            sentence_of[(str(part), work_id, start)] = texts[work_id][start:end]
    lower = Counter(word for text in texts_seen for word in LOWER_WORD.findall(text))
    distinct = sorted(set(sentence_of.values()), key=len)
    set_gpu_allocator("pytorch")
    require_gpu(0)
    nlp = load_transformer()
    docs = dict(zip(distinct, nlp.pipe(distinct, batch_size=64), strict=True))
    out = []
    for row in rows.iter_rows(named=True):
        sentence = sentence_of[(row["part"], row["work_id"], row["sentence_start"])]
        doc = docs[sentence]
        start = row["target_start"] - row["sentence_start"]
        end = row["target_end"] - row["sentence_start"]
        out.append({key: row[key] for key in JOIN} | describe(doc, start, end, row, lower))
    return pl.DataFrame(out)


def describe(
    doc: Doc, start: int, end: int, row: dict[str, Any], lower: Counter[str]
) -> dict[str, Any]:
    """The features of the entity over the target, and of the sentence."""
    letters = [t for t in doc if t.is_alpha]
    capital_share = sum(t.text[0].isupper() for t in letters) / len(letters) if letters else 0.0
    text = doc.text
    if text[start].islower():
        shape = "lower"
    elif not text[:start].strip(" \"'‘“([-"):
        shape = "initial"
    else:
        shape = "capital"
    base = {
        "label_stored": row["entity"],
        "shape": shape,
        "capital_share": round(capital_share, 3),
    }
    entity = entity_over(doc, start, end)
    if entity is None:
        return base | {"label_now": None, "entity_text": None}
    inside = [t for t in entity if t.idx < end and start < t.idx + len(t.text)]
    others = [t for t in entity if t not in inside and t.is_alpha and not t.is_stop]
    after = text[entity.end_char :]
    before = doc[entity.start - 1].lower_ if entity.start > 0 else ""
    return base | {
        "label_now": entity.label_,
        "entity_text": entity.text,
        "tokens": len(entity),
        "target_last": bool(inside) and inside[-1].i == entity.end - 1,
        "others": len(others),
        "others_common": all(lower[t.lower_] >= COMMON for t in others) if others else None,
        "acronym_after": bool(ACRONYM_AFTER.match(after)),
        "digit_or_caps": any(
            any(c.isdigit() for c in t.text) or (t.text.isupper() and len(t.text) > 1)
            for t in entity
        ),
        "the_before": before == "the",
    }


def judged(audit: Path, words: dict[str, Word]) -> pl.DataFrame:
    """The judged left-out items, `real_use` true where the judges found them wrongly left out."""
    key = json.loads((audit / "key.json").read_text())
    final, _, _ = final_verdicts(audit, key)
    lemma_of = {word.reported_name: word.lemma for word in words.values()}
    return pl.DataFrame(
        [
            {
                "part": entry["part"],
                "work_id": entry["work_id"],
                "word": lemma_of[entry["word"]],
                "target_start": entry["target_start"],
                "real_use": final[item_id] == "wrong",
            }
            for item_id, entry in key.items()
            if entry["kind"] == "left_out"
        ]
    )


Rule = Callable[[dict[str, Any]], bool]
RULES: dict[str, Rule] = {}


def rate(frame: pl.DataFrame, fallback: float) -> float:
    """The share of real uses among judged items, or the fallback with too few."""
    judged_items = frame.filter(pl.col("real_use").is_not_null())
    if judged_items.height < MIN_ITEMS:
        return fallback
    return float(judged_items["real_use"].mean())  # type: ignore[arg-type]


def forecast(rule: Rule | None, left: pl.DataFrame, in_scope: dict[str, int]) -> dict[str, Any]:
    """A rule's effect on the name rows: usages and judged items it keeps, and the two forecasts."""
    names = left.filter(pl.col("reason").str.starts_with("name:"))
    keeps = [bool(rule and rule(row)) for row in names.iter_rows(named=True)]
    names = names.with_columns(pl.Series("kept", keeps))
    others = left.filter(~pl.col("reason").str.starts_with("name:"))
    overall = rate(left, 0.0)
    pos_rate = {reason: rate(group, overall) for (reason,), group in others.group_by("reason")}
    moved = names.filter("kept")
    moved_rate = rate(moved, overall)
    still = names.filter(~pl.col("kept"))
    still_overall = rate(still, overall)
    label_rate = {
        reason: rate(group, still_overall) for (reason,), group in still.group_by("reason")
    }
    by_word: dict[str, Counter[str]] = defaultdict(Counter)
    for word, reason in others.select("word", "reason").rows():
        by_word[word][reason] += 1
    for word, reason in still.select("word", "reason").rows():
        by_word[word]["still|" + reason] += 1
    moved_by_word = Counter(moved["word"].to_list())
    sent_back = precision_added = 0.0
    for word in set(by_word) | set(moved_by_word):
        counts = by_word[word]
        total = sum(counts.values())
        if total:
            real = sum(
                n * (label_rate[r[6:]] if r.startswith("still|") else pos_rate.get(r, overall))
                for r, n in counts.items()
            )
            sent_back += float(
                binom.sf(SENT_BACK_AT - 1, min(LEFT_OUT_PER_WORD, total), real / total)
            )
        base = in_scope.get(word, 0) + moved_by_word[word]
        if base:
            precision_added += KEPT_PER_WORD * moved_by_word[word] * (1 - moved_rate) / base
    judged_moved = moved.filter(pl.col("real_use").is_not_null())
    return {
        "usages_kept": moved.height,
        "judged_kept": judged_moved.height,
        "of_them_real_uses": int(judged_moved["real_use"].sum()),
        "estimated_real_uses_recovered": round(moved.height * moved_rate),
        "estimated_errors_added_to_kept": round(moved.height * (1 - moved_rate)),
        "forecast_words_sent_back": round(sent_back, 1),
        "forecast_errors_added_to_precision_draw": round(precision_added, 1),
    }


def feature_tables(names: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """For each feature value, overall and by label: judged items, real uses, usages."""
    tables = {}
    for feature in [
        "label_now",
        "tokens",
        "target_last",
        "others_common",
        "acronym_after",
        "digit_or_caps",
        "the_before",
        "shape",
        "capital_band",
    ]:
        rows = []
        for (label, value), group in names.group_by("reason", feature):
            items = group.filter(pl.col("real_use").is_not_null())
            rows.append(
                {
                    "label": label,
                    "value": str(value),
                    "usages": group.height,
                    "judged": items.height,
                    "real_uses": int(items["real_use"].sum()),
                }
            )
        tables[feature] = sorted(rows, key=lambda r: (r["label"], -r["usages"]))
    return tables


def main() -> None:
    """Compute or read the features, then write the feature tables and the rules' forecasts."""
    audit, features_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if Path.cwd().resolve() in features_path.resolve().parents:
        raise SystemExit("write the features outside the repository: they quote the entities")
    words = {word.lemma: word for word in load_words(read_catalog())}
    usages = load_usages(words)
    in_scope = dict(usages.filter("in_scope").group_by("word").len().iter_rows())
    left = usages.filter(~pl.col("in_scope"))
    name_rows = left.filter(pl.col("reason").str.starts_with("name:"))
    if not features_path.exists():
        features(name_rows).write_parquet(features_path)
    described = pl.read_parquet(features_path)
    items = judged(audit, words)
    left = left.join(items, on=JOIN, how="left").join(described, on=JOIN, how="left")
    left = left.with_columns(
        pl.col("capital_share")
        .cut([0.3, 0.6], labels=["under 0.3", "0.3 to 0.6", "over 0.6"])
        .cast(pl.String)
        .alias("capital_band")
    )
    names = left.filter(pl.col("reason").str.starts_with("name:"))
    report = {
        "judged_left_out_items_now_in_scope": items.height
        - int(left["real_use"].is_not_null().sum()),
        "name_rows": names.height,
        "label_agreement": round(
            float((names["label_now"] == names["label_stored"]).fill_null(value=False).mean()), 4
        ),
        "status_quo": forecast(None, left, in_scope),
        "rules": {name: forecast(rule, left, in_scope) for name, rule in RULES.items()},
        "features": feature_tables(names),
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(
        json.dumps({k: v for k, v in report.items() if k != "features"}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
