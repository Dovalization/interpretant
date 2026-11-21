"""What each keep-or-drop rule of the usages table touches, by decade (`docs/usages_audit.md`).

A rule earns its place if what it drops does not carry the word's meaning (or what it keeps does),
and if getting it wrong would not move the measures: a class whose share changes over the decades,
or whose errors fall on one sense, can pose as drift. This profile reads the rebuilt usages table
and the first audit's judged items and reports:
- per decade, the usages kept and left out by class: left out for each part of speech or name
  label; kept as a hyphen part (head or modifier), with the proper-noun or adjective tag, inside an
  entity the generic-term rule keeps, or as a phrase target;
- per class, the judged kept items' errors and the judged left-out items' real uses;
- for the names behind the first table's exclusions (FEATURES.parquet, from
  `name_features.py`): how many distinct names there are, how many recur, how concentrated the
  recurring ones are in one decade, and the judged real uses by how often the name recurs;
- the usages whose sentence and target recur in another work (duplicated abstracts or sentences);
- per word, the share of hyphen-part usages in the first and last decades.

Words appear by their reported names; no sentence or entity text is written. Run from the
repository root on the machine holding the table, the audit items and the features:
`python scripts/evidence/usage_rule_profile.py AUDIT_DIR FEATURES.parquet OUT.json`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))
from score_usages import final_verdicts, load_items

from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import Word, load_words
from interpretant.usages.usage_selection import NAME_LABELS

COLUMNS = ["work_id", "word", "content_key", "target_start", "position", "pos", "entity"]
COLUMNS += ["in_scope", "drawn"]
JOIN = ["part", "work_id", "word", "target_start"]
RECURRING = 10
BURST_SHARE = 0.5


def load_usages() -> pl.DataFrame:
    """Every usage of the rebuilt table, with its part and decade."""
    catalog = read_catalog()
    return pl.concat(
        [
            pl.read_parquet(path, columns=COLUMNS).with_columns(
                pl.lit(path.stem).alias("part"),
                pl.lit(int(path.stem.rsplit("_", 1)[1])).alias("decade"),
            )
            for path in sorted(catalog.usages.path.glob("*.parquet"))
        ]
    )


def classify(usages: pl.DataFrame, words: dict[str, Word]) -> pl.DataFrame:
    """Each usage's class: why it is left out, or how it is kept."""
    classes = []
    rows = usages.select("word", "position", "pos", "entity", "in_scope").rows()
    for word, position, pos, entity, in_scope in rows:
        allowed = pos in words[word].pos or (position is not None and pos != "X")
        if not in_scope:
            classes.append(f"out:name:{entity}" if allowed else f"out:pos:{pos}")
        elif entity in NAME_LABELS:
            classes.append("kept:generic entity")
        elif position is not None:
            classes.append(f"kept:hyphen {position}")
        elif " " in word:
            classes.append("kept:phrase")
        elif pos == "PROPN":
            classes.append("kept:proper-noun tag")
        elif pos == "ADJ":
            classes.append("kept:adjective tag")
        else:
            classes.append("kept:plain")
    return usages.with_columns(pl.Series("class", classes))


def by_decade(usages: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Each class's share of the decade's usages, in percent."""
    totals = dict(usages.group_by("decade").len().iter_rows())
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for decade, name, n in usages.group_by("decade", "class").len().iter_rows():
        table[name][str(decade)] = round(100 * n / totals[decade], 3)
    table["all usages"] = {str(d): float(n) for d, n in sorted(totals.items())}
    return {name: dict(sorted(shares.items())) for name, shares in sorted(table.items())}


def judged_by_class(audit: Path, words: dict[str, Word], usages: pl.DataFrame) -> dict[str, Any]:
    """The first audit's judged kept and left-out items, by their class in the rebuilt table."""
    key = json.loads((audit / "key.json").read_text())
    final, _, _ = final_verdicts(audit, key)
    items = load_items(audit)
    lemma_of = {word.reported_name: word.lemma for word in words.values()}
    rows = [
        {
            "part": entry["part"],
            "work_id": entry["work_id"],
            "word": lemma_of[entry["word"]],
            "target_start": entry["target_start"],
            "kind": entry["kind"],
            "verdict": final[item_id],
            "first_reason": items[item_id].get("reason", ""),
        }
        for item_id, entry in key.items()
        if entry["kind"] in {"kept", "left_out"}
    ]
    matched = pl.DataFrame(rows).join(usages.select(*JOIN, "class"), on=JOIN, how="left")
    out: dict[str, Any] = {"unmatched": int(matched["class"].null_count())}
    for kind, verdict_of_interest in (("kept", "wrong"), ("left_out", "wrong")):
        tally = Counter()
        flagged = Counter()
        for name, verdict in (
            matched.filter(pl.col("kind") == kind).select("class", "verdict").rows()
        ):
            tally[str(name)] += 1
            flagged[str(name)] += verdict == verdict_of_interest
        label = "kept_errors" if kind == "kept" else "left_out_real_uses"
        out[label] = {
            name: {"judged": n, "flagged": flagged[name]} for name, n in tally.most_common()
        }
    return out


def names(features: Path, audit: Path, words: dict[str, Word]) -> dict[str, Any]:
    """How the first table's excluded names spread: distinct, recurring, bursty, judged."""
    described = pl.read_parquet(features).filter(pl.col("entity_text").is_not_null())
    described = described.with_columns(
        pl.col("entity_text").str.to_lowercase().str.strip_chars(" .,;:()\"'").alias("name"),
        pl.col("part").str.extract(r"_(\d{4})$").alias("decade"),
    )
    per_name = described.group_by("name").agg(
        pl.len().alias("usages"),
        pl.col("decade").mode().first().alias("top_decade"),
        (pl.col("decade") == pl.col("decade").mode().first()).mean().alias("top_share"),
    )
    total = described.height
    recurring = per_name.filter(pl.col("usages") >= RECURRING)
    bursty = recurring.filter(pl.col("top_share") >= BURST_SHARE)
    key = json.loads((audit / "key.json").read_text())
    final, _, _ = final_verdicts(audit, key)
    lemma_of = {word.reported_name: word.lemma for word in words.values()}
    judged = pl.DataFrame(
        [
            {
                "part": e["part"],
                "work_id": e["work_id"],
                "word": lemma_of[e["word"]],
                "target_start": e["target_start"],
                "real_use": final[item_id] == "wrong",
            }
            for item_id, e in key.items()
            if e["kind"] == "left_out"
        ]
    )
    joined = judged.join(described.select(*JOIN, "name"), on=JOIN, how="inner").join(
        per_name.select("name", "usages"), on="name", how="left"
    )
    bands = {
        "seen once": (1, 1),
        "2 to 9 times": (2, RECURRING - 1),
        "10 or more": (RECURRING, None),
    }
    by_band = {}
    for band, (low, high) in bands.items():
        mask = pl.col("usages") >= low
        if high is not None:
            mask &= pl.col("usages") <= high
        group = joined.filter(mask)
        by_band[band] = {
            "judged": group.height,
            "real_uses": int(group["real_use"].sum()),
            "usages": int(per_name.filter(mask)["usages"].sum()),
        }
    return {
        "usages": total,
        "distinct_names": per_name.height,
        f"names_recurring_{RECURRING}_or_more": recurring.height,
        "usages_in_recurring_names": int(recurring["usages"].sum()),
        "recurring_names_half_in_one_decade": bursty.height,
        "usages_in_those": int(bursty["usages"].sum()),
        "judged_by_how_often_the_name_recurs": by_band,
    }


def duplicates(usages: pl.DataFrame) -> dict[str, Any]:
    """Usages whose sentence and target recur in another work, by decade."""
    works_per_key = usages.group_by("content_key").agg(pl.col("work_id").n_unique().alias("works"))
    flagged = usages.join(works_per_key, on="content_key").with_columns(
        (pl.col("works") > 1).alias("shared")
    )
    per_decade = flagged.group_by("decade").agg(
        pl.len().alias("usages"),
        pl.col("shared").sum().alias("shared"),
        (pl.col("shared") & pl.col("drawn")).sum().alias("shared_drawn"),
    )
    return {
        "usages_shared_with_another_work": int(flagged["shared"].sum()),
        "drawn_usages_shared": int((flagged["shared"] & flagged["drawn"]).sum()),
        "by_decade": {
            str(d): {"usages": u, "shared": s, "shared_drawn": sd}
            for d, u, s, sd in per_decade.sort("decade").rows()
        },
    }


def hyphen_shift(usages: pl.DataFrame, words: dict[str, Word]) -> list[dict[str, Any]]:
    """Per word, the in-scope share of hyphen parts in its first and last decades with 100 uses."""
    kept = usages.filter("in_scope")
    shares = (
        kept.group_by("word", "decade")
        .agg(pl.len().alias("n"), pl.col("position").is_not_null().mean().alias("hyphen"))
        .filter(pl.col("n") >= 100)
    )
    out = []
    for (word,), group in shares.group_by("word"):
        group = group.sort("decade")
        first, last = group.row(0, named=True), group.row(-1, named=True)
        out.append(
            {
                "word": words[str(word)].reported_name,
                "first_decade": first["decade"],
                "first_share": round(first["hyphen"], 4),
                "last_decade": last["decade"],
                "last_share": round(last["hyphen"], 4),
            }
        )
    return sorted(out, key=lambda r: -(r["last_share"] - r["first_share"]))


def main() -> None:
    """Profile the rules and write the report."""
    audit, features, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    words = {word.lemma: word for word in load_words(read_catalog())}
    usages = classify(load_usages(), words)
    report = {
        "usages": usages.height,
        "by_decade_percent": by_decade(usages),
        "judged_by_class": judged_by_class(audit, words, usages),
        "names": names(features, audit, words),
        "duplicates": duplicates(usages),
        "hyphen_share_by_word": hyphen_shift(usages, words),
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(f"{usages.height} usages profiled; written to {out}\n")


if __name__ == "__main__":
    main()
