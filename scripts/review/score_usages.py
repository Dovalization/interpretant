"""Score the matching audit (`docs/usages_audit.md`): the judges, the adjudicator, the bars.

Reads an audit folder's key, counts and calibration list, and the judges' verdicts in its
`verdicts/` folder: `judge_A_part_*.json` and `judge_B_part_*.json`, each mapping an item id to
`{"verdict": "right" | "wrong", "reason": ..., "note": ...}`. Where the two judges differ, the Opus
adjudicator's verdict stands (`verdicts/opus_adjudication.json` or its part files). With
`pending`, the script writes the items the adjudicator must see, with both judges' verdicts, to
`adjudicate.jsonl` (they quote corpus sentences and stay out of git). Otherwise it scores the
pre-registered bars:

- precision: the kept items' errors pooled, against the smallest number that passes at least 96% of
  the time at a true error rate of 2% for the number judged; and per word, at most 2 errors;
- scope: a word with 3 or more of its left-out items wrongly left out goes back;
- recall: each word's estimated misses (candidates times the share judged right) over its in-scope
  usages plus the misses, at most 2% per word and 1% pooled;
- calibration, given the author's verdicts (a JSON object of item id to `right` or `wrong`): the
  agents' final verdicts agree on at least 54 of 60.

Words appear by their reported names (held-out words by a hash). Writes `results.json`. Run from the
repository root: `python scripts/review/score_usages.py AUDIT_DIR pending|score [AUTHOR.json]`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import binom

KEPT_ERRORS_PER_WORD = 2
LEFT_OUT_WRONG_PER_WORD = 3
MISSED_PER_WORD = 0.02
MISSED_POOLED = 0.01
CALIBRATION_AGREEMENT = 54


def load_judge(folder: Path, judge: str) -> dict[str, dict[str, Any]]:
    """One judge's verdicts, merged over its part files.

    Raises:
        SystemExit: if an item is judged twice.
    """
    return load_parts(folder, f"judge_{judge}")


def load_parts(folder: Path, name: str) -> dict[str, dict[str, Any]]:
    """Verdicts merged over ``NAME.json`` and ``NAME_part_*.json``, not the judges' manifests.

    Raises:
        SystemExit: if an item is judged twice.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    paths = [folder / f"{name}.json", *sorted(folder.glob(f"{name}_part_*.json"))]
    for path in paths:
        if not path.exists() or path.name.endswith(".manifest.json"):
            continue
        for item_id, verdict in json.loads(path.read_text()).items():
            if item_id in verdicts:
                raise SystemExit(f"{name} judged {item_id} twice")
            verdicts[item_id] = verdict
    return verdicts


def load_items(audit: Path) -> dict[str, dict[str, Any]]:
    """Every item, from the batch files."""
    items = {}
    for path in sorted(audit.glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            items[item["id"]] = item
    return items


def pooled_bar(judged: int) -> int:
    """The most kept errors that pass: the smallest count passing 96% of the time at 2%."""
    return next(count for count in range(judged + 1) if binom.cdf(count, judged, 0.02) >= 0.96)


def final_verdicts(audit: Path, key: dict[str, Any]) -> tuple[dict[str, str], int, list[str]]:
    """Each item's final verdict, how many the judges agreed on, and the items still unsettled.

    Raises:
        SystemExit: if a judge left an item without a verdict.
    """
    folder = audit / "verdicts"
    first, second = load_judge(folder, "A"), load_judge(folder, "B")
    missing = sorted(set(key) - set(first) | set(key) - set(second))
    if missing:
        raise SystemExit(f"{len(missing)} items lack a judge's verdict, such as {missing[:5]}")
    settled = load_parts(folder, "opus_adjudication")
    final, agreed, unsettled = {}, 0, []
    for item_id in key:
        one, two = first[item_id]["verdict"], second[item_id]["verdict"]
        if one == two:
            final[item_id] = one
            agreed += 1
        elif item_id in settled:
            final[item_id] = settled[item_id]["verdict"]
        else:
            unsettled.append(item_id)
    return final, agreed, unsettled


def write_pending(audit: Path, key: dict[str, Any]) -> None:
    """Write the items the judges disagree on, with both verdicts, for the adjudicator."""
    folder = audit / "verdicts"
    first, second = load_judge(folder, "A"), load_judge(folder, "B")
    _, agreed, unsettled = final_verdicts(audit, key)
    items = load_items(audit)
    with (audit / "adjudicate.jsonl").open("w", encoding="utf-8") as handle:
        for item_id in unsettled:
            row = items[item_id] | {"judge_1": first[item_id], "judge_2": second[item_id]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.stdout.write(
        f"judges agree on {agreed} of {len(key)}; {len(unsettled)} items for the adjudicator\n"
    )


def score(audit: Path, key: dict[str, Any], author: dict[str, str] | None) -> dict[str, Any]:
    """The bars, from the final verdicts and the counts.

    Raises:
        SystemExit: if an item is still unsettled.
    """
    final, agreed, unsettled = final_verdicts(audit, key)
    if unsettled:
        raise SystemExit(f"{len(unsettled)} items still need the adjudicator: run pending first")
    counts = json.loads((audit / "counts.json").read_text())
    by_kind: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    reasons: Counter[str] = Counter()
    first = load_judge(audit / "verdicts", "A")
    for item_id, entry in key.items():
        by_kind[entry["kind"]][entry["word"]][final[item_id]] += 1
        if entry["kind"] == "kept" and final[item_id] == "wrong":
            reasons[str(first[item_id].get("reason") or "unstated")] += 1
    kept = by_kind["kept"]
    kept_judged = sum(sum(tally.values()) for tally in kept.values())
    kept_errors = sum(tally["wrong"] for tally in kept.values())
    bar = pooled_bar(kept_judged)
    failed_words = sorted(w for w, tally in kept.items() if tally["wrong"] > KEPT_ERRORS_PER_WORD)
    back_words = sorted(
        w for w, tally in by_kind["left_out"].items() if tally["wrong"] >= LEFT_OUT_WRONG_PER_WORD
    )
    recall = {}
    estimated_total = in_scope_total = 0.0
    for word, in_scope in counts["in_scope_usages"].items():
        tally = by_kind["missed"].get(word, Counter())
        judged = sum(tally.values())
        candidates = counts["candidate_misses"].get(word, 0)
        estimated = candidates * (tally["right"] / judged) if judged else 0.0
        share = estimated / (in_scope + estimated) if in_scope + estimated else 0.0
        recall[word] = {
            "candidates": candidates,
            "judged": judged,
            "right": tally["right"],
            "estimated_misses": round(estimated, 1),
            "missed_share": round(share, 4),
        }
        estimated_total += estimated
        in_scope_total += in_scope
    pooled_missed = estimated_total / (in_scope_total + estimated_total)
    recall_failed = sorted(w for w, r in recall.items() if r["missed_share"] > MISSED_PER_WORD)
    results: dict[str, Any] = {
        "items": len(key),
        "judges_agreed": agreed,
        "precision": {
            "judged": kept_judged,
            "errors": kept_errors,
            "bar": bar,
            "passes": kept_errors <= bar,
            "error_reasons": dict(reasons),
            "words_over_two_errors": failed_words,
        },
        "scope": {"words_sent_back": back_words},
        "recall": {
            "pooled_missed_share": round(pooled_missed, 4),
            "passes_pooled": pooled_missed <= MISSED_POOLED,
            "words_over_two_percent": recall_failed,
            "by_word": recall,
        },
    }
    if author is not None:
        ids = json.loads((audit / "calibration.json").read_text())
        agree = sum(author[item_id] == final[item_id] for item_id in ids)
        results["calibration"] = {
            "items": len(ids),
            "agree": agree,
            "passes": agree >= CALIBRATION_AGREEMENT,
            "disagreements": [i for i in ids if author[i] != final[i]],
        }
    return results


def main() -> None:
    """Write the adjudicator's items, or score the audit and write the results."""
    audit, mode = Path(sys.argv[1]), sys.argv[2]
    key = json.loads((audit / "key.json").read_text())
    if mode == "pending":
        write_pending(audit, key)
        return
    author = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None
    results = score(audit, key, author)
    (audit / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    precision, recall = results["precision"], results["recall"]
    verdict = "passes" if precision["passes"] else "fails"
    over_two = len(precision["words_over_two_errors"])
    sent_back = len(results["scope"]["words_sent_back"])
    recall_failed = len(recall["words_over_two_percent"])
    sys.stdout.write(
        f"precision: {precision['errors']} errors in {precision['judged']} "
        f"(bar {precision['bar']}, {verdict}); words over two errors: {over_two}; "
        f"words sent back on scope: {sent_back}; missed share pooled "
        f"{recall['pooled_missed_share']:.2%}, words over 2%: {recall_failed}\n"
    )
    if "calibration" in results:
        calibration = results["calibration"]
        sys.stdout.write(f"calibration: {calibration['agree']} of {calibration['items']}\n")


if __name__ == "__main__":
    main()
