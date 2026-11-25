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
- scope: a word with 3 or more of its left-out items wrongly left out is screened; when the counts
  hold each word's left-out usages (the redraw's), scope is read by its cost (the amendment of
  2026-11-20): each word's estimated real uses wrongly left out (left-out usages times the share
  judged wrongly left out) over its in-scope usages plus those, at most 2% per word and 1% pooled,
  with the loss per decade and each word's combined loss, missed plus left out;
- recall: each word's estimated misses (candidates times the share judged right) over its in-scope
  usages plus the misses, at most 2% per word and 1% pooled;
- with `--calibration DIR`, an audit folder holding the author's verdicts (`author.json`, item id
  to `right` or `wrong`, on its `calibration.json` items): whether its agents' final verdicts agree
  on at least 54 of 60, and as a check each kind's pooled rate corrected for the judges'
  misclassification on those items.

Words appear by their reported names (held-out words by a hash). Writes `results.json`. Run from the
repository root:
`python scripts/review/score_usages.py AUDIT_DIR pending|score [--calibration DIR]
[--amendment COMMIT]`. Scoring runs only from a checkout that keeps the pre-registration
(`check_scoring`): `--amendment` names the commit that last changed the plan before scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom

from interpretant.pipelines.preregistration import PreregistrationError, check_scoring

KEPT_ERRORS_PER_WORD = 2
LEFT_OUT_WRONG_PER_WORD = 3
MISSED_POOLED = 0.01
CALIBRATION_AGREEMENT = 54
# A word's share of real uses lost, missed or wrongly left out (recall, and scope by its cost).
LOST_PER_WORD = 0.02
LOST_POOLED = 0.01
DRAWS = 20_000
CORRECTION_SEED = 20261307


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


def score(audit: Path, key: dict[str, Any], calibration: Path | None) -> dict[str, Any]:
    """The bars, from the final verdicts and the counts; with a calibration, its checks.

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
    screened = sorted(
        w for w, tally in by_kind["left_out"].items() if tally["wrong"] >= LEFT_OUT_WRONG_PER_WORD
    )
    recall = estimated_losses(
        by_kind["missed"], counts["candidate_misses"], counts["in_scope_usages"], "right"
    )
    pooled_missed = pooled_share(recall, counts["in_scope_usages"])
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
        "scope": {"words_screened": screened},
        "recall": {
            "pooled_missed_share": round(pooled_missed, 4),
            "passes_pooled": pooled_missed <= MISSED_POOLED,
            "words_over_two_percent": over_two_percent(recall),
            "by_word": recall,
        },
    }
    if "left_out_usages" in counts:
        results["scope"] |= scope_cost(key, final, by_kind["left_out"], counts, recall)
    if calibration is not None:
        results["calibration"] = calibration_agreement(calibration)
        observed = {
            "kept": (kept_errors, kept_judged),
            "left_out": _positives(by_kind["left_out"], "wrong"),
            "missed": _positives(by_kind["missed"], "right"),
        }
        results["corrected_for_judge_error"] = corrected_rates(calibration, observed)
    return results


def estimated_losses(
    tallies: dict[str, Counter[str]],
    pools: dict[str, int],
    in_scope: dict[str, int],
    real: str,
) -> dict[str, dict[str, Any]]:
    """Each word's real uses lost from a pool: its size times the share of its items judged real.

    The pool is the candidate misses (real uses judged right) or the left-out usages (real uses
    judged wrongly left out); the share lost is over the word's in-scope usages plus the loss.
    """
    losses = {}
    for word, base in in_scope.items():
        tally = tallies.get(word, Counter())
        judged = sum(tally.values())
        pool = pools.get(word, 0)
        estimated = pool * tally[real] / judged if judged else 0.0
        losses[word] = {
            "pool": pool,
            "judged": judged,
            "real": tally[real],
            "estimated": round(estimated, 1),
            "share": round(estimated / (base + estimated), 4) if base + estimated else 0.0,
        }
    return losses


def pooled_share(losses: dict[str, dict[str, Any]], in_scope: dict[str, int]) -> float:
    """The pooled share lost: every word's estimated loss over in-scope usages plus the losses."""
    lost = sum(loss["estimated"] for loss in losses.values())
    return lost / (sum(in_scope.values()) + lost)


def over_two_percent(losses: dict[str, dict[str, Any]]) -> list[str]:
    """The words losing more than 2% of their uses."""
    return sorted(word for word, loss in losses.items() if loss["share"] > LOST_PER_WORD)


def scope_cost(
    key: dict[str, Any],
    final: dict[str, str],
    left_out: dict[str, Counter[str]],
    counts: dict[str, Any],
    recall: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The scope bar read by its cost (the amendment of 2026-11-20), by word and by decade."""
    in_scope = counts["in_scope_usages"]
    lost = estimated_losses(left_out, counts["left_out_usages"], in_scope, "wrong")
    pooled = pooled_share(lost, in_scope)
    combined = {}
    for word, base in in_scope.items():
        both = lost[word]["estimated"] + recall[word]["estimated"]
        combined[word] = round(both / (base + both), 4) if base + both else 0.0
    by_decade: dict[str, Counter[str]] = defaultdict(Counter)
    for item_id, entry in key.items():
        if entry["kind"] == "left_out":
            by_decade[entry["part"].rsplit("_", 1)[1]][final[item_id]] += 1
    decades = estimated_losses(
        by_decade, counts["left_out_by_decade"], counts["in_scope_by_decade"], "wrong"
    )
    return {
        "pooled_lost_share": round(pooled, 4),
        "passes_pooled": pooled <= LOST_POOLED,
        "words_over_two_percent": over_two_percent(lost),
        "by_word": lost,
        "by_decade": decades,
        "combined_loss_by_word": combined,
    }


def _positives(tallies: dict[str, Counter[str]], positive: str) -> tuple[int, int]:
    return (
        sum(tally[positive] for tally in tallies.values()),
        sum(sum(tally.values()) for tally in tallies.values()),
    )


def calibration_agreement(calibration: Path) -> dict[str, Any]:
    """The author's calibration verdicts against that audit's final verdicts."""
    key = json.loads((calibration / "key.json").read_text())
    final, _, _ = final_verdicts(calibration, key)
    author = json.loads((calibration / "author.json").read_text())
    ids = json.loads((calibration / "calibration.json").read_text())
    agree = sum(author[item_id] == final[item_id] for item_id in ids)
    return {
        "audit": str(calibration),
        "items": len(ids),
        "agree": agree,
        "passes": agree >= CALIBRATION_AGREEMENT,
        "disagreements": [i for i in ids if author[i] != final[i]],
    }


# The verdict that marks the rate each kind's bar reads: an error, a real use left out, a miss.
POSITIVE = {"kept": "wrong", "left_out": "wrong", "missed": "right"}


def corrected_rates(
    calibration: Path, observed: dict[str, tuple[int, int]]
) -> dict[str, dict[str, Any]]:
    """Each kind's pooled rate corrected for the judges' misclassification (Rogan–Gladen).

    The author's calibration verdicts give each kind's sensitivity and specificity, drawn from
    their posteriors under uniform priors; the observed rate is drawn the same way; the corrected
    rate is (observed + specificity - 1) / (sensitivity + specificity - 1), clipped to [0, 1].
    """
    key = json.loads((calibration / "key.json").read_text())
    final, _, _ = final_verdicts(calibration, key)
    author = json.loads((calibration / "author.json").read_text())
    ids = json.loads((calibration / "calibration.json").read_text())
    rng = np.random.default_rng(CORRECTION_SEED)
    out = {}
    for kind, (positives, total) in observed.items():
        positive = POSITIVE[kind]
        pairs = [
            (author[i] == positive, final[i] == positive) for i in ids if key[i]["kind"] == kind
        ]
        table = {
            "both": sum(a and g for a, g in pairs),
            "author_only": sum(a and not g for a, g in pairs),
            "judges_only": sum(g and not a for a, g in pairs),
            "neither": sum(not a and not g for a, g in pairs),
        }
        sensitivity = rng.beta(1 + table["both"], 1 + table["author_only"], DRAWS)
        specificity = rng.beta(1 + table["neither"], 1 + table["judges_only"], DRAWS)
        rate = rng.beta(1 + positives, 1 + total - positives, DRAWS)
        informative = sensitivity + specificity > 1
        corrected = np.clip(
            (rate[informative] + specificity[informative] - 1)
            / (sensitivity[informative] + specificity[informative] - 1),
            0,
            1,
        )
        low, middle, high = np.percentile(corrected, [2.5, 50, 97.5])
        out[kind] = {
            "observed": round(positives / total, 4) if total else 0.0,
            "corrected": round(float(middle), 4),
            "interval": [round(float(low), 4), round(float(high), 4)],
            "calibration": table,
        }
    return out


def main() -> None:
    """Write the adjudicator's items, or score the audit and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audit", type=Path)
    parser.add_argument("mode", choices=["pending", "score"])
    parser.add_argument(
        "--calibration", type=Path, help="the audit folder holding the author's author.json"
    )
    parser.add_argument(
        "--amendment",
        help="score: the commit of the plan's last change before scoring (the pre-registration)",
    )
    options = parser.parse_args()
    audit = options.audit
    key = json.loads((audit / "key.json").read_text())
    if options.mode == "pending":
        write_pending(audit, key)
        return
    if options.amendment is None:
        parser.error("score needs --amendment: the commit that last changed the plan")
    try:
        check_scoring(Path.cwd(), audit, options.amendment, options.calibration)
    except PreregistrationError as error:
        raise SystemExit(f"not scored: {error}") from error
    results = score(audit, key, options.calibration)
    (audit / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    precision, scope, recall = results["precision"], results["scope"], results["recall"]
    verdict = "passes" if precision["passes"] else "fails"
    lines = [
        f"precision: {precision['errors']} errors in {precision['judged']} "
        f"(bar {precision['bar']}, {verdict}); "
        f"words over two errors: {len(precision['words_over_two_errors'])}",
        f"scope: {len(scope['words_screened'])} words screened (3 or more of 10)",
        f"recall: missed share pooled {recall['pooled_missed_share']:.2%}, words over 2%: "
        f"{len(recall['words_over_two_percent'])}",
    ]
    if "pooled_lost_share" in scope:
        lines.append(
            f"scope cost: lost share pooled {scope['pooled_lost_share']:.2%}, words over 2%: "
            f"{len(scope['words_over_two_percent'])}"
        )
    if "calibration" in results:
        calibration = results["calibration"]
        lines.append(f"calibration: {calibration['agree']} of {calibration['items']}")
        for kind, rate in results["corrected_for_judge_error"].items():
            low, high = rate["interval"]
            lines.append(
                f"{kind}: observed {rate['observed']:.2%}, corrected {rate['corrected']:.2%} "
                f"({low:.2%} to {high:.2%})"
            )
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
