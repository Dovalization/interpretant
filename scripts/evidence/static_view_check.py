"""Step 4's check: the rebuilt static view against the view of record R_B.

`docs/preprocessing_phase.md` (step 4) fixes the bar before the rebuild. R_B (`daadb4ed`) is the
view with the lemma exceptions and neither map; the rebuild adds the spelling map and the compound
map (`docs/spellings.md`, `docs/compounds.md`). Each entry of R_B is rewritten as the view would
rewrite it (spelled per hyphen segment, then the one-token compound map), and the new view is read
against that:

- **Alignment:** the same works in the same parts, each with R_B's starts.
- **Changes explained:** every changed entry is its rewrite (a spelling, a one-token compound), or
  a spaced compound's standard followed by `""`. The reverse, a rewrite the new view did not
  apply, is allowed only for accented tokens, which both maps leave alone; the text decides.
- **Targets** (kept works, removed ranges skipped): no target entry of R_B changes, each target's
  word count is R_B's plus the words written into it, and no merged pair holds a word of the
  table. Held-out words are counted, never printed.
- **Straddles:** merged pairs with one token inside a removed range, in kept works.
- **Reconciliation:** merges and one-token rewrites in kept works against the builder's counts.
- **Spot check:** 80 changed entries in kept works, drawn by the smallest sha256 of (seed, work
  id, start), with R_B's and the new entries around them and the text near the token. The items
  quote corpus text, so they stay out of git; the key (blind id to work, start and kind) does not.

Usage, from the repository root on the run machine:
`python scripts/evidence/static_view_check.py R_B_DIR PROPOSAL.parquet OUT [PROCESSES]`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.compounds import load_compounds
from interpretant.catalog.config import read_catalog
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.static_view import static_view_parts
from interpretant.catalog.words import load_words
from interpretant.cli.progress import LiveProgress
from interpretant.corpus.compound_spelling import build_compound_map
from interpretant.corpus.spelling import has_accent, spell
from interpretant.pipelines.workers import map_parts

SEED = 20261126
SPOT_CHECK = 80
CONTEXT = 8
SNIPPET = 120
EXAMPLES = 20
_WORD = re.compile(r"\S+")
# A spot-check candidate: draw key, work id, start, kind, R_B's entries, new entries, position.
Candidate = tuple[str, str, int, str, list[str], list[str], int]


@dataclass(frozen=True)
class PartJob:
    """What a worker needs for one part (picklable)."""

    old_path: Path
    new_path: Path
    spellings: dict[str, str]
    one_token: dict[str, str]
    spaced: dict[tuple[str, str], str]
    targets: frozenset[str]
    phrases: frozenset[str]
    held_out: frozenset[str]
    removals: dict[str, list[tuple[int, int]]]


def draw_key(work_id: str, start: int) -> str:
    """The spot check's draw key."""
    return hashlib.sha256(f"{SEED}:{work_id}:{start}".encode()).hexdigest()


def measured_entries(starts: list[int], ranges: list[tuple[int, int]] | None) -> list[bool]:
    """Which entries a kept work's cells read: those outside its removed ranges."""
    if ranges is None:
        return [False] * len(starts)
    lows = [low for low, _ in ranges]
    return [
        (index := bisect_right(lows, start) - 1) < 0 or start >= ranges[index][1]
        for start in starts
    ]


@dataclass
class PartTally:
    """One part's counts, examples and spot-check candidates."""

    counts: Counter[str] = field(default_factory=Counter)
    target_old: Counter[str] = field(default_factory=Counter)
    target_new: Counter[str] = field(default_factory=Counter)
    target_gained: Counter[str] = field(default_factory=Counter)
    target_changed: Counter[str] = field(default_factory=Counter)
    merged_forms: Counter[str] = field(default_factory=Counter)
    rewritten_forms: Counter[str] = field(default_factory=Counter)
    unexplained: list[tuple[str, int, str, str, str]] = field(default_factory=list)
    not_applied: list[tuple[str, int]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)

    def keep(self, candidate: Candidate) -> None:
        """Keep a candidate, pruning to the smallest draw keys as they pile up."""
        self.candidates.append(candidate)
        if len(self.candidates) > 4 * SPOT_CHECK:
            self.candidates = sorted(self.candidates)[:SPOT_CHECK]


def check_part(part: str, job: PartJob) -> dict[str, Any]:
    """One part's tally, as plain data."""
    columns = ["work_id", "tokens", "starts"]
    old = pl.read_parquet(job.old_path, columns=columns)
    new = {row[0]: row for row in pl.read_parquet(job.new_path, columns=columns).iter_rows()}
    tally = PartTally()
    counts = tally.counts
    old_ids = set(old["work_id"].to_list())
    counts["works_only_in_r_b"] = len(old_ids - new.keys())
    counts["works_only_in_new"] = len(new.keys() - old_ids)
    cache: dict[str, str] = {}

    def rewrite(word: str) -> str:
        if word not in cache:
            spelled = spell(word, job.spellings)
            cache[word] = job.one_token.get(spelled, spelled)
        return cache[word]

    for work_id, before, starts in old.iter_rows():
        counts["works"] += 1
        if work_id not in new:
            continue
        _, after, new_starts = new[work_id]
        if new_starts != starts:
            counts["works_misaligned"] += 1
            continue
        counts["entries"] += len(before)
        wanted = [rewrite(word) for word in before] + [""]
        ranges = job.removals.get(work_id)
        measured = measured_entries(starts, ranges)
        index = 0
        while index < len(before):
            standard = job.spaced.get((wanted[index], wanted[index + 1]))
            span = 1
            if standard is not None and after[index] == standard and after[index + 1] == "":
                span, kind = 2, "merge"
                pieces = {wanted[index], wanted[index + 1]}
                phrase = f"{wanted[index]} {wanted[index + 1]}"
                counts["merges_holding_a_word"] += bool(
                    pieces & job.targets or phrase in job.phrases
                )
                counts["merges_holding_a_held_out_word"] += bool(pieces & job.held_out)
                if measured[index] != measured[index + 1] and ranges is not None:
                    counts["straddles"] += 1
                elif measured[index]:
                    tally.merged_forms[f"{before[index]} {before[index + 1]}"] += 1
            else:
                counts["spaced_pairs_not_merged"] += standard is not None
                kind = classify(before[index], after[index], wanted[index])
            counts[f"entries_{kind}"] += 1
            if kind == "not_applied":
                tally.not_applied.append((work_id, starts[index]))
            elif kind == "unexplained" and len(tally.unexplained) < EXAMPLES:
                tally.unexplained.append(
                    (work_id, starts[index], before[index], after[index], wanted[index])
                )
            elif kind in ("rewrite", "split") and measured[index]:
                spelled = spell(before[index], job.spellings)
                tally.rewritten_forms["spelling"] += spelled != before[index]
                if spelled in job.one_token:
                    tally.rewritten_forms[spelled] += 1
            for position in range(index, index + span):
                if measured[position]:
                    count_targets(tally, job, before[position], after[position])
            if kind in ("merge", "rewrite", "split", "unexplained") and measured[index]:
                low, high = max(0, index - CONTEXT), index + span + CONTEXT
                tally.keep(
                    (
                        draw_key(work_id, starts[index]),
                        work_id,
                        starts[index],
                        kind,
                        before[low:high],
                        after[low:high],
                        index - low,
                    )
                )
            index += span
    return {
        "counts": dict(counts),
        "target_old": dict(tally.target_old),
        "target_new": dict(tally.target_new),
        "target_gained": dict(tally.target_gained),
        "target_changed": dict(tally.target_changed),
        "merged_forms": dict(tally.merged_forms),
        "rewritten_forms": dict(tally.rewritten_forms),
        "unexplained": tally.unexplained,
        "not_applied": tally.not_applied,
        "candidates": sorted(tally.candidates)[:SPOT_CHECK],
    }


def classify(before: str, after: str, wanted: str) -> str:
    """An entry that is not a merge: unchanged, rewritten as wanted, not rewritten, or else."""
    if after == wanted:
        if after == before:
            return "unchanged"
        return "split" if " " in after else "rewrite"
    return "not_applied" if after == before else "unexplained"


def count_targets(tally: PartTally, job: PartJob, before: str, after: str) -> None:
    """Count a measured entry's target words in R_B and in the new view."""
    if before in job.targets:
        tally.target_old[before] += 1
        tally.target_changed[before] += after != before
    for word in after.split():
        if word in job.targets:
            tally.target_new[word] += 1
            tally.target_gained[word] += before != word


def token_text(text: str | None, start: int) -> str:
    """The whitespace-delimited text at a token's start."""
    found = _WORD.match(text or "", start)
    return found.group(0) if found else ""


def main() -> None:
    """Run the check over every part, apply the bar, and write the summary and the spot check."""
    r_b, proposal_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    processes = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    new_parts = static_view_parts(catalog)
    old_parts = sorted(path.stem for path in r_b.glob("*.parquet"))
    words = load_words(catalog)
    targets = frozenset(word.lemma for word in words if not word.is_phrase)
    held_out = frozenset(word.lemma for word in words if word.held_out)
    compounds = build_compound_map(
        (compound.form, compound.standard) for compound in load_compounds(catalog)
    )
    kept = (
        load_abstracts(catalog)
        .filter(pl.col("dropped_reason").is_null())
        .select("work_id", "removals")
        .collect()
    )
    removals = {
        work_id: sorted((r["start"], r["end"]) for r in ranges)
        for work_id, ranges in kept.iter_rows()
    }
    jobs = {}
    for part in sorted(set(old_parts) & set(new_parts)):
        ids = pl.read_parquet(r_b / f"{part}.parquet", columns=["work_id"])["work_id"]
        jobs[part] = PartJob(
            old_path=r_b / f"{part}.parquet",
            new_path=catalog.static_view.path / f"{part}.parquet",
            spellings={spelling.form: spelling.standard for spelling in load_spellings(catalog)},
            one_token=dict(compounds.one_token),
            spaced=dict(compounds.spaced),
            targets=targets,
            phrases=frozenset(word.lemma for word in words if word.is_phrase),
            held_out=held_out,
            removals={work_id: removals[work_id] for work_id in ids if work_id in removals},
        )
    with LiveProgress(Console()) as live:
        live.stage("checking parts", total=len(jobs))
        results = map_parts(check_part, jobs, process_count=processes, progress=live)

    def total(name: str) -> Counter[str]:
        summed: Counter[str] = Counter()
        for result in results.values():
            summed.update(result[name])
        return summed

    counts = total("counts")
    target_old, target_new = total("target_old"), total("target_new")
    target_gained, target_changed = total("target_gained"), total("target_changed")
    not_applied = [entry for result in results.values() for entry in result["not_applied"]]
    drawn: list[Candidate] = sorted(
        candidate for result in results.values() for candidate in result["candidates"]
    )[:SPOT_CHECK]
    needed = sorted({work_id for work_id, _ in not_applied} | {row[1] for row in drawn})
    texts = dict(
        load_abstracts(catalog)
        .filter(pl.col("work_id").is_in(needed))
        .select("work_id", "text")
        .collect()
        .iter_rows()
    )
    accents = Counter(
        "accented" if has_accent(token_text(texts.get(work_id), start)) else "not_accented"
        for work_id, start in not_applied
    )
    built = dict(pl.read_parquet(proposal_path).select("form", "form_count").iter_rows())
    merged, rewritten = total("merged_forms"), total("rewritten_forms")
    bar = {
        "alignment": counts["works_misaligned"] == 0
        and counts["works_only_in_r_b"] == 0
        and counts["works_only_in_new"] == 0
        and old_parts == new_parts,
        "changes_explained": counts["entries_unexplained"] == 0 and accents["not_accented"] == 0,
        "targets": sum(target_changed.values()) == 0
        and counts["merges_holding_a_word"] == 0
        and all(target_new[w] == target_old[w] + target_gained[w] for w in targets),
    }
    summary = {
        "parts": {"r_b": len(old_parts), "new": len(new_parts), "checked": len(jobs)},
        "counts": dict(sorted(counts.items())),
        "rewrites_not_applied": dict(accents),
        "targets": {
            word: {
                "r_b": target_old[word],
                "gained": target_gained[word],
                "new": target_new[word],
                "changed": target_changed[word],
            }
            for word in sorted(targets - held_out)
        },
        "held_out_words_changed_or_gaining": sum(
            1 for word in held_out if target_changed[word] or target_gained[word]
        ),
        "reconciliation": {
            "merges_seen": sum(merged.values()),
            "merges_counted_by_the_builder": sum(c for f, c in built.items() if " " in f),
            "one_token_rewrites_seen": sum(c for f, c in rewritten.items() if f != "spelling"),
            "one_token_counted_by_the_builder": sum(c for f, c in built.items() if " " not in f),
            "spelled_entries_seen": rewritten["spelling"],
            "largest_merge_differences": sorted(
                ((f, merged[f], c) for f, c in built.items() if " " in f and merged[f] != c),
                key=lambda row: -abs(row[1] - row[2]),
            )[:EXAMPLES],
        },
        "unexplained_examples": [row for r in results.values() for row in r["unexplained"]][
            :EXAMPLES
        ],
        "bar": bar,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    key = {}
    with (out / "spot_check.jsonl").open("w", encoding="utf-8") as handle:
        for number, (_, work_id, start, kind, before, after, position) in enumerate(drawn):
            item_id = f"S{number:02d}"
            key[item_id] = {"work_id": work_id, "start": start, "kind": kind}
            text = texts.get(work_id) or ""
            item = {
                "id": item_id,
                "r_b": before,
                "new": after,
                "position": position,
                "text": text[max(0, start - SNIPPET) : start + SNIPPET],
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "spot_check_key.json").write_text(json.dumps(key, indent=1) + "\n")
    shown = {name: summary[name] for name in ("counts", "rewrites_not_applied", "bar")}
    sys.stdout.write(json.dumps(shown, indent=1) + "\n")
    sys.stdout.write(f"spot check: {len(drawn)} items in {out / 'spot_check.jsonl'}\n")


if __name__ == "__main__":
    main()
