"""Score the junk gate's calibration and build its labels (`docs/junk_gate.md`).

Reads a draw folder's key (`interpretant draw-junk-gate`) and the judges' answers in its
`verdicts/` folder: `judge_A_part_*.json` and `judge_B_part_*.json`, each mapping an item id to the
codebook's answer (`note`, `verdict`, `kind`, `sentence`, `damaged`), and the adjudicator's
(`opus_adjudication.json` or its part files). The adjudicator settles the items where one judge
says `junk` and the other does not, or one says the marked sentence is `other` and the other does
not; on every other difference the first judge's answer stands. A record is junk when its final
verdict is `junk`; `abstract` and `mixed` are not. Run from the repository root as
`python scripts/review/score_junk_gate.py MODE ...`:

- `pending FOLDER`: writes the items the adjudicator must see, with both judges' answers, to
  `FOLDER/adjudicate.jsonl` (it quotes the corpus and stays out of git);
- `calibrate RUN3_FOLDER`: step 1's three bars against the author's verdicts on run 3
  (`data/review/run3/verdicts.jsonl`: `not_abstract` is junk, `unsure` is left out, r018 is junk),
  and once the author's second look is saved (`data/review/junk_gate/run3_second_look.json`), the
  judges' confusion table against it; writes `data/review/junk_gate/calibration.json`;
- `labels SAMPLE_FOLDER [CONFIRMATION_FOLDER]`: writes the catalog's `junk_labels` file from the
  final answers without their notes, refusing answers made under another codebook than the
  committed one, or still waiting for the adjudicator;
- `fresh [--redraw]`: step 3's 30 records for the author to judge blind
  (`data/review/junk_gate/fresh.json`), and their agreement with the judges once the author's
  answers are saved (`fresh_author.json`); `--redraw` is the redraw after a failed bar
  (`fresh_redraw.json`, `fresh_redraw_author.json`).

The author's answers map an item id to `abstract`, `junk` or `unsure`, or to the answer the review
page saves (`{"verdict": ...}`). Words appear by their reported names. The gate's final scoring is
not here yet: it comes with the classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from interpretant.catalog.config import read_catalog
from interpretant.catalog.junk_gate import (
    JunkLabel,
    JunkLabels,
    Run3Verdict,
    load_junk_labels,
    load_run3_verdicts,
    save_junk_labels,
)
from interpretant.review import verdicts
from interpretant.review.judging import load_items
from interpretant.review.sampling_plans import ConfusionMatrix

Answer = verdicts.Answer
CODEBOOK = Path("curated/judging/junk_gate/prompt.md")
GATE_FOLDER = Path("data/review/junk_gate")
# Step 1 (docs/junk_gate.md, "The calibration, in three steps"): the reference's sizes once r018
# counts as junk and the unsure records are left out, and the three bars on them
RUN3_RECORDS = 197
RUN3_JUNK = 31
RUN3_ABSTRACTS = 166
AGREEMENT_BAR = 178
JUNK_FOUND_BAR = 27
ABSTRACTS_CALLED_JUNK_BAR = 6
# The author counts r018 as junk (2026-11-15), whatever the verdicts file says
RECODED_AS_JUNK = frozenset({"r018"})
FRESH_SEED = 20261403
REDRAW_SEED = 20261409
FRESH_AGREEMENT_BAR = 27


@dataclass(frozen=True)
class FreshGroup:
    """One group of step 3's fresh records: the set it comes from, the judges' call, its size."""

    name: str
    item_set: str
    is_junk: bool
    size: int


FRESH_GROUPS = (
    FreshGroup("labelled_junk", "labelled", is_junk=True, size=12),
    FreshGroup("labelled_abstract", "labelled", is_junk=False, size=8),
    FreshGroup("enrichment_abstract", "enrichment", is_junk=False, size=10),
)


def needs_adjudicator(first: Answer, second: Answer) -> bool:
    """Whether two answers differ on junk, or on whether the marked sentence is another text's."""
    return is_junk(first) != is_junk(second) or (first["sentence"] == "other") != (
        second["sentence"] == "other"
    )


def is_junk(answer: Answer) -> bool:
    """Whether an answer calls the record junk: ``mixed`` and ``abstract`` are not."""
    return bool(answer["verdict"] == "junk")


def load_parts(folder: Path, name: str) -> dict[str, Answer]:
    """Answers merged over ``NAME.json`` and ``NAME_part_*.json``, not the judges' manifests.

    Raises:
        SystemExit: if an item is answered twice.
    """
    try:
        return verdicts.load_parts(folder, name)
    except verdicts.VerdictsError as exc:
        raise SystemExit(str(exc)) from exc


def load_key(folder: Path) -> dict[str, dict[str, Any]]:
    """The draw's key entries by item id; a confirmation key lists its cells beside them."""
    key: dict[str, Any] = json.loads((folder / "key.json").read_text())
    return dict(key["items"]) if "cells" in key else key


def final_answers(folder: Path, key: Mapping[str, Any]) -> tuple[dict[str, Answer], int, list[str]]:
    """Each item's final answer, how many the judges agreed on, and the items still pending.

    Raises:
        SystemExit: if a judge left an item without an answer.
    """
    first = load_parts(folder / "verdicts", "judge_A")
    second = load_parts(folder / "verdicts", "judge_B")
    missing = sorted(set(key) - set(first) | set(key) - set(second))
    if missing:
        raise SystemExit(f"{len(missing)} items lack a judge's answer, such as {missing[:5]}")
    return verdicts.merge_verdicts(
        {item_id: first[item_id] for item_id in key},
        {item_id: second[item_id] for item_id in key},
        load_parts(folder / "verdicts", "opus_adjudication"),
        differ=needs_adjudicator,
    )


def write_pending(folder: Path) -> str:
    """Write the items the adjudicator must see, with both judges' answers; a summary line."""
    key = load_key(folder)
    _, agreed, pending = final_answers(folder, key)
    rows = verdicts.build_pending(
        load_items(folder),
        load_parts(folder / "verdicts", "judge_A"),
        load_parts(folder / "verdicts", "judge_B"),
        pending,
    )
    with (folder / "adjudicate.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return f"judges agree on {agreed} of {len(key)}; {len(pending)} items for the adjudicator"


def build_reference(author: Sequence[Run3Verdict]) -> dict[str, bool]:
    """Whether the author called each of run 3's records junk; the unsure records are left out."""
    return {
        verdict.id: verdict.id in RECODED_AS_JUNK or verdict.verdict == "not_abstract"
        for verdict in author
        if verdict.verdict != "unsure" or verdict.id in RECODED_AS_JUNK
    }


def calibrate(
    folder: Path, author: Sequence[Run3Verdict], second_look: Mapping[str, str]
) -> dict[str, Any]:
    """Step 1's bars: the judges' final verdicts on run 3 against the author's.

    ``second_look`` holds the author's second look at disputed records (empty before it is
    made): the reference for the judges' confusion table takes it where given and the first
    verdict elsewhere.

    Raises:
        SystemExit: if an item is still pending, or the reference is not step 1's 197 records
            (31 junk, 166 abstracts).
    """
    final, agreed, pending = final_answers(folder, load_key(folder))
    if pending:
        raise SystemExit(f"{len(pending)} items still need the adjudicator: run pending first")
    reference = build_reference(author)
    junk_count = sum(reference.values())
    if (len(reference), junk_count) != (RUN3_RECORDS, RUN3_JUNK):
        raise SystemExit(
            f"the reference holds {len(reference)} records and {junk_count} junk, not step 1's "
            f"{RUN3_RECORDS} and {RUN3_JUNK}"
        )
    unjudged = sorted(set(reference) - set(final))
    if unjudged:
        raise SystemExit(f"{len(unjudged)} of run 3's records were not judged: {unjudged[:5]}")
    called = {item_id: is_junk(final[item_id]) for item_id in reference}
    agree = sum(called[item_id] == is_junk_record for item_id, is_junk_record in reference.items())
    found = sum(called[item_id] for item_id, is_junk_record in reference.items() if is_junk_record)
    false_junk = sum(
        called[item_id] for item_id, is_junk_record in reference.items() if not is_junk_record
    )
    bars = {
        "agreement": _bar(agree, RUN3_RECORDS, AGREEMENT_BAR, passes=agree >= AGREEMENT_BAR),
        "junk_found": _bar(found, RUN3_JUNK, JUNK_FOUND_BAR, passes=found >= JUNK_FOUND_BAR),
        "abstracts_called_junk": _bar(
            false_junk,
            RUN3_ABSTRACTS,
            ABSTRACTS_CALLED_JUNK_BAR,
            passes=false_junk <= ABSTRACTS_CALLED_JUNK_BAR,
        ),
    }
    results: dict[str, Any] = {
        "records": len(reference),
        "judges_agreed": agreed,
        "bars": bars,
        "passes": all(bar["passes"] for bar in bars.values()),
        "disputed": sorted(
            item_id for item_id in reference if called[item_id] != reference[item_id]
        ),
    }
    if second_look:
        results["second_look"] = compare_with_second_look(final, reference, second_look)
    return results


def compare_with_second_look(
    final: Mapping[str, Answer], reference: Mapping[str, bool], second_look: Mapping[str, str]
) -> dict[str, Any]:
    """The judges' confusion table against the second look where given, the first verdict elsewhere.

    The author's ``unsure`` on a second look leaves the first verdict standing.
    """
    looked = {
        item_id: answer == "junk"
        for item_id, answer in second_look.items()
        if answer in ("abstract", "junk")
    }
    corrected = dict(reference) | looked
    pairs = [
        (is_junk_record, is_junk(final[item_id])) for item_id, is_junk_record in corrected.items()
    ]
    confusion = ConfusionMatrix(
        true_positives=sum(truth and called for truth, called in pairs),
        false_negatives=sum(truth and not called for truth, called in pairs),
        false_positives=sum(called and not truth for truth, called in pairs),
        true_negatives=sum(not truth and not called for truth, called in pairs),
    )
    return {"records": len(pairs), "second_looks": len(looked), "confusion": asdict(confusion)}


def check_codebook(folder: Path, codebook_sha256: str) -> None:
    """Refuse answers made under another codebook: every judge's manifest must name this one.

    Raises:
        SystemExit: if the folder holds no manifest, or one names another codebook.
    """
    manifests = sorted((folder / "verdicts").glob("*.manifest.json"))
    if not manifests:
        raise SystemExit(f"{folder} holds no judge's manifest: the codebook judged with is unknown")
    foreign = [
        path.name
        for path in manifests
        if json.loads(path.read_text()).get("codebook_sha256") != codebook_sha256
    ]
    if foreign:
        raise SystemExit(f"answers made under another codebook than the committed one: {foreign}")


def build_labels(folders: Sequence[Path], codebook_sha256: str) -> JunkLabels:
    """The labels of every item drawn into ``folders``: each final answer without its note.

    Raises:
        SystemExit: if a folder's answers were made under another codebook, an item is still
            pending, or two folders hold the same item id.
    """
    labels: dict[str, JunkLabel] = {}
    for folder in folders:
        check_codebook(folder, codebook_sha256)
        key = load_key(folder)
        final, _, pending = final_answers(folder, key)
        if pending:
            raise SystemExit(
                f"{len(pending)} items of {folder} still need the adjudicator: run pending first"
            )
        repeated = sorted(set(key) & set(labels))
        if repeated:
            raise SystemExit(f"items drawn twice: {repeated[:5]}")
        for item_id, entry in key.items():
            answer = final[item_id]
            labels[item_id] = JunkLabel(
                set=entry["set"],
                stratum=entry.get("stratum"),
                work_id=entry["work_id"],
                part=entry["part"],
                decade=entry["decade"],
                word=entry["word"],
                content_key=entry["content_key"],
                verdict=answer["verdict"],
                kind=answer["kind"],
                sentence=answer["sentence"],
                damaged=answer["damaged"],
            )
    return JunkLabels(codebook_sha256=codebook_sha256, items=labels)


def draw_fresh(labels: JunkLabels, seed: int) -> list[dict[str, str]]:
    """Step 3's records: each group's items with the smallest sha256 of ``seed:item id``, shuffled.

    12 of L the judges call junk, 8 of L and 10 of E they call abstracts (``mixed`` included). An
    item whose record is already drawn is passed over, so the author judges 30 records.

    Raises:
        SystemExit: if a group has too few items.
    """
    ordered = sorted(
        labels.items, key=lambda item_id: hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()
    )
    drawn: list[dict[str, str]] = []
    work_ids: set[str] = set()
    for group in FRESH_GROUPS:
        members = [
            item_id
            for item_id in ordered
            if labels.items[item_id].set == group.item_set
            and (labels.items[item_id].verdict == "junk") == group.is_junk
        ]
        picked: list[str] = []
        for item_id in members:
            if len(picked) == group.size:
                break
            if labels.items[item_id].work_id not in work_ids:
                picked.append(item_id)
                work_ids.add(labels.items[item_id].work_id)
        if len(picked) < group.size:
            raise SystemExit(f"{group.name} has {len(picked)} records, not {group.size}")
        drawn += [{"id": item_id, "group": group.name} for item_id in picked]
    random.Random(seed).shuffle(drawn)
    return drawn


def score_fresh(
    drawn: Sequence[Mapping[str, str]], labels: JunkLabels, author: Mapping[str, str]
) -> dict[str, Any]:
    """The author's answers on the fresh records against the judges' final verdicts.

    An ``unsure`` answer agrees with neither verdict.

    Raises:
        SystemExit: if the author's answers are not on exactly the drawn items.
    """
    item_ids = [entry["id"] for entry in drawn]
    if set(author) != set(item_ids):
        raise SystemExit("the author's answers are not on the drawn items")
    agreeing = [
        item_id
        for item_id in item_ids
        if author[item_id] in ("abstract", "junk")
        and (author[item_id] == "junk") == (labels.items[item_id].verdict == "junk")
    ]
    return {
        "items": len(item_ids),
        "agree": len(agreeing),
        "unsure": sum(author[item_id] == "unsure" for item_id in item_ids),
        "bar": FRESH_AGREEMENT_BAR,
        "passes": len(agreeing) >= FRESH_AGREEMENT_BAR,
        "disagreements": sorted(set(item_ids) - set(agreeing)),
    }


def load_author_answers(path: Path) -> dict[str, str]:
    """The author's answers by item id: each a verdict, or the review page's answer holding one."""
    if not path.exists():
        return {}
    answers: dict[str, Any] = json.loads(path.read_text())
    return {
        item_id: answer["verdict"] if isinstance(answer, dict) else answer
        for item_id, answer in answers.items()
    }


def compute_committed_sha256(path: Path) -> str:
    """The sha256 of a file as committed at HEAD."""
    committed = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"], capture_output=True, check=True
    ).stdout
    return hashlib.sha256(committed).hexdigest()


def run_pending(options: argparse.Namespace) -> str:
    """The ``pending`` mode."""
    return write_pending(options.folder)


def run_calibrate(options: argparse.Namespace) -> str:
    """The ``calibrate`` mode."""
    author = load_run3_verdicts(read_catalog())
    second_look = load_author_answers(GATE_FOLDER / "run3_second_look.json")
    results = calibrate(options.folder, author, second_look)
    _write_json(GATE_FOLDER / "calibration.json", results)
    lines = [
        f"{name}: {bar['count']} of {bar['of']} (bar {bar['bar']}, "
        f"{'passes' if bar['passes'] else 'fails'})"
        for name, bar in results["bars"].items()
    ]
    lines.append(f"{len(results['disputed'])} records for the second look")
    if "second_look" in results:
        lines.append(f"against the second look: {results['second_look']['confusion']}")
    return "\n".join(lines)


def run_labels(options: argparse.Namespace) -> str:
    """The ``labels`` mode."""
    folders = [options.sample, *([options.confirmation] if options.confirmation else [])]
    labels = build_labels(folders, compute_committed_sha256(CODEBOOK))
    path = save_junk_labels(read_catalog(), labels)
    return f"{len(labels.items)} labels written to {path}"


def run_fresh(options: argparse.Namespace) -> str:
    """The ``fresh`` mode."""
    seed, stem = (REDRAW_SEED, "fresh_redraw") if options.redraw else (FRESH_SEED, "fresh")
    labels = load_junk_labels(read_catalog())
    drawn = draw_fresh(labels, seed)
    results: dict[str, Any] = {"seed": seed, "items": drawn}
    author = load_author_answers(GATE_FOLDER / f"{stem}_author.json")
    summary = f"{len(drawn)} fresh records drawn with seed {seed}"
    if author:
        results["author"] = score_fresh(drawn, labels, author)
        verdict = "passes" if results["author"]["passes"] else "fails"
        summary += f"; the author agrees on {results['author']['agree']} ({verdict})"
    _write_json(GATE_FOLDER / f"{stem}.json", results)
    return summary


MODES: dict[str, Callable[[argparse.Namespace], str]] = {
    "pending": run_pending,
    "calibrate": run_calibrate,
    "labels": run_labels,
    "fresh": run_fresh,
}


def main() -> None:
    """Run one mode and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("pending", help="the adjudicator's items").add_argument("folder", type=Path)
    modes.add_parser("calibrate", help="step 1's bars").add_argument("folder", type=Path)
    labels = modes.add_parser("labels", help="the catalog's junk_labels file")
    labels.add_argument("sample", type=Path)
    labels.add_argument("confirmation", type=Path, nargs="?")
    fresh = modes.add_parser("fresh", help="step 3's fresh records")
    fresh.add_argument("--redraw", action="store_true", help="the redraw after a failed bar")
    options = parser.parse_args()
    sys.stdout.write(MODES[options.mode](options) + "\n")


def _bar(count: int, of: int, bar: int, *, passes: bool) -> dict[str, Any]:
    return {"count": count, "of": of, "bar": bar, "passes": passes}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")


if __name__ == "__main__":
    main()
