"""Compare the junk gate's local judges with the author and the Claude judges (`docs/junk_gate.md`).

The local trial's two stages, from the answers `interpretant judge-ollama` saved under a draw
folder's `verdicts/local/` (`SETUP_part_*.json` and `SETUP.manifest.json`, one setup per name: a
model, its precision and its machine). Run from the repository root:

- `stage1 RUN3_FOLDER`: each setup's answers on run 3 against the author's verdicts
  (`data/review/run3/verdicts.jsonl`: `not_abstract` is junk, `unsure` is left out, r018 is junk):
  the agreement on the 197, the junk found of 31, the abstracts called junk of 166 and step 1's
  three bars, with its speed and its share on the graphics card. Then every panel (two setups, a
  third settling their junk differences), cascade (a decision model, with the records whose junk
  probability falls in a band sent to a chat model) and jury (the majority of three), each with
  its accuracy and its estimated time for L and E. Setups and combinations go on to stage 2 when
  none is both more accurate (the agreement, then the junk found) and faster, and their time for
  L and E is under 12 hours. Writes `data/review/junk_gate/local_trial_stage1.json`.
- `stage2 SAMPLE_FOLDER SETUP...`: each named setup's answers on L against the Claude judges'
  labels (sensitivity, specificity and agreement), on the author's fresh 30, and for a decision
  model the area under the ROC curve of its junk probability beside the classifier's score's.
  Writes `data/review/junk_gate/local_trial_stage2.json`.

A combination's time: parts on different machines run together, parts on one machine in turn, and
a part that waits for others' answers (a panel's third setup, a cascade's chat model) starts after
them. Setups are named by their verdict files; machines by their manifests' host labels, never an
address.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score

from interpretant.catalog.config import read_catalog
from interpretant.catalog.junk_gate import Run3Verdict, load_junk_labels, load_run3_verdicts
from interpretant.catalog.junk_gate_tables import load_junk_scores
from interpretant.review.sampling_plans import compute_wilson_interval
from interpretant.review.verdicts import Answer, load_parts

GATE_FOLDER = Path("data/review/junk_gate")
LOCAL = Path("verdicts/local")
RECODED_AS_JUNK = frozenset({"r018"})
AGREEMENT_BAR = 178
JUNK_FOUND_BAR = 27
ABSTRACTS_CALLED_JUNK_BAR = 6
# L and E: the items stage 2 judges, and the most hours a setup may take for them
STAGE2_ITEMS = 4600
STAGE2_HOURS = 12.0
CASCADE_BANDS = ((0.2, 0.8), (0.1, 0.9))
FIT_SENSITIVITY = 0.80
FIT_SPECIFICITY = 0.97


@dataclass(frozen=True)
class Setup:
    """One local setup: its answers by item id, its kind, machine and seconds per item."""

    name: str
    answers: dict[str, Answer]
    kind: str
    machine: str
    seconds_per_item: float
    gpu_share: float | None


@dataclass(frozen=True)
class Candidate:
    """A setup or a combination: its junk calls by item id and its time for L and E."""

    name: str
    shape: str
    calls: dict[str, bool]
    hours: float


def is_junk(answer: Answer) -> bool:
    """Whether an answer calls the record junk: ``mixed`` and ``abstract`` are not."""
    return bool(answer["verdict"] == "junk")


def junk_probability(answer: Answer) -> float:
    """A decision model's probability that the record is junk (one minus the own-prose answer)."""
    return 1.0 - float(answer["probabilities"]["own_prose"])


def build_reference(author: Sequence[Run3Verdict]) -> dict[str, bool]:
    """Whether the author called each of run 3's records junk; the unsure records are left out."""
    return {
        verdict.id: verdict.id in RECODED_AS_JUNK or verdict.verdict == "not_abstract"
        for verdict in author
        if verdict.verdict != "unsure" or verdict.id in RECODED_AS_JUNK
    }


def load_setups(folder: Path) -> list[Setup]:
    """Every setup whose manifest sits in the folder's ``verdicts/local``, sorted by name."""
    local = folder / LOCAL
    setups = []
    for manifest_path in sorted(local.glob("*.manifest.json")):
        name = manifest_path.name.removesuffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs = manifest["runs"]
        judged = sum(run["items_judged"] for run in runs)
        seconds = sum(run["seconds_total"] for run in runs)
        last = runs[-1]
        setups.append(
            Setup(
                name=name,
                answers=load_parts(local, name),
                kind=manifest["kind"],
                machine=last["host_label"],
                seconds_per_item=seconds / judged if judged else float("inf"),
                gpu_share=(last["size_vram"] / last["size"]) if last.get("size") else None,
            )
        )
    return setups


def score_calls(calls: Mapping[str, bool], reference: Mapping[str, bool]) -> dict[str, Any]:
    """Agreement, junk found and abstracts called junk against a reference, with step 1's bars."""
    shared = [item_id for item_id in reference if item_id in calls]
    agree = sum(calls[item_id] == reference[item_id] for item_id in shared)
    junk = [item_id for item_id in shared if reference[item_id]]
    abstracts = [item_id for item_id in shared if not reference[item_id]]
    found = sum(calls[item_id] for item_id in junk)
    called = sum(calls[item_id] for item_id in abstracts)
    return {
        "records": len(shared),
        "agreement": agree,
        "junk_found": found,
        "junk": len(junk),
        "abstracts_called_junk": called,
        "abstracts": len(abstracts),
        "meets_bars": len(shared) == len(reference)
        and agree >= AGREEMENT_BAR
        and found >= JUNK_FOUND_BAR
        and called <= ABSTRACTS_CALLED_JUNK_BAR,
    }


def hours_for(seconds_per_item: float, share: float = 1.0) -> float:
    """Hours a part takes for its share of L and E."""
    return seconds_per_item * share * STAGE2_ITEMS / 3600


def parallel_hours(parts: Sequence[tuple[str, float]]) -> float:
    """Hours for parts that run at once: together across machines, in turn on one machine."""
    by_machine: dict[str, float] = defaultdict(float)
    for machine, hours in parts:
        by_machine[machine] += hours
    return max(by_machine.values())


def build_candidates(setups: Sequence[Setup], items: Sequence[str]) -> list[Candidate]:
    """Every setup alone, and every panel, cascade and jury of them, on the given items."""
    calls = {setup.name: {item: is_junk(setup.answers[item]) for item in items} for setup in setups}
    alone = {setup.name: hours_for(setup.seconds_per_item) for setup in setups}
    candidates = [
        Candidate(setup.name, "alone", calls[setup.name], alone[setup.name]) for setup in setups
    ]
    by_name = {setup.name: setup for setup in setups}
    for first, second in itertools.combinations(sorted(by_name), 2):
        split = [item for item in items if calls[first][item] != calls[second][item]]
        start = parallel_hours(
            [(by_name[first].machine, alone[first]), (by_name[second].machine, alone[second])]
        )
        for third in sorted(set(by_name) - {first, second}):
            panel = {
                item: calls[third][item] if item in split else calls[first][item] for item in items
            }
            hours = start + hours_for(by_name[third].seconds_per_item, len(split) / len(items))
            candidates.append(
                Candidate(f"panel {first} + {second} / {third}", "panel", panel, hours)
            )
    for trio in itertools.combinations(sorted(by_name), 3):
        jury = {item: sum(calls[name][item] for name in trio) >= 2 for item in items}
        hours = parallel_hours([(by_name[name].machine, alone[name]) for name in trio])
        candidates.append(Candidate(f"jury {' + '.join(trio)}", "jury", jury, hours))
    decisions = [setup for setup in setups if setup.kind == "decision"]
    chats = [setup for setup in setups if setup.kind == "chat"]
    for decision, chat in itertools.product(decisions, chats):
        probability = {item: junk_probability(decision.answers[item]) for item in items}
        for low, high in CASCADE_BANDS:
            band = [item for item in items if low <= probability[item] <= high]
            cascade = {
                item: calls[chat.name][item] if item in band else probability[item] > 0.5
                for item in items
            }
            hours = alone[decision.name] + hours_for(chat.seconds_per_item, len(band) / len(items))
            name = f"cascade {decision.name} [{low}, {high}] -> {chat.name}"
            candidates.append(Candidate(name, "cascade", cascade, hours))
    return candidates


def select_for_stage2(scored: Sequence[dict[str, Any]]) -> list[str]:
    """The candidates no other beats on both accuracy and time, and that fit in 12 hours."""

    def accuracy(entry: Mapping[str, Any]) -> tuple[int, int]:
        return entry["run3"]["agreement"], entry["run3"]["junk_found"]

    return [
        entry["name"]
        for entry in scored
        if entry["hours_for_l_and_e"] <= STAGE2_HOURS
        and not any(
            accuracy(other) > accuracy(entry)
            and other["hours_for_l_and_e"] < entry["hours_for_l_and_e"]
            for other in scored
        )
    ]


def run_stage1(
    run3: Path, author: Sequence[Run3Verdict], second_look: Mapping[str, str]
) -> dict[str, Any]:
    """Stage 1's table: each setup, each combination, and those that go on to stage 2."""
    reference = build_reference(author)
    with_second_look = {
        item: (second_look[item] == "junk") if item in second_look else junk
        for item, junk in reference.items()
    }
    setups = load_setups(run3)
    missing = {setup.name: sorted(set(reference) - set(setup.answers)) for setup in setups}
    incomplete = {name: ids for name, ids in missing.items() if ids}
    if incomplete:
        raise SystemExit(f"setups without an answer to every run 3 record: {sorted(incomplete)}")
    candidates = build_candidates(setups, sorted(reference))
    scored = [
        {
            "name": candidate.name,
            "shape": candidate.shape,
            "hours_for_l_and_e": round(candidate.hours, 3),
            "run3": score_calls(candidate.calls, reference),
            "run3_second_look": score_calls(candidate.calls, with_second_look),
        }
        for candidate in candidates
    ]
    return {
        "setups": [
            {
                "name": setup.name,
                "kind": setup.kind,
                "machine": setup.machine,
                "items_per_minute": round(60 / setup.seconds_per_item, 2),
                "gpu_share": None if setup.gpu_share is None else round(setup.gpu_share, 3),
            }
            for setup in setups
        ],
        "candidates": scored,
        "stage2": select_for_stage2(scored),
    }


def run_stage2(
    sample: Path,
    names: Sequence[str],
    labels: Mapping[str, Mapping[str, Any]],
    fresh_author: Mapping[str, str],
    classifier_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Each named setup against the Claude judges on L, the author's fresh 30 and the classifier."""
    setups = {setup.name: setup for setup in load_setups(sample)}
    unknown = sorted(set(names) - set(setups))
    if unknown:
        raise SystemExit(f"no local answers for {unknown}")
    labelled = {item: label for item, label in labels.items() if label["set"] == "labelled"}
    claude = {item: label["verdict"] == "junk" for item, label in labelled.items()}
    report = {}
    for name in names:
        setup = setups[name]
        shared = [item for item in claude if item in setup.answers]
        local = {item: is_junk(setup.answers[item]) for item in shared}
        report[name] = {
            "kind": setup.kind,
            "machine": setup.machine,
            "items_per_minute": round(60 / setup.seconds_per_item, 2),
            "labelled": _agreement(local, {item: claude[item] for item in shared}),
            "fresh": _fresh_agreement(setup.answers, fresh_author),
        }
        if setup.kind == "decision":
            report[name]["auc"] = _areas(setup, shared, claude, labelled, classifier_scores)
    return report


def _agreement(local: Mapping[str, bool], claude: Mapping[str, bool]) -> dict[str, Any]:
    junk = [item for item in claude if claude[item]]
    abstracts = [item for item in claude if not claude[item]]
    found = sum(local[item] for item in junk)
    cleared = sum(not local[item] for item in abstracts)
    sensitivity = found / len(junk) if junk else None
    specificity = cleared / len(abstracts) if abstracts else None
    return {
        "items": len(claude),
        "agreement": sum(local[item] == claude[item] for item in claude),
        "sensitivity": sensitivity,
        "sensitivity_interval": compute_wilson_interval(found, len(junk)) if junk else None,
        "specificity": specificity,
        "specificity_interval": compute_wilson_interval(cleared, len(abstracts))
        if abstracts
        else None,
        "fit_for_labelling_against_claude": sensitivity is not None
        and specificity is not None
        and sensitivity >= FIT_SENSITIVITY
        and specificity >= FIT_SPECIFICITY,
    }


def _fresh_agreement(answers: Mapping[str, Answer], author: Mapping[str, str]) -> dict[str, int]:
    judged = [item for item in author if item in answers]
    agree = sum(
        author[item] != "unsure" and is_junk(answers[item]) == (author[item] == "junk")
        for item in judged
    )
    return {"items": len(judged), "agree": agree}


def _areas(
    setup: Setup,
    shared: Sequence[str],
    claude: Mapping[str, bool],
    labelled: Mapping[str, Mapping[str, Any]],
    classifier_scores: Mapping[str, float],
) -> dict[str, float | None]:
    truth = [claude[item] for item in shared]
    if len(set(truth)) < 2:
        return {"decision_model": None, "classifier": None}
    probabilities = [junk_probability(setup.answers[item]) for item in shared]
    scored = [item for item in shared if labelled[item]["work_id"] in classifier_scores]
    classifier = (
        float(
            roc_auc_score(
                [claude[item] for item in scored],
                [classifier_scores[labelled[item]["work_id"]] for item in scored],
            )
        )
        if len({claude[item] for item in scored}) == 2
        else None
    )
    return {"decision_model": float(roc_auc_score(truth, probabilities)), "classifier": classifier}


def _read_answers(path: Path) -> dict[str, str]:
    """The author's answers, as plain strings or as a page's ``{verdict, note}`` objects."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        item: value["verdict"] if isinstance(value, dict) else value for item, value in raw.items()
    }


def _write(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return f"wrote {path}"


def main() -> None:
    """Run the chosen stage from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stages = parser.add_subparsers(dest="stage", required=True)
    first = stages.add_parser("stage1", help="every setup and combination on run 3")
    first.add_argument("run3", type=Path)
    second = stages.add_parser("stage2", help="the chosen setups on L and E")
    second.add_argument("sample", type=Path)
    second.add_argument("setups", nargs="+")
    options = parser.parse_args()
    catalog = read_catalog()
    actions: dict[str, Callable[[], str]] = {
        "stage1": lambda: _write(
            GATE_FOLDER / "local_trial_stage1.json",
            run_stage1(
                options.run3,
                load_run3_verdicts(catalog),
                _read_answers(GATE_FOLDER / "run3_second_look.json"),
            ),
        ),
        "stage2": lambda: _write(
            GATE_FOLDER / "local_trial_stage2.json",
            run_stage2(
                options.sample,
                options.setups,
                {
                    item: label.model_dump()
                    for item, label in load_junk_labels(catalog).items.items()
                },
                _read_answers(GATE_FOLDER / "fresh_author.json"),
                dict(load_junk_scores(catalog).select("work_id", "score").iter_rows()),
            ),
        ),
    }
    sys.stdout.write(actions[options.stage]() + "\n")


if __name__ == "__main__":
    main()
