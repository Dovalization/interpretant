"""Round Q (Q1-Q7): sense lists induced from DWUG's texts, three labellers.

Committed before any step was run (docs/predictions_labels.md). The steps, in order:

    uv run --extra labels python scripts/score_labels_induced.py evidence     # step A, local
    uv run --extra labels python scripts/score_labels_induced.py consolidate  # step B, Opus
    uv run --extra labels python scripts/score_labels_induced.py coverage     # Sonnet on half A
    uv run --extra labels python scripts/score_labels_induced.py revise       # Opus, one pass
    uv run --extra labels python scripts/score_labels_induced.py freeze       # lists written
    uv run --extra labels python scripts/score_labels_induced.py label sonnet|jev|substitutes
    uv run python scripts/score_labels_induced.py score

The rule-built arm, from step A's evidence alone:

    uv run --extra labels python scripts/score_labels_induced.py rules
    uv run --extra labels python scripts/score_labels_induced.py label sonnet|jev|substitutes rules
    uv run python scripts/score_labels_induced.py score_rules

Everything under data/labels/round_q/ except the label files stays local; the frozen lists and
community map go to data/sense_lists/. No DWUG text is committed.
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
import score_labels_benchmark as round_y  # noqa: E402

from interpretant.drift.induction import (  # noqa: E402
    Community,
    Substituter,
    cleaned,
    communities,
    kept_communities,
    neighbours,
    placed,
    stop_words,
    wiktionary_entries,
    wordnet_entries,
)
from interpretant.drift.labels import (  # noqa: E402
    FAILED,
    OTHER,
    REFUSED,
    REPEAT_SUFFIX,
    UNCLEAR,
    BatchLabeller,
    Item,
    JevLabeller,
    Sense,
    SenseList,
    load_sense_lists,
    run_batch,
    shown,
)
from interpretant.drift.pairs import dwug_graded_change  # noqa: E402
from interpretant.predictions import report  # noqa: E402

OUT = Path("data/labels/round_q")
KAIKKI = Path("data/external/kaikki")
LISTS = Path("data/sense_lists/dwug_en_induced.toml")
MAP = Path("data/sense_lists/dwug_en_induced_communities.json")
RULE_LISTS = Path("data/sense_lists/dwug_en_rules.toml")
RULE_MAP = Path("data/sense_lists/dwug_en_rules_communities.json")
ANY_USE = "any_use"
SEED = 2026
CONSOLIDATOR = "claude-opus-5-5"
SONNET = BatchLabeller(model="claude-sonnet-5-5", effort="low")
JEV = JevLabeller(model="jev-1.14.0")
COVERAGE = 0.10
REVISION_USAGES = 40
MAX_SENSES = 8
SHUFFLES = 20
POS = {"nn": ("n", "noun"), "vb": ("v", "verb")}

CONSOLIDATION_PROMPT = """You build the list of senses in which a word is used in a set of texts.
You are given evidence gathered from those texts and from dictionaries: groups of substitute words
that a language model proposed for the word in context (C1, C2, ..., each with its most frequent
substitutes and example usages, the word marked with ⟦ and ⟧), the words most often found right
before or after it (N1, N2, ...), and dictionary glosses from WordNet (W1, ...) and Wiktionary
(K1, ...).

Write the list of senses the texts show, following these rules.
1. Every sense cites at least one substitute group (C...) or neighbour (N...) as evidence. A
   dictionary entry (W..., K...) may be cited as well, never alone. Leave out senses the texts do
   not show.
2. Use the fewest senses a careful reader could tell apart in context, and merge senses a reader
   could not reliably tell apart. Between 1 and 8 senses.
3. Map every substitute group to the one sense it shows, or to "none" if it shows no sense of the
   word (for example, a group of unrelated or generic words).
4. Give each sense a short descriptive key in lowercase letters and underscores, a one-line
   definition, and a short invented example sentence that is not copied from the usages."""

REVISION_PROMPT = """You extend a list of the senses in which a word is used in a set of texts.
You are given the current list, and usages (U1, U2, ...) that the list did not fit, each with
substitute words a language model proposed for the word in context; the word is marked with ⟦
and ⟧. Add the senses these usages show that the list lacks, following these rules.
1. Every sense you add cites at least one of the usages (U...) as evidence.
2. Do not change or repeat the current senses. Add only senses a careful reader could tell apart
   from them in context. The list may hold at most 8 senses in all, so add at most the number
   stated.
3. Give each added sense a short descriptive key in lowercase letters and underscores, a
   one-line definition, and a short invented example sentence that is not copied from the
   usages. Adding nothing is a valid answer."""

SENSE_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "definition": {"type": "string"},
        "example": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["key", "definition", "example", "evidence"],
    "additionalProperties": False,
}
CONSOLIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "senses": {"type": "array", "items": SENSE_SCHEMA},
        "community_map": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"community": {"type": "string"}, "sense": {"type": "string"}},
                "required": ["community", "sense"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["senses", "community_map"],
    "additionalProperties": False,
}
REVISION_SCHEMA = {
    "type": "object",
    "properties": {"added": {"type": "array", "items": SENSE_SCHEMA}},
    "required": ["added"],
    "additionalProperties": False,
}
KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def halves() -> dict[str, str]:
    """Each kept DWUG usage's half, "A" or "B", as the predictions fix it."""
    found = round_y.usages()
    rng = random.Random(SEED)
    half = dict.fromkeys(found, "A")
    for word in sorted({lemma for lemma, _, _ in found.values()}):
        for period in (1, 2):
            keys = sorted(k for k in found if found[k][0] == word and found[k][1] == period)
            rng.shuffle(keys)
            half.update(dict.fromkeys(keys[: len(keys) // 2], "B"))
    return half


def substitutes() -> dict[str, list[str]]:
    """Every kept usage's cleaned substitutes, computed once and stored."""
    path = OUT / "substitutes.json"
    if path.exists():
        stored: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
        return stored
    found = sorted(round_y.usages().items())
    raw = Substituter().fillers([usage for _, (_, _, usage) in found])
    stops = stop_words()
    result = {
        key: cleaned(fillers, lemma.rsplit("_", 1)[0], u.text[u.start : u.end], stops)
        for (key, (lemma, _, u)), fillers in zip(found, raw, strict=True)
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


def kaikki(word: str) -> Path:
    path = KAIKKI / f"{word}.jsonl"
    if not path.exists():
        KAIKKI.mkdir(parents=True, exist_ok=True)
        url = f"https://kaikki.org/dictionary/English/meaning/{word[0]}/{word[:2]}/{word}.jsonl"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            path.write_bytes(response.read())
        with (KAIKKI / "downloaded.tsv").open("a", encoding="utf-8") as log:
            log.write(f"{word}\t{date.today().isoformat()}\t{url}\n")
    return path


def evidence() -> None:
    """Step A: each word's evidence from half A, written to data/labels/round_q/evidence/."""
    found = round_y.usages()
    half = halves()
    found_substitutes = substitutes()
    stops = stop_words()
    folder = OUT / "evidence"
    folder.mkdir(parents=True, exist_ok=True)
    for lemma in sorted({lemma for lemma, _, _ in found.values()}):
        word, part = lemma.rsplit("_", 1)
        keys = sorted(k for k in found if found[k][0] == lemma and half[k] == "A")
        lists = [found_substitutes[k] for k in keys]
        kept = kept_communities(lists, communities(lists))
        groups = []
        for group in kept:
            members = [k for k, s in zip(keys, lists, strict=True) if placed(s, kept) == group.name]
            examples = sorted(
                keys, key=lambda k: (-len(set(found_substitutes[k]) & group.members), k)
            )[:3]
            groups.append(
                {
                    "name": group.name,
                    "members": sorted(group.members),
                    "top": list(group.top),
                    "usages": len(members),
                    "examples": [shown(found[k][2]) for k in examples],
                }
            )
        record = {
            "lemma": lemma,
            "word": word,
            "pos": POS[part][1],
            "half_a": len(keys),
            "communities": groups,
            "neighbours": neighbours([found[k][2] for k in keys], stops),
            "wordnet": wordnet_entries(word, POS[part][0]),
            "wiktionary": wiktionary_entries(kaikki(word), POS[part][1]),
        }
        (folder / f"{lemma}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"{lemma}: {len(groups)} communities, {len(record['wiktionary'])} Wiktionary senses")


def evidence_record(lemma: str) -> dict[str, object]:
    record: dict[str, object] = json.loads(
        (OUT / "evidence" / f"{lemma}.json").read_text(encoding="utf-8")
    )
    return record


def evidence_text(record: dict[str, object]) -> str:
    lines = [f"Word: {record['word']} ({record['pos']}), {record['half_a']} usages", ""]
    lines.append("Substitute groups:")
    for group in record["communities"]:  # type: ignore[attr-defined]
        lines.append(
            f"{group['name']}: {', '.join(group['top'])} "
            f"(holds {group['usages']} of {record['half_a']} usages)"
        )
        lines += [f"  example: {text}" for text in group["examples"]]
    lines += ["", "Neighbours:"]
    lines += [
        f"N{i}: {item} ({count})"
        for i, (item, count) in enumerate(record["neighbours"], start=1)  # type: ignore[arg-type]
    ]
    lines += ["", "WordNet:"]
    lines += [f"W{i}: {gloss}" for i, gloss in enumerate(record["wordnet"], start=1)]  # type: ignore[arg-type]
    lines += ["", "Wiktionary:"]
    lines += [f"K{i}: {gloss}" for i, gloss in enumerate(record["wiktionary"], start=1)]  # type: ignore[arg-type]
    return "\n".join(lines)


def request(custom_id: str, system: str, text: str, schema: dict[str, object]) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "params": {
            "model": CONSOLIDATOR,
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": text}],
            "output_config": {
                "effort": "high",
                "format": {"type": "json_schema", "schema": schema},
            },
        },
    }


def valid_list(answer: dict[str, object], record: dict[str, object]) -> str | None:
    """Why a consolidated list breaks the rules, or None if it keeps them."""
    senses = answer.get("senses")
    if not isinstance(senses, list) or not 1 <= len(senses) <= MAX_SENSES:
        return "needs 1 to 8 senses"
    keys = [sense["key"] for sense in senses]
    if len(set(keys)) != len(keys) or not all(KEY.match(key) for key in keys):
        return "keys must be distinct, lowercase and underscores"
    if not all(any(e[:1] in "CN" for e in sense["evidence"]) for sense in senses):
        return "a sense cites no substitute group or neighbour"
    names = [group["name"] for group in record["communities"]]  # type: ignore[attr-defined]
    mapped = {entry["community"]: entry["sense"] for entry in answer["community_map"]}  # type: ignore[attr-defined]
    if sorted(mapped) != sorted(names) or not set(mapped.values()) <= {*keys, "none"}:
        return "every substitute group must map to one listed sense or to none"
    return None


def consolidate() -> None:
    """Step B: one Opus request per word; a word whose list breaks the rules is sent once more."""
    lemmas = sorted(path.stem for path in (OUT / "evidence").glob("*.json"))
    records = {lemma: evidence_record(lemma) for lemma in lemmas}
    accepted: dict[str, dict[str, object]] = {}
    for attempt in (1, 2):
        pending = [lemma for lemma in lemmas if lemma not in accepted]
        if not pending:
            break
        stored = run_batch(
            [
                request(
                    f"q{i}",
                    CONSOLIDATION_PROMPT,
                    evidence_text(records[lemma]),
                    CONSOLIDATION_SCHEMA,
                )
                for i, lemma in enumerate(pending)
            ],
            OUT / "raw" / f"consolidate_{attempt}",
            {"lemmas": pending},
        )
        by_id = {str(r["custom_id"]): r for r in stored}
        for i, lemma in enumerate(pending):
            result = by_id.get(f"q{i}", {})
            try:
                answer = json.loads(str(result.get("text")))
            except json.JSONDecodeError:
                print(f"{lemma}: attempt {attempt} unreadable")
                continue
            problem = valid_list(answer, records[lemma])
            if problem is None:
                accepted[lemma] = answer
            else:
                print(f"{lemma}: attempt {attempt} rejected, {problem}")
    (OUT / "consolidated.json").write_text(json.dumps(accepted, indent=1), encoding="utf-8")
    missing = [lemma for lemma in lemmas if lemma not in accepted]
    print(f"{len(accepted)} lists accepted; without a list: {missing or 'none'}")


def as_sense_list(word: str, senses: list[dict[str, object]]) -> SenseList:
    return SenseList(
        word, tuple(Sense(str(s["key"]), str(s["definition"]), str(s["example"])) for s in senses)
    )


def draft_lists() -> dict[str, SenseList]:
    consolidated = json.loads((OUT / "consolidated.json").read_text(encoding="utf-8"))
    return {
        lemma: as_sense_list(lemma.rsplit("_", 1)[0], answer["senses"])
        for lemma, answer in consolidated.items()
    }


def coverage() -> None:
    """Sonnet labels every half-A usage with the draft lists."""
    found = round_y.usages()
    half = halves()
    lists = draft_lists()
    work = [
        (lists[found[k][0]], Item(k, found[k][2]))
        for k in sorted(found)
        if half[k] == "A" and found[k][0] in lists
    ]
    labels = SONNET.label(work, OUT / "raw" / "coverage")
    (OUT / "coverage_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    for lemma in sorted(lists):
        own = [label for key, label in labels.items() if found[key][0] == lemma]
        share = statistics.mean(label in (OTHER, UNCLEAR) for label in own)
        print(f"{lemma}: {share:.0%} other or unclear on half A")


def revise() -> None:
    """One Opus pass for each word whose half-A usages are over 10% other or unclear."""
    found = round_y.usages()
    found_substitutes = substitutes()
    consolidated = json.loads((OUT / "consolidated.json").read_text(encoding="utf-8"))
    labels = json.loads((OUT / "coverage_labels.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    pending, texts = [], []
    for lemma in sorted(consolidated):
        own = {k: v for k, v in labels.items() if found[k][0] == lemma}
        unfit = sorted(k for k, v in own.items() if v in (OTHER, UNCLEAR))
        room = MAX_SENSES - len(consolidated[lemma]["senses"])
        if len(unfit) <= COVERAGE * len(own) or room == 0:
            continue
        sample = rng.sample(unfit, min(REVISION_USAGES, len(unfit)))
        current = "\n".join(
            f"- {s['key']}: {s['definition']}" for s in consolidated[lemma]["senses"]
        )
        usages = "\n".join(
            f"U{i}: {shown(found[k][2])}\n    substitutes: {', '.join(found_substitutes[k][:10])}"
            for i, k in enumerate(sample, start=1)
        )
        word = lemma.rsplit("_", 1)[0]
        texts.append(
            f"Word: {word}\n\nCurrent senses:\n{current}\n\nYou may add at most {room}.\n\n"
            f"Usages the list did not fit:\n{usages}"
        )
        pending.append(lemma)
    if not pending:
        print("no word needs a revision")
        (OUT / "revised.json").write_text(json.dumps(consolidated, indent=1), encoding="utf-8")
        return
    stored = run_batch(
        [request(f"r{i}", REVISION_PROMPT, text, REVISION_SCHEMA) for i, text in enumerate(texts)],
        OUT / "raw" / "revise",
        {"lemmas": pending},
    )
    by_id = {str(r["custom_id"]): r for r in stored}
    for i, lemma in enumerate(pending):
        try:
            added = json.loads(str(by_id.get(f"r{i}", {}).get("text")))["added"]
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"{lemma}: revision unreadable, list kept as it was")
            continue
        senses = consolidated[lemma]["senses"]
        keys = {s["key"] for s in senses}
        for sense in added:
            cites_usage = any(str(e).startswith("U") for e in sense["evidence"])
            fresh = sense["key"] not in keys and KEY.match(sense["key"])
            if cites_usage and fresh and len(senses) < MAX_SENSES:
                senses.append(sense)
                keys.add(sense["key"])
        print(f"{lemma}: {len(senses)} senses after revision")
    (OUT / "revised.json").write_text(json.dumps(consolidated, indent=1), encoding="utf-8")


def freeze() -> None:
    """The frozen lists and the community map, written to data/sense_lists/."""
    revised = json.loads((OUT / "revised.json").read_text(encoding="utf-8"))
    lines = [
        "# Round Q's sense lists (docs/predictions_labels.md), induced from half A of DWUG EN by",
        "# scripts/score_labels_induced.py; not edited by hand. Evidence ids refer to",
        "# data/labels/round_q/evidence/ (C substitute groups, N neighbours, W WordNet, K",
        "# Wiktionary, U usages in the revision pass).",
        "",
    ]
    community_map = {}
    for lemma, answer in sorted(revised.items()):
        lines += [f"[{lemma}]", f"word = {json.dumps(lemma.rsplit('_', 1)[0])}"]
        for sense in answer["senses"]:
            lines += [
                f"[[{lemma}.senses]]",
                f"key = {json.dumps(sense['key'])}",
                f"definition = {json.dumps(sense['definition'])}",
                f"example = {json.dumps(sense['example'])}",
                f"evidence = {json.dumps(sense['evidence'])}",
            ]
        lines.append("")
        record = evidence_record(lemma)
        community_map[lemma] = {
            "communities": {
                group["name"]: group["members"]
                for group in record["communities"]  # type: ignore[attr-defined]
            },
            "map": {entry["community"]: entry["sense"] for entry in answer["community_map"]},
        }
    LISTS.write_text("\n".join(lines), encoding="utf-8")
    MAP.write_text(json.dumps(community_map, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{len(revised)} lists frozen")


def rules() -> None:
    """The rule-built lists: every kept community of step A becomes one sense, described by its
    substitutes, with no language model; a word with no kept community gets one sense, the word
    in any use. Written to data/sense_lists/ with their community map."""
    lines = [
        "# Round Q's rule-built sense lists (docs/predictions_labels.md): every kept substitute",
        "# community of step A is one sense, described by its ten most frequent substitutes;",
        "# generated by scripts/score_labels_induced.py, no language model, not edited by hand.",
        "",
    ]
    community_map = {}
    for path in sorted((OUT / "evidence").glob("*.json")):
        record = evidence_record(path.stem)
        groups = record["communities"]  # type: ignore[assignment]
        lines += [f"[{path.stem}]", f"word = {json.dumps(record['word'])}"]
        senses: dict[str, str] = {}
        for group in groups:  # type: ignore[attr-defined]
            key = "_".join(group["top"][:2])
            while key in senses.values():
                key = f"{key}_{group['name'].lower()}"
            senses[group["name"]] = key
            definition = "the sense in which the word could be replaced by " + ", ".join(
                group["top"]
            )
            lines += [
                f"[[{path.stem}.senses]]",
                f"key = {json.dumps(key)}",
                f"definition = {json.dumps(definition)}",
                'example = "no example"',
            ]
        entry: dict[str, object] = {
            "communities": {g["name"]: g["members"] for g in groups},  # type: ignore[attr-defined]
            "map": senses,
        }
        if not senses:
            lines += [
                f"[[{path.stem}.senses]]",
                f"key = {json.dumps(ANY_USE)}",
                'definition = "the word in any use"',
                'example = "no example"',
            ]
            entry["default"] = ANY_USE
        lines.append("")
        community_map[path.stem] = entry
    RULE_LISTS.write_text("\n".join(lines), encoding="utf-8")
    RULE_MAP.write_text(json.dumps(community_map, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{len(community_map)} rule-built lists written")


def half_b_work(lists_path: Path) -> list[tuple[SenseList, Item]]:
    found = round_y.usages()
    half = halves()
    lists = load_sense_lists(lists_path)
    items = [
        (lists[found[k][0]], Item(k, found[k][2]))
        for k in sorted(found)
        if half[k] == "B" and found[k][0] in lists
    ]
    rng = random.Random(SEED)
    repeats = [
        (senses, Item(item.key + REPEAT_SUFFIX, item.usage))
        for senses, item in rng.sample(items, round(round_y.REPEATS * len(items)))
    ]
    return items + repeats


def substitute_labels(map_path: Path) -> dict[str, str]:
    """The labeller with no generative model: a usage's community, read through the map; a word
    with no kept community takes the map's default, if it has one."""
    found = round_y.usages()
    half = halves()
    found_substitutes = substitutes()
    maps = json.loads(map_path.read_text(encoding="utf-8"))
    labels = {}
    for key in sorted(found):
        lemma = found[key][0]
        if half[key] != "B" or lemma not in maps:
            continue
        groups = [
            Community(name, frozenset(members), ())
            for name, members in maps[lemma]["communities"].items()
        ]
        groups.sort(key=lambda g: int(g.name[1:]))
        name = placed(found_substitutes[key], groups)
        default = maps[lemma].get("default", "none")
        sense = maps[lemma]["map"].get(name, "none") if name else default
        labels[key] = OTHER if sense == "none" else sense
    return labels


def label(name: str, arm: str) -> None:
    """Label half B with one labeller and one arm's lists: "induced" (Opus) or "rules"."""
    lists_path, map_path, suffix = {
        "induced": (LISTS, MAP, ""),
        "rules": (RULE_LISTS, RULE_MAP, "_rules"),
    }[arm]
    if name == "substitutes":
        labels = substitute_labels(map_path)
    else:
        labeller = {"sonnet": SONNET, "jev": JEV}[name]
        labels = labeller.label(half_b_work(lists_path), OUT / "raw" / f"half_b_{name}{suffix}")
    with (OUT / f"{name}{suffix}_labels.tsv").open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t")
        writer.writerow(["usage", "label"])
        writer.writerows(sorted(labels.items()))
    broken = sum(value in (FAILED, REFUSED) for value in labels.values())
    print(f"{name} ({arm}): {len(labels)} labels; failed or refused {broken}")


def read(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as table:
        return {row["usage"]: row["label"] for row in csv.DictReader(table, delimiter="\t")}


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> float:
    """Spearman of x and y with z partialled out: ranks, linear residuals on z's ranks."""
    rx, ry, rz = (rankdata(values) for values in (x, y, z))

    def residual(values: np.ndarray) -> np.ndarray:
        slope, intercept = np.polyfit(rz, values, 1)
        return np.asarray(values - (slope * rz + intercept))

    return float(np.corrcoef(residual(rx), residual(ry))[0, 1])


def measures(labels: dict[str, str], lists: dict[str, SenseList]) -> tuple[float, float, float]:
    """On half B: Spearman with the graded change, the partial Spearman controlling for the mean
    shuffled divergence, and the mean adjusted Rand index with DWUG's clusters."""
    found = round_y.usages()
    half = halves()
    graded = dwug_graded_change(round_y.ROOT)
    words = sorted(graded)
    keys_of = {w: [k for k in sorted(found) if found[k][0] == w and half[k] == "B"] for w in words}
    half_b = {k: found[k] for w in words for k in keys_of[w]}
    real = round_y.divergences(labels, half_b, lists)
    noise = []
    for word in words:
        values = []
        for seed in range(SHUFFLES):
            rng = random.Random(seed)
            periods = [found[k][1] for k in keys_of[word]]
            rng.shuffle(periods)
            moved = {k: (word, p, found[k][2]) for k, p in zip(keys_of[word], periods, strict=True)}
            values.append(round_y.divergences(labels, moved, lists)[word])
        noise.append(statistics.mean(values))
    x = [real[w] for w in words]
    target = [graded[w] for w in words]
    rho = float(spearmanr(x, target).statistic)
    partial = partial_spearman(x, target, noise)
    cluster_of = round_y.clusters()
    rand = [
        adjusted_rand_score([cluster_of[k] for k in keys_of[w]], [labels[k] for k in keys_of[w]])
        for w in words
        if len({cluster_of[k] for k in keys_of[w]}) >= 2
    ]
    return rho, partial, float(np.mean(rand))


def score() -> None:
    induced = load_sense_lists(LISTS)
    written = load_sense_lists(round_y.LISTS)
    half = halves()
    found = {}
    for name in ("sonnet", "jev", "substitutes"):
        path = OUT / f"{name}_labels.tsv"
        if not path.exists():
            continue
        labels = read(path)
        originals = {k: v for k, v in labels.items() if not k.endswith(REPEAT_SUFFIX)}
        found[name] = originals
        rho, partial, rand = measures(originals, induced)
        unfit = statistics.mean(v in (OTHER, UNCLEAR) for v in originals.values())
        broken = statistics.mean(v in (FAILED, REFUSED) for v in originals.values())
        print(f"{name}: {len(originals)} half-B usages; failed or refused {broken:.1%}")
        report(
            f"Q1 ({name}) the induced lists rank the benchmark's changes",
            rho >= 0.70,
            f"Spearman {rho:.3f} on half B (needed 0.70; gold clusters 0.966)",
        )
        if name != "substitutes":
            hand = read(round_y.OUT / f"{name}_labels.tsv")
            hand_b = {k: v for k, v in hand.items() if half.get(k) == "B"}
            rho_y, _, rand_y = measures(hand_b, written)
            report(
                f"Q2 ({name}) induced lists do as well as hand-written ones",
                rho >= rho_y - 0.10,
                f"induced {rho:.3f}, round Y's labels on half B {rho_y:.3f} (at most 0.10 below)",
            )
        report(
            f"Q3 ({name}) the measure tracks change beyond variety",
            partial >= 0.40,
            f"partial Spearman {partial:.3f} (needed 0.40; gold clusters 0.849)",
        )
        report(
            f"Q4 ({name}) the lists cover the texts",
            unfit <= COVERAGE,
            f"{unfit:.1%} of half-B usages other or unclear (at most 10%)",
        )
        if name != "substitutes":
            repeats = [k for k in labels if k.endswith(REPEAT_SUFFIX)]
            consistency = statistics.mean(
                labels[k] == labels[k.removesuffix(REPEAT_SUFFIX)] for k in repeats
            )
            report(
                f"Q5 ({name}) the labels are consistent",
                consistency >= 0.85,
                f"{consistency:.1%} of {len(repeats)} repeats labelled the same (needed 85%)",
            )
            report(
                f"Q6 ({name}) grouping",
                "descriptive",
                f"mean ARI on half B {rand:.3f}; round Y's labels on half B {rand_y:.3f}",
            )
        else:
            report(f"Q6 ({name}) grouping", "descriptive", f"mean ARI on half B {rand:.3f}")
    names = sorted(found)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = sorted(set(found[first]) & set(found[second]))
            agreement = statistics.mean(found[first][k] == found[second][k] for k in shared)
            report(
                f"Q7 {first} and {second} agree",
                "descriptive",
                f"same label on {agreement:.1%} of {len(shared)} half-B usages",
            )


def score_rules() -> None:
    """The rule-built arm (R1-R5), beside the induced arm for the same labeller."""
    rule_lists = load_sense_lists(RULE_LISTS)
    induced = load_sense_lists(LISTS)
    found = {}
    for name in ("sonnet", "jev", "substitutes"):
        path = OUT / f"{name}_rules_labels.tsv"
        if not path.exists():
            continue
        labels = read(path)
        originals = {k: v for k, v in labels.items() if not k.endswith(REPEAT_SUFFIX)}
        found[name] = originals
        rho, partial, rand = measures(originals, rule_lists)
        unfit = statistics.mean(v in (OTHER, UNCLEAR) for v in originals.values())
        report(
            f"R1 ({name}) rule-built lists rank the benchmark's changes",
            rho >= 0.70,
            f"Spearman {rho:.3f} on half B (needed 0.70)",
        )
        induced_path = OUT / f"{name}_labels.tsv"
        if induced_path.exists():
            induced_labels = {
                k: v for k, v in read(induced_path).items() if not k.endswith(REPEAT_SUFFIX)
            }
            rho_q, _, _ = measures(induced_labels, induced)
            report(
                f"R2 ({name}) the language model's list-building changes little",
                rho >= rho_q - 0.10,
                f"rule-built {rho:.3f}, induced by Opus {rho_q:.3f} (at most 0.10 below)",
            )
        report(
            f"R3 ({name}) the measure tracks change beyond variety",
            partial >= 0.40,
            f"partial Spearman {partial:.3f} (needed 0.40)",
        )
        report(
            f"R4 ({name}) the lists cover the texts",
            unfit <= COVERAGE,
            f"{unfit:.1%} of half-B usages other or unclear (at most 10%)",
        )
        if name != "substitutes":
            repeats = [k for k in labels if k.endswith(REPEAT_SUFFIX)]
            consistency = statistics.mean(
                labels[k] == labels[k.removesuffix(REPEAT_SUFFIX)] for k in repeats
            )
            report(
                f"R5 ({name}) the labels are consistent",
                consistency >= 0.85,
                f"{consistency:.1%} of {len(repeats)} repeats labelled the same (needed 85%)",
            )
        report(f"R6 ({name}) grouping", "descriptive", f"mean ARI on half B {rand:.3f}")
    names = sorted(found)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = sorted(set(found[first]) & set(found[second]))
            agreement = statistics.mean(found[first][k] == found[second][k] for k in shared)
            report(
                f"R7 {first} and {second} agree on rule-built lists",
                "descriptive",
                f"same label on {agreement:.1%} of {len(shared)} half-B usages",
            )


if __name__ == "__main__":
    steps = {
        "evidence": evidence,
        "consolidate": consolidate,
        "coverage": coverage,
        "revise": revise,
        "freeze": freeze,
        "rules": rules,
        "score": score,
        "score_rules": score_rules,
    }
    if sys.argv[1] == "label":
        label(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "induced")
    else:
        steps[sys.argv[1]]()
