"""How often targets occur inside hyphenated compounds, as head or as modifier."""
import gzip, json, re
from collections import Counter
from pathlib import Path

targets = ["attention", "network", "learning", "model", "platform", "alignment", "prompt",
           "intelligence", "embodiment", "representation", "interface", "agent"]
stats = {t: Counter() for t in targets}
examples = {t: {"head": Counter(), "modifier": Counter()} for t in targets}
token = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
seen = set()
for raw in (Path("data/raw/openalex"), Path("data/raw/openalex_full")):
    for path in sorted(raw.glob("*/*.jsonl.gz")):
        for line in gzip.open(path, "rt", encoding="utf-8"):
            record = json.loads(line)
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            for word in token.findall(record["abstract"].lower()):
                parts = word.split("-")
                for target in targets:
                    forms = (target, target + "s")
                    if not any(part in forms for part in parts):
                        continue
                    if len(parts) == 1:
                        stats[target]["plain"] += 1
                    elif parts[-1] in forms:
                        stats[target]["head"] += 1
                        examples[target]["head"][word] += 1
                    else:
                        stats[target]["modifier"] += 1
                        examples[target]["modifier"][word] += 1
for target in targets:
    s = stats[target]; total = sum(s.values())
    print(f"{target:15} {total:>8,}  head {s['head'] / total:5.1%}  modifier {s['modifier'] / total:5.1%}")
    print("    head:    ", ", ".join(f"{w} {n}" for w, n in examples[target]["head"].most_common(5)))
    print("    modifier:", ", ".join(f"{w} {n}" for w, n in examples[target]["modifier"].most_common(5)))
