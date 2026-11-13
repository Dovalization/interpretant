"""Multiple mentions per abstract, and where the first mention sits."""
import gzip, json, random, re
from collections import Counter
from pathlib import Path

targets = ["attention", "network", "learning", "model", "platform", "alignment", "prompt",
           "embodiment", "representation", "consciousness"]
sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
rng = random.Random(1)
stats = {t: Counter() for t in targets}
seen = set()
for raw in (Path("data/raw/openalex"), Path("data/raw/openalex_full")):
    for path in sorted(raw.glob("*/*.jsonl.gz")):
        for line in gzip.open(path, "rt", encoding="utf-8"):
            record = json.loads(line)
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            text = record["abstract"]
            sentences = sentence_end.split(text)
            for target in targets:
                pattern = re.compile(rf"\b{target}s?\b", re.I)
                hits = [i for i, s in enumerate(sentences) for _ in pattern.finditer(s)]
                if not hits:
                    continue
                s = stats[target]
                s["abstracts"] += 1
                s["mentions"] += len(hits)
                if len(hits) > 1:
                    s["multi"] += 1
                    s["first_in_first_sentence"] += hits[0] == 0
                    s["random_in_first_sentence"] += rng.choice(hits) == 0
for target in targets:
    s = stats[target]
    print(f"{target:15} abstracts {s['abstracts']:>7,}  mentions/abstract {s['mentions'] / s['abstracts']:.2f}  "
          f"several mentions {s['multi'] / s['abstracts']:5.1%}  | among those, in sentence 1: "
          f"first {s['first_in_first_sentence'] / max(s['multi'], 1):5.1%} vs random {s['random_in_first_sentence'] / max(s['multi'], 1):5.1%}")
