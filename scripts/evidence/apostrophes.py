"""How often each apostrophe occurs in the cleaned abstracts, and the fused types it leaves."""
import re
from collections import Counter
from pathlib import Path

from interpretant.corpus.preprocessing import preprocess_text
from interpretant.drift.senses import cased_abstracts

MARKS = {"'": "straight", "’": "right curly", "‘": "left curly", "ʼ": "modifier", "´": "acute", "`": "backtick"}
pattern = re.compile(r"([A-Za-z]+)([" + "".join(MARKS) + r"])([A-Za-z]+)")
marks = Counter()
fused = Counter()
abstracts = 0
seen = set()
for raw_dir in (Path("data/raw/openalex"), Path("data/raw/openalex_full")):
    for path in sorted(raw_dir.glob("*/*.jsonl.gz")):
        for work_id, text in cased_abstracts(raw_dir, path.parent.name, int(path.name.split(".")[0])):
            if work_id in seen:
                continue
            seen.add(work_id)
            abstracts += 1
            for left, mark, right in pattern.findall(text):
                marks[MARKS[mark]] += 1
                if mark != "'":
                    fused[preprocess_text(left + mark + right)] += 1
print(f"{abstracts:,} abstracts")
for name, count in marks.most_common():
    print(f"  {name:12} {count:>10,}")
print(f"fused types: {len(fused):,}, tokens {sum(fused.values()):,}")
print(fused.most_common(40))
