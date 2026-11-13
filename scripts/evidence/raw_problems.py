"""How many raw abstracts show each text problem."""
import gzip, json, re
from collections import Counter
from pathlib import Path

checks = {
    "mojibake (Ã, â€, Â)": re.compile(r"Ã[\x80-\xbf]|â€|Â[\xa0-\xbf]"),
    "HTML entity (&amp; &lt; &#..;)": re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-f]+);", re.I),
    "HTML/JATS tag (<p>, <jats:…>, <i>)": re.compile(r"</?(?:jats:|p\b|i\b|b\b|sup\b|sub\b|br\b|italic|bold|scp)", re.I),
    "LaTeX command (\\alpha, \\frac)": re.compile(r"\\[a-zA-Z]{2,}"),
    "dollar math ($...$ with \\ ^ _)": re.compile(r"\$[^$]*[\\^_][^$]*\$"),
    "soft hyphen U+00AD": re.compile("­"),
    "curly apostrophe or quote": re.compile("[‘’“”]"),
    "control or format char (Cc/Cf)": re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f​-‏ -‮⁠-⁤﻿]"),
    "ligature (ﬁ ﬂ)": re.compile("[ﬀ-ﬆ]"),
    "URL": re.compile(r"https?://|www\.", re.I),
    "DOI without scheme": re.compile(r"(?<![/\w])10\.\d{4,9}/\S+"),
    "copyright sign or 'all rights reserved'": re.compile(r"©|\ball rights reserved\b", re.I),
}
hits, total, seen = Counter(), 0, set()
for raw in (Path("data/raw/openalex"), Path("data/raw/openalex_full")):
    for path in sorted(raw.glob("*/*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as cell:
            for line in cell:
                record = json.loads(line)
                if record["id"] in seen:
                    continue
                seen.add(record["id"])
                total += 1
                text = record["abstract"]
                for name, pattern in checks.items():
                    if pattern.search(text):
                        hits[name] += 1
print(f"{total:,} unique raw abstracts")
for name in checks:
    print(f"  {name:42} {hits[name]:>9,}  {hits[name] / total:6.2%}")
