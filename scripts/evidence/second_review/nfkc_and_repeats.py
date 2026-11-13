"""Two checks on raw abstracts.

1. What NFKC changes besides ligatures (step 2.2), and does it fuse words with digits?
2. How often an abstract repeats a sentence, which would give two mentions the same usage id
   under A11 (work id, word, hash of the sentence text and the target's offsets in it).
"""

import gzip
import itertools
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path("data/raw/openalex")
PER_CELL = 4000
TARGETS = ("attention", "network", "platform", "alignment", "prompt", "hallucination", "component")

cells = sorted(ROOT.glob("*/*.jsonl.gz"))
print(f"{len(cells)} cells; reading up to {PER_CELL} abstracts from each", file=sys.stderr)

changed_characters: Counter[str] = Counter()
abstracts = changed_abstracts = 0
fused_examples: Counter[str] = Counter()
fused_abstracts = spacing_mark_abstracts = 0
repeat_abstracts = repeat_with_target = 0
repeat_examples: list[str] = []

SUPERSCRIPT_AFTER_WORD = re.compile(r"[A-Za-z]{3,}[¹²³⁰-⁹₀-₉]+")
SPACING_MARKS = re.compile("[´¨¯¸˘-˝]")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

for cell in cells:
    with gzip.open(cell, "rt", encoding="utf-8") as lines:
        for line in itertools.islice(lines, PER_CELL):
            text = json.loads(line).get("abstract") or ""
            if not text:
                continue
            abstracts += 1
            normal = unicodedata.normalize("NFKC", text)
            if normal != text:
                changed_abstracts += 1
                for character in set(text):
                    if unicodedata.normalize("NFKC", character) != character:
                        changed_characters[character] += 1
            found = SUPERSCRIPT_AFTER_WORD.findall(text)
            if found:
                fused_abstracts += 1
                for match in found[:3]:
                    fused_examples[f"{match} -> {unicodedata.normalize('NFKC', match)}"] += 1
            if SPACING_MARKS.search(text):
                spacing_mark_abstracts += 1
            sentences = [part.strip() for part in SENTENCE.split(text) if len(part.strip()) >= 40]
            repeated = [sentence for sentence, count in Counter(sentences).items() if count > 1]
            if repeated:
                repeat_abstracts += 1
                with_target = [
                    sentence
                    for sentence in repeated
                    if any(re.search(rf"\b{word}s?\b", sentence, re.I) for word in TARGETS)
                ]
                if with_target:
                    repeat_with_target += 1
                    if len(repeat_examples) < 3:
                        repeat_examples.append(with_target[0][:140])

print(f"abstracts read: {abstracts:,}")
print(f"\n1. NFKC changes {changed_abstracts:,} abstracts ({changed_abstracts / abstracts:.1%})")
print("   characters it rewrites, by number of abstracts containing them:")
for character, count in changed_characters.most_common(28):
    name = unicodedata.name(character, "?")
    result = unicodedata.normalize("NFKC", character)
    print(f"     {count:>6}  U+{ord(character):04X} {name[:38]:<38} -> {result!r}")
print(f"   a word followed directly by a superscript or subscript digit: {fused_abstracts:,} abstracts "
      f"({fused_abstracts / abstracts:.2%}); most common:")
for example, count in fused_examples.most_common(12):
    print(f"     {count:>5}  {example}")
print(f"   spacing accents that NFKC turns into a space plus a combining mark: "
      f"{spacing_mark_abstracts:,} abstracts ({spacing_mark_abstracts / abstracts:.2%})")
print(f"\n2. abstracts repeating a sentence of 40+ characters: {repeat_abstracts:,} "
      f"({repeat_abstracts / abstracts:.2%}); with one of seven target words in the repeated "
      f"sentence: {repeat_with_target:,} ({repeat_with_target / abstracts:.3%})")
for example in repeat_examples:
    print(f"     e.g. {example}")
