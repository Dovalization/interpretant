"""Share of abstracts holding a target word, and DocBin bytes per token on a sample."""
import gzip, json, random, re
from pathlib import Path

import spacy
from spacy.tokens import DocBin

words = [w.strip().replace("_", " ") for w in open("data/words_of_interest.txt") if w.strip() and not w.startswith("#")]
pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")s?\b", re.I)
total = holding = 0
sample = []
rng = random.Random(3)
for path in sorted(Path("data/raw/openalex").glob("*/*.jsonl.gz")):
    for line in gzip.open(path, "rt", encoding="utf-8"):
        text = json.loads(line)["abstract"]
        total += 1
        holding += bool(pattern.search(text))
        if rng.random() < 0.002:
            sample.append(text)
print(f"{holding:,} of {total:,} abstracts ({holding / total:.0%}) contain one of the {len(words)} target words")
nlp = spacy.load("en_core_web_sm", exclude=["parser"])
nlp.enable_pipe("senter")
docs = list(nlp.pipe(sample, batch_size=64))
tokens = sum(len(d) for d in docs)
for attrs, label in ((None, "default attributes"), (["ORTH", "LEMMA", "TAG", "POS", "MORPH", "SENT_START", "ENT_IOB", "ENT_TYPE"], "tokens, lemmas, tags, sentences, entities")):
    bin_ = DocBin(attrs=attrs) if attrs else DocBin()
    for d in docs:
        bin_.add(d)
    size = len(bin_.to_bytes())
    print(f"DocBin ({label}): {size / tokens:.1f} bytes per token over {len(docs):,} abstracts; "
          f"whole corpus (~250M tokens) ≈ {size / tokens * 250e6 / 1e9:.1f} GB")
