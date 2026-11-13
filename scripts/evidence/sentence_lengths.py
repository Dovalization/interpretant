"""Length of the sentences that hold target words, with spaCy's senter and abbreviation rules."""
import gzip, json, random, re
from pathlib import Path

import numpy as np
import spacy
from spacy.symbols import ORTH
from transformers import AutoTokenizer

targets = ["attention", "network", "learning", "model", "platform", "alignment", "prompt",
           "embodiment", "representation", "consciousness"]
pattern = re.compile(r"\b(" + "|".join(targets) + r")s?\b", re.I)
rng = random.Random(7)
paths = sorted(Path("data/raw/openalex").glob("*/*.jsonl.gz"))
texts = []
for path in paths:
    lines = gzip.open(path, "rt", encoding="utf-8").readlines()
    for line in rng.sample(lines, min(60, len(lines))):
        text = json.loads(line)["abstract"]
        if pattern.search(text):
            texts.append(text)
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "lemmatizer"])
nlp.enable_pipe("senter")
for abbreviation in ["cf.", "Fig.", "Figs.", "Sect.", "Eq.", "Eqs.", "al.", "approx.", "ca.", "resp.", "Ref.", "Refs.", "Tab.", "Ch.", "pp.", "vol.", "no.", "i.e.", "e.g."]:
    nlp.tokenizer.add_special_case(abbreviation, [{ORTH: abbreviation}])
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")  # XL-LEXEME's tokenizer family
words, subwords = [], []
for doc in nlp.pipe(texts, batch_size=64):
    for sentence in doc.sents:
        if pattern.search(sentence.text):
            words.append(sum(1 for token in sentence if not token.is_punct))
            subwords.append(len(tokenizer(sentence.text)["input_ids"]))
words, subwords = np.array(words), np.array(subwords)
print(f"{len(texts):,} abstracts sampled, {len(words):,} target sentences")
print("words per sentence: median", int(np.median(words)), " 5th pct", int(np.percentile(words, 5)), " 95th pct", int(np.percentile(words, 95)), " max", words.max())
for bound in (5, 8, 10):
    print(f"  under {bound} words: {np.mean(words < bound):.1%}")
for bound in (60, 100):
    print(f"  over {bound} words: {np.mean(words > bound):.1%}")
print(f"  over 128 subword tokens (XL-LEXEME's window): {np.mean(subwords > 128):.1%}")
