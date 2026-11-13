"""Point 4 of finding 11: what survives a DocBin round trip, and how long does applying lemma
exceptions to cached documents take compared with parsing?"""

import gzip
import itertools
import json
import time

import spacy
from spacy.tokens import Doc, DocBin

Doc.set_extension("sentence_languages", default=None)

nlp = spacy.load("en_core_web_sm", exclude=["parser"])
nlp.enable_pipe("senter")
nlp.tokenizer.add_special_case("cf.", [{"ORTH": "cf.", "NORM": "compare"}])

text = (
    "Our senses of colour differ, cf. earlier work. The motives were unclear. "
    "Attention is all you need."
)
doc = nlp(text)
doc._.sentence_languages = ["en", "en", "en"]
doc.spans["removed"] = [doc[0:2]]
print("before saving:", [s.text[:20] for s in doc.sents], "| cf. norm:", doc[5].norm_,
      "| colour norm:", doc[3].norm_)


def round_trip(**options: object) -> Doc:
    cache = DocBin(**options)
    cache.add(doc)
    return next(DocBin().from_bytes(cache.to_bytes()).get_docs(nlp.vocab))


for label, options in (
    ("default DocBin", {}),
    ("attrs + SENT_START, NORM; store_user_data",
     {"attrs": ["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_KB_ID", "LEMMA",
                "MORPH", "POS", "SENT_START", "NORM"], "store_user_data": True}),
):
    loaded = round_trip(**options)
    sentences = (
        [s.text[:20] for s in loaded.sents] if loaded.has_annotation("SENT_START") else "LOST"
    )
    print(f"\n{label}:")
    print("  sentences:", sentences)
    print("  cf. norm:", loaded[5].norm_, "| colour norm:", loaded[3].norm_)
    print("  sentence languages:", loaded._.sentence_languages)
    print("  span group 'removed':", [span.text for span in loaded.spans.get("removed", [])])
    print("  lemmas:", [(token.text, token.lemma_) for token in loaded if token.text in
                        ("senses", "motives")])

# Applying lemma exceptions to cached documents
ruler = nlp.get_pipe("attribute_ruler").__class__(nlp.vocab, name="lemma_exceptions")
ruler.add([[{"LOWER": "senses", "TAG": "NNS"}]], {"LEMMA": "sense"})
ruler.add([[{"LOWER": "motives", "TAG": "NNS"}]], {"LEMMA": "motive"})
fixed = ruler(round_trip(attrs=["ORTH", "TAG", "LEMMA", "POS", "MORPH", "SENT_START", "NORM"]))
print("\nafter exceptions on the cached doc:",
      [(token.text, token.lemma_) for token in fixed if token.text in ("senses", "motives")])

# Timing on 2,000 real abstracts
with gzip.open("data/raw/openalex/psychology/2010.jsonl.gz", "rt", encoding="utf-8") as cell:
    texts = [json.loads(line)["abstract"] for line in itertools.islice(cell, 2000)]
start = time.perf_counter()
docs = list(nlp.pipe(texts, batch_size=64))
parse_seconds = time.perf_counter() - start
cache = DocBin(attrs=["ORTH", "TAG", "LEMMA", "POS", "MORPH", "SENT_START", "NORM"])
for parsed in docs:
    cache.add(parsed)
payload = cache.to_bytes()
start = time.perf_counter()
for cached in DocBin().from_bytes(payload).get_docs(nlp.vocab):
    ruler(cached)
apply_seconds = time.perf_counter() - start
print(f"\n2,000 abstracts: parsing {parse_seconds:.1f} s; loading the cache and applying the "
      f"exceptions {apply_seconds:.1f} s ({parse_seconds / apply_seconds:.0f}x faster)")
