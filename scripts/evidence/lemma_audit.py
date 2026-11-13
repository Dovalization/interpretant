"""Audit of the voted lemma map: every mapped entry and the plural-like types left unmapped."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import polars as pl
from gensim.models import Word2Vec

from interpretant.corpus.files import source_texts
from interpretant.corpus.lemmas import (
    PROPER_TAGS,
    _known,
    _lemmatizer,
    count_types,
    load_votes,
    read_overrides,
    voted_lemma,
)

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
COUNTS_CACHE = OUT / "type_counts.parquet"

if COUNTS_CACHE.exists():
    counts = Counter(dict(pl.read_parquet(COUNTS_CACHE).iter_rows()))
else:
    counts = count_types(source_texts(Path("data/processed"), ("openalex", "openalex_full")))
    pl.DataFrame(list(counts.items()), schema=["type", "count"], orient="row").write_parquet(
        COUNTS_CACHE
    )
total_tokens = sum(counts.values())

votes_frame = pl.read_parquet("data/processed/lemma_votes.parquet")
votes = load_votes(Path("data/processed/lemma_votes.parquet"))
overrides = read_overrides(Path("data/lemma_overrides.tsv"))
mapping = pl.read_csv("data/processed/lemmas_voted.tsv", separator="\t", quote_char=None)
mapped = dict(zip(mapping["token"], mapping["lemma"], strict=True))
rules = dict(zip(mapping["token"], mapping["rule"], strict=True))

compass = Word2Vec.load("models/twec_grid_unlemmatised/compass.model").wv
lookups = _lemmatizer().lookups
noun_exceptions = lookups.get_table("lemma_exc")["noun"]
verb_exceptions = lookups.get_table("lemma_exc")["verb"]


def cosine(a: str, b: str) -> float | None:
    if a in compass.key_to_index and b in compass.key_to_index:
        return float(compass.similarity(a, b))
    return None


def readings(token: str) -> dict[str, float]:
    rows = votes.get(token, [])
    total = sum(c for _, _, c in rows)
    proper = sum(c for t, _, c in rows if t in PROPER_TAGS)
    common = total - proper
    nns = sum(c for t, _, c in rows if t == "NNS")
    vbz = sum(c for t, _, c in rows if t == "VBZ")
    nns_self = sum(c for t, lem, c in rows if t == "NNS" and lem == token)
    named = Counter()
    for t, lem, c in rows:
        if t in ("NNS", "VBZ"):
            named[lem] += c
    nns_lemmas = Counter()
    vbz_lemmas = Counter()
    for t, lem, c in rows:
        if t == "NNS":
            nns_lemmas[lem] += c
        if t == "VBZ":
            vbz_lemmas[lem] += c
    return {
        "total": total,
        "proper_share": proper / total if total else 0.0,
        "nns": nns,
        "vbz": vbz,
        "inflected_share": (nns + vbz) / common if common else 0.0,
        "nns_self_share": nns_self / nns if nns else 0.0,
        "spacy_lemma": named.most_common(1)[0][0] if named else "",
        "nns_lemma": nns_lemmas.most_common(1)[0][0] if nns_lemmas else "",
        "vbz_lemma": vbz_lemmas.most_common(1)[0][0] if vbz_lemmas else "",
    }


# ---- every mapped entry, with its signals ----
rows = []
for token, lemma in mapped.items():
    signal = readings(token)
    rows.append(
        {
            "token": token,
            "lemma": lemma,
            "rule": rules[token],
            "count": counts[token],
            "lemma_count": counts[lemma],
            "lemma_ratio": counts[lemma] / counts[token] if counts[token] else 0.0,
            "lemma_known": _known(lemma),
            "chained": lemma in mapped,
            "cosine": cosine(token, lemma),
            **signal,
        }
    )
entries = pl.DataFrame(rows)
entries.write_parquet(OUT / "entries.parquet")

# Baseline: cosine between regular plurals and singulars over frequent, clean merges.
clean = entries.filter(
    (pl.col("rule") == "plural") & (pl.col("count") >= 200) & pl.col("cosine").is_not_null()
)
quantiles = clean.select(
    [pl.col("cosine").quantile(q).alias(f"q{int(q * 100)}") for q in (0.05, 0.1, 0.25, 0.5)]
)
print("cosine of plural/singular, frequent plural merges:", quantiles.to_dicts()[0])

# ---- unmapped types that look like inflections ----
unmapped_rows = []
for token, rows_ in votes.items():
    if token in mapped or token in overrides or counts[token] < 5:
        continue
    signal = readings(token)
    if signal["inflected_share"] < 0.6 or signal["nns"] + signal["vbz"] < 5:
        continue
    lemma = signal["spacy_lemma"]
    if not lemma or lemma == token:
        continue
    without_gate = voted_lemma(token, votes, None)
    unmapped_rows.append(
        {
            "token": token,
            "spacy_lemma": lemma,
            "count": counts[token],
            "lemma_count": counts[lemma],
            "lemma_ratio": counts[lemma] / counts[token],
            "in_noun_exceptions": token in noun_exceptions,
            "in_verb_exceptions": token in verb_exceptions,
            "lemma_known": _known(lemma),
            "blocked_by_gate": without_gate is not None,
            "ungated_lemma": without_gate[0] if without_gate else "",
            "cosine": cosine(token, lemma),
            **{k: v for k, v in signal.items() if k != "spacy_lemma"},
        }
    )
unmapped = pl.DataFrame(unmapped_rows)
unmapped.write_parquet(OUT / "unmapped.parquet")
print(f"mapped entries: {entries.height:,}; unmapped inflection-like types: {unmapped.height:,}")
print(f"total tokens: {total_tokens:,}")
