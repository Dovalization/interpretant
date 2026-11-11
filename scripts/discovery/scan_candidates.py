"""Exploratory corpus-wide candidate scan (read-only): words whose company changed within a field
between consecutive decades, and words whose company differs between fields in the same decade,
from the aligned per-cell TWEC models (unlemmatised grid). Frequency-matched percentiles keep rare
words from dominating. Writes candidates.json."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP

M = Path("models/twec_grid_unlemmatised/openalex")
FIELDS = sorted({p.stem.rsplit("_", 1)[0] for p in M.glob("*.kv")})
DECADES = list(range(1960, 2030, 10))
MIN_EARLY, MIN_LATE, MIN_VOCAB, K = 30, 50, 20, 25
CHECK = {"attention", "prompt", "platform", "network", "alignment", "transformer", "cloud", "embedding", "hallucination", "agent", "web", "embodiment", "token", "stream"}

def ok(w): return w.isalpha() and len(w) > 2 and w not in STOP

cells = {}
for f in FIELDS:
    for d in DECADES:
        p = M / f"{f}_{d}.kv"
        if not p.exists(): continue
        kv = KeyedVectors.load(str(p))
        words = [w for w in kv.index_to_key if kv.get_vecattr(w, "count") >= MIN_VOCAB and "_" not in w]
        if len(words) < 2000: continue
        X = np.array([kv[w] for w in words]); X /= np.linalg.norm(X, axis=1, keepdims=True)
        cells[(f, d)] = {"words": words, "index": {w: i for i, w in enumerate(words)}, "X": X,
                         "count": {w: kv.get_vecattr(w, "count") for w in words}}
    print(f, "loaded", flush=True)

def neighbours(cell, ws):
    idx = [cell["index"][w] for w in ws]
    S = cell["X"][idx] @ cell["X"].T
    S[np.arange(len(idx)), idx] = -1
    top = np.argpartition(-S, K, axis=1)[:, :K]
    return [set(cell["words"][j] for j in row) for row in top]

def freq_matched_percentile(values, freqs):
    """Percentile of each value within its frequency decile (higher = more change)."""
    values, freqs = np.asarray(values), np.log(np.asarray(freqs))
    edges = np.quantile(freqs, np.linspace(0, 1, 11)); out = np.zeros(len(values))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs <= hi)
        if m.sum(): out[m] = (np.argsort(np.argsort(values[m])) + 0.5) / m.sum()
    return out

within = defaultdict(list)   # word -> [(score, field, early, late, cos_dist, jaccard, c_early, c_late)]
for f in FIELDS:
    ds = [d for d in DECADES if (f, d) in cells]
    for a, b in zip(ds, ds[1:]):
        A, B = cells[(f, a)], cells[(f, b)]
        ws = [w for w in A["words"] if ok(w) and w in B["index"] and A["count"][w] >= MIN_EARLY and B["count"][w] >= MIN_LATE]
        if len(ws) < 200: continue
        va = A["X"][[A["index"][w] for w in ws]]; vb = B["X"][[B["index"][w] for w in ws]]
        cosd = 1 - np.sum(va * vb, axis=1)
        na, nb = neighbours(A, ws), neighbours(B, ws)
        jac = np.array([len(x & y) / len(x | y) for x, y in zip(na, nb)])
        freqs = [min(A["count"][w], B["count"][w]) for w in ws]
        score = (freq_matched_percentile(cosd, freqs) + freq_matched_percentile(1 - jac, freqs)) / 2
        for w, s, c, j in zip(ws, score, cosd, jac):
            within[w].append((float(s), f, a, b, float(c), float(j), A["count"][w], B["count"][w]))
    print(f, "within done", flush=True)

across = defaultdict(list)   # word -> [(score, field1, field2, decade, cos_dist, c1, c2)]
for d in (2000, 2010, 2020):
    fs = [f for f in FIELDS if (f, d) in cells]
    for i, f1 in enumerate(fs):
        for f2 in fs[i + 1:]:
            A, B = cells[(f1, d)], cells[(f2, d)]
            ws = [w for w in A["words"] if ok(w) and w in B["index"] and A["count"][w] >= 100 and B["count"][w] >= 100]
            if len(ws) < 200: continue
            va = A["X"][[A["index"][w] for w in ws]]; vb = B["X"][[B["index"][w] for w in ws]]
            cosd = 1 - np.sum(va * vb, axis=1)
            score = freq_matched_percentile(cosd, [min(A["count"][w], B["count"][w]) for w in ws])
            for w, s, c in zip(ws, score, cosd):
                across[w].append((float(s), f1, f2, d, float(c), A["count"][w], B["count"][w]))
    print(d, "across done", flush=True)

json.dump({"within": within, "across": across}, open(sys.argv[1], "w"), default=int)
print("words with within-field scores:", len(within), "| across-field:", len(across))
for w in sorted(CHECK):
    best = max(within.get(w, [(0,)]), key=lambda t: t[0])
    print(f"check {w:13s} best within-field {best[:4] if len(best) > 1 else 'n/a'}")
