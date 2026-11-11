"""Exploratory: a long-list of candidate words for the XL-LEXEME scan, flagged by both embedding
seeds (within-field company change, across-field difference) or as arrivals, without artefacts."""
import json, sys
from collections import defaultdict
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP
D = Path(sys.argv[1]); a = json.load(open(D / "candidates.json")); b = json.load(open(D / "candidates43.json"))
ENGLISH = {w.strip().lower() for w in open("/usr/share/dict/words")}
SHORTLIST = ["presence", "strength", "fusion", "agency", "interpretation", "image"]
TARGETS = {"attention", "prompt", "platform", "network", "alignment", "component", "having", "hallucination", "transformer", "cloud", "embedding"}
CONTROLS = {l.strip() for l in open("data/control_words_rung3.txt") if l.strip() and not l.startswith("#")}
ARTEFACTS = {"abstract", "abstracts", "keywords", "keyword", "figure", "fig", "dumps", "braindumps", "abstrak"}
def idx(c, kind):
    out = {}
    for w, es in c[kind].items():
        for e in es: out[(w,) + tuple(e[1:4])] = e[0]
    return out
wa, wb, xa, xb = idx(a, "within"), idx(b, "within"), idx(a, "across"), idx(b, "across")
within = defaultdict(float)
for w, es in a["within"].items():
    for s, f, e, l, cosd, jac, ce, cl in es:
        s2 = wb.get((w, f, e, l))
        if s2 and min(s, s2) >= 0.98 and min(ce, cl) >= 80: within[w] = max(within[w], min(s, s2))
across = defaultdict(int)
for w, es in a["across"].items():
    for s, f1, f2, d, cosd, c1, c2 in es:
        s2 = xb.get((w, f1, f2, d))
        if s2 and min(s, s2) >= 0.97 and min(c1, c2) >= 150 and d >= 2000: across[w] += 1
M = Path("models/twec_grid_unlemmatised/openalex"); arrivals = {}
for f in sorted({p.stem.rsplit("_", 1)[0] for p in M.glob("*.kv")}):
    for e, l in [(1990, 2000), (2000, 2010), (2010, 2020)]:
        A, B = KeyedVectors.load(str(M / f"{f}_{e}.kv")), KeyedVectors.load(str(M / f"{f}_{l}.kv"))
        for w in B.index_to_key[:20000]:
            if not (w.isalpha() and len(w) > 2 and w not in STOP): continue
            cb = B.get_vecattr(w, "count"); ce = A.get_vecattr(w, "count") if w in A.key_to_index else 0
            if cb >= 150 and ce <= 25:
                nbrs = [x for x, _ in B.most_similar(w, topn=10)]
                if sum(x in ENGLISH for x in nbrs) >= 5:
                    arrivals[w] = max(arrivals.get(w, 0), cb / (ce + 5))
ok = lambda w: w not in ARTEFACTS and w not in TARGETS and w not in CONTROLS
pick_within = [w for w, _ in sorted(within.items(), key=lambda x: -x[1]) if ok(w)][:80]
pick_across = [w for w, _ in sorted(across.items(), key=lambda x: -x[1]) if ok(w) and w not in pick_within][:60]
pick_arrivals = [w for w, _ in sorted(arrivals.items(), key=lambda x: -x[1]) if ok(w) and w not in pick_within and w not in pick_across][:40]
longlist = {"shortlist": SHORTLIST, "within": pick_within, "across": pick_across, "arrivals": pick_arrivals}
json.dump(longlist, open(D / "longlist.json", "w"), indent=1)
allw = set(SHORTLIST) | set(pick_within) | set(pick_across) | set(pick_arrivals)
print(f"long-list: {len(allw)} words (within {len(pick_within)}, across {len(pick_across)}, arrivals {len(pick_arrivals)}; shortlist inside: {all(w in allw or w in SHORTLIST for w in SHORTLIST)})")
for k in ("within", "across", "arrivals"): print(f"\n{k}: " + ", ".join(longlist[k]))
