import json
import sys
from collections import defaultdict
from pathlib import Path

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOP

D = Path(sys.argv[1])
M = Path("models/twec_grid_unlemmatised/openalex")
with open(D / "candidates.json") as candidates_file:
    c = json.load(candidates_file)
kvs: dict[tuple[str, int], KeyedVectors] = {}


def kv(f: str, d: int) -> KeyedVectors:
    if (f, d) not in kvs:
        kvs[(f, d)] = KeyedVectors.load(str(M / f"{f}_{d}.kv"))
    return kvs[(f, d)]


def nb(f: str, d: int, w: str, n: int = 8) -> list[str]:
    k = kv(f, d)
    return [
        x
        for x, _ in k.most_similar(w, topn=60)
        if x.isalpha() and k.get_vecattr(x, "count") >= 20 and x not in STOP
    ][:n]


# 1. within-field company change: strong score, both cells well populated
rows = []
for w, entries in c["within"].items():
    for s, f, a, b, _cosd, jac, ca, cb in entries:
        if s >= 0.985 and min(ca, cb) >= 100:
            rows.append((s, min(ca, cb), w, f, a, b, jac))
rows.sort(key=lambda r: (-r[0], -r[1]))
seen, within = set(), []
for r in rows:
    if r[2] in seen:
        continue
    seen.add(r[2])
    within.append(r)
# 2. arrivals: rare before, common after, in the same field
arrivals = []
fields = sorted({p.stem.rsplit("_", 1)[0] for p in M.glob("*.kv")})
for f in fields:
    for a, b in [(1990, 2000), (2000, 2010), (2010, 2020)]:
        A, B = kv(f, a), kv(f, b)
        for w in B.index_to_key[:20000]:
            if not (w.isalpha() and len(w) > 2 and w not in STOP):
                continue
            cb = B.get_vecattr(w, "count")
            ca = A.get_vecattr(w, "count") if w in A.key_to_index else 0
            if cb >= 150 and ca <= 25:
                arrivals.append((cb / (ca + 5), cb, ca, w, f, a, b))
arrivals.sort(reverse=True)
# 3. across fields: same decade, different company, flagged in several field pairs
agg = defaultdict(list)
for w, entries in c["across"].items():
    for s, f1, f2, d, _cosd, c1, c2 in entries:
        if s >= 0.97 and min(c1, c2) >= 200 and d >= 2010:
            agg[w].append((s, f1, f2, d))
across = sorted(agg.items(), key=lambda kv_: (-len(kv_[1]), -max(x[0] for x in kv_[1])))

out = {"within": [], "arrivals": [], "across": []}
print("=== WITHIN A FIELD: company changed between decades (top 30) ===")
for s, mn, w, f, a, b, _jac in within[:30]:
    early, late = nb(f, a, w), nb(f, b, w)
    out["within"].append([w, f, a, b, s, mn, early, late])
    print(
        f"{w:16s} {f:13s} {a}s→{b}s  pct {s:.3f}  uses≥{mn:<5d} | "
        f"{', '.join(early[:6])}  →  {', '.join(late[:6])}"
    )
print("\n=== ARRIVALS: rare before, common after (top 30, one per word) ===")
seenw = set()
for _ratio, cb, ca, w, f, a, b in arrivals:
    if w in seenw:
        continue
    seenw.add(w)
    if len(out["arrivals"]) >= 30:
        break
    late = nb(f, b, w, 6)
    out["arrivals"].append([w, f, a, b, ca, cb, late])
    print(f"{w:16s} {f:13s} {a}s→{b}s  {ca:>4d} → {cb:<5d} | {', '.join(late)}")
print("\n=== ACROSS FIELDS: same word, different company, 2010s/2020s (top 25) ===")
for w, flags in across[:25]:
    s, f1, f2, d = max(flags)
    out["across"].append([w, len(flags), f1, f2, d])
    print(
        f"{w:16s} flagged in {len(flags):2d} field pairs; strongest {f1} vs {f2} {d}s | "
        f"{', '.join(nb(f1, d, w, 5))}  ||  {', '.join(nb(f2, d, w, 5))}"
    )
with open(D / "shortlist.json", "w") as shortlist_file:
    json.dump(out, shortlist_file, indent=1, default=int)
