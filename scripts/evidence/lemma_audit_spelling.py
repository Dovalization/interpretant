import re
import sys
import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(22); pl.Config.set_tbl_hide_dataframe_shape(True)
d = sys.argv[1]
e = pl.read_parquet(f"{d}/entries.parquet")
counts = dict(pl.read_parquet(f"{d}/type_counts.parquet").iter_rows())
mapped = dict(zip(e["token"], e["lemma"]))
sp = e.filter(pl.col("rule") == "spelling")
print("spelling entries, most frequent:")
print(sp.sort("count", descending=True).select("token", "lemma", "count", "lemma_count", "cosine").head(25))
# British -is- forms whose -iz- twin exists in the corpus but which stay unmerged (after mapping)
final = lambda w: mapped.get(w, w)
rows = []
for word, count in counts.items():
    if count < 20 or not re.search(r"is(e|es|ed|ing|ation|ations|er|ers)$", word):
        continue
    twin = re.sub(r"is(e|es|ed|ing|ation|ations|er|ers)$", lambda m: "iz" + m.group(1), word)
    if counts.get(twin, 0) >= 20 and final(word) != final(twin):
        rows.append((word, final(word), twin, final(twin), count, counts[twin]))
left = pl.DataFrame(rows, schema=["british", "maps_to", "american", "american_maps_to", "count", "american_count"], orient="row")
print(f"\nBritish -is- forms not joined with an -iz- twin in the corpus: {left.height} types, {left['count'].sum():,} tokens")
print(left.sort("count", descending=True).head(40))
