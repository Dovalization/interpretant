import sys
import polars as pl
pl.Config.set_tbl_rows(45); pl.Config.set_fmt_str_lengths(22); pl.Config.set_tbl_width_chars(200)
pl.Config.set_float_precision(2); pl.Config.set_tbl_hide_dataframe_shape(True)
d = sys.argv[1]
e = pl.read_parquet(f"{d}/entries.parquet")
cols = ["token", "lemma", "rule", "count", "lemma_count", "lemma_ratio", "cosine", "proper_share", "nns_self_share", "spacy_lemma"]
def show(title, frame, n=40, by="count"):
    print(f"\n### {title}: {frame.height} types, {frame['count'].sum():,} tokens")
    print(frame.sort(by, descending=True).select(cols).head(n))
print("rules:", e.group_by("rule").agg(pl.len(), pl.col("count").sum()).sort("rule").rows())
show("A1 rare lemma (lemma < 10% of token or < 5 uses)", e.filter((pl.col("lemma_ratio") < 0.1) | (pl.col("lemma_count") < 5)))
show("A2 mostly proper-noun readings (>50%)", e.filter(pl.col("proper_share") > 0.5))
show("A3 spaCy names a different lemma", e.filter((pl.col("spacy_lemma") != pl.col("lemma")) & (pl.col("spacy_lemma") != "") & (pl.col("rule") != "override") & (pl.col("rule") != "spelling")))
show("A4 noun and verb readings both large, different lemmas", e.filter((pl.col("nns") >= 0.2 * (pl.col("nns") + pl.col("vbz"))) & (pl.col("vbz") >= 0.2 * (pl.col("nns") + pl.col("vbz"))) & (pl.col("nns_lemma") != pl.col("vbz_lemma"))).with_columns(pl.col("nns_lemma"), pl.col("vbz_lemma")))
show("A6 spaCy keeps the plural as its own lemma in >=20% of NNS readings", e.filter((pl.col("nns_self_share") >= 0.2) & (pl.col("rule") != "override")))
show("A7 chained (lemma is itself mapped)", e.filter(pl.col("chained")))
show("A8 lemma not in spaCy's word list", e.filter(~pl.col("lemma_known") & (pl.col("rule") != "override")))
show("A9 short tokens (<=4 chars)", e.filter(pl.col("token").str.len_chars() <= 4))
show("A5 low cosine (< 0.3) plural/singular, both frequent", e.filter((pl.col("cosine") < 0.3) & (pl.col("count") >= 100) & (pl.col("lemma_count") >= 100)), by="count")
