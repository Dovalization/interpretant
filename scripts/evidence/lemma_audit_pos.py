import sys
import polars as pl
pl.Config.set_tbl_rows(70); pl.Config.set_fmt_str_lengths(18); pl.Config.set_tbl_width_chars(220)
pl.Config.set_float_precision(2); pl.Config.set_tbl_hide_dataframe_shape(True)
d = sys.argv[1]
e = pl.read_parquet(f"{d}/entries.parquet")
v = pl.read_parquet("data/processed/lemma_votes.parquet").filter(~pl.col("tag").is_in(["NNP", "NNPS"]))
pos = v.group_by("type").agg(
    lemma_total=pl.col("count").sum(),
    nn=pl.col("count").filter(pl.col("tag") == "NN").sum(),
    vb=pl.col("count").filter(pl.col("tag").is_in(["VB", "VBP"])).sum(),
    top_tag=pl.col("tag").sort_by("count", descending=True).first(),
)
e = e.join(pos, left_on="lemma", right_on="type", how="left").with_columns(
    match_share=pl.when(pl.col("rule") == "verb").then(pl.col("vb") / pl.col("lemma_total")).otherwise(pl.col("nn") / pl.col("lemma_total"))
)
e.write_parquet(f"{d}/entries_pos.parquet")
cols = ["token", "lemma", "rule", "count", "lemma_count", "cosine", "match_share", "top_tag", "proper_share"]
base = e.filter(pl.col("rule").is_in(["plural", "verb"]))
for lo, hi in [(0, 0.2), (0.2, 0.35), (0.35, 0.5)]:
    band = base.filter((pl.col("match_share") >= lo) & (pl.col("match_share") < hi))
    print(f"\n### singular read as the matching POS {lo:.0%}-{hi:.0%}: {band.height} types, {band['count'].sum():,} tokens")
    print(band.sort("count", descending=True).select(cols).head(45 if lo == 0 else 30))
print("\nmean cosine by band:")
print(base.with_columns(band=pl.col("match_share").cut([0.2, 0.35, 0.5, 0.7])).group_by("band").agg(pl.len(), pl.col("count").sum(), pl.col("cosine").mean()).sort("band"))
