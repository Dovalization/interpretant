import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(30)
votes = pl.read_parquet("data/processed/lemma_votes.parquet")
mapped = pl.read_csv("data/processed/lemmas_voted.tsv", separator="\t")["type"].to_list() if False else \
    pl.read_csv("data/processed/lemmas_voted.tsv", separator="\t", has_header=True).get_column(pl.read_csv("data/processed/lemmas_voted.tsv", separator="\t", has_header=True).columns[0])
common = votes.filter(~pl.col("tag").is_in(["NNP", "NNPS"]))
per_type = common.group_by("type").agg(
    total=pl.col("count").sum(),
    plural=pl.col("count").filter(pl.col("tag").is_in(["NNS", "VBZ"])).sum(),
    lemma=pl.col("lemma").filter(pl.col("tag") == "NNS").sort_by(pl.col("count").filter(pl.col("tag") == "NNS"), descending=True).first(),
).with_columns(share=pl.col("plural") / pl.col("total"))
unmapped = per_type.filter(~pl.col("type").is_in(mapped.implode()) & (pl.col("total") >= 5) & pl.col("lemma").is_not_null() & (pl.col("lemma") != pl.col("type")))
irregular = unmapped.filter((pl.col("share") >= 0.8) & ~pl.col("type").str.ends_with("s"))
print("irregular (>=80%, not ending in s):", irregular.height, irregular["total"].sum())
print(irregular.sort("total", descending=True).head(40))
band = unmapped.filter((pl.col("share") >= 0.6) & (pl.col("share") < 0.8))
print("60-80% band:", band.height, band["total"].sum())
print(band.sort("total", descending=True).head(40))
regular_missed = unmapped.filter((pl.col("share") >= 0.8) & pl.col("type").str.ends_with("s"))
print(">=80% ending in s but unmapped:", regular_missed.height, regular_missed["total"].sum())
print(regular_missed.sort("total", descending=True).head(40))

from interpretant.corpus.lemmas import _lemmatizer
lookups = _lemmatizer().lookups
index = set(lookups.get_table("lemma_index")["noun"])
exceptions = lookups.get_table("lemma_exc")["noun"]
guarded = irregular.with_columns(
    in_exceptions=pl.col("type").map_elements(lambda t: t in exceptions, return_dtype=pl.Boolean),
    in_index=pl.col("lemma").map_elements(lambda w: w in index, return_dtype=pl.Boolean),
)
print(guarded.group_by("in_exceptions", "in_index").agg(pl.len(), pl.col("total").sum()))
print("kept by exceptions:", guarded.filter("in_exceptions").sort("total", descending=True)["type"].head(60).to_list())
print("dropped:", guarded.filter(~pl.col("in_exceptions")).sort("total", descending=True)["type"].head(40).to_list())
print("dropped lemmas:", guarded.filter(~pl.col("in_exceptions")).sort("total", descending=True)["lemma"].head(40).to_list())
