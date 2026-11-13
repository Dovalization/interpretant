import sys
import polars as pl
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(18); pl.Config.set_tbl_width_chars(220)
pl.Config.set_float_precision(2); pl.Config.set_tbl_hide_dataframe_shape(True)
d = sys.argv[1]
u = pl.read_parquet(f"{d}/unmapped.parquet")
cols = ["token", "spacy_lemma", "count", "lemma_count", "inflected_share", "proper_share", "cosine", "blocked_by_gate", "ungated_lemma"]
def show(title, frame, n=30):
    print(f"\n### {title}: {frame.height} types, {frame['count'].sum():,} tokens")
    print(frame.sort("count", descending=True).select(cols).head(n))
strong = pl.col("inflected_share") >= 0.8
show("B1 noun exception list (irregular)", u.filter(pl.col("in_noun_exceptions") & strong))
show("B1b verb exception list", u.filter(pl.col("in_verb_exceptions") & strong & ~pl.col("in_noun_exceptions")))
show("B2 -ves", u.filter(pl.col("token").str.ends_with("ves") & strong))
show("B3 -men", u.filter(pl.col("token").str.ends_with("men") & strong & ~pl.col("in_noun_exceptions")))
show("B4 blocked only by the frequency gate", u.filter(pl.col("blocked_by_gate")))
rest = u.filter(strong & ~pl.col("in_noun_exceptions") & ~pl.col("in_verb_exceptions") & ~pl.col("token").str.ends_with("ves") & ~pl.col("token").str.ends_with("men") & ~pl.col("blocked_by_gate"))
show("B5 other >=80% inflected, unmapped", rest, 45)
