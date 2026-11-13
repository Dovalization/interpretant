"""Point 5 of finding 11: if the lemma review goes most frequent first, how many types until the
unreviewed rest covers under 0.1% (or 0.01%) of tokens?"""

import re

import polars as pl

AUDIT = "lemma_audit"  # the output folder of lemma_audit.py
entries = pl.read_parquet(f"{AUDIT}/entries.parquet")
total_tokens = pl.read_parquet(f"{AUDIT}/type_counts.parquet")["count"].sum()

disagree = entries.filter(
    (pl.col("spacy_lemma").fill_null("") != "")
    & (pl.col("spacy_lemma") != pl.col("token"))
    & (pl.col("lemma") != pl.col("token"))
    & (pl.col("spacy_lemma") != pl.col("lemma"))
)

SPELLING = [("our", "or"), ("ise", "ize"), ("yse", "yze"), ("isation", "ization"), ("tre", "ter"),
            ("ence", "ense"), ("ogue", "og"), ("ae", "e"), ("oe", "e"), ("ll", "l"), ("dgement", "dgment"), ("mme", "m")]


def spelling_only(british: str, american: str) -> bool:
    return any(british.replace(a, b) == american for a, b in SPELLING) or (
        re.sub(r"[sz]", "z", british) == re.sub(r"[sz]", "z", american)
    )


disagree = disagree.with_columns(
    spelling=pl.struct("spacy_lemma", "lemma").map_elements(
        lambda row: spelling_only(row["spacy_lemma"], row["lemma"]), return_dtype=pl.Boolean
    )
)
review = disagree.filter(~pl.col("spelling")).sort("count", descending=True)
review_tokens = review["count"].sum()
print(f"all tokens: {total_tokens:,}")
print(f"types where both merge and differ: {disagree.height:,} "
      f"({disagree['count'].sum() / total_tokens:.2%} of tokens)")
print(f"  spelling only (handled by the spelling table, not reviewed): {disagree['spelling'].sum():,}")
print(f"  to review: {review.height:,} types, {review_tokens:,} tokens "
      f"({review_tokens / total_tokens:.3%} of all tokens)")

cumulative = review["count"].cum_sum()
for threshold in (0.001, 0.0001):
    rest = review_tokens - cumulative
    needed = int((rest >= threshold * total_tokens).sum()) + 1
    needed = min(needed, review.height)
    print(f"  unreviewed rest under {threshold:.2%} of tokens after the top {needed:,} types")
for top in (50, 100, 200, 400):
    covered = cumulative[min(top, review.height) - 1] / review_tokens
    print(f"  top {top:>3} types cover {covered:.0%} of the disagreeing tokens")
print("\nmost frequent to review:")
print(review.select("token", "lemma", "spacy_lemma", "count").head(25))
