"""Each merged compound family's share of its forms per decade, overall and per field.

`docs/compounds.md` reports how the open, hyphenated and closed ways of writing a compound drift
across decades, a finding the merge would otherwise hide. The builder's `shares.parquet` counts
every form of every proposed family over kept works per field and decade (spaced forms only where
they pass the noun-compound gate). This script keeps the families of the audited map
(`curated/compounds.tsv`), labels each form open, hyphenated or closed, and prints:

- the share of each way of writing per decade, pooled over all families and over the varying
  families (those whose standard holds under 90% of their tokens; the others are mostly
  line-break remnants such as *meth-od*), overall and per field;
- the standard's share per decade for the families whose other forms have most tokens.

A share over fewer than 20 tokens is left empty. With OUT, every family's share per decade, overall
and per field, is written there. Usage:
`python scripts/evidence/compound_shares.py SHARES.parquet [TOP] [OUT.parquet]`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from interpretant.catalog.compounds import load_compounds
from interpretant.catalog.config import read_catalog

WAYS = ("closed", "hyphenated", "open")
VARYING_BELOW = 0.9
MIN_TOKENS = 20


def way_of_writing(form: str) -> str:
    """Whether a form is written closed, hyphenated or open."""
    if " " in form:
        return "open"
    return "hyphenated" if "-" in form else "closed"


def family_forms() -> pl.DataFrame:
    """Each form of the audited map's families, with its family's standard and way of writing."""
    compounds = load_compounds(read_catalog())
    rows = {(compound.standard, compound.standard) for compound in compounds}
    rows |= {(compound.form, compound.standard) for compound in compounds}
    return pl.DataFrame(
        [(form, standard, way_of_writing(form)) for form, standard in sorted(rows)],
        schema=["form", "standard", "way"],
        orient="row",
    )


def way_shares(counts: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Each way of writing's share of the families' tokens per decade, within the groups ``by``."""
    totals = counts.group_by([*by, "decade", "way"]).agg(pl.col("count").sum())
    wide = totals.pivot(on="way", index=[*by, "decade"], values="count").fill_null(0)
    return (
        wide.with_columns(pl.sum_horizontal(*WAYS).alias("tokens"))
        .with_columns((pl.col(way) / pl.col("tokens")).round(3).alias(way) for way in WAYS)
        .select(*by, "decade", "tokens", *WAYS)
        .sort(*by, "decade")
    )


def family_shares(counts: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Each family's tokens and standard's share per decade within ``by``, empty under 20 tokens."""
    is_standard = pl.col("form") == pl.col("standard")
    return (
        counts.group_by("standard", *by, "decade")
        .agg(
            pl.col("count").sum().alias("tokens"),
            pl.col("count").filter(is_standard).sum().alias("standard_tokens"),
        )
        .with_columns(
            pl.when(pl.col("tokens") >= MIN_TOKENS)
            .then((pl.col("standard_tokens") / pl.col("tokens")).round(2))
            .alias("share")
        )
        .sort("standard", *by, "decade")
    )


def main() -> None:
    """Join the counts to the audited families, print the tables, and write the family shares."""
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    forms = family_forms()
    counts = pl.read_parquet(sys.argv[1]).join(forms, on="form", how="inner")
    missing = forms.join(counts.select("form").unique(), on="form", how="anti")
    families = (
        counts.group_by("standard")
        .agg(
            pl.col("count").sum().alias("tokens"),
            pl.col("count").filter(pl.col("form") != pl.col("standard")).sum().alias("changed"),
        )
        .with_columns((1 - pl.col("changed") / pl.col("tokens")).alias("standard_share"))
    )
    varying = families.filter(pl.col("standard_share") < VARYING_BELOW)["standard"]
    varying_counts = counts.filter(pl.col("standard").is_in(varying.to_list()))
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(160)
    write = sys.stdout.write
    write(
        f"{families.height} families, {forms.height} forms, {counts['count'].sum()} tokens "
        f"({families['changed'].sum()} in forms other than the standard); "
        f"{missing.height} forms without a count\n"
        f"{varying.len()} varying families (standard under {VARYING_BELOW:.0%} of tokens), "
        f"{varying_counts['count'].sum()} tokens\n\n"
    )
    write(f"Ways of writing per decade, all families:\n{way_shares(counts, [])}\n\n")
    write(f"Ways of writing per decade, varying families:\n{way_shares(varying_counts, [])}\n\n")
    write(f"Per field, varying families:\n{way_shares(varying_counts, ['field'])}\n\n")
    overall = family_shares(counts, [])
    largest = families.sort("changed", "standard", descending=[True, False]).head(top)
    per_decade = overall.filter(pl.col("standard").is_in(largest["standard"].to_list())).pivot(
        on="decade", index="standard", values="share"
    )
    table = (
        largest.select("standard", "tokens", "changed")
        .join(per_decade, on="standard")
        .sort("changed", descending=True)
    )
    write(f"The standard's share per decade, the {top} families with most changed tokens:\n")
    write(f"{table}\n")
    if len(sys.argv) > 3:
        out = Path(sys.argv[3])
        by_field = family_shares(counts, ["field"])
        all_fields = overall.with_columns(pl.lit("all").alias("field"))
        (
            pl.concat([all_fields, by_field], how="diagonal")
            .select("standard", "field", "decade", "tokens", "standard_tokens", "share")
            .write_parquet(out)
        )
        write(f"family shares written to {out}\n")


if __name__ == "__main__":
    main()
