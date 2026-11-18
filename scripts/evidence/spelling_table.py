"""The spelling map's proposal, its audit sample and its census (agenda 3.4; phase step 2).

Every token type of the static view (R_B, `data/interim/static_view`) and every hyphen segment is
looked up in spaCy's English spelling table (`lexeme_norm`, frozen inside en_core_web_sm 3.8.0;
its keys are hashed, so the corpus's own types are the probe). The table's pairs are filtered:

- a key or value with white space or no letter is dropped;
- informal norms (*cos* → *because*) and pairs too far apart to be spellings (a character ratio
  under 0.6) are dropped;
- pairs known to merge different words are dropped (*analyses* → *analyzes*, *buses* →
  *busses*, *cheques* → *checks*, *storey* → *story*).

The project's own rules (`corpus.lemmas.american`) add the derived words the table lacks
(*optimisation*, *categorisation*). Each pair is then written as the corpus's more frequent form
(the author, 2026-11-17), a word-table lemma keeping its own spelling, and chains are resolved.

Writes OUT/proposal.parquet (form, standard, source, counts, whether the British form won),
OUT/sample.parquet (150 pairs, seed 20261119: 75 drawn by tokens, 75 uniform over pairs) and
OUT/census.parquet (every pair touching the word table, the 50 with most tokens, every pair whose
British form won), and prints the summary. Run from the repository root on the run machine:
`python scripts/evidence/spelling_table.py OUT`.
"""

from __future__ import annotations

import difflib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import spacy

from interpretant.catalog.config import read_catalog
from interpretant.catalog.words import load_words
from interpretant.corpus.lemmas import american

SEED = 20261119
SAMPLE_BY_TOKENS = 75
SAMPLE_UNIFORM = 75
TOP_BY_TOKENS = 50
MIN_RATIO = 0.6
INFORMAL = frozenset(
    [
        "cos",
        "cuz",
        "fav",
        "fave",
        "misc",
        "pls",
        "plz",
        "thx",
        "doin",
        "goin",
        "havin",
        "lovin",
        "nothin",
        "somethin",
        "tho",
        "thru",
        "wanna",
        "gonna",
    ]
)
# Pairs whose two forms are different words in this corpus, not spellings of one word.
DIFFERENT_WORDS = frozenset(
    [
        "analyses",
        "buses",
        "busing",
        "bussing",
        "gases",
        "storey",
        "storeys",
        "tonne",
        "tonnes",
        "philtre",
        "tranquilly",
        "cheque",
        "cheques",
        "pummelled",
        "pummelling",
    ]
)
# Forms of other languages the project's rules read as British spellings (Portuguese *analise*).
NOT_ENGLISH = frozenset({"analise", "analize"})


def type_counts(view: Path) -> tuple[Counter[str], Counter[str]]:
    """Every token type's count, and every hyphen segment's count, in the static view."""
    tokens = (
        pl.scan_parquet(view / "*.parquet")
        .select(pl.col("tokens").explode().alias("token"))
        .group_by("token")
        .len()
        .collect()
    )
    whole = Counter(dict(tokens.iter_rows()))
    segments: Counter[str] = Counter()
    for token, count in whole.items():
        if token and "-" in token:
            for segment in token.split("-"):
                if segment:
                    segments[segment] += count
    return whole, segments


def table_pairs(forms: list[str]) -> dict[str, str]:
    """The spelling table's norm for each form it changes, before filtering."""
    norms = spacy.load("en_core_web_sm").vocab.lookups.get_table("lexeme_norm")
    pairs: dict[str, str] = {}
    for form in forms:
        norm = norms.get(form)
        if isinstance(norm, str) and norm != form:
            pairs[form] = norm
    return pairs


def kept_pair(form: str, norm: str) -> bool:
    """Whether a table pair is a spelling variant this corpus can merge."""
    if any(c.isspace() for c in form + norm) or not any(c.isalpha() for c in norm):
        return False
    if form in INFORMAL or form in DIFFERENT_WORDS:
        return False
    # A plural noun in -yses beside a verb in -yzes (analyses, psychoanalyses): different words.
    if form.endswith("yses") or norm.endswith("yzes"):
        return False
    return difflib.SequenceMatcher(None, form, norm).ratio() >= MIN_RATIO


def oriented(
    pairs: dict[str, tuple[str, str]], counts: Counter[str], lemmas: frozenset[str]
) -> list[dict[str, object]]:
    """Each pair written toward the corpus's more frequent form, word-table lemmas kept."""
    rows = []
    for british, (american_form, source) in pairs.items():
        # The British form wins only when the American form occurs and is rarer: with one form
        # absent, the table's direction stands.
        british_more = counts[american_form] > 0 and counts[british] > counts[american_form]
        if american_form in lemmas or (british not in lemmas and not british_more):
            form, standard, british_won = british, american_form, False
        else:
            form, standard, british_won = american_form, british, True
        rows.append(
            {
                "form": form,
                "standard": standard,
                "source": source,
                "form_count": counts[form],
                "standard_count": counts[standard],
                "british_won": british_won,
            }
        )
    by_form: dict[str, dict[str, object]] = {}
    for row in sorted(rows, key=lambda r: -int(str(r["form_count"]))):
        by_form.setdefault(str(row["form"]), row)
    standards = {form: str(row["standard"]) for form, row in by_form.items()}
    for row in by_form.values():
        seen = {str(row["form"])}
        target = str(row["standard"])
        while target in standards and target not in seen:
            seen.add(target)
            target = standards[target]
        row["standard"] = target
    return [row for row in by_form.values() if row["form"] != row["standard"]]


def main() -> None:
    """Build the proposal, the sample and the census, and print the summary."""
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog()
    lemmas = frozenset(word.lemma for word in load_words(catalog) if not word.is_phrase)
    whole, segments = type_counts(catalog.static_view.path)
    counts = whole + segments
    forms = sorted(
        form
        for form in counts
        if form and form.isascii() and form.replace("-", "").isalpha() and "-" not in form
    )
    raw = table_pairs(forms)
    kept = {form: norm for form, norm in raw.items() if kept_pair(form, norm)}
    pairs: dict[str, tuple[str, str]] = {form: (norm, "norm_table") for form, norm in kept.items()}
    in_table = set(raw) | set(raw.values())
    for form in forms:
        if form in pairs or form in in_table:
            continue
        candidate = american(form, counts)
        if candidate is not None and candidate != form:
            pairs[form] = (candidate, "rules")
    pairs = {
        form: pair
        for form, pair in pairs.items()
        if form not in lemmas and form not in NOT_ENGLISH and pair[0] not in NOT_ENGLISH
    }
    proposal = pl.DataFrame(oriented(pairs, counts, lemmas)).sort("form_count", descending=True)
    proposal.write_parquet(out / "proposal.parquet")

    rng = np.random.default_rng(SEED)
    weights = proposal["form_count"].to_numpy().astype(float)
    by_tokens = rng.choice(
        proposal.height, size=SAMPLE_BY_TOKENS, replace=False, p=weights / weights.sum()
    )
    rest = np.setdiff1d(np.arange(proposal.height), by_tokens)
    uniform = rng.choice(rest, size=SAMPLE_UNIFORM, replace=False)
    sample = pl.concat(
        [
            proposal[by_tokens.tolist()].with_columns(drawn=pl.lit("by_tokens")),
            proposal[uniform.tolist()].with_columns(drawn=pl.lit("uniform")),
        ]
    )
    sample.write_parquet(out / "sample.parquet")
    touching = proposal.filter(
        pl.col("form").is_in(sorted(lemmas)) | pl.col("standard").is_in(sorted(lemmas))
    )
    census = pl.concat(
        [
            touching.with_columns(reason=pl.lit("word_table")),
            proposal.head(TOP_BY_TOKENS).with_columns(reason=pl.lit("top_tokens")),
            proposal.filter(pl.col("british_won")).with_columns(reason=pl.lit("british_won")),
        ]
    ).unique(subset=["form"], keep="first", maintain_order=True)
    census.write_parquet(out / "census.parquet")

    changed = int(proposal["form_count"].sum())
    total = sum(whole.values())
    lines = [
        f"types probed: {len(forms):,}; table pairs found {len(raw):,}, kept {len(kept):,}; "
        f"rule pairs {sum(1 for _, s in pairs.values() if s == 'rules'):,}",
        f"proposal: {proposal.height:,} forms, {changed:,} tokens ({changed / total:.2%} of the view); "
        f"British form won {int(proposal['british_won'].sum())}; by source "
        f"{dict(proposal.group_by('source').len().iter_rows())}",
        f"sample {sample.height}; census {census.height} ({dict(census.group_by('reason').len().iter_rows())})",
        f"word-table lemmas mapped away: {sorted(set(proposal['form']) & lemmas)}",
        "pairs touching the word table: "
        + ", ".join(f"{f}->{s}" for f, s in touching.select("form", "standard").iter_rows()),
        "top 20: "
        + ", ".join(
            f"{f}->{s} ({n:,})"
            for f, s, n in proposal.head(20).select("form", "standard", "form_count").iter_rows()
        ),
        "British won, top 15: "
        + ", ".join(
            f"{f}->{s} ({n:,} vs {m:,})"
            for f, s, n, m in proposal.filter(pl.col("british_won"))
            .head(15)
            .select("form", "standard", "form_count", "standard_count")
            .iter_rows()
        ),
    ]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
