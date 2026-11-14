"""Bibliographic values every source writes in its own form: page ranges and ISSNs."""

from __future__ import annotations

import polars as pl

# A start and an end page, each perhaps after letters ("EL108-EL114"), joined by any dash.
_PAGE_RANGE = r"^\s*[A-Za-z]*(\d+)\s*[-‐‑‒–—]\s*[A-Za-z]*(\d+)\s*$"
# A longer span is a parsing accident (a volume's whole run, a misread number), not one work.
MAX_PAGES = 2_000


def count_pages(column: str) -> pl.Expr:
    """The number of pages a range in ``column`` covers, null where it holds no single range.

    "538-539" is 2. PubMed shortens the end to the digits that change, so "1649-53" is
    1649-1653, 5 pages. A single page number says only where a work starts, and several ranges
    are rare, so both give null, as does a span over 2,000 pages.
    """
    start = pl.col(column).str.extract(_PAGE_RANGE, 1)
    end = pl.col(column).str.extract(_PAGE_RANGE, 2)
    shortened = end.str.len_chars() < start.str.len_chars()
    head = start.str.slice(0, start.str.len_chars() - end.str.len_chars())
    full_end = pl.when(shortened).then(head + end).otherwise(end)
    pages = full_end.cast(pl.Int64, strict=False) - start.cast(pl.Int64, strict=False) + 1
    return pl.when(pages.is_between(1, MAX_PAGES)).then(pages)


def is_valid_issn(issn: str) -> bool:
    """Whether an ISSN has its form (####-###X) and its check digit is right (ISO 3297)."""
    digits = issn.replace("-", "")
    if len(issn) != 9 or issn[4] != "-" or not digits[:7].isdigit():
        return False
    total = sum(
        int(digit) * weight for digit, weight in zip(digits[:7], range(8, 1, -1), strict=True)
    )
    check = (11 - total % 11) % 11
    return digits[7] == ("X" if check == 10 else str(check))
