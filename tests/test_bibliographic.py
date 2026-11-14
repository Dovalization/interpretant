"""Page ranges and ISSNs as the sources write them."""

from __future__ import annotations

import polars as pl
import pytest

from interpretant.corpus.bibliographic import count_pages, is_valid_issn


@pytest.mark.parametrize(
    ("page", "pages"),
    [
        ("538-539", 2),
        ("EL108-EL114", 7),
        ("1649-53", 5),
        ("12–14", 3),
        ("79", None),
        ("iii-x", None),
        ("1-3, 5-7", None),
        ("1 – 4000", None),
        (None, None),
    ],
    ids=["range", "letters before the numbers", "PubMed's shortened end", "en dash",
         "single page", "roman numerals", "several ranges", "beyond a work's length", "none"],
)  # fmt: skip
def test_count_pages_reads_a_single_range(page: str | None, pages: int | None) -> None:
    table = pl.DataFrame({"page": [page]}, schema={"page": pl.String()})
    assert table.select(count_pages("page")).item() == pages


@pytest.mark.parametrize("issn", ["0018-9340", "0016-934X"], ids=["right", "check digit X"])
def test_is_valid_issn_accepts_a_right_check_digit(issn: str) -> None:
    assert pl.select(is_valid_issn(pl.lit(issn))).item()


@pytest.mark.parametrize(
    "issn",
    ["0016-9340", "1234-988X", "00189340", "abcd-efgh"],
    ids=["typo", "wrong check digit", "no hyphen", "letters"],
)
def test_is_valid_issn_refuses_a_malformed_issn(issn: str) -> None:
    assert not pl.select(is_valid_issn(pl.lit(issn))).item()
