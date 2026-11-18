"""Checks of the usages table against the repaired text (decision agenda 4.5).

A usage points into the abstracts table instead of copying text, so its content key must equal the
hash recomputed from the sentence and the target's offsets in it, and the target's span must hold
the word's letters. These checks read the text back and list every usage that fails.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from interpretant.corpus.static_view import fold
from interpretant.usages.identifiers import build_content_key


def check_usages_against_text(usages: pl.DataFrame, texts: Mapping[str, str]) -> list[str]:
    """Every usage whose content key or target span does not match its work's text.

    Returns a list of problems, each naming the work, the word and what failed; empty if all pass.
    """
    problems = []
    for row in usages.iter_rows(named=True):
        text = texts.get(row["work_id"])
        if text is None:
            problems.append(f"{row['work_id']} {row['word']}: no text")
            continue
        sentence = text[row["sentence_start"] : row["sentence_end"]]
        start = row["target_start"] - row["sentence_start"]
        end = row["target_end"] - row["sentence_start"]
        if build_content_key(sentence, start, end) != row["content_key"]:
            problems.append(f"{row['work_id']} {row['word']}: content key")
        target = fold(text[row["target_start"] : row["target_end"]])
        if not target.strip() or target[:1] != row["word"][:1]:
            problems.append(f"{row['work_id']} {row['word']}: target span {target!r}")
    return problems
