"""How the corpus commands print what a stage wrote."""

from __future__ import annotations

from rich.table import Table

from interpretant.pipelines.corpus import CorpusResult
from interpretant.pipelines.detect_languages import LanguageResult


def result_counts(result: CorpusResult) -> dict[str, int]:
    """A stage's headline counts for the run log."""
    return {
        "abstracts": result.abstract_count,
        "dropped": sum(result.dropped_counts.values()),
        "parts": len(result.parts),
    }


def render_result(result: CorpusResult) -> Table:
    """One table: the run's totals, then the count of each drop, removal and flag reason."""
    table = Table(title=f"{result.abstract_count:,} abstracts in {len(result.parts)} parts")
    table.add_column("Rule")
    table.add_column("Reason")
    table.add_column("Count", justify="right")
    for rule, counts in (
        ("dropped", result.dropped_counts),
        ("removed range", result.removal_counts),
        ("flagged", result.flag_counts),
    ):
        for reason, count in counts.items():
            table.add_row(rule, reason, f"{count:,}")
    return table


def render_language_result(result: LanguageResult) -> Table:
    """One table: the sentences lingua gave each language."""
    total = sum(result.sentence_counts.values())
    table = Table(
        title=f"{total:,} sentences of {result.abstract_count:,} abstracts in "
        f"{len(result.parts)} parts"
    )
    table.add_column("Language")
    table.add_column("Sentences", justify="right")
    table.add_column("Share", justify="right")
    for language, count in result.sentence_counts.items():
        table.add_row(language, f"{count:,}", f"{count / total:.2%}")
    return table
