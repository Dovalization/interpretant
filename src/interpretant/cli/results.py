"""How the corpus commands print what a stage wrote."""

from __future__ import annotations

from rich.table import Table

from interpretant.pipelines.corpus import CorpusResult


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
