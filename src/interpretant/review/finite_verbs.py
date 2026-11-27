"""Whether a text keeps a finite verb once its removed ranges are left out.

Round S1 took a record with no finite verb outside its removals as likely junk
(`docs/predictions_cleaning.md`); the junk gate draws such records as a stratum and uses the same
test as a feature (`docs/junk_gate.md`).
"""

from __future__ import annotations

from collections.abc import Sequence

Range = tuple[int, int]


def has_finite_verb_outside(
    starts: Sequence[int], finite_verbs: Sequence[bool], removals: Sequence[Range]
) -> bool:
    """Whether a finite verb lies outside every removed range."""
    return any(
        finite and not any(start <= offset < end for start, end in removals)
        for offset, finite in zip(starts, finite_verbs, strict=True)
    )
