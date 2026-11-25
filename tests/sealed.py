"""Failing a check on the word table without printing its words, so held-out words stay sealed.

The phase's ground rule (docs/preprocessing_phase.md): held-out words are never printed, only
hashed. pytest's assertion rewriting would print a failing list in full, so the invariant checks
on the word table fail through ``fail_sealed`` instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest

from interpretant.catalog.config import read_catalog

# The held-out words are tracked by DVC, not git, so a checkout without them (CI) skips the checks
# that read the real table
requires_held_out = pytest.mark.skipif(
    not read_catalog().held_out_words.path.exists(),
    reason="the held-out words are tracked by DVC (dvc pull curated/held_out.tsv)",
)


def fail_sealed(words: Sequence[str], what: str) -> None:
    """Fail the test if ``words`` is not empty, naming only their count and hash prefixes."""
    if words:
        hashes = ", ".join(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8] for word in words)
        pytest.fail(f"{len(words)} words {what}: sha256 {hashes}", pytrace=False)
