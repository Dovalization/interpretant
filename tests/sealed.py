"""Failing a check on the word table without printing its words, so held-out words stay sealed.

The phase's ground rule (docs/preprocessing_phase.md): held-out words are never printed, only
hashed. pytest's assertion rewriting would print a failing list in full, so the invariant checks
on the word table fail through ``fail_sealed`` instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest


def fail_sealed(words: Sequence[str], what: str) -> None:
    """Fail the test if ``words`` is not empty, naming only their count and hash prefixes."""
    if words:
        hashes = ", ".join(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8] for word in words)
        pytest.fail(f"{len(words)} words {what}: sha256 {hashes}", pytrace=False)
