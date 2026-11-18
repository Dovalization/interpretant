"""The usages' identifiers, each defined once (decision agenda 4.5).

- **content key:** the sha256 of exactly the text a sentence encoder reads and the target's place in
  it, ``"{start}:{end}:{sentence}"`` with offsets relative to the sentence, first 20 hex characters.
  Identical text is encoded once and shared; the same sentence with the target elsewhere, or a wider
  context, gets another key.
- **draw key:** the sha256 of the seed, the work, the word and the content key. The draw keeps the
  smallest key per work and word, so one abstract's draw never depends on other abstracts or on row
  order, and the word is in the key so different words' draws are independent (Kohavi et al. 2009
  found correlations between experiments unless each hashed its own id with the unit's).

A usage's id is (work id, word, content key); it changes only if that mention's sentence changes.
"""

from __future__ import annotations

import hashlib

CONTENT_KEY_LENGTH = 20


def build_content_key(sentence: str, start: int, end: int) -> str:
    """The content key of a target at ``sentence[start:end]``."""
    digest = hashlib.sha256(f"{start}:{end}:{sentence}".encode()).hexdigest()
    return digest[:CONTENT_KEY_LENGTH]


def build_draw_key(seed: int, work_id: str, word: str, content_key: str) -> str:
    """The draw key of a usage: the smallest per work and word is drawn."""
    return hashlib.sha256(f"{seed}:{work_id}:{word}:{content_key}".encode()).hexdigest()
