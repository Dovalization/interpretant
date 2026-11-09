"""Helpers shared by the embedding trainers."""

from collections.abc import Iterator

# gensim's compiled training loop silently truncates a sentence after this many tokens.
MAX_SENTENCE_TOKENS = 10_000


def training_sentences(line: str) -> Iterator[list[str]]:
    """Tokenise one document line into sentences gensim trains on in full.

    Books are stored one per line (about 120,000 tokens on average); passed whole, gensim
    would train on the first 10,000 tokens of each book and ignore the rest.
    """
    tokens = line.split()
    for start in range(0, len(tokens), MAX_SENTENCE_TOKENS):
        yield tokens[start : start + MAX_SENTENCE_TOKENS]
