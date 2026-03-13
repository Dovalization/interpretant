"""Text preprocessing utilities for corpus documents."""

import re
import unicodedata


def preprocess_text(text: str) -> str:
    """Normalise and clean a raw document string.

    Steps applied in order:
    1. Unicode NFKC normalisation
    2. Lowercase
    3. Strip LaTeX-style math spans (``$...$`` and ``$$...$$``)
    4. Collapse punctuation runs to a single space
    5. Collapse whitespace

    Args:
        text: Raw document string.

    Returns:
        Cleaned, lowercased string.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Remove display math
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    # Remove inline math
    text = re.sub(r"\$[^$]+\$", " ", text)
    # Replace punctuation / non-alpha with space (keep hyphens inside words)
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    # Collapse multiple hyphens or leading/trailing hyphens in tokens
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_long_enough(text: str, min_tokens: int = 20) -> bool:
    """Return True if text contains at least min_tokens whitespace-separated tokens.

    Args:
        text: Preprocessed document string.
        min_tokens: Minimum number of tokens required.

    Returns:
        True if the token count meets or exceeds min_tokens.
    """
    return len(text.split()) >= min_tokens


def filter_empty(texts: list[str]) -> list[str]:
    """Remove empty or whitespace-only strings from a list.

    Args:
        texts: List of text strings, possibly containing empty entries.

    Returns:
        New list with blank entries removed.
    """
    return [t for t in texts if t.strip()]
