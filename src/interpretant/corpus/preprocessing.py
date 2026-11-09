"""Text preprocessing utilities for corpus documents."""

import re
import unicodedata

# Plurals and British spellings of the target words (data/words_of_interest.txt), mapped
# onto the lemma so each target gets one vector instead of several split ones. Forms that
# are mostly verbs ("controls") or carry a separate sense ("media") are left alone.
TARGET_VARIANTS: dict[str, str] = {
    "representations": "representation",
    "perceptions": "perception",
    "experiences": "experience",
    "cognitions": "cognition",
    "minds": "mind",
    "meanings": "meaning",
    "intelligences": "intelligence",
    "computations": "computation",
    "memories": "memory",
    "models": "model",
    "networks": "network",
    "systems": "system",
    "complexities": "complexity",
    "behaviour": "behavior",
    "behaviours": "behavior",
    "behaviors": "behavior",
    "symbols": "symbol",
    "signs": "sign",
    "affordances": "affordance",
    "embodiments": "embodiment",
    "interfaces": "interface",
    "users": "user",
    "agents": "agent",
    "simulations": "simulation",
    "images": "image",
    "mediums": "medium",
}
_VARIANT_RE = re.compile(r"\b(" + "|".join(TARGET_VARIANTS) + r")\b")


def preprocess_text(text: str) -> str:
    """Normalise and clean a raw document string.

    Steps applied in order:
    1. Unicode NFKC normalisation
    2. Lowercase
    3. NFKD transliteration to ASCII (café→cafe, übermensch→ubermensch, différance→differance)
    4. Strip LaTeX-style math spans (``$...$`` and ``$$...$$``)
    5. Collapse punctuation runs to a single space
    6. Collapse whitespace
    7. Map target-word variants onto their lemma (``TARGET_VARIANTS``)

    Args:
        text: Raw document string.

    Returns:
        Cleaned, lowercased string.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Rejoin PDF line-break artifacts: "var¬ \niables" → "variables".
    # U+00AC (NOT SIGN) is used by some PDF extractors as a soft-hyphen substitute.
    text = re.sub(r"\u00ac\s*", "", text)
    # Transliterate accented characters to ASCII base (café→cafe, übermensch→ubermensch,
    # différance→differance) so diacritics don't corrupt tokens when punctuation is stripped.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
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
    return _VARIANT_RE.sub(lambda match: TARGET_VARIANTS[match.group(1)], text)


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
