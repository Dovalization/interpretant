"""A type-level lemma map: plurals and British spellings, nothing else.

The cells are not lemmatised beyond the target words, so *structure* and *structures*,
*embedding* and *embeddings*, *conceptualisation* and *conceptualization* train as separate
words. A general lemmatiser is the wrong tool here: it turns *learning* into *learn*,
*meaning* into *mean* and *embodied* into *embody*, which destroys the nominalisations the
targets live on. So the map accepts simplemma's lemma only when the token is a regular plural
of it, normalises British spellings to American through rules gated on simplemma's lexicon
and an exception list, and leaves every other form alone. The map is learned once from the
vocabulary, written to a reviewable file, and applied line by line like the phrase model.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import simplemma

LANGUAGE = "en"
MIN_SPELLING_LENGTH = 6
RULE_OVERRIDE = "override"
RULE_PLURAL = "plural"
RULE_SPELLING = "spelling"

# Words ending in -ise where -ise is not the British suffix (and their derivatives).
ISE_EXCEPTIONS = (
    "advertise",
    "advise",
    "arise",
    "comprise",
    "compromise",
    "concise",
    "demise",
    "despise",
    "devise",
    "disguise",
    "enterprise",
    "excise",
    "exercise",
    "expertise",
    "franchise",
    "improvise",
    "likewise",
    "merchandise",
    "otherwise",
    "paradise",
    "praise",
    "precise",
    "premise",
    "promise",
    "raise",
    "revise",
    "supervise",
    "surprise",
    "televise",
    "treatise",
)
OUR_STEMS = (
    "behaviour",
    "colour",
    "endeavour",
    "favour",
    "flavour",
    "harbour",
    "honour",
    "humour",
    "labour",
    "neighbour",
    "odour",
    "rigour",
    "savour",
    "tumour",
    "vapour",
    "vigour",
)
RE_STEMS = ("calibre", "centre", "fibre", "litre", "metre", "sombre", "spectre", "theatre")
SPELLINGS = {
    "analogue": "analog",
    "artefact": "artifact",
    "catalogue": "catalog",
    "defence": "defense",
    "dialogue": "dialog",
    "grey": "gray",
    "judgement": "judgment",
    "licence": "license",
    "offence": "offense",
    "pretence": "pretense",
    "programme": "program",
    "sceptic": "skeptic",
    "sceptical": "skeptical",
    "scepticism": "skepticism",
}
_ISE_SUFFIX = re.compile(r"(is)(ation|ations|e|ed|es|ing|er|ers|able)$")
_YSE_SUFFIX = re.compile(r"(ys)(e|ed|es|ing|er|ers)$")
_DOUBLE_L = re.compile(r"[aeiou]ll(ing|ed|er|ers)$")
_WORD = re.compile(r"^[a-z]+$")


def _known(word: str) -> bool:
    return bool(simplemma.is_known(word, lang=LANGUAGE))


def regular_plural(token: str, lemma: str) -> bool:
    """Whether ``token`` is a regular English plural of ``lemma``."""
    if lemma == token or len(lemma) < 3:
        return False
    if token in (lemma + "s", lemma + "es"):
        return True
    if lemma.endswith("y") and token == lemma[:-1] + "ies":
        return True
    return lemma.endswith("is") and token == lemma[:-2] + "es"


def singular(token: str) -> str | None:
    """The singular of a regular plural, or ``None``."""
    lemma = simplemma.lemmatize(token, lang=LANGUAGE)
    return lemma if regular_plural(token, lemma) else None


def _spelling_candidate(token: str) -> str | None:
    if token in SPELLINGS:
        return SPELLINGS[token]
    if token.startswith(ISE_EXCEPTIONS):
        return None
    if _ISE_SUFFIX.search(token):
        return _ISE_SUFFIX.sub(lambda match: "iz" + match.group(2), token)
    if _YSE_SUFFIX.search(token):
        return _YSE_SUFFIX.sub(lambda match: "yz" + match.group(2), token)
    for stem in OUR_STEMS:
        if token.startswith(stem):
            return stem[:-3] + "or" + token[len(stem) :]
    for stem in RE_STEMS:
        if token == stem:
            return stem[:-2] + "er"
        if token.startswith(stem[:-1]) and token[len(stem) - 1 :] in ("ed", "ing"):
            return stem[:-2] + "er" + token[len(stem) - 1 :]
    if (match := _DOUBLE_L.search(token)) is not None:
        return token[: match.start() + 1] + "l" + match.group(1)
    return None


def american(token: str) -> str | None:
    """The American spelling of a British form, or ``None`` when the token is not one."""
    if len(token) < MIN_SPELLING_LENGTH:
        return None
    candidate = _spelling_candidate(token)
    if candidate is None or candidate == token or not _known(candidate):
        return None
    return candidate


@dataclass
class LemmaMap:
    """Token type to normal form, with the rule that produced each entry."""

    mapping: dict[str, str] = field(default_factory=dict)
    rules: dict[str, str] = field(default_factory=dict)

    def apply(self, line: str) -> str:
        return " ".join(self.mapping.get(token, token) for token in line.split())

    def save(self, path: Path, counts: Counter[str] | None = None) -> None:
        counts = counts or Counter()
        rows = sorted(self.mapping, key=lambda token: (-counts[token], token))
        path.write_text(
            "token\tlemma\tcount\trule\n"
            + "".join(
                f"{token}\t{self.mapping[token]}\t{counts[token]}\t{self.rules[token]}\n"
                for token in rows
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> LemmaMap:
        lemma_map = cls()
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            token, lemma, _count, rule = line.split("\t")
            lemma_map.mapping[token] = lemma
            lemma_map.rules[token] = rule
        return lemma_map


def read_overrides(path: Path) -> dict[str, str]:
    """Read ``token<TAB>lemma`` lines; an empty lemma keeps the token as it is."""
    overrides: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        token, _, lemma = line.partition("\t")
        overrides[token.strip()] = lemma.strip()
    return overrides


def normal_form(token: str, protected: set[str], overrides: dict[str, str]) -> tuple[str, str]:
    """Return ``(lemma, rule)`` for one token type; ``(token, "")`` when it stays as it is."""
    if token in protected or not _WORD.match(token):
        return token, ""
    if token in overrides:
        lemma = overrides[token] or token
        return lemma, RULE_OVERRIDE if lemma != token else ""
    lemma, rule = token, ""
    if (single := singular(token)) is not None:
        lemma, rule = single, RULE_PLURAL
    if lemma in overrides:
        return overrides[lemma] or lemma, rule or RULE_OVERRIDE
    if lemma not in protected and (spelled := american(lemma)) is not None:
        lemma, rule = spelled, RULE_SPELLING if rule == "" else rule
    return lemma, rule


def learn_lemma_map(
    counts: Counter[str], protected: Iterable[str], overrides: dict[str, str]
) -> LemmaMap:
    """Map every type in ``counts`` that has a normal form other than itself."""
    protected_set = set(protected)
    lemma_map = LemmaMap()
    for token in counts:
        lemma, rule = normal_form(token, protected_set, overrides)
        if lemma != token:
            lemma_map.mapping[token] = lemma
            lemma_map.rules[token] = rule
    return lemma_map


def count_types(paths: Iterable[Path]) -> Counter[str]:
    """Count token types over processed files (one document per line)."""
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                counts.update(line.split())
    return counts
