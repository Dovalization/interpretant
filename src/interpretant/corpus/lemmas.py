"""A type-level lemma map: -s inflections, plural nouns and British spellings, nothing else.

The cells are not lemmatised beyond the target words, so *structure* and *structures*,
*embedding* and *embeddings*, *conceptualisation* and *conceptualization* train as separate
words. A general lemmatiser is the wrong tool here: it turns *learning* into *learn*,
*meaning* into *mean* and *embodied* into *embody*, which destroys the nominalisations the
targets live on. So the map merges a word only into the singular or base form spaCy gives it
in context (a plural noun, a third-person verb), when the readings agree and the word is a regular
-s form of it or one of spaCy's irregular plurals; it normalises British spellings to American
through rules gated on spaCy's word list or the corpus and an exception list, and leaves every
other form alone. The votes come from one spaCy pass over the raw abstracts (``lemma_votes``);
the map is learned once from them, written to a reviewable file, and applied line by line like
the phrase model.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from functools import cache
from itertools import islice
from pathlib import Path
from typing import cast

import polars as pl
import spacy
from spacy.language import Language
from spacy.pipeline import Lemmatizer
from spacy.tokens import Doc
from thinc.api import require_gpu

from interpretant.corpus.preprocessing import preprocess_text

SPACY_MODEL = "en_core_web_sm"
# Parts of speech in spaCy's lemma index (WordNet's base forms).
INDEX_POS = ("noun", "verb", "adj", "adv")
MIN_SPELLING_LENGTH = 6
RULE_OVERRIDE = "override"
RULE_PLURAL = "plural"
RULE_VERB = "verb"
RULE_SPELLING = "spelling"
# Fine-grained tags of the -s inflections a word may merge from: plural noun, third-person verb.
INFLECTED_TAGS = {"NNS": RULE_PLURAL, "VBZ": RULE_VERB}
PROPER_TAGS = frozenset({"NNP", "NNPS"})
# The readings that mark a candidate as the singular of each kind of inflection.
SINGULAR_TAGS = {RULE_PLURAL: frozenset({"NN"}), RULE_VERB: frozenset({"VB", "VBP"})}
# Stem endings after which English spells the plural or third person with -es.
ES_STEM_ENDINGS = ("s", "x", "z", "ch", "sh", "o")
MEN_SUFFIX = "men"
# A word merges only with at least this many tagged occurrences, at least this share of them
# read as an -s inflection of one lemma.
MIN_VOTES = 5
MIN_VOTE_SHARE = 0.8
VOTE_CHUNK = 1_000
# Chunks waiting per CPU worker: enough to keep each busy, few enough that the texts are read
# as the workers need them, not all held in memory at once.
CHUNKS_IN_FLIGHT_PER_WORKER = 2

# Stems (without the final e) of words where -ise is not the British suffix: *surprise*,
# *surprising*, *arising*; and the suffixes -wise and -rise, which never are.
ISE_EXCEPTION_STEMS = (
    "advertis",
    "advis",
    "apprais",
    "aris",
    "chastis",
    "circumcis",
    "compris",
    "compromis",
    "concis",
    "demis",
    "despis",
    "devis",
    "disguis",
    "enterpris",
    "excis",
    "exercis",
    "exorcis",
    "expertis",
    "franchis",
    "guis",
    "improvis",
    "incis",
    "merchandis",
    "paradis",
    "prais",
    "precis",
    "premis",
    "promis",
    "rais",
    "repris",
    "revis",
    "supervis",
    "surmis",
    "surpris",
    "televis",
    "treatis",
)
ISE_EXCEPTION_SUFFIXES = ("wise",)
# The root *rise* (*rise*, *sunrise*, *uprising*) starts the word or follows a consonant; after a
# vowel, *-rise* is the -ise suffix (*characterise*, *theorise*, *summarise*).
_RISE_ROOT = re.compile(r"(?:^|[^aeiou])ris(?:e|es|ed|ing|en)$")
# Verbs whose final l doubles before a suffix in British spelling only.
DOUBLE_L_STEMS = (
    "bevel",
    "cancel",
    "channel",
    "chisel",
    "counsel",
    "dial",
    "enamel",
    "equal",
    "fuel",
    "funnel",
    "grovel",
    "initial",
    "jewel",
    "label",
    "level",
    "libel",
    "marvel",
    "mislabel",
    "model",
    "panel",
    "parallel",
    "pencil",
    "quarrel",
    "refuel",
    "relabel",
    "remodel",
    "revel",
    "rival",
    "shovel",
    "signal",
    "spiral",
    "stencil",
    "total",
    "travel",
    "tunnel",
    "unlabel",
    "unravel",
)
DOUBLE_L_SUFFIXES = ("ing", "ed", "er", "ers", "ation", "ations")
# A spelling merge must also exist in the corpus: the candidate's count must be at least this
# share of the token's, which stops the word list from accepting forms that never occur
# (*controled*).
MIN_SHARE = 0.1
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
_WORD = re.compile(r"^[a-z]+$")


@cache
def _lemmatizer() -> Lemmatizer:
    """The rule lemmatizer of spaCy's small English model; the other components are not loaded."""
    pipeline = spacy.load(SPACY_MODEL, exclude=["tok2vec", "tagger", "parser", "attribute_ruler"])
    return cast(Lemmatizer, pipeline.get_pipe("lemmatizer"))


@cache
def _index(pos: str) -> frozenset[str]:
    return frozenset(_lemmatizer().lookups.get_table("lemma_index")[pos])


@cache
def _lemmas(word: str, pos: str) -> tuple[str, ...]:
    """spaCy's rule-based lemmas of ``word`` read as ``pos``, out of context."""
    token = Doc(_lemmatizer().vocab, words=[word])[0]
    token.pos_ = pos.upper()
    return tuple(_lemmatizer().rule_lemmatize(token))


def _known(word: str) -> bool:
    """Whether the word reduces to a base form in spaCy's lemma index."""
    return any(lemma in _index(pos) for pos in INDEX_POS for lemma in _lemmas(word, pos))


def _common_enough(candidate: str, token: str, counts: Counter[str] | None) -> bool:
    return counts is None or counts[candidate] >= max(1, MIN_SHARE * counts[token])


def regular_plural(token: str, lemma: str) -> bool:
    """Whether ``token`` is a regular English plural of ``lemma``."""
    if lemma == token or len(lemma) < 3:
        return False
    if token in (lemma + "s", lemma + "es"):
        return True
    if lemma.endswith("y") and token == lemma[:-1] + "ies":
        return True
    return lemma.endswith("is") and token == lemma[:-2] + "es"


Votes = Mapping[str, Sequence[tuple[str, str, int]]]


@cache
def _tagger(model: str) -> Language:
    """An English pipeline with what lemmas need: tagger, attribute ruler, lemmatizer."""
    return spacy.load(model, exclude=["parser", "ner", "senter"])


def _vote_chunk(texts: Sequence[str], model: str = SPACY_MODEL) -> Counter[tuple[str, str, str]]:
    votes: Counter[tuple[str, str, str]] = Counter()
    normal: dict[str, str] = {}
    for doc in _tagger(model).pipe(texts, batch_size=256):
        for token in doc:
            if not token.is_alpha:
                continue
            form = normal.setdefault(token.text, preprocess_text(token.text))
            if _WORD.match(form):
                lemma = normal.setdefault(token.lemma_, preprocess_text(token.lemma_))
                votes[(form, token.tag_, lemma)] += 1
    return votes


def _chunks(texts: Iterable[str], size: int) -> Iterator[list[str]]:
    iterator = iter(texts)
    while chunk := list(islice(iterator, size)):
        yield chunk


def lemma_votes(
    texts: Iterable[str], workers: int = 1, model: str = SPACY_MODEL, gpu: bool = False
) -> Counter[tuple[str, str, str]]:
    """Count ``(word type, tag, lemma)`` over every alphabetic token spaCy reads in ``texts``.

    Texts keep their case and punctuation, so the tagger reads real sentences; the word type and
    the lemma are normalised as the processed cells are, so the votes key on the types the map
    is applied to. With ``gpu`` the pipeline runs in this process on the GPU (``workers`` is
    ignored); otherwise ``workers`` processes share the CPU.
    """
    votes: Counter[tuple[str, str, str]] = Counter()
    if gpu:
        require_gpu()
    if gpu or workers <= 1:
        for chunk in _chunks(texts, VOTE_CHUNK):
            votes.update(_vote_chunk(chunk, model))
        return votes
    limit = CHUNKS_IN_FLIGHT_PER_WORKER * workers
    pending: set[Future[Counter[tuple[str, str, str]]]] = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk in _chunks(texts, VOTE_CHUNK):
            if len(pending) >= limit:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    votes.update(future.result())
            pending.add(pool.submit(_vote_chunk, chunk, model))
        for future in wait(pending).done:
            votes.update(future.result())
    return votes


def save_votes(votes: Counter[tuple[str, str, str]], path: Path) -> None:
    rows = [(form, tag, lemma, count) for (form, tag, lemma), count in votes.items()]
    frame = pl.DataFrame(rows, schema=["type", "tag", "lemma", "count"], orient="row")
    frame.sort(["type", "count"], descending=[False, True]).write_parquet(path)


def load_votes(path: Path) -> dict[str, list[tuple[str, str, int]]]:
    votes: dict[str, list[tuple[str, str, int]]] = {}
    for form, tag, lemma, count in pl.read_parquet(path).iter_rows():
        votes.setdefault(form, []).append((tag, lemma, count))
    return votes


def _regular_singulars(token: str) -> list[str]:
    """The forms ``token`` can be a regular -s inflection of, by English spelling: strip -s;
    strip -es only after *s*, *x*, *z*, *ch*, *sh* or *o*; -ies to -y; -ses to -sis."""
    candidates = [token[:-1]]
    if token.endswith("es") and token[:-2].endswith(ES_STEM_ENDINGS):
        candidates.append(token[:-2])
    if token.endswith("ies"):
        candidates.append(token[:-3] + "y")
    if token.endswith("ses"):
        candidates.append(token[:-2] + "is")
    return [form for form in dict.fromkeys(candidates) if regular_plural(token, form)]


@cache
def _noun_exceptions() -> Mapping[str, Sequence[str]]:
    """spaCy's irregular plural nouns and their singulars (*mice* → *mouse*)."""
    return cast(Mapping[str, Sequence[str]], _lemmatizer().lookups.get_table("lemma_exc")["noun"])


def _men_singular(token: str) -> str | None:
    """The -man singular of a -men plural (*freshmen* → *freshman*), a guess by spelling alone."""
    if token.endswith(MEN_SUFFIX) and len(token) > len(MEN_SUFFIX) + 1:
        return token[: -len(MEN_SUFFIX)] + "man"
    return None


def _irregular_singulars(token: str) -> list[str]:
    """Singulars English spelling rules cannot reach: spaCy's irregular nouns (*wives*, *testes*,
    *appendices*), then -men to -man."""
    candidates = [lemma for lemma in _noun_exceptions().get(token, ()) if lemma != token]
    if (men := _men_singular(token)) is not None:
        candidates.append(men)
    return candidates


def _tags(word: str, votes: Votes) -> Counter[str]:
    """The word's readings by tag, proper-noun readings left out."""
    tags: Counter[str] = Counter()
    for tag, _, count in votes.get(word, ()):
        if tag not in PROPER_TAGS:
            tags[tag] += count
    return tags


def voted_lemma(
    token: str, votes: Votes, counts: Counter[str] | None = None
) -> tuple[str, str] | None:
    """``(lemma, rule)`` when spaCy mostly reads ``token`` as an -s inflection or a plural noun,
    or ``None``.

    spaCy's tags decide whether the token is an inflection: at least ``MIN_VOTE_SHARE`` of at
    least ``MIN_VOTES`` readings (proper nouns left out, *United States*) must be plural nouns
    or third-person verbs. English spelling limits the possible singulars (no *plan* for
    *planes*); a plural noun also takes the singulars of spaCy's irregular nouns (*mice*,
    *shelves* → *shelf*, *testes* → *testis*) and -men to -man. Among the candidates the one
    spaCy most often reads as the matching part of speech wins (a singular noun for a plural, a
    base verb for a third-person verb), then the most frequent, then the one spaCy names.
    spaCy's own lemmas of *-es* words are not used: they are unreliable (*uses* → *us*,
    *senses* → *sens*). A singular spaCy does not read as that part of speech at least
    ``MIN_VOTES`` times must, with ``counts``, occur at least ``MIN_SHARE`` as often as the token
    (no *polis* → *poli*); a -man singular must always be read so, and more often as a common
    noun than as a name, as the rule is only a guess (*specimen* has no *speciman*, *hymen* is
    not the plural of the surname *Hyman*). Plurals that are words of their own (*aesthetics*,
    *ethics*) are kept by the override file, not by this rule.
    """
    tags = _tags(token, votes)
    total = sum(tags.values())
    inflected = sum(tags[tag] for tag in INFLECTED_TAGS)
    if total < MIN_VOTES or inflected / total < MIN_VOTE_SHARE:
        return None
    rule = RULE_PLURAL if tags["NNS"] >= tags["VBZ"] else RULE_VERB
    candidates = _regular_singulars(token)
    if rule == RULE_PLURAL:
        candidates = list(dict.fromkeys(candidates + _irregular_singulars(token)))
    if not candidates:
        return None
    matching = SINGULAR_TAGS[rule]
    named: Counter[str] = Counter()
    for tag, lemma_named, count in votes.get(token, ()):
        if tag in INFLECTED_TAGS:
            named[lemma_named] += count

    def preference(form: str) -> tuple[int, int, int]:
        readings = _tags(form, votes)
        frequency = counts[form] if counts else 0
        return sum(readings[tag] for tag in matching), frequency, named[form]

    lemma = max(candidates, key=preference)
    recognised = preference(lemma)[0] >= MIN_VOTES
    if lemma == _men_singular(token) and lemma not in _noun_exceptions().get(token, ()):
        noun_readings = preference(lemma)[0]
        proper_readings = sum(count for tag, _, count in votes.get(lemma, ()) if tag in PROPER_TAGS)
        return (lemma, rule) if recognised and noun_readings > proper_readings else None
    if not recognised and not _common_enough(lemma, token, counts):
        return None
    return lemma, rule


def _spelling_candidate(token: str) -> str | None:
    if token in SPELLINGS:
        return SPELLINGS[token]
    if (
        token.startswith(ISE_EXCEPTION_STEMS)
        or token.endswith(ISE_EXCEPTION_SUFFIXES)
        or _RISE_ROOT.search(token)
    ):
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
    for stem in DOUBLE_L_STEMS:
        if token.startswith(stem + "l") and token[len(stem) + 1 :] in DOUBLE_L_SUFFIXES:
            return token[: len(stem)] + token[len(stem) + 1 :]
    return None


def american(token: str, counts: Counter[str] | None = None) -> str | None:
    """The American spelling of a British form, or ``None`` when the token is not one.

    Without ``counts``, the American form must be in spaCy's word list. With ``counts``, it must
    occur in the corpus at least ``MIN_SHARE`` as often as the token, and at least ``MIN_VOTES``
    times when the word list lacks it: the list has few derived words (*contextualize*,
    *personalization*), so the corpus is the evidence for those.
    """
    if len(token) < MIN_SPELLING_LENGTH:
        return None
    candidate = _spelling_candidate(token)
    if candidate is None or candidate == token:
        return None
    if counts is None:
        return candidate if _known(candidate) else None
    attested = _known(candidate) or counts[candidate] >= MIN_VOTES
    return candidate if attested and _common_enough(candidate, token, counts) else None


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


def normal_form(
    token: str,
    protected: set[str],
    overrides: dict[str, str],
    votes: Votes,
    counts: Counter[str] | None = None,
) -> tuple[str, str]:
    """Return ``(lemma, rule)`` for one token type; ``(token, "")`` when it stays as it is."""
    if token in protected or not _WORD.match(token):
        return token, ""
    if token in overrides:
        lemma = overrides[token] or token
        return lemma, RULE_OVERRIDE if lemma != token else ""
    lemma, rule = token, ""
    if (voted := voted_lemma(token, votes, counts)) is not None:
        lemma, rule = voted
    if lemma in overrides:
        return overrides[lemma] or lemma, rule or RULE_OVERRIDE
    if lemma not in protected and (spelled := american(lemma, counts)) is not None:
        lemma, rule = spelled, RULE_SPELLING if rule == "" else rule
    return lemma, rule


def learn_lemma_map(
    counts: Counter[str], votes: Votes, protected: Iterable[str], overrides: dict[str, str]
) -> LemmaMap:
    """Map every type in ``counts`` that has a normal form other than itself.

    A normal form that is itself mapped is followed to the end of the chain, so applying the map
    once is final (*particularises* → *particularizes* → *particularize*).
    """
    protected_set = set(protected)
    lemma_map = LemmaMap()
    for token in counts:
        lemma, rule = normal_form(token, protected_set, overrides, votes, counts)
        if lemma != token:
            lemma_map.mapping[token] = lemma
            lemma_map.rules[token] = rule
    for token in list(lemma_map.mapping):
        lemma = _end_of_chain(token, lemma_map.mapping)
        if lemma == token:
            del lemma_map.mapping[token], lemma_map.rules[token]
        else:
            lemma_map.mapping[token] = lemma
    return lemma_map


def _end_of_chain(token: str, mapping: Mapping[str, str]) -> str:
    seen = {token}
    lemma = mapping[token]
    while lemma in mapping and lemma not in seen:
        seen.add(lemma)
        lemma = mapping[lemma]
    return lemma


def count_types(paths: Iterable[Path]) -> Counter[str]:
    """Count token types over processed files (one document per line)."""
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                counts.update(line.split())
    return counts
