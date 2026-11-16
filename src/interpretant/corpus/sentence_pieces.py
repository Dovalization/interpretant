"""A text cut at its sentences' edges, so the word repairs change only its English sentences.

The word repairs (decision agenda 2.3) were written for English: in a foreign sentence they put
English words in (Spanish *considera* into *consider a*), and a foreign sentence with English
words in it can read as English to lingua, so it would be kept. The base sentence languages,
read before the repairs (docs/plan_early_languages.md), say which sentences are English. A text
is cut into pieces, its sentences and the gaps between them; a repair changes only the pieces it
may: an English sentence, or an undetermined one in an abstract mostly English. Joining the
pieces gives the repaired text, and each sentence's new character range follows from the pieces'
lengths, so the languages stay on the text the later stages read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

ENGLISH = "en"
UNDETERMINED = "und"

# A sentence as the language tables give it: start, end, language, lingua's confidence.
SentenceLanguage = tuple[int, int, str, float]


@dataclass(frozen=True)
class SentencePiece:
    """Which piece a sentence is, and its language with lingua's confidence."""

    piece: int
    language: str
    confidence: float


@dataclass(frozen=True)
class PiecedText:
    """A work's text as pieces, whether the repairs may change each, and its sentences."""

    work_id: str
    pieces: tuple[str, ...]
    repairable: tuple[bool, ...]
    sentences: tuple[SentencePiece, ...]

    @property
    def text(self) -> str:
        """The text the pieces make."""
        return "".join(self.pieces)


def cut_into_pieces(work_id: str, text: str, sentences: Iterable[SentenceLanguage]) -> PiecedText:
    """The text cut at its sentences' edges, each piece marked repairable or not.

    A sentence is repairable when it is English, or undetermined in an abstract with at least as
    many characters in English sentences as in sentences of other languages. The gaps between
    sentences, and any text outside them, are never repaired.
    """
    ordered = sorted(sentences)
    english = sum(end - start for start, end, language, _ in ordered if language == ENGLISH)
    foreign = sum(
        end - start
        for start, end, language, _ in ordered
        if language not in (ENGLISH, UNDETERMINED)
    )
    mostly_english = english >= foreign
    pieces: list[str] = []
    repairable: list[bool] = []
    placed: list[SentencePiece] = []
    position = 0
    for start, end, language, confidence in ordered:
        if start > position:
            pieces.append(text[position:start])
            repairable.append(False)
        placed.append(SentencePiece(len(pieces), language, confidence))
        pieces.append(text[start:end])
        repairable.append(language == ENGLISH or (language == UNDETERMINED and mostly_english))
        position = end
    if position < len(text):
        pieces.append(text[position:])
        repairable.append(False)
    return PiecedText(work_id, tuple(pieces), tuple(repairable), tuple(placed))


def repair_pieces(pieced: PiecedText, repair: Callable[[str], str]) -> PiecedText:
    """The pieced text with ``repair`` applied to each piece it may change."""
    pieces = tuple(
        repair(piece) if may else piece
        for piece, may in zip(pieced.pieces, pieced.repairable, strict=True)
    )
    return replace(pieced, pieces=pieces)


def sentence_ranges(pieced: PiecedText) -> list[SentenceLanguage]:
    """Each sentence's character range in the text the pieces make, with its language."""
    starts = [0]
    for piece in pieced.pieces:
        starts.append(starts[-1] + len(piece))
    return [
        (starts[sentence.piece], starts[sentence.piece + 1], sentence.language, sentence.confidence)
        for sentence in pieced.sentences
    ]


def pieced_texts(
    texts: Iterable[tuple[str, str]], sentences: Sequence[tuple[str, int, int, str, float]]
) -> list[PiecedText]:
    """Each work's text cut into pieces, from its sentences' rows (work id first)."""
    by_work: dict[str, list[SentenceLanguage]] = {}
    for work_id, start, end, language, confidence in sentences:
        by_work.setdefault(work_id, []).append((start, end, language, confidence))
    return [cut_into_pieces(work_id, text, by_work.get(work_id, ())) for work_id, text in texts]
