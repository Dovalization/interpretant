"""The usages table (decision agenda 4.1-4.6; `docs/preprocessing_phase.md` step 6), part by part.

Each worker process takes whole parts. For every kept work it reads the cached spaCy document into
the small model's vocabulary, checks that its text is the abstracts table's, applies the lemma
exceptions, and matches the word table's targets (`usages/target_matching.py`). A mention inside a
removed range, or in a sentence the sentence view marks removed, is never a usage. Each mention is
placed in its sentence of the contextual sentence view; the sentence's entities are recognised on
it alone (`usages/entities.py`), and the mention gets its content key, draw key and scope. Then one
in-scope usage per work and word, in a sentence no exclusion keeps out, is drawn
(`usages/usage_selection.py`), and the part is written and checked against the table's schema.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import polars as pl
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from spacy.language import Language
from spacy.tokens import Doc, Span

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.annotations import WORK_ID, load_annotations
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.lemma_exceptions import LemmaException, load_lemma_exceptions
from interpretant.catalog.sentence_view import REMOVED, load_sentence_view_part
from interpretant.catalog.spellings import load_spellings
from interpretant.catalog.static_view import load_static_view
from interpretant.catalog.usages import USAGES_COLUMNS, save_usages
from interpretant.catalog.words import Word, load_words
from interpretant.corpus.lemma_exceptions import lemma_exception_ruler
from interpretant.corpus.sentence_view import merge_ranges
from interpretant.pipelines.corpus import corpus_cells, corpus_parts
from interpretant.pipelines.params import CorpusSelection, ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.pipelines.workers import map_parts
from interpretant.usages.checks import validate_usages_against_text
from interpretant.usages.entities import (
    entity_label,
    load_entity_recognizer,
    recognise_sentence_entities,
)
from interpretant.usages.identifiers import build_content_key, build_draw_key
from interpretant.usages.target_matching import TargetMention, build_target_matcher, match_targets
from interpretant.usages.usage_selection import is_in_scope, mark_drawn


class UsagesSettings(ParamsSettings):
    """The ``build_usages`` section of ``params.yaml``: the draw's seed, the model, the names.

    Only the small model annotates the table until the model comparison (agenda 4.9) decides.
    """

    model_config = SettingsConfigDict(yaml_config_section="build_usages")

    seed: int
    model: Literal["en_core_web_sm"]
    name_labels: tuple[str, ...] = Field(min_length=1)


# Sentences queued before their entities are recognised, and sentences per recognizer batch.
_SENTENCES_PER_FLUSH = 2048
_ENTITY_BATCH_SIZE = 256
_COLUMN_NAMES = USAGES_COLUMNS.names()
_POS = _COLUMN_NAMES.index("pos")
_ENTITY = _COLUMN_NAMES.index("entity")
_IN_SCOPE = _COLUMN_NAMES.index("in_scope")


class TextMismatchError(ValueError):
    """A cached document's text is not its work's text in the abstracts table."""


@dataclass(frozen=True)
class UsagesResult:
    """What the stage wrote: the parts, and the counts of mentions by what became of them."""

    parts: list[str]
    counts: dict[str, int]


@dataclass(frozen=True)
class _PartJob:
    """What a worker needs for its parts (picklable)."""

    catalog: CatalogConfig
    settings: UsagesSettings
    words: tuple[Word, ...]
    spellings: dict[str, str]
    exceptions: tuple[LemmaException, ...]


class _Sentence(NamedTuple):
    """One sentence of the sentence view, as a usage records it."""

    number: int
    start: int
    end: int
    word_count: int
    language: str
    confidence: float
    exclusion: str | None


@dataclass(frozen=True)
class _Work:
    """One kept work: its document, text, merged removed ranges, sentences and static entries."""

    work_id: str
    doc: Doc
    text: str
    removed: list[tuple[int, int]]
    sentences: list[_Sentence]
    static: dict[int, str]


def build_usages(
    catalog: CatalogConfig,
    selection: CorpusSelection,
    settings: UsagesSettings,
    *,
    process_count: int = 1,
    progress: Progress = SILENT,
) -> UsagesResult:
    """Write the usages table of every part of the selection.

    Raises:
        ValueError: if no raw cell matches the selection, or a curated file is malformed.
        TextMismatchError: if a cached document's text is not its work's text.
        FileNotFoundError: if a part's cache, abstracts, sentence view or static view is missing.
        pandera.errors.SchemaError: if a part's rows break the usages table's schema, or a
            usage's content key or offsets do not match its text.
    """
    parts = corpus_parts(corpus_cells(catalog, selection))
    job = _PartJob(
        catalog,
        settings,
        tuple(load_words(catalog)),
        {spelling.form: spelling.standard for spelling in load_spellings(catalog)},
        tuple(load_lemma_exceptions(catalog)),
    )
    # Largest parts start first, so a long part does not finish alone (`docs/performance.md`).
    sizes = {part: _kept_work_count(catalog, part) for part in parts}
    progress.stage("building the usages", total=sum(sizes.values()))
    results = map_parts(
        _build_part,
        dict.fromkeys(parts, job),
        process_count=process_count,
        progress=progress,
        sizes=sizes,
    )
    counts: Counter[str] = Counter()
    for result in results.values():
        counts.update(result)
    return UsagesResult(parts=parts, counts=dict(counts))


def _build_part(part: str, job: _PartJob) -> Counter[str]:
    """Write one part's usages; its counts of mentions by what became of them."""
    recognizer = load_entity_recognizer()
    ruler = lemma_exception_ruler(recognizer.vocab, job.exceptions)
    matcher = build_target_matcher(job.words, job.spellings)
    words = {word.lemma: word for word in job.words}
    texts, removals = _kept_works(job.catalog, part)
    sentences = _sentences(job.catalog, part)
    static = {
        work_id: dict(zip(starts, tokens, strict=True))
        for work_id, tokens, starts in load_static_view(
            job.catalog, part, columns=["work_id", "tokens", "starts"]
        ).iter_rows()
    }
    rows: list[tuple[Any, ...]] = []
    counts: Counter[str] = Counter()
    batch = _EntityBatch(recognizer, job.settings.name_labels)
    for doc in load_annotations(job.catalog, part, recognizer.vocab):
        work_id = doc.user_data[WORK_ID]
        if work_id not in texts:
            continue
        if doc.text != texts[work_id]:
            raise TextMismatchError(f"the cached document of {work_id} is not its text")
        work = _Work(
            work_id,
            doc,
            texts[work_id],
            removals[work_id],
            sentences.get(work_id, []),
            static.get(work_id, {}),
        )
        mentions = match_targets(ruler(doc), matcher)
        counts["mentions"] += len(mentions)
        _add_usages(work, mentions, words, job.settings, counts, batch)
        if batch.is_full:
            rows.extend(batch.flush())
    rows.extend(batch.flush())
    table = mark_drawn(pl.DataFrame(rows, schema=USAGES_COLUMNS, orient="row"))
    validate_usages_against_text(table, texts)
    save_usages(job.catalog, table, part=part)
    counts["usages"] += table.height
    counts["in_scope"] += int(table["in_scope"].sum())
    counts["drawn"] += int(table["drawn"].sum())
    return counts


@dataclass(frozen=True)
class _Waiting:
    """A usage's row waiting for its entity and scope, and where its target is in its sentence."""

    row: list[Any]
    sentence: tuple[str, int]
    start: int
    end: int
    word: Word
    hyphen_part: bool


class _EntityBatch:
    """Usages waiting for their sentences' entities, recognised a batch of sentences at a time.

    The recognizer's ``pipe`` predicts many sentences per call, which the profile of one call per
    sentence showed spending a third of its time on the call itself (`docs/performance.md`).
    """

    def __init__(self, recognizer: Language, name_labels: tuple[str, ...]) -> None:
        self.recognizer = recognizer
        self.name_labels = name_labels
        self.spans: dict[tuple[str, int], Span] = {}
        self.waiting: list[_Waiting] = []

    @property
    def is_full(self) -> bool:
        """Whether enough sentences wait to recognise them."""
        return len(self.spans) >= _SENTENCES_PER_FLUSH

    def add(
        self, work: _Work, sentence: _Sentence, mention: TargetMention, row: list[Any], word: Word
    ) -> None:
        """Queue a usage's row, and its sentence if no usage queued it yet.

        Raises:
            ValueError: if the sentence covers no token.
        """
        key = (work.work_id, sentence.number)
        span = self.spans.get(key)
        if span is None:
            span = work.doc.char_span(sentence.start, sentence.end, alignment_mode="expand")
            if span is None:
                raise ValueError(f"{work.work_id}: sentence {sentence.number} has no tokens")
            self.spans[key] = span
        self.waiting.append(
            _Waiting(
                row,
                key,
                mention.start_char - span.start_char,
                mention.end_char - span.start_char,
                word,
                mention.position is not None,
            )
        )

    def flush(self) -> list[tuple[Any, ...]]:
        """Recognise the queued sentences' entities and return their usages' finished rows."""
        keys = list(self.spans)
        recognised = recognise_sentence_entities(
            (self.spans[key] for key in keys), self.recognizer, batch_size=_ENTITY_BATCH_SIZE
        )
        entity_docs = dict(zip(keys, recognised, strict=True))
        rows = []
        for waiting in self.waiting:
            entity = entity_label(entity_docs[waiting.sentence], waiting.start, waiting.end)
            row = waiting.row
            row[_ENTITY] = entity
            row[_IN_SCOPE] = is_in_scope(
                row[_POS], waiting.word, entity, self.name_labels, hyphen_part=waiting.hyphen_part
            )
            rows.append(tuple(row))
        self.spans.clear()
        self.waiting.clear()
        return rows


def _add_usages(
    work: _Work,
    mentions: Sequence[TargetMention],
    words: Mapping[str, Word],
    settings: UsagesSettings,
    counts: Counter[str],
    batch: _EntityBatch,
) -> None:
    """Queue one work's usages, in the table's column order, for their entities and scope."""
    sentence_starts = [sentence.start for sentence in work.sentences]
    removed_starts = [start for start, _ in work.removed]
    for mention in mentions:
        inside = bisect_right(removed_starts, mention.start_char) - 1
        if inside >= 0 and mention.start_char < work.removed[inside][1]:
            counts["inside_removed_ranges"] += 1
            continue
        index = bisect_right(sentence_starts, mention.start_char) - 1
        if index < 0 or mention.end_char > work.sentences[index].end:
            counts["outside_sentences"] += 1
            continue
        sentence = work.sentences[index]
        if sentence.exclusion == REMOVED:
            counts["in_removed_sentences"] += 1
            continue
        head = work.doc[mention.token_index]
        content_key = build_content_key(
            work.text[sentence.start : sentence.end],
            mention.start_char - sentence.start,
            mention.end_char - sentence.start,
        )
        row = [
            work.work_id,
            mention.word,
            content_key,
            build_draw_key(settings.seed, work.work_id, mention.word, content_key),
            sentence.number,
            sentence.start,
            sentence.end,
            mention.start_char,
            mention.end_char,
            mention.compound,
            mention.position,
            head.pos_,
            head.tag_,
            str(head.morph),
            head.lemma_,
            None,  # the entity, set by the flush
            work.static.get(head.idx),
            sentence.word_count,
            sentence.language,
            sentence.confidence,
            sentence.exclusion,
            None,  # in scope, set by the flush
            False,
            settings.model,
        ]
        batch.add(work, sentence, mention, row, words[mention.word])


def _kept_work_count(catalog: CatalogConfig, part: str) -> int:
    """How many works of an abstracts part are kept."""
    return load_abstracts_part(catalog, part, ["dropped_reason"])["dropped_reason"].null_count()


def _kept_works(
    catalog: CatalogConfig, part: str
) -> tuple[dict[str, str], dict[str, list[tuple[int, int]]]]:
    """Each kept work's text and merged removed ranges in one abstracts part."""
    abstracts = load_abstracts_part(
        catalog, part, ["work_id", "text", "removals", "dropped_reason"]
    ).filter(pl.col("dropped_reason").is_null())
    texts = dict(zip(abstracts["work_id"], abstracts["text"], strict=True))
    removals = {
        work_id: merge_ranges((removal["start"], removal["end"]) for removal in ranges)
        for work_id, ranges in abstracts.select("work_id", "removals").iter_rows()
    }
    return texts, removals


def _sentences(catalog: CatalogConfig, part: str) -> dict[str, list[_Sentence]]:
    """Each work's sentence-view sentences, in text order."""
    view = load_sentence_view_part(catalog, part).sort("work_id", "start")
    found: dict[str, list[_Sentence]] = {}
    for work_id, *sentence in view.select(
        "work_id",
        "sentence_number",
        "start",
        "end",
        "word_count",
        "language",
        "confidence",
        "exclusion",
    ).iter_rows():
        found.setdefault(work_id, []).append(_Sentence(*sentence))
    return found
