"""The usages table (decision agenda 4.1-4.6; `docs/preprocessing_phase.md` step 6), part by part.

Each worker process takes whole parts. For every kept work it reads the cached spaCy document,
checks that its text is the abstracts table's, applies the lemma exceptions, and matches the word
table's targets (`usages/target_matching.py`). A mention inside a removed range, or in a sentence
the sentence view marks removed, is never a usage. Each mention is placed in its sentence of the
contextual sentence view and gets its content key and draw key.

The model then annotates each usage (agenda 4.9). With the small model, the cached document's tags
stand and entities are recognised on each sentence alone, in batches. With the transformer, which
the model comparison chose (`docs/usages_audit.md`), the workers stage each usage and the tokens of
each target sentence; then a few processes sharing the GPU annotate every target sentence once, in
float16 and sorted by length (`docs/performance.md`), and its part of speech, tag, morphology,
lemma (after the lemma exceptions) and entity replace the cached ones. Either way each usage gets
its scope, one in-scope usage per work and word in a sentence no exclusion keeps out is drawn
(`usages/usage_selection.py`), and the part is checked against the text and the schema and written.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, NamedTuple

import polars as pl
import spacy
import torch
from pydantic import Field
from pydantic_settings import SettingsConfigDict
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from thinc.api import require_gpu, set_gpu_allocator

from interpretant.catalog.abstracts import load_abstracts_part
from interpretant.catalog.annotation_cache import (
    ANNOTATION_CACHE_COLUMNS,
    load_annotation_cache,
    save_annotation_cache,
)
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
from interpretant.pipelines.workers import Report, map_parts, map_parts_reporting
from interpretant.usages.checks import validate_usages_against_text
from interpretant.usages.entities import (
    SMALL_MODEL,
    SPANS_PER_FORWARD,
    TRANSFORMER_MODEL,
    entity_over_target,
    in_named_instrument,
    load_entity_recognizer,
    load_transformer,
    recognise_sentence_entities,
    require_entity_annotation,
)
from interpretant.usages.identifiers import build_content_key, build_draw_key
from interpretant.usages.target_matching import TargetMention, build_target_matcher, match_targets
from interpretant.usages.usage_selection import is_in_scope, mark_drawn


class UsagesSettings(ParamsSettings):
    """The ``build_usages`` section of ``params.yaml``: the draw's seed, the model, the names."""

    model_config = SettingsConfigDict(yaml_config_section="build_usages")

    seed: int
    model: Literal["en_core_web_sm", "en_core_web_trf"]
    name_labels: tuple[str, ...] = Field(min_length=1)


# Sentences queued before their entities are recognised, and sentences per recognizer batch.
_SENTENCES_PER_FLUSH = 2048
_ENTITY_BATCH_SIZE = 256
# A transformer batch holds at most this many documents, and by default this many tokens: its GPU
# memory grows with its tokens (about 0.35 MB each), so long sentences go in small batches; a
# smaller cap lets more processes share the GPU (`docs/performance.md`).
_TRANSFORMER_BATCH_SIZE = 64
BATCH_TOKENS = 1000
# A sentence longer than this is read as a window of half that either side of each target.
_LONG_SENTENCE_TOKENS = 512
# GPU memory left for other programs (the desktop held 1.5 GB), and each process's CUDA context,
# which PyTorch's allocator does not count (measured 0.3 GB); the rest is split between the
# processes (`docs/performance.md`).
_OTHER_PROGRAMS_GB = 1.6
_CUDA_CONTEXT_GB = 0.3
_REPORT_EVERY = 1000
# Bumped whenever the code that annotates a sentence changes, so the cache is rebuilt.
_ANNOTATION_VERSION = 1
# The packages whose versions shape the annotation; an upgrade starts a new cache.
_ANNOTATION_PACKAGES = ("spacy", "spacy-curated-transformers", TRANSFORMER_MODEL)
_COLUMN_NAMES = USAGES_COLUMNS.names()
_POS = _COLUMN_NAMES.index("pos")
_ENTITY = _COLUMN_NAMES.index("entity")
_INSIDE_NAME = _COLUMN_NAMES.index("inside_name")
_IN_SCOPE = _COLUMN_NAMES.index("in_scope")
_WINDOWED = _COLUMN_NAMES.index("windowed")


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
    staging: str = ""
    gpu_process_count: int = 1
    batch_tokens: int = BATCH_TOKENS
    reuse_cache: bool = False


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
    gpu_process_count: int = 7,
    batch_tokens: int = BATCH_TOKENS,
    resume: bool = False,
    reuse_cache: bool = False,
    progress: Progress = SILENT,
) -> UsagesResult:
    """Write the usages table of every part of the selection.

    With the transformer, ``gpu_process_count`` processes share the GPU, each within an equal share
    of its memory (`docs/performance.md`); with ``resume``, parts the transformer already wrote with
    the current table's columns are kept and skipped. Every sentence is annotated afresh and the
    annotation cache replaced, as the run of record needs; with ``reuse_cache``, sentences found in
    the cache are not annotated again.

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
    counts: Counter[str] = Counter()
    if settings.model == SMALL_MODEL:
        progress.stage("building the usages", total=sum(sizes.values()))
        results = map_parts(
            _build_part,
            dict.fromkeys(parts, job),
            process_count=process_count,
            progress=progress,
            sizes=sizes,
        )
        for result in results.values():
            counts.update(result)
        return UsagesResult(parts=parts, counts=dict(counts))
    if resume:
        done = [part for part in parts if _annotated_by_transformer(catalog, part)]
        counts["skipped_parts"] = len(done)
        parts = [part for part in parts if part not in done]
        sizes = {part: sizes[part] for part in parts}
    catalog.usages.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=catalog.usages.path.parent, prefix="staging_") as staging:
        staged_job = _PartJob(
            job.catalog,
            settings,
            job.words,
            job.spellings,
            job.exceptions,
            staging,
            gpu_process_count,
            batch_tokens,
            reuse_cache,
        )
        progress.stage("finding the mentions", total=sum(sizes.values()))
        found = map_parts(
            _stage_part,
            dict.fromkeys(parts, staged_job),
            process_count=process_count,
            progress=progress,
            sizes=sizes,
        )
        sentences = {part: found[part]["staged_sentences"] for part in parts}
        progress.stage("annotating with the transformer", total=sum(sentences.values()))
        annotated = map_parts_reporting(
            _annotate_part,
            dict.fromkeys(parts, staged_job),
            process_count=gpu_process_count,
            progress=progress,
            sizes=sentences,
        )
    for part in parts:
        counts.update(found[part])
        counts.update(annotated[part])
    return UsagesResult(parts=parts, counts=dict(counts))


def _build_part(part: str, job: _PartJob) -> Counter[str]:
    """Write one part's usages with the small model; its counts of mentions."""
    recognizer = load_entity_recognizer()
    batch = _EntityBatch(recognizer, job.settings.name_labels)
    texts, counts = _match_part(part, job, recognizer.vocab, batch)
    table = pl.DataFrame(batch.rows, schema=USAGES_COLUMNS, orient="row")
    return counts + _finish_part(part, job, table, texts)


def _match_part(
    part: str, job: _PartJob, vocab: Vocab, sink: _EntityBatch | _StagedUsages
) -> tuple[dict[str, str], Counter[str]]:
    """Find one part's mentions and give each usage's row to the sink; the kept texts, counts."""
    ruler = lemma_exception_ruler(vocab, job.exceptions)
    matcher = build_target_matcher(job.words, job.spellings)
    words = {word.lemma: word for word in job.words}
    texts, removals = _kept_works(job.catalog, part)
    sentences = _sentences(job.catalog, part)
    # Each work's static entries are read from the frame when its document comes up: a dict for
    # every work at once took 270 MB a worker (`docs/performance.md`).
    static_view = load_static_view(job.catalog, part, columns=["work_id", "tokens", "starts"])
    static_rows = {work_id: index for index, work_id in enumerate(static_view["work_id"])}
    counts: Counter[str] = Counter()
    for doc in load_annotations(job.catalog, part, vocab):
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
            _static_entries(static_view, static_rows.get(work_id)),
        )
        mentions = match_targets(ruler(doc), matcher)
        counts["mentions"] += len(mentions)
        _add_usages(work, mentions, words, job.settings, counts, sink)
        if sink.is_full:
            sink.flush()
    sink.flush()
    return texts, counts


def _finish_part(
    part: str, job: _PartJob, table: pl.DataFrame, texts: Mapping[str, str]
) -> Counter[str]:
    """Draw one usage per work and word, check the part against the text, write it; its counts."""
    table = mark_drawn(table)
    validate_usages_against_text(table, texts)
    save_usages(job.catalog, table, part=part)
    return Counter(
        {
            "usages": table.height,
            "in_scope": int(table["in_scope"].sum()),
            "drawn": int(table["drawn"].sum()),
        }
    )


# A staged usage: its row with the cached annotation, and where its target is in its sentence.
_STAGED_COLUMNS = pl.Schema(
    dict(USAGES_COLUMNS)
    | {
        "head_offset": pl.Int64(),
        "span_start": pl.Int64(),
        "hyphen_part": pl.Boolean(),
        "window_start": pl.Int64(),
    }
)
_STAGED_SENTENCES = pl.Schema(
    {
        "work_id": pl.String(),
        "sentence_number": pl.Int64(),
        "window_start": pl.Int64(),
        "words": pl.List(pl.String()),
        "spaces": pl.List(pl.Boolean()),
    }
)


class _StagedUsages:
    """Usages waiting for the transformer, and the tokens of each of their sentences."""

    is_full = False

    def __init__(self) -> None:
        self.rows: list[tuple[Any, ...]] = []
        self.sentences: dict[tuple[str, int, int], tuple[list[str], list[bool]]] = {}

    def add(
        self, work: _Work, sentence: _Sentence, mention: TargetMention, row: list[Any], word: Word
    ) -> None:
        """Stage a usage's row, and its sentence's tokens if no usage staged them yet.

        Raises:
            ValueError: if the sentence covers no token.
        """
        del word  # the transformer's reading decides the scope later
        span = work.doc.char_span(sentence.start, sentence.end, alignment_mode="expand")
        if span is None:
            raise ValueError(f"{work.work_id}: sentence {sentence.number} has no tokens")
        head = mention.token_index - span.start
        window_start = 0
        if len(span) > _LONG_SENTENCE_TOKENS:
            half = _LONG_SENTENCE_TOKENS // 2
            window_start = max(0, head - half)
            span = span[window_start : min(len(span), head + half + 1)]
            row[_WINDOWED] = True
        key = (work.work_id, sentence.number, window_start)
        if key not in self.sentences:
            self.sentences[key] = (
                [token.text for token in span],
                [bool(token.whitespace_) for token in span],
            )
        hyphen_part = mention.position is not None
        self.rows.append((*row, head - window_start, span.start_char, hyphen_part, window_start))

    def flush(self) -> None:
        """Nothing waits to be flushed: the staged rows are written once the part is matched."""


def _stage_part(part: str, job: _PartJob) -> Counter[str]:
    """Stage one part's usages and target sentences for the transformer; its counts."""
    sink = _StagedUsages()
    _, counts = _match_part(part, job, spacy.blank("en").vocab, sink)
    folder = Path(job.staging)
    pl.DataFrame(sink.rows, schema=_STAGED_COLUMNS, orient="row").write_parquet(
        folder / f"{part}.rows.parquet"
    )
    pl.DataFrame(
        [(w, n, s, words, spaces) for (w, n, s), (words, spaces) in sink.sentences.items()],
        schema=_STAGED_SENTENCES,
        orient="row",
    ).write_parquet(folder / f"{part}.sentences.parquet")
    counts["staged_sentences"] = len(sink.sentences)
    return counts


@cache
def _transformer(process_count: int) -> Language:
    """The transformer on the GPU, loaded once per process, within its share of the memory.

    PyTorch keeps the memory of a process's largest batch reserved, so processes sharing the GPU
    would otherwise each hold their peak and run it out (`docs/performance.md`); expandable
    segments make that reserve fit the memory in use more closely.
    """
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    set_gpu_allocator("pytorch")
    require_gpu(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    share = (total - _OTHER_PROGRAMS_GB - _CUDA_CONTEXT_GB * process_count) / process_count
    torch.cuda.set_per_process_memory_fraction(share / total)
    return load_transformer()


def _batches(lengths: Sequence[int], tokens_cap: int = BATCH_TOKENS) -> list[range]:
    """Consecutive batches of documents within the document and token caps."""
    batches, start, tokens = [], 0, 0
    for index, length in enumerate(lengths):
        full = index - start >= _TRANSFORMER_BATCH_SIZE or tokens + length > tokens_cap
        if index > start and full:
            batches.append(range(start, index))
            start, tokens = index, 0
        tokens += length
    if start < len(lengths):
        batches.append(range(start, len(lengths)))
    return batches


def _annotate_part(part: str, job: _PartJob, report: Report) -> Counter[str]:
    """Annotate one part's target sentences, from the cache or with the transformer; write it.

    A sentence whose key is in the part's annotation cache is rebuilt from it without the GPU; the
    others go through the transformer, once for each distinct sentence, in batches capped by
    documents and tokens, and are added to the cache. The model loads only if some sentence is
    missing from the cache.
    """
    words = {word.lemma: word for word in job.words}
    folder = Path(job.staging)
    rows = pl.read_parquet(folder / f"{part}.rows.parquet")
    sentences = pl.read_parquet(folder / f"{part}.sentences.parquet").sort(
        pl.col("words").list.len()
    )
    waiting: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    key_columns = ["work_id", "sentence_number", "window_start"]
    for index, key in enumerate(rows.select(key_columns).iter_rows()):
        waiting[key].append(index)
    staged = rows.select(
        "word", "target_start", "target_end", "head_offset", "span_start", "hyphen_part"
    ).rows()
    annotation: dict[str, list[Any]] = {name: [None] * rows.height for name in _ANNOTATED}
    texts_of = sentences.select("words", "spaces").rows()
    keys = sentences.select(key_columns).rows()
    settings_hash = _annotation_settings(job.exceptions)
    folder_name = settings_hash[:16]
    cache_keys = [_annotation_key(settings_hash, words_, spaces) for words_, spaces in texts_of]
    earlier = (
        load_annotation_cache(job.catalog, folder_name, part)
        if job.reuse_cache
        else pl.DataFrame(schema=ANNOTATION_CACHE_COLUMNS)
    )
    cached = {row[0]: row[1:] for row in earlier.iter_rows()}
    added: list[tuple[Any, ...]] = []
    done = 0

    def read(position: int, doc: Doc) -> None:
        nonlocal done
        for index in waiting[keys[position]]:
            word, start, end, head_offset, span_start, hyphen_part = staged[index]
            head = doc[head_offset]
            entity, inside_entity = entity_over_target(
                doc, start - span_start, end - span_start, job.settings.name_labels
            )
            inside_name = inside_entity or in_named_instrument(
                doc, start - span_start, end - span_start
            )
            values = {
                "pos": head.pos_,
                "tag": head.tag_,
                "morph": str(head.morph),
                "lemma": head.lemma_,
                "entity": entity,
                "inside_name": inside_name,
                "in_scope": is_in_scope(
                    head.pos_, words[word], inside_name=inside_name, hyphen_part=hyphen_part
                ),
            }
            for name, value in values.items():
                annotation[name][index] = value
        done += 1
        if done % _REPORT_EVERY == 0:
            report(_REPORT_EVERY)

    hits, sharing = _split_by_cache(cache_keys, cached)
    for position in hits:
        words_, spaces = texts_of[position]
        read(position, _cached_doc(words_, spaces, cached[cache_keys[position]]))
    missing = [positions[0] for positions in sharing.values()]
    if missing:
        transformer = _transformer(job.gpu_process_count)
        exceptions = lemma_exception_ruler(transformer.vocab, job.exceptions)
        lengths = [len(texts_of[position][0]) for position in missing]
        for batch in _batches(lengths, job.batch_tokens):
            positions = [missing[i] for i in batch]
            docs = [
                Doc(transformer.vocab, words=texts_of[i][0], spaces=texts_of[i][1])
                for i in positions
            ]
            pipe = transformer.pipe(docs, batch_size=len(docs))
            for position, doc in zip(positions, pipe, strict=True):
                doc = exceptions(doc)
                require_entity_annotation(doc)
                added.append((cache_keys[position], *_cache_row(doc)))
                for same in sharing[cache_keys[position]]:
                    read(same, doc)
        # A part's longest sentences come last; give their memory back to the other processes.
        torch.cuda.empty_cache()
    report(done % _REPORT_EVERY)
    if added:
        new = pl.DataFrame(added, schema=ANNOTATION_CACHE_COLUMNS, orient="row")
        save_annotation_cache(job.catalog, pl.concat([earlier, new]), folder_name, part)
    table = rows.with_columns(
        *(
            pl.Series(name, values, dtype=USAGES_COLUMNS[name])
            for name, values in annotation.items()
        ),
        pl.lit(TRANSFORMER_MODEL).alias("model"),
    ).select(USAGES_COLUMNS.names())
    texts, _ = _kept_works(job.catalog, part)
    counts = _finish_part(part, job, table, texts)
    waiting_for_transformer = sum(len(positions) for positions in sharing.values())
    counts["cache_hits"] += len(cache_keys) - waiting_for_transformer
    counts["cache_misses"] += len(missing)
    counts["shared_sentences"] += waiting_for_transformer - len(missing)
    return counts


def _split_by_cache(
    cache_keys: Sequence[str], cached: Mapping[str, Any]
) -> tuple[list[int], dict[str, list[int]]]:
    """The sentences found in the cache, and the others grouped by key in order of appearance.

    Identical sentences (boilerplate, near-duplicate abstracts) share a key, so the transformer
    annotates the first of each group once for all of them.
    """
    hits: list[int] = []
    sharing: dict[str, list[int]] = defaultdict(list)
    for position, cache_key in enumerate(cache_keys):
        if cache_key in cached:
            hits.append(position)
        else:
            sharing[cache_key].append(position)
    return hits, dict(sharing)


def _annotation_settings(exceptions: Sequence[LemmaException]) -> str:
    """The hash of everything besides the tokens that shapes a sentence's annotation."""
    settings = {
        "model": TRANSFORMER_MODEL,
        "packages": {package: _installed_version(package) for package in _ANNOTATION_PACKAGES},
        "spans_per_forward": SPANS_PER_FORWARD,
        "float16": True,
        "version": _ANNOTATION_VERSION,
        "exceptions": sorted((e.form, e.tag, e.lemma) for e in exceptions),
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def _installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _annotation_key(settings_hash: str, words: Sequence[str], spaces: Sequence[bool]) -> str:
    """A sentence's key in the annotation cache: its tokens hashed with the settings' hash."""
    tokens = "\x1f".join(words) + "\x1e" + "".join("1" if space else "0" for space in spaces)
    return hashlib.sha256(f"{settings_hash}\x1d{tokens}".encode()).hexdigest()[:32]


def _cache_row(doc: Doc) -> tuple[list[str], ...]:
    """A document's annotation as the cache keeps it: tags, parts of speech, morphs, lemmas, IOB."""
    return (
        [token.tag_ for token in doc],
        [token.pos_ for token in doc],
        [str(token.morph) for token in doc],
        [token.lemma_ for token in doc],
        [f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else "O" for token in doc],
    )


def _cached_doc(words: list[str], spaces: list[bool], row: tuple[Any, ...]) -> Doc:
    """A sentence rebuilt from its cached annotation."""
    tags, pos, morphs, lemmas, ents = row
    return Doc(
        _blank_vocab(),
        words=words,
        spaces=spaces,
        tags=tags,
        pos=pos,
        morphs=morphs,
        lemmas=lemmas,
        ents=ents,
    )


@cache
def _blank_vocab() -> Vocab:
    """A vocabulary for documents rebuilt from the cache, one per process."""
    vocab: Vocab = spacy.blank("en").vocab
    return vocab


def _annotated_by_transformer(catalog: CatalogConfig, part: str) -> bool:
    """Whether a part's saved usages were written by the transformer with the current columns.

    A part without usages counts as done: there is nothing to annotate in it.
    """
    path = catalog.usages.path / f"{part}.parquet"
    if not path.exists():
        return False
    if list(pl.read_parquet_schema(path)) != USAGES_COLUMNS.names():
        return False
    models = pl.read_parquet(path, columns=["model"])["model"]
    return models.len() == 0 or set(models) == {TRANSFORMER_MODEL}


# The columns the transformer's reading sets.
_ANNOTATED = ("pos", "tag", "morph", "lemma", "entity", "inside_name", "in_scope")


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
        self.rows: list[tuple[Any, ...]] = []

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

    def flush(self) -> None:
        """Recognise the queued sentences' entities and keep their usages' finished rows."""
        keys = list(self.spans)
        recognised = recognise_sentence_entities(
            (self.spans[key] for key in keys), self.recognizer, batch_size=_ENTITY_BATCH_SIZE
        )
        entity_docs = dict(zip(keys, recognised, strict=True))
        for waiting in self.waiting:
            doc = entity_docs[waiting.sentence]
            entity, inside_entity = entity_over_target(
                doc, waiting.start, waiting.end, self.name_labels
            )
            inside_name = inside_entity or in_named_instrument(doc, waiting.start, waiting.end)
            row = waiting.row
            row[_ENTITY] = entity
            row[_INSIDE_NAME] = inside_name
            row[_IN_SCOPE] = is_in_scope(
                row[_POS], waiting.word, inside_name=inside_name, hyphen_part=waiting.hyphen_part
            )
            self.rows.append(tuple(row))
        self.spans.clear()
        self.waiting.clear()


def _add_usages(
    work: _Work,
    mentions: Sequence[TargetMention],
    words: Mapping[str, Word],
    settings: UsagesSettings,
    counts: Counter[str],
    sink: _EntityBatch | _StagedUsages,
) -> None:
    """Give one work's usages, in the table's column order, to the sink for their annotation."""
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
            None,  # inside a name, set by the flush
            None,  # in scope, set by the flush
            False,
            settings.model,
            False,  # windowed, set by the staging for a long sentence
        ]
        sink.add(work, sentence, mention, row, words[mention.word])


def _static_entries(view: pl.DataFrame, row: int | None) -> dict[int, str]:
    """One work's static-view entries by character offset; none if the view has no row for it."""
    if row is None:
        return {}
    _, tokens, starts = view.row(row)
    return dict(zip(starts, tokens, strict=True))


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
