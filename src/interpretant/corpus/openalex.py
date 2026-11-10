"""OpenAlex corpus source — seeded abstract samples per field and decade.

OpenAlex labels every work with one topic taxonomy, so philosophy, psychology, neuroscience,
computer science and the social sciences share field labels across the whole series. Each
cell of the field × decade grid is a seeded random sample of English research prose with
abstracts (``RESEARCH_TYPES``), stored as ``data/raw/openalex/<field>/<decade>.jsonl.gz``
with one work per line. Ingest drops duplicate abstracts and archive catalogue records
("Letter from X to Y dated ...", "Script from ..."), which OpenAlex indexes as works, then
caps any author at 1% of a cell's works (``interpretant.corpus.caps``).

Requests need an API key (``OPENALEX_API_KEY`` in the environment or ``.env``): a list page
of 200 works costs one credit from the key's daily budget.
"""

from __future__ import annotations

import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from interpretant.corpus.caps import MAX_AUTHOR_SHARE, cap_author_works
from interpretant.corpus.preprocessing import preprocess_text

API_URL = "https://api.openalex.org/works"
PAGE_SIZE = 200
MAX_SAMPLE = 10_000  # OpenAlex rejects larger samples

# A cell is a field's research prose: one genre rule for every field, so a change between
# decades is less likely to be a change of genre. Book reviews, editorials, letters, preprints
# (often the same paper as a published article), datasets and archive records are left out.
RESEARCH_TYPES = ("article", "review", "book-chapter", "book", "dissertation", "conference-paper")

_CATALOGUE_RECORD = re.compile(
    r"^\s*(letter|script|telegram|memo(randum)?|postcard|note)\s+(from|to|by)\b", re.IGNORECASE
)
# Publisher pages scraped in place of an abstract: a paywall banner at the start, citation
# and alert widgets, the Altmetric donut, or the share bar ("Facebook Twitter LinkedIn ...").
# A lone "Twitter" or "Facebook" is left alone: social media are a research topic.
_WEB_PAGE = re.compile(
    r"^\s*no ?access|export citation|save to binder|add to favorites|track citations|"
    r"e-alerts|clicking on the donut|facebook\W{0,3}twitter\W{0,3}(linkedin|reddit|e-?mail)|"
    r"download citation file|easybib|refworks|request reuse permissions|rft_id|(%2[0-9a-f]){3}",
    re.IGNORECASE,
)
# A citation string standing in for an abstract: "Volume 3, Issue 2, ..., Pages 120-123".
_CITATION_RECORD = re.compile(r"\bvolume \d+,? (issue|number) \d+\b.*\bpages? \d+", re.IGNORECASE)
# Footers and promotions appended to real abstracts: JSTOR's download stamp and
# "Click here to purchase a PDF version of this article ...".
_APPENDED_FOOTER = re.compile(
    r"this content downloaded from [\d.]+.*?utc(\s*all use subject to\s*\S+)?|click here[^.]*\.",
    re.IGNORECASE,
)
# "© 2019 Elsevier B.V. All rights reserved." or, without the formula, to the sentence end.
_COPYRIGHT = re.compile(
    r"(©|\(c\)|copyright).{0,150}?all rights reserved\.?|(©|\(c\)|copyright)\s*\d{4}\b[^.]*\.?",
    re.IGNORECASE,
)
MIN_ABSTRACT_WORDS = 5

# Function words for a cheap language check: OpenAlex labels some French, Spanish and
# Portuguese abstracts (and mastheads with no prose at all) as English.
_ENGLISH_WORDS = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "that",
        "for",
        "are",
        "with",
        "as",
        "by",
        "this",
        "be",
        "on",
        "from",
        "which",
        "it",
        "an",
        "at",
        "or",
        "not",
        "have",
        "we",
        "these",
        "their",
        "was",
        "were",
        "has",
        "been",
        "can",
    ]
)
_OTHER_WORDS = frozenset(
    [
        "le",
        "la",
        "les",
        "des",
        "du",
        "une",
        "est",
        "que",
        "qui",
        "dans",
        "pour",
        "sur",
        "pas",
        "au",
        "aux",
        "ce",
        "cette",
        "ces",
        "sont",
        "avec",
        "par",
        "nous",
        "el",
        "los",
        "las",
        "del",
        "por",
        "para",
        "una",
        "con",
        "como",
        "pero",
        "sus",
        "esta",
        "este",
        "entre",
        "der",
        "die",
        "das",
        "und",
        "ist",
        "nicht",
        "mit",
        "sich",
        "auf",
        "dem",
        "den",
        "zur",
        "os",
        "um",
        "uma",
        "dos",
        "das",
        "pelo",
        "pela",
        "nao",
        "sao",
    ]
)


def looks_english(text: str) -> bool:
    """Return whether a preprocessed abstract reads as English prose."""
    tokens = text.split()
    english = sum(token in _ENGLISH_WORDS for token in tokens)
    other = sum(token in _OTHER_WORDS for token in tokens)
    return english > other and english >= 0.05 * len(tokens)


def strip_boilerplate(abstract: str) -> str:
    """Remove copyright footers, download stamps and promotions from a real abstract.

    URLs and DOIs are removed later by ``preprocess_text``, for every source.
    """
    return _COPYRIGHT.sub(" ", _APPENDED_FOOTER.sub(" ", abstract))


# Field slug → OpenAlex filter. Slugs are alphanumeric so processed files follow the
# ``<source>_<decade>.txt`` convention the audit reads.
FIELDS: dict[str, str] = {
    "philosophy": "primary_topic.subfield.id:1211",
    "histphilsci": "primary_topic.subfield.id:1207",
    "linguistics": "primary_topic.subfield.id:1203",
    "psychology": "primary_topic.field.id:32",
    "cogpsych": "primary_topic.subfield.id:3205",
    "neuroscience": "primary_topic.field.id:28",
    "compsci": "primary_topic.field.id:17",
    "ai": "primary_topic.subfield.id:1702",
    "hci": "primary_topic.subfield.id:1709",
    "sociology": "primary_topic.subfield.id:3312",
    "communication": "primary_topic.subfield.id:3315",
}


class BudgetExhaustedError(RuntimeError):
    """The API key's daily credit budget is spent; downloads can resume after the reset."""


class OpenAlexSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openalex_api_key: str


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    """Rebuild abstract text from OpenAlex's ``{word: [positions]}`` index."""
    if not index:
        return ""
    positions = {position: word for word, places in index.items() for position in places}
    return " ".join(positions[position] for position in sorted(positions))


def work_record(work: dict[str, Any]) -> dict[str, object]:
    """Keep what the grid needs from a work: id, year, type, author ids and abstract text."""
    authors = [
        str(authorship["author"]["id"])
        for authorship in work.get("authorships") or []
        if (authorship.get("author") or {}).get("id")
    ]
    return {
        "id": work["id"],
        "year": work["publication_year"],
        "type": work.get("type", ""),
        "authors": authors,
        "abstract": abstract_from_inverted_index(work.get("abstract_inverted_index")),
    }


def cell_filter(field: str, decade: int) -> str:
    return (
        f"{FIELDS[field]},publication_year:{decade}-{decade + 9},language:en,"
        f"has_abstract:true,type:{'|'.join(RESEARCH_TYPES)}"
    )


@dataclass
class CleanedCell:
    """A cell's documents after cleaning, with what each rule removed."""

    documents: list[str] = dataclass_field(default_factory=list)
    ids: list[str] = dataclass_field(default_factory=list)
    empty: int = 0
    duplicates: int = 0
    catalogue_records: int = 0
    web_pages: int = 0
    non_english: int = 0
    author_capped: int = 0


def clean_cell(
    records: Iterable[dict[str, Any]], max_author_share: float = MAX_AUTHOR_SHARE, seed: int = 42
) -> CleanedCell:
    """Preprocess a cell's abstracts and apply the cleaning rules in order.

    Catalogue records, scraped web pages, citation strings, abstracts too short to be
    abstracts, non-English abstracts and exact duplicates go first, so they do not count
    towards any author's share;
    then no author may appear in more than ``max_author_share`` of the remaining works. URLs
    and copyright footers are stripped from the abstracts that stay.
    """
    cell = CleanedCell()
    seen: set[str] = set()
    kept: list[tuple[list[str], str, str]] = []
    for record in records:
        abstract = str(record["abstract"])
        if _CATALOGUE_RECORD.match(abstract) or _CITATION_RECORD.search(abstract):
            cell.catalogue_records += 1
            continue
        if _WEB_PAGE.search(abstract):
            cell.web_pages += 1
            continue
        text = preprocess_text(strip_boilerplate(abstract))
        if len(text.split()) < MIN_ABSTRACT_WORDS:
            cell.empty += 1
        elif not looks_english(text):
            cell.non_english += 1
        elif text in seen:
            cell.duplicates += 1
        else:
            seen.add(text)
            kept.append((list(record.get("authors") or []), text, str(record.get("id") or "")))
    allowed = cap_author_works([authors for authors, _, _ in kept], max_author_share, seed)
    cell.author_capped = len(kept) - len(allowed)
    cell.documents = [text for index, (_, text, _) in enumerate(kept) if index in allowed]
    cell.ids = [work_id for index, (_, _, work_id) in enumerate(kept) if index in allowed]
    return cell


def cell_path(raw_dir: Path, field: str, decade: int) -> Path:
    return raw_dir / field / f"{decade}.jsonl.gz"


_SELECT = "id,publication_year,type,authorships,abstract_inverted_index"
IDS_PER_REQUEST = 50  # OpenAlex accepts up to 50 values in one OR filter


class OpenAlexClient:
    """Fetches seeded sample pages or whole cells, waiting out short rate limits."""

    def __init__(self, api_key: str, pause_seconds: float = 0.1, max_attempts: int = 8) -> None:
        self.api_key = api_key
        self.pause_seconds = pause_seconds
        self.max_attempts = max_attempts

    def count(self, filters: str) -> int:
        return int(self._get({"filter": filters, "per-page": 1, "select": "id"})["meta"]["count"])

    def sample_page(self, filters: str, sample: int, seed: int, page: int) -> list[dict[str, Any]]:
        params = {
            "filter": filters,
            "sample": sample,
            "seed": seed,
            "per-page": PAGE_SIZE,
            "page": page,
            "select": _SELECT,
        }
        results: list[dict[str, Any]] = self._get(params)["results"]
        return results

    def cursor_page(self, filters: str, cursor: str) -> tuple[list[dict[str, Any]], str | None]:
        """Return one page of every matching work and the cursor for the next page."""
        params = {"filter": filters, "per-page": PAGE_SIZE, "cursor": cursor, "select": _SELECT}
        data = self._get(params)
        return data["results"], data["meta"].get("next_cursor")

    def sources_for(self, ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
        """Return each work's primary source (id, name), fetched ``IDS_PER_REQUEST`` at a time."""
        sources: dict[str, tuple[str | None, str | None]] = {}
        for start in range(0, len(ids), IDS_PER_REQUEST):
            batch = ids[start : start + IDS_PER_REQUEST]
            short = [work_id.rsplit("/", 1)[-1] for work_id in batch]
            results = self._get(
                {
                    "filter": "ids.openalex:" + "|".join(short),
                    "per-page": IDS_PER_REQUEST,
                    "select": "id,primary_location",
                }
            )["results"]
            for work in results:
                source = (work.get("primary_location") or {}).get("source") or {}
                sources[str(work["id"])] = (source.get("id"), source.get("display_name"))
            for work_id in batch:
                sources.setdefault(work_id, (None, None))
        return sources

    def _get(self, params: dict[str, object]) -> dict[str, Any]:
        url = f"{API_URL}?{urllib.parse.urlencode({**params, 'api_key': self.api_key})}"
        for attempt in range(self.max_attempts):
            time.sleep(self.pause_seconds)
            try:
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    return json.load(response)  # type: ignore[no-any-return]
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    raise
                retry_after = _retry_after(error)
                if retry_after > 600:
                    raise BudgetExhaustedError(f"retry after {retry_after}s") from error
                time.sleep(max(retry_after, 2**attempt))
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAlex request failed {self.max_attempts} times: {params}")


def _retry_after(error: urllib.error.HTTPError) -> int:
    try:
        return int(json.load(error).get("retryAfter", 0))
    except (ValueError, AttributeError, json.JSONDecodeError):
        return 0


def download_cell(
    client: OpenAlexClient,
    raw_dir: Path,
    field: str,
    decade: int,
    sample_size: int = MAX_SAMPLE,
    seed: int = 42,
    full: bool = False,
) -> int:
    """Download one cell and return its number of works; existing cells are kept.

    A cell is a seeded sample of up to ``sample_size`` works, or every matching work when
    ``full`` is set (for a word too rare to measure in a sample). The file is written under a
    temporary name and renamed when complete, so an interrupted download leaves no partial
    cell behind.
    """
    path = cell_path(raw_dir, field, decade)
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as existing:
            return sum(1 for _ in existing)
    filters = cell_filter(field, decade)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    written = 0
    with gzip.open(partial, "wt", encoding="utf-8") as output:
        for work in _cell_works(client, filters, sample_size, seed, full):
            output.write(json.dumps(work_record(work), ensure_ascii=False) + "\n")
            written += 1
    partial.rename(path)
    return written


def fetch_sources(client: OpenAlexClient, raw_dir: Path, field: str, decade: int) -> Path:
    """Look up the primary source of every work in a downloaded cell.

    Writes ``<field>/sources_<decade>.jsonl`` (id, source_id, source_name) next to the cell and
    skips ids already written, so an interrupted run resumes.
    """
    output = raw_dir / field / f"sources_{decade}.jsonl"
    done: set[str] = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            done = {json.loads(line)["id"] for line in handle if line.strip()}
    with gzip.open(cell_path(raw_dir, field, decade), "rt", encoding="utf-8") as cell:
        pending = [
            record["id"]
            for record in (json.loads(line) for line in cell)
            if record["id"] not in done
        ]
    with output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), IDS_PER_REQUEST * 10):
            batch = pending[start : start + IDS_PER_REQUEST * 10]
            for work_id, (source_id, source_name) in client.sources_for(batch).items():
                handle.write(
                    json.dumps(
                        {"id": work_id, "source_id": source_id, "source_name": source_name},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
    return output


def _cell_works(
    client: OpenAlexClient, filters: str, sample_size: int, seed: int, full: bool
) -> Iterator[dict[str, Any]]:
    if full:
        cursor: str | None = "*"
        while cursor:
            works, cursor = client.cursor_page(filters, cursor)
            if not works:
                return
            yield from works
        return
    sample = min(sample_size, MAX_SAMPLE, client.count(filters))
    for page in range(1, -(-sample // PAGE_SIZE) + 1):
        yield from client.sample_page(filters, sample, seed, page)


class OpenAlexSource:
    """Loads one field's downloaded abstract samples, one document per abstract."""

    def __init__(self, raw_dir: Path, field: str) -> None:
        self.raw_dir = raw_dir
        self.field = field

    def _records(self, decade: int) -> Iterator[dict[str, Any]]:
        path = cell_path(self.raw_dir, self.field, decade)
        if not path.exists():
            return
        with gzip.open(path, "rt", encoding="utf-8") as cell:
            for line in cell:
                yield json.loads(line)

    def cell(self, decade: int) -> CleanedCell:
        return clean_cell(self._records(decade))
