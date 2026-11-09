"""OpenAlex corpus source — seeded abstract samples per field and decade.

OpenAlex labels every work with one topic taxonomy, so philosophy, psychology, neuroscience,
computer science and the social sciences share field labels across the whole series. Each
cell of the field × decade grid is a seeded random sample of English works with abstracts,
stored as ``data/raw/openalex/<field>/<decade>.jsonl.gz`` with one work per line.

Requests need an API key (``OPENALEX_API_KEY`` in the environment or ``.env``): a list page
of 200 works costs one credit from the key's daily budget.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from interpretant.corpus.base import CorpusSource
from interpretant.corpus.preprocessing import is_long_enough, preprocess_text

API_URL = "https://api.openalex.org/works"
PAGE_SIZE = 200
MAX_SAMPLE = 10_000  # OpenAlex rejects larger samples

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
    """Keep what the grid needs from a work: id, year, author ids and abstract text."""
    authors = [
        str(authorship["author"]["id"])
        for authorship in work.get("authorships") or []
        if (authorship.get("author") or {}).get("id")
    ]
    return {
        "id": work["id"],
        "year": work["publication_year"],
        "authors": authors,
        "abstract": abstract_from_inverted_index(work.get("abstract_inverted_index")),
    }


def cell_filter(field: str, decade: int) -> str:
    return f"{FIELDS[field]},publication_year:{decade}-{decade + 9},language:en,has_abstract:true"


def cell_path(raw_dir: Path, field: str, decade: int) -> Path:
    return raw_dir / field / f"{decade}.jsonl.gz"


class OpenAlexClient:
    """Fetches seeded sample pages, waiting out short rate limits."""

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
            "select": "id,publication_year,authorships,abstract_inverted_index",
        }
        results: list[dict[str, Any]] = self._get(params)["results"]
        return results

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
) -> int:
    """Download one cell's sample and return its number of works; existing cells are kept.

    The file is written under a temporary name and renamed when complete, so an interrupted
    download leaves no partial cell behind.
    """
    path = cell_path(raw_dir, field, decade)
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as existing:
            return sum(1 for _ in existing)
    filters = cell_filter(field, decade)
    sample = min(sample_size, MAX_SAMPLE, client.count(filters))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    written = 0
    with gzip.open(partial, "wt", encoding="utf-8") as output:
        for page in range(1, -(-sample // PAGE_SIZE) + 1):
            for work in client.sample_page(filters, sample, seed, page):
                output.write(json.dumps(work_record(work), ensure_ascii=False) + "\n")
                written += 1
    partial.rename(path)
    return written


class OpenAlexSource(CorpusSource):
    """Loads one field's downloaded abstract samples, one document per abstract."""

    def __init__(self, raw_dir: Path, field: str) -> None:
        self.raw_dir = raw_dir
        self.field = field

    def _records(self, decade: int) -> Iterator[dict[str, object]]:
        path = cell_path(self.raw_dir, self.field, decade)
        if not path.exists():
            return
        with gzip.open(path, "rt", encoding="utf-8") as cell:
            for line in cell:
                yield json.loads(line)

    def _decades(self) -> list[int]:
        return sorted(
            int(path.name.split(".")[0]) for path in (self.raw_dir / self.field).glob("*.jsonl.gz")
        )

    def iter_documents(self) -> Iterator[str]:
        for decade in self._decades():
            for record in self._records(decade):
                text = preprocess_text(str(record["abstract"]))
                if text:
                    yield text

    def iter_decade_slices(
        self,
        start: int,
        end: int,
        step: int,
        min_tokens: int = 0,
        workers: int = 1,
    ) -> Iterator[tuple[int, list[str]]]:
        for decade in range(start, end, step):
            documents = []
            for record in self._records(decade):
                text = preprocess_text(str(record["abstract"]))
                if text and (min_tokens == 0 or is_long_enough(text, min_tokens)):
                    documents.append(text)
            yield decade, documents

    def document_count(self) -> int:
        return sum(sum(1 for _ in self._records(decade)) for decade in self._decades())
