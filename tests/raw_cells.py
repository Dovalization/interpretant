"""Raw OpenAlex cells for tests, shaped like the downloads."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable

from interpretant.catalog.config import CatalogConfig


def write_raw_cell(catalog: CatalogConfig, field: str, decade: int, works: Iterable[int]) -> None:
    """Write a raw cell of the given work numbers, each with a short abstract."""
    path = catalog.raw_abstracts.path / field / f"{decade}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as cell:
        for work in works:
            record = {
                "id": f"https://openalex.org/W{work}",
                "year": decade,
                "type": "article",
                "authors": [],
                "abstract": "Models learn senses.",
            }
            cell.write(json.dumps(record) + "\n")
