"""Gzipped JSON Lines files that grow by appending, as the fetches write them.

Each append is one gzip member, compressed in memory and written in one call, so a fetch stopped
between batches leaves a whole file (gzip reads the members as one stream).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import polars as pl


def append_json_lines(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Append records to a gzipped JSON Lines file, one gzip member per call."""
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(gzip.compress(lines.encode("utf-8")))


def read_json_lines(path: Path) -> Iterator[str]:
    """The lines of a gzipped JSON Lines file, read as one stream across its members.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        yield from lines


def read_cell_files(folder: Path, schema: pl.Schema) -> pl.DataFrame:
    """Every ``<field>/<decade>.jsonl.gz`` file under ``folder`` as one table, with these fields.

    Polars reads the gzipped files directly; fields a record lacks are null, fields the schema
    does not name are left out. No file gives an empty table.
    """
    files = sorted(folder.glob("*/*.jsonl.gz"))
    if not files:
        return pl.DataFrame(schema=schema)
    return pl.read_ndjson(files, schema=schema)
