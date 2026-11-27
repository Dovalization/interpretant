"""Write run 3's records for the junk gate: its judged records with the text they had in run 3.

Reads a past run's abstracts table (a folder of `<field>_<decade>.parquet` parts with `work_id`,
`text` and `removals`) and run 3's verdicts (`data/review/run3/verdicts.jsonl`), and writes the
judged records' rows to the catalog's `run3_records` file, refusing if a judged record is missing.
The file quotes the corpus: it is run once by hand, from the repository root, and the file is then
registered with `dvc add`: `python scripts/review/junk_gate_run3.py ABSTRACTS_DIR`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Collection
from pathlib import Path

import polars as pl

from interpretant.catalog.config import read_catalog
from interpretant.catalog.junk_gate import (
    RUN3_RECORDS_COLUMNS,
    load_run3_verdicts,
    save_run3_records,
)


def build_run3_records(abstracts: Path, work_ids: Collection[str]) -> pl.DataFrame:
    """The judged records' work ids, texts and removed ranges, from a past abstracts table.

    Raises:
        SystemExit: if a judged record is missing, or sits in two parts.
    """
    wanted = pl.DataFrame({"work_id": sorted(work_ids)}, schema={"work_id": pl.String()})
    parts = sorted(abstracts.glob("*_*.parquet"))
    records = pl.concat(
        [pl.DataFrame(schema=RUN3_RECORDS_COLUMNS)]
        + [
            pl.read_parquet(path, columns=list(RUN3_RECORDS_COLUMNS)).join(
                wanted, on="work_id", how="semi"
            )
            for path in parts
        ]
    )
    if records["work_id"].n_unique() != records.height:
        raise SystemExit("a judged record sits in two parts of the abstracts table")
    missing = sorted(set(work_ids) - set(records["work_id"]))
    if missing:
        raise SystemExit(
            f"{len(missing)} judged records are missing from {abstracts}, such as {missing[:5]}"
        )
    return records.sort("work_id")


def main() -> None:
    """Write run 3's judged records to the catalog's run3_records file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("abstracts", type=Path, help="the past run's abstracts table (a folder)")
    options = parser.parse_args()
    catalog = read_catalog()
    work_ids = {verdict.work_id for verdict in load_run3_verdicts(catalog)}
    path = save_run3_records(catalog, build_run3_records(options.abstracts, work_ids))
    sys.stdout.write(f"{len(work_ids)} records written to {path}; register it with dvc add\n")


if __name__ == "__main__":
    main()
