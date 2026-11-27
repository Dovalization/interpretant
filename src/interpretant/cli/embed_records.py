"""The ``embed-records`` command: the frozen encoder's vector of every record behind a usage."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.embed_records import (
    RecordEncoderSettings,
    build_record_encoder,
    embed_records,
)
from interpretant.pipelines.progress import Progress


@click.command("embed-records")
@click.option(
    "--catalog",
    "catalog_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CATALOG_FILE,
    show_default=True,
    help="The catalog file naming the inputs and outputs.",
)
def embed_records_command(catalog_file: Path) -> None:
    """Encode the kept text of every record behind a drawn usage, for the junk gate's classifier.

    The encoder is params.yaml's junk_gate.encoder, frozen: the first 512 tokens, the first
    token's state, normalised to length 1 (docs/junk_gate.md). Each record is encoded once, in the
    part of the usages table it is drawn in, on the GPU when there is one. Progress is shown live:
    a bar in a terminal, a line every half minute in a log file.
    """
    run_on_command_line(
        "embed-records",
        catalog_file,
        lambda catalog, progress, console: run_embed_records(
            catalog, progress=progress, console=console
        ),
    )


def run_embed_records(
    catalog: CatalogConfig, *, progress: Progress, console: Console
) -> dict[str, int]:
    """Write the record vectors; the records, parts and works drawn in two parts for the run log."""
    settings = RecordEncoderSettings()
    result = embed_records(catalog, build_record_encoder(settings), progress=progress)
    console.print(
        f"{result.record_count:,} records encoded by {settings.model} into "
        f"{len(result.parts)} parts; {result.repeated_count} works drawn in two parts or more"
    )
    return {
        "records": result.record_count,
        "parts": len(result.parts),
        "works_in_several_parts": result.repeated_count,
    }
