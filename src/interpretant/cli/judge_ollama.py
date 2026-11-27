"""The ``judge-ollama`` command: review items judged by a local model on an Ollama server."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.judge_ollama import (
    JUDGE_KINDS,
    JudgeKind,
    OllamaJudging,
    build_http_transport,
    build_merged_manifest,
    generate_verdicts,
)
from interpretant.pipelines.progress import Progress
from interpretant.review.judging import Answers, split_parts


@click.command("judge-ollama")
@click.argument("items", type=click.Path(exists=True, path_type=Path))
@click.argument("verdicts", type=click.Path(file_okay=False, path_type=Path))
@click.option("--name", required=True, help="The verdict files' prefix, e.g. qwen_runner.")
@click.option(
    "--codebook",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The codebook: a chat model's whole system prompt, a decision model's criteria.",
)
@click.option(
    "--answers",
    type=click.Choice(["records"]),
    required=True,
    help="The schema: records only, the junk gate's answers.",
)
@click.option("--model", required=True, help="The model, as the Ollama server names it.")
@click.option(
    "--kind",
    type=click.Choice(JUDGE_KINDS),
    required=True,
    help="chat: answers in the schema; decision: probabilities, mapped to the schema.",
)
@click.option("--host", required=True, help="The Ollama server's URL; no file records it.")
@click.option("--host-label", required=True, help="The server's name in the manifest, e.g. runner.")
@click.option(
    "--parallel", type=click.IntRange(min=1), default=1, show_default=True, help="Requests at once."
)
@click.option(
    "--keep-alive",
    default="2m",
    show_default=True,
    help="How long the server keeps the model loaded after the last request.",
)
def judge_ollama_command(
    items: Path,
    verdicts: Path,
    name: str,
    codebook: Path,
    answers: Answers,
    model: str,
    kind: JudgeKind,
    host: str,
    host_label: str,
    parallel: int,
    keep_alive: str,
) -> None:
    """Judge review items with a local model on an Ollama server (the junk gate's local trial).

    ITEMS is a folder of batch_*.jsonl files, or one .jsonl file cut into parts of 50; each part's
    answers go to VERDICTS/NAME_part_NN.json once all of its items have answered, and parts already
    judged are skipped. One item goes in each request. Each run's speed and the model's share of
    the graphics card accumulate in NAME.manifest.json under VERDICTS, which names the server by
    its label only. Fails if any part failed; rerun to retry those.
    """
    judging = OllamaJudging(
        name=name,
        codebook=codebook,
        answers=answers,
        model=model,
        kind=kind,
        host_label=host_label,
        parallel=parallel,
        keep_alive=keep_alive,
    )
    counts = run_on_command_line(
        f"judge-ollama {model} on {host_label}",
        CATALOG_FILE,
        lambda catalog, progress, console: run_judge_ollama(
            catalog, items, verdicts, judging, host, progress=progress, console=console
        ),
    )
    if counts["failed_parts"]:
        raise click.ClickException(f"{counts['failed_parts']} parts failed; rerun to retry them")


def run_judge_ollama(
    catalog: CatalogConfig,
    items: Path,
    verdicts: Path,
    judging: OllamaJudging,
    host: str,
    *,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Judge the parts not yet judged and add this run to the manifest; the items and failures.

    The catalog is unused: judging reads and writes only the paths it is given. ``host`` reaches
    only the transport.
    """
    del catalog
    run = generate_verdicts(
        split_parts(items), judging, verdicts, build_http_transport(host), progress=progress
    )
    manifest_path = verdicts / f"{judging.name}.manifest.json"
    earlier = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    codebook_sha256 = hashlib.sha256(judging.codebook.read_bytes()).hexdigest()
    manifest = build_merged_manifest(earlier, run, judging, codebook_sha256)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    record = manifest["runs"][-1]
    console.print(
        f"{run.items_judged} items judged and {run.items_failed} failed on {judging.host_label}, "
        f"{record['items_per_minute']} a minute; {len(run.failed)} parts failed"
    )
    return {
        "judged": run.judged,
        "items_judged": run.items_judged,
        "items_failed": run.items_failed,
        "failed_parts": len(run.failed),
    }
