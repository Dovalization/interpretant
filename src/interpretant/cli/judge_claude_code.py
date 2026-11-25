"""The ``judge-claude-code`` command: review items judged by Claude Code, on the subscription."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.judge_claude_code import (
    USAGE_FIELDS,
    CommandLineJudging,
    build_merged_manifest,
    generate_verdicts,
    generate_with_claude,
)
from interpretant.pipelines.progress import Progress
from interpretant.review.judging import EFFORTS, SCHEMAS, Answers, Effort, split_parts


@click.command("judge-claude-code")
@click.argument("items", type=click.Path(exists=True, path_type=Path))
@click.argument("verdicts", type=click.Path(file_okay=False, path_type=Path))
@click.option("--name", required=True, help="The verdict files' prefix, e.g. judge_A.")
@click.option(
    "--codebook",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The codebook, sent as the whole system prompt.",
)
@click.option("--answers", type=click.Choice(sorted(SCHEMAS)), required=True, help="The schema.")
@click.option("--model", required=True, help="The model, as Claude Code names it.")
@click.option("--effort", type=click.Choice(EFFORTS), required=True, help="The model's effort.")
@click.option(
    "--slice",
    "items_per_call",
    default=1,
    show_default=True,
    help="Items judged in one call.",
)
@click.option("--parallel", default=4, show_default=True, help="Calls running at once.")
@click.option("--claude", default="claude", show_default=True, help="Claude Code's executable.")
def judge_claude_code_command(
    items: Path,
    verdicts: Path,
    name: str,
    codebook: Path,
    answers: Answers,
    model: str,
    effort: Effort,
    items_per_call: int,
    parallel: int,
    claude: str,
) -> None:
    """Judge review items with Claude Code's command line, on the subscription.

    ITEMS is a folder of batch_*.jsonl files, or one .jsonl file cut into parts of 50; each part's
    answers go to VERDICTS/NAME_part_NN.json, and parts already judged are skipped. Each call sees
    only the codebook and its items. The usage accumulates in NAME.manifest.json under VERDICTS
    and goes to the run log. Fails if any part failed; rerun to retry those.
    """
    judging = CommandLineJudging(
        name=name,
        codebook=codebook,
        answers=answers,
        model=model,
        effort=effort,
        items_per_call=items_per_call,
        parallel=parallel,
        claude=claude,
    )
    counts = run_on_command_line(
        f"judge-claude-code {model}",
        CATALOG_FILE,
        lambda catalog, progress, console: run_judge_claude_code(
            catalog, items, verdicts, judging, progress=progress, console=console
        ),
    )
    if counts["failed_parts"]:
        raise click.ClickException(f"{counts['failed_parts']} parts failed; rerun to retry them")


def run_judge_claude_code(
    catalog: CatalogConfig,
    items: Path,
    verdicts: Path,
    judging: CommandLineJudging,
    *,
    progress: Progress,
    console: Console,
) -> dict[str, int]:
    """Judge the parts not yet judged and add this run to the manifest; the usage and failures.

    The catalog is unused: judging reads and writes only the paths it is given.
    """
    del catalog
    version = subprocess.run(
        [judging.claude, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    run = generate_verdicts(
        split_parts(items), judging, verdicts, generate_with_claude, progress=progress
    )
    manifest_path = verdicts / f"{judging.name}.manifest.json"
    earlier = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = build_merged_manifest(
        earlier,
        run,
        {
            "model": judging.model,
            "effort": judging.effort,
            "claude_version": version,
            "codebook_sha256": hashlib.sha256(judging.codebook.read_bytes()).hexdigest(),
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    console.print(
        f"{run['judged']} judged this run, {len(run['failed'])} parts failed; "
        f"models {', '.join(run['models'])}; tokens {run['tokens']}"
    )
    counts = {name: int(run["tokens"].get(name, 0)) for name in USAGE_FIELDS}
    return counts | {
        "judged": run["judged"],
        "failed_parts": len(run["failed"]),
        "notional_micro_usd": int(run["tokens"].get("notional_micro_usd", 0)),
    }
