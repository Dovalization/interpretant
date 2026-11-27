"""The ``judge-api`` command: review items judged through the Message Batches API, within a cap."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import anthropic
import click
from rich.console import Console

from interpretant.catalog.config import CATALOG_FILE, CatalogConfig
from interpretant.cli.jobs import run_on_command_line
from interpretant.pipelines.judge_api import (
    PRICES,
    USAGE_FIELDS,
    ApiJudging,
    ApiKeySettings,
    Thinking,
    generate_verdicts,
    render_report,
)
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.review.judging import EFFORTS, SCHEMAS, Answers, Effort, load_items


@click.command("judge-api")
@click.argument("items", type=click.Path(exists=True, path_type=Path))
@click.argument("out", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--codebook",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The codebook, sent as the cached system prompt.",
)
@click.option("--answers", type=click.Choice(sorted(SCHEMAS)), required=True, help="The schema.")
@click.option("--model", type=click.Choice(sorted(PRICES)), required=True, help="The model.")
@click.option("--effort", type=click.Choice(EFFORTS), required=True, help="The model's effort.")
@click.option(
    "--max-dollars",
    type=float,
    required=True,
    help="The run's cap at batch prices; a chunk that could pass it is not sent.",
)
@click.option("--chunk", default=500, show_default=True, help="Requests in one batch.")
@click.option("--max-tokens", default=4096, show_default=True, help="Most output tokens a request.")
@click.option(
    "--thinking",
    type=click.Choice(["adaptive", "off"]),
    default="adaptive",
    show_default=True,
    help="off: the model answers without reasoning.",
)
@click.option(
    "--second-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="A second judge with the same settings, writing here; the two run side by side, each "
    "within --max-dollars.",
)
def judge_api_command(
    items: Path,
    out: Path,
    codebook: Path,
    answers: Answers,
    model: str,
    effort: Effort,
    max_dollars: float,
    chunk: int,
    max_tokens: int,
    thinking: Thinking,
    second_out: Path | None,
) -> None:
    """Judge review items through the Message Batches API, within a cap.

    ITEMS is a folder of batch_*.jsonl files or one .jsonl file; OUT is a JSON file of item id to
    answer, and a stopped run resumes where it left off. Every item is its own request, with the
    codebook cached for an hour. The usage and cost go to OUT's .manifest.json and the run log.
    With --second-out, a second judge with the same settings judges the same items beside the
    first, so a pair of judges takes the time of one. The API key comes from the checkout's .env.
    """
    judging = ApiJudging(
        codebook=codebook,
        answers=answers,
        model=model,
        effort=effort,
        max_dollars=max_dollars,
        chunk=chunk,
        max_tokens=max_tokens,
        thinking=thinking,
    )
    run_on_command_line(
        build_record_name(model, second_out),
        CATALOG_FILE,
        lambda catalog, progress, console: run_judge_api(
            catalog, items, out, judging, progress=progress, console=console, second_out=second_out
        ),
    )


def build_record_name(model: str, second_out: Path | None) -> str:
    """The run log's name for a run: one judge, or a pair."""
    return f"judge-api {model}" if second_out is None else f"judge-api {model} pair"


def run_judge_api(
    catalog: CatalogConfig,
    items: Path,
    out: Path,
    judging: ApiJudging,
    *,
    progress: Progress,
    console: Console,
    second_out: Path | None = None,
) -> dict[str, int]:
    """Judge the items, write each judge's manifest and print its report; the tokens and cost.

    With ``second_out``, a second judge judges the same items in its own thread, with its own
    client, cap, verdicts and manifest. The live progress follows the first judge; the counts add
    both. The catalog is unused: judging reads and writes only the paths it is given.
    """
    del catalog
    api_key = ApiKeySettings().anthropic_api_key
    loaded = load_items(items)
    outs = [out] if second_out is None else [out, second_out]
    reporters: list[Progress] = [progress, SILENT][: len(outs)]
    with ThreadPoolExecutor(max_workers=len(outs)) as pool:
        manifests = list(
            pool.map(
                lambda path, reporter: _judge_into(path, loaded, judging, api_key, reporter),
                outs,
                reporters,
            )
        )
    counts: Counter[str] = Counter()
    for manifest in manifests:
        console.print(render_report(manifest))
        counts.update({name: int(manifest["tokens"].get(name, 0)) for name in USAGE_FIELDS})
        counts["judged"] += manifest["judged"]
        counts["cost_micro_usd"] += int(manifest["spent_dollars"] * 1e6)
    return dict(counts)


def _judge_into(
    out: Path,
    items: list[dict[str, Any]],
    judging: ApiJudging,
    api_key: str,
    progress: Progress,
) -> dict[str, Any]:
    """One judge's run: its verdicts saved to ``out``, resumed if it exists, and its manifest."""
    client = anthropic.Anthropic(api_key=api_key)
    verdicts = json.loads(out.read_text()) if out.exists() else {}
    manifest = generate_verdicts(items, judging, client, verdicts, out, progress=progress)
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
