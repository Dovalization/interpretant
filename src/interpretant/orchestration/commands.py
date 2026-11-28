"""The command line for running jobs in Prefect: ``interpretant submit`` and ``interpretant runs``.

Prefect is imported only when one of these commands runs, so the rest of the command line works
without the ``orchestration`` extra.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from uuid import UUID

import click

from interpretant.cli.annotate_corpus import annotate_corpus_command
from interpretant.cli.build_corpus import build_corpus_command
from interpretant.cli.build_junk_gate import build_junk_gate_command
from interpretant.cli.build_linked_tables import build_linked_tables_command
from interpretant.cli.build_sentence_view import build_sentence_view_command
from interpretant.cli.build_static_view import build_static_view_command
from interpretant.cli.build_usages import build_usages_command
from interpretant.cli.clean_corpus import clean_corpus_command
from interpretant.cli.count_word_repairs import count_word_repairs_command
from interpretant.cli.detect_languages import detect_languages_command
from interpretant.cli.draw_junk_gate import draw_junk_gate_command
from interpretant.cli.draw_usages_audit import draw_usages_audit_command
from interpretant.cli.embed_records import embed_records_command
from interpretant.cli.fetch_linked_sources import fetch_linked_sources_command
from interpretant.cli.fetch_metadata import fetch_metadata_command
from interpretant.cli.judge_api import judge_api_command
from interpretant.cli.judge_claude_code import judge_claude_code_command
from interpretant.cli.judge_ollama import judge_ollama_command
from interpretant.cli.report_corpus import report_corpus_command
from interpretant.cli.score_records import score_records_command
from interpretant.orchestration.registry import DVC_LOCK, JOBS

REPOSITORY = Path(__file__).resolve().parents[3]
# The jobs' own commands, whose options their submissions take (but --catalog)
JOB_COMMANDS: tuple[click.Command, ...] = (
    build_linked_tables_command,
    count_word_repairs_command,
    annotate_corpus_command,
    detect_languages_command,
    clean_corpus_command,
    build_corpus_command,
    report_corpus_command,
    build_static_view_command,
    build_sentence_view_command,
    build_usages_command,
    embed_records_command,
    score_records_command,
    build_junk_gate_command,
    fetch_metadata_command,
    fetch_linked_sources_command,
    draw_usages_audit_command,
    draw_junk_gate_command,
    judge_api_command,
    judge_claude_code_command,
    judge_ollama_command,
)


@click.group()
@click.option(
    "--stay-awake", is_flag=True, help="Keep the runner awake after the run instead of sleeping."
)
@click.pass_context
def submit(context: click.Context, *, stay_awake: bool) -> None:
    """Run a job on the runner from this commit (pushed to GitHub), follow it, collect its record.

    To run a job on this machine instead, run its command (`interpretant <job>`).
    """
    context.obj = {"stay_awake": stay_awake}


@submit.command("command", context_settings={"ignore_unknown_options": True})
@click.option("--name", help="The run's name, and its log file's (default: the command's).")
@click.option("--pull", multiple=True, help="A DVC-tracked path to pull first (repeatable).")
@click.option(
    "--output",
    "outputs",
    multiple=True,
    type=click.Path(path_type=Path),
    help="A file or folder the command writes that git keeps, committed with the runner's record (repeatable).",  # noqa: E501
)
@click.option("--push-outputs", is_flag=True, help="dvc push after the run succeeds.")
@click.argument("command", nargs=-1, type=click.UNPROCESSED, required=True)
@click.pass_context
def submit_command(
    context: click.Context,
    name: str | None,
    pull: tuple[str, ...],
    outputs: tuple[Path, ...],
    command: tuple[str, ...],
    *,
    push_outputs: bool,
) -> None:
    """Run any command in the checkout: one-off scripts and the scorers."""
    parameters = {
        "command": list(command),
        "name": name or Path(command[0]).stem,
        "pull": list(pull),
        "outputs": [build_parameter(output) for output in outputs],
        "push_outputs": push_outputs,
    }
    _submit_and_follow(context.obj, "command", parameters)


@submit.command("repro")
@click.argument(
    "stage", type=click.Choice(sorted(job.stage for job in JOBS.values() if job.stage is not None))
)
@click.option("--pull", multiple=True, help="A DVC-tracked path to pull first (repeatable).")
@click.pass_context
def submit_repro(context: click.Context, stage: str, pull: tuple[str, ...]) -> None:
    """Run a DVC stage with `dvc repro` on the runner, recording its outputs in dvc.lock.

    DVC runs the stage's command from dvc.yaml if its inputs changed, then the outputs are pushed
    to the DVC remote, and dvc.lock and the stage's tracked outputs come back with the record. A
    reproduced stage starts from deleted outputs, so it doesn't resume a stopped run.
    """
    [job] = [job for job in JOBS.values() if job.stage == stage]
    parameters = {
        "command": ["dvc", "repro", stage],
        "name": f"repro-{job.name}",
        "pull": list(pull),
        "outputs": [DVC_LOCK, *job.tracked_outputs],
        "push_outputs": True,
    }
    _submit_and_follow(context.obj, "command", parameters)


def build_job_submission(command: click.Command) -> click.Command:
    """The ``interpretant submit <job>`` command for a job's own command.

    It takes the job's options and ``--pull``; their values are the job's flow's parameters, by
    name. A stage whose outputs are to be recorded runs through ``submit repro`` instead.
    """
    assert command.name is not None
    job = JOBS[command.name]
    options = [build_unchecked(param) for param in command.params if param.name != "catalog_file"]
    options.append(
        click.Option(
            ["--pull"], multiple=True, help="A DVC-tracked path to pull first (repeatable)."
        )
    )

    @click.pass_context
    def submit_job(context: click.Context, /, **values: object) -> None:
        parameters = {name: build_parameter(value) for name, value in values.items()}
        _submit_and_follow(context.obj, job.name, parameters)

    return click.Command(
        job.name,
        params=options,
        callback=submit_job,
        help=command.help,
        short_help=command.get_short_help_str(),
    )


def build_unchecked(param: click.Parameter) -> click.Parameter:
    """A copy of a job's option that doesn't require its path to exist here.

    The run's checkout is the one that must hold it: review items, for one, exist only on the
    runner.
    """
    if not isinstance(param.type, click.Path) or not param.type.exists:
        return param
    unchecked = copy.copy(param)
    unchecked.type = copy.copy(param.type)
    unchecked.type.exists = False
    return unchecked


def build_parameter(value: object) -> object:
    """An option's value as the flow's parameter: a list for a tuple, a path relative to the repo.

    A relative path lets the run find it in its own checkout, whatever folder it was submitted from.
    """
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(REPOSITORY))
        except ValueError as exc:
            raise click.BadParameter(f"{value} is outside the repository") from exc
    return value


for _command in JOB_COMMANDS:
    submit.add_command(build_job_submission(_command))


@click.group()
def runs() -> None:
    """Follow a Prefect run, or bring back run records left on GitHub."""


@runs.command("follow")
@click.argument("flow_run_id", type=click.UUID)
def follow_command(flow_run_id: UUID) -> None:
    """Print a run's states and log lines until it ends."""
    from interpretant.orchestration import submit as submission

    final = submission.follow_run(flow_run_id, echo=click.echo)
    raise SystemExit(0 if final == "Completed" else 1)


@runs.command("collect")
def collect_command() -> None:
    """Cherry-pick every run record still on GitHub into this branch."""
    from interpretant.orchestration import submit as submission

    left = submission.collect_records(REPOSITORY)
    for branch in left:
        click.echo(f"couldn't apply {branch}; cherry-pick it by hand from GitHub")
    raise SystemExit(1 if left else 0)


def _submit_and_follow(options: dict[str, Any], flow_name: str, parameters: dict[str, Any]) -> None:
    from interpretant.orchestration import submit as submission

    try:
        commit = submission.get_published_commit(REPOSITORY)
        flow_run_id = submission.start_run(
            submission.build_submission(
                flow_name, parameters, commit=commit, stay_awake=options["stay_awake"]
            )
        )
    except submission.SubmissionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"started {flow_name} on the runner as {flow_run_id}")
    try:
        final = submission.follow_run(flow_run_id, echo=click.echo)
    except KeyboardInterrupt:
        click.echo(f"\nstill running; follow it with: interpretant runs follow {flow_run_id}")
        raise SystemExit(130) from None
    if final == "Completed" and not submission.collect_record(flow_run_id, REPOSITORY):
        click.echo("its record didn't apply here; run `interpretant runs collect`")
    raise SystemExit(0 if final == "Completed" else 1)
