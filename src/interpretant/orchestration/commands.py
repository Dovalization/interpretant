"""The command line for running jobs in Prefect: ``interpretant submit`` and ``interpretant runs``.

Prefect is imported only when one of these commands runs, so the rest of the command line works
without the ``orchestration`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import click

from interpretant.cli.annotate_corpus import annotate_corpus_command
from interpretant.cli.build_corpus import build_corpus_command
from interpretant.cli.build_linked_tables import build_linked_tables_command
from interpretant.cli.build_sentence_view import build_sentence_view_command
from interpretant.cli.build_static_view import build_static_view_command
from interpretant.cli.build_usages import build_usages_command
from interpretant.cli.clean_corpus import clean_corpus_command
from interpretant.cli.count_word_repairs import count_word_repairs_command
from interpretant.cli.detect_languages import detect_languages_command
from interpretant.cli.fetch_linked_sources import fetch_linked_sources_command
from interpretant.cli.fetch_metadata import fetch_metadata_command
from interpretant.cli.report_corpus import report_corpus_command
from interpretant.orchestration.registry import JOBS
from interpretant.orchestration.steps import Role

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
    fetch_metadata_command,
    fetch_linked_sources_command,
)


@click.group()
@click.option(
    "--where",
    type=click.Choice(["runner", "local"]),
    default="runner",
    show_default=True,
    help="runner: the machine with the GPU, from the committed code; local: this checkout as it is.",  # noqa: E501
)
@click.option(
    "--stay-awake", is_flag=True, help="Keep the runner awake after the run instead of sleeping."
)
@click.pass_context
def submit(context: click.Context, where: Role, *, stay_awake: bool) -> None:
    """Run a job as a Prefect run, follow it, and bring its record back."""
    context.obj = {"where": where, "stay_awake": stay_awake}


@submit.command("command", context_settings={"ignore_unknown_options": True})
@click.option("--name", help="The run's name, and its log file's (default: the command's).")
@click.option("--pull", multiple=True, help="A DVC-tracked path to pull first (repeatable).")
@click.option("--push-outputs", is_flag=True, help="dvc push after the run succeeds.")
@click.argument("command", nargs=-1, type=click.UNPROCESSED, required=True)
@click.pass_context
def submit_command(
    context: click.Context,
    name: str | None,
    pull: tuple[str, ...],
    command: tuple[str, ...],
    *,
    push_outputs: bool,
) -> None:
    """Run any command in the checkout: one-off scripts and the scorers."""
    parameters = {
        "command": list(command),
        "name": name or Path(command[0]).stem,
        "pull": list(pull),
        "push_outputs": push_outputs,
    }
    _submit_and_follow(context.obj, "command", parameters)


def build_job_submission(command: click.Command) -> click.Command:
    """The ``interpretant submit <job>`` command for a job's own command.

    It takes the job's options, ``--pull``, and for a DVC stage ``--save-outputs``; their values
    are the job's flow's parameters, by name.
    """
    assert command.name is not None
    job = JOBS[command.name]
    options = [param for param in command.params if param.name != "catalog_file"]
    options.append(
        click.Option(
            ["--pull"], multiple=True, help="A DVC-tracked path to pull first (repeatable)."
        )
    )
    if job.stage is not None:
        options.append(
            click.Option(
                ["--save-outputs"],
                is_flag=True,
                help=f"After the run, record the outputs in dvc.lock (dvc commit {job.stage}) and push them.",  # noqa: E501
            )
        )

    @click.pass_context
    def submit_job(context: click.Context, /, **values: object) -> None:
        parameters = {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in values.items()
        }
        _submit_and_follow(context.obj, job.name, parameters)

    return click.Command(
        job.name,
        params=options,
        callback=submit_job,
        help=command.help,
        short_help=command.get_short_help_str(),
    )


for _command in JOB_COMMANDS:
    submit.add_command(build_job_submission(_command))


@click.group()
def runs() -> None:
    """Follow a Prefect run, or bring back run records left on the relay."""


@runs.command("follow")
@click.argument("flow_run_id", type=click.UUID)
def follow_command(flow_run_id: UUID) -> None:
    """Print a run's states and log lines until it ends."""
    from interpretant.orchestration import submit as submission

    final = submission.follow_run(flow_run_id, echo=click.echo)
    raise SystemExit(0 if final == "Completed" else 1)


@runs.command("collect")
def collect_command() -> None:
    """Cherry-pick every run record still on the relay into this branch."""
    from interpretant.orchestration import submit as submission

    left = submission.collect_records(REPOSITORY)
    for branch in left:
        click.echo(f"couldn't apply {branch}; cherry-pick it by hand from the relay")
    raise SystemExit(1 if left else 0)


def _submit_and_follow(options: dict[str, Any], flow_name: str, parameters: dict[str, Any]) -> None:
    from interpretant.orchestration import submit as submission

    role: Role = options["where"]
    try:
        commit = (
            submission.publish_commit(REPOSITORY)
            if role == "runner"
            else submission.get_commit(REPOSITORY)
        )
        flow_run_id = submission.start_run(
            submission.build_submission(
                flow_name, role, parameters, commit=commit, stay_awake=options["stay_awake"]
            )
        )
    except submission.SubmissionError as exc:
        raise click.ClickException(str(exc)) from exc
    where = "on the runner" if role == "runner" else "locally"
    click.echo(f"started {flow_name} {where} as {flow_run_id}")
    try:
        final = submission.follow_run(flow_run_id, echo=click.echo)
    except KeyboardInterrupt:
        click.echo(f"\nstill running; follow it with: interpretant runs follow {flow_run_id}")
        raise SystemExit(130) from None
    if (
        final == "Completed"
        and role == "runner"
        and not submission.collect_record(flow_run_id, REPOSITORY)
    ):
        click.echo("its record didn't apply here; run `interpretant runs collect`")
    raise SystemExit(0 if final == "Completed" else 1)
