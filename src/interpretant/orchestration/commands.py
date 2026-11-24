"""The command line for running jobs in Prefect: ``interpretant submit`` and ``interpretant runs``.

Prefect is imported only when one of these commands runs, so the rest of the command line works
without the ``orchestration`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import click

from interpretant.orchestration.steps import Role

REPOSITORY = Path(__file__).resolve().parents[3]


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
    click.echo(f"started {flow_name} on the {role} as {flow_run_id}")
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
