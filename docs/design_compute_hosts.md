# Where runs happen, and judging as a run

Status: agreed with the author, 2026-11-19; the launcher replaced by Prefect runs, 2026-11-25.

## The problem

Work happens on two kinds of machine: the **local machine**, where code is edited, tested and
committed, and an optional **remote runner** with more cores and a GPU. Runs have been launched by
hand on the remote runner, and their outputs copied back by hand: review items for the judges,
keys, metrics lines. That couples the work to one particular remote machine, duplicates data, and
leaves pushes refused when a run's metrics line is left uncommitted on the remote side.

## Principles

1. **Any run can happen on either machine.** The remote runner is preferred, for its cores and GPU
   and to keep the local machine free. The local machine is the alternative the author chooses, and
   everything except GPU-heavy stages works there.
2. **Data moves through DVC, small outputs through git, nothing by hand.** The DVC remote is the
   store of record for stage outputs. A machine that runs a stage pulls its inputs and pushes its
   outputs. Keys, counts, scores and run records are small and committed; verdicts stay where they
   were judged (the author, 2026-11-25), as their reasons can quote corpus sentences and name
   held-out words.
3. **Review files are regenerated, not copied.** Item batches, adjudication lists and review pages
   are deterministic functions of a table and a seed, gitignored because they quote corpus
   sentences. Whichever machine holds the table builds them.
4. **Every run is visible wherever it runs.** Runs keep printing the progress lines of
   `interpretant.cli.progress` to a log file. Collecting them into a dashboard is outside this
   repository.
5. **Nothing personal in the repository.** Host names, addresses, wake services and dashboard URLs
   live outside it: in Prefect's variables, like `.dvc/config.local` holds the DVC store's address.

## Running a job: `interpretant submit`

Since 2026-11-25 every run goes through Prefect (`src/interpretant/orchestration`). The launcher
script and its local hosts file (`scripts/run.py`, `hosts.local.yaml`), agreed on 2026-11-19, are
retired.

- **One flow per job.** Each job (`orchestration/registry.py`) is a Prefect flow taking its
  command's options, and `prefect.yaml` deploys it twice: `<job>/runner`, on the remote runner from
  a submitted commit, and `<job>/local`, in the working checkout as it is. `command` runs any other
  command, for one-off scripts and the scorers.
- **`interpretant submit [--where runner|local] <job> [options]`** checks Prefect is reachable,
  starts the deployment run, follows its states and log lines, and exits with its outcome. Nothing
  falls back silently: a runner run waits, queued, until the runner is up, and a job runs locally
  only with `--where local`.
- **Code.** For a runner run the committed tree is pushed to a private relay repository
  (`refs/heads/submitted`), and the runner checks that commit out before the flow is imported; it
  refuses a checkout holding someone's changes. Uncommitted changes to tracked files refuse the
  submission.
- **Data.** `--pull PATH` runs `dvc pull` first; a stage's `--save-outputs` records its outputs in
  `dvc.lock` (`dvc commit`) and pushes them after the run.
- **Limits.** The runner takes one interpretant run at a time (one checkout); on it, the GPU and CPU
  slots are Prefect concurrency limits, and git and DVC take a lock per machine.
- **Records.** The run's output goes to `logs/<UTC time>-<name>.log`, its progress to a Prefect
  artifact. On success its record is appended to `metrics/runs.jsonl` and committed under the git
  lock, with the files git keeps that it wrote (a stage's tracked outputs, an audit's key and
  counts, a scorer's results via `submit command --output`); on the runner the commit goes to the
  relay as `runs/<flow run id>`, and `submit` cherry-picks it into the local branch
  (`interpretant runs collect` brings back any left behind). A failed run records nothing.
- **Machines.** Host names, folders, pools and worker commands are Prefect variables set outside
  this repository, so nothing here names a host (principle 5).

## Judging as a run

Agent judging moves from interactive subagents to a script that calls the Claude API, so it runs
on whichever machine holds the review files, shows up like any run, and needs no copying.

- **`interpretant judge-api ITEMS OUT`** (first `scripts/review/judge.py`) sends every item as its own request
  through the Message Batches API (asynchronous, half the price, up to 100,000 requests per batch).
  One item per request keeps judgements independent and lets a failed item be retried alone.
- **Models.** Judges A and B: `claude-sonnet-5-5` ($2 / $10 per million input and output tokens
  before the batch discount); the adjudicator: `claude-opus-5-5` ($4 / $20), only on the items the
  judges disagree on. Effort is set explicitly (medium for judges, high for the adjudicator) and
  recorded.
- **Prompt.** The codebook is the system prompt, byte-identical across requests and marked for
  caching, so repeated requests read it from the cache when it is long enough to cache; the item is
  the user message. Answers are constrained by structured output (`output_config.format`, a JSON
  schema: verdict, reason, note), so no free-text parsing.
- **Records.** Verdicts are written as the existing `verdicts/judge_*.json` files, kept on the
  machine that judged them (principle 2). A manifest records the model ids, effort, the codebook's sha256, batch ids and token usage. Raw API
  responses are not kept. A refusal or an errored item is listed in the manifest and resubmitted
  once; what still fails goes to the adjudicator.
- **Cost of one matching audit** (about 4,500 items, two judges, an adjudicator on about a tenth):
  roughly $10–20 at batch prices, mostly output tokens.
- **A spending cap** (the author's requirement). The Batches API reports usage only when a batch
  ends, so requests go out in chunks (500 by default). A chunk is submitted only if the dollars
  spent so far plus the chunk's worst case (every input token written to the cache, every request
  using its full output allowance, at batch prices) stay within `--max-dollars`, so a run never
  spends more than its cap; what the cap leaves is listed, and a later run resumes from it. Prices
  are kept in one table in the script, with the date they were read.
- **A usage report** (the author's requirement). Every run writes, and prints at the end, the
  input, cache-write, cache-read and output tokens per model and per judge, and their cost at batch
  prices, into its manifest; the totals also go into the run record (`metrics/runs.jsonl`).
- **Credentials.** `ANTHROPIC_API_KEY` from the checkout's `.env` on each machine, never the
  environment (it would move Claude Code itself to API billing) and never the repository.
  `judge-claude-code` judges on the Claude Code subscription and never sees the key.
- **The author's calibration** stays a page the author answers; its build runs where the items are,
  and only the finished page is published.

## Local-only mode

With no remote runner, every job runs with `--where local` and the local process count. The
transformer path is slow there. Data comes from the DVC remote.

## Decisions (the author, 2026-11-19)

- The 27 mentions of personal host names in 8 files were replaced by "the remote runner" (and its
  specification where a measurement needs it). No commit message names one.
- The launcher is a script, `scripts/run.py` (retired 2026-11-25 for Prefect runs).
- The judging script has a spending cap and reports its usage: tokens and cost.

## Decisions (the author, 2026-11-24 and 2026-11-25)

- Runs go through Prefect, one flow per job, with code reaching the runner through a private relay
  repository rather than GitHub.
- The judges run on the runner too, where the items are, and their verdicts stay there; the scorer
  runs there as a command job and brings back only its results.
- The record commit is made with `--no-verify`: it holds the run log and data files, which the
  hooks don't check.
