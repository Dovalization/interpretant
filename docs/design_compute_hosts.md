# Where runs happen, and judging as a run

Status: agreed with the author, 2026-11-19. Building it is the next work.

## The problem

Work happens on two kinds of machine: the **local machine**, where code is edited, tested and
committed, and an optional **remote runner** with more cores and a GPU. Runs have been launched by
hand on the remote runner, and their outputs copied back by hand: review items for the judges,
keys, metrics lines. That couples the work to one particular remote machine, duplicates data, and
leaves pushes refused when a run's metrics line is left uncommitted on the remote side.

## Principles

1. **Any run can happen on either machine.** The remote runner is preferred when it is available,
   for its cores and GPU and to keep the local machine free. The local machine is the fallback, and
   everything except GPU-heavy stages works there.
2. **Data moves through DVC, small outputs through git, nothing by hand.** The DVC remote is the
   store of record for stage outputs. A machine that runs a stage pulls its inputs and pushes its
   outputs. Keys, counts, verdicts and run records are small and committed.
3. **Review files are regenerated, not copied.** Item batches, adjudication lists and review pages
   are deterministic functions of a table and a seed, gitignored because they quote corpus
   sentences. Whichever machine holds the table builds them.
4. **Every run is visible wherever it runs.** Runs keep printing the progress lines of
   `interpretant.cli.progress` to a log file. Collecting them into a dashboard is outside this
   repository.
5. **Nothing personal in the repository.** Host names, addresses, wake services and dashboard URLs
   live in a gitignored local file, like `.dvc/config.local` holds the DVC store's address.

## Local configuration

`hosts.local.yaml` at the repository root, gitignored, with `hosts.example.yaml` committed:

```yaml
remote:                # leave out to run everything locally
  ssh: gpu-box         # an ssh host alias from ~/.ssh/config
  path: ~/Developer/interpretant
  git_remote: gpu-box  # the git remote that pushes to that checkout
  wake: null           # optional command that powers the runner on (e.g. a wake-on-LAN request)
  wake_timeout: 180    # seconds to wait for ssh after waking
  processes: 12        # CPU workers there
  gpu: true
local:
  processes: 6
```

## The launcher

A script (the author's choice), `python scripts/run.py [--where auto|local|remote] -- <command>`:

1. **Choose the machine.** `auto` uses the remote runner if ssh answers. If not, it runs the
   `wake` command (if configured) and waits up to `wake_timeout`; if the runner still does not
   answer, it falls back to the local machine. A command marked as needing a GPU (the transformer
   path) refuses to fall back unless `--allow-cpu` is given, since it would take hours.
2. **Sync code.** It requires a clean, committed tree and pushes it to the chosen checkout.
3. **Ensure data.** `dvc pull` for the command's inputs where they are missing.
4. **Run** from the repository directory, stdout and stderr to `logs/<time>-<command>.log`
   (gitignored), so a collector watching the repository finds it.
5. **Record.** On success it commits `metrics/runs.jsonl` on that machine with a fixed message,
   `dvc push`es stage outputs, and pulls the commit back to the local machine. The run record names
   the machine by its specification (system, CPUs, GPU), as it already does, never by host name.

A failed run leaves its log and records nothing.

## Judging as a run

Agent judging moves from interactive subagents to a script that calls the Claude API, so it runs
on whichever machine holds the review files, shows up like any run, and needs no copying.

- **`scripts/review/judge.py ITEMS_DIR CODEBOOK JUDGE`** sends every item as its own request
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
- **Records.** Verdicts are written as the existing `verdicts/judge_*.json` files and committed. A
  manifest records the model ids, effort, the codebook's sha256, batch ids and token usage. Raw API
  responses are not kept. A refusal or an errored item is listed in the manifest and resubmitted
  once; what still fails goes to the adjudicator.
- **Cost of one matching audit** (about 4,500 items, two judges, an adjudicator on about a tenth):
  roughly $10–20 at batch prices, mostly output tokens.
- **A spending cap** (the author's requirement). Before submitting, the script counts every
  request's input tokens (the token-counting endpoint, which is free) and prices them with a
  ceiling on output per item; it refuses to submit if the estimate passes `--max-dollars`. While a
  batch runs it is checked again with the usage reported so far, and cancelled if the actual cost
  passes the cap. Prices are kept in one table in the script, with the date they were read.
- **A usage report** (the author's requirement). Every run writes, and prints at the end, the
  input, cache-write, cache-read and output tokens per model and per judge, and their cost at batch
  prices, into its manifest; the totals also go into the run record (`metrics/runs.jsonl`).
- **Credentials.** `ANTHROPIC_API_KEY` from the environment (or an `ant auth login` profile) on each
  machine, never in the repository.
- **The author's calibration** stays a page the author answers; its build runs where the items are,
  and only the finished page is published.

## Mac-only (local-only) mode

With no remote runner, every stage and script runs locally with the local process count. The
transformer path runs only with `--allow-cpu`, or waits. Data comes from the DVC remote.

## Order of work

1. `hosts.example.yaml`, the local file's loader and the launcher, with tests that fake ssh.
2. `scripts/review/judge.py` and its manifest, tried on the pilot's items before any audit of
   record.
3. Then step 6 resumes: the transformer path, the table of record, the audit drawn and judged
   through the launcher and the judging script.

## Decisions (the author, 2026-11-19)

- The 27 mentions of personal host names in 8 files were replaced by "the remote runner" (and its
  specification where a measurement needs it). No commit message names one.
- The launcher is a script, `scripts/run.py`.
- The judging script has a spending cap and reports its usage: tokens and cost.
