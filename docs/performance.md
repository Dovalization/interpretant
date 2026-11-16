# Pipeline performance

*Started 2026-11-16. Where the corpus pipeline's time goes, what was changed to cut it, and what
each change did. Figures come from full runs of the pipeline on the homelab machine (Linux
x86_64, 12 CPUs, ten worker processes) unless they say otherwise. Speed work before this file is
recorded in the decision agenda (run 7c and the first round of changes); from here it is
recorded here.*

## How speed work is done

- **Profile before changing anything.** py-spy records the whole pipeline (`py-spy record
  --subprocesses --nonblocking --rate 20 --format speedscope`, started as the parent of `dvc
  repro`), and a CPU snapshot of every process is taken every 30 seconds. The profile names the
  process that limits each stage and the functions it spends its time in. Twice the bottleneck
  guessed beforehand was wrong: srsly's registry lookups, then spaCy tokenizing in the main
  process.
- **Check every change against the outputs.** A full run follows each round, and every stage's
  output hash is compared with the previous run's (`dvc.lock`). A change meant to keep the
  outputs must give the same hashes; a change that alters them (a new hash for near-duplicates)
  has its differences listed and reviewed.
- **Use the library's intended path.** spaCy's, Polars' and datasketch's documentation are
  checked for the supported way to do a step before any code of the project's own.

## Full runs

Seconds per stage; the total is the whole `dvc repro`.

| Run | Date | Total | Word repairs | spaCy pass | Sentence languages | Cleaning | Corpus build | Report |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Run 7c | 2026-11-15 | 42.4 min | 155 | 1,482 | 347 | 272 | 219 | – |
| First round's check | 2026-11-16 | 28.9 min | 145 | 942 | 246 | 214 | 157 | 27 |
| Staging-table check (build and report only) | 2026-11-16 | – | – | – | – | – | 128 | 1 |
| Second round's check | 2026-11-16 | *running* | | | | | | |

## Changes

### First round (2026-11-16): outputs identical, 42.4 to 28.9 minutes

| Change | What the profile showed | Commit |
|---|---|---|
| srsly's msgpack extensions looked up once per process wherever the cache is read | the lookups took 28% of the language stage and 10% of the cleaning workers | `9b0b50ed` |
| One MinHash's permutations reused for every signature | making them again took 29% of the corpus build | `c8caf435` |
| Texts repaired once by the word-repair stage and sent to spaCy as plain strings | the pass's main process tokenized every text (22%) and repaired it (21.5%) | `06dc5ea1` |
| The works' metadata checked from their ids with Polars | 3 s instead of about 30 | `7f420b83` |

Recorded in `75797bb3`. Afterwards the pass's main process was 69% busy and its workers 84%; in
the workers the attribute ruler took 30%.

### Second round (2026-11-16)

| Change | What was measured | Commit |
|---|---|---|
| The review flags and the report's sources read from OpenAlex's staging table | pydantic checked every metadata record for fields the table holds: 28 s in the build, 24 in the report, now 1 s and under 1; outputs identical (`f1b73dc9`) | `890fce10` |
| The 13 attribute rules that test the dependency label dropped | only the parser sets it, and the pass leaves the parser out, so they never match; 13% of the ruler's time | `e716cc99` |
| Near-duplicate shingles hashed with MurmurHash3 (`mmh3`) | datasketch's default SHA-1 took 24% of the build; 37 ns per shingle against 295; the signatures change | `6748a688` |
| The glued- and split-word counts as lazy Polars queries | `Series` methods ran on one core; as a query the same steps ran four to eight times faster with the same counts | `8aaee46d` |
| The spaCy pass gives each worker whole parts to annotate and cache | spaCy's `n_process` sent every document back to the main process, which packed it again into the cache: 69% of that process | `610e178a` |
| The attribute ruler's single-token rules found by dictionary lookup | spaCy's matcher tries every rule at every token, about 80 ns a rule; 12.8 µs per token down to 2.6, the same tags, parts of speech, morphology and lemmas on 3,000 abstracts | `deaa283e` |
| Sentence languages detected by part in worker processes | reading the cache and walking the sentences ran in one process, half of the stage | `b98213a6` |
| Near-duplicate signatures made by part in worker processes | shingling and hashing ran in the main process; only the comparison needs an order | `128a89f6` |

The check run's results go here when it finishes.

## Measured and not adopted

- **Splitting the attribute ruler's `IN` tests into exact values** made it slower (14.3 µs per
  token against 12.9): the matcher's cost is per rule, not per Python call.
- **Polars' streaming engine for the string steps written eagerly** gained nothing: an eager
  `select` already runs in parallel. The gain came from replacing `Series` methods.
- **Skipping works before the pass** would save at most 9% of it, the share of works dropped at
  all, and most drops need the pass's sentences.

## Open candidates

- **The cleaning patterns.** The first-token regular expressions (URLs, DOIs, LaTeX, spam) take
  42% of the cleaning ruler's time: 6.0 µs per token, 3.5 without them; the `IN` tests and the
  quantifiers cost 1.2 to 1.7 µs more. A faster form must keep every match the same.
- **tok2vec**, 36 to 39% of the pass's worker time, is the model itself; nothing found to change.
- **The word-repair stage** sends every text to its workers and back three times, once per step.
- **Reusing unchanged documents** between runs: the pass annotates every text on every run.
- **`DocBin.add`** walks every token in Python to collect strings, now in the pass's workers.
