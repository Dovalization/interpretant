# Pipeline performance

*Where the pipelines' time goes, what was changed to cut it, and what each change did. Started
2026-11-16 from the speed records then spread over the decision agenda and `docs/methods.md`.
Figures come from the homelab machine (Linux x86_64, 12 CPUs, an NVIDIA RTX 4070 SUPER; ten
worker processes) unless they say otherwise.*

## How speed work is done

- **Profile before changing anything.** py-spy records the whole pipeline (`py-spy record
  --subprocesses --nonblocking --rate 20 --format speedscope`, started as the parent of `dvc
  repro`, since the machine lets it trace only processes it starts), and a CPU snapshot of every
  process is taken every 30 seconds. The profile names the process that limits each stage and
  the functions it spends its time in. Twice the bottleneck guessed beforehand was wrong:
  srsly's registry lookups, then spaCy tokenizing in the main process.
- **Check every change against the outputs.** A full run follows each round, and every stage's
  output hash is compared with the previous run's (`dvc.lock`). A change meant to keep the
  outputs must give the same hashes; a change that alters them (a new hash for near-duplicates)
  has its differences listed and reviewed.
- **Use the library's intended path.** spaCy's, Polars' and datasketch's documentation are
  checked for the supported way to do a step before any code of the project's own.
- **Measure the gain after the change too.** The store's batching (below) was measured only
  after it was built, and gained 5%.

## How each corpus stage uses processes

Every worker pool starts its processes with `spawn`: Polars runs its own threads, and its
documentation warns that `fork` can then deadlock a worker. The run log records each run's peak
memory (the main process and the largest worker, from `getrusage`).

- **Word repairs:** repairing, splitting glued words and joining split words run by part in
  worker processes; the corpus-wide counts between them are lazy Polars queries in the main
  process.
- **spaCy pass:** each worker takes whole parts, largest first (a part holds at most 10,000
  texts, 1.4% of the corpus), annotates them and writes their cache files; no document crosses
  processes.
- **Sentence languages:** each worker takes whole parts with its own copy of lingua's models
  (about 690 MB) and runs lingua on one core (`RAYON_NUM_THREADS=1`).
- **Cleaning:** each worker takes whole parts.
- **Corpus build:** the corpus-level rules read the whole table in the main process; the
  near-duplicate signatures are made by part in worker processes, and only the comparison runs
  in order.

## Full runs

From run 7c, seconds per stage; the total is the whole `dvc repro`.

| Run | Date | Total | Word repairs | spaCy pass | Sentence languages | Cleaning | Corpus build | Report |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Run 7c | 2026-11-15 | 42.4 min | 155 | 1,482 | 347 | 272 | 219 | – |
| First round's check | 2026-11-16 | 28.9 min | 145 | 942 | 246 | 214 | 157 | 27 |
| Staging-table check (build and report only) | 2026-11-16 | – | – | – | – | – | 128 | 1 |
| Second round's check | 2026-11-16 | *running* | | | | | | |

Earlier runs, as recorded at the time:

- **Run 1**, the first full build (705,283 abstracts): the spaCy pass about 47 minutes across
  restarts, `build-corpus` about 4.5 minutes, of which near-duplicates 2 min 54 s.
- **Run 3:** `detect-languages` 5 min 30 s, against about 1 minute hoped for, because only
  lingua's own calls ran in parallel; loading each part's documents and walking their sentences
  ran on one core. Fixed in the second round.
- **Run 7b:** about 40 minutes: the spaCy pass 23.6, sentence languages 5.9, cleaning 4.5, the
  corpus build 3.6 (near-duplicates 2.9), the glued-word count 2.0, and the same 31-second
  metadata check in four stages. It changed 199 of 705,283 texts.
- **Run 7c:** joining the split words took 43 s on one process over all texts (in worker
  processes since the first round).

## Changes

### Before the rebuild's speed rounds (2026-11-13)

- **Named entity recognition left out of the pass** (`named_entity_recognition: false`; the
  decision is in the decision agenda, 4.9). It took about 44% of a worker's time: a
  10,000-abstract part took 30 s instead of 44, the 705,283 abstracts about 27 minutes instead
  of 40. For comparison, the last full pass of the earlier corpus (lemma votes with the tagger
  and lemmatizer) read 1,274,029 abstracts, 206 million tokens, in 30 minutes on 10 processes.
- **Stage 2 split three ways** (the spaCy pass, sentence languages, cleaning; the decision is in
  the decision agenda). lingua was 10% of the pass (about 1.2 minutes over 10 processes as its
  own stage), and the patterns and keeping rules ran at 725 cached documents a second in one
  process (about 1.6 minutes over 10). A pattern or threshold change then reran about 6–7
  minutes of stages instead of 30, a language setting about 8.

### Profile of run 7c (2026-11-15)

py-spy on every process, 236,841 samples. The spaCy pass's main process was its limit: 1,507
CPU seconds in a pass of 1,482 (100% of a core, matching the CPU snapshots), while the ten
workers used 7,663 together, about half of each core. In the main process, tokenizing took 22%:
with `as_tuples=True`, spaCy 3.8's `Language.pipe` makes every `Doc` in the calling process and
`_multiprocessing_pipe` sends it packed with `Doc.to_bytes` (spaCy's source), so the workers
never tokenized. Repair took 21.5%, unpacking the workers' documents (`Doc.from_bytes`) about
12%, the `DocBin` cache about 20%, the glued-word split and the join 4%. In the workers,
tok2vec took 39% and the attribute ruler 30%. The srsly registry lookups that the pass cached
(2026-11-13) still cost 28% of the language stage and 10% of the cleaning workers; the language
stage spent 54% of its time loading cached documents, in one process; near-duplicates spent 29%
of the corpus build making each of 645,542 MinHashes' permutations again (datasketch takes
`permutations` from another MinHash "for faster initialization") and 17% in its default SHA-1
hash; the repeated metadata check was pydantic's JSON validation and gzip.

### First round (2026-11-16): outputs identical, 42.4 to 28.9 minutes

| Change | What the profile showed | Commit |
|---|---|---|
| srsly's msgpack extensions looked up once per process wherever the cache is read | the lookups took 28% of the language stage and 10% of the cleaning workers | `9b0b50ed` |
| One MinHash's permutations reused for every signature | making them again took 29% of the corpus build | `c8caf435` |
| Texts repaired once by the word-repair stage (`data/interim/repaired_texts`, kept out of the DVC store) and sent to spaCy as plain strings | the pass's main process tokenized every text (22%) and repaired it (21.5%) | `06dc5ea1` |
| The works' metadata checked from their ids with Polars | 3 s instead of about 30 | `7f420b83` |

Every stage's outputs had the same hashes as run 7c's (`75797bb3`). Afterwards the pass's main
process was 69% busy and its workers 84%; in the workers the attribute ruler took 30%.

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

## The vector store's encoding

Every usage the rounds and scans need is encoded once with XL-LEXEME (`docs/methods.md`). The
text work runs on the CPU first (the manifest, in parallel by field). The same sentences,
encoded different ways:

| Setup | Sentences per second | The whole store |
|---|---|---|
| Laptop GPU (Apple M3, full precision) | about 20 | about 25 hours |
| RTX 4070 SUPER, full precision | about 490 | about 60 minutes |
| RTX 4070 SUPER, float16 autocast on its tensor cores | about 1,600 | about 20 minutes |

The gains came from the hardware and the arithmetic. Of the pipeline changes, batching
sentences of similar length and building the next batch's token ids in worker processes removed
idle time between batches, but added about 5%, because the GPU's arithmetic was the bottleneck,
not the tokeniser; that was measured only after the change was made. Batching by a token budget
and writing shards on a background thread followed for later stores.

## Planned: reusing unchanged documents

Planned 2026-11-15 (the author), measured first. A full run reannotates every text, while run
7b changed 199 of 705,283. Reading cached documents runs at 7,600–13,000 a second and writing
them at 2,500–4,400 a second on one process, so copying unchanged documents costs under a minute
over 10 processes. The plan: each cached document keyed by the hash of its repaired text and a
fingerprint of the pipeline (spaCy and model versions, the annotation code, the stage's
settings), and only changed texts annotated, as the vector store reuses vectors by text hash;
the same for sentence languages. A full pass without reuse must give identical documents. Known
non-English works need no rule of their own: their documents are reused while their text is
unchanged, and no language decision is frozen. The plan's other steps, one metadata check and
the repaired texts written once, were done in the first round. The reuse waits until more
passes are expected.

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
- **Reusing unchanged documents** (above).
- **`DocBin.add`** walks every token in Python to collect strings, now in the pass's workers.
