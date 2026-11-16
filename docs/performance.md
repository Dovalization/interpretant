# Pipeline performance

*Where the pipelines' time and memory go, what was changed to cut them, and what each change
did. Started 2026-11-16 from the speed records then spread over the decision agenda and
`docs/methods.md`. Figures come from the homelab machine (Linux x86_64, 12 CPUs, which are 6
physical cores with two threads each; an NVIDIA RTX 4070 SUPER; 31 GB of memory; ten worker
processes) unless they say otherwise.*

## How speed work is done

- **Profile before changing anything.** py-spy records the whole pipeline (`uvx py-spy@0.4.3
  record --subprocesses --nonblocking --rate 20 --format speedscope`, started as the parent of
  `dvc repro`, since the machine lets it trace only processes it starts; it follows the workers
  through DVC's shell), and a CPU snapshot of every process is taken every 30 seconds. py-spy is
  not a project dependency. It costs 12–18% of one core: the profiled pass of run 7c ran at 538
  texts a second, as unprofiled run 7b had. The profile names the process that limits each stage
  and the functions it spends its time in. Twice the bottleneck guessed beforehand was wrong:
  srsly's registry lookups, then spaCy tokenizing in the main process; a third guess (moving the
  row-building into the workers, "30–40% faster") was measured as no change.
- **Check every change against the outputs.** A full run follows each round, and every stage's
  output hash is compared with the previous run's (`dvc.lock`). A change meant to keep the
  outputs must give the same hashes; a change that alters them (a new hash for near-duplicates)
  has its differences listed and reviewed.
- **Use the library's intended path.** spaCy's, Polars' and datasketch's documentation are
  checked for the supported way to do a step before any code of the project's own.
- **Measure the gain after the change too.** The store's batching (below) was measured only
  after it was built, and gained 5%.
- **Do not rerun what cannot change.** A code change that cannot affect a stage's outputs (most
  often in the catalog package, which every stage lists) is recorded with `dvc commit -f
  <stage>` instead of a full rerun.

## How each corpus stage uses processes

Every worker pool starts its processes with `spawn`: Polars runs its own threads, and its
documentation warns that `fork` can then deadlock a worker.

- **Word repairs:** repairing, splitting glued words and joining split words run by part in
  worker processes; the corpus-wide counts between them are lazy Polars queries in the main
  process.
- **spaCy pass:** each worker takes whole parts, largest first (a part holds at most 10,000
  texts, 1.4% of the corpus), annotates them in batches of 256 texts and writes their cache
  files; no document crosses processes. The pipeline ends with spaCy's `doc_cleaner`, which drops
  the tok2vec tensor: it is never cached or read, and it made each document about three times as
  large to hold (2026-11-13).
- **Sentence languages:** each worker takes whole parts with its own copy of lingua's models
  (about 690 MB) and runs lingua on one core (`RAYON_NUM_THREADS=1`).
- **Cleaning:** each worker takes whole parts.
- **Corpus build:** the corpus-level rules read the whole table in the main process; the
  near-duplicate signatures are made by part in worker processes, each abstract's shingles built
  as it is visited and only its signature kept (about 1 kB; the corpus's shingles held at once
  would take about 15 GB, a calculation); only the comparison runs in order.

## Full runs

From run 7c, seconds per stage. Run 7c's total is the whole `dvc repro`; the later totals add
up the stages.

| Run | Date | Total | Word repairs | spaCy pass | Sentence languages | Cleaning | Corpus build | Report |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Run 7c | 2026-11-15 | 42.4 min | 155 | 1,482 | 347 | 272 | 219 | – |
| First round's check | 2026-11-16 | 28.9 min | 145 | 942 | 246 | 214 | 157 | 27 |
| Staging-table check (build and report only) | 2026-11-16 | – | – | – | – | – | 128 | 1 |
| Second round's check | 2026-11-16 | 19.3 min | 115 | 657 | 122 | 211 | 51 | 1 |

Earlier runs, as recorded at the time:

- **Run 1**, the first full build (705,283 abstracts): the spaCy pass about 240–350 abstracts a
  second, about 47 minutes across restarts, with a peak of about 16 GB; `build-corpus` about 4.5
  minutes, of which near-duplicates 2 min 54 s. On the Mac, run 1's workers spent 44% in named
  entity recognition, 21% in tok2vec, 12% in the attribute ruler, 8% in lingua, 6% in the
  lemmatizer and 3.5% in the span ruler.
- **Run 2:** the spaCy pass about 500 abstracts a second. It held every part's rows until the
  end and reached 25 of the machine's 31 GB.
- **Run 3:** the spaCy pass 19 min 40 s (about 654 a second), sentence languages 5 min 30 s,
  cleaning 3 min 30 s and 40 s, the corpus build about 3.6 minutes. The machine's peak was about
  15 GB (the main process 5.2 GB). A pattern change reran about 9 minutes of stages, a language
  setting about 14.
- **Run 4:** the spaCy pass 22 min 50 s, sentence languages 5 min 34 s, cleaning 4 min 05 s,
  the corpus build 3 min 36 s.
- **Run 7b:** about 40 minutes: the spaCy pass 23.6, sentence languages 5.9, cleaning 4.5, the
  corpus build 3.6 (near-duplicates 2.9), the glued-word count 2.0, and the same 31-second
  metadata check in four stages. It changed 199 of 705,283 texts.
- **Run 7c:** joining the split words took 43 s on one process over all texts (in worker
  processes since the first round).

## Memory

Peaks from the run log (`getrusage` in each command's main process): the corpus build about
13 GB, the word repairs 11–12 GB, the report 4.6–6.4 GB, cleaning 3.3–4.2 GB, sentence
languages about 2.6 GB, the spaCy pass's main process 5.6 GB in run 7c and 3.6 GB after the
first round. In the second round the pass's and the language stage's main processes peaked at
about 0.7 GB, each spaCy worker held about 2 GB, and the word repairs' main process 9.1 GB.
The log's figure for the largest worker has equalled the main process's in every record, so it
is not yet a worker's peak (an open item, below).

- **`gc.freeze` before spaCy forked its workers** (2026-11-13, `6d4c9d41`): one part on ten
  workers used 17 GB, about 1 GB of it private to each worker, as the workers' garbage
  collection copied the pages of the objects they inherited; freezing saved about 1.5 GB per
  run. Gone since the second round, whose workers are spawned and read their own parts.
- **The old lemma-vote pass** queued all of its input through `ProcessPoolExecutor.map` (19 GB);
  a bounded queue of two chunks per worker kept its memory flat.

## Changes

### Before the rebuild's speed rounds (2026-11-12 and 13)

- **The small spaCy model for the pass** (2026-11-12): one CPU process ran 19,270 words a
  second, `en_core_web_trf` on the GPU 11,332, so the full pass would take about 5.5 hours with
  the transformer against 25–30 minutes on ten processes (the full-pass times extrapolated).
- **srsly's msgpack extensions looked up once in the pass** (`b1805b41`): srsly asked its
  registries on every pack and unpack, a third of the main process. The fix took the function
  calls on 4,000 abstracts from 83.6 million to 23.6 million and rebuilding the documents from
  7.1 s to 0.8 s (cProfile on the Mac); on the homelab machine the main process went from 95% to
  84% busy, the workers from 60–67% to 78–90%, and the pass from about 285 to 350 abstracts a
  second (+23%). The gain was smaller than the profile's third suggested; the 6 physical cores
  were given as the reason, not measured.
- **Batches of 256 texts:** batches of 1,000 were slower in the same timed test, 10,000
  abstracts on ten workers in about 95 s, about 40 s of it fixed setup.
- **Named entity recognition left out of the pass** (`named_entity_recognition: false`; the
  decision is in the decision agenda, 4.9). It took about 44% of a worker's time: a
  10,000-abstract part took 30 s instead of 44, the 705,283 abstracts about 27 minutes instead
  of 40. For comparison, the last full pass of the earlier corpus (lemma votes with the tagger
  and lemmatizer) read 1,274,029 abstracts, 206 million tokens, in 30 minutes on 10 processes.
- **lingua's candidate languages** (measured on the Mac, 2,000 sentences): 5 languages ran
  9,900 sentences a second, 21 languages 4,300, all 75 1,300 with about 1.2 GB per worker, so
  all 75 were rejected. The list of about 20 made the pass about 4% slower (10% had been
  estimated). The Mac's "about 150 MB" of models was far too low: 19 languages take about
  690 MB on Linux.
- **Stage 2 split three ways** (the spaCy pass, sentence languages, cleaning; the decision is in
  the decision agenda). Estimated beforehand: lingua 10% of the pass (about 1.2 minutes over 10
  processes as its own stage), the patterns and keeping rules 725 cached documents a second in
  one process (about 1.6 minutes over 10), and a pattern change rerunning about 6–7 minutes of
  stages instead of 30, a language setting about 8. Run 3 measured about 9 and 14 minutes.
- **Sentence languages with lingua's multi-threaded methods** (`a3c36a71`): one process and one
  copy of the models instead of spawned workers each with their own. Reading the documents then
  ran on one core, the cause of run 3's 5 min 30 s; the second round went back to workers, each
  with lingua on one core (`b98213a6`).
- **Work metadata fetched with several requests at once** (`9b9a6abb`): one request at a time
  took about 1.5 s; eight threads made about five a second (the free key allows 30), about 40
  minutes in all. Crossref takes 200 DOIs a request (300 make the address too long), three
  requests at once with a contact email, about 500 works a second; PubMed about 100.

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

The check run (profiled, 2026-11-16) took 19.3 minutes of stages instead of 28.9: the word
repairs 115 s (145), the spaCy pass 657 (942), sentence languages 122 (246), cleaning 211 (214),
the corpus build 51 (157), the report 1 (27). Every output but the corpus build's and the
report's had the same hash as before. The new hash changed 491 near-duplicate decisions of
645,542 (251 works newly marked, 240 no longer), and one work moved into the author cap and one
out: 639,898 abstracts kept instead of 639,909. The changed pairs lie on both sides of the
threshold of 0.8: their exact Jaccard similarity on the same shingles has a median of 0.833 for
the newly marked and 0.818 for the others, seven in ten within 0.05 of the threshold,
from 0.68 to 0.92 at most. MinHash with 128 permutations estimates a similarity of 0.8 with a
standard error of about 0.035, so a new hash moves the pairs near the threshold either way.

In the profile, the spaCy workers used 6,423 CPU seconds, 98% of their ten processes (the first
round's used about 7,950 and the main process about 650 more). The attribute ruler now takes 4.7%
of them; tok2vec's layers about 40%, the project's sentence rules 6.5%, the tokenizer 6.1%. The
pass is limited by its workers alone.

## The vector store's encoding

Every usage the rounds and scans need is encoded once with XL-LEXEME (`docs/methods.md`). The
text work runs on the CPU first (the manifest, in parallel by field on 11 processes, about ten
times faster than on the Mac). The same sentences, encoded different ways:

| Setup | Sentences per second | The whole store |
|---|---|---|
| Laptop GPU (Apple M3, full precision) | about 20 | about 25 hours |
| RTX 4070 SUPER, full precision | about 490 | about 60 minutes |
| RTX 4070 SUPER, float16 autocast on its tensor cores | about 1,600 | about 20 minutes |

A short test sentence ran at 775 a second against 490 on the store's real sentences, which is
why the first estimate missed. DeBERTa-v3-large ran at about 900 a second. The gains came from
the hardware and the arithmetic. Of the pipeline changes, batching sentences of similar length
and building the next batch's token ids in worker processes removed idle time between batches,
but added about 5%, because the GPU's arithmetic was the bottleneck, not the tokeniser; that
was measured only after the change was made. Batching by a token budget and writing shards on a
background thread followed for later stores.

## Planned: reusing unchanged documents

Planned 2026-11-15 (the author), measured first. A full run reannotates every text, while run
7b changed 199 of 705,283. Reading cached documents ran at about 5,000 a second when first
measured (2026-11-13) and at 7,600–13,000 a second on 2026-11-15, writing them at 2,500–4,400 a
second on one process, so copying unchanged documents costs under a minute over 10 processes.
The plan: each cached document keyed by the hash of its repaired text and a fingerprint of the
pipeline (spaCy and model versions, the annotation code, the stage's settings), and only changed
texts annotated, as the vector store reuses vectors by text hash; the same for sentence
languages. A full pass without reuse must give identical documents. Known non-English works need
no rule of their own: their documents are reused while their text is unchanged, and no language
decision is frozen. The plan's other steps, one metadata check and the repaired texts written
once, were done in the first round. The reuse waits until more passes are expected.

## Measured and not adopted

- **Moving the row-building into the spaCy workers** (2026-11-13): the same time with or
  without it, against a guess of 30–40% faster.
- **Splitting the attribute ruler's `IN` tests into exact values** made it slower (14.3 µs per
  token against 12.9): the matcher's cost is per rule, not per Python call.
- **Polars' streaming engine for the string steps written eagerly** gained nothing: an eager
  `select` already runs in parallel. The gain came from replacing `Series` methods.
- **Skipping works before the pass** would save at most 9% of it, the share of works dropped at
  all, and most drops need the pass's sentences.

## Open candidates

- **The worker count.** With the pass's main process idle, 11 or 12 workers may help, or not, on
  6 physical cores; a timed test on part of the corpus decides.
- **The sentence rules**, the project's own component, take 6.5% of the pass's workers.
- **The cleaning patterns**, now the slowest stage after the pass (211 s, 67% of its workers in
  spaCy's span ruler). The first-token regular expressions (URLs, DOIs, LaTeX, spam) take
  42% of the cleaning ruler's time: 6.0 µs per token, 3.5 without them; the `IN` tests and the
  quantifiers cost 1.2 to 1.7 µs more. A faster form must keep every match the same.
- **tok2vec**, 36 to 39% of the pass's worker time, is the model itself; nothing found to change.
- **The word-repair stage** sends every text to its workers and back three times, once per step.
- **Reusing unchanged documents** (above).
- **`DocBin.add`** walks every token in Python to collect strings, now in the pass's workers.
- **The run log's worker memory:** the largest worker's peak equals the main process's in every
  record, so the measurement needs fixing before worker memory can be read from it.
