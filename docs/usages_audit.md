# The usages table: model comparison and matching audit (agenda 4.1–4.6, 4.9; phase step 6)

Pre-registered 2026-11-18, before any sample was drawn or any item judged; amended 2026-11-19 for the
held-out words, still before any draw.

## What the table is

One row per mention of a word-table target in a kept work (`catalog/usages.py`, built by
`interpretant build-usages`). A mention matches by its spelled lemma or form, as a part of a
hyphenated word, or as a phrase (`usages/target_matching.py`). A mention inside a removed range is
left out. Each row records:
- its sentence of the contextual sentence view: span, word count, language, exclusion;
- the target's span, its compound and position when it is part of a hyphenated word;
- the head token's part of speech, tag, morphology and lemma;
- the entity it sits in, recognised by the small model on the sentence alone (agenda 4.1);
- the static view's entry for the token;
- whether it is in scope, and whether it is the usage drawn for its work and word.

A usage is in scope when its word allows its part of speech and it sits in no name. The names are
all 11 of spaCy's name labels: PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW,
LANGUAGE (the author, 2026-11-18). A target that is one part of a hyphenated word is in scope
whatever the whole word's tag, as agenda 4.2 decided (*attention* in *attention-based*). One usage
is drawn per work and word: the eligible one (in scope, in a sentence that is not cut off) with the
smallest draw key, seed 20261301 (`params.yaml`).

**The working table.** The comparison and the audit sample a table built with the small model over
all 88 parts (10 min on the remote runner, code `5881332`), not the 8 development parts the phase plan named:
the whole table costs little, and a corpus-wide sample measures the whole corpus. It is not the
record. The locked `dvc repro build_usages` comes after step 7's pre-registration, as the plan
says, and no count per word and decade is read for drift.

## Decisions (the author, 2026-11-18 and 19)

- **Short sentences stay in the draw.** In the working table, 1.74% of eligible usages sit in
  sentences under 10 words (words holding a letter). A 10-word floor would lose 10,036 of 881,800
  draws (1.1%), and the share falls by decade from 2.5% in the 1950s to 1.2% in the 2020s, so the
  loss would be uneven over time. A floor of 5 words would lose 805 draws
  (`scripts/evidence/short_sentences.py`). Sentence length stays in the table, and the measures
  run a pre-registered sensitivity check without usages under 10 words. Reweighting or topping up
  after a floor would restore the counts but not the composition: a lost draw is a work whose
  only mention sits in a short fragment.
  - **The benchmarks.** Only SemEval-2020 Task 1 and DWUG set a numeric floor: sentence length
    "> 9 (before removal of punctuation)" (the English corpus page) and "a minimum of ten tokens"
    (Schlechtweg et al. 2021), with no stated reason. Tokens there include punctuation, so the
    table's unit is stricter. DURel, SURel, RuSemShift, RuShiftEval, LSCDiscovery, NorDiaChange
    and DiscoWUG state no floor. ChiWUG removed "insufficient contexts" by hand, and DIACR-Ita
    filtered OCR paragraphs only. The DWUG and DIACR-Ita analyses trace annotator disagreement
    partly to short contexts. No study found isolates sentence length in contextual change scores.
  - **One usage per work** (agenda 4.3) has no benchmark precedent: none caps usages per document.
    Kutuzov et al. (2022) show why a cap helps: 27 sentences from one 1994 article made a spurious
    change.
  - **Name uses.** No benchmark removes name usages (DiscoWUG removes name targets). Laicher et
    al. (2021) found proper names' influence on BERT clusters small but above the baseline, and
    Montariol et al. (2021) suggest entity filtering when the change studied is in concepts, not
    names. The table keeps the filter, and the measures check with and without it (phase plan).
- **The held-out set is drafted now,** before the comparison and the audit, so both cover it
  (agenda 6.2). Fixed 2026-11-19 (`docs/held_out.md`): 24 documented changes and 22 stable words
  joined `words.tsv`, marked held out, so the table has 99 words, and the working table is
  rebuilt. Their usages are compared and judged like the others, but no committed file names
  them: keys and per-word results carry `Word.reported_name`, a hash.

## The model comparison (agenda 4.9), set before reading

- **The sample.** 300 usages per word (all usages, in scope or not), seed 20261302, spread over the
  decades as evenly as the word's usages allow: each decade gets an equal share, a decade with
  fewer usages gives all it has, and the rest is spread over the others. Within a decade the
  usages with the smallest `sha256(seed:work:word:content_key)` are taken.
- **The transformer** re-annotates each sampled usage's sentence from the cached tokens
  (`usages/entities.load_transformer`: float16 autocast and the lookup attribute ruler, measured
  to change 5 tags and 4 lemmas in 192,828 tokens and no entity, `docs/performance.md`), then the
  lemma exceptions. Its part of speech, lemma and entity give the usage's scope under the same
  rules.
- **What is measured,** pooled over the sample (each word weighs the same):
  - **scope flips:** the usage is in scope under one model and not the other;
  - **match flips:** under the transformer's lemma the token stops being the target. Matching
    reads the spelled lemma or the form, so a lemma change that keeps the match changes nothing in
    the table. Raw lemma differences are reported too. Targets that are part of a hyphenated word
    are matched by their letters, so only their scope is compared.
- **The decision.** The transformer annotates the table if more than 2% of the sampled usages
  flip (scope or match) **and** it is right more often on the judged disagreements: 50
  disagreements drawn with seed 20261303 are judged
  (`curated/judging/model_comparison/prompt.md`, the two readings shown as A and B in random
  order), and among those where exactly one reading is right, the transformer must be right on
  more than half. Otherwise the small model stays. The decision goes into `params.yaml`.
- **The sentence context check.** On the same sample the small model also recognises entities on
  the whole abstract. Agenda 4.1 measured 99.87% agreement with the sentence alone; if the
  agreement here falls under 99%, the author decides whether to recognise on the abstract.
- **Not covered:** the sample comes from the small model's matches, so mentions only the
  transformer would match are left to the recall check below.

## The matching audit, set before reading

**What is judged** (`curated/judging/usages/prompt.md`): whether a marked place is a use of the
target word in one of its parts of speech, outside a name, with the marks on the word. Every sense
of the word counts; a mention of the word as a word counts. Three kinds of item:

- **Precision (kept).** 20 drawn in-scope usages per word, seed 20261304, spread over decades as
  in the comparison: 1,980 items for 99 words (fewer if a word has fewer usages).
  - **The pooled bar:** at most 51 errors in 1,980. That passes 96.8% of the time at a true error
    rate of 2% and 0.04% of the time at 4% (the plan's bar of 23 was set for about 800 items; the
    rule is unchanged: the smallest number that passes at least 96% of the time at 2%; if fewer
    items are drawn, the bar is recomputed by the same rule before judging).
  - **Per word, as screening:** at most 2 errors in 20 (a word with a true rate of 5% passes 92%
    of the time, at 10% 68%). A word that fails is reviewed: its errors are read and the rule or
    its parts of speech fixed.
- **Scope (left out).** 10 usages per word that the table left out of scope, seed 20261304. If 3
  or more were wrongly left out, that word's rule (its parts of speech or the name labels) goes
  back for review. A true rate of 10% sends a word back 7% of the time, 30% 62%.
- **Recall (missed).** An independent case-insensitive search runs over every kept work's text
  outside removed ranges. For each word it looks for the word, its plurals (*-s*, *-es*, *-ies*),
  its other spellings from the spelling map and their plurals, alone or as a hyphen part. A hit
  that overlaps no mention is a candidate miss. 20 candidates per word are judged (all, if fewer),
  seed 20261304. The estimated misses of a word are its candidates times the share judged right,
  and the missed share is those misses over its in-scope usages plus the misses.
  - **The bar:** at most 2% per word and at most 1% pooled.

**The judges.** Two Sonnet judges each judge every item on their own, with the saved prompt; an
Opus adjudicator settles their differences. Items carry no field, decade or model label beyond
what the prompt shows.

**The calibration.** The author judges 60 items blind: 30 kept, 15 left out, 15 missed, drawn from
the judged items with seed 20261305. The agents' verdicts stand if they agree with the author on
at least 54 of 60. Agents agreeing 90% of the time reach 54 in 61% of draws, at 95% in 97%.

**The pilot round and the amended codebook (2026-11-19).** The first draw was taken from the
small model's table and fully judged by one pair of judges (wave 1) before the model comparison was
decided; the comparison then chose the transformer, so that draw does not audit the table of
record. It is kept as a pilot (`data/review/usages_audit_pilot/`), and its reports showed the
judges reading three cases differently. Before any item of the table of record was judged, the
codebook was amended: words in titles of cited works are names, while the abstract's own title and
headings are uses (the author); an -ing form counts for a noun target only where it works as a noun
or noun modifier (the author); and noun modifiers, coined names and identifiers are settled as the
judges had mostly read them. The audit of record is drawn from the transformer's table with the
same seeds and judged fresh under the amended codebook; no pilot verdict is reused.

**The judging setup (2026-11-19, before any verdict of record was scored).** The judges are
`claude-sonnet-5-5` and the adjudicator `claude-opus-5-5`, called through Claude Code's command
line (`scripts/review/judge_cli.py`) at high effort, one item per call. Each call sees only the saved
codebook, as its whole system prompt, and the item: no tools, settings, MCP servers or project
files, no saved session, and an answer constrained to the codebook's schema. A first run through
the Message Batches API at low effort was stopped after 1,000 items per judge, and none of its
verdicts is used: at low effort Sonnet does not think (69 output tokens per item with thinking on,
64 with it off, in the trial), so the verdict comes before the note, and two of the 50 notes
checked contradicted their own verdicts. On those 50 items (`batch_00`), the setups compared:

| Setup | Agrees with the API judges | Notes match verdicts |
|---|---|---|
| API, low effort, one item per request (the two judges agree on all 50) | — | 48 of 50 |
| Command line, low effort, 50 items per call | 44 of 50 | several contradict |
| Command line, low effort, one item per call | 47 of 50 | 50 of 50 |
| Command line, high effort, one item per call (chosen) | 45 of 50 | 50 of 50 |

Where the chosen setup differs from the API judges, its reading follows the codebook (a cited
title or a product's name left out as a name, an -ing form taking an object read as a verb, a
hyphen part counted).

**If a bar fails.** The rule is fixed and the failed part of the audit is drawn again, seed
20261306, after a note here records the fix before the redraw. Any change to `words.tsv` needs
the author. If the calibration fails, the prompt is amended on the disagreements and 60 fresh
items are drawn with seed 20261306. A second failure goes to the author.
