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
64 with it off, in the trial), so the verdict comes before the note, and two of the notes read
(those of the nine items the setups disputed) contradicted their own verdicts. On those 50 items (`batch_00`), the setups compared:

| Setup | Agrees with the API judges | Notes match verdicts |
|---|---|---|
| API, low effort, one item per request (the two judges agree on all 50) | — | 2 of 9 disputed contradict |
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

## The audit of record's results (2026-11-20)

The two judges agreed on 4,385 of 4,442 items and the adjudicator settled the other 57
(`data/review/usages_audit/`). **The calibration passes:** the agents' verdicts agree with the
author's on 57 of 60 (bar 54). The three disagreements: an adjective in a generic staining
technique (*… Golgi method*) that the author read as a name; a noun in a list of chapter titles,
which the codebook counts as a table-of-contents heading but the calibration page's summary of the
rules left out; and a paper title followed by a description, which the judges themselves split on
(the record's own title, or a bibliography entry).

| Bar | Result |
|---|---|
| Precision, pooled | passes: 42 errors in 1,980 (bar 51): 21 names, 10 other words, 8 not English, 3 parts of speech |
| Precision, per word | 3 words over 2 errors, all held out |
| Scope | fails: 290 of 950 left-out items were real uses; 47 words had 3 or more |
| Recall, pooled | passes: 0.42% missed (1,174 of the 1,512 judged candidates were real uses) |
| Recall, per word | 6 words over 2%: *neural network* (6.8%), *intentionality* (2.4%), *user interface* (2.2%) and three held-out words (17.1%, 7.5%, 3.7%) |

**Why scope failed** (`scripts/evidence/usages_rule_options.py`, `rule_options.json`). Of the
86,198 left-out usages, the judged items give each reason's rate of real uses: a noun word tagged
as a verb 11% (52,703 usages), ORG 22% (9,027), PRODUCT 46% (7,949), WORK_OF_ART 18% (6,091), a
noun word tagged as an adjective 80% (6,080), LAW 31% (869). A model drawing each word's 10 items
from its strata at these rates forecasts 51 words sent back against the 47 observed. The bar sends
a word back once roughly 15% of its left-out usages are real uses, so no rule on a whole label or
tag brings the forecast below about 30. About 1% of real uses are left out overall, and the share
is not even over time: the judged left-out items were real uses 19–20% of the time in the 1950s to
1970s and 29–37% from the 1980s.

**The adjectives are word by word.** All judged adjective-tagged items of seven noun words were
noun modifiers (*component failure*): *component* 8 of 8, *feedback* 2 of 2 and five held-out
words (9 of 9, 10 of 10, 7 of 7, 3 of 3, 2 of 2; one of them has more adjective-tagged usages than
in-scope ones), while none of *medium*'s 10 was.

**No feature of the entity separates names from generic terms** (`scripts/evidence/name_features.py`:
the 26,014 usages left out for a name re-annotated, 99.0% with the same label). An entity that is
the target alone held real uses in 48 of 77 judged items (62%), a one-token entity in 36 of 53,
and longer entities in 15–35%; a bracketed acronym after the entity, digits or capitals in it,
and whether its other words are common lower-case words made little difference.

**The precision words' errors are noise the rules cannot reach,** except one: six of one word's
eight errors are French reflexives (*lui-même*), hyphen parts the transformer tags X in sentences
the language filter labelled English. The others are OCR fragments of longer words and a Spanish
sentence (one word), and an acronym, a name and a page's HTML (the other).

**The missed uses are three matcher gaps and one tokenizer gap.** Unfamiliar plurals the small
tagger reads as singular proper nouns (NNP) keep their form as lemma, and the plural fallback
covered only NNS and NNPS; a capitalised plural phrase (*Neural Networks*) found no phrase, as its
head's lemma is the plural; a phrase reaching into a hyphenated token (*multi-user interfaces*,
*neural network-based*) was not searched. The tokenizer gap: words glued to punctuation in
Springer tables of contents (*Intentionality.-*) and footnote marks (*attention,2*) are one token
the matcher cannot read; most of *intentionality*'s 26 candidate misses are these.

## The fixes, recorded before the redraw (2026-11-20)

1. **Seven noun words accept ADJ** (`curated/words.tsv`, the author): *component*, *feedback* and
   the five held-out words above.
2. **A generic term labelled as a name stays in scope** (`usages/entities.py`, the author): an
   entity that is the target alone (its other tokens punctuation or stop words), or a PRODUCT
   whose last token is the target, no longer puts the target inside a name. On the judged items
   these held 81 real uses in 137 (59%); the forecast was 34 words sent back on scope instead of
   48 and 4.6 more errors in the precision draw.
3. **A hyphen part tagged X is out of scope** (`usages/usage_selection.py`): 51 usages in the
   table, the six judged all French.
4. **The matcher** (`usages/target_matching.py`): NNP takes the plural fallback; a spaced phrase's
   head may be a plural; a spaced phrase may start at a hyphenated token's last part or end at
   one's first part.

Left as they are: the tokenizer gap (fixing it means re-annotating the corpus), the noise behind
the other precision errors, and the per-decade unevenness, which the drift measures must allow for.

**The redraw.** The usages table is rebuilt with these rules, and all three parts are drawn again
for every word with seed 20261306, since the fixes change every word's kept and left-out usages:
20 kept, 10 left out and 20 candidate misses per word. The judges, the codebook and the bars are
unchanged; the calibration stands, as the codebook has not changed. A second failure goes to the
author.

## Amendment, before the redraw is drawn or scored (2026-11-20)

**Scope is read by its cost** (the author, after the literature check below). The 3-of-10 rule
measures how clean a word's left-out pool is, not what the exclusions cost the measures, and with
10 items it cannot keep both of its errors low (a word with 10% real uses among its left-out
usages is sent back 7% of the time, one with 30% 62%; acceptance sampling gives no better rule at
that size). From the redraw on:

- **The bar:** a word's estimated real uses wrongly left out are its left-out usages times the
  share of its judged left-out items that were real uses; over its in-scope usages plus those, at
  most 2% per word and 1% pooled, as for recall.
- **The screen:** a word with 3 or more of its 10 left-out items wrongly left out is reviewed, as a
  word with more than 2 kept errors is, without failing the audit.
- **Reported:** the estimated share lost per decade (the judged left-out items pooled over words by
  decade), and each word's combined loss, missed plus wrongly left out.

**Judge error is reported as a check.** The agents' verdicts decide the bars. Beside each pooled
rate the scorer reports the rate corrected for the judges' misclassification as the author's 60
calibration items measure it, kind by kind (the Rogan–Gladen estimator, with uniform priors on each
kind's sensitivity and specificity, and a 95% interval from their posterior draws). With 60 items
the interval is wide; it is reported, not used to pass or fail.

**The literature check** (2026-11-20) behind both:

- Leaving names out is standard in semantic change work, as their contexts change with events
  rather than meaning, but it is done word by word (Soni, Lerman and Eisenstein, *Follow the
  Leader*, on legal opinions and scientific abstracts) or by keeping proper names apart from common
  nouns (Kutuzov, Velldal and Øvrelid 2022), not usage by usage with a news-trained recognizer,
  whose rarer labels are its least reliable even on news (WORK_OF_ART, LAW, PRODUCT).
- Preprocessing errors of the kind this audit measures (missed targets, tagging and lemmatisation
  errors) were found in the field's main benchmark, with a call to document and audit
  preprocessing (Phan-Tat et al. 2026, *Evaluating the Evaluator*).
- An LLM annotator is validated on an expert's random sample against a codebook, refined on its
  errors and tested on fresh items; agreement between LLM runs is no evidence of validity (Pangakis
  et al. 2023; Ziems et al. 2024).
- Even accurate LLM labels can bias low rates estimated from them and break their intervals unless
  expert labels correct them (Egami et al. 2023).

**The rules reviewed against the corpus and the literature** (the author, 2026-11-20, before the
redraw; `scripts/evidence/usage_rule_profile.py`, `rule_profile.json`). Each rule was weighed on
whether what it drops does not carry the word's meaning, and on whether its errors could pose as
drift, a class whose share moves over the decades or whose errors fall on one sense.

- **Kept as they are:** the parts of speech per word (the norm in DWUG, SemEval-2020,
  NorDiaChange; the verb-tagged uses of noun words left out are 2.5–2.9% of every decade, 10% of
  them real uses); usages inside names left out, with the generic-term rule (dropping usages whose
  target sits in a name is the best-supported variant, with a small measured effect, Martinc et
  al. 2020; keeping every name would add about 25,000 usages, about 70% of them not uses, and half
  the excluded names occur once and are names three times in four); hyphen parts and noun
  modifiers (judged right 72 times in 75); one usage per abstract (one sense per discourse, Gale et
  al. 1992); titles of cited works as names (Del Tredici et al. 2019 on referential contexts).
- **A named instrument is a name.** The proper-noun tag lets in names the recognizer misses: 14 of
  the 26 judged kept items tagged PROPN were names (named scales and tests, a principle, a journal,
  a model), and such usages rise from 1.1% of the 1950s' usages to 2.9% of the 2020s'. A target in
  a capitalised run ending in an instrument word (*Scale*, *Inventory*, *Test*, *Questionnaire*,
  *Checklist*, *Battery* and the like) is inside a name: on the judged items the rule matched 30,
  27 of them names, and it removes 4 of the 42 kept errors for 1 real use.
- **Copies are not drawn.** 6,688 usages (3,536 drawn) sit in a sentence another work holds at the
  same place, 0.6% of the 1980s' and 1990s' usages; 365 pairs of works share three or more, the
  same paper indexed twice below the near-duplicate threshold. Diachronic datasets de-duplicate
  before sampling (NorDiaChange; Shoemark et al. 2019; Antoniak and Mimno 2018). When two works
  share three or more target sentences at the same places, the later is a copy: its usages stay
  in the table, marked with the work it copies, and are never drawn.
- **Each usage records its role, and the measures are checked with and without it.** Contextual
  vectors cluster by syntactic role (Giulianelli et al. 2020; Kutuzov et al. 2022), and the role
  mix moves: hyphen-part modifiers are 0.9% of the 1950s' usages and 1.9% of the 2020s', and for
  single words much more (*agent* 0.6% to 14%, *prompt* 0 to 11%, *user* 2.7% to 9%). Each usage
  is a head, a noun modifier (the next token a noun) or a compound part, and step 7's
  pre-registration must report the drift measures with and without modifiers and compound parts.
- **Flags, reported per decade, decided at step 7:** a capitalised target (not opening its
  sentence), so the measures can be rerun without capitalised terms, where the name rules are
  least reliable; the word mentioned as a word (*the term "attention"*); and a list-like sentence
  (most of its words capitalised, or a table of contents' separators), where titles and
  instrument lists sit.

## The redraw's results (2026-11-21)

The two judges agreed on 4,171 of 4,224 items and the adjudicator settled the other 53
(`data/review/usages_audit_redraw/`). The calibration of record stands (57 of 60).

| Bar | Result |
|---|---|
| Precision, pooled | fails: 55 errors in 1,980 (bar 51): 40 names, 5 other words, 4 not English, 4 parts of speech, 2 unstated |
| Precision, per word (screen) | 4 words over 2 errors: *medium* (3 parts of speech, 1 name) and three held-out words (3 not English and 2 other words and 1 unstated; 3 names and 1 unstated; 2 other words, 1 name, 1 not English) |
| Scope cost, pooled | passes: 0.32% of real uses wrongly left out (bar 1%) |
| Scope cost, per word | fails: 6 words over 2%: *prompt* (14.1%), *information processing* (5.3%), *agency* (2.7%) and three held-out words (6.4%, 3.8%, 2.1%) |
| Scope screen | 29 words with 3 or more of their 10 left-out items real uses |
| Recall, pooled | passes: 0.21% missed |
| Recall, per word | fails: *intentionality* (2.4%), the tokenizer gap left as it is |
| Judge error (check) | kept 2.78% observed, 0.78% corrected (0.00–9.22%); left out 19.1% and 1.8% (0.0–20.3%); missed 88.2% and 93.2% (0–100%). The calibration holds one kept error in 30, so the intervals say little |

**Where the kept errors are.** Targets capitalised mid-sentence are 87 of the 1,980 kept items
and hold 32 of the 55 errors (37% of them wrong, against 1.2% of the rest); proper-noun-tagged
targets were wrong 16 times in 34, the generic-term rule 3 times in 5. The errors fall with time:
43 in the 1,117 items before 2000 (3.9%), 12 in the 863 from 2000 (1.4%). Capitalised targets were
wrong 24 times in 56 before 2000 and 8 times in 31 from 2000, where most are technical terms
written in capitals (*Artificial Intelligence*, *User Interface*, *Neural Networks*). No rule on
what the table records removes errors without losing as many real uses. With each word weighted by
its kept usages, leaving out capitalised targets would bring the kept error rate from 2.8% to 0.7%
and lose 2.3% of real uses; proper-noun tags 1.1% and 1.3%; capitalised proper-noun tags 1.1% and
1.2%; list-like sentences 2.2% and 1.1%. The losses would fall mostly on the recent decades' terms.

**Where the scope cost is.** *information processing*: the transformer tags the phrase's -ing head
as a verb, and 6 of the 7 judged verb-tagged items were real uses. *prompt*'s 14.1% rests on one
real use in 10, an adjective-tagged noun modifier. Adjective-tagged *prompt* before a noun runs at
12–22 usages a decade until the 2010s and 130 in the 2020s (*prompt engineering*), about a tenth of
the 2020s' uses; adding the tag would also let in the adjective (*prompt response*), about 25
usages a decade, as many as all of *prompt*'s kept uses in the 1960s. *agency* (3 of 10 inside
organisation names) and the three held-out words (1 to 5 of 10 over several labels and tags) show
no single pattern. The per-word bar reads 10 items, so it cannot resolve a word whose left-out
usages outnumber its kept ones: one real use among 10 of *prompt*'s 2,799 left-out usages is 14% of
its 1,705 kept ones. This was not checked when the bar was set; the bar stands as recorded.

**The author's decisions (2026-11-21).**

- **Precision is accepted as measured:** 2.8% of kept usages, 3.9% before 2000 and 1.4% after.
  Step 7's pre-registration must rerun the drift measures without capitalised targets and report
  both, as a required check.
- **information processing takes VERB** (`curated/words.tsv`). The fix is checked without
  redrawing the audit: 20 drawn usages it brings into scope, seed 20261308, are judged like the
  precision items and pass with at most 2 errors, the precision screen's bar per word
  (`scripts/review/usages_fix_check.py`, `data/review/usages_fix_check/`). The rebuilt table holds
  213 more usages in scope (171 of them drawn) and nothing else changed; the check **passes**, 1
  error in 20, the two judges agreeing on all 20.
- **The other five words are recorded, not fixed:** *prompt*'s 2020s noun modifiers stay lost to
  the adjective tag (step 7's runs without modifiers do not depend on them), and *agency* and the
  three held-out words keep their rules.
- **Three more name tests were run and none is used.** Each was set before it ran and read the 151
  capitalised kept items of both audits (51 wrong, 100 right; `data/review/usages_audit_redraw/`):
  - **A second recognizer** (`scripts/evidence/gliner_names.py`, `gliner_names.json`): GLiNER
    (large v2.1, threshold 0.5, the codebook's name types as labels) put 25 of the wrong items and
    31 of the right ones inside a name of more than the target (26 and 40 for any name covering
    it), short of its bar (half the wrong items caught, twice as many wrong items as right ones
    removed). Like the transformer's recognizer, it does not tell a name from a term in capitals.
  - **The lower-case check** (`scripts/evidence/capitalised_runs.py`, `capitalised_runs.json`;
    Mikheev 2002): a capitalised run the corpus writes in lower case under a tenth of the time,
    seen at least 3 times, removed 9 wrong items for 5 right ones (7 for 2 on the first audit's
    items, 2 for 3 on the redraw's), short of its bar (5 wrong items and twice as many as right
    ones). Counting runs seen fewer than 3 times as names removed 39 for 44: most of the names are
    cited titles seen once, where the corpus holds no evidence.
  - **Publication and organisation words** in the run (*Journal*, *Society*, *Institute* and the
    like, as the instrument rule's test words): 2 wrong items for 1 right one.
- **Further work waits on step 7** (the author). If the drift measures with and without
  capitalised targets agree, the names are left as they are. If they differ, a classifier is
  trained on capitalised uses labelled by the judges (about 1,000) and confirmed by a fresh
  precision draw. A filter judging every capitalised use with a language model (about 83,000) was
  set aside for its cost and for putting a language model inside the pipeline.
