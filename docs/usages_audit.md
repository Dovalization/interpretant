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
