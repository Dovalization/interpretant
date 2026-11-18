# The lemma review (agenda 3.3)

Pre-registered 2026-11-17, before any form was judged. The list comes from
`scripts/evidence/lemma_review_list.py` (commit 1e8bfb2d) over the kept text of the build of record
75f36a82: 9,677,037 plural-noun and third-person-verb tokens, auxiliaries aside.

## What the list holds

The static view writes spaCy's in-context lemma (3.2), so the review looks for spaCy's errors,
with the voted lemma map as the tool that finds them. Forms with at least 20 uses unless stated.

*Correction (2026-11-17):* the map the list read, here called "the voted map", is
`data/processed/lemmas.tsv`, the frozen grid's type-level map from simplemma's lemmas of bare word
types (36,726 rows, `docs/corpus.md`), not the map voted from spaCy's tagger in context
(`data/processed/lemmas_voted.tsv`, 22,313 rows). It shaped which forms are `disagree` and the
voted lemma of the tail check; it is not rerun, since round L1 found the exceptions below the
static measurement's noise (`docs/predictions_lemmas.md`). Below, "the voted map" keeps that
meaning.

| Kind | Forms | Tokens | What it is |
|---|---:|---:|---|
| disagree | 489 | 24,745 | Both merge, to different lemmas, spelling aside (3.3's list; any count) |
| override | 158 | 340,517 | spaCy merges a form the override file decides (*data* → *datum*, *media* → *medium*, *ethics* → *ethic*) |
| homograph | 715 | 164,961 | spaCy merges into a singular rarely of the same part of speech (*others* → *other*, *goods* → *good*, but also *embeddings* → *embedding*) |
| suspect_merge | 330 | 27,243 | spaCy merges into a lemma rare beside the form (*diabetes* → *diabete*, *cosmos* → *cosmo*) or the form is mostly a name (*socrates*) |
| policy_keep | 12 | 17,616 | Candidates under the override file's 2026-11-11 rule: own noun entry, singular mostly an adjective (*contemporaries*, *majors*) |
| hyphenated | 1,091 | 2,714 | spaCy's lemma differs from the last segment's lemma rejoined (3.5's note; any count) |

Two findings came with the list:

- **The overrides were not applied.** The thin static view takes spaCy's lemma alone, so it writes
  *datum* for *data* (139,839 uses) and *medium* for *media* (57,608), merging the mass media into
  the target word *medium*. The override file
  carries into the exceptions file, so these are fixed when the view is rebuilt with the
  exceptions (3.6).
- **Auxiliaries were lemmatised** (*is* → *be*, *has* → *have*; 1.47 million tokens), against 3.2.
  Fixed in commit 1d9f6fc7; the view is rebuilt once, with the exceptions.

The agenda's named examples: spaCy keeps *forceps*, and *mises*, *hans* and *rogers* have under 20
uses; *cosmos* is a suspect and *thanks* and *currents* are homographs; *novels* → *novel* (1,215
uses) and *logos* → *logo* (370) are on no list, as their singulars are used as nouns often enough
(a fifth and two thirds of their uses).

## Who judges what

- **The author** (the judging page): the 100 most frequent `disagree` forms (3.3); 30 drawn from
  the other 389 (seed 20261116), where the voted map's lemma right on at least 27 accepts it for
  the untested tail (3.3); and the 12 `policy_keep` candidates.
- **Agent judges**: every `suspect_merge` and `homograph`, the 30 drawn `hyphenated` forms, and,
  for calibration, the author's 142 forms. Two Sonnet judges see each form independently, as the
  page shows it (the candidate lemmas in random order, their sources hidden, three corpus
  sentences); an Opus judge settles the forms where they differ.
- **The author, after the agents** (amended 2026-11-17, after the agents judged and before any
  author answer): the page opens a second part once the 142 blind answers are in. First a second
  look at the forms where the author's blind answer differs from the agents', with both shown;
  the blind answers stay as they were and score the calibration, and the second look's answer is
  the author's final one. Then the 50 most frequent suspect merges and homographs (105,228 of
  their 192,204 tokens), with spaCy's lemma and the agents' shown and the agents' suggested; the
  author's answer there replaces the agents'. The calibration forms are mostly spelling-style
  disagreements, so these 50 put the agents' policy calls (*others* kept, *professionals* merged)
  in front of the author.
- **No judging**: the override file carries into the exceptions file, except *goes*, which 3.2
  decides later (ordinary verbs merge: *goes* → *go*). *does* stays: outside auxiliary use spaCy
  reads it as a plural noun (*doe*).

## Rules, set before reading

1. **Calibration.** The agents' final lemma is compared with the author's blind answer on the
   author's 142 forms (keeping the form counts as a lemma). If they agree on at least 90%, the
   agents' verdicts are accepted for the suspects and homographs, except where the author's
   answer on the 50 calls replaces them. If not, the author reviews the agents' other changes to
   spaCy's lemma on those forms, most frequent first. The tail check (3.3) uses the author's final
   answers; the blind count is reported beside it. *senses* (a worked example before judging),
   *formes* and *guilders* (asked about while judging) were talked through with the author, so
   they are left out of the calibration (139 forms).
2. **The hyphenated rule.** If the agents' final lemma equals the rejoined segment on at least 27
   of the 30 drawn forms, the rule applies to all 1,091; otherwise more are drawn.
3. **Targets.** After the exceptions, no target word may be mapped to another word.

Approved corrections go into `curated/lemma_exceptions.tsv` with the carried overrides (3.2).

## Results

Judged 2026-11-17. The author answered all 142 forms blind, gave a final answer on the 15 where
the second look showed a difference, and answered the 50 calls.

1. **Calibration: fails.** The agents agree with the author's blind answers on 124 of 139 forms
   (89.2%; the rule asks 90%). The 15 differences:
   - 7 the author changed to the agents' answer on the second look (*bases* → *base*,
     *auspices*, *sceptics* → *sceptic*, *interspecies*, *idiosyncracies* → *idiosyncrasy*,
     *reveries* → *reverie*, *discuses* → *discuss*);
   - 5 junk tokens of one use each in the tail sample (*aeries*, *lalencies*,
     *onknownfrequencies*, *examineinconsistencies*, *propositionalindependencies*), kept by the
     author and singularised or corrected by the agents. The instructions differed here: the
     judging page carried the Opus judge's sentence on keeping junk, which the two Sonnet judges
     never had. Without them the agreement is 124 of 134 (92.5%); reported, not used, since the
     scoring was set before reading;
   - 3 differences the author kept: *ancients* (the agents merge), *fortes* (the agents give
     *forte*) and *preprocesses* (the agents give *preprocesse*).
2. **The hyphenated rule: passes**, 27 of 30. The three misses are two fragments with a leading
   hyphen (*-fluxes*, *-furnishes*) and one junk compound.
3. **The tail check: fails.** Against the author's final answers the voted map is right on 20 of
   the 30 drawn forms (the rule asks 27; blind also 20), spaCy on none and the agents on 25 (their
   misses are the five junk tokens).
4. **The calls** (reported, not a rule): the author took the agents' lemma on 45 of 50. All 14
   calls where the agents changed spaCy's lemma were accepted; of the 36 where they kept spaCy's
   merge, 5 were overridden: *blacks*, *whites*, *normals* and *hypnotics* stay apart from their
   mostly adjectival singulars, and *ss* stays (spaCy and the agents: *s*).

### Decisions after the results (the author, 2026-11-17)

- **The calibration counts as passed.** Five of the fifteen differences are the junk tokens on
  which the author and the agents had different instructions (above); on the other 134 forms the
  agents agree with the author on 124 (92.5%). The agents' verdicts are therefore accepted for
  the suspects and homographs, except where the author's answer on the 50 calls replaces them,
  and the registered fallback (the author reviewing the agents' changes) does not run. A follow-up
  review of 274 forms, briefly on the page, was withdrawn before any answer. Known limit: four of
  the author's five overrides on the calls kept a plural apart from a mostly adjectival singular
  (*blacks*, *whites*, *normals*, *hypnotics*); the agents' other merges of that shape stand as
  they decided.
- **The untested tail.** The agenda's consequence of a failed tail check is that more forms are
  reviewed (3.3). All 359 untested `disagree` forms (804 tokens, at most 10 uses each) are
  reviewed by the agents as above (two Sonnet judges, the Opus judge on their differences), now
  all with the sentence on keeping junk, since the agents were right on 25 of the 30 drawn and
  spaCy on none. Their contexts come from `lemma_review_list.py --tail`, drawn with their own
  seed so every other form's contexts are unchanged (checked: 2,436 of 2,436 identical).

## The consistency pass (set 2026-11-17, before it was applied)

The case-by-case answers on plurals whose singular is another part of speech disagree with each
other: of the 302 listed plural nouns whose singular is mostly an adjective, 45 were kept apart
(*others*, *goods*, *blacks*, *academics*, *normals*) and 257 merged (*humans*, *professionals*,
*metrics*, *officials*, *seconds*). The author chose one policy for all of them: **keep apart**.
Merging two words makes a word's neighbourhood depend on their mix, which can change across
decades and look like drift; keeping apart only splits some noun plurals from their singular
(*metrics*, *bounds*), which thins counts.

- **The rule.** A form read mostly as a plural noun that spaCy merges into a singular keeps its
  form when the singular's spelling has at least 20 uses in the kept text, fewer than half of
  them tagged as nouns (common or proper), and WordNet (spaCy's lemma index) gives it an
  adjective, verb or adverb reading. Hyphenated forms stay with the hyphenated rule.
- **Scope.** Every such plural in the kept text (`plural_merges.parquet`, written by
  `lemma_review_list.py`), not only the listed ones, so it also reaches forms no list caught
  (*novels*).
- **Precedence.** The override file, then this rule, then the author's and the agents' answers,
  then the hyphenated rule. The rule overrides earlier answers in its scope, the author's
  included; the results list every one it changes, and any override entry it disagrees with.
- **Left out.** Plurals whose singular is mostly a noun with another meaning (*logos*, *arms*):
  a part-of-speech rule cannot see them, so they stay as decided or are listed by hand. The other
  answers (spellings, names, junk, irregular plurals) stand.
- **How the rule was chosen.** Its tests were tried on the listed forms before this was written:
  the tag share alone took in ordinary nouns that spaCy often tags as adjectives in compounds
  (*astrocytes*, *mammals*, *neuropeptides*: 79 of 381), which the WordNet test removes. The
  thresholds (20 uses, half) are the review list's own.
- **Amended before use: adjectives only** (the author, 2026-11-17). Applied to the whole kept text,
  the rule as written kept 1,929 plurals: 1,531 whose singular is mostly an adjective, 299 mostly
  a verb (*estimates*, *gains*, *moves*, *thanks*), 63 numbers or determiners (*ones*,
  *thousands*) and 36 adverbs. A verb's plural noun is mostly the noun of the same word (*an
  estimate*, *to estimate*), so keeping it apart would split one word, not separate two, and a
  part-of-speech test cannot tell *thanks* from *estimates*. The rule therefore covers only
  singulars with at least half of their uses tagged as adjectives and an adjective reading in
  WordNet. Verb and number plurals keep their earlier answers (*thanks* and *ones* stay apart,
  *beings*, *estimates* and *thousands* merge), which stay judgement calls.
- **The review list's ties.** A rerun showed the list depended on the order the worker processes
  finished in when a form's lemmas tie (*edifies*: *edifie* 1, *edify* 1). Fixed in 3bef2a8f; two
  runs are now byte-identical. The judged list stays the first run's output (kept in
  `data/processed/lemma_review/`); the fixed script differs from it on 7 rare forms with tied
  lemmas and on 9 sampled forms near ties (7 of the tail sample, 1 hyphenated, 1 at the top-100
  boundary), all of which were judged either way. The rule reads the fixed run's
  `plural_merges.parquet`.

### The consistency pass: results

The adjective rule keeps 1,471 plurals apart. It overturns 1,331 earlier answers: 1,074 forms
(166,459 tokens) that spaCy merged and no list caught, such as *characteristics* (29,989),
*individuals* (29,604), *adolescents*, *females*, *males*, *standards* and *novels*; 235 agent
verdicts (26,063 tokens); and 22 of the author's answers (39,658 tokens): *humans*,
*professionals*, *metrics*, *officials*, *seconds*, *relatives*, *contemporaries*,
*intellectuals*, *universals*, *abstracts*, *majors*, *bilinguals*, *constants*, *primitives*,
*complexes*, *twins*, *fricatives*, *extremes*, *numerals*, *particulars*, *connectives* and
*minors*. It disagrees with no override entry. As expected, it also splits some noun plurals from
a singular mostly used attributively (*keys*, *tokens*, *firms*, *squares*).

### The tail and the exceptions file

- **The tail.** The two Sonnet judges agree on 349 of the 359 forms (97.2%); the Opus judge
  settled the other 10. The agents give spaCy's lemma for 39, the voted map's for 204, keep the
  form for 53 and give another lemma for 63.
- **The record.** Every judgement is in `curated/lemma_review_judgements.tsv`, one row per judged
  form (1,576): spaCy's and the voted map's lemmas, the agents' final lemma and the author's blind,
  final and call answers.
- **The exceptions file.** `curated/lemma_exceptions.tsv`, built by
  `scripts/evidence/lemma_exceptions.py`, holds 3,297 forms: 1,374 by the adjective rule, 1,091 by
  the hyphenated rule, 538 by the agents, 181 from the override file (all but *goes*), 104 by the
  author's final answers and 9 by the author's calls; 1,816 keep their form. Each entry names the
  reading it applies to (`NNS` or `VBZ` as judged, empty for the override file's entries), so
  *thanks* as a verb still merges. No target word is mapped to another word, and no exception
  merges a form into a target word. The check on spaCy's own lemmas of the target words runs on
  the rebuilt view (3.6).
- **The target words on the cache (3.6, `scripts/evidence/target_lemmas.py`, 2 min 34 s).**
  Over the kept text, no token of a target's own spelling is written as another word with the
  exceptions (0), so the contingency of round L1 (`docs/predictions_lemmas.md`) does not apply.
  One target changes: *medium* falls from 66,287 tokens to 8,684, as the 57,603 uses of *media*
  stop counting as it; 87% of what the view counted as *medium* was *media*. The other 31
  single-word targets keep the same tokens and the same forms.
- **Amended 2026-11-17 (3.6).** The override file's entries now apply to plural nouns only
  (`NNS`): every override form is read mostly as one (*goes*, the only verb, is not carried), and
  with an empty tag *means* would have stayed unmerged as a verb too ("this means that").
