# The lemma review (agenda 3.3)

Pre-registered 2026-11-17, before any form was judged. The list comes from
`scripts/evidence/lemma_review_list.py` (commit 1e8bfb2d) over the kept text of the build of record
75f36a82: 9,677,037 plural-noun and third-person-verb tokens, auxiliaries aside.

## What the list holds

The static view writes spaCy's in-context lemma (3.2), so the review looks for spaCy's errors,
with the voted lemma map as the tool that finds them. Forms with at least 20 uses unless stated.

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
  *datum* for *data* (139,839 uses) and *medium* for the target *media* (57,608). The override file
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

Neither failure's consequence fully fit what the results showed, so the author chose:

- **Follow-up review.** The registered consequence of a failed calibration is that the author
  reviews the agents' other changes to spaCy's lemma, most frequent first. The calls found no
  error among the agents' changes and five among their merges, four of them plurals merged into a
  mostly adjectival singular. The review therefore covers every other suspect or homograph with
  at least 50 uses where the agents either changed spaCy's lemma (115 forms) or merged a plural
  into a singular whose adjective share is at least 0.5 (159), most frequent first, on the judging
  page with the agents' lemma suggested. The author's answers replace the agents'; the agents'
  verdicts stand for the forms with fewer uses.
- **The untested tail.** No consequence was registered for a failed tail check. The 359 untested
  `disagree` forms (804 tokens, at most 10 uses each) are judged by the agents as above (two
  Sonnet judges, the Opus judge on their differences), now all with the sentence on keeping junk,
  since the agents were right on 25 of the 30 drawn and spaCy on none.

To be completed: the follow-up review's figures, the tail's verdicts and the exceptions file.
