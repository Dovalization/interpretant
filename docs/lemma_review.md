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
- **No judging**: the override file carries into the exceptions file, except *goes*, which 3.2
  decides later (ordinary verbs merge: *goes* → *go*). *does* stays: outside auxiliary use spaCy
  reads it as a plural noun (*doe*).

## Rules, set before reading

1. **Calibration.** The agents' final lemma is compared with the author's on the author's 142
   forms (keeping the form counts as a lemma). If they agree on at least 90%, the agents'
   verdicts are accepted for the suspects and homographs. If not, the author reviews the agents'
   changes to spaCy's lemma on those forms, most frequent first.
2. **The hyphenated rule.** If the agents' final lemma equals the rejoined segment on at least 27
   of the 30 drawn forms, the rule applies to all 1,091; otherwise more are drawn.
3. **Targets.** After the exceptions, no target word may be mapped to another word.

Approved corrections go into `curated/lemma_exceptions.tsv` with the carried overrides (3.2).

## Results

To be written after judging.
