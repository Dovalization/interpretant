# The held-out set (agenda 6.2)

Fixed 2026-11-19, before the model comparison, the matching audit and any measurement. **Not for
use until the confirmation plan is committed.** Development outputs never name these words: the
word table marks them `held_out: yes`, committed outputs show `Word.reported_name` (a hash), and
the tests check the set by its hash.

## How it was drafted

- **Candidates.** Two agents drafted 80 candidates (48 changes, 32 stable words) from the project's
  fields, single words only, none in the 362-word exclusion set: the word table, every control
  list, the discovery long list, the sense lists' words, the DWUG and SemEval-2020 English targets
  and the kaikki extracts' words. Each change carries its old and new sense, the decade its new
  sense entered these fields' research literature, and its dating sources (Wiktionary senses and
  dates, and the paper or event that introduced the sense). Each stable word carries why it is
  stable. No change score was computed and no usage was read.
- **Corpus counts per decade** (`scripts/evidence/held_out_counts.py`): each candidate's kept
  abstracts and static-view tokens per decade, written outside the repository.
- **The rule of 50** (the author, 2026-11-19). A change needs at least 50 abstracts before its date
  and 50 from it on, so the comparison has uses on both sides. The 1950s–60s corpus is about 13
  times smaller than the 2020s, so early changes rarely pass; a second draft of 14 changes arriving
  in the 1980s or later topped the set up. The kept changes therefore lean to the 1980s onward.
- **Claude's recommendations** dropped candidates that failed the rule, whose old sense would
  swamp the new one, with low confidence, verbs among the stable words (no change is a verb), and
  stable words with a possible drift or a frequency far from the changes'.
- **The author's review** (2026-11-19, on a review page): all recommendations accepted, except two
  changes kept against them. Their new senses appear beside development words (*network*,
  *hallucination*), but no score or sense list was ever made for them; the cost is some
  correlation with those development cases.

## The set

- **24 documented changes,** all judged documented by the author, and **22 stable words.** Every
  word is matched by lemma; nouns accept PROPN, as in the rest of the table.
- **The list's sha256** (the sorted lemmas, one per line, each followed by a newline):
  `18dfd3a5fcf3c08289d11e62e5252e9e47af5827de3cb78312ce3601b23aa1b6`, checked by
  `tests/test_words.py`.
- **The evidence** (`curated/held_out_evidence.jsonl`: senses, dates, sources, the author's label,
  abstracts per decade) has sha256
  `12db0d6409d227f1153e20bf8103256561343f9c21645d8216d9f1a891bb183c`.
- **Leakage checklist,** for the confirmation plan (after Kapoor and Narayanan 2023): fixed before
  any measurement of these words; drawn from words used nowhere in development; filtered from
  development outputs by their reported name; their labels, if any, are made after the plan is
  frozen.
