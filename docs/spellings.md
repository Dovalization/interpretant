# The spelling map (agenda 3.4; `docs/preprocessing_phase.md`, step 2)

Pre-registered 2026-11-17, before any pair of the sample or the census was judged.

## What the map is

Spelling variants of one word (*behaviour* and *behavior*, *analyse* and *analyze*, *centre* and
*center*) split one word's counts and give it separate, noisier vectors (Nguyen and Grieve 2020),
and the share of British and American spellings drifts across decades (Gonçalves et al. 2018). The
map writes each variant as one standard spelling, in the static view and later in target
matching. It applies to each hyphen segment of a folded token, one token for one
(`corpus/spelling.py`).

- **Sources.** spaCy's English spelling table (`lexeme_norm`, frozen inside en_core_web_sm 3.8.0)
  is probed with every token type of the static view R_B (688,190 types and hyphen segments).
  - Its pairs are filtered: white space or no letter; informal norms (*cos* → *because*); pairs
    under a character ratio of 0.6; and pairs known to merge different words (*analyses* →
    *analyzes*, *buses* → *busses*, *cheques* → *checks*, *storey* → *story*).
  - The project's rules (`corpus.lemmas.american`) add the derived words the table lacks
    (*optimisation*, *categorisation*). Portuguese *analise* is left out.
- **Direction** (the author, 2026-11-17). Each pair is written as the form the corpus uses more
  often, so *dialogue* stays (11,031 against 1,383 for *dialog*) while *behaviour* becomes
  *behavior*. A word-table lemma keeps its own spelling.
- **The proposal** (`scripts/evidence/spelling_table.py`, code `8e8e6522`): 1,954 forms
  (1,116 from the table, 838 from the rules), 172,600 tokens, 0.17% of the view. The British form
  won 116 pairs. No word-table lemma is mapped away; the only pair touching the table is
  *behaviour* → *behavior*.
- **Corrected before judging (2026-11-17).** A first build let a British form used once win over
  an American form that never occurs, mapping the absent form onto it (*civilizes* → *civilises*,
  57 such pairs). The British form now wins only over an American form that occurs and is rarer.
  Pairs of a plural noun in *-yses* and a verb in *-yzes* (*psychoanalyses*) are dropped as
  different words.

## The audit, set before reading

- **What is judged.** Whether a pair's two forms are spellings of one English word with one
  meaning, as the corpus uses them. A pair is **wrong** if:
  - the two are different words or meanings in this corpus;
  - one is not English, or is a misspelling of a third word;
  - or the standard is not a real spelling.

  The direction is the rule above and is not judged.
- **The sample.** 150 pairs, seed 20261119: 75 drawn with probability proportional to tokens and
  75 uniform over the other pairs. The map passes if at most 3 are wrong. Passing chances are:
  - 94% at a true error rate of 1% (the acceptable level);
  - 65% at 2%;
  - 5% at 5% (the rate that must fail).
- **The census, fixed one by one.** The census's errors are fixed one by one and do not count
  against the bar. It covers 163 pairs: the one pair touching the word table, the 49 others with
  most tokens, and the 113 others whose British form won.
- **The judges.**
  - Two Claude Sonnet judges and a Claude Opus adjudicator for their differences, with the prompt
    saved in `curated/judging/spellings/prompt.md`.
  - Each pair is shown with up to two corpus sentences for each form, from kept works.
  - The author judges 30 pairs drawn from the sample (seed 20261121), blind. The agents' verdicts
    stand if they agree with the author on at least 27 of the 30; otherwise the author judges the
    150.
  - The items, which quote corpus sentences, stay out of the public repository. The key
    (`data/review/spellings/key.json`) and the verdicts are kept in `data/review/spellings/`.
- **If the sample fails,** the agents review every pair of the failing source (table or rules),
  its errors are fixed, and a new sample (seed 20261120) is judged once against the same bar.
- **Then.** `curated/spellings.tsv` is written from the proposal, minus the pairs judged wrong,
  and the check that no word-table lemma is mapped away runs on it.

## Results

To be written after judging.
