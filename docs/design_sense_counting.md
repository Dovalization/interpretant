# Design draft: counting the senses

*Draft, 2026-11-12, for review. It proposes how the senses of each usage are counted on the
rebuilt store, from what `docs/methodology_comparison.md` records; generative language models
stay a measured comparison, not the instrument.*

## Where counting stands

- **Finding the senses is open; hand-written lists are the best so far.** They scored 0.69–0.80
  on the human benchmark (rounds Y and Q), above the configurations of automatic lists tried
  (substitutes and neighbours 0.39–0.59, WordNet about 0.18 lower); a Wiktionary check found two
  gaps in four words (generic *transformer*, network *cloud*). Clustering failed in one design
  only (SciBERT, KMeans, silhouette); other designs are untried (`docs/decision_agenda.md`,
  step 7a).
- **Counting is the weak step.** The cue classifier (cue words label the easy usages, a classifier
  learns the rest) agrees 84–88% with the development labels, but labelled 24% of AI's 2000s
  *prompt* usages with the language-model sense, which did not exist yet (Sonnet: 3%). Cue words
  exist only in some eras, so the classifier learns the era along with the sense.
- **Corrected counts** have been used since round A, but their error rates come from the easy,
  cue-labelled usages, which understates errors on the hard ones.

## Proposal: three counters on the same usages

1. **The current cue classifier**, as the baseline.
2. **Cues as spaCy patterns.** The cue syntax of `drift/classifier.py` (`<word`, `>>word`,
   `!word` in a five-word window) rewritten as `Matcher` and `DependencyMatcher` patterns that can
   require a part of speech and a syntactic relation: *prompt* as the object of *design* or
   *engineer*, not merely near them. A usage spaCy reads as a verb never gets a noun sense, which
   removes the *prompt* anachronism at its source.
3. **A definition reader.** Each sense in the hand-written lists already has a definition; a model
   scores how well a usage fits each definition and picks the best, with no cue words at all.
   Candidates, cheapest first: the usage's XL-LEXEME vector against each definition's vector
   (nearest definition); a gloss-selection model of the GlossReader kind. Both are encoders, not
   generative models.

Each counter labels the same usages, so their disagreements show where each fails.

## How they are judged

| Check | What it catches |
|---|---|
| The human benchmark (DWUG EN), graded change and sense groups | Agreement with people |
| The author's blind judgements (100, plus the 48 disagreements of the head-to-head round) | Labelling accuracy on this corpus |
| Anachronisms: a sense's share before it existed (*prompt*'s language-model sense before 2020) | Era learned in place of sense |
| Planted changes without cue words | Whether counting depends on cues |
| A few hand labels across eras (about 30 per word, spread over decades) | Old senses with few examples |
| Sonnet and JEV on a sample | The comparison arm, as before |

The corrected counts are re-estimated with error rates from the hand labels, not from the
cue-labelled usages.

## Order

1. Rewrite the cues as spaCy patterns (counter 2) on the store's annotated documents, and measure
   the *prompt* anachronism against the cue classifier.
2. The nearest-definition reader (counter 3, first candidate), on the same usages.
3. The author's hand labels across eras, about 30 per word, which every counter is scored on.
4. The gloss-selection model only if the nearest-definition reader falls short.

## Decisions needed

1. Which words are counted first (the four documented changes, *prompt*, *hallucination*).
2. Whether you can label about 30 usages per word across eras, and how they are sampled.
3. Whether a definition reader may be fine-tuned on public gloss data, or must be used as it is.
