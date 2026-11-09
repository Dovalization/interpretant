# Methods

What the pipeline computes, with the settings in force since November 2026. The corpus itself
is described in [`corpus.md`](corpus.md); the results and their tests are in
[`predictions.md`](predictions.md) and [`predictions_cross_perplexity.md`](predictions_cross_perplexity.md).

## Text

`preprocess_text` (`corpus/preprocessing.py`) applies NFKC normalisation, lowercasing, removal
of the soft-hyphen artifact, transliteration to ASCII, removal of LaTeX spans, URLs and DOIs,
replacement of non-alphanumeric characters (except in-word hyphens) with spaces, and whitespace
collapsing. Plurals and British spellings of the target words are mapped onto one lemma
(`TARGET_VARIANTS`). There is no stop-word removal: skip-gram's frequent-word subsampling does
that job without changing the context windows.

`corpus lemmas learn|apply` (`corpus/lemmas.py`) then normalises the rest of the vocabulary at
the type level, and only in two ways: regular plurals collapse onto their singular (simplemma's
lemma, accepted only when the token is a regular plural of it), and British spellings onto
American ones (rules gated on simplemma's lexicon, an exception list and the American form's
own corpus frequency). Verbs, participles and *-ing* nominalisations stay as they are, because
a general lemmatiser maps *learning* to *learn* and *embedding* to *embed*. The map is written
to `data/processed/lemmas.tsv` for review; the hand-kept part is `data/lemma_overrides.tsv`.
Round 4 of the predictions tests which findings survive this change.

`corpus phrases learn` fits one gensim `Phrases` model (normalised PMI ≥ 0.5, minimum count 20)
on a balanced sample of 300,000 tokens from every cell, so the large modern cells do not decide
the phrases of the 1850s; `corpus phrases apply` joins the learned phrases with `_` in every
cell. The phrase targets in `data/words_of_interest.txt` are always merged, and no learned phrase
absorbs a single-word target.

## Embeddings: one model per cell, one space

Word vectors trained separately share no coordinate system. TWEC (Di Carlo, Bianchi and
Palmonari, 2019) trains a compass first and then trains each slice with the compass's context
vectors frozen, so every slice's target vectors land in the compass's space and need no
alignment. `twec grid` builds the compass from the same token budget from every main cell
(200,000 tokens; `books` and `openalex`), then trains every cell, including the full cells, with
100 dimensions, a 5-word window, 10 epochs, 5 negative samples and a minimum count of 5 in the
compass vocabulary. Lines are split into sentences of at most 10,000 tokens before training,
because gensim's compiled loop silently stops reading a sentence there and the books are stored
one per line (`embedding/base.py`). The second seed (43) repeats the whole run.

`ppmi grid` is the count-based check (Church, 2017: word2vec approximates a PMI factorisation;
Hamilton, Leskovec and Jurafsky, 2016: SVD is more stable on small corpora). It uses the tuned
settings of Levy, Goldberg and Dagan (2015) so the comparison is fair: the same 5-word window,
context-distribution smoothing with α = 0.75, no PMI shift, and eigenvalue weighting of 0.5 on a
truncated SVD.

## Drift against a corpus-wide noise floor

For a pair of cells (the same field in consecutive decades, or two fields in one decade), a word
is *measured* only if it has at least 150 occurrences in both (`MEASURE_MIN`); its neighbours are
drawn from words with at least 50 occurrences in both (`NEIGHBOUR_MIN`), so a change of
vocabulary cannot pose as a change of neighbourhood. Two measures per word:

- **cosine drift**: 1 minus the cosine between the word's vectors in the two cells;
- **neighbourhood shift**: the Jaccard distance between its 25 nearest neighbours in each cell.

Retraining a cell with a new seed moves an ordinary word's neighbours by a median Jaccard
distance of about 0.4, against about 0.6 for a change of decade, so raw values are not read. In
each pair the measured words are grouped into equal-sized frequency bins (at least 100 words per
bin, at most 10) by the geometric mean of their two counts, and each word gets its percentile
rank within its bin on both measures. A word is a **candidate** only above the 95th percentile on
both. Everyday control words (`data/control_words.txt`, 38 words; 15 cross-field controls) check
the floor and are flagged in under 1% of measurements, except *food*, which genuinely differs
between psychology, sociology and neuroscience. `drift grid` writes one row per word and pair to
`data/drift/within_field.parquet` and `cross_field.parquet`.

A candidate becomes a finding only after the robustness checks: the second seed, PPMI-SVD, and a
prediction committed before the evidence (below).

## Cross-perplexity: who predicts whom

`drift perplexity` fits one bigram language model with interpolated Kneser-Ney smoothing
(discount 0.75) per field and decade on the main OpenAlex grid, and scores every field's
held-out abstracts under every field's model. The **excess cross-entropy** E(A → B) is the
cross-entropy, in bits per token, of B's held-out text under A's model minus its cross-entropy
under B's own model; low E(A → B) means B's language is predictable from A's (Vilhena et al.
2014, "cultural holes"). The corrected design (`--drop-shared --fixed-budget 200000`,
seeds 1–5) removes any abstract that appears in more than one cell of a decade, trains every
model on exactly 200,000 tokens, holds out 1,000 abstracts per cell, uses one vocabulary per
decade (words with at least 5 occurrences in that decade's training samples), and repeats the
whole run with five seeds; a comparison holds only if it goes the same way in all five.

## Pre-registration

Every analysis is scored against predictions written and committed before its evidence
existed. The predictions name the comparison, the threshold and what falsifies them; a scorer
script is committed next, run once, and its verdicts are appended beneath the predictions,
which are never edited afterwards. A prediction that fails stays failed in the record; a new
reading is written as a new section with new predictions. Rounds so far: the trained models
(round 1), a second seed with decades and field pairs the first round never looked at
(round 2), PPMI-SVD (round 3), and four rounds on cross-perplexity (C, D, F, H), including one
whose published interpretation was reversed by its own follow-up.

## Why "interpretant"

In Peirce's theory of signs, the interpretant is neither the word nor the thing it points to;
it is the meaning the sign produces in a community of minds: the further sign, the mental
response, the understanding that arises when a community encounters a representation. Peirce
distinguishes three registers:

1. **Immediate**: what a sign is interpretable as, prior to any specific context.
2. **Dynamic**: the actual effect it produces on a mind at a given moment.
3. **Final**: the ideal effect that would be reached at the convergent end of unlimited inquiry.

A cell-trained word vector corresponds to none of these directly, but corpus statistics across
a community and a period capture the aggregate of dynamic interpretant events, and that
aggregate is what the pipeline measures. The vectors are social facts, not semantic facts: they
record what a community did with a sign, which words it appeared near, which arguments it
enabled, which conceptual neighbours it acquired.

*Interpretant drift* is therefore the precise description of the measurement. The standard
framing of semantic drift is descriptive: the word moved. It does not say what changed, the
word, the community or the relation between them. Interpretant drift treats meaning as a
property of a community's relationship to signs: the community changed what it did with the
sign, and the vector displacement encodes that change. *Consciousness* did not drift;
philosophers stopped reading Husserl and started reading Crick. The word is the trace; the
community is the event. That is also why the interpretant of *intelligence* in philosophy and
in computer science are not two measurements of one thing but two things, and why the corpus is
split by field on theoretical, not only methodological, grounds.

Infinite semiosis, Peirce's observation that the interpretant of a sign is itself a sign that
produces a further interpretant, indefinitely (Eco's "unlimited semiosis"), frames what the
pipeline makes legible: a term absorbs new associations, sheds old ones, gets borrowed by
adjacent fields that use it in their own sign relations. Each cell's vector is a moment in a
process that has no principled end.

## References

- Church, K. W. (2017). Word2Vec. *Natural Language Engineering*, 23(1).
- Di Carlo, V., Bianchi, F., & Palmonari, M. (2019). Training temporal word embeddings with a
  compass. *AAAI*.
- Giulianelli, M., Del Tredici, M., & Fernández, R. (2020). Analysing lexical semantic change
  with contextualised word representations. *ACL*.
- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal
  statistical laws of semantic change. *ACL*.
- Levy, O., Goldberg, Y., & Dagan, I. (2015). Improving distributional similarity with lessons
  learned from word embeddings. *TACL*, 3.
- Vilhena, D. A., et al. (2014). Finding cultural holes: how structure and culture diverge in
  networks of scholarly communication. *Sociological Science*, 1.
