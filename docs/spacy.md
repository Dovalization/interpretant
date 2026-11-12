# spaCy in this project

*Recorded 2026-11-11, when spaCy joined the stack (README, Stack). This page lists where its
features fit the project's jobs (`docs/methodology_comparison.md`), what each would replace, and
when it comes in. It is a plan, checked against spaCy's documentation, not a record of results.*

## The lesson that shaped it

spaCy works in steps: the tagger reads each word in its sentence and assigns its part of speech
and morphology (singular or plural, verb form), then the rule lemmatizer applies suffix rules for
that kind of word and leaves base forms alone. Used out of context, with the tagger skipped, the
lemmatizer has to guess, and it guesses badly: on the corpus vocabulary it read every word ending
in *s* as a possible plural noun (*loss* → *los*, *discuss* → *discus*, *aesthetics* →
*aesthetic*) and missed third-person verbs (*uses*, *identifies*). The type-level lemma map
therefore takes its merges from votes: one spaCy pass over the raw abstracts counts how each word
type is read in context (`corpus lemmas votes`, then `corpus lemmas learn`).

Even in context, three things needed rules of the project's own:

- **spaCy's tags decide whether a word merges, not its lemmas.** Its lemmas of *-es* words are
  unreliable (*uses* → *us*, *senses* → *sens*, *taxes* → *taxis*); spaCy's own source calls the
  ordering of candidate lemmas "dodgy ... until we get frequencies". English spelling limits the
  possible singulars (*-es* only after *s*, *x*, *z*, *ch*, *sh* or *o*), and among them the one
  read as the matching part of speech wins, then the most frequent.
- **Irregular plurals come from spaCy's exception table**, not its lemmas in context: a plural
  noun also takes the singulars the lemmatizer lists for it (*mice*, *wives*, *shelves* →
  *shelf*, *testes* → *testis*), and -men takes -man when spaCy reads the -man form as a common
  noun (*freshmen*, not *specimen*; added 2026-11-12 after an audit of the map).
- **Proper-noun readings are left out** (*United States*, *Academy of Sciences*).
- **Plurals that are words of their own** (*aesthetics*, *hermeneutics*, *odds*, *arrears*) are
  kept by `data/lemma_overrides.tsv`, an exception list in the way spaCy's own rule lemmatizer
  takes WordNet's. The 135 added on 2026-11-11 were found in the votes (merged as plurals,
  their own WordNet entry, a singular mostly read as an adjective) and reviewed by hand.

The general rule: use spaCy on sentences, not on bare words, and keep a reviewed list for the
words no rule can tell apart.

## Where it fits

| Feature | Job | Replaces or adds |
|---|---|---|
| Tokenizer with special cases and infix rules, `norm_`, token flags | Cleaning: the processed text written from the same pass as the votes and the store (below) | The regex cleaning of `preprocess_text` in `corpus/preprocessing.py` |
| Sentence segmentation (parser or `senter`) | Usage contexts in the store; shared-sentence detection | The regex sentence splitters in `corpus/quality.py` and `drift/senses.py`, which break on *e.g.*, *et al.* and *Fig. 2* |
| Part of speech and morphology (tagger) | Lemma votes; a usage feature for counting senses | Adds a check on anachronisms: *prompt* read as a verb cannot carry the language-model sense |
| Lemmas in context | Finding target words in the store | `TARGET_VARIANTS` in `corpus/preprocessing.py` and `word_pattern` in `drift/senses.py` |
| `AttributeRuler` (patterns that set `LEMMA`) | Lemma exceptions in the store's pipeline: it loads `data/lemma_overrides.tsv`, so the store's lemmas in context follow the same reviewed list as the type-level map | spaCy's lemmas left as they are, which would merge *aesthetics* into *aesthetic* again |
| `Matcher` (patterns on lemma, part of speech, morphology) | Cue words for counting senses; boilerplate patterns | The cue syntax of `drift/classifier.py` (`<word`, `>>word`, `!word` in a five-word window) |
| `PhraseMatcher` | Phrase targets (*neural network*, *social network*) | Phrase handling that rests on the grid's gensim phrase model |
| `DependencyMatcher` (needs the parser) | Syntactic cues: *prompt* as the object of *design* or *engineer*, *attention* modified by *self-* or *multi-head* | Cues by proximity, which cannot tell syntax from neighbourhood |
| Dependency collocations | Detecting change: which verbs take the word as object, which modifiers it takes, compared across periods | A cleaner form of the collocations method, not yet tried |
| Named entities | Flagging proper-name uses (a product called Transformer, a person called Williams, company names with *Cloud*) | Adds a filter; these now enter the usages unseen |
| Noun chunks | Compound terms around a target (*attention mechanism*, *cloud computing*) as evidence for senses and list gaps | Complements the Wiktionary check on written lists |
| Span groups, custom attributes, `DocBin` | Each target occurrence stored as a labelled span; annotated documents saved once and reused by every round | Start and end offsets in the store's records |
| `nlp.pipe`, `prefer_gpu`, the transformer model | Throughput on the CPU; accuracy on the homelab GPU | — |
| lingua as a custom component | Language checks inside the same pipeline | A separate pass |
| `textcat`, `spancat` | An experiment for counting senses with a classifier on the transformer model | An alternative to the scikit-learn cue classifier |
| Entity linker | Possibly: link each usage to a sense entry with a definition, close to the definition-reading model | Needs training data |

## Cleaning text with spaCy

*Checked 2026-11-12 against the documentation and on a sample of academic prose.*

The audit of the lemma map traced two faults to `preprocess_text`, the regex cleaner every
processed file goes through:

- **Curly apostrophes fuse words.** A straight apostrophe becomes a space (*Kant's* → *kant s*);
  a curly one is deleted by the ASCII fold (*Kant’s* → *kants*). The abstracts hold 537,558
  curly apostrophes against 722,671 straight ones, which leave 542,162 fused tokens in 36,665
  types: possessives that look like plurals (*kants*, *platos*), clashes with real words
  (*one’s* → *ones*, *God’s* → *gods*, *people’s* → *peoples*, *it’s* → *its*) and junk types
  (*dont*, *doesnt*).
- **Two tokenisations disagree.** The votes come from spaCy's tokens, the type counts from the
  regex's. spaCy splits words at hyphens and the regex keeps them whole, so none of the 47,670
  hyphenated types ending in *s* (*trade-offs*, *self-reports*) ever received a vote; hyphenated
  types are 1.2% of tokens.

What spaCy offers instead:

- **The tokenizer** applies special cases first, then prefixes and suffixes, then infixes (such
  as hyphens), and `nlp.tokenizer.explain(text)` reports which rule made each token. It is
  non-destructive and splits `’s` and `n’t` whatever the apostrophe.
- **Abbreviations.** Only some are built in (*e.g.*, *vs.*); *cf.*, *Fig.*, *Sect.*, *Eq.*,
  *et al.* and *approx.* are not, and their full stops end sentences. Added as special cases,
  they took the parser from seven wrong sentences on the sample to the right four.
- **Hyphens.** One documented infix rule splits a hyphen between letters; without it,
  *self-organising*, *trade-offs* and *mother-in-law* stay single tokens and are tagged whole.
- **`norm_`.** The documentation describes norms set by tokenizer exceptions (*n't* → *not*).
  The trained pipeline also ships a `lexeme_norm` table from `spacy-lookups-data`, not
  described in the API pages: 3,510 entries, mostly British to American spellings
  (*characterise* → *characterize*, *behavioural*, *labelled*, *programme*), plus some
  informal ones (*cos* → *because*). It lacks most derived words (*personalisation*,
  *operationalise*) and *judgement*, so the map's own spelling rules stay as a backstop, and the
  table is read as data and checked, not relied on as an interface.
- **Flags** for what to drop: `is_punct`, `is_quote`, `is_bracket`, `is_currency`, `like_url`,
  `like_email`, `like_num`, `is_alpha`, `is_ascii`, `shape_`. `like_num` is also true for
  number words (*one*), so it cannot drop numbers alone.
- **Sentences.** The parser is the most accurate; `senter` is about ten times faster, ships
  disabled, and matched the parser on the sample once the special cases were in; the rule-based
  `sentencizer` split the sample into eleven pieces; a custom component can set
  `is_sent_start` before the parser.

What stays outside spaCy: Unicode normalisation (NFKC) before tokenising, folding to ASCII
after it, stripping LaTeX math and DOIs without a scheme, language identification (lingua) and
near-duplicates (datasketch).

The cleaning step this points to, for the store rebuild: normalise and strip math; run spaCy
with the abbreviation special cases, no hyphen infix, `senter`, the tagger, the lemmatizer and
the `AttributeRuler` with the override file; drop punctuation, URLs, e-mail addresses and the
possessive *'s* (its tag, POS, tells it from *it's*); write each remaining token's `norm_`,
lowercased and folded to ASCII. The votes, the type counts and the lemma map then share one
tokenisation.

## When

1. **Store rebuild:** cleaning (above), sentences, part of speech and morphology, lemmas in
   context with the override list in an `AttributeRuler`, named entities, `PhraseMatcher` for
   targets, lingua as a component, span groups with `DocBin`, the GPU path.
2. **Counting senses:** `Matcher` and `DependencyMatcher` cues, noun chunks.
3. **Later experiments:** dependency collocations for detection, trainable classifiers, the entity
   linker.

## Considered and left out

- **Weasel** (spaCy's workflow tool, formerly spaCy projects) overlaps with the content-hash
  manifest and would add a workflow layer the project does not need.
- **Out-of-context lemmatisation** of bare word types, for the reason above.

## Sources

From spaCy's documentation, read 2026-11-11.

- Lemmatizer, including the source of its rule lemmatization: <https://spacy.io/api/lemmatizer>;
  trained pipeline design and the lemmatizer's dependence on the tagger:
  <https://spacy.io/models>; exception lists and the trainable lemmatizer:
  <https://spacy.io/usage/linguistic-features>.
- `AttributeRuler`, and the move of lemma exceptions to it in spaCy 3:
  <https://spacy.io/api/attributeruler>, <https://spacy.io/usage/v3>.
- Rule-based matching (`Matcher`, `PhraseMatcher`, `DependencyMatcher`):
  <https://spacy.io/usage/rule-based-matching>.
- Processing pipelines (`nlp.pipe`, multiprocessing, disabling components):
  <https://spacy.io/usage/processing-pipelines>.
- Saving and loading (`DocBin`): <https://spacy.io/usage/saving-loading>.

Read 2026-11-12, for cleaning:

- The tokenizer's algorithm, special cases, infix rules, `Tokenizer.explain`, the retokenizer
  and the four ways to set sentence boundaries:
  <https://spacy.io/usage/linguistic-features>; `Tokenizer.add_special_case`:
  <https://spacy.io/api/tokenizer>.
- Token attributes (`norm_`, the `is_*` and `like_*` flags): <https://spacy.io/api/token>.
- `senter` against the parser: <https://spacy.io/models>.
