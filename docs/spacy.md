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
type is read in context, and a type merges only when spaCy mostly reads it as a plural noun or a
third-person verb of one lemma (`corpus lemmas votes`, then `corpus lemmas learn`).

The general rule: use spaCy on sentences, not on bare words.

## Where it fits

| Feature | Job | Replaces or adds |
|---|---|---|
| Sentence segmentation (parser or `senter`) | Usage contexts in the store; shared-sentence detection | The regex sentence splitters in `corpus/quality.py` and `drift/senses.py`, which break on *e.g.*, *et al.* and *Fig. 2* |
| Part of speech and morphology (tagger) | Lemma votes; a usage feature for counting senses | Adds a check on anachronisms: *prompt* read as a verb cannot carry the language-model sense |
| Lemmas in context | Finding target words in the store | `TARGET_VARIANTS` in `corpus/preprocessing.py` and `word_pattern` in `drift/senses.py` |
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

## When

1. **Store rebuild:** sentences, part of speech and morphology, lemmas in context, named entities,
   `PhraseMatcher` for targets, lingua as a component, span groups with `DocBin`, the GPU path.
2. **Counting senses:** `Matcher` and `DependencyMatcher` cues, noun chunks.
3. **Later experiments:** dependency collocations for detection, trainable classifiers, the entity
   linker.

## Considered and left out

- **Weasel** (spaCy's workflow tool, formerly spaCy projects) overlaps with the content-hash
  manifest and would add a workflow layer the project does not need.
- **Out-of-context lemmatisation** of bare word types, for the reason above.

## Sources

From spaCy's documentation, read 2026-11-11.

- Lemmatizer: <https://spacy.io/api/lemmatizer>; trained pipeline design and the lemmatizer's
  dependence on the tagger: <https://spacy.io/models>.
- Rule-based matching (`Matcher`, `PhraseMatcher`, `DependencyMatcher`):
  <https://spacy.io/usage/rule-based-matching>.
- Processing pipelines (`nlp.pipe`, multiprocessing, disabling components):
  <https://spacy.io/usage/processing-pipelines>.
- Saving and loading (`DocBin`): <https://spacy.io/usage/saving-loading>.
