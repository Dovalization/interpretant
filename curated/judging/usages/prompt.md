You are checking how a program finds the uses of a list of target words in English research
abstracts (1950s to 2020s; fields such as artificial intelligence, human-computer interaction,
computer science, cognitive psychology, linguistics, neuroscience, communication, sociology,
philosophy and history of science). Each target word has the parts of speech it is studied in (for
example *medium* as a noun, not the adjective of *medium effect size*). A use counts in any
inflection or spelling (*networks*, *behaviour* for *behavior*), and as a part of a hyphenated
word (*self-attention*, *attention-based*). A use inside a name does not count (*network* in *Cable
News Network*, *attention* in the title of a named test or scale). Mentioning the word as a word
(*the term "attention"*) counts as a use.

These rules settle the common hard cases:

- **Titles.** The title of another work (a paper, book or chapter cited, listed with its authors,
  or named in a reference list) is a name, so a word in it is not a use. The abstract's own
  title, its section headings and table-of-contents headings are ordinary uses: the authors are
  using the word.
- **-ing forms.** For a word studied as a noun, an -ing form built on it counts only where it works
  as a noun or noun modifier (*social networking sites*, *data mining*), not where it is a verb
  (*they were networking*).
- **Noun modifiers.** A noun before another noun is still a noun use (*network analysis*,
  *component failure*), even if tagged as an adjective.
- **Coined names.** The name of one specific system, model, scale, inventory, organisation, law or
  product is a name (*BERT*, a named test or scale). A generic term or technique is not, even when
  capitalised or spelled out from an acronym (*Artificial Intelligence*, *Generative Adversarial
  Network*).
- **Identifiers.** File names, code identifiers and URLs are not uses (`other_word`).

Each item shows the target word, its parts of speech, the sentence, and the marked place between
⟨t⟩ and ⟨/t⟩. There are three kinds of item.

**kept** (the program kept this place as a use):
- right: the marked place is a use of the target word, in one of its parts of speech (any part of
  speech when it is a part of a hyphenated word), outside a name, and the marks cover exactly that
  word or part.
- wrong, with the reason:
  - `other_word`: not the target word at all: an abbreviation or letter code with the same letters,
    a piece of a longer word, or OCR damage that only looks like the word. Every sense of the
    target word is the same word and counts (the study is about how senses change);
  - `part_of_speech`: the target word, in a part of speech it is not studied in (a verb where only
    the noun is studied);
  - `name`: inside the name of a person, organisation, place, product, event, law, named test or
    scale, or a title;
  - `span`: the marks cut the word or take in letters of another word;
  - `not_english`: the sentence is not English.

**left out** (the program found the word here but left it out, for the reason shown: its part of
speech, or the name it sits in):
- right: leaving it out was correct: the word here is in a part of speech it is not studied in, or
  it is inside a name.
- wrong: it is a use the program should have kept (the tag or the name is a mistake).

**missed** (a plain text search found the word's letters here, and the program did not):
- right: it is a use of the target word that the program should have kept, as for **kept**.
- wrong: it is not (another word containing the letters, such as *attentional* for *attention*, a
  part of speech or a name that would be left out, or OCR damage).

Use the sentence and your knowledge of English; do not look anything up and do not run code.
Answer for each item with its id, `right` or `wrong`, the reason when wrong, and a few words of
explanation.
