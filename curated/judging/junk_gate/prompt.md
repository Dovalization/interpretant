You are checking records from a corpus of English research abstracts (1950s to 2020s; fields such
as artificial intelligence, human-computer interaction, computer science, cognitive psychology,
linguistics, neuroscience, communication, sociology, philosophy and history of science). Each
record should hold a work's abstract. Some hold other text. Your job is to say which.

Each item shows one record's text. One sentence may be marked between ⟦ and ⟧. The mark
`[… cut …]` shows where text was left out for length; it is not part of the record.

## The question

Is the text **the work's own prose**: the authors writing about their own work, in English?

- **`abstract`**: the text is the work's own prose. All of these count:
  - an abstract or summary, with or without headings (*Background*, *Methods*, *Results*);
  - the opening paragraphs of an article, used where the work has no abstract;
  - a book review, an editorial, a comment or a reply: each argues in its own words;
  - an abstract of a talk, a workshop paper, a thesis, a report or a book chapter;
  - a summary of a book's argument or content, in prose;
  - a short abstract, even one sentence, or an abstract cut off before its end;
  - an abstract with small leftovers around it: a copyright line, a label, a stray reference, some
    damaged words.
- **`junk`**: the text is not the work's own prose. Give its kind:
  - `citation_or_metadata`: a citation line, author and journal details, a DOI or page range, a
    thesis catalogue line (*Thesis: Ph. D., …*), a list of keywords with no sentence;
  - `publisher_or_web`: text from a publisher's or repository's page: access or login notices,
    navigation, pricing, praise for a book, series information;
  - `contents_or_list`: a table of contents, a list of titles, a list of reviewed books, a list of
    examples or names with no prose around it;
  - `archive_or_catalogue`: a description of an archive item: a photograph, a newspaper page, a
    letter in a collection, a catalogue note;
  - `other_work`: text about something other than the work, or another work's text standing in
    for it: a quotation used as an epigraph, a chronology, a biography of an interviewee;
  - `letter_obituary_erratum_news`: a letter to the editor, an obituary or death notice, a
    correction notice, a news item or announcement;
  - `fragment`: a piece with no complete sentence of its own, such as a few words or a broken line;
  - `not_english`: text that is mostly not in English;
  - `other`: junk of another kind (say which in the note).
- **`mixed`**: the work's own prose **and** a block of other material in the same record, for
  example an abstract followed by a reference list, or a contents list before a summary. Give the
  kind of the other material. A small leftover does not make a record mixed: use `mixed` when the
  other material is at least a few lines or about a fifth of the text.

## Rules for the hard cases

- **Judge the text, not its quality.** A dull, short, old-fashioned or badly written abstract is
  still an abstract. So is one from an unusual venue.
- **A book review is an abstract; a list of reviewed books is junk.** *Book reviews in this
  article: …* followed by titles is `contents_or_list`.
- **A biographical or historical article is an abstract; an obituary or death notice is junk.**
  An article that studies a person's work is the authors' own prose.
- **A letter to the editor is junk** (*To the Editor:*, *Sir,*), even when it argues a point. An
  editorial or a comment is an abstract.
- **A description of what a book or issue contains** is an abstract when it summarises the
  argument in prose, and junk when it lists contents or sells the book.
- **A passage of an older or primary text** (a philosopher's, a classic author's, a scripture's,
  a historical document's) standing as the record's text is the work's own text: the work is
  usually an edition, a translation, an anthology or a study that opens with it. It is junk only
  when the record plainly belongs to something else, such as a quotation printed above a
  publisher's description.
- **Damage is not junk.** Glued or split words, stray characters and lost punctuation do not change
  the verdict. They are recorded separately (below).
- **When unsure between `abstract` and `junk`,** ask whether a reader would learn what the work
  argues or found. If yes, it is an abstract.

## The marked sentence

- **`sentence`**: is the marked sentence part of the work's own prose?
  - `own`: yes;
  - `other`: no, it belongs to the other material (a citation, a heading list, a notice, a
    quotation of another work);
  - `none`: no sentence is marked.
- In a `junk` record the marked sentence is `other`. In an `abstract` it is usually `own`, but a
  marked leftover (a copyright line, a stray reference) is `other`.

## Damage

- **`damaged`**: `yes` if the marked sentence shows damage from scanning or conversion: words glued
  together or split apart, letters swapped for look-alikes (*l* for *I*, *rn* for *m*), stray
  characters, or two sentences run together with no stop. Otherwise `no`. With no marked sentence,
  judge the whole text.

## Your answer

For each item give, in this order:

- `note`: one short sentence saying what the text is, written before you decide;
- `verdict`: `abstract`, `mixed` or `junk`;
- `kind`: the kind of the junk or of the other material, or `none` for an abstract;
- `sentence`: `own`, `other` or `none`;
- `damaged`: `yes` or `no`.
