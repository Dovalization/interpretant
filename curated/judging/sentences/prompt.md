You are checking where a sentence splitter cut research abstracts into sentences. The corpus holds
English research abstracts from the 1950s to the 2020s (fields such as artificial intelligence,
human-computer interaction, computer science, cognitive psychology, linguistics, communication,
sociology, philosophy and history of science). Older texts often carry OCR damage. Check only where
the cuts fall, not the text itself: spelling, spacing and noise do not matter.

Rules for every kind of item:

- **Headings are not errors.** A heading glued to the start of a sentence (*Abstract*, *ABSTRACT*,
  *Introduction*, *Background:*, *Methods:*, *Results:*, *Conclusions:*) is fine: do not mark a
  break after it, and do not count a sentence that starts with one as wrong. A heading left as its
  own row is fine too.
- **Text that is not prose,** such as reference lists, tables, contents or catalogue lines: mark
  only cuts that are plainly wrong.
- **OCR noise and other languages.** A cut caused by OCR damage is judged like any other: a stray
  period that splits one sentence makes a wrong cut; a lost full stop that leaves two sentences in
  one makes a missed break. Judge text in another language by the same rules if you can read it;
  if you cannot, do not mark it.

There are three kinds of item.

**A whole abstract** shows its sentences in order, numbered from 0. Between each two sentences is a
cut. Mark:

- **a wrong cut:** a cut where no sentence ends, for example after an abbreviation such as p., et
  al. or e.g., after an initial or a number, or inside brackets. Give the number of the sentence
  that follows the cut.
- **a missed break:** two sentences left in one. Give the sentence's number and the word that
  should start the second sentence, copied exactly as it appears.

Most abstracts need no marks.

**One sentence** shows the sentence to judge, marked TARGET, with the sentence before it and the
sentence after it as the splitter cut them. Judge only the target:

- **right:** both cuts around it fall where a sentence ends, and it holds one sentence.
- **wrong:** the cut before it is wrong (it starts in the middle of a sentence), the cut after it is
  wrong (it stops in the middle of a sentence), or it holds two sentences. Say which.

**A joined cut** shows two pieces the splitter first cut apart and a rule then joined back into one
sentence, with the text around them.

- **right:** the two pieces belong to one sentence (or one heading, reference or line of non-prose
  text), so joining them was right.
- **wrong:** the first piece ends one sentence and the second starts another, so the join glued two
  sentences together.

Do not look anything up and do not run code.
