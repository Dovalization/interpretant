You are spot-checking a rebuilt word list for a corpus of English research abstracts (1950s to
2020s; fields such as artificial intelligence, human-computer interaction, computer science,
cognitive psychology, linguistics, communication, sociology, philosophy and history of science).
The word list has one entry per word token: lowercase, accents removed, plural nouns and
third-person singular verbs written as their lemma. The rebuild applies two audited maps to it:

- a spelling map writes each spelling variant in one standard way ("colour" as "color",
  "behaviour" as "behavior");
- a compound map writes each two-part compound in one standard way: open ("data base"),
  hyphenated ("data-base") or closed ("database"). A token can become two words ("user-interface"
  as "user interface"). Two tokens can become one: the first entry holds the compound and the
  second is left empty (shown as "").

Each item shows one changed place: the entries around it before the rebuild (R_B) and after it
(NEW), the changed entry marked by its position (counting from 0; a merge spans that entry and the
next), and the text near the token. Which way of writing is the standard was set by a rule; do not
judge the direction.

- right: NEW writes the same word or compound as R_B there, only spelled or joined in the
  standard way. A stray hyphen left by a line break counts as right ("meth-od" as "method").
- wrong: NEW writes a different word than the text means at that place ("re-creation", creating
  again, as "recreation", leisure); it joins two words that do not form one compound in this text
  (as in a name followed by a noun, or "green house" for a house painted green); or it changes a
  name, a letter code, an abbreviation or a word in another language into an English word.

Use the text and your knowledge of English; do not look anything up and do not run code.
