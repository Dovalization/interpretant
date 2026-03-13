# Books Corpus

## Curatorial philosophy

This is an **anchor corpus** rather than a bulk-ingested one. Every title was selected because it occupies a specific conceptual position across four fields — philosophy of mind, AI/CS, design, and arts — and spans the full temporal range of the project (1890–2020s).

The deliberate curation is a methodological choice: rather than maximising token count, the goal is to track how a small, well-understood set of canonical texts shapes — or reveals — semantic drift in adjacent, large-scale corpora (PubMed, ACL Anthology, arXiv). A book on perception by Merleau-Ponty should produce different neighbourhood structures around words like *embodiment*, *affordance*, or *representation* than an ACL abstract. That contrast is the signal.

Bulk ingestion would dilute this: adding 10,000 digitised 19th-century texts would swamp the signal from the 46 anchor texts and push the corpus toward general historical language rather than disciplinary conceptual history.

## Corpus table

| Author | Title | Year | Fields | Status |
|--------|-------|------|--------|--------|
| William James | The Principles of Psychology | 1890 | philosophy_of_mind, psychology | public_domain |
| Edmund Husserl | Logical Investigations | 1900 | philosophy_of_mind, phenomenology | public_domain |
| Henri Bergson | Creative Evolution | 1907 | philosophy_of_mind, philosophy | public_domain |
| Bertrand Russell | The Analysis of Mind | 1921 | philosophy_of_mind, philosophy | public_domain |
| Ludwig Wittgenstein | Tractatus Logico-Philosophicus | 1921 | philosophy_of_mind, philosophy | public_domain |
| Gilbert Ryle | The Concept of Mind | 1949 | philosophy_of_mind, philosophy | open_access |
| Maurice Merleau-Ponty | Phenomenology of Perception | 1945 | philosophy_of_mind, phenomenology | pending |
| Ludwig Wittgenstein | Philosophical Investigations | 1953 | philosophy_of_mind, philosophy | pending |
| Jerry Fodor | The Language of Thought | 1975 | philosophy_of_mind, cognitive_science | pending |
| Daniel Dennett | Brainstorms | 1978 | philosophy_of_mind, cognitive_science | pending |
| Daniel Dennett | Consciousness Explained | 1991 | philosophy_of_mind, cognitive_science | pending |
| Varela, Thompson, Rosch | The Embodied Mind | 1991 | philosophy_of_mind, cognitive_science | pending |
| David Chalmers | The Conscious Mind | 1996 | philosophy_of_mind | pending |
| Andy Clark | Being There | 1997 | philosophy_of_mind, cognitive_science | pending |
| Norbert Wiener | Cybernetics | 1948 | ai_cs, systems_theory | open_access |
| Minsky & Papert | Perceptrons | 1969 | ai_cs | pending |
| Newell & Simon | Human Problem Solving | 1972 | ai_cs, cognitive_science | pending |
| Douglas Hofstadter | Gödel, Escher, Bach | 1979 | ai_cs, philosophy_of_mind | pending |
| Winograd & Flores | Understanding Computers and Cognition | 1986 | ai_cs, philosophy_of_mind, design | pending |
| Marvin Minsky | The Society of Mind | 1986 | ai_cs, cognitive_science | pending |
| Rodney Brooks | Cambrian Intelligence | 1999 | ai_cs | pending |
| Herbert A. Simon | The Sciences of the Artificial | 1969 | design, ai_cs | pending |
| Christopher Alexander | Notes on the Synthesis of Form | 1964 | design | pending |
| Victor Papanek | Design for the Real World | 1971 | design | pending |
| Edward Tufte | The Visual Display of Quantitative Information | 1983 | design | pending |
| Donald Norman | The Design of Everyday Things | 1988 | design, cognitive_science | pending |
| William J. Mitchell | City of Bits | 1995 | design, arts | open_access |
| John Thackara | In the Bubble | 2005 | design | pending |
| Tim Brown | Change by Design | 2009 | design | pending |
| Ezio Manzini | Design, When Everybody Designs | 2015 | design | pending |
| Walter Benjamin | Illuminations | 1955 | arts, philosophy | author_posted |
| Marshall McLuhan | Understanding Media | 1964 | arts, design | pending |
| Susan Sontag | Against Interpretation | 1966 | arts | pending |
| John Berger | Ways of Seeing | 1972 | arts | pending |
| Roland Barthes | Image Music Text | 1977 | arts, philosophy | pending |
| Jean Baudrillard | Simulacra and Simulation | 1981 | arts, philosophy | author_posted |
| Fredric Jameson | Postmodernism | 1991 | arts, philosophy | pending |
| Lev Manovich | The Language of New Media | 2001 | arts, design | author_posted |
| Boris Groys | Art Power | 2008 | arts | pending |
| Thomas S. Kuhn | The Structure of Scientific Revolutions | 1962 | cross_field, philosophy | pending |
| Gregory Bateson | Steps to an Ecology of Mind | 1972 | cross_field, systems_theory | author_posted |
| Paul Virilio | Speed and Politics | 1977 | cross_field, arts | pending |
| Donna Haraway | Simians, Cyborgs, and Women | 1991 | cross_field, philosophy_of_mind | author_posted |
| Bruno Latour | We Have Never Been Modern | 1991 | cross_field, philosophy | pending |
| N. Katherine Hayles | How We Became Posthuman | 1999 | cross_field, ai_cs, philosophy_of_mind | pending |
| Yuk Hui | Recursivity and Contingency | 2019 | cross_field, philosophy_of_mind, ai_cs | open_access |

## How to add a book

1. Extract the text (PDF → `.txt`) using the pipeline:
   ```bash
   uv run interpretant pdf extract path/to/book.pdf \
     --output data/external/books/1960s/author_title.txt \
     --author "Author Name" --title "Book Title" --year 1964 \
     --decade 1960s --fields design --add-to-manifest
   ```
   Or place a pre-extracted `.txt` file directly in the correct decade folder.

2. Add an entry to `manifest.json` with all required fields:
   `author`, `title`, `year`, `decade`, `fields`, `status`, `source`, `source_url`, `filename`, `notes`.

3. Validate:
   ```bash
   uv run interpretant corpus books validate
   ```
   The entry should appear as `✓` present.

4. Re-run stats to confirm word counts update:
   ```bash
   uv run interpretant corpus books stats
   ```

## Provenance note

Public domain texts (pre-1928) are sourced from Project Gutenberg or the Internet Archive and are free to redistribute. Author-posted texts are linked to stable URLs maintained by the authors or their institutions (e.g., Marxists Internet Archive for Benjamin). Entries marked `pending` must be added manually — either by purchasing and extracting a legitimate copy, or by locating a freely available scan. Do not commit copyrighted `.txt` files to the repository.
