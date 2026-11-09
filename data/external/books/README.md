# Books corpus

The books are the pre-1950 layer of the grid and a within-field series of their own: 358
canonical works in philosophy of mind, AI, design and cultural theory, 1820s–2020s, about 42
million tokens. They are curated, not bulk-ingested: each title was chosen for the conceptual
position it occupies, and the layer is read for trajectories within that tradition, not as a
sample of English.

## Rules

- **Dated by the English edition in the file.** A translation is the translator's English and
  belongs to the decade of that translation; `original_year`, `original_language` and
  `translator` record the source. Every edition decision, with the evidence from each file's
  front matter, is in `scripts/redate_books.py`, which applies them to the manifest.
- **Author cap.** No author contributes more than 250,000 tokens to a decade
  (`corpus books ingest`); `--leave-out-largest-author` checks a finding against the cap.
- **Folders.** `<decade>/` holds the texts the loader reads. `excluded/` holds duplicates
  whose text is already in another file; `canon/` holds English originals before the first
  decade (Coleridge, *Frankenstein*). Entries there carry an `exclusion` reason and never
  enter training.
- **Files are not committed.** Only `manifest.json` and this page are tracked. Source files
  are copyrighted or large; their filenames are not recorded in the repository.

## Manifest

`manifest.json` is a list of entries:

| Field | Meaning |
|---|---|
| `author`, `title` | As on the title page of the English edition |
| `year`, `decade` | The English edition in the file; `decade` is its folder |
| `fields` | Tags such as `philosophy`, `philosophy_of_mind`, `ai_cs`, `design`, `arts`, `cross_field` |
| `status`, `source`, `source_url` | Provenance (`public_domain`, `open_access`, ...) |
| `filename` | The `.txt` in the decade folder |
| `translator`, `original_language`, `original_year` | Present for every translation |
| `notes` | Free text: edition, key vocabulary |
| `exclusion` | Reason, for entries in `excluded/` or `canon/` |

## Adding a book

```bash
uv run interpretant pdf extract path/to/book.pdf \
  --output data/external/books/1960s/author_title.txt \
  --author "Author Name" --title "Book Title" --year 1964 \
  --decade 1960s --fields design --add-to-manifest
uv run interpretant corpus books validate
uv run interpretant corpus books ingest
```

For a translation, add `translator`, `original_language` and `original_year` to the entry by
hand, and date it by the translation. Re-run `corpus audit` afterwards: it reports any entry
whose dating misrepresents the English text it contributes.
