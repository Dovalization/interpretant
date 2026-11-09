"""Extract field-distinctive vocabulary from the books corpus using TF-IDF.

Groups books by manifest field labels, builds a TF-IDF matrix over all books
(one row per book), averages scores within each field group, and writes the
top-N terms per field to data/processed/field_lexicons.json.

Run:  uv run python scripts/extract_field_lexicons.py
Output: data/processed/field_lexicons.json
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Field mapping ────────────────────────────────────────────────────────────
# Maps our 6 display field names to the manifest field labels that feed them.

FIELD_MAP: dict[str, list[str]] = {
    "ML / deep learning": ["ai_cs"],
    "Neuroscience / biology": ["neuroscience", "psychology"],
    "Philosophy of mind": ["philosophy_of_mind", "phenomenology"],
    "Cognitive science": ["cognitive_science"],
    "Cybernetics / systems": ["systems_theory"],
    "Social theory": ["sociology", "political_philosophy", "anthropology"],
}

# Invert: manifest_field → display_field
_manifest_to_display: dict[str, str] = {
    mf: df for df, mfs in FIELD_MAP.items() for mf in mfs
}

TOPN = 60          # words to emit per field
MIN_BOOKS = 3      # skip fields with fewer than this many books

# ─── System dictionary ────────────────────────────────────────────────────────

_sys_dict_path = pathlib.Path("/usr/share/dict/words")
_valid_words: set[str] = set()
if _sys_dict_path.exists():
    _valid_words = {w.lower() for w in _sys_dict_path.read_text().splitlines() if w.isalpha()}

STOPWORDS = {
    # function words
    "that", "this", "with", "from", "they", "have", "been", "were", "will",
    "which", "their", "there", "what", "when", "then", "also", "into", "some",
    "more", "over", "only", "such", "than", "very", "even", "much", "most",
    "those", "these", "would", "could", "should", "upon", "about", "after",
    "before", "other", "every", "itself", "himself", "herself", "both", "each",
    "must", "made", "make", "said", "says", "just", "here", "where", "while",
    "being", "having", "does", "doing", "used", "like", "well", "come", "came",
    "thus", "therefore", "however", "between", "without", "through", "another",
    "because", "although", "different", "certain", "something", "nothing",
    "everything", "anything", "whether", "perhaps", "always", "never", "often",
    "still", "again", "might", "shall", "since", "though", "under", "until",
    "among", "along", "around", "within", "during", "against", "whose",
    "itself", "rather", "quite", "seems", "appear", "become", "became",
    "indeed", "merely", "simply", "already", "almost", "either", "neither",
    "cannot", "possible", "course", "little", "subject", "account", "matter",
    "approach", "important", "similar", "various", "following", "called",
    "suggest", "provide", "consider", "regard", "include", "shows", "based",
    "basic", "definition", "formulation", "attitude", "center", "fresh",
    # French words that appear in translated texts (Bergson, Merleau-Ponty, etc.)
    "abord", "valeur", "champ", "termes", "donne", "corps", "sidenote",
    "frazer", "inconscient",
    # overly generic nouns/adjectives that appear everywhere
    "world", "first", "human", "nature", "point", "sense", "object", "whole",
    "place", "thing", "things", "given", "found", "present", "general",
    "particular", "problem", "theory", "system", "example", "number", "value",
    "state", "space", "order", "level", "model", "process", "function",
    "figure", "chapter", "relationship", "result", "results", "question",
    "manner", "means", "basis", "terms", "cases", "fact", "facts", "form",
    "forms", "class", "sense", "kinds", "stage", "parts", "paper", "book",
    "three", "second", "third", "large", "small", "great", "might", "think",
    "known", "taken", "gives", "given", "using", "takes", "follows", "itself",
    # Darwin-era OCR / naturalist noise bleeding into neuroscience field
    "nearly", "comparatively", "chiefly", "upwards", "orang", "monkey",
    "savage", "memoir", "naturalist", "genus", "tropical", "extinct",
    "probable", "pollen", "organic", "domestication", "borneo", "marine",
    # Too generic across cognitive / ML fields to be distinctive
    "stimulus", "response", "behavior", "behaviour", "computer", "input",
    "output", "program", "programme",
}


def is_valid(w: str) -> bool:
    return (
        len(w) >= 5
        and w in _valid_words
        and w not in STOPWORDS
        and not w.isdigit()
    )


def tokenise(text: str) -> str:
    """Return the text as a whitespace-joined string of valid tokens (for sklearn)."""
    tokens = re.findall(r"[a-z]{5,}", text.lower())
    return " ".join(t for t in tokens if is_valid(t))


# ─── Load manifest and build (text, field_list) pairs ────────────────────────

manifest_path = pathlib.Path("data/external/books/manifest.json")
manifest = json.loads(manifest_path.read_text())

docs: list[str] = []
doc_fields: list[list[str]] = []

n_loaded = 0
n_skipped = 0

for book in manifest:
    decade = book["decade"]
    book_fields = book.get("fields", [])

    display_fields = list({
        _manifest_to_display[mf]
        for mf in book_fields
        if mf in _manifest_to_display
    })
    if not display_fields:
        n_skipped += 1
        continue

    if "filename" in book:
        path = pathlib.Path("data/external/books") / decade / book["filename"]
    elif "file" in book:
        path = pathlib.Path(book["file"])
    else:
        n_skipped += 1
        continue
    if not path.exists():
        n_skipped += 1
        continue

    text = tokenise(path.read_text(errors="ignore"))
    if not text.strip():
        n_skipped += 1
        continue

    docs.append(text)
    doc_fields.append(display_fields)
    n_loaded += 1

print(f"Books loaded: {n_loaded}, skipped: {n_skipped}")

# Report per-field book counts
field_doc_counts: dict[str, int] = {}
for fields in doc_fields:
    for df in fields:
        field_doc_counts[df] = field_doc_counts.get(df, 0) + 1
for df, cnt in sorted(field_doc_counts.items()):
    print(f"  {df}: {cnt} books")

# ─── Fit TF-IDF over all books ────────────────────────────────────────────────
# sublinear_tf=True dampens term frequency (log(1+tf)) so long books don't dominate.
# min_df=3 requires a term to appear in at least 3 books total.
# IDF is computed over all documents in the corpus.

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    min_df=3,
    max_df=0.85,    # exclude terms appearing in >85% of books (too generic)
    token_pattern=r"(?u)\b[a-z]{5,}\b",
    norm="l2",
)
tfidf_matrix = vectorizer.fit_transform(docs)   # shape: (n_books, n_terms)
vocab = vectorizer.get_feature_names_out()       # shape: (n_terms,)

# ─── Average TF-IDF per field and extract top terms ───────────────────────────

field_lexicons: dict[str, list[str]] = {}

all_display_fields = sorted(field_doc_counts.keys())

for df in all_display_fields:
    n_docs = field_doc_counts[df]
    if n_docs < MIN_BOOKS:
        field_lexicons[df] = []
        print(f"\n{df} — skipped (only {n_docs} books)")
        continue

    # Indices of books belonging to this field
    indices = [i for i, fields in enumerate(doc_fields) if df in fields]
    # Mean TF-IDF score for each term across books in this field
    field_scores: np.ndarray = np.asarray(tfidf_matrix[indices].mean(axis=0)).flatten()

    # Rank by mean score, filter to valid dictionary words
    ranked = sorted(enumerate(field_scores), key=lambda x: x[1], reverse=True)
    top_words = [
        vocab[idx]
        for idx, score in ranked
        if score > 0 and vocab[idx] not in STOPWORDS and vocab[idx] in _valid_words
    ][:TOPN]

    field_lexicons[df] = top_words
    print(f"\n{df} — top 20:")
    print("  " + ", ".join(top_words[:20]))

# ─── Save ─────────────────────────────────────────────────────────────────────

out_path = pathlib.Path("data/processed/field_lexicons.json")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(field_lexicons, indent=2))
print(f"\nsaved → {out_path}")
