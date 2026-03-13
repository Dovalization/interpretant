# Corpus

## Supported sources

### S2ORC (Semantic Scholar Open Research Corpus)

A large-scale corpus of academic papers with full text and metadata.

- Download: https://allenai.org/data/s2orc
- Config: `conf/corpus/s2orc.yaml`
- Loader: `interpretant.corpus.S2ORCSource`

### arXiv

Bulk data from the arXiv preprint server.

- Download: https://arxiv.org/help/bulk_data_s3
- Config: `conf/corpus/arxiv.yaml`
- Loader: `interpretant.corpus.ArxivSource`

## Preprocessing

`preprocess_text(text)` applies the following pipeline:

1. Unicode NFKC normalisation
2. Lowercase
3. Strip LaTeX math spans (`$...$` and `$$...$$`)
4. Replace non-alphanumeric characters with space
5. Collapse whitespace

`filter_empty(texts)` removes blank entries from document lists.
