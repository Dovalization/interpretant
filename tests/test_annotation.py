"""The spaCy pass, with the repository's cleaning rules and patterns.

It loads en_core_web_sm (the pass's own pipeline, a pinned dependency) once for the session.
"""

from __future__ import annotations

import pytest
from lingua import LanguageDetector
from spacy.language import Language

from interpretant.catalog import (
    CatalogConfig,
    CleaningPatterns,
    load_annotations,
    load_cleaning_patterns,
    read_catalog,
    save_annotations,
)
from interpretant.corpus.annotation import (
    SPANS_KEY,
    AnnotatedAbstract,
    RawAbstract,
    annotate_abstracts,
    build_language_detector,
    build_pipeline,
)

ENGLISH_THRESHOLD = 0.9


@pytest.fixture(scope="session")
def patterns() -> CleaningPatterns:
    return load_cleaning_patterns(read_catalog())


@pytest.fixture(scope="session")
def nlp(patterns: CleaningPatterns) -> Language:
    return build_pipeline(patterns)


@pytest.fixture(scope="session")
def detector() -> LanguageDetector:
    return build_language_detector(["en", "de", "fr", "es"])


def raw(text: str, work: int = 1) -> RawAbstract:
    return RawAbstract(
        work_id=f"https://openalex.org/W{work}",
        year=2017,
        fields=("ai",),
        source="regular",
        document_type="article",
        author_ids=("https://openalex.org/A1",),
        text=text,
    )


def annotate(
    text: str, nlp: Language, detector: LanguageDetector, patterns: CleaningPatterns
) -> AnnotatedAbstract:
    ((_, row),) = annotate_abstracts(
        [raw(text)], nlp, detector, patterns, language_threshold=ENGLISH_THRESHOLD
    )
    return row


def removed_texts(row: AnnotatedAbstract) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for removal in row.removals:
        found.setdefault(removal.reason, []).append(row.text[removal.start : removal.end])
    return found


@pytest.mark.parametrize(
    ("text", "reason", "removed"),
    [
        ("Results are online at https://example.org today.", "url", "https://example.org"),
        ("Write to someone@example.org for data.", "email", "someone@example.org"),
        ("Data are at doi:10.1000/xyz123 for reuse.", "doi", "doi:10.1000/xyz123"),
        ("We bound \\alpha by a constant.", "latex", "\\alpha"),
        ("Abstract: We study attention.", "abstract_label", "Abstract:"),
        (
            "We study attention. © 2019 Elsevier Ltd. All rights reserved.",
            "copyright",
            "© 2019 Elsevier Ltd.",
        ),
        (
            "We study attention. This content downloaded from 128.0.0.1 on Tue, 01 Jan 2019.",
            "footer",
            "This content downloaded from 128.0.0.1 on Tue, 01 Jan 2019.",
        ),
    ],
    ids=["url", "email", "doi", "latex", "leading label", "copyright", "jstor footer"],
)
def test_annotate_abstracts_removes_each_kind_of_text_with_its_reason(
    text: str,
    reason: str,
    removed: str,
    nlp: Language,
    detector: LanguageDetector,
    patterns: CleaningPatterns,
) -> None:
    row = annotate(text, nlp, detector, patterns)
    assert removed in removed_texts(row).get(reason, [])
    assert row.dropped_reason is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Letter from John Smith to the editor, 1890, about the harvest.", "catalogue_record"),
        ("Journal of Things, Volume 3, Issue 2, 2010, Pages 120-123.", "citation_record"),
        ("Export citation. Save to Binder. Track citations.", "web_scrape"),
        ("Get the latest brain dumps for certification exams here.", "exam_spam"),
        ("cafÃ© society in the 1920s.", "mojibake"),
    ],
    ids=["catalogue record", "citation record", "web scrape", "exam spam", "mojibake"],
)
def test_annotate_abstracts_drops_records_that_are_not_abstracts(
    text: str,
    reason: str,
    nlp: Language,
    detector: LanguageDetector,
    patterns: CleaningPatterns,
) -> None:
    assert annotate(text, nlp, detector, patterns).dropped_reason == reason


@pytest.mark.parametrize(
    "text",
    [
        "We received a letter from the editor about the harvest.",
        "Twitter use among teenagers rose sharply over the decade.",
        "The abstract: a summary of attention research.",
    ],
    ids=["letter mid-sentence", "a lone social network", "label not at the start"],
)
def test_annotate_abstracts_keeps_look_alikes_of_the_rules(
    text: str, nlp: Language, detector: LanguageDetector, patterns: CleaningPatterns
) -> None:
    row = annotate(text, nlp, detector, patterns)
    assert row.dropped_reason is None
    assert row.removals == []


def test_annotate_abstracts_keeps_abbreviations_inside_sentences(
    nlp: Language, detector: LanguageDetector, patterns: CleaningPatterns
) -> None:
    text = "As shown in Fig. 2, the effect holds (Smith et al. 2019), cf. Sect. 3. It is robust."
    row = annotate(text, nlp, detector, patterns)
    sentences = [row.text[sentence.start : sentence.end] for sentence in row.sentences]
    assert sentences == [
        "As shown in Fig. 2, the effect holds (Smith et al. 2019), cf. Sect. 3.",
        "It is robust.",
    ]


def test_build_pipeline_keeps_hyphenated_words_whole(nlp: Language) -> None:
    assert [token.text for token in nlp("self-attention and trade-offs in COVID-19")] == [
        "self-attention",
        "and",
        "trade-offs",
        "in",
        "COVID-19",
    ]


def test_annotate_abstracts_reads_an_english_abstract_as_english(
    nlp: Language, detector: LanguageDetector, patterns: CleaningPatterns
) -> None:
    text = "Attention mechanisms improve machine translation. The effect is large."
    row = annotate(text, nlp, detector, patterns)
    assert [sentence.language for sentence in row.sentences] == ["en", "en"]


def test_annotate_abstracts_removes_a_sentence_in_another_language(
    nlp: Language, detector: LanguageDetector, patterns: CleaningPatterns
) -> None:
    text = (
        "Attention mechanisms improve machine translation in many settings. "
        "Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung."
    )
    row = annotate(text, nlp, detector, patterns)
    assert [sentence.language for sentence in row.sentences] == ["en", "de"]
    assert removed_texts(row)["not_english"] == [
        "Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung."
    ]


def test_annotation_cache_keeps_sentences_lemmas_and_removal_spans(
    catalog: CatalogConfig, nlp: Language, detector: LanguageDetector, patterns: CleaningPatterns
) -> None:
    pairs = list(
        annotate_abstracts(
            [raw("Abstract: Models use senses. See https://example.org now.")],
            nlp,
            detector,
            patterns,
            language_threshold=ENGLISH_THRESHOLD,
        )
    )
    save_annotations(catalog, [doc for doc, _ in pairs], part="ai_2010")
    (cached,) = load_annotations(catalog, "ai_2010", nlp.vocab)
    original = pairs[0][0]
    assert [sentence.text for sentence in cached.sents] == [s.text for s in original.sents]
    assert [token.lemma_ for token in cached] == [token.lemma_ for token in original]
    assert [span.label_ for span in cached.spans[SPANS_KEY]] == ["abstract_label", "url"]


def test_annotate_abstracts_gives_the_same_rows_in_worker_processes(
    nlp: Language, detector: LanguageDetector, patterns: CleaningPatterns
) -> None:
    abstracts = [
        raw("Abstract: Self-attention helps. See https://example.org now.", work=1),
        raw("As shown in Fig. 2, doi:10.1000/xyz123 holds the data.", work=2),
        raw("Models learn senses. The effect is large.", work=3),
    ]

    def rows(process_count: int) -> list[AnnotatedAbstract]:
        pairs = annotate_abstracts(
            abstracts,
            nlp,
            detector,
            patterns,
            language_threshold=ENGLISH_THRESHOLD,
            batch_size=2,
            process_count=process_count,
        )
        return [row for _, row in pairs]

    assert rows(2) == rows(1)


def test_build_language_detector_requires_english() -> None:
    with pytest.raises(ValueError, match="English"):
        build_language_detector(["de", "fr"])
