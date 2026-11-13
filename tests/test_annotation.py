"""The spaCy pass, with the repository's cleaning rules and patterns.

The pipeline (en_core_web_sm, a pinned dependency, with lingua as its last component) is loaded
once per session in conftest.py.
"""

from __future__ import annotations

import pytest
from spacy.language import Language

from interpretant.catalog import CatalogConfig, CleaningPatterns, load_annotations, save_annotations
from interpretant.corpus.annotation import (
    MALFORMED,
    SPANS_KEY,
    AnnotatedAbstract,
    RawAbstract,
    annotate_abstracts,
    build_pipeline,
)
from interpretant.corpus.language import (
    CONFIDENCE_ATTRIBUTE,
    LANGUAGE_ATTRIBUTE,
    SENTENCE_LANGUAGES,
)

# The share of an abstract's judged text in another language that drops it.
SHARE = 0.5
ENGLISH_SENTENCE = "Attention mechanisms improve machine translation in many settings and tasks."
GERMAN_SENTENCE = "Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung."


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


def annotate(text: str, nlp: Language, patterns: CleaningPatterns) -> AnnotatedAbstract:
    ((_, row),) = annotate_abstracts([raw(text)], nlp, patterns, max_not_english_share=SHARE)
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
        ("Write to someone@example.org for the full data.", "email", "someone@example.org"),
        (
            "The data are available at doi:10.1000/xyz123 for reuse.",
            "doi",
            "doi:10.1000/xyz123",
        ),
        ("We bound \\alpha by a constant.", "latex", "\\alpha"),
        ("Abstract: We study attention in neural models.", "abstract_label", "Abstract:"),
        (
            "We study attention in neural models. © 2019 Elsevier Ltd. All rights reserved.",
            "copyright",
            "© 2019 Elsevier Ltd.",
        ),
        (
            "We study attention in neural models. "
            "This content downloaded from 128.0.0.1 on Tue, 01 Jan 2019.",
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
    patterns: CleaningPatterns,
) -> None:
    row = annotate(text, nlp, patterns)
    assert removed in removed_texts(row).get(reason, [])
    assert row.dropped_reason is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Letter from John Smith to the editor, 1890, about the harvest.", "catalogue_record"),
        ("Journal of Things, Volume 3, Issue 2, 2010, Pages 120-123.", "citation_record"),
        ("Export citation. Save to Binder. Track citations.", "web_scrape"),
        ("Get the latest brain dumps for certification exams here.", "exam_spam"),
        ("cafÃ© society in the 1920s and after the war.", "mojibake"),
        ("Abstract: Short note.", "too_short"),
        (f"{GERMAN_SENTENCE} {ENGLISH_SENTENCE[:30]}.", "not_english"),
    ],
    ids=[
        "catalogue record",
        "citation record",
        "web scrape",
        "exam spam",
        "mojibake",
        "too short",
        "mostly another language",
    ],
)
def test_annotate_abstracts_drops_records_that_are_not_abstracts(
    text: str,
    reason: str,
    nlp: Language,
    patterns: CleaningPatterns,
) -> None:
    assert annotate(text, nlp, patterns).dropped_reason == reason


@pytest.mark.parametrize(
    "text",
    [
        "We received a letter from the editor about the harvest.",
        "Twitter use among teenagers rose sharply over the decade.",
        "The abstract: a summary of attention research.",
        "Students prepare for English certification exams with self-directed strategies.",
        "This paper concentrates on two actual questions of nerve growth factor research.",
    ],
    ids=[
        "letter mid-sentence",
        "a lone social network",
        "label not at the start",
        "research on certification exams",
        "actual questions meaning current ones",
    ],
)
def test_annotate_abstracts_keeps_look_alikes_of_the_rules(
    text: str, nlp: Language, patterns: CleaningPatterns
) -> None:
    row = annotate(text, nlp, patterns)
    assert row.dropped_reason is None
    assert row.removals == []


def test_annotate_abstracts_keeps_abbreviations_inside_sentences(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = "As shown in Fig. 2, the effect holds (Smith et al. 2019), cf. Sect. 3. It is robust."
    row = annotate(text, nlp, patterns)
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


def test_build_pipeline_runs_lingua_then_drops_the_tensors(nlp: Language) -> None:
    assert nlp.pipe_names[-2:] == [SENTENCE_LANGUAGES, "doc_cleaner"]
    settings = nlp.config["components"][SENTENCE_LANGUAGES]
    assert (settings["languages"], settings["english_threshold"]) == (["en", "de", "fr", "es"], 0.9)
    assert nlp("Attention helps models.").tensor is None


def test_build_pipeline_runs_named_entity_recognition_only_when_asked(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    assert "ner" not in nlp.pipe_names
    assert not nlp("Ada Lovelace worked in London.").has_annotation("ENT_IOB")
    with_entities = build_pipeline(
        patterns, languages=["en"], english_threshold=0.9, named_entity_recognition=True
    )
    assert "ner" in with_entities.pipe_names


def test_annotate_abstracts_reads_an_english_abstract_as_english(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = "Attention mechanisms improve machine translation. The effect is large."
    row = annotate(text, nlp, patterns)
    assert [sentence.language for sentence in row.sentences] == ["en", "en"]


def test_annotate_abstracts_removes_a_sentence_in_another_language(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = f"{ENGLISH_SENTENCE} {ENGLISH_SENTENCE} {GERMAN_SENTENCE}"
    row = annotate(text, nlp, patterns)
    assert [sentence.language for sentence in row.sentences] == ["en", "en", "de"]
    assert removed_texts(row)["not_english"] == [GERMAN_SENTENCE]
    assert row.dropped_reason is None


def test_annotate_abstracts_flags_a_sentence_without_punctuation(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    run_on = " ".join(["models learn senses from attention in context"] * 22)
    row = annotate(f"{run_on}.", nlp, patterns)
    assert row.flags == [MALFORMED]
    assert row.dropped_reason is None


def test_annotation_cache_keeps_sentences_norms_languages_and_removal_spans(
    catalog: CatalogConfig, nlp: Language, patterns: CleaningPatterns
) -> None:
    text = f"Abstract: Models use senses. See https://example.org now. {GERMAN_SENTENCE}"
    ((original, _),) = annotate_abstracts([raw(text)], nlp, patterns, max_not_english_share=SHARE)
    save_annotations(catalog, [original], part="ai_2010")
    (cached,) = load_annotations(catalog, "ai_2010", nlp.vocab)
    assert [sentence.text for sentence in cached.sents] == [s.text for s in original.sents]
    assert [token.norm_ for token in cached] == [token.norm_ for token in original]
    assert [token.lemma_ for token in cached] == [token.lemma_ for token in original]
    assert [span.label_ for span in cached.spans[SPANS_KEY]] == ["abstract_label", "url"]
    assert [sentence._.get(LANGUAGE_ATTRIBUTE) for sentence in cached.sents] == ["en", "en", "de"]
    assert [sentence._.get(CONFIDENCE_ATTRIBUTE) for sentence in cached.sents] == [
        sentence._.get(CONFIDENCE_ATTRIBUTE) for sentence in original.sents
    ]


def test_annotate_abstracts_gives_the_same_rows_in_worker_processes(
    nlp: Language, patterns: CleaningPatterns
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
            patterns,
            max_not_english_share=SHARE,
            batch_size=2,
            process_count=process_count,
        )
        return [row for _, row in pairs]

    assert rows(2) == rows(1)
