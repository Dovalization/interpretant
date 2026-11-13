"""The spaCy pass, with the repository's cleaning rules and patterns.

The pipeline (en_core_web_sm, a pinned dependency, with lingua as its last component) is loaded
once per session in conftest.py.
"""

from __future__ import annotations

import pytest
from spacy.language import Language
from spacy.tokenizer import Tokenizer

from interpretant.catalog import CatalogConfig, CleaningPatterns, load_annotations, save_annotations
from interpretant.corpus.annotation import (
    MALFORMED,
    SPANS_KEY,
    AnnotatedAbstract,
    RawAbstract,
    annotate_abstracts,
    build_pipeline,
    kept_text,
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
# Numbers and abbreviations the glued-punctuation rules must leave whole.
WHOLE_TOKENS = ("0.05", "U.S.", "Fig.", "Eq.", "0.922", "3.5")


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


@pytest.mark.parametrize(
    ("text", "removed"),
    [
        ("Scores are listed at www.example.org for all teams.", "www.example.org"),
        ("Scores are listed at example.org/scores for all teams.", "example.org/scores"),
        ("Attention scores were tracked by altmetric.com over time.", "altmetric.com"),
    ],
    ids=["www", "path", "domain ending"],
)
def test_annotate_abstracts_removes_links_without_a_scheme(
    text: str, removed: str, nlp: Language, patterns: CleaningPatterns
) -> None:
    assert removed_texts(annotate(text, nlp, patterns))["url"] == [removed]


STAMP = (
    "This content downloaded from 157.55.39.1 on Tue, 01 Jan 2019 12:34:56 UTC "
    "All use subject to http://about.jstor.org/terms"
)


@pytest.mark.parametrize(
    ("text", "reason", "removed"),
    [
        (
            "Models learn senses from context in five corpora Copyright © 2019 Elsevier Ltd.",
            "copyright",
            "Copyright © 2019 Elsevier Ltd.",
        ),
        (
            f"{ENGLISH_SENTENCE} (PsycINFO Database Record (c) 2006 APA, all rights reserved).",
            "copyright",
            "(PsycINFO Database Record (c) 2006 APA, all rights reserved).",
        ),
        (
            f"{ENGLISH_SENTENCE} To register or login click here.",
            "footer",
            "To register or login click here.",
        ),
        (
            f"Deduction is developed as a process of composition {STAMP} in Rule XII, replacing "
            "definition as the means to clarity.",
            "download_stamp",
            STAMP,
        ),
        (
            f"{ENGLISH_SENTENCE} Additional Resources FiguresReferencesRelatedDetailsCited by "
            "Richard Luchsinger and Gottfried E. Arnold (1970).",
            "footer",
            "Additional Resources FiguresReferencesRelatedDetailsCited by Richard Luchsinger "
            "and Gottfried E. Arnold (1970).",
        ),
    ],
    ids=[
        "copyright after text",
        "psycinfo record",
        "login prompt",
        "jstor stamp mid-text",
        "platform page metadata",
    ],
)
def test_annotate_abstracts_removes_a_notice_and_keeps_the_text_before_it(
    text: str, reason: str, removed: str, nlp: Language, patterns: CleaningPatterns
) -> None:
    row = annotate(text, nlp, patterns)
    assert removed_texts(row)[reason] == [removed]
    kept = kept_text(row.text, row.removals)
    assert kept.split() == text.replace(removed, " ").split()
    assert row.dropped_reason is None


def test_annotate_abstracts_removes_an_encoded_library_link_and_keeps_the_abstract(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    link = "{openurl}?query=rft.jtitle%253DJAMA%26rft_id%253Dinfo%253Apmid%252F10086433"
    text = f"{ENGLISH_SENTENCE} {ENGLISH_SENTENCE} [1]: {link} [2]: see the trial."
    row = annotate(text, nlp, patterns)
    assert removed_texts(row) == {"encoded_link": [link]}
    assert row.dropped_reason is None


def test_annotate_abstracts_keeps_dotted_words_that_are_not_links(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = "We coded the interviews in ATLAS.ti and served Node.js apps of.the E.coli lab."
    assert "url" not in removed_texts(annotate(text, nlp, patterns))


def test_annotate_abstracts_keeps_the_audit_abbreviations_inside_sentences(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = (
        "Two measures, viz. recall and precision, rose (pp. 12-14, Vol. 3, vol. 2), "
        "resp. 4% and 6%, from ca. 1950 on. It is robust."
    )
    row = annotate(text, nlp, patterns)
    sentences = [row.text[sentence.start : sentence.end] for sentence in row.sentences]
    assert sentences == [text.removesuffix(" It is robust."), "It is robust."]


@pytest.mark.parametrize(
    ("text", "first"),
    [
        ("The fit was good (r = 0.922).In addition, it held.", "The fit was good (r = 0.922)."),
        ("This was shown before (Ogles, 2004).This focus matters.", "(Ogles, 2004)."),
        ("The commission met in 1950.French delegates came.", "The commission met in 1950."),
        ("He called it the “New Frontier”.The phrase spread.", "the “New Frontier”."),
        ("The score rose to 3.5.Then it fell back.", "The score rose to 3.5."),
        ("Models were trained on two corpora.Results were mixed.", "on two corpora."),
        ("Does attention help?Yes, in every setting.", "Does attention help?"),
    ],
    ids=["bracket", "citation", "year", "quote", "decimal", "two words", "question"],
)
def test_build_pipeline_starts_a_sentence_after_glued_punctuation(
    text: str, first: str, nlp: Language
) -> None:
    sentences = [sentence.text for sentence in nlp(text).sents]
    assert len(sentences) == 2
    assert sentences[0].endswith(first)


@pytest.mark.parametrize(
    ("text", "first"),
    [
        (
            "This was studied by Goodman [2] and Haberman [3]. Examples covered both cases.",
            "This was studied by Goodman [2] and Haberman [3].",
        ),
        (
            "Prior work [4, 5] used attention. We extend it to longer texts.",
            "Prior work [4, 5] used attention.",
        ),
        (
            "The target moves fast [7][8][9]. Therefore, tracking is hard.",
            "The target moves fast [7][8][9].",
        ),
        (
            "Results agree with [12–14].As shown in Fig. 2, the gain holds.",
            "Results agree with [12–14].",
        ),
        (
            "Adults discriminate [ta] from [da] and rate sleep quality [PSQI] well. Others do not.",
            "Adults discriminate [ta] from [da] and rate sleep quality [PSQI] well.",
        ),
        (
            "The problem is solved efficiently [2]. [3] claims that it is not.",
            "The problem is solved efficiently [2].",
        ),
    ],
    ids=[
        "single numbers",
        "list",
        "consecutive",
        "range then glued stop",
        "phonetic symbols and an abbreviation",
        "citation starting a sentence",
    ],
)
def test_build_pipeline_starts_no_sentence_inside_square_brackets(
    text: str, first: str, nlp: Language
) -> None:
    sentences = [sentence.text for sentence in nlp(text).sents]
    assert sentences == [first, text.removeprefix(first).lstrip()]


def test_build_pipeline_splits_glued_punctuation_and_brackets_off(nlp: Language) -> None:
    tokenizer = nlp.tokenizer
    assert isinstance(tokenizer, Tokenizer)
    assert [text for _, text in tokenizer.explain("(r = 0.922).In")] == [
        "(",
        "r",
        "=",
        "0.922",
        ")",
        ".",
        "In",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "The effect was significant at p < 0.05 in the U.S. sample.",
        "As shown in Fig. 2 and Eq. 3, the gain is 0.922 on average.",
        "Results improved by 3.5% (see Table 2).",
    ],
    ids=["p value and country", "figure and equation", "percentage"],
)
def test_build_pipeline_keeps_numbers_and_abbreviations_whole(text: str, nlp: Language) -> None:
    doc = nlp(text)
    assert len(list(doc.sents)) == 1
    tokens = {token.text for token in doc}
    assert {kept for kept in WHOLE_TOKENS if kept in text} <= tokens


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
