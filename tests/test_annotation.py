"""The spaCy pass, the sentence languages and the cleaning, on single abstracts.

The spaCy pipeline (en_core_web_sm, a pinned dependency) is loaded once per session in
conftest.py; each test runs an abstract through the three steps the corpus stages run.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from typing import Any

import pytest
import spacy
from lingua import LanguageDetector
from spacy.language import Language
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc, Span

from interpretant.catalog.annotations import WORK_ID, load_annotations, save_annotations
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.curated import CleaningPatterns, load_cleaning_patterns
from interpretant.corpus.annotation import annotate_texts, build_pipeline
from interpretant.corpus.language import (
    SentenceLanguage,
    build_language_detector,
    detect_sentence_languages,
)
from interpretant.corpus.removals import (
    MALFORMED,
    SPANS_KEY,
    AnnotatedAbstract,
    Removal,
    build_cleaning_pipeline,
    clean_abstract,
    find_removals,
    kept_text,
)
from interpretant.corpus.repair import compile_signatures, repair_text
from interpretant.corpus.works import RawAbstract

# The English words an abstract with text in another language needs to stay.
MIN_ENGLISH_WORDS = 30
ENGLISH_SENTENCE = "Attention mechanisms improve machine translation in many settings and tasks."
GERMAN_SENTENCE = "Die Aufmerksamkeit ist für die Übersetzung von großer Bedeutung."
# Numbers and abbreviations that repair and the tokenizer must leave whole.
WHOLE_TOKENS = ("0.05", "U.S.", "Fig.", "Eq.", "0.922", "3.5")


def as_document(doc: Doc) -> tuple[str, list[tuple[int, int]]]:
    """A document as the language stage reads it: its text and its sentences' ranges."""
    return doc.text, [(sentence.start_char, sentence.end_char) for sentence in doc.sents]


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


def repaired(abstract: RawAbstract) -> tuple[str, str]:
    """A work's id and its text repaired, as the count-word-repairs stage gives them to the pass."""
    return abstract.work_id, repair_text(abstract.text)


def annotate(text: str, nlp: Language, patterns: CleaningPatterns) -> AnnotatedAbstract:
    """One abstract through the spaCy pass, the sentence languages and the cleaning."""
    abstract = raw(text)
    (doc,) = annotate_texts([repaired(abstract)], nlp)
    (languages,) = detect_sentence_languages(
        [as_document(doc)],
        language_detector(),
        english_threshold=0.9,
        minimum_relative_distance=0.1,
    )
    return clean_abstract(
        doc,
        abstract,
        languages,
        cleaning=cleaning_pipeline(),
        patterns=patterns,
        signatures=compile_signatures(patterns.mojibake_signatures),
        min_english_words=MIN_ENGLISH_WORDS,
    )


@cache
def language_detector() -> LanguageDetector:
    return build_language_detector(["en", "de", "fr", "es"])


@cache
def cleaning_pipeline() -> Language:
    return build_cleaning_pipeline(load_cleaning_patterns(read_catalog()))


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
        (
            "(1985). Towards a theory of predication for English. Australian Journal of "
            "Linguistics: Vol. 5, No. 2, pp. 143-196.",
            "citation_record",
        ),
        (
            "The Origins of Modern Sociology. Journal of Sociology, Vol. 21, No. 2 (Apr., 1993), "
            "pp. 166-171",
            "citation_line",
        ),
        (
            "Notes on the Psalms. The Journal of Theological Studies, Volume XXVII, Issue 2, "
            "1 April 1991, Pages 100–112",
            "citation_line",
        ),
        (
            "Notes on the Psalms. The Journal of Theological Studies, Volume 8, Issue January, "
            "January 1953, Pages 91–93",
            "citation_line",
        ),
        ("Export citation. Save to Binder. Track citations.", "web_scrape"),
        (
            "No AccessJournal of Speech and Hearing DisordersArticle1 Dec 1954A Simple Adapter "
            "William R. Tiffany Google Scholar More articles by this author",
            "web_scrape",
        ),
        (
            "Designing for users Ann Smith View Profile Authors Info & Claims CHI '91: Proceedings "
            "of the conference on human factors in computing systems 2Downloads",
            "web_scrape",
        ),
        (
            "Preview this article: Digitalis in heart failure, a review of the evidence",
            "web_scrape",
        ),
        ("Get access Share Icon Share The museum and its public in the 1850s", "web_scrape"),
        ("Get the latest brain dumps for certification exams here.", "exam_spam"),
        (
            "To the Editor.\u2014 The recent article by Parker and Brown deserves a comment.",
            "letter",
        ),
        ("Abstract: To the Editor: We read with great interest the study by Yoo et al.", "letter"),
        (
            "SIR,-A scientific shrine of the greatest interest to geologists is Rugbourne Farm.",
            "letter",
        ),
        ("Dear Editor, with great interest I read the meta-analysis by Michaud et al.", "letter"),
        ("Sir.\u2014Many causes of sudden infant death syndrome have been proposed.", "letter"),
        ("To the Editor.\u2014The report by Dr Gellert and colleagues deserves a reply.", "letter"),
        ("Sir .—I was amused to read the note on the habits of the cuckoo.", "letter"),
        ("To the EditorWe read with interest the article by Yoo and colleagues.", "letter"),
        ("Dear SirMultinational companies now set the terms of the debate.", "letter"),
        (
            "(To the Editor of ‘The Daily News’.) Sir, the harvest was poor this year.",
            "letter",
        ),
        (
            "Letter to the editor regarding the use of placebo in surgical trials.",
            "catalogue_record",
        ),
        (
            "Abstract: Letter from Charles Darwin to Joseph Hooker, 1858, on the species question.",
            "catalogue_record",
        ),
        (
            "Thesis (Ph.D.)--Massachusetts Institute of Technology, Dept. of Physics, 1980.",
            "too_short",
        ),
        ("cafÃ© society in the 1920s and after the war.", "mojibake"),
        ("Abstract: Short note.", "too_short"),
        (f"{GERMAN_SENTENCE} {ENGLISH_SENTENCE[:30]}.", "not_english"),
    ],
    ids=[
        "catalogue record",
        "citation record",
        "citation stub starting with its year",
        "jstor citation line",
        "oup citation line",
        "oup citation line with a month for its issue",
        "web scrape",
        "page with no access glued",
        "acm digital library page",
        "publisher preview",
        "publisher access prompt",
        "exam spam",
        "letter to the editor",
        "letter after an abstract label",
        "letter opening sir with a dash",
        "letter opening dear editor",
        "letter opening sir glued to its dash",
        "letter opening to the editor glued to its dash",
        "letter opening sir with a dash glued to the next word",
        "letter opening to the editor glued to the next word",
        "letter opening dear sir glued to the next word",
        "letter opening to the editor in brackets",
        "catalogue letter that also holds to the editor",
        "catalogue letter after an abstract label",
        "thesis notes alone",
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
        "Sir Isaac Newton's correspondence with Cotes shows how the Principia was revised.",
        "Letters to the editor are a neglected genre in the history of medicine.",
        "We argue that to the editor of a journal the referee's report is advice, not a verdict.",
        "“ Sir, we know our will is free, and there's an end on't,” Johnson told Boswell.",
        "SirJohn Herschel's correspondence with Babbage shows how the society began.",
    ],
    ids=[
        "a man named sir",
        "a study of letters",
        "to the editor mid-sentence",
        "a quotation opening with sir",
        "a man named sir glued to his name",
    ],
)
def test_annotate_abstracts_keeps_texts_that_only_mention_letters(
    text: str, nlp: Language, patterns: CleaningPatterns
) -> None:
    assert annotate(text, nlp, patterns).dropped_reason is None


@pytest.mark.parametrize(
    "text",
    [
        "We received a letter from the editor about the harvest.",
        "Twitter use among teenagers rose sharply over the decade.",
        "The abstract: a summary of attention research.",
        "Students prepare for English certification exams with self-directed strategies.",
        "This paper concentrates on two actual questions of nerve growth factor research.",
        "No accessible records of the dialect survive from before the war.",
        "The first survey of the dialect (1985). A second one followed a decade later.",
        "The thesis (Ph.D. work by the author) argues that the dialect survived the war.",
        "Get access to care early, the campaign told rural families in its first year.",
    ],
    ids=[
        "letter mid-sentence",
        "a lone social network",
        "label not at the start",
        "research on certification exams",
        "actual questions meaning current ones",
        "no accessible at the start",
        "a bracketed year not at the start",
        "a thesis mentioned after the first word",
        "get access to in a real sentence",
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


def test_annotate_abstracts_removes_a_citation_line_after_real_prose(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    citation = "Child Development, Vol. 55, No. 4 (Aug., 1984), pp. 1479-1485."
    row = annotate(f"{ENGLISH_SENTENCE} " * 4 + citation, nlp, patterns)
    assert removed_texts(row) == {"citation_line": [citation]}
    assert row.dropped_reason is None


def test_annotate_abstracts_removes_thesis_notes_and_keeps_the_abstract_after_them(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    notes = "Thesis (M.A.)--Georgetown University, 2009.; Includes bibliographical references."
    abstract = "This study reads the cyborg of science fiction film as a figure of identity."
    row = annotate(f"{notes} {abstract}", nlp, patterns)
    assert removed_texts(row) == {"thesis_notes": [notes]}
    assert row.dropped_reason is None


def test_find_removals_takes_a_notice_to_the_end_of_the_sentence_it_ends_in(
    nlp: Language,
) -> None:
    words = ["Models", "learn", "senses", ".", "(", "c", ")", "2006", "APA", "press", "."]
    starts: list[bool | int | None] = [True, False, False, False, True]
    starts += [False, False, True, False, False, False]
    doc = Doc(nlp.vocab, words=words, sent_starts=starts)
    doc.spans[SPANS_KEY] = [Span(doc, 4, 8, label="copyright")]
    english = [SentenceLanguage("en", 1.0)] * len(list(doc.sents))
    removals, _ = find_removals(doc, {"copyright": "rest_of_sentence"}, english)
    assert removals == [Removal(doc[4].idx, doc[-1].idx + len(doc[-1]), "copyright")]


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


def test_annotate_abstracts_keeps_lower_case_figure_and_section_references_inside_sentences(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = (
        "The system is shown as a block schematic in fig. 1. and figs. 2-3, derived in "
        "sect. 4 from eq. 5 and eqs. 6-7 of the model. It is robust."
    )
    row = annotate(text, nlp, patterns)
    sentences = [row.text[sentence.start : sentence.end] for sentence in row.sentences]
    assert sentences[-1] == "It is robust."
    assert all(not sentence.startswith(("1.", "2-3", "4 ", "5 ", "6-7")) for sentence in sentences)


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
        ('He said that the model learns."Acquaintance with the data grows.', 'model learns."'),
    ],
    ids=["bracket", "citation", "year", "quote", "decimal", "two words", "question",
         "closing quote"],
)  # fmt: skip
def test_build_pipeline_starts_a_sentence_after_glued_punctuation_once_repaired(
    text: str, first: str, nlp: Language
) -> None:
    sentences = [sentence.text for sentence in nlp(repair_text(text)).sents]
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
            "Results agree with [12–14]. As shown in Fig. 2, the gain holds.",
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
        "range",
        "phonetic symbols and an abbreviation",
        "citation starting a sentence",
    ],
)
def test_build_pipeline_starts_no_sentence_inside_square_brackets(
    text: str, first: str, nlp: Language
) -> None:
    sentences = [sentence.text for sentence in nlp(text).sents]
    assert sentences == [first, text.removeprefix(first).lstrip()]


@pytest.mark.parametrize(
    ("text", "first"),
    [
        (
            "The effect lasted for 10 min. after the dose and then fell. It rose again later.",
            "The effect lasted for 10 min. after the dose and then fell.",
        ),
        (
            "It covers electronics, etc.) with management in mind. A survey followed.",
            "It covers electronics, etc.) with management in mind.",
        ),
        (
            "According to previous research in spatial reasoning (Toma &. Cross, 1998), it "
            "holds. A survey followed.",
            "According to previous research in spatial reasoning (Toma &. Cross, 1998), it holds.",
        ),
        (
            "Anxiety was evaluated [with the long form of the inventory developed for this "
            "study]. A survey followed.",
            "Anxiety was evaluated [with the long form of the inventory developed for this study].",
        ),
        (
            "The program ran for three weeks.4 Just prior to the end, it stopped.",
            "The program ran for three weeks.4",
        ),
        (
            "Responses were markedly decreased.4. Cocaine potentiated responses in both areas.",
            "Responses were markedly decreased.4.",
        ),
    ],
    ids=[
        "lowercase after a unit",
        "lowercase after etc.)",
        "inside parentheses",
        "after a long square bracket",
        "glued footnote number",
        "glued footnote number with its own full stop",
    ],
)
def test_build_pipeline_applies_the_run_2_sentence_rules(
    text: str, first: str, nlp: Language
) -> None:
    sentences = [sentence.text for sentence in nlp(text).sents]
    assert sentences == [first, text.removeprefix(first).lstrip()]


def test_build_pipeline_lets_a_word_with_an_inner_capital_start_a_sentence(nlp: Language) -> None:
    doc = nlp("We measured expression in every sample. mRNA levels rose after the treatment.")
    assert [sentence.text for sentence in doc.sents][-1] == "mRNA levels rose after the treatment."


def test_build_pipeline_splits_dashes_between_words_but_not_en_dashes(nlp: Language) -> None:
    tokens = [token.text for token in nlp("the theory—namely first--applied sleep–wake cycles")]
    assert tokens == [
        "the",
        "theory",
        "—",
        "namely",
        "first",
        "--",
        "applied",
        "sleep–wake",
        "cycles",
    ]


def test_build_pipeline_keeps_dotted_numbers_and_volumes_whole(nlp: Language) -> None:
    tokens = [token.text for token in nlp("See Vol.3 and Fig.4 with 3.5 points.")]
    assert {"Vol.3", "Fig.4", "3.5"} <= set(tokens)


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


def test_build_pipeline_ends_by_dropping_the_tensors(nlp: Language) -> None:
    assert nlp.pipe_names[-1] == "doc_cleaner"
    assert "span_ruler" not in nlp.pipe_names
    assert nlp("Attention helps models.").tensor is None


def attribute_rules(pipeline: Language) -> list[dict[str, Any]]:
    ruler = pipeline.get_pipe("attribute_ruler")
    assert isinstance(ruler, AttributeRuler)
    return ruler.patterns


def test_build_pipeline_drops_only_the_attribute_rules_that_need_the_parser(
    nlp: Language,
) -> None:
    stock = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
    rules = attribute_rules(nlp)
    assert len(rules) == len(attribute_rules(stock)) - 13
    assert not any("DEP" in token for rule in rules for p in rule["patterns"] for token in p)
    text = "It is not what I said, but they do have to be there. Y'all ain't seen nothin' yet."
    annotated = [(t.tag_, t.pos_, str(t.morph), t.lemma_) for t in nlp(text)]
    assert annotated == [(t.tag_, t.pos_, str(t.morph), t.lemma_) for t in stock(text)]


def test_build_pipeline_runs_named_entity_recognition_only_when_asked(nlp: Language) -> None:
    assert "ner" not in nlp.pipe_names
    assert not nlp("Ada Lovelace worked in London.").has_annotation("ENT_IOB")
    assert "ner" in build_pipeline(named_entity_recognition=True).pipe_names


def test_annotate_abstracts_reads_an_english_abstract_as_english(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = "Attention mechanisms improve machine translation. The effect is large."
    row = annotate(text, nlp, patterns)
    assert [sentence.language for sentence in row.sentences] == ["en", "en"]


def test_annotate_abstracts_removes_a_sentence_in_another_language(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = f"{ENGLISH_SENTENCE} {ENGLISH_SENTENCE} {ENGLISH_SENTENCE} {GERMAN_SENTENCE}"
    row = annotate(text, nlp, patterns)
    assert [sentence.language for sentence in row.sentences] == ["en", "en", "en", "de"]
    assert removed_texts(row)["not_english"] == [GERMAN_SENTENCE]
    assert row.dropped_reason is None


def test_annotate_abstracts_keeps_the_english_summary_of_a_bilingual_abstract(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = f"{ENGLISH_SENTENCE} {ENGLISH_SENTENCE} {ENGLISH_SENTENCE} " + " ".join(
        [GERMAN_SENTENCE] * 6
    )
    row = annotate(text, nlp, patterns)
    assert len(removed_texts(row)["not_english"]) == 6
    assert row.dropped_reason is None


def test_annotate_abstracts_drops_too_little_english_beside_another_language(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    text = f"{ENGLISH_SENTENCE} {GERMAN_SENTENCE} {GERMAN_SENTENCE} {GERMAN_SENTENCE}"
    assert annotate(text, nlp, patterns).dropped_reason == "not_english"


def test_annotate_abstracts_keeps_a_short_english_record_with_a_shorter_foreign_part(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    row = annotate(f"{ENGLISH_SENTENCE} {ENGLISH_SENTENCE} {GERMAN_SENTENCE}", nlp, patterns)
    assert removed_texts(row)["not_english"] == [GERMAN_SENTENCE]
    assert row.dropped_reason is None


def test_annotate_abstracts_keeps_a_short_english_abstract(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    assert annotate(ENGLISH_SENTENCE, nlp, patterns).dropped_reason is None


def test_annotate_abstracts_flags_a_sentence_without_punctuation(
    nlp: Language, patterns: CleaningPatterns
) -> None:
    run_on = " ".join(["models learn senses from attention in context"] * 22)
    row = annotate(f"{run_on}.", nlp, patterns)
    assert row.flags == [MALFORMED]
    assert row.dropped_reason is None


def test_annotation_cache_keeps_sentences_norms_lemmas_and_the_work_id(
    catalog: CatalogConfig, nlp: Language
) -> None:
    text = f"Abstract: Models use senses. See https://example.org now. {GERMAN_SENTENCE}"
    (original,) = annotate_texts([repaired(raw(text, work=7))], nlp)
    save_annotations(catalog, [original], part="ai_2010")
    (cached,) = load_annotations(catalog, "ai_2010", nlp.vocab)
    assert [sentence.text for sentence in cached.sents] == [s.text for s in original.sents]
    assert [token.norm_ for token in cached] == [token.norm_ for token in original]
    assert [token.lemma_ for token in cached] == [token.lemma_ for token in original]
    assert cached.user_data[WORK_ID] == "https://openalex.org/W7"
    assert SPANS_KEY not in cached.spans


def test_annotate_texts_raises_an_error_in_its_input(nlp: Language) -> None:
    def works_then_failure() -> Iterator[tuple[str, str]]:
        for work in range(4):
            yield f"https://openalex.org/W{work}", f"Models learn senses {work}."
        raise ValueError("cannot read the next text")

    with pytest.raises(ValueError, match="cannot read the next text"):
        list(annotate_texts(works_then_failure(), nlp, batch_size=2))
