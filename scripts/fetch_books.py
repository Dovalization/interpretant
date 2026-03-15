#!/usr/bin/env python3
"""Download freely available book texts into data/external/books/.

Run from the project root:
    uv run python scripts/fetch_books.py

Each book is fetched, cleaned, and written to the correct decade folder.
Already-present files are skipped unless --force is passed.
"""

from __future__ import annotations

import re
import sys
import tempfile
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BOOKS_DIR = Path("data/external/books")
RATE_LIMIT_SECONDS = 1.5  # polite delay between requests


# ---------------------------------------------------------------------------
# Fetch + clean helpers
# ---------------------------------------------------------------------------


def _fetch(url: str) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "interpretant-corpus-fetcher/1.0 (research)"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    upper = text.upper()
    start_idx = 0
    for marker in ("*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"):
        idx = upper.find(marker)
        if idx != -1:
            start_idx = text.find("\n", idx) + 1
            break
    end_idx = len(text)
    for marker in ("*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"):
        idx = upper.find(marker)
        if idx != -1:
            end_idx = idx
            break
    return text[start_idx:end_idx].strip()


class _HtmlToText(HTMLParser):
    """Minimal HTML → plain text converter."""

    _BLOCK_TAGS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "li", "blockquote", "tr"}
    _SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._depth += 1
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._depth == 0:
            self._parts.append(data)

    def result(self) -> str:
        raw = "".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _html_to_text(html: str) -> str:
    parser = _HtmlToText()
    parser.feed(html)
    return parser.result()


def _strip_ia_watermarks(text: str) -> str:
    """Remove Internet Archive download watermark lines."""
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if not re.match(r"^\s*Downloaded from\s+https?://", line, re.IGNORECASE)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def fetch_gutenberg(ebook_id: str) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
    return _strip_gutenberg(_fetch(url))


def fetch_html(url: str) -> str:
    return _html_to_text(_fetch(url))


def fetch_ia_text(url: str) -> str:
    return _strip_ia_watermarks(_fetch(url))


def fetch_pdf_url(url: str) -> str:
    """Download a PDF from url and extract text via PyMuPDF."""
    import fitz  # type: ignore[import-untyped]  # PyMuPDF

    req = urllib.request.Request(
        url, headers={"User-Agent": "interpretant-corpus-fetcher/1.0 (research)"}
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            tmp_path.write_bytes(resp.read())
        doc = fitz.open(str(tmp_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(pages)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Book definitions
# ---------------------------------------------------------------------------
# Each entry: (decade_label, filename, list_of_(label, url_or_callable))
# Multiple items in the last list are concatenated with a separator.

_SEP = "\n\n" + ("—" * 60) + "\n\n"

BOOKS: list[tuple[str, str, list[tuple[str, str]]]] = [
    # --- Epistemology foundations (public domain, filed under 1890s as anchor texts) ---
    (
        "1890s",
        "descartes_meditations.txt",
        [("Descartes Meditations on First Philosophy", "gutenberg:59")],
    ),
    (
        "1890s",
        "kant_critique_pure_reason.txt",
        [("Kant Critique of Pure Reason", "gutenberg:4280")],
    ),
    # --- 1820s ---
    (
        "1820s",
        "schopenhauer_world_as_will_vol1.txt",
        [
            (
                "Schopenhauer World as Will and Representation Vol 1",
                "ia:https://archive.org/download/worldaswillandre00scho/worldaswillandre00scho_djvu.txt",
            )
        ],
    ),
    (
        "1820s",
        "hegel_philosophy_right.txt",
        [("Hegel Philosophy of Right", "gutenberg:68951")],
    ),
    (
        "1820s",
        "hegel_phenomenology_spirit.txt",
        [("Hegel Phenomenology of Mind", "gutenberg:2082")],
    ),
    # --- 1830s ---
    (
        "1830s",
        "hegel_philosophy_history.txt",
        [("Hegel Lectures on the Philosophy of History", "gutenberg:48766")],
    ),
    (
        "1830s",
        "carlyle_sartor_resartus.txt",
        [("Carlyle Sartor Resartus", "gutenberg:1051")],
    ),
    (
        "1830s",
        "emerson_nature.txt",
        [("Emerson Nature", "gutenberg:29433")],
    ),
    (
        "1830s",
        "tocqueville_democracy_america.txt",
        [("Tocqueville Democracy in America", "gutenberg:816")],
    ),
    # --- 1840s ---
    (
        "1840s",
        "feuerbach_essence_christianity.txt",
        [("Feuerbach The Essence of Christianity", "gutenberg:5851")],
    ),
    (
        "1840s",
        "emerson_essays_first_series.txt",
        [("Emerson Essays First Series", "gutenberg:2944")],
    ),
    (
        "1840s",
        "kierkegaard_concept_anxiety.txt",
        [("Kierkegaard The Concept of Anxiety", "gutenberg:4596")],
    ),
    (
        "1840s",
        "kierkegaard_fear_trembling.txt",
        [("Kierkegaard Fear and Trembling", "gutenberg:36484")],
    ),
    (
        "1840s",
        "engels_condition_working_class.txt",
        [("Engels The Condition of the Working Class in England", "gutenberg:17306")],
    ),
    (
        "1840s",
        "mill_system_logic.txt",
        [("Mill A System of Logic", "gutenberg:26495")],
    ),
    (
        "1840s",
        "mill_principles_political_economy.txt",
        [("Mill Principles of Political Economy", "gutenberg:30107")],
    ),
    (
        "1840s",
        "marx_communist_manifesto.txt",
        [("Marx Engels Communist Manifesto", "gutenberg:61530")],
    ),
    # --- 1850s ---
    (
        "1850s",
        "melville_moby_dick.txt",
        [("Melville Moby-Dick", "gutenberg:2701")],
    ),
    (
        "1850s",
        "thoreau_walden.txt",
        [("Thoreau Walden", "gutenberg:205")],
    ),
    (
        "1850s",
        "darwin_origin_species.txt",
        [("Darwin On the Origin of Species", "gutenberg:1228")],
    ),
    (
        "1850s",
        "mill_on_liberty.txt",
        [("Mill On Liberty", "gutenberg:34901")],
    ),
    (
        "1850s",
        "marx_critique_political_economy.txt",
        [("Marx A Contribution to the Critique of Political Economy", "gutenberg:11458")],
    ),
    # --- 1860s ---
    (
        "1860s",
        "mill_utilitarianism.txt",
        [("Mill Utilitarianism", "gutenberg:11224")],
    ),
    (
        "1860s",
        "marx_capital_vol1.txt",
        [
            (
                "Marx Capital Vol 1",
                "ia:https://archive.org/download/capitalcritiqueo01marx/capitalcritiqueo01marx_djvu.txt",
            )
        ],
    ),
    (
        "1860s",
        "dostoevsky_notes_underground.txt",
        [("Dostoevsky Notes from Underground", "gutenberg:600")],
    ),
    (
        "1860s",
        "dostoevsky_crime_punishment.txt",
        [("Dostoevsky Crime and Punishment", "gutenberg:2554")],
    ),
    (
        "1860s",
        "galton_hereditary_genius.txt",
        [("Galton Hereditary Genius", "gutenberg:7124")],
    ),
    # --- 1870s ---
    (
        "1870s",
        "darwin_descent_man.txt",
        [("Darwin The Descent of Man", "gutenberg:2300")],
    ),
    (
        "1870s",
        "spencer_study_of_sociology.txt",
        [("Spencer The Study of Sociology", "gutenberg:37850")],
    ),
    (
        "1870s",
        "pater_renaissance.txt",
        [("Pater Studies in the History of the Renaissance", "gutenberg:2398")],
    ),
    (
        "1870s",
        "sidgwick_methods_ethics.txt",
        [("Sidgwick The Methods of Ethics", "gutenberg:46743")],
    ),
    (
        "1870s",
        "brentano_psychology_empirical.txt",
        [
            (
                "Brentano Psychology from an Empirical Standpoint",
                "ia:https://archive.org/download/psychologyfromanempirical00bren/psychologyfromanempirical00bren_djvu.txt",
            )
        ],
    ),
    # --- 1880s ---
    (
        "1880s",
        "frege_foundations_arithmetic.txt",
        [("Frege The Foundations of Arithmetic", "gutenberg:48312")],
    ),
    (
        "1880s",
        "spencer_man_versus_state.txt",
        [("Spencer The Man Versus the State", "gutenberg:6836")],
    ),
    (
        "1880s",
        "nietzsche_birth_tragedy.txt",
        [("Nietzsche The Birth of Tragedy", "gutenberg:51356")],
    ),
    (
        "1880s",
        "nietzsche_beyond_good_evil.txt",
        [("Nietzsche Beyond Good and Evil", "gutenberg:4363")],
    ),
    (
        "1880s",
        "nietzsche_genealogy_morality.txt",
        [("Nietzsche On the Genealogy of Morality", "gutenberg:998")],
    ),
    (
        "1880s",
        "nietzsche_zarathustra.txt",
        [("Nietzsche Thus Spoke Zarathustra", "gutenberg:1998")],
    ),
    (
        "1880s",
        "dostoevsky_brothers_karamazov.txt",
        [("Dostoevsky Brothers Karamazov", "gutenberg:28054")],
    ),
    (
        "1880s",
        "nietzsche_gay_science.txt",
        [("Nietzsche The Gay Science (The Joyful Wisdom)", "gutenberg:52881")],
    ),
    # --- 1890s ---
    (
        "1890s",
        "wilde_intentions.txt",
        [("Wilde Intentions", "gutenberg:887")],
    ),
    (
        "1890s",
        "tolstoy_what_is_art.txt",
        [("Tolstoy What is Art", "gutenberg:64908")],
    ),
    (
        "1890s",
        "frazer_golden_bough.txt",
        [("Frazer The Golden Bough", "gutenberg:3035")],
    ),
    # --- 1900s ---
    (
        "1900s",
        "poincare_science_hypothesis.txt",
        [("Poincare Science and Hypothesis", "gutenberg:39713")],
    ),
    (
        "1900s",
        "croce_aesthetic.txt",
        [("Croce Aesthetic", "gutenberg:12330")],
    ),
    (
        "1900s",
        "poincare_value_science.txt",
        [("Poincare The Value of Science", "gutenberg:39528")],
    ),
    (
        "1900s",
        "freud_interpretation_dreams.txt",
        [
            (
                "Freud The Interpretation of Dreams",
                "ia:https://archive.org/download/interpretationof00freu/interpretationof00freu_djvu.txt",
            )
        ],
    ),
    (
        "1900s",
        "freud_psychopathology_everyday_life.txt",
        [
            (
                "Freud Psychopathology of Everyday Life",
                "ia:https://archive.org/download/psychopathologyo00freu/psychopathologyo00freu_djvu.txt",
            )
        ],
    ),
    (
        "1900s",
        "veblen_theory_leisure_class.txt",
        [("Veblen Theory of the Leisure Class", "gutenberg:833")],
    ),
    (
        "1900s",
        "james_essays_radical_empiricism.txt",
        [("James Essays in Radical Empiricism", "gutenberg:5235")],
    ),
    # --- 1910s ---
    (
        "1900s",
        "james_pragmatism.txt",
        [("James Pragmatism", "gutenberg:5116")],
    ),
    (
        "1900s",
        "james_varieties_religious_experience.txt",
        [("James The Varieties of Religious Experience", "gutenberg:621")],
    ),
    (
        "1910s",
        "dewey_how_we_think.txt",
        [("Dewey How We Think", "gutenberg:37423")],
    ),
    (
        "1910s",
        "bell_art.txt",
        [("Bell Art", "gutenberg:16917")],
    ),
    (
        "1910s",
        "einstein_relativity.txt",
        [("Einstein Relativity", "html:https://www.gutenberg.org/cache/epub/5001/pg5001-images.html")],
    ),
    (
        "1910s",
        "russell_intro_mathematical_philosophy.txt",
        [("Russell Introduction to Mathematical Philosophy", "html:https://www.gutenberg.org/cache/epub/41654/pg41654-images.html")],
    ),
    (
        "1910s",
        "russell_knowledge_external_world.txt",
        [("Russell Our Knowledge of the External World", "gutenberg:37090")],
    ),
    (
        "1910s",
        "bergson_matter_memory.txt",
        [
            (
                "Bergson Matter and Memory",
                "ia:https://archive.org/download/mattermemory00berguoft/mattermemory00berguoft_djvu.txt",
            )
        ],
    ),
    (
        "1910s",
        "husserl_ideas_1.txt",
        [
            (
                "Husserl Ideas General Introduction to Pure Phenomenology",
                "ia:https://archive.org/download/ideasgeneralintro00huss/ideasgeneralintro00huss_djvu.txt",
            )
        ],
    ),
    (
        "1910s",
        "russell_problems_philosophy.txt",
        [("Russell The Problems of Philosophy", "gutenberg:5827")],
    ),
    (
        "1910s",
        "freud_totem_taboo.txt",
        [("Freud Totem and Taboo", "gutenberg:37850")],
    ),
    (
        "1910s",
        "kafka_metamorphosis.txt",
        [("Kafka The Metamorphosis", "gutenberg:5200")],
    ),
    # --- 1920s ---
    (
        "1920s",
        "eliot_sacred_wood.txt",
        [("Eliot The Sacred Wood", "gutenberg:34953")],
    ),
    (
        "1920s",
        "fry_vision_design.txt",
        [("Fry Vision and Design", "gutenberg:54154")],
    ),
    (
        "1920s",
        "kafka_trial.txt",
        [("Kafka The Trial", "gutenberg:7849")],
    ),
    (
        "1920s",
        "dewey_experience_nature.txt",
        [
            (
                "Dewey Experience and Nature",
                "ia:https://archive.org/download/experienceandnat029343mbp/experienceandnat029343mbp_djvu.txt",
            )
        ],
    ),
    (
        "1920s",
        "whitehead_process_reality.txt",
        [
            (
                "Whitehead Process and Reality",
                "ia:https://archive.org/download/processrealityes0000unse/processrealityes0000unse_djvu.txt",
            )
        ],
    ),
    (
        "1920s",
        "ogden_richards_meaning_of_meaning.txt",
        [
            (
                "Ogden Richards The Meaning of Meaning",
                "ia:https://archive.org/download/meaningofmeaning0000unse_k8y3/meaningofmeaning0000unse_k8y3_djvu.txt",
            )
        ],
    ),
    (
        "1920s",
        "cassirer_philosophy_symbolic_forms.txt",
        [
            (
                "Cassirer Philosophy of Symbolic Forms Vol 1",
                "pdf:https://uberty.org/wp-content/uploads/2015/12/Ernst_Cassirer_The_Philosophy_of_Symbolic_Forms1c.pdf",
            )
        ],
    ),
    # --- 1940s (additions) ---
    (
        "1940s",
        "wiener_human_use_human_beings.txt",
        [
            (
                "Wiener The Human Use of Human Beings",
                "pdf:https://monoskop.org/images/9/90/Wiener_Norbert_The_Human_Use_of_Human_Beings_1950.pdf",
            )
        ],
    ),
    # --- 1950s (additions) ---
    (
        "1950s",
        "turing_computing_machinery_intelligence.txt",
        [
            (
                "Turing Computing Machinery and Intelligence",
                "pdf:https://courses.cs.umbc.edu/471/papers/turing.pdf",
            )
        ],
    ),
    (
        "1950s",
        "austin_how_to_do_things_words.txt",
        [
            (
                "Austin How to Do Things with Words",
                "pdf:https://archive.org/download/austin-how-to-do-things-with-words/Austin%2C%20how%20to%20do%20things%20with%20words_text.pdf",
            )
        ],
    ),
    (
        "1950s",
        "levi_strauss_structural_anthropology.txt",
        [
            (
                "Levi-Strauss Structural Anthropology",
                "pdf:https://monoskop.org/images/e/e8/Levi-Strauss_Claude_Structural_Anthropology_1963.pdf",
            )
        ],
    ),
    # --- 1990s (additions) ---
    (
        "1990s",
        "castells_rise_network_society.txt",
        [
            (
                "Castells The Rise of the Network Society",
                "ia:https://archive.org/download/NetworkSocietyManuelCastells/Network%20Society%20-%20Manuel%20Castells_djvu.txt",
            )
        ],
    ),
    (
        "1990s",
        "stiegler_technics_time.txt",
        [
            (
                "Stiegler Technics and Time Vol 1",
                "pdf:https://monoskop.org/images/6/6f/Stiegler_Bernard_Technics_and_Time_1_The_Fault_of_Epimetheus.pdf",
            )
        ],
    ),
    # --- 2000s (additions) ---
    (
        "2000s",
        "lessig_code_v2.txt",
        [
            (
                "Lessig Code Version 2.0",
                "pdf:https://upload.wikimedia.org/wikipedia/commons/f/fd/Code_v2.pdf",
            )
        ],
    ),
    # --- 2010s (additions) ---
    (
        "2010s",
        "goodfellow_deep_learning.txt",
        [
            ("Goodfellow Deep Learning ch1 intro", "html:https://www.deeplearningbook.org/contents/intro.html"),
            ("Goodfellow Deep Learning ch2 linear algebra", "html:https://www.deeplearningbook.org/contents/linear_algebra.html"),
            ("Goodfellow Deep Learning ch3 probability", "html:https://www.deeplearningbook.org/contents/prob.html"),
            ("Goodfellow Deep Learning ch5 ml basics", "html:https://www.deeplearningbook.org/contents/ml.html"),
            ("Goodfellow Deep Learning ch6 deep networks", "html:https://www.deeplearningbook.org/contents/mlp.html"),
            ("Goodfellow Deep Learning ch7 regularization", "html:https://www.deeplearningbook.org/contents/regularization.html"),
            ("Goodfellow Deep Learning ch8 optimization", "html:https://www.deeplearningbook.org/contents/optimization.html"),
            ("Goodfellow Deep Learning ch9 convnets", "html:https://www.deeplearningbook.org/contents/convnets.html"),
            ("Goodfellow Deep Learning ch10 recurrent", "html:https://www.deeplearningbook.org/contents/rnn.html"),
            ("Goodfellow Deep Learning ch11 practical", "html:https://www.deeplearningbook.org/contents/guidelines.html"),
            ("Goodfellow Deep Learning ch15 representation", "html:https://www.deeplearningbook.org/contents/representation.html"),
        ],
    ),
    (
        "2010s",
        "sutton_barto_reinforcement_learning.txt",
        [
            (
                "Sutton Barto Reinforcement Learning 2nd ed",
                "pdf:http://incompleteideas.net/book/RLbook2020.pdf",
            )
        ],
    ),
    (
        "2010s",
        "pasquale_black_box_society.txt",
        [
            (
                "Pasquale The Black Box Society",
                "pdf:https://raley.english.ucsb.edu/wp-content/Engl800/Pasquale-blackbox.pdf",
            )
        ],
    ),
    # --- existing entries ---
    (
        "1910s",
        "saussure_course_general_linguistics.txt",
        [("Saussure Course in General Linguistics", "ia:https://archive.org/download/courseingenerall00saus/courseingenerall00saus_djvu.txt")],
    ),
    (
        "1910s",
        "dewey_democracy_education.txt",
        [("Dewey Democracy and Education", "ia:https://archive.org/download/cu31924032505756/cu31924032505756_djvu.txt")],
    ),
    (
        "1930s",
        "peirce_collected_papers.txt",
        [
            (
                "Peirce Collected Papers Vol 1",
                "ia:https://archive.org/download/collected-papers-of-charles-sanders-peirce-volume-1/"
                "Collected%20Papers%20of%20Charles%20Sanders%20Peirce%2C%20Volume%201_djvu.txt",
            )
        ],
    ),
    (
        "1930s",
        "heidegger_poetry_language_thought.txt",
        [
            (
                "Heidegger Poetry Language Thought",
                "ia:https://archive.org/download/ApolloHumanRightsBooks/"
                "76553938-Martin-Heidegger-Poetry-Language-Thought_djvu.txt",
            )
        ],
    ),
    (
        "1940s",
        "shannon_weaver_mathematical_theory_communication.txt",
        [
            (
                "Shannon Weaver Mathematical Theory of Communication",
                "ia:https://archive.org/download/in.ernet.dli.2015.503815/"
                "2015.503815.The-Mathematical_djvu.txt",
            )
        ],
    ),
    (
        "1950s",
        "barthes_mythologies.txt",
        [
            (
                "Barthes Mythologies",
                "ia:https://archive.org/download/barthes-roland-mythologies-en-1972/"
                "Barthes_Roland_Mythologies_EN_1972_djvu.txt",
            )
        ],
    ),
    (
        "1970s",
        "foucault_discipline_punish.txt",
        [
            (
                "Foucault Discipline and Punish",
                "ia:https://archive.org/download/DisciplinePunishTheBirthOfThePrison/"
                "Discipline%20%26%20Punish-The%20Birth%20of%20the%20Prison_djvu.txt",
            )
        ],
    ),
    (
        "1980s",
        "deleuze_guattari_thousand_plateaus.txt",
        [
            (
                "Deleuze Guattari A Thousand Plateaus",
                "ia:https://archive.org/download/AThousandPlateaus_20180115/"
                "A-Thousand-Plateaus_djvu.txt",
            )
        ],
    ),
    (
        "1890s",
        "james_principles_psychology.txt",
        [
            ("James PoP Vol 1", "gutenberg:57628"),
            ("James PoP Vol 2", "gutenberg:57634"),
        ],
    ),
    (
        "1900s",
        "bergson_creative_evolution.txt",
        [("Bergson Creative Evolution", "gutenberg:26163")],
    ),
    (
        "1920s",
        "russell_analysis_mind.txt",
        [("Russell Analysis of Mind", "gutenberg:2529")],
    ),
    (
        "1920s",
        "wittgenstein_tractatus.txt",
        [
            (
                "Wittgenstein Tractatus",
                "html:https://standardebooks.org/ebooks/ludwig-wittgenstein/"
                "tractatus-logico-philosophicus/c-k-ogden/text/single-page",
            )
        ],
    ),
    (
        "1950s",
        "benjamin_illuminations.txt",
        [
            (
                "Benjamin Work of Art",
                "html:https://www.marxists.org/reference/subject/philosophy/works/ge/benjamin.htm",
            ),
            (
                "Benjamin Theses on History",
                "html:https://www.marxists.org/reference/archive/benjamin/1940/history.htm",
            ),
            (
                "Benjamin Author as Producer",
                "html:https://www.marxists.org/reference/archive/benjamin/1970/author-producer.htm",
            ),
        ],
    ),
    (
        "1980s",
        "baudrillard_simulacra_simulation.txt",
        [
            (
                "Baudrillard Simulacra",
                "ia:https://archive.org/download/"
                "simulacra-and-simulation-jean-baudrillard-translated-by-sheila-faria-glaser/"
                "Simulacra%20and%20Simulation%20-%20by%20Jean%20Baudrillard%3B"
                "%20translated%20by%20Sheila%20Faria%20Glaser_djvu.txt",
            )
        ],
    ),
    (
        "1990s",
        "mitchell_city_bits.txt",
        [
            (
                "Mitchell City of Bits",
                "ia:https://archive.org/download/"
                "mit_press_book_9780262279956/"
                "mit_press_book_9780262279956_djvu.txt",
            )
        ],
    ),
    (
        "1940s",
        "wiener_cybernetics.txt",
        [
            (
                "Wiener Cybernetics",
                "ia:https://archive.org/download/"
                "cybernetics-norbert-wiener/"
                "CYBERNETICS%20-%20NORBERT%20WIENER_djvu.txt",
            )
        ],
    ),
    (
        "1940s",
        "ryle_concept_mind.txt",
        [
            (
                "Ryle Concept of Mind",
                "ia:https://archive.org/download/"
                "gilbert-ryle-concept-of-mind/"
                "Gilbert%20Ryle%2C%20Concept%20of%20Mind_djvu.txt",
            )
        ],
    ),
    (
        "1960s",
        "kuhn_structure_scientific_revolutions.txt",
        [
            (
                "Kuhn Structure of Scientific Revolutions",
                "ia:https://archive.org/download/"
                "the-structure-of-scientific-revolutions-thomas-s.-kuhn/"
                "The%20Structure%20of%20Scientific%20Revolutions%20-%20Thomas%20S.%20Kuhn_djvu.txt",
            )
        ],
    ),
    (
        "1960s",
        "mcluhan_understanding_media.txt",
        [
            (
                "McLuhan Understanding Media",
                "ia:https://archive.org/download/"
                "ETC0624/"
                "Understanding%20Media%20-%20The%20Extensions%20of%20Man_djvu.txt",
            )
        ],
    ),
    (
        "1970s",
        "bateson_ecology_mind.txt",
        [
            (
                "Bateson Steps to an Ecology of Mind",
                "pdf:https://monoskop.org/images/b/bf/"
                "Bateson_Gregory_Steps_to_an_Ecology_of_Mind.pdf",
            )
        ],
    ),
    (
        "1990s",
        "haraway_simians_cyborgs_women.txt",
        [
            (
                "Haraway Simians Cyborgs and Women",
                "ia:https://archive.org/download/"
                "harawaydonnajsimianscyborgsandwomenthereinventionofnature/"
                "Haraway_Donna_J_Simians_Cyborgs_and_Women_The_Reinvention_of_Nature_djvu.txt",
            )
        ],
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fetch_part(scheme_url: str) -> str:
    if scheme_url.startswith("gutenberg:"):
        ebook_id = scheme_url.split(":", 1)[1]
        return fetch_gutenberg(ebook_id)
    elif scheme_url.startswith("html:"):
        url = scheme_url.split(":", 1)[1]
        return fetch_html(url)
    elif scheme_url.startswith("ia:"):
        url = scheme_url.split(":", 1)[1]
        return fetch_ia_text(url)
    elif scheme_url.startswith("pdf:"):
        url = scheme_url.split(":", 1)[1]
        return fetch_pdf_url(url)
    else:
        raise ValueError(f"Unknown scheme in: {scheme_url!r}")


def main(force: bool = False) -> None:
    if not BOOKS_DIR.exists():
        print(f"ERROR: {BOOKS_DIR} does not exist. Run from project root.", file=sys.stderr)
        sys.exit(1)

    total = len(BOOKS)
    for i, (decade, filename, parts) in enumerate(BOOKS, 1):
        dest = BOOKS_DIR / decade / filename
        print(f"[{i}/{total}] {filename}", end="  ")

        if dest.exists() and not force:
            words = len(dest.read_text(encoding="utf-8").split())
            print(f"skip (already present, {words:,} words)")
            continue

        segments: list[str] = []
        for label, scheme_url in parts:
            print(f"\n         fetching {label} ...", end="", flush=True)
            try:
                text = _fetch_part(scheme_url)
                segments.append(text)
                words = len(text.split())
                print(f" {words:,} words", end="")
            except Exception as exc:  # noqa: BLE001
                print(f" ERROR: {exc}", file=sys.stderr)
                segments.append(f"[FETCH ERROR: {label}: {exc}]")
            time.sleep(RATE_LIMIT_SECONDS)

        combined = _SEP.join(segments)
        dest.write_text(combined, encoding="utf-8")
        total_words = len(combined.split())
        print(f"\n         → saved {total_words:,} words to {dest}")

    print("\nDone. Run `uv run interpretant corpus books validate` to check status.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
