"""Fetch books from Project Gutenberg and Internet Archive.

Downloads text files, strips boilerplate, saves to decade folders,
and updates data/external/books/manifest.json.

Run: uv run python scripts/fetch_open_access.py
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser

try:
    import fitz  # PyMuPDF

    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

BOOKS_DIR = pathlib.Path("data/external/books")
MANIFEST_PATH = pathlib.Path("data/external/books/manifest.json")


# ─── Book definitions ─────────────────────────────────────────────────────────
# (decade, output_filename, author, title, year, fields, source, source_id)
# source: "gutenberg" | "archive" | "gutenberg_merge" (list of ids to concatenate)

BOOKS: list[tuple] = [
    # 1820s
    ("1820s", "brown_lectures_philosophy_mind.txt",
     "Thomas Brown", "Lectures on the Philosophy of the Human Mind, Vol. 1",
     1820, ["psychology", "philosophy_of_mind"], "gutenberg", 43116),
    ("1820s", "shelley_frankenstein.txt",
     "Mary Wollstonecraft Shelley", "Frankenstein; or, The Modern Prometheus",
     1818, ["philosophy_of_mind"], "gutenberg", 84),
    ("1820s", "stendhal_on_love.txt",
     "Stendhal", "On Love",
     1822, ["psychology", "phenomenology"], "gutenberg", 53720),
    ("1820s", "de_quincey_confessions_opium_eater.txt",
     "Thomas De Quincey", "Confessions of an English Opium-Eater",
     1821, ["phenomenology"], "gutenberg", 2040),

    # 1830s
    ("1830s", "darwin_voyage_beagle.txt",
     "Charles Darwin", "The Voyage of the Beagle",
     1839, ["neuroscience"], "gutenberg", 944),
    ("1830s", "herschel_preliminary_discourse.txt",
     "John Herschel", "A Preliminary Discourse on the Study of Natural Philosophy",
     1830, ["cognitive_science"], "gutenberg", 54897),
    ("1830s", "martineau_society_in_america.txt",
     "Harriet Martineau", "Society in America (Vols. 1 & 2)",
     1837, ["sociology"], "gutenberg_merge", [52621, 52685]),
    ("1830s", "babbage_reflections_decline_science.txt",
     "Charles Babbage", "Reflections on the Decline of Science in England",
     1830, ["systems_theory", "cognitive_science"], "gutenberg", 1216),
    ("1830s", "quetelet_treatise_on_man.txt",
     "Adolphe Quetelet", "A Treatise on Man",
     1835, ["sociology", "cognitive_science"], "archive", "treatiseonmandev00quet"),

    # 1840s
    ("1840s", "carlyle_on_heroes.txt",
     "Thomas Carlyle", "On Heroes, Hero-Worship, and the Heroic in History",
     1841, ["sociology", "political_philosophy"], "gutenberg", 1091),
    ("1840s", "stirner_ego_and_his_own.txt",
     "Max Stirner", "The Ego and His Own",
     1844, ["philosophy_of_mind", "political_philosophy"], "gutenberg", 34580),
    ("1840s", "humboldt_cosmos_vol1.txt",
     "Alexander von Humboldt",
     "Cosmos: A Sketch of the Physical Description of the Universe, Vol. 1",
     1845, ["systems_theory"], "gutenberg", 14565),
    ("1840s", "poe_eureka.txt",
     "Edgar Allan Poe", "Eureka: A Prose Poem",
     1848, ["philosophy_of_mind", "systems_theory"], "gutenberg", 32037),
    ("1840s", "douglass_narrative.txt",
     "Frederick Douglass", "Narrative of the Life of Frederick Douglass",
     1845, ["sociology", "political_philosophy"], "gutenberg", 23),
    ("1840s", "kierkegaard_either_or_pt1.txt",
     "Søren Kierkegaard", "Either/Or, Part I",
     1843, ["phenomenology", "philosophy_of_mind"],
     "archive_epub", "either-or-part-1-kierkegaards-writings_compress"),
    ("1840s", "kierkegaard_philosophical_fragments.txt",
     "Søren Kierkegaard", "Philosophical Fragments",
     1844, ["philosophy_of_mind"],
     "archive_pdf", "philosophical-fragments-or-a-fragment-of-philosophy-kierkegaard-soren-1962"),

    # 1850s
    ("1850s", "whitman_leaves_of_grass.txt",
     "Walt Whitman", "Leaves of Grass",
     1855, ["phenomenology"], "gutenberg", 1322),
    ("1850s", "douglass_my_bondage_my_freedom.txt",
     "Frederick Douglass", "My Bondage and My Freedom",
     1855, ["sociology", "political_philosophy"], "gutenberg", 202),
    ("1850s", "marx_eighteenth_brumaire.txt",
     "Karl Marx", "The Eighteenth Brumaire of Louis Bonaparte",
     1852, ["political_philosophy", "sociology"], "gutenberg", 1346),
    ("1850s", "mayhew_london_labour_vol1.txt",
     "Henry Mayhew", "London Labour and the London Poor, Vol. 1",
     1851, ["sociology"], "gutenberg", 55998),
    ("1850s", "spencer_social_statics.txt",
     "Herbert Spencer", "Social Statics",
     1851, ["sociology", "political_philosophy"], "archive", "socialstaticsorc0000herb"),

    # 1860s
    ("1860s", "mill_subjection_of_women.txt",
     "John Stuart Mill", "The Subjection of Women",
     1869, ["sociology", "political_philosophy"], "gutenberg", 27083),
    ("1860s", "spencer_principles_biology_vol1.txt",
     "Herbert Spencer", "The Principles of Biology, Vol. 1",
     1864, ["neuroscience"], "gutenberg", 54612),
    ("1860s", "bernard_introduction_experimental_medicine.txt",
     "Claude Bernard", "An Introduction to the Study of Experimental Medicine",
     1865, ["neuroscience"], "archive", "b21270557"),
    ("1860s", "lyell_geological_evidences_antiquity.txt",
     "Charles Lyell", "The Geological Evidences of the Antiquity of Man",
     1863, ["neuroscience"], "archive", "geologicalevide03lyelgoog"),
    ("1860s", "marx_wages_price_profit.txt",
     "Karl Marx", "Wages, Price and Profit",
     1865, ["political_philosophy", "sociology"], "archive", "k.-marx"),

    # 1870s
    ("1870s", "nietzsche_thoughts_out_of_season_pt1.txt",
     "Friedrich Nietzsche", "Thoughts Out of Season, Part I",
     1873, ["philosophy_of_mind", "phenomenology"], "gutenberg", 51710),
    ("1870s", "nietzsche_thoughts_out_of_season_pt2.txt",
     "Friedrich Nietzsche", "Thoughts Out of Season, Part II",
     1876, ["philosophy_of_mind", "phenomenology"], "gutenberg", 38226),
    ("1870s", "nietzsche_human_all_too_human.txt",
     "Friedrich Nietzsche", "Human, All Too Human",
     1878, ["philosophy_of_mind"], "gutenberg", 38145),
    ("1870s", "haeckel_history_of_creation_vol1.txt",
     "Ernst Haeckel", "The History of Creation, Vol. 1",
     1876, ["neuroscience"], "gutenberg", 40472),
    ("1870s", "mill_autobiography.txt",
     "John Stuart Mill", "Autobiography",
     1873, ["sociology", "philosophy_of_mind"], "gutenberg", 10378),
    ("1870s", "taine_on_intelligence.txt",
     "Hippolyte Taine", "On Intelligence",
     1871, ["cognitive_science", "psychology"], "archive", "onintelligence00tain"),
    ("1870s", "galton_english_men_of_science.txt",
     "Francis Galton", "English Men of Science",
     1874, ["cognitive_science", "sociology"], "archive", "englishmenofscie00galtuoft"),

    # 1880s
    ("1880s", "huxley_science_and_culture.txt",
     "T.H. Huxley", "Science and Culture, and Other Essays",
     1881, ["cognitive_science"], "gutenberg", 52344),
    ("1880s", "mach_analysis_of_sensations.txt",
     "Ernst Mach", "Contributions to the Analysis of the Sensations",
     1886, ["phenomenology", "cognitive_science"], "archive", "contributionstoa00machrich"),
    ("1880s", "romanes_mental_evolution_animals.txt",
     "George Romanes", "Mental Evolution in Animals",
     1883, ["neuroscience", "cognitive_science"], "archive", "mentalevolutioninan00roma"),

    # 1890s
    ("1890s", "santayana_sense_of_beauty.txt",
     "George Santayana", "The Sense of Beauty",
     1896, ["phenomenology"], "gutenberg", 26842),
    ("1890s", "nietzsche_twilight_idols.txt",
     "Friedrich Nietzsche", "Twilight of the Idols / The Antichrist",
     1889, ["philosophy_of_mind", "phenomenology"], "gutenberg", 52263),
    ("1890s", "james_psychology_briefer_course.txt",
     "William James", "Psychology: Briefer Course",
     1892, ["psychology", "cognitive_science"], "gutenberg", 55262),
    ("1890s", "durkheim_division_of_labor.txt",
     "Émile Durkheim", "The Division of Labor in Society",
     1893, ["sociology"], "archive", "in.ernet.dli.2015.233884"),
    ("1890s", "pearson_grammar_of_science.txt",
     "Karl Pearson", "The Grammar of Science",
     1892, ["cognitive_science"], "archive", "grammarofscience00pearrich"),

    # 1900s
    ("1900s", "moore_principia_ethica.txt",
     "G.E. Moore", "Principia Ethica",
     1903, ["philosophy_of_mind"], "gutenberg", 53430),
    ("1900s", "james_pluralistic_universe.txt",
     "William James", "A Pluralistic Universe",
     1909, ["philosophy_of_mind", "phenomenology"], "gutenberg", 11984),
    ("1900s", "bergson_introduction_to_metaphysics.txt",
     "Henri Bergson", "An Introduction to Metaphysics",
     1903, ["phenomenology", "philosophy_of_mind"], "archive", "introductiontome0000henr"),
    ("1900s", "russell_principles_of_mathematics.txt",
     "Bertrand Russell", "The Principles of Mathematics",
     1903, ["cognitive_science"], "archive", "principlesofmath01russ"),

    # 1910s
    ("1910s", "jung_psychology_unconscious.txt",
     "C.G. Jung", "Psychology of the Unconscious",
     1912, ["psychology"], "gutenberg", 65903),
    ("1910s", "freud_general_introduction_psychoanalysis.txt",
     "Sigmund Freud", "A General Introduction to Psychoanalysis",
     1917, ["psychology"], "gutenberg", 38219),
    ("1910s", "dewey_essays_experimental_logic.txt",
     "John Dewey", "Essays in Experimental Logic",
     1916, ["cognitive_science", "philosophy_of_mind"], "gutenberg", 40794),
    ("1910s", "russell_why_men_fight.txt",
     "Bertrand Russell", "Why Men Fight",
     1916, ["political_philosophy", "sociology"], "gutenberg", 55610),

    # 1920s
    ("1920s", "watson_behaviorism.txt",
     "John B. Watson", "Behaviorism",
     1924, ["psychology", "cognitive_science"], "archive", "behaviorism00wats"),

    # 1930s
    ("1930s", "benjamin_work_of_art.txt",
     "Walter Benjamin", "The Work of Art in the Age of Mechanical Reproduction",
     1935, ["sociology"], "archive_pdf_url",
     "https://archive.org/download/the-work-of-art-in-the-age-of-mechanical-reproduction-walter-benjamin/"
     "The%20Work%20Of%20Art%20In%20The%20Age%20Of%20Mechanical%20Reproduction%20-%20Walter%20Benjamin.pdf"),
    ("1930s", "ayer_language_truth_logic.txt",
     "A.J. Ayer", "Language, Truth and Logic",
     1936, ["philosophy_of_mind", "cognitive_science"], "archive_pdf_url",
     "https://archive.org/download/ayer-language-truth-and-logic/Ayer%2C%20Language%2C%20Truth%20And%20Logic.pdf"),
    ("1930s", "mannheim_ideology_utopia.txt",
     "Karl Mannheim", "Ideology and Utopia",
     1929, ["sociology", "political_philosophy"], "archive", "in.ernet.dli.2015.187027"),
    ("1930s", "sartre_transcendence_ego.txt",
     "Jean-Paul Sartre", "The Transcendence of the Ego",
     1936, ["phenomenology", "philosophy_of_mind"],
     "archive_pdf_url",
     "https://archive.org/download/the-transcendence-of-the-ego-a-sketch-for-a-sartre/"
     "The%20Transcendence%20Of%20The%20Ego%20A%20Sketch%20For%20A%20Sartre.pdf"),

    # 1940s
    ("1940s", "cassirer_essay_on_man.txt",
     "Ernst Cassirer", "An Essay on Man",
     1944, ["philosophy_of_mind", "phenomenology"], "archive", "in.ernet.dli.2015.236097"),

    # 1950s
    ("1950s", "skinner_verbal_behavior.txt",
     "B.F. Skinner", "Verbal Behavior",
     1957, ["psychology", "cognitive_science"], "archive_epub",
     "VerbalBehavior"),

    # ── BATCH 2 ───────────────────────────────────────────────────────────────

    # 1820s
    ("1820s", "bell_anatomy_philosophy_expression.txt",
     "Charles Bell", "The Anatomy and Philosophy of Expression",
     1824, ["neuroscience", "phenomenology"],
     "archive", "BellCharlesTheAnatomyAndPhilosophyOfExpression"),
    ("1820s", "combe_system_phrenology.txt",
     "George Combe", "A System of Phrenology",
     1825, ["neuroscience", "psychology"],
     "archive", "systehro00comb"),
    ("1820s", "brown_lectures_philosophy_mind_vol2.txt",
     "Thomas Brown", "Lectures on the Philosophy of the Human Mind, Vol. 2",
     1820, ["psychology", "philosophy_of_mind"],
     "archive", "lecturesonphilo09browgoog"),
    ("1820s", "maine_de_biran_influence_of_habit.txt",
     "Maine de Biran", "The Influence of Habit on the Faculty of Thinking",
     1803, ["psychology", "philosophy_of_mind"],
     "archive", "influenceofhabit00main"),
    ("1820s", "hamilton_lectures_metaphysics_logic.txt",
     "William Hamilton", "Lectures on Metaphysics and Logic",
     1859, ["philosophy_of_mind", "cognitive_science"],
     "archive", "lecturesonmetaph01hamiiala"),
    ("1820s", "cousin_introduction_history_philosophy.txt",
     "Victor Cousin", "Introduction to the History of Philosophy",
     1828, ["philosophy_of_mind"],
     "archive", "introductiontoh00cousgoog"),

    # 1830s
    ("1830s", "martineau_how_to_observe.txt",
     "Harriet Martineau", "How to Observe Morals and Manners",
     1838, ["sociology"],
     "gutenberg", 33944),
    ("1830s", "martineau_illustrations_political_economy.txt",
     "Harriet Martineau", "Illustrations of Political Economy (selection)",
     1832, ["sociology", "political_philosophy"],
     "gutenberg_merge", [11196, 11197, 11198]),
    ("1830s", "mill_essays_unsettled_questions.txt",
     "John Stuart Mill", "Essays on Some Unsettled Questions of Political Economy",
     1844, ["political_philosophy", "sociology"],
     "gutenberg", 12004),
    ("1830s", "whewell_philosophy_inductive_sciences.txt",
     "William Whewell", "The Philosophy of the Inductive Sciences",
     1840, ["cognitive_science"],
     "archive", "philosophyinduc00whewgoog"),
    ("1830s", "tocqueville_memoir_on_pauperism.txt",
     "Alexis de Tocqueville", "Memoir on Pauperism",
     1835, ["sociology", "political_philosophy"],
     "archive", "tocqueville-1st-letter-pauperism-1835"),
    ("1830s", "brewster_letters_natural_magic.txt",
     "David Brewster", "Letters on Natural Magic",
     1832, ["cognitive_science", "systems_theory"],
     "archive", "lettersonnatural00brew"),

    # 1840s
    ("1840s", "kierkegaard_philosophical_fragments.txt",
     "Søren Kierkegaard", "Philosophical Fragments",
     1844, ["philosophy_of_mind"],
     "archive", "soren-kierkegaard-philosophical-fragments-or-a-fragment-of-philosophy"),
    ("1840s", "marx_german_ideology.txt",
     "Karl Marx", "The German Ideology",
     1846, ["political_philosophy", "sociology"],
     "marxists", "https://www.marxists.org/archive/marx/works/1845/german-ideology/index.htm"),

    # 1850s
    ("1850s", "comte_catechism_positive_religion.txt",
     "Auguste Comte", "The Catechism of Positive Religion",
     1852, ["sociology"],
     "archive", "catechismofposit01comt"),
    ("1850s", "lewes_biographical_history_philosophy.txt",
     "George Henry Lewes", "The Biographical History of Philosophy",
     1857, ["philosophy_of_mind", "cognitive_science"],
     "archive", "biographicalhist00lewerich"),
    ("1850s", "mill_dissertations_discussions_vol1.txt",
     "John Stuart Mill", "Dissertations and Discussions, Vol. 1",
     1859, ["philosophy_of_mind", "political_philosophy"],
     "archive", "dissertationsand01milluoft"),
    ("1850s", "harriet_taylor_mill_enfranchisement_women.txt",
     "Harriet Taylor Mill", "The Enfranchisement of Women",
     1851, ["sociology", "political_philosophy"],
     "gutenberg", 73404),
    ("1850s", "chambers_vestiges_natural_history.txt",
     "Robert Chambers", "Vestiges of the Natural History of Creation",
     1844, ["neuroscience"],
     "gutenberg", 18559),
    ("1850s", "buckle_history_civilisation_vol2.txt",
     "Henry Thomas Buckle", "History of Civilisation in England, Vol. 2",
     1861, ["sociology"],
     "archive", "historyofciviliz033013mbp"),

    # 1860s
    ("1860s", "marx_wages_price_profit.txt",
     "Karl Marx", "Wages, Price and Profit",
     1865, ["political_philosophy", "sociology"],
     "marxists", "https://www.marxists.org/archive/marx/works/1865/value-price-profit/index.htm"),
    ("1860s", "von_hartmann_philosophy_unconscious.txt",
     "Eduard von Hartmann", "Philosophy of the Unconscious",
     1869, ["philosophy_of_mind", "psychology"],
     "archive", "philosophyofunco01hart"),
    ("1860s", "spencer_education.txt",
     "Herbert Spencer", "Education: Intellectual, Moral, and Physical",
     1861, ["cognitive_science", "sociology"],
     "archive", "spencereducation00spen"),
    ("1860s", "tyndall_fragments_of_science.txt",
     "John Tyndall", "Fragments of Science",
     1871, ["cognitive_science", "systems_theory"],
     "gutenberg", 24527),
    ("1860s", "tylor_researches_early_history_mankind.txt",
     "Edward Tylor", "Researches into the Early History of Mankind",
     1865, ["anthropology", "sociology"],
     "archive", "researchesintoea00tylouoft"),
    ("1860s", "maudsley_physiology_of_mind.txt",
     "Henry Maudsley", "The Physiology of Mind",
     1876, ["neuroscience", "psychology"],
     "archive", "39002086171213.med.yale.edu"),

    # 1870s
    ("1870s", "marx_civil_war_in_france.txt",
     "Karl Marx", "The Civil War in France",
     1871, ["political_philosophy", "sociology"],
     "marxists", "https://www.marxists.org/archive/marx/works/1871/civil-war-france/index.htm"),
    ("1870s", "spencer_principles_sociology_vol1.txt",
     "Herbert Spencer", "The Principles of Sociology, Vol. 1",
     1876, ["sociology", "systems_theory"],
     "archive", "principlessocio01spengoog"),

    # 1880s
    ("1880s", "leslie_stephen_science_ethics.txt",
     "Leslie Stephen", "The Science of Ethics",
     1882, ["philosophy_of_mind", "sociology"],
     "archive", "thescienceofethi00stepuoft"),
    ("1880s", "royce_religious_aspect_philosophy.txt",
     "Josiah Royce", "The Religious Aspect of Philosophy",
     1885, ["philosophy_of_mind"],
     "archive", "religiousaspecto0000royc"),
    ("1880s", "romanes_mental_evolution_man.txt",
     "George Romanes", "Mental Evolution in Man",
     1888, ["neuroscience", "cognitive_science"],
     "gutenberg", 50382),

    # 1890s
    ("1890s", "bradley_appearance_and_reality.txt",
     "Francis Herbert Bradley", "Appearance and Reality",
     1893, ["philosophy_of_mind"],
     "archive", "appearanceandre00bradgoog"),

    # 1910s
    ("1910s", "trotter_instincts_herd.txt",
     "Wilfred Trotter", "Instincts of the Herd in Peace and War",
     1916, ["sociology", "psychology"],
     "gutenberg", 53453),

    # 1930s
    ("1930s", "ayer_language_truth_logic.txt",
     "A.J. Ayer", "Language, Truth and Logic",
     1936, ["philosophy_of_mind", "cognitive_science"],
     "archive", "ayer-language-truth-and-logic"),
    ("1930s", "mannheim_ideology_utopia.txt",
     "Karl Mannheim", "Ideology and Utopia",
     1929, ["sociology", "political_philosophy"],
     "archive",
     "ideology-and-utopia-an-introduction-to-the-sociology-of-knowledge-karl-mannheim"),
    ("1930s", "sartre_transcendence_ego.txt",
     "Jean-Paul Sartre", "The Transcendence of the Ego",
     1936, ["phenomenology", "philosophy_of_mind"],
     "archive", "thetranscendenceoftheego"),
    ("1930s", "lewin_dynamic_theory_personality.txt",
     "Kurt Lewin", "A Dynamic Theory of Personality",
     1935, ["psychology", "cognitive_science"],
     "archive", "in.ernet.dli.2015.18586"),

    # ── BATCH 3 ───────────────────────────────────────────────────────────────

    # 1820s — Bell (correct identifier)
    ("1820s", "bell_anatomy_philosophy_expression.txt",
     "Charles Bell", "The Anatomy and Philosophy of Expression",
     1824, ["neuroscience", "phenomenology"],
     "archive", "anatomyphilos00belluoft"),

    # 1830s — Tocqueville merged from two letters
    ("1830s", "tocqueville_memoir_on_pauperism.txt",
     "Alexis de Tocqueville", "Memoir on Pauperism",
     1835, ["sociology", "political_philosophy"],
     "archive_merge",
     ["tocqueville-1st-letter-pauperism-1835", "tocqueville-2nd-letter-pauperism-1837"]),

    # 1840s — Marx German Ideology via Marxists.org chapter crawl
    ("1840s", "marx_german_ideology.txt",
     "Karl Marx", "The German Ideology",
     1846, ["political_philosophy", "sociology"],
     "marxists_toc", "https://www.marxists.org/archive/marx/works/1845/german-ideology/"),

    # 1860s — Marx Wages, Price and Profit via Marxists.org
    ("1860s", "marx_wages_price_profit.txt",
     "Karl Marx", "Wages, Price and Profit",
     1865, ["political_philosophy", "sociology"],
     "marxists_toc", "https://www.marxists.org/archive/marx/works/1865/value-price-profit/"),

    # 1870s — Marx Civil War in France via Archive
    ("1870s", "marx_civil_war_in_france.txt",
     "Karl Marx", "The Civil War in France",
     1871, ["political_philosophy", "sociology"],
     "archive", "the-civil-war-in-france-marx"),

    # 1890s — Freud & Breuer Studies on Hysteria (English 1936 Brill translation)
    ("1890s", "freud_breuer_studies_hysteria.txt",
     "Sigmund Freud and Josef Breuer", "Studies on Hysteria",
     1895, ["psychology", "neuroscience"],
     "archive", "breuer-freud-1936-hysteria"),

    # 1890s — Durkheim Division of Labor (retry with second identifier)
    ("1890s", "durkheim_division_of_labor.txt",
     "Émile Durkheim", "The Division of Labor in Society",
     1893, ["sociology"],
     "archive", "dli.ernet.6156"),

    # 1900s — Durkheim Moral Education
    ("1900s", "durkheim_moral_education.txt",
     "Émile Durkheim", "Moral Education",
     1902, ["sociology"],
     "archive", "dli.ernet.28979"),

    # 1910s — Weber Economy and Society
    ("1910s", "weber_economy_and_society.txt",
     "Max Weber", "Economy and Society",
     1922, ["sociology", "political_philosophy"],
     "archive", "MaxWeberEconomyAndSociety"),

    # 1930s — retry with confirmed-good identifiers
    ("1930s", "ayer_language_truth_logic.txt",
     "A.J. Ayer", "Language, Truth and Logic",
     1936, ["philosophy_of_mind", "cognitive_science"],
     "archive", "AlfredAyer"),

    ("1930s", "mannheim_ideology_utopia.txt",
     "Karl Mannheim", "Ideology and Utopia",
     1929, ["sociology", "political_philosophy"],
     "archive", "ideologyutopiain00mann"),

    ("1930s", "sartre_transcendence_ego.txt",
     "Jean-Paul Sartre", "The Transcendence of the Ego",
     1936, ["phenomenology", "philosophy_of_mind"],
     "archive", "thetranscendenceoftheego"),

    ("1930s", "lewin_dynamic_theory_personality.txt",
     "Kurt Lewin", "A Dynamic Theory of Personality",
     1935, ["psychology", "cognitive_science"],
     "archive", "in.ernet.dli.2015.499226"),

    ("1930s", "cassirer_symbolic_forms_vol3.txt",
     "Ernst Cassirer", "The Philosophy of Symbolic Forms, Vol. 3",
     1929, ["philosophy_of_mind", "phenomenology"],
     "archive", "the-philosophy-of-symbolic-forms-Ernst-Cassirer"),

    # 1940s
    ("1940s", "cassirer_myth_of_state.txt",
     "Ernst Cassirer", "The Myth of the State",
     1946, ["philosophy_of_mind", "political_philosophy"],
     "archive", "in.ernet.dli.2015.506295"),

    ("1940s", "popper_open_society_vol2.txt",
     "Karl Popper", "The Open Society and Its Enemies, Vol. 2",
     1945, ["political_philosophy", "sociology"],
     "archive", "open-society-and-its-enemies-vol.2-karl-popper"),

    # 1950s — retry Skinner + new Chomsky and Bruner
    ("1950s", "skinner_verbal_behavior.txt",
     "B.F. Skinner", "Verbal Behavior",
     1957, ["psychology", "cognitive_science"],
     "archive", "VerbalBehavior"),

    ("1950s", "chomsky_syntactic_structures.txt",
     "Noam Chomsky", "Syntactic Structures",
     1957, ["cognitive_science", "psychology"],
     "archive", "NoamChomskySyntcaticStructures"),

    ("1950s", "bruner_study_of_thinking.txt",
     "Jerome Bruner", "A Study of Thinking",
     1956, ["cognitive_science", "psychology"],
     "archive", "img-0922"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_url(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ERROR: {exc}")
    return None


def strip_gutenberg(text: str) -> str:
    start = re.search(r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    if start:
        rest = text[start.end():]
        nl = rest.find("\n")
        text = rest[nl + 1:] if nl >= 0 else rest
    end = re.search(r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    if end:
        text = text[: end.start()]
    return text.strip()


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.parts.append(data)


def extract_epub(data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if n.endswith((".html", ".xhtml", ".htm"))]
            # Try to respect spine order from OPF
            opf_list = [n for n in zf.namelist() if n.endswith(".opf")]
            if opf_list:
                opf_text = zf.read(opf_list[0]).decode("utf-8", errors="ignore")
                spine_ids = re.findall(r'<itemref[^>]+idref="([^"]+)"', opf_text)
                id_href = dict(re.findall(r'<item[^>]+id="([^"]+)"[^>]+href="([^"]+)"', opf_text))
                prefix = opf_list[0].rsplit("/", 1)[0] + "/" if "/" in opf_list[0] else ""
                ordered = [prefix + id_href[s] for s in spine_ids if s in id_href]
                ordered = [n for n in ordered if n in zf.namelist()]
                if ordered:
                    names = ordered
            parts: list[str] = []
            for name in names:
                html = zf.read(name).decode("utf-8", errors="ignore")
                p = _HTMLTextExtractor()
                p.feed(html)
                parts.extend(p.parts)
        return " ".join(parts)
    except Exception as exc:
        print(f"    EPUB error: {exc}")
        return None


def extract_pdf(data: bytes) -> str | None:
    if not HAS_FITZ:
        print("    PyMuPDF not available; skipping PDF")
        return None
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        print(f"    PDF error: {exc}")
        return None


def fetch_gutenberg_text(gid: int) -> str | None:
    # Try cache URL first (most reliable)
    for url in [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]:
        data = fetch_url(url)
        if data:
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = data.decode("latin-1", errors="ignore")
            return strip_gutenberg(text)
    return None


def fetch_archive_text(identifier: str) -> str | None:
    base = f"https://archive.org/download/{identifier}"
    # Plain OCR text
    for suffix in [f"/{identifier}_djvu.txt", f"/{identifier}.txt"]:
        data = fetch_url(base + suffix)
        if data:
            return data.decode("utf-8", errors="ignore")
    # EPUB
    data = fetch_url(f"{base}/{identifier}.epub")
    if data:
        result = extract_epub(data)
        if result:
            return result
    # PDF
    data = fetch_url(f"{base}/{identifier}.pdf")
    if data:
        return extract_pdf(data)
    return None


def fetch_archive_epub(identifier: str) -> str | None:
    base = f"https://archive.org/download/{identifier}"
    data = fetch_url(f"{base}/{identifier}.epub")
    if data:
        return extract_epub(data)
    return fetch_archive_text(identifier)  # fallback


def fetch_archive_pdf(identifier: str) -> str | None:
    base = f"https://archive.org/download/{identifier}"
    data = fetch_url(f"{base}/{identifier}.pdf")
    if data:
        return extract_pdf(data)
    return fetch_archive_text(identifier)


def fetch_pdf_url(url: str) -> str | None:
    data = fetch_url(url)
    if data:
        return extract_pdf(data)
    return None


def fetch_marxists(url: str) -> str | None:
    """Fetch and strip HTML from marxists.org."""
    data = fetch_url(url)
    if not data:
        return None
    html = data.decode("utf-8", errors="ignore")
    p = _HTMLTextExtractor()
    p.feed(html)
    return " ".join(p.parts)


def fetch_archive_merge(identifiers: list[str]) -> str | None:
    """Fetch and concatenate multiple Archive.org items."""
    parts = []
    for identifier in identifiers:
        text = fetch_archive_text(identifier)
        if text:
            parts.append(text)
        time.sleep(0.5)
    return "\n\n".join(parts) if parts else None


def fetch_marxists_toc(base_url: str) -> str | None:
    """Fetch all chapter pages from a Marxists.org work index."""
    index_data = fetch_url(base_url)
    if not index_data:
        return None
    html = index_data.decode("utf-8", errors="ignore")

    # Collect relative .htm links within the same work directory
    base_path = base_url.rstrip("/")
    raw_links = re.findall(r'href="([^"#]*\.html?)"', html, re.IGNORECASE)
    seen: set[str] = set()
    chapter_urls: list[str] = []
    for link in raw_links:
        if link.startswith(("http://", "https://")):
            full_url = link
        else:
            full_url = urllib.parse.urljoin(base_url, link)
        # Only follow links within the same work directory
        if full_url.startswith(base_path) and full_url != base_url and full_url not in seen:
            seen.add(full_url)
            chapter_urls.append(full_url)

    if not chapter_urls:
        # No chapter links — fall back to treating the index as the full text
        p = _HTMLTextExtractor()
        p.feed(html)
        return " ".join(p.parts)

    parts: list[str] = []
    for url in chapter_urls:
        text = fetch_marxists(url)
        if text and len(text.strip()) > 100:
            parts.append(text)
        time.sleep(0.3)

    return "\n\n".join(parts) if parts else None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    existing_filenames = {b.get("filename") for b in manifest}

    fetched = 0
    skipped = 0
    failed: list[str] = []

    for entry in BOOKS:
        decade, filename, author, title, year, fields, source, source_id = entry

        dest = BOOKS_DIR / decade / filename

        if dest.exists():
            print(f"skip  {filename} (already exists)")
            skipped += 1
            continue

        print(f"\n→ {title} ({decade})")

        text: str | None = None

        if source == "gutenberg":
            text = fetch_gutenberg_text(int(source_id))  # type: ignore[arg-type]

        elif source == "gutenberg_merge":
            parts = []
            for gid in source_id:  # type: ignore[union-attr]
                t = fetch_gutenberg_text(gid)
                if t:
                    parts.append(t)
            text = "\n\n".join(parts) if parts else None

        elif source == "archive":
            text = fetch_archive_text(str(source_id))

        elif source == "archive_epub":
            text = fetch_archive_epub(str(source_id))

        elif source == "archive_pdf":
            text = fetch_archive_pdf(str(source_id))

        elif source == "archive_pdf_url":
            text = fetch_pdf_url(str(source_id))

        elif source == "marxists":
            text = fetch_marxists(str(source_id))

        elif source == "archive_merge":
            text = fetch_archive_merge(list(source_id))  # type: ignore[arg-type]

        elif source == "marxists_toc":
            text = fetch_marxists_toc(str(source_id))

        if not text or len(text.strip()) < 500:
            print("  FAILED — no usable text retrieved")
            failed.append(f"{decade}/{filename}")
            time.sleep(1)
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        words = len(text.split())
        print(f"  saved {words:,} words → {dest}")

        if filename not in existing_filenames:
            manifest.append({
                "title": title,
                "author": author,
                "decade": decade,
                "year": year,
                "filename": filename,
                "fields": fields,
                "status": "available",
            })
            existing_filenames.add(filename)

        fetched += 1
        time.sleep(0.5)  # be polite to servers

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Done. Fetched: {fetched}, Skipped: {skipped}, Failed: {len(failed)}")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
