"""Re-date the books manifest by the English edition each file contains (November 2026 audit).

A translation carries the translator's English, so its vocabulary belongs to the decade of the
English edition, not of the original. This script applies the decisions below to
``data/external/books/manifest.json`` and moves each file into the folder of its new decade.
Books that cannot be dated inside the grid go to ``canon/`` (pre-1820 English originals) or
``excluded/`` (wrong or unusable text); the loader ignores both folders.

Evidence for each edition is the file's own front matter (copyright, "translated by", "first
published") unless the comment says otherwise. Run it once; a second run is a no-op.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

BOOKS_DIR = Path("data/external/books")
GRID_START = 1820


@dataclass(frozen=True)
class Translation:
    year: int
    translator: str
    original_language: str
    original_year: int


# filename -> English edition in the file.
TRANSLATIONS: dict[str, Translation] = {
    "hegel_phenomenology_spirit.txt": Translation(2018, "Michael Inwood", "German", 1807),
    "nietzsche_beyond_good_evil.txt": Translation(1907, "Helen Zimmern", "German", 1886),
    "nietzsche_genealogy_morality.txt": Translation(1910, "Horace B. Samuel", "German", 1887),
    "nietzsche_zarathustra.txt": Translation(1909, "Thomas Common", "German", 1883),
    "marx_capital_vol1.txt": Translation(1887, "Samuel Moore and Edward Aveling", "German", 1867),
    "marx_communist_manifesto.txt": Translation(1888, "Samuel Moore", "German", 1848),
    "dostoevsky_brothers_karamazov.txt": Translation(1912, "Constance Garnett", "Russian", 1880),
    "dostoevsky_notes_underground.txt": Translation(1918, "Constance Garnett", "Russian", 1864),
    "dostoevsky_crime_punishment.txt": Translation(1914, "Constance Garnett", "Russian", 1866),
    "freud_interpretation_dreams.txt": Translation(1913, "A. A. Brill", "German", 1899),
    "freud_psychopathology_everyday_life.txt": Translation(1914, "A. A. Brill", "German", 1901),
    "freud_totem_taboo.txt": Translation(1950, "James Strachey", "German", 1913),
    "kafka_metamorphosis.txt": Translation(2002, "David Wyllie", "German", 1915),
    "kafka_trial.txt": Translation(2003, "David Wyllie", "German", 1925),
    "cassirer_philosophy_symbolic_forms.txt": Translation(1953, "Ralph Manheim", "German", 1923),
    "levi_strauss_structural_anthropology.txt": Translation(
        1963, "Claire Jacobson and Brooke Grundfest Schoepf", "French", 1958
    ),
    "stiegler_technics_time.txt": Translation(
        1998, "Richard Beardsworth and George Collins", "French", 1994
    ),
    "saussure_course_general_linguistics.txt": Translation(1959, "Wade Baskin", "French", 1916),
    "heidegger_poetry_language_thought.txt": Translation(1971, "Albert Hofstadter", "German", 1935),
    "barthes_mythologies.txt": Translation(1972, "Annette Lavers", "French", 1957),
    "foucault_discipline_punish.txt": Translation(1977, "Alan Sheridan", "French", 1975),
    "deleuze_guattari_thousand_plateaus.txt": Translation(1987, "Brian Massumi", "French", 1980),
    "husserl_logical_investigations.txt": Translation(1970, "J. N. Findlay", "German", 1900),
    "bergson_creative_evolution.txt": Translation(1911, "Arthur Mitchell", "French", 1907),
    "wittgenstein_tractatus.txt": Translation(1922, "C. K. Ogden", "German", 1921),
    "merleau_ponty_phenomenology_perception.txt": Translation(1962, "Colin Smith", "French", 1945),
    "wittgenstein_philosophical_investigations.txt": Translation(
        1953, "G. E. M. Anscombe", "German", 1953
    ),
    "benjamin_illuminations.txt": Translation(1968, "Harry Zohn", "German", 1955),
    # Second edition (1957) with corrections; the translation dates from 1944.
    "kierkegaard_concept_anxiety.txt": Translation(1944, "Walter Lowrie", "Danish", 1844),
    # Austin's revised edition (1953), English pages only.
    "frege_foundations_arithmetic.txt": Translation(1953, "J. L. Austin", "German", 1884),
    "barthes_image_music_text.txt": Translation(1977, "Stephen Heath", "French", 1977),
    "baudrillard_simulacra_simulation.txt": Translation(
        1994, "Sheila Faria Glaser", "French", 1981
    ),
    "debord_society_spectacle.txt": Translation(2002, "Ken Knabb", "French", 1967),
    "eco_open_work.txt": Translation(1989, "Anna Cancogni", "Italian", 1962),
    "lyotard_postmodern_condition.txt": Translation(
        1984, "Geoff Bennington and Brian Massumi", "French", 1979
    ),
    "virilio_speed_politics.txt": Translation(1986, "Mark Polizzotti", "French", 1977),
    "latour_never_been_modern.txt": Translation(1993, "Catherine Porter", "French", 1991),
    "weber_protestant_ethic.txt": Translation(1930, "Talcott Parsons", "German", 1905),
    "freud_ego_id.txt": Translation(1927, "Joan Riviere", "German", 1923),
    "heidegger_being_time.txt": Translation(
        1962, "John Macquarrie and Edward Robinson", "German", 1927
    ),
    "jung_psychological_types.txt": Translation(
        1971, "H. G. Baynes, revised by R. F. C. Hull", "German", 1921
    ),
    "kafka_castle.txt": Translation(1930, "Willa and Edwin Muir", "German", 1926),
    "lukacs_history_class_consciousness.txt": Translation(
        1971, "Rodney Livingstone", "German", 1923
    ),
    "freud_civilization_discontents.txt": Translation(1961, "James Strachey", "German", 1930),
    "gramsci_prison_notebooks.txt": Translation(
        1971, "Quintin Hoare and Geoffrey Nowell Smith", "Italian", 1935
    ),
    "jung_archetypes_collective_unconscious.txt": Translation(
        1959, "R. F. C. Hull", "German", 1934
    ),
    "camus_myth_of_sisyphus.txt": Translation(1955, "Justin O'Brien", "French", 1942),
    "piaget_psychology_intelligence.txt": Translation(
        1950, "Malcolm Piercy and D. E. Berlyne", "French", 1947
    ),
    "sartre_being_nothingness.txt": Translation(1956, "Hazel E. Barnes", "French", 1943),
    "adorno_minima_moralia.txt": Translation(1974, "E. F. N. Jephcott", "German", 1951),
    "bachelard_poetics_space.txt": Translation(1964, "Maria Jolas", "French", 1958),
    "simondon_mode_existence_technical_objects.txt": Translation(
        2017, "Cecile Malaspina and John Rogove", "French", 1958
    ),
    "adorno_negative_dialectics.txt": Translation(1973, "E. B. Ashton", "German", 1966),
    "althusser_for_marx.txt": Translation(1969, "Ben Brewster", "French", 1965),
    "deleuze_difference_repetition.txt": Translation(1994, "Paul Patton", "French", 1968),
    "derrida_of_grammatology.txt": Translation(1976, "Gayatri Chakravorty Spivak", "French", 1967),
    "derrida_writing_difference.txt": Translation(1978, "Alan Bass", "French", 1967),
    "foucault_order_of_things.txt": Translation(1970, "Alan Sheridan (uncredited)", "French", 1966),
    "gadamer_truth_method.txt": Translation(
        1989, "Joel Weinsheimer and Donald G. Marshall", "German", 1960
    ),
    "habermas_knowledge_human_interests.txt": Translation(
        1971, "Jeremy J. Shapiro", "German", 1968
    ),
    # No front matter; Fink's complete translation is the 2006 Norton edition.
    "lacan_ecrits.txt": Translation(2006, "Bruce Fink", "French", 1966),
    "ricoeur_conflict_interpretations.txt": Translation(1974, "Don Ihde (ed.)", "French", 1969),
    "vygotsky_mind_in_society.txt": Translation(
        1978,
        "Michael Cole, Vera John-Steiner, Sylvia Scribner and Ellen Souberman (eds.)",
        "Russian",
        1930,
    ),
    # Nice's translation appeared in 1990; the file has no copyright page.
    "bourdieu_logic_practice.txt": Translation(1990, "Richard Nice", "French", 1980),
    "habermas_theory_communicative_action.txt": Translation(
        1984, "Thomas McCarthy", "German", 1981
    ),
    "freud_beyond_pleasure_principle.txt": Translation(1961, "James Strachey", "German", 1920),
    "adorno_horkheimer_dialectic_enlightenment.txt": Translation(
        2002, "Edmund Jephcott", "German", 1944
    ),
    "bourdieu_distinction.txt": Translation(1984, "Richard Nice", "French", 1979),
    "borges_ficciones.txt": Translation(1962, "Anthony Kerrigan (ed.)", "Spanish", 1944),
    "merleau_ponty_visible_invisible.txt": Translation(1968, "Alphonso Lingis", "French", 1964),
    "durkheim_elementary_forms.txt": Translation(1915, "Joseph Ward Swain", "French", 1912),
    "descartes_meditations.txt": Translation(1853, "John Veitch", "Latin", 1641),
    "kant_critique_pure_reason.txt": Translation(1855, "J. M. D. Meiklejohn", "German", 1781),
    "popper_logic_scientific_discovery.txt": Translation(
        1959, "Karl Popper, Julius Freed and Lan Freed", "German", 1934
    ),
    "castoriadis_imaginary_institution.txt": Translation(1987, "Kathleen Blamey", "French", 1975),
    "kittler_gramophone_film_typewriter.txt": Translation(
        1999, "Geoffrey Winthrop-Young and Michael Wutz", "German", 1986
    ),
    "beauvoir_second_sex.txt": Translation(1953, "H. M. Parshley", "French", 1949),
    "luhmann_social_systems.txt": Translation(
        1995, "John Bednarz Jr. with Dirk Baecker", "German", 1984
    ),
    "de_certeau_practice_everyday_life.txt": Translation(1984, "Steven Rendall", "French", 1980),
    # The file is the Reaktion edition, Anthony Mathews' translation (2000).
    "flusser_philosophy_photography.txt": Translation(2000, "Anthony Mathews", "German", 1983),
    "mauss_the_gift.txt": Translation(1954, "Ian Cunnison", "French", 1925),
    "husserl_ideas_1.txt": Translation(1931, "W. R. Boyce Gibson", "German", 1913),
    "bergson_matter_memory.txt": Translation(
        1911, "Nancy Margaret Paul and W. Scott Palmer", "French", 1896
    ),
    "schopenhauer_world_as_will_vol1.txt": Translation(1958, "E. F. J. Payne", "German", 1818),
    # Preface and introduction from Dyde (1896), the rest from Knox (1942).
    "hegel_philosophy_right.txt": Translation(1942, "T. M. Knox and S. W. Dyde", "German", 1820),
    "tocqueville_democracy_america.txt": Translation(1835, "Henry Reeve", "French", 1835),
    "brentano_psychology_empirical.txt": Translation(
        1973, "Antos C. Rancurello, D. B. Terrell and Linda L. McAlister", "German", 1874
    ),
    "nietzsche_birth_tragedy.txt": Translation(1909, "William A. Haussmann", "German", 1872),
    "engels_condition_working_class.txt": Translation(
        1887, "Florence Kelley Wischnewetzky", "German", 1845
    ),
    "marx_critique_political_economy.txt": Translation(1904, "N. I. Stone", "German", 1859),
    "hegel_philosophy_history.txt": Translation(1857, "J. Sibree", "German", 1837),
    "feuerbach_essence_christianity.txt": Translation(1854, "George Eliot", "German", 1841),
    "tolstoy_what_is_art.txt": Translation(1898, "Aylmer Maude", "Russian", 1897),
    "poincare_science_hypothesis.txt": Translation(1913, "George Bruce Halsted", "French", 1902),
    "croce_aesthetic.txt": Translation(1909, "Douglas Ainslie", "Italian", 1902),
    "einstein_relativity.txt": Translation(1920, "Robert W. Lawson", "German", 1916),
    "duhem_aim_structure_physical_theory.txt": Translation(
        1954, "Philip P. Wiener", "French", 1906
    ),
    "carnap_unity_science.txt": Translation(1934, "Max Black", "German", 1932),
    "beck_risk_society.txt": Translation(1992, "Mark Ritter", "German", 1986),
    "habermas_between_facts_norms.txt": Translation(1996, "William Rehg", "German", 1992),
    "meillassoux_after_finitude.txt": Translation(2008, "Ray Brassier", "French", 2006),
    "stiegler_technics_time_3.txt": Translation(2011, "Stephen Barker", "French", 2001),
    "ranciere_emancipated_spectator.txt": Translation(2009, "Gregory Elliott", "French", 2008),
    "nietzsche_gay_science.txt": Translation(1910, "Thomas Common", "German", 1882),
    "wundt_principles_physiological_psychology.txt": Translation(
        1904, "Edward Bradford Titchener", "German", 1874
    ),
    # Fourth edition: Helmer's translation "thoroughly revised" by Jan Tarski.
    "tarski_introduction_logic.txt": Translation(1994, "Olaf Helmer, revised", "Polish", 1936),
    # Martineau's condensed translation; the file is the 1896 Bell reprint.
    "comte_positive_philosophy.txt": Translation(1853, "Harriet Martineau", "French", 1830),
    "comte_general_view_positivism.txt": Translation(1865, "J. H. Bridges", "French", 1848),
    "schleiermacher_on_religion.txt": Translation(1893, "John Oman", "German", 1799),
    "helmholtz_physiological_optics_v3.txt": Translation(
        1925, "James P. C. Southall", "German", 1867
    ),
    # Second edition, revised (1892); Wallace's first edition was 1873.
    "hegel_science_of_logic.txt": Translation(1892, "William Wallace", "German", 1817),
    "bergson_time_and_free_will.txt": Translation(1910, "F. L. Pogson", "French", 1889),
    "pavlov_conditioned_reflexes.txt": Translation(1927, "G. V. Anrep", "Russian", 1926),
    "stendhal_on_love.txt": Translation(
        1915, "Philip Sidney Woolf and Cecil N. Sidney Woolf", "French", 1822
    ),
    "quetelet_treatise_on_man.txt": Translation(1842, "R. Knox", "French", 1835),
    "stirner_ego_and_his_own.txt": Translation(1907, "Steven T. Byington", "German", 1844),
    "humboldt_cosmos_vol1.txt": Translation(1849, "E. C. Otté", "German", 1845),
    "kierkegaard_either_or_pt1.txt": Translation(
        1987, "Howard V. Hong and Edna H. Hong", "Danish", 1843
    ),
    # Daniel De Leon's translation (1897 serial, 1898 book).
    "marx_eighteenth_brumaire.txt": Translation(1898, "Daniel De Leon", "German", 1852),
    "bernard_introduction_experimental_medicine.txt": Translation(
        1927, "Henry Copley Greene", "French", 1865
    ),
    "nietzsche_thoughts_out_of_season_pt1.txt": Translation(
        1909, "Anthony M. Ludovici", "German", 1873
    ),
    "nietzsche_thoughts_out_of_season_pt2.txt": Translation(1909, "Adrian Collins", "German", 1874),
    "nietzsche_human_all_too_human.txt": Translation(1908, "Alexander Harvey", "German", 1878),
    "haeckel_history_of_creation_vol1.txt": Translation(
        1876, "E. Ray Lankester (rev.)", "German", 1868
    ),
    "taine_on_intelligence.txt": Translation(1871, "T. D. Haye", "French", 1870),
    "mach_analysis_of_sensations.txt": Translation(1897, "C. M. Williams", "German", 1886),
    "nietzsche_twilight_idols.txt": Translation(1911, "Anthony M. Ludovici", "German", 1889),
    "bergson_introduction_to_metaphysics.txt": Translation(1912, "T. E. Hulme", "French", 1903),
    "jung_psychology_unconscious.txt": Translation(1916, "Beatrice M. Hinkle", "German", 1912),
    "freud_general_introduction_psychoanalysis.txt": Translation(
        1920, "G. Stanley Hall (pref.)", "German", 1917
    ),
    "maine_de_biran_influence_of_habit.txt": Translation(
        1929, "Margaret Donaldson Boehm", "French", 1803
    ),
    "cousin_introduction_history_philosophy.txt": Translation(
        1832, "Henning Gotfried Linberg", "French", 1828
    ),
    "comte_catechism_positive_religion.txt": Translation(1858, "Richard Congreve", "French", 1852),
    "von_hartmann_philosophy_unconscious.txt": Translation(
        1884, "William Chatterton Coupland", "German", 1869
    ),
    # Marxists Internet Archive text, from Marx-Engels Collected Works vol. 5 (1976).
    "marx_german_ideology.txt": Translation(1976, "Marx-Engels Collected Works", "German", 1846),
    "weber_economy_and_society.txt": Translation(
        1968, "Guenther Roth and Claus Wittich (eds.)", "German", 1922
    ),
    "mannheim_ideology_utopia.txt": Translation(
        1936, "Louis Wirth and Edward Shils", "German", 1929
    ),
}

# filename -> corrected year for English originals filed in the wrong decade.
ENGLISH_YEARS: dict[str, int] = {
    "wiener_human_use_human_beings.txt": 1950,
    "veblen_theory_leisure_class.txt": 1899,
    "james_essays_radical_empiricism.txt": 1912,
    "taylor_sources_self.txt": 1989,
    "hamilton_lectures_metaphysics_logic.txt": 1859,
    "mill_essays_unsettled_questions.txt": 1844,
    "whewell_philosophy_inductive_sciences.txt": 1840,
    "chambers_vestiges_natural_history.txt": 1844,
    "buckle_history_civilisation_vol2.txt": 1861,
    "tyndall_fragments_of_science.txt": 1871,
    "maudsley_physiology_of_mind.txt": 1876,
    # The 1922 abridgment by Frazer himself, not the 1890 first edition.
    "frazer_golden_bough.txt": 1922,
}

CANON: dict[str, str] = {
    "coleridge_biographia_literaria.txt": "English original of 1817, before the first decade",
    "shelley_frankenstein.txt": "English original of 1818, before the first decade",
}

# Files that are not the manifest's book, and have no public-domain English edition to
# replace them with yet.
EXCLUDED: dict[str, str] = {
    "poincare_value_science.txt": (
        "wrong text (Kroeber, Mohave Pottery); The Value of Science is inside "
        "poincare_science_hypothesis.txt"
    ),
    "benjamin_work_of_art.txt": "duplicate: the essay is in benjamin_illuminations.txt",
}

# Entries dropped from the manifest: the file was a different book (Bayard Taylor, A History
# of Germany), and the book adds little target vocabulary next to The Concept of Dread.
REMOVED: set[str] = {"kierkegaard_fear_trembling.txt"}

# Replacements downloaded for files that held the wrong book (November 2026).
REPLACED: dict[str, tuple[str, str]] = {
    "nietzsche_genealogy_morality.txt": (
        "Project Gutenberg",
        "https://www.gutenberg.org/ebooks/52319",
    ),
    "marx_communist_manifesto.txt": ("Project Gutenberg", "https://www.gutenberg.org/ebooks/61"),
    "james_essays_radical_empiricism.txt": (
        "Project Gutenberg",
        "https://www.gutenberg.org/ebooks/32547",
    ),
    "frazer_golden_bough.txt": ("Project Gutenberg", "https://www.gutenberg.org/ebooks/3623"),
    "feuerbach_essence_christianity.txt": (
        "Project Gutenberg",
        "https://www.gutenberg.org/ebooks/47025",
    ),
    "eliot_sacred_wood.txt": ("Project Gutenberg", "https://www.gutenberg.org/ebooks/57795"),
    "chambers_vestiges_natural_history.txt": (
        "Project Gutenberg",
        "https://www.gutenberg.org/ebooks/7116",
    ),
    "marx_critique_political_economy.txt": (
        "Project Gutenberg",
        "https://www.gutenberg.org/ebooks/46423",
    ),
    "croce_aesthetic.txt": ("Project Gutenberg", "https://www.gutenberg.org/ebooks/9306"),
    "galton_hereditary_genius.txt": (
        "Internet Archive",
        "https://archive.org/details/hereditarygeniu01galtgoog",
    ),
    "spencer_study_of_sociology.txt": (
        "Internet Archive",
        "https://archive.org/details/studyofsociology00spen_0",
    ),
    "spencer_man_versus_state.txt": (
        "Internet Archive",
        "https://archive.org/details/manversusstate00spen",
    ),
    "martineau_illustrations_political_economy.txt": (
        "Internet Archive",
        "https://archive.org/details/illustrationsofp06martuoft",
    ),
    "hegel_philosophy_history.txt": (
        "Internet Archive",
        "https://archive.org/details/lecturesonphilos00hegeiala",
    ),
    "descartes_meditations.txt": (
        "Internet Archive",
        "https://archive.org/details/meditationsselec00desc",
    ),
    # The old file had a broken font encoding; this text is OCR of a scan of the Braziller
    # reprint of the 1946 Prentice-Hall edition, supplied by the project owner.
    "morris_signs_language_behavior.txt": ("Manually sourced (OCR of scanned PDF)", ""),
    # Supplied by the project owner: Princeton PDF, Austin PDF (OCR) and the Zohn ebook,
    # trimmed to the ten essays.
    "kierkegaard_concept_anxiety.txt": ("Manually sourced (PDF text layer)", ""),
    "frege_foundations_arithmetic.txt": ("Manually sourced (OCR of scanned PDF)", ""),
    "benjamin_illuminations.txt": ("Manually sourced (EPUB)", ""),
}

TITLES: dict[str, str] = {
    "kierkegaard_concept_anxiety.txt": "The Concept of Dread",
    "benjamin_illuminations.txt": "Illuminations",
    "poincare_science_hypothesis.txt": (
        "The Foundations of Science "
        "(Science and Hypothesis, The Value of Science, Science and Method)"
    ),
    "martineau_illustrations_political_economy.txt": "Illustrations of Political Economy, Vol. 6",
}


def entry_filename(entry: dict[str, object]) -> str:
    return str(entry.get("filename") or Path(str(entry.get("file", ""))).name)


def target_folder(entry: dict[str, object]) -> str:
    filename = entry_filename(entry)
    if filename in EXCLUDED:
        return "excluded"
    if filename in CANON:
        return "canon"
    return f"{int(str(entry['year'])) // 10 * 10}s"


def apply(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Apply the decisions to manifest entries, dropping duplicate entries for one file."""
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for entry in entries:
        filename = entry_filename(entry)
        if filename in seen or filename in REMOVED:
            continue
        seen.add(filename)
        entry.pop("file", None)
        entry["filename"] = filename
        if filename in TRANSLATIONS:
            translation = TRANSLATIONS[filename]
            entry["year"] = translation.year
            entry["original_year"] = translation.original_year
            entry["original_language"] = translation.original_language
            entry["translator"] = translation.translator
        if filename in ENGLISH_YEARS:
            entry["year"] = ENGLISH_YEARS[filename]
        if filename in REPLACED:
            entry["source"], entry["source_url"] = REPLACED[filename]
        if filename in TITLES:
            entry["title"] = TITLES[filename]
        entry.pop("exclusion", None)
        reason = EXCLUDED.get(filename) or CANON.get(filename)
        if reason:
            entry["exclusion"] = reason
        entry["decade"] = target_folder(entry)
        result.append(entry)
    return result


def main() -> None:
    manifest_path = BOOKS_DIR / "manifest.json"
    entries: list[dict[str, object]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_paths = {entry_filename(entry): BOOKS_DIR / str(entry["decade"]) for entry in entries}
    updated = apply(entries)
    moved = 0
    for entry in updated:
        filename = str(entry["filename"])
        source = old_paths[filename] / filename
        destination = BOOKS_DIR / str(entry["decade"]) / filename
        if source == destination or not source.is_file():
            continue
        destination.parent.mkdir(exist_ok=True)
        shutil.move(source, destination)
        moved += 1
    manifest_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(updated)} entries, {moved} files moved")


if __name__ == "__main__":
    main()
