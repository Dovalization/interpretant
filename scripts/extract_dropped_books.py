#!/usr/bin/env python3
"""Extract and move manually-dropped book files into the corpus.

Run from project root:
    uv run python scripts/extract_dropped_books.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import fitz  # type: ignore[import-untyped]

BOOKS_DIR = Path("data/external/books")

# (source glob pattern, decade, output filename)
MAPPINGS: list[tuple[str, str, str]] = [
    # 1820s — Schopenhauer (scan-heavy, may need Docling)
    ("*Schopenhauer*World*Will*Representation*", "1820s", "schopenhauer_world_as_will_vol1.txt"),
    # 1820s — Hegel Phenomenology of Spirit
    ("*Phenomenology*Spirit*Hegel*", "1820s", "hegel_phenomenology_spirit.txt"),
    ("*Hegel*Phenomenology*Spirit*", "1820s", "hegel_phenomenology_spirit.txt"),
    # 1820s — Hegel Philosophy of Right
    ("*Hegel*Philosophy*Right*", "1820s", "hegel_philosophy_right.txt"),
    # 1870s — Brentano (scan-heavy, may need Docling)
    ("*Brentano*Psychology*Empirical*", "1870s", "brentano_psychology_empirical.txt"),
    # 1870s — Wundt
    ("*Wundt*Physiological*Psychology*", "1870s", "wundt_principles_physiological_psychology.txt"),
    ("*Wundt*Principles*Psychology*", "1870s", "wundt_principles_physiological_psychology.txt"),

    # 1900s — Duhem
    ("*Duhem*Aim*Structure*Physical*", "1900s", "duhem_aim_structure_physical_theory.txt"),
    # 1910s — Bergson Matter and Memory
    ("*Bergson*Matter*Memory*", "1910s", "bergson_matter_memory.txt"),
    # 1920s — Whitehead Science and the Modern World
    ("*whitehead*Science*Modern*World*", "1920s", "whitehead_science_modern_world.txt"),
    ("*Science*Modern*World*whitehead*", "1920s", "whitehead_science_modern_world.txt"),
    ("*Science*Modern*World*North*", "1920s", "whitehead_science_modern_world.txt"),
    # 1930s — Carnap, Neurath
    ("*Carnap*unity*science*", "1930s", "carnap_unity_science.txt"),
    ("*Neurath*Carnap*Encyclopedia*Unified*", "1930s", "neurath_carnap_encyclopedia_unified_science.txt"),
    ("*International*Encyclopedia*Unified*Science*", "1930s", "neurath_carnap_encyclopedia_unified_science.txt"),
    # 1830s — Comte
    ("*Comte*Positive*Philosophy*", "1830s", "comte_positive_philosophy.txt"),
    ("*Positive*Philosophy*Comte*", "1830s", "comte_positive_philosophy.txt"),
    # 1930s — Tarski
    ("*Tarski*Introduction*Logic*", "1930s", "tarski_introduction_logic.txt"),
    # 1940s — Langer
    ("*Langer*Philosophy*New*Key*", "1940s", "langer_philosophy_new_key.txt"),
    ("*Langer*Feeling*Form*", "1950s", "langer_feeling_form.txt"),
    # 1940s — Morris semiotics
    ("*Morris*Signs*Language*Behavior*", "1940s", "morris_signs_language_behavior.txt"),
    ("*Signs*Language*Behavior*Morris*", "1940s", "morris_signs_language_behavior.txt"),
    # 1990s — social theory
    ("*Taylor*Sources*self*", "1990s", "taylor_sources_self.txt"),
    ("*Sources*self*Taylor*", "1990s", "taylor_sources_self.txt"),
    ("*Giddens*Consequences*Modernity*", "1990s", "giddens_consequences_modernity.txt"),
    ("*Beck*Risk*Society*", "1990s", "beck_risk_society.txt"),
    ("*Bauman*Life*fragments*", "1990s", "bauman_life_fragments.txt"),
    ("*Habermas*Between*Facts*Norms*", "1990s", "habermas_between_facts_norms.txt"),
    # 2010s — speculative realism + Stiegler + Rancière
    ("*Meillassoux*After*Finitude*", "2010s", "meillassoux_after_finitude.txt"),
    ("*Harman*Weird*Realism*", "2010s", "harman_weird_realism.txt"),
    ("*Stiegler*Technics*Time*3*", "2010s", "stiegler_technics_time_3.txt"),
    ("*Ranci*re*Emancipated*Spectator*", "2010s", "ranciere_emancipated_spectator.txt"),
    # Partials to replace
    ("*Totem*Taboo*Routledge*", "1910s", "freud_totem_taboo.txt"),
    ("*Castells*Rise*Network*", "1990s", "castells_rise_network_society.txt"),
    ("*Pasquale*Black*Box*", "2010s", "pasquale_black_box_society.txt"),
    # Psychoanalysis
    ("*Beyond*Pleasure*", "1920s", "freud_beyond_pleasure_principle.txt"),
    ("*Ego*and*Id*", "1920s", "freud_ego_id.txt"),
    ("*Civilization*Discontents*", "1930s", "freud_civilization_discontents.txt"),
    ("*Jung*Psychological*Types*", "1920s", "jung_psychological_types.txt"),
    ("*Archetypes*Collective*", "1930s", "jung_archetypes_collective_unconscious.txt"),
    ("*Fromm*Escape*Freedom*", "1940s", "fromm_escape_from_freedom.txt"),
    ("*Lacan*Ecrits*", "1960s", "lacan_ecrits.txt"),
    # Frankfurt School
    ("*Horkheimer*Adorno*Dialectic*enlightenment*", "1940s", "adorno_horkheimer_dialectic_enlightenment.txt"),
    ("*Minima*Moralia*Radi*", "1950s", "adorno_minima_moralia.txt"),
    ("*Negative*dialectics*", "1960s", "adorno_negative_dialectics.txt"),
    ("*Eros*Civilization*", "1950s", "marcuse_eros_civilization.txt"),
    ("*One-Dimensional*Man*", "1960s", "marcuse_one_dimensional_man.txt"),
    ("*Knowledge*Human*Interests*", "1960s", "habermas_knowledge_human_interests.txt"),
    ("*Theory*Communicative*Action*Volume*1*", "1980s", "habermas_theory_communicative_action.txt"),
    # Western Marxism / Sociology
    ("*Protestant*Ethic*", "1900s", "weber_protestant_ethic.txt"),
    ("*Elementary*Forms*Religious*", "1910s", "durkheim_elementary_forms.txt"),
    ("*History*Class*Consciousness*", "1920s", "lukacs_history_class_consciousness.txt"),
    ("*Prison*Notebooks*Gramsci*", "1930s", "gramsci_prison_notebooks.txt"),
    ("*Althusser*For*Marx*", "1960s", "althusser_for_marx.txt"),
    ("*Bourdieu*Logic*Practice*", "1980s", "bourdieu_logic_practice.txt"),
    ("*Distinction*Social*Critique*", "1980s", "bourdieu_distinction.txt"),
    # Continental Philosophy
    ("*Heidegger*Being*time*", "1920s", "heidegger_being_time.txt"),
    ("*Sartre*Being*Nothingness*", "1940s", "sartre_being_nothingness.txt"),
    ("*Camus*Myth*Sisyphus*", "1940s", "camus_myth_of_sisyphus.txt"),
    ("*Bachelard*poetics*space*", "1950s", "bachelard_poetics_space.txt"),
    ("*Mode*Existence*Technical*", "1950s", "simondon_mode_existence_technical_objects.txt"),
    ("*Gadamer*Truth*Method*", "1960s", "gadamer_truth_method.txt"),
    ("*Merleau-Ponty*Visible*Invisible*", "1960s", "merleau_ponty_visible_invisible.txt"),
    ("*Deleuze*Difference*Repetition*", "1960s", "deleuze_difference_repetition.txt"),
    ("*Derrida*Of*Grammatology*", "1960s", "derrida_of_grammatology.txt"),
    ("*Derrida*Writing*difference*", "1960s", "derrida_writing_difference.txt"),
    ("*Foucault*Order*Things*", "1960s", "foucault_order_of_things.txt"),
    ("*Ricoeur*Conflict*Interpretations*", "1970s", "ricoeur_conflict_interpretations.txt"),
    ("*Vygotsky*Mind*Society*", "1970s", "vygotsky_mind_in_society.txt"),
    # Structuralism
    ("*Jakobson*Language*Literature*", "1980s", "jakobson_language_literature.txt"),
    # Constructivism
    ("*Piaget*Psychology*Intelligence*", "1940s", "piaget_psychology_intelligence.txt"),
    # Literature
    ("*Borges*Ficciones*", "1940s", "borges_ficciones.txt"),
    ("*Kafka*Castle*", "1920s", "kafka_castle.txt"),
    # Contemporary
    ("*Haraway*Staying*Trouble*", "2010s", "haraway_staying_trouble.txt"),
    ("*Bridle*New*Dark*Age*", "2010s", "bridle_new_dark_age.txt"),
    ("*Zuboff*Surveillance*Capitalism*", "2010s", "zuboff_surveillance_capitalism.txt"),
    # Adorno Minima Moralia plain text version
    ("Adorno*Theodor*W*-*Minima*Moralia*-*libgen*", "1950s", "adorno_minima_moralia.txt"),
    # Philosophy of science / Epistemology
    ("Popper*Logic*Scientific Discovery*", "1950s", "popper_logic_scientific_discovery.txt"),
    ("Karl Popper*Conjectures*", "1960s", "popper_conjectures_refutations.txt"),
    ("Paul Feyerabend*Against Method*", "1970s", "feyerabend_against_method.txt"),
    ("Imre Lakatos*Methodology*", "1970s", "lakatos_scientific_research_programmes.txt"),
    ("Quine*Logical Point Of View*", "1950s", "quine_logical_point_of_view.txt"),
    ("Michael Polanyi*Personal Knowledge*", "1950s", "polanyi_personal_knowledge.txt"),
    ("Laboratory Life*libgen*", "1970s", "latour_woolgar_laboratory_life.txt"),
    ("Bruno Latour - Science in Action*", "1980s", "latour_science_in_action.txt"),
    ("Ian Hacking*Representing*Intervening*", "1980s", "hacking_representing_intervening.txt"),
    ("Hilary Putnam*Reason*Truth*History*", "1980s", "putnam_reason_truth_history.txt"),
    ("Nelson Goodman*Ways*Worldmaking*", "1970s", "goodman_ways_of_worldmaking.txt"),
    # Searle (Intentionality is a scan — handled by Docling)
    ("John Searle - Speech Acts*", "1960s", "searle_speech_acts.txt"),
    ("*Representation*Mind*Searle*Rediscovery*", "1990s", "searle_rediscovery_mind.txt"),
    ("John Searle - Intentionality*", "1980s", "searle_intentionality.txt"),
    # Gap-filling books
    ("*Husserl*Ideas*general*introduction*pure*", "1910s", "husserl_ideas_1.txt"),
    ("*Marcel Mauss*Gift*", "1920s", "mauss_the_gift.txt"),
    ("*George Herbert Mead*Mind*Self*", "1930s", "mead_mind_self_society.txt"),
    ("*Dewey*Art*Experience*", "1930s", "dewey_art_as_experience.txt"),
    ("*Second Sex*de Beauvoir*", "1940s", "beauvoir_second_sex.txt"),
    ("*Hannah Arendt*Human Condition*", "1950s", "arendt_human_condition.txt"),
    ("*Goffman*Presentation*Self*", "1950s", "goffman_presentation_self.txt"),
    ("*Ulric Neisser*Cognitive Psychology*", "1960s", "neisser_cognitive_psychology.txt"),
    ("*Chomsky*Aspects*Theory*Syntax*", "1960s", "chomsky_aspects_theory_syntax.txt"),
    ("*Debord*", "1960s", "debord_society_spectacle.txt"),
    ("*Umberto Eco*Open Work*", "1960s", "eco_open_work.txt"),
    ("*Clifford Geertz*Interpretation*Cultures*", "1970s", "geertz_interpretation_cultures.txt"),
    ("*Castoriadis*Imaginary Institution*", "1970s", "castoriadis_imaginary_institution.txt"),
    ("*Gibson*Ecological Approach*Visual*", "1970s", "gibson_ecological_approach.txt"),
    ("*Umberto Eco*Theory*Semiotics*", "1970s", "eco_theory_semiotics.txt"),
    ("*Lakoff*Metaphors*Live By*", "1980s", "lakoff_johnson_metaphors.txt"),
    ("*Michel de Certeau*Practice*Everyday*", "1980s", "de_certeau_practice_everyday_life.txt"),
    ("Marr D*Vision*", "1980s", "marr_vision.txt"),
    ("*Flusser*Philosophy*Photography*", "1980s", "flusser_philosophy_photography.txt"),
    ("*Niklas Luhmann*Social Systems*", "1980s", "luhmann_social_systems.txt"),
    ("*Rumelhart*McClelland*Parallel Distributed*", "1980s", "rumelhart_mcclelland_pdp.txt"),
    ("*Lyotard*Postmodern Condition*", "1980s", "lyotard_postmodern_condition.txt"),
    ("*Tufte*Visual Display*Quantitative*", "1980s", "tufte_visual_display.txt"),
    ("*Butler*Gender Trouble*", "1990s", "butler_gender_trouble.txt"),
    ("*Damasio*descartes error*", "1990s", "damasio_descartes_error.txt"),
    ("*Stuart Russell*Peter Norvig*Artificial Intelligence*", "1990s", "russell_norvig_ai.txt"),
    ("*Kittler*Gramophone*Film*Typewriter*", "1990s", "kittler_gramophone_film_typewriter.txt"),
    # 1960s — McLuhan
    ("*McLuhan*Gutenberg*Galaxy*", "1960s", "mcluhan_gutenberg_galaxy.txt"),
    # 2000s — media theory
    ("*Galloway*Protocol*", "2000s", "galloway_protocol.txt"),
    ("*New*Philosophy*New*Media*Hansen*", "2000s", "hansen_new_philosophy_new_media.txt"),
    ("*New*Philosophy*for*New*Media*", "2000s", "hansen_new_philosophy_new_media.txt"),
    ("*Bruno Latour*Reassembling*Social*", "2000s", "latour_reassembling_social.txt"),
    ("*Supersizing*Mind*Clark*", "2000s", "clark_supersizing_mind.txt"),
    ("*Mind*Life*Thompson*", "2000s", "thompson_mind_in_life.txt"),
    ("*Wegner*Illusion*Conscious*Will*", "2000s", "wegner_illusion_conscious_will.txt"),
    ("*Hofstadter*Strange*Loop*", "2000s", "hofstadter_i_am_strange_loop.txt"),
]


def _is_scan(doc: fitz.Document) -> bool:
    if doc.name.endswith(".epub"):
        return False
    pages = [doc[i].get_text() for i in range(min(5, len(doc)))]
    avg = sum(len(t) for t in pages) / len(pages) if pages else 0
    return avg < 100


def _extract_fitz(path: Path) -> str:
    doc = fitz.open(str(path))
    if _is_scan(doc):
        doc.close()
        return ""  # signal: needs Docling
    pages = [page.get_text() for page in doc]
    doc.close()
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(pages)).strip()


def _extract_djvu(path: Path) -> str:
    result = subprocess.run(
        ["djvutxt", str(path)], capture_output=True, text=True, errors="replace"
    )
    return re.sub(r"\n{3,}", "\n\n", result.stdout).strip()


def _extract_txt(path: Path) -> str:
    return re.sub(r"\n{3,}", "\n\n", path.read_text(errors="replace")).strip()


def _extract_html(path: Path) -> str:
    from html.parser import HTMLParser

    class _Strip(HTMLParser):
        _BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "li", "blockquote", "tr"}
        _SKIP = {"script", "style", "nav", "header", "footer", "aside"}

        def __init__(self) -> None:
            super().__init__()
            self._parts: list[str] = []
            self._depth = 0

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in self._SKIP:
                self._depth += 1
            if tag in self._BLOCK:
                self._parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in self._SKIP:
                self._depth = max(0, self._depth - 1)

        def handle_data(self, data: str) -> None:
            if self._depth == 0:
                self._parts.append(data)

        def result(self) -> str:
            return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()

    parser = _Strip()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.result()


def _extract(path: Path) -> tuple[str, bool]:
    """Return (text, needs_docling)."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _extract_txt(path), False
    if suffix in {".html", ".htm"}:
        return _extract_html(path), False
    if suffix == ".djvu":
        return _extract_djvu(path), False
    if suffix in {".pdf", ".epub", ".mobi", ".azw3"}:
        text = _extract_fitz(path)
        if not text:
            return "", True
        return text, False
    return "", False


def find_file(pattern: str) -> Path | None:
    matches = list(Path("data").glob(pattern))
    if not matches:
        return None
    return matches[0]


def main() -> None:
    results: list[tuple[str, str, int | None]] = []

    for pattern, decade, output_name in MAPPINGS:
        dest = BOOKS_DIR / decade / output_name

        src = find_file(pattern)
        if src is None:
            print(f"NOT FOUND  {pattern}")
            results.append((output_name, "not_found", None))
            continue

        if dest.exists():
            words = len(dest.read_text(errors="replace").split())
            print(f"skip  {output_name} (already present, {words:,} words)")
            results.append((output_name, "skip", words))
            continue

        print(f"extracting  {src.name[:70]} ...", end="", flush=True)
        text, needs_docling = _extract(src)

        if needs_docling:
            print(f" SCAN — run: uv run interpretant pdf extract '{src}' --output '{dest}'")
            results.append((output_name, "needs_docling", None))
            continue

        if not text or len(text.split()) < 500:
            print(f" ERROR — extracted only {len(text.split())} words")
            results.append((output_name, "error", len(text.split())))
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        words = len(text.split())
        print(f" {words:,} words → {dest}")
        results.append((output_name, "ok", words))

    print("\n=== Summary ===")
    for name, status, words in results:
        w = f"{words:,}" if words is not None else "-"
        print(f"  {status:<14} {w:>10}  {name}")


if __name__ == "__main__":
    main()
