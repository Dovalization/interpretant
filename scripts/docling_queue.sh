#!/usr/bin/env bash
# Run Docling PDF extractions sequentially.
# Each job runs to completion before the next starts.
# Already-present output files are skipped.
# Run from project root: bash scripts/docling_queue.sh

set -euo pipefail

extract() {
    local src="$1"
    local dest="$2"
    local name
    name=$(basename "$dest")

    if [[ -f "$dest" ]]; then
        local words
        words=$(wc -w < "$dest")
        echo "skip  $name (already present, $words words)"
        return
    fi

    echo ""
    echo "=== [$(date +%H:%M:%S)] extracting: $(basename "$src") ==="
    uv run interpretant pdf extract "$src" --output "$dest"
    echo "=== [$(date +%H:%M:%S)] done: $name — $(wc -w < "$dest") words ==="
}

extract \
    'data/Dewey John - Art as Experience - libgen.li.pdf' \
    'data/external/books/1930s/dewey_art_as_experience.txt'

extract \
    'data/The Second Sex{Simone de Beauvoir}(2011){113364835} libgen.li.pdf' \
    'data/external/books/1940s/beauvoir_second_sex.txt'

extract \
    'data/James J. Gibson - The Ecological Approach To Visual Perception (1986, Psychology Press) - libgen.li.pdf' \
    'data/external/books/1970s/gibson_ecological_approach.txt'

extract \
    'data/Vilem Flusser, Vilm Flusser - Towards a Philosophy of Photography (2000, Reaktion Books) - libgen.li.pdf' \
    'data/external/books/1980s/flusser_philosophy_photography.txt'

extract \
    'data/Niklas Luhmann - Social Systems (1995, Stanford University Press) - libgen.li.pdf' \
    'data/external/books/1980s/luhmann_social_systems.txt'

extract \
    'data/[Writing Science] Friedrich Kittler - Gramophone, Film, Typewriter (1999, Stanford University Press) - libgen.li.pdf' \
    'data/external/books/1990s/kittler_gramophone_film_typewriter.txt'

extract \
    'data/Duhem, Pierre Maurice Marie, 1861-1916 - The Aim and Structure of Physical Theory - libgen.li.pdf' \
    'data/external/books/1900s/duhem_aim_structure_physical_theory.txt'

extract \
    'data/[Key texts. Classical studies in the History of Ideas.] Carnap, Rudolf - The unity of science (1934,1996, Thoemmes) - libgen.li.pdf' \
    'data/external/books/1930s/carnap_unity_science.txt'

extract \
    'data/Charles Taylor - Sources of the self_ the making of the modern identity (Harvard University Press) - libgen.li.pdf' \
    'data/external/books/1990s/taylor_sources_self.txt'

extract \
    'data/Zygmunt Bauman - Life in fragments _ essays in postmodern morality (1995, Blackwell Publishers) - libgen.li.pdf' \
    'data/external/books/1990s/bauman_life_fragments.txt'

extract \
    'data/[Meridian_ Crossing Aesthetics (Book 3)] Bernard Stiegler - Technics and Time, 3_ Cinematic Time and the Question of Malaise (Meridian_ Crossing Aesthetics) (2010, Stanford University Press) - libgen.li.pdf' \
    'data/external/books/2010s/stiegler_technics_time_3.txt'

extract \
    'data/Jacques Rancière, Gregory Elliott - The Emancipated Spectator (2009, Verso) - libgen.li.pdf' \
    'data/external/books/2010s/ranciere_emancipated_spectator.txt'

echo ""
echo "=== All done. Run: uv run interpretant corpus books validate ==="
