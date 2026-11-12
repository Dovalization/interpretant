# Design draft: filling the thin cells and the control words

*Draft, 2026-11-12, for review; nothing is downloaded until it is approved. It proposes what to
add to the store's sources before the rebuild (`docs/design_store_rebuild.md`).*

## What exists

`scripts/fetch_sense_words.py` already does word-targeted downloads. For every word, field and
decade a round plans, it counts the grid's abstracts that contain the word; a word whose planned
cells all reach `MIN_USES` (500) is read from the grid, and any other word has *all* its planned
cells downloaded by word: a seeded sample of up to `TARGET` (1,000) works whose abstract contains
it, into `data/raw/openalex_words/<word>/`. Each word's cells therefore come from one source,
never a mix of grid and download. Without `--download` it only prints the gaps.

Ten words have been downloaded this way: *agency*, *cloud*, *consciousness*, *embedding*,
*embodiment*, *fusion*, *hallucination*, *intelligence*, *prompt*, *strength*.

## Gaps

**Thin cells** (the 2026-11-11 audit): *attention* in AI, 186 and 174 usages in the 1990s and
2000s; *platform* in communication, 142 in the 2000s; *alignment* in AI, 105 in the 2010s. Small
cells are where the across-minus-within measure gave its one false alarm (round X).
*attention* was left out of the earlier plan on purpose (it was round C's development data, read
from the grid); the fresh start has no such reason.

**Control words.** 15 of round X's 30 are missing from the store: *abstract*, *comparison*,
*conducted*, *defined*, *description*, *important*, *life*, *methodology*, *necessary*, *people*,
*report*, *successful*, *techniques*, *theory*, *weight*. They were chosen because the grid holds
hundreds of their uses per cell (`data/control_words_senses.txt`), so they need no download: the
store rebuild exports them from the grid.

**Half-decade windows.** If the detection experiments use half-decades
(`docs/design_detection_experiments.md`), each window needs its own floor of usages, which
roughly doubles what the thin words need.

## Proposal

1. Add *attention*, *platform* and *alignment* to the plan, with all their planned cells (one
   source per word), and print the gaps first.
2. Size downloads for half-decades only if the detection design keeps them: `TARGET` per
   half-decade instead of per decade.
3. Export the 15 control words from the grid in the store rebuild.

## Cost

The OpenAlex key allows 10,000 credits a day; the full grid download (88 cells, up to 10,000
works each) cost about 3,800, roughly one credit per page of up to 200 works. A word-targeted
cell of up to 1,000 works is then about 5 to 6 credits, and the three words' cells together
likely under 200. The gap print, which only counts, costs one credit per cell and should confirm
the figure before anything is downloaded.

## Decisions needed

1. The words and cells to add (the three above, or more from the detection design's cases).
2. Decade or half-decade sizing.
3. The floor a cell must reach to be read from the grid (500 now).
