"""Draw the known-answers essay's figures (docs/known_answers.md) into figures/.

`known_answers_grid.png` places round 5's documented changes and quiet transitions against the
grid's candidate rule, in both seeds. `two_readings.png` sets round N's reading of the 1990s
classifier (counting assignments) beside round W's (reading probabilities) on the same samples.
The numbers come from the frozen scorers' own functions, so the figures show what was scored.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import score_grid_calibration as round5  # noqa: E402
import score_reference_frame as round_w  # noqa: E402

from interpretant.drift.direction import relative_gain  # noqa: E402
from interpretant.visualization.figures import (  # noqa: E402
    ReadingPanel,
    known_answer_ranks,
    readings,
)

OUTPUT = Path("figures")
SEEDS = (1, 2, 3, 4, 5)
CASES = (
    ("alignment, AI 2010s→20s", True, ("ai", 2010, "alignment")),
    ("attention, AI 2010s→20s", True, ("ai", 2010, "attention")),
    ("network, AI 1980s→90s", True, ("ai", 1980, "network")),
    ("platform, communication 2000s→10s", True, ("communication", 2000, "platform")),
    ("attention, AI 1990s→2000s", False, ("ai", 1990, "attention")),
    ("attention, psychology 2010s→20s", False, ("psychology", 2010, "attention")),
)


def grid_ranks() -> pd.DataFrame:
    rows = []
    for case, documented, (field, decade, word) in CASES:
        for seed in (42, 43):
            row = round5.within(seed, field, decade, word)
            assert row is not None, f"{word} unmeasured in {field} {decade}s, seed {seed}"
            rows.append(
                {
                    "case": case,
                    "documented": documented,
                    "seed": seed,
                    "cosine_rank": float(row["cosine_rank"]),
                    "shift_rank": float(row["shift_rank"]),
                }
            )
    return pd.DataFrame(rows)


def round_n() -> list[tuple[str, list[float]]]:
    table = pd.read_parquet("data/drift/direction_assignments.parquet")
    return [
        (label, [relative_gain(table, seed, 1990, 2020, "hci", targets) for seed in SEEDS])
        for label, targets in (
            ("towards computing", round_w.COMPUTING),
            ("towards behavioural\nand social sciences", round_w.BEHAVIOURAL_SOCIAL),
        )
    ]


def round_w_rows() -> list[tuple[str, list[float]]]:
    rows = []
    for label, slug in (("classifier", "classifier"), ("language models", "language_models")):
        costs = pd.read_parquet(f"data/drift/reference_frame_costs_{slug}.parquet")
        whole = costs[costs.cell == "whole"]
        rows.append((label, round_w.movements(whole, "test_decade", 1990, 2020, round_w.COMPUTING)))
    return rows


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    figures = {
        "known_answers_grid": known_answer_ranks(
            grid_ranks(),
            title="The grid flags little, and what it flags is real",
            subtitle="Four documented sense changes and two quiet transitions, in both training seeds",  # noqa: E501
            label_offsets={
                "alignment, AI 2010s→20s": (2.0, 13.0, "right"),
                "attention, AI 2010s→20s": (-8.0, -9.0, "right"),
            },
            footnote="TWEC on the lemmatised 10,000-abstract grid (round 5). A word is flagged when it is in the top 5% of its frequency band on both measures; each case's two seeds are joined by a line. Only alignment was flagged; neither quiet transition was.",  # noqa: E501
        ),
        "two_readings": readings(
            [
                ReadingPanel(
                    "Counting assignments (round N)",
                    "HCI's gain towards each group, net of other fields\n(share of abstracts)",
                    round_n(),
                ),
                ReadingPanel(
                    "Reading probabilities (round W)",
                    "HCI's movement towards computing over the behavioural\nand social sciences, net of other fields (bits)",  # noqa: E501
                    round_w_rows(),
                    left_caption="← behavioural and social",
                    right_caption="computing →",
                ),
            ],
            title="The same classifier, read two ways, points two ways",
            subtitle="HCI's change from the 1990s to the 2020s under a classifier trained on the 1990s, same samples",  # noqa: E501
            footnote="Bars: mean of five resampled runs; points: the runs. Round N counts the abstracts the classifier assigns to each field; round W, fitted on the same samples (its assignments reproduce round N's exactly), reads its probabilities in a two-group contrast. Round W's language models are shown for comparison. Round W's predictions were untestable as scored; these are its measured values.",  # noqa: E501
        ),
    }
    for stem, figure in figures.items():
        path = OUTPUT / f"{stem}.png"
        figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
        print(f"{path}")
    means = {label: statistics.mean(values) for label, values in round_n() + round_w_rows()}
    print({label: round(value, 3) for label, value in means.items()})


if __name__ == "__main__":
    main()
