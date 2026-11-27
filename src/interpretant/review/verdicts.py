"""Two judges' answers and an adjudicator's, merged into each item's final answer.

Every audit judged by agents settles its items the same way: each judge's answers arrive in part
files, the adjudicator answers the items where the judges differ in a way the audit's plan sends
to it, and everywhere else the first judge's answer stands.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

# One judge's answer to one item, shaped by its codebook's schema (``review.judging.SCHEMAS``)
Answer = dict[str, Any]


class VerdictsError(ValueError):
    """The judges' answers can't be merged as they stand."""


def load_parts(folder: Path, name: str) -> dict[str, Answer]:
    """One judge's answers by item id, merged over ``NAME.json`` and ``NAME_part_*.json``.

    The judges' manifests (``*.manifest.json``) sit beside the answers and are left out.

    Raises:
        VerdictsError: if an item is judged twice.
    """
    answers: dict[str, Answer] = {}
    paths = [folder / f"{name}.json", *sorted(folder.glob(f"{name}_part_*.json"))]
    for path in paths:
        if not path.exists() or path.name.endswith(".manifest.json"):
            continue
        for item_id, answer in json.loads(path.read_text(encoding="utf-8")).items():
            if item_id in answers:
                raise VerdictsError(f"{name} judged {item_id} twice")
            answers[item_id] = answer
    return answers


def merge_verdicts(
    first: Mapping[str, Answer],
    second: Mapping[str, Answer],
    adjudicated: Mapping[str, Answer],
    *,
    differ: Callable[[Answer, Answer], bool],
) -> tuple[dict[str, Answer], int, list[str]]:
    """Each item's final answer, from the two judges' answers and the adjudicator's.

    Where the judges' answers do not differ, the first judge's stands: a difference the plan does
    not send to the adjudicator is settled for the first judge. Where they differ, the
    adjudicator's stands; an item the adjudicator has not answered yet is pending and has no final
    answer.

    Args:
        first: The first judge's answer to every item, by item id. Items are merged in its order.
        second: The second judge's answers to the same items.
        adjudicated: The adjudicator's answers so far, by item id; answers to other items are
            ignored.
        differ: Whether two answers to one item need the adjudicator.

    Returns:
        The final answers by item id, the number of items the judges agreed on, and the ids of the
        items still waiting for the adjudicator.

    Raises:
        VerdictsError: if the judges did not answer the same items.
    """
    unmatched = sorted(first.keys() ^ second.keys())
    if unmatched:
        raise VerdictsError(
            f"{len(unmatched)} items have one judge's answer only, such as {unmatched[:5]}"
        )
    final: dict[str, Answer] = {}
    agreed_count = 0
    pending_ids: list[str] = []
    for item_id, first_answer in first.items():
        if not differ(first_answer, second[item_id]):
            final[item_id] = first_answer
            agreed_count += 1
        elif item_id in adjudicated:
            final[item_id] = adjudicated[item_id]
        else:
            pending_ids.append(item_id)
    return final, agreed_count, pending_ids


def build_pending(
    items: Iterable[dict[str, Any]],
    first: Mapping[str, Answer],
    second: Mapping[str, Answer],
    pending_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """The items the adjudicator must see, in the order of ``pending_ids``.

    Each item gains both judges' answers as ``judge_1`` and ``judge_2``, which
    ``review.judging.render`` shows under it.
    """
    items_by_id = {item["id"]: item for item in items}
    return [
        items_by_id[item_id] | {"judge_1": first[item_id], "judge_2": second[item_id]}
        for item_id in pending_ids
    ]
