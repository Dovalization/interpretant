"""The score-records stage on synthetic records, with a stand-in Ollama server.

In the 1990s, records 1 to 20 and 101 to 110 and 141 to 150 are junk, the rest prose. L labels
records 1 to 100, E records 101 to 110, and 111 to 150 are unlabelled. The stand-in answers a junk
text as not its own prose and a prose text as its own, so the screening model's lowest cut-off
falls between them and the confirming model gets L's, E's and the unlabelled junk records.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.junk_gate import JunkLabels, save_junk_labels
from interpretant.catalog.record_decisions import load_decisions, read_pass_manifest
from interpretant.pipelines.judge_ollama import OllamaRequestError
from interpretant.pipelines.score_records import (
    DecisionSettings,
    ScoringIncompleteError,
    find_lowest_screen_cut_off,
    labelled_records,
    score_records,
)
from interpretant.review.junk_gate import OPEN_MARK
from tests.junk_gate_records import (
    SyntheticRecord,
    build_label,
    work_id,
    write_records,
    write_words,
)

REPOSITORY = Path(__file__).resolve().parents[1]
SETTINGS = DecisionSettings(
    codebook=REPOSITORY / "curated" / "judging" / "junk_gate" / "prompt.md",
    screen_model="screen-model",
    confirm_model="confirm-model",
    specificity_target=0.97,
    folds=5,
    keep_alive="1m",
    chunk_size=7,
)
JUNK_NUMBERS = {*range(1, 21), *range(101, 111), *range(141, 151)}


class StandInServer:
    """Answers own-prose questions: 0.1 for a junk text, 0.9 for prose; unloads answer empty.

    The requests whose positions are in ``failing`` (counting decision requests from 0) fail.
    """

    def __init__(self, failing: frozenset[int] = frozenset()) -> None:
        self.failing = failing
        self.decisions: list[dict[str, Any]] = []
        self.unloaded: list[str] = []

    def __call__(self, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        assert body is not None
        if path == "/api/generate":
            self.unloaded.append(body["model"])
            return {}
        position = len(self.decisions)
        self.decisions.append(body)
        if position in self.failing:
            raise OllamaRequestError("the server refused")
        own_prose = 0.1 if "CONTENTS" in body["state"] else 0.9
        return {"answers": {"own_prose": {"type": "noul", "noul": own_prose}}}

    def models(self) -> list[str]:
        return [body["model"] for body in self.decisions]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    folder = tmp_path_factory.mktemp("score_records")
    shutil.copy(REPOSITORY / "catalog.yaml", folder / "catalog.yaml")
    catalog = read_catalog(folder / "catalog.yaml")
    write_words(catalog)
    records = [
        SyntheticRecord(number, 1995, is_junk=number in JUNK_NUMBERS, has_network=False)
        for number in range(1, 151)
    ]
    usages = write_records(catalog, records)
    labels = [
        build_label(
            usages[number, "attention"],
            "labelled" if number <= 100 else "enrichment",
            decade=1990,
            verdict="junk" if number in JUNK_NUMBERS else "abstract",
        )
        for number in range(1, 111)
    ]
    save_junk_labels(
        catalog,
        JunkLabels(
            codebook_sha256="c" * 64,
            items={f"S{index:04d}": label for index, label in enumerate(labels, start=1)},
        ),
    )
    return SimpleNamespace(catalog=catalog)


def run(catalog: CatalogConfig, root: Path, server: StandInServer, **options: Any) -> Any:
    return score_records(catalog, SETTINGS, server, root=root, **options)


def test_every_record_is_screened_and_the_ones_it_could_call_junk_are_confirmed(
    corpus: SimpleNamespace, tmp_path: Path
) -> None:
    server = StandInServer()

    result = run(corpus.catalog, tmp_path, server)

    screen = load_decisions(tmp_path, "screen")
    confirm = load_decisions(tmp_path, "confirm")
    assert set(screen["work_id"]) == {work_id(number) for number in range(1, 151)}
    expected = {work_id(number) for number in [*range(1, 111), *range(141, 151)]}
    assert set(confirm["work_id"]) == expected
    assert result.screen.needed == result.screen.scored == 150
    assert result.confirm.needed == result.confirm.scored == len(expected)
    lowest = find_lowest_screen_cut_off(labelled_records(corpus.catalog), screen, SETTINGS)
    assert result.lowest_screen_cut_off == lowest
    prose, junk = 1 - 0.9, 1 - 0.1
    assert prose < lowest <= junk
    np.testing.assert_allclose(sorted(set(screen["probability"])), [0.1, 0.9])


def test_one_model_runs_at_a_time_and_no_sentence_is_marked(
    corpus: SimpleNamespace, tmp_path: Path
) -> None:
    server = StandInServer()

    run(corpus.catalog, tmp_path, server)

    models = server.models()
    assert models == sorted(models, key=lambda model: model != "screen-model")
    assert server.unloaded == ["screen-model", "confirm-model"]
    assert all(OPEN_MARK not in body["state"] for body in server.decisions)
    assert all(list(body["questions"]) == ["own_prose"] for body in server.decisions)


def test_a_stopped_run_resumes_with_the_records_it_lacks(
    corpus: SimpleNamespace, tmp_path: Path
) -> None:
    with pytest.raises(ScoringIncompleteError, match="1 records failed the screening pass"):
        run(corpus.catalog, tmp_path, StandInServer(failing=frozenset({2, 3})))
    assert load_decisions(tmp_path, "screen").height == 149
    server = StandInServer()

    result = run(corpus.catalog, tmp_path, server)

    assert result.screen.scored == 1
    assert server.models().count("screen-model") == 1
    assert load_decisions(tmp_path, "screen").height == 150


def test_scores_made_with_another_model_are_refused_and_kept(
    corpus: SimpleNamespace, tmp_path: Path
) -> None:
    run(corpus.catalog, tmp_path, StandInServer())
    other = SETTINGS.model_copy(update={"screen_model": "another-model"})

    with pytest.raises(ValueError, match="another model; move"):
        score_records(corpus.catalog, other, StandInServer(), root=tmp_path)
    assert read_pass_manifest(tmp_path, "screen")["model"] == "screen-model"
    assert load_decisions(tmp_path, "screen").height == 150


def test_a_trial_scores_only_the_first_records_by_both_models(
    corpus: SimpleNamespace, tmp_path: Path
) -> None:
    result = run(corpus.catalog, tmp_path, StandInServer(), limit=3)

    assert result.screen.scored == result.confirm.scored == 3
    assert result.lowest_screen_cut_off is None
    assert load_decisions(tmp_path, "screen")["work_id"].equals(
        load_decisions(tmp_path, "confirm")["work_id"]
    )


def test_the_params_hold_the_deviations_models_and_cut_off_rule() -> None:
    settings = DecisionSettings()

    assert (settings.screen_model, settings.confirm_model) == (
        "d3-2b-16k",
        "pplx-decider-flash:4b",
    )
    assert (settings.specificity_target, settings.folds) == (0.97, 5)
    assert settings.codebook == Path("curated/judging/junk_gate/prompt.md")
