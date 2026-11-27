"""The embed-records stage on the synthetic corpus, with a stand-in for the frozen encoder."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from numpy.typing import NDArray
from rich.console import Console

from interpretant.catalog.abstracts import load_abstracts
from interpretant.catalog.config import CatalogConfig, read_catalog
from interpretant.catalog.record_vectors import VECTOR_SIZE, load_record_vectors
from interpretant.catalog.usages import load_usages, load_usages_part, save_usages
from interpretant.cli import embed_records as embed_records_command
from interpretant.corpus.removals import Removal, kept_text
from interpretant.pipelines.build_corpus import BuildCorpusSettings, build_corpus
from interpretant.pipelines.build_sentence_view import build_sentence_view
from interpretant.pipelines.build_static_view import build_static_view
from interpretant.pipelines.build_usages import UsagesSettings, build_usages
from interpretant.pipelines.embed_records import (
    RecordEncoderSettings,
    embed_records,
    encode_token_ids,
)
from interpretant.pipelines.progress import SILENT, Progress
from interpretant.usages.usage_selection import NAME_LABELS
from tests.corpus_catalog import SELECTION, prepare_annotated_corpus, write_corpus_catalog

USAGES_SETTINGS = UsagesSettings(
    seed=7, model="en_core_web_sm", name_labels=tuple(sorted(NAME_LABELS))
)


class StandInEncoder:
    """Encodes each text as a unit vector seeded by its hash, and keeps the texts it was given."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: Sequence[str], progress: Progress) -> NDArray[np.float32]:
        self.texts.extend(texts)
        vectors = np.stack(
            [
                np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
                .standard_normal(VECTOR_SIZE)
                .astype(np.float32)
                for text in texts
            ]
        )
        progress.advance(len(texts))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def usages_catalog(tmp_path_factory: pytest.TempPathFactory) -> CatalogConfig:
    """The synthetic corpus with every stage up to the usages run, as ``dvc repro`` runs them."""
    catalog = read_catalog(write_corpus_catalog(tmp_path_factory.mktemp("corpus")))
    prepare_annotated_corpus(catalog)
    build_corpus(catalog, SELECTION, BuildCorpusSettings())
    build_sentence_view(catalog)
    build_static_view(catalog, SELECTION)
    build_usages(catalog, SELECTION, USAGES_SETTINGS)
    return catalog


def drawn_work_ids(catalog: CatalogConfig) -> set[str]:
    return set(load_usages(catalog).filter("drawn").select("work_id").collect()["work_id"])


def test_embed_records_writes_one_unit_vector_per_record_behind_a_drawn_usage(
    usages_catalog: CatalogConfig,
) -> None:
    result = embed_records(usages_catalog, StandInEncoder())

    vectors = load_record_vectors(usages_catalog).collect()
    assert result.parts == ["ai_2010", "compsci_2010"]
    assert sorted(path.stem for path in usages_catalog.record_vectors.path.iterdir()) == [
        "ai_2010",
        "compsci_2010",
    ]
    assert drawn_work_ids(usages_catalog)
    assert set(vectors["work_id"]) == drawn_work_ids(usages_catalog)
    assert vectors["work_id"].is_unique().all()
    assert result.record_count == vectors.height
    assert vectors.schema["vector"] == pl.Array(pl.Float32, VECTOR_SIZE)
    lengths = np.linalg.norm(np.stack(vectors["vector"].to_numpy()), axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_embed_records_encodes_each_kept_text_once_and_the_same_way_every_time(
    usages_catalog: CatalogConfig,
) -> None:
    encoder = StandInEncoder()

    embed_records(usages_catalog, encoder)
    first = load_record_vectors(usages_catalog).collect()
    embed_records(usages_catalog, StandInEncoder())
    again = load_record_vectors(usages_catalog).collect()

    assert first.equals(again)
    abstracts = load_abstracts(usages_catalog).select("work_id", "text", "removals").collect()
    kept = {
        work_id: kept_text(text, [Removal(**removal) for removal in removals])
        for work_id, text, removals in abstracts.iter_rows()
    }
    assert sorted(encoder.texts) == sorted(kept[work_id] for work_id in first["work_id"])


def test_a_work_drawn_in_two_parts_is_written_in_the_first_only(
    tmp_path: Path, usages_catalog: CatalogConfig
) -> None:
    catalog = usages_catalog.model_copy(
        update={
            "usages": usages_catalog.usages.model_copy(update={"path": tmp_path / "usages"}),
            "record_vectors": usages_catalog.record_vectors.model_copy(
                update={"path": tmp_path / "record_vectors"}
            ),
        }
    )
    first_part = load_usages_part(usages_catalog, "ai_2010")
    [work_id, *_] = sorted(set(first_part.filter("drawn")["work_id"]))
    copied = first_part.filter(pl.col("work_id") == work_id)
    second_part = load_usages_part(usages_catalog, "compsci_2010")
    save_usages(catalog, first_part, "ai_2010")
    save_usages(catalog, pl.concat([second_part, copied]), "compsci_2010")

    result = embed_records(catalog, StandInEncoder())

    assert result.repeated_count == 1
    in_first = pl.read_parquet(tmp_path / "record_vectors" / "ai_2010.parquet")
    in_second = pl.read_parquet(tmp_path / "record_vectors" / "compsci_2010.parquet")
    assert work_id in set(in_first["work_id"])
    assert work_id not in set(in_second["work_id"])


def test_encode_token_ids_batches_by_length_and_returns_the_vectors_in_order() -> None:
    token_ids = [[5, 1], [7, 2, 2, 2], [9], [3, 4, 4]]
    widths: list[int] = []

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        widths.append(input_ids.shape[1])
        assert attention_mask.sum(dim=1).tolist() == [int((row != 0).sum()) for row in input_ids]
        return torch.stack([input_ids[:, 0].float(), attention_mask.sum(dim=1).float()], dim=1)

    advanced: list[int] = []

    class Counting:
        def stage(self, name: str, total: int | None = None) -> None:
            return

        def advance(self, amount: int = 1) -> None:
            advanced.append(amount)

        def skip(self, amount: int) -> None:
            return

    vectors = encode_token_ids(token_ids, forward, pad_id=0, batch_size=2, progress=Counting())

    assert widths == [4, 2]
    assert advanced == [2, 2]
    expected = np.array([[5, 2], [7, 4], [9, 1], [3, 3]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(vectors, expected, rtol=1e-6)
    assert vectors.dtype == np.float32


def test_the_command_encodes_with_the_pre_registered_encoder(
    usages_catalog: CatalogConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[RecordEncoderSettings] = []

    def build_stand_in(settings: RecordEncoderSettings) -> StandInEncoder:
        built.append(settings)
        return StandInEncoder()

    monkeypatch.setattr(embed_records_command, "build_record_encoder", build_stand_in)

    counts = embed_records_command.run_embed_records(
        usages_catalog, progress=SILENT, console=Console(file=io.StringIO())
    )

    [settings] = built
    assert settings.model == "BAAI/bge-base-en-v1.5"
    assert settings.revision == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    assert settings.max_tokens == 512
    assert counts["records"] == len(drawn_work_ids(usages_catalog))
    assert counts["works_in_several_parts"] == 0
