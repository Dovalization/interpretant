"""The record vectors (`docs/junk_gate.md`, the classifier): every record behind a drawn usage.

Every work with at least one drawn usage is encoded once, from its abstract's kept text (the text
outside removed ranges, which the measures read), by the frozen text encoder of ``params.yaml``'s
``junk_gate.encoder`` section: the text's first ``max_tokens`` tokens, the last layer's state of
the first token, normalised to length 1. A work's vector goes in the part of the usages table it
is drawn in. The usages table puts each work in its abstracts part, so a work drawn in two parts
should not occur; if one does, it is encoded once and written in the first of them by sorted part
name, and counted.

The encoder runs on CUDA when there is a GPU, in float32 under inference mode, in batches of texts
of similar length, the longest first; the vectors come back in the texts' order. It is passed in,
so tests encode with a stand-in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from interpretant.catalog.abstracts import abstracts_parts, load_abstracts_part
from interpretant.catalog.config import CatalogConfig
from interpretant.catalog.record_vectors import (
    RECORD_VECTORS_COLUMNS,
    VECTOR_SIZE,
    save_record_vectors,
)
from interpretant.catalog.usages import load_usages_part
from interpretant.corpus.removals import Removal, kept_text
from interpretant.pipelines.params import ParamsSettings
from interpretant.pipelines.progress import SILENT, Progress

# Texts to unit vectors in their order; each batch's texts are reported to the progress as done.
Encode = Callable[[Sequence[str], Progress], NDArray[np.float32]]
# A batch's token ids, padded to one width, and its attention mask to each text's first state.
Forward = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class RecordEncoderSettings(ParamsSettings):
    """The ``junk_gate.encoder`` section of ``params.yaml``: the frozen encoder and its batches."""

    model_config = SettingsConfigDict(yaml_config_section="junk_gate.encoder")

    model: str
    revision: str
    max_tokens: int = Field(gt=0)
    batch_size: int = Field(gt=0)


@dataclass(frozen=True)
class RecordVectorsResult:
    """What the stage wrote: its parts, the records encoded, and the works drawn in two parts."""

    parts: list[str]
    record_count: int
    repeated_count: int


def embed_records(
    catalog: CatalogConfig, encode: Encode, *, progress: Progress = SILENT
) -> RecordVectorsResult:
    """Write the vector of every record behind a drawn usage, one part per part of the corpus.

    A part with no new record is written empty, so the table has the usages table's parts.

    Raises:
        FileNotFoundError: if the abstracts table has no part, or a part of the usages table is
            missing.
        ValueError: if a drawn usage's work is not in its part of the abstracts table.
        pandera.errors.SchemaError: if a part's vectors break the table's schema.
    """
    parts = abstracts_parts(catalog)
    if not parts:
        raise FileNotFoundError(f"no part of the abstracts table under {catalog.abstracts.path}")
    progress.stage("finding the records behind drawn usages", total=len(parts))
    works_by_part: dict[str, list[str]] = {}
    seen: set[str] = set()
    repeated_count = 0
    for part in parts:
        usages = load_usages_part(catalog, part, ["work_id", "drawn"])
        drawn = usages.filter("drawn")["work_id"].unique().sort().to_list()
        works_by_part[part] = [work_id for work_id in drawn if work_id not in seen]
        repeated_count += len(drawn) - len(works_by_part[part])
        seen.update(works_by_part[part])
        progress.advance()
    progress.stage("encoding the records", total=len(seen))
    for part, work_ids in works_by_part.items():
        if work_ids:
            vectors = encode(_kept_texts(catalog, part, work_ids), progress)
        else:
            vectors = np.empty((0, VECTOR_SIZE), dtype=np.float32)
        table = pl.DataFrame(
            {"work_id": work_ids, "vector": vectors}, schema=RECORD_VECTORS_COLUMNS
        )
        save_record_vectors(catalog, table, part)
    return RecordVectorsResult(parts=parts, record_count=len(seen), repeated_count=repeated_count)


def build_record_encoder(settings: RecordEncoderSettings) -> Encode:
    """The frozen encoder of the settings, loaded once, on CUDA when there is a GPU, in float32.

    Each text is cut to its first ``max_tokens`` tokens, the special tokens included; its vector
    is the last layer's state of the first token, normalised to length 1 (BGE's sentence vector).
    """
    # Imported here: transformers takes seconds to import, and every command loads this module
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(settings.model, revision=settings.revision)  # type: ignore[no-untyped-call]
    model = AutoModel.from_pretrained(settings.model, revision=settings.revision)
    model = model.to(device=device, dtype=torch.float32).eval()
    pad_id = int(tokenizer.pad_token_id)

    def first_token_states(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        states: torch.Tensor = output.last_hidden_state[:, 0].cpu()
        return states

    def encode_texts(texts: Sequence[str], progress: Progress) -> NDArray[np.float32]:
        encoded = tokenizer(list(texts), truncation=True, max_length=settings.max_tokens)
        return encode_token_ids(
            encoded["input_ids"],
            first_token_states,
            pad_id=pad_id,
            batch_size=settings.batch_size,
            progress=progress,
        )

    return encode_texts


def encode_token_ids(
    token_ids: Sequence[Sequence[int]],
    forward: Forward,
    *,
    pad_id: int,
    batch_size: int,
    progress: Progress = SILENT,
) -> NDArray[np.float32]:
    """One vector of length 1 per text's token ids, in their order, the texts batched by length.

    The longest texts go first, so a batch too large for the GPU fails at once; texts of equal
    length keep their order, so the batches are the same in every run.

    Raises:
        ValueError: if there are no token ids.
    """
    if not token_ids:
        raise ValueError("no texts to encode")
    order = sorted(range(len(token_ids)), key=lambda index: len(token_ids[index]), reverse=True)
    batches = []
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        width = len(token_ids[batch[0]])
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, index in enumerate(batch):
            length = len(token_ids[index])
            input_ids[row, :length] = torch.tensor(token_ids[index], dtype=torch.long)
            attention_mask[row, :length] = 1
        with torch.inference_mode():
            states = forward(input_ids, attention_mask).float()
        batches.append(torch.nn.functional.normalize(states, dim=1).numpy())
        progress.advance(len(batch))
    stacked = np.concatenate(batches)
    vectors = np.empty_like(stacked, dtype=np.float32)
    vectors[order] = stacked
    return vectors


def _kept_texts(catalog: CatalogConfig, part: str, work_ids: Sequence[str]) -> list[str]:
    """The kept text of each work, in the order given, from one part of the abstracts table."""
    abstracts = load_abstracts_part(catalog, part, ["work_id", "text", "removals"]).filter(
        pl.col("work_id").is_in(list(work_ids))
    )
    texts = {
        work_id: kept_text(text, [Removal(**removal) for removal in removals])
        for work_id, text, removals in abstracts.iter_rows()
    }
    missing = [work_id for work_id in work_ids if work_id not in texts]
    if missing:
        raise ValueError(f"{len(missing)} drawn works are missing from abstracts part {part}")
    return [texts[work_id] for work_id in work_ids]
