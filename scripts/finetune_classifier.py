"""Fine-tune XL-LEXEME's top layers with one small head per word on the cue-labelled usages, and
label the development usages (development only; nothing here is a test).

    uv run python scripts/finetune_classifier.py LIST OUT.json [--layers 4] [--epochs 2]

Training: every word's cue-seeded usages in the main store's manifest (quality rules version 2),
out-of-scope usages and every development text left out; square-root balanced class weights;
10% of the seeds held out to watch for overfitting. The encoder reads usages exactly as the
vector store does (``usage_input_ids``, mean pooling); its bottom layers stay frozen, its top
``--layers`` layers and the heads train under float16 autocast. Output: each development
usage's label and probabilities, keyed by its set and id, for ``dev_harness.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import tomllib
from pathlib import Path

import numpy as np

from interpretant.drift.classifier import cue_sense, out_of_scope, sqrt_balanced
from interpretant.drift.pairs import XL_LEXEME, XL_LEXEME_REVISION, Usage, usage_input_ids

STORE = Path("data/store")
WORDS = ("attention", "prompt", "platform", "network", "component", "having")


def usage(row: dict[str, object]) -> Usage:
    return Usage(str(row["text"]), int(row["start"]), int(row["end"]))  # type: ignore[call-overload]


def load(list_path: Path) -> tuple[dict, list, list]:  # type: ignore[type-arg]
    table = tomllib.loads(list_path.read_text(encoding="utf-8"))
    with gzip.open(STORE / "dev_usages.jsonl.gz", "rt", encoding="utf-8") as source:
        dev = [json.loads(line) for line in source]
    dev_texts = {r["text"] for r in dev}
    seeds: list[tuple[str, dict[str, object], str]] = []
    seen: set[tuple[str, str]] = set()
    with gzip.open(STORE / "usages.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            r = json.loads(line)
            word = r["word"]
            if word not in WORDS or (word, r["text"]) in seen or r["text"] in dev_texts:
                continue
            seen.add((word, r["text"]))
            u = usage(r)
            if out_of_scope(u, table[word].get("out_of_scope", [])):
                continue
            cues = {s["key"]: s["cues"] for s in table[word]["senses"]}
            sense = cue_sense(u, cues)
            if sense is not None:
                seeds.append((word, r, sense))
    return table, seeds, [r for r in dev if r["word"] in WORDS]


def main() -> None:
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("list_path", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    table, seeds, dev = load(args.list_path)
    senses = {w: [s["key"] for s in table[w]["senses"]] for w in WORDS}
    tokenizer = AutoTokenizer.from_pretrained(XL_LEXEME, revision=XL_LEXEME_REVISION)  # type: ignore[no-untyped-call]
    encoder = AutoModel.from_pretrained(XL_LEXEME, revision=XL_LEXEME_REVISION).cuda()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    layers = encoder.encoder.layer
    for layer in layers[len(layers) - args.layers :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    heads = nn.ModuleDict({w: nn.Linear(1024, len(senses[w])) for w in WORDS}).cuda()
    weights = {
        w: torch.tensor(
            [sqrt_balanced([s for ww, _, s in seeds if ww == w]).get(k, 1.0) for k in senses[w]],
            dtype=torch.float32,
        ).cuda()
        for w in WORDS
    }

    def ids_of(row: dict[str, object]) -> list[int]:
        return usage_input_ids(tokenizer, usage(row))

    def forward(rows: list[dict[str, object]], word: str) -> torch.Tensor:
        batch = [ids_of(r) for r in rows]
        width = max(len(b) for b in batch)
        pad = int(tokenizer.pad_token_id)
        ids = torch.tensor([b + [pad] * (width - len(b)) for b in batch]).cuda()
        mask = torch.tensor([[1] * len(b) + [0] * (width - len(b)) for b in batch]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        hidden = hidden.float()
        w8 = mask.unsqueeze(-1).float()
        pooled = (hidden * w8).sum(1) / w8.sum(1).clamp(min=1e-9)
        pooled = nn.functional.normalize(pooled, dim=-1)
        logits: torch.Tensor = heads[word](pooled)
        return logits

    order = list(range(len(seeds)))
    random.shuffle(order)
    held = set(order[: len(order) // 10])
    train = [seeds[i] for i in order if i not in held]
    holdout = [seeds[i] for i in order if i in held]
    print(
        f"{len(train):,} training seeds, {len(holdout):,} held out, {len(dev):,} dev usages",
        flush=True,
    )

    optimiser = torch.optim.AdamW(
        [
            {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": args.lr},
            {"params": heads.parameters(), "lr": 1e-3},
        ],
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]

    def batches(items: list[tuple[str, dict[str, object], str]], shuffle: bool) -> list:  # type: ignore[type-arg]
        by_word: dict[str, list] = {}  # type: ignore[type-arg]
        for item in items:
            by_word.setdefault(item[0], []).append(item)
        out = []
        for word, group in by_word.items():
            if shuffle:
                random.shuffle(group)
            out += [(word, group[k : k + args.batch]) for k in range(0, len(group), args.batch)]
        if shuffle:
            random.shuffle(out)
        return out

    def holdout_accuracy() -> float:
        encoder.eval()
        right = 0
        with torch.no_grad():
            for word, group in batches(holdout, shuffle=False):
                logits = forward([r for _, r, _ in group], word)
                right += sum(
                    senses[word][int(i)] == s
                    for i, (_, _, s) in zip(logits.argmax(1), group, strict=True)
                )
        encoder.train()
        return right / max(1, len(holdout))

    print(f"before training: held-out accuracy {holdout_accuracy():.3f}", flush=True)
    encoder.train()
    for epoch in range(args.epochs):
        total, steps = 0.0, 0
        for word, group in batches(train, shuffle=True):
            targets = torch.tensor([senses[word].index(s) for _, _, s in group]).cuda()
            logits = forward([r for _, r, _ in group], word)
            loss = nn.functional.cross_entropy(logits, targets, weight=weights[word])
            optimiser.zero_grad()
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.step(optimiser)
            scaler.update()
            total += float(loss)
            steps += 1
        print(
            f"epoch {epoch + 1}: mean loss {total / steps:.4f}, "
            f"held-out accuracy {holdout_accuracy():.3f}",
            flush=True,
        )

    encoder.eval()
    predictions: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for word in WORDS:
            rows = [r for r in dev if r["word"] == word]
            for k in range(0, len(rows), 64):
                chunk = rows[k : k + 64]
                probs = torch.softmax(forward(chunk, word), dim=-1).cpu().numpy()
                for r, p in zip(chunk, probs, strict=True):
                    predictions[f"{r['set']}|{r['id']}"] = {
                        "label": senses[word][int(np.argmax(p))],
                        "probabilities": {
                            s: float(v) for s, v in zip(senses[word], p, strict=True)
                        },
                    }
    args.out.write_text(
        json.dumps(
            {
                "settings": vars(args) | {"list_path": str(args.list_path), "out": str(args.out)},
                "predictions": predictions,
            }
        ),
        encoding="utf-8",
    )
    print(f"{len(predictions):,} dev predictions written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
