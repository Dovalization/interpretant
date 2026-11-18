"""The transformer on target sentences with the lookup attribute ruler and several GPU processes.

`transformer_speed.py` found float16 autocast only 1.2 times faster than full precision on target
sentences: the transformer took a quarter of the pipeline's time, the entity recognizer, the
attribute ruler and the lemmatizer the rest, on the CPU, while the GPU waited. This script measures
the two remedies: the project's lookup attribute ruler (`corpus/attribute_rules.py`) in place of
spaCy's, checked equal token by token first, and 1 to 4 processes each holding the model on the one
GPU, so one process's CPU work overlaps another's GPU work (spaCy warns against `n_process` with a
GPU model, W114, so each process owns its model). Sentences are sorted by length, batches are 64
documents and 128 spans. Run from the repository root on the GPU machine, with
`PYTORCH_ALLOC_CONF=expandable_segments:True`:
`python scripts/evidence/transformer_parallel.py [SENTENCES]`.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
import traceback
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Barrier
from typing import Any

from spacy.language import Language

Sentences = list[tuple[list[str], list[bool]]]
BATCH_SIZE = 64
SPANS_PER_FORWARD = 128


def build(*, lookup: bool) -> Language:
    """The transformer in float16 without its parser, with the lookup attribute ruler or not."""
    import spacy

    from interpretant.corpus.attribute_rules import LOOKUP_ATTRIBUTE_RULER

    nlp = spacy.load(
        "en_core_web_trf",
        exclude=["parser"],
        config={
            "components": {
                "transformer": {
                    "model": {
                        "mixed_precision": True,
                        "with_spans": {"batch_size": SPANS_PER_FORWARD},
                    }
                }
            }
        },
    )
    if lookup:
        ruler: Any = nlp.get_pipe("attribute_ruler")
        rules = [
            rule
            for rule in ruler.patterns
            if not any("DEP" in token for pattern in rule["patterns"] for token in pattern)
        ]
        replaced: Any = nlp.replace_pipe("attribute_ruler", LOOKUP_ATTRIBUTE_RULER)
        replaced.add_patterns(rules)
    return nlp


def worker(share: Sentences, ready: Barrier, go: Barrier, out: Queue[float | None]) -> None:
    """Load the model on the GPU, wait for the others, annotate the share, report the seconds."""
    try:
        from spacy.tokens import Doc
        from thinc.api import require_gpu, set_gpu_allocator

        set_gpu_allocator("pytorch")
        require_gpu(0)
        nlp = build(lookup=True)
        warm = [Doc(nlp.vocab, words=w, spaces=s) for w, s in share[:128]]
        list(nlp.pipe(warm, batch_size=BATCH_SIZE))
        ready.wait(timeout=600)
        go.wait(timeout=600)
        started = time.perf_counter()
        docs = (Doc(nlp.vocab, words=w, spaces=s) for w, s in share)
        list(nlp.pipe(docs, batch_size=BATCH_SIZE))
        out.put(time.perf_counter() - started)
    except BaseException:
        sys.stdout.write("worker failed:\n" + traceback.format_exc())
        out.put(None)


def equal_tokens(sentences: Sentences) -> tuple[int, int]:
    """How many tokens get the same annotation with the lookup ruler as with spaCy's, of all."""
    from spacy.tokens import Doc
    from thinc.api import require_gpu, set_gpu_allocator

    set_gpu_allocator("pytorch")
    require_gpu(0)
    plain, lookup = build(lookup=False), build(lookup=True)
    first = list(plain.pipe([Doc(plain.vocab, words=w, spaces=s) for w, s in sentences]))
    second = list(lookup.pipe([Doc(lookup.vocab, words=w, spaces=s) for w, s in sentences]))
    same = total = 0
    for one, two in zip(first, second, strict=True):
        for a, b in zip(one, two, strict=True):
            total += 1
            same += (a.tag_, a.pos_, str(a.morph), a.lemma_, a.ent_type_) == (
                b.tag_,
                b.pos_,
                str(b.morph),
                b.lemma_,
                b.ent_type_,
            )
    return same, total


def main() -> None:
    """Check the lookup ruler in a child process, then time 1 to 4 GPU processes."""
    sys.path.insert(0, "scripts/evidence")
    from transformer_speed import target_sentences  # type: ignore[import-not-found]  # a sibling

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    sentences = sorted(target_sentences(limit), key=lambda sentence: len(sentence[0]))
    context = mp.get_context("spawn")
    with context.Pool(1) as pool:
        same, total = pool.apply(equal_tokens, (sentences[:2000],))
    sys.stdout.write(f"lookup ruler equal on {same} of {total} tokens\n")
    for processes in (1, 2, 3, 4):
        shares = [sentences[index::processes] for index in range(processes)]
        ready, go = context.Barrier(processes + 1), context.Barrier(processes + 1)
        out: Queue[float | None] = context.Queue()
        workers = [context.Process(target=worker, args=(s, ready, go, out)) for s in shares]
        for process in workers:
            process.start()
        ready.wait(timeout=900)
        started = time.perf_counter()
        go.wait(timeout=900)
        times = [out.get() for _ in workers]
        wall = time.perf_counter() - started
        for process in workers:
            process.join()
        if None in times:
            raise SystemExit(f"a worker failed with {processes} processes")
        sys.stdout.write(
            f"{processes} processes: {wall:.1f} s, {len(sentences) / wall:.0f} sentences a second\n"
        )


if __name__ == "__main__":
    main()
