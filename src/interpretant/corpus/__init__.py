"""Corpus compilation: collecting abstracts, repairing and annotating them, and cleaning.

The rebuilt modules (repair, and the annotation and cleaning that follow) are exported here;
the pre-rebuild modules (openalex, lemmas, phrases, quality, caps, files, audit,
preprocessing) are imported by path until the rebuild replaces them.
"""

from interpretant.corpus.repair import compile_signatures, has_mojibake, repair_text

__all__ = ["compile_signatures", "has_mojibake", "repair_text"]
