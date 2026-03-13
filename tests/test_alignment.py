"""Tests for alignment utilities."""

import pytest

from interpretant.alignment.procrustes import ProcrustesAligner
from interpretant.alignment.twec import TWECAligner


class TestProcrustesAligner:
    def test_instantiation(self) -> None:
        aligner = ProcrustesAligner(reference_decade=2010)
        assert aligner.reference_decade == 2010

    def test_default_reference_decade(self) -> None:
        aligner = ProcrustesAligner()
        assert aligner.reference_decade == 2010

    def test_empty_aligner_has_no_decades(self) -> None:
        aligner = ProcrustesAligner()
        assert aligner.decades() == []

    def test_empty_aligner_has_empty_shared_vocab(self) -> None:
        aligner = ProcrustesAligner()
        assert aligner.shared_vocabulary() == []

    def test_fit_raises_without_models(self, tmp_path: object) -> None:
        """fit() with no real model files should raise an error."""
        aligner = ProcrustesAligner()
        with pytest.raises(ValueError):
            aligner.fit({})


class TestTWECAligner:
    def test_fit_raises_not_implemented(self) -> None:
        aligner = TWECAligner()
        with pytest.raises(NotImplementedError):
            aligner.fit({})

    def test_align_raises_not_implemented(self, tmp_path: object) -> None:
        aligner = TWECAligner()
        with pytest.raises(NotImplementedError):
            import pathlib

            aligner.align(pathlib.Path(str(tmp_path)))
