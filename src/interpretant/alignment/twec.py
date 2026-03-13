"""TWEC (Temporal Word Embeddings with a Compass) aligner.

.. note::

    TWEC requires a custom gensim fork that is not available on PyPI.
    Install it with::

        pip install git+https://github.com/valedica/gensim.git

    The standard gensim package will NOT work for this aligner.
    Use the Procrustes aligner as a drop-in replacement if you cannot
    install the custom fork.
"""

from pathlib import Path

import numpy as np

from interpretant.alignment.base import Aligner


class TWECAligner(Aligner):
    """TWEC temporal alignment.

    Trains a shared 'compass' embedding and then aligns each decade's
    model to it using the TWEC training procedure.
    """

    def fit(self, model_paths: dict[int, Path]) -> None:
        raise NotImplementedError(
            "TWECAligner requires the custom gensim fork from "
            "https://github.com/valedica/gensim.git — see module docstring."
        )

    def align(self, output_dir: Path) -> dict[int, Path]:
        raise NotImplementedError

    def get_vector(self, word: str, decade: int) -> np.ndarray:
        raise NotImplementedError

    def shared_vocabulary(self) -> list[str]:
        raise NotImplementedError

    def decades(self) -> list[int]:
        raise NotImplementedError

    def load_aligned(self, model_dir: Path) -> None:
        raise NotImplementedError
