"""Heatmap visualisation for pairwise drift between decades."""

import numpy as np
import plotly.graph_objects as go


def build_heatmap_figure(
    word: str,
    vectors: dict[int, np.ndarray],
) -> go.Figure:
    """Build a heatmap of pairwise cosine distances across all decade pairs.

    Args:
        word: The word being examined.
        vectors: Mapping of decade → aligned embedding vector.

    Returns:
        A Plotly Figure showing the pairwise distance heatmap.

    Raises:
        NotImplementedError: Stub — not yet implemented.
    """
    raise NotImplementedError(
        "build_heatmap_figure is not yet implemented."
    )
