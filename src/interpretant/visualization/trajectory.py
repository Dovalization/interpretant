"""Trajectory scatter plot for semantic drift visualisation."""

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def build_trajectory_figure(
    word: str,
    vectors: dict[int, np.ndarray],
    color_scale: str = "Viridis",
) -> go.Figure:
    """Build a 2-D PCA scatter plot of a word's vector trajectory across decades.

    Each point represents the word's position in the embedding space for one
    decade. Points are connected by lines to show the trajectory over time.

    Args:
        word: The word whose trajectory is being visualised.
        vectors: Mapping of decade → aligned embedding vector.
        color_scale: Plotly colour scale name for decade colouring.

    Returns:
        A Plotly Figure with the trajectory scatter plot.
    """
    decades = sorted(vectors.keys())
    matrix = np.stack([vectors[decade] for decade in decades])

    if matrix.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(matrix)
        explained = pca.explained_variance_ratio_.sum()
        axis_label = f"PCA ({explained:.1%} variance)"
    else:
        coords = matrix
        axis_label = "Dim"

    decade_labels = [str(decade) for decade in decades]
    decade_indices = list(range(len(decades)))

    fig = go.Figure()

    # Trajectory lines
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="lines",
            line={"color": "rgba(100,100,100,0.4)", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Points
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            marker={
                "size": 12,
                "color": decade_indices,
                "colorscale": color_scale,
                "showscale": True,
                "colorbar": {
                    "title": "Decade",
                    "tickvals": decade_indices,
                    "ticktext": decade_labels,
                },
            },
            text=decade_labels,
            textposition="top center",
            name=word,
            hovertemplate="<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Semantic trajectory: <i>{word}</i>",
        xaxis_title=f"{axis_label} 1",
        yaxis_title=f"{axis_label} 2",
        height=480,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig
