"""Streamlit dashboard for interactive semantic drift exploration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

VOCAB = ["network", "atom", "information", "model", "evolution", "structure"]
DECADES = [1960, 1970, 1980, 1990, 2000, 2010]
VECTOR_DIM = 300
SEED = 42


def _make_synthetic_vectors() -> dict[str, dict[int, np.ndarray]]:
    """Return seeded random vectors for the demo vocab across all decades.

    Vectors are unit-normalised and jittered per decade to simulate drift.
    """
    rng = np.random.default_rng(SEED)
    base_vectors = {word: rng.standard_normal(VECTOR_DIM) for word in VOCAB}
    # Normalise bases
    base_vectors = {w: v / np.linalg.norm(v) for w, v in base_vectors.items()}

    result: dict[str, dict[int, np.ndarray]] = {word: {} for word in VOCAB}
    for decade_idx, decade in enumerate(DECADES):
        noise_scale = 0.05 * (decade_idx + 1)
        for word in VOCAB:
            noise = rng.standard_normal(VECTOR_DIM) * noise_scale
            vector = base_vectors[word] + noise
            result[word][decade] = vector / np.linalg.norm(vector)
    return result


def _group_vectors_by_decade(
    vectors: dict[str, dict[int, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    """Invert the {word: {decade: vector}} index to {decade: {word: vector}}."""
    by_decade: dict[int, dict[str, np.ndarray]] = {}
    for word, decade_dict in vectors.items():
        for decade, vector in decade_dict.items():
            by_decade.setdefault(decade, {})[word] = vector
    return by_decade


def _decade_drift_series(word_vectors: dict[int, np.ndarray]) -> pd.Series:
    """Compute consecutive cosine-distance drift for one word across decades."""
    from interpretant.drift.metrics import cosine_distance

    sorted_decades = sorted(word_vectors.keys())
    return pd.Series(
        {
            t1: cosine_distance(word_vectors[t0], word_vectors[t1])
            for t0, t1 in zip(sorted_decades, sorted_decades[1:], strict=False)
        },
        name="drift",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def main(demo: bool = False) -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="interpretant — semantic drift tracker",
        page_icon="📖",
        layout="wide",
    )

    st.title("interpretant")
    st.caption("Diachronic semantic drift tracker for academic corpora")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Settings")

        if demo:
            st.info("Running in **demo mode** with synthetic data.")
            vectors = _make_synthetic_vectors()
            available_words = sorted(vectors.keys())
        else:
            aligned_dir = Path("models/aligned")
            if not aligned_dir.exists() or not any(aligned_dir.glob("*.kv")):
                st.warning(
                    "No aligned models found in `models/aligned/`. "
                    "Run `make pipeline` first, or launch with `make app-demo`."
                )
                st.stop()
            else:
                from interpretant.alignment.procrustes import ProcrustesAligner

                aligner = ProcrustesAligner()
                aligner.load_aligned(aligned_dir)
                available_words = aligner.shared_vocabulary()
                vectors = {
                    word: {decade: aligner.get_vector(word, decade) for decade in aligner.decades()}
                    for word in available_words
                }

        selected_word = st.selectbox("Select word", available_words, index=0)
        k_neighbors = st.slider("Neighbours (k)", min_value=5, max_value=30, value=10)

    # ---- Layout: two columns ----
    col_left, col_right = st.columns([3, 2])

    word_vectors = vectors[selected_word]

    # Panel 1: Trajectory scatter
    with col_left:
        st.subheader("Semantic trajectory")
        from interpretant.visualization.trajectory import build_trajectory_figure

        fig = build_trajectory_figure(selected_word, word_vectors)
        st.plotly_chart(fig, use_container_width=True)

    # Panel 2: Drift line chart
    with col_right:
        st.subheader("Drift over time (cosine distance)")
        word_drift = _decade_drift_series(word_vectors)
        st.line_chart(word_drift, use_container_width=True, height=240)

        st.metric(
            "Total trajectory dispersion",
            f"{word_drift.sum():.4f}",
            help="Sum of consecutive cosine distances across all decades.",
        )

    # Panel 3 & 4: Neighbour comparison table
    st.subheader("Nearest-neighbour comparison")
    from interpretant.visualization.neighbors import build_neighbors_table

    neighbors_df = build_neighbors_table(
        selected_word, _group_vectors_by_decade(vectors), k=k_neighbors
    )
    st.dataframe(neighbors_df, use_container_width=True)


if __name__ == "__main__":
    # Detect --demo flag when launched via `streamlit run main.py -- --demo`
    demo_mode = "--demo" in sys.argv
    main(demo=demo_mode)
