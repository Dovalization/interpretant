# Visualization

## Streamlit dashboard

Launch with:

```bash
make app-demo   # synthetic data
make app        # real models (requires make pipeline first)
```

## Panels

### Semantic trajectory

2-D PCA projection of a word's embedding vector across decades, connected as a trajectory. Built by `build_trajectory_figure()`.

### Drift line chart

Per-decade cosine-distance series showing how quickly the word's meaning changed in each period.

### Nearest-neighbour comparison

A table showing the top-k nearest neighbours for each decade. Built by `build_neighbors_table()`.

## Plotly figures

All figures are `plotly.graph_objects.Figure` objects and can be embedded in Jupyter notebooks or other Streamlit pages.
