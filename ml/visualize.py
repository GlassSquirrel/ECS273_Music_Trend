"""
visualize.py
============
Generate UMAP-based visualisations from VAE latent vectors and cluster labels:

    1. Interactive 3D UMAP scatter plot (Plotly HTML)
    2. Static 2D screenshot of the scatter plot (requires kaleido)
    3. Stacked area chart of cluster proportions by decade (ThemeRiver-style)

UMAP is only used here for 2D/3D layout; all clustering was performed
upstream in cluster.py on the full 32-dim latent space.

Input (from results/):
    latent_vectors.npy   -- (N, LATENT_DIM) from train_vae.py
    cluster_labels.npy   -- (N,) from cluster.py
    msd_clustered.csv    -- metadata + cluster column from cluster.py

Output (saved to results/figures/):
    viz_umap_3d.html          -- interactive Plotly 3D scatter (open in browser)
    viz_umap_3d_static.png    -- static PNG (requires `pip install kaleido`)
    viz_decade_trend.png      -- cluster proportion by decade (stacked area)

Usage (from the `ml/` directory):
    python visualize.py

    Re-running is fast because the UMAP embedding is cached to
    results/umap_coords_3d.npy. Delete that file to recompute from scratch.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from logger import setup_logger

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RESULTS_DIR  = "results"
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")  # output folder for all plots
LATENT_PATH  = os.path.join(RESULTS_DIR, "latent_vectors.npy")
LABELS_PATH  = os.path.join(RESULTS_DIR, "cluster_labels.npy")
META_PATH    = os.path.join(RESULTS_DIR, "msd_clustered.csv")
UMAP_CACHE   = os.path.join(RESULTS_DIR, "umap_coords_3d.npy")  # cached so UMAP only runs once

RANDOM_SEED  = 42
SAVE_PREFIX  = os.path.join(FIGURES_DIR, "viz")

logger = setup_logger("visualize")

np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────
# UMAP projection
# ──────────────────────────────────────────────

def compute_umap(latent_vectors: np.ndarray,
                 n_components: int = 3,
                 cache_path: str = UMAP_CACHE) -> np.ndarray:
    """
    Project high-dimensional latent vectors into n_components dimensions
    using UMAP.

    A PCA pre-reduction to min(50, D) dims is applied first to speed up
    UMAP's graph construction on large datasets.

    The result is cached to disk so repeated visualise calls are instant.

    Parameters
    ----------
    latent_vectors : (N, D) float32
    n_components   : target dimensionality (2 or 3)
    cache_path     : .npy file to cache/load the embedding

    Returns
    -------
    coords : (N, n_components) float32
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached UMAP embedding from {cache_path} ...")
        return np.load(cache_path)

    logger.info(f"Computing UMAP {n_components}D embedding "
          f"(may take 1-3 min on CPU) ...")

    # Pre-reduce with PCA to speed up UMAP neighbourhood graph construction
    n_pca = min(50, latent_vectors.shape[1])
    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    latent_pca = pca.fit_transform(latent_vectors)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=10,
        min_dist=0.05,
        random_state=RANDOM_SEED,
    )
    coords = reducer.fit_transform(latent_pca).astype(np.float32)

    np.save(cache_path, coords)
    logger.info(f"Saved UMAP embedding cache: {cache_path}")
    return coords


# ──────────────────────────────────────────────
# 3D interactive scatter (Plotly)
# ──────────────────────────────────────────────

def plot_umap_3d(coords: np.ndarray,
                 labels: np.ndarray,
                 save_prefix: str = SAVE_PREFIX):
    """
    Render an interactive 3D scatter plot with Plotly and save as HTML.

    Each point represents one track, coloured by its cluster assignment.
    The HTML file can be opened in any browser for rotation, zoom, and hover.

    Also attempts to save a static PNG (requires `pip install kaleido`).
    """
    import plotly.express as px

    df_3d = pd.DataFrame({
        "UMAP-1":  coords[:, 0],
        "UMAP-2":  coords[:, 1],
        "UMAP-3":  coords[:, 2] if coords.shape[1] >= 3 else np.zeros(len(coords)),
        "Cluster": [f"Cluster {lbl}" for lbl in labels],
    })

    fig = px.scatter_3d(
        df_3d, x="UMAP-1", y="UMAP-2", z="UMAP-3",
        color="Cluster",
        opacity=0.5,
        title="VAE Latent Space — UMAP 3D",
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(legend=dict(itemsizing="constant"))

    html_path = f"{save_prefix}_umap_3d.html"
    fig.write_html(html_path)
    logger.info(f"Saved: {html_path}  (open in a browser to interact)")

    # Static PNG export — optional, requires kaleido
    png_path = f"{save_prefix}_umap_3d_static.png"
    try:
        fig.write_image(png_path, width=900, height=700)
        logger.info(f"Saved: {png_path}")
    except Exception:
        logger.info("(Static PNG skipped — install kaleido: pip install kaleido)")


# ──────────────────────────────────────────────
# Decade trend chart (ThemeRiver-style)
# ──────────────────────────────────────────────

def plot_decade_trend(df: pd.DataFrame,
                      save_prefix: str = SAVE_PREFIX):
    """
    Plot a stacked area chart of normalised cluster proportions per decade.

    This is the ThemeRiver-style overview that shows when each musical style
    emerged, peaked, and declined. Only tracks with a known release year are
    included.

    Parameters
    ----------
    df : DataFrame with columns 'cluster' and 'year'
    """
    df_valid = df[df["year"] > 0].copy()
    df_valid["decade"] = (df_valid["year"] // 10 * 10).astype(int)

    # Count tracks per (decade, cluster) and normalise within each decade
    pivot = (df_valid
             .groupby(["decade", "cluster"])
             .size()
             .unstack(fill_value=0))
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    pivot_norm.plot(
        kind="area", stacked=True, figsize=(10, 5),
        colormap="tab10", alpha=0.8,
    )
    plt.title("Cluster Distribution by Decade (Normalised)")
    plt.xlabel("Decade")
    plt.ylabel("Proportion")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    save_path = f"{save_prefix}_decade_trend.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")

    return pivot, pivot_norm


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # ── Create output directories ────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("Loading inputs ...")
    latent_vectors = np.load(LATENT_PATH)
    labels         = np.load(LABELS_PATH)
    df             = pd.read_csv(META_PATH)

    logger.info(f"  Latent vectors : {latent_vectors.shape}")
    logger.info(f"  Cluster labels : {labels.shape}  "
          f"(K={len(np.unique(labels))} clusters)")

    # ── Compute UMAP (or load from cache) ────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Step 1: UMAP projection ...")
    coords = compute_umap(latent_vectors, n_components=3, cache_path=UMAP_CACHE)

    # ── 3D scatter plot ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Step 2: 3D UMAP scatter plot ...")
    plot_umap_3d(coords, labels, save_prefix=SAVE_PREFIX)

    # ── Decade trend chart ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Step 3: Decade trend chart ...")
    if "year" not in df.columns:
        logger.warning("  'year' column not found in metadata -- skipping trend chart.")
    else:
        pivot, pivot_norm = plot_decade_trend(df, save_prefix=SAVE_PREFIX)
        logger.info("\nNormalised cluster proportions by decade:")
        logger.info(pivot_norm.round(3).to_string())

    logger.info("\nOutput files:")
    logger.info(f"  {SAVE_PREFIX}_umap_3d.html         — interactive 3D scatter")
    logger.info(f"  {SAVE_PREFIX}_umap_3d_static.png   — static screenshot (needs kaleido)")
    logger.info(f"  {SAVE_PREFIX}_decade_trend.png      — ThemeRiver-style trend chart")


if __name__ == "__main__":
    main()