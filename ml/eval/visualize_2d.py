"""
visualize_2d.py
===============
Generate a 2D UMAP visualisation from the tri-modal VAE latent vectors
and cluster labels, for side-by-side comparison with the 3D UMAP (RQ3).

Location: ml/eval/visualize_2d.py
Reads inputs from: ml/results/        (../results/)
Writes outputs to: ml/eval/results/   (results/)

Input (from ../results/):
    latent_vectors.npy   -- (N, 32) from train_vae.py
    cluster_labels.npy   -- (N,)    from cluster.py

Output (saved to results/figures/):
    viz_umap_2d.html         -- interactive Plotly 2D scatter
    viz_umap_2d_static.png   -- static PNG (requires kaleido)
    viz_umap_2d_mpl.png      -- matplotlib fallback PNG (always works)

Usage (from the ml/eval/ directory):
    python visualize_2d.py

Delete results/umap_coords_2d.npy to force UMAP recomputation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from logger import setup_logger

# =============================================================
# Configuration
# This script lives in ml/eval/ so:
#   inputs  come from ../results/  (the VAE training outputs)
#   outputs go    to   results/    (local eval/results/)
# =============================================================

# -- Input paths (read from ml/results/) --
VAE_RESULTS_DIR = "../results"
LATENT_PATH     = os.path.join(VAE_RESULTS_DIR, "latent_vectors.npy")
LABELS_PATH     = os.path.join(VAE_RESULTS_DIR, "cluster_labels.npy")

# -- Output paths (written to ml/eval/results/) --
RESULTS_DIR   = "results"
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
UMAP_CACHE_2D = os.path.join(RESULTS_DIR, "umap_coords_2d.npy")
SAVE_PREFIX   = os.path.join(FIGURES_DIR, "viz")

# -- UMAP hyperparameters: identical to visualize.py for fair comparison --
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST    = 0.05
RANDOM_SEED      = 42

logger = setup_logger("visualize_2d")
np.random.seed(RANDOM_SEED)


# =============================================================
# UMAP 2D projection
# =============================================================

def compute_umap_2d(latent_vectors, cache_path=UMAP_CACHE_2D):
    """
    Project latent vectors into 2D using UMAP.

    Uses identical hyperparameters to the 3D projection in visualize.py
    so any visual difference reflects dimensionality only, not settings.
    Result is cached to ml/eval/results/umap_coords_2d.npy.
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached 2D UMAP from {cache_path} ...")
        return np.load(cache_path)

    logger.info("Computing UMAP 2D embedding (may take 1-3 min on CPU) ...")

    n_pca = min(50, latent_vectors.shape[1])
    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    latent_pca = pca.fit_transform(latent_vectors)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=RANDOM_SEED,
    )
    coords = reducer.fit_transform(latent_pca).astype(np.float32)

    np.save(cache_path, coords)
    logger.info(f"Saved 2D UMAP cache: {cache_path}")
    return coords


# =============================================================
# 2D scatter plot
# =============================================================

def plot_umap_2d(coords, labels, save_prefix=SAVE_PREFIX):
    """
    Render an interactive 2D scatter plot (Plotly HTML) and save
    static PNGs. The matplotlib fallback always works without kaleido.
    """
    import plotly.express as px

    df_2d = pd.DataFrame({
        "UMAP-1":  coords[:, 0],
        "UMAP-2":  coords[:, 1],
        "Cluster": [f"Cluster {lbl}" for lbl in labels],
    })

    fig = px.scatter(
        df_2d, x="UMAP-1", y="UMAP-2",
        color="Cluster",
        opacity=0.5,
        title="VAE Latent Space - UMAP 2D",
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend=dict(itemsizing="constant"), width=900, height=700)

    # HTML
    html_path = f"{save_prefix}_umap_2d.html"
    fig.write_html(html_path)
    logger.info(f"Saved: {html_path}")

    # Static PNG via kaleido (abspath avoids kaleido path resolution bug)
    png_path = os.path.abspath(f"{save_prefix}_umap_2d_static.png")
    try:
        fig.write_image(png_path, width=900, height=700)
        logger.info(f"Saved: {png_path}")
    except Exception as e:
        logger.warning(f"Kaleido PNG skipped: {e}")
        logger.warning("pip install kaleido  to enable static PNG export")

    # Matplotlib fallback PNG - always works, no kaleido needed
    mpl_path = os.path.abspath(f"{save_prefix}_umap_2d_mpl.png")
    fig_mpl, ax = plt.subplots(figsize=(9, 7))
    unique_clusters = sorted(df_2d["Cluster"].unique())
    colors = plt.cm.tab10.colors
    for i, cluster in enumerate(unique_clusters):
        mask = df_2d["Cluster"] == cluster
        ax.scatter(
            df_2d.loc[mask, "UMAP-1"],
            df_2d.loc[mask, "UMAP-2"],
            s=2, alpha=0.5,
            color=colors[i % len(colors)],
            label=cluster,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("VAE Latent Space - UMAP 2D")
    ax.legend(markerscale=4, bbox_to_anchor=(1.05, 1),
              loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(mpl_path, dpi=150)
    plt.close()
    logger.info(f"Saved (matplotlib): {mpl_path}")


# =============================================================
# Main
# =============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logger.info("=" * 55)
    logger.info("Loading inputs ...")
    logger.info(f"  Latent vectors : {LATENT_PATH}")
    logger.info(f"  Cluster labels : {LABELS_PATH}")

    latent_vectors = np.load(LATENT_PATH)
    labels         = np.load(LABELS_PATH)

    logger.info(f"  Latent shape   : {latent_vectors.shape}")
    logger.info(f"  Labels shape   : {labels.shape}  "
                f"(K={len(np.unique(labels))} clusters)")

    logger.info("\n" + "=" * 55)
    logger.info("Step 1: 2D UMAP projection ...")
    coords_2d = compute_umap_2d(latent_vectors, cache_path=UMAP_CACHE_2D)
    logger.info(f"  2D coords shape: {coords_2d.shape}")

    logger.info("\n" + "=" * 55)
    logger.info("Step 2: 2D UMAP scatter plot ...")
    plot_umap_2d(coords_2d, labels, save_prefix=SAVE_PREFIX)

    logger.info("\nOutput files written to: " + os.path.abspath(FIGURES_DIR))
    logger.info(f"  viz_umap_2d.html       - interactive scatter")
    logger.info(f"  viz_umap_2d_static.png - kaleido PNG")
    logger.info(f"  viz_umap_2d_mpl.png    - matplotlib PNG")


if __name__ == "__main__":
    main()
