"""
cluster.py
==========
Find the optimal number of clusters K using Silhouette + Elbow analysis,
run KMeans on the VAE latent vectors, and save cluster assignments.

Clustering is performed directly on the full 32-dimensional latent space
(not on UMAP projections) to preserve as much geometric information as
possible. UMAP is used only for visualisation in visualize.py.

Input:
    results/latent_vectors.npy        -- (N, LATENT_DIM) from train_vae.py
    ../data/processed/meta.csv        -- track metadata (track_id, year, etc.)

Output:
    results/cluster_labels.npy        -- (N,) integer cluster assignments
    results/msd_clustered.csv         -- meta.csv with 'cluster' column added
    results/figures/cluster_selection.png -- Elbow + Silhouette plots for K selection

Usage (from the `ml/` directory):
    python cluster.py [--k K]

    If --k is provided, that value is used directly (skips K search).
    Otherwise the script sweeps K_RANGE and picks the best by Silhouette.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from logger import setup_logger

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RESULTS_DIR  = "results"
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")  # output folder for all plots
LATENT_PATH  = os.path.join(RESULTS_DIR, "latent_vectors.npy")
META_PATH    = "../data/processed/meta.csv"
K_RANGE      = range(4, 11)      # K values to evaluate during search
RANDOM_SEED  = 42

logger = setup_logger("cluster")

np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────
# K selection
# ──────────────────────────────────────────────

def find_optimal_k(vectors: np.ndarray,
                   k_range=K_RANGE,
                   save_path: str = os.path.join(FIGURES_DIR, "cluster_selection.png")) -> int:
    """
    Sweep k_range and evaluate each K using Inertia (Elbow) and Silhouette.

    Silhouette score is computed with cosine distance on a sample of up to
    2000 points for speed. The K that maximises Silhouette is returned.

    Parameters
    ----------
    vectors   : (N, D) float32 latent vectors
    k_range   : iterable of candidate K values
    save_path : path for the diagnostic plot

    Returns
    -------
    best_k : int
    """
    inertias, silhouettes = [], []

    logger.info(f"Sweeping K in {list(k_range)} on {vectors.shape[1]}-dim latent space ...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(vectors)
        inertias.append(km.inertia_)
        sil = silhouette_score(
            vectors, labels,
            metric="cosine",
            sample_size=min(2000, len(vectors)),
            random_state=RANDOM_SEED,
        )
        silhouettes.append(sil)
        logger.info(f"  k={k:2d} | inertia={km.inertia_:10.1f} | silhouette={sil:.4f}")

    # Plot Elbow and Silhouette side by side
    k_list = list(k_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(k_list, inertias, "o-")
    ax1.set(title="Elbow Method", xlabel="K", ylabel="Inertia")
    ax2.plot(k_list, silhouettes, "o-", color="orange")
    ax2.set(title="Silhouette Score", xlabel="K", ylabel="Score")
    best_k = k_list[int(np.argmax(silhouettes))]
    ax2.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")

    logger.info(f"\nBest K by Silhouette score: {best_k}")
    return best_k


# ──────────────────────────────────────────────
# KMeans clustering
# ──────────────────────────────────────────────

def run_clustering(vectors: np.ndarray,
                   n_clusters: int,
                   n_init: int = 20) -> tuple[np.ndarray, KMeans]:
    """
    Fit KMeans with n_init restarts and return cluster labels.

    Using more restarts (n_init=20) than the default reduces the chance of
    landing in a poor local minimum on the 32-dim latent space.

    Returns
    -------
    labels : np.ndarray (N,) int
    km     : fitted KMeans object
    """
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=n_init)
    labels = km.fit_predict(vectors)
    final_sil = silhouette_score(vectors, labels, metric="cosine",
                                 sample_size=min(2000, len(vectors)),
                                 random_state=RANDOM_SEED)
    logger.info(f"KMeans (k={n_clusters}, n_init={n_init}): "
          f"silhouette (cosine) = {final_sil:.4f}")
    return labels, km


# ──────────────────────────────────────────────
# Cluster summary
# ──────────────────────────────────────────────

def print_cluster_summary(df: pd.DataFrame):
    """Print per-cluster size, decade distribution, and top artist tags."""
    summary_cols = [c for c in ["loudness", "tempo", "mode"] if c in df.columns]

    logger.info("\nCluster sizes:")
    logger.info(df["cluster"].value_counts().sort_index().to_string())

    if summary_cols:
        logger.info("\nCluster means (selected acoustic features):")
        logger.info(df.groupby("cluster")[summary_cols].mean().round(3).to_string())

    if "year" in df.columns:
        df["decade"] = (df["year"].clip(lower=1) // 10 * 10).astype(int)
        logger.info("\nDecade × cluster counts:")
        logger.info(df[df["year"] > 0]
              .groupby(["decade", "cluster"])
              .size()
              .unstack(fill_value=0)
              .to_string())

    if "artist_terms" in df.columns:
        logger.info("\nTop 5 artist tags per cluster:")
        for c in sorted(df["cluster"].unique()):
            terms = []
            for raw in df.loc[df["cluster"] == c, "artist_terms"].dropna():
                terms.extend([t.strip() for t in str(raw).split(";") if t.strip()])
            top5 = [t for t, _ in Counter(terms).most_common(5)]
            logger.info(f"  Cluster {c}: {top5}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cluster VAE latent vectors")
    parser.add_argument("--k", type=int, default=None,
                        help="Fix K (skips automatic search if provided)")
    args = parser.parse_args()

    # ── Load latent vectors ──────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("Loading latent vectors ...")
    vectors = np.load(LATENT_PATH)
    logger.info(f"  Shape: {vectors.shape}")

    # ── Find optimal K (or use the user-supplied value) ──────────────────────
    logger.info("\n" + "=" * 55)
    if args.k is not None:
        best_k = args.k
        logger.info(f"Using user-specified K = {best_k} (skipping K search)")
    else:
        logger.info("Step 1: K selection")
        best_k = find_optimal_k(vectors, k_range=K_RANGE)

    # ── Run KMeans ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info(f"Step 2: KMeans clustering with K={best_k} ...")
    labels, km = run_clustering(vectors, n_clusters=best_k)

    # ── Save outputs ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("Saving outputs ...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    labels_path = os.path.join(RESULTS_DIR, "cluster_labels.npy")
    np.save(labels_path, labels)
    logger.info(f"Saved: {labels_path}")

    # Attach labels to metadata and save
    meta_df = pd.read_csv(META_PATH)
    meta_df["cluster"] = labels
    clustered_path = os.path.join(RESULTS_DIR, "msd_clustered.csv")
    meta_df.to_csv(clustered_path, index=False)
    logger.info(f"Saved: {clustered_path}")

    # ── Print summary ────────────────────────────────────────────────────────
    print_cluster_summary(meta_df)

if __name__ == "__main__":
    main()