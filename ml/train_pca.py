"""
train_pca_baseline.py
=====================
PCA baseline for clustering comparison (RQ1).

This script mirrors the role of train_vae.py in the pipeline:
it performs dimensionality reduction only, saving a 32-dimensional
representation for downstream clustering by cluster.py.

PCA is applied to the concatenated tri-modal feature matrix
(acoustic + lyric + tags), reducing it to PCA_DIM=32 dimensions
to match the VAE latent dimensionality for a fair comparison.

The key distinction from the VAE: PCA is a linear method that finds
directions of maximum variance globally, whereas the VAE learns a
non-linear probabilistic latent space regularised to be smooth and
semantically organised.

Input (from ../data/processed/):
    acoustic.npy, lyric.npy, tags.npy

Output:
    results_pca/latent_vectors.npy         -- (N, 32) PCA-reduced vectors
                                              (named identically to VAE output
                                               so cluster.py works unchanged)
    results_pca/figures/
        explained_variance_pca.png         -- cumulative explained variance plot

Usage (from the `ml/` directory):
    python train_pca_baseline.py

Then run clustering exactly as you would for the VAE:
    python cluster.py --latent results_pca/latent_vectors.npy
                      --output results_pca/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from logger import setup_logger

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROCESSED_DIR = "../data/processed"
RESULTS_DIR   = "results_pca"
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
PCA_DIM       = 32        # match VAE latent dim for fair comparison
RANDOM_SEED   = 42

logger = setup_logger("train_pca_baseline")
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────
# Feature concatenation
# ──────────────────────────────────────────────

def load_and_concatenate(processed_dir: str) -> np.ndarray:
    """
    Load all three modalities and concatenate into a single feature matrix.

    Concatenation order matches the VAE's modality ordering:
        [acoustic (41-dim) | lyric (50-dim) | tags (100-dim)] → 191-dim total

    No additional scaling is applied because each modality was already
    standardised in data_preprocess.py.

    Returns
    -------
    features : np.ndarray (N, 191) float32
    """
    acoustic = np.load(os.path.join(processed_dir, "acoustic.npy"))
    lyric    = np.load(os.path.join(processed_dir, "lyric.npy"))
    tags     = np.load(os.path.join(processed_dir, "tags.npy"))

    logger.info(f"  acoustic : {acoustic.shape}")
    logger.info(f"  lyric    : {lyric.shape}")
    logger.info(f"  tags     : {tags.shape}")

    features = np.concatenate([acoustic, lyric, tags], axis=1).astype(np.float32)
    logger.info(f"  Concatenated shape : {features.shape}")
    return features


# ──────────────────────────────────────────────
# PCA reduction
# ──────────────────────────────────────────────

def apply_pca(features: np.ndarray,
              n_components: int = PCA_DIM) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA on the concatenated feature matrix and reduce to n_components dims.

    Also saves a cumulative explained variance plot as a diagnostic,
    analogous to the training history plot saved by train_vae.py.

    Parameters
    ----------
    features     : (N, D) float32
    n_components : target dimensionality (default: 32 to match VAE)

    Returns
    -------
    reduced : (N, n_components) float32
    pca     : fitted PCA object
    """
    logger.info(f"Fitting PCA: {features.shape[1]}-dim → {n_components}-dim ...")
    pca     = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(features).astype(np.float32)

    explained = np.cumsum(pca.explained_variance_ratio_)
    logger.info(f"  Cumulative explained variance at {n_components} components: "
                f"{explained[-1]:.4f}")

    # Diagnostic plot — mirrors training_history.png from train_vae.py
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, n_components + 1), explained, "o-")
    plt.axhline(explained[-1], color="red", linestyle="--",
                label=f"{explained[-1]:.2%} variance retained")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance (Baseline)")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "explained_variance_pca.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")

    return reduced, pca


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load and concatenate features ────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("Step 1: Loading and concatenating tri-modal features ...")
    features = load_and_concatenate(PROCESSED_DIR)

    # ── PCA reduction ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info(f"Step 2: PCA reduction to {PCA_DIM} dimensions ...")
    pca_vectors, pca = apply_pca(features, n_components=PCA_DIM)

    # ── Save latent vectors ──────────────────────────────────────────────────
    # Named identically to train_vae.py output so cluster.py works unchanged
    logger.info("\n" + "=" * 55)
    logger.info("Step 3: Saving PCA latent vectors ...")
    latent_path = os.path.join(RESULTS_DIR, "latent_vectors.npy")
    np.save(latent_path, pca_vectors)
    logger.info(f"Saved: {latent_path}  {pca_vectors.shape}")

    logger.info("\nDone. Run clustering with:")
    logger.info(f"  python cluster.py --latent {latent_path} "
                f"--output {RESULTS_DIR}/")


if __name__ == "__main__":
    main()