"""
test_02_clustering.py - Audio feature clustering test variant
Algorithm: StandardScaler -> PCA (whitening) -> KMeans

Produces:
  figures/test_cluster_elbow.png    - inertia + silhouette vs k
  figures/test_cluster_pca.png      - PCA 2D scatter colored by cluster
  figures/test_cluster_umap.png     - UMAP 2D scatter colored by cluster (if umap installed)
  figures/test_cluster_profile.png  - normalised feature heatmap per cluster
  figures/test_cluster_radar.png    - radar chart of cluster audio signatures
  results/test_cluster_labels.csv   - selected metadata + cluster assignment
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[INFO] umap-learn not found - skipping UMAP plot. Install with: pip install umap-learn")

from test_config import (
    DATA_PATH,
    FIGURES_DIR,
    RESULTS_DIR,
    CLUSTER_FEATURES,
    PALETTE,
    STYLE,
    FIG_DPI,
)

plt.style.use(STYLE)
RNG = 42


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_df, feature_matrix_df) with circular key encoding."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace("", np.nan)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    df["year"] = df["year"].where(df["year"] > 0, np.nan)
    df["key_sin"] = np.sin(2 * np.pi * df["key"] / 12)
    df["key_cos"] = np.cos(2 * np.pi * df["key"] / 12)

    feat_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    feat_df = df[feat_cols].copy()
    return df, feat_df


def preprocess(feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Impute -> scale -> PCA-whiten. Returns (X_scaled, X_pca)."""
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(feat_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95, whiten=True, random_state=RNG)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA: {X_pca.shape[1]} components explain >=95% variance")

    return X_scaled, X_pca


def find_optimal_k(X_pca: np.ndarray, k_range=range(2, 13)) -> int:
    inertias, silhouettes = [], []
    print("  Computing elbow curve...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RNG, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, labels, sample_size=3000, random_state=RNG))
        print(f"    k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Choosing the Number of Clusters (Test Variant)", fontsize=13, fontweight="bold")

    ks = list(k_range)
    ax1.plot(ks, inertias, "o-", color="#2980b9", lw=2, ms=7)
    ax1.set_xlabel("k", fontsize=11)
    ax1.set_ylabel("Inertia (within-cluster SSE)", fontsize=11)
    ax1.set_title("Elbow Method", fontsize=12)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))

    ax2.plot(ks, silhouettes, "s-", color="#e74c3c", lw=2, ms=7)
    ax2.set_xlabel("k", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Silhouette Score", fontsize=12)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1))

    best_k = ks[int(np.argmax(silhouettes))]
    ax2.axvline(best_k, color="#e74c3c", linestyle="--", lw=1.5, label=f"Best k = {best_k}")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    path = f"{FIGURES_DIR}/test_cluster_elbow.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")
    return best_k


def fit_kmeans(X_pca: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=RNG, n_init=20, max_iter=500)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=5000, random_state=RNG)
    print(f"  Final KMeans k={k}: silhouette = {sil:.4f}")
    return labels


def plot_pca(X_scaled: np.ndarray, labels: np.ndarray, k: int):
    pca2 = PCA(n_components=2, random_state=RNG)
    coords = pca2.fit_transform(X_scaled)

    cmap = plt.get_cmap(PALETTE)
    colors = [cmap(i / k) for i in range(k)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for c_idx in range(k):
        mask = labels == c_idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=colors[c_idx],
            alpha=0.5,
            s=8,
            rasterized=True,
            label=f"Cluster {c_idx}",
        )

    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}% var)", fontsize=11)
    ax.set_title(f"Test KMeans Clusters in PCA Space (k = {k})", fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=10, framealpha=0.8, loc="best", ncol=2 if k > 6 else 1)

    path = f"{FIGURES_DIR}/test_cluster_pca.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_umap(X_scaled: np.ndarray, labels: np.ndarray, k: int):
    if not UMAP_AVAILABLE:
        return

    print("  Running UMAP (this may take ~1-2 min)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=RNG,
    )
    emb = reducer.fit_transform(X_scaled)

    cmap = plt.get_cmap(PALETTE)
    colors = [cmap(i / k) for i in range(k)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for c_idx in range(k):
        mask = labels == c_idx
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            color=colors[c_idx],
            alpha=0.5,
            s=8,
            rasterized=True,
            label=f"Cluster {c_idx}",
        )

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(f"Test KMeans Clusters in UMAP Space (k = {k})", fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=10, framealpha=0.8, loc="best", ncol=2 if k > 6 else 1)
    ax.set_xticks([])
    ax.set_yticks([])

    path = f"{FIGURES_DIR}/test_cluster_umap.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_profile(df: pd.DataFrame, labels: np.ndarray, k: int):
    profile_cols = [
        "tempo",
        "loudness",
        "avg_segment_loudness_max",
        "key_sin",
        "key_cos",
        "avg_timbre_0",
        "avg_timbre_1",
        "avg_timbre_2",
        "avg_timbre_3",
    ]
    profile_cols = [c for c in profile_cols if c in df.columns]

    sub = df[profile_cols].copy()
    for col in profile_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    sub["cluster"] = labels
    means = sub.groupby("cluster")[profile_cols].mean()
    z = (means - means.mean()) / (means.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, max(4, k * 0.9)))
    sns.heatmap(
        z,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 8},
        ax=ax,
        cbar_kws={"label": "Z-score (across clusters)"},
    )
    ax.set_title("Test Cluster Audio Profiles", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cluster", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)

    plt.tight_layout()
    path = f"{FIGURES_DIR}/test_cluster_profile.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_radar(df: pd.DataFrame, labels: np.ndarray, k: int):
    radar_cols = [
        "tempo",
        "loudness",
        "avg_segment_loudness_max",
        "avg_timbre_0",
        "avg_timbre_1",
        "avg_timbre_2",
    ]
    radar_cols = [c for c in radar_cols if c in df.columns]

    sub = df[radar_cols].copy()
    for col in radar_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub["cluster"] = labels

    means = sub.groupby("cluster")[radar_cols].mean()
    normed = (means - means.min()) / (means.max() - means.min() + 1e-9)

    n = len(radar_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    cmap = plt.get_cmap(PALETTE)

    for c_idx, row in normed.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, lw=2, label=f"Cluster {c_idx}", color=cmap(c_idx / k))
        ax.fill(angles, vals, alpha=0.1, color=cmap(c_idx / k))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ["Tempo", "Loudness", "Seg Loud Max", "Timbre-0", "Timbre-1", "Timbre-2"][:n],
        fontsize=10,
    )
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25%", "50%", "75%"], fontsize=8)
    ax.set_title("Test Cluster Audio Signature (Radar)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    plt.tight_layout()
    path = f"{FIGURES_DIR}/test_cluster_radar.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def save_labels(df: pd.DataFrame, labels: np.ndarray):
    out = df[
        [
            "track_id",
            "title",
            "artist_name",
            "year",
            "tempo",
            "loudness",
            "avg_segment_loudness_max",
            "key",
        ]
    ].copy()
    out["cluster"] = labels
    path = f"{RESULTS_DIR}/test_cluster_labels.csv"
    out.to_csv(path, index=False)
    print(f"  saved -> {path}")

    summary = df.copy()
    summary["cluster"] = labels
    for col in ["tempo", "loudness", "avg_segment_loudness_max"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    grp = summary.groupby("cluster").agg(
        count=("track_id", "count"),
        avg_tempo=("tempo", "mean"),
        avg_loudness=("loudness", "mean"),
        avg_segment_loudness_max=("avg_segment_loudness_max", "mean"),
    ).round(3)
    print("\nCluster summary:")
    print(grp.to_string())


if __name__ == "__main__":
    print("Loading and engineering test features...")
    df, feat_df = load_features()
    print(f"  {len(df):,} songs, {len(feat_df.columns)} cluster features")

    print("\nPreprocessing (impute -> scale -> PCA)...")
    X_scaled, X_pca = preprocess(feat_df)

    print("\n[1] Elbow + silhouette analysis...")
    best_k = find_optimal_k(X_pca)
    k = best_k
    print(f"  -> Using k = {k}")

    print(f"\n[2] Fitting final KMeans (k={k})...")
    labels = fit_kmeans(X_pca, k)

    print("\n[3] PCA 2D scatter...")
    plot_pca(X_scaled, labels, k)

    print("\n[4] UMAP 2D scatter...")
    plot_umap(X_scaled, labels, k)

    print("\n[5] Cluster profile heatmap...")
    plot_profile(df, labels, k)

    print("\n[6] Radar chart...")
    plot_radar(df, labels, k)

    print("\n[7] Saving cluster labels...")
    save_labels(df, labels)

    print("\nTest clustering complete. Figures ->", FIGURES_DIR)
