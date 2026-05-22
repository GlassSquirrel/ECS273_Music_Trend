"""
01_eda.py — Exploratory Data Analysis
Produces:
  figures/eda_distributions.png   — KDE histograms of 8 key features
  figures/eda_correlation.png     — heatmap of audio feature correlations
  figures/eda_missing.png         — % missing values per column
  figures/eda_years.png           — song count by year
  figures/eda_popularity.png      — artist hotness vs song hotness scatter
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from config import DATA_PATH, FIGURES_DIR, EDA_FEATURES, STYLE, FIG_DPI

plt.style.use(STYLE)


# ── helpers ──────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace("", np.nan)
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    # sentinel: year==0 means unknown
    df["year"] = df["year"].where(df["year"] > 0, np.nan)
    return df


def save(fig: plt.Figure, name: str):
    path = f"{FIGURES_DIR}/{name}"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ── Figure 1: feature distributions ──────────────────────────────────────────

def plot_distributions(df: pd.DataFrame):
    features = list(EDA_FEATURES.keys())
    labels   = list(EDA_FEATURES.values())
    n_cols, n_rows = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7))
    fig.suptitle("Distribution of Key Audio & Popularity Features", fontsize=15, fontweight="bold")

    colors = plt.get_cmap("tab10").colors

    for ax, feat, label, color in zip(axes.flat, features, labels, colors):
        series = df[feat].dropna()
        # clip extreme outliers for readability (keep 1st–99th pct)
        lo, hi = series.quantile(0.01), series.quantile(0.99)
        series = series.clip(lo, hi)

        ax.hist(series, bins=50, color=color, alpha=0.7, edgecolor="none", density=True)

        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(series, bw_method=0.15)
        xs  = np.linspace(lo, hi, 300)
        ax.plot(xs, kde(xs), color=color, lw=2)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))

    plt.tight_layout()
    save(fig, "eda_distributions.png")


# ── Figure 2: correlation heatmap ─────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame):
    # select interpretable audio features only
    cols = [
        "tempo", "loudness", "duration", "sections_count", "bars_count",
        "tatums_count", "segments_count",
        "avg_segment_loudness_max", "avg_segment_loudness_start",
        "key", "mode", "time_signature",
        "song_hotttnesss", "artist_familiarity", "artist_hotttnesss",
        "avg_timbre_0", "avg_timbre_1", "avg_timbre_2",
        "avg_pitch_0", "avg_pitch_1", "avg_pitch_2",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # keep lower triangle
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", center=0,
        cmap="RdBu_r", vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    save(fig, "eda_correlation.png")


# ── Figure 3: missing values ──────────────────────────────────────────────────

def plot_missing(df: pd.DataFrame):
    pct_missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    pct_missing = pct_missing[pct_missing > 0]

    fig, ax = plt.subplots(figsize=(12, max(4, len(pct_missing) * 0.28)))
    colors = ["#e74c3c" if p > 50 else "#e67e22" if p > 20 else "#3498db" for p in pct_missing]
    bars = ax.barh(pct_missing.index, pct_missing.values, color=colors, edgecolor="none")
    ax.set_xlabel("% Missing", fontsize=11)
    ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")
    ax.axvline(x=50, color="red", linestyle="--", lw=1, alpha=0.5, label="50% threshold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, pct_missing.values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    save(fig, "eda_missing.png")


# ── Figure 4: songs by year ───────────────────────────────────────────────────

def plot_years(df: pd.DataFrame):
    years = df["year"].dropna().astype(int)
    years = years[years >= 1950]

    fig, ax = plt.subplots(figsize=(12, 5))
    bins = range(1950, 2012)
    counts, edges, patches = ax.hist(years, bins=bins, color="#2980b9", edgecolor="white", lw=0.4)

    # color bars by era
    era_colors = {(1950, 1970): "#8e44ad", (1970, 1985): "#e74c3c",
                  (1985, 1995): "#e67e22", (1995, 2011): "#27ae60"}
    for patch, left in zip(patches, edges):
        for (lo, hi), c in era_colors.items():
            if lo <= left < hi:
                patch.set_facecolor(c)
                break

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Songs", fontsize=12)
    ax.set_title("Song Count by Year (subset with known year)", fontsize=14, fontweight="bold")

    for (lo, hi), c in era_colors.items():
        ax.axvspan(lo, hi, alpha=0.05, color=c)

    ax.text(0.01, 0.95, f"n = {len(years):,} songs with known year",
            transform=ax.transAxes, fontsize=10, va="top", color="gray")
    plt.tight_layout()
    save(fig, "eda_years.png")


# ── Figure 5: popularity scatter ─────────────────────────────────────────────

def plot_popularity(df: pd.DataFrame):
    sub = df[["artist_hotttnesss", "song_hotttnesss", "artist_familiarity",
              "year"]].dropna(subset=["artist_hotttnesss", "song_hotttnesss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Popularity & Familiarity Landscape", fontsize=14, fontweight="bold")

    # left: artist hotness vs song hotness
    ax = axes[0]
    sc = ax.scatter(
        sub["artist_hotttnesss"], sub["song_hotttnesss"],
        c=sub["artist_familiarity"], cmap="plasma",
        alpha=0.4, s=10, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Artist Familiarity", fontsize=9)
    ax.set_xlabel("Artist Hotness", fontsize=11)
    ax.set_ylabel("Song Hotness", fontsize=11)
    ax.set_title("Hotness Space", fontsize=12)

    # right: familiarity vs artist hotness, colored by year
    sub_year = sub.dropna(subset=["year"])
    ax2 = axes[1]
    sc2 = ax2.scatter(
        sub_year["artist_familiarity"], sub_year["artist_hotttnesss"],
        c=sub_year["year"], cmap="coolwarm",
        alpha=0.4, s=10, rasterized=True,
    )
    cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.02)
    cbar2.set_label("Year", fontsize=9)
    ax2.set_xlabel("Artist Familiarity", fontsize=11)
    ax2.set_ylabel("Artist Hotness", fontsize=11)
    ax2.set_title("Familiarity vs Hotness (colored by year)", fontsize=12)

    plt.tight_layout()
    save(fig, "eda_popularity.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data…")
    df = load()
    print(f"  {len(df):,} songs, {len(df.columns)} columns")

    print("\n[1/5] Feature distributions")
    plot_distributions(df)

    print("[2/5] Correlation heatmap")
    plot_correlation(df)

    print("[3/5] Missing values")
    plot_missing(df)

    print("[4/5] Songs by year")
    plot_years(df)

    print("[5/5] Popularity scatter")
    plot_popularity(df)

    print("\nEDA complete. All figures saved to", FIGURES_DIR)
