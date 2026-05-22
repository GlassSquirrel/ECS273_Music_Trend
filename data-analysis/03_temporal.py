"""
03_temporal.py — Temporal Trend Analysis
Analyses how music has evolved from the 1950s to the 2010s.

Produces:
  figures/temporal_song_count.png     — songs per year + cumulative share
  figures/temporal_feature_trends.png — multi-feature trend lines by year
  figures/temporal_decade_boxes.png   — feature distributions by decade
  figures/temporal_loudness_war.png   — the "loudness war" visualisation
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import DATA_PATH, FIGURES_DIR, STYLE, FIG_DPI

plt.style.use(STYLE)


# ── helpers ───────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace("", np.nan)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    df["year"] = df["year"].where(df["year"] > 0, np.nan)
    return df


def with_decade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= 1950]
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "s"
    return df


def save(fig, name):
    path = f"{FIGURES_DIR}/{name}"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Figure 1: songs per year ─────────────────────────────────────────────────

def plot_song_count(df: pd.DataFrame):
    tdf = with_decade(df)
    year_counts = tdf.groupby("year").size().reset_index(name="count")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Song Count Over Time", fontsize=14, fontweight="bold")

    # decade background bands
    decade_colors = {"1950s": "#f9ebea", "1960s": "#fef9e7", "1970s": "#eafaf1",
                     "1980s": "#eaf4fb", "1990s": "#f4ecf7", "2000s": "#fdfefe"}
    for dec, col in decade_colors.items():
        yr = int(dec[:-1])
        ax1.axvspan(yr, yr + 10, alpha=0.4, color=col)

    ax1.bar(year_counts["year"], year_counts["count"],
            color="#2c3e50", alpha=0.8, width=0.8)

    # 5-year rolling mean
    yr_series = year_counts.set_index("year")["count"]
    yr_series = yr_series.reindex(range(1950, 2011), fill_value=0)
    rolling   = yr_series.rolling(5, center=True).mean()
    ax1.plot(rolling.index, rolling.values, color="#e74c3c", lw=2.5,
             label="5-year rolling mean")
    ax1.set_ylabel("Number of Songs", fontsize=11)
    ax1.legend(fontsize=10)

    # cumulative share
    cum = yr_series.cumsum() / yr_series.sum()
    ax2.fill_between(cum.index, cum.values, alpha=0.6, color="#3498db")
    ax2.set_ylabel("Cumulative\nShare", fontsize=9)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.axhline(0.5, color="gray", lw=1, linestyle="--")

    plt.tight_layout()
    save(fig, "temporal_song_count.png")


# ── Figure 2: multi-feature trend ────────────────────────────────────────────

def plot_feature_trends(df: pd.DataFrame):
    tdf = with_decade(df)

    features = {
        "tempo":    ("Tempo (BPM)", "#e74c3c"),
        "loudness": ("Loudness (dB)", "#8e44ad"),
        "avg_timbre_0": ("Timbre-0 (Power)", "#2980b9"),
        "sections_count": ("# Sections", "#27ae60"),
        "song_hotttnesss": ("Song Hotness", "#e67e22"),
    }
    features = {k: v for k, v in features.items() if k in tdf.columns}

    # annual medians
    annual = tdf.groupby("year")[list(features.keys())].median()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_title("Music Feature Trends by Year (annual median, min-max normalised)",
                 fontsize=13, fontweight="bold")

    for col, (label, color) in features.items():
        s = annual[col].dropna()
        # min-max normalise to [0,1] for comparison
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        ax.plot(s_norm.index, s_norm.values, lw=1.5, alpha=0.35, color=color)
        # smooth
        smooth = s_norm.rolling(5, center=True).mean()
        ax.plot(smooth.index, smooth.values, lw=2.5, color=color, label=label)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Normalised Value [0, 1]", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(1950, 2011)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    save(fig, "temporal_feature_trends.png")


# ── Figure 3: feature distributions by decade ────────────────────────────────

def plot_decade_boxes(df: pd.DataFrame):
    tdf = with_decade(df)
    decade_order = sorted(tdf["decade"].unique())

    features = {
        "tempo":    "Tempo (BPM)",
        "loudness": "Loudness (dB)",
        "sections_count": "# Sections",
        "avg_timbre_0":   "Timbre-0 (Power)",
    }
    features = {k: v for k, v in features.items() if k in tdf.columns}

    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    fig.suptitle("Feature Distributions by Decade", fontsize=13, fontweight="bold")

    palette = sns.color_palette("muted", len(decade_order))

    for ax, (col, label) in zip(axes, features.items()):
        data = tdf[["decade", col]].dropna()
        sns.boxplot(
            data=data, x="decade", y=col,
            order=decade_order, palette=palette,
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
            ax=ax,
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Decade", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save(fig, "temporal_decade_boxes.png")


# ── Figure 4: loudness war ────────────────────────────────────────────────────

def plot_loudness_war(df: pd.DataFrame):
    tdf = with_decade(df)
    tdf = tdf.dropna(subset=["loudness"])

    # medians and IQR bands per year
    g = tdf.groupby("year")["loudness"]
    med  = g.median()
    q25  = g.quantile(0.25)
    q75  = g.quantile(0.75)
    idx  = med.index

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(idx, q25, q75, alpha=0.25, color="#e74c3c", label="IQR (25–75%)")
    ax.plot(idx, med.rolling(3, center=True).mean(), color="#c0392b", lw=2.5,
            label="Median loudness (3-yr smoothed)")
    ax.plot(idx, med, color="#c0392b", lw=0.8, alpha=0.4)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Loudness (dB)", fontsize=11)
    ax.set_title("The Loudness War: Average Track Loudness Over Time",
                 fontsize=13, fontweight="bold")
    ax.axhline(med.mean(), color="gray", linestyle="--", lw=1,
               label=f"Overall mean ({med.mean():.1f} dB)")
    ax.legend(fontsize=10)

    # annotation for loudness war era
    ax.annotate("Loudness\nWar era",
                xy=(1995, med.loc[1995] if 1995 in med.index else med.iloc[-15]),
                xytext=(1985, med.mean() + 2),
                arrowprops={"arrowstyle": "->", "color": "#7f8c8d"},
                fontsize=10, color="#7f8c8d")

    plt.tight_layout()
    save(fig, "temporal_loudness_war.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data…")
    df = load()
    valid_yr = df["year"].notna().sum()
    print(f"  {valid_yr:,} songs with known year (≥1950)")

    print("\n[1/4] Song count per year")
    plot_song_count(df)

    print("[2/4] Multi-feature trends")
    plot_feature_trends(df)

    print("[3/4] Decade box plots")
    plot_decade_boxes(df)

    print("[4/4] Loudness war")
    plot_loudness_war(df)

    print("\nTemporal analysis complete. Figures → ", FIGURES_DIR)
