"""
05_lyrics.py — Lyrics Analysis
Works on the 2,350 songs that have musiXmatch bag-of-words data.

Produces:
  figures/lyrics_coverage.png      — coverage pie + word count histogram
  figures/lyrics_top_words.png     — top 40 words bar chart
  figures/lyrics_richness.png      — vocabulary richness distribution
  figures/lyrics_audio_compare.png — audio feature comparison (lyrics vs no lyrics)
  figures/lyrics_scatter.png       — richness × hotness scatter + timbre correlation
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

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
    df["lyrics_available"] = pd.to_numeric(df["lyrics_available"], errors="coerce").fillna(0).astype(int)
    return df


def save(fig, name):
    path = f"{FIGURES_DIR}/{name}"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def with_lyrics(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["lyrics_available"] == 1].copy()


# ── Figure 1: coverage + word count histogram ─────────────────────────────────

def plot_coverage(df: pd.DataFrame):
    has  = (df["lyrics_available"] == 1).sum()
    no   = len(df) - has
    ldf  = with_lyrics(df)

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # left: pie
    ax1 = fig.add_subplot(gs[0])
    wedges, texts, autotexts = ax1.pie(
        [has, no],
        labels=["Has lyrics", "No lyrics"],
        colors=["#2ecc71", "#bdc3c7"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(12)
    ax1.set_title(f"Lyrics Coverage\n(n = {len(df):,} songs)", fontsize=12, fontweight="bold")

    # right: word count histogram
    ax2 = fig.add_subplot(gs[1])
    wc  = ldf["lyrics_total_words"].dropna()
    ax2.hist(wc, bins=60, color="#2ecc71", edgecolor="none", alpha=0.8)
    ax2.axvline(wc.median(), color="#e74c3c", lw=2, linestyle="--",
                label=f"Median: {wc.median():.0f} words")
    ax2.axvline(wc.mean(), color="#e67e22", lw=2, linestyle=":",
                label=f"Mean: {wc.mean():.0f} words")
    ax2.set_xlabel("Total Word Count (BoW tokens)", fontsize=11)
    ax2.set_ylabel("Number of Songs", fontsize=11)
    ax2.set_title("Word Count Distribution\n(songs with lyrics)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)

    fig.suptitle("musiXmatch Lyrics Data Overview", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "lyrics_coverage.png")


# ── Figure 2: top 40 words ────────────────────────────────────────────────────

def plot_top_words(df: pd.DataFrame):
    ldf = with_lyrics(df)

    word_freq: dict[str, int] = {}
    for _, row in ldf.iterrows():
        words  = str(row.get("lyrics_top5_words",  "") or "").split(";")
        counts = str(row.get("lyrics_top5_counts", "") or "").split(";")
        for w, c in zip(words, counts):
            w = w.strip()
            if w and w != "nan":
                try:
                    word_freq[w] = word_freq.get(w, 0) + int(c)
                except ValueError:
                    pass

    if not word_freq:
        print("  No word frequency data available — skipping")
        return

    top40 = sorted(word_freq.items(), key=lambda x: -x[1])[:40]
    words, freqs = zip(*top40)

    # colour by frequency tier
    max_f = max(freqs)
    colors = plt.get_cmap("YlOrRd")([f / max_f * 0.8 + 0.2 for f in freqs])

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(words)), freqs, color=colors, edgecolor="none")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Cumulative Count Across All Songs", fontsize=11)
    ax.set_title("Top 40 Words in musiXmatch Lyrics (bag-of-words aggregated)",
                 fontsize=13, fontweight="bold")

    # add value labels on top bars
    for bar, val in zip(bars[:10], freqs[:10]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save(fig, "lyrics_top_words.png")


# ── Figure 3: vocabulary richness ─────────────────────────────────────────────

def plot_richness(df: pd.DataFrame):
    ldf = with_lyrics(df).copy()
    ldf = ldf.dropna(subset=["lyrics_total_words", "lyrics_unique_words"])
    ldf = ldf[ldf["lyrics_total_words"] > 0]

    ldf["richness"] = ldf["lyrics_unique_words"] / ldf["lyrics_total_words"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Vocabulary Richness (unique words / total words)",
                 fontsize=13, fontweight="bold")

    # left: richness histogram
    ax = axes[0]
    r  = ldf["richness"].clip(0, 1)
    ax.hist(r, bins=50, color="#3498db", edgecolor="none", alpha=0.8)
    ax.axvline(r.median(), color="#e74c3c", lw=2, linestyle="--",
               label=f"Median: {r.median():.2f}")
    ax.set_xlabel("Richness Score", fontsize=11)
    ax.set_ylabel("Songs", fontsize=11)
    ax.set_title("Richness Distribution", fontsize=12)
    ax.legend(fontsize=10)

    # right: unique vs total scatter
    ax2 = axes[1]
    sc = ax2.scatter(
        ldf["lyrics_total_words"], ldf["lyrics_unique_words"],
        c=ldf["richness"], cmap="coolwarm", alpha=0.5, s=12, rasterized=True,
    )
    fig.colorbar(sc, ax=ax2, label="Richness", pad=0.02)

    # reference lines
    for frac, label in [(0.3, "30%"), (0.5, "50%"), (0.7, "70%")]:
        x_max = ldf["lyrics_total_words"].quantile(0.98)
        ax2.plot([0, x_max], [0, x_max * frac], lw=1, linestyle="--",
                 color="gray", alpha=0.5)
        ax2.text(x_max * 0.95, x_max * frac * 0.95, label,
                 fontsize=8, color="gray")

    ax2.set_xlabel("Total Words", fontsize=11)
    ax2.set_ylabel("Unique Words", fontsize=11)
    ax2.set_title("Unique vs Total Word Count", fontsize=12)

    plt.tight_layout()
    save(fig, "lyrics_richness.png")


# ── Figure 4: audio comparison (lyrics vs no-lyrics) ─────────────────────────

def plot_audio_compare(df: pd.DataFrame):
    features = {
        "tempo":    "Tempo (BPM)",
        "loudness": "Loudness (dB)",
        "duration": "Duration (s)",
        "song_hotttnesss": "Song Hotness",
        "artist_familiarity": "Artist Familiarity",
        "sections_count": "# Sections",
    }
    features = {k: v for k, v in features.items() if k in df.columns}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Audio Features: Songs With Lyrics vs Without",
                 fontsize=13, fontweight="bold")

    for ax, (col, label) in zip(axes.flat, features.items()):
        grp = df[[col, "lyrics_available"]].dropna()
        grp[col] = pd.to_numeric(grp[col], errors="coerce")
        grp = grp.dropna()

        has  = grp[grp["lyrics_available"] == 1][col]
        no   = grp[grp["lyrics_available"] == 0][col]

        # clip to IQR×3 for readability
        lo = grp[col].quantile(0.01)
        hi = grp[col].quantile(0.99)

        sns.kdeplot(has.clip(lo, hi), ax=ax, color="#2ecc71",
                    fill=True, alpha=0.35, label=f"With lyrics (n={len(has):,})")
        sns.kdeplot(no.clip(lo, hi),  ax=ax, color="#e74c3c",
                    fill=True, alpha=0.35, label=f"No lyrics (n={len(no):,})")

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(has.dropna(), no.dropna(), alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{label}  (p={p:.3f} {sig})", fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.8)

    plt.tight_layout()
    save(fig, "lyrics_audio_compare.png")


# ── Figure 5: richness × hotness scatter ─────────────────────────────────────

def plot_richness_scatter(df: pd.DataFrame):
    ldf = with_lyrics(df).copy()
    ldf = ldf.dropna(subset=["lyrics_total_words", "lyrics_unique_words",
                             "song_hotttnesss"])
    ldf = ldf[ldf["lyrics_total_words"] > 0]
    ldf["richness"] = (ldf["lyrics_unique_words"] / ldf["lyrics_total_words"]).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Lyrics Richness vs Audio Features", fontsize=13, fontweight="bold")

    # left: richness vs song hotness
    ax = axes[0]
    sc = ax.scatter(ldf["richness"], ldf["song_hotttnesss"],
                    c=ldf["lyrics_total_words"], cmap="viridis",
                    alpha=0.5, s=15, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Total Words", pad=0.02)

    # regression line
    m, b, r, p, _ = stats.linregress(ldf["richness"].fillna(0),
                                      ldf["song_hotttnesss"].fillna(0))
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, m * xs + b, color="#e74c3c", lw=2,
            label=f"r = {r:.2f}  (p = {p:.3f})")
    ax.set_xlabel("Vocabulary Richness", fontsize=11)
    ax.set_ylabel("Song Hotness", fontsize=11)
    ax.set_title("Richness vs Song Hotness", fontsize=12)
    ax.legend(fontsize=10)

    # right: correlation bar chart of richness vs audio features
    ax2 = axes[1]
    corr_cols = ["tempo", "loudness", "duration", "sections_count",
                 "avg_timbre_0", "avg_timbre_1", "avg_timbre_2",
                 "artist_familiarity", "song_hotttnesss"]
    corr_cols = [c for c in corr_cols if c in ldf.columns]

    corrs = {}
    for col in corr_cols:
        vals = pd.to_numeric(ldf[col], errors="coerce")
        valid = ldf["richness"].notna() & vals.notna()
        if valid.sum() > 10:
            r_val, _ = stats.pearsonr(ldf.loc[valid, "richness"], vals[valid])
            corrs[col] = r_val

    corr_series = pd.Series(corrs).sort_values()
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_series]
    corr_series.plot(kind="barh", ax=ax2, color=colors, edgecolor="none")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Pearson r with Vocabulary Richness", fontsize=10)
    ax2.set_title("Correlation of Richness with Audio Features", fontsize=12)
    ax2.set_xlim(-0.5, 0.5)

    plt.tight_layout()
    save(fig, "lyrics_scatter.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data…")
    df = load()
    n_lyrics = (df["lyrics_available"] == 1).sum()
    print(f"  {n_lyrics:,} / {len(df):,} songs have lyrics")

    print("\n[1/5] Coverage + word count")
    plot_coverage(df)

    print("[2/5] Top 40 words")
    plot_top_words(df)

    print("[3/5] Vocabulary richness")
    plot_richness(df)

    print("[4/5] Audio feature comparison")
    plot_audio_compare(df)

    print("[5/5] Richness × hotness scatter")
    plot_richness_scatter(df)

    print("\nLyrics analysis complete. Figures → ", FIGURES_DIR)
