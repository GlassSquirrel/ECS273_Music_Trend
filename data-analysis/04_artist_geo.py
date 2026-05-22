"""
04_artist_geo.py — Artist Analysis & Geographic Distribution

Produces:
  figures/artist_top20.png        — top 20 artists by song count
  figures/artist_hotness.png      — familiarity × hotness scatter
  figures/artist_wordcloud.png    — artist genre tags word cloud
  figures/artist_geo.html         — interactive geographic scatter (Plotly)
  figures/artist_geo_static.png   — static world map (matplotlib)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

try:
    from wordcloud import WordCloud
    WC_AVAILABLE = True
except ImportError:
    WC_AVAILABLE = False
    print("[INFO] wordcloud not installed — skipping word cloud. pip install wordcloud")

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[INFO] plotly not installed — skipping interactive map. pip install plotly")

from config import DATA_PATH, FIGURES_DIR, STYLE, FIG_DPI

plt.style.use(STYLE)


# ── helpers ───────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace("", np.nan)
    for col in ["artist_familiarity", "artist_hotttnesss", "song_hotttnesss",
                "artist_latitude", "artist_longitude", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df["year"].where(df["year"] > 0, np.nan)
    return df


def save(fig, name):
    path = f"{FIGURES_DIR}/{name}"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Figure 1: top 20 artists ──────────────────────────────────────────────────

def plot_top_artists(df: pd.DataFrame):
    top = (
        df.groupby("artist_name")
        .agg(
            song_count=("track_id", "count"),
            avg_hotness=("artist_hotttnesss", "mean"),
            avg_familiarity=("artist_familiarity", "mean"),
        )
        .sort_values("song_count", ascending=False)
        .head(20)
        .iloc[::-1]   # flip for horizontal bar
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    cmap    = plt.get_cmap("plasma")
    norm    = mcolors.Normalize(top["avg_hotness"].min(), top["avg_hotness"].max())
    colors  = [cmap(norm(v)) for v in top["avg_hotness"]]

    bars = ax.barh(top.index, top["song_count"], color=colors, edgecolor="none")
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Artist Hotness", pad=0.02)

    for bar, cnt in zip(bars, top["song_count"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(int(cnt)), va="center", fontsize=9)

    ax.set_xlabel("Number of Songs in Subset", fontsize=11)
    ax.set_title("Top 20 Artists by Song Count\n(color = artist hotness)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "artist_top20.png")


# ── Figure 2: familiarity × hotness scatter ───────────────────────────────────

def plot_hotness_scatter(df: pd.DataFrame):
    sub = df[["artist_name", "artist_familiarity", "artist_hotttnesss",
              "song_hotttnesss", "year"]].dropna(
                  subset=["artist_familiarity", "artist_hotttnesss"])

    # one point per artist (aggregate)
    artists = (
        sub.groupby("artist_name")
        .agg(
            familiarity=("artist_familiarity", "mean"),
            artist_hot=("artist_hotttnesss", "mean"),
            song_hot=("song_hotttnesss", "mean"),
            count=("artist_name", "count"),
        )
        .reset_index()
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        artists["familiarity"], artists["artist_hot"],
        c=artists["song_hot"], cmap="YlOrRd",
        s=artists["count"] * 8 + 10,
        alpha=0.65, edgecolors="none", rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Avg Song Hotness", fontsize=9)

    # label top artists
    top_hot = artists.nlargest(8, "artist_hot")
    for _, row in top_hot.iterrows():
        ax.annotate(row["artist_name"],
                    (row["familiarity"], row["artist_hot"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, alpha=0.85)

    ax.set_xlabel("Artist Familiarity", fontsize=11)
    ax.set_ylabel("Artist Hotness", fontsize=11)
    ax.set_title("Artist Familiarity vs Hotness\n(dot size = song count, color = avg song hotness)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "artist_hotness.png")


# ── Figure 3: word cloud of genre tags ───────────────────────────────────────

def plot_wordcloud(df: pd.DataFrame):
    if not WC_AVAILABLE:
        return

    # collect all artist terms
    all_terms: dict[str, float] = {}
    for _, row in df.iterrows():
        terms = str(row.get("artist_terms", "") or "")
        if not terms or terms == "nan":
            continue
        for t in terms.split(";"):
            t = t.strip()
            if t:
                all_terms[t] = all_terms.get(t, 0) + 1

    if not all_terms:
        print("  No artist terms found — skipping word cloud")
        return

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="plasma",
        max_words=150,
        prefer_horizontal=0.9,
        collocations=False,
    ).generate_from_frequencies(all_terms)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Artist Genre Tags — Word Cloud", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    save(fig, "artist_wordcloud.png")


# ── Figure 4a: interactive Plotly geo map ─────────────────────────────────────

def plot_geo_interactive(df: pd.DataFrame):
    if not PLOTLY_AVAILABLE:
        return

    geo = df.dropna(subset=["artist_latitude", "artist_longitude"]).copy()
    geo = geo[
        (geo["artist_latitude"].between(-90, 90)) &
        (geo["artist_longitude"].between(-180, 180))
    ]

    fig = px.scatter_geo(
        geo,
        lat="artist_latitude",
        lon="artist_longitude",
        color="artist_hotttnesss",
        color_continuous_scale="Plasma",
        hover_name="artist_name",
        hover_data={"title": True, "year": True,
                    "artist_familiarity": ":.2f",
                    "artist_latitude": False, "artist_longitude": False},
        size_max=8,
        opacity=0.65,
        projection="natural earth",
        title=f"Geographic Distribution of Artists  (n = {len(geo):,})",
        labels={"artist_hotttnesss": "Artist Hotness"},
    )
    fig.update_layout(
        coloraxis_colorbar={"title": "Hotness"},
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=550,
    )
    path = f"{FIGURES_DIR}/artist_geo.html"
    fig.write_html(path)
    print(f"  saved → {path}")


# ── Figure 4b: static matplotlib geo scatter ──────────────────────────────────

def plot_geo_static(df: pd.DataFrame):
    geo = df.dropna(subset=["artist_latitude", "artist_longitude"]).copy()
    geo = geo[
        (geo["artist_latitude"].between(-90, 90)) &
        (geo["artist_longitude"].between(-180, 180))
    ]

    fig, ax = plt.subplots(figsize=(15, 8))

    # simple ocean / land background via coloured rectangles
    ax.set_facecolor("#d6eaf8")
    ax.axhspan(-90, 90, xmin=0, xmax=1, color="#d6eaf8", zorder=0)

    sc = ax.scatter(
        geo["artist_longitude"], geo["artist_latitude"],
        c=geo["artist_hotttnesss"], cmap="plasma",
        s=15, alpha=0.6, edgecolors="none", rasterized=True, zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Artist Hotness", fontsize=9)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(f"Artist Geographic Distribution  (n = {len(geo):,} songs with location data)",
                 fontsize=13, fontweight="bold")

    # region labels
    regions = [
        ("North America", -100, 45), ("Europe", 15, 50),
        ("Latin America", -60, -15), ("Asia", 100, 35),
        ("Africa", 20, 5), ("Oceania", 140, -25),
    ]
    for name, lon, lat in regions:
        ax.text(lon, lat, name, fontsize=8.5, color="#5d6d7e",
                ha="center", style="italic", alpha=0.7)

    plt.tight_layout()
    save(fig, "artist_geo_static.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data…")
    df = load()

    print("\n[1/4] Top 20 artists")
    plot_top_artists(df)

    print("[2/4] Hotness scatter")
    plot_hotness_scatter(df)

    print("[3/4] Genre tag word cloud")
    plot_wordcloud(df)

    print("[4/4] Geographic map")
    plot_geo_interactive(df)
    plot_geo_static(df)

    print("\nArtist/geo analysis complete. Figures → ", FIGURES_DIR)
