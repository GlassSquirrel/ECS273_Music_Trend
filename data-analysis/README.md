# Million Song Dataset — Analysis Project

This project extracts, enriches, and analyses audio features and lyrics from the [Million Song Subset](http://millionsongdataset.com/) (10,000 songs) combined with lyric data from [musiXmatch](http://millionsongdataset.com/musixmatch/).

---

## Project Structure

```
ECS273_Music_Trend/
├── data/
│   ├── msd_subset.csv       # extracted audio/metadata (10,000 songs × 78 cols)
│   └── msd_merged.csv       # above + musiXmatch lyrics columns (× 85 cols)
└── data_analysis/
    ├── config.py
    ├── requirements.txt
    ├── 01_eda.py
    ├── 02_clustering.py
    ├── 03_temporal.py
    ├── 04_artist_geo.py
    ├── 05_lyrics.py
    ├── figures/             # generated plots
    └── results/             # generated CSVs
```

---

## Step 1 — Set Up the Python Environment

```bash
cd data-analysis
pip install -r requirements.txt
```

All packages are standard. The only optional one is `umap-learn` — if it's not installed, `02_clustering.py` will skip the UMAP plot but everything else still runs fine.

---

## Step 2 — Run the Analysis Scripts

Run each script:

```bash
python3 01_eda.py
python3 02_clustering.py
python3 03_temporal.py
python3 04_artist_geo.py
python3 05_lyrics.py
```

Figures are saved to `data-analysis/figures/` and any result CSVs to `data-analysis/results/`.

---

## What Each Script Does

### `01_eda.py` — Exploratory Data Analysis

A quick first look at the whole dataset before any modelling. Produces five figures:

| Figure | What it shows |
|--------|---------------|
| `eda_distributions.png` | KDE histograms for tempo, loudness, duration, sections, hotness, and familiarity |
| `eda_correlation.png` | Pearson correlation heatmap across ~20 audio + popularity features |
| `eda_missing.png` | Bar chart of % missing values per column (useful for knowing which fields to trust) |
| `eda_years.png` | Song count by year from 1950 to 2010, with era colour bands |
| `eda_popularity.png` | Artist hotness × song hotness scatter, coloured by familiarity and year |

---

### `02_clustering.py` — Audio Feature Clustering

The main machine learning step. Groups songs into clusters based purely on their sonic characteristics — no genre labels involved.

**Algorithm:** `StandardScaler → PCA (95% variance retained) → KMeans`

A few design choices worth knowing:
- `energy` and `danceability` are excluded — they're all 0.0 in this subset (not computed by Echo Nest).
- Musical `key` (0–11) is encoded as sin/cos so that C and B are adjacent in the feature space, not 11 units apart.
- The 12 Echo Nest timbre coefficients (MFCC-like) and 12 chroma/pitch features are the most discriminative inputs.

The script first runs an elbow + silhouette analysis across k = 2…12, picks the best k automatically, then fits the final model.

| Figure / File | What it shows |
|---------------|---------------|
| `cluster_elbow.png` | Inertia and silhouette score vs k — how we pick the number of clusters |
| `cluster_pca.png` | 2D PCA scatter coloured by cluster assignment |
| `cluster_umap.png` | 2D UMAP scatter (richer structure than PCA; requires `umap-learn`) |
| `cluster_profile.png` | Z-score heatmap of feature means per cluster — what defines each cluster |
| `cluster_radar.png` | Radar/spider chart of 6 key audio features per cluster |
| `results/cluster_labels.csv` | Original metadata + cluster assignment for every song |

---

### `03_temporal.py` — Trends Over Time

Explores how music has changed across the decades. Only uses the ~4,680 songs that have a known recording year (1950–2010).

| Figure | What it shows |
|--------|---------------|
| `temporal_song_count.png` | Songs per year (bar chart) with a 5-year rolling mean and cumulative share |
| `temporal_feature_trends.png` | Normalised trend lines for tempo, loudness, timbre, sections, and hotness — all on the same axis for easy comparison |
| `temporal_decade_boxes.png` | Box plots of tempo, loudness, sections, and timbre by decade |
| `temporal_loudness_war.png` | Median loudness over time with IQR band — the classic "loudness war" in one chart |

---

### `04_artist_geo.py` — Artist Analysis & Geography

Zooms out from individual songs to look at artists and where they come from (~3,740 songs have location data).

| Figure | What it shows |
|--------|---------------|
| `artist_top20.png` | Horizontal bar chart of the 20 most-represented artists, coloured by their hotness score |
| `artist_hotness.png` | Artist familiarity vs artist hotness scatter; dot size = number of songs, colour = avg song hotness |
| `artist_wordcloud.png` | Word cloud built from all artist genre tags across the dataset |
| `artist_geo.html` | **Interactive** Plotly map — hover over any dot to see the artist name, song title, and year |
| `artist_geo_static.png` | Same map as a static PNG for embedding in reports |

---

### `05_lyrics.py` — Lyrics Analysis

Works on the 2,350 songs (23.5%) that have musiXmatch bag-of-words data. The lyrics come as word counts, not raw text, so the analysis focuses on frequency and vocabulary metrics.

| Figure | What it shows |
|--------|---------------|
| `lyrics_coverage.png` | Pie chart of lyrics coverage + histogram of total word counts per song |
| `lyrics_top_words.png` | Bar chart of the 40 most common words across all songs with lyrics |
| `lyrics_richness.png` | Vocabulary richness (unique / total words) histogram + unique-vs-total scatter |
| `lyrics_audio_compare.png` | KDE plots comparing audio features between songs with and without lyrics; p-values from Mann-Whitney U tests shown on each panel |
| `lyrics_scatter.png` | Vocabulary richness vs song hotness scatter with a regression line, plus a bar chart of Pearson correlations between richness and other audio features |

---

## Configuration

All shared settings live in `data-analysis/config.py`:

- **`DATA_PATH`** — points to `data/msd_merged.csv` by default; change this if your file is somewhere else.
- **`CLUSTER_FEATURES`** — the full list of features fed into the clustering pipeline.
- **`FIG_DPI`** — output resolution (default 150 dpi; raise to 300 for print-quality).
- **`STYLE`** — matplotlib style applied globally across all scripts.
