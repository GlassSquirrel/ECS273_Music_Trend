# Music Trend Analysis — Frontend

Interactive visualization dashboard for multi-modal music trend analysis (1960–2010), built on the Million Song Dataset (MSD) with VAE-based latent embeddings and KMeans clustering.

## Running the App

```bash
cd client
npm install      # first time only
npm run dev      # starts dev server at http://localhost:5173
```

## Tech Stack

| | |
|---|---|
| Framework | React 18 (Vite) |
| 2D Charts | D3.js v7 |
| 3D Rendering | Three.js v0.184 |
| Styling | Inline styles + CSS variables (light / dark mode) |

---

## Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│  Header                                             │
├──────────────────────────────┬──────────────────────┤
│  ThemeRiver                  │  Stats               │
│                              ├──────────────────────┤
├──────────────────┬───────────┤  UMAP 3D Scatter     │
│  Audio Features  │  Artist   │  (expandable)        │
│                  │  Tags     │                      │
└──────────────────┴───────────┴──────────────────────┘
```

All panels share a single `activeCluster` state (0–7, or `null` = all clusters). Clicking a cluster in any panel highlights it across the entire dashboard.

---

## Panels

### ThemeRiver

**What it shows** — How the relative share of each cluster's songs evolves from 1960 to 2010. Wider bands in a given year mean more songs assigned to that cluster. The wiggle layout (D3 `stackOffsetWiggle`) centres the river around a baseline to reduce visual distortion.

**Data source** — `src/themeRiverData.json`, pre-computed from `ml/results/msd_clustered.csv`.

**Computation** — Songs with unknown year (`year = 0`) are excluded. The remaining 4,634 tracks (1960–2010) are grouped by `(year, cluster)` and counted. The result is a 51-row × 8-cluster count matrix.

**Interaction** — Hover to read the exact year and cluster count. Click a cluster label in the legend to highlight it; click again to return to the all-clusters view.

---

### UMAP 3D Cluster Scatter

**What it shows** — A 3-D spatial layout of all 10,000 songs, colour-coded by cluster. Songs that sound and feel similar (in the VAE latent space) cluster together. Three orthogonal axes (UMAP-1, UMAP-2, UMAP-3) are shown with a semi-transparent grid box for spatial reference.

**Data source** — `src/clusterData.json` — 2,000 uniformly sampled points from `ml/results/umap_coords_3d.npy` (UMAP embedding) and `ml/results/cluster_labels.npy`, joined with track metadata from `ml/results/msd_clustered.csv`.

**Computation pipeline**
1. VAE (32-dim latent space) trained on fused acoustic + lyric + tag features.
2. PCA pre-reduction → UMAP (3 components, cached in `umap_coords_3d.npy`).
3. Coordinates normalised to a [−6, 6]³ cube for rendering.
4. Each cluster rendered as a separate `THREE.Points` cloud with `NormalBlending` and a soft-circle sprite texture.

**Interaction**
- **Drag** to orbit, **scroll** to zoom. Auto-rotation resumes 3.5 s after interaction ends.
- **Hover** a point to see the song title, artist, year, and cluster.
- **Click** a point to select its cluster.
- **↗ button** (top-right of card) expands to full-screen. Press **Esc**, click the **✕** button, or click the dark backdrop to exit.

---

### Audio Features

**What it shows** — Six acoustic characteristics averaged across all tracks in the selected cluster, displayed as normalised bar charts (0 = lowest across all clusters, 1 = highest).

**Data source** — `src/audioFeatureData.json`, pre-computed from `data/processed/acoustic.npy`.

**Computation**
1. `acoustic.npy` holds 39 standardised acoustic features (StandardScaler fit on the full dataset).
2. The scaler is inverted (`transformers.pkl → acoustic_scaler`) to recover original-scale values.
3. Per-cluster means are computed over all tracks in that cluster.
4. Each of the six selected features is min-max normalised across the eight cluster means so relative differences are preserved.

**Features displayed**

| Label | Source field | Meaning |
|---|---|---|
| Loudness | `loudness` | Average track loudness (dB) |
| Tempo | `tempo` | Beats per minute |
| Major / Minor | `mode` | 1 = major key, 0 = minor key |
| Key Clarity | `key_confidence` | Confidence in detected key (0–1) |
| Brightness | `avg_timbre_1` | Spectral brightness (MIR timbre dim 2) |
| Rhythm Clarity | `time_signature_confidence` | Confidence in detected time signature |

---

### Artist Tags

**What it shows** — The most characteristic genre/style tags for the selected cluster (or, in all-clusters mode, the most frequent tags across the entire dataset). Font size scales with tag weight.

**Data source** — `src/wordCloudData.json`, pre-computed from the `artist_terms` column in `ml/results/msd_clustered.csv`.

**Computation — per-cluster view (keys `"0"`–`"7"`)**
Each tag is scored with cluster-level TF-IDF:

```
score(tag, cluster) = (doc_freq_in_cluster / cluster_size)
                    × (log((N_clusters + 1) / (N_clusters_with_tag + 1)) + 0.5)
```

Tags that appear in fewer than 2% of a cluster's tracks are excluded. Generic tags (`rock`, `pop`, `alternative`) are in a stop-list to emphasise cluster-specific characteristics. Top 15 tags per cluster are retained, with weights normalised to [0, 1].

**Computation — all-clusters view (key `"all"`)**
Raw document frequency across all 10,000 tracks, no IDF. Tags must appear in at least 1% of all tracks. Top 18 tags are retained.

---

## Data Files (`src/`)

| File | Rows | Description |
|---|---|---|
| `clusterData.json` | 2,000 | UMAP 3-D coords + cluster + metadata (sampled) |
| `themeRiverData.json` | 51 | Year (1960–2010) × 8-cluster song counts |
| `audioFeatureData.json` | 8 clusters | Per-cluster normalised acoustic feature means |
| `wordCloudData.json` | 9 keys (0–7 + all) | Top tags per cluster + global overview |
