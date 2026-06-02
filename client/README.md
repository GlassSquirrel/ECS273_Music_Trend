# Frontend

React dashboard for the music trend analysis project. For setup and the full data-flow explanation, see the [root README](../README.md).

## Run

```bash
cd client
npm install
npm run dev   # http://localhost:5173
```

Requires the backend to be running at `localhost:8000` (Vite proxies `/api/*` there).

## Stack

React 18 · Vite · D3.js v7 · Three.js

## Dashboard panels

### ThemeRiver
Stacked stream graph (D3 `stackOffsetWiggle`) showing genre share by year. Band width encodes song count; a dashed centre baseline anchors the wiggle. Hover to see exact count and percentage of that year; click a band to select its cluster.

**Data** — `year × cluster` song counts from `msd_clustered.csv`, 1960–2010 (4,634 songs with known year).

### UMAP 3D Cluster Scatter
Three.js point cloud of 2,000 sampled songs in UMAP 3D space, coloured by cluster. Three grid planes and labelled axes (UMAP-1/2/3) provide spatial reference. Drag to orbit, scroll to zoom, click a point to select its cluster, ↗ to expand fullscreen.

**Data** — `umap_coords_3d.npy` + `cluster_labels.npy`, joined with track metadata.

### Audio Features
Horizontal bar chart of 6 acoustic feature means per cluster, normalised to [0, 1] across clusters.

| Display label | Source field | Meaning |
|---|---|---|
| Loudness | `loudness` | Average track loudness (dB) |
| Tempo | `tempo` | Beats per minute |
| Major / Minor | `mode` | 1 = major, 0 = minor |
| Key Clarity | `key_confidence` | Confidence in detected key |
| Brightness | `avg_timbre_1` | Spectral brightness (MIR timbre dim 2) |
| Rhythm Clarity | `time_signature_confidence` | Confidence in detected time signature |

**Data** — `acoustic.npy` inverse-scaled via `transformers.pkl`, per-cluster mean, min-max normalised.

### Artist Tags
Word cloud of the most distinctive genre/style tags per cluster, sized by TF-IDF weight. In all-clusters mode, shows the globally most frequent tags.

**Data** — `artist_terms` column from `msd_clustered.csv`.

## Cluster colours

| Cluster | Colour | Dominant tags |
|---|---|---|
| 0 | `#7F77DD` | metal · punk · hardcore |
| 1 | `#1D9E75` | jazz · folk · blues · country |
| 2 | `#D85A30` | electronic · hip hop · techno |
| 3 | `#D4537E` | jazz · easy listening · classical |
| 4 | `#378ADD` | electronic · folk · singer-songwriter |
| 5 | `#639922` | hip hop · rap · reggae · funk |
| 6 | `#BA7517` | electronic · guitar · classic rock |
| 7 | `#9966CC` | jazz · soul · folk · acoustic |
| *(all)* | `#58A4B0` | — |
