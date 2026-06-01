# Multi-modal Music Trend Analysis - Machine Learning Pipeline

## Overview

This pipeline performs multi-modal music trend analysis on the Million Song
Dataset (MSD) subset (10,000 tracks) fused with musiXmatch lyric database.

It proceeds through five stages:

1. **data_merge.py**: joins the MSD acoustic feature CSV with raw
   musiXmatch Bag-of-Words lyric counts via shared track IDs.
2. **data_preprocess.py**: standardises acoustic features; 
   applies stop word removal, TF-IDF weighting, and TruncatedSVD 
   compression on lyric features; encodes artist tags as a multi-hot binary matrix.
   Rows with missing acoustic values are dropped first and the same
   row mask is applied to all modalities to keep all output arrays
   aligned by index.
3. **train_vae.py**: trains a tri-modal Variational Autoencoder (VAE)
   that fuses acoustic, lyric, and tag features into a shared 32-dimensional
   latent space, then extracts deterministic latent vectors for all tracks.
4. **cluster.py**: sweeps K ∈ {4, ..., 10} using Silhouette score to find
   the optimal number of clusters, then runs KMeans on the latent vectors.
5. **visualize.py**: projects latent vectors into 3D via UMAP and renders
   an interactive scatter plot and a ThemeRiver-style decade trend chart.

## Dependencies

To install all required packages:

```bash
pip install -r requirements.txt
```

in which, `kaleido` is optional and only needed for static PNG export in `visualize.py`.

## Data setup
The merged file `data/msd_mxm_merged.csv` (~107 MB) exceeds GitHub's file size
limit and is not included in this repository. Please follow the steps below to
generate it locally.

1. Download the raw data `msd_subset.csv` and musiXmatch files `mxm_dataset_train.txt` and `mxm_dataset_test.txt` to the `data/` directory.

    You can also download the raw data by yourself:
    - MSD subset (`msd_subset.csv`)): Download from [the Million Song Dataset - Getting the dataset](http://millionsongdataset.com/pages/getting-dataset/)
    - musicXmatch lyrics (`mxm_dataset_train.txt` and `mxm_dataset_test.txt`): Download from [the Million Song Dataset - The musiXmatch Dataset](http://labrosa.ee.columbia.edu/millionsong/musixmatch)
2. Generate the merged dataset
```bash
cd ml
python data_merge.py
```
3. Run data preprocessing
```bash
python data_preprocess.py
```

## Running the machine learning pipeline

Run all scripts from the `ml/` directory.

```bash
# Step 1: re-run when you change VAE architecture, EPOCHS, LR, or LATENT_DIM
python train_vae.py

# Step 2: re-run when you change K or want to try a different cluster count
python cluster.py            # auto-selects best K via Silhouette
python cluster.py --k 8      # fix K=8, skip the search

# Step 3: re-run for new plots; UMAP is cached so it only recomputes once
python visualize.py
```

## Directory layout

```
ml/
├── data_merge.py        # Merge MSD subset with raw musiXmatch BoW counts
├── data_preprocess.py   # Stop word removal, TF-IDF, SVD, artist tag encoding
├── train_vae.py         # Tri-modal VAE training + latent vector extraction
├── cluster.py           # Optimal K search + KMeans on latent vectors
├── visualize.py         # UMAP 3D scatter + ThemeRiver decade trend chart
└── results/
    ├── vae_weights.pt
    ├── latent_vectors.npy
    ├── cluster_labels.npy
    ├── msd_clustered.csv
    ├── umap_coords_3d.npy
    └── figures/
        ├── training_history.png
        ├── cluster_selection.png
        ├── viz_umap_3d.html
        ├── viz_umap_3d_static.png
        └── viz_decade_trend.png
../
data/
├── msd_subset.csv              (input)
├── mxm_dataset_train.txt       (input)
├── mxm_dataset_test.txt        (input)
├── msd_mxm_merged.csv          (output of data_merge.py)
└── processed/                  (output of preprocess.py)
    ├── acoustic.npy
    ├── lyric.npy
    ├── tags.npy
    ├── has_lyrics.npy
    ├── has_tags.npy
    ├── track_ids.npy
    ├── meta.csv
    └── transformers.pkl
```