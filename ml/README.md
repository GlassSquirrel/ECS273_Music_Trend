# ML Pipeline

Turns raw MSD + musiXmatch data into cluster labels and UMAP coordinates. All outputs are already committed — you only need to re-run if you want to experiment with the model or clustering parameters.

For a full description of the algorithms, see the [root README](../README.md#ml-pipeline-reference).

## Re-running

All scripts run from the `ml/` directory.

```bash
cd ml

# 1. Rebuild the merged dataset (only needed if raw data changed)
python data_merge.py
python data_preprocess.py

# 2. Retrain the VAE
python train_vae.py

# 3. Re-cluster (auto K-search, or fix K)
python cluster.py
python cluster.py --k 8

# 4. Regenerate UMAP plots
python visualize.py
```

UMAP is cached in `results/umap_coords_3d.npy` — delete it to force a recompute.

## Results

```
results/
├── vae_weights.pt          trained VAE
├── latent_vectors.npy      (10000, 32) latent means
├── cluster_labels.npy      (10000,) cluster assignments
├── msd_clustered.csv       metadata + cluster column
├── umap_coords_3d.npy      (10000, 3) UMAP embedding
└── figures/
    ├── training_history.png
    ├── cluster_selection.png   Elbow + Silhouette sweep
    ├── viz_umap_3d.html        interactive Plotly scatter
    ├── viz_umap_3d_static.png
    └── viz_decade_trend.png
```

## Data sources

- **MSD subset** (`data/msd_subset.csv`): 10K tracks with acoustic features. [millionsongdataset.com](http://millionsongdataset.com/pages/getting-dataset/)
- **musiXmatch** (`data/mxm_dataset_*.txt`): bag-of-words lyric counts. [labrosa.ee.columbia.edu/millionsong/musixmatch](http://labrosa.ee.columbia.edu/millionsong/musixmatch)

`data/msd_mxm_merged.csv` (~107 MB) is not committed. Run `data_merge.py` to generate it locally.
