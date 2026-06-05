# ML Pipeline

Turns raw MSD + musiXmatch data into cluster labels and UMAP coordinates. All outputs are already committed — you only need to re-run if you want to experiment with the model or clustering parameters.

For a full description of the algorithms, see the [root README](../README.md#ml-pipeline-reference).

## Prerequisite
To install all required packages:

```bash
pip install -r requirements.txt
```

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
python cluster.py --latent results/latent_vectors.npy --output results/
python cluster.py --latent results/latent_vectors.npy --output results/ --k 8

# 4. Regenerate UMAP plots
python visualize.py
```

UMAP is cached in `results/umap_coords_3d.npy` — delete it to force a recompute.

## Evaluation

There are several scripts in the folder to reproduce the comparisions used in the evaluation section of the report.

### RQ1: Clustering Validity and Semantic Coherence

To reproduce the result of the PCA model:

```bash
python train_pca.py
python cluster.py --latent results_pca/latent_vectors.npy --output results_pca/
```

To reproduce the result of the bi-modality VAE model:

```bash
python train_vae_2.py
python cluster.py --latent results_2modal/latent_vectors.npy --output results_2modal/
```

### RQ2: 2D vs 3D UMAP Projection

```bash
cd ml/eval

# Generate 2D UMAP embedding and static PNG
python visualize_2d.py

# Compute trustworthiness scores for 2D vs 3D UMAP
python trustworthiness.py
```

Results are saved to `ml/eval/results/figures/`.

## Results

```
results/                        tri-modal VAE (main model)
├── vae_weights.pt
├── latent_vectors.npy          (10000, 32) latent means
├── cluster_labels.npy          (10000,) cluster assignments
├── msd_clustered.csv           metadata + cluster column
├── umap_coords_3d.npy          (10000, 3) UMAP embedding
└── figures/
    ├── training_history.png
    ├── cluster_selection.png   Elbow + Silhouette sweep
    ├── viz_umap_3d.html        interactive Plotly scatter
    ├── viz_umap_3d_static.png
    └── viz_decade_trend.png

results_pca/                    PCA baseline
├── latent_vectors.npy
├── cluster_labels.npy
├── msd_clustered.csv
└── figures/
    ├── explained_variance_pca.png
    └── cluster_selection.png

results_2modal/                 bi-modal VAE baseline
├── vae_weights.pt
├── latent_vectors.npy
├── cluster_labels.npy
├── msd_clustered.csv
└── figures/
    ├── training_history.png
    └── cluster_selection.png

eval/results/                   RQ2 evaluation outputs
├── umap_coords_2d.npy
└── figures/
    ├── viz_umap_2d.html
    ├── viz_umap_2d_mpl.png
    └── trustworthiness_comparison.png
```

## Data sources

- **MSD subset** (`data/msd_subset.csv`): 10K tracks with acoustic features. [millionsongdataset.com](http://millionsongdataset.com/pages/getting-dataset/)
- **musiXmatch** (`data/mxm_dataset_*.txt`): bag-of-words lyric counts. [labrosa.ee.columbia.edu/millionsong/musixmatch](http://labrosa.ee.columbia.edu/millionsong/musixmatch)

`data/msd_mxm_merged.csv` (~107 MB) is not committed. Run `data_merge.py` to generate it locally.
