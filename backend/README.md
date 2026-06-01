# Backend README

This backend serves the frontend through MongoDB-backed APIs.

It uses two MongoDB collections:

- `songs`
- `visualization_cache`

## 1. Import Data Into MongoDB

### Prerequisites

Make sure the following are ready before importing:

- MongoDB is running locally or you have a reachable MongoDB URI
- The ML pipeline outputs already exist in this repo:
  - `ml/results/msd_clustered.csv`
  - `ml/results/umap_coords_3d.npy`
  - `data/processed/acoustic.npy`
  - `data/processed/transformers.pkl`


### Install Import Dependencies

From the repo root:

```bash
pip install -r backend/requirements.txt
```

### Run the Import Script

From the repo root:

```bash
python backend/scripts/import_data.py
```

What this does:

- imports song-level records into `music_trend.songs`
- imports frontend-ready visualization data into `music_trend.visualization_cache`

Expected success output:

```bash
Imported 10000 songs into 'music_trend.songs'
Upserted 4 cache docs into 'music_trend.visualization_cache'
```

After this, you should be able to see both collections in MongoDB Compass.

## 2. Start The  Project

The final project has two running parts:

- backend API server
- frontend Vite app

You need two terminals.

### Step A: Start the Backend

Go into `backend/` and install Node dependencies once:

```bash
cd backend
npm install
```

Then start the backend:

```bash
npm run dev
```

Expected output:

```bash
API listening on http://localhost:8000
```

You can verify the backend with:

```bash
http://localhost:8000/api/health
http://localhost:8000/api/bootstrap
```

### Step B: Start the Frontend

Open a second terminal and run:

```bash
cd client
npm install
npm run dev
```

Expected output includes:

```bash
Local: http://localhost:5173/
```

Open:

```bash
http://localhost:5173/
```

### How The Final Project Works

The frontend does not read local JSON files directly anymore.

The runtime flow is:

1. browser opens the frontend at `http://localhost:5173`
2. frontend requests `/api/bootstrap`
3. Vite proxies `/api/*` to `http://localhost:8000`
4. backend reads MongoDB
5. backend returns:
   - `clusterData`
   - `themeRiverData`
   - `audioFeatureData`
   - `wordCloudData`
   - `stats`
6. frontend renders the dashboard

## Quick Start Summary

From the repo root:

```bash
pip install -r backend/requirements.txt
python backend/scripts/import_data.py
```

Terminal 1:

```bash
cd backend
npm install
npm run dev
```

Terminal 2:

```bash
cd client
npm install
npm run dev
```

Then open:

```bash
http://localhost:5173/
```
