# Backend

Express API server that reads from MongoDB and serves the frontend dashboard.

## Setup & run

See the [root README](../README.md) for the full quick-start guide (data import + running both servers).

```bash
# from backend/
npm install
npm run dev   # http://localhost:8000
```

## Environment

Copy `.env.example` to `.env` and adjust if your MongoDB is not on the default local port:

```
PORT=8000
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DB=music_trend
```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/bootstrap` | Returns all dashboard data in one payload |

`/api/bootstrap` response shape:

```json
{
  "clusterData":      [...],
  "themeRiverData":   [...],
  "audioFeatureData": { "features": {}, "display": {} },
  "wordCloudData":    {},
  "stats":            { "totalSongs": 10000, "clusteredSongs": 10000, "clusters": 8 }
}
```

## Collections

| Collection | Contents |
|---|---|
| `music_trend.songs` | 10,000 track records (metadata + cluster label + UMAP coords) |
| `music_trend.visualization_cache` | Pre-aggregated data for each dashboard panel |

Populated by running `python backend/scripts/import_data.py` from the repo root.
