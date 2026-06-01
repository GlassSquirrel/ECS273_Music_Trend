"""
Import project visualization data into MongoDB using two collections:

1. songs
2. visualization_cache

Run from repo root:
    python backend/scripts/import_data.py

Environment variables:
    MONGODB_URI   default: mongodb://127.0.0.1:27017
    MONGODB_DB    default: music_trend
"""

from __future__ import annotations

import math
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient, ReplaceOne


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "ml" / "results"

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGODB_DB", "music_trend")

META_PATH = RESULTS_DIR / "msd_clustered.csv"
UMAP_PATH = RESULTS_DIR / "umap_coords_3d.npy"
ACOUSTIC_PATH = DATA_DIR / "acoustic.npy"
TRANSFORMERS_PATH = DATA_DIR / "transformers.pkl"

SCATTER_SAMPLE_SIZE = 2000
FEATURE_KEYS = [
    "loudness",
    "tempo",
    "mode",
    "key_confidence",
    "avg_timbre_1",
    "time_signature_confidence",
]
FEATURE_LABELS = {
    "loudness": "Loudness",
    "tempo": "Tempo",
    "mode": "Major / Minor",
    "key_confidence": "Key Clarity",
    "avg_timbre_1": "Brightness",
    "time_signature_confidence": "Rhythm Clarity",
}
WORDCLOUD_STOP = {"rock", "pop", "alternative"}


def load_inputs():
    meta_df = pd.read_csv(META_PATH)
    umap_coords = np.load(UMAP_PATH)
    acoustic_scaled = np.load(ACOUSTIC_PATH)

    with open(TRANSFORMERS_PATH, "rb") as file:
        transformers = pickle.load(file)

    if len(meta_df) != len(umap_coords) or len(meta_df) != len(acoustic_scaled):
        raise ValueError("Input files are not row-aligned.")

    acoustic_cols = transformers["acoustic_cols"]
    acoustic_scaler = transformers["acoustic_scaler"]
    acoustic_raw = acoustic_scaler.inverse_transform(acoustic_scaled)
    acoustic_df = pd.DataFrame(acoustic_raw, columns=acoustic_cols)

    songs_df = pd.concat(
        [
            meta_df.reset_index(drop=True),
            acoustic_df.reset_index(drop=True),
        ],
        axis=1,
    )
    songs_df["umap_x"] = umap_coords[:, 0]
    songs_df["umap_y"] = umap_coords[:, 1]
    songs_df["umap_z"] = umap_coords[:, 2]
    return songs_df


def pick_scatter_rows(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()
    indices = np.linspace(0, len(df) - 1, num=sample_size, dtype=int)
    return df.iloc[indices].copy()


def build_cluster_data(df: pd.DataFrame):
    sampled = pick_scatter_rows(df, SCATTER_SAMPLE_SIZE)
    return [
        {
            "x": round(float(row.umap_x), 4),
            "y": round(float(row.umap_y), 4),
            "z": round(float(row.umap_z), 4),
            "cluster": int(row.cluster),
            "year": int(row.year) if not pd.isna(row.year) else 0,
            "title": row.title if isinstance(row.title, str) else "",
            "artist": row.artist_name if isinstance(row.artist_name, str) else "",
        }
        for row in sampled.itertuples(index=False)
    ]


def build_theme_river_data(df: pd.DataFrame):
    valid = df[(df["year"] >= 1960) & (df["year"] <= 2010)].copy()
    counts = (
        valid.groupby(["year", "cluster"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1960, 2011), fill_value=0)
    )

    cluster_ids = sorted(int(c) for c in df["cluster"].dropna().unique())
    for cluster_id in cluster_ids:
        if cluster_id not in counts.columns:
            counts[cluster_id] = 0
    counts = counts[cluster_ids]

    rows = []
    for year, values in counts.iterrows():
        row = {"year": int(year)}
        for cluster_id, count in values.items():
            row[f"cluster_{int(cluster_id)}"] = int(count)
        rows.append(row)
    return rows


def build_audio_feature_data(df: pd.DataFrame):
    cluster_means = (
        df.groupby("cluster")[FEATURE_KEYS]
        .mean()
        .sort_index()
    )

    features = {}
    for cluster_id, row in cluster_means.iterrows():
        features[str(int(cluster_id))] = {}

    for key in FEATURE_KEYS:
        values = cluster_means[key].astype(float)
        low = float(values.min())
        high = float(values.max())
        denom = high - low
        for cluster_id, value in values.items():
            norm = 0.0 if denom == 0 else (float(value) - low) / denom
            features[str(int(cluster_id))][key] = round(norm, 3)

    return {
        "features": features,
        "display": FEATURE_LABELS,
    }


def parse_terms(raw_value):
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []
    return [term.strip() for term in raw_value.split(";") if term.strip()]


def build_word_cloud_data(df: pd.DataFrame):
    cluster_ids = sorted(int(c) for c in df["cluster"].dropna().unique())
    terms_per_cluster = {}
    cluster_doc_freq = {}

    for cluster_id in cluster_ids:
        cluster_rows = df[df["cluster"] == cluster_id]
        doc_terms = [set(parse_terms(value)) for value in cluster_rows["artist_terms"]]
        terms_per_cluster[cluster_id] = doc_terms
        counter = Counter()
        for terms in doc_terms:
            counter.update(terms)
        cluster_doc_freq[cluster_id] = counter

    clusters_with_tag = Counter()
    for cluster_id in cluster_ids:
        for tag in cluster_doc_freq[cluster_id]:
            clusters_with_tag[tag] += 1

    result = {}
    n_clusters = len(cluster_ids)

    for cluster_id in cluster_ids:
        doc_terms = terms_per_cluster[cluster_id]
        cluster_size = max(len(doc_terms), 1)
        threshold = max(1, math.ceil(cluster_size * 0.02))
        scores = []

        for tag, doc_freq in cluster_doc_freq[cluster_id].items():
            if tag in WORDCLOUD_STOP or doc_freq < threshold:
                continue
            idf = math.log((n_clusters + 1) / (clusters_with_tag[tag] + 1)) + 0.5
            score = (doc_freq / cluster_size) * idf
            scores.append((tag, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        top_scores = scores[:15]
        max_score = top_scores[0][1] if top_scores else 1.0
        result[str(cluster_id)] = [
            {"word": tag, "w": round(score / max_score, 3)}
            for tag, score in top_scores
        ]

    all_doc_terms = [set(parse_terms(value)) for value in df["artist_terms"]]
    all_freq = Counter()
    for terms in all_doc_terms:
        all_freq.update(terms)

    min_global = max(1, math.ceil(len(all_doc_terms) * 0.01))
    global_scores = [
        (tag, freq)
        for tag, freq in all_freq.items()
        if freq >= min_global
    ]
    global_scores.sort(key=lambda item: item[1], reverse=True)
    global_scores = global_scores[:18]
    global_max = global_scores[0][1] if global_scores else 1.0
    result["all"] = [
        {"word": tag, "w": round(freq / global_max, 3)}
        for tag, freq in global_scores
    ]
    return result


def build_song_documents(df: pd.DataFrame):
    docs = []
    for row in df.itertuples(index=False):
        doc = {
            "_id": row.track_id,
            "track_id": row.track_id,
            "title": row.title if isinstance(row.title, str) else "",
            "artist_name": row.artist_name if isinstance(row.artist_name, str) else "",
            "year": int(row.year) if not pd.isna(row.year) else 0,
            "artist_terms": parse_terms(row.artist_terms),
            "lyrics_available": bool(int(row.lyrics_available)) if not pd.isna(row.lyrics_available) else False,
            "cluster": int(row.cluster) if not pd.isna(row.cluster) else None,
            "umap": {
                "x": round(float(row.umap_x), 6),
                "y": round(float(row.umap_y), 6),
                "z": round(float(row.umap_z), 6),
            },
        }
        for key in FEATURE_KEYS:
            doc[key] = round(float(getattr(row, key)), 6)
        docs.append(doc)
    return docs


def upsert_many(collection, docs, key="_id"):
    operations = [
        ReplaceOne({key: doc[key]}, doc, upsert=True)
        for doc in docs
    ]
    if operations:
        collection.bulk_write(operations, ordered=False)


def main():
    songs_df = load_inputs()

    cluster_data = build_cluster_data(songs_df)
    theme_river_data = build_theme_river_data(songs_df)
    audio_feature_data = build_audio_feature_data(songs_df)
    word_cloud_data = build_word_cloud_data(songs_df)
    song_docs = build_song_documents(songs_df)

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    songs = db["songs"]
    visualization_cache = db["visualization_cache"]

    songs.create_index("cluster")
    songs.create_index("year")

    upsert_many(songs, song_docs)

    cache_docs = [
        {"_id": "clusterData", "data": cluster_data},
        {"_id": "themeRiverData", "data": theme_river_data},
        {"_id": "audioFeatureData", "data": audio_feature_data},
        {"_id": "wordCloudData", "data": word_cloud_data},
    ]
    upsert_many(visualization_cache, cache_docs)

    print(f"Imported {len(song_docs)} songs into '{MONGO_DB}.songs'")
    print(f"Upserted {len(cache_docs)} cache docs into '{MONGO_DB}.visualization_cache'")


if __name__ == "__main__":
    main()
