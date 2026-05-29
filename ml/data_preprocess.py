"""
data_preprocess.py
=============
Transform the raw merged dataset (msd_mxm_merged.csv) into model-ready
feature matrices and save them as .npy / .npz files.
 
Processing steps:
    1. Load msd_mxm_merged.csv (raw BoW counts + acoustic features)
    2. Acoustic features: select columns, impute, StandardScaler
    3. Lyric features:
       a. Remove stop words (zero out their columns)
       b. Select top MAX_VOCAB words by corpus frequency (on lyrics-available rows)
       c. Apply TF-IDF weighting on the 1000-word matrix
       d. Compress with TruncatedSVD → 50-dim dense vectors
       e. StandardScaler on SVD output (fit on lyrics-available rows only)
    4. Artist tags: build top-100 multi-hot binary matrix
    5. Save all arrays + metadata to ../data/processed/
 
Input:
    ../data/msd_mxm_merged.csv
 
Outputs (under ../data/processed/):
    acoustic.npy        -- (N, n_acoustic)  float32, standardised
    lyric.npy           -- (N, 50)          float32, SVD-compressed & standardised
    tags.npy            -- (N, 100)         float32, multi-hot artist tags
    has_lyrics.npy      -- (N,)             float32  binary mask
    has_tags.npy        -- (N,)             float32  binary mask
    track_ids.npy       -- (N,)             str      track IDs (order preserved)
    meta.csv            -- original df columns needed for visualisation
                           (track_id, title, artist_name, year, cluster=NaN)
 
Usage (from the `ml/` directory):
    python preprocess.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from logger import setup_logger

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATA_DIR        = "../data"
INPUT_PATH      = os.path.join(DATA_DIR, "msd_mxm_merged.csv")
OUTPUT_DIR      = os.path.join(DATA_DIR, "processed")
RANDOM_SEED     = 42
MAX_VOCAB       = 1000   # top content words kept after stop word removal
SVD_COMPONENTS  = 50     # lyric compression target dimension
TOP_TAGS        = 100    # number of most-frequent artist tags to encode

logger = setup_logger("data_preprocess")

# Acoustic columns to use.
# Note: danceability and energy are all-zero in the MSD subset and are excluded.
ACOUSTIC_COLS = (
    ["loudness", "tempo", "mode", "key", "duration",
     "key_confidence", "mode_confidence", "time_signature",
     "time_signature_confidence", "bars_count", "beats_count",
     "sections_count", "segments_count",
     "avg_segment_loudness_max", "avg_segment_loudness_start"]
    + [f"avg_timbre_{i}" for i in range(12)]
    + [f"avg_pitch_{i}"  for i in range(12)]
)
 
# Common lyric stop words to remove before TF-IDF selection.
# These are high-frequency function words and filler syllables that carry
# little semantic meaning for tag/style discrimination.
LYRIC_STOP_WORDS = {
    "i", "the", "you", "to", "and", "a", "me", "it", "not", "my",
    "we", "of", "is", "in", "do", "your", "am", "on", "are", "she",
    "he", "will", "be", "that", "this", "for", "have", "what", "all",
    "no", "so", "but", "if", "or", "at", "by", "an", "as", "was",
    "with", "from", "her", "his", "they", "them", "their", "our",
    "its", "been", "has", "had", "would", "could", "should", "may",
    "can", "did", "does", "let", "got", "get", "just", "now", "up",
    "out", "about", "when", "there", "here", "than", "then", "more",
    "some", "who", "how", "into", "like", "one", "oh", "yeah", "la",
    "na", "da", "hey", "ah", "uh", "ooh", "gonna", "wanna", "gotta",
    "cause", "em", "ya", "y", "de", "que", "babi", "come", "go",
    "know", "said", "say", "see", "make", "take", "give", "came",
    "way", "time", "day", "back",
}
 
 
# ──────────────────────────────────────────────
# Step 1: process acoustic features
# ──────────────────────────────────────────────
 
def process_acoustic(df: pd.DataFrame):
    """
    Select acoustic columns, drop rows with NaN, and apply StandardScaler.
 
    Returns
    -------
    acoustic_scaled : np.ndarray (N, n_cols)  float32
    scaler          : fitted StandardScaler (saved for inference)
    valid_mask      : pd.Series bool mask of kept rows
    """
    # Drop rows where any acoustic column is missing
    valid_mask = df[ACOUSTIC_COLS].notna().all(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"  Dropping {n_dropped} rows with missing acoustic values")
 
    acoustic_raw = df.loc[valid_mask, ACOUSTIC_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    acoustic_scaled = scaler.fit_transform(acoustic_raw).astype(np.float32)
 
    logger.info(f"  Acoustic features : {acoustic_scaled.shape}")
    return acoustic_scaled, scaler, valid_mask
 
 
# ──────────────────────────────────────────────
# Step 2: process lyric features
# ──────────────────────────────────────────────
 
def process_lyrics(df: pd.DataFrame, max_vocab: int = MAX_VOCAB,
                   svd_components: int = SVD_COMPONENTS):
    """
    Transform raw BoW counts into compressed, standardised lyric vectors.
 
    Pipeline:
        raw counts (5000-dim)
        → stop word zeroing
        → Top max_vocab word selection by corpus frequency
          (frequency counted on lyrics-available rows only, matching
           the original data_merge_1.py behaviour)
        → TF-IDF weighting on the reduced 1000-dim matrix
        → TruncatedSVD (50 dim)
        → StandardScaler (fit on content rows only)
 
    Parameters
    ----------
    df             : DataFrame with 'bow_<word>' columns and 'lyrics_available'
    max_vocab      : number of content words to keep before SVD
    svd_components : output dimensionality of TruncatedSVD
 
    Returns
    -------
    lyric_dense    : np.ndarray (N, svd_components)  float32
    has_lyrics     : np.ndarray (N,)                 float32 binary mask
    selected_words : list[str]  the max_vocab words chosen
    svd            : fitted TruncatedSVD
    lyric_scaler   : fitted StandardScaler
    """
    bow_cols = [c for c in df.columns if c.startswith("bow_")]
    vocab = [c[len("bow_"):] for c in bow_cols]

    raw_matrix = df[bow_cols].values.astype(np.float32)  # (N, V)
    has_lyrics = (df["lyrics_available"].values == 1).astype(np.float32)

    logger.info(f"  Raw BoW matrix    : {raw_matrix.shape}")
    logger.info(f"  Songs with lyrics : {int(has_lyrics.sum())} / {len(has_lyrics)}")

    # 1. Zero out stop word columns
    stop_indices = [i for i, w in enumerate(vocab) if w in LYRIC_STOP_WORDS]
    logger.info(f"  Stop words removed: {len(stop_indices)}")
    raw_matrix[:, stop_indices] = 0.0

    # 2. Select Top max_vocab words by corpus frequency.
    #    Frequency is counted on lyrics-available rows only, consistent with
    #    data_merge_1.py which built the vocabulary from matched MSD songs.
    has_content_mask = raw_matrix.sum(axis=1) > 0
    word_totals = np.asarray(raw_matrix[has_content_mask].sum(axis=0)).flatten()
    top_indices = np.argsort(word_totals)[::-1][:max_vocab]
    top_indices = np.sort(top_indices)
    selected_words = [vocab[i] for i in top_indices]
    raw_matrix_selected = raw_matrix[:, top_indices]  # (N, max_vocab)

    logger.info(f"  Selected words    : {len(selected_words)} "
                f"(from {len(vocab)} total, stop words excluded)")
    logger.info(f"  Top 10 words      : {selected_words[:10]}")

    # 3. TF-IDF weighting on the 1000-dim matrix.
    #    fit_transform on all rows (including zero rows) matches the
    #    original data_merge_1.py behaviour.
    tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(raw_matrix_selected)
    tfidf_selected = tfidf_matrix.toarray().astype(np.float32)  # (N, max_vocab)

    # 4. TruncatedSVD: fit only on rows that have actual TF-IDF content.
    #    Zero rows (no lyrics or all words filtered out) are kept as zeros.
    svd_fit_mask = tfidf_selected.sum(axis=1) > 0

    svd = TruncatedSVD(n_components=svd_components, random_state=RANDOM_SEED)

    lyric_dense = np.zeros((len(df), svd_components), dtype=np.float32)
    lyric_dense[svd_fit_mask] = svd.fit_transform(
        tfidf_selected[svd_fit_mask]
    ).astype(np.float32)

    # 5. StandardScaler fitted on content-available rows only.
    lyric_scaler = StandardScaler()
    lyric_scaler.fit(lyric_dense[svd_fit_mask])
    lyric_dense[svd_fit_mask] = lyric_scaler.transform(
        lyric_dense[svd_fit_mask]
    ).astype(np.float32)

    logger.info(f"  SVD fit rows      : {int(svd_fit_mask.sum())} "
                f"(excluded {int(has_lyrics.sum()) - int(svd_fit_mask.sum())} "
                f"zero-content rows from {int(has_lyrics.sum())} lyrics_available)")
    logger.info(f"  Lyric dense shape : {lyric_dense.shape}")

    return lyric_dense, has_lyrics, selected_words, svd, lyric_scaler
 
 
# ──────────────────────────────────────────────
# Step 3: process artist tag into multi-hot encoding
# ──────────────────────────────────────────────
 
def process_tags(df: pd.DataFrame, top_n: int = TOP_TAGS):
    """
    Build a multi-hot binary matrix from semi-colon-separated artist_terms.
 
    Only the top_n most frequent tags across the whole corpus are kept.
 
    Returns
    -------
    tags_matrix : np.ndarray (N, top_n)   float32
    has_tags    : np.ndarray (N,)         float32 binary mask
    top_terms   : list[str]               vocabulary of kept tags
    """
    def parse_terms(row):
        val = row.get("artist_terms", "")
        if isinstance(val, str) and val.strip():
            return [t.strip() for t in val.split(";") if t.strip()]
        return []
 
    terms_list = df.apply(parse_terms, axis=1).tolist()
 
    # Count all tags and keep the most frequent top_n
    all_terms = [t for terms in terms_list for t in terms]
    term_counts = Counter(all_terms)
    top_terms = [t for t, _ in term_counts.most_common(top_n)]
    term2idx = {t: i for i, t in enumerate(top_terms)}
 
    logger.info(f"  Artist tag vocab  : {len(top_terms)} (from {len(term_counts)} unique tags)")
    logger.info(f"  Top 10 tags       : {top_terms[:10]}")
 
    tags_matrix = np.zeros((len(df), top_n), dtype=np.float32)
    for i, terms in enumerate(terms_list):
        for t in terms:
            if t in term2idx:
                tags_matrix[i, term2idx[t]] = 1.0
 
    has_tags = (tags_matrix.sum(axis=1) > 0).astype(np.float32)
    logger.info(f"  Songs with tags   : {int(has_tags.sum())} / {len(df)}")
 
    return tags_matrix, has_tags, top_terms
 
 
# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # 1. Load merged dataset
    logger.info("=" * 55)
    logger.info("Loading msd_mxm_merged.csv ...")
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"  Raw shape: {df.shape}")
 
    # 2. Acoustic
    logger.info("\n" + "=" * 55)
    logger.info("Processing acoustic features ...")
    acoustic, acoustic_scaler, valid_mask = process_acoustic(df)
 
    # Apply the same valid_mask to all modalities so row order is consistent
    df = df.loc[valid_mask].reset_index(drop=True)
 
    # 3. Lyrics
    logger.info("\n" + "=" * 55)
    logger.info("Processing lyric features ...")
    lyric, has_lyrics, selected_words, svd, lyric_scaler = process_lyrics(df)
 
    # 4. Artist tags
    logger.info("\n" + "=" * 55)
    logger.info("Processing artist tags ...")
    tags, has_tags, top_terms = process_tags(df)
 
    # 5. Save feature arrays
    logger.info("\n" + "=" * 55)
    logger.info(f"Saving processed arrays to {OUTPUT_DIR} ...")
 
    np.save(os.path.join(OUTPUT_DIR, "acoustic.npy"),   acoustic)
    np.save(os.path.join(OUTPUT_DIR, "lyric.npy"),      lyric)
    np.save(os.path.join(OUTPUT_DIR, "tags.npy"),       tags)
    np.save(os.path.join(OUTPUT_DIR, "has_lyrics.npy"), has_lyrics)
    np.save(os.path.join(OUTPUT_DIR, "has_tags.npy"),   has_tags)
    np.save(os.path.join(OUTPUT_DIR, "track_ids.npy"),
            np.array(df["track_id"].tolist(), dtype=object))
 
    # Save metadata columns used later for visualisation
    meta_cols = [c for c in ["track_id", "title", "artist_name", "year",
                              "artist_terms", "lyrics_available"]
                 if c in df.columns]
    df[meta_cols].to_csv(os.path.join(OUTPUT_DIR, "meta.csv"), index=False)
 
    # Save fitted transformers for inference / reproducibility
    with open(os.path.join(OUTPUT_DIR, "transformers.pkl"), "wb") as f:
        pickle.dump({
            "acoustic_scaler": acoustic_scaler,
            "svd":             svd,
            "lyric_scaler":    lyric_scaler,
            "selected_words":  selected_words,
            "top_terms":       top_terms,
            "acoustic_cols":   ACOUSTIC_COLS,
        }, f)
 
    logger.info("\nSummary:")
    logger.info(f"  acoustic.npy   : {acoustic.shape}")
    logger.info(f"  lyric.npy      : {lyric.shape}")
    logger.info(f"  tags.npy       : {tags.shape}")
    logger.info(f"  has_lyrics.npy : {has_lyrics.shape}  (positive: {int(has_lyrics.sum())})")
    logger.info(f"  has_tags.npy   : {has_tags.shape}    (positive: {int(has_tags.sum())})")
    logger.info(f"  meta.csv       : {df[meta_cols].shape}")
 
 
if __name__ == "__main__":
    main()