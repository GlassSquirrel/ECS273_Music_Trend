"""
data_merge.py
=============
Merge the MSD subset CSV with musiXmatch Bag-of-Words lyric database.

Input files (under `../data/`):
    msd_subset.csv          -- 10,000 MSD songs with acoustic features
    mxm_dataset_train.txt   -- musiXmatch training split (BoW)
    mxm_dataset_test.txt    -- musiXmatch test split (BoW)

Output files (under `../data/`):
    msd_mxm_merged.csv      -- MSD acoustic features + raw BoW lyric counts
                               (one bow_<word> column per vocabulary word)

Usage (from the `ml/` directory):
    python data_merge.py
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from logger import setup_logger

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATA_DIR       = "../data"
MSD_PATH       = os.path.join(DATA_DIR, "msd_subset.csv")
MXM_TRAIN_PATH = os.path.join(DATA_DIR, "mxm_dataset_train.txt")
MXM_TEST_PATH  = os.path.join(DATA_DIR, "mxm_dataset_test.txt")
OUTPUT_PATH    = os.path.join(DATA_DIR, "msd_mxm_merged.csv")

logger = setup_logger("data_merge")


# ──────────────────────────────────────────────
# Step 1: Parse musiXmatch .txt files
# ──────────────────────────────────────────────

def parse_mxm_file(filepath: str):
    """
    Parse a musiXmatch dataset file (train or test split).

    File format:
        Lines starting with '#'  -> comments, skipped
        Line starting with '%'   -> comma-separated vocabulary (5000 words), one line only
        All other lines          -> song records:
                                    track_id,mxm_tid,word_idx:count,...

    Returns
    -------
    vocab   : list[str]   -- the 5000-word vocabulary in index order
    records : list[dict]  -- each entry has keys:
                             'track_id', 'mxm_tid', 'word_counts'
                             where word_counts = {1-based word index: raw count}
    """
    vocab   = None
    records = []

    logger.info(f"Parsing {os.path.basename(filepath)} ...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue

            # Vocabulary line
            if line.startswith("%"):
                vocab = line[1:].split(",")
                logger.info(f"  Vocabulary size: {len(vocab)}")
                continue

            # Song record line
            parts = line.split(",")
            if len(parts) < 2:
                continue

            track_id = parts[0]
            mxm_tid  = parts[1]

            # Parse sparse word counts: "word_idx:count" (1-based index)
            word_counts = {}
            for token in parts[2:]:
                token = token.strip()
                if ":" in token:
                    try:
                        idx, cnt = token.split(":")
                        word_counts[int(idx)] = int(cnt)
                    except ValueError:
                        continue

            records.append({
                "track_id":    track_id,
                "mxm_tid":     mxm_tid,
                "word_counts": word_counts,
            })

    logger.info(f"  Songs parsed: {len(records)}")
    return vocab, records


# ──────────────────────────────────────────────
# Step 2: Build raw BoW DataFrame for MSD songs
# ──────────────────────────────────────────────

def build_raw_bow_df(vocab: list, records: list, msd_track_ids: set):
    """
    Convert parsed musiXmatch records into a raw count DataFrame,
    keeping only the songs present in the MSD subset.

    Parameters
    ----------
    vocab          : vocabulary list from parse_mxm_file
    records        : record list from parse_mxm_file
    msd_track_ids  : set of track IDs present in msd_subset.csv

    Returns
    -------
    bow_df : pd.DataFrame  -- index = track_id, columns = 'bow_<word>'
                              values are raw integer word counts
    """
    # Filter to MSD songs only
    filtered = [r for r in records if r["track_id"] in msd_track_ids]
    logger.info(f"Songs matched with MSD subset: {len(filtered)} / {len(records)}")

    if len(filtered) == 0:
        raise ValueError(
            "No matching track IDs found. "
            "Check that msd_subset.csv and musiXmatch files use the same ID format."
        )

    n_songs = len(filtered)
    n_vocab = len(vocab)
    track_ids = [r["track_id"] for r in filtered]

    # Build sparse count matrix (songs x vocabulary)
    logger.info(f"Building raw BoW count matrix ({n_songs} x {n_vocab}) ...")
    count_matrix = lil_matrix((n_songs, n_vocab), dtype=np.int32)

    for i, record in enumerate(filtered):
        for word_idx, count in record["word_counts"].items():
            col = word_idx - 1  # convert 1-based to 0-based
            if 0 <= col < n_vocab:
                count_matrix[i, col] = count

    # Convert to dense DataFrame with 'bow_<word>' column names
    col_names = [f"bow_{w}" for w in vocab]
    bow_df = pd.DataFrame(
        count_matrix.toarray(),
        index=track_ids,
        columns=col_names,
        dtype=np.int32,
    )
    bow_df.index.name = "track_id"

    logger.info(f"Raw BoW DataFrame shape: {bow_df.shape}")
    return bow_df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # Step 1: Load MSD subset
    logger.info("=" * 55)
    logger.info("Step 1: Loading MSD subset")
    msd_df = pd.read_csv(MSD_PATH)
    logger.info(f"  MSD subset shape: {msd_df.shape}")
    msd_track_ids = set(msd_df["track_id"].tolist())
    logger.info(f"  Unique track IDs: {len(msd_track_ids)}")

    # Step 2: Parse musiXmatch train + test files
    logger.info("\n" + "=" * 55)
    logger.info("Step 2: Parsing musiXmatch files")
    vocab_train, records_train = parse_mxm_file(MXM_TRAIN_PATH)
    vocab_test,  records_test  = parse_mxm_file(MXM_TEST_PATH)

    assert vocab_train == vocab_test, (
        "Train and test vocabulary mismatch -- files may be from different releases."
    )
    all_records = records_train + records_test
    logger.info(f"  Total musiXmatch songs (train + test): {len(all_records)}")

    # Step 3: Build raw BoW DataFrame
    logger.info("\n" + "=" * 55)
    logger.info("Step 3: Building raw BoW DataFrame")
    bow_df = build_raw_bow_df(vocab_train, all_records, msd_track_ids)

    # Step 4: Left-join MSD with BoW (songs without lyrics get 0s)
    logger.info("\n" + "=" * 55)
    logger.info("Step 4: Merging MSD subset with raw BoW lyrics")
    merged_df = msd_df.merge(
        bow_df.reset_index(),
        on="track_id",
        how="left",
    )

    # Fill missing lyrics with 0 (songs not in musiXmatch)
    bow_cols = [f"bow_{w}" for w in vocab_train]
    merged_df[bow_cols] = merged_df[bow_cols].fillna(0).astype(np.int32)

    # Binary flag: 1 if the song has at least one lyric word count
    has_lyrics = merged_df["track_id"].isin(set(bow_df.index))
    merged_df["lyrics_available"] = has_lyrics.astype(np.int8)

    logger.info(f"  Merged shape      : {merged_df.shape}")
    logger.info(f"  Songs with lyrics : {merged_df['lyrics_available'].sum()} / {len(merged_df)}")
    logger.info(f"  BoW columns       : {len(bow_cols)} (raw counts, no TF-IDF)")

    # Step 5: Save
    logger.info("\n" + "=" * 55)
    logger.info("Step 5: Saving")
    merged_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"  Saved: {OUTPUT_PATH}")

    logger.info("\nColumn overview:")
    logger.info(f"  Original MSD columns : {len(msd_df.columns)}")
    logger.info(f"  BoW word columns     : {len(bow_cols)}  (bow_<word>, raw counts)")
    logger.info(f"  lyrics_available flag: 1 column")
    logger.info(f"  Total columns        : {merged_df.shape[1]}")


if __name__ == "__main__":
    main()