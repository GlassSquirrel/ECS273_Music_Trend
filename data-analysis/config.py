"""Shared paths, constants, and feature definitions for all analysis scripts."""
import os

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(_HERE, ".."))

DATA_PATH   = os.path.join(ROOT, "data", "msd_merged.csv")
FIGURES_DIR = os.path.join(_HERE, "figures")
RESULTS_DIR = os.path.join(_HERE, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Clustering feature set ────────────────────────────────────────────────────
# energy / danceability are all 0.0 in this subset (not computed by Echo Nest)
# key is circular: encoded as sin + cos of (key / 12 * 2π)
CLUSTER_FEATURES = [
    # Tempo / rhythm
    "tempo",
    # Loudness
    "loudness",
    "avg_segment_loudness_max",
    "avg_segment_loudness_start",
    # Duration / structure
    "duration",
    "sections_count",
    "bars_count",
    # Harmonic (circular key encoding + mode)
    "key_sin",
    "key_cos",
    "mode",
    # Time signature (3/4 vs 4/4 etc.)
    "time_signature",
    # Timbral: 12 MFCC-like Echo Nest coefficients
    *[f"avg_timbre_{i}" for i in range(12)],
    # Chroma / pitch class profile
    *[f"avg_pitch_{i}" for i in range(12)],
]

# Human-readable labels for a subset of features (for EDA plots)
EDA_FEATURES = {
    "tempo":              "Tempo (BPM)",
    "loudness":           "Loudness (dB)",
    "duration":           "Duration (s)",
    "sections_count":     "# Sections",
    "bars_count":         "# Bars",
    "song_hotttnesss":    "Song Hotness",
    "artist_familiarity": "Artist Familiarity",
    "artist_hotttnesss":  "Artist Hotness",
}

# ── Aesthetics ────────────────────────────────────────────────────────────────
PALETTE   = "tab10"
FIG_DPI   = 150
STYLE     = "seaborn-v0_8-whitegrid"
