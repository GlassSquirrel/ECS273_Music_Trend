"""Shared paths, constants, and feature definitions for test clustering."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))

DATA_PATH = os.path.join(ROOT, "data", "msd_merged.csv")
FIGURES_DIR = os.path.join(_HERE, "figures")
RESULTS_DIR = os.path.join(_HERE, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# key is circular: encoded as sin + cos of (key / 12 * 2π)
CLUSTER_FEATURES = [
    "tempo",
    "loudness",
    "avg_segment_loudness_max",
    "key_sin",
    "key_cos",
    *[f"avg_timbre_{i}" for i in range(12)],
]

PALETTE = "tab10"
FIG_DPI = 150
STYLE = "seaborn-v0_8-whitegrid"
