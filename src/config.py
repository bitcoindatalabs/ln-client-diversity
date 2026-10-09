"""
Configuration and Path Management for Paper 01 Experiment Pipeline.
"""

from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SRC_DIR.parent
REPO_ROOT = PAPER_ROOT  # this repo; it sits in dev/github/, next to lightning-data/

# Output Paths (Saved directly into paper LaTeX directory)
PAPER_DIR = PAPER_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
DATA_DIR = PAPER_ROOT / "data"

# Local Data Connectors (pointing to ../lightning-data/data; used only by src/archive_v1/)
EXTERNAL_DATA_DIR = REPO_ROOT.parent / "lightning-data" / "data"
CHANNEL_PROFILE_PARQUET = EXTERNAL_DATA_DIR / "channel_profile.parquet"
NODE_FEATURE_PARQUET = EXTERNAL_DATA_DIR / "node_feature.parquet"
NODE_PROFILE_PARQUET = EXTERNAL_DATA_DIR / "node_profile.parquet"
NODE_TYPES_JSON = EXTERNAL_DATA_DIR / "ln_node_types.json"
GRAPH_JSON = EXTERNAL_DATA_DIR / "graph" / "gall.json"

# Data cutoff: the last gossip snapshot used in the paper. The data lake keeps growing; every stage truncates
# its inputs here so that reruns reproduce the published numbers.
DATA_CUTOFF = "20261004"

# Simulation Hyperparameters
RANDOM_SEED = 42
MONTE_CARLO_ITERATIONS = 50
PERCOLATION_STEPS = 21  # 0.0 to 1.0 in 0.05 increments
PAYMENT_SAMPLES = 5000

# Client Color Mapping for Publication Figures — same validated palette as the deep-dive report
# (dataviz validate_palette.js, light surface: all checks pass; worst CVD pair 7.8 → direct-label series)
CLIENT_PALETTE = {
    "LND": "#2a78d6",
    "CLN": "#D4603A",
    "Eclair": "#1a9e6e",
    "LDK": "#4a3aa7",
    "Unknown": "#8893A0",
}
