"""
Configuration and Path Management for Paper 01 Experiment Pipeline.
"""

from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SRC_DIR.parent
REPO_ROOT = PAPER_ROOT.parent.parent

# Output Paths (Saved directly into paper LaTeX directory)
PAPER_DIR = PAPER_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
DATA_DIR = PAPER_ROOT / "data"

# Local Data Connectors (pointing to ../../lightning-data/data)
EXTERNAL_DATA_DIR = REPO_ROOT / "lightning-data" / "data"
CHANNEL_PROFILE_PARQUET = EXTERNAL_DATA_DIR / "channel_profile.parquet"
NODE_FEATURE_PARQUET = EXTERNAL_DATA_DIR / "node_feature.parquet"
NODE_PROFILE_PARQUET = EXTERNAL_DATA_DIR / "node_profile.parquet"
NODE_TYPES_JSON = EXTERNAL_DATA_DIR / "ln_node_types.json"
GRAPH_JSON = EXTERNAL_DATA_DIR / "graph" / "gall.json"

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
