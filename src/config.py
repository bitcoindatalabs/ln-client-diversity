"""
Paths and parameters shared by the pipeline stages.
"""

import os
from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

# Output Paths (written directly into the LaTeX directory)
PAPER_DIR = REPO_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
DATA_DIR = REPO_ROOT / "data"

# Input: daily gossip snapshots and on-chain extracts as Parquet (layout in src/lnmetrics/incident_metrics.py).
# Set LN_PARQUET_DIR to their location; the default is data/parquet inside this repo (not committed).
LN_PARQUET_DIR = Path(os.environ.get("LN_PARQUET_DIR", DATA_DIR / "parquet"))

# Data cutoff: the last gossip snapshot used in the paper. The snapshots keep growing; every stage truncates
# its inputs here so that reruns reproduce the published numbers.
DATA_CUTOFF = "20261004"

# Simulation Hyperparameters
RANDOM_SEED = 42
MONTE_CARLO_ITERATIONS = 50
PERCOLATION_STEPS = 21  # 0.0 to 1.0 in 0.05 increments
PAYMENT_SAMPLES = 5000

# Client colours for the figures (checked for colour-vision deficiency; series are also labelled directly)
CLIENT_PALETTE = {
    "LND": "#2a78d6",
    "CLN": "#D4603A",
    "Eclair": "#1a9e6e",
    "LDK": "#4a3aa7",
    "Unknown": "#8893A0",
}
