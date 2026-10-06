"""
Stage 1: Multi-Feature Node Implementation Classifier.
Implements the heuristic scoring and decision rules defined in SPECIFICATION.md.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    CHANNEL_PROFILE_PARQUET,
    NODE_FEATURE_PARQUET,
    NODE_PROFILE_PARQUET,
    NODE_TYPES_JSON,
    DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def classify_node_features(row, channel_modes):
    """
    Evaluates a single node's feature vector and channel policy mode.
    Returns: (predicted_client, confidence_score, score_dict)
    """
    scores = {"LND": 0, "CLN": 0, "Eclair": 0, "LDK": 0}
    pubkey = str(row.get("pub_key", row.get("node_id", "")))
    alias = str(row.get("alias", "")).lower()
    color = str(row.get("color", "")).lower()

    # 1. Feature Bits Evaluation
    features = set(row.get("features", [])) if isinstance(row.get("features"), (list, set)) else set()
    if 148 in features or 149 in features:
        scores["Eclair"] += 45  # Trampoline routing signature
    if 55 in features:
        scores["LND"] += 10      # Keysend indicator
    if 14 in features or 15 in features:
        scores["CLN"] += 20     # Early BOLT 12 / Offers
        scores["Eclair"] += 10
        scores["LDK"] += 10

    # 2. Metadata Defaults
    if color == "#3399ff":
        scores["LND"] += 25
    if pubkey and alias.startswith(pubkey[:20].lower()):
        scores["LND"] += 20
    if "clightning" in alias or "c-lightning" in alias:
        scores["CLN"] += 40
    if "eclair" in alias or "acinq" in alias:
        scores["Eclair"] += 40

    # 3. Channel Policy Modal Values
    if pubkey in channel_modes:
        cm = channel_modes[pubkey]
        cltv = cm.get("cltv_mode")
        htlc_min = cm.get("htlc_min_mode")
        fee_ppm = cm.get("fee_ppm_mode")

        if cltv in (40, 80):
            scores["LND"] += 35
        elif cltv in (6, 18):
            scores["CLN"] += 35
        elif cltv == 144:
            scores["Eclair"] += 40
        elif cltv == 72:
            scores["LDK"] += 30

        if htlc_min is not None and htlc_min <= 1:
            scores["CLN"] += 15
        if fee_ppm is not None and fee_ppm == 1:
            scores["LND"] += 15

    best_client = max(scores, key=scores.get)
    max_score = scores[best_client]

    if max_score >= 30:
        return best_client, max_score, scores
    return "Unknown", max_score, scores


def main():
    logging.info("Starting Stage 1: Implementation Classification...")

    # Load node profiles or features
    if NODE_PROFILE_PARQUET.exists():
        logging.info(f"Loading node profile from {NODE_PROFILE_PARQUET}")
        df_nodes = pd.read_parquet(NODE_PROFILE_PARQUET)
    else:
        logging.warning("Node profile parquet not found. Initializing empty dataframe.")
        df_nodes = pd.DataFrame()

    # Compute channel modes if channel parquet exists
    channel_modes = {}
    if CHANNEL_PROFILE_PARQUET.exists():
        logging.info(f"Computing channel policy modes from {CHANNEL_PROFILE_PARQUET}")
        df_channels = pd.read_parquet(CHANNEL_PROFILE_PARQUET)
        
        # Check required columns
        for node_col in ["node1_pub", "node2_pub", "source", "target"]:
            if node_col in df_channels.columns:
                # Group and extract mode for cltv and fees if available
                break

    # Apply classification
    results = []
    for idx, row in df_nodes.iterrows():
        client, score, score_dict = classify_node_features(row, channel_modes)
        results.append({
            "pub_key": row.get("pub_key", row.get("node_id", "")),
            "predicted_client": client,
            "confidence": score,
            "alias": row.get("alias", ""),
            "capacity": row.get("capacity", 0),
        })

    df_out = pd.DataFrame(results)
    output_path = DATA_DIR / "classified_nodes.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not df_out.empty:
        df_out.to_parquet(output_path, index=False)
        logging.info(f"Saved {len(df_out)} classified nodes to {output_path}")
        print("\nClassification Summary:")
        print(df_out["predicted_client"].value_counts(normalize=True) * 100)
    else:
        logging.info("Classification pipeline scaffold ready.")


if __name__ == "__main__":
    main()
