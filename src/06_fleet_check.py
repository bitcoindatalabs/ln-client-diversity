"""
Stage 6: robustness of the node-count statistics to one managed fleet.

From mid-2025 one operator ran hundreds to about two thousand small public LND nodes with aliases "LNT.<name>",
each with one or two channels, most of them to the operator's hubs (identified in our follow-up paper on upgrade
dynamics). Every statistic that counts nodes counts each of them as an independent operator. A node belongs to the
fleet if its alias starts with "LNT.", or if at least half of its channels go to a fleet hub (an "LNT." node with at
least 50 channels).

For the monthly snapshots of stage 1 and the latest snapshot, this stage reports the fleet's size and capacity and
every client's node share with and without it; for the routable core of 2026-08-25 (stage 4), the fleet's share
of the core. Capacity shares change by less than the fleet's capacity, which is reported alongside.

Outputs
  data/fleet_check.json

Usage:  python src/06_fleet_check.py
"""

import importlib.util
import json
import logging

import numpy as np
import pandas as pd

from config import DATA_CUTOFF, DATA_DIR, SRC_DIR

from lnmetrics import incident_metrics as im

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


census = _load("census", "01_client_census.py")
res = _load("resilience", "04_resilience.py")

FLEET_ALIAS_PREFIX = "LNT."
FLEET_HUB_MIN_CHANNELS = 50
CORE_DATE = "20260825"


def fleet_flags(date: str, channels: pd.Series) -> pd.Series:
    """Fleet membership of every node in `channels` (index: pub_key, value: channel count) on `date`."""
    alias = im.read_nodes(date, ["pub_key", "alias"]).set_index("pub_key")["alias"].reindex(channels.index).fillna("")
    named = alias.str.startswith(FLEET_ALIAS_PREFIX)
    hubs = set(named.index[named & (channels >= FLEET_HUB_MIN_CHANNELS)])
    ch = im.read_policies(date, ["channel_id", "node_pub", "peer_pub"]).drop_duplicates("channel_id")
    ends = pd.concat([pd.DataFrame({"pk": ch["node_pub"], "hub": ch["peer_pub"].isin(hubs)}),
                      pd.DataFrame({"pk": ch["peer_pub"], "hub": ch["node_pub"].isin(hubs)})])
    to_hub = ends.groupby("pk")["hub"].mean().reindex(channels.index).fillna(0)
    return named | (to_hub >= 0.5)


def snapshot_check(date: str) -> dict:
    snap = census.snapshot(date)
    snap["fleet"] = fleet_flags(date, snap["channels"])
    with_f, without = census.shares(snap), census.shares(snap[~snap["fleet"]])
    f = snap[snap["fleet"]]
    return {
        "date": f"{date[:4]}-{date[4:6]}-{date[6:]}", "nodes": len(snap), "fleet_nodes": len(f),
        "fleet_node_share": round(100 * len(f) / len(snap), 2),
        "fleet_clients": f["client"].value_counts().to_dict(),
        "fleet_cap_btc": round(f["capacity"].sum() / 1e8, 3),
        "fleet_cap_share": round(100 * f["capacity"].sum() / snap["capacity"].sum(), 3),
        "node_share": {c: round(with_f[c]["node_share"], 2) for c in census.CLIENTS},
        "node_share_exfleet": {c: round(without[c]["node_share"], 2) for c in census.CLIENTS},
        "node_share_high_exfleet": {c: round(without[c]["node_share_high"], 2) for c in census.CLIENTS},
        "cap_share_exfleet": {c: round(without[c]["cap_share"], 2) for c in census.CLIENTS},
    }


def core_check(labels: pd.DataFrame) -> dict:
    G = res.Graph(CORE_DATE, labels)
    pol = im.read_policies(CORE_DATE, ["channel_id", "node_pub", "peer_pub"]).drop_duplicates("channel_id")
    channels = pd.concat([pol["node_pub"], pol["peer_pub"]]).value_counts()
    fl = fleet_flags(CORE_DATE, channels)
    in_fleet = np.array([bool(fl.get(k, False)) for k in G.pub])
    lnd = G.client == "LND"
    return {
        "date": CORE_DATE, "core_nodes": int(G.n), "fleet_nodes": int(in_fleet.sum()),
        "fleet_share": round(100 * in_fleet.mean(), 2),
        "fleet_cap_share": round(100 * G.node_cap[in_fleet].sum() / G.node_cap.sum(), 3),
        "lnd_node_share": round(100 * lnd.mean(), 2),
        "lnd_node_share_exfleet": round(100 * lnd[~in_fleet].mean(), 2),
        "fleet_median_degree": float(np.median(G.degree[in_fleet])) if in_fleet.any() else None,
    }


def main():
    monthly = [snapshot_check(d) for d in census.monthly_dates()]
    for m in monthly:
        log.info("%s: fleet %d of %d nodes; LND %.1f%% -> %.1f%% without", m["date"], m["fleet_nodes"], m["nodes"],
                 m["node_share"]["LND"], m["node_share_exfleet"]["LND"])
    latest = max(d for d in im.gossip_dates() if d <= DATA_CUTOFF)
    out = {"data_cutoff": DATA_CUTOFF, "alias_prefix": FLEET_ALIAS_PREFIX, "hub_min_channels": FLEET_HUB_MIN_CHANNELS,
           "latest": snapshot_check(latest), "monthly": monthly, "core": core_check(im.node_labels())}
    epoch_b = [m for m in monthly if m["date"] >= "2024-07-01"]
    out["epoch_b_lnd_node_share"] = {
        "with_fleet": [min(m["node_share"]["LND"] for m in epoch_b), max(m["node_share"]["LND"] for m in epoch_b)],
        "without_fleet": [min(m["node_share_exfleet"]["LND"] for m in epoch_b),
                          max(m["node_share_exfleet"]["LND"] for m in epoch_b)],
        "first": epoch_b[0]["date"], "last": epoch_b[-1]["date"],
        "first_month_with_fleet": next((m["date"] for m in monthly if m["fleet_nodes"] > 0), None),
        "max_fleet_month": max(monthly, key=lambda m: m["fleet_nodes"])["date"],
        "max_fleet_nodes": max(m["fleet_nodes"] for m in monthly),
    }
    path = DATA_DIR / "fleet_check.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("Latest: %s", out["latest"])
    log.info("Core: %s", out["core"])
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
