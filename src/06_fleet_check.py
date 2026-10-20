"""
Stage 6: robustness of the node-count statistics to one managed fleet.

From mid-2025 one operator ran hundreds to about two thousand small public LND nodes with aliases "LNT.<name>",
each with one or two channels, most of them to the operator's hubs (identified in our follow-up paper on upgrade
dynamics). Every statistic that counts nodes counts each of them as an independent operator. A node belongs to the
fleet if its alias starts with "LNT.", or if at least half of its channels go to a fleet hub (an "LNT." node with at
least 50 channels); the definition is incident_metrics.fleet_flags_from().

For the monthly snapshots of stage 1 and the latest snapshot, this stage reports the fleet's size and capacity and
every client's node share with and without it; for the routable core of 2026-08-25 (stage 4), the fleet's share
of the core and the reachability loss of the observed outage footprint and of the two-day placebo windows of stage 5
when only pairs of nodes outside the fleet are counted (fleet nodes still carry paths). Capacity shares change by
less than the fleet's capacity, which is reported alongside.

Outputs
  data/fleet_check.json

Usage:  python src/06_fleet_check.py
"""

import importlib.util
import json
import logging

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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
sup = _load("supporting", "05_supporting_analyses.py")

CORE_DATE = "20260825"                  # routable core of stage 4, the day before the CLN call
EVENT_AFTER = "20260827"
AMOUNT = 100_000


def snapshot_check(date: str) -> dict:
    snap = census.snapshot(date)
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


def graph_fleet(G) -> np.ndarray:
    """Fleet membership of the nodes of a stage-4 graph, from all announced channels of its snapshot."""
    pol = im.read_policies(G.date, ["channel_id", "node_pub", "peer_pub"]).drop_duplicates("channel_id")
    fl = im.fleet_flags(G.date, pd.concat([pol["node_pub"], pol["peer_pub"]]).value_counts())
    return np.array([bool(fl.get(k, False)) for k in G.pub])


def r_all_among(G, failed: np.ndarray, count: np.ndarray, amount: int = AMOUNT) -> float:
    """R_all(a) of stage 4 counting only pairs of `count` nodes: connected pairs among surviving `count` nodes over
    all pairs of `count` nodes; every surviving node can carry paths."""
    alive = ~failed
    keep = (G.cap >= amount) & alive[G.u] & alive[G.v]
    g = coo_matrix((np.ones(keep.sum()), (G.u[keep], G.v[keep])), shape=(G.n, G.n))
    _, comp = connected_components(g, directed=False)
    k = np.bincount(comp[alive & count])
    n = count.sum()
    return float((k * (k - 1)).sum() / (n * (n - 1)))


def footprint(G0, G2, fleet: np.ndarray) -> dict:
    """Two-day footprint (nodes of G0 with no usable channel in G2): change in R_all, all pairs and non-fleet pairs."""
    gone = ~np.isin(G0.pub, G2.pub)
    none, everyone = np.zeros(G0.n, bool), np.ones(G0.n, bool)
    return {"n_failed": int(gone.sum()), "n_failed_fleet": int((gone & fleet).sum()),
            "dR_all": r_all_among(G0, gone, everyone) - r_all_among(G0, none, everyone),
            "dR_all_exfleet": r_all_among(G0, gone, ~fleet) - r_all_among(G0, none, ~fleet),
            "R_all_base": r_all_among(G0, none, everyone), "R_all_base_exfleet": r_all_among(G0, none, ~fleet)}


def core_check(labels: pd.DataFrame) -> dict:
    G = res.Graph(CORE_DATE, labels)
    in_fleet = graph_fleet(G)
    lnd = G.client == "LND"
    out = {
        "date": CORE_DATE, "core_nodes": int(G.n), "fleet_nodes": int(in_fleet.sum()),
        "fleet_share": round(100 * in_fleet.mean(), 2),
        "fleet_cap_share": round(100 * G.node_cap[in_fleet].sum() / G.node_cap.sum(), 3),
        "lnd_node_share": round(100 * lnd.mean(), 2),
        "lnd_node_share_exfleet": round(100 * lnd[~in_fleet].mean(), 2),
        "fleet_median_degree": float(np.median(G.degree[in_fleet])) if in_fleet.any() else None,
    }
    # reachability of the observed footprint and of the placebo windows, all pairs vs non-fleet pairs (100k sat)
    obs = footprint(G, res.Graph(EVENT_AFTER, labels), in_fleet)
    dset = {d for d in im.gossip_dates() if sup.WINDOW_START <= d <= DATA_CUTOFF}
    graphs, rows = {}, []

    def graph(d):
        if d not in graphs:
            graphs[d] = res.Graph(d, labels)
        return graphs[d]

    for d0 in pd.date_range("2026-05-01", "2026-08-23"):
        a, b = d0.strftime("%Y%m%d"), (d0 + pd.Timedelta(days=2)).strftime("%Y%m%d")
        if a not in dset or b not in dset:
            continue
        kind = "tor" if a in sup.TOR_WINDOWS else "close-wave" if a in sup.CLOSE_WAVE_WINDOWS else "ordinary"
        rows.append({"from": a, "kind": kind, **footprint(graph(a), graph(b), graph_fleet(graph(a)))})
    pl = pd.DataFrame(rows)
    o = pl[pl["kind"] == "ordinary"]
    out["reachability_100k"] = {
        "observed": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in obs.items()},
        "ordinary_windows": len(o),
        "ordinary_dR_all_median": round(float(o["dR_all"].median()), 4),
        "ordinary_dR_all_exfleet_median": round(float(o["dR_all_exfleet"].median()), 4),
        "ordinary_worse_than_outage": int((o["dR_all"] <= obs["dR_all"]).sum()),
        "ordinary_worse_than_outage_exfleet": int((o["dR_all_exfleet"] <= obs["dR_all_exfleet"]).sum()),
        "tor_windows": pl[pl["kind"] == "tor"].round(4).to_dict(orient="records"),
        "series": pl.round(4).to_dict(orient="records"),
    }
    return out


def main():
    monthly = [snapshot_check(d) for d in census.monthly_dates()]
    for m in monthly:
        log.info("%s: fleet %d of %d nodes; LND %.1f%% -> %.1f%% without", m["date"], m["fleet_nodes"], m["nodes"],
                 m["node_share"]["LND"], m["node_share_exfleet"]["LND"])
    latest = max(d for d in im.gossip_dates() if d <= DATA_CUTOFF)
    out = {"data_cutoff": DATA_CUTOFF, "alias_prefix": im.FLEET_ALIAS_PREFIX,
           "hub_min_channels": im.FLEET_HUB_MIN_CHANNELS,
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
    log.info("Core: %s", {k: v for k, v in out["core"].items() if k != "reachability_100k"})
    r = out["core"]["reachability_100k"]
    log.info("Reachability @100k: outage %s; ordinary median %.4f / %.4f ex-fleet; %d / %d ordinary windows worse",
             r["observed"], r["ordinary_dR_all_median"], r["ordinary_dR_all_exfleet_median"],
             r["ordinary_worse_than_outage"], r["ordinary_worse_than_outage_exfleet"])
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
