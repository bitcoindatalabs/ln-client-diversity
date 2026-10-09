"""
incident_metrics.py — measurements of how the Lightning Network responded to incidents.

Outage and close metrics used by every stage of this repo, defined once.

Inputs (LN_PARQUET_DIR, set in src/config.py):
  gossip_daily/nodes/date=YYYYMMDD.parquet      node announcements per day
  gossip_daily/policies/date=YYYYMMDD.parquet   channel directions per day (incl. `disabled`)
  onchain/ln_close_txs.parquet                  channel closes, from spends of funding outputs on chain
  onchain/ln_commitment_spends.parquet          spends of force-close outputs

Conventions:
  - Client labels are fixed per node at a pre-incident reference date (default 2026-08-01), so
    cohorts don't change because of the incident itself (nodes that go dark drop out of gossip).
  - Outage = the *peer* side: share of channel directions toward a cohort's nodes that peers mark
    disabled (an offline node cannot announce anything itself).
  - Force closes are split into "mass-close events" (one node party to >= MASS_CLOSE_MIN force
    closes in a UTC day: an operator closing many channels at once) and the organic remainder.
    Chain data does not reveal which side broadcast a commitment, so this is how single-operator
    spikes are kept from being read as network-wide effects.
  - Sweep fees: one sweep tx often spends many outputs; its fee is allocated per input
    (fee / n_inputs) so totals are not counted once per swept output.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config import LN_PARQUET_DIR
from .client_fingerprint import RULES_VERSION, classify_node

PARQUET = Path(LN_PARQUET_DIR)
GOSSIP_NODES = PARQUET / "gossip_daily" / "nodes"
GOSSIP_POLICIES = PARQUET / "gossip_daily" / "policies"
CLOSES_PATH = PARQUET / "onchain" / "ln_close_txs.parquet"
SPENDS_PATH = PARQUET / "onchain" / "ln_commitment_spends.parquet"
ANALYSIS_DIR = PARQUET / "analysis"

CLIENTS = ["LND", "CLN", "Eclair", "LDK", "Unknown"]
REF_DATE = "20260801"
MASS_CLOSE_MIN = 20          # force closes by one node in one UTC day → mass-close event
DARK_SHARE = 0.8             # node is "dark" when >= 80% of directions toward it are disabled

# Incident timeline; dates in UTC, verified against primary sources on 2026-10-07 (sources: paper/references.bib).
INCIDENTS = [
    {"date": "2026-07-10", "label": "Tor DoS wave: onion-service directories collapse; Tor-only nodes lose peers", "client": "all", "observed": True,
     "source": "https://1aeo.com/blog/onion-service-directory-capacity-june-2026.html"},
    {"date": "2026-07-19", "label": "CLN memory-exhaustion DoS disclosed (Delving Bitcoin, fixed in 26.04/26.06)", "client": "CLN",
     "source": "https://delvingbitcoin.org/t/vulnerability-disclosure-twin-memory-exhaustion-dos-vulnerabilities-in-core-lightning/2731"},
    {"date": "2026-08-07", "label": "BTCPay < 2.4.2 advisory: LND macaroons exfiltrated, funds stolen", "client": "LND",
     "source": "https://blog.btcpayserver.org/security-advisory-btcpay-server-2-4-2/"},
    {"date": "2026-08-11", "label": "LND: 10 advisories (3 high), all < 0.19", "client": "LND",
     "source": "https://security.lightning.engineering/"},
    {"date": "2026-08-26", "label": "CLN: upgrade or restart with --offline", "client": "CLN",
     "source": "https://x.com/Core_LN/status/2092755509510283423"},
    {"date": "2026-08-28", "label": "CLN 26.06.7 released", "client": "CLN",
     "source": "https://github.com/ElementsProject/lightning/releases/tag/v26.06.7"},
    {"date": "2026-09-09", "label": "LDK 0.2.6 security release", "client": "LDK",
     "source": "https://github.com/lightningdevkit/rust-lightning/releases/tag/v0.2.6"},
    {"date": "2026-09-14", "label": "Eclair 0.14.3 security release", "client": "Eclair",
     "source": "https://github.com/ACINQ/eclair/releases/tag/v0.14.3"},
    {"date": "2026-09-21", "label": "Lightning Labs: 8 advisories (4 lnd, 4 btcd)", "client": "LND",
     "source": "https://security.lightning.engineering/"},
    {"date": "2026-09-22", "label": "CLN 26.06.8 released", "client": "CLN",
     "source": "https://github.com/ElementsProject/lightning/releases/tag/v26.06.8"},
    {"date": "2026-10-02", "label": "CLN: attackers targeting nodes <= 26.06.7", "client": "CLN",
     "source": "https://x.com/Core_LN/status/2105835841847242976"},
]


# ---------------------------------------------------------------------------------------
# Gossip helpers
# ---------------------------------------------------------------------------------------
QUALITY_PATH = PARQUET / "gossip_daily" / "_quality.parquet"
# Vantage epochs of our gossip view (observed 2026-10-06 from the extracts themselves):
#   A  ..2024-04-30  ~42k channels / ~8.4k nodes with channels — narrower view
#   gap 2024-05..06 partial dumps (~3–8k channels); 2024-07-24..08-05 and 2025-03-27..05-18 "wallet locked"
#   B  2024-07-01..  ~41–50k channels / ~11–13k nodes
# Compare levels within an epoch; shares (ratios) are less sensitive but still state the epoch.
EPOCHS = [("A", "00000000", "20240430"), ("gap", "20240501", "20240630"), ("B", "20240701", "99999999")]


def _all_gossip_dates() -> List[str]:
    def dates(d):
        return {Path(f).stem.split("=")[1] for f in glob.glob(str(d / "date=*.parquet"))}
    return sorted(dates(GOSSIP_NODES) & dates(GOSSIP_POLICIES))


def gossip_quality(refresh: bool = False) -> pd.DataFrame:
    """Per-day quality manifest (incremental, cached in gossip_daily/_quality.parquet):
    channels, nodes with channels, capacity, share of channels updated within 14 days, epoch, status:
      ok | partial (< 50% of the epoch's median channel count) | stale (< 50% of channels fresh)."""
    have = pd.read_parquet(QUALITY_PATH) if QUALITY_PATH.exists() and not refresh else pd.DataFrame()
    done = set(have["date"]) if len(have) else set()
    rows = []
    for d in _all_gossip_dates():
        if d in done:
            continue
        p = read_policies(d, ["channel_id", "node_pub", "peer_pub", "capacity", "last_update"])
        ch = p.drop_duplicates("channel_id")
        ts = pd.Timestamp(d).timestamp() + 86400
        rows.append({"date": d, "channels": len(ch), "nodes": int(pd.concat([ch["node_pub"], ch["peer_pub"]]).nunique()),
                     "cap_btc": ch["capacity"].sum() / 1e8,
                     "fresh14_share": float((ch["last_update"] >= ts - 14 * 86400).mean()) if len(ch) else 0.0})
    q = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if rows else have
    q = q.drop(columns=[c for c in ("epoch", "status") if c in q], errors="ignore").sort_values("date")
    q["epoch"] = "B"
    for name, lo, hi in EPOCHS:
        q.loc[(q["date"] >= lo) & (q["date"] <= hi), "epoch"] = name
    med = q.groupby("epoch")["channels"].transform("median")
    q["status"] = np.select([q["fresh14_share"] < 0.5, (q["channels"] < 0.5 * med) | (q["epoch"] == "gap")],
                            ["stale", "partial"], "ok")
    tmp = str(QUALITY_PATH) + ".tmp"
    q.to_parquet(tmp, index=False)
    os.replace(tmp, QUALITY_PATH)
    return q.reset_index(drop=True)


def gossip_dates(include_bad: bool = False) -> List[str]:
    """Dates (YYYYMMDD) with both gossip extracts; by default only days whose quality status is "ok"."""
    if include_bad:
        return _all_gossip_dates()
    q = gossip_quality()
    return sorted(q.loc[q["status"] == "ok", "date"])


def read_policies(date: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(GOSSIP_POLICIES / f"date={date}.parquet", columns=columns)


def read_nodes(date: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(GOSSIP_NODES / f"date={date}.parquet", columns=columns)


def _cltv_modes(pol: pd.DataFrame) -> pd.Series:
    """Modal time_lock_delta of the policies each node sets itself."""
    return (pol.groupby(["node_pub", "time_lock_delta"]).size().reset_index(name="n")
               .sort_values(["node_pub", "n"], ascending=[True, False])
               .drop_duplicates("node_pub").set_index("node_pub")["time_lock_delta"])


# ---------------------------------------------------------------------------------------
# Client labels
# ---------------------------------------------------------------------------------------
def node_labels(ref_date: str = REF_DATE, refresh: bool = False) -> pd.DataFrame:
    """One client label per node ever seen in gossip, taken from the snapshot closest to ref_date
    (latest on/before it, else earliest after). Cached in analysis/node_client_labels_<ref>.parquet."""
    cache = ANALYSIS_DIR / f"node_client_labels_{ref_date}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        if (df["rules_version"] == RULES_VERSION).all():
            return df

    dates = gossip_dates()
    order = [d for d in reversed(dates) if d <= ref_date] + [d for d in dates if d > ref_date]
    rows: Dict[str, dict] = {}
    for d in order:
        nodes = read_nodes(d, ["pub_key", "alias", "color", "feature_bits"])
        nodes = nodes[~nodes["pub_key"].isin(rows.keys())]
        if nodes.empty:
            continue
        cltv = _cltv_modes(read_policies(d, ["node_pub", "time_lock_delta"]))
        for r in nodes.itertuples(index=False):
            bits = list(r.feature_bits) if r.feature_bits is not None else []
            mode = cltv.get(r.pub_key)
            client, conf, reason = classify_node(bits, r.color, r.alias, r.pub_key,
                                                 float(mode) if mode is not None else None)
            rows[r.pub_key] = {"pub_key": r.pub_key, "alias": r.alias, "client": client,
                               "confidence": conf, "reason": reason, "label_date": d,
                               "has_features": bool(bits)}
    df = pd.DataFrame(rows.values())
    df["rules_version"] = RULES_VERSION
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def label_map(labels: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(labels["pub_key"], labels["client"]))


# ---------------------------------------------------------------------------------------
# Q1 / RQ3: observed outage
# ---------------------------------------------------------------------------------------
def outage_daily(labels: pd.DataFrame, dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Per cohort × day: directions toward the cohort, share disabled by peers (count and capacity),
    plus cohort size (nodes with >= 1 channel, node capacity) from the same snapshot.
    Node capacity counts each channel for both endpoints (standard 'node capacity')."""
    lab = label_map(labels)
    out = []
    for d in dates or gossip_dates():
        p = read_policies(d, ["channel_id", "capacity", "node_pub", "peer_pub", "disabled"])
        p["target"] = p["peer_pub"].map(lab).fillna("Unknown")
        p["cap_dis"] = np.where(p["disabled"], p["capacity"], 0)
        g = p.groupby("target").agg(dirs=("disabled", "size"), disabled=("disabled", "sum"),
                                    cap_btc=("capacity", "sum"), cap_disabled_btc=("cap_dis", "sum"),
                                    nodes=("peer_pub", "nunique"))
        # robustness: drop the single node with the most disabled capacity in each cohort that day
        per_node = p.groupby(["target", "peer_pub"]).agg(cap=("capacity", "sum"), cap_dis=("cap_dis", "sum"))
        top = per_node.sort_values("cap_dis").groupby(level=0).tail(1).droplevel(1)
        g["cap_btc_x1"] = g["cap_btc"] - top["cap"].reindex(g.index).fillna(0)
        g["cap_disabled_btc_x1"] = g["cap_disabled_btc"] - top["cap_dis"].reindex(g.index).fillna(0)
        g[["cap_btc", "cap_disabled_btc", "cap_btc_x1", "cap_disabled_btc_x1"]] /= 1e8
        g["pct_disabled"] = 100 * g["disabled"] / g["dirs"]
        g["pct_cap_disabled"] = 100 * g["cap_disabled_btc"] / g["cap_btc"]
        g["pct_cap_disabled_x1"] = 100 * g["cap_disabled_btc_x1"] / g["cap_btc_x1"]
        g["date"] = pd.Timestamp(d)
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def node_network(date: str = REF_DATE) -> Dict[str, str]:
    """tor-only / clearnet-only / both / none, from the node announcement on `date`."""
    n = read_nodes(date, ["pub_key", "has_tor", "has_clearnet"])
    net = np.select([n["has_tor"] & ~n["has_clearnet"], ~n["has_tor"] & n["has_clearnet"], n["has_tor"] & n["has_clearnet"]],
                    ["tor-only", "clearnet-only", "both"], default="none")
    return dict(zip(n["pub_key"], net))


def outage_by_network(labels: pd.DataFrame, dates: Optional[Iterable[str]] = None,
                      ref_date: str = REF_DATE) -> pd.DataFrame:
    """Disabled share per cohort × target network type × day (transport confound check: Tor outages
    hit Tor-only nodes of every client at once, e.g. 2026-07-10)."""
    lab, net = label_map(labels), node_network(ref_date)
    out = []
    for d in dates or gossip_dates():
        p = read_policies(d, ["peer_pub", "disabled"])
        p["target"] = p["peer_pub"].map(lab).fillna("Unknown")
        p["net"] = p["peer_pub"].map(net).fillna("unknown")
        g = p.groupby(["target", "net"]).agg(dirs=("disabled", "size"), disabled=("disabled", "sum")).reset_index()
        g["pct_disabled"] = 100 * g["disabled"] / g["dirs"]
        g["date"] = pd.Timestamp(d)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def outage_effect(daily: pd.DataFrame, baseline=("2026-08-01", "2026-08-25"),
                  peak_day="2026-08-27") -> pd.DataFrame:
    """Baseline mean ± sd, peak value and effect in sd units per cohort (count and capacity)."""
    rows = []
    for t, g in daily.groupby("target"):
        b = g[(g["date"] >= baseline[0]) & (g["date"] <= baseline[1])]
        pk = g[g["date"] == peak_day]
        row = {"target": t}
        for m in [x for x in ["pct_disabled", "pct_cap_disabled", "pct_cap_disabled_x1", "pct_disabled_clearnet"] if x in g]:
            mu, sd = b[m].mean(), b[m].std()
            v = pk[m].iloc[0] if len(pk) else np.nan
            row.update({f"{m}_base": mu, f"{m}_sd": sd, f"{m}_peak": v, f"{m}_z": (v - mu) / sd if sd else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def node_dark_panel(labels: pd.DataFrame, client: str = "CLN",
                    dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Node × day panel for one cohort: share of directions toward the node disabled by peers,
    node capacity, and a 'dark' flag. Basis for compliance and recovery."""
    members = set(labels.loc[labels["client"] == client, "pub_key"])
    out = []
    for d in dates or gossip_dates():
        p = read_policies(d, ["capacity", "peer_pub", "disabled"])
        p = p[p["peer_pub"].isin(members)]
        g = p.groupby("peer_pub").agg(dirs=("disabled", "size"), disabled=("disabled", "sum"),
                                      cap_btc=("capacity", "sum"))
        g["cap_btc"] /= 1e8
        g["share_disabled"] = g["disabled"] / g["dirs"]
        g["dark"] = g["share_disabled"] >= DARK_SHARE
        g["date"] = pd.Timestamp(d)
        out.append(g.reset_index().rename(columns={"peer_pub": "pub_key"}))
    return pd.concat(out, ignore_index=True)


def compliance_by_size(panel: pd.DataFrame, before="2026-08-25", after="2026-08-27",
                       n_bins: int = 5) -> pd.DataFrame:
    """Share of cohort nodes that went dark after the shutdown call, by pre-incident capacity bin.
    Only nodes visible and not dark on `before` are counted."""
    b = panel[(panel["date"] == before) & ~panel["dark"]].set_index("pub_key")
    a = panel[panel["date"] == after].set_index("pub_key")
    j = b[["cap_btc"]].join(a[["dark"]], how="left")
    j["dark"] = j["dark"].fillna(True)   # dropped out of gossip entirely → counted as dark
    j["size_bin"] = pd.qcut(j["cap_btc"].rank(method="first"), n_bins,
                            labels=[f"Q{i + 1}" for i in range(n_bins)])
    r = j.groupby("size_bin", observed=True).agg(nodes=("dark", "size"), went_dark=("dark", "sum"),
                                                 min_cap_btc=("cap_btc", "min"), max_cap_btc=("cap_btc", "max"))
    r["pct_went_dark"] = 100 * r["went_dark"] / r["nodes"]
    return r.reset_index()


def went_dark(panel: pd.DataFrame, event: str = "2026-08-26") -> set:
    """Nodes visible and not dark the day before `event`, and dark — or absent from gossip — the day after."""
    ev = pd.Timestamp(event)
    pre = panel[(panel["date"] == ev - pd.Timedelta(days=1)) & ~panel["dark"]]
    nxt = panel[panel["date"] == ev + pd.Timedelta(days=1)].set_index("pub_key")["dark"]
    return {pk for pk in pre["pub_key"] if nxt.get(pk, True)}


def recovery_curve(panel: pd.DataFrame, event="2026-08-26", horizon_days: int = 40) -> pd.DataFrame:
    """Of the nodes that went dark (see went_dark), share still dark t days after `event`.
    A node absent from gossip on a day counts as dark that day (not censored)."""
    ev = pd.Timestamp(event)
    cohort = went_dark(panel, event)
    seen_dates = set(panel["date"].unique())
    rows = []
    for t in range(1, horizon_days + 1):
        d = ev + pd.Timedelta(days=t)
        if d not in seen_dates:
            continue
        snap = panel[(panel["date"] == d) & panel["pub_key"].isin(cohort)]
        recovered = int((~snap["dark"]).sum())
        rows.append({"day": t, "date": d, "cohort": len(cohort),
                     "share_still_dark": 1 - recovered / len(cohort) if cohort else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Q2 / Q5: closes, commitment spends, fees
# ---------------------------------------------------------------------------------------
def load_closes(labels: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    c = pd.read_parquet(CLOSES_PATH)
    c["date"] = pd.to_datetime(c["close_time"], unit="s").dt.normalize()
    if labels is not None:
        lab = label_map(labels)
        c["client1"] = c["node1_pub"].map(lab).fillna("Unknown")
        c["client2"] = c["node2_pub"].map(lab).fillna("Unknown")
    c["mass_event"] = False
    f = c["close_type"] == "force"
    if f.any():
        sides = pd.concat([c.loc[f, ["date", "node1_pub"]].rename(columns={"node1_pub": "pub"}),
                           c.loc[f, ["date", "node2_pub"]].rename(columns={"node2_pub": "pub"})])
        per = sides.groupby(["date", "pub"]).size()
        mass = set(per[per >= MASS_CLOSE_MIN].index)
        c.loc[f, "mass_event"] = [(d, a) in mass or (d, b) in mass for d, a, b in
                                  zip(c.loc[f, "date"], c.loc[f, "node1_pub"], c.loc[f, "node2_pub"])]
    return c


def load_spends() -> pd.DataFrame:
    s = pd.read_parquet(SPENDS_PATH)
    s["date"] = pd.to_datetime(s["spend_time"], unit="s").dt.normalize()
    # pre-anchor to_remote outputs are plain P2WPKH (older extracts label them "other")
    s.loc[(s["spend_type"] == "other") & (s["output_role"] == "to_remote_p2wpkh"), "spend_type"] = "to_remote_p2wpkh_spend"
    s["fee_alloc_sat"] = s["spend_fee_sat"] / s["spend_n_inputs"].clip(lower=1)
    return s


def involves(c: pd.DataFrame, client: str) -> pd.Series:
    return (c["client1"] == client) | (c["client2"] == client)


def closes_daily(c: pd.DataFrame) -> pd.DataFrame:
    """Daily close counts by type; force closes split into mass-event vs organic, and organic force
    closes by cohort involvement (a channel involving CLN and LND counts for both)."""
    d = c.groupby(["date", "close_type"]).size().unstack(fill_value=0)
    f = c[c["close_type"] == "force"]
    d["force_mass"] = f[f["mass_event"]].groupby("date").size()
    org = f[~f["mass_event"]]
    d["force_organic"] = org.groupby("date").size()
    for cl in ["LND", "CLN", "Eclair", "LDK"]:
        d[f"force_organic_{cl}"] = org[involves(org, cl)].groupby("date").size()
    d["force_organic_LND_only"] = org[(org["client1"] == "LND") & (org["client2"] == "LND")].groupby("date").size()
    return d.fillna(0).astype(int).reset_index()


def mass_events(c: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate description of mass-close events: day, client of the closing-hub node, count,
    capacity, and the hub node's identity (pub key + alias)."""
    f = c[(c["close_type"] == "force") & c["mass_event"]]
    sides = pd.concat([f[["date", "node1_pub", "capacity"]].rename(columns={"node1_pub": "pub"}),
                       f[["date", "node2_pub", "capacity"]].rename(columns={"node2_pub": "pub"})])
    per = sides.groupby(["date", "pub"]).agg(force_closes=("capacity", "size"), cap_btc=("capacity", "sum"))
    per = per[per["force_closes"] >= MASS_CLOSE_MIN].reset_index()
    per["cap_btc"] /= 1e8
    per["hub_client"] = per["pub"].map(label_map(labels)).fillna("Unknown")
    per["hub_alias"] = per["pub"].map(dict(zip(labels["pub_key"], labels["alias"])))
    return per.rename(columns={"pub": "hub_pub"}).sort_values("date")


def top_closers(c: pd.DataFrame, labels: pd.DataFrame, start: str, end: str, n: int = 10) -> pd.DataFrame:
    """Nodes party to the most closes in [start, end], split by close type."""
    w = c[(c["date"] >= start) & (c["date"] <= end)]
    sides = pd.concat([w[["node1_pub", "close_type", "capacity"]].rename(columns={"node1_pub": "pub"}),
                       w[["node2_pub", "close_type", "capacity"]].rename(columns={"node2_pub": "pub"})])
    t = sides.groupby(["pub", "close_type"]).size().unstack(fill_value=0)
    t["closes"] = t.sum(axis=1)
    t["cap_btc"] = sides.groupby("pub")["capacity"].sum() / 1e8
    t = t.sort_values("closes", ascending=False).head(n).reset_index()
    t["alias"] = t["pub"].map(dict(zip(labels["pub_key"], labels["alias"])))
    t["client"] = t["pub"].map(label_map(labels)).fillna("Unknown")
    t["share_of_window"] = 100 * t["closes"] / len(w)
    return t


def channels_by_cohort(labels: pd.DataFrame, dates: Iterable[str]) -> pd.DataFrame:
    """Open public channels per day involving each cohort (exposure denominator for close rates)."""
    lab = label_map(labels)
    rows = []
    for d in dates:
        p = read_policies(d, ["channel_id", "node_pub", "peer_pub"]).drop_duplicates("channel_id")
        a, b = p["node_pub"].map(lab).fillna("Unknown"), p["peer_pub"].map(lab).fillna("Unknown")
        row = {"date": pd.Timestamp(d), "all": len(p)}
        for cl in ["LND", "CLN", "Eclair", "LDK"]:
            row[cl] = int(((a == cl) | (b == cl)).sum())
        row["LND_only"] = int(((a == "LND") & (b == "LND")).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def force_close_rates(daily: pd.DataFrame, exposure: pd.DataFrame,
                      baseline=("2026-07-01", "2026-08-25"), window=("2026-08-26", "2026-09-08")) -> pd.DataFrame:
    """Organic force closes per 1,000 open channels per day, baseline vs window, per cohort, and the
    difference-in-differences against LND-only channels."""
    m = daily.merge(exposure, on="date", how="inner")
    rows = []
    for cl, col in [("CLN", "CLN"), ("Eclair", "Eclair"), ("LDK", "LDK"), ("LND_only", "LND_only")]:
        num = m[f"force_organic_{cl}"]
        rate = 1000 * num / m[col]
        def mean(r):
            mask = (m["date"] >= r[0]) & (m["date"] <= r[1])
            return rate[mask].mean(), int(num[mask].sum())
        (b, nb), (w, nw) = mean(baseline), mean(window)
        rows.append({"cohort": cl, "baseline_rate": b, "window_rate": w, "ratio": w / b if b else np.nan,
                     "baseline_n": nb, "window_n": nw})
    r = pd.DataFrame(rows)
    lnd = r.loc[r["cohort"] == "LND_only", "ratio"].iloc[0]
    r["ratio_vs_LND_only"] = r["ratio"] / lnd
    return r


def spend_events_daily(s: pd.DataFrame) -> pd.DataFrame:
    """Daily counts of commitment-output resolutions; breach and HTLC classes are the interesting ones."""
    return s.groupby(["date", "spend_type"]).size().unstack(fill_value=0).reset_index()


def fee_summary(c: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Fees per close type: closing-tx fee + allocated sweep fees (force closes only), sats."""
    sweep = s.groupby("channel_id")["fee_alloc_sat"].sum()
    c = c.assign(sweep_fee_sat=c["channel_id"].map(sweep).fillna(0))
    c["total_fee_sat"] = c["fee_sat"].fillna(0) + c["sweep_fee_sat"]
    return c.groupby("close_type").agg(closes=("fee_sat", "size"), close_fee_median=("fee_sat", "median"),
                                       total_fee_median=("total_fee_sat", "median"),
                                       total_fee_mean=("total_fee_sat", "mean"),
                                       total_fee_btc=("total_fee_sat", lambda x: x.sum() / 1e8)).reset_index()
