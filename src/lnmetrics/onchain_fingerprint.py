"""
onchain_fingerprint.py — on-chain evidence for a node's Lightning implementation, from the wallet that
builds its channel funding transactions.

A funding tx is constructed by the *funder's* wallet. From same-client channels (both endpoints labeled the
same with high confidence; whoever funded, the wallet is known) two wallet families separate cleanly:

  LND   btcwallet: nLockTime 0, input nSequence 0x0 (≈ 82%), taproot inputs/change
  CORE  Core/BDK-style wallet — CLN's wallet, Eclair via bitcoind, BDK (e.g. LDK Node), Sparrow/Core PSBT
        funding: anti-fee-sniping nLockTime = height, nSequence 0xfffffffd (RBF). Read it as "not LND's
        wallet", never as "CLN".

What it measures is the *wallet*, not the daemon: LND operators who fund opens from an external wallet
(PSBT) show CORE (confirmed for several batch funders), and nodes that migrated LND → CLN can show LND-wallet
opens from before the switch. Feature bits therefore stay authoritative where they are decisive.
Held-out agreement with high-confidence feature-bit labels (5-fold, 2026-10-06, all 226k funding txs):
99.2–99.3% when decided on 25,234 nodes; 98% of nodes with ≥ 5 funding txs get a decision.

LDK and Eclair cannot be fingerprinted on-chain in general: LDK is a library whose funding txs are built by
each integrator's wallet (our same-client "LDK" sample is ~99% one operator, Block), and the Eclair sample is
essentially ACINQ. So the on-chain evidence answers "LND wallet or Core-style wallet?"; feature bits still
separate CLN from Eclair (client_fingerprint.py).

Model. Chain data doesn't say which endpoint funded a channel. For node X and each of its channels with a
fetched funding tx (signature f), the tx was built by X with probability θ, otherwise by the peer:
    P(f | X runs K) = θ·P(f | K) + (1 − θ)·P(f | peer)
with P(f | peer) = P(f | peer's label) for LND/CORE peers, else the network-wide distribution.
Evidence = log-likelihood ratio  LLR = Σ_channels log P(f | X=CORE) − log P(f | X=LND).
Batch opens (several funding outputs in one tx whose channels share one endpoint) identify the funder: that
endpoint gets the tx with θ = 1, the others get nothing from it.

Usage (research):
    from lnmetrics import onchain_fingerprint as of
    ev = of.node_evidence(labels)          # pub_key, n_txs, n_batch, llr, onchain_family
    cv = of.cross_validate(labels)         # held-out agreement with feature-bit labels
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .incident_metrics import PARQUET

FUNDING_PATH = PARQUET / "onchain" / "ln_funding_txs.parquet"
CHANNELS_PATH = PARQUET / "channels_unique.parquet"

FAMILIES = ["LND", "CORE"]
CLIENT_TO_FAMILY = {"LND": "LND", "CLN": "CORE", "Eclair": "CORE"}
THETA = 0.5            # prior share of a node's channels it funded itself
ALPHA = 1.0            # Laplace smoothing over signatures
MIN_SIG_COUNT = 50     # rarer signatures fold into "other"
LLR_DECISIVE = 4.6     # |LLR| ≥ ln(100): ≥ 100:1 odds
LLR_SUPPORT = 2.3      # |LLR| ≥ ln(10)


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
def _seq_class(s: str) -> str:
    v = set(json.loads(s))
    if len(v) != 1:
        return "mixed"
    x = v.pop()
    return x if x in ("0x0", "0xfffffffd", "0xffffffff", "0xfffffffe") else "other"


def funding_signatures() -> pd.DataFrame:
    """One row per fetched funding tx: txid, signature, number of funding outputs."""
    f = pd.read_parquet(FUNDING_PATH, columns=["txid", "locktime_kind", "input_sequences", "input_types",
                                               "n_funding_outputs", "confirm_height"])
    f["sig"] = (f["locktime_kind"] + "|" + f["input_sequences"].map(_seq_class) + "|"
                + np.where(f["input_types"].str.contains("taproot"), "tr", "notr"))
    common = f["sig"].value_counts()
    f.loc[~f["sig"].isin(common[common >= MIN_SIG_COUNT].index), "sig"] = "other"
    return f[["txid", "sig", "n_funding_outputs", "confirm_height"]]


def channel_txs(sigs: pd.DataFrame) -> pd.DataFrame:
    """Channels joined to their funding-tx signature; batch-open funder resolved where possible."""
    ch = pd.read_parquet(CHANNELS_PATH, columns=["chan_point", "node1_pub", "node2_pub"])
    ch = ch[ch["chan_point"].notna() & ~ch["chan_point"].str.startswith("0" * 64)].drop_duplicates("chan_point")
    ch["txid"] = ch["chan_point"].str.split(":").str[0]
    ch = ch.merge(sigs, on="txid", how="inner")
    # batch opens: the endpoint common to every channel funded by the tx is the funder
    multi = ch[ch.groupby("txid")["txid"].transform("size") > 1]
    funder = {}
    for txid, g in multi.groupby("txid"):
        common = set(g["node1_pub"]) | set(g["node2_pub"])
        for a, b in zip(g["node1_pub"], g["node2_pub"]):
            common &= {a, b}
        if len(common) == 1:
            funder[txid] = common.pop()
    ch["funder"] = ch["txid"].map(funder)
    return ch


# ---------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------
def signature_model(ch: pd.DataFrame, fam: Dict[str, str], exclude: Optional[set] = None) -> pd.DataFrame:
    """P(sig | family) from channels whose two endpoints share a family (optionally excluding nodes),
    plus the network-wide distribution ('ALL'). Rows: signatures; columns: LND, CORE, ALL."""
    f1, f2 = ch["node1_pub"].map(fam), ch["node2_pub"].map(fam)
    train = ch[(f1 == f2) & f1.notna()].assign(family=f1)
    if exclude:
        train = train[~train["node1_pub"].isin(exclude) & ~train["node2_pub"].isin(exclude)]
    train = train.drop_duplicates("txid")
    sigs = sorted(ch["sig"].unique())
    tab = pd.crosstab(train["sig"], train["family"]).reindex(index=sigs, columns=FAMILIES, fill_value=0)
    tab = (tab + ALPHA) / (tab + ALPHA).sum()
    allp = ch.drop_duplicates("txid")["sig"].value_counts().reindex(sigs, fill_value=0) + ALPHA
    tab["ALL"] = allp / allp.sum()
    return tab


def _evidence(ch: pd.DataFrame, model: pd.DataFrame, fam: Dict[str, str], nodes: Optional[set] = None) -> pd.DataFrame:
    """Per-node LLR (CORE vs LND) and counts, from the channel × signature table."""
    rows = []
    for side, other in (("node1_pub", "node2_pub"), ("node2_pub", "node1_pub")):
        d = ch[["txid", "sig", "funder", side, other]].rename(columns={side: "pub_key", other: "peer"})
        if nodes is not None:
            d = d[d["pub_key"].isin(nodes)]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    # batch opens where someone else is the funder carry no evidence about this node
    d = d[d["funder"].isna() | (d["funder"] == d["pub_key"])]
    theta = np.where(d["funder"] == d["pub_key"], 1.0, THETA)
    peer_fam = d["peer"].map(fam)
    p_lnd, p_core = model.loc[d["sig"], "LND"].to_numpy(), model.loc[d["sig"], "CORE"].to_numpy()
    p_peer = np.where(peer_fam == "LND", p_lnd, np.where(peer_fam == "CORE", p_core, model.loc[d["sig"], "ALL"].to_numpy()))
    d["llr"] = np.log(theta * p_core + (1 - theta) * p_peer) - np.log(theta * p_lnd + (1 - theta) * p_peer)
    d["is_batch_funder"] = d["funder"] == d["pub_key"]
    g = d.groupby("pub_key").agg(n_txs=("txid", "nunique"), n_batch=("is_batch_funder", "sum"), llr=("llr", "sum"))
    g["onchain_family"] = np.select([g["llr"] >= LLR_SUPPORT, g["llr"] <= -LLR_SUPPORT], ["CORE", "LND"], "unclear")
    g["onchain_strength"] = np.select([g["llr"].abs() >= LLR_DECISIVE, g["llr"].abs() >= LLR_SUPPORT], ["decisive", "supporting"], "weak")
    return g.reset_index()


def node_evidence(labels: pd.DataFrame, min_confidence=("high", "operator")) -> pd.DataFrame:
    """On-chain wallet-family evidence for every node with fetched funding txs. Signature model trained on
    channels between feature-bit-labeled nodes of `min_confidence`."""
    fam = _family_map(labels, min_confidence)
    ch = channel_txs(funding_signatures())
    return _evidence(ch, signature_model(ch, fam), fam)


def _family_map(labels: pd.DataFrame, min_confidence) -> Dict[str, str]:
    hi = labels[labels["confidence"].isin(min_confidence) & labels["client"].isin(CLIENT_TO_FAMILY)]
    return dict(zip(hi["pub_key"], hi["client"].map(CLIENT_TO_FAMILY)))


def cross_validate(labels: pd.DataFrame, folds: int = 5, seed: int = 42,
                   min_confidence=("high", "operator")) -> pd.DataFrame:
    """Held-out evaluation: nodes split into folds; the signature model for a fold is trained without any
    channel touching that fold's nodes, and those nodes' peers keep their labels. Returns one row per
    labeled node with on-chain evidence: true family (from feature bits) vs on-chain family."""
    fam = _family_map(labels, min_confidence)
    ch = channel_txs(funding_signatures())
    nodes = np.array(sorted(fam))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(nodes, rng.integers(0, folds, len(nodes))))
    out = []
    for k in range(folds):
        test = {n for n in nodes if fold_of[n] == k}
        model = signature_model(ch, fam, exclude=test)
        ev = _evidence(ch, model, fam, nodes=test)
        ev["true_family"] = ev["pub_key"].map(fam)
        ev["fold"] = k
        out.append(ev)
    return pd.concat(out, ignore_index=True)
