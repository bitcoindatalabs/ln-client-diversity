"""
Stage 2 (RQ2): How accurate is gossip-based client inference?

Two independent ground truths, neither used by the rules (shared/lightning/client_fingerprint.py):

  GT1  self-identifying aliases ("…[LND]", "…-CLN", "c-lightning", "eclair", "ldk-node", …), pooled over all
       monthly gossip snapshots (quality "ok"); each node evaluated on the latest snapshot where it has a channel
       and its alias still self-identifies. LND's default alias (pubkey prefix) is excluded — the rules use it.
  GT2  on-chain wallet family of the node's channel funding txs (shared/lightning/onchain_fingerprint.py):
       LND btcwallet vs Core/BDK-style. Measures the wallet, not the daemon, so it grades LND vs non-LND only.
       High-confidence labels are graded with held-out (5-fold) evidence; lower tiers were never used for
       training, so their evidence is out-of-sample by construction.

Also: a learned baseline on the raw feature-bit vector, trained on GT1 and compared to the rules on the same
held-out folds — k-nearest neighbours (Jaccard, k = 3) instead of the decision tree of Espinasa-Vilarrasa et al.,
to stay dependency-free (scikit-learn is not installed in this environment); and share bounds for the latest snapshot.

Outputs
  paper/tables/table2_classifier_eval.tex      GT1 precision/recall per client; GT2 agreement per rule tier
  paper/figures/fig2_classifier_validation.pdf GT1 confusion matrix + GT2 agreement by tier
  data/classifier_eval.json

Usage:  python src/02_classifier_eval.py
"""

import json
import logging
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import CLIENT_PALETTE, DATA_CUTOFF, DATA_DIR, FIGURES_DIR, RANDOM_SEED, REPO_ROOT, TABLES_DIR  # noqa: E402

sys.path.insert(0, str(REPO_ROOT.parent.parent / "python" / "automation"))
from shared.lightning import incident_metrics as im  # noqa: E402
from shared.lightning import onchain_fingerprint as of  # noqa: E402
from shared.lightning.client_fingerprint import KNOWN_OPERATOR_CLIENTS, RULES_VERSION, classify_node  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

CLIENTS = ["LND", "CLN", "Eclair", "LDK"]
ALIAS_PATTERNS = {
    "CLN": r"(?<![a-z])(?:cln|c-?lightning|core-?lightning|corelightning)(?![a-z])",
    "Eclair": r"(?<![a-z])eclair",          # word start: excludes pastry names (framboiseclair, …)
    "LDK": r"(?<![a-z])ldk(?![a-z])|ldk[-_ ]?(?:node|server)",
    "LND": r"(?<![a-z])lnd(?![a-z])",
}

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def tier(reason: str) -> str:
    """Rule tier from the reason string (stable across label confidence)."""
    r = reason.split("; on-chain")[0].split("; conflicts")[0]   # keep "no features; …" intact
    if r.startswith("operator-attributed"):
        return "operator"
    if "lnd-only feature bit" in r:
        return "LND-only bits (2023/31)"
    if "eclair feature bits" in r:
        return "Eclair bits (37, 29+61)"
    if "cln feature set" in r:
        return "CLN set + keysend"
    if "without keysend" in r:
        return "CLN-like set, no keysend → Eclair"
    if "ldk-like" in r:
        return "LDK-like set"
    if "lnd default color/alias" in r:
        return "LND default colour/alias" + (" (no features)" if r.startswith("no features") else "")
    if "cltv default" in r:
        return "CLTV default only" + (" (no features)" if r.startswith("no features") else "")
    return "none"


# ---------------------------------------------------------------------------------------
# GT1: alias anchors
# ---------------------------------------------------------------------------------------
def alias_anchors() -> pd.DataFrame:
    """One row per self-identifying node: truth, rule prediction at its latest snapshot with channels, features."""
    first = {}
    for d in im.gossip_dates():
        if d > DATA_CUTOFF:
            break
        first.setdefault(d[:6], d)
    rows = {}
    for d in [first[k] for k in sorted(first)]:
        nodes = im.read_nodes(d, ["pub_key", "alias", "color", "feature_bits"])
        a = nodes["alias"].fillna("").str.lower()
        truth = pd.Series(None, index=nodes.index, dtype=object)
        n_hits = pd.Series(0, index=nodes.index)
        for c, pat in ALIAS_PATTERNS.items():
            hit = a.str.contains(pat, regex=True)
            truth[hit] = c
            n_hits += hit
        sel = nodes[(n_hits == 1) & truth.notna()].assign(truth=truth[(n_hits == 1) & truth.notna()])
        sel = sel[~sel["pub_key"].isin(KNOWN_OPERATOR_CLIENTS)]
        if sel.empty:
            continue
        pol = im.read_policies(d, ["node_pub", "peer_pub", "time_lock_delta"])
        has = set(pol["node_pub"]) | set(pol["peer_pub"])
        cltv = im._cltv_modes(pol[["node_pub", "time_lock_delta"]])
        for r in sel[sel["pub_key"].isin(has)].itertuples(index=False):
            bits = [int(b) for b in r.feature_bits] if r.feature_bits is not None else []
            m = cltv.get(r.pub_key)
            client, conf, reason = classify_node(bits, r.color, r.alias, r.pub_key, float(m) if m is not None else None)
            rows[r.pub_key] = {"pub_key": r.pub_key, "alias": r.alias, "truth": r.truth, "date": d, "pred": client,
                               "confidence": conf, "tier": tier(reason), "bits": bits,
                               "cltv": int(m) if m is not None else -1}
    df = pd.DataFrame(rows.values())
    log.info("GT1 anchors: %s", df["truth"].value_counts().to_dict())
    return df


def prf(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    out = []
    for c in CLIENTS:
        tp = ((df["truth"] == c) & (df[pred_col] == c)).sum()
        fp = ((df["truth"] != c) & (df[pred_col] == c)).sum()
        fn = ((df["truth"] == c) & (df[pred_col] != c)).sum()
        unk = ((df["truth"] == c) & (df[pred_col] == "Unknown")).sum()
        out.append({"client": c, "n": int((df["truth"] == c).sum()),
                    "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan,
                    "unknown": int(unk)})
    return pd.DataFrame(out)


def _folds(y: np.ndarray, k: int, rng) -> list:
    """Stratified k-fold indices."""
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    return [np.array(f) for f in folds]


def _macro_f1(truth, pred, labels) -> float:
    f = []
    for c in labels:
        tp = sum((t == c) & (p == c) for t, p in zip(truth, pred))
        fp = sum((t != c) & (p == c) for t, p in zip(truth, pred))
        fn = sum((t == c) & (p != c) for t, p in zip(truth, pred))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


def tree_baseline(anchors: pd.DataFrame, seed: int = RANDOM_SEED, k_nn: int = 3) -> dict:
    """Learned baseline: k-NN (Jaccard) on the feature-bit vector + CLTV-default one-hot, held-out stratified folds
    (5 repeats); the rules are scored on the same held-out nodes. Anchors without announced features excluded
    (both methods are blind there)."""
    df = anchors[anchors["bits"].map(len) > 0].reset_index(drop=True)
    vocab = sorted({b for bits in df["bits"] for b in bits})
    col = {b: i for i, b in enumerate(vocab)}
    X = np.zeros((len(df), len(vocab) + 6), dtype=bool)
    for i, bits in enumerate(df["bits"]):
        for b in bits:
            X[i, col[b]] = True
    for j, v in enumerate([34, 40, 72, 80, 144]):
        X[:, len(vocab) + j] = (df["cltv"] == v).to_numpy()
    X[:, -1] = (~df["cltv"].isin([34, 40, 72, 80, 144])).to_numpy()
    y = df["truth"].to_numpy()
    k = max(2, int(min(5, pd.Series(y).value_counts().min())))
    rng = np.random.default_rng(seed)
    Xi = X.astype(np.int32)
    inter = Xi @ Xi.T
    sizes = Xi.sum(axis=1)
    jac = inter / np.maximum(sizes[:, None] + sizes[None, :] - inter, 1)
    knn_pred, rule_pred, truth = [], [], []
    for _ in range(5):
        for te in _folds(y, k, rng):
            tr = np.setdiff1d(np.arange(len(y)), te)
            for i in te:
                nn = tr[np.argsort(-jac[i, tr], kind="stable")[:k_nn]]
                knn_pred.append(pd.Series(y[nn]).value_counts().index[0])
                rule_pred.append(df["pred"].iat[i])
                truth.append(y[i])
    labels = sorted(set(y))
    return {"method": f"kNN (Jaccard, k={k_nn})", "n": len(df), "folds": k, "repeats": 5,
            "tree_macro_f1": _macro_f1(truth, knn_pred, labels), "rules_macro_f1": _macro_f1(truth, rule_pred, labels),
            "tree_acc": float(np.mean(np.array(knn_pred) == np.array(truth))),
            "rules_acc": float(np.mean(np.array(rule_pred) == np.array(truth)))}


# ---------------------------------------------------------------------------------------
# GT2: on-chain wallet family by rule tier
# ---------------------------------------------------------------------------------------
def onchain_by_tier(date: str) -> pd.DataFrame:
    nodes = im.read_nodes(date, ["pub_key", "alias", "color", "feature_bits"])
    cltv = im._cltv_modes(im.read_policies(date, ["node_pub", "time_lock_delta"]))
    rows = []
    for r in nodes.itertuples(index=False):
        bits = list(r.feature_bits) if r.feature_bits is not None else []
        m = cltv.get(r.pub_key)
        client, conf, reason = classify_node(bits, r.color, r.alias, r.pub_key, float(m) if m is not None else None)
        rows.append({"pub_key": r.pub_key, "client": client, "confidence": conf, "tier": tier(reason)})
    lab = pd.DataFrame(rows)
    high = lab["confidence"].isin(["high", "operator"])
    labels_for_model = lab.rename(columns={})[["pub_key", "client", "confidence"]].assign(alias=None)
    cv = of.cross_validate(labels_for_model)                                  # held-out evidence for high tiers
    ev = of.node_evidence(labels_for_model)                                   # model trained on high tiers only
    e = pd.concat([cv[cv["pub_key"].isin(lab.loc[high, "pub_key"])][["pub_key", "onchain_family", "onchain_strength"]],
                   ev[ev["pub_key"].isin(lab.loc[~high, "pub_key"])][["pub_key", "onchain_family", "onchain_strength"]]])
    d = lab.merge(e, on="pub_key", how="inner")
    d = d[d["onchain_strength"] == "decisive"]
    d["expected"] = d["client"].map({"LND": "LND", "CLN": "CORE", "Eclair": "CORE", "LDK": "CORE"})
    g = d.groupby(["tier", "client"]).agg(n=("pub_key", "size"), lnd_wallet=("onchain_family", lambda s: (s == "LND").mean()))
    g = g.reset_index()
    g["agree"] = np.where(g["client"] == "LND", g["lnd_wallet"], 1 - g["lnd_wallet"])
    g.loc[g["client"] == "Unknown", "agree"] = np.nan
    return g.sort_values(["client", "n"], ascending=[True, False])


def share_bounds(date: str) -> pd.DataFrame:
    snap_mod = __import__("01_client_census") if False else None  # noqa: F841 (census logic kept in 01)
    pol = im.read_policies(date, ["channel_id", "capacity", "node_pub", "peer_pub", "time_lock_delta"])
    ch = pol.drop_duplicates("channel_id")
    ends = pd.concat([ch[["node_pub", "capacity"]].rename(columns={"node_pub": "pub_key"}),
                      ch[["peer_pub", "capacity"]].rename(columns={"peer_pub": "pub_key"})])
    per = ends.groupby("pub_key")["capacity"].sum()
    nodes = im.read_nodes(date, ["pub_key", "alias", "color", "feature_bits"]).set_index("pub_key")
    cltv = im._cltv_modes(pol[["node_pub", "time_lock_delta"]])
    rows = []
    for pk, cap in per.items():
        n = nodes.loc[pk] if pk in nodes.index else None
        bits = list(n["feature_bits"]) if n is not None and n["feature_bits"] is not None else []
        m = cltv.get(pk)
        c, conf, _ = classify_node(bits, n["color"] if n is not None else None, n["alias"] if n is not None else None,
                                   pk, float(m) if m is not None else None)
        rows.append((c, conf, cap))
    df = pd.DataFrame(rows, columns=["client", "conf", "cap"])
    out = []
    for c in CLIENTS + ["Unknown"]:
        s = df[df["client"] == c]
        hi = s[s["conf"].isin(["high", "operator"])]
        out.append({"client": c, "node_share": 100 * len(s) / len(df), "node_share_high": 100 * len(hi) / len(df),
                    "cap_share": 100 * s["cap"].sum() / df["cap"].sum(),
                    "cap_share_high": 100 * hi["cap"].sum() / df["cap"].sum()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------
def fig_validation(anchors: pd.DataFrame, tiers: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1, 1.5]})
    cols = CLIENTS + ["Unknown"]
    m = pd.crosstab(anchors["truth"], anchors["pred"]).reindex(index=CLIENTS, columns=cols, fill_value=0)
    ax = axes[0]
    ax.imshow(m.div(m.sum(axis=1).replace(0, 1), axis=0).to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            share = v / max(m.iloc[i].sum(), 1)
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="white" if share > 0.6 else "#222222")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(CLIENTS)), CLIENTS)
    ax.set_xlabel("rule prediction")
    ax.set_ylabel("self-identified (alias)")
    ax.set_title("(a) Alias anchors", loc="left")

    ax = axes[1]
    t = tiers[(tiers["client"] != "Unknown") & (tiers["n"] >= 10)].copy()
    t["label"] = t["client"] + ": " + t["tier"]
    t = t.sort_values("agree")
    y = np.arange(len(t))
    ax.barh(y, 100 * t["agree"], color=[CLIENT_PALETTE.get(c, "#888888") for c in t["client"]], height=0.6)
    for yi, (a, n) in enumerate(zip(t["agree"], t["n"])):
        ax.text(100 * a + 1, yi, f"{100 * a:.0f}% (n={n:,})", va="center", fontsize=6.5)
    ax.set_yticks(y, t["label"], fontsize=6.5)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% agreeing with on-chain wallet family")
    ax.set_title("(b) On-chain check by rule tier", loc="left")
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_classifier_validation.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Wrote %s", out)


def table_eval(p: pd.DataFrame, tb: dict, tiers: pd.DataFrame, n_anchor: int):
    r1 = "\n".join(f"{r.client} & {r.n} & {('--' if np.isnan(r.precision) else f'{100 * r.precision:.1f}')} & "
                   f"{('--' if np.isnan(r.recall) else f'{100 * r.recall:.1f}')} & {r.unknown} \\\\" for r in p.itertuples())
    t = tiers[(tiers["client"] != "Unknown") & (tiers["n"] >= 10)]
    arrow = r"$\rightarrow$"  # backslash inside an f-string expression needs Python 3.12+
    r2 = "\n".join(f"{r.client} & {r.tier.replace('→', arrow)} & {r.n:,} & {100 * r.agree:.1f} \\\\"
                   for r in t.itertuples())
    tex = r"""\begin{table}[t]
\centering
\caption{Validation of client inference (rules """ + RULES_VERSION.replace("_", r"\_") + r"""). (a) """ + str(n_anchor) + r""" nodes whose
alias names their implementation (pooled over monthly snapshots, evaluated on the latest snapshot where they have a
channel); ``Unknown'' counts nodes the rules leave unlabeled (mostly no announced features). A learned baseline
($k$-NN, Jaccard, on the feature-bit vector) vs.\ rules on the same held-out folds: macro-F1 """ + f"{tb['tree_macro_f1']:.2f} vs.\\ {tb['rules_macro_f1']:.2f}" + \
        r""" ($n=""" + str(tb["n"]) + r"""$). (b) Share of nodes in each rule tier whose channel-funding wallet agrees with the
label (LND btcwallet for LND, Core/BDK-style otherwise), nodes with decisive on-chain evidence; high tiers graded with
held-out evidence.}
\label{tab:classifier-eval}
\small
\begin{tabular}{lrrrr}
\toprule
\multicolumn{5}{l}{(a) Alias anchors} \\
Client & $n$ & Precision (\%) & Recall (\%) & Unknown \\
\midrule
""" + r1 + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{llrr}
\toprule
\multicolumn{4}{l}{(b) On-chain wallet family} \\
Label & Rule tier & $n$ & Agree (\%) \\
\midrule
""" + r2 + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out = TABLES_DIR / "table2_classifier_eval.tex"
    out.write_text(tex, encoding="utf-8")
    log.info("Wrote %s", out)


def main():
    for d in (FIGURES_DIR, TABLES_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    latest = max(d for d in im.gossip_dates() if d <= DATA_CUTOFF)
    anchors = alias_anchors()
    p = prf(anchors)
    log.info("GT1 precision/recall:\n%s", p.round(3).to_string(index=False))
    tb = tree_baseline(anchors)
    log.info("Decision tree vs rules (held-out): %s", {k: round(v, 3) if isinstance(v, float) else v for k, v in tb.items()})
    tiers = onchain_by_tier(latest)
    log.info("GT2 by tier (%s):\n%s", latest, tiers.round(3).to_string(index=False))
    bounds = share_bounds(latest)
    log.info("Share bounds (%s):\n%s", latest, bounds.round(1).to_string(index=False))
    fig_validation(anchors, tiers)
    table_eval(p, tb, tiers, len(anchors))
    mism = anchors[(anchors["pred"] != anchors["truth"]) & (anchors["pred"] != "Unknown")]
    out = {"rules": RULES_VERSION, "latest": latest, "anchors": anchors["truth"].value_counts().to_dict(),
           "gt1": p.to_dict(orient="records"), "tree_baseline": tb, "gt2_tiers": tiers.to_dict(orient="records"),
           "share_bounds": bounds.to_dict(orient="records"),
           "gt1_misclassified": mism[["alias", "truth", "pred", "confidence", "tier", "date"]].to_dict(orient="records")}
    (DATA_DIR / "classifier_eval.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    log.info("GT1 misclassified (excluding Unknown):\n%s", mism[["alias", "truth", "pred", "tier"]].to_string(index=False))


if __name__ == "__main__":
    main()
