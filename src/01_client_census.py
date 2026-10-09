"""
Stage 1 (C1 / RQ1): Client concentration of the public Lightning Network, May 2023 – Oct 2026.

For the first gossip snapshot of every month, each public node with at least one channel is labeled with
src/lnmetrics/client_fingerprint.py (feature bits + modal CLTV; rules version recorded) and we report
client shares by node count, by channel endpoints and by node capacity (each channel counts for both
endpoints).

Only gossip days with quality status "ok" are used (incident_metrics.gossip_quality: wallet-locked error
dumps, partial May–Jun 2024 dumps and stale days excluded). Our gossip view changed in mid-2024 (vantage epoch
A → B: ~42k → ~46–50k channels); the figure marks the boundary and levels are only compared within an epoch.

Label uncertainty is reported, not hidden: for every client the share is given as a range from
"high-confidence labels only" (lower bound) to "all labels" (point estimate). The rules were derived from
2026 feature sets, so older snapshots carry more low-confidence and Unknown labels.

Outputs
  paper/figures/fig1_client_share.pdf          monthly share of nodes and of capacity (100% stacked)
  paper/tables/table1_client_distribution.tex  census on the latest snapshot (+ Gini of node capacity)
  data/client_census_monthly.json              the monthly series (for text and checks)

Usage:  python src/01_client_census.py
"""

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import CLIENT_PALETTE, DATA_CUTOFF, DATA_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402

from lnmetrics import incident_metrics as im  # noqa: E402
from lnmetrics.client_fingerprint import RULES_VERSION, classify_node  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

CLIENTS = ["LND", "CLN", "Eclair", "LDK", "Unknown"]
HIGH = {"high", "operator"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def monthly_dates() -> list:
    """First available gossip date of each month."""
    first = {}
    for d in im.gossip_dates():
        if d > DATA_CUTOFF:
            break
        first.setdefault(d[:6], d)
    return [first[k] for k in sorted(first)]


def snapshot(date: str) -> pd.DataFrame:
    """One row per node with ≥ 1 channel: client label, confidence, channels, node capacity (sats)."""
    pol = im.read_policies(date, ["channel_id", "capacity", "node_pub", "peer_pub", "time_lock_delta"])
    ch = pol.drop_duplicates("channel_id")
    ends = pd.concat([ch[["node_pub", "capacity"]].rename(columns={"node_pub": "pub_key"}),
                      ch[["peer_pub", "capacity"]].rename(columns={"peer_pub": "pub_key"})])
    per = ends.groupby("pub_key").agg(channels=("capacity", "size"), capacity=("capacity", "sum"))
    cltv = im._cltv_modes(pol[["node_pub", "time_lock_delta"]])
    nodes = im.read_nodes(date, ["pub_key", "alias", "color", "feature_bits"]).set_index("pub_key")
    rows = []
    for pk in per.index:
        if pk in nodes.index:
            n = nodes.loc[pk]
            bits = list(n["feature_bits"]) if n["feature_bits"] is not None else []
            color, alias = n["color"], n["alias"]
        else:
            bits, color, alias = [], None, None
        mode = cltv.get(pk)
        client, conf, _ = classify_node(bits, color, alias, pk, float(mode) if mode is not None else None)
        rows.append((pk, client, conf))
    lab = pd.DataFrame(rows, columns=["pub_key", "client", "confidence"]).set_index("pub_key")
    return per.join(lab)


def shares(snap: pd.DataFrame) -> dict:
    tot_n, tot_c, tot_e = len(snap), snap["capacity"].sum(), snap["channels"].sum()
    out = {}
    for c in CLIENTS:
        s = snap[snap["client"] == c]
        hi = s[s["confidence"].isin(HIGH)]
        out[c] = {"nodes": int(len(s)), "node_share": 100 * len(s) / tot_n,
                  "node_share_high": 100 * len(hi) / tot_n,
                  "endpoint_share": 100 * s["channels"].sum() / tot_e,
                  "cap_btc": s["capacity"].sum() / 1e8, "cap_share": 100 * s["capacity"].sum() / tot_c,
                  "cap_share_high": 100 * hi["capacity"].sum() / tot_c}
    return out


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def fig_share(series: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    for ax, key, title in [(axes[0], "node_share", "(a) Share of public nodes"),
                           (axes[1], "cap_share", "(b) Share of node capacity")]:
        w = series.pivot(index="date", columns="client", values=key)[CLIENTS]
        # one stack per vantage epoch, so the excluded mid-2024 months aren't drawn as a transition
        for part in (w[w.index < "2024-07-01"], w[w.index >= "2024-07-01"]):
            if len(part):
                ax.stackplot(part.index, *[part[c].to_numpy() for c in CLIENTS],
                             colors=[CLIENT_PALETTE[c] for c in CLIENTS], edgecolor="white", linewidth=0.6)
        ax.set_ylim(0, 100)
        ax.set_xlim(w.index.min(), w.index.max())
        ax.axvline(pd.Timestamp("2024-06-01"), color="#888888", lw=0.8, ls=(0, (2, 2)))
        ax.annotate("vantage\nchange", (pd.Timestamp("2024-07-01"), 50), xytext=(3, 0), textcoords="offset points",
                    fontsize=6, color="white", va="center")
        ax.set_title(title, loc="left")
        ax.set_ylabel("%")
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        # direct labels at the right edge, centred in each band (skip bands too thin to hold text)
        last = w.iloc[-1]
        cum = 0.0
        for c in CLIENTS:
            mid = cum + last[c] / 2
            if last[c] >= 2.5:
                ax.annotate(f"{c} {last[c]:.0f}%", (w.index[-1], mid), xytext=(3, 0), textcoords="offset points",
                            fontsize=6.5, va="center", color=CLIENT_PALETTE[c], annotation_clip=False)
            cum += last[c]
    handles = [matplotlib.patches.Patch(color=CLIENT_PALETTE[c]) for c in CLIENTS]
    fig.legend(handles, CLIENTS, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 0.95, 1))
    out = FIGURES_DIR / "fig1_client_share.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Wrote %s", out)


def table_census(date: str, snap: pd.DataFrame, sh: dict):
    rows = []
    for c in CLIENTS:
        s = snap[snap["client"] == c]
        v = sh[c]
        rows.append(
            f"{c} & {v['nodes']:,} & {v['node_share']:.1f} ({v['node_share_high']:.1f}) & {v['endpoint_share']:.1f} & "
            f"{v['cap_btc']:,.0f} & {v['cap_share']:.1f} ({v['cap_share_high']:.1f}) & "
            f"{(s['capacity'].mean() / 1e8 if len(s) else 0):.2f} & {gini(s['capacity'].to_numpy()):.2f} \\\\")
    tot = snap["capacity"].sum() / 1e8
    tex = r"""\begin{table*}[t]
\centering
\caption{Client implementations of public Lightning nodes with at least one channel, gossip snapshot """ + \
        f"{date[:4]}-{date[4:6]}-{date[6:]}" + r""" (rules """ + RULES_VERSION.replace("_", r"\_") + r"""). Shares in \%; in parentheses, the share
counting only high-confidence labels (lower bound). Node capacity counts each channel for both endpoints
(total """ + f"{tot:,.0f}" + r""" BTC). Gini: inequality of node capacity within the cohort.}
\label{tab:client-census}
\small
\begin{tabular}{lrrrrrrr}
\toprule
Client & Nodes & Node share & Endpoints & Capacity (BTC) & Cap.\ share & Mean (BTC) & Gini \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    out = TABLES_DIR / "table1_client_distribution.tex"
    out.write_text(tex, encoding="utf-8")
    log.info("Wrote %s", out)


def main():
    for d in (FIGURES_DIR, TABLES_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    dates = monthly_dates()
    log.info("Rules %s; %d monthly snapshots %s–%s", RULES_VERSION, len(dates), dates[0], dates[-1])
    recs, snap, sh = [], None, None
    for d in dates:
        snap = snapshot(d)
        sh = shares(snap)
        for c, v in sh.items():
            recs.append({"date": pd.Timestamp(d), "client": c, **v})
        log.info("%s: %d nodes; LND %.1f%% of nodes / %.1f%% of capacity; Unknown %.1f%% of nodes",
                 d, len(snap), sh["LND"]["node_share"], sh["LND"]["cap_share"], sh["Unknown"]["node_share"])
    series = pd.DataFrame(recs)
    fig_share(series)
    latest = max(d for d in im.gossip_dates() if d <= DATA_CUTOFF)
    snap_l = snapshot(latest)
    table_census(latest, snap_l, shares(snap_l))
    out = DATA_DIR / "client_census_monthly.json"
    series.assign(date=series["date"].dt.strftime("%Y-%m-%d")).round(3).to_json(out, orient="records")
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
