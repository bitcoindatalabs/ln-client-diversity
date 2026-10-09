"""
Stage 4 (C3 / RQ4): Exposure-aware resilience, calibrated by the observed CLN outage.

Graph: the routable core of the public graph on a pre-incident snapshot (default 2026-08-25, the day before the
CLN shutdown call). A channel is usable only if *both* directions are announced and enabled — ~73% of announced
channels; nodes whose channels are all disabled on one side (mostly long-offline leaves) drop out; it can carry a payment of `a` sats if its capacity ≥ a (optimistic,
"full") or ≥ 2a ("split": balance assumed 50/50). Payments need a path of usable channels, so the measure is
connectivity of the thresholded graph — an upper bound on what routing achieves (no fees, no in-flight HTLCs).

Metrics
  R_all(a) reachable-pair share of the *pre-failure* network: Σ_i n_i(n_i − 1) / (N₀(N₀ − 1)); pairs involving a
           failed node count as lost (main measure)
  R(a)   same among nodes still online (service to survivors; can rise when weakly connected nodes drop out)
  R_c(a) same, restricted to pairs of client c's online nodes (minority isolation)
  LCC capacity share: capacity of channels inside the largest component / capacity of all usable channels

Scenarios
  observed      nodes of the 2026-08-25 core with no usable channel left on 2026-08-27, removed from the Aug-25
                graph — the outage's measured footprint (all clients, and CLN only)
  calibrated    a random p̂ of CLN nodes fails (p̂ from 03_observed_outage.py), 50 Monte Carlo runs
  curves        for each client c and p ∈ {0, .05, …, 1}: random fraction p of c fails (50 MC); x-axis is the
                share of network node capacity removed, so cohorts of different size are comparable
  baselines     the same number of nodes removed uniformly at random / by degree / by capacity
  dividend      equal failed capacity drawn (i) within one client vs (ii) across all clients
  dependence    within-client reachability with and without LND nodes; client × client capacity matrix

Outputs
  paper/figures/fig6_resilience_curves.pdf      R(a) vs capacity removed, per client + baselines + observed
  paper/figures/fig7_client_capacity_matrix.pdf share of each client's channel capacity by peer client
  paper/tables/table6_scenarios.tex             scenario comparison
  data/resilience_results.json

Usage:  python src/04_resilience.py [--date 20260801] [--mc 50]
"""

import argparse
import json
import warnings
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from config import CLIENT_PALETTE, DATA_CUTOFF, DATA_DIR, FIGURES_DIR, RANDOM_SEED, TABLES_DIR  # noqa: E402

from lnmetrics import incident_metrics as im  # noqa: E402
from lnmetrics.client_fingerprint import RULES_VERSION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("fontTools").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean over cohorts with < 2 online nodes
log = logging.getLogger(__name__)

CLIENTS = ["LND", "CLN", "Eclair", "LDK"]
AMOUNTS = [10_000, 100_000, 1_000_000]
P_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E8E8E8",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------------------
class Graph:
    """Undirected channel graph of one gossip snapshot (channels usable in both directions)."""

    def __init__(self, date: str, labels: pd.DataFrame):
        p = im.read_policies(date, ["channel_id", "capacity", "node_pub", "peer_pub", "disabled"])
        both = p.groupby("channel_id").agg(cap=("capacity", "first"), a=("node_pub", "min"), b=("node_pub", "max"),
                                           dirs=("disabled", "size"), dis=("disabled", "sum"))
        ch = both[(both["dirs"] == 2) & (both["dis"] == 0) & (both["a"] != both["b"])]
        self.date = date
        nodes = pd.Index(sorted(set(ch["a"]) | set(ch["b"])))
        self.pub = nodes.to_numpy()
        self.n = len(nodes)
        self.u = nodes.get_indexer(ch["a"])
        self.v = nodes.get_indexer(ch["b"])
        self.cap = ch["cap"].to_numpy(dtype=float)
        lab = im.label_map(labels)
        self.client = np.array([lab.get(k, "Unknown") for k in self.pub])
        self.node_cap = np.bincount(self.u, self.cap, self.n) + np.bincount(self.v, self.cap, self.n)
        self.degree = np.bincount(self.u, minlength=self.n) + np.bincount(self.v, minlength=self.n)
        log.info("Graph %s: %d nodes, %d usable channels, %.0f BTC", date, self.n, len(self.cap), self.cap.sum() / 1e8)

    def measure(self, failed: np.ndarray, amount: int, split: bool = False) -> dict:
        """R(a), per-client R_c(a), LCC capacity share, with `failed` (bool mask) nodes removed."""
        need = 2 * amount if split else amount
        alive = ~failed
        keep = (self.cap >= need) & alive[self.u] & alive[self.v]
        g = coo_matrix((np.ones(keep.sum()), (self.u[keep], self.v[keep])), shape=(self.n, self.n))
        _, comp = connected_components(g, directed=False)
        comp_alive = comp[alive]
        sizes = np.bincount(comp_alive)
        n_alive = alive.sum()
        pairs = float((sizes * (sizes - 1)).sum())
        out = {"R": pairs / (n_alive * (n_alive - 1)) if n_alive > 1 else 0.0,
               "R_all": pairs / (self.n * (self.n - 1))}
        for c in CLIENTS:
            m = alive & (self.client == c)
            k = np.bincount(comp[m])
            K, K0 = m.sum(), (self.client == c).sum()
            out[f"R_{c}"] = float((k * (k - 1)).sum() / (K * (K - 1))) if K > 1 else float("nan")
            out[f"R_{c}_all"] = float((k * (k - 1)).sum() / (K0 * (K0 - 1))) if K0 > 1 else float("nan")
        usable_cap = self.cap[(self.cap >= need)].sum()
        lcc = np.argmax(sizes) if len(sizes) else -1
        in_lcc = keep & (comp[self.u] == lcc)
        out["lcc_cap"] = float(self.cap[in_lcc].sum() / usable_cap) if usable_cap else 0.0
        return out

    def cap_share(self, failed: np.ndarray) -> float:
        return float(self.node_cap[failed].sum() / self.node_cap.sum())


# ---------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------
def mc(G: Graph, draw, runs: int, rng, amounts=AMOUNTS, split=False) -> dict:
    """Average measures over `runs` random failure draws; draw(rng) → bool mask."""
    acc, caps = {}, []
    for _ in range(runs):
        f = draw(rng)
        caps.append(G.cap_share(f))
        for a in amounts:
            for k, v in G.measure(f, a, split).items():
                acc.setdefault((a, k), []).append(v)
    res = {"cap_removed": float(np.mean(caps)), "n_failed": int(draw(rng).sum())}
    for (a, k), vals in acc.items():
        res[f"{k}@{a}"] = float(np.nanmean(vals))
        res[f"{k}@{a}_sd"] = float(np.nanstd(vals))
    return res


def within_client(G: Graph, c: str, p: float):
    idx = np.flatnonzero(G.client == c)
    k = int(round(p * len(idx)))

    def draw(rng):
        f = np.zeros(G.n, bool)
        f[rng.choice(idx, k, replace=False)] = True
        return f
    return draw


def random_nodes(G: Graph, k: int):
    def draw(rng):
        f = np.zeros(G.n, bool)
        f[rng.choice(G.n, k, replace=False)] = True
        return f
    return draw


def top_nodes(G: Graph, k: int, by: np.ndarray):
    order = np.argsort(-by, kind="stable")[:k]
    f = np.zeros(G.n, bool)
    f[order] = True
    return lambda rng: f


def capacity_budget(G: Graph, pool: np.ndarray, q: float):
    """Random nodes from `pool` until failed node capacity reaches q of the network total."""
    target = q * G.node_cap.sum()

    def draw(rng):
        order = rng.permutation(pool)
        cum = np.cumsum(G.node_cap[order])
        f = np.zeros(G.n, bool)
        f[order[: int(np.searchsorted(cum, target)) + 1]] = True
        return f
    return draw


def capacity_matrix(G: Graph) -> pd.DataFrame:
    """Share of each client's channel capacity that goes to each peer client (rows sum to 1)."""
    a, b = G.client[G.u], G.client[G.v]
    df = pd.DataFrame({"x": np.concatenate([a, b]), "y": np.concatenate([b, a]), "cap": np.concatenate([G.cap, G.cap])})
    m = df.pivot_table(index="x", columns="y", values="cap", aggfunc="sum", fill_value=0)
    order = CLIENTS + ["Unknown"]
    m = m.reindex(index=order, columns=order, fill_value=0)
    return m.div(m.sum(axis=1), axis=0)


# ---------------------------------------------------------------------------------------
# Figures & tables
# ---------------------------------------------------------------------------------------
def fig_curves(curves: pd.DataFrame, base: pd.DataFrame, calib: dict, observed: dict, amounts=(100_000, 1_000_000)):
    fig, axes = plt.subplots(1, len(amounts), figsize=(7.0, 2.7), sharey=True)
    for ax, a in zip(axes, amounts):
        key = f"R_all@{a}"
        for c in CLIENTS:
            s = curves[curves["client"] == c].sort_values("cap_removed")
            ax.plot(100 * s["cap_removed"], 100 * s[key], color=CLIENT_PALETTE[c], lw=1.3, marker="o", ms=2.2)
            last = s.iloc[-1]
            ax.annotate(c, (100 * last["cap_removed"], 100 * last[key]), xytext=(3, 0), textcoords="offset points",
                        fontsize=6.5, color=CLIENT_PALETTE[c], va="center")
        for name, ls in [("random", (0, (1, 1.5))), ("top-capacity", (0, (4, 2)))]:
            s = base[base["baseline"] == name].sort_values("cap_removed")
            ax.plot(100 * s["cap_removed"], 100 * s[key], color="#555555", lw=1.0, ls=ls, label=f"{name} nodes")
        ax.scatter([100 * calib["cap_removed"]], [100 * calib[key]], s=28, color=CLIENT_PALETTE["CLN"],
                   edgecolor="black", lw=0.6, zorder=5, label="calibrated CLN outage")
        if observed:
            ax.scatter([100 * observed["cap_removed"]], [100 * observed[key]], s=30, marker="D", color="white",
                       edgecolor=CLIENT_PALETTE["CLN"], lw=1.2, zorder=5, label="observed (Aug 27)")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, 100)
        ax.set_title(f"Payment of {a:,} sats", loc="left")
        ax.set_xlabel("% of network node capacity failed")
    axes[0].set_ylabel("% of node pairs still reachable")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = FIGURES_DIR / "fig6_resilience_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Wrote %s", out)


def fig_matrix(m: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    im_ = ax.imshow(100 * m.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(m.index)), m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = 100 * m.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else "#222222")
    ax.set_xlabel("peer's client")
    ax.set_ylabel("node's client")
    ax.set_title("Channel capacity by peer client (%)", loc="left", fontsize=8.5)
    fig.colorbar(im_, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIGURES_DIR / "fig7_client_capacity_matrix.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Wrote %s", out)


def table_scenarios(rows: list, a: int = 100_000):
    lines = []
    for name, r in rows:
        cells = [name, f"{r['n_failed']:,}", f"{100 * r['cap_removed']:.1f}", f"{100 * r[f'R_all@{a}']:.1f}",
                 f"{100 * r[f'R@{a}']:.1f}"] + \
                [("--" if np.isnan(r.get(f"R_{c}@{a}", np.nan)) else f"{100 * r[f'R_{c}@{a}']:.1f}") for c in ["CLN", "Eclair", "LDK"]] + \
                [f"{100 * r[f'lcc_cap@{a}']:.1f}"]
        lines.append(" & ".join(cells) + r" \\")
    g = rows[0][1].get("graph_date", "")
    tex = r"""\begin{table*}[t]
\centering
\caption{Failure scenarios on the """ + f"{g[:4]}-{g[4:6]}-{g[6:]}" + r""" public graph (channels enabled in both
directions), payment size """ + f"{a:,}" + r""" sats, optimistic liquidity. $R_{\mathrm{all}}$: \% of pre-failure node pairs still connected (pairs with a failed
node count as lost); $R$: \% of online node pairs connected by a path of
channels that can carry the payment; $R_c$: same as $R$, for pairs of surviving nodes of client $c$; LCC cap.: \% of usable capacity in the largest
component. Random scenarios: mean of the Monte Carlo runs.}
\label{tab:scenarios}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
Scenario & Failed & Cap.\ (\%) & $R_{\mathrm{all}}$ & $R$ & $R_{\mathrm{CLN}}$ & $R_{\mathrm{Eclair}}$ & $R_{\mathrm{LDK}}$ & LCC cap. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    out = TABLES_DIR / "table6_scenarios.tex"
    out.write_text(tex, encoding="utf-8")
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260825", help="pre-incident gossip snapshot (day before the CLN call)")
    ap.add_argument("--mc", type=int, default=50)
    args = ap.parse_args()
    for d in (FIGURES_DIR, TABLES_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    labels = im.node_labels()
    cal = json.loads((DATA_DIR / "observed_outage_calibration.json").read_text(encoding="utf-8"))
    log.info("Rules %s; calibration p̂ = %.3f of CLN nodes", RULES_VERSION, cal["p_hat_nodes"])

    G = Graph(args.date, labels)
    none = np.zeros(G.n, bool)
    intact = {"n_failed": 0, "cap_removed": 0.0, "graph_date": args.date}
    for a in AMOUNTS:
        intact.update({f"{k}@{a}": v for k, v in G.measure(none, a).items()})
        intact.update({f"{k}@{a}_split": v for k, v in G.measure(none, a, split=True).items()})

    # observed: measured graphs around the outage, each against itself (no simulation)
    observed = {}
    for d in ("20260825", "20260827"):
        Gd = Graph(d, labels)
        observed[d] = {"n_nodes_usable": Gd.n, **{f"{k}@{a}": v for a in AMOUNTS for k, v in Gd.measure(np.zeros(Gd.n, bool), a).items()}}
    # express the observed outage on the baseline graph: nodes that lost every usable channel on Aug 27
    G27 = Graph("20260827", labels) if args.date != "20260827" else G
    gone = ~np.isin(G.pub, G27.pub)
    obs_on_base = {"n_failed": int(gone.sum()), "cap_removed": G.cap_share(gone),
                   **{f"{k}@{a}": v for a in AMOUNTS for k, v in G.measure(gone, a).items()}}
    obs_cln = {"n_failed": int((gone & (G.client == "CLN")).sum()), "cap_removed": G.cap_share(gone & (G.client == "CLN")),
               **{f"{k}@{a}": v for a in AMOUNTS for k, v in G.measure(gone & (G.client == "CLN"), a).items()}}
    log.info("Observed: %d nodes lost all usable channels by Aug 27 (%d CLN)", obs_on_base["n_failed"], obs_cln["n_failed"])

    # placebo: the same two-day footprint measure on ordinary days before the call (normal churn)
    placebo = []
    for d0, d2 in [("20260804", "20260806"), ("20260808", "20260810"), ("20260811", "20260813"),
                   ("20260815", "20260817"), ("20260818", "20260820"), ("20260821", "20260823")]:
        try:
            G0, G2 = Graph(d0, labels), Graph(d2, labels)
        except FileNotFoundError:
            continue
        g2 = ~np.isin(G0.pub, G2.pub)
        m = G0.measure(np.zeros(G0.n, bool), 100_000)
        f = G0.measure(g2, 100_000)
        placebo.append({"from": d0, "to": d2, "n_failed": int(g2.sum()), "n_failed_cln": int((g2 & (G0.client == "CLN")).sum()),
                        "dR_all": f["R_all"] - m["R_all"], "dR_CLN_all": f["R_CLN_all"] - m["R_CLN_all"]})
    placebo = pd.DataFrame(placebo)
    m25 = G.measure(np.zeros(G.n, bool), 100_000)
    obs_d = {"dR_all": obs_on_base["R_all@100000"] - m25["R_all"], "dR_CLN_all": obs_on_base["R_CLN_all@100000"] - m25["R_CLN_all"]}
    log.info("Placebo two-day windows (normal churn) @100k:\n%s", placebo.round(4).to_string(index=False))
    log.info("Outage window Aug 25→27: lost %d nodes (%d CLN), dR_all %+.4f, dR_CLN_all %+.4f",
             obs_on_base["n_failed"], obs_cln["n_failed"], obs_d["dR_all"], obs_d["dR_CLN_all"])

    # calibrated CLN outage
    calib = mc(G, within_client(G, "CLN", cal["p_hat_nodes"]), args.mc, rng)
    k = calib["n_failed"]

    # per-client curves
    rows = []
    for c in CLIENTS:
        for p in P_GRID:
            r = mc(G, within_client(G, c, p), args.mc if 0 < p < 1 else 1, rng)
            rows.append({"client": c, "p": p, **r})
        log.info("curve %s done", c)
    curves = pd.DataFrame(rows)

    # baselines across capacity budgets matching the curves' range
    base_rows = []
    for k_b in sorted({int(x) for x in np.unique(np.geomspace(5, G.n * 0.9, 18).astype(int))}):
        base_rows.append({"baseline": "random", **mc(G, random_nodes(G, k_b), args.mc, rng)})
        base_rows.append({"baseline": "top-capacity", **mc(G, top_nodes(G, k_b, G.node_cap), 1, rng)})
        base_rows.append({"baseline": "top-degree", **mc(G, top_nodes(G, k_b, G.degree), 1, rng)})
    base = pd.DataFrame(base_rows)

    # diversity dividend: same failed capacity, within one client vs across all clients
    dividend = []
    for q in (0.01, 0.02, 0.035, 0.05, 0.10, 0.20):
        across = mc(G, capacity_budget(G, np.arange(G.n), q), args.mc, rng)
        for c in CLIENTS:
            pool = np.flatnonzero(G.client == c)
            if G.node_cap[pool].sum() < q * G.node_cap.sum():
                continue
            within = mc(G, capacity_budget(G, pool, q), args.mc, rng)
            dividend.append({"q": q, "client": c, **{f"within_{x}": within[x] for x in within},
                             **{f"across_{x}": across[x] for x in across}})
    dividend = pd.DataFrame(dividend)

    # dependence on LND
    no_lnd = G.client == "LND"
    dep = {f"{k}@{a}": v for a in AMOUNTS for k, v in G.measure(no_lnd, a).items()}
    matrix = capacity_matrix(G)

    fig_curves(curves, base, calib, obs_cln)
    fig_matrix(matrix)
    same_k = [("Intact", intact),
              ("Observed: lost all usable channels, Aug 27", obs_on_base),
              ("  of which CLN nodes only", obs_cln),
              (f"Calibrated CLN ($\\hat p$ = {cal['p_hat_nodes']:.2f})", calib),
              ("Same count, random nodes", mc(G, random_nodes(G, k), args.mc, rng)),
              ("Same count, top degree", mc(G, top_nodes(G, k, G.degree), 1, rng)),
              ("Same count, top capacity", mc(G, top_nodes(G, k, G.node_cap), 1, rng))]
    for c in CLIENTS:
        same_k.append((f"All {c} nodes fail", mc(G, within_client(G, c, 1.0), 1, rng)))
    table_scenarios(same_k)

    res = {"graph_date": args.date, "rules": RULES_VERSION, "mc_runs": args.mc, "intact": intact,
           "observed_graphs": observed, "observed_on_base": obs_on_base, "observed_cln_on_base": obs_cln,
           "observed_delta_100k": obs_d, "placebo_windows": placebo.to_dict(orient="records"),
           "calibrated": calib, "scenarios": {n: r for n, r in same_k}, "without_lnd": dep,
           "capacity_matrix": matrix.round(4).to_dict(), "curves": curves.to_dict(orient="records"),
           "baselines": base.to_dict(orient="records"), "dividend": dividend.to_dict(orient="records")}
    out = DATA_DIR / "resilience_results.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log.info("Wrote %s", out)
    a = 100_000
    for name, r in same_k:
        log.info("%-46s failed %5d cap %5.1f%%  R_all %.3f  R %.3f  R_CLN_all %.3f", name, r["n_failed"],
                 100 * r["cap_removed"], r[f"R_all@{a}"], r[f"R@{a}"], r[f"R_CLN_all@{a}"])
    log.info("Within-client reachability without LND @100k: %s",
             {c: round(dep[f"R_{c}@{a}"], 3) for c in ["CLN", "Eclair", "LDK"]})
    log.info("Dividend @100k (R within vs across):\n%s",
             dividend[["q", "client", f"within_R_all@{a}", f"across_R_all@{a}"]].round(4).to_string(index=False) if len(dividend) else "-")


if __name__ == "__main__":
    main()
