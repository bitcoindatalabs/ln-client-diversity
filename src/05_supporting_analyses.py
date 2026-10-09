"""
Stage 5: supporting numbers and robustness checks — every number quoted in the text of main.tex that the
stages 01–04 do not already write to a table, plus the checks a reviewer asks for.

Data description      gossip coverage and quality; closes, commitment-output spends and funding txs at the cutoff
Outage robustness     daily rate at which CLN / LND nodes go dark in ordinary windows (churn baseline);
                      compliance by transport; recovery milestones; CLN disabled share on late dates
Placebo distribution  the outage's two-day reachability footprint vs every ordinary two-day window May–Aug 2026
                      (the 2026-07-10 Tor event and the early-August closing waves are reported separately)
Mixing null model     client labels permuted within node-capacity deciles: expected share of each client's capacity
                      facing LND, same-client share, and within-client reachability without LND
Closing behaviour     mass-close events, single-operator days, force-close cost, closes toward nodes that went dark,
                      and the August–October contraction of the public graph (who closed, how, vs earlier windows)
Stratified calibration  the calibrated CLN outage drawn with the observed per-cell compliance (size quintile ×
                      transport) instead of uniformly at random
Hub criticality       reachability lost when each of the 300 largest nodes fails alone; client mix of the largest
                      and most critical nodes

Outputs
  paper/figures/fig8_placebo.pdf          two-day reachability loss: ordinary windows vs the CLN outage and the Tor event
  data/supporting_numbers.json

Usage:  python src/05_supporting_analyses.py
"""

import importlib.util
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import DATA_CUTOFF, DATA_DIR, FIGURES_DIR, RANDOM_SEED, SRC_DIR  # noqa: E402

from lnmetrics import incident_metrics as im  # noqa: E402

_spec = importlib.util.spec_from_file_location("resilience", SRC_DIR / "04_resilience.py")
res = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(res)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)

CUT = pd.Timestamp(DATA_CUTOFF)
END = CUT + pd.Timedelta(days=1)
EVENT = "2026-08-26"
WINDOW_START = "20260501"                       # outage-analysis window (daily gossip complete from here)
TOR_WINDOWS = {"20260708", "20260709", "20260710"}                         # windows spanning 2026-07-10
CLOSE_WAVE_WINDOWS = {"20260801", "20260802", "20260803", "20260804"}      # early-August closing waves
NULL_DRAWS = 200
HUB_CANDIDATES = 300

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E8E8E8",
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ---------------------------------------------------------------------------------------
# Data description
# ---------------------------------------------------------------------------------------
def data_description(c: pd.DataFrame, s: pd.DataFrame) -> dict:
    q = im.gossip_quality()
    q = q[q["date"] <= DATA_CUTOFF]
    span = pd.date_range(pd.Timestamp(q["date"].min()), CUT)
    ok = q[q["status"] == "ok"]
    epochs = {}
    for ep in ["A", "B"]:
        e = ok[ok["epoch"] == ep]
        epochs[ep] = {"days": len(e), "first": e["date"].min(), "last": e["date"].max(),
                      "channels_p5_median_p95": [int(e["channels"].quantile(p)) for p in (.05, .5, .95)],
                      "nodes_p5_median_p95": [int(e["nodes"].quantile(p)) for p in (.05, .5, .95)]}
    cut_height = int(c["close_height"].max())
    cu = pd.read_parquet(im.PARQUET / "channels_unique.parquet", columns=["chan_point", "birth_block"])
    cu = cu[cu["chan_point"].notna() & ~cu["chan_point"].str.startswith("0" * 64)].drop_duplicates("chan_point")
    f = pd.read_parquet(im.PARQUET / "onchain" / "ln_funding_txs.parquet", columns=["txid", "confirm_height"])
    return {
        "gossip_calendar_days": len(span), "gossip_days_with_extract": len(q),
        "gossip_days_missing": len(span) - len(q), "gossip_status": q["status"].value_counts().to_dict(),
        "gossip_usable_share": r(len(ok) / len(span), 3), "epochs": epochs,
        "gossip_usable_days_since_2026_05_01": int((ok["date"] >= WINDOW_START).sum()),
        "calendar_days_since_2026_05_01": len(pd.date_range("2026-05-01", CUT)),
        "cutoff_height": cut_height,
        "watched_funding_outputs": int((cu["birth_block"] <= cut_height).sum()),
        "closes_first_date": str(c["date"].min().date()), "closes": len(c),
        "closes_by_type": c["close_type"].value_counts().to_dict(),
        "commitment_output_spends": len(s),
        "justice_spends": int(s["spend_type"].str.contains("revocation_breach").sum()),
        "last_justice_spend": str(s.loc[s["spend_type"].str.contains("revocation_breach"), "date"].max().date()),
        "funding_txs": int((f["confirm_height"] <= cut_height).sum()),
    }


# ---------------------------------------------------------------------------------------
# Outage robustness
# ---------------------------------------------------------------------------------------
def dark_transitions(labels: pd.DataFrame, client: str, dates: list) -> pd.DataFrame:
    """For each day d with d−1 and d+1 in gossip: share of the cohort's nodes not dark on d−1 that are dark
    (or absent) on d+1 — the same definition as the outage cohort (incident_metrics.went_dark)."""
    panel = im.node_dark_panel(labels, client, dates)
    by_day = {d: g.set_index("pub_key")["dark"] for d, g in panel.groupby("date")}
    rows = []
    for d in pd.to_datetime(dates):
        a, b = d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)
        if a not in by_day or b not in by_day:
            continue
        pre = by_day[a]
        pre = pre.index[~pre.to_numpy()]
        nxt = by_day[b].reindex(pre).fillna(True).to_numpy(dtype=bool)
        rows.append({"date": d, "n": len(pre), "rate": float(nxt.mean())})
    return pd.DataFrame(rows)


def outage_robustness(labels: pd.DataFrame, dates: list) -> dict:
    out = {}
    for cl in ("CLN", "LND"):
        t = dark_transitions(labels, cl, dates)
        ordinary = t[(t["date"] < EVENT) & ~t["date"].between("2026-07-09", "2026-07-11")]
        out[f"dark_rate_{cl}"] = {"event": r(t.loc[t["date"] == EVENT, "rate"].iloc[0]),
                                  "ordinary_median": r(ordinary["rate"].median()),
                                  "ordinary_p95": r(ordinary["rate"].quantile(.95)),
                                  "ordinary_max": r(ordinary["rate"].max()), "ordinary_days": len(ordinary),
                                  "tor_event_2026_07_10": r(t.loc[t["date"] == "2026-07-10", "rate"].iloc[0])}

    panel = im.node_dark_panel(labels, "CLN", dates)
    cohort = im.went_dark(panel, EVENT)
    pre = panel[(panel["date"] == pd.Timestamp(EVENT) - pd.Timedelta(days=1)) & ~panel["dark"]].set_index("pub_key")
    net = im.node_network(im.REF_DATE)
    tr = pd.DataFrame({"net": [net.get(k, "none") for k in pre.index], "dark": pre.index.isin(cohort)})
    out["compliance_by_transport"] = {k: {"nodes": int(len(g)), "went_dark": r(g["dark"].mean(), 3)} for k, g in tr.groupby("net")}

    rec = im.recovery_curve(panel, event=EVENT, horizon_days=60).set_index("day")["share_still_dark"]
    out["recovery_back_within_3_7_days"] = [r(1 - rec.get(3)), r(1 - rec.get(7))]

    daily = im.outage_daily(labels, dates)
    cln = daily[daily["target"] == "CLN"].set_index("date")["pct_disabled"]
    out["D_CLN_late"] = {d: r(cln.get(pd.Timestamp(d)), 2) for d in ("2026-10-02", "2026-10-04")}

    bynet = im.outage_by_network(labels, dates)
    for tgt in ("CLN", "LND"):
        for n in ("tor-only", "clearnet-only", "both"):
            x = bynet[(bynet["target"] == tgt) & (bynet["net"] == n)].set_index("date")["pct_disabled"]
            out[f"disabled_{tgt}_{n}"] = {"baseline_mean": r(x["2026-08-01":"2026-08-25"].mean(), 1),
                                          "aug27": r(x.get(pd.Timestamp("2026-08-27")), 1),
                                          "aug28": r(x.get(pd.Timestamp("2026-08-28")), 1),
                                          "aug29": r(x.get(pd.Timestamp("2026-08-29")), 1)}
    pooled = bynet.groupby(["date", "net"])[["dirs", "disabled"]].sum().reset_index()
    pooled["pct"] = 100 * pooled["disabled"] / pooled["dirs"]
    tor = pooled[pooled["net"] == "tor-only"].set_index("date")["pct"]
    out["tor_only_disabled_pooled"] = {"jul09": r(tor.get(pd.Timestamp("2026-07-09")), 1),
                                       "jul10": r(tor.get(pd.Timestamp("2026-07-10")), 1),
                                       "jun20_jul08_mean": r(tor["2026-06-20":"2026-07-08"].mean(), 1),
                                       "jul24": r(tor.get(pd.Timestamp("2026-07-24")), 1)}
    return out, panel, cohort


# ---------------------------------------------------------------------------------------
# Placebo distribution of the two-day reachability footprint
# ---------------------------------------------------------------------------------------
def placebo(labels: pd.DataFrame, dates: list, observed: dict) -> dict:
    dset, graphs, rows = set(dates), {}, []

    def G(d):
        if d not in graphs:
            graphs[d] = res.Graph(d, labels)
        return graphs[d]

    for d0 in pd.date_range("2026-05-01", "2026-08-23"):
        a, b = d0.strftime("%Y%m%d"), (d0 + pd.Timedelta(days=2)).strftime("%Y%m%d")
        if a not in dset or b not in dset:
            continue
        G0, G2 = G(a), G(b)
        gone = ~np.isin(G0.pub, G2.pub)
        m, f = G0.measure(np.zeros(G0.n, bool), 100_000), G0.measure(gone, 100_000)
        rows.append({"from": a, "n_failed": int(gone.sum()), "dR_all": f["R_all"] - m["R_all"],
                     "dR_CLN_all": f["R_CLN_all"] - m["R_CLN_all"],
                     "kind": "tor" if a in TOR_WINDOWS else "close-wave" if a in CLOSE_WAVE_WINDOWS else "ordinary"})
    pl = pd.DataFrame(rows)
    o = pl[pl["kind"] == "ordinary"]
    obs_all, obs_cln = observed["dR_all"], observed["dR_CLN_all"]
    fig_placebo(pl, obs_all, obs_cln)
    return {"windows": len(pl), "ordinary_windows": len(o),
            "ordinary_dR_all_median_p5_min": [r(o["dR_all"].median()), r(o["dR_all"].quantile(.05)), r(o["dR_all"].min())],
            "ordinary_dR_CLN_all_median_p5_min": [r(o["dR_CLN_all"].median()), r(o["dR_CLN_all"].quantile(.05)), r(o["dR_CLN_all"].min())],
            "ordinary_windows_worse_than_outage_all": int((o["dR_all"] <= obs_all).sum()),
            "ordinary_windows_worse_than_outage_cln": int((o["dR_CLN_all"] <= obs_cln).sum()),
            "ordinary_nodes_lost_median_max": [int(o["n_failed"].median()), int(o["n_failed"].max())],
            "outage": {"dR_all": r(obs_all), "dR_CLN_all": r(obs_cln)},
            "tor_windows": pl[pl["kind"] == "tor"].round(4).to_dict(orient="records"),
            "close_wave_windows": pl[pl["kind"] == "close-wave"].round(4).to_dict(orient="records"),
            "series": pl.round(4).to_dict(orient="records")}


def fig_placebo(pl: pd.DataFrame, obs_all: float, obs_cln: float):
    o = pl[pl["kind"] == "ordinary"]
    tor = pl[pl["from"].isin(["20260708", "20260709"])]       # the two windows spanning the Jul 10 onset
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, key, obs, title in [(axes[0], "dR_all", obs_all, "(a) All node pairs"),
                                (axes[1], "dR_CLN_all", obs_cln, "(b) Pairs of CLN nodes")]:
        bins = np.arange(-30, 1.01, 1.0)
        ax.hist(100 * o[key], bins=bins, color="#A3ACB8", edgecolor="white", lw=0.4, label="ordinary two-day windows")
        ax.axvline(100 * obs, color=res.CLIENT_PALETTE["CLN"], lw=1.6, label="CLN shutdown (Aug 25→27)")
        for k, v in enumerate(tor[key]):
            ax.axvline(100 * v, color="#4A5563", lw=1.2, ls=(0, (3, 1.5)),
                       label="Tor DoS (Jul 8→10, 9→11)" if k == 0 else None)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Change in reachable pairs, 100k sat (pp)")
        ax.set_xlim(-30, 1)
    axes[0].set_ylabel("Windows")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = FIGURES_DIR / "fig8_placebo.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------------------
# Mixing null model, stratified calibration, hub criticality (routable core of 2026-08-25)
# ---------------------------------------------------------------------------------------
def core_analyses(labels: pd.DataFrame, panel: pd.DataFrame, cohort: set, rng) -> dict:
    G = res.Graph("20260825", labels)
    out = {}
    truth = G.client.copy()
    deciles = pd.qcut(pd.Series(G.node_cap).rank(method="first"), 10, labels=False).to_numpy()
    obs_m = res.capacity_matrix(G)
    r_obs = G.measure(truth == "LND", 100_000)
    draws_m, draws_r = [], []
    for _ in range(NULL_DRAWS):
        perm = truth.copy()
        for k in range(10):
            idx = np.flatnonzero(deciles == k)
            perm[idx] = rng.permutation(truth[idx])
        G.client = perm
        draws_m.append(res.capacity_matrix(G).to_numpy())
        rr = G.measure(perm == "LND", 100_000)
        draws_r.append([rr["R_CLN"], rr["R_Eclair"], rr["R_LDK"]])
    G.client = truth
    nm = pd.DataFrame(np.mean(draws_m, axis=0), index=obs_m.index, columns=obs_m.columns)
    dr = np.array(draws_r)
    out["mixing_null_model"] = {
        "draws": NULL_DRAWS, "stratification": "node-capacity deciles",
        "share_facing_LND": {c: {"observed": r(obs_m.loc[c, "LND"], 3), "null": r(nm.loc[c, "LND"], 3)} for c in ["CLN", "Eclair", "LDK"]},
        "same_client_share": {c: {"observed": r(obs_m.loc[c, c], 3), "null": r(nm.loc[c, c], 3)} for c in ["CLN", "Eclair", "LDK"]},
        "R_without_LND_100k": {c: {"observed": r(r_obs[f"R_{c}"], 3), "null_mean": r(np.nanmean(dr[:, i]), 3),
                                   "null_p95": r(np.nanpercentile(dr[:, i], 95), 3)} for i, c in enumerate(["CLN", "Eclair", "LDK"])},
    }

    # stratified calibration: per-node failure probability = observed compliance of its (size quintile × transport) cell
    pre = panel[(panel["date"] == pd.Timestamp(EVENT) - pd.Timedelta(days=1)) & ~panel["dark"]].set_index("pub_key")[["cap_btc"]]
    pre["dark"] = pre.index.isin(cohort)
    pre["q"] = pd.qcut(pre["cap_btc"].rank(method="first"), 5, labels=False)
    net = im.node_network(im.REF_DATE)
    pre["net"] = [net.get(k, "none") for k in pre.index]
    cell = pre.groupby(["q", "net"])["dark"].mean()
    overall = pre["dark"].mean()
    cln_idx = np.flatnonzero(G.client == "CLN")
    probs = np.array([cell.get((pre.at[G.pub[i], "q"], pre.at[G.pub[i], "net"]), overall) if G.pub[i] in pre.index else overall
                      for i in cln_idx])

    def draw(g):
        f = np.zeros(G.n, bool)
        f[cln_idx[g.random(len(cln_idx)) < probs]] = True
        return f
    strat = res.mc(G, draw, 50, rng, amounts=[100_000])
    out["calibrated_stratified_100k"] = {k: r(v) for k, v in strat.items() if not k.endswith("_sd")}

    # hub criticality
    base = G.measure(np.zeros(G.n, bool), 100_000)
    order = np.argsort(-G.node_cap, kind="stable")
    rows = []
    for i in order[:HUB_CANDIDATES]:
        f = np.zeros(G.n, bool)
        f[i] = True
        m = G.measure(f, 100_000)
        rows.append({"pub": G.pub[i], "client": G.client[i], "cap_share": G.node_cap[i] / G.node_cap.sum(),
                     "cap_rank": len(rows) + 1, "dR": m["R"] - base["R"]})
    h = pd.DataFrame(rows)
    crit = h.sort_values("dR").head(20)
    alias = im.read_nodes("20260825", ["pub_key", "alias"]).set_index("pub_key")["alias"]
    cln = G.measure(G.client == "CLN", 100_000)
    out["hubs"] = {
        "all_CLN_dR_pp": r(100 * (cln["R"] - base["R"]), 2),
        # ranked by single-node criticality; no keys or aliases (the paper names only ACINQ)
        "top10_critical": [{"rank": k + 1, "client": x.client, "cap_rank": int(x.cap_rank), "cap_share": r(x.cap_share, 4),
                            "dR_pp": r(100 * x.dR, 2)} for k, x in enumerate(crit.head(10).itertuples())],
        "top1_critical_alias": alias.get(crit.iloc[0]["pub"], ""),
        "client_mix_top_by_capacity": {k: G.client[order[:k]].tolist().count("LND") / k for k in (10, 50, 100)},
        "top20_critical_by_client": crit["client"].value_counts().to_dict(),
        "top20_critical_dR_median_pp": r(100 * crit["dR"].median(), 2),
        "top1_critical": {"client": crit.iloc[0]["client"], "dR_pp": r(100 * crit.iloc[0]["dR"], 2),
                          "cap_share": r(crit.iloc[0]["cap_share"], 4)},
        "top10_capacity_clients": pd.Series(G.client[order[:10]]).value_counts().to_dict(),
        "top100_capacity_clients": pd.Series(G.client[order[:100]]).value_counts().to_dict(),
    }
    return out


# ---------------------------------------------------------------------------------------
# Closing behaviour
# ---------------------------------------------------------------------------------------
def closing(labels: pd.DataFrame, c: pd.DataFrame, s: pd.DataFrame, cohort: set, panel: pd.DataFrame) -> dict:
    out = {}
    may = c[c["date"] >= "2026-05-01"]
    me = im.mass_events(may, labels)
    out["mass_events_since_may"] = {"events": len(me), "by_hub_client": me["hub_client"].value_counts().to_dict()}
    for d in ("2026-08-03", "2026-08-26"):
        x = me[me["date"] == d].sort_values("force_closes", ascending=False).head(1)
        out[f"largest_mass_event_{d}"] = x[["hub_alias", "hub_client", "force_closes"]].to_dict(orient="records")
    tc = im.top_closers(c, labels, "2026-08-02", "2026-08-16", n=1)
    out["top_closer_aug02_16"] = tc[["alias", "client", "closes", "share_of_window"]].round(1).to_dict(orient="records")
    fs = im.fee_summary(may, s).set_index("close_type")
    out["fees_since_may"] = {k: {"closes": int(fs.at[k, "closes"]), "median_total_sat": r(fs.at[k, "total_fee_median"], 0),
                                 "total_btc": r(fs.at[k, "total_fee_btc"], 3)} for k in fs.index}

    # closes toward nodes that went dark vs CLN nodes that stayed up (two weeks after the call)
    pre = set(panel.loc[(panel["date"] == pd.Timestamp(EVENT) - pd.Timedelta(days=1)) & ~panel["dark"], "pub_key"])
    w = c[(c["date"] >= EVENT) & (c["date"] <= "2026-09-08")]
    p25 = im.read_policies("20260825", ["channel_id", "node_pub", "peer_pub"]).drop_duplicates("channel_id")
    for name, nodes in [("went_dark", cohort), ("stayed_up", pre - cohort)]:
        x = w[w["node1_pub"].isin(nodes) | w["node2_pub"].isin(nodes)]
        nch = int((p25["node_pub"].isin(nodes) | p25["peer_pub"].isin(nodes)).sum())
        nf = int((x["close_type"] == "force").sum())
        out[f"closes_two_weeks_{name}"] = {"nodes": len(nodes), "channels_aug25": nch, "closes": len(x), "force": nf,
                                           "force_per_1000_channels": r(1000 * nf / nch, 1)}

    # contraction of the public graph, 2026-08-01 → cutoff
    p1 = im.read_policies("20260801", ["channel_id", "node_pub", "peer_pub", "capacity"]).drop_duplicates("channel_id")
    pe = set(im.read_policies(DATA_CUTOFF, ["channel_id"])["channel_id"])
    gone = ~p1["channel_id"].isin(pe)
    closed = c[(c["date"] >= "2026-08-01") & c["channel_id"].isin(set(p1.loc[gone, "channel_id"]))]
    sides = pd.concat([closed["node1_pub"], closed["node2_pub"]]).value_counts()
    alias = dict(zip(labels["pub_key"], labels["alias"]))
    lm = im.label_map(labels)
    c1, c2 = p1["node_pub"].map(lm).fillna("Unknown"), p1["peer_pub"].map(lm).fillna("Unknown")

    def turnover(d0, d1):
        a = set(im.read_policies(d0, ["channel_id"])["channel_id"])
        b = set(im.read_policies(d1, ["channel_id"])["channel_id"])
        return {"gone": r(len(a - b) / len(a), 3), "new": r(len(b - a) / len(a), 3)}
    out["contraction"] = {
        "channels_aug01": len(p1), "channels_cutoff": len(pe), "gone": int(gone.sum()), "new": len(pe - set(p1["channel_id"])),
        "gone_closed_onchain_share": r(closed["channel_id"].nunique() / gone.sum(), 3),
        "gone_closed_in_august": int(closed.loc[closed["date"] < "2026-09-01", "channel_id"].nunique()),
        "capacity_share_gone": r(p1.loc[gone, "capacity"].sum() / p1["capacity"].sum(), 3),
        "closed_by_type": closed["close_type"].value_counts().to_dict(),
        "top10_operators_share_of_closes": r(sides.head(10).sum() / len(closed), 3),
        "top10_operators": [(alias.get(k), lm.get(k, "Unknown"), int(v)) for k, v in sides.head(10).items()],
        "gone_share_by_client": {k: r(gone[(c1 == k) | (c2 == k)].mean(), 3) for k in ["LND", "CLN", "Eclair", "LDK"]},
        "same_length_window_2026_05_01_07_04": turnover("20260501", "20260704"),
        "same_length_window_2026_06_01_08_04": turnover("20260601", "20260804"),
    }
    return out


# ---------------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    labels = im.node_labels()
    dates = [d for d in im.gossip_dates() if WINDOW_START <= d <= DATA_CUTOFF]
    c = im.load_closes(labels)
    c = c[c["date"] < END]
    s = im.load_spends()
    s = s[s["date"] < END]

    out = {"data_cutoff": DATA_CUTOFF}
    out["data"] = data_description(c, s)
    log.info("data: %s", out["data"])
    rob, panel, cohort = outage_robustness(labels, dates)
    out["outage"] = rob
    log.info("outage robustness: %s", rob)
    resil = json.loads((DATA_DIR / "resilience_results.json").read_text(encoding="utf-8"))
    out["placebo"] = placebo(labels, dates, resil["observed_delta_100k"])
    log.info("placebo: %s", out["placebo"])
    out.update(core_analyses(labels, panel, cohort, rng))
    out["closing"] = closing(labels, c, s, cohort, panel)
    path = DATA_DIR / "supporting_numbers.json"
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
