"""
Stage 3 (C2 / RQ3): The observed common-mode outage — Core Lightning, 2026-08-26.

On 2026-08-26 CLN maintainers asked operators to take nodes offline. A node that goes offline cannot
announce anything, so the outage is measured on the *peer* side: the share of channel directions
toward a cohort's nodes that peers mark `disabled` in gossip (D_c(t), SPECIFICATION.md §4.2).

Metrics come from python/automation/shared/lightning/incident_metrics.py, the same module that powers
the TABConf deep dive, so the paper and the public report cannot drift apart.

Outputs
  paper/figures/fig3_observed_outage.pdf       D_c(t), count- and capacity-weighted, Jul–Oct 2026
  paper/figures/fig4_cln_recovery.pdf          share of dark CLN nodes still dark, by day
  paper/tables/table3_outage_effect.tex        baseline mean ± sd, peak, z, DiD vs LND (block bootstrap)
  paper/tables/table4_outage_compliance.tex    went dark by capacity quintile; force-close rate ratios
  paper/tables/table5_outage_robustness.tex    CLN effect by target transport (Tor/clearnet) and without its largest node
  paper/figures/fig5_tor_event.pdf             the 2026-07-10 transport failure: disabled share by network type
  data/observed_outage_calibration.json        p̂ and T̂ for the exposure scenarios in 04_resilience.py

Usage:  python src/03_observed_outage.py [--refresh-labels]
"""

import argparse
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import CLIENT_PALETTE, DATA_DIR, FIGURES_DIR, RANDOM_SEED, REPO_ROOT, TABLES_DIR  # noqa: E402

sys.path.insert(0, str(REPO_ROOT.parent.parent / "python" / "automation"))
from shared.lightning import incident_metrics as im  # noqa: E402
from shared.lightning.client_fingerprint import RULES_VERSION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)

CLIENTS = ["LND", "CLN", "Eclair", "LDK"]
EVENT = "2026-08-26"
PEAK = "2026-08-27"
BASELINE = ("2026-08-01", "2026-08-25")
WINDOW = ("2026-08-27", "2026-09-09")          # two weeks after the call
PLOT_FROM = "2026-07-01"
BOOT_ITERS = 2000
BOOT_BLOCK = 7                                  # days; gossip shares are autocorrelated

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.6, "axes.edgecolor": "#888888",
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------
def block_bootstrap_did(daily: pd.DataFrame, metric: str, treated: str = "CLN", control: str = "LND",
                        iters: int = BOOT_ITERS, block: int = BOOT_BLOCK, seed: int = RANDOM_SEED):
    """DiD = (treated − control) in WINDOW minus (treated − control) in BASELINE, on daily values.
    95% CI from a moving-block bootstrap over days within each period."""
    w = daily.pivot(index="date", columns="target", values=metric)
    gap = (w[treated] - w[control]).dropna()
    base = gap[(gap.index >= BASELINE[0]) & (gap.index <= BASELINE[1])].to_numpy()
    win = gap[(gap.index >= WINDOW[0]) & (gap.index <= WINDOW[1])].to_numpy()
    point = win.mean() - base.mean()
    rng = np.random.default_rng(seed)

    def resample(x):
        n = len(x)
        starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
        return np.concatenate([x[s:s + block] for s in starts])[:n]

    boots = np.array([resample(win).mean() - resample(base).mean() for _ in range(iters)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, len(base), len(win)


def calibration(panel: pd.DataFrame, daily: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """p̂: share of CLN nodes (and of CLN capacity) that went dark after the call, among those not
    dark before; T̂: median days dark among nodes that came back, plus the share that never did."""
    j = panel[(panel["date"] == pd.Timestamp(EVENT) - pd.Timedelta(days=1)) & ~panel["dark"]].set_index("pub_key")
    cohort = im.went_dark(panel, EVENT)
    j["dark"] = j.index.isin(cohort)
    p_nodes = float(j["dark"].mean())
    p_cap = float(j.loc[j["dark"], "cap_btc"].sum() / j["cap_btc"].sum())

    later = panel[(panel["date"] > PEAK) & panel["pub_key"].isin(cohort)].sort_values("date")
    first_back = later[~later["dark"]].groupby("pub_key")["date"].min()
    days_dark = (first_back - pd.Timestamp(PEAK)).dt.days + 1
    still = 1 - len(first_back) / len(cohort) if cohort else float("nan")

    cln = daily[daily["target"] == "CLN"].set_index("date")
    peak_excess = float(cln.loc[PEAK, "pct_disabled"] - cln.loc[BASELINE[0]:BASELINE[1], "pct_disabled"].mean())
    return {
        "event": EVENT, "peak_day": PEAK, "client_rules": RULES_VERSION, "label_ref_date": im.REF_DATE,
        "dark_definition": f"share of directions toward node disabled by peers >= {im.DARK_SHARE}, or node absent from gossip",
        "cln_nodes_at_risk": int(len(j)), "cln_nodes_went_dark": int(len(cohort)),
        "p_hat_nodes": round(p_nodes, 4), "p_hat_capacity": round(p_cap, 4),
        "peak_excess_disabled_pct_points": round(peak_excess, 2),
        "T_hat_median_days_dark_returners": float(days_dark.median()) if len(days_dark) else None,
        "T_hat_p25_p75_days": [float(days_dark.quantile(.25)), float(days_dark.quantile(.75))] if len(days_dark) else None,
        "share_never_returned_by_last_snapshot": round(float(still), 4),
        "share_dark_on_last_snapshot": round(float(rec["share_still_dark"].iloc[-1]), 4),
        "last_snapshot": str(panel["date"].max().date()),
    }


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
def incident_lines(ax):
    for inc in im.INCIDENTS:
        if inc["date"] >= PLOT_FROM:
            ax.axvline(pd.Timestamp(inc["date"]), color="#999999", lw=0.6, ls="-", zorder=0)


def fig_outage(daily: pd.DataFrame):
    d = daily[daily["date"] >= PLOT_FROM]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    for ax, metric, title in [(axes[0], "pct_disabled", "(a) Share of channels disabled by peers"),
                              (axes[1], "pct_cap_disabled", "(b) Share of capacity disabled by peers")]:
        incident_lines(ax)
        ends = []
        for c in CLIENTS:
            s = d[d["target"] == c].set_index("date")[metric].asfreq("D")   # gossip gaps stay gaps
            ax.plot(s.index, s.values, color=CLIENT_PALETTE[c], lw=1.4 if c == "CLN" else 1.0)
            last = s.dropna()
            ends.append([last.iloc[-1], c, last.index[-1]])
        ax.set_ylim(bottom=0)
        # direct end labels, nudged apart so they never overlap
        gap = 0.075 * ax.get_ylim()[1]
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for y, c, x in ends:
            ax.annotate(c, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7,
                        color=CLIENT_PALETTE[c], va="center", annotation_clip=False)
        ax.set_title(title, loc="left")
        ax.set_ylabel("%")
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    top = axes[0].get_ylim()[1]
    axes[0].annotate('CLN: "take nodes offline"', (pd.Timestamp(EVENT), top * 0.97), xytext=(-4, 0),
                     textcoords="offset points", ha="right", va="top", fontsize=7, color="#444444")
    handles = [matplotlib.lines.Line2D([], [], color=CLIENT_PALETTE[c], lw=1.4) for c in CLIENTS]
    fig.legend(handles, CLIENTS, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 0.98, 1))
    out = FIGURES_DIR / "fig3_observed_outage.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Wrote %s", out)


def fig_recovery(rec: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.concatenate([[0], rec["day"].to_numpy()])
    y = np.concatenate([[1.0], rec["share_still_dark"].to_numpy()])
    ax.step(x, 100 * y, where="post", color=CLIENT_PALETTE["CLN"], lw=1.4)
    ax.set_xlabel(f"Days after {EVENT}")
    ax.set_ylabel("% still dark")
    ax.set_ylim(0, 102)
    ax.set_title(f"CLN nodes dark on {PEAK} (n = {int(rec['cohort'].iloc[0])})", loc="left")
    ax.annotate(f"{100 * y[-1]:.0f}%", (x[-1], 100 * y[-1]), xytext=(0, 6), textcoords="offset points",
                ha="right", fontsize=7)
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_cln_recovery.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------
def table_effect(daily: pd.DataFrame, eff: pd.DataFrame):
    rows = []
    for c in CLIENTS:
        e = eff[eff["target"] == c].iloc[0]
        cells = [c]
        for m in ["pct_disabled", "pct_cap_disabled"]:
            cells += [f"{e[m + '_base']:.1f} $\\pm$ {e[m + '_sd']:.1f}", f"{e[m + '_peak']:.1f}", f"{e[m + '_z']:.1f}"]
        rows.append(" & ".join(cells) + r" \\")
    did = []
    for m, lab in [("pct_disabled", "channels"), ("pct_cap_disabled", "capacity")]:
        pt, lo, hi, nb, nw = block_bootstrap_did(daily, m)
        did.append(f"{lab}: {pt:+.1f} pp [{lo:+.1f}, {hi:+.1f}]")
    tex = r"""\begin{table}[t]
\centering
\caption{Share of channel directions toward each client cohort disabled by peers (\%). Baseline: daily mean $\pm$ sd, """ + \
        f"{BASELINE[0]} to {BASELINE[1]}; peak: {PEAK}; $z$ = (peak $-$ mean)/sd." + r""" Difference-in-differences, CLN vs.\ LND, """ + \
        f"{WINDOW[0]} to {WINDOW[1]} vs.\\ baseline (95\\% moving-block bootstrap CI, block {BOOT_BLOCK} days): " + "; ".join(did) + r""".}
\label{tab:outage-effect}
\small
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{3}{c}{By channel count} & \multicolumn{3}{c}{By capacity} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Client & Baseline & Peak & $z$ & Baseline & Peak & $z$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out = TABLES_DIR / "table3_outage_effect.tex"
    out.write_text(tex, encoding="utf-8")
    log.info("Wrote %s (%s)", out, "; ".join(did))


def table_compliance(comp: pd.DataFrame, rates: pd.DataFrame):
    crow = "\n".join(f"{r.size_bin} & {r.min_cap_btc:.3f}--{r.max_cap_btc:.3f} & {r.nodes} & {r.pct_went_dark:.1f} \\\\"
                     for r in comp.itertuples())
    names = {"CLN": "CLN", "Eclair": "Eclair", "LDK": "LDK", "LND_only": r"LND$\leftrightarrow$LND"}
    rrow = "\n".join(f"{names[r.cohort]} & {r.baseline_rate:.2f} & {r.window_rate:.2f} & {r.ratio:.2f} & {r.ratio_vs_LND_only:.2f} \\\\"
                     for r in rates.itertuples())
    tex = r"""\begin{table}[t]
\centering
\caption{(a) CLN nodes that went dark after the shutdown call (not dark on 2026-08-25, dark on 2026-08-28), by
pre-incident node capacity quintile. (b) Organic force closes per 1{,}000 open public channels per day involving
each cohort, 2026-07-01--08-25 (baseline) vs.\ 2026-08-26--09-08 (window); mass-close events (one node in
$\geq """ + str(im.MASS_CLOSE_MIN) + r"""$ force closes in a UTC day) excluded.}
\label{tab:outage-compliance}
\small
\begin{tabular}{lrrr}
\toprule
\multicolumn{4}{l}{(a) Compliance by size} \\
Quintile & Capacity (BTC) & Nodes & Went dark (\%) \\
\midrule
""" + crow + r"""
\midrule
\multicolumn{4}{l}{(b) Organic force-close rate} \\
\end{tabular}
\begin{tabular}{lrrrr}
Cohort & Baseline & Window & Ratio & vs.\ LND$\leftrightarrow$LND \\
\midrule
""" + rrow + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out = TABLES_DIR / "table4_outage_compliance.tex"
    out.write_text(tex, encoding="utf-8")
    log.info("Wrote %s", out)


def robustness(daily: pd.DataFrame, bynet: pd.DataFrame):
    """CLN effect re-estimated (i) on clearnet-only and Tor-only targets, (ii) capacity-weighted without the
    single node with the most disabled capacity each day. Guards against a Tor-heavy CLN population and
    against one large node (e.g. a ~50 BTC CLN node dark 2026-07-28..30) driving the result."""
    variants = [("All targets, by channels", daily, "pct_disabled"),
                ("All targets, by capacity", daily, "pct_cap_disabled"),
                ("By capacity, largest node removed", daily, "pct_cap_disabled_x1")]
    for net in ["clearnet-only", "tor-only", "both"]:
        sub = bynet[bynet["net"] == net][["date", "target", "pct_disabled"]]
        variants.append((f"{net.capitalize()} targets, by channels", sub, "pct_disabled"))
    rows = []
    for name, df, m in variants:
        w = df.pivot(index="date", columns="target", values=m)
        b = w.loc[BASELINE[0]:BASELINE[1], "CLN"]
        pk = w.loc[PEAK, "CLN"]
        pt, lo, hi, _, _ = block_bootstrap_did(df, m)
        rows.append(f"{name} & {b.mean():.1f} & {pk:.1f} & {(pk - b.mean()) / b.std():.1f} & {pt:+.1f} [{lo:+.1f}, {hi:+.1f}] \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Robustness of the CLN outage effect. Baseline and peak as in Table~\ref{tab:outage-effect};
DiD vs.\ LND targets in the same subset, 95\% moving-block bootstrap CI. Target transport from node
announcements on """ + im.REF_DATE + r""".}
\label{tab:outage-robustness}
\small
\begin{tabular}{lrrrr}
\toprule
Variant & Baseline (\%) & Peak (\%) & $z$ & DiD (pp) \
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out = TABLES_DIR / "table5_outage_robustness.tex"
    out.write_text(tex, encoding="utf-8")
    log.info("Wrote %s", out)
    for r in rows:
        log.info("  %s", r)


def fig_tor_event(bynet: pd.DataFrame):
    """All clients pooled: disabled share toward Tor-only vs clearnet-only vs dual-stack nodes."""
    d = bynet[(bynet["date"] >= "2026-06-20") & (bynet["date"] <= "2026-08-10")]
    pooled = d.groupby(["date", "net"])[["dirs", "disabled"]].sum().reset_index()
    pooled["pct"] = 100 * pooled["disabled"] / pooled["dirs"]
    colors = {"tor-only": "#4A5563", "clearnet-only": "#A3ACB8", "both": "#C9A24A"}
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axvline(pd.Timestamp("2026-07-10"), color="#999999", lw=0.6)
    for net, col in colors.items():
        s = pooled[pooled["net"] == net].set_index("date")["pct"].asfreq("D")
        ax.plot(s.index, s.values, color=col, lw=1.3)
        last = s.dropna()
        ax.annotate(net, (last.index[-1], last.iloc[-1]), xytext=(3, 0), textcoords="offset points",
                    fontsize=7, color=col, va="center", annotation_clip=False)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("% of channels disabled by peers")
    ax.set_title("All clients, by target transport", loc="left")
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    out = FIGURES_DIR / "fig5_tor_event.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-labels", action="store_true")
    args = ap.parse_args()
    for d in (FIGURES_DIR, TABLES_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    labels = im.node_labels(refresh=args.refresh_labels)
    dates = im.gossip_dates()
    log.info("Client rules %s; %d gossip days %s–%s", RULES_VERSION, len(dates), dates[0], dates[-1])

    daily = im.outage_daily(labels, dates)
    daily = daily[daily["target"].isin(CLIENTS)]
    eff = im.outage_effect(daily, baseline=BASELINE, peak_day=PEAK)
    panel = im.node_dark_panel(labels, "CLN", dates)
    comp = im.compliance_by_size(panel)
    rec = im.recovery_curve(panel, event=EVENT, horizon_days=60)

    closes = im.load_closes(labels)
    cd = im.closes_daily(closes)
    exposure = im.channels_by_cohort(labels, dates)
    rates = im.force_close_rates(cd, exposure, baseline=("2026-07-01", "2026-08-25"), window=("2026-08-26", "2026-09-08"))

    bynet = im.outage_by_network(labels, dates)
    robustness(daily, bynet)
    fig_tor_event(bynet)
    fig_outage(daily)
    fig_recovery(rec)
    table_effect(daily, eff)
    table_compliance(comp, rates)

    cal = calibration(panel, daily, rec)
    out = DATA_DIR / "observed_outage_calibration.json"
    out.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    log.info("Wrote %s: %s", out, cal)


if __name__ == "__main__":
    main()
