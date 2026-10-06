# Engineering & Research Specification (v2)

## Working title
**One Bug Away? Measuring Lightning Network Client Concentration and Common-Mode Failure — Calibrated by the 2026 Security Wave**

*Alternative:* "Client Monoculture in the Lightning Network: Empirical Concentration, an Observed Common-Mode Outage, and Exposure-Aware Resilience"

**Target:** arXiv (`cs.CR`, cross-list `cs.NI`) by **2026-10-18**; then FC / AFT submission.
**Companion:** `deep-dives/analyses/2026-10-lightning-network-analysis` (TABConf, 2026-10-11) supplies the incident measurements.
**v1 (superseded):** `_migration_backups/papers_20261004/SPECIFICATION.md.orig`

---

## 1. Positioning — why this paper, why now

Prior work measured LN topology and resilience to *random* or *degree-targeted* node removal
(Rohrer et al. 2019; Seres et al. 2020; Lin et al. 2020) and inferred client implementations
(Zabka et al. 2021; Espinasa-Vilarrasa et al. 2024). What is missing is the link between **who runs
which software** and **how failures actually correlate**. Removing "all LND nodes" from a graph where
LND is ~85% of capacity is trivially catastrophic and not a contribution. Our contributions:

- **C1 — Validated client inference + longitudinal concentration (2023–2026).**
  Feature-bit fingerprints from daily gossip, with explicit precision/recall on labeled anchors,
  cross-checked by *on-chain* fingerprints (closing-transaction locktime/sequence/fee patterns) for
  nodes without announcements. Client shares by node count *and* capacity over three years.
- **C2 — An observed common-mode failure.** On 2026-08-26 CLN maintainers asked operators to take
  nodes offline. From daily gossip we measure it directly: channels toward CLN nodes disabled by peers
  rose from 11.1% ± 0.9 to 37.9% the next day (capacity: 4.1% → 30.2%), while other cohorts stayed flat;
  recovery ≈ 1–4 weeks. To our knowledge, the first measured cohort-wide outage in LN.
- **C3 — Exposure-aware resilience.** 2026 advisories were *version-scoped* (e.g. LND < 0.19), and
  bugs across implementations were *uncorrelated*. We model failure as "fraction p of cohort c is
  exposed / unavailable for duration T" (not deletion), calibrate p and T on C2, and report
  payment-reachability and capacity-weighted metrics — not just giant-component size.
- **C4 — Policy defaults and routing (secondary).** Do client CLTV/fee defaults shift simulated route
  selection? Isolated one parameter at a time, under each client's actual pathfinding model.

## 2. Research questions
- **RQ1** How concentrated are LN implementations by nodes vs capacity, and how did that change 2023→2026?
- **RQ2** How accurate is gossip-based client inference, and do on-chain fingerprints agree?
- **RQ3** What happened during the observed CLN outage: magnitude, compliance by node size, recovery, effect on reachability?
- **RQ4** Calibrated to RQ3, what is the reachability / capacity loss when a fraction p of each cohort fails, versus random and degree baselines? Where are minority clients dependent on LND bridges?
- **RQ5** (secondary) Do default policies bias simulated route selection between clients?

## 3. Data
| Dataset | Source | Coverage |
|---|---|---|
| Daily gossip (nodes, channel policies incl. `disabled`) | own LND `describegraph` → `gossip_daily` parquet | 2023–2026 daily (documented gaps) |
| On-chain closes + commitment-output spends | own Bitcoin Core (unpruned, txindex) → `ln_close_tx_scanner.py` | 2026-05 → now; backfill to 2023 |
| Labels | operators with known implementations (ACINQ, Blockstream, Block, Lightning Labs-run, LSPs), self-identifying aliases (`*-ldk`, `ldk_server`, CLN random aliases) | ~100–300 nodes, expand |
| Advisories | Lightning Labs, CLN, Eclair, LDK | 2024–2026 |

Data lake and data-quality notes: `python/automation/docs/LN_DATA_LAKE_AND_FINDINGS.md`.
**Threats to validity (state in paper):** single vantage point; public (announced) graph only — LDK and
Eclair (Phoenix) usage behind private channels is under-counted; feature sets evolve by release.

## 4. Methods

### 4.1 Client inference (C1/RQ2)
Rule set v1 = `shared/lightning/client_fingerprint.py` (implementation-specific feature bits:
LND 2023/31; Eclair 29/37/61; CLN 11+39/43; LDK 39/43/63 w/o 11; fallbacks on CLTV defaults/LND colour).
- Ground truth: labeled anchor set; report per-class precision/recall, confusion matrix, and how
  shares change if every low-confidence node is relabeled (bounds).
- Baseline comparison: decision tree on the full feature-bit vector (Espinasa-Vilarrasa et al. style),
  trained on the anchors; agreement with rules.
- On-chain cross-check: for closes where the closer's client is known, learn which tx features
  (mutual-close nLockTime/nSequence, fee rounding, output ordering, anchor usage) separate clients;
  apply to nodes without gossip features.

### 4.2 Observed outage (C2/RQ3)
- Metric: `D_c(t)` = share of channel directions with `peer_pub ∈ c` that peers mark `disabled`;
  capacity-weighted variant. Baseline = mean ± sd over a pre-window; effect in sd units.
- Compliance: node-level "dark" indicator (≥ 80% of its channels disabled by peers) by capacity decile.
- Recovery: time to return within 2 sd of baseline; survival curve per node.
- Control: other cohorts over the same days (difference-in-differences).
- Network effect: payment reachability before/during (see 4.3) using the observed disabled set.

### 4.3 Exposure-aware resilience (C3/RQ4)
- Graph: directed channel graph with capacity; a node "fails" = all its channels unusable for T.
- Scenarios: for each cohort c and p ∈ {0, .05, …, 1}: random fraction p of c fails (50 MC runs);
  calibrated point p̂, T̂ from 4.2. Baselines: same number of nodes random / top-degree / top-capacity.
- Metrics: (a) **reachable pair share** for payments of size a ∈ {10k, 100k, 1M sats} given
  capacity ≥ a along a path (uniform liquidity assumption; sensitivity with 50/50 split); (b) capacity
  in the largest component; (c) minority-client isolation: share of CLN/Eclair/LDK pairs still connected;
  (d) **bridge dependence**: share of each cohort's capacity whose shortest paths require LND nodes.
- Correlated-failure contrast: same total failed capacity, drawn (i) within one client vs (ii) across
  clients — quantifies the diversity dividend.

### 4.4 Routing defaults (C4/RQ5, secondary)
- Sample 5,000 (s, t) pairs; amounts {1k, 100k, 1M} sats.
- Pathfinding per client model (verify formulas against each codebase before use): LND probability-
  weighted cost with time-lock risk factor; CLN `riskfactor`; LDK ProbabilisticScorer-style; Eclair heuristics.
- Counterfactuals, one at a time: set a cohort's CLTV to the network median; set fees to median.
  Report change in route share per cohort. Wording: *simulated* route selection (no flow data;
  probing-based validation after first draft).

## 5. Figures & tables (target)
1. Client share by nodes vs capacity, monthly 2023–2026 (stacked area, two panels).
2. Classifier validation: confusion matrix + share bounds.
3. **Observed outage**: D_c(t) daily Jul–Oct 2026 with incident markers (CLN vs others), + capacity-weighted.
4. Recovery survival curve of CLN nodes after 2026-08-26.
5. Resilience curves: reachable pair share vs p per cohort, with baselines and the calibrated point.
6. Bridge-dependence heatmap (cohort × cohort capacity).
7. (secondary) Route-share change under default counterfactuals.
Tables: client census; outage effect sizes; scenario comparison.

All figures: matplotlib, vector PDF, colorblind-safe palette, no dual axes.

## 6. Ethics & disclosure
Aggregates only; no per-node exposure lists while advisories are active. Share the outage findings
with implementation teams before posting. Data and code released (aggregated datasets + scripts).

## 7. References (verify every entry before citing — v1 list contained unverifiable items)
Core: Poon & Dryja 2016; BOLT specs (2, 3, 7, 9).
Topology & resilience: Rohrer, Malliaris, Tschorsch 2019; Seres et al. 2020; Lin et al. 2020; Martinazzi & Flori 2020.
Client inference: Zabka et al. 2021; Espinasa-Vilarrasa et al. 2024.
Attacks: Harris & Zohar 2020 (Flood & Loot); Mizrahi & Zohar 2021 (congestion); jamming literature.
Monoculture: Geer et al. 2003; Birman & Schneider 2009; Ethereum client-diversity incidents.
Pathfinding: Pickhardt & Richter 2021; client documentation for LND / CLN / LDK / Eclair scorers.
**Remove** `weintraub2024payout` unless it can be verified as cited.

## 8. Code layout
```
papers/01-ln-client-monoculture-fragility/
  src/01_client_census.py       C1: shares 2023–2026 (uses shared client_fingerprint)
  src/02_classifier_eval.py     RQ2: anchors, P/R, bounds, decision-tree baseline, on-chain cross-check
  src/03_observed_outage.py     C2: D_c(t), compliance, recovery, DiD
  src/04_resilience.py          C3: exposure scenarios, baselines, bridge dependence
  src/05_routing_defaults.py    C4 (secondary)
  paper/  main.tex, references.bib, figures/, tables/
```
Inputs come from the shared data lake (gossip_daily, onchain); no data copied into the repo except
small aggregated tables for figures.

## 9. Timeline
| Date | Milestone |
|---|---|
| Oct 5–7 | C2 measurements (shared with deep dive); C1 census; anchor labels |
| Oct 8–10 | Classifier validation; resilience scenarios v1 |
| Oct 11 | TABConf: present deep dive; share early paper findings with LN researchers |
| Oct 12–16 | C3 complete, C4 secondary, writing |
| Oct 17–18 | Full draft, references verified, arXiv package |
| after | Probing-based validation; FC/AFT submission |
