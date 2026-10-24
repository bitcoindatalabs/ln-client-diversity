# Does Client Diversity Protect the Lightning Network? Evidence from the 2026 Core Lightning Shutdown

Saurabh Kumar, Bitcoin Data Labs

**Paper:** [arXiv:2610.20952](https://arxiv.org/abs/2610.20952) ([HTML](https://arxiv.org/html/2610.20952),
[PDF](https://arxiv.org/pdf/2610.20952)) · DOI [10.48550/arXiv.2610.20952](https://doi.org/10.48550/arXiv.2610.20952)
· [Bitcoin Data Labs research](https://bitcoindatalabs.org/research.html#ln-client-diversity)

Code, aggregate data and LaTeX source for the paper. arXiv categories: `cs.CR`, cross-list `cs.NI`.
Data cutoff: 2026-10-04. The arXiv versions are tagged `v1.0-arxiv` (arXiv:2610.20952v1) and `v2.0-arxiv`
(v2: client rules v4, which separate CLN from Eclair by bit 5 instead of keysend, and node shares reported with and
without one operator's managed fleet).

## Summary
LND runs about 84% of public Lightning capacity and 80% of public nodes (84% counting one operator's fleet of
about 2,000 tiny LND nodes). On 2026-08-26 Core Lightning maintainers told operators
who could not upgrade to take their nodes off the network: a client-correlated outage, observed in production.
The paper measures it from daily gossip and on-chain closes (against churn baselines, a control client and 94
ordinary two-day windows), calibrates a failure model on it, and tests how clients connect against a null model.
Finding: the outage stayed inside the CLN cohort because the nodes that complied were peripheral; what decides the
blast radius of a correlated failure is whether it reaches the hubs, not how many nodes run the failing software.

## Key findings

**How diverse is Lightning's client software?** Not very, by capacity. LND runs 80% of public nodes (84% counting one
operator's fleet of about 2,000 tiny nodes) and holds 84% of public channel capacity, a share that has barely moved
since 2023.

**What happened when Core Lightning nodes went offline on 26 August 2026?** 22% of CLN nodes went dark overnight, ten
times the usual rate, and a third were still dark five weeks later. Reachability among all node pairs fell by 7.8
percentage points, half the loss during a Tor denial-of-service wave in July; among CLN pairs it fell by 24.5 points.

**What decides how far a failure in one client spreads?** Whether it reaches the hubs, not how many nodes run the
failing software. Losing ACINQ's Eclair node alone costs the surviving network four times the reachability lost with
all 619 routable CLN nodes.

**What should be tracked instead of node counts?** Which software the hubs run.

## Repository structure
* `paper/`: LaTeX source (`main.tex`, `references.bib`, `figures/`, `tables/`), the compiled `main.pdf` and a
  build script.
* `src/01_*.py` … `src/06_*.py`: the six pipeline stages.
* `src/lnmetrics/`: client inference from gossip (`client_fingerprint.py`), outage and close metrics
  (`incident_metrics.py`) and on-chain wallet evidence (`onchain_fingerprint.py`).
* `data/`: aggregate JSON outputs behind every figure, table and in-text number.
* `scripts/package_arxiv.py`: builds the arXiv upload bundle.

## Reproducing the results

### 1. Dependencies
```bash
pip install -r requirements.txt
```

### 2. Input data
The stages read daily snapshots of the public Lightning gossip graph and on-chain channel extracts as Parquet
files; `src/lnmetrics/incident_metrics.py` documents the expected layout. The underlying gossip and blockchain
data are public; our daily snapshots are available from the author on request. Point the pipeline at them with
```bash
export LN_PARQUET_DIR=/path/to/parquet        # default: data/parquet
```

### 3. Pipeline
Each stage truncates its inputs at `DATA_CUTOFF` (`src/config.py`) and writes its figure(s) to `paper/figures/`,
its table(s) to `paper/tables/` and its numbers to `data/*.json`.
```bash
python src/01_client_census.py        # monthly client shares 2023-2026                -> fig1, table1
python src/02_classifier_eval.py      # client-inference validation (aliases, on-chain) -> fig2, table2
python src/03_observed_outage.py      # CLN shutdown of 2026-08-26                      -> fig3-5, tables 3-5
python src/04_resilience.py           # resilience scenarios, calibrated on stage 3      -> fig6-7, table6
python src/05_supporting_analyses.py  # other in-text numbers + robustness checks       -> fig8, supporting_numbers.json
python src/06_fleet_check.py          # node shares and reachability without one fleet  -> fleet_check.json
```

### 4. Paper
`paper/build.ps1` (Windows, MiKTeX) runs pdflatex → bibtex → pdflatex until references settle; `build.bat` runs it
from cmd. On other systems:
```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
`python scripts/package_arxiv.py` then writes the arXiv bundle to `dist/`.

## How to cite
Please cite the paper (GitHub's *Cite this repository* button gives the same entry from [`CITATION.cff`](./CITATION.cff)):
```bibtex
@misc{kumar2026clientdiversity,
  title         = {Does Client Diversity Protect the Lightning Network? Evidence from the 2026 Core Lightning Shutdown},
  author        = {Saurabh Kumar},
  year          = {2026},
  eprint        = {2610.20952},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2610.20952},
  url           = {https://arxiv.org/abs/2610.20952}
}
```

## License
Code and aggregate data are released under the MIT License (see [`LICENSE`](./LICENSE)).
