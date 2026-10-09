# Does Client Diversity Protect the Lightning Network? Evidence from the 2026 Core Lightning Shutdown

Saurabh Kumar, Bitcoin Data Labs

Code, aggregate data and LaTeX source for the paper. arXiv categories: `cs.CR`, cross-list `cs.NI`.
Data cutoff: 2026-10-04. The version submitted to arXiv is tagged `v1.0-arxiv`.

## Summary
LND runs about 84% of public Lightning nodes and capacity. On 2026-08-26 Core Lightning maintainers told operators
who could not upgrade to take their nodes off the network: a client-correlated outage, observed in production.
The paper measures it from daily gossip and on-chain closes (against churn baselines, a control client and 94
ordinary two-day windows), calibrates a failure model on it, and tests how clients connect against a null model.
Finding: the outage stayed inside the CLN cohort because the nodes that complied were peripheral; what decides the
blast radius of a correlated failure is whether it reaches the hubs, not how many nodes run the failing software.

## Repository structure
* `paper/`: LaTeX source (`main.tex`, `references.bib`, `figures/`, `tables/`), the compiled `main.pdf` and a
  build script.
* `src/01_*.py` … `src/05_*.py`: the five pipeline stages.
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
```

### 4. Paper
`paper/build.ps1` (Windows, MiKTeX) runs pdflatex → bibtex → pdflatex until references settle; `build.bat` runs it
from cmd. On other systems:
```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
`python scripts/package_arxiv.py` then writes the arXiv bundle to `dist/`.

## License
Code and aggregate data are released under the MIT License (see [`LICENSE`](./LICENSE)).
