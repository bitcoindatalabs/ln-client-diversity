# Lightning Network Client Diversity and the 2026 CLN Shutdown

Code, aggregate data and LaTeX source for the paper below, by Saurabh Kumar (Bitcoin Data Labs).

## Title
**Does Client Diversity Protect the Lightning Network? Evidence from the 2026 Core Lightning Shutdown**

**Target Categories:** arXiv `cs.CR`, cross-list `cs.NI`
**Status:** Full draft; data cutoff 2026-10-04

---

## 📖 Overview
LND runs about 84% of public Lightning nodes and capacity. On 2026-08-26 Core Lightning maintainers told operators
who could not upgrade to take their nodes off the network: a client-correlated outage, observed in production.
The paper measures it from daily gossip and on-chain closes (against churn baselines, a control client and 94
ordinary two-day windows), calibrates a failure model on it, and tests how clients connect against a null model.
Finding: the outage stayed inside the CLN cohort because the nodes that complied were peripheral; what decides the
blast radius of a correlated failure is whether it reaches the hubs, not how many nodes run the failing software.

For the research plan and its history, see **[`SPECIFICATION.md`](./SPECIFICATION.md)**.

---

## 🛠️ Quickstart

### 1. Environment & Dependencies
Use your global Python environment (no virtual environment required):
```bash
pip install -r requirements.txt
```

### 2. Run Experiment Pipeline
Each stage reads the local Lightning data lake (via `python/automation/shared/lightning/`), truncates its inputs at
`DATA_CUTOFF` (`src/config.py`), and writes its figure(s) to `paper/figures/`, its table(s) to `paper/tables/` and
its numbers to `data/*.json`.
```bash
python src/01_client_census.py        # monthly client shares 2023-2026               -> fig1, table1
python src/02_classifier_eval.py      # client-inference validation (aliases, on-chain) -> fig2, table2
python src/03_observed_outage.py      # CLN shutdown of 2026-08-26                      -> fig3-5, tables 3-5
python src/04_resilience.py           # resilience scenarios, calibrated on stage 3      -> fig6-7, table6
python src/05_supporting_analyses.py  # every other in-text number + robustness checks -> fig8, supporting_numbers.json
```
Generated tables are overwritten on each run; fix table layout in the generator, not in `paper/tables/`.

### 3. Compile LaTeX Paper (MiKTeX, Windows)
```powershell
cd paper
.\build -Open      # pdflatex -> bibtex -> pdflatex until references settle; prints overfull/undefined warnings
.\build -Clean     # after editing references.bib
```
`build.bat` runs `build.ps1` with the execution policy bypassed (Windows blocks unsigned scripts by default);
`build.ps1` finds MiKTeX even when it is not on PATH. For edit-and-preview, VS Code's LaTeX Workshop
extension with `-synctex=1` gives click-to-jump between source and PDF.

### 4. Package for arXiv
```bash
python scripts/package_arxiv.py   # -> dist/ln-client-diversity-arxiv.tar.gz (build the paper first)
```

---

## 📂 Repository Structure
* `SPECIFICATION.md`: research plan (v2) and what changed from it.
* `paper/`: LaTeX source (`main.tex`, `references.bib`, `figures/`, `tables/`) and `build.ps1`.
* `src/`: the five pipeline stages; `src/archive_v1/` holds the superseded v1 scripts.
* `data/`: aggregate JSON outputs (committed); raw data stay in the data lake.
* `notebooks/`: scratch EDA notebooks (ignored by git).
* `scripts/package_arxiv.py`: builds the arXiv upload bundle.
