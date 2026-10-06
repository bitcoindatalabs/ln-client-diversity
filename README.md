# Paper 01: Lightning Network Client Monoculture & Fragility

## Title
**The Fragility of Decentralization: Quantifying Common-Mode Failure and Algorithmic Routing Distortion under Lightning Network Client Monoculture**

**Target Categories:** arXiv `cs.CR`, `cs.NI`  
**Status:** In Progress

---

## 📖 Overview
This research paper empirically evaluates the systemic risk of software monoculture on the Bitcoin Lightning Network (LN). Combining multi-feature client fingerprinting with network percolation modeling and pathfinding distortion analysis, it quantifies the blast radius of correlated common-mode failures (CMF) and demonstrates how compiled client defaults create algorithmic feedback loops in routing.

For the complete technical and experimental guide, see **[`SPECIFICATION.md`](./SPECIFICATION.md)**.

---

## 🛠️ Quickstart

### 1. Environment & Dependencies
Use your global Python environment (no virtual environment required):
```bash
pip install -r requirements.txt
```

### 2. Run Experiment Pipeline
```bash
# 1. Classify nodes using gossip features
python src/01_classify_nodes.py

# 2. Construct directed liquidity graph and compute census metrics
python src/02_build_graph.py

# 3. Simulate Common-Mode Failure (CMF) & Percolation phase transitions
python src/03_simulate_percolation.py

# 4. Simulate pathfinding routing bias & default parameter tax
python src/04_simulate_routing.py
```

### 3. Compile LaTeX Paper
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📂 Repository Structure
* `SPECIFICATION.md`: Detailed instructions for the coding agent.
* `paper/`: Complete LaTeX source files (`main.tex`, `references.bib`, `figures/`, `tables/`).
* `src/`: Modular Python experiment scripts.
* `data/`: Local storage for datasets (strictly ignored by git).
* `notebooks/`: Scratch EDA notebooks (strictly ignored by git).
