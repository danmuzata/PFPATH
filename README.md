# PfPATH

**PfPATH** (Plasmodium falciparum Pathways of Adaptive Trajectories under Heterogeneous selection)
is an end-to-end computational framework for modeling **protein evolution under structural,
biophysical, and population-genetic constraints**, with a primary focus on **PfDHFR antifolate resistance**.

PfPATH integrates **sequence-based analyses**, **structure- and dynamics-informed metrics**, and
**evolutionary simulations** into a single, modular, reproducible pipeline.

---

## 🔬 What PfPATH Does

PfPATH connects multiple biological layers into one coherent evolutionary model:

### 1. Sequence Layer
- Homolog retrieval (optional BLASTp)
- Multiple sequence alignment (MUSCLE / Clustal Omega)
- Occupancy and Shannon entropy
- Mutual information (MI) matrices
- Deep mutational scanning (DMS) integration

### 2. Network Layer
- MI- and DCCM-based residue interaction networks
- Maximal clique analysis
- Community detection (Louvain)

### 3. Structure & Dynamics Layer
- Molecular dynamics trajectory analysis
- RMSD / RMSF
- Dynamic cross-correlation matrices (DCCM)
- Hydrogen-bond analysis
- DSSP secondary-structure mapping
- 2D and 3D free-energy landscapes (FELs)

### 4. Evolutionary Layer (PfPATH Engine)
- Composite fitness model integrating:
  - DMS
  - Thermodynamic stability (ΔΔG)
  - Co-evolution (MI)
  - Structural coupling (DCCM)
- Wright–Fisher population simulations
- Drug on/off and cycling regimes
- Haplotype-specific selection (triple / quadruple DHFR mutants)
- Stochastic adaptive walks in sequence space

---

## 🧠 Core Philosophy

PfPATH is built on three principles:

1. **Biology-aware constraints**  
   Evolution is shaped by structure, stability, and epistasis — not fitness alone.

2. **Modularity**  
   Each layer (sequence, structure, network, evolution) can be run independently.

3. **Reproducibility first**  
   All inputs, parameters, and outputs are logged per run.

---

## 📁 Repository Structure
PfPATH/
├── app.py # Entry point (CLI / future GUI)
├── pfpath/
│ ├── engine/ # PfPATH evolutionary simulator
│ ├── sequence/ # MSA, entropy, MI, DMS
│ ├── structure/ # MD & structural analyses
│ ├── network/ # Clique & community analysis
│ ├── io/ # Validation & path handling
│ └── utils/ # Logging & normalization
├── runs/ # Auto-generated run outputs
├── examples/ # Minimal example datasets
├── environment.yml # Conda environment
├── README.md
└── docs/ # Extended documentation