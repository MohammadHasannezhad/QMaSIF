# QMaSIF: E(3)-Invariant Geometric Quantum Learning for Protein Interface Site Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane Verified](https://img.shields.io/badge/PennyLane-Parameter--Shift%20Verified-009688.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official reproducible codebase for **QMaSIF**:  
*QMaSIF: E(3)-Invariant Geometric Quantum Learning via Data Re-uploading SU(2) Rotations and Distance-Conditioned Ising-ZZ Entanglement for Protein Interface Site Prediction*

- **Author:** Mohammad Hasannezhad Soufyani  
- **Affiliation:** Department of Computer Science, University of Tabriz, Tabriz, Iran  
- **Contact:** `Mohammadhasannejad810@gmail.com`

---

## Overview

**QMaSIF** is an ultra-lightweight, strictly E(3)-invariant hybrid quantum-classical architecture for predicting protein-protein interaction (PPI) interface sites directly from unbound monomer structures:

1. **Unbound Monomer Surface Fields:** Deterministic Farthest-Point Sampling (FPS, $R_{\text{cut}} = 8.0\,\text{\AA}$) over exposed side-chain centroids using three Gaussian-smoothed E(3)-invariant fields: Debye-Hückel electrostatics ($q$), interfacial hydrophobicity ($h$), and half-sphere surface exposure ($s$).
2. **Data Re-uploading Geometric VQC ($k=4$ qubits, $L=3$ layers):** Combines shared-scale $SU(2)$ feature re-uploading, all-to-all Euclidean distance-conditioned Ising-$ZZ$ entanglement ($\theta_{ij}^{(l)} = \gamma_l / (d_{ij} + \epsilon)$), and global $SU(2)$ mixing gates that break $Z$-basis commutation.
3. **Dual Permutation-Handling Readouts:**
   - **Flagship Pseudo-Canonical Readout (23 trainable parameters):** Practical permutation invariance on generic continuous protein coordinates ($0.00$ deviation).
   - **Sorting-Free Symmetric Readout (21 trainable parameters):** Analytically $S_k$-permutation invariant across all $4!=24$ permutations ($\le 2.98 \times 10^{-7}$ error even under adversarial exact key ties).
4. **Hardware-Compatible Verification:** Vectorized PyTorch statevector and density-matrix simulators match PennyLane `default.qubit` (Parameter-Shift Rule) and `default.mixed` within float32 machine precision ($\le 3.58 \times 10^{-7}$).

---

## Summary of Primary Benchmark Results

Evaluated via zero-leakage **5-fold cluster-level cross-validation** across all **60 Docking Benchmark 5.5 (DB5.5) complexes** clustered into **37 homology-independent clusters** ($\le 30\%$ BLOSUM62 Smith-Waterman identity; verified max cross-fold identity **26.67%**; 5,154 balanced patches and **17,416 full-surface residues**):

| Model Architecture | Params | Pooled OOF | 5-Fold Mean ± SD | Cluster Mean [95% CI] | Full-Surface OOF ($N=17{,}416$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QMaSIF ($L=3$, Flagship)** | **23** | **65.37%** | **65.60 ± 1.83%** | **64.19% [59.26, 68.90]** | **66.20%** |
| QMaSIF ($L=3$, Symmetric Readout, No Sort) | 21 | 63.80% | 64.37 ± 3.06% | 62.71% [58.13, 67.07] | 64.46% |
| QMaSIF ($L=2$, Shallow) | 17 | 64.37% | 64.34 ± 4.54% | 62.22% [57.58, 66.80] | 64.63% |
| Classical EGNN (E(3)-invariant) | 3,345 (145×) | 63.30% | 63.48 ± 1.58% | 61.52% [56.92, 65.80] | 64.39% |
| Fair Classical MLP (18-dim) | 321 (14×) | 61.64% | 61.78 ± 3.85% | 61.35% [57.04, 65.47] | 61.83% |
| Shared-Param Classical GNN (1:1 Mirror) | 23 (1×) | 61.64% | 61.87 ± 2.87% | 61.11% [56.70, 65.17] | 62.08% |
| Iso-Param Tiny DeepSet MLP | 17 | 61.63% | 62.71 ± 1.98% | 61.39% [57.05, 65.39] | 64.08% |
| Iso-Param Linear (18-dim) | 19 | 61.10% | 62.07 ± 3.00% | 60.81% [56.18, 65.13] | 62.57% |
| Shared-Param Ising-GNN (1:1 Mirror) | 23 (1×) | 60.71% | 61.33 ± 0.87% | 61.13% [56.81, 64.99] | 60.99% |
| Center-Only Logistic | 4 | 59.21% | 60.17 ± 1.64% | 59.85% [55.87, 63.74] | 59.56% |

- **Statistical Rigor:** Significant against all 7 classical baselines at the 60-complex level after Holm–Bonferroni correction ($p_{60}^{\text{Holm}} \le 0.0333$); 6/7 significant unadjusted at the independent 37-cluster level ($p_{37} \le 0.0423$) with strictly positive 95% cluster bootstrap CIs against 5/7 baselines.
- **Key Ablations ($L=3$):** Removing data re-uploading (**−5.55%**, $p_{37}=0.0001$), static distance-blind $ZZ$ (**−3.37%**, $p_{60}=0.0032$), and removing entanglement (**−1.90%**, $p_{60}=0.0469$; fold SD $\pm 1.83\% \to \pm 4.25\%$).
- **3D Interpretability (`1AK4`):** **83.09%** full-surface ROC-AUC ($r = -0.483, p = 5.40 \times 10^{-8}$) and monotonic 3-zone spatial enrichment across 2,855 held-out residues (Mann-Whitney $p = 1.84 \times 10^{-21}$).

---

## Repository Structure

```text
QMaSIF/
├── README.md                                  # Project overview & quickstart
├── requirements.txt                           # Pip dependencies
├── environment.yml                            # Reproducible Conda environment
├── CITATION.cff                               # GitHub native citation metadata
├── LICENSE                                    # MIT License
├── .gitignore                                 # Excludes caches & raw PDB downloads
├── run_all.py                                 # Unified reproduction CLI (--quick / --full / --step N)
├── organize_workspace.py                      # Non-destructive folder organizer & pre-push checker
├── qmasif_core.py                             # Core vectorized PyTorch VQC & canonical sorting
├── step1_verify_core.py                       # PennyLane parity, gradient & symmetry tests (Table 1)
├── step2_build_pdb_dataset.py                 # RCSB PDB download, fields & <=30% BLOSUM62 clustering (Table 2)
├── step3_train_and_benchmark.py               # Protocol B holdout benchmark & SO(3) stress test (Table 6)
├── step4_ablations.py                         # Qubit (k) & depth (L) scaling sweeps (Table 7)
├── step5_nisq_and_bp.py                       # Density-matrix depolarizing noise & shot simulations (Fig. 8)
├── step6_pymol_interpretability.py            # Full-surface inference & 1AK4 PyMOL export (Fig. 9)
├── step7_priority1_cluster_cv.py              # Protocol A 5-fold cluster CV & iso-param baselines (Tables 3-4)
├── step8_priority2_3_ablation_and_symmetry.py # Fair L=3 ablations, Symmetric Readout & 24-perm ties (Table 5)
├── step9_final_reviewer_checks.py             # Matched 23p mirrors, Holm-Bonferroni & 10-patch trainability (Fig. 10)
├── update_fig2_for_overleaf.py                # Generates final 9-model Figure 2
├── qmasif_best_model.pt                       # Saved validation-selected checkpoint (Protocol B)
├── docs/
│   ├── REPRODUCIBILITY.md                     # Detailed table-by-table & figure-by-figure reproduction guide
│   └── DATA_AND_CODE_AVAILABILITY.txt         # Updated manuscript availability text (LaTeX & Word)
├── figures/                                   # Curated publication-grade PNG figures (Figs. 2-10)
├── results/                                   # Archived raw JSON benchmark & statistical outputs
└── pymol/                                     # 1AK4_qmasif_annotated.pdb & visualize_1ak4_qmasif.pml
```

---

## Quickstart

### 1. Environment Setup

**Option A — Conda (Recommended):**
```bash
git clone https://github.com/MohammadHasannezhad/QMaSIF.git
cd QMaSIF
conda env create -f environment.yml
conda activate qmasif
```

**Option B — Pip:**
```bash
git clone https://github.com/MohammadHasannezhad/QMaSIF.git
cd QMaSIF
pip install -r requirements.txt
```

### 2. Run Verification & Reproduction

```bash
# Check workspace integrity (scripts, checkpoint, PyMOL files)
python run_all.py --check

# Quick verification (~1 min): PennyLane parity, dataset build & PyMOL interpretability
python run_all.py --quick

# Full end-to-end reproduction (all Tables 1-7 and Figures 2-10)
python run_all.py --full
```

For exact step-by-step mapping to every table, figure, and runtime estimate, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

### 3. 3D PyMOL Visualization (`1AK4`)

Open PyMOL in the `pymol/` directory and run:
```text
@visualize_1ak4_qmasif.pml
```

---

## Citation

```bibtex
@article{hasannezhad2026qmasif,
  title   = {QMaSIF: E(3)-Invariant Geometric Quantum Learning via Data Re-uploading SU(2) Rotations and Distance-Conditioned Ising-ZZ Entanglement for Protein Interface Site Prediction},
  author  = {Hasannezhad Soufyani, Mohammad},
  year    = {2026},
  note    = {Department of Computer Science, University of Tabriz, Iran}
}
```

## License

Released under the [MIT License](LICENSE).
