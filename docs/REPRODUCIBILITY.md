# Reproducibility Guide — QMaSIF

This document maps every table, figure, and statistical claim in the QMaSIF manuscript to its exact Python script, input requirements, saved JSON artifacts, and approximate CPU runtime.

---

## Hardware & Software Notes

- **Hardware:** All experiments run on standard multi-core CPU without requiring a GPU or external surface-mesh binaries (MSMS/APBS).
- **PDB Structures:** All 60 experimental Docking Benchmark 5.5 (DB5.5) `.pdb` files are downloaded automatically from the RCSB PDB servers during Step 2 and cached locally in `pdb_cache/`.
- **Determinism:** All dataset splits, cross-validation folds, random initializations, and bootstrap resamples use fixed deterministic seeds.

---

## Step-by-Step Mapping to Manuscript Artifacts

| Step | Script | Manuscript Tables / Figures | Output Files Generated | Approx. CPU Time |
| :---: | :--- | :--- | :--- | :---: |
| **1** | `step1_verify_core.py` | **Table 1** (PennyLane parity, non-zero gradients, physical sensitivity, E(3) & generic permutation tests) | Console verification report | ~5 s |
| **2** | `step2_build_pdb_dataset.py` | **Table 2** (60 DB5.5 PDBs, 37 BLOSUM62 $\le 30\%$ clusters, zero-leakage verification) | `qmasif_real_pdb_site_dataset.pt`, `pdb_cache/` | ~15 s (cached) / ~2 min (fresh download) |
| **3** | `step3_train_and_benchmark.py` | **Table 6**, **Figs. 4–6** (Protocol B 5-seed holdout benchmark & $SO(3)+50\,\text{\AA}$ rotational stress test) | `qmasif_best_model.pt`, `step3_benchmark_results.json`, `Fig2_*.png`, `Fig3_*.png`, `Fig4a_*.png` | ~8 min |
| **4** | `step4_ablations.py` | **Table 7**, **Fig. 7** (Protocol B qubit register $k \in \{2,4,6,8\}$ and depth $L \in \{1,2,3,4\}$ scaling sweeps) | `step4_ablation_results.json`, `Fig4b_Ablations_and_Scaling.png` | ~12 min |
| **5** | `step5_nisq_and_bp.py` | **Fig. 8** (Vectorized density-matrix depolarizing noise $p \in [0\%,15\%]$ verified against `default.mixed`, plus finite-shot sampling) | `step5_nisq_bp_results.json`, `Fig5a_NISQ_Noise_and_Shots.png` | ~3 min |
| **6** | `step6_pymol_interpretability.py` | **Fig. 9** (Full-surface inference on 2,855 unseen residues, 3-zone spatial enrichment, `1AK4` PyMOL export) | `step6_interpretability_results.json`, `Fig5bc_Structural_Interpretability.png`, `1AK4_qmasif_annotated.pdb`, `visualize_1ak4_qmasif.pml` | ~20 s |
| **7** | `step7_priority1_cluster_cv.py` | **Tables 3–4** (Protocol A 5-fold cluster-level CV across all 60 complexes & 37 clusters + iso-parameter baselines) | `step7_cluster_cv_results.json`, `Fig_Priority1_ClusterCV_IsoParam.png` | ~20 min |
| **8** | `step8_priority2_3_ablation_and_symmetry.py` | **Table 1** (Adversarial 24-perm tie test), **Table 5**, **Fig. 3** (Fair $L=3$ 5-fold CV ablations & Symmetric Readout) | `step8_ablation_cv_results.json`, `Fig_Priority2_Ablation_ClusterCV.png` | ~45 min |
| **9** | `step9_final_reviewer_checks.py` | **Tables 3–4** (Structurally matched 23p classical mirrors & Holm–Bonferroni $m=7$), **Fig. 10** (10-patch gradient trainability) | `step9_final_reviewer_results.json`, `Fig6_Barren_Plateaus.png` | ~12 min |
| **10** | `update_fig2_for_overleaf.py` | **Fig. 2** (Final 9-model efficiency frontier & 7-baseline paired cluster advantage) | `Fig_Priority1_ClusterCV_IsoParam.png` | ~2 s |

---

## Running Individual or Grouped Steps

```bash
# Verify repository files before running
python run_all.py --check

# Quick smoke test (Steps 1, 2, 6 using saved checkpoint)
python run_all.py --quick

# Run a single step (e.g., Step 9)
python run_all.py --step 9

# Resume from Step 7 through Step 10
python run_all.py --from-step 7

# Full reproduction from scratch (Steps 1 through 10)
python run_all.py --full
```

---

## Verifying Against Saved Reference Outputs

Pre-computed raw numerical outputs from our manuscript runs are archived inside `results/` (`.json` files) alongside the validation-selected Protocol B checkpoint (`qmasif_best_model.pt`). Reviewers can inspect exact per-complex AUCs, per-cluster bootstrap intervals, Holm–Bonferroni adjusted $p$-values, and 10-patch gradient variances directly without re-running multi-fold training.
