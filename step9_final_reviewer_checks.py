import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from qmasif_core import QMaSIFQuantumLayer, canonical_sort_batch
from step7_priority1_cluster_cv import (
    load_all_60_complexes_per_complex,
    cluster_complexes_and_make_5folds,
    concat_records,
    train_fold_model,
    safe_wilcoxon,
    cluster_bootstrap_ci,
    cluster_bootstrap_diff_ci,
)


# ==============================================================================
# 1. Structurally Matched 23-Parameter Classical Counterparts (Exact 1:1 Mirror)
# ==============================================================================
class SharedParamClassicalGNN(nn.Module):
    """
    Exact 23-parameter classical mirror of flagship QMaSIF (k=4, L=3):
      - Same 4 canonically sorted nodes & 3 features (q, h, s)
      - Same L=3 layers of shared-weight data re-uploading (omega: 3x3 = 9 params)
      - Same all-to-all 1/(d_ij + eps) pairwise coupling (gamma: 3 params)
      - Same shared global node mixing (beta: 3x2 = 6 params)
      - Same 5-parameter canonical linear readout (4 weights + 1 bias)
      Total trainable parameters: 9 + 3 + 6 + 5 = 23 (EXACT 1:1 parity with QMaSIF L=3).
    """

    def __init__(self, num_nodes: int = 4, num_layers: int = 3,
                 bilinear_ising: bool = False, eps: float = 1e-5):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.bilinear_ising = bilinear_ising
        self.eps = eps

        self.omega = nn.Parameter(torch.tensor([[1.0, 1.5, 1.2]], dtype=torch.float32).repeat(num_layers, 1))
        self.gamma = nn.Parameter(torch.linspace(0.3, 0.6, num_layers))
        self.beta = nn.Parameter(torch.tensor([[0.8, 0.0]], dtype=torch.float32).repeat(num_layers, 1))
        self.readout = nn.Linear(num_nodes, 1)

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        pos_s, feats_s = canonical_sort_batch(pos, feats, center_pos=center_pos)
        B, k, _ = pos_s.shape

        dist_mat = torch.cdist(pos_s, pos_s, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        off_diag = (~torch.eye(k, dtype=torch.bool, device=pos.device)).unsqueeze(0)
        inv_d = torch.where(off_diag, 1.0 / (dist_mat + self.eps), torch.zeros_like(dist_mat))

        q_f = feats_s[:, :, 0]
        h_f = feats_s[:, :, 1] - 0.5
        s_f = feats_s[:, :, 2] - 0.5

        h = torch.zeros((B, k), dtype=torch.float32, device=pos.device)
        for l in range(self.num_layers):
            # 1. Shared-scale data re-uploading (3 shared params)
            a = torch.tanh(h + self.omega[l, 0] * q_f + self.omega[l, 1] * h_f + self.omega[l, 2] * s_f)

            # 2. All-to-all 1/d_ij pairwise message passing (1 shared param)
            msg = torch.bmm(inv_d, a.unsqueeze(-1)).squeeze(-1)  # sum_{j!=i} a_j / (d_ij + eps)
            if self.bilinear_ising:
                msg = self.gamma[l] * (a * msg)                  # classical Ising-like bilinear product
            else:
                msg = self.gamma[l] * msg                        # standard additive GNN message

            # 3. Shared global node mixing (2 shared params)
            h = torch.tanh(self.beta[l, 0] * a + msg + self.beta[l, 1])

        return self.readout(h).squeeze(-1)


def holm_bonferroni(pvals):
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = pvals[idx] * (m - rank)
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


# ==============================================================================
# 2. Part 1: 5-Fold Cluster-Level CV for Matched 23-Param Classical Models
# ==============================================================================
def run_part1_matched_classical_cv():
    print("=" * 90)
    print("PART 1: 5-FOLD CLUSTER-LEVEL CV FOR STRUCTURALLY MATCHED 23-PARAM CLASSICAL MODELS")
    print("=" * 90)

    records = load_all_60_complexes_per_complex(k=4, radius=8.0)
    clusters, fold_clusters, max_cross_pid = cluster_complexes_and_make_5folds(
        records, identity_thresh=30.0, n_folds=5
    )

    n_complexes = len(records)
    n_clusters = len(clusters)

    new_specs = {
        "Shared-Param Classical GNN (23p)": {
            "fn": lambda: SharedParamClassicalGNN(num_nodes=4, num_layers=3, bilinear_ising=False),
            "lr": 0.03, "wd": 0.0,
        },
        "Shared-Param Ising-GNN (23p)": {
            "fn": lambda: SharedParamClassicalGNN(num_nodes=4, num_layers=3, bilinear_ising=True),
            "lr": 0.03, "wd": 0.0,
        },
    }

    new_results = {}
    for name, spec in new_specs.items():
        n_params = sum(p.numel() for p in spec["fn"]().parameters() if p.requires_grad)
        assert n_params == 23, f"Expected exact 23 params, got {n_params}"
        print(f"\n>>> Running 5-Fold Cluster-Level CV: {name} | Exact Params: {n_params}")

        oof_bal_probs = [None] * n_complexes
        oof_full_probs = [None] * n_complexes
        per_complex_bal_auc = np.zeros(n_complexes, dtype=np.float64)
        per_complex_full_auc = np.zeros(n_complexes, dtype=np.float64)
        fold_bal_aucs = []
        t0 = time.time()

        for f in range(5):
            test_idx = [idx for _, grp in fold_clusters[f] for idx in grp]
            rem_clusters = [grp for f_other in range(5) if f_other != f for _, grp in fold_clusters[f_other]]
            n_val_c = max(4, len(rem_clusters) // 5)
            shift = (f * n_val_c) % len(rem_clusters)
            rem_rot = rem_clusters[shift:] + rem_clusters[:shift]
            val_idx = [idx for grp in rem_rot[:n_val_c] for idx in grp]
            train_idx = [idx for grp in rem_rot[n_val_c:] for idx in grp]

            train_t = concat_records(records, train_idx)
            val_t = concat_records(records, val_idx)

            model = train_fold_model(
                spec["fn"], train_t, val_t,
                seed=42 + f * 100, epochs=65, batch_size=128,
                lr=spec["lr"], wd=spec["wd"],
            )

            f_p, f_y = [], []
            with torch.no_grad():
                for c_idx in test_idx:
                    rec = records[c_idx]
                    bp = torch.sigmoid(model(rec["bal_pos"], rec["bal_feats"], center_pos=rec["bal_ctr"])).numpy()
                    by = rec["bal_labels"].numpy().astype(int)
                    oof_bal_probs[c_idx] = bp
                    per_complex_bal_auc[c_idx] = roc_auc_score(by, bp) * 100.0
                    f_p.append(bp)
                    f_y.append(by)

                    fp = torch.sigmoid(model(rec["full_pos"], rec["full_feats"], center_pos=rec["full_ctr"])).numpy()
                    fy = rec["full_labels"].numpy().astype(int)
                    oof_full_probs[c_idx] = fp
                    per_complex_full_auc[c_idx] = roc_auc_score(fy, fp) * 100.0

            fold_auc = roc_auc_score(np.concatenate(f_y), np.concatenate(f_p)) * 100.0
            fold_bal_aucs.append(float(fold_auc))
            print(f"    Fold {f+1}/5 | Fold Balanced ROC-AUC: {fold_auc:5.2f}%")

        all_bal_p = np.concatenate(oof_bal_probs)
        all_bal_y = np.concatenate([r["bal_labels"].numpy().astype(int) for r in records])
        all_full_p = np.concatenate(oof_full_probs)
        all_full_y = np.concatenate([r["full_labels"].numpy().astype(int) for r in records])

        per_cluster_bal_auc = np.array([
            float(np.mean([per_complex_bal_auc[idx] for idx in grp])) for grp in clusters
        ], dtype=np.float64)
        cl_mean, cl_lo, cl_hi = cluster_bootstrap_ci(per_cluster_bal_auc, n_boot=2000, seed=42)

        new_results[name] = {
            "params": n_params,
            "time_s": float(time.time() - t0),
            "fold_bal_aucs": fold_bal_aucs,
            "fold_mean_std": [float(np.mean(fold_bal_aucs)), float(np.std(fold_bal_aucs, ddof=1))],
            "pooled_bal_roc_auc": float(roc_auc_score(all_bal_y, all_bal_p) * 100.0),
            "pooled_bal_pr_auc": float(average_precision_score(all_bal_y, all_bal_p) * 100.0),
            "pooled_full_roc_auc": float(roc_auc_score(all_full_y, all_full_p) * 100.0),
            "cluster_bal_mean_ci95": [cl_mean, cl_lo, cl_hi],
            "per_complex_bal_aucs": per_complex_bal_auc.tolist(),
            "per_cluster_bal_aucs": per_cluster_bal_auc.tolist(),
        }

    # Combine with saved Step 7 results & compute Holm-Bonferroni across all baselines
    with open("step7_cluster_cv_results.json", "r") as f:
        s7 = json.load(f)

    all_models = dict(s7["models"])
    all_models.update(new_results)

    q_name = "QMaSIF (L=3, 23p)"
    q_comp = np.array(all_models[q_name]["per_complex_bal_aucs"])
    q_clus = np.array(all_models[q_name]["per_cluster_bal_aucs"])

    baselines_order = [
        "Center-Only Logistic (4p)",
        "Iso-Param Linear (19p)",
        "Iso-Param Tiny MLP (17p)",
        "Shared-Param Classical GNN (23p)",
        "Shared-Param Ising-GNN (23p)",
        "Fair Classical MLP (321p)",
        "Classical EGNN (3345p)",
    ]

    print("\n" + "=" * 112)
    print("COMPLETE 5-FOLD CLUSTER-LEVEL CV SUMMARY (INCLUDING STRUCTURALLY MATCHED 23p CLASSICAL MIRRORS)")
    print("=" * 112)
    hdr = f"{'Model Architecture':<34} | {'Params':>6} | {'Pooled OOF':>10} | {'5-Fold Mean±SD':>14} | {'Cluster Mean [95% CI]':>23} | {'Full-Surf OOF':>13}"
    print(hdr)
    print("-" * len(hdr))
    for m_name in ["QMaSIF (L=3, 23p)", "QMaSIF (L=2, 17p)"] + baselines_order:
        r = all_models[m_name]
        cm, clo, chi = r["cluster_bal_mean_ci95"]
        print(
            f"{m_name:<34} | {r['params']:>6d} | {r['pooled_bal_roc_auc']:9.2f}% | "
            f"{r['fold_mean_std'][0]:5.2f} ± {r['fold_mean_std'][1]:4.2f}% | "
            f"{cm:5.2f}% [{clo:5.2f}, {chi:5.2f}] | {r['pooled_full_roc_auc']:12.2f}%"
        )

    raw_p60, raw_p37, stats_rows = [], [], {}
    for b_name in baselines_order:
        b_comp = np.array(all_models[b_name]["per_complex_bal_aucs"])
        b_clus = np.array(all_models[b_name]["per_cluster_bal_aucs"])
        _, p60 = safe_wilcoxon(q_comp, b_comp)
        _, p37 = safe_wilcoxon(q_clus, b_clus)
        dm, dlo, dhi = cluster_bootstrap_diff_ci(q_clus, b_clus, n_boot=2000, seed=42)
        wins = int(np.sum(q_comp > b_comp))
        raw_p60.append(p60)
        raw_p37.append(p37)
        stats_rows[b_name] = {
            "cluster_delta_ci95": [dm, dlo, dhi],
            "p60_raw": p60,
            "p37_raw": p37,
            "wins_60": wins,
        }

    holm_p60 = holm_bonferroni(raw_p60)
    holm_p37 = holm_bonferroni(raw_p37)
    for b_name, hp60, hp37 in zip(baselines_order, holm_p60, holm_p37):
        stats_rows[b_name]["p60_holm"] = hp60
        stats_rows[b_name]["p37_holm"] = hp37

    print(f"\n[PAIRED WILCOXON + HOLM-BONFERRONI (m={len(baselines_order)}): Flagship QMaSIF (L=3, 23p) vs ALL CLASSICAL BASELINES]")
    print(f"{'Baseline':<34} | {'ΔCluster [95% CI]':>22} | {'p60 (Holm)':>16} | {'p37 (Holm)':>16} | {'Wins':>6}")
    print("-" * 108)
    for b_name in baselines_order:
        sr = stats_rows[b_name]
        dm, dlo, dhi = sr["cluster_delta_ci95"]
        s60 = "*" if sr["p60_holm"] < 0.05 else " "
        s37 = "*" if sr["p37_holm"] < 0.05 else " "
        print(
            f"{b_name:<34} | {dm:+5.2f}% [{dlo:+5.2f}, {dhi:+5.2f}] | "
            f"{sr['p60_raw']:.4f} ({sr['p60_holm']:.4f}{s60}) | "
            f"{sr['p37_raw']:.4f} ({sr['p37_holm']:.4f}{s37}) | {sr['wins_60']:2d}/60"
        )

    return new_results, stats_rows


# ==============================================================================
# 3. Part 2: Multi-Patch Empirical Gradient Trainability Analysis (10 Diverse Patches)
# ==============================================================================
def select_10_diverse_patches(test_dict):
    """Select 5 interface + 5 non-interface patches evenly spanning the unseen test set."""
    labels = test_dict["labels"].numpy().astype(int)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    chosen_pos = pos_idx[np.linspace(0, len(pos_idx) - 1, 5, dtype=int)]
    chosen_neg = neg_idx[np.linspace(0, len(neg_idx) - 1, 5, dtype=int)]
    chosen = np.concatenate([chosen_pos, chosen_neg])

    pos_10 = test_dict["pos"][chosen].clone()
    feats_10 = test_dict["feats"][chosen].clone()
    ctr_10 = test_dict["center"][chosen].clone()
    lbl_10 = labels[chosen].tolist()

    print(f"\n[PATCHES] Selected 10 diverse unseen PDB patches (5 interface + 5 non-interface):")
    for idx_i, (g_idx, y) in enumerate(zip(chosen, lbl_10)):
        f0 = feats_10[idx_i, 0].numpy()
        print(f"    Patch {idx_i+1:2d} (test idx {g_idx:3d}, y={y}) | center (q={f0[0]:+.2f}, h={f0[1]:.2f}, s={f0[2]:.2f})")
    return pos_10, feats_10, ctr_10, lbl_10


def compute_multipatch_grad_variance(pos_10, feats_10, ctr_10, k: int, L: int,
                                     n_inits: int = 100, seed: int = 42):
    """
    Computes Var[d<Z_0>/d gamma_0] and Var[d<Z_0>/d beta_{0,x}] over `n_inits` uniform
    random initializations in [-pi, pi] independently for EACH of the 10 diverse patches.
    """
    pos_k, feats_k = canonical_sort_batch(pos_10[:, :k, :], feats_10[:, :k, :], center_pos=ctr_10)
    dist_k = torch.cdist(pos_k, pos_k, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    n_patches = pos_k.shape[0]

    qlayer = QMaSIFQuantumLayer(num_qubits=k, num_layers=L)
    rng = torch.Generator().manual_seed(seed + k * 31 + L * 7)

    grads_gamma = np.zeros((n_patches, n_inits), dtype=np.float64)
    grads_beta = np.zeros((n_patches, n_inits), dtype=np.float64)

    for s in range(n_inits):
        with torch.no_grad():
            qlayer.omega.uniform_(-np.pi, np.pi, generator=rng)
            qlayer.gamma.uniform_(-np.pi, np.pi, generator=rng)
            qlayer.beta.uniform_(-np.pi, np.pi, generator=rng)

        q_out = qlayer(feats_k, dist_k)  # [10, k]
        for p in range(n_patches):
            g_gam, g_bet = torch.autograd.grad(
                q_out[p, 0],
                (qlayer.gamma, qlayer.beta),
                retain_graph=(p < n_patches - 1),
            )
            grads_gamma[p, s] = float(g_gam[0].item())
            grads_beta[p, s] = float(g_bet[0, 0].item())

    per_patch_var_g = np.var(grads_gamma, axis=1, ddof=1)
    per_patch_var_b = np.var(grads_beta, axis=1, ddof=1)

    return {
        "var_gamma_mean": float(np.mean(per_patch_var_g)),
        "var_gamma_std": float(np.std(per_patch_var_g, ddof=1)),
        "var_gamma_min": float(np.min(per_patch_var_g)),
        "var_gamma_max": float(np.max(per_patch_var_g)),
        "var_beta_mean": float(np.mean(per_patch_var_b)),
        "var_beta_std": float(np.std(per_patch_var_b, ddof=1)),
        "var_beta_min": float(np.min(per_patch_var_b)),
        "var_beta_max": float(np.max(per_patch_var_b)),
        "per_patch_var_gamma": per_patch_var_g.tolist(),
        "per_patch_var_beta": per_patch_var_b.tolist(),
    }


def run_part2_multipatch_trainability():
    print("\n" + "=" * 90)
    print("PART 2: EMPIRICAL GRADIENT TRAINABILITY ANALYSIS (10 DIVERSE PATCHES x 100 INITS = 1,000 EVALS/PT)")
    print("=" * 90)

    data = torch.load("qmasif_real_pdb_site_dataset.pt", weights_only=False)
    pos_10, feats_10, ctr_10, _ = select_10_diverse_patches(data["test"])

    print("\n--- Depth Sweep (k=4, L in 1..8) ---")
    depth_stats = {}
    for L in range(1, 9):
        st = compute_multipatch_grad_variance(pos_10, feats_10, ctr_10, k=4, L=L, n_inits=100, seed=42)
        depth_stats[str(L)] = st
        print(
            f"  L={L} | ZZ Var[∂<Z0>/∂γ0]: {st['var_gamma_mean']:.3e} "
            f"(range [{st['var_gamma_min']:.2e}, {st['var_gamma_max']:.2e}]) | "
            f"RX Var[∂<Z0>/∂β0]: {st['var_beta_mean']:.3e}"
        )

    print("\n--- Qubit Width Sweep (k in {2,4,6,8}, L=2 and L=3) ---")
    qubit_stats = {"L2": {}, "L3": {}}
    for L_tag, L_val in [("L2", 2), ("L3", 3)]:
        for k in [2, 4, 6, 8]:
            st = compute_multipatch_grad_variance(pos_10, feats_10, ctr_10, k=k, L=L_val, n_inits=100, seed=100)
            qubit_stats[L_tag][str(k)] = st
            print(
                f"  L={L_val}, k={k} | ZZ Var: {st['var_gamma_mean']:.3e} "
                f"(range [{st['var_gamma_min']:.2e}, {st['var_gamma_max']:.2e}]) | "
                f"RX Var: {st['var_beta_mean']:.3e}"
            )

    # Generate Updated Publication-Grade Fig6_Barren_Plateaus.png (Mean + Min-Max Shaded Band across 10 Patches)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0), dpi=300)

    ls = list(range(1, 9))
    vg_m = [depth_stats[str(L)]["var_gamma_mean"] for L in ls]
    vg_lo = [depth_stats[str(L)]["var_gamma_min"] for L in ls]
    vg_hi = [depth_stats[str(L)]["var_gamma_max"] for L in ls]
    vb_m = [depth_stats[str(L)]["var_beta_mean"] for L in ls]
    vb_lo = [depth_stats[str(L)]["var_beta_min"] for L in ls]
    vb_hi = [depth_stats[str(L)]["var_beta_max"] for L in ls]

    ax1.semilogy(ls, vg_m, "-o", color="#E11D48", lw=2.4, markersize=6.5, label=r"$\mathrm{Var}[\partial \langle Z_0 \rangle / \partial \gamma_0]$ (ZZ, 10-Patch Mean)")
    ax1.fill_between(ls, vg_lo, vg_hi, color="#E11D48", alpha=0.18, label="10-Patch Min–Max Range (ZZ)")
    ax1.semilogy(ls, vb_m, "--s", color="#0284C7", lw=2.2, markersize=6, label=r"$\mathrm{Var}[\partial \langle Z_0 \rangle / \partial \beta_{0,x}]$ (RX, 10-Patch Mean)")
    ax1.fill_between(ls, vb_lo, vb_hi, color="#0284C7", alpha=0.14)
    ax1.axvspan(1, 3, color="#16A34A", alpha=0.09, label=r"Shallow Operating Regime ($L \leq 3$)")
    ax1.set_ylim(2e-4, 3.0e-1)
    ax1.set_xlabel("Circuit Layer Depth ($L$)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Gradient Variance (Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Gradient Trainability vs. Depth ($k=4$, 10 Diverse Patches)", fontsize=11, fontweight="bold")
    ax1.legend(loc="center right", frameon=True, fontsize=8.3, facecolor="white", framealpha=0.95)
    ax1.grid(True, which="both", ls="--", alpha=0.45)

    ks = [2, 4, 6, 8]
    vg2_m = [qubit_stats["L2"][str(k)]["var_gamma_mean"] for k in ks]
    vg3_m = [qubit_stats["L3"][str(k)]["var_gamma_mean"] for k in ks]
    vg3_lo = [qubit_stats["L3"][str(k)]["var_gamma_min"] for k in ks]
    vg3_hi = [qubit_stats["L3"][str(k)]["var_gamma_max"] for k in ks]
    vb3_m = [qubit_stats["L3"][str(k)]["var_beta_mean"] for k in ks]

    ax2.semilogy(ks, vg2_m, "-o", color="#E11D48", lw=2.3, markersize=6.5, label="ZZ Grad Var ($L=2$, 10-Patch Mean)")
    ax2.semilogy(ks, vg3_m, "-^", color="#9333EA", lw=2.4, markersize=7, label="ZZ Grad Var ($L=3$, 10-Patch Mean)")
    ax2.fill_between(ks, vg3_lo, vg3_hi, color="#9333EA", alpha=0.16, label="10-Patch Min–Max Range ($L=3$)")
    ax2.semilogy(ks, vb3_m, "--s", color="#0284C7", lw=2.2, markersize=6, label="RX Grad Var ($L=3$, 10-Patch Mean)")
    ax2.set_xticks(ks)
    ax2.set_ylim(8e-5, 3.0e-1)
    ax2.set_xlabel("Number of Qubits ($k$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Gradient Variance (Log Scale)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Gradient Trainability vs. Qubit Width (10 Diverse Patches)", fontsize=11, fontweight="bold")
    ax2.legend(loc="center left", bbox_to_anchor=(0.02, 0.56), frameon=True, fontsize=8.3, facecolor="white", framealpha=0.95)
    ax2.grid(True, which="both", ls="--", alpha=0.45)

    plt.tight_layout()
    plt.savefig("Fig6_Barren_Plateaus.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\n  [OK] Updated 'Fig6_Barren_Plateaus.png' with 10-patch mean & min-max bands.")

    return {"depth_sweep": depth_stats, "qubit_sweep": qubit_stats}


if __name__ == "__main__":
    t0 = time.time()
    new_models, stats_holm = run_part1_matched_classical_cv()
    grad_stats = run_part2_multipatch_trainability()

    with open("step9_final_reviewer_results.json", "w") as f:
        json.dump({
            "matched_23p_classical_models": new_models,
            "wilcoxon_and_holm_bonferroni": stats_holm,
            "multipatch_gradient_trainability": grad_stats,
        }, f, indent=2)

    print(f"\n✅ STEP 9 COMPLETED in {time.time() - t0:.1f}s!")
    print("   Saved: 'step9_final_reviewer_results.json' and updated 'Fig6_Barren_Plateaus.png'")