import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from qmasif_core import QMaSIFSiteModel, QMaSIFQuantumLayer, canonical_sort_batch
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
# 1. Exact Analytically Permutation-Invariant Variant (Priority 3)
# ==============================================================================
class SymmetricReadoutQMaSIF(nn.Module):
    """
    Analytically E(3)- and S_k-permutation-invariant QMaSIF WITHOUT canonical sorting.
    Because QMaSIFQuantumLayer is strictly permutation-equivariant (shared omega, all-to-all
    commuting ZZ(gamma_l / d_ij), shared beta), separating the unique center node (d=0)
    and symmetrically averaging the k-1 neighbor expectations guarantees EXACT 0.00e+00
    permutation error even under adversarial exact key ties.

    Parameter budget (k=4, L=3): 18 quantum + 3 symmetric readout (w_ctr, w_nbr, b) = 21 params.
    """

    def __init__(self, num_qubits: int = 4, num_layers: int = 3,
                 use_entanglement: bool = True, distance_conditioned: bool = True):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.quantum_layer = QMaSIFQuantumLayer(
            num_qubits=num_qubits,
            num_layers=num_layers,
            use_entanglement=use_entanglement,
            distance_conditioned=distance_conditioned,
        )
        if not use_entanglement:
            self.quantum_layer.gamma.requires_grad = False

        self.w_ctr = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.w_nbr = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        B, k, _ = pos.shape
        if center_pos is None:
            center_pos = pos[:, 0, :]

        d_ctr = torch.norm(pos - center_pos.unsqueeze(1), dim=-1)  # [B, k]
        ctr_idx = torch.argmin(d_ctr, dim=1)                       # [B]

        dist_mat = torch.cdist(pos, pos, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        q_out = self.quantum_layer(feats, dist_mat)                # [B, k]

        e_ctr = q_out[torch.arange(B, device=pos.device), ctr_idx]
        e_nbr = (q_out.sum(dim=1) - e_ctr) / max(1, k - 1)
        return self.w_ctr * e_ctr + self.w_nbr * e_nbr + self.bias


# ==============================================================================
# 2. Clean Ablation Variants for L=3 (Accurate Parameter Counting)
# ==============================================================================
class NoEntanglementQMaSIF(QMaSIFSiteModel):
    """Exact U_ent = I ablation with unused gamma parameters frozen (20 params for L=3)."""
    def __init__(self, num_qubits: int = 4, num_layers: int = 3):
        super().__init__(num_qubits=num_qubits, num_layers=num_layers, use_entanglement=False)
        self.quantum_layer.gamma.requires_grad = False


class ShuffledGeometryQMaSIF(QMaSIFSiteModel):
    """Destroys 3D pairwise distance geometry via deterministic cross-sample permutation."""
    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        pos, feats = canonical_sort_batch(pos, feats, center_pos=center_pos)
        dist_mat = torch.cdist(pos, pos, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        gen = torch.Generator(device="cpu").manual_seed(9999)
        perm = torch.randperm(dist_mat.shape[0], generator=gen).to(dist_mat.device)
        q_out = self.quantum_layer(feats, dist_mat[perm])
        return self.readout(q_out).squeeze(-1)


class SingleEncodingQuantumLayer(QMaSIFQuantumLayer):
    """Encodes features ONLY once before variational layers (no data re-uploading)."""
    def __init__(self, num_qubits: int = 4, num_layers: int = 3, eps: float = 1e-5):
        super().__init__(num_qubits=num_qubits, num_layers=num_layers, eps=eps)
        self.omega = nn.Parameter(torch.tensor([[1.2, 2.0, 1.5]], dtype=torch.float32))

    def forward(self, feats: torch.Tensor, dist_mat: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        dim = 1 << self.num_qubits
        device = feats.device

        state = torch.zeros((B, dim), dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j

        for i in range(self.num_qubits):
            q_val = feats[:, i, 0]
            h_val = feats[:, i, 1] - 0.5
            s_val = feats[:, i, 2] - 0.5
            zeros = torch.zeros_like(q_val)
            zero_c = torch.complex(zeros, zeros)

            ry_h = 0.5 * (0.5 * np.pi + self.omega[0, 0] * q_val)
            cy, sy = torch.cos(ry_h), torch.sin(ry_h)
            ry_mat = torch.stack([
                torch.stack([torch.complex(cy, zeros), torch.complex(-sy, zeros)], dim=-1),
                torch.stack([torch.complex(sy, zeros), torch.complex(cy, zeros)], dim=-1),
            ], dim=-2)
            state = self._apply_1q_batched(state, ry_mat, i)

            rz_h = 0.5 * (self.omega[0, 1] * h_val)
            cz, sz = torch.cos(rz_h), torch.sin(rz_h)
            rz_mat = torch.stack([
                torch.stack([torch.complex(cz, -sz), zero_c], dim=-1),
                torch.stack([zero_c, torch.complex(cz, sz)], dim=-1),
            ], dim=-2)
            state = self._apply_1q_batched(state, rz_mat, i)

            rx_h = 0.5 * (self.omega[0, 2] * s_val)
            cx, sx = torch.cos(rx_h), torch.sin(rx_h)
            rx_mat = torch.stack([
                torch.stack([torch.complex(cx, zeros), torch.complex(zeros, -sx)], dim=-1),
                torch.stack([torch.complex(zeros, -sx), torch.complex(cx, zeros)], dim=-1),
            ], dim=-2)
            state = self._apply_1q_batched(state, rx_mat, i)

        for l in range(self.num_layers):
            total_half_phase = torch.zeros((B, dim), dtype=torch.float32, device=device)
            for p_idx, (i, j) in enumerate(self.pairs):
                d_ij = dist_mat[:, i, j] + self.eps
                theta_ij = self.gamma[l] / d_ij
                total_half_phase = total_half_phase + 0.5 * theta_ij.unsqueeze(1) * self.pair_signs[p_idx].unsqueeze(0)
            zz_phase = torch.complex(torch.cos(total_half_phase), -torch.sin(total_half_phase))
            state = state * zz_phase

            rx_h = 0.5 * self.beta[l, 0]
            cx, sx = torch.cos(rx_h), torch.sin(rx_h)
            z0 = torch.zeros_like(cx)
            rx_mix = torch.stack([
                torch.stack([torch.complex(cx, z0), torch.complex(z0, -sx)], dim=-1),
                torch.stack([torch.complex(z0, -sx), torch.complex(cx, z0)], dim=-1),
            ], dim=-2)
            ry_h = 0.5 * self.beta[l, 1]
            cy, sy = torch.cos(ry_h), torch.sin(ry_h)
            ry_mix = torch.stack([
                torch.stack([torch.complex(cy, z0), torch.complex(-sy, z0)], dim=-1),
                torch.stack([torch.complex(sy, z0), torch.complex(cy, z0)], dim=-1),
            ], dim=-2)
            mix_mat = torch.matmul(ry_mix, rx_mix)
            for i in range(self.num_qubits):
                state = self._apply_1q_batched(state, mix_mat, i)

        probs = state.real ** 2 + state.imag ** 2
        return torch.matmul(probs, self.z_signs.t())


class SingleEncodingQMaSIF(QMaSIFSiteModel):
    def __init__(self, num_qubits: int = 4, num_layers: int = 3):
        super().__init__(num_qubits=num_qubits, num_layers=num_layers)
        self.quantum_layer = SingleEncodingQuantumLayer(num_qubits=num_qubits, num_layers=num_layers)


# ==============================================================================
# 3. Priority 3 Adversarial Tie-Breaking Permutation Stress Test
# ==============================================================================
@torch.no_grad()
def run_adversarial_tie_breaking_test():
    """
    Tests permutation invariance under ADVERSARIAL EXACT KEY TIES:
    Neighbors 1 & 2 have IDENTICAL distance to center (5.0A) and IDENTICAL (q, h, s) features,
    while lying at asymmetric distances to neighbor 3 (breaking internal graph symmetry).
    """
    print("\n" + "=" * 84)
    print("PRIORITY 3: ADVERSARIAL EXACT-KEY-TIE PERMUTATION INVARIANCE STRESS TEST")
    print("=" * 84)

    torch.manual_seed(42)
    # Center at origin; nodes 1 & 2 both at radius 5.0A with identical (q, h, s)
    # Node 3 sits closer to node 1 than node 2, so swapping 1 & 2 swaps their roles in sorted order!
    pos = torch.tensor([[[0.0, 0.0, 0.0],
                         [5.0, 0.0, 0.0],
                         [-5.0, 0.0, 0.0],
                         [4.0, 3.0, 0.0]]], dtype=torch.float32)
    feats = torch.tensor([[[0.20, 0.65, 0.40],
                           [-0.35, 0.80, 0.55],   # Identical key to node 2
                           [-0.35, 0.80, 0.55],   # Identical key to node 1
                           [0.10, 0.30, 0.70]]], dtype=torch.float32)
    ctr = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    m_canon = QMaSIFSiteModel(num_qubits=4, num_layers=3, use_canonical_sort=True)
    m_sym = SymmetricReadoutQMaSIF(num_qubits=4, num_layers=3)
    m_canon.eval()
    m_sym.eval()

    # Give canonical readout distinct weights so tie sensitivity is exposed
    m_canon.readout.weight.copy_(torch.tensor([[0.8, 1.2, -0.7, 0.4]]))
    m_canon.readout.bias.copy_(torch.tensor([0.1]))

    base_canon = m_canon(pos, feats, center_pos=ctr).item()
    base_sym = m_sym(pos, feats, center_pos=ctr).item()

    # Test ALL 4! = 24 exact permutations of the 4 nodes!
    import itertools
    max_diff_canon, max_diff_sym = 0.0, 0.0
    for perm in itertools.permutations(range(4)):
        idx = torch.tensor(perm, dtype=torch.long)
        out_c = m_canon(pos[:, idx], feats[:, idx], center_pos=ctr).item()
        out_s = m_sym(pos[:, idx], feats[:, idx], center_pos=ctr).item()
        max_diff_canon = max(max_diff_canon, abs(out_c - base_canon))
        max_diff_sym = max(max_diff_sym, abs(out_s - base_sym))

    # Also test E(3) invariance (SO(3) + 50A translation + reflection) on SymmetricReadoutQMaSIF
    q, r = torch.linalg.qr(torch.randn(3, 3))
    R = q @ torch.diag(torch.sign(torch.diag(r)))
    t = torch.tensor([[[45.0, -32.0, 18.0]]], dtype=torch.float32)
    pos_e3 = torch.matmul(pos, R) + t
    ctr_e3 = torch.matmul(ctr, R) + t.squeeze(1)
    e3_diff_sym = abs(m_sym(pos_e3, feats, center_pos=ctr_e3).item() - base_sym)

    print(f"  [Adversarial 24/24 Permutations w/ Exact Key Ties]")
    print(f"    Canonical-Sort QMaSIF (distinct w_i)  -> Max |Δlogit| = {max_diff_canon:.3e}  (Tie-sensitive!)")
    print(f"    Symmetric-Readout QMaSIF (NO sort)    -> Max |Δlogit| = {max_diff_sym:.3e}  (EXACT S_k invariant!)")
    print(f"  [E(3) Stress Test on Symmetric-Readout QMaSIF (SO(3) + 50Å + Reflection)]")
    print(f"    Symmetric-Readout QMaSIF              -> Max |Δlogit| = {e3_diff_sym:.3e}")
    assert max_diff_sym < 1e-6, f"Symmetric readout permutation error too high: {max_diff_sym}"
    assert e3_diff_sym < 1e-5, f"Symmetric readout E(3) error too high: {e3_diff_sym}"
    print("  STATUS: PASS (Analytic S_k & E(3) invariance proven under all 24 permutations & exact ties!)")

    return {
        "canonical_sort_adversarial_tie_max_diff": float(max_diff_canon),
        "symmetric_readout_adversarial_tie_max_diff": float(max_diff_sym),
        "symmetric_readout_e3_max_diff": float(e3_diff_sym),
    }


def make_shuffled_records(records, seed: int = 777):
    """Deterministically permute node ordering inside every patch to simulate unordered point clouds."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for rec in records:
        new_rec = dict(rec)
        for prefix in ("bal", "full"):
            pos = rec[f"{prefix}_pos"].clone()
            feats = rec[f"{prefix}_feats"].clone()
            B, k, _ = pos.shape
            for b in range(B):
                p = torch.randperm(k, generator=gen)
                pos[b] = pos[b, p]
                feats[b] = feats[b, p]
            new_rec[f"{prefix}_pos"] = pos
            new_rec[f"{prefix}_feats"] = feats
        out.append(new_rec)
    return out


# ==============================================================================
# 4. Main Priority 2 & 3 5-Fold Cluster-Level Ablation Suite (L=3)
# ==============================================================================
def run_step8():
    t_start = time.time()
    print("=" * 84)
    print("PRIORITY 2 & 3: FAIR 5-FOLD CLUSTER-LEVEL ABLATION SUITE AT L=3 (60 PDBs, 37 CLUSTERS)")
    print("=" * 84)

    # 1. Run Priority 3 adversarial tie-breaking verification first
    tie_stats = run_adversarial_tie_breaking_test()

    # 2. Load exact same 60 complexes & 5 homology-independent folds as Priority 1
    records = load_all_60_complexes_per_complex(k=4, radius=8.0)
    clusters, fold_clusters, max_cross_pid = cluster_complexes_and_make_5folds(
        records, identity_thresh=30.0, n_folds=5
    )
    records_shuf = make_shuffled_records(records, seed=777)

    n_complexes = len(records)
    n_clusters = len(clusters)

    results = {}

    # Anchor: Reuse exact Full QMaSIF (L=3, 23p) results from Priority 1 if available (100% consistent)
    if os.path.exists("step7_cluster_cv_results.json"):
        with open("step7_cluster_cv_results.json", "r") as f:
            s7 = json.load(f)
        if "QMaSIF (L=3, 23p)" in s7["models"]:
            results["Full QMaSIF (L=3, 23p)"] = s7["models"]["QMaSIF (L=3, 23p)"]
            print("\n[INFO] Loaded exact anchor 'Full QMaSIF (L=3, 23p)' from 'step7_cluster_cv_results.json'.")

    ablation_specs = [
        ("Symmetric Readout (No Sort, 21p)", lambda: SymmetricReadoutQMaSIF(4, 3), True),
        ("No Entanglement (U_ent=I, 20p)", lambda: NoEntanglementQMaSIF(4, 3), False),
        ("Unconditioned ZZ (Static, 23p)", lambda: QMaSIFSiteModel(4, 3, distance_conditioned=False), False),
        ("Random Geometry (Shuf d_ij, 23p)", lambda: ShuffledGeometryQMaSIF(4, 3), False),
        ("No Data Re-uploading (17p)", lambda: SingleEncodingQMaSIF(4, 3), False),
        ("No Canonical Sort (Unord., 23p)", lambda: QMaSIFSiteModel(4, 3, use_canonical_sort=False), True),
    ]

    if "Full QMaSIF (L=3, 23p)" not in results:
        ablation_specs.insert(0, ("Full QMaSIF (L=3, 23p)", lambda: QMaSIFSiteModel(4, 3), False))

    for name, model_fn, use_shuffled in ablation_specs:
        active_recs = records_shuf if use_shuffled else records
        n_params = sum(p.numel() for p in model_fn().parameters() if p.requires_grad)
        print(f"\n>>> Running 5-Fold Cluster-Level CV: {name} | Params: {n_params} | Shuffled Nodes: {use_shuffled}")

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

            train_t = concat_records(active_recs, train_idx)
            val_t = concat_records(active_recs, val_idx)

            model = train_fold_model(
                model_fn, train_t, val_t,
                seed=42 + f * 100, epochs=65, batch_size=128, lr=0.03, wd=0.0,
            )

            f_p, f_y = [], []
            with torch.no_grad():
                for c_idx in test_idx:
                    rec = active_recs[c_idx]
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
        all_bal_y = np.concatenate([r["bal_labels"].numpy().astype(int) for r in active_recs])
        all_full_p = np.concatenate(oof_full_probs)
        all_full_y = np.concatenate([r["full_labels"].numpy().astype(int) for r in active_recs])

        per_cluster_bal_auc = np.array([
            float(np.mean([per_complex_bal_auc[idx] for idx in grp])) for grp in clusters
        ], dtype=np.float64)

        cl_mean, cl_lo, cl_hi = cluster_bootstrap_ci(per_cluster_bal_auc, n_boot=2000, seed=42)

        results[name] = {
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

    # ==========================================================================
    # 5. Summary Table & Paired Statistical Tests vs Full QMaSIF (L=3)
    # ==========================================================================
    print("\n" + "=" * 108)
    print(f"PRIORITY 2 & 3 ABLATION SUMMARY AT L=3 (5-FOLD CLUSTER CV | {n_complexes} COMPLEXES | {n_clusters} CLUSTERS)")
    print("=" * 108)
    hdr = (
        f"{'Variant':<34} | {'Params':>6} | {'Pooled OOF':>10} | "
        f"{'5-Fold Mean±SD':>14} | {'Cluster Mean [95% CI]':>23} | {'Full-Surf OOF':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        cm, clo, chi = r["cluster_bal_mean_ci95"]
        print(
            f"{name:<34} | {r['params']:>6d} | {r['pooled_bal_roc_auc']:9.2f}% | "
            f"{r['fold_mean_std'][0]:5.2f} ± {r['fold_mean_std'][1]:4.2f}% | "
            f"{cm:5.2f}% [{clo:5.2f}, {chi:5.2f}] | {r['pooled_full_roc_auc']:12.2f}%"
        )

    anchor_key = "Full QMaSIF (L=3, 23p)"
    a_comp = np.array(results[anchor_key]["per_complex_bal_aucs"])
    a_clus = np.array(results[anchor_key]["per_cluster_bal_aucs"])
    stat_vs_full = {}

    print(f"\n[PAIRED CLUSTER & COMPLEX WILCOXON: {anchor_key} vs ABLATED VARIANTS]")
    for name, r in results.items():
        if name == anchor_key:
            continue
        b_comp = np.array(r["per_complex_bal_aucs"])
        b_clus = np.array(r["per_cluster_bal_aucs"])
        _, p_c = safe_wilcoxon(a_comp, b_comp)
        _, p_cl = safe_wilcoxon(a_clus, b_clus)
        dm, dlo, dhi = cluster_bootstrap_diff_ci(a_clus, b_clus, n_boot=2000, seed=42)
        wins = int(np.sum(a_comp > b_comp))
        stat_vs_full[name] = {
            "cluster_delta_ci95": [dm, dlo, dhi],
            "wilcoxon_p_60_complexes": p_c,
            "wilcoxon_p_37_clusters": p_cl,
            "wins_out_of_60": wins,
        }
        print(
            f"  vs {name:<34} | ΔCluster: {dm:+5.2f}% [{dlo:+5.2f}, {dhi:+5.2f}] | "
            f"p(60)={p_c:.4f}, p(37)={p_cl:.4f} | Wins: {wins}/60"
        )

    with open("step8_ablation_cv_results.json", "w") as f:
        json.dump({
            "adversarial_tie_test": tie_stats,
            "variants": results,
            "stat_vs_full": stat_vs_full,
        }, f, indent=2)

    # Clean Horizontal Bar Chart (Cluster Mean + 95% CI & Pooled OOF)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)

    names = list(results.keys())[::-1]
    y_pos = np.arange(len(names))
    pooled_vals = [results[n]["pooled_bal_roc_auc"] for n in names]
    full_vals = [results[n]["pooled_full_roc_auc"] for n in names]
    colors = []
    for n in names:
        if "Full QMaSIF" in n:
            colors.append("#0284C7")
        elif "Symmetric Readout" in n:
            colors.append("#16A34A")
        else:
            colors.append("#94A3B8")

    bars = ax.barh(y_pos, pooled_vals, height=0.58, color=colors, edgecolor="#0F172A", lw=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Out-of-Fold Pooled ROC-AUC Across All 60 PDB Complexes (%)", fontsize=10, fontweight="bold")
    ax.set_xlim(55.0, max(pooled_vals) + 4.5)
    ax.set_title("5-Fold Cluster-Level Architectural Ablation at L=3 (37 Independent Clusters)", fontsize=11.5, fontweight="bold")

    for bar, pv, fv in zip(bars, pooled_vals, full_vals):
        ax.text(pv + 0.25, bar.get_y() + bar.get_height() / 2,
                f"{pv:.2f}% (Full-Surf: {fv:.2f}%)", va="center", ha="left", fontsize=8.8, fontweight="bold")

    plt.tight_layout()
    plt.savefig("Fig_Priority2_Ablation_ClusterCV.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ PRIORITY 2 & 3 COMPLETED in {(time.time() - t_start) / 60.0:.1f} min!")
    print("   Saved: 'step8_ablation_cv_results.json' and 'Fig_Priority2_Ablation_ClusterCV.png'")


if __name__ == "__main__":
    run_step8()