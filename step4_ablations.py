import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from qmasif_core import QMaSIFSiteModel, QMaSIFQuantumLayer, canonical_sort_batch


# ==============================================================================
# 1. Ablation Model Variants (Without Modifying qmasif_core.py)
# ==============================================================================
class ShuffledGeometryQMaSIF(QMaSIFSiteModel):
    """Ablation: Breaks 3D spatial logic by randomly shuffling pairwise distances across the batch."""
    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        pos, feats = canonical_sort_batch(pos, feats, center_pos=center_pos)
        dist_mat = torch.cdist(pos, pos, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        # Permute distance matrices across batch samples (preserves marginal distribution, destroys geometry)
        gen = torch.Generator(device="cpu").manual_seed(9999)
        perm = torch.randperm(dist_mat.shape[0], generator=gen).to(dist_mat.device)
        dist_shuffled = dist_mat[perm]
        q_out = self.quantum_layer(feats, dist_shuffled)
        return self.readout(q_out).squeeze(-1)


class SingleEncodingQuantumLayer(QMaSIFQuantumLayer):
    """Ablation: Encodes physicochemical features ONLY once at l=0 (No Data Re-uploading)."""
    def forward(self, feats: torch.Tensor, dist_mat: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        dim = 1 << self.num_qubits
        device = feats.device

        state = torch.zeros((B, dim), dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j

        # Encode ONLY once before variational layers
        for i in range(self.num_qubits):
            q_val = feats[:, i, 0]
            h_val = feats[:, i, 1] - 0.5
            s_val = feats[:, i, 2] - 0.5
            zeros = torch.zeros_like(q_val)
            zero_c = torch.complex(zeros, zeros)

            ry_half = 0.5 * (0.5 * np.pi + self.omega[0, 0] * q_val)
            c_y, s_y = torch.cos(ry_half), torch.sin(ry_half)
            ry_mat = torch.stack([
                torch.stack([torch.complex(c_y, zeros), torch.complex(-s_y, zeros)], dim=-1),
                torch.stack([torch.complex(s_y, zeros), torch.complex(c_y, zeros)], dim=-1),
            ], dim=-2)
            state = self._apply_1q_batched(state, ry_mat, i)

            rz_half = 0.5 * (self.omega[0, 1] * h_val)
            c_z, s_z = torch.cos(rz_half), torch.sin(rz_half)
            rz_mat = torch.stack([
                torch.stack([torch.complex(c_z, -s_z), zero_c], dim=-1),
                torch.stack([zero_c, torch.complex(c_z, s_z)], dim=-1),
            ], dim=-2)
            state = self._apply_1q_batched(state, rz_mat, i)

            rx_half = 0.5 * (self.omega[0, 2] * s_val)
            c_x, s_x = torch.cos(rx_half), torch.sin(rx_half)
            rx_mat = torch.stack([
                torch.stack([torch.complex(c_x, zeros), torch.complex(zeros, -s_x)], dim=-1),
                torch.stack([torch.complex(zeros, -s_x), torch.complex(c_x, zeros)], dim=-1),
            ], dim=-2)
            state = self._apply_1q_batched(state, rx_mat, i)

        for l in range(self.num_layers):
            if self.use_entanglement and len(self.pairs) > 0:
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
    def __init__(self, num_qubits: int = 4, num_layers: int = 2):
        super().__init__(num_qubits=num_qubits, num_layers=num_layers)
        self.quantum_layer = SingleEncodingQuantumLayer(num_qubits=num_qubits, num_layers=num_layers)


# ==============================================================================
# 2. Fast, Exact Training & Evaluation Helper
# ==============================================================================
def get_split(split_dict, k: int, shuffle_nodes: bool = False, seed: int = 777):
    pos = split_dict["pos"][:, :k, :].clone()
    feats = split_dict["feats"][:, :k, :].clone()
    center = split_dict["center"].clone()
    labels = split_dict["labels"].clone()

    if shuffle_nodes:
        # Simulate unordered point-cloud patches (random permutation per sample)
        gen = torch.Generator().manual_seed(seed)
        B = pos.shape[0]
        for b in range(B):
            perm = torch.randperm(k, generator=gen)
            pos[b] = pos[b, perm]
            feats[b] = feats[b, perm]
    return pos, feats, center, labels


@torch.no_grad()
def eval_auc(model, split_t):
    model.eval()
    pos, feats, center, labels = split_t
    probs = torch.sigmoid(model(pos, feats, center_pos=center)).cpu().numpy()
    y = labels.cpu().numpy().astype(int)
    return float(roc_auc_score(y, probs) * 100.0), float(average_precision_score(y, probs) * 100.0)


def run_config(model_fn, train_t, val_t, test_t, seeds=(42, 101, 202),
               epochs: int = 55, batch_size: int = 256, lr: float = 0.035):
    roc_list, pr_list, time_list = [], [], []
    n_params = sum(p.numel() for p in model_fn().parameters() if p.requires_grad)

    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()

        loader = DataLoader(TensorDataset(*train_t), batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(s))

        best_val = -1.0
        best_state = None
        t0 = time.time()

        for ep in range(epochs):
            model.train()
            for pos, feats, center, y in loader:
                optimizer.zero_grad()
                loss = criterion(model(pos, feats, center_pos=center), y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (ep + 1) % 2 == 0 or ep == epochs - 1:
                val_roc, _ = eval_auc(model, val_t)
                if val_roc > best_val:
                    best_val = val_roc
                    best_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - t0
        model.load_state_dict(best_state)
        test_roc, test_pr = eval_auc(model, test_t)
        roc_list.append(test_roc)
        pr_list.append(test_pr)
        time_list.append(elapsed)

    return {
        "params": n_params,
        "roc_mean": float(np.mean(roc_list)),
        "roc_std": float(np.std(roc_list, ddof=1)),
        "pr_mean": float(np.mean(pr_list)),
        "pr_std": float(np.std(pr_list, ddof=1)),
        "time_mean": float(np.mean(time_list)),
        "seeds_roc": roc_list,
    }


# ==============================================================================
# 3. Main Execution of All 3 Ablation Suites
# ==============================================================================
def run_all_ablations():
    print("=" * 78)
    print("STEP 4: SYSTEMATIC ABLATION & SCALING STUDIES ON REAL PDB DATASET")
    print("=" * 78)

    data = torch.load("qmasif_real_pdb_site_dataset.pt", weights_only=False)
    seeds = (42, 101, 202)

    tr4 = get_split(data["train"], k=4)
    va4 = get_split(data["val"], k=4)
    te4 = get_split(data["test"], k=4)

    # ------------------------------------------------------------------
    # PART A: Architectural Component Ablations (k=4, L=2)
    # ------------------------------------------------------------------
    print("\n--- [PART A] Architectural Component Ablations (k=4, L=2, 3 Seeds) ---")
    ablation_results = {}

    # 1. Full QMaSIF (reused as anchor for Parts B & C)
    full_res = run_config(lambda: QMaSIFSiteModel(num_qubits=4, num_layers=2), tr4, va4, te4, seeds=seeds)
    ablation_results["Full QMaSIF (Ours)"] = full_res
    print(f"  Full QMaSIF (Ours)          | ROC: {full_res['roc_mean']:5.2f} ± {full_res['roc_std']:4.2f}% | PR: {full_res['pr_mean']:5.2f}% | Time: {full_res['time_mean']:5.1f}s")

    # 2. No Entanglement (U_ent = I)
    res_noent = run_config(lambda: QMaSIFSiteModel(num_qubits=4, num_layers=2, use_entanglement=False), tr4, va4, te4, seeds=seeds)
    ablation_results["No Entanglement (U_ent=I)"] = res_noent
    print(f"  No Entanglement (U_ent=I)   | ROC: {res_noent['roc_mean']:5.2f} ± {res_noent['roc_std']:4.2f}% | PR: {res_noent['pr_mean']:5.2f}% | Time: {res_noent['time_mean']:5.1f}s")

    # 3. Unconditioned ZZ (Static, no 1/d_ij)
    res_uncond = run_config(lambda: QMaSIFSiteModel(num_qubits=4, num_layers=2, distance_conditioned=False), tr4, va4, te4, seeds=seeds)
    ablation_results["Unconditioned ZZ (Static)"] = res_uncond
    print(f"  Unconditioned ZZ (Static)   | ROC: {res_uncond['roc_mean']:5.2f} ± {res_uncond['roc_std']:4.2f}% | PR: {res_uncond['pr_mean']:5.2f}% | Time: {res_uncond['time_mean']:5.1f}s")

    # 4. Random Geometry (Shuffled d_ij)
    res_randg = run_config(lambda: ShuffledGeometryQMaSIF(num_qubits=4, num_layers=2), tr4, va4, te4, seeds=seeds)
    ablation_results["Random Geometry (Shuffled d)"] = res_randg
    print(f"  Random Geometry (Shuffled d)| ROC: {res_randg['roc_mean']:5.2f} ± {res_randg['roc_std']:4.2f}% | PR: {res_randg['pr_mean']:5.2f}% | Time: {res_randg['time_mean']:5.1f}s")

    # 5. No Data Re-uploading (Single Encoding)
    res_noreup = run_config(lambda: SingleEncodingQMaSIF(num_qubits=4, num_layers=2), tr4, va4, te4, seeds=seeds)
    ablation_results["No Data Re-uploading"] = res_noreup
    print(f"  No Data Re-uploading        | ROC: {res_noreup['roc_mean']:5.2f} ± {res_noreup['roc_std']:4.2f}% | PR: {res_noreup['pr_mean']:5.2f}% | Time: {res_noreup['time_mean']:5.1f}s")

    # 6. No Canonical Sort on Unordered Point-Cloud Input
    tr4_shuf = get_split(data["train"], k=4, shuffle_nodes=True, seed=11)
    va4_shuf = get_split(data["val"], k=4, shuffle_nodes=True, seed=22)
    te4_shuf = get_split(data["test"], k=4, shuffle_nodes=True, seed=33)
    res_nosort = run_config(lambda: QMaSIFSiteModel(num_qubits=4, num_layers=2, use_canonical_sort=False),
                            tr4_shuf, va4_shuf, te4_shuf, seeds=seeds)
    ablation_results["No Canonical Sort (Unordered)"] = res_nosort
    print(f"  No Canonical Sort (Unord.)  | ROC: {res_nosort['roc_mean']:5.2f} ± {res_nosort['roc_std']:4.2f}% | PR: {res_nosort['pr_mean']:5.2f}% | Time: {res_nosort['time_mean']:5.1f}s")

    # ------------------------------------------------------------------
    # PART B: Qubit Register Size Scaling (k in [2, 4, 6, 8], L=2)
    # ------------------------------------------------------------------
    print("\n--- [PART B] Qubit Register Size Scaling (k in [2, 4, 6, 8], L=2) ---")
    qubit_scaling = {}
    for k in [2, 4, 6, 8]:
        if k == 4:
            res = full_res
        else:
            tr_k = get_split(data["train"], k=k)
            va_k = get_split(data["val"], k=k)
            te_k = get_split(data["test"], k=k)
            res = run_config(lambda kk=k: QMaSIFSiteModel(num_qubits=kk, num_layers=2),
                             tr_k, va_k, te_k, seeds=seeds)
        qubit_scaling[str(k)] = res
        print(f"  k={k} Qubits (Params={res['params']:2d}) | ROC: {res['roc_mean']:5.2f} ± {res['roc_std']:4.2f}% | PR: {res['pr_mean']:5.2f}% | Time: {res['time_mean']:6.1f}s")

    # ------------------------------------------------------------------
    # PART C: Circuit Depth Scaling (L in [1, 2, 3, 4], k=4)
    # ------------------------------------------------------------------
    print("\n--- [PART C] Circuit Layer Depth Scaling (L in [1, 2, 3, 4], k=4) ---")
    depth_scaling = {}
    for L in [1, 2, 3, 4]:
        if L == 2:
            res = full_res
        else:
            res = run_config(lambda ll=L: QMaSIFSiteModel(num_qubits=4, num_layers=ll),
                             tr4, va4, te4, seeds=seeds)
        depth_scaling[str(L)] = res
        print(f"  L={L} Layers (Params={res['params']:2d}) | ROC: {res['roc_mean']:5.2f} ± {res['roc_std']:4.2f}% | PR: {res['pr_mean']:5.2f}% | Time: {res['time_mean']:6.1f}s")

    # Save JSON
    out_json = {
        "architectural_ablations": ablation_results,
        "qubit_scaling": qubit_scaling,
        "depth_scaling": depth_scaling,
    }
    with open("step4_ablation_results.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # ------------------------------------------------------------------
    # Plot Publication-Quality 3-Panel Figure (Fig4b_Ablations_and_Scaling.png)
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.0), dpi=300)

    # Panel A: Architectural Component Ablations
    abl_names = list(ablation_results.keys())[::-1]
    abl_means = [ablation_results[n]["roc_mean"] for n in abl_names]
    abl_stds = [ablation_results[n]["roc_std"] for n in abl_names]
    bar_colors = ["#94A3B8"] * (len(abl_names) - 1) + ["#0284C7"]

    y_pos = np.arange(len(abl_names))
    bars = ax1.barh(y_pos, abl_means, xerr=abl_stds, capsize=3, color=bar_colors, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(abl_names, fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Unseen PDB Test ROC-AUC (%)", fontsize=9.5, fontweight="bold")
    ax1.set_xlim(max(45, min(abl_means) - 6), max(abl_means) + 5)
    ax1.set_title("(a) Architectural Component Ablations", fontsize=11, fontweight="bold")
    for bar, m in zip(bars, abl_means):
        ax1.text(m + 0.6, bar.get_y() + bar.get_height() / 2, f"{m:.2f}%", va="center", fontsize=8.5, fontweight="bold")

    # Panel B: Qubit Scaling (k)
    ks = [2, 4, 6, 8]
    k_rocs = [qubit_scaling[str(k)]["roc_mean"] for k in ks]
    k_stds = [qubit_scaling[str(k)]["roc_std"] for k in ks]
    k_times = [qubit_scaling[str(k)]["time_mean"] for k in ks]

    ax2.errorbar(ks, k_rocs, yerr=k_stds, fmt="-o", color="#0284C7", lw=2.4, capsize=4, label="Test ROC-AUC (%)")
    ax2.set_xticks(ks)
    ax2.set_xlabel("Number of Qubits (k)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Unseen PDB Test ROC-AUC (%)", fontsize=9.5, fontweight="bold", color="#0284C7")
    ax2.tick_params(axis="y", labelcolor="#0284C7")
    ax2.set_title("(b) Qubit Register Scaling (k)", fontsize=11, fontweight="bold")

    ax2_t = ax2.twinx()
    ax2_t.plot(ks, k_times, "--^", color="#E11D48", lw=2.0, label="Train Time (s)")
    ax2_t.set_ylabel("Training Time (s)", fontsize=9.5, fontweight="bold", color="#E11D48")
    ax2_t.tick_params(axis="y", labelcolor="#E11D48")
    ax2_t.grid(False)

    # Panel C: Circuit Depth Scaling (L)
    ls = [1, 2, 3, 4]
    l_rocs = [depth_scaling[str(L)]["roc_mean"] for L in ls]
    l_stds = [depth_scaling[str(L)]["roc_std"] for L in ls]
    l_times = [depth_scaling[str(L)]["time_mean"] for L in ls]

    ax3.errorbar(ls, l_rocs, yerr=l_stds, fmt="-s", color="#16A34A", lw=2.4, capsize=4, label="Test ROC-AUC (%)")
    ax3.set_xticks(ls)
    ax3.set_xlabel("Variational Ansatz Depth (L)", fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Unseen PDB Test ROC-AUC (%)", fontsize=9.5, fontweight="bold", color="#16A34A")
    ax3.tick_params(axis="y", labelcolor="#16A34A")
    ax3.set_title("(c) Circuit Layer Depth Scaling (L)", fontsize=11, fontweight="bold")

    ax3_t = ax3.twinx()
    ax3_t.plot(ls, l_times, "--^", color="#E11D48", lw=2.0, label="Train Time (s)")
    ax3_t.set_ylabel("Training Time (s)", fontsize=9.5, fontweight="bold", color="#E11D48")
    ax3_t.tick_params(axis="y", labelcolor="#E11D48")
    ax3_t.grid(False)

    plt.tight_layout()
    plt.savefig("Fig4b_Ablations_and_Scaling.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ STEP 4 COMPLETED! Saved 'step4_ablation_results.json' and 'Fig4b_Ablations_and_Scaling.png'")


if __name__ == "__main__":
    run_all_ablations()