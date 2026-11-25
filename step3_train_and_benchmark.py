import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt

from qmasif_core import QMaSIFSiteModel, canonical_sort_batch


def random_so3_batch(batch_size: int, device: torch.device, seed: int = 123) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mats = []
    for _ in range(batch_size):
        q, r = torch.linalg.qr(torch.randn(3, 3, generator=gen))
        d = torch.diag(r)
        q = q @ torch.diag(d / torch.abs(d))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        mats.append(q)
    return torch.stack(mats, dim=0).to(device)


def apply_independent_se3(pos: torch.Tensor, center: torch.Tensor, seed: int = 2026):
    """Apply identical random SO(3) rotation + 50A translation to both patch nodes and center."""
    B = pos.shape[0]
    R = random_so3_batch(B, pos.device, seed=seed)
    gen = torch.Generator(device="cpu").manual_seed(seed + 999)
    t = (torch.randn(B, 1, 3, generator=gen) * 50.0).to(pos.device)

    pos_rot = torch.bmm(pos, R) + t
    ctr_rot = torch.bmm(center.unsqueeze(1), R).squeeze(1) + t.squeeze(1)
    return pos_rot, ctr_rot


class FairClassicalMLP(nn.Module):
    """
    Receives the exact same 18 E(3)-invariant canonically sorted features as QMaSIF:
      4 * 3 node features (q, h, s) + 6 pairwise Euclidean distances = 18 dims.
    Total trainable parameters: 321.
    """
    def __init__(self, num_nodes: int = 4):
        super().__init__()
        in_dim = num_nodes * 3 + (num_nodes * (num_nodes - 1)) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
        self.register_buffer("triu_i", idx[0])
        self.register_buffer("triu_j", idx[1])

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        pos_s, feats_s = canonical_sort_batch(pos, feats, center_pos=center_pos)
        dist_mat = torch.cdist(pos_s, pos_s, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        d_ij = dist_mat[:, self.triu_i, self.triu_j] / 10.0
        x = torch.cat([feats_s.reshape(pos.shape[0], -1), d_ij], dim=-1)
        return self.net(x).squeeze(-1)


class ClassicalEGNN(nn.Module):
    """
    True E(3)-invariant & permutation-invariant EGNN (Satorras et al., ICML 2021)
    combining center-node representation and global patch pooling.
    """
    def __init__(self, in_dim: int = 3, hidden_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.node_in = nn.Linear(in_dim + 1, hidden_dim)  # +1 for distance to center
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        ])
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim * 2, 1)

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        B, k, _ = pos.shape
        if center_pos is None:
            center_pos = pos.mean(dim=1)

        d_ctr = torch.norm(pos - center_pos.unsqueeze(1), dim=-1)  # [B, k]
        ctr_idx = torch.argmin(d_ctr, dim=1)                       # [B]

        dist_mat = torch.cdist(pos, pos, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        d_sq = (dist_mat ** 2) / 100.0
        mask = (~torch.eye(k, dtype=torch.bool, device=pos.device)).unsqueeze(0).unsqueeze(-1).float()

        h = self.node_in(torch.cat([feats, (d_ctr / 10.0).unsqueeze(-1)], dim=-1))
        for l in range(self.num_layers):
            h_i = h.unsqueeze(2).expand(B, k, k, -1)
            h_j = h.unsqueeze(1).expand(B, k, k, -1)
            m_ij = self.edge_mlps[l](torch.cat([h_i, h_j, d_sq.unsqueeze(-1)], dim=-1)) * mask
            m_i = m_ij.sum(dim=2) / max(1, k - 1)
            h = h + self.node_mlps[l](torch.cat([h, m_i], dim=-1))

        h_center = h[torch.arange(B, device=pos.device), ctr_idx]
        h_pool = h.mean(dim=1)
        return self.readout(torch.cat([h_center, h_pool], dim=-1)).squeeze(-1)


class RawCoordMLP(nn.Module):
    """
    Non-invariant baseline taking raw 3D coordinates + features directly.
    """
    def __init__(self, num_nodes: int = 4):
        super().__init__()
        in_dim = num_nodes * 3 + num_nodes * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        x = torch.cat([pos.reshape(pos.shape[0], -1) / 20.0, feats.reshape(pos.shape[0], -1)], dim=-1)
        return self.net(x).squeeze(-1)


def get_split(split_dict, k: int = 4):
    return (
        split_dict["pos"][:, :k, :],
        split_dict["feats"][:, :k, :],
        split_dict["center"],
        split_dict["labels"],
    )


@torch.no_grad()
def evaluate_model(model, split_tensors, threshold: float = None, rotate_se3: bool = False):
    model.eval()
    pos, feats, center, labels = split_tensors
    if rotate_se3:
        pos, center = apply_independent_se3(pos, center, seed=2026)

    logits = model(pos, feats, center_pos=center)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = labels.cpu().numpy().astype(int)

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = float(auc(fpr, tpr))
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, probs)
    pr_auc = float(average_precision_score(y_true, probs))

    if threshold is None:
        j_idx = int(np.argmax(tpr - fpr))
        threshold = float(thresholds[j_idx])

    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    pos_scores = probs[y_true == 1]
    neg_scores = probs[y_true == 0]

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "threshold": threshold,
        "margin": float(np.mean(pos_scores) - np.mean(neg_scores)),
        "cm": cm.tolist(),
        "fpr": fpr,
        "tpr": tpr,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
    }


def train_one_run(model_fn, train_t, val_t, test_t,
                  seed: int = 42, epochs: int = 75, batch_size: int = 128,
                  lr: float = 0.015, weight_decay: float = 1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(TensorDataset(*train_t), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best_val_auc = -1.0
    best_state = None
    loss_history = []

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for pos, feats, center, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(pos, feats, center_pos=center), y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1
        scheduler.step()
        loss_history.append(ep_loss / max(1, nb))

        val_res = evaluate_model(model, val_t, threshold=None, rotate_se3=False)
        if val_res["roc_auc"] > best_val_auc:
            best_val_auc = val_res["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    # Fit threshold strictly on Validation Set
    val_final = evaluate_model(model, val_t, threshold=None, rotate_se3=False)
    tau_star = val_final["threshold"]

    # Evaluate once on unseen Test Set (original & independently rotated+translated)
    test_orig = evaluate_model(model, test_t, threshold=tau_star, rotate_se3=False)
    test_rot = evaluate_model(model, test_t, threshold=tau_star, rotate_se3=True)

    return model, loss_history, train_time, val_final, test_orig, test_rot


def run_benchmark():
    print("=" * 78)
    print("STEP 3 (v3): REAL PDB SURFACE-FIELD BENCHMARK (5 INDEPENDENT RUNS)")
    print("=" * 78)

    data = torch.load("qmasif_real_pdb_site_dataset.pt", weights_only=False)
    train_t = get_split(data["train"], k=4)
    val_t = get_split(data["val"], k=4)
    test_t = get_split(data["test"], k=4)

    print(f"[INFO] Balanced Surface Patches -> Train: {len(train_t[3])} | Val: {len(val_t[3])} | Test: {len(test_t[3])}")

    seeds = [42, 101, 202, 303, 404]

    model_specs = {
        "QMaSIF (Ours)": {
            "fn": lambda: QMaSIFSiteModel(num_qubits=4, num_layers=2, use_canonical_sort=True),
            "lr": 0.03, "wd": 0.0,
        },
        "Fair Classical MLP (18-dim)": {
            "fn": lambda: FairClassicalMLP(num_nodes=4),
            "lr": 0.01, "wd": 1e-4,
        },
        "Classical EGNN": {
            "fn": lambda: ClassicalEGNN(in_dim=3, hidden_dim=16, num_layers=2),
            "lr": 0.005, "wd": 1e-4,
        },
        "Raw-Coord MLP (Non-Invariant)": {
            "fn": lambda: RawCoordMLP(num_nodes=4),
            "lr": 0.01, "wd": 1e-4,
        },
    }

    all_runs, summary = {}, {}
    best_qmasif, best_qmasif_res, best_q_val_auc = None, None, -1.0

    for name, spec in model_specs.items():
        n_params = sum(p.numel() for p in spec["fn"]().parameters() if p.requires_grad)
        print(f"\n>>> Benchmarking: {name} | Trainable Params: {n_params}")
        runs = []
        for s in seeds:
            model, loss_hist, t_sec, val_res, test_orig, test_rot = train_one_run(
                spec["fn"], train_t, val_t, test_t,
                seed=s, epochs=75, batch_size=128, lr=spec["lr"], weight_decay=spec["wd"]
            )
            runs.append({
                "seed": s, "loss_hist": loss_hist, "time_s": t_sec,
                "val_auc": val_res["roc_auc"], "test_orig": test_orig, "test_rot": test_rot,
            })
            print(
                f"    Seed {s:3d} | Val AUC: {val_res['roc_auc']*100:5.2f}% | "
                f"Test AUC: {test_orig['roc_auc']*100:5.2f}% | "
                f"Test PR: {test_orig['pr_auc']*100:5.2f}% | "
                f"Rot AUC: {test_rot['roc_auc']*100:5.2f}% | "
                f"Acc: {test_orig['accuracy']*100:5.2f}% | "
                f"τ*: {test_orig['threshold']:.3f} | Time: {t_sec:.1f}s"
            )
            # Select checkpoint strictly by Validation AUC (zero test selection bias)
            if name == "QMaSIF (Ours)" and val_res["roc_auc"] > best_q_val_auc:
                best_q_val_auc = val_res["roc_auc"]
                best_qmasif = copy.deepcopy(model)
                best_qmasif_res = test_orig

        all_runs[name] = runs

        def agg(key, sub="test_orig"):
            vals = [r[sub][key] * 100.0 for r in runs]
            return float(np.mean(vals)), float(np.std(vals, ddof=1))

        summary[name] = {
            "params": n_params,
            "roc_auc_mean_std": agg("roc_auc"),
            "pr_auc_mean_std": agg("pr_auc"),
            "acc_mean_std": agg("accuracy"),
            "f1_mean_std": agg("f1"),
            "sens_mean_std": agg("sensitivity"),
            "spec_mean_std": agg("specificity"),
            "rot_auc_mean_std": agg("roc_auc", sub="test_rot"),
            "avg_time_s": float(np.mean([r["time_s"] for r in runs])),
            "roc_auc_seeds": [r["test_orig"]["roc_auc"] * 100.0 for r in runs],
        }

    torch.save({
        "state_dict": best_qmasif.state_dict(),
        "quantum_gamma": best_qmasif.quantum_layer.gamma.detach().cpu().tolist(),
        "quantum_beta": best_qmasif.quantum_layer.beta.detach().cpu().tolist(),
        "threshold": best_qmasif_res["threshold"],
        "val_roc_auc": best_q_val_auc,
        "test_roc_auc": best_qmasif_res["roc_auc"],
    }, "qmasif_best_model.pt")

    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY TABLE (MEAN ± STD ACROSS 5 INDEPENDENT SEEDS ON REAL PDB)")
    print("=" * 78)
    header = f"{'Model':<30} | {'Params':>6} | {'ROC-AUC (%)':>13} | {'PR-AUC (%)':>13} | {'Acc (%)':>13} | {'Rot-AUC (%)':>13}"
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(
            f"{name:<30} | {s['params']:>6d} | "
            f"{s['roc_auc_mean_std'][0]:5.2f} ± {s['roc_auc_mean_std'][1]:4.2f} | "
            f"{s['pr_auc_mean_std'][0]:5.2f} ± {s['pr_auc_mean_std'][1]:4.2f} | "
            f"{s['acc_mean_std'][0]:5.2f} ± {s['acc_mean_std'][1]:4.2f} | "
            f"{s['rot_auc_mean_std'][0]:5.2f} ± {s['rot_auc_mean_std'][1]:4.2f}"
        )

    qs = summary["QMaSIF (Ours)"]
    print("\n[DETAILED QMaSIF METRICS (5-Run Mean ± Std)]")
    print(f"  Sensitivity : {qs['sens_mean_std'][0]:.2f}% ± {qs['sens_mean_std'][1]:.2f}%")
    print(f"  Specificity : {qs['spec_mean_std'][0]:.2f}% ± {qs['spec_mean_std'][1]:.2f}%")
    print(f"  F1-Score    : {qs['f1_mean_std'][0]:.2f}% ± {qs['f1_mean_std'][1]:.2f}%")
    print(f"  Best-Val Q-Params -> gamma={np.round(best_qmasif.quantum_layer.gamma.detach().numpy(), 4)}, "
          f"beta={np.round(best_qmasif.quantum_layer.beta.detach().numpy(), 4)}")

    print("\n[STATISTICAL SIGNIFICANCE: PAIRED T-TEST ON ROC-AUC (5 SEEDS)]")
    q_aucs = summary["QMaSIF (Ours)"]["roc_auc_seeds"]
    stat_tests = {}
    for other in ["Fair Classical MLP (18-dim)", "Classical EGNN", "Raw-Coord MLP (Non-Invariant)"]:
        t_val, p_val = stats.ttest_rel(q_aucs, summary[other]["roc_auc_seeds"])
        stat_tests[other] = {"t_stat": float(t_val), "p_value": float(p_val)}
        print(f"  QMaSIF vs {other:<30}: t = {t_val:+.3f}, p = {p_val:.4e}")

    q_rot = [r["test_rot"]["roc_auc"] * 100.0 for r in all_runs["QMaSIF (Ours)"]]
    raw_rot = [r["test_rot"]["roc_auc"] * 100.0 for r in all_runs["Raw-Coord MLP (Non-Invariant)"]]
    t_rot, p_rot = stats.ttest_rel(q_rot, raw_rot)
    stat_tests["QMaSIF_vs_RawMLP_Rotated"] = {"t_stat": float(t_rot), "p_value": float(p_rot)}
    print(f"  QMaSIF vs Raw-Coord MLP (UNDER 3D ROTATION) : t = {t_rot:+.3f}, p = {p_rot:.4e}")

    with open("step3_benchmark_results.json", "w") as f:
        json.dump({"summary": summary, "stat_tests": stat_tests}, f, indent=2)

    # Plots
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "QMaSIF (Ours)": "#0284C7",
        "Fair Classical MLP (18-dim)": "#16A34A",
        "Classical EGNN": "#9333EA",
        "Raw-Coord MLP (Non-Invariant)": "#64748B",
    }

    # Fig 2: ROC & PR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=300)
    for name in ["QMaSIF (Ours)", "Fair Classical MLP (18-dim)", "Classical EGNN"]:
        runs = all_runs[name]
        rep = runs[np.argsort([r["test_orig"]["roc_auc"] for r in runs])[len(runs) // 2]]["test_orig"]
        rm, rs = summary[name]["roc_auc_mean_std"]
        pm, ps = summary[name]["pr_auc_mean_std"]
        lw = 2.6 if name == "QMaSIF (Ours)" else 2.0
        ax1.plot(rep["fpr"], rep["tpr"], color=colors[name], lw=lw, label=f"{name} ({rm:.2f}% ± {rs:.2f}%)")
        ax2.plot(rep["rec_curve"], rep["prec_curve"], color=colors[name], lw=lw, label=f"{name} ({pm:.2f}% ± {ps:.2f}%)")

    ax1.plot([0, 1], [0, 1], color="#94A3B8", lw=1.4, linestyle="--")
    ax1.set_title("(a) Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    ax2.axhline(0.5, color="#94A3B8", lw=1.4, linestyle="--")
    ax2.set_title("(b) Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig("Fig2_ROC_PR_Curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 3: Confusion Matrix & Probability Distribution
    q_runs = all_runs["QMaSIF (Ours)"]
    q_rep = q_runs[np.argsort([r["test_orig"]["roc_auc"] for r in q_runs])[len(q_runs) // 2]]["test_orig"]
    cm = np.array(q_rep["cm"])
    tau = q_rep["threshold"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=300)
    im = ax1.matshow(cm, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontweight="bold", fontsize=15)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Non-Interface (0)", "Interface (1)"], fontsize=9, fontweight="bold")
    ax1.set_yticklabels(["Non-Interface (0)", "Interface (1)"], fontsize=9, fontweight="bold")
    ax1.xaxis.set_ticks_position("bottom")
    ax1.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax1.set_title(f"(a) Confusion Matrix on Unseen PDB Test (N={cm.sum()}, τ*={tau:.3f})", fontsize=11, fontweight="bold")
    ax1.grid(False)

    ax2.hist(q_rep["pos_scores"], bins=25, alpha=0.6, color="#0284C7", edgecolor="black", lw=0.4, label="Interface Sites (1)")
    ax2.hist(q_rep["neg_scores"], bins=25, alpha=0.6, color="#EF4444", edgecolor="black", lw=0.4, label="Non-Interface Surface (0)")
    ax2.axvline(tau, color="#16A34A", linestyle="--", lw=2.2, label=f"Val-Fitted τ* = {tau:.3f}")
    ax2.set_title("(b) Predicted Interface Probability Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Predicted Interface Probability", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Frequency (Surface Patches)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig("Fig3_Confusion_and_Similarity.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 4a: Loss & SE(3) Stress Test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for name in model_specs.keys():
        mean_loss = np.mean([r["loss_hist"] for r in all_runs[name]], axis=0)
        ax1.plot(range(1, len(mean_loss) + 1), mean_loss, label=name, color=colors[name], lw=2.2)
    ax1.set_title("(a) Mean BCE Loss Convergence (5 Runs)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax1.legend(frameon=True, fontsize=8.5)

    model_names = list(model_specs.keys())
    short_names = [f"{n.split(' (')[0]}\n({summary[n]['params']}p)" for n in model_names]
    orig_means = [summary[n]["roc_auc_mean_std"][0] for n in model_names]
    orig_stds = [summary[n]["roc_auc_mean_std"][1] for n in model_names]
    rot_means = [summary[n]["rot_auc_mean_std"][0] for n in model_names]
    rot_stds = [summary[n]["rot_auc_mean_std"][1] for n in model_names]

    x_pos = np.arange(len(model_names))
    w = 0.35
    b1 = ax2.bar(x_pos - w / 2, orig_means, w, yerr=orig_stds, capsize=4, color="#0284C7", label="Original PDB Orientation")
    b2 = ax2.bar(x_pos + w / 2, rot_means, w, yerr=rot_stds, capsize=4, color="#F59E0B", label="Random SO(3) + 50Å Shift")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Test ROC-AUC (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(40, 100)
    ax2.set_title("(b) Rotational & Translational Stress Test", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    for bar in list(b1) + list(b2):
        h_val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h_val + 1.5, f"{h_val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig("Fig4a_Benchmark_and_Rotation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ STEP 3 (v3) COMPLETED!")


if __name__ == "__main__":
    run_benchmark()