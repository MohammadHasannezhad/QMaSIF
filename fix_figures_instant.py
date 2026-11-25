import os
import json
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from qmasif_core import QMaSIFSiteModel
from step2_build_pdb_dataset import parse_pdb_chains, compute_unbound_surface_fields
from step6_pymol_interpretability import score_entire_chain_surface

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def fix_fig3():
    """اصلاح کنتراست اعداد سفید/مشکی ماتریس درهم‌ریختگی و انتقال Legend به بالا-راست."""
    data = torch.load("qmasif_real_pdb_site_dataset.pt", weights_only=False)
    ckpt = torch.load("qmasif_best_model.pt", weights_only=False)

    model = QMaSIFSiteModel(num_qubits=4, num_layers=2, use_canonical_sort=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(data["test"]["pos"][:, :4, :], data["test"]["feats"][:, :4, :], center_pos=data["test"]["center"])
        probs = torch.sigmoid(logits).cpu().numpy()

    y_true = data["test"]["labels"].cpu().numpy().astype(int)
    tau = float(ckpt["threshold"])
    y_pred = (probs >= tau).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=300)
    im = ax1.imshow(cm, cmap="Blues", aspect="auto")
    thresh = (cm.max() + cm.min()) / 2.0
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            pct = 100.0 * cm[i, j] / row_sums[i, 0]
            txt_color = "white" if cm[i, j] > thresh else "#0F172A"
            ax1.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                     fontweight="bold", fontsize=14, color=txt_color)

    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Non-Interface (0)", "Interface (1)"], fontsize=9.5, fontweight="bold")
    ax1.set_yticklabels(["Non-Interface (0)", "Interface (1)"], fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax1.set_title(f"(a) Confusion Matrix on Unseen PDB Test (N={cm.sum()}, $\\tau^*$={tau:.3f})", fontsize=11, fontweight="bold")
    ax1.grid(False)

    ax2.hist(probs[y_true == 1], bins=25, alpha=0.62, color="#0284C7", edgecolor="black", lw=0.4, label="Interface Sites (1)")
    ax2.hist(probs[y_true == 0], bins=25, alpha=0.58, color="#EF4444", edgecolor="black", lw=0.4, label="Non-Interface Surface (0)")
    ax2.axvline(tau, color="#16A34A", linestyle="--", lw=2.3, label=f"Val-Fitted $\\tau^*$ = {tau:.3f}")
    ax2.set_title("(b) Predicted Interface Probability Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Predicted Interface Probability", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Frequency (Surface Patches)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5, facecolor="white", framealpha=0.95)

    plt.tight_layout()
    plt.savefig("Fig3_Confusion_and_Similarity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Fixed Fig3_Confusion_and_Similarity.png")


def fix_fig4b():
    """رفع برخورد اعداد با میله‌های خطا در پنل (a) و افزودن Legend به پنل‌های (b) و (c)."""
    with open("step4_ablation_results.json", "r") as f:
        res = json.load(f)

    abl = res["architectural_ablations"]
    q_sc = res["qubit_scaling"]
    d_sc = res["depth_scaling"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=300)

    # Panel A
    abl_names = list(abl.keys())[::-1]
    abl_means = [abl[n]["roc_mean"] for n in abl_names]
    abl_stds = [abl[n]["roc_std"] for n in abl_names]
    bar_colors = ["#94A3B8"] * (len(abl_names) - 1) + ["#0284C7"]

    y_pos = np.arange(len(abl_names))
    bars = ax1.barh(y_pos, abl_means, xerr=abl_stds, capsize=3.5, color=bar_colors, height=0.58,
                    error_kw=dict(ecolor="#0F172A", lw=1.4))
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(abl_names, fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Unseen PDB Test ROC-AUC (%)", fontsize=10, fontweight="bold")
    x_right = max(m + s for m, s in zip(abl_means, abl_stds)) + 3.2
    ax1.set_xlim(55.0, x_right)
    ax1.set_title("(a) Architectural Component Ablations", fontsize=11.5, fontweight="bold")
    for bar, m, s in zip(bars, abl_means, abl_stds):
        ax1.text(m + s + 0.35, bar.get_y() + bar.get_height() / 2, f"{m:.2f}%",
                 va="center", ha="left", fontsize=9, fontweight="bold", color="#0F172A")

    # Panel B
    ks = [2, 4, 6, 8]
    k_rocs = [q_sc[str(k)]["roc_mean"] for k in ks]
    k_stds = [q_sc[str(k)]["roc_std"] for k in ks]
    k_times = [q_sc[str(k)]["time_mean"] for k in ks]

    l1 = ax2.errorbar(ks, k_rocs, yerr=k_stds, fmt="-o", color="#0284C7", lw=2.4, capsize=4, label="Test ROC-AUC (%)")
    ax2.set_xticks(ks)
    ax2.set_ylim(61.8, 65.8)
    ax2.set_xlabel("Number of Qubits ($k$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Unseen PDB Test ROC-AUC (%)", fontsize=10, fontweight="bold", color="#0284C7")
    ax2.tick_params(axis="y", labelcolor="#0284C7")
    ax2.set_title("(b) Qubit Register Scaling ($k$)", fontsize=11.5, fontweight="bold")

    ax2_t = ax2.twinx()
    l2 = ax2_t.plot(ks, k_times, "--^", color="#E11D48", lw=2.1, label="Training Time (s)")
    ax2_t.set_ylim(0, 165)
    ax2_t.set_ylabel("Training Time (s)", fontsize=10, fontweight="bold", color="#E11D48")
    ax2_t.tick_params(axis="y", labelcolor="#E11D48")
    ax2_t.grid(False)

    h1, lab1 = ax2.get_legend_handles_labels()
    h2, lab2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, lab1 + lab2, loc="upper left", frameon=True, fontsize=9, facecolor="white", framealpha=0.95)

    # Panel C
    ls = [1, 2, 3, 4]
    l_rocs = [d_sc[str(L)]["roc_mean"] for L in ls]
    l_stds = [d_sc[str(L)]["roc_std"] for L in ls]
    l_times = [d_sc[str(L)]["time_mean"] for L in ls]

    ax3.errorbar(ls, l_rocs, yerr=l_stds, fmt="-s", color="#16A34A", lw=2.4, capsize=4, label="Test ROC-AUC (%)")
    ax3.set_xticks(ls)
    ax3.set_ylim(55.5, 68.5)
    ax3.set_xlabel("Variational Ansatz Depth ($L$)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Unseen PDB Test ROC-AUC (%)", fontsize=10, fontweight="bold", color="#16A34A")
    ax3.tick_params(axis="y", labelcolor="#16A34A")
    ax3.set_title("(c) Circuit Layer Depth Scaling ($L$)", fontsize=11.5, fontweight="bold")

    ax3_t = ax3.twinx()
    ax3_t.plot(ls, l_times, "--^", color="#E11D48", lw=2.1, label="Training Time (s)")
    ax3_t.set_ylim(10, 75)
    ax3_t.set_ylabel("Training Time (s)", fontsize=10, fontweight="bold", color="#E11D48")
    ax3_t.tick_params(axis="y", labelcolor="#E11D48")
    ax3_t.grid(False)

    h1, lab1 = ax3.get_legend_handles_labels()
    h2, lab2 = ax3_t.get_legend_handles_labels()
    ax3.legend(h1 + h2, lab1 + lab2, loc="lower right", frameon=True, fontsize=9, facecolor="white", framealpha=0.95)

    plt.tight_layout()
    plt.savefig("Fig4b_Ablations_and_Scaling.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Fixed Fig4b_Ablations_and_Scaling.png")


def fix_fig5a_and_fig6():
    """اصلاح زیرنویس ریاضی Fig5a و انتقال کادرهای Legend روی فضای خالی میانی Fig6."""
    with open("step5_nisq_bp_results.json", "r") as f:
        res = json.load(f)

    # --- Fig 5a ---
    noise_results = res["depolarizing_noise"]
    shot_results = res["finite_shots"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300)
    ps_pct = [r["p"] * 100.0 for r in noise_results]
    rocs_n = [r["roc_auc"] for r in noise_results]
    margins_n = [r["margin"] for r in noise_results]

    ax1.plot(ps_pct, rocs_n, "-o", color="#0284C7", lw=2.5, label="Test ROC-AUC (%)")
    ax1.axhline(50.0, color="#EF4444", linestyle="--", lw=1.4, label="Random Chance (50%)")
    ax1.set_xlabel("2-Qubit Gate Depolarizing Noise Rate $p$ (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Unseen PDB Test ROC-AUC (%)", fontsize=10, fontweight="bold", color="#0284C7")
    ax1.tick_params(axis="y", labelcolor="#0284C7")
    ax1.set_ylim(47, max(rocs_n) + 4)
    ax1.set_title("(a) NISQ Depolarizing Noise Resilience", fontsize=11.5, fontweight="bold")

    ax1_t = ax1.twinx()
    ax1_t.plot(ps_pct, margins_n, "--s", color="#F59E0B", lw=2.0, label="Prob. Margin $\\Delta$")
    ax1_t.set_ylabel("Probability Margin ($\\mu_{\\mathrm{pos}} - \\mu_{\\mathrm{neg}}$)", fontsize=10, fontweight="bold", color="#D97706")
    ax1_t.tick_params(axis="y", labelcolor="#D97706")
    ax1_t.grid(False)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1_t.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="lower left", frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)

    x_labels = [str(r["shots"]) for r in shot_results]
    s_means = [r["roc_mean"] for r in shot_results]
    s_stds = [r["roc_std"] for r in shot_results]
    ideal_roc = noise_results[0]["roc_auc"]

    ax2.errorbar(x_labels, s_means, yerr=s_stds, fmt="-s", color="#16A34A", ecolor="#86EFAC",
                 elinewidth=2.2, capsize=4, lw=2.5, label="Finite-Shot Test ROC-AUC (10 Trials)")
    ax2.axhline(ideal_roc, color="#0284C7", linestyle="--", lw=1.8, label=f"Exact Statevector ({ideal_roc:.2f}%)")
    ax2.set_xlabel("Number of Measurement Shots ($S$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Unseen PDB Test ROC-AUC (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(min(s_means) - 4.5, ideal_roc + 2.8)
    ax2.set_title("(b) Finite-Shot Measurement Stability", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)

    plt.tight_layout()
    plt.savefig("Fig5a_NISQ_Noise_and_Shots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Fixed Fig5a_NISQ_Noise_and_Shots.png")

    # --- Fig 6 ---
    depth_bp = res["barren_plateau_depth"]
    qubit_bp = res["barren_plateau_qubits"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300)
    ls = list(range(1, 9))
    vg_l = [depth_bp[str(L)]["var_gamma0"] for L in ls]
    vb_l = [depth_bp[str(L)]["var_beta0"] for L in ls]

    ax1.semilogy(ls, vg_l, "-o", color="#E11D48", lw=2.4, markersize=7, label=r"$\mathrm{Var}[\partial \langle Z_0 \rangle / \partial \gamma_0]$ (ZZ)")
    ax1.semilogy(ls, vb_l, "--s", color="#0284C7", lw=2.2, markersize=6, label=r"$\mathrm{Var}[\partial \langle Z_0 \rangle / \partial \beta_0]$ (RX)")
    ax1.axvspan(1, 3, color="#16A34A", alpha=0.10, label=r"Shallow Regime ($L \leq 3$)")
    ax1.set_ylim(4e-4, 2.5e-1)
    ax1.set_xlabel("Circuit Layer Depth ($L$)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Gradient Variance (Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Gradient Variance vs. Circuit Depth ($k=4$)", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="center right", frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)
    ax1.grid(True, which="both", ls="--", alpha=0.45)

    ks = [2, 4, 6, 8]
    vg_k2 = [qubit_bp["L2"][str(k)]["var_gamma0"] for k in ks]
    vg_k3 = [qubit_bp["L3"][str(k)]["var_gamma0"] for k in ks]
    vb_k2 = [qubit_bp["L2"][str(k)]["var_beta0"] for k in ks]

    ax2.semilogy(ks, vg_k2, "-o", color="#E11D48", lw=2.4, markersize=7, label="ZZ Grad Var ($L=2$)")
    ax2.semilogy(ks, vg_k3, "-^", color="#9333EA", lw=2.4, markersize=7, label="ZZ Grad Var ($L=3$)")
    ax2.semilogy(ks, vb_k2, "--s", color="#0284C7", lw=2.2, markersize=6, label="RX Grad Var ($L=2$)")
    ax2.set_xticks(ks)
    ax2.set_ylim(1e-4, 2.5e-1)
    ax2.set_xlabel("Number of Qubits ($k$)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Gradient Variance (Log Scale)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Gradient Variance vs. Qubit Register Size ($k$)", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="center left", bbox_to_anchor=(0.02, 0.56), frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)
    ax2.grid(True, which="both", ls="--", alpha=0.45)

    plt.tight_layout()
    plt.savefig("Fig6_Barren_Plateaus.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Fixed Fig6_Barren_Plateaus.png")


def fix_fig5bc():
    """حذف دیواره مصنوعی در 30Å در پنل (a) و فاصله‌گذاری تمیز برچسب‌های میانگین در پنل (b)."""
    data = torch.load("qmasif_real_pdb_site_dataset.pt", weights_only=False)
    ckpt = torch.load("qmasif_best_model.pt", weights_only=False)

    model = QMaSIFSiteModel(num_qubits=4, num_layers=2, use_canonical_sort=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_probs, all_dists = [], []
    for entry in data["test"]["complexes"]:
        pdb_id, cid_a, cid_b = entry.split("_")
        chains = parse_pdb_chains(os.path.join("pdb_cache", f"{pdb_id.lower()}.pdb"))
        ca = compute_unbound_surface_fields(chains[cid_a])
        cb = compute_unbound_surface_fields(chains[cid_b])
        ra = score_entire_chain_surface(model, ca, cb, k=4, radius=8.0)
        rb = score_entire_chain_surface(model, cb, ca, k=4, radius=8.0)
        all_probs.extend([ra["probs"], rb["probs"]])
        all_dists.extend([ra["dists"], rb["dists"]])

    all_probs = np.concatenate(all_probs)
    all_dists = np.concatenate(all_dists)
    r_all, p_all = stats.pearsonr(all_dists, all_probs)
    rho_all, _ = stats.spearmanr(all_dists, all_probs)

    z_core = all_probs[all_dists <= 5.0]
    z_rim = all_probs[(all_dists > 5.0) & (all_dists <= 10.0)]
    z_distal = all_probs[all_dists > 10.0]
    _, u_pval = stats.mannwhitneyu(z_core, z_distal, alternative="greater")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=300)

    # Panel A: نمایش واقعی تا 28Å بدون Clip کردن روی خط 30Å
    vis_mask = all_dists <= 28.0
    ax1.scatter(all_dists[vis_mask], all_probs[vis_mask], s=13, alpha=0.22, color="#0284C7",
                edgecolors="none", label=f"Unseen Surface Residues (N={len(all_probs)})")

    bins = np.linspace(2.0, 26.0, 13)
    bc, bm, bsem = [], [], []
    for i in range(len(bins) - 1):
        m = (all_dists >= bins[i]) & (all_dists < bins[i + 1])
        if np.sum(m) >= 5:
            bc.append(0.5 * (bins[i] + bins[i + 1]))
            bm.append(float(np.mean(all_probs[m])))
            bsem.append(float(np.std(all_probs[m], ddof=1) / np.sqrt(np.sum(m))))

    ax1.errorbar(bc, bm, yerr=bsem, fmt="-o", color="#E11D48", lw=2.4, capsize=3.5, markersize=6, label="Binned Mean $\\pm$ SEM")
    ax1.axvline(5.0, color="#16A34A", linestyle="--", lw=1.7, label="Interface Cutoff (5.0 Å)")
    ax1.set_xlim(1.5, 28.0)
    ax1.set_xlabel("True Heavy-Atom Distance to Binding Partner (Å)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Predicted QMaSIF Interface Score", fontsize=10, fontweight="bold")
    ax1.set_title(f"(a) Spatial Enrichment on Unseen PDBs\n($r = {r_all:+.3f}$, $\\rho = {rho_all:+.3f}$, $p = {p_all:.2e}$)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)

    # Panel B: فاصله‌گذاری تمیز برچسب‌های میانگین بیرون از لبه جعبه
    bp = ax2.boxplot([z_core, z_rim, z_distal], widths=0.42, patch_artist=True, showfliers=False,
                     medianprops={"color": "#0F172A", "linewidth": 2.0})
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels([
        f"Core Interface\n($\\leq$5Å, n={len(z_core)})",
        f"Interface Rim\n(5–10Å, n={len(z_rim)})",
        f"Distal Surface\n(>10Å, n={len(z_distal)})",
    ], fontsize=9.5, fontweight="bold")

    for patch, col in zip(bp["boxes"], ["#EF4444", "#F59E0B", "#3B82F6"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    for idx_z, arr in enumerate([z_core, z_rim, z_distal], 1):
        ax2.scatter([idx_z], [arr.mean()], color="black", marker="D", s=45, zorder=5,
                    label="Zone Mean" if idx_z == 1 else None)
        ax2.text(idx_z + 0.25, arr.mean(), f"$\\mu$={arr.mean():.3f}", fontsize=9.2, fontweight="bold", va="center", ha="left")

    ax2.set_xlim(0.55, 3.65)
    ax2.set_ylabel("Predicted QMaSIF Interface Score", fontsize=10, fontweight="bold")
    ax2.set_title(f"(b) 3-Zone Structural Enrichment\n(Mann-Whitney U $p = {u_pval:.2e}$)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9, facecolor="white", framealpha=0.95)

    plt.tight_layout()
    plt.savefig("Fig5bc_Structural_Interpretability.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Fixed Fig5bc_Structural_Interpretability.png")


if __name__ == "__main__":
    print("Applying instant publication-grade fixes to 5 figures...")
    fix_fig3()
    fix_fig4b()
    fix_fig5a_and_fig6()
    fix_fig5bc()
    print("✅ All 5 figures fixed!")