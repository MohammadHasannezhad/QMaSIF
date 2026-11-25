import os
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt

from qmasif_core import QMaSIFSiteModel
from step2_build_pdb_dataset import parse_pdb_chains, compute_unbound_surface_fields
from step6_pymol_interpretability import score_entire_chain_surface

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def fmt_sci_tex(val: float) -> str:
    """Format a float into clean LaTeX scientific notation (e.g., 1.22 \\times 10^{-5})."""
    exp = int(np.floor(np.log10(abs(val))))
    mant = val / (10.0 ** exp)
    return rf"{mant:.2f} \times 10^{{{exp}}}"


def regenerate_fig7_panels_bc():
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

    # Panel (b): Distance-to-Interface Spatial Enrichment
    vis_mask = all_dists <= 28.0
    ax1.scatter(
        all_dists[vis_mask], all_probs[vis_mask],
        s=13, alpha=0.22, color="#0284C7", edgecolors="none",
        label=f"Unseen Surface Residues (N={len(all_probs)})",
    )

    bins = np.linspace(2.0, 26.0, 13)
    bc, bm, bsem = [], [], []
    for i in range(len(bins) - 1):
        m = (all_dists >= bins[i]) & (all_dists < bins[i + 1])
        if np.sum(m) >= 5:
            bc.append(0.5 * (bins[i] + bins[i + 1]))
            bm.append(float(np.mean(all_probs[m])))
            bsem.append(float(np.std(all_probs[m], ddof=1) / np.sqrt(np.sum(m))))

    ax1.errorbar(
        bc, bm, yerr=bsem, fmt="-o", color="#E11D48",
        lw=2.4, capsize=3.5, markersize=6, label="Binned Mean $\\pm$ SEM",
    )
    ax1.axvline(5.0, color="#16A34A", linestyle="--", lw=1.7, label="Interface Cutoff (5.0 Å)")
    ax1.set_xlim(1.5, 28.0)
    ax1.set_xlabel("True Heavy-Atom Distance to Binding Partner (Å)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Predicted QMaSIF Interface Score", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"(b) Spatial Enrichment on Unseen PDBs\n"
        f"($r = {r_all:+.3f}$, $\\rho = {rho_all:+.3f}$, $p = {fmt_sci_tex(p_all)}$)",
        fontsize=11, fontweight="bold",
    )
    ax1.legend(loc="upper right", frameon=True, fontsize=8.8, facecolor="white", framealpha=0.95)

    # Panel (c): 3-Zone Structural Enrichment
    bp = ax2.boxplot(
        [z_core, z_rim, z_distal], widths=0.42, patch_artist=True,
        showfliers=False, medianprops={"color": "#0F172A", "linewidth": 2.0},
    )
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
        ax2.scatter(
            [idx_z], [arr.mean()], color="black", marker="D", s=45, zorder=5,
            label="Zone Mean" if idx_z == 1 else None,
        )
        ax2.text(
            idx_z + 0.25, arr.mean(), f"$\\mu$={arr.mean():.3f}",
            fontsize=9.2, fontweight="bold", va="center", ha="left",
        )

    ax2.set_xlim(0.55, 3.65)
    ax2.set_ylabel("Predicted QMaSIF Interface Score", fontsize=10, fontweight="bold")
    ax2.set_title(
        f"(c) 3-Zone Structural Enrichment\n(Mann-Whitney $U$, $p = {fmt_sci_tex(u_pval)}$)",
        fontsize=11, fontweight="bold",
    )
    ax2.legend(loc="upper right", frameon=True, fontsize=9, facecolor="white", framealpha=0.95)

    plt.tight_layout()
    plt.savefig("Fig5bc_Structural_Interpretability.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Updated 'Fig5bc_Structural_Interpretability.png' with panels (b) and (c)!")


if __name__ == "__main__":
    regenerate_fig7_panels_bc()