import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.5), dpi=300)

models_p1 = [
    ("QMaSIF (L=3, 23p)",                23,   64.19, 59.26, 68.90, "#E11D48", "D", 9.5,  0.00),
    ("QMaSIF (L=2, 17p)",                17,   62.22, 57.58, 66.80, "#0284C7", "o", 9.0,  0.00),
    ("Classical EGNN (3345p)",           3345, 61.52, 56.92, 65.80, "#9333EA", "X", 9.0,  0.00),
    ("Iso-Param Tiny MLP (17p)",         17,   61.39, 57.05, 65.39, "#D97706", "P", 8.8, -0.04),
    ("Fair Classical MLP (321p)",        321,  61.35, 57.04, 65.47, "#16A34A", "v", 8.5,  0.00),
    ("Shared-Param Ising-GNN (23p)",     23,   61.13, 56.81, 64.99, "#0D9488", "h", 8.8,  0.06),
    ("Shared-Param Classical GNN (23p)", 23,   61.11, 56.70, 65.17, "#4F46E5", "p", 8.8, -0.06),
    ("Iso-Param Linear (19p)",           19,   60.81, 56.18, 65.13, "#F59E0B", "s", 8.0,  0.00),
    ("Center-Only Logistic (4p)",        4,    59.85, 55.87, 63.74, "#64748B", "^", 8.5,  0.00),
]

for name, params, mean, lo, hi, col, mk, ms, x_jit in models_p1:
    ax1.errorbar(
        params * (10 ** x_jit), mean,
        yerr=[[mean - lo], [hi - mean]],
        fmt=mk, color=col, markersize=ms, capsize=3.8, lw=1.9,
        label=f"{name}: {mean:.2f}%",
        zorder=5 if "QMaSIF" in name else 3,
    )

ax1.set_xscale("log")
ax1.set_xlim(2.5, 6500)
ax1.set_ylim(54.2, 71.5)
ax1.set_xlabel("Trainable Parameters (Log Scale)", fontsize=10.5, fontweight="bold")
ax1.set_ylabel("Out-of-Fold Cluster-Level ROC-AUC (% $\\pm$ 95% CI)", fontsize=10, fontweight="bold")
ax1.set_title("(a) Parameter Efficiency Across All 60 Complexes (37 Clusters)", fontsize=11, fontweight="bold")
ax1.legend(loc="upper right", frameon=True, fontsize=7.9, facecolor="white", framealpha=0.96)
ax1.grid(True, which="both", ls="--", alpha=0.45)

diffs_p2 = [
    ("Center-Only Logistic (4p)",        4.34,  0.05, 8.16),
    ("Iso-Param Linear (19p)",           3.38, -1.13, 7.35),
    ("Iso-Param Tiny MLP (17p)",         2.81,  0.61, 5.06),
    ("Shared-Param Ising-GNN (23p)",     3.06,  0.24, 5.92),
    ("Shared-Param Classical GNN (23p)", 3.08,  0.39, 5.90),
    ("Fair Classical MLP (321p)",        2.84, -0.52, 6.16),
    ("Classical EGNN (3345p)",           2.68,  0.75, 4.81),
]

y_pos = np.arange(len(diffs_p2))
ax2.axvline(0.0, color="#0F172A", linestyle="--", lw=1.4)

for i, (bname, dm, lo, hi) in enumerate(diffs_p2):
    col = "#E11D48" if lo > 0 else "#0284C7"
    ax2.errorbar(
        dm, i, xerr=[[dm - lo], [hi - dm]],
        fmt="D", color=col, capsize=4.0, lw=2.2, markersize=6.8,
        label="Strictly Positive 95% Cluster CI" if i == 0 else ("95% Cluster CI Crosses Zero" if i == 1 else None),
    )
    ax2.text(hi + 0.18, i, f"+{dm:.2f}%", va="center", ha="left", fontsize=8.5, fontweight="bold", color="#0F172A")

ax2.set_yticks(y_pos)
ax2.set_yticklabels([d[0] for d in diffs_p2], fontsize=9.0, fontweight="bold")
ax2.set_ylim(-0.65, len(diffs_p2) - 0.15)
ax2.set_xlim(-2.0, 9.7)
ax2.set_xlabel(r"Cluster-Level $\Delta$ROC-AUC (%) with 95% Bootstrap CI", fontsize=10, fontweight="bold")
ax2.set_title(r"(b) Flagship QMaSIF ($L=3$, 23p) $-$ Baseline ($N=37$ Clusters)", fontsize=11, fontweight="bold")
ax2.legend(loc="upper right", frameon=True, fontsize=8.4, facecolor="white", framealpha=0.96)
ax2.grid(True, axis="x", ls="--", alpha=0.45)

plt.tight_layout()
plt.savefig("Fig_Priority1_ClusterCV_IsoParam.png", dpi=300, bbox_inches="tight")
plt.close()
print("Updated Fig_Priority1_ClusterCV_IsoParam.png successfully.")