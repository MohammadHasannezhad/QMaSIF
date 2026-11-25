import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def polish_priority1_fig():
    with open("step7_cluster_cv_results.json", "r") as f:
        s7 = json.load(f)

    results = s7["models"]
    stat_comp = s7["stat_comparisons"]
    n_comp = s7["n_complexes"]
    n_clus = s7["n_clusters"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4), dpi=300)

    palette = {
        "QMaSIF (L=3, 23p)": ("#E11D48", "D", 9.5),
        "QMaSIF (L=2, 17p)": ("#0284C7", "o", 9.0),
        "Classical EGNN (3345p)": ("#9333EA", "X", 9.0),
        "Fair Classical MLP (321p)": ("#16A34A", "v", 8.5),
        "Iso-Param Tiny MLP (17p)": ("#D97706", "P", 9.0),
        "Iso-Param Linear (19p)": ("#F59E0B", "s", 8.0),
        "Center-Only Logistic (4p)": ("#64748B", "^", 8.5),
    }

    for name, (col, mk, ms) in palette.items():
        r = results[name]
        cm, clo, chi = r["cluster_bal_mean_ci95"]
        ax1.errorbar(
            r["params"], cm,
            yerr=[[cm - clo], [chi - cm]],
            fmt=mk, color=col, markersize=ms, capsize=4, lw=2.0,
            label=f"{name}: {cm:.2f}%",
        )

    ax1.set_xscale("log")
    ax1.set_xlim(2.5, 6500)
    ax1.set_ylim(54.0, 71.0)
    ax1.set_xlabel("Trainable Parameters (Log Scale)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Out-of-Fold Cluster-Level ROC-AUC (% $\\pm$ 95% CI)", fontsize=10, fontweight="bold")
    ax1.set_title(f"(a) Parameter Efficiency Across All {n_comp} Complexes ({n_clus} Clusters)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.4, facecolor="white", framealpha=0.96)
    ax1.grid(True, which="both", ls="--", alpha=0.45)

    bases = [
        "Center-Only Logistic (4p)",
        "Iso-Param Linear (19p)",
        "Iso-Param Tiny MLP (17p)",
        "Fair Classical MLP (321p)",
        "Classical EGNN (3345p)",
    ]
    short_b = [b.split(" (")[0] for b in bases]
    y_pos = np.arange(len(bases))
    h = 0.34

    d2 = [stat_comp["QMaSIF (L=2, 17p)"][b]["cluster_delta_mean_ci95"] for b in bases]
    d3 = [stat_comp["QMaSIF (L=3, 23p)"][b]["cluster_delta_mean_ci95"] for b in bases]

    ax2.axvline(0.0, color="#0F172A", linestyle="--", lw=1.4)
    ax2.errorbar(
        [x[0] for x in d2], y_pos + h / 2,
        xerr=[[x[0] - x[1] for x in d2], [x[2] - x[0] for x in d2]],
        fmt="o", color="#0284C7", capsize=3.5, lw=2.0, markersize=6.5, label="QMaSIF ($L=2$, 17p) $-$ Baseline",
    )
    ax2.errorbar(
        [x[0] for x in d3], y_pos - h / 2,
        xerr=[[x[0] - x[1] for x in d3], [x[2] - x[0] for x in d3]],
        fmt="D", color="#E11D48", capsize=3.5, lw=2.0, markersize=6.5, label="QMaSIF ($L=3$, 23p) $-$ Baseline",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_b, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(-0.65, len(bases) - 0.15)
    ax2.set_xlim(-3.0, 9.2)
    ax2.set_xlabel(r"Cluster-Level $\Delta$ROC-AUC (%) with 95% Bootstrap CI", fontsize=10, fontweight="bold")
    ax2.set_title(f"(b) Paired Cluster-Level Advantage ($N={n_clus}$ Independent Clusters)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.6, facecolor="white", framealpha=0.96)
    ax2.grid(True, axis="x", ls="--", alpha=0.45)

    plt.tight_layout()
    plt.savefig("Fig_Priority1_ClusterCV_IsoParam.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Polished Fig_Priority1_ClusterCV_IsoParam.png")


def polish_priority2_fig():
    with open("step8_ablation_cv_results.json", "r") as f:
        s8 = json.load(f)

    results = s8["variants"]
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

    fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=300)
    bars = ax.barh(y_pos, pooled_vals, height=0.58, color=colors, edgecolor="#0F172A", lw=0.65)
    ax.axvline(results["Full QMaSIF (L=3, 23p)"]["pooled_bal_roc_auc"], color="#0284C7", ls="--", lw=1.3, alpha=0.75)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Out-of-Fold Pooled ROC-AUC Across All 60 PDB Complexes (%)", fontsize=10, fontweight="bold")
    ax.set_xlim(56.0, 71.2)
    ax.set_title("5-Fold Cluster-Level Architectural Ablation at $L=3$ (37 Independent Clusters)", fontsize=11.5, fontweight="bold")

    for bar, pv, fv in zip(bars, pooled_vals, full_vals):
        ax.text(
            pv + 0.22, bar.get_y() + bar.get_height() / 2,
            f"{pv:.2f}%  (Full-Surf: {fv:.2f}%)",
            va="center", ha="left", fontsize=8.8, fontweight="bold", color="#0F172A",
        )

    plt.tight_layout()
    plt.savefig("Fig_Priority2_Ablation_ClusterCV.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] Polished Fig_Priority2_Ablation_ClusterCV.png")


if __name__ == "__main__":
    polish_priority1_fig()
    polish_priority2_fig()
    print("✅ Both priority figures polished and ready for LaTeX!")