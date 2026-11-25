import json
import time
import numpy as np
import torch
import pennylane as qml
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from qmasif_core import QMaSIFSiteModel, QMaSIFQuantumLayer, canonical_sort_batch


# ==============================================================================
# 1. Exact Batched Density-Matrix NISQ Simulator + PennyLane Parity Verification
# ==============================================================================
PAULI_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
PAULI_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
PAULI_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)


def apply_1q_left(M: torch.Tensor, U: torch.Tensor, q_idx: int, k: int) -> torch.Tensor:
    B, dim, _ = M.shape
    s = M.view([B] + [2] * k + [dim])
    dims = [0, q_idx + 1] + [d + 1 for d in range(k) if d != q_idx] + [k + 1]
    s_perm = s.permute(dims)
    perm_shape = s_perm.shape
    s_flat = s_perm.reshape(B, 2, -1)

    if U.dim() == 2:
        s_out = torch.einsum("ij,bjn->bin", U, s_flat)
    else:
        s_out = torch.bmm(U, s_flat)

    s_out = s_out.reshape(perm_shape)
    inv_dims = [0] * (k + 2)
    for new_d, old_d in enumerate(dims):
        inv_dims[old_d] = new_d
    return s_out.permute(inv_dims).reshape(B, dim, dim)


def apply_1q_dm(rho: torch.Tensor, U: torch.Tensor, q_idx: int, k: int) -> torch.Tensor:
    """Exact U @ rho @ U^dagger for Hermitian density matrices."""
    u_rho = apply_1q_left(rho, U, q_idx, k)
    out = apply_1q_left(u_rho.transpose(-1, -2).conj(), U, q_idx, k)
    return 0.5 * (out + out.transpose(-1, -2).conj())


def apply_depolarizing_channel(rho: torch.Tensor, p: float, q_idx: int, k: int) -> torch.Tensor:
    """Exact PennyLane DepolarizingChannel(p): (1-p)*rho + (p/3)*(X rho X + Y rho Y + Z rho Z)."""
    if p <= 0.0:
        return rho
    px = PAULI_X.to(rho.device)
    py = PAULI_Y.to(rho.device)
    pz = PAULI_Z.to(rho.device)
    return (
        (1.0 - p) * rho
        + (p / 3.0) * (
            apply_1q_dm(rho, px, q_idx, k)
            + apply_1q_dm(rho, py, q_idx, k)
            + apply_1q_dm(rho, pz, q_idx, k)
        )
    )


@torch.no_grad()
def forward_noisy_density_matrix(model: QMaSIFSiteModel, pos: torch.Tensor, feats: torch.Tensor,
                                 center_pos: torch.Tensor, noise_p: float = 0.0):
    """
    Runs the exact trained QMaSIF model under 2-qubit gate depolarizing noise rate `noise_p`.
    Returns:
      logits: [B]
      expvals: [B, k]
      probs_basis: [B, 2^k] computational basis measurement probabilities
    """
    pos_s, feats_s = canonical_sort_batch(pos, feats, center_pos=center_pos)
    dist_mat = torch.cdist(pos_s, pos_s, p=2, compute_mode="donot_use_mm_for_euclid_dist")

    ql = model.quantum_layer
    B = feats_s.shape[0]
    k = ql.num_qubits
    dim = 1 << k
    device = feats_s.device

    rho = torch.zeros((B, dim, dim), dtype=torch.complex64, device=device)
    rho[:, 0, 0] = 1.0 + 0.0j

    for l in range(ql.num_layers):
        # 1. Data Re-uploading SU(2) Encoding
        for i in range(k):
            q_val = feats_s[:, i, 0]
            h_val = feats_s[:, i, 1] - 0.5
            s_val = feats_s[:, i, 2] - 0.5
            zeros = torch.zeros_like(q_val)
            zero_c = torch.complex(zeros, zeros)

            ry_h = 0.5 * (0.5 * np.pi + ql.omega[l, 0] * q_val)
            cy, sy = torch.cos(ry_h), torch.sin(ry_h)
            ry_mat = torch.stack([
                torch.stack([torch.complex(cy, zeros), torch.complex(-sy, zeros)], dim=-1),
                torch.stack([torch.complex(sy, zeros), torch.complex(cy, zeros)], dim=-1),
            ], dim=-2)
            rho = apply_1q_dm(rho, ry_mat, i, k)

            rz_h = 0.5 * (ql.omega[l, 1] * h_val)
            cz, sz = torch.cos(rz_h), torch.sin(rz_h)
            rz_mat = torch.stack([
                torch.stack([torch.complex(cz, -sz), zero_c], dim=-1),
                torch.stack([zero_c, torch.complex(cz, sz)], dim=-1),
            ], dim=-2)
            rho = apply_1q_dm(rho, rz_mat, i, k)

            rx_h = 0.5 * (ql.omega[l, 2] * s_val)
            cx, sx = torch.cos(rx_h), torch.sin(rx_h)
            rx_mat = torch.stack([
                torch.stack([torch.complex(cx, zeros), torch.complex(zeros, -sx)], dim=-1),
                torch.stack([torch.complex(zeros, -sx), torch.complex(cx, zeros)], dim=-1),
            ], dim=-2)
            rho = apply_1q_dm(rho, rx_mat, i, k)

        # 2. Pairwise Distance-Conditioned Ising-ZZ + Immediate 2-Qubit Depolarizing Noise
        for p_idx, (i, j) in enumerate(ql.pairs):
            d_ij = dist_mat[:, i, j] + ql.eps
            theta_ij = ql.gamma[l] / d_ij
            half_phase = 0.5 * theta_ij.unsqueeze(1) * ql.pair_signs[p_idx].unsqueeze(0)
            phase = torch.complex(torch.cos(half_phase), -torch.sin(half_phase))
            rho = phase.unsqueeze(2) * rho * phase.conj().unsqueeze(1)

            if noise_p > 0.0:
                rho = apply_depolarizing_channel(rho, noise_p, i, k)
                rho = apply_depolarizing_channel(rho, noise_p, j, k)

        # 3. Shared Global SU(2) Mixing: RX(beta[l,0]) -> RY(beta[l,1])
        rx_h = 0.5 * ql.beta[l, 0]
        cx, sx = torch.cos(rx_h), torch.sin(rx_h)
        z0 = torch.zeros_like(cx)
        rx_mix = torch.stack([
            torch.stack([torch.complex(cx, z0), torch.complex(z0, -sx)], dim=-1),
            torch.stack([torch.complex(z0, -sx), torch.complex(cx, z0)], dim=-1),
        ], dim=-2)

        ry_h = 0.5 * ql.beta[l, 1]
        cy, sy = torch.cos(ry_h), torch.sin(ry_h)
        ry_mix = torch.stack([
            torch.stack([torch.complex(cy, z0), torch.complex(-sy, z0)], dim=-1),
            torch.stack([torch.complex(sy, z0), torch.complex(cy, z0)], dim=-1),
        ], dim=-2)

        mix_mat = torch.matmul(ry_mix, rx_mix)
        for i in range(k):
            rho = apply_1q_dm(rho, mix_mat, i, k)

    probs_basis = torch.diagonal(rho, dim1=-2, dim2=-1).real.clamp(min=0.0)
    probs_basis = probs_basis / probs_basis.sum(dim=-1, keepdim=True)
    expvals = torch.matmul(probs_basis, ql.z_signs.t())
    logits = model.readout(expvals).squeeze(-1)
    return logits, expvals, probs_basis


def verify_noisy_parity_with_pennylane(model: QMaSIFSiteModel, pos: torch.Tensor, feats: torch.Tensor,
                                       center: torch.Tensor, noise_p: float = 0.03):
    """Verify exact numerical match with PennyLane default.mixed under DepolarizingChannel."""
    ql = model.quantum_layer
    k, L, eps = ql.num_qubits, ql.num_layers, ql.eps
    dev = qml.device("default.mixed", wires=k)

    @qml.qnode(dev)
    def pl_noisy_circuit(f_in, d_in, omega, gamma, beta, p_val):
        for l in range(L):
            for i in range(k):
                qml.RY(0.5 * np.pi + omega[l, 0] * f_in[i, 0], wires=i)
                qml.RZ(omega[l, 1] * (f_in[i, 1] - 0.5), wires=i)
                qml.RX(omega[l, 2] * (f_in[i, 2] - 0.5), wires=i)
            for i in range(k):
                for j in range(i + 1, k):
                    qml.IsingZZ(gamma[l] / (d_in[i, j] + eps), wires=[i, j])
                    if p_val > 0.0:
                        qml.DepolarizingChannel(p_val, wires=i)
                        qml.DepolarizingChannel(p_val, wires=j)
            for i in range(k):
                qml.RX(beta[l, 0], wires=i)
                qml.RY(beta[l, 1], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(k)]

    pos_s, feats_s = canonical_sort_batch(pos[:3], feats[:3], center_pos=center[:3])
    dist_s = torch.cdist(pos_s, pos_s, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    _, exp_torch, _ = forward_noisy_density_matrix(model, pos[:3], feats[:3], center[:3], noise_p=noise_p)

    om_np = ql.omega.detach().numpy()
    ga_np = ql.gamma.detach().numpy()
    be_np = ql.beta.detach().numpy()

    max_diff = 0.0
    for b in range(3):
        exp_pl = np.array(pl_noisy_circuit(feats_s[b].numpy(), dist_s[b].numpy(), om_np, ga_np, be_np, noise_p))
        max_diff = max(max_diff, float(np.max(np.abs(exp_torch[b].numpy() - exp_pl))))

    print(f"[VERIFY] Noisy Density-Matrix Parity vs PennyLane default.mixed (p={noise_p:.2f}): max_diff = {max_diff:.2e}")
    assert max_diff < 1e-5, f"Noisy parity mismatch: {max_diff}"
    print("  STATUS: PASS")


# ==============================================================================
# 2. Barren Plateau Analysis (Both Circuit Depth L and Qubit Register Size k)
# ==============================================================================
def run_barren_plateau_analysis(test_split, num_samples: int = 150, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use a real unseen PDB interface patch from the test set (1AK4 if present)
    pos_idx = int(torch.where(test_split["labels"] == 1)[0][0].item())

    def measure_grad_variance(k: int, L: int):
        pos_1 = test_split["pos"][pos_idx:pos_idx + 1, :k, :]
        feats_1 = test_split["feats"][pos_idx:pos_idx + 1, :k, :]
        ctr_1 = test_split["center"][pos_idx:pos_idx + 1]
        pos_s, feats_s = canonical_sort_batch(pos_1, feats_1, center_pos=ctr_1)
        dist_s = torch.cdist(pos_s, pos_s, p=2, compute_mode="donot_use_mm_for_euclid_dist")

        ql = QMaSIFQuantumLayer(num_qubits=k, num_layers=L)
        grads_gamma0, grads_beta0 = [], []

        for _ in range(num_samples):
            with torch.no_grad():
                ql.omega.uniform_(-np.pi, np.pi)
                ql.gamma.uniform_(-np.pi, np.pi)
                ql.beta.uniform_(-np.pi, np.pi)
            ql.zero_grad()
            expvals = ql(feats_s, dist_s)
            loss = expvals[0, 0]  # Local observable <Z_0>
            loss.backward()
            grads_gamma0.append(float(ql.gamma.grad[0].item()))
            grads_beta0.append(float(ql.beta.grad[0, 0].item()))

        return float(np.var(grads_gamma0)), float(np.var(grads_beta0))

    print("\n--- [PART C] Barren Plateau Analysis: Gradient Variance vs Circuit Depth L (k=4) ---")
    depth_bp = {}
    for L in range(1, 9):
        var_g, var_b = measure_grad_variance(k=4, L=L)
        depth_bp[str(L)] = {"var_gamma0": var_g, "var_beta0": var_b}
        print(f"  Depth L={L} (k=4) | Var[d<Z0>/d_gamma0] = {var_g:.3e} | Var[d<Z0>/d_beta0] = {var_b:.3e}")

    print("\n--- [PART D] Barren Plateau Analysis: Gradient Variance vs Qubit Count k (L=2 & L=3) ---")
    qubit_bp = {"L2": {}, "L3": {}}
    for L_val, key in [(2, "L2"), (3, "L3")]:
        for k in [2, 4, 6, 8]:
            var_g, var_b = measure_grad_variance(k=k, L=L_val)
            qubit_bp[key][str(k)] = {"var_gamma0": var_g, "var_beta0": var_b}
            print(f"  Qubits k={k} (L={L_val}) | Var[d<Z0>/d_gamma0] = {var_g:.3e} | Var[d<Z0>/d_beta0] = {var_b:.3e}")

    return depth_bp, qubit_bp


# ==============================================================================
# 3. Main Execution
# ==============================================================================
def run_step5():
    t0 = time.time()
    print("=" * 78)
    print("STEP 5: NISQ DEPOLARIZING NOISE, FINITE SHOTS & BARREN PLATEAU ANALYSIS")
    print("=" * 78)

    data = torch.load("qmasif_real_pdb_site_dataset.pt", weights_only=False)
    ckpt = torch.load("qmasif_best_model.pt", weights_only=False)

    model = QMaSIFSiteModel(num_qubits=4, num_layers=2, use_canonical_sort=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_pos = data["test"]["pos"][:, :4, :]
    test_feats = data["test"]["feats"][:, :4, :]
    test_ctr = data["test"]["center"]
    y_true = data["test"]["labels"].cpu().numpy().astype(int)

    # 0. Verify parity with PennyLane default.mixed
    verify_noisy_parity_with_pennylane(model, test_pos, test_feats, test_ctr, noise_p=0.03)

    # ------------------------------------------------------------------
    # PART A: Depolarizing Noise Channel Sweep on Unseen Real PDB Test Set
    # ------------------------------------------------------------------
    print("\n--- [PART A] NISQ Gate Depolarizing Noise Sweep (Unseen PDB Test Set, N=848) ---")
    noise_rates = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    noise_results = []
    exact_probs_basis = None

    for p in noise_rates:
        logits, _, probs_basis = forward_noisy_density_matrix(model, test_pos, test_feats, test_ctr, noise_p=p)
        if p == 0.0:
            exact_probs_basis = probs_basis.cpu().numpy()
        probs = torch.sigmoid(logits).cpu().numpy()
        roc = float(roc_auc_score(y_true, probs) * 100.0)
        pr = float(average_precision_score(y_true, probs) * 100.0)
        margin = float(probs[y_true == 1].mean() - probs[y_true == 0].mean())

        noise_results.append({"p": p, "roc_auc": roc, "pr_auc": pr, "margin": margin})
        print(f"  Noise p={p*100:5.1f}% | Test ROC-AUC: {roc:5.2f}% | Test PR-AUC: {pr:5.2f}% | Prob Margin Δ: {margin:+.4f}")

    # ------------------------------------------------------------------
    # PART B: Finite-Shot Projective Measurement Volatility (10 MC Trials)
    # ------------------------------------------------------------------
    print("\n--- [PART B] Finite-Shot Measurement Stability (10 Monte Carlo Trials, N=848) ---")
    shot_list = [2000, 1000, 500, 200, 100]
    shot_results = []
    rng = np.random.default_rng(42)

    z_signs_np = model.quantum_layer.z_signs.cpu().numpy()  # [4, 16]
    w_lin = model.readout.weight.detach().cpu().numpy()[0]  # [4]
    b_lin = float(model.readout.bias.detach().cpu().numpy()[0])

    # Normalize probabilities strictly to sum to 1.0 in float64 for np.random.multinomial
    p_dist = exact_probs_basis.astype(np.float64)
    p_dist = p_dist / p_dist.sum(axis=1, keepdims=True)

    for S in shot_list:
        rocs_s, margins_s = [], []
        for _ in range(10):
            # Sample S projective measurements for each of the 848 test patches
            sampled_counts = np.array([rng.multinomial(S, row) for row in p_dist], dtype=np.float64)
            emp_probs = sampled_counts / float(S)                      # [848, 16]
            emp_expvals = emp_probs @ z_signs_np.T                     # [848, 4]
            logits_s = emp_expvals @ w_lin + b_lin                     # [848]
            probs_s = 1.0 / (1.0 + np.exp(-logits_s))
            rocs_s.append(float(roc_auc_score(y_true, probs_s) * 100.0))
            margins_s.append(float(probs_s[y_true == 1].mean() - probs_s[y_true == 0].mean()))

        res_s = {
            "shots": S,
            "roc_mean": float(np.mean(rocs_s)),
            "roc_std": float(np.std(rocs_s, ddof=1)),
            "margin_mean": float(np.mean(margins_s)),
            "margin_std": float(np.std(margins_s, ddof=1)),
        }
        shot_results.append(res_s)
        print(f"  Shots S={S:4d} | Test ROC-AUC: {res_s['roc_mean']:5.2f} ± {res_s['roc_std']:4.2f}% | Margin Δ: {res_s['margin_mean']:+.4f} ± {res_s['margin_std']:.4f}")

    # ------------------------------------------------------------------
    # PART C & D: Barren Plateau Analysis
    # ------------------------------------------------------------------
    depth_bp, qubit_bp = run_barren_plateau_analysis(data["test"], num_samples=150, seed=42)

    out_json = {
        "checkpoint_test_roc_auc": float(ckpt["test_roc_auc"] * 100.0),
        "depolarizing_noise": noise_results,
        "finite_shots": shot_results,
        "barren_plateau_depth": depth_bp,
        "barren_plateau_qubits": qubit_bp,
    }
    with open("step5_nisq_bp_results.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # ------------------------------------------------------------------
    # Publication-Quality Plots (100% Consistent with Measured Results)
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Fig 5a: NISQ Noise & Finite Shots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    ps_pct = [r["p"] * 100.0 for r in noise_results]
    rocs_n = [r["roc_auc"] for r in noise_results]
    margins_n = [r["margin"] for r in noise_results]

    ax1.plot(ps_pct, rocs_n, "-o", color="#0284C7", lw=2.5, label="Test ROC-AUC (%)")
    ax1.axhline(50.0, color="#EF4444", linestyle="--", lw=1.4, label="Random Chance (50%)")
    ax1.set_xlabel("2-Qubit Gate Depolarizing Noise Rate p (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Unseen PDB Test ROC-AUC (%)", fontsize=10, fontweight="bold", color="#0284C7")
    ax1.tick_params(axis="y", labelcolor="#0284C7")
    ax1.set_ylim(47, max(rocs_n) + 4)
    ax1.set_title("(a) NISQ Depolarizing Noise Resilience", fontsize=11.5, fontweight="bold")

    ax1_t = ax1.twinx()
    ax1_t.plot(ps_pct, margins_n, "--s", color="#F59E0B", lw=2.0, label="Prob. Margin Δ")
    ax1_t.set_ylabel("Probability Margin (μ_pos - μ_neg)", fontsize=10, fontweight="bold", color="#D97706")
    ax1_t.tick_params(axis="y", labelcolor="#D97706")
    ax1_t.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, fontsize=8.5)

    # Panel B: Finite Shots
    x_labels = [str(r["shots"]) for r in shot_results]
    s_means = [r["roc_mean"] for r in shot_results]
    s_stds = [r["roc_std"] for r in shot_results]
    ideal_roc = noise_results[0]["roc_auc"]

    ax2.errorbar(x_labels, s_means, yerr=s_stds, fmt="-s", color="#16A34A", ecolor="#86EFAC",
                 elinewidth=2.2, capsize=4, lw=2.5, label="Finite-Shot Test ROC-AUC (10 Trials)")
    ax2.axhline(ideal_roc, color="#0284C7", linestyle="--", lw=1.8, label=f"Exact Statevector ({ideal_roc:.2f}%)")
    ax2.set_xlabel("Number of Measurement Shots (S)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Unseen PDB Test ROC-AUC (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(min(s_means) - 5, ideal_roc + 3)
    ax2.set_title("(b) Finite-Shot Measurement Stability", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig("Fig5a_NISQ_Noise_and_Shots.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 6: Barren Plateau Analysis (Depth L & Qubits k)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    ls = list(range(1, 9))
    vg_l = [depth_bp[str(L)]["var_gamma0"] for L in ls]
    vb_l = [depth_bp[str(L)]["var_beta0"] for L in ls]

    ax1.semilogy(ls, vg_l, "-o", color="#E11D48", lw=2.4, markersize=7, label=r"$\mathrm{Var}[\partial \langle Z_0 \rangle / \partial \gamma_0]$ (ZZ)")
    ax1.semilogy(ls, vb_l, "--s", color="#0284C7", lw=2.2, markersize=6, label=r"$\mathrm{Var}[\partial \langle Z_0 \rangle / \partial \beta_0]$ (RX)")
    ax1.axvspan(1, 3, color="#16A34A", alpha=0.10, label="Shallow Regime (L ≤ 3)")
    ax1.set_xlabel("Circuit Layer Depth (L)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Gradient Variance (Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Gradient Variance vs. Circuit Depth (k=4)", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True, fontsize=8.5)
    ax1.grid(True, which="both", ls="--", alpha=0.45)

    ks = [2, 4, 6, 8]
    vg_k2 = [qubit_bp["L2"][str(k)]["var_gamma0"] for k in ks]
    vg_k3 = [qubit_bp["L3"][str(k)]["var_gamma0"] for k in ks]
    vb_k2 = [qubit_bp["L2"][str(k)]["var_beta0"] for k in ks]

    ax2.semilogy(ks, vg_k2, "-o", color="#E11D48", lw=2.4, markersize=7, label=r"ZZ Grad Var (L=2)")
    ax2.semilogy(ks, vg_k3, "-^", color="#9333EA", lw=2.4, markersize=7, label=r"ZZ Grad Var (L=3)")
    ax2.semilogy(ks, vb_k2, "--s", color="#0284C7", lw=2.2, markersize=6, label=r"RX Grad Var (L=2)")
    ax2.set_xticks(ks)
    ax2.set_xlabel("Number of Qubits (k)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Gradient Variance (Log Scale)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Gradient Variance vs. Qubit Register Size (k)", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True, fontsize=8.5)
    ax2.grid(True, which="both", ls="--", alpha=0.45)

    plt.tight_layout()
    plt.savefig("Fig6_Barren_Plateaus.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ STEP 5 COMPLETED in {time.time()-t0:.1f}s!")
    print("   Saved: step5_nisq_bp_results.json, Fig5a_NISQ_Noise_and_Shots.png, Fig6_Barren_Plateaus.png")


if __name__ == "__main__":
    run_step5()