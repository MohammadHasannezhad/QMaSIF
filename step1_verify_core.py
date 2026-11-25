import numpy as np
import torch
import pennylane as qml
from qmasif_core import QMaSIFSiteModel, canonical_sort_batch


def get_random_o3_matrix(reflection: bool = False) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(3, 3))
    d = torch.diag(r)
    q = q @ torch.diag(d / torch.abs(d))
    det = torch.det(q)
    if (not reflection and det < 0) or (reflection and det > 0):
        q[:, 0] = -q[:, 0]
    return q


def build_pennylane_qnode(num_qubits: int = 4, num_layers: int = 2, eps: float = 1e-5):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(feats, dist_mat, omega, gamma, beta):
        for l in range(num_layers):
            for i in range(num_qubits):
                qml.RY(0.5 * np.pi + omega[l, 0] * feats[i, 0], wires=i)
                qml.RZ(omega[l, 1] * (feats[i, 1] - 0.5), wires=i)
                qml.RX(omega[l, 2] * (feats[i, 2] - 0.5), wires=i)
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    d_ij = dist_mat[i, j] + eps
                    qml.IsingZZ(gamma[l] / d_ij, wires=[i, j])
            for i in range(num_qubits):
                qml.RX(beta[l, 0], wires=i)
                qml.RY(beta[l, 1], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return circuit


def run_all_verifications():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 72)
    print("STEP 1 (v3): DATA RE-UPLOADING QMaSIF CORE VERIFICATION")
    print("=" * 72)

    model = QMaSIFSiteModel(num_qubits=4, num_layers=2)
    counts = model.count_parameters()
    print(f"[INFO] Parameter Budget -> Quantum: {counts['quantum']} | Linear Readout: {counts['classical_readout']} | Total: {counts['total']}")

    pos = (torch.rand(4, 4, 3) * 2.0 - 1.0) * 8.0
    feats = torch.cat([torch.rand(4, 4, 1) * 2.0 - 1.0, torch.rand(4, 4, 1), torch.rand(4, 4, 1)], dim=-1)

    pos_s, feats_s = canonical_sort_batch(pos[:1], feats[:1])
    dist_s = torch.cdist(pos_s, pos_s, p=2, compute_mode="donot_use_mm_for_euclid_dist")

    model.zero_grad()
    out_torch = model.quantum_layer(feats_s, dist_s)[0]
    out_torch.sum().backward()
    g_omega = model.quantum_layer.omega.grad.detach().clone()
    g_gamma = model.quantum_layer.gamma.grad.detach().clone()
    g_beta = model.quantum_layer.beta.grad.detach().clone()

    pl_circuit = build_pennylane_qnode(num_qubits=4, num_layers=2)
    om_pl = model.quantum_layer.omega.detach().clone().double().requires_grad_(True)
    ga_pl = model.quantum_layer.gamma.detach().clone().double().requires_grad_(True)
    be_pl = model.quantum_layer.beta.detach().clone().double().requires_grad_(True)
    out_pl = torch.stack(pl_circuit(feats_s[0].double(), dist_s[0].double(), om_pl, ga_pl, be_pl))
    out_pl.sum().backward()

    fwd_diff = torch.max(torch.abs(out_torch.detach() - out_pl.detach().float())).item()
    om_diff = torch.max(torch.abs(g_omega - om_pl.grad.float())).item()
    ga_diff = torch.max(torch.abs(g_gamma - ga_pl.grad.float())).item()
    be_diff = torch.max(torch.abs(g_beta - be_pl.grad.float())).item()

    print(f"\n[TEST 1] PennyLane Parameter-Shift Parity -> Fwd: {fwd_diff:.2e} | dOmega: {om_diff:.2e} | dGamma: {ga_diff:.2e} | dBeta: {be_diff:.2e}")
    assert max(fwd_diff, om_diff, ga_diff, be_diff) < 1e-5
    print("  STATUS: PASS")

    print(f"\n[TEST 2] Non-Zero Gradients -> min|dOmega|: {g_omega.abs().min():.4f} | min|dGamma|: {g_gamma.abs().min():.4f} | min|dBeta|: {g_beta.abs().min():.4f}")
    assert g_omega.abs().min() > 1e-4 and g_gamma.abs().min() > 1e-4 and g_beta.abs().min() > 1e-4
    print("  STATUS: PASS")

    with torch.no_grad():
        base_q = model.quantum_layer(feats_s, dist_s)
        d_geom = torch.max(torch.abs(base_q - model.quantum_layer(feats_s, dist_s * 1.5))).item()
        f_q = feats_s.clone(); f_q[:, :, 0] = torch.clamp(f_q[:, :, 0] + 0.3, -1, 1)
        d_q = torch.max(torch.abs(base_q - model.quantum_layer(f_q, dist_s))).item()
        f_h = feats_s.clone(); f_h[:, :, 1] = torch.clamp(f_h[:, :, 1] + 0.3, 0, 1)
        d_h = torch.max(torch.abs(base_q - model.quantum_layer(f_h, dist_s))).item()
        f_s = feats_s.clone(); f_s[:, :, 2] = torch.clamp(f_s[:, :, 2] + 0.3, 0, 1)
        d_s = torch.max(torch.abs(base_q - model.quantum_layer(f_s, dist_s))).item()

    print(f"\n[TEST 3] Physical Sensitivity -> d_ij: {d_geom:.4f} | q: {d_q:.4f} | h: {d_h:.4f} | s: {d_s:.4f}")
    assert min(d_geom, d_q, d_h, d_s) > 1e-2
    print("  STATUS: PASS")

    model.eval()
    with torch.no_grad():
        z_orig = model(pos, feats)
        R_rot = get_random_o3_matrix(reflection=False)
        t_vec = torch.randn(1, 1, 3) * 50.0
        err_se3 = torch.max(torch.abs(z_orig - model(torch.matmul(pos, R_rot) + t_vec, feats))).item()
        R_ref = get_random_o3_matrix(reflection=True)
        err_ref = torch.max(torch.abs(z_orig - model(torch.matmul(pos, R_ref) - t_vec, feats))).item()

        max_perm_err = 0.0
        for _ in range(20):
            perm = torch.randperm(4)
            max_perm_err = max(max_perm_err, torch.max(torch.abs(z_orig - model(pos[:, perm, :], feats[:, perm, :]))).item())

    print(f"\n[TEST 4 & 5] Symmetries -> SO(3)+50A: {err_se3:.2e} | Reflection: {err_ref:.2e} | Perm: {max_perm_err:.2e}")
    assert max(err_se3, err_ref, max_perm_err) < 1e-5
    print("  STATUS: PASS")
    print("=" * 72)


if __name__ == "__main__":
    run_all_verifications()