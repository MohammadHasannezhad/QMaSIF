import numpy as np
import torch
import torch.nn as nn


def canonical_sort_batch(pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
    """
    Algorithm 1: Vectorized Deterministic Pseudo-Canonical Neighborhood Ordering.
    Ensures strict permutation invariance and E(3) invariance with zero Python loops.
    """
    if center_pos is None:
        center_pos = pos.mean(dim=1)

    d_center = torch.norm(pos - center_pos.unsqueeze(1), dim=-1)

    d_r = torch.round(d_center * 1e4)
    q_r = torch.round(feats[:, :, 0] * 1e4)
    h_r = torch.round(feats[:, :, 1] * 1e4)
    s_r = torch.round(feats[:, :, 2] * 1e4)

    idx = torch.argsort(s_r, dim=1, stable=True)
    idx = torch.gather(idx, 1, torch.argsort(torch.gather(h_r, 1, idx), dim=1, stable=True))
    idx = torch.gather(idx, 1, torch.argsort(torch.gather(q_r, 1, idx), dim=1, stable=True))
    idx = torch.gather(idx, 1, torch.argsort(torch.gather(d_r, 1, idx), dim=1, stable=True))

    idx_pos = idx.unsqueeze(-1).expand(-1, -1, pos.shape[-1])
    idx_feats = idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1])
    return torch.gather(pos, 1, idx_pos), torch.gather(feats, 1, idx_feats)


class QMaSIFQuantumLayer(nn.Module):
    """
    Vectorized exact statevector implementation of QMaSIF VQC with Data Re-uploading.
    Mathematically identical to PennyLane default.qubit with L layers of:
      1) Shared-scale SU(2) Data Re-uploading per qubit i:
           RY(pi/2 + omega_{l,0}*q_i) -> RZ(omega_{l,1}*(h_i - 0.5)) -> RX(omega_{l,2}*(s_i - 0.5))
      2) All-to-all distance-conditioned IsingZZ(gamma_l / (d_ij + eps))
      3) Shared global SU(2) mixing: RX(beta_{l,0}) -> RY(beta_{l,1})
      4) Pauli-Z expectation measurement on all k qubits.

    For L=2: exactly 12 trainable quantum parameters (6 per layer), strictly E(3) & permutation invariant.
    """

    def __init__(self, num_qubits: int = 4, num_layers: int = 2, eps: float = 1e-5,
                 use_entanglement: bool = True, distance_conditioned: bool = True):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.eps = eps
        self.use_entanglement = use_entanglement
        self.distance_conditioned = distance_conditioned

        # 6 shared quantum parameters per layer: 3 encoding scales + 1 ZZ coupling + 2 mixing angles
        self.omega = nn.Parameter(torch.tensor([[1.2, 2.0, 1.5]], dtype=torch.float32).repeat(num_layers, 1))
        self.gamma = nn.Parameter(torch.linspace(0.5, 1.0, num_layers))
        self.beta = nn.Parameter(torch.tensor([[0.5, 0.3]], dtype=torch.float32).repeat(num_layers, 1))

        dim = 1 << num_qubits
        indices = torch.arange(dim, dtype=torch.long)

        z_signs = torch.empty((num_qubits, dim), dtype=torch.float32)
        for q in range(num_qubits):
            bit = (indices >> (num_qubits - 1 - q)) & 1
            z_signs[q] = torch.where(bit == 0, 1.0, -1.0)
        self.register_buffer("z_signs", z_signs)

        pairs, pair_signs = [], []
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pairs.append((i, j))
                pair_signs.append(z_signs[i] * z_signs[j])
        self.pairs = pairs
        if pair_signs:
            self.register_buffer("pair_signs", torch.stack(pair_signs, dim=0))
        else:
            self.register_buffer("pair_signs", torch.empty((0, dim), dtype=torch.float32))

    def _apply_1q_batched(self, state: torch.Tensor, U: torch.Tensor, q_idx: int) -> torch.Tensor:
        B = state.shape[0]
        k = self.num_qubits
        s = state.view([B] + [2] * k)

        dims = [0, q_idx + 1] + [d + 1 for d in range(k) if d != q_idx]
        s_perm = s.permute(dims)
        perm_shape = s_perm.shape
        s_flat = s_perm.reshape(B, 2, -1)

        if U.dim() == 2:
            s_out = torch.einsum("ij,bjn->bin", U, s_flat)
        else:
            s_out = torch.bmm(U, s_flat)

        s_out = s_out.reshape(perm_shape)
        inv_dims = [0] * (k + 1)
        for new_d, old_d in enumerate(dims):
            inv_dims[old_d] = new_d
        return s_out.permute(inv_dims).reshape(B, -1)

    def forward(self, feats: torch.Tensor, dist_mat: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        dim = 1 << self.num_qubits
        device = feats.device

        state = torch.zeros((B, dim), dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j

        for l in range(self.num_layers):
            # 1. Data Re-uploading SU(2) Encoding
            for i in range(self.num_qubits):
                q_val = feats[:, i, 0]
                h_val = feats[:, i, 1] - 0.5
                s_val = feats[:, i, 2] - 0.5

                zeros = torch.zeros_like(q_val)
                zero_c = torch.complex(zeros, zeros)

                # RY(pi/2 + omega[l,0] * q)
                ry_half = 0.5 * (0.5 * np.pi + self.omega[l, 0] * q_val)
                c_y, s_y = torch.cos(ry_half), torch.sin(ry_half)
                ry_mat = torch.stack([
                    torch.stack([torch.complex(c_y, zeros), torch.complex(-s_y, zeros)], dim=-1),
                    torch.stack([torch.complex(s_y, zeros), torch.complex(c_y, zeros)], dim=-1),
                ], dim=-2)
                state = self._apply_1q_batched(state, ry_mat, i)

                # RZ(omega[l,1] * (h - 0.5))
                rz_half = 0.5 * (self.omega[l, 1] * h_val)
                c_z, s_z = torch.cos(rz_half), torch.sin(rz_half)
                rz_mat = torch.stack([
                    torch.stack([torch.complex(c_z, -s_z), zero_c], dim=-1),
                    torch.stack([zero_c, torch.complex(c_z, s_z)], dim=-1),
                ], dim=-2)
                state = self._apply_1q_batched(state, rz_mat, i)

                # RX(omega[l,2] * (s - 0.5))
                rx_half = 0.5 * (self.omega[l, 2] * s_val)
                c_x, s_x = torch.cos(rx_half), torch.sin(rx_half)
                rx_mat = torch.stack([
                    torch.stack([torch.complex(c_x, zeros), torch.complex(zeros, -s_x)], dim=-1),
                    torch.stack([torch.complex(zeros, -s_x), torch.complex(c_x, zeros)], dim=-1),
                ], dim=-2)
                state = self._apply_1q_batched(state, rx_mat, i)

            # 2. Distance-Conditioned Ising-ZZ Entanglement
            if self.use_entanglement and len(self.pairs) > 0:
                total_half_phase = torch.zeros((B, dim), dtype=torch.float32, device=device)
                for p_idx, (i, j) in enumerate(self.pairs):
                    if self.distance_conditioned:
                        d_ij = dist_mat[:, i, j] + self.eps
                        theta_ij = self.gamma[l] / d_ij
                    else:
                        theta_ij = self.gamma[l].expand(B)
                    total_half_phase = total_half_phase + 0.5 * theta_ij.unsqueeze(1) * self.pair_signs[p_idx].unsqueeze(0)

                zz_phase = torch.complex(torch.cos(total_half_phase), -torch.sin(total_half_phase))
                state = state * zz_phase

            # 3. Shared Global SU(2) Mixing: RX(beta[l,0]) -> RY(beta[l,1])
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

            mix_mat = torch.matmul(ry_mix, rx_mix)  # RY(beta_1) @ RX(beta_0)
            for i in range(self.num_qubits):
                state = self._apply_1q_batched(state, mix_mat, i)

        probs = state.real ** 2 + state.imag ** 2
        return torch.matmul(probs, self.z_signs.t())


class QMaSIFSiteModel(nn.Module):
    """
    Ultra-lightweight E(3)-Invariant QMaSIF with Data Re-uploading.
    For k=4, L=2:
      - Quantum parameters : 12 (6 shared params per layer)
      - Linear readout     : 5 (4 weights + 1 bias)
      - Total trainable    : 17 parameters
    """

    def __init__(self, num_qubits: int = 4, num_layers: int = 2,
                 use_canonical_sort: bool = True, use_entanglement: bool = True,
                 distance_conditioned: bool = True):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.use_canonical_sort = use_canonical_sort

        self.quantum_layer = QMaSIFQuantumLayer(
            num_qubits=num_qubits,
            num_layers=num_layers,
            use_entanglement=use_entanglement,
            distance_conditioned=distance_conditioned,
        )
        self.readout = nn.Linear(num_qubits, 1)

    def count_parameters(self):
        q_params = sum(p.numel() for p in self.quantum_layer.parameters() if p.requires_grad)
        c_params = sum(p.numel() for p in self.readout.parameters() if p.requires_grad)
        return {"quantum": q_params, "classical_readout": c_params, "total": q_params + c_params}

    def forward(self, pos: torch.Tensor, feats: torch.Tensor,
                center_pos: torch.Tensor = None, return_quantum: bool = False):
        if self.use_canonical_sort:
            pos, feats = canonical_sort_batch(pos, feats, center_pos=center_pos)

        dist_mat = torch.cdist(pos, pos, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        q_out = self.quantum_layer(feats, dist_mat)
        logits = self.readout(q_out).squeeze(-1)

        if return_quantum:
            return logits, q_out
        return logits