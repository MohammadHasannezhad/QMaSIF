import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score
from Bio.Align import PairwiseAligner, substitution_matrices
import matplotlib.pyplot as plt

from qmasif_core import QMaSIFSiteModel, canonical_sort_batch
from step2_build_pdb_dataset import (
    parse_pdb_chains,
    compute_unbound_surface_fields,
    extract_surface_patch,
)


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ==============================================================================
# 1. Exact Sequence Extraction & BLOSUM62 Smith-Waterman <=30% Clustering
# ==============================================================================
def get_chain_sequence(chain_dict, pdb_path: str, cid: str) -> str:
    if "seq" in chain_dict and isinstance(chain_dict["seq"], str) and len(chain_dict["seq"]) > 0:
        return chain_dict["seq"]
    if "res_names" in chain_dict:
        return "".join(AA3_TO_1.get(r, "X") for r in chain_dict["res_names"])

    seen = set()
    seq = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  "):
                continue
            atom_name = line[12:16].strip()
            ch = line[21].strip() or "_"
            if ch != cid or atom_name != "CA":
                continue
            res_id = line[22:27].strip()
            if res_id in seen:
                continue
            seen.add(res_id)
            res_name = line[17:20].strip()
            seq.append(AA3_TO_1.get(res_name, "X"))
    return "".join(seq)


def build_blosum62_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    return aligner


def sw_percent_identity(aligner, seq1: str, seq2: str, min_score: float = 50.0) -> float:
    s1 = "".join(c for c in seq1 if c in AA3_TO_1.values())
    s2 = "".join(c for c in seq2 if c in AA3_TO_1.values())
    if len(s1) < 10 or len(s2) < 10:
        return 0.0
    alns = aligner.align(s1, s2)
    if len(alns) == 0:
        return 0.0
    best = alns[0]
    if best.score < min_score:
        return 0.0
    blocks_a, blocks_b = best.aligned
    matches = 0
    for (sa, ea), (sb, eb) in zip(blocks_a, blocks_b):
        matches += sum(c1 == c2 for c1, c2 in zip(s1[sa:ea], s2[sb:eb]))
    return 100.0 * matches / float(min(len(s1), len(s2)))


def cluster_complexes_and_make_5folds(complex_records, identity_thresh: float = 30.0, n_folds: int = 5):
    """
    Exact BLOSUM62 Smith-Waterman <=30% single-linkage clustering across all complexes,
    followed by balanced 5-fold assignment at the cluster level + cross-fold leakage audit.
    """
    n = len(complex_records)
    aligner = build_blosum62_aligner()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pair_id_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        seqs_i = [complex_records[i]["seq_a"], complex_records[i]["seq_b"]]
        for j in range(i + 1, n):
            seqs_j = [complex_records[j]["seq_a"], complex_records[j]["seq_b"]]
            max_pid = 0.0
            for si in seqs_i:
                for sj in seqs_j:
                    pid = sw_percent_identity(aligner, si, sj)
                    if pid > max_pid:
                        max_pid = pid
            pair_id_mat[i, j] = pair_id_mat[j, i] = max_pid
            if max_pid >= identity_thresh:
                union(i, j)

    cluster_map = {}
    for i in range(n):
        r = find(i)
        cluster_map.setdefault(r, []).append(i)

    # Sort clusters deterministically (largest first for balanced bin-packing across 5 folds)
    clusters = sorted(
        cluster_map.values(),
        key=lambda grp: (-len(grp), complex_records[grp[0]]["entry"]),
    )

    fold_clusters = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for cid_idx, grp in enumerate(clusters):
        best_f = int(np.argmin(fold_sizes))
        fold_clusters[best_f].append((cid_idx, grp))
        fold_sizes[best_f] += len(grp)

    # Verify strict zero cross-fold homology leakage (<30%)
    max_cross_fold_pid = 0.0
    for f1 in range(n_folds):
        idx1 = [idx for _, grp in fold_clusters[f1] for idx in grp]
        for f2 in range(f1 + 1, n_folds):
            idx2 = [idx for _, grp in fold_clusters[f2] for idx in grp]
            sub = pair_id_mat[np.ix_(idx1, idx2)]
            if sub.size > 0:
                max_cross_fold_pid = max(max_cross_fold_pid, float(np.max(sub)))

    print(f"[CLUSTERING] {n} complexes -> {len(clusters)} homology-independent clusters (<= {identity_thresh:.0f}% BLOSUM62 SW)")
    print(f"[CLUSTERING] 5-Fold complex counts: {fold_sizes}")
    print(f"[VERIFY]     Max Cross-Fold Sequence Identity: {max_cross_fold_pid:.2f}% (Target < {identity_thresh:.0f}%)")
    assert max_cross_fold_pid < identity_thresh, f"Homology leakage detected: {max_cross_fold_pid:.2f}%"
    print("             STATUS: PASS (Zero cross-fold homology leakage verified!)")

    return clusters, fold_clusters, max_cross_fold_pid


# ==============================================================================
# 2. Per-Complex Real PDB Patch Extraction (Balanced + Full Unbalanced Surface)
# ==============================================================================
def compute_exact_heavy_distances(chain_target, chain_partner) -> np.ndarray:
    partner_atoms = np.vstack(chain_partner["heavy_coords"])
    tree = cKDTree(partner_atoms)
    min_dists = np.empty(len(chain_target["heavy_coords"]), dtype=np.float32)
    for i, res_atoms in enumerate(chain_target["heavy_coords"]):
        d, _ = tree.query(res_atoms, k=1)
        min_dists[i] = float(np.min(d))
    return min_dists


def load_all_60_complexes_per_complex(k: int = 4, radius: float = 8.0):
    data = torch.load("qmasif_real_pdb_site_dataset.pt", weights_only=False)
    all_entries = (
        list(data["train"]["complexes"])
        + list(data["val"]["complexes"])
        + list(data["test"]["complexes"])
    )
    print(f"[DATA] Extracting per-complex patches & full surfaces for all {len(all_entries)} real PDB complexes...")

    records = []
    rng = np.random.default_rng(42)

    for entry in all_entries:
        pdb_id, cid_a, cid_b = entry.split("_")
        pdb_path = os.path.join("pdb_cache", f"{pdb_id.lower()}.pdb")
        chains = parse_pdb_chains(pdb_path)
        ca = compute_unbound_surface_fields(chains[cid_a])
        cb = compute_unbound_surface_fields(chains[cid_b])
        seq_a = get_chain_sequence(ca, pdb_path, cid_a)
        seq_b = get_chain_sequence(cb, pdb_path, cid_b)

        all_pos, all_feat, all_ctr, all_dists = [], [], [], []
        for target_c, partner_c in [(ca, cb), (cb, ca)]:
            dists = compute_exact_heavy_distances(target_c, partner_c)
            for idx in np.where(target_c["surf_mask"])[0]:
                patch = extract_surface_patch(target_c, int(idx), k_max=max(k, 8), radius=radius)
                if patch is None:
                    continue
                p_pos, p_feat, p_ctr = patch
                all_pos.append(p_pos[:k])
                all_feat.append(p_feat[:k])
                all_ctr.append(p_ctr)
                all_dists.append(float(dists[idx]))

        pos_arr = np.array(all_pos, dtype=np.float32)
        feat_arr = np.array(all_feat, dtype=np.float32)
        ctr_arr = np.array(all_ctr, dtype=np.float32)
        dist_arr = np.array(all_dists, dtype=np.float32)
        full_labels = (dist_arr <= 5.0).astype(np.float32)

        # Balanced patches (<=5A positive vs >10A negative, 1:1 per complex)
        pos_idx = np.where(dist_arr <= 5.0)[0]
        neg_idx = np.where(dist_arr > 10.0)[0]
        n_bal = min(len(pos_idx), len(neg_idx))
        if n_bal == 0:
            continue
        chosen_pos = rng.choice(pos_idx, size=n_bal, replace=False) if len(pos_idx) > n_bal else pos_idx
        chosen_neg = rng.choice(neg_idx, size=n_bal, replace=False) if len(neg_idx) > n_bal else neg_idx
        bal_idx = np.concatenate([chosen_pos, chosen_neg])

        records.append({
            "entry": entry,
            "seq_a": seq_a,
            "seq_b": seq_b,
            # Balanced split tensors (for training, validation, and balanced CV evaluation)
            "bal_pos": torch.tensor(pos_arr[bal_idx], dtype=torch.float32),
            "bal_feats": torch.tensor(feat_arr[bal_idx], dtype=torch.float32),
            "bal_ctr": torch.tensor(ctr_arr[bal_idx], dtype=torch.float32),
            "bal_labels": torch.tensor(full_labels[bal_idx], dtype=torch.float32),
            # Full natural surface tensors (for unbalanced full-surface evaluation)
            "full_pos": torch.tensor(pos_arr, dtype=torch.float32),
            "full_feats": torch.tensor(feat_arr, dtype=torch.float32),
            "full_ctr": torch.tensor(ctr_arr, dtype=torch.float32),
            "full_labels": torch.tensor(full_labels, dtype=torch.float32),
        })

    tot_bal = sum(len(r["bal_labels"]) for r in records)
    tot_full = sum(len(r["full_labels"]) for r in records)
    print(f"[DATA] Loaded {len(records)} complexes | Balanced patches: {tot_bal} | Full surface residues: {tot_full}")
    return records


def concat_records(records, indices):
    return (
        torch.cat([records[i]["bal_pos"] for i in indices], dim=0),
        torch.cat([records[i]["bal_feats"] for i in indices], dim=0),
        torch.cat([records[i]["bal_ctr"] for i in indices], dim=0),
        torch.cat([records[i]["bal_labels"] for i in indices], dim=0),
    )


# ==============================================================================
# 3. Model Zoo (Quantum + Iso-Parameter Classical + Full Classical Baselines)
# ==============================================================================
class CenterOnlyLogistic(nn.Module):
    """4-parameter linear baseline using ONLY the central surface residue's (q, h, s)."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        _, feats_s = canonical_sort_batch(pos, feats, center_pos=center_pos)
        return self.lin(feats_s[:, 0, :]).squeeze(-1)


class IsoParamLinear18(nn.Module):
    """19-parameter linear model on the exact 18 canonically sorted invariant features."""
    def __init__(self, num_nodes: int = 4):
        super().__init__()
        in_dim = num_nodes * 3 + (num_nodes * (num_nodes - 1)) // 2
        self.lin = nn.Linear(in_dim, 1)
        idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
        self.register_buffer("triu_i", idx[0])
        self.register_buffer("triu_j", idx[1])

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        pos_s, feats_s = canonical_sort_batch(pos, feats, center_pos=center_pos)
        dist_mat = torch.cdist(pos_s, pos_s, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        d_ij = dist_mat[:, self.triu_i, self.triu_j] / 10.0
        x = torch.cat([feats_s.reshape(pos.shape[0], -1), d_ij], dim=-1)
        return self.lin(x).squeeze(-1)


class IsoParamTinyDeepSetMLP(nn.Module):
    """
    EXACT 17-parameter non-linear E(3)- and permutation-invariant classical baseline!
      - Shared node encoder: Linear(4, 2) + SiLU on [q_i, h_i, s_i, d_{0i}/10] -> 10 params
      - Symmetric readout on [h_center(2), mean(h_nbrs)(2), mean(d_ctr)(1), mean(d_pair)(1)] -> Linear(6, 1) -> 7 params
      Total = 17 trainable parameters (exact 1:1 parity with QMaSIF L=2).
    """
    def __init__(self, num_nodes: int = 4):
        super().__init__()
        self.node_enc = nn.Linear(4, 2)
        self.act = nn.SiLU()
        self.readout = nn.Linear(6, 1)
        idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
        self.register_buffer("triu_i", idx[0])
        self.register_buffer("triu_j", idx[1])

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        B, k, _ = pos.shape
        if center_pos is None:
            center_pos = pos[:, 0, :]
        d_ctr = torch.norm(pos - center_pos.unsqueeze(1), dim=-1)  # [B, k]
        ctr_idx = torch.argmin(d_ctr, dim=1)                       # [B]

        node_in = torch.cat([feats, (d_ctr / 10.0).unsqueeze(-1)], dim=-1)  # [B, k, 4]
        h = self.act(self.node_enc(node_in))                                # [B, k, 2]

        h_center = h[torch.arange(B, device=pos.device), ctr_idx]           # [B, 2]
        h_nbr_mean = (h.sum(dim=1) - h_center) / max(1, k - 1)              # [B, 2]

        dist_mat = torch.cdist(pos, pos, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        mean_d_pair = (dist_mat[:, self.triu_i, self.triu_j] / 10.0).mean(dim=1, keepdim=True)
        mean_d_ctr = (d_ctr.sum(dim=1, keepdim=True) / (10.0 * max(1, k - 1)))

        z = torch.cat([h_center, h_nbr_mean, mean_d_ctr, mean_d_pair], dim=-1)  # [B, 6]
        return self.readout(z).squeeze(-1)


class FairClassicalMLP(nn.Module):
    """321-parameter 2-layer ReLU MLP on the exact 18 canonically sorted invariant features."""
    def __init__(self, num_nodes: int = 4):
        super().__init__()
        in_dim = num_nodes * 3 + (num_nodes * (num_nodes - 1)) // 2
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 1))
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
    """3,345-parameter 2-layer E(3)-invariant EGNN."""
    def __init__(self, in_dim: int = 3, hidden_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.node_in = nn.Linear(in_dim + 1, hidden_dim)
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
            for _ in range(num_layers)
        ])
        self.node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim * 2, 1)

    def forward(self, pos: torch.Tensor, feats: torch.Tensor, center_pos: torch.Tensor = None):
        B, k, _ = pos.shape
        if center_pos is None:
            center_pos = pos[:, 0, :]
        d_ctr = torch.norm(pos - center_pos.unsqueeze(1), dim=-1)
        ctr_idx = torch.argmin(d_ctr, dim=1)

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


# ==============================================================================
# 4. Training & Strict Out-of-Fold Evaluation + Statistical Analysis
# ==============================================================================
def train_fold_model(model_fn, train_t, val_t, seed: int, epochs: int, batch_size: int, lr: float, wd: float):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(*train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_val_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    v_pos, v_feats, v_ctr, v_y = val_t
    v_y_np = v_y.numpy().astype(int)

    for _ in range(epochs):
        model.train()
        for pos, feats, ctr, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(pos, feats, center_pos=ctr), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_probs = torch.sigmoid(model(v_pos, v_feats, center_pos=v_ctr)).numpy()
        v_auc = float(roc_auc_score(v_y_np, v_probs))
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


def safe_wilcoxon(a: np.ndarray, b: np.ndarray):
    diff = a - b
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    stat, pval = stats.wilcoxon(a, b, alternative="two-sided")
    return float(stat), float(pval)


def cluster_bootstrap_ci(cluster_scores: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Resample entire independent homology clusters (N=37) with replacement."""
    rng = np.random.default_rng(seed)
    m = len(cluster_scores)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        boot_means[b] = np.mean(cluster_scores[idx])
    return (
        float(np.mean(cluster_scores)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def cluster_bootstrap_diff_ci(scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    m = len(scores_a)
    diffs = scores_a - scores_b
    boot_d = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        boot_d[b] = np.mean(diffs[idx])
    return (
        float(np.mean(diffs)),
        float(np.percentile(boot_d, 2.5)),
        float(np.percentile(boot_d, 97.5)),
    )


def run_step7():
    t_global = time.time()
    print("=" * 84)
    print("PRIORITY 1: 5-FOLD CLUSTER-LEVEL CV (ALL 60 PDBs, 37 CLUSTERS) + ISO-PARAM BASELINES")
    print("=" * 84)

    records = load_all_60_complexes_per_complex(k=4, radius=8.0)
    clusters, fold_clusters, max_cross_pid = cluster_complexes_and_make_5folds(
        records, identity_thresh=30.0, n_folds=5
    )

    model_specs = {
        "QMaSIF (L=2, 17p)": {
            "fn": lambda: QMaSIFSiteModel(num_qubits=4, num_layers=2, use_canonical_sort=True),
            "lr": 0.03, "wd": 0.0,
        },
        "QMaSIF (L=3, 23p)": {
            "fn": lambda: QMaSIFSiteModel(num_qubits=4, num_layers=3, use_canonical_sort=True),
            "lr": 0.03, "wd": 0.0,
        },
        "Center-Only Logistic (4p)": {
            "fn": lambda: CenterOnlyLogistic(),
            "lr": 0.02, "wd": 1e-4,
        },
        "Iso-Param Linear (19p)": {
            "fn": lambda: IsoParamLinear18(num_nodes=4),
            "lr": 0.02, "wd": 1e-4,
        },
        "Iso-Param Tiny MLP (17p)": {
            "fn": lambda: IsoParamTinyDeepSetMLP(num_nodes=4),
            "lr": 0.02, "wd": 1e-4,
        },
        "Fair Classical MLP (321p)": {
            "fn": lambda: FairClassicalMLP(num_nodes=4),
            "lr": 0.01, "wd": 1e-4,
        },
        "Classical EGNN (3345p)": {
            "fn": lambda: ClassicalEGNN(in_dim=3, hidden_dim=16, num_layers=2),
            "lr": 0.005, "wd": 1e-4,
        },
    }

    n_complexes = len(records)
    n_clusters = len(clusters)

    results = {}
    for name, spec in model_specs.items():
        n_params = sum(p.numel() for p in spec["fn"]().parameters() if p.requires_grad)
        print(f"\n>>> Running 5-Fold Cluster-Level CV: {name} | Exact Params: {n_params}")

        oof_bal_probs = [None] * n_complexes
        oof_full_probs = [None] * n_complexes
        per_complex_bal_auc = np.zeros(n_complexes, dtype=np.float64)
        per_complex_full_auc = np.zeros(n_complexes, dtype=np.float64)
        fold_bal_aucs = []
        t0 = time.time()

        for f in range(5):
            test_idx = [idx for _, grp in fold_clusters[f] for idx in grp]

            # Build strictly disjoint internal validation split (~20% of remaining clusters)
            rem_clusters = [grp for f_other in range(5) if f_other != f for _, grp in fold_clusters[f_other]]
            n_val_c = max(4, len(rem_clusters) // 5)
            # Rotate which remaining clusters are used for validation across folds
            shift = (f * n_val_c) % len(rem_clusters)
            rem_rot = rem_clusters[shift:] + rem_clusters[:shift]
            val_idx = [idx for grp in rem_rot[:n_val_c] for idx in grp]
            train_idx = [idx for grp in rem_rot[n_val_c:] for idx in grp]

            train_t = concat_records(records, train_idx)
            val_t = concat_records(records, val_idx)

            model = train_fold_model(
                spec["fn"], train_t, val_t,
                seed=42 + f * 100, epochs=65, batch_size=128,
                lr=spec["lr"], wd=spec["wd"],
            )

            f_probs_l, f_labels_l = [], []
            with torch.no_grad():
                for c_idx in test_idx:
                    rec = records[c_idx]
                    bp = torch.sigmoid(model(rec["bal_pos"], rec["bal_feats"], center_pos=rec["bal_ctr"])).numpy()
                    by = rec["bal_labels"].numpy().astype(int)
                    oof_bal_probs[c_idx] = bp
                    per_complex_bal_auc[c_idx] = roc_auc_score(by, bp) * 100.0
                    f_probs_l.append(bp)
                    f_labels_l.append(by)

                    fp = torch.sigmoid(model(rec["full_pos"], rec["full_feats"], center_pos=rec["full_ctr"])).numpy()
                    fy = rec["full_labels"].numpy().astype(int)
                    oof_full_probs[c_idx] = fp
                    per_complex_full_auc[c_idx] = roc_auc_score(fy, fp) * 100.0

            fold_auc = roc_auc_score(np.concatenate(f_labels_l), np.concatenate(f_probs_l)) * 100.0
            fold_bal_aucs.append(float(fold_auc))
            print(f"    Fold {f+1}/5 (Test: {len(test_idx):2d} complexes) | Fold Balanced ROC-AUC: {fold_auc:5.2f}%")

        elapsed = time.time() - t0

        # Pooled out-of-fold metrics across ALL 60 complexes
        all_bal_p = np.concatenate(oof_bal_probs)
        all_bal_y = np.concatenate([r["bal_labels"].numpy().astype(int) for r in records])
        pooled_bal_roc = float(roc_auc_score(all_bal_y, all_bal_p) * 100.0)
        pooled_bal_pr = float(average_precision_score(all_bal_y, all_bal_p) * 100.0)

        all_full_p = np.concatenate(oof_full_probs)
        all_full_y = np.concatenate([r["full_labels"].numpy().astype(int) for r in records])
        pooled_full_roc = float(roc_auc_score(all_full_y, all_full_p) * 100.0)

        # Strictly independent per-cluster AUCs (N=37 independent clusters)
        per_cluster_bal_auc = np.array([
            float(np.mean([per_complex_bal_auc[idx] for idx in grp])) for grp in clusters
        ], dtype=np.float64)
        per_cluster_full_auc = np.array([
            float(np.mean([per_complex_full_auc[idx] for idx in grp])) for grp in clusters
        ], dtype=np.float64)

        cl_mean, cl_lo, cl_hi = cluster_bootstrap_ci(per_cluster_bal_auc, n_boot=2000, seed=42)
        fl_mean, fl_lo, fl_hi = cluster_bootstrap_ci(per_cluster_full_auc, n_boot=2000, seed=42)

        results[name] = {
            "params": n_params,
            "time_s": float(elapsed),
            "fold_bal_aucs": fold_bal_aucs,
            "fold_mean_std": [float(np.mean(fold_bal_aucs)), float(np.std(fold_bal_aucs, ddof=1))],
            "pooled_bal_roc_auc": pooled_bal_roc,
            "pooled_bal_pr_auc": pooled_bal_pr,
            "pooled_full_roc_auc": pooled_full_roc,
            "per_complex_bal_mean_std": [float(np.mean(per_complex_bal_auc)), float(np.std(per_complex_bal_auc, ddof=1))],
            "per_complex_full_mean_std": [float(np.mean(per_complex_full_auc)), float(np.std(per_complex_full_auc, ddof=1))],
            "cluster_bal_mean_ci95": [cl_mean, cl_lo, cl_hi],
            "cluster_full_mean_ci95": [fl_mean, fl_lo, fl_hi],
            "per_complex_bal_aucs": per_complex_bal_auc.tolist(),
            "per_complex_full_aucs": per_complex_full_auc.tolist(),
            "per_cluster_bal_aucs": per_cluster_bal_auc.tolist(),
            "per_cluster_full_aucs": per_cluster_full_auc.tolist(),
        }

    # ==========================================================================
    # 5. Summary Table & Complex-/Cluster-Level Statistical Tests
    # ==========================================================================
    print("\n" + "=" * 108)
    print(f"5-FOLD CLUSTER-LEVEL CV SUMMARY (ALL {n_complexes} COMPLEXES | {n_clusters} INDEPENDENT CLUSTERS | MAX CROSS-FOLD ID: {max_cross_pid:.2f}%)")
    print("=" * 108)
    hdr = (
        f"{'Model Architecture':<27} | {'Params':>6} | {'Pooled OOF':>10} | "
        f"{'5-Fold Mean±SD':>14} | {'Cluster Mean [95% CI]':>23} | {'Full-Surf OOF':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        cm, clo, chi = r["cluster_bal_mean_ci95"]
        print(
            f"{name:<27} | {r['params']:>6d} | {r['pooled_bal_roc_auc']:9.2f}% | "
            f"{r['fold_mean_std'][0]:5.2f} ± {r['fold_mean_std'][1]:4.2f}% | "
            f"{cm:5.2f}% [{clo:5.2f}, {chi:5.2f}] | {r['pooled_full_roc_auc']:12.2f}%"
        )

    stat_comparisons = {}
    for q_name in ["QMaSIF (L=2, 17p)", "QMaSIF (L=3, 23p)"]:
        print(f"\n[STATISTICAL TESTS: {q_name} vs CLASSICAL BASELINES (N={n_complexes} Complexes & N={n_clusters} Clusters)]")
        q_comp = np.array(results[q_name]["per_complex_bal_aucs"])
        q_clus = np.array(results[q_name]["per_cluster_bal_aucs"])
        stat_comparisons[q_name] = {}

        for base in [
            "Center-Only Logistic (4p)",
            "Iso-Param Linear (19p)",
            "Iso-Param Tiny MLP (17p)",
            "Fair Classical MLP (321p)",
            "Classical EGNN (3345p)",
        ]:
            b_comp = np.array(results[base]["per_complex_bal_aucs"])
            b_clus = np.array(results[base]["per_cluster_bal_aucs"])

            w_c, p_c = safe_wilcoxon(q_comp, b_comp)
            w_cl, p_cl = safe_wilcoxon(q_clus, b_clus)
            dm, dlo, dhi = cluster_bootstrap_diff_ci(q_clus, b_clus, n_boot=2000, seed=42)
            wins_comp = int(np.sum(q_comp > b_comp))

            stat_comparisons[q_name][base] = {
                "complex_delta_mean": float(np.mean(q_comp - b_comp)),
                "complex_wilcoxon_p": p_c,
                "complex_wins_out_of_60": wins_comp,
                "cluster_delta_mean_ci95": [dm, dlo, dhi],
                "cluster_wilcoxon_p": p_cl,
            }
            print(
                f"  vs {base:<26} | ΔCluster: {dm:+5.2f}% [{dlo:+5.2f}, {dhi:+5.2f}] | "
                f"Wilcoxon p(60 comp)={p_c:.4f}, p(37 clus)={p_cl:.4f} | Wins: {wins_comp}/{n_complexes}"
            )

    with open("step7_cluster_cv_results.json", "w") as f:
        json.dump({
            "n_complexes": n_complexes,
            "n_clusters": n_clusters,
            "max_cross_fold_seq_id": max_cross_pid,
            "complex_entries": [r["entry"] for r in records],
            "models": results,
            "stat_comparisons": stat_comparisons,
        }, f, indent=2)

    # ==========================================================================
    # 6. Publication-Quality Figure: Efficiency Frontier & Cluster-Level ΔAUC
    # ==========================================================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2), dpi=300)

    # Panel (a): Parameter Efficiency Frontier (Cluster Mean AUC + 95% Bootstrap CI vs Log Params)
    palette = {
        "QMaSIF (L=2, 17p)": ("#0284C7", "o", 10),
        "QMaSIF (L=3, 23p)": ("#0369A1", "D", 10),
        "Center-Only Logistic (4p)": ("#64748B", "^", 8),
        "Iso-Param Linear (19p)": ("#F59E0B", "s", 8),
        "Iso-Param Tiny MLP (17p)": ("#D97706", "P", 9),
        "Fair Classical MLP (321p)": ("#16A34A", "v", 8),
        "Classical EGNN (3345p)": ("#9333EA", "X", 9),
    }

    for name, r in results.items():
        col, mk, ms = palette[name]
        cm, clo, chi = r["cluster_bal_mean_ci95"]
        ax1.errorbar(
            r["params"], cm,
            yerr=[[cm - clo], [chi - cm]],
            fmt=mk, color=col, markersize=ms, capsize=4, lw=2.0,
            label=f"{name}: {cm:.2f}%",
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Trainable Parameters (Log Scale)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Out-of-Fold Cluster-Level ROC-AUC (% ± 95% CI)", fontsize=10, fontweight="bold")
    ax1.set_title(f"(a) Parameter Efficiency Across All {n_complexes} Complexes ({n_clusters} Clusters)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.3, facecolor="white", framealpha=0.95)
    ax1.grid(True, which="both", ls="--", alpha=0.45)

    # Panel (b): Cluster-Level Bootstrap ΔAUC (QMaSIF L=3 & L=2 vs Classical Baselines)
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

    d2 = [stat_comparisons["QMaSIF (L=2, 17p)"][b]["cluster_delta_mean_ci95"] for b in bases]
    d3 = [stat_comparisons["QMaSIF (L=3, 23p)"][b]["cluster_delta_mean_ci95"] for b in bases]

    ax2.axvline(0.0, color="#0F172A", linestyle="--", lw=1.4)
    ax2.errorbar(
        [x[0] for x in d2], y_pos + h / 2,
        xerr=[[x[0] - x[1] for x in d2], [x[2] - x[0] for x in d2]],
        fmt="o", color="#0284C7", capsize=3.5, lw=2.0, markersize=6.5, label="QMaSIF (L=2, 17p) − Baseline",
    )
    ax2.errorbar(
        [x[0] for x in d3], y_pos - h / 2,
        xerr=[[x[0] - x[1] for x in d3], [x[2] - x[0] for x in d3]],
        fmt="D", color="#E11D48", capsize=3.5, lw=2.0, markersize=6.5, label="QMaSIF (L=3, 23p) − Baseline",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_b, fontsize=9.5, fontweight="bold")
    ax2.set_xlabel(r"Cluster-Level $\Delta$ROC-AUC (%) with 95% Bootstrap CI", fontsize=10, fontweight="bold")
    ax2.set_title(f"(b) Paired Cluster-Level Advantage (N={n_clusters} Clusters)", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5, facecolor="white", framealpha=0.95)
    ax2.grid(True, axis="x", ls="--", alpha=0.45)

    plt.tight_layout()
    plt.savefig("Fig_Priority1_ClusterCV_IsoParam.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ PRIORITY 1 COMPLETED in {(time.time() - t_global) / 60.0:.1f} min!")
    print("   Saved: 'step7_cluster_cv_results.json' and 'Fig_Priority1_ClusterCV_IsoParam.png'")


if __name__ == "__main__":
    run_step7()