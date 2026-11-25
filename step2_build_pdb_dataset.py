import os
import time
import urllib.request
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from Bio.Align import PairwiseAligner, substitution_matrices

_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "local"
_ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")
_ALIGNER.open_gap_score = -11.0
_ALIGNER.extend_gap_score = -1.0

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Standard Kyte-Doolittle Hydropathy scale [-4.5, 4.5]
KYTE_DOOLITTLE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}

# Standard Wimley-White whole-residue interfacial hydrophobicity scale (inverted: higher = more interfacial)
WIMLEY_WHITE_IFACE = {
    "TRP": 1.85, "PHE": 1.13, "TYR": 0.94, "LEU": 0.56, "ILE": 0.31,
    "CYS": 0.24, "MET": 0.23, "VAL": -0.07, "SER": -0.13, "THR": -0.14,
    "ALA": -0.17, "GLY": -0.01, "HIS": -0.96, "ASN": -0.42, "PRO": -0.45,
    "GLN": -0.58, "ARG": -0.81, "LYS": -0.99, "ASP": -1.23, "GLU": -2.02,
}

AMBER_NET_CHARGE = {
    "ARG": +1.0, "LYS": +1.0, "HIS": +0.25,
    "ASP": -1.0, "GLU": -1.0,
}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

DB55_PDB_IDS = [
    "1AK4", "1AY7", "1BRS", "1BVN", "1CGI", "1CLV", "1D6R", "1DFJ", "1E6E", "1E96",
    "1EAW", "1EWY", "1EZU", "1F34", "1F51", "1FLE", "1FQJ", "1GCQ", "1GHQ", "1GL1",
    "1GLA", "1GPW", "1GRN", "1GXD", "1H1V", "1H9D", "1HE1", "1HE8", "1I2M", "1I4D",
    "1IBR", "1J2J", "1JIW", "1JK9", "1JTD", "1JTG", "1K5D", "1K74", "1KAC", "1KTZ",
    "1KXP", "1LFD", "1M10", "1MAH", "1ML0", "1MQ8", "1NW9", "1OPH", "1OYV", "1PPE",
    "1PXV", "1QA9", "1R0R", "1S1Q", "1T6B", "1TMQ", "1UDI", "1US7", "1WQ1", "1XD3",
]


def download_pdb(pdb_id: str, cache_dir: str = "pdb_cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{pdb_id.lower()}.pdb")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        return out_path
    except Exception:
        return ""


def parse_pdb_chains(pdb_path: str):
    raw_chains = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM  "):
                continue
            if line[16] not in (" ", "A"):
                continue
            res_name = line[17:20].strip()
            if res_name not in AA3_TO_1:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if element == "H" or (element == "" and atom_name.startswith("H")):
                continue
            chain_id = line[21].strip() or "_"
            res_seq = line[22:27].strip()
            try:
                coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue

            raw_chains.setdefault(chain_id, {}).setdefault(
                res_seq, {"res_name": res_name, "ca": None, "sc": [], "heavy": []}
            )
            entry = raw_chains[chain_id][res_seq]
            entry["heavy"].append(coord)
            if atom_name == "CA":
                entry["ca"] = coord
            if atom_name not in BACKBONE_ATOMS:
                entry["sc"].append(coord)

    parsed = {}
    for cid, res_dict in raw_chains.items():
        ca_list, sc_list, heavy_list, res_names, res_ids, seq_chars = [], [], [], [], [], []
        for rseq, info in res_dict.items():
            if info["ca"] is None or len(info["heavy"]) == 0:
                continue
            ca_list.append(info["ca"])
            sc_centroid = np.mean(info["sc"], axis=0) if len(info["sc"]) > 0 else info["ca"]
            sc_list.append(sc_centroid)
            heavy_list.append(np.array(info["heavy"], dtype=np.float32))
            res_names.append(info["res_name"])
            res_ids.append(rseq)
            seq_chars.append(AA3_TO_1[info["res_name"]])

        if len(ca_list) >= 25:
            parsed[cid] = {
                "chain_id": cid,
                "ca_coords": np.array(ca_list, dtype=np.float32),
                "sc_coords": np.array(sc_list, dtype=np.float32),
                "heavy_coords": heavy_list,
                "res_names": res_names,
                "res_ids": res_ids,
                "sequence": "".join(seq_chars),
            }
    return parsed


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    if seq1 == seq2:
        return 1.0
    score = _ALIGNER.score(seq1, seq2)
    if score < 50.0:
        return 0.0
    aln = _ALIGNER.align(seq1, seq2)[0]
    n_ident, aligned_len = 0, 0
    for (s1_a, s1_b), (s2_a, s2_b) in zip(aln.aligned[0], aln.aligned[1]):
        aligned_len += (s1_b - s1_a)
        n_ident += sum(a == b for a, b in zip(seq1[s1_a:s1_b], seq2[s2_a:s2_b]))
    if aligned_len < 20:
        return 0.0
    return min(1.0, n_ident / float(min(len(seq1), len(seq2))))


def compute_unbound_surface_fields(chain_data):
    """
    Compute 3 strictly E(3)-invariant, Gaussian-smoothed local surface fields
    SOLELY from the single unbound monomer chain (zero partner information):
      - q_field in [-1, 1] : Local screened electrostatic potential (Debye-Huckel, AMBER99)
      - h_field in [0, 1]  : Local solvent-exposed interfacial hydrophobicity (Wimley-White + KD)
      - s_field in [0, 1]  : Local geometric surface shape (Half-Sphere Exposure + atomic packing)
    """
    ca = chain_data["ca_coords"]
    sc = chain_data["sc_coords"]
    res_names = chain_data["res_names"]
    N = len(ca)

    dist_sc = cdist(sc, sc)
    dist_ca = cdist(ca, ca)

    # 1. Exact foreign heavy-atom packing within 10A (fast SASA/burial proxy)
    all_heavy = np.vstack(chain_data["heavy_coords"])
    own_counts = np.array([len(h) for h in chain_data["heavy_coords"]], dtype=np.float32)
    tree = cKDTree(all_heavy)
    n_heavy_10 = np.array(tree.query_ball_point(sc, r=10.0, return_length=True), dtype=np.float32) - own_counts
    n_heavy_10 = np.maximum(n_heavy_10, 0.0)

    # Exposed surface mask: exclude deeply buried hydrophobic core (top 30% densest atoms)
    surf_mask = n_heavy_10 <= np.percentile(n_heavy_10, 70)
    surf_float = surf_mask.astype(np.float32)

    # 2. Half-Sphere Exposure (HSE-up) along outward side-chain vector (strictly E(3)-invariant)
    hse_up = np.zeros(N, dtype=np.float32)
    for i in range(N):
        u = sc[i] - ca[i]
        norm_u = np.linalg.norm(u)
        if norm_u < 0.1:
            nbrs_8 = np.where((dist_ca[i] > 0.1) & (dist_ca[i] <= 10.0))[0]
            if len(nbrs_8) > 0:
                u = ca[i] - ca[nbrs_8].mean(axis=0)
                norm_u = np.linalg.norm(u)
        if norm_u > 1e-6:
            u = u / norm_u
            nbrs_13 = np.where((dist_ca[i] > 0.1) & (dist_ca[i] <= 13.0))[0]
            if len(nbrs_13) > 0:
                dots = (ca[nbrs_13] - ca[i]) @ u
                hse_up[i] = float(np.sum(dots > 0.0))

    # Surface Gaussian smoothing kernel (sigma = 6.0A) over exposed surface residues ONLY
    W_surf = np.exp(-0.5 * (dist_sc / 6.0) ** 2) * (dist_sc <= 12.0) * surf_float[None, :]
    W_norm = W_surf / (W_surf.sum(axis=1, keepdims=True) + 1e-6)

    # Field 0: Local Electrostatic / Charge Field q in [-1, 1]
    raw_q = np.array([AMBER_NET_CHARGE.get(r, 0.0) for r in res_names], dtype=np.float32)
    dh_kernel = np.exp(-dist_sc / 8.0) / (dist_sc + 2.0)
    dh_pot = np.tanh((dh_kernel @ raw_q) * 2.5)
    q_field = (W_norm @ dh_pot).astype(np.float32)

    # Field 1: Local Surface Interfacial Hydrophobicity Patch Field h in [0, 1]
    ww_raw = np.array([(WIMLEY_WHITE_IFACE[r] + 2.02) / 3.87 for r in res_names], dtype=np.float32)
    kd_raw = np.array([(KYTE_DOOLITTLE[r] + 4.5) / 9.0 for r in res_names], dtype=np.float32)
    res_hydro = 0.65 * ww_raw + 0.35 * kd_raw
    # Smooth over exposed surface neighborhood -> lights up surface hydrophobic/aromatic patches!
    h_smooth = W_norm @ res_hydro
    h_field = np.clip((h_smooth - 0.25) / 0.50, 0.0, 1.0).astype(np.float32)

    # Field 2: Local Geometric Flatness / Exposure Field s in [0, 1]
    # Combines normalized HSE-up (concavity/flatness) and relative surface exposure
    hse_norm = np.clip(hse_up / 20.0, 0.0, 1.0)
    pack_norm = np.clip(n_heavy_10 / 160.0, 0.0, 1.0)
    geom_local = 0.5 * hse_norm + 0.5 * pack_norm
    s_field = np.clip(W_norm @ geom_local, 0.0, 1.0).astype(np.float32)

    chain_data["surf_mask"] = surf_mask
    chain_data["q_field"] = q_field
    chain_data["h_field"] = h_field
    chain_data["s_field"] = s_field
    chain_data["dist_sc"] = dist_sc
    return chain_data


def compute_partner_heavy_distances(chain_a, chain_b):
    ca_dist = cdist(chain_a["ca_coords"], chain_b["ca_coords"])
    Na, Nb = ca_dist.shape
    min_a = np.full(Na, 999.0, dtype=np.float32)
    min_b = np.full(Nb, 999.0, dtype=np.float32)

    cand_a, cand_b = np.where(ca_dist <= 16.0)
    for i, j in zip(cand_a, cand_b):
        d = float(cdist(chain_a["heavy_coords"][i], chain_b["heavy_coords"][j]).min())
        if d < min_a[i]:
            min_a[i] = d
        if d < min_b[j]:
            min_b[j] = d
    return min_a, min_b


def select_best_heterodimer_pair(chains):
    cids = sorted(chains.keys())
    if len(cids) < 2:
        return None
    best_hetero, best_any = None, None
    for i1 in range(len(cids)):
        for i2 in range(i1 + 1, len(cids)):
            c1, c2 = chains[cids[i1]], chains[cids[i2]]
            if cdist(c1["ca_coords"], c2["ca_coords"]).min() > 10.0:
                continue
            min_a, min_b = compute_partner_heavy_distances(c1, c2)
            n_contacts = int(np.sum(min_a <= 5.0) + np.sum(min_b <= 5.0))
            if n_contacts < 20:
                continue
            seq_id = compute_sequence_identity(c1["sequence"], c2["sequence"])
            entry = (n_contacts, cids[i1], cids[i2], min_a, min_b)
            if best_any is None or n_contacts > best_any[0]:
                best_any = entry
            if seq_id < 0.95 and (best_hetero is None or n_contacts > best_hetero[0]):
                best_hetero = entry
    return best_hetero if best_hetero is not None else best_any


def extract_surface_patch(chain_data, center_idx: int, k_max: int = 8, radius: float = 8.0):
    """
    Extract k_max surface sub-patch nodes around center_idx within radius=8.0A
    using deterministic FPS (node 0 = center_idx, rest spread across the 8A patch).
    """
    surf_indices = np.where(chain_data["surf_mask"])[0]
    dists = chain_data["dist_sc"][center_idx, surf_indices]

    in_rad = surf_indices[dists <= radius]
    if len(in_rad) < 4:
        in_rad = surf_indices[dists <= 10.0]
        if len(in_rad) < 4:
            return None
    if len(in_rad) < k_max:
        in_rad = surf_indices[dists <= 12.0]
        if len(in_rad) < k_max:
            return None

    selected = [center_idx]
    candidates = [int(idx) for idx in in_rad if int(idx) != center_idx]
    dist_sc = chain_data["dist_sc"]
    h_arr, s_arr = chain_data["h_field"], chain_data["s_field"]

    min_dists = np.array([dist_sc[c, center_idx] for c in candidates], dtype=np.float32)

    for _ in range(k_max - 1):
        best_pos = 0
        best_key = (-1.0, -999.0, -999.0)
        for idx_c, cand in enumerate(candidates):
            key = (round(float(min_dists[idx_c]), 4), round(float(h_arr[cand]), 4), round(float(s_arr[cand]), 4))
            if key > best_key:
                best_key = key
                best_pos = idx_c
        chosen = candidates.pop(best_pos)
        min_dists = np.delete(min_dists, best_pos)
        selected.append(chosen)
        if candidates:
            new_d = np.array([dist_sc[c, chosen] for c in candidates], dtype=np.float32)
            min_dists = np.minimum(min_dists, new_d)

    sel_arr = np.array(selected, dtype=np.int64)
    pos = chain_data["sc_coords"][sel_arr].copy()
    feats = np.stack([
        chain_data["q_field"][sel_arr],
        chain_data["h_field"][sel_arr],
        chain_data["s_field"][sel_arr],
    ], axis=-1).astype(np.float32)
    center_pos = chain_data["sc_coords"][center_idx].copy()
    return pos, feats, center_pos


def cluster_complexes_by_homology(complex_records, identity_threshold: float = 0.30):
    M = len(complex_records)
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(M):
        for j in range(i + 1, M):
            if any(compute_sequence_identity(s1, s2) >= identity_threshold
                   for s1 in complex_records[i]["sequences"]
                   for s2 in complex_records[j]["sequences"]):
                union(i, j)

    clusters = {}
    for i in range(M):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def build_site_dataset(k_max: int = 8, radius: float = 8.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    t0 = time.time()

    print("=" * 76)
    print("STEP 2 (v3): REAL PDB SURFACE-FIELD DATASET (ZERO-LEAKAGE + DIAGNOSTIC)")
    print("=" * 76)

    complex_records = []
    for pdb_id in DB55_PDB_IDS:
        pdb_path = download_pdb(pdb_id)
        if not pdb_path:
            continue
        chains = parse_pdb_chains(pdb_path)
        pair = select_best_heterodimer_pair(chains)
        if pair is None:
            continue

        _, cid_a, cid_b, min_a, min_b = pair
        chain_a = compute_unbound_surface_fields(chains[cid_a])
        chain_b = compute_unbound_surface_fields(chains[cid_b])

        samples = []
        for chain, min_d in [(chain_a, min_a), (chain_b, min_b)]:
            pos_indices = np.where(chain["surf_mask"] & (min_d <= 5.0))[0]
            neg_indices = np.where(chain["surf_mask"] & (min_d > 10.0))[0]

            pos_patches = []
            for idx in pos_indices:
                p = extract_surface_patch(chain, int(idx), k_max=k_max, radius=radius)
                if p is not None:
                    pos_patches.append((p, 1, float(min_d[idx]), chain["chain_id"], chain["res_ids"][idx]))

            neg_patches = []
            rng.shuffle(neg_indices)
            for idx in neg_indices:
                p = extract_surface_patch(chain, int(idx), k_max=k_max, radius=radius)
                if p is not None:
                    neg_patches.append((p, 0, float(min_d[idx]), chain["chain_id"], chain["res_ids"][idx]))
                if len(neg_patches) >= len(pos_patches):
                    break

            n_bal = min(len(pos_patches), len(neg_patches))
            if n_bal > 0:
                samples.extend(pos_patches[:n_bal])
                samples.extend(neg_patches[:n_bal])

        if len(samples) < 12:
            continue

        complex_records.append({
            "pdb_id": pdb_id,
            "chains": (cid_a, cid_b),
            "sequences": [chain_a["sequence"], chain_b["sequence"]],
            "samples": samples,
        })

    print(f"[INFO] Processed {len(complex_records)} real PDB complexes in {time.time()-t0:.1f}s.")
    clusters = cluster_complexes_by_homology(complex_records, identity_threshold=0.30)
    print(f"[INFO] Homology clusters (<=30% BLOSUM62 SW): {len(clusters)}")

    ak4_c = next((i for i, c in enumerate(clusters) if any(complex_records[m]["pdb_id"] == "1AK4" for m in c)), None)
    others = [i for i in range(len(clusters)) if i != ak4_c]
    rng.shuffle(others)

    target_test = max(1, int(round(0.15 * len(complex_records))))
    target_val = max(1, int(round(0.15 * len(complex_records))))

    test_ids, val_ids, train_ids = [], [], []
    if ak4_c is not None:
        test_ids.extend(clusters[ak4_c])
    for c_idx in others:
        mem = clusters[c_idx]
        if len(test_ids) < target_test and (len(test_ids) + len(mem) <= target_test + 2):
            test_ids.extend(mem)
        elif len(val_ids) < target_val and (len(val_ids) + len(mem) <= target_val + 2):
            val_ids.extend(mem)
        else:
            train_ids.extend(mem)

    for sa, sb, na, nb in [(train_ids, val_ids, "Train", "Val"), (train_ids, test_ids, "Train", "Test"), (val_ids, test_ids, "Val", "Test")]:
        max_id = max(
            compute_sequence_identity(s1, s2)
            for i in sa for j in sb
            for s1 in complex_records[i]["sequences"]
            for s2 in complex_records[j]["sequences"]
        )
        print(f"  Max cross-split seq identity ({na} vs {nb}): {max_id*100:.2f}%")
        assert max_id < 0.30
    print("  STATUS: PASS (Zero homology leakage verified!)")

    def pack(comp_indices):
        pos_l, feat_l, ctr_l, y_l, d_l, meta_l, comp_names = [], [], [], [], [], [], []
        for idx in comp_indices:
            rec = complex_records[idx]
            comp_names.append(f"{rec['pdb_id']}_{rec['chains'][0]}_{rec['chains'][1]}")
            for (p_pos, p_feat, p_ctr), label, dist_iface, cid, rid in rec["samples"]:
                pos_l.append(p_pos); feat_l.append(p_feat); ctr_l.append(p_ctr)
                y_l.append(label); d_l.append(dist_iface); meta_l.append((rec["pdb_id"], cid, rid))
        return {
            "pos": torch.tensor(np.array(pos_l), dtype=torch.float32),
            "feats": torch.tensor(np.array(feat_l), dtype=torch.float32),
            "center": torch.tensor(np.array(ctr_l), dtype=torch.float32),
            "labels": torch.tensor(np.array(y_l), dtype=torch.float32),
            "dist_to_interface": torch.tensor(np.array(d_l), dtype=torch.float32),
            "residue_meta": meta_l,
            "complexes": comp_names,
        }

    dataset = {
        "train": pack(train_ids),
        "val": pack(val_ids),
        "test": pack(test_ids),
        "metadata": {
            "total_complexes": len(complex_records),
            "total_chains": len(complex_records) * 2,
            "num_clusters": len(clusters),
            "seq_identity_cutoff": 0.30,
            "interface_cutoff": 5.0,
            "non_interface_cutoff": 10.0,
            "patch_radius": radius,
            "k_max": k_max,
        },
    }
    torch.save(dataset, "qmasif_real_pdb_site_dataset.pt")

    # Quick Biophysical Signal Diagnostic (Train -> Val ONLY, Test remains untouched!)
    print("\n[DIAGNOSTIC] Real Biophysical Signal Check (Train -> Val, k=4):")
    tr_f, tr_p, tr_y = dataset["train"]["feats"][:, :4, :].numpy(), dataset["train"]["pos"][:, :4, :].numpy(), dataset["train"]["labels"].numpy()
    va_f, va_p, va_y = dataset["val"]["feats"][:, :4, :].numpy(), dataset["val"]["pos"][:, :4, :].numpy(), dataset["val"]["labels"].numpy()

    def build_flat(p_arr, f_arr):
        idx_i, idx_j = np.triu_indices(4, k=1)
        d_mat = np.linalg.norm(p_arr[:, :, None, :] - p_arr[:, None, :, :], axis=-1)
        d_ij = d_mat[:, idx_i, idx_j] / 10.0
        return np.concatenate([f_arr.reshape(len(p_arr), -1), d_ij], axis=1)

    for fi, fname in enumerate(["Electrostatics (q)", "Surface Hydrophobicity (h)", "Surface Flatness/HSE (s)"]):
        auc_single = roc_auc_score(va_y, va_f[:, 0, fi])
        auc_abs = roc_auc_score(va_y, -np.abs(va_f[:, 0, fi]))
        print(f"  Center {fname:<28}: Val ROC-AUC = {max(auc_single, 1-auc_single)*100:.2f}% (|f| AUC = {max(auc_abs, 1-auc_abs)*100:.2f}%)")

    X_tr, X_va = build_flat(tr_p, tr_f), build_flat(va_p, va_f)
    lr = LogisticRegression(max_iter=500).fit(X_tr, tr_y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42).fit(X_tr, tr_y)
    print(f"  Combined 18-dim Logistic Regression : Val ROC-AUC = {roc_auc_score(va_y, lr.predict_proba(X_va)[:, 1])*100:.2f}%")
    print(f"  Combined 18-dim Random Forest       : Val ROC-AUC = {roc_auc_score(va_y, rf.predict_proba(X_va)[:, 1])*100:.2f}%")

    n_tr, n_va, n_te = len(dataset["train"]["labels"]), len(dataset["val"]["labels"]), len(dataset["test"]["labels"])
    print("\n" + "=" * 76)
    print(f"DATASET SAVED: Train={n_tr} | Val={n_va} | Test={n_te} (Total={n_tr+n_va+n_te} balanced patches)")
    print("=" * 76)


if __name__ == "__main__":
    build_site_dataset()