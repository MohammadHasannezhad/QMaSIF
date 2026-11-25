"""
Unified reproduction CLI for QMaSIF.

Examples:
    python run_all.py --check         # Verify all required scripts and checkpoints exist
    python run_all.py --quick         # Run Steps 1, 2, and 6 (~1 min using saved checkpoint)
    python run_all.py --step 9        # Run only Step 9
    python run_all.py --from-step 7   # Run Steps 7 through 10
    python run_all.py --full          # Run full end-to-end reproduction (Steps 1-10)
"""

import argparse
import os
import subprocess
import sys
import time


PIPELINE = [
    (1, "step1_verify_core.py", "Mathematical & PennyLane parameter-shift verification (Table 1)"),
    (2, "step2_build_pdb_dataset.py", "DB5.5 PDB download, surface fields & <=30% BLOSUM62 clustering (Table 2)"),
    (3, "step3_train_and_benchmark.py", "Protocol B multi-seed holdout & SO(3)+50A stress test (Table 6, Figs. 4-6)"),
    (4, "step4_ablations.py", "Protocol B qubit width (k) & circuit depth (L) scaling (Table 7, Fig. 7)"),
    (5, "step5_nisq_and_bp.py", "Simulated depolarizing noise & finite-shot robustness (Fig. 8)"),
    (6, "step6_pymol_interpretability.py", "Full-surface interpretability & 1AK4 PyMOL export (Fig. 9)"),
    (7, "step7_priority1_cluster_cv.py", "Protocol A: 5-fold cluster-level CV & iso-param baselines (Tables 3-4)"),
    (8, "step8_priority2_3_ablation_and_symmetry.py", "Fair L=3 CV ablations, Symmetric Readout & 24-perm tie test (Table 5, Fig. 3)"),
    (9, "step9_final_reviewer_checks.py", "Matched 23p classical mirrors, Holm-Bonferroni & 10-patch trainability (Fig. 10)"),
    (10, "update_fig2_for_overleaf.py", "Generate final updated Figure 2 (9 models, 7 baselines)"),
]


def resolve_script(filename: str):
    if os.path.exists(filename):
        return filename
    if filename == "step1_verify_core.py" and os.path.exists("tep1_verify_core.py"):
        return "tep1_verify_core.py"
    return None


def run_check():
    print("=" * 84)
    print("QMaSIF REPOSITORY INTEGRITY CHECK")
    print("=" * 84)
    missing = []
    for fname in ["qmasif_core.py", "qmasif_best_model.pt", "README.md", "requirements.txt"]:
        ok = os.path.exists(fname)
        print(f"  [{'OK' if ok else 'MISSING':>7}] {fname}")
        if not ok:
            missing.append(fname)

    for step_num, script_name, _ in PIPELINE:
        found = resolve_script(script_name)
        status = "OK" if found else "MISSING"
        extra = f" (found as {found})" if found and found != script_name else ""
        print(f"  [{status:>7}] Step {step_num:02d}: {script_name}{extra}")
        if not found:
            missing.append(script_name)

    for folder in ["docs", "figures", "results", "pymol"]:
        ok = os.path.isdir(folder)
        n_items = len([f for f in os.listdir(folder) if not f.startswith(".")]) if ok else 0
        print(f"  [{'OK' if ok else 'MISSING':>7}] {folder}/ ({n_items} files)")

    if missing:
        print(f"\n[WARN] Missing {len(missing)} item(s): {missing}")
        sys.exit(1)
    print("\n[PASS] All core modules and pipeline scripts are ready!")


def main():
    parser = argparse.ArgumentParser(description="Reproduce QMaSIF paper experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Verify all scripts and folders exist")
    group.add_argument("--quick", action="store_true", help="Run Steps 1, 2, and 6 (using saved checkpoint)")
    group.add_argument("--full", action="store_true", help="Run full pipeline (Steps 1 through 10)")
    group.add_argument("--step", type=int, choices=range(1, 11), metavar="N", help="Run only Step N")
    group.add_argument("--from-step", type=int, choices=range(1, 11), metavar="N", help="Run Steps N through 10")
    args = parser.parse_args()

    if args.check:
        run_check()
        return

    if args.quick:
        selected = [s for s in PIPELINE if s[0] in (1, 2, 6)]
    elif args.full:
        selected = PIPELINE
    elif args.step is not None:
        selected = [s for s in PIPELINE if s[0] == args.step]
    else:
        selected = [s for s in PIPELINE if s[0] >= args.from_step]

    t_total = time.time()
    for step_num, script_name, desc in selected:
        script_path = resolve_script(script_name)
        if script_path is None:
            raise FileNotFoundError(f"Required script not found: '{script_name}'")
        print("=" * 84)
        print(f"[STEP {step_num:02d}] {desc}")
        print(f"          Running: {sys.executable} {script_path}")
        print("=" * 84)
        t0 = time.time()
        subprocess.run([sys.executable, script_path], check=True)
        print(f"[DONE] Step {step_num:02d} finished in {time.time() - t0:.1f}s\n")

    print("=" * 84)
    print(f"All selected QMaSIF steps completed in {(time.time() - t_total) / 60.0:.1f} min.")
    print("=" * 84)


if __name__ == "__main__":
    main()
